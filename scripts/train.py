"""
Training script for Adaptive-MPNet with DDP multi-GPU support.

Usage:
    # Single GPU
    python train.py --config configs/adaptive_mpnet.yaml

    # Multi-GPU DDP (4 GPUs)
    torchrun --nproc_per_node=4 train.py --config configs/adaptive_mpnet.yaml
"""

import argparse
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from src.models.adaptive_mpnet import AdaptiveMPNet, build_adaptive_mpnet
from src.data.dataset import (PICIADataset, ModalityDropout, ArtifactSimulator,
                               RandomFlip3D, ForegroundCenteredCrop, CenterCrop3D,
                               Compose)
from src.utils.losses import DiceFocalTverskyLoss, DeepSupervisionWrapper, DiceCELoss
from src.utils.metrics import MetricTracker
from src.utils.gpu_monitor import init_nvml, shutdown_nvml, format_gpu_stats, format_all_gpu_stats

logger = logging.getLogger(__name__)

GLOBAL_SEED = 42


def set_seed(seed: int = GLOBAL_SEED, rank: int = 0):
    """Fix all random sources for reproducibility. Each DDP rank gets seed+rank."""
    s = seed + rank
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int):
    """Seed each DataLoader worker deterministically."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def setup_logging(rank: int):
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format=f"[Rank {rank}] %(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def setup_ddp():
    """Initialize DDP if launched with torchrun."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        from datetime import timedelta
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        torch.cuda.set_device(0)
        return 0, 0, 1


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def build_dataloaders(config: dict, rank: int, world_size: int):
    """Build train/val DataLoaders with optional DDP samplers."""
    data_cfg = config.get("data", {})
    aug_cfg = config.get("augmentation", {})
    train_cfg = config.get("training", {})

    data_config = {
        "root_dir": data_cfg.get("dataset_root", "/workspace/P1-MAR/data"),
        "target_spacing": data_cfg.get("target_spacing", [0.5, 0.5, 3.0]),
        "crop_size": data_cfg.get("crop_size", [128, 128, 32]),
        "split_seed": data_cfg.get("seed", 42),
        "lesion_labels_dir": data_cfg.get("lesion_labels_dir",
                                          "labels/csPCa_lesion_delineations/human_expert/merged"),
        "split_source": data_cfg.get("split_source", None),
        "fold": data_cfg.get("fold", 0),
    }
    # Legacy random split ratios (only used if split_source is not set)
    if data_config["split_source"] is None:
        split = data_cfg.get("train_val_test_split", [0.7, 0.15, 0.15])
        data_config["train_ratio"] = split[0]
        data_config["val_ratio"] = split[1]
        data_config["test_ratio"] = split[2]

    # Patch crop size: (D, H, W) from config (H, W, D)
    patch_crop_cfg = data_cfg.get("patch_crop_size", None)
    if patch_crop_cfg:
        patch_crop_dhw = (patch_crop_cfg[2], patch_crop_cfg[0], patch_crop_cfg[1])
        fg_prob = data_cfg.get("foreground_crop_prob", 0.75)
    else:
        patch_crop_dhw = None

    train_transform_list = []
    if patch_crop_dhw:
        train_transform_list.append(
            ForegroundCenteredCrop(crop_size=patch_crop_dhw, fg_prob=fg_prob))
    train_transform_list.extend([
        RandomFlip3D(p=0.5),
        ArtifactSimulator(
            noise_std=aug_cfg.get("gaussian_noise_std", 0.1),
            motion_p=aug_cfg.get("motion_artifact_prob", 0.15),
        ),
        ModalityDropout(p=aug_cfg.get("modality_dropout_prob", 0.3)),
    ])
    train_transforms = Compose(train_transform_list)

    val_transform_list = []
    if patch_crop_dhw:
        val_transform_list.append(CenterCrop3D(crop_size=patch_crop_dhw))
    val_transforms = Compose(val_transform_list) if val_transform_list else None

    train_ds = PICIADataset(data_config["root_dir"], "train", data_config,
                            transform=train_transforms)
    oversample = train_cfg.get("oversample_positive_factor", 1)
    if oversample > 1:
        train_ds.oversample_positives(factor=oversample)
    val_ds = PICIADataset(data_config["root_dir"], "val", data_config,
                          transform=val_transforms)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                       rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size,
                                     rank=rank, shuffle=False) if world_size > 1 else None

    batch_size = train_cfg.get("batch_size", 4)
    num_workers = data_cfg.get("num_workers", 8)

    g = torch.Generator()
    g.manual_seed(GLOBAL_SEED)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    return train_loader, val_loader, train_sampler


def build_optimizer(model: nn.Module, config: dict):
    train_cfg = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("learning_rate", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )
    return optimizer


def build_scheduler(optimizer, config: dict, steps_per_epoch: int):
    train_cfg = config["training"]
    sched_type = train_cfg.get("scheduler", "cosine_warmup")

    if sched_type == "reduce_on_plateau":
        # ReduceLROnPlateau (stepped per epoch after warmup)
        patience = train_cfg.get("plateau_patience", 20)
        factor = train_cfg.get("plateau_factor", 0.5)
        min_lr = train_cfg.get("min_lr", 1e-6)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=patience, factor=factor,
            min_lr=min_lr,
        )
        return scheduler
    else:
        # Cosine warmup (stepped per batch)
        num_epochs = train_cfg.get("num_epochs", 300)
        warmup_epochs = train_cfg.get("warmup_epochs", 10)
        min_lr = train_cfg.get("min_lr", 1e-6)
        total_steps = num_epochs * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(min_lr / train_cfg.get("learning_rate", 1e-4),
                       0.5 * (1 + math.cos(math.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _get_lr(optimizer, scheduler):
    """Get current LR, compatible with both LambdaLR and ReduceLROnPlateau."""
    if hasattr(scheduler, 'get_last_lr'):
        try:
            return scheduler.get_last_lr()[0]
        except Exception:
            pass
    return optimizer.param_groups[0]['lr']


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler,
                    scaler, device, epoch, num_epochs, config, writer,
                    global_step, rank, local_rank):
    model.train()
    train_cfg = config["training"]
    grad_clip = train_cfg.get("gradient_clip", 1.0)
    use_amp = train_cfg.get("mixed_precision", True)
    is_plateau = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    warmup_epochs = train_cfg.get("warmup_epochs", 10)
    base_lr = train_cfg.get("learning_rate", 1e-4)

    total_loss = 0.0
    num_batches = 0

    # Manual warmup for ReduceLROnPlateau
    if is_plateau and epoch < warmup_epochs:
        warmup_lr = base_lr * (epoch + 1) / warmup_epochs
        for pg in optimizer.param_groups:
            pg['lr'] = warmup_lr

    # tqdm progress bar only on rank 0
    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{num_epochs}",
        leave=True,
        dynamic_ncols=True,
        disable=(rank != 0),
    )

    for batch_idx, batch in enumerate(pbar):
        t2w = batch["t2w"].to(device, non_blocking=True)
        hbv = batch["hbv"].to(device, non_blocking=True)
        adc = batch["adc"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            outputs = model([t2w, hbv, adc])
            # Handle deep supervision: outputs may be (logits, [aux_list])
            if isinstance(outputs, (tuple, list)) and len(outputs) == 2:
                logits, aux_list = outputs
            else:
                logits = outputs
                aux_list = []
            if logits.shape[2:] != label.shape[2:]:
                logits = F.interpolate(
                    logits, size=label.shape[2:], mode='trilinear', align_corners=False)
            if aux_list and isinstance(criterion, DeepSupervisionWrapper):
                loss = criterion((logits, aux_list), label)
            else:
                loss = criterion(logits, label)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # Step per-batch schedulers (cosine warmup), not plateau
        if not is_plateau:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1
        global_step += 1

        # Update progress bar
        if rank == 0:
            lr = _get_lr(optimizer, scheduler)
            avg = total_loss / num_batches
            pbar.set_postfix_str(
                f"loss={loss.item():.4f} avg={avg:.4f} lr={lr:.2e}"
            )

        if writer and batch_idx % 20 == 0:
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/lr", _get_lr(optimizer, scheduler), global_step)

    # GPU stats after epoch
    if rank == 0:
        gpu_str = format_gpu_stats(local_rank)
        pbar.set_postfix_str(
            f"avg_loss={total_loss/max(num_batches,1):.4f} | {gpu_str}"
        )

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, global_step


@torch.no_grad()
def validate(model, val_loader, criterion, device, config, rank=0):
    model.eval()
    use_amp = config["training"].get("mixed_precision", True)
    tracker = MetricTracker()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(
        val_loader,
        desc="  Validating",
        leave=False,
        dynamic_ncols=True,
        disable=(rank != 0),
    )

    for batch in pbar:
        t2w = batch["t2w"].to(device, non_blocking=True)
        hbv = batch["hbv"].to(device, non_blocking=True)
        adc = batch["adc"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            outputs = model([t2w, hbv, adc])
            if isinstance(outputs, (tuple, list)):
                logits = outputs[0]
            else:
                logits = outputs
            if logits.shape[2:] != label.shape[2:]:
                logits = F.interpolate(
                    logits, size=label.shape[2:], mode='trilinear', align_corners=False)
            loss = criterion(logits, label)

        total_loss += loss.item()
        num_batches += 1
        tracker.update(logits.float(), label)

    metrics = tracker.compute()
    metrics["loss"] = total_loss / max(num_batches, 1)
    return metrics


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, metrics,
                    config, filepath):
    state = {
        "epoch": epoch,
        "model_state_dict": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "metrics": metrics,
        "config": config,
    }
    torch.save(state, filepath)


def main():
    parser = argparse.ArgumentParser(description="Train Adaptive-MPNet")
    parser.add_argument("--config", type=str, default="configs/adaptive_mpnet.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--reset_scheduler", action="store_true",
                        help="Reset LR scheduler state when resuming")
    parser.add_argument("--name", type=str, default=None,
                        help="Run name for checkpoint/log directories (default: from config model.name)")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED,
                        help="Global random seed (default: 42)")
    parser.add_argument("--output_prefix", type=str, default=None,
                        help="Optional prefix for checkpoint/log dirs (e.g. 'ablation_no_deepsup')")
    args = parser.parse_args()

    # Setup DDP
    rank, local_rank, world_size = setup_ddp()
    setup_logging(rank)
    seed = args.seed
    set_seed(seed, rank)
    device = torch.device(f"cuda:{local_rank}")

    # Init GPU monitoring
    nvml_ok = init_nvml() if rank == 0 else False

    # Load config
    config_path = Path(__file__).parent / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    train_cfg = config["training"]

    # Create output dirs (seed-aware)
    name = args.name or config["model"].get("name", "AdaptiveMPNet")
    seed_tag = f"seed{seed}"
    base_ckpt = Path(train_cfg.get("checkpoint_dir", "outputs/checkpoints"))
    base_log = Path(train_cfg.get("log_dir", "outputs/logs"))
    if args.output_prefix:
        base_ckpt = base_ckpt / args.output_prefix
        base_log = base_log / args.output_prefix
    ckpt_dir = base_ckpt / seed_tag / name
    log_dir = base_log / seed_tag / name
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

    # Build model
    model = build_adaptive_mpnet(config).to(device)
    if world_size > 1:
        # find_unused_parameters needed when deep_supervision=false
        # (aux heads exist but aren't used in loss)
        find_unused = not train_cfg.get("deep_supervision", True)
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=find_unused)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        logger.info("=" * 60)
        logger.info("Adaptive-MPNet Training")
        logger.info("=" * 60)
        logger.info("Parameters: %.2fM", param_count / 1e6)
        logger.info("GPUs: %d | Batch/GPU: %d | Effective batch: %d",
                     world_size, train_cfg.get("batch_size", 4),
                     world_size * train_cfg.get("batch_size", 4))

    # Build data
    train_loader, val_loader, train_sampler = build_dataloaders(config, rank, world_size)
    if rank == 0:
        logger.info("Train: %d batches (%d samples), Val: %d batches (%d samples)",
                     len(train_loader), len(train_loader.dataset),
                     len(val_loader), len(val_loader.dataset))
        if nvml_ok:
            logger.info("GPU status: %s", format_all_gpu_stats(world_size))
        logger.info("-" * 60)

    # Loss, optimizer, scheduler
    loss_name = train_cfg.get("loss", "dice_focal_tversky")
    if loss_name in ("dice_focal_tversky", "dice_focal"):
        base_loss = DiceFocalTverskyLoss(
            dice_weight=train_cfg.get("dice_weight", 1.0),
            focal_weight=train_cfg.get("focal_weight", 1.0),
            tversky_weight=train_cfg.get("tversky_weight", 1.0),
            focal_alpha=train_cfg.get("focal_alpha", 0.75),
            focal_gamma=train_cfg.get("focal_gamma", 2.0),
            tversky_alpha=train_cfg.get("tversky_alpha", 0.3),
            tversky_beta=train_cfg.get("tversky_beta", 0.7),
        )
    else:
        base_loss = DiceCELoss(
            dice_weight=train_cfg.get("dice_weight", 1.0),
            ce_weight=train_cfg.get("ce_weight", 1.0),
            ce_class_weight=train_cfg.get("ce_class_weight", None),
        )
    if train_cfg.get("deep_supervision", True):
        criterion = DeepSupervisionWrapper(base_loss)
    else:
        criterion = base_loss
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, len(train_loader))
    scaler = GradScaler(enabled=train_cfg.get("mixed_precision", True))

    # Tensorboard
    writer = SummaryWriter(str(log_dir)) if rank == 0 else None

    # Resume
    start_epoch = 0
    best_dice = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        raw_model = model.module if isinstance(model, DDP) else model
        raw_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        if args.reset_scheduler:
            # Override LR from config and reset scheduler + best_dice
            new_lr = train_cfg.get("learning_rate", 1e-4)
            for pg in optimizer.param_groups:
                pg['lr'] = new_lr
            logger.info("Scheduler reset: LR set to %.2e", new_lr)
            best_dice = 0.0
        else:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            scaler.load_state_dict(ckpt["scaler_state_dict"])
            best_dice = ckpt.get("metrics", {}).get("dice", 0.0)
        logger.info("Resumed from epoch %d, best dice %.4f", start_epoch, best_dice)

    # Training loop
    num_epochs = train_cfg.get("num_epochs", 300)
    global_step = start_epoch * len(train_loader)
    is_plateau = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    early_stop_patience = train_cfg.get("early_stopping_patience", 0)
    epochs_without_improvement = 0

    for epoch in range(start_epoch, num_epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        t0 = time.time()
        train_loss, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, epoch, num_epochs, config, writer,
            global_step, rank, local_rank)
        train_time = time.time() - t0

        # Validate every 5 epochs
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            val_metrics = validate(model, val_loader, criterion, device,
                                   config, rank)

            # Synchronize all ranks after validation to prevent NCCL timeouts
            if dist.is_initialized():
                dist.barrier()

            # Step ReduceLROnPlateau with val Dice (after warmup)
            warmup_epochs = train_cfg.get("warmup_epochs", 10)
            if is_plateau and epoch >= warmup_epochs:
                scheduler.step(val_metrics["dice"])

            if rank == 0:
                gpu_info = ""
                if nvml_ok:
                    gpu_info = f" | {format_gpu_stats(local_rank)}"

                logger.info(
                    "Epoch %d/%d | Train: %.4f | Val: %.4f | "
                    "Dice: %.4f | Sens: %.4f | Spec: %.4f | LR: %.2e | %.1fs%s",
                    epoch + 1, num_epochs, train_loss, val_metrics["loss"],
                    val_metrics["dice"], val_metrics["sensitivity"],
                    val_metrics["specificity"],
                    _get_lr(optimizer, scheduler), train_time, gpu_info,
                )
                if writer:
                    writer.add_scalar("val/loss", val_metrics["loss"], epoch)
                    writer.add_scalar("val/dice", val_metrics["dice"], epoch)
                    writer.add_scalar("val/sensitivity", val_metrics["sensitivity"], epoch)
                    writer.add_scalar("val/specificity", val_metrics["specificity"], epoch)

                # Save best model
                if val_metrics["dice"] > best_dice:
                    best_dice = val_metrics["dice"]
                    epochs_without_improvement = 0
                    save_checkpoint(model, optimizer, scheduler, scaler,
                                    epoch, val_metrics, config,
                                    ckpt_dir / "best_model.pth")
                    logger.info(">>> New best Dice: %.4f — checkpoint saved", best_dice)
                else:
                    epochs_without_improvement += 5  # we validate every 5 epochs

                # Early stopping
                if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
                    logger.info("Early stopping triggered after %d epochs without improvement. "
                                "Best Dice: %.4f", epochs_without_improvement, best_dice)
                    break
        else:
            # Non-validation epoch: brief summary
            if rank == 0:
                logger.info(
                    "Epoch %d/%d | Train Loss: %.4f | %.1fs",
                    epoch + 1, num_epochs, train_loss, train_time,
                )

        # Save periodic checkpoint
        if rank == 0 and epoch % 50 == 0:
            save_checkpoint(model, optimizer, scheduler, scaler,
                            epoch, {}, config,
                            ckpt_dir / f"checkpoint_epoch{epoch:04d}.pth")

    # Final save
    if rank == 0:
        save_checkpoint(model, optimizer, scheduler, scaler,
                        num_epochs - 1, {}, config,
                        ckpt_dir / "final_model.pth")
        logger.info("=" * 60)
        logger.info("Training complete. Best Dice: %.4f", best_dice)
        logger.info("=" * 60)

    if writer:
        writer.close()
    if rank == 0:
        shutdown_nvml()
    cleanup_ddp()


if __name__ == "__main__":
    main()
