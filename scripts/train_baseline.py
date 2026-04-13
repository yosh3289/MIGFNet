"""
Training script for baseline models (nnU-Net, SwinUNETR, Vanilla U-Mamba).

Usage:
    torchrun --nproc_per_node=4 train_baseline.py --config configs/adaptive_mpnet.yaml --model nnUNet_concat
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from src.models.baselines import build_baseline
from src.utils.losses import DiceFocalTverskyLoss, DeepSupervisionWrapper, DiceCELoss
from src.utils.metrics import MetricTracker
from src.utils.gpu_monitor import init_nvml, shutdown_nvml, format_gpu_stats, format_all_gpu_stats
from train import (
    setup_logging, setup_ddp, cleanup_ddp,
    build_dataloaders, build_optimizer, build_scheduler,
    validate, save_checkpoint, _get_lr,
    set_seed, GLOBAL_SEED,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Baseline Model")
    parser.add_argument("--config", type=str, default="configs/adaptive_mpnet.yaml")
    parser.add_argument("--model", type=str, required=True,
                        choices=["nnUNet_concat", "SwinUNETR", "UMamba_concat", "OfficialUNet"])
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    rank, local_rank, world_size = setup_ddp()
    setup_logging(rank)
    set_seed(GLOBAL_SEED, rank)
    device = torch.device(f"cuda:{local_rank}")

    nvml_ok = init_nvml() if rank == 0 else False

    config_path = Path(__file__).parent / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    train_cfg = config["training"]
    model_name = args.model

    # SwinUNETR requires all spatial dims >= 32 (divisible by 2^5)
    if model_name == "SwinUNETR":
        data_cfg = config.get("data", {})
        patch = data_cfg.get("patch_crop_size", data_cfg.get("crop_size", [128, 128, 32]))
        # patch is [H, W, D], ensure D >= 32
        if patch[2] < 32:
            patch[2] = 32
            data_cfg["patch_crop_size"] = patch
            if rank == 0:
                logger.info("SwinUNETR: overriding patch_crop_size D to 32 -> %s", patch)

    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "outputs/checkpoints")) / model_name
    log_dir = Path(train_cfg.get("log_dir", "outputs/logs")) / model_name
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

    # Build baseline model
    model = build_baseline(model_name, config).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        logger.info("=" * 60)
        logger.info("Baseline Training: %s", model_name)
        logger.info("=" * 60)
        logger.info("Parameters: %.2fM", param_count / 1e6)
        logger.info("GPUs: %d | Batch/GPU: %d | Effective batch: %d",
                     world_size, train_cfg.get("batch_size", 4),
                     world_size * train_cfg.get("batch_size", 4))

    train_loader, val_loader, train_sampler = build_dataloaders(config, rank, world_size)
    if rank == 0:
        logger.info("Train: %d batches (%d samples), Val: %d batches (%d samples)",
                     len(train_loader), len(train_loader.dataset),
                     len(val_loader), len(val_loader.dataset))
        if nvml_ok:
            logger.info("GPU status: %s", format_all_gpu_stats(world_size))
        logger.info("-" * 60)

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
    writer = SummaryWriter(str(log_dir)) if rank == 0 else None

    start_epoch = 0
    best_dice = 0.0
    num_epochs = train_cfg.get("num_epochs", 300)
    grad_clip = train_cfg.get("gradient_clip", 1.0)
    use_amp = train_cfg.get("mixed_precision", True)
    is_plateau = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    warmup_epochs = train_cfg.get("warmup_epochs", 10)
    base_lr = train_cfg.get("learning_rate", 1e-4)
    early_stop_patience = train_cfg.get("early_stopping_patience", 0)
    epochs_without_improvement = 0

    for epoch in range(start_epoch, num_epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        model.train()
        total_loss = 0.0
        num_batches = 0
        t0 = time.time()

        # Manual warmup for ReduceLROnPlateau
        if is_plateau and epoch < warmup_epochs:
            warmup_lr = base_lr * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        pbar = tqdm(
            train_loader,
            desc=f"[{model_name}] Epoch {epoch+1}/{num_epochs}",
            leave=True,
            dynamic_ncols=True,
            disable=(rank != 0),
        )

        for batch in pbar:
            t2w = batch["t2w"].to(device, non_blocking=True)
            hbv = batch["hbv"].to(device, non_blocking=True)
            adc = batch["adc"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                outputs = model([t2w, hbv, adc])
                if isinstance(outputs, (tuple, list)) and len(outputs) == 2:
                    logits, aux_list = outputs
                else:
                    logits = outputs
                    aux_list = []
                if logits.shape[2:] != label.shape[2:]:
                    logits = torch.nn.functional.interpolate(
                        logits, size=label.shape[2:], mode='trilinear',
                        align_corners=False)
                if aux_list:
                    loss = criterion((logits, aux_list), label)
                else:
                    loss = criterion(logits, label)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            if not is_plateau:
                scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            if rank == 0:
                lr = _get_lr(optimizer, scheduler)
                avg = total_loss / num_batches
                pbar.set_postfix_str(f"loss={loss.item():.4f} avg={avg:.4f} lr={lr:.2e}")

        avg_loss = total_loss / max(num_batches, 1)
        train_time = time.time() - t0

        # GPU stats at end of epoch
        if rank == 0 and nvml_ok:
            pbar.set_postfix_str(
                f"avg_loss={avg_loss:.4f} | {format_gpu_stats(local_rank)}"
            )

        if epoch % 5 == 0 or epoch == num_epochs - 1:
            val_metrics = validate(model, val_loader, criterion, device,
                                   config, rank)

            # Synchronize all ranks after validation to prevent NCCL timeouts
            if dist.is_initialized():
                dist.barrier()

            # Step ReduceLROnPlateau with val Dice (after warmup)
            if is_plateau and epoch >= warmup_epochs:
                scheduler.step(val_metrics["dice"])

            if rank == 0:
                gpu_info = ""
                if nvml_ok:
                    gpu_info = f" | {format_gpu_stats(local_rank)}"

                logger.info(
                    "[%s] Epoch %d/%d | Train: %.4f | Val: %.4f | "
                    "Dice: %.4f | Sens: %.4f | Spec: %.4f | LR: %.2e | %.1fs%s",
                    model_name, epoch + 1, num_epochs, avg_loss,
                    val_metrics["loss"], val_metrics["dice"],
                    val_metrics["sensitivity"], val_metrics["specificity"],
                    _get_lr(optimizer, scheduler), train_time, gpu_info,
                )
                if writer:
                    writer.add_scalar("val/loss", val_metrics["loss"], epoch)
                    writer.add_scalar("val/dice", val_metrics["dice"], epoch)
                    writer.add_scalar("val/sensitivity", val_metrics["sensitivity"], epoch)
                    writer.add_scalar("val/specificity", val_metrics["specificity"], epoch)

                if val_metrics["dice"] > best_dice:
                    best_dice = val_metrics["dice"]
                    epochs_without_improvement = 0
                    save_checkpoint(model, optimizer, scheduler, scaler,
                                    epoch, val_metrics, config,
                                    ckpt_dir / "best_model.pth")
                    logger.info(">>> [%s] New best Dice: %.4f — checkpoint saved",
                                model_name, best_dice)
                else:
                    epochs_without_improvement += 5

                if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
                    logger.info("[%s] Early stopping after %d epochs without improvement. "
                                "Best Dice: %.4f", model_name, epochs_without_improvement, best_dice)
                    break
        else:
            if rank == 0:
                logger.info("[%s] Epoch %d/%d | Train Loss: %.4f | %.1fs",
                            model_name, epoch + 1, num_epochs, avg_loss, train_time)

        if rank == 0 and epoch % 50 == 0:
            save_checkpoint(model, optimizer, scheduler, scaler,
                            epoch, {}, config,
                            ckpt_dir / f"checkpoint_epoch{epoch:04d}.pth")

    if rank == 0:
        logger.info("=" * 60)
        logger.info("[%s] Training complete. Best Dice: %.4f", model_name, best_dice)
        logger.info("=" * 60)
    if writer:
        writer.close()
    if rank == 0:
        shutdown_nvml()
    cleanup_ddp()


if __name__ == "__main__":
    main()
