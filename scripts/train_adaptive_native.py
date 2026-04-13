"""
Training script for adaptive native models (nnUNet/UNet/Conv3D/SwinUNETR + 3 modules).

Reuses the same training infrastructure as train.py (DDP, AMP, early stopping, etc.)
but builds models from adaptive_native.py or adaptive_mpnet.py (for conv3d).

Usage:
    torchrun --nproc_per_node=4 train_adaptive_native.py --model adaptive_nnunet
    torchrun --nproc_per_node=4 train_adaptive_native.py --model adaptive_unet
    torchrun --nproc_per_node=4 train_adaptive_native.py --model adaptive_conv3d
    torchrun --nproc_per_node=4 train_adaptive_native.py --model adaptive_swinunetr
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml
import torch

sys.path.insert(0, str(Path(__file__).parent))

# Reuse everything from train.py
from train import (
    setup_ddp, cleanup_ddp, setup_logging,
    build_dataloaders, build_optimizer, build_scheduler,
    train_one_epoch, validate, save_checkpoint, _get_lr,
    set_seed, GLOBAL_SEED,
)
from src.models.adaptive_native import build_adaptive_native
from src.models.adaptive_mpnet import build_adaptive_mpnet
from src.utils.losses import DiceFocalTverskyLoss, DeepSupervisionWrapper
from src.utils.gpu_monitor import init_nvml, shutdown_nvml, format_all_gpu_stats

import time
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

ALL_MODELS = ["adaptive_nnunet", "adaptive_unet", "adaptive_conv3d", "adaptive_swinunetr"]


def build_model(name: str, config: dict):
    if name == "adaptive_conv3d":
        # Use AdaptiveMPNet with conv3d backbone
        config_copy = {**config}
        config_copy["model"] = {**config["model"], "backbone": "conv3d"}
        return build_adaptive_mpnet(config_copy)
    else:
        return build_adaptive_native(name, config)


def main():
    parser = argparse.ArgumentParser(description="Train Adaptive Native Models")
    parser.add_argument("--model", type=str, required=True, choices=ALL_MODELS)
    parser.add_argument("--config", type=str, default="configs/adaptive_mpnet.yaml")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED,
                        help="Global random seed (default: 42)")
    parser.add_argument("--output_prefix", type=str, default=None,
                        help="Optional prefix for checkpoint/log dirs (e.g. 'ablation_nnunet/A1')")
    args = parser.parse_args()

    model_name = args.model
    seed = args.seed

    # Setup DDP
    rank, local_rank, world_size = setup_ddp()
    setup_logging(rank)
    set_seed(seed, rank)
    device = torch.device(f"cuda:{local_rank}")
    nvml_ok = init_nvml() if rank == 0 else False

    # Load config
    config_path = Path(__file__).parent / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    train_cfg = config["training"]

    # SwinUNETR needs D >= 32 for patch_crop_size
    if model_name == "adaptive_swinunetr":
        pcs = config["data"].get("patch_crop_size", [64, 64, 16])
        if pcs[2] < 32:
            config["data"]["patch_crop_size"] = [pcs[0], pcs[1], 32]
            if rank == 0:
                logger.info("SwinUNETR: overriding patch_crop D to 32")

    # Build model
    model = build_model(model_name, config).to(device)
    if world_size > 1:
        find_unused = not train_cfg.get("deep_supervision", True)
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=find_unused)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Output dirs (seed-aware, with optional prefix)
    seed_tag = f"seed{seed}"
    base_ckpt = Path(train_cfg.get("checkpoint_dir", "outputs/checkpoints"))
    base_log = Path(train_cfg.get("log_dir", "outputs/logs"))
    if args.output_prefix:
        base_ckpt = base_ckpt / args.output_prefix
        base_log = base_log / args.output_prefix
    ckpt_dir = base_ckpt / seed_tag / model_name
    log_dir = base_log / seed_tag / model_name
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        logger.info("=" * 60)
        logger.info("Training: %s", model_name)
        logger.info("=" * 60)
        logger.info("Parameters: %.2fM", param_count / 1e6)
        logger.info("GPUs: %d | Batch/GPU: %d", world_size, train_cfg.get("batch_size", 8))

    # Build data
    train_loader, val_loader, train_sampler = build_dataloaders(config, rank, world_size)
    if rank == 0:
        logger.info("Train: %d batches, Val: %d batches", len(train_loader), len(val_loader))

    # Loss
    base_loss = DiceFocalTverskyLoss(
        dice_weight=train_cfg.get("dice_weight", 1.0),
        focal_weight=train_cfg.get("focal_weight", 1.0),
        tversky_weight=train_cfg.get("tversky_weight", 0.0),
        focal_alpha=train_cfg.get("focal_alpha", 0.9),
        focal_gamma=train_cfg.get("focal_gamma", 2.0),
        tversky_alpha=train_cfg.get("tversky_alpha", 0.3),
        tversky_beta=train_cfg.get("tversky_beta", 0.7),
    )
    if train_cfg.get("deep_supervision", True):
        criterion = DeepSupervisionWrapper(base_loss)
    else:
        criterion = base_loss

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, len(train_loader))
    scaler = GradScaler(enabled=train_cfg.get("mixed_precision", True))
    writer = SummaryWriter(str(log_dir)) if rank == 0 else None

    # Training loop
    num_epochs = train_cfg.get("num_epochs", 300)
    global_step = 0
    best_dice = 0.0
    is_plateau = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    early_stop_patience = train_cfg.get("early_stopping_patience", 0)
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
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
            val_metrics = validate(model, val_loader, criterion, device, config, rank)

            if dist.is_initialized():
                dist.barrier()

            warmup_epochs = train_cfg.get("warmup_epochs", 10)
            if is_plateau and epoch >= warmup_epochs:
                scheduler.step(val_metrics["dice"])

            if rank == 0:
                logger.info(
                    "[%s] Epoch %d/%d | Train: %.4f | Val: %.4f | "
                    "Dice: %.4f | Sens: %.4f | Spec: %.4f | LR: %.2e | %.1fs",
                    model_name, epoch + 1, num_epochs, train_loss, val_metrics["loss"],
                    val_metrics["dice"], val_metrics["sensitivity"],
                    val_metrics["specificity"],
                    _get_lr(optimizer, scheduler), train_time,
                )

                if val_metrics["dice"] > best_dice:
                    best_dice = val_metrics["dice"]
                    epochs_without_improvement = 0
                    save_checkpoint(model, optimizer, scheduler, scaler,
                                    epoch, val_metrics, config,
                                    ckpt_dir / "best_model.pth")
                    logger.info(">>> [%s] New best Dice: %.4f", model_name, best_dice)
                else:
                    epochs_without_improvement += 5

                if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
                    logger.info("[%s] Early stopping at epoch %d. Best Dice: %.4f",
                                model_name, epoch + 1, best_dice)
                    break
        else:
            if rank == 0:
                logger.info("[%s] Epoch %d/%d | Train Loss: %.4f | %.1fs",
                            model_name, epoch + 1, num_epochs, train_loss, train_time)

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
