"""
Train official picai_baseline U-Net architecture with DDP multi-GPU support.

Uses MONAI UNet with the same architecture/hyperparameters as picai_baseline:
- Channels: [32, 64, 128, 256, 512, 1024]
- Strides: [(2,2,2), (1,2,2), (1,2,2), (1,2,2), (2,2,2)]
- Focal Loss (gamma=1.0)
- Adam optimizer, LR=0.001, polynomial decay
- Input: 3-channel (T2W, HBV, ADC), shape 20x256x256

Usage:
    torchrun --nproc_per_node=4 train_official_unet_ddp.py --fold 0 --epochs 250
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import UNet
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OVERVIEWS_DIR = Path("/workspace/P1-MAR/outputs/official_unet/overviews")
WEIGHTS_DIR = Path("/workspace/P1-MAR/outputs/official_unet/weights")


class FocalLoss(nn.Module):
    """Focal loss for binary segmentation."""

    def __init__(self, gamma=1.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        # logits: (B, 2, D, H, W), targets: (B, D, H, W) long
        probs = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets.long(), num_classes=2).permute(0, 4, 1, 2, 3).float()

        pt = (probs * targets_oh).sum(dim=1)  # (B, D, H, W)
        focal_weight = (1 - pt) ** self.gamma
        ce = -torch.log(pt.clamp(min=1e-7))
        loss = (focal_weight * ce).mean()
        return loss


class PICIAUNetDataset(Dataset):
    """Dataset for picai_baseline UNet format (preprocessed stacked MHA)."""

    def __init__(self, overview_path: str, augment: bool = False):
        with open(overview_path) as f:
            data = json.load(f)
        self.image_paths = data["image_paths"]
        self.label_paths = data["label_paths"]
        self.case_labels = data["case_label"]
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load stacked image (3, 20, 256, 256)
        img = sitk.ReadImage(self.image_paths[idx])
        img_arr = sitk.GetArrayFromImage(img).astype(np.float32)  # (3, 20, 256, 256)

        # Z-score normalization per channel
        for c in range(img_arr.shape[0]):
            ch = img_arr[c]
            mean = ch.mean()
            std = ch.std()
            if std > 1e-8:
                img_arr[c] = (ch - mean) / std

        # Load label (20, 256, 256)
        lbl = sitk.ReadImage(self.label_paths[idx])
        lbl_arr = sitk.GetArrayFromImage(lbl).astype(np.int64)  # (20, 256, 256)

        # Simple augmentation: random flips
        if self.augment:
            if np.random.random() > 0.5:
                img_arr = np.flip(img_arr, axis=-1).copy()  # horizontal flip
                lbl_arr = np.flip(lbl_arr, axis=-1).copy()
            if np.random.random() > 0.5:
                img_arr = np.flip(img_arr, axis=-2).copy()  # vertical flip
                lbl_arr = np.flip(lbl_arr, axis=-2).copy()

        return {
            "image": torch.from_numpy(img_arr),
            "label": torch.from_numpy(lbl_arr),
            "case_label": self.case_labels[idx],
        }


def setup_ddp():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        torch.cuda.set_device(0)
        return 0, 0, 1


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch,
                    num_epochs, rank):
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}",
                leave=True, dynamic_ncols=True, disable=(rank != 0))

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=True):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

        if rank == 0:
            pbar.set_postfix_str(f"loss={loss.item():.4f} avg={total_loss/num_batches:.4f}")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, loader, criterion, device, rank):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    # Track dice per positive case
    dice_pos = []
    sens_pos = []
    spec_all = []

    for batch in tqdm(loader, desc="  Validating", leave=False, disable=(rank != 0)):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with autocast(enabled=True):
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss += loss.item()
        num_batches += 1

        # Compute metrics
        pred = logits.argmax(dim=1).float()  # (B, D, H, W)
        target = labels.float()

        for i in range(pred.shape[0]):
            p = pred[i].reshape(-1)
            t = target[i].reshape(-1)
            has_lesion = t.sum() > 0

            # Specificity (all cases)
            tn = ((1 - p) * (1 - t)).sum()
            fp = (p * (1 - t)).sum()
            spec = (tn / (tn + fp + 1e-5)).item()
            spec_all.append(spec)

            if has_lesion:
                # Dice
                intersection = (p * t).sum()
                dice = (2 * intersection / (p.sum() + t.sum() + 1e-5)).item()
                dice_pos.append(dice)
                # Sensitivity
                tp = (p * t).sum()
                fn = (t * (1 - p)).sum()
                sens = (tp / (tp + fn + 1e-5)).item()
                sens_pos.append(sens)

    metrics = {
        "loss": total_loss / max(num_batches, 1),
        "dice": np.mean(dice_pos) if dice_pos else 0.0,
        "sensitivity": np.mean(sens_pos) if sens_pos else 0.0,
        "specificity": np.mean(spec_all) if spec_all else 0.0,
        "num_positive": len(dice_pos),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train picai_baseline UNet with DDP")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    rank, local_rank, world_size = setup_ddp()
    if rank == 0:
        logging.basicConfig(level=logging.INFO, force=True)
    device = torch.device(f"cuda:{local_rank}")

    # Build model — official picai_baseline architecture
    model = UNet(
        spatial_dims=3,
        in_channels=3,
        out_channels=2,
        channels=(32, 64, 128, 256, 512, 1024),
        strides=((2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)),
        num_res_units=0,
    ).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        logger.info("=" * 60)
        logger.info("Official picai_baseline U-Net (DDP)")
        logger.info("Parameters: %.2fM", param_count / 1e6)
        logger.info("GPUs: %d | Batch/GPU: %d | Effective batch: %d",
                     world_size, args.batch_size, world_size * args.batch_size)
        logger.info("=" * 60)

    # Data
    train_overview = OVERVIEWS_DIR / f"PI-CAI_train-fold-{args.fold}.json"
    val_overview = OVERVIEWS_DIR / f"PI-CAI_val-fold-{args.fold}.json"

    train_ds = PICIAUNetDataset(str(train_overview), augment=True)
    val_ds = PICIAUNetDataset(str(val_overview), augment=False)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                        rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size,
                                      rank=rank, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=(train_sampler is None),
                               sampler=train_sampler, num_workers=4,
                               pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                             shuffle=False, sampler=val_sampler,
                             num_workers=4, pin_memory=True)

    if rank == 0:
        logger.info("Train: %d cases (%d batches), Val: %d cases (%d batches)",
                     len(train_ds), len(train_loader), len(val_ds), len(val_loader))

    # Loss, optimizer, scheduler
    criterion = FocalLoss(gamma=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)

    # Polynomial LR decay (official: lr * (1 - epoch/num_epochs)^0.95)
    def poly_lr(epoch):
        return max(1e-7 / args.lr, (1 - epoch / args.epochs) ** 0.95)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, poly_lr)
    scaler = GradScaler(enabled=True)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    best_dice = 0.0

    for epoch in range(args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                      scaler, device, epoch, args.epochs, rank)
        scheduler.step()
        elapsed = time.time() - t0

        # Validate every 10 epochs (matching official)
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            val_metrics = validate(model, val_loader, criterion, device, rank)

            if dist.is_initialized():
                dist.barrier()

            if rank == 0:
                lr = optimizer.param_groups[0]['lr']
                logger.info(
                    "Epoch %d/%d | Train: %.4f | Val: %.4f | "
                    "Dice: %.4f | Sens: %.4f | Spec: %.4f | LR: %.2e | %.1fs",
                    epoch + 1, args.epochs, train_loss, val_metrics["loss"],
                    val_metrics["dice"], val_metrics["sensitivity"],
                    val_metrics["specificity"], lr, elapsed)

                if val_metrics["dice"] > best_dice:
                    best_dice = val_metrics["dice"]
                    raw_model = model.module if isinstance(model, DDP) else model
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": raw_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "metrics": val_metrics,
                    }, WEIGHTS_DIR / f"unet_F{args.fold}.pt")
                    logger.info(">>> New best Dice: %.4f — checkpoint saved", best_dice)
        else:
            if rank == 0 and epoch % 25 == 0:
                logger.info("Epoch %d/%d | Train Loss: %.4f | %.1fs",
                            epoch + 1, args.epochs, train_loss, elapsed)

    # Final save
    if rank == 0:
        raw_model = model.module if isinstance(model, DDP) else model
        torch.save({
            "epoch": args.epochs - 1,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, WEIGHTS_DIR / f"unet_F{args.fold}_final.pt")
        logger.info("=" * 60)
        logger.info("Training complete. Best Dice: %.4f", best_dice)
        logger.info("=" * 60)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
