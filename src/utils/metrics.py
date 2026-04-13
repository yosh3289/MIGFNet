"""Evaluation metrics for 3D medical image segmentation."""

import torch
import torch.nn.functional as F
import numpy as np


def dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    """Compute 3D Dice Similarity Coefficient.

    Args:
        pred: predicted binary mask [B, 1, D, H, W] or [B, D, H, W]
        target: ground truth binary mask, same shape
        smooth: smoothing factor to avoid division by zero
    Returns:
        Per-sample Dice score [B]
    """
    if pred.dim() == 5:
        pred = pred[:, 1] if pred.shape[1] > 1 else pred[:, 0]
    if target.dim() == 5:
        target = target[:, 0]

    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    return (2.0 * intersection + smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)


def sensitivity(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    """Compute sensitivity (recall / true positive rate).

    Args:
        pred: predicted binary mask [B, D, H, W] or [B, 1, D, H, W]
        target: ground truth, same shape
    Returns:
        Per-sample sensitivity [B]
    """
    if pred.dim() == 5:
        pred = pred[:, 1] if pred.shape[1] > 1 else pred[:, 0]
    if target.dim() == 5:
        target = target[:, 0]

    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)

    tp = (pred_flat * target_flat).sum(dim=1)
    fn = (target_flat * (1 - pred_flat)).sum(dim=1)
    return (tp + smooth) / (tp + fn + smooth)


def specificity(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    """Compute specificity (true negative rate).

    Args:
        pred: predicted binary mask [B, D, H, W] or [B, 1, D, H, W]
        target: ground truth, same shape
    Returns:
        Per-sample specificity [B]
    """
    if pred.dim() == 5:
        pred = pred[:, 1] if pred.shape[1] > 1 else pred[:, 0]
    if target.dim() == 5:
        target = target[:, 0]

    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)

    tn = ((1 - pred_flat) * (1 - target_flat)).sum(dim=1)
    fp = (pred_flat * (1 - target_flat)).sum(dim=1)
    return (tn + smooth) / (tn + fp + smooth)


class MetricTracker:
    """Accumulates predictions and computes metrics over a full dataset.

    Tracks positive cases (has lesion) and negative cases separately.
    Dice and Sensitivity are computed on positive cases only (standard in
    medical image segmentation). Case-level detection metrics are also reported.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.dice_pos = []       # Dice on positive cases only
        self.sens_pos = []       # Sensitivity on positive cases only
        self.spec_scores = []    # Specificity on all cases
        self.has_lesion = []     # Whether GT has any positive voxels
        self.pred_any = []       # Whether prediction has any positive voxels

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """Update with a batch of predictions.

        Args:
            logits: [B, num_classes, D, H, W] raw model output
            targets: [B, 1, D, H, W] binary ground truth
        """
        if logits.shape[1] > 1:
            pred = (logits.argmax(dim=1, keepdim=False)).float()
        else:
            pred = (torch.sigmoid(logits[:, 0]) > 0.5).float()

        target = targets[:, 0].float()

        pred_flat = pred.reshape(pred.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)
        has_lesion = target_flat.sum(dim=1) > 0
        pred_any = pred_flat.sum(dim=1) > 0

        self.has_lesion.append(has_lesion.cpu())
        self.pred_any.append(pred_any.cpu())

        d = dice_score(pred, target).cpu()
        s = sensitivity(pred, target).cpu()
        sp = specificity(pred, target).cpu()

        if has_lesion.any():
            self.dice_pos.append(d[has_lesion.cpu()])
            self.sens_pos.append(s[has_lesion.cpu()])

        self.spec_scores.append(sp)

    def compute(self) -> dict[str, float]:
        """Compute metrics. Dice/Sensitivity on positive cases only."""
        all_has = torch.cat(self.has_lesion)
        all_pred = torch.cat(self.pred_any)
        all_spec = torch.cat(self.spec_scores)
        n_pos = all_has.sum().item()
        n_neg = (~all_has).sum().item()
        n_total = len(all_has)

        # Case-level detection
        tp_case = (all_has & all_pred).sum().item()
        fn_case = (all_has & ~all_pred).sum().item()
        fp_case = (~all_has & all_pred).sum().item()
        tn_case = (~all_has & ~all_pred).sum().item()
        case_sens = tp_case / max(tp_case + fn_case, 1)
        case_spec = tn_case / max(tn_case + fp_case, 1)

        if self.dice_pos:
            all_dice_pos = torch.cat(self.dice_pos)
            all_sens_pos = torch.cat(self.sens_pos)
            dice_val = all_dice_pos.mean().item()
            dice_std = all_dice_pos.std().item() if len(all_dice_pos) > 1 else 0.0
            sens_val = all_sens_pos.mean().item()
            sens_std = all_sens_pos.std().item() if len(all_sens_pos) > 1 else 0.0
        else:
            dice_val = dice_std = sens_val = sens_std = 0.0

        return {
            "dice": dice_val,
            "dice_std": dice_std,
            "sensitivity": sens_val,
            "sensitivity_std": sens_std,
            "specificity": all_spec.mean().item(),
            "specificity_std": all_spec.std().item() if len(all_spec) > 1 else 0.0,
            "num_samples": n_total,
            "num_positive": int(n_pos),
            "num_negative": int(n_neg),
            "case_sensitivity": case_sens,
            "case_specificity": case_spec,
        }
