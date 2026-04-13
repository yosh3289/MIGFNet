"""Loss functions for medical image segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss for binary/multi-class segmentation (foreground only)."""

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        target_int = targets[:, 0].long()
        one_hot = F.one_hot(target_int, num_classes).permute(0, 4, 1, 2, 3).float()
        dims = (0, 2, 3, 4)
        intersection = (probs * one_hot).sum(dim=dims)
        cardinality = (probs + one_hot).sum(dim=dims)
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_per_class[1:].mean()


class FocalLoss(nn.Module):
    """Focal Loss for class-imbalanced segmentation.

    Reduces loss for well-classified (easy) examples, focusing training on
    hard negatives. FL(p) = -alpha * (1-p)^gamma * log(p)
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        target_long = targets[:, 0].long()
        ce = F.cross_entropy(logits, target_long, reduction='none')
        probs = F.softmax(logits, dim=1)
        # Gather probability of the correct class
        pt = probs.gather(1, target_long.unsqueeze(1)).squeeze(1)
        # Alpha weighting: alpha for foreground, (1-alpha) for background
        alpha_t = torch.where(target_long == 1, self.alpha, 1.0 - self.alpha)
        focal_weight = alpha_t * (1.0 - pt) ** self.gamma
        return (focal_weight * ce).mean()


class TverskyLoss(nn.Module):
    """Tversky Loss — generalizes Dice with separate FP/FN penalties.

    When alpha > beta, penalizes false negatives more heavily (good for
    detecting small lesions). alpha=beta=0.5 is equivalent to Dice.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-5):
        super().__init__()
        self.alpha = alpha  # FP weight (lower = tolerate more FP)
        self.beta = beta    # FN weight (higher = penalize missed lesions more)
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)[:, 1]  # foreground probability
        target = targets[:, 0].float()
        dims = (1, 2, 3)  # spatial dims, keep batch
        tp = (probs * target).sum(dim=dims)
        fp = (probs * (1.0 - target)).sum(dim=dims)
        fn = ((1.0 - probs) * target).sum(dim=dims)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky.mean()


class DiceFocalTverskyLoss(nn.Module):
    """Combined Dice + Focal + Tversky loss for severely imbalanced segmentation.

    Dice: overlap-based, scale-invariant for foreground
    Focal: down-weights easy background voxels
    Tversky: penalizes false negatives more than false positives
    """

    def __init__(self, dice_weight: float = 1.0, focal_weight: float = 1.0,
                 tversky_weight: float = 1.0, focal_alpha: float = 0.75,
                 focal_gamma: float = 2.0, tversky_alpha: float = 0.3,
                 tversky_beta: float = 0.7):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.tversky_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        if self.dice_weight > 0:
            loss = loss + self.dice_weight * self.dice_loss(logits, targets)
        if self.focal_weight > 0:
            loss = loss + self.focal_weight * self.focal_loss(logits, targets)
        if self.tversky_weight > 0:
            loss = loss + self.tversky_weight * self.tversky_loss(logits, targets)
        return loss


class DeepSupervisionWrapper(nn.Module):
    """Wraps a loss function to apply at multiple decoder scales.

    Expects model to return (final_logits, [aux_logits_1, aux_logits_2, ...])
    when training. Weights decrease for deeper (lower-resolution) outputs.
    """

    def __init__(self, loss_fn: nn.Module, weights: list[float] | None = None):
        super().__init__()
        self.loss_fn = loss_fn
        self.weights = weights  # set dynamically if None

    def forward(self, outputs, targets: torch.Tensor) -> torch.Tensor:
        if isinstance(outputs, (tuple, list)):
            main_logits = outputs[0]
            aux_logits_list = outputs[1] if len(outputs) > 1 else []
        else:
            return self.loss_fn(outputs, targets)

        # Main loss at full resolution
        loss = self.loss_fn(main_logits, targets)

        if not aux_logits_list:
            return loss

        # Weights: halve for each deeper level
        weights = self.weights or [0.5 ** (i + 1) for i in range(len(aux_logits_list))]

        for w, aux in zip(weights, aux_logits_list):
            # Resize targets to match auxiliary output resolution
            if aux.shape[2:] != targets.shape[2:]:
                t_down = F.interpolate(
                    targets.float(), size=aux.shape[2:],
                    mode='nearest'
                )
            else:
                t_down = targets
            loss = loss + w * self.loss_fn(aux, t_down)

        return loss


# Legacy compatibility
class DiceCELoss(nn.Module):
    """Combined Dice + Cross-Entropy loss with optional class weighting."""

    def __init__(self, dice_weight: float = 1.0, ce_weight: float = 1.0,
                 smooth: float = 1e-5,
                 ce_class_weight: list[float] | None = None):
        super().__init__()
        self.dice_loss = DiceLoss(smooth=smooth)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        weight = torch.tensor(ce_class_weight, dtype=torch.float32) if ce_class_weight else None
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(logits, targets)
        if self.ce_loss.weight is not None and self.ce_loss.weight.device != logits.device:
            self.ce_loss.weight = self.ce_loss.weight.to(logits.device)
        ce = self.ce_loss(logits, targets[:, 0].long())
        return self.dice_weight * dice + self.ce_weight * ce
