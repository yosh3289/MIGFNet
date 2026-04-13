"""
Evaluation script for Adaptive-MPNet with PI-CAI official metrics + robustness testing.

Primary metrics (via picai_eval): AUROC, AP, Ranking Score
Secondary metrics: Dice, Sensitivity, Specificity

Tests across scenarios:
  1. Ideal: Full T2W + HBV + ADC
  2. Missing modality: Zero-out HBV or ADC
  3. Artifact: Inject severe noise into HBV or ADC

Usage:
    python evaluate.py --config configs/adaptive_mpnet.yaml --checkpoint outputs/checkpoints/best_model.pth
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from src.models.adaptive_mpnet import build_adaptive_mpnet
from src.models.adaptive_native import build_adaptive_native
from src.models.baselines import build_baseline
from src.data.dataset import PICIADataset, CenterCrop3D, Compose
from src.utils.metrics import MetricTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def apply_test_scenario(batch: dict, scenario: dict) -> dict:
    """Apply a test scenario to a batch of data.

    Supports:
    - drop_modalities: list of modality keys to zero-out
    - artifact_modalities: list of modality keys to add noise to
    - artifact_noise_std: noise standard deviation
    """
    batch = {k: v.clone() if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}

    # Zero-out modalities
    for mod in scenario.get("drop_modalities", []):
        if mod in batch:
            batch[mod] = torch.zeros_like(batch[mod])

    # Add artifact noise
    for mod in scenario.get("artifact_modalities", []):
        if mod in batch:
            noise_std = scenario.get("artifact_noise_std", 0.5)
            noise = torch.randn_like(batch[mod]) * noise_std
            batch[mod] = batch[mod] + noise

    return batch


@torch.no_grad()
def evaluate_scenario(model, dataloader, scenario, device):
    """Evaluate model on a single test scenario.

    Returns both traditional metrics (via MetricTracker) and per-case
    softmax predictions + ground truth labels for picai_eval.
    """
    model.eval()
    tracker = MetricTracker()

    # Collect per-case predictions and labels for picai_eval
    all_y_det = []  # softmax detection maps (numpy arrays)
    all_y_true = []  # binary ground truth labels (numpy arrays)
    all_subject_ids = []

    for batch in dataloader:
        batch = apply_test_scenario(batch, scenario)

        t2w = batch["t2w"].to(device, non_blocking=True)
        hbv = batch["hbv"].to(device, non_blocking=True)
        adc = batch["adc"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        patient_ids = batch["patient_id"]

        with autocast():
            outputs = model([t2w, hbv, adc])
            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            if logits.shape[2:] != label.shape[2:]:
                logits = F.interpolate(logits, size=label.shape[2:],
                                       mode='trilinear', align_corners=False)

        tracker.update(logits.float(), label)

        # Extract softmax class-1 probabilities as detection maps
        if logits.shape[1] > 1:
            softmax_maps = torch.softmax(logits.float(), dim=1)[:, 1]  # [B, D, H, W]
        else:
            softmax_maps = torch.sigmoid(logits.float()[:, 0])  # [B, D, H, W]

        label_np = label[:, 0].cpu().numpy()  # [B, D, H, W]

        for i in range(softmax_maps.shape[0]):
            all_y_det.append(softmax_maps[i].cpu().numpy().astype(np.float32))
            all_y_true.append((label_np[i] >= 1).astype(np.int32))
            all_subject_ids.append(patient_ids[i])

    # Compute traditional metrics
    traditional_metrics = tracker.compute()

    # Compute PI-CAI official metrics using picai_eval
    picai_metrics = compute_picai_metrics(all_y_det, all_y_true, all_subject_ids)

    # Merge all metrics
    return {**traditional_metrics, **picai_metrics}


def compute_picai_metrics(y_det, y_true, subject_list=None):
    """Compute PI-CAI official evaluation metrics.

    Converts softmax volumes to detection maps (lesion candidates) before
    passing to picai_eval, as required by the official evaluation protocol.

    Args:
        y_det: list of numpy arrays, softmax probability maps per case
        y_true: list of numpy arrays, binary ground truth labels per case
        subject_list: optional list of subject IDs

    Returns:
        dict with AUROC, AP, ranking_score
    """
    try:
        from picai_eval import evaluate
        from report_guided_annotation import extract_lesion_candidates

        # Convert softmax volumes to detection maps (lesion candidates)
        detection_maps = []
        for softmax_vol in y_det:
            det_map = extract_lesion_candidates(softmax_vol)[0]
            detection_maps.append(det_map)

        metrics = evaluate(
            y_det=detection_maps,
            y_true=y_true,
            subject_list=subject_list,
        )

        return {
            "auroc": float(metrics.auroc),
            "ap": float(metrics.AP),
            "ranking_score": float(metrics.score),
        }
    except Exception as e:
        logger.warning("picai_eval failed: %s. Returning NaN for official metrics.", e)
        return {
            "auroc": float("nan"),
            "ap": float("nan"),
            "ranking_score": float("nan"),
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Adaptive-MPNet")
    parser.add_argument("--config", type=str, default="configs/adaptive_mpnet.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="AdaptiveMPNet",
                        help="Model name for results table")
    parser.add_argument("--output", type=str, default="outputs/eval_results.json")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load config
    config_path = Path(__file__).parent / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Build model and load checkpoint
    ADAPTIVE_NATIVE = ["adaptive_nnunet", "adaptive_unet", "adaptive_conv3d", "adaptive_swinunetr"]
    if args.model_name == "AdaptiveMPNet":
        model = build_adaptive_mpnet(config).to(device)
    elif args.model_name in ADAPTIVE_NATIVE:
        model = build_adaptive_native(args.model_name, config).to(device)
    else:
        model = build_baseline(args.model_name, config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info("Loaded checkpoint from %s", args.checkpoint)

    # Build test dataloader
    data_cfg = config.get("data", {})
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

    # Apply center crop if patch_crop_size is configured
    patch_crop_cfg = data_cfg.get("patch_crop_size", None)
    test_transform = None
    if patch_crop_cfg:
        patch_crop = list(patch_crop_cfg)
        # SwinUNETR requires all spatial dims >= 32 (divisible by 2^5)
        if args.model_name == "SwinUNETR" and patch_crop[2] < 32:
            patch_crop[2] = 32
            logger.info("SwinUNETR: overriding patch D to 32")
        patch_crop_dhw = (patch_crop[2], patch_crop[0], patch_crop[1])
        test_transform = Compose([CenterCrop3D(crop_size=patch_crop_dhw)])

    # For fold-based splits, val and test are the same set
    eval_split = "val" if data_config.get("split_source") == "picai_pub" else "test"
    test_ds = PICIADataset(data_config["root_dir"], eval_split, data_config,
                           transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"],
                             shuffle=False, num_workers=4, pin_memory=True)
    logger.info("Eval set (%s): %d samples", eval_split, len(test_ds))

    # Evaluate across all 7 scenarios
    scenarios = config.get("evaluation", {}).get("test_scenarios", [
        {"name": "ideal", "drop_modalities": []},
        {"name": "missing_t2w", "drop_modalities": ["t2w"]},
        {"name": "missing_hbv", "drop_modalities": ["hbv"]},
        {"name": "missing_adc", "drop_modalities": ["adc"]},
        {"name": "artifact_t2w", "artifact_modalities": ["t2w"], "artifact_noise_std": 0.5},
        {"name": "artifact_hbv", "artifact_modalities": ["hbv"], "artifact_noise_std": 0.5},
        {"name": "artifact_adc", "artifact_modalities": ["adc"], "artifact_noise_std": 0.5},
    ])

    param_count = sum(p.numel() for p in model.parameters())
    results = {"model": args.model_name, "params_M": round(param_count / 1e6, 2), "scenarios": {}}

    for scenario in scenarios:
        name = scenario["name"]
        logger.info("Evaluating scenario: %s", name)
        metrics = evaluate_scenario(model, test_loader, scenario, device)
        results["scenarios"][name] = metrics
        logger.info(
            "  %s: AUROC=%.4f | AP=%.4f | Score=%.4f | "
            "Dice=%.4f | Sens=%.4f | Spec=%.4f | Pos=%d Neg=%d",
            name, metrics.get("auroc", 0), metrics.get("ap", 0),
            metrics.get("ranking_score", 0),
            metrics["dice"], metrics["sensitivity"], metrics["specificity"],
            metrics["num_positive"], metrics["num_negative"],
        )

    # Print comparison table
    print("\n" + "=" * 100)
    print(f"Model: {args.model_name}")
    first_scenario = scenarios[0]["name"]
    print(f"  Eval set: {results['scenarios'][first_scenario]['num_positive']} positive, "
          f"{results['scenarios'][first_scenario]['num_negative']} negative cases")
    print("-" * 100)
    print(f"{'Scenario':<16} {'AUROC':>7} {'AP':>7} {'Score':>7} "
          f"{'Dice(+)':>8} {'Sens(+)':>8} {'Spec':>8} "
          f"{'CaseSens':>9} {'CaseSpec':>9}")
    print("-" * 100)
    for name, m in results["scenarios"].items():
        print(f"{name:<16} {m.get('auroc', 0):>7.4f} {m.get('ap', 0):>7.4f} "
              f"{m.get('ranking_score', 0):>7.4f} "
              f"{m['dice']:>8.4f} {m['sensitivity']:>8.4f} "
              f"{m['specificity']:>8.4f} {m['case_sensitivity']:>9.4f} "
              f"{m['case_specificity']:>9.4f}")
    print("=" * 100)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
