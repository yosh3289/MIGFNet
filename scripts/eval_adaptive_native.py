"""
Evaluation script for adaptive native models.
Same 7-scenario evaluation as evaluate.py but for adaptive native models.
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
from src.models.adaptive_native import build_adaptive_native
from src.models.adaptive_mpnet import build_adaptive_mpnet
from src.data.dataset import PICIADataset, CenterCrop3D, Compose
from src.utils.metrics import MetricTracker
from evaluate import apply_test_scenario, evaluate_scenario

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ALL_MODELS = ["adaptive_nnunet", "adaptive_unet", "adaptive_conv3d", "adaptive_swinunetr"]


def build_model(name: str, config: dict):
    if name == "adaptive_conv3d":
        config_copy = {**config}
        config_copy["model"] = {**config["model"], "backbone": "conv3d"}
        return build_adaptive_mpnet(config_copy)
    else:
        return build_adaptive_native(name, config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=ALL_MODELS)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/adaptive_mpnet.yaml")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config_path = Path(__file__).parent / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_name = args.model

    # SwinUNETR needs D >= 32
    if model_name == "adaptive_swinunetr":
        pcs = config["data"].get("patch_crop_size", [64, 64, 16])
        if pcs[2] < 32:
            config["data"]["patch_crop_size"] = [pcs[0], pcs[1], 32]

    model = build_model(model_name, config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    param_count = sum(p.numel() for p in model.parameters())
    logger.info("Loaded %s (%.2fM params) from %s", model_name, param_count / 1e6, args.checkpoint)

    # Build test dataloader
    data_cfg = config.get("data", {})
    data_config = {
        "root_dir": data_cfg.get("dataset_root", "/workspace/P1-MAR/data"),
        "target_spacing": data_cfg.get("target_spacing", [0.5, 0.5, 3.0]),
        "crop_size": data_cfg.get("crop_size", [128, 128, 32]),
        "split_seed": data_cfg.get("seed", 42),
        "lesion_labels_dir": data_cfg.get("lesion_labels_dir",
                                          "labels/csPCa_lesion_delineations/human_expert/merged"),
        "split_source": data_cfg.get("split_source", "picai_pub"),
        "fold": data_cfg.get("fold", 0),
    }

    patch_crop_cfg = data_cfg.get("patch_crop_size", None)
    test_transform = None
    if patch_crop_cfg:
        patch_crop = list(patch_crop_cfg)
        patch_crop_dhw = (patch_crop[2], patch_crop[0], patch_crop[1])
        test_transform = Compose([CenterCrop3D(crop_size=patch_crop_dhw)])

    eval_split = "val" if data_config.get("split_source") == "picai_pub" else "test"
    test_ds = PICIADataset(data_config["root_dir"], eval_split, data_config,
                           transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"],
                             shuffle=False, num_workers=4, pin_memory=True)
    logger.info("Eval set (%s): %d samples", eval_split, len(test_ds))

    # 7 scenarios
    scenarios = [
        {"name": "ideal", "drop_modalities": []},
        {"name": "missing_t2w", "drop_modalities": ["t2w"]},
        {"name": "missing_hbv", "drop_modalities": ["hbv"]},
        {"name": "missing_adc", "drop_modalities": ["adc"]},
        {"name": "artifact_t2w", "artifact_modalities": ["t2w"], "artifact_noise_std": 0.5},
        {"name": "artifact_hbv", "artifact_modalities": ["hbv"], "artifact_noise_std": 0.5},
        {"name": "artifact_adc", "artifact_modalities": ["adc"], "artifact_noise_std": 0.5},
    ]

    results = {"model": model_name, "params_M": round(param_count / 1e6, 2), "scenarios": {}}

    for scenario in scenarios:
        name = scenario["name"]
        logger.info("Evaluating scenario: %s", name)
        metrics = evaluate_scenario(model, test_loader, scenario, device)
        results["scenarios"][name] = metrics
        logger.info(
            "  %s: AUROC=%.4f | AP=%.4f | Score=%.4f | "
            "Dice=%.4f | Sens=%.4f | Spec=%.4f | CaseSens=%.4f | CaseSpec=%.4f",
            name, metrics.get("auroc", 0), metrics.get("ap", 0),
            metrics.get("ranking_score", 0),
            metrics["dice"], metrics["sensitivity"], metrics["specificity"],
            metrics["case_sensitivity"], metrics["case_specificity"],
        )

    # Print table
    print("\n" + "=" * 130)
    print(f"Model: {model_name} ({param_count/1e6:.2f}M params)")
    print("-" * 130)
    print(f"{'Scenario':<16} {'AUROC':>7} {'AP':>7} {'Score':>7} "
          f"{'Dice(+)':>8} {'Sens(+)':>8} {'Spec':>8} "
          f"{'CaseSens':>9} {'CaseSpec':>9}")
    print("-" * 130)
    for name, m in results["scenarios"].items():
        print(f"{name:<16} {m.get('auroc', 0):>7.4f} {m.get('ap', 0):>7.4f} "
              f"{m.get('ranking_score', 0):>7.4f} "
              f"{m['dice']:>8.4f} {m['sensitivity']:>8.4f} "
              f"{m['specificity']:>8.4f} {m['case_sensitivity']:>9.4f} "
              f"{m['case_specificity']:>9.4f}")
    print("=" * 130)

    # Save
    output_path = Path(args.output or f"outputs/results/eval_{model_name}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
