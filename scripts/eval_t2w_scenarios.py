"""
Evaluate missing_t2w and artifact_t2w scenarios for all MAR models.
Appends results to existing eval JSON files.
"""
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
from src.models.baselines import build_baseline
from src.data.dataset import PICIADataset, CenterCrop3D, Compose
from src.utils.metrics import MetricTracker
from evaluate import apply_test_scenario, evaluate_scenario

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

T2W_SCENARIOS = [
    {"name": "missing_t2w", "drop_modalities": ["t2w"]},
    {"name": "artifact_t2w", "artifact_modalities": ["t2w"], "artifact_noise_std": 0.5},
]

# MAR models: (display_name, model_name_for_build, checkpoint_path, eval_json_path)
MAR_MODELS = [
    ("AdaptiveMPNet", "AdaptiveMPNet", "outputs/checkpoints/AdaptiveMPNet/best_model.pth",
     "outputs/results/eval_AdaptiveMPNet.json"),
    ("OfficialUNet", "OfficialUNet", "outputs/checkpoints/OfficialUNet/best_model.pth",
     "outputs/results/eval_OfficialUNet_unified.json"),
    ("UMamba_concat", "UMamba_concat", "outputs/checkpoints/UMamba_concat/best_model.pth",
     "outputs/results/eval_UMamba_concat.json"),
    ("nnUNet_concat", "nnUNet_concat", "outputs/checkpoints/nnUNet_concat/best_model.pth",
     "outputs/results/eval_nnUNet_concat_unified.json"),
    ("SwinUNETR", "SwinUNETR", "outputs/checkpoints/SwinUNETR/best_model.pth",
     "outputs/results/eval_SwinUNETR.json"),
]


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config_path = Path(__file__).parent / "configs" / "adaptive_mpnet.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    for display_name, model_name, ckpt_path, eval_json_path in MAR_MODELS:
        ckpt_full = Path(__file__).parent / ckpt_path
        eval_full = Path(__file__).parent / eval_json_path

        if not ckpt_full.exists():
            logger.warning("No checkpoint for %s at %s, skipping", display_name, ckpt_full)
            continue
        if not eval_full.exists():
            logger.warning("No eval JSON for %s at %s, skipping", display_name, eval_full)
            continue

        logger.info("=" * 60)
        logger.info("Evaluating T2W scenarios: %s", display_name)
        logger.info("=" * 60)

        # Build model
        if model_name == "AdaptiveMPNet":
            model = build_adaptive_mpnet(config).to(device)
        else:
            model = build_baseline(model_name, config).to(device)

        ckpt = torch.load(str(ckpt_full), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        # Build dataloader
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
            if model_name == "SwinUNETR" and patch_crop[2] < 32:
                patch_crop[2] = 32
            patch_crop_dhw = (patch_crop[2], patch_crop[0], patch_crop[1])
            test_transform = Compose([CenterCrop3D(crop_size=patch_crop_dhw)])

        eval_split = "val" if data_config.get("split_source") == "picai_pub" else "test"
        test_ds = PICIADataset(data_config["root_dir"], eval_split, data_config,
                               transform=test_transform)
        test_loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"],
                                 shuffle=False, num_workers=4, pin_memory=True)

        # Load existing results
        with open(eval_full) as f:
            results = json.load(f)

        for scenario in T2W_SCENARIOS:
            name = scenario["name"]
            logger.info("  Scenario: %s", name)
            metrics = evaluate_scenario(model, test_loader, scenario, device)
            results["scenarios"][name] = metrics
            logger.info(
                "  %s: AUROC=%.4f | AP=%.4f | Score=%.4f | "
                "Dice=%.4f | CaseSens=%.4f | CaseSpec=%.4f",
                name, metrics.get("auroc", 0), metrics.get("ap", 0),
                metrics.get("ranking_score", 0),
                metrics["dice"], metrics["case_sensitivity"], metrics["case_specificity"],
            )

        # Save updated results
        with open(eval_full, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Updated %s", eval_full)

        del model
        torch.cuda.empty_cache()

    # Print summary
    print("\n" + "=" * 120)
    print("T2W SCENARIO RESULTS — MAR Models")
    print("=" * 120)
    scenarios = ["ideal", "missing_t2w", "missing_hbv", "missing_adc",
                 "artifact_t2w", "artifact_hbv", "artifact_adc"]
    header = f"{'Model':<18}"
    for s in scenarios:
        header += f" {s:>14}"
    print(header)
    print("-" * 120)
    for display_name, _, _, eval_json_path in MAR_MODELS:
        ep = Path(__file__).parent / eval_json_path
        if not ep.exists():
            continue
        with open(ep) as f:
            r = json.load(f)
        line = f"{display_name:<18}"
        for s in scenarios:
            sc = r["scenarios"].get(s, {}).get("ranking_score", 0)
            line += f" {sc:>14.4f}"
        print(line)
    print("=" * 120)


if __name__ == "__main__":
    main()
