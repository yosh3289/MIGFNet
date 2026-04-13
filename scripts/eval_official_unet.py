"""
Evaluate Official U-Net with on-the-fly preprocessing from source MHA files.

Preprocessed data was deleted for disk space. This script streams source images,
resamples to target spacing (0.5, 0.5, 3.0), crops/pads to (20, 256, 256),
Z-score normalizes, runs inference, and evaluates with picai_eval.

Tests across 5 scenarios (matching evaluate.py):
  1. ideal: Full T2W + HBV + ADC
  2. missing_hbv: Zero-out HBV channel
  3. missing_adc: Zero-out ADC channel
  4. artifact_hbv: Add noise (std=0.5) to HBV channel
  5. artifact_adc: Add noise (std=0.5) to ADC channel

Usage:
    CUDA_VISIBLE_DEVICES=1 python eval_official_unet.py
"""

import json
import logging
import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from monai.networks.nets import UNet
from torch.amp import autocast

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

IMAGES_DIR = Path("/workspace/P1-MAR/data/images")
GT_DIR = Path("/workspace/P1-MAR/data/labels/csPCa_lesion_delineations/human_expert/merged")
SPLIT_PATH = Path("/workspace/P1-MAR/configs/fold0_split.json")
WEIGHTS_PATH = Path("/workspace/P1-MAR/outputs/official_unet/weights/unet_F0.pt")
OUTPUT_PATH = Path("/workspace/P1-MAR/outputs/results/eval_Official_UNet.json")

TARGET_SPACING = [0.5, 0.5, 3.0]   # (x, y, z) SimpleITK convention
TARGET_SHAPE = [20, 256, 256]       # (z, y, x)

# Channel indices: 0=T2W, 1=HBV, 2=ADC
SCENARIOS = [
    {"name": "ideal"},
    {"name": "missing_hbv", "zero_channels": [1]},
    {"name": "missing_adc", "zero_channels": [2]},
    {"name": "artifact_hbv", "noise_channels": [1], "noise_std": 0.5},
    {"name": "artifact_adc", "noise_channels": [2], "noise_std": 0.5},
]


def resample_to_target(image_path: str, target_spacing, target_shape, is_label=False):
    """Resample and center-crop/pad image to target spacing and shape."""
    img = sitk.ReadImage(str(image_path))

    orig_spacing = np.array(img.GetSpacing())
    orig_size = np.array(img.GetSize())
    new_size = np.round(orig_size * orig_spacing / np.array(target_spacing)).astype(int).tolist()

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetOutputPixelType(sitk.sitkUInt8)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)
    img = resampler.Execute(img)

    arr = sitk.GetArrayFromImage(img)
    for ax in range(3):
        src_sz = arr.shape[ax]
        tgt_sz = target_shape[ax]
        if src_sz > tgt_sz:
            start = (src_sz - tgt_sz) // 2
            idx = [slice(None)] * 3
            idx[ax] = slice(start, start + tgt_sz)
            arr = arr[tuple(idx)]
        elif src_sz < tgt_sz:
            pad_before = (tgt_sz - src_sz) // 2
            pad_after = tgt_sz - src_sz - pad_before
            pad_widths = [(0, 0)] * 3
            pad_widths[ax] = (pad_before, pad_after)
            arr = np.pad(arr, pad_widths, mode='constant', constant_values=0)

    return arr


def preprocess_case(study_id: str, patient_dir: Path):
    """Load and preprocess a single case from source MHA files.

    Returns:
        image: numpy array (3, 20, 256, 256), Z-score normalized per channel
        label: numpy array (20, 256, 256), binary
    """
    channels = []
    for mod in ["t2w", "hbv", "adc"]:
        mod_path = patient_dir / f"{study_id}_{mod}.mha"
        arr = resample_to_target(str(mod_path), TARGET_SPACING, TARGET_SHAPE, is_label=False)
        channels.append(arr)

    image = np.stack(channels, axis=0).astype(np.float32)

    for c in range(image.shape[0]):
        ch = image[c]
        mean = ch.mean()
        std = ch.std()
        if std > 1e-8:
            image[c] = (ch - mean) / std

    gt_path = GT_DIR / f"{study_id}.nii.gz"
    label = resample_to_target(str(gt_path), TARGET_SPACING, TARGET_SHAPE, is_label=True)
    label = (label >= 1).astype(np.int64)

    return image, label


def apply_scenario(image: np.ndarray, scenario: dict, rng: np.random.Generator) -> np.ndarray:
    """Apply a test scenario to an image. Returns a modified copy."""
    img = image.copy()

    for ch in scenario.get("zero_channels", []):
        img[ch] = 0.0

    for ch in scenario.get("noise_channels", []):
        noise_std = scenario.get("noise_std", 0.5)
        img[ch] = img[ch] + rng.normal(0, noise_std, size=img[ch].shape).astype(np.float32)

    return img


def find_study_dirs():
    """Map study_id -> patient_dir for all available studies."""
    studies = {}
    for patient_dir in sorted(IMAGES_DIR.iterdir()):
        if not patient_dir.is_dir():
            continue
        for f in patient_dir.glob("*_t2w.mha"):
            study_id = f.stem.rsplit("_", 1)[0]
            studies[study_id] = patient_dir
    return studies


def compute_voxel_metrics(pred: np.ndarray, gt: np.ndarray):
    """Compute voxel-level Dice, sensitivity, specificity."""
    pred_flat = pred.astype(bool).ravel()
    gt_flat = gt.astype(bool).ravel()

    tp = (pred_flat & gt_flat).sum()
    fp = (pred_flat & ~gt_flat).sum()
    fn = (~pred_flat & gt_flat).sum()
    tn = (~pred_flat & ~gt_flat).sum()

    dice = float((2 * tp) / (2 * tp + fp + fn + 1e-8))
    sensitivity = float(tp / (tp + fn + 1e-8))
    specificity = float(tn / (tn + fp + 1e-8))

    return dice, sensitivity, specificity


def compute_picai_metrics(y_det, y_true, subject_list):
    """Compute PI-CAI official evaluation metrics."""
    try:
        from picai_eval import evaluate
        from report_guided_annotation import extract_lesion_candidates

        detection_maps = []
        for softmax_vol in y_det:
            det_map = extract_lesion_candidates(softmax_vol)[0]
            detection_maps.append(det_map)

        metrics_obj = evaluate(
            y_det=detection_maps,
            y_true=y_true,
            subject_list=subject_list,
        )
        return {
            "auroc": float(metrics_obj.auroc),
            "ap": float(metrics_obj.AP),
            "ranking_score": float(metrics_obj.score),
        }
    except Exception as e:
        logger.warning("picai_eval failed: %s", e)
        return {"auroc": float("nan"), "ap": float("nan"), "ranking_score": float("nan")}


def evaluate_scenario(model, images, labels, study_ids, scenario, device):
    """Evaluate model on a single test scenario."""
    rng = np.random.default_rng(42)

    dice_pos = []
    sens_pos = []
    spec_all = []
    has_lesion_list = []
    pred_any_list = []
    y_det = []
    y_true = []

    with torch.no_grad():
        for image, label, study_id in zip(images, labels, study_ids):
            # Apply scenario modification
            img_modified = apply_scenario(image, scenario, rng)

            # Inference
            img_tensor = torch.from_numpy(img_modified).unsqueeze(0).to(device)
            with autocast(device_type="cuda"):
                logits = model(img_tensor)

            softmax = torch.softmax(logits.float(), dim=1)[0, 1]
            softmax_np = softmax.cpu().numpy().astype(np.float32)
            pred_binary = (softmax_np > 0.5).astype(np.uint8)

            has_lesion = bool(label.max() > 0)
            pred_has_any = bool(pred_binary.max() > 0)
            has_lesion_list.append(has_lesion)
            pred_any_list.append(pred_has_any)

            dice, sens, spec = compute_voxel_metrics(pred_binary, label)
            spec_all.append(spec)
            if has_lesion:
                dice_pos.append(dice)
                sens_pos.append(sens)

            y_det.append(softmax_np)
            y_true.append(label.astype(np.int32))

    # Aggregate
    n_pos = sum(has_lesion_list)
    n_neg = len(has_lesion_list) - n_pos

    # Case-level detection
    has_arr = np.array(has_lesion_list, dtype=bool)
    pred_arr = np.array(pred_any_list, dtype=bool)
    tp_case = int((has_arr & pred_arr).sum())
    fn_case = int((has_arr & ~pred_arr).sum())
    fp_case = int((~has_arr & pred_arr).sum())
    tn_case = int((~has_arr & ~pred_arr).sum())

    # picai_eval
    picai = compute_picai_metrics(y_det, y_true, study_ids)

    return {
        "dice": float(np.mean(dice_pos)) if dice_pos else 0.0,
        "dice_std": float(np.std(dice_pos)) if len(dice_pos) > 1 else 0.0,
        "sensitivity": float(np.mean(sens_pos)) if sens_pos else 0.0,
        "sensitivity_std": float(np.std(sens_pos)) if len(sens_pos) > 1 else 0.0,
        "specificity": float(np.mean(spec_all)) if spec_all else 0.0,
        "specificity_std": float(np.std(spec_all)) if len(spec_all) > 1 else 0.0,
        "num_samples": len(study_ids),
        "num_positive": int(n_pos),
        "num_negative": int(n_neg),
        "case_sensitivity": float(tp_case / max(tp_case + fn_case, 1)),
        "case_specificity": float(tn_case / max(tn_case + fp_case, 1)),
        **picai,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Official U-Net")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Build model
    model = UNet(
        spatial_dims=3,
        in_channels=3,
        out_channels=2,
        channels=(32, 64, 128, 256, 512, 1024),
        strides=((2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)),
        num_res_units=0,
    ).to(device)

    ckpt = torch.load(str(WEIGHTS_PATH), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Loaded checkpoint from %s (epoch %d)", WEIGHTS_PATH, ckpt.get("epoch", -1))

    # Load fold-0 val split
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    val_ids = split["val"]
    logger.info("Fold-0 val split: %d cases", len(val_ids))

    studies = find_study_dirs()
    logger.info("Found %d studies in image directory", len(studies))

    # === Phase 1: Preprocess all cases once ===
    logger.info("Preprocessing all validation cases...")
    all_images = []
    all_labels = []
    valid_ids = []

    for i, study_id in enumerate(val_ids):
        if study_id not in studies:
            continue
        patient_dir = studies[study_id]
        if not all(
            (patient_dir / f"{study_id}_{mod}.mha").is_file()
            for mod in ["t2w", "hbv", "adc"]
        ):
            continue
        if not (GT_DIR / f"{study_id}.nii.gz").exists():
            continue

        image, label = preprocess_case(study_id, patient_dir)
        all_images.append(image)
        all_labels.append(label)
        valid_ids.append(study_id)

        if (i + 1) % 50 == 0:
            logger.info("  Preprocessed %d/%d", i + 1, len(val_ids))

    logger.info("Preprocessed %d cases (skipped %d)", len(valid_ids), len(val_ids) - len(valid_ids))

    # === Phase 2: Evaluate all scenarios ===
    results = {"model": "Official_UNet", "scenarios": {}}

    for scenario in SCENARIOS:
        name = scenario["name"]
        logger.info("Evaluating scenario: %s", name)
        metrics = evaluate_scenario(model, all_images, all_labels, valid_ids, scenario, device)
        results["scenarios"][name] = metrics
        logger.info(
            "  %s: AUROC=%.4f | AP=%.4f | Score=%.4f | "
            "Dice=%.4f | Sens=%.4f | Spec=%.4f | CaseSens=%.4f | CaseSpec=%.4f",
            name, metrics["auroc"], metrics["ap"], metrics["ranking_score"],
            metrics["dice"], metrics["sensitivity"], metrics["specificity"],
            metrics["case_sensitivity"], metrics["case_specificity"],
        )

    # Print comparison table
    print("\n" + "=" * 100)
    print("Model: Official U-Net")
    m0 = results["scenarios"]["ideal"]
    print(f"  Eval set: {m0['num_positive']} positive, {m0['num_negative']} negative cases")
    print("-" * 100)
    print(f"{'Scenario':<16} {'AUROC':>7} {'AP':>7} {'Score':>7} "
          f"{'Dice(+)':>8} {'Sens(+)':>8} {'Spec':>8} "
          f"{'CaseSens':>9} {'CaseSpec':>9}")
    print("-" * 100)
    for name, m in results["scenarios"].items():
        print(f"{name:<16} {m['auroc']:>7.4f} {m['ap']:>7.4f} "
              f"{m['ranking_score']:>7.4f} "
              f"{m['dice']:>8.4f} {m['sensitivity']:>8.4f} "
              f"{m['specificity']:>8.4f} {m['case_sensitivity']:>9.4f} "
              f"{m['case_specificity']:>9.4f}")
    print("=" * 100)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
