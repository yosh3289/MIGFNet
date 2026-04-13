"""
Evaluate nnU-Net v2 with robustness testing.

For the ideal scenario, loads pre-existing binary validation predictions.
For robustness scenarios (missing/artifact modalities), loads the nnU-Net model
and runs inference with modified inputs using nnU-Net's predictor API.

Usage:
    CUDA_VISIBLE_DEVICES=0 python eval_nnunet.py
"""

import json
import logging
import argparse
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PRED_DIR = Path(
    "/workspace/P1-MAR/nnunet_data/nnUNet_results/Dataset100_PICAI/"
    "nnUNetTrainerFocalDiceLoss__nnUNetPlans__3d_fullres/fold_0/validation"
)
MODEL_DIR = Path(
    "/workspace/P1-MAR/nnunet_data/nnUNet_results/Dataset100_PICAI/"
    "nnUNetTrainerFocalDiceLoss__nnUNetPlans__3d_fullres"
)
IMAGES_DIR = Path("/workspace/P1-MAR/data/images")
GT_DIR = Path("/workspace/P1-MAR/data/labels/csPCa_lesion_delineations/human_expert/merged")
SPLIT_PATH = Path("/workspace/P1-MAR/configs/fold0_split.json")
OUTPUT_PATH = Path("/workspace/P1-MAR/outputs/results/eval_nnUNet_v2.json")

# Channel indices: 0=T2W, 1=HBV, 2=ADC
SCENARIOS = [
    {"name": "ideal"},
    {"name": "missing_hbv", "zero_channels": [1]},
    {"name": "missing_adc", "zero_channels": [2]},
    {"name": "artifact_hbv", "noise_channels": [1], "noise_std": 0.5},
    {"name": "artifact_adc", "noise_channels": [2], "noise_std": 0.5},
]


def study_id_to_nnunet_filename(study_id: str) -> str:
    """Convert study_id like '10000_1000000' to nnU-Net filename '100001000000'."""
    return study_id.replace("_", "")


def resample_to_reference(image, reference, is_label=True):
    """Resample image to match reference image's physical space."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(image)


def compute_voxel_metrics(pred, gt):
    """Compute voxel-level Dice, sensitivity, specificity."""
    pred_flat = pred.astype(bool).ravel()
    gt_flat = (gt >= 1).ravel()

    tp = int((pred_flat & gt_flat).sum())
    fp = int((pred_flat & ~gt_flat).sum())
    fn = int((~pred_flat & gt_flat).sum())
    tn = int((~pred_flat & ~gt_flat).sum())

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
        for vol in y_det:
            det_map = extract_lesion_candidates(vol)[0]
            detection_maps.append(det_map)

        metrics = evaluate(y_det=detection_maps, y_true=y_true, subject_list=subject_list)
        return {
            "auroc": float(metrics.auroc),
            "ap": float(metrics.AP),
            "ranking_score": float(metrics.score),
        }
    except Exception as e:
        logger.warning("picai_eval failed: %s", e)
        return {"auroc": float("nan"), "ap": float("nan"), "ranking_score": float("nan")}


def aggregate_metrics(dice_pos, sens_pos, spec_all, has_lesion_list, pred_any_list,
                      y_det, y_true, subject_list):
    """Aggregate per-case metrics into scenario results."""
    n_pos = sum(has_lesion_list)
    n_neg = len(has_lesion_list) - n_pos

    has_arr = np.array(has_lesion_list, dtype=bool)
    pred_arr = np.array(pred_any_list, dtype=bool)
    tp_case = int((has_arr & pred_arr).sum())
    fn_case = int((has_arr & ~pred_arr).sum())
    fp_case = int((~has_arr & pred_arr).sum())
    tn_case = int((~has_arr & ~pred_arr).sum())

    picai = compute_picai_metrics(y_det, y_true, subject_list)

    return {
        "dice": float(np.mean(dice_pos)) if dice_pos else 0.0,
        "dice_std": float(np.std(dice_pos)) if len(dice_pos) > 1 else 0.0,
        "sensitivity": float(np.mean(sens_pos)) if sens_pos else 0.0,
        "sensitivity_std": float(np.std(sens_pos)) if len(sens_pos) > 1 else 0.0,
        "specificity": float(np.mean(spec_all)) if spec_all else 0.0,
        "specificity_std": float(np.std(spec_all)) if len(spec_all) > 1 else 0.0,
        "num_samples": len(subject_list),
        "num_positive": int(n_pos),
        "num_negative": int(n_neg),
        "case_sensitivity": float(tp_case / max(tp_case + fn_case, 1)),
        "case_specificity": float(tn_case / max(tn_case + fp_case, 1)),
        **picai,
    }


def eval_ideal(val_ids):
    """Evaluate ideal scenario from pre-existing predictions."""
    logger.info("Evaluating ideal scenario from saved predictions...")

    dice_pos, sens_pos, spec_all = [], [], []
    has_lesion_list, pred_any_list = [], []
    y_det, y_true, subject_list = [], [], []

    for i, study_id in enumerate(val_ids):
        nn_fn = study_id_to_nnunet_filename(study_id) + ".nii.gz"
        pred_path = PRED_DIR / nn_fn
        gt_path = GT_DIR / f"{study_id}.nii.gz"

        if not pred_path.exists() or not gt_path.exists():
            continue

        pred_img = sitk.ReadImage(str(pred_path))
        gt_img = sitk.ReadImage(str(gt_path))
        gt_resampled = resample_to_reference(gt_img, pred_img, is_label=True)

        gt_arr = (sitk.GetArrayFromImage(gt_resampled) >= 1).astype(np.uint8)
        pred_arr = sitk.GetArrayFromImage(pred_img).astype(np.uint8)

        has_lesion = bool(gt_arr.max() > 0)
        pred_has_any = bool(pred_arr.max() > 0)
        has_lesion_list.append(has_lesion)
        pred_any_list.append(pred_has_any)

        dice, sens, spec = compute_voxel_metrics(pred_arr, gt_arr)
        spec_all.append(spec)
        if has_lesion:
            dice_pos.append(dice)
            sens_pos.append(sens)

        y_det.append(pred_arr.astype(np.float32))
        y_true.append(gt_arr.astype(np.int32))
        subject_list.append(study_id)

        if (i + 1) % 50 == 0:
            logger.info("  Processed %d/%d", i + 1, len(val_ids))

    return aggregate_metrics(dice_pos, sens_pos, spec_all, has_lesion_list,
                             pred_any_list, y_det, y_true, subject_list)


def eval_robustness(val_ids, scenario, device_id=0):
    """Evaluate a robustness scenario by re-running nnU-Net inference with modified inputs."""
    os.environ['nnUNet_results'] = '/workspace/P1-MAR/nnunet_data/nnUNet_results'

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    import torch

    # Initialize predictor (fast: no mirroring, larger step size)
    predictor = nnUNetPredictor(
        tile_step_size=0.75,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device(f'cuda:{device_id}'),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(MODEL_DIR),
        use_folds=(0,),
        checkpoint_name='checkpoint_best.pth',
    )

    rng = np.random.default_rng(42)

    # Load plans for preprocessing properties
    with open(MODEL_DIR / "plans.json") as f:
        plans = json.load(f)
    spacing = plans["configurations"]["3d_fullres"]["spacing"]  # [3.0, 0.5, 0.5]

    dice_pos, sens_pos, spec_all = [], [], []
    has_lesion_list, pred_any_list = [], []
    y_det, y_true, subject_list = [], [], []

    # Map study_id -> patient_dir
    study_dirs = {}
    for patient_dir in sorted(IMAGES_DIR.iterdir()):
        if not patient_dir.is_dir():
            continue
        for f in patient_dir.glob("*_t2w.mha"):
            sid = f.stem.rsplit("_", 1)[0]
            study_dirs[sid] = patient_dir

    for i, study_id in enumerate(val_ids):
        if study_id not in study_dirs:
            continue
        patient_dir = study_dirs[study_id]
        gt_path = GT_DIR / f"{study_id}.nii.gz"
        if not gt_path.exists():
            continue

        # Load source images as separate channels
        channels = []
        props_ref = None
        for mod in ["t2w", "hbv", "adc"]:
            mod_path = patient_dir / f"{study_id}_{mod}.mha"
            if not mod_path.exists():
                break
            img = sitk.ReadImage(str(mod_path))
            arr = sitk.GetArrayFromImage(img).astype(np.float32)
            channels.append(arr)
            if props_ref is None:
                props_ref = {
                    "sitk_stuff": {
                        "spacing": img.GetSpacing(),
                        "origin": img.GetOrigin(),
                        "direction": img.GetDirection(),
                    },
                    "shape_before_cropping": arr.shape,
                    "bbox_used_for_cropping": [[0, s] for s in arr.shape],
                    "shape_after_cropping_and_before_resampling": arr.shape,
                    "spacing": list(img.GetSpacing()),
                }
        else:
            # Apply scenario modifications to raw channels
            for ch_idx in scenario.get("zero_channels", []):
                channels[ch_idx] = np.zeros_like(channels[ch_idx])
            for ch_idx in scenario.get("noise_channels", []):
                noise_std = scenario.get("noise_std", 0.5)
                # Apply noise relative to channel statistics
                ch = channels[ch_idx]
                ch_std = ch.std()
                if ch_std > 1e-8:
                    noise = rng.normal(0, noise_std * ch_std, size=ch.shape).astype(np.float32)
                    channels[ch_idx] = ch + noise

            # Stack into multi-channel image: (C, Z, Y, X)
            # All channels should have same shape (from same study), but they might differ
            # Resample all channels to T2W reference shape if needed
            ref_shape = channels[0].shape
            for c_idx in range(1, len(channels)):
                if channels[c_idx].shape != ref_shape:
                    # Use SimpleITK to resample
                    ch_img = sitk.ReadImage(str(patient_dir / f"{study_id}_{['t2w','hbv','adc'][c_idx]}.mha"))
                    ref_img = sitk.ReadImage(str(patient_dir / f"{study_id}_t2w.mha"))
                    resampler = sitk.ResampleImageFilter()
                    resampler.SetReferenceImage(ref_img)
                    resampler.SetInterpolator(sitk.sitkLinear)
                    ch_resampled = resampler.Execute(ch_img)
                    channels[c_idx] = sitk.GetArrayFromImage(ch_resampled).astype(np.float32)
                    if c_idx in scenario.get("zero_channels", []):
                        channels[c_idx] = np.zeros_like(channels[c_idx])
                    elif c_idx in scenario.get("noise_channels", []):
                        ch = channels[c_idx]
                        ch_std = ch.std()
                        if ch_std > 1e-8:
                            noise = rng.normal(0, noise_std * ch_std, size=ch.shape).astype(np.float32)
                            channels[c_idx] = ch + noise

            image = np.stack(channels, axis=0)  # (3, Z, Y, X)

            # Run prediction through nnU-Net (handles preprocessing internally)
            ret = predictor.predict_from_list_of_npy_arrays(
                image_or_list_of_images=[image],
                segs_from_prev_stage_or_list_of_segs_from_prev_stage=[None],
                properties_or_list_of_properties=[props_ref],
                truncated_ofname=[None],
                save_probabilities=False,
            )
            pred_arr = ret[0].astype(np.uint8)  # (Z, Y, X) binary segmentation

            # Load GT and resample to prediction space
            gt_img = sitk.ReadImage(str(gt_path))
            # Create reference image with prediction's properties
            pred_sitk = sitk.GetImageFromArray(pred_arr)
            pred_sitk.SetSpacing(props_ref["sitk_stuff"]["spacing"])
            pred_sitk.SetOrigin(props_ref["sitk_stuff"]["origin"])
            pred_sitk.SetDirection(props_ref["sitk_stuff"]["direction"])

            gt_resampled = resample_to_reference(gt_img, pred_sitk, is_label=True)
            gt_arr = (sitk.GetArrayFromImage(gt_resampled) >= 1).astype(np.uint8)

            # Handle shape mismatch by cropping to min shape
            min_shape = tuple(min(a, b) for a, b in zip(pred_arr.shape, gt_arr.shape))
            pred_arr = pred_arr[:min_shape[0], :min_shape[1], :min_shape[2]]
            gt_arr = gt_arr[:min_shape[0], :min_shape[1], :min_shape[2]]

            has_lesion = bool(gt_arr.max() > 0)
            pred_has_any = bool(pred_arr.max() > 0)
            has_lesion_list.append(has_lesion)
            pred_any_list.append(pred_has_any)

            dice, sens, spec = compute_voxel_metrics(pred_arr, gt_arr)
            spec_all.append(spec)
            if has_lesion:
                dice_pos.append(dice)
                sens_pos.append(sens)

            y_det.append(pred_arr.astype(np.float32))
            y_true.append(gt_arr.astype(np.int32))
            subject_list.append(study_id)

            if (i + 1) % 50 == 0:
                logger.info("  Processed %d/%d", i + 1, len(val_ids))
            continue  # skip the else-block's fallthrough

        # if the for loop broke (missing modality file), skip this case
        continue

    return aggregate_metrics(dice_pos, sens_pos, spec_all, has_lesion_list,
                             pred_any_list, y_det, y_true, subject_list)


def main():
    parser = argparse.ArgumentParser(description="Evaluate nnU-Net v2")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ideal-only", action="store_true",
                        help="Only evaluate ideal scenario (from saved predictions)")
    args = parser.parse_args()

    with open(SPLIT_PATH) as f:
        split = json.load(f)
    val_ids = split["val"]
    logger.info("Fold-0 val split: %d cases", len(val_ids))

    results = {"model": "nnUNet_v2", "scenarios": {}}

    for scenario in SCENARIOS:
        name = scenario["name"]

        if name == "ideal":
            metrics = eval_ideal(val_ids)
        elif args.ideal_only:
            continue
        else:
            logger.info("Evaluating scenario: %s (re-running inference)", name)
            metrics = eval_robustness(val_ids, scenario, device_id=args.gpu)

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
    print("Model: nnU-Net v2")
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

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
