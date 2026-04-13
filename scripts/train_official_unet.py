"""
Train official picai_baseline U-Net on PI-CAI data with human expert labels.

This wraps the picai_baseline UNet training pipeline, generating the required
data overview JSON files and calling the official training loop.

Usage:
    python train_official_unet.py --fold 0
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PICAI_ROOT = Path("/workspace/P1-MAR/data")
IMAGES_DIR = PICAI_ROOT / "images"
LABELS_DIR = PICAI_ROOT / "labels" / "csPCa_lesion_delineations" / "human_expert" / "merged"
OVERVIEWS_DIR = Path("/workspace/P1-MAR/outputs/official_unet/overviews")
WEIGHTS_DIR = Path("/workspace/P1-MAR/outputs/official_unet/weights")

# Official picai_baseline U-Net expects preprocessed images at specific spacing
# Default: 3.0 x 0.5 x 0.5 mm, shape 20 x 256 x 256
TARGET_SPACING = [0.5, 0.5, 3.0]  # (x, y, z) SimpleITK convention
TARGET_SHAPE = [20, 256, 256]  # (z, y, x)


def get_fold_split(fold: int = 0):
    """Load official PI-CAI fold split."""
    try:
        from picai_baseline.splits.picai import valid_splits, train_splits
        return {
            "train": train_splits[fold]["subject_list"],
            "val": valid_splits[fold]["subject_list"],
        }
    except ImportError:
        split_path = Path(__file__).parent / "configs" / "fold0_split.json"
        with open(split_path) as f:
            return json.load(f)


def find_study_dirs():
    """Map study_id -> patient_dir for all studies."""
    studies = {}
    for patient_dir in sorted(IMAGES_DIR.iterdir()):
        if not patient_dir.is_dir():
            continue
        for f in patient_dir.glob("*.mha"):
            parts = f.stem.rsplit("_", 1)
            if len(parts) == 2 and parts[1] in ("t2w", "hbv", "adc"):
                prefix = parts[0]
                if prefix not in studies:
                    studies[prefix] = patient_dir
    return studies


def resample_to_target(image_path: str, target_spacing, target_shape, is_label=False):
    """Resample and crop/pad image to target spacing and shape."""
    img = sitk.ReadImage(str(image_path))

    # Resample
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

    # Center crop/pad to target shape (z, y, x)
    arr = sitk.GetArrayFromImage(img)  # (z, y, x)
    result = np.zeros(target_shape, dtype=arr.dtype)

    # Compute crop/pad for each axis
    for ax in range(3):
        src_sz = arr.shape[ax]
        tgt_sz = target_shape[ax]
        if src_sz >= tgt_sz:
            start = (src_sz - tgt_sz) // 2
            slc = slice(start, start + tgt_sz)
        else:
            slc = slice(0, src_sz)

        # Build slicing
        src_slices = [slice(None)] * 3
        dst_slices = [slice(None)] * 3
        if src_sz >= tgt_sz:
            src_slices[ax] = slc
        else:
            pad_start = (tgt_sz - src_sz) // 2
            src_slices[ax] = slice(0, src_sz)
            dst_slices[ax] = slice(pad_start, pad_start + src_sz)

    # Simpler approach: just crop/pad
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

    out_img = sitk.GetImageFromArray(arr)
    out_img.SetSpacing(target_spacing)
    return out_img


def prepare_preprocessed_data(fold: int = 0):
    """Preprocess all images and labels for picai_baseline UNet format.

    Creates preprocessed MHA files at target spacing/shape and generates
    the JSON overview files needed by picai_baseline.
    """
    preproc_dir = Path("/workspace/P1-MAR/outputs/official_unet/preprocessed")
    images_preproc = preproc_dir / "images"
    labels_preproc = preproc_dir / "labels"
    images_preproc.mkdir(parents=True, exist_ok=True)
    labels_preproc.mkdir(parents=True, exist_ok=True)
    OVERVIEWS_DIR.mkdir(parents=True, exist_ok=True)

    fold_split = get_fold_split(fold)
    studies = find_study_dirs()

    for split_name, subject_list in [("train", fold_split["train"]),
                                      ("val", fold_split["val"])]:
        image_paths = []
        label_paths = []
        case_labels = []

        for study_id in subject_list:
            if study_id not in studies:
                logger.warning("Study %s not found in images, skipping", study_id)
                continue

            patient_dir = studies[study_id]
            label_path = LABELS_DIR / f"{study_id}.nii.gz"
            if not label_path.is_file():
                continue

            # Check all modalities exist
            mods_ok = all(
                (patient_dir / f"{study_id}_{mod}.mha").is_file()
                for mod in ["t2w", "hbv", "adc"]
            )
            if not mods_ok:
                continue

            # Preprocess images (3-channel stacked MHA)
            stacked_path = images_preproc / f"{study_id}.mha"
            if not stacked_path.exists():
                channels = []
                for mod in ["t2w", "hbv", "adc"]:
                    mod_path = patient_dir / f"{study_id}_{mod}.mha"
                    resampled = resample_to_target(
                        str(mod_path), TARGET_SPACING, TARGET_SHAPE, is_label=False
                    )
                    channels.append(sitk.GetArrayFromImage(resampled))
                stacked = np.stack(channels, axis=0).astype(np.float32)  # (3, z, y, x)
                stacked_img = sitk.GetImageFromArray(stacked)
                stacked_img.SetSpacing(TARGET_SPACING)
                sitk.WriteImage(stacked_img, str(stacked_path))

            # Preprocess label (binarize)
            lbl_out_path = labels_preproc / f"{study_id}.nii.gz"
            if not lbl_out_path.exists():
                lbl_resampled = resample_to_target(
                    str(label_path), TARGET_SPACING, TARGET_SHAPE, is_label=True
                )
                lbl_arr = sitk.GetArrayFromImage(lbl_resampled)
                lbl_binary = (lbl_arr >= 1).astype(np.uint8)
                lbl_img = sitk.GetImageFromArray(lbl_binary)
                lbl_img.SetSpacing(TARGET_SPACING)
                sitk.WriteImage(lbl_img, str(lbl_out_path))

            # Case-level label (1 if any positive voxel)
            lbl_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(lbl_out_path)))
            has_lesion = int(lbl_arr.max() > 0)

            image_paths.append(str(stacked_path))
            label_paths.append(str(lbl_out_path))
            case_labels.append(has_lesion)

        # Save overview JSON
        overview = {
            "image_paths": image_paths,
            "label_paths": label_paths,
            "case_label": case_labels,
        }
        overview_path = OVERVIEWS_DIR / f"PI-CAI_{split_name}-fold-{fold}.json"
        with open(overview_path, "w") as f:
            json.dump(overview, f, indent=2)

        n_pos = sum(case_labels)
        n_neg = len(case_labels) - n_pos
        logger.info("%s fold-%d: %d cases (%d pos, %d neg)",
                    split_name, fold, len(case_labels), n_pos, n_neg)


def main():
    parser = argparse.ArgumentParser(description="Train official picai_baseline U-Net")
    parser.add_argument("--fold", type=int, default=0, help="Fold number")
    parser.add_argument("--epochs", type=int, default=250, help="Number of training epochs")
    parser.add_argument("--preprocess_only", action="store_true",
                        help="Only preprocess data, don't train")
    args = parser.parse_args()

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Preprocess data and generate overview JSONs
    logger.info("Preparing preprocessed data for fold-%d...", args.fold)
    prepare_preprocessed_data(args.fold)

    if args.preprocess_only:
        logger.info("Preprocessing complete. Skipping training.")
        return

    # Step 2: Train using picai_baseline
    logger.info("Starting official U-Net training...")
    train_cmd = [
        sys.executable, "-m", "picai_baseline.unet.train",
        "--weights_dir", str(WEIGHTS_DIR),
        "--overviews_dir", str(OVERVIEWS_DIR),
        "--folds", str(args.fold),
        "--num_epochs", str(args.epochs),
        "--batch_size", "8",
        "--image_shape", "20", "256", "256",
        "--num_channels", "3",
        "--num_classes", "2",
        "--base_lr", "0.001",
        "--focal_loss_gamma", "1.0",
        "--model_type", "unet",
        "--enable_da", "1",
    ]
    logger.info("Command: %s", " ".join(train_cmd))
    subprocess.run(train_cmd, check=True)

    logger.info("Official U-Net training complete. Weights saved to: %s", WEIGHTS_DIR)


if __name__ == "__main__":
    main()
