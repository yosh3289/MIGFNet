"""
Prepare nnU-Net v2 raw data format for PI-CAI official baseline training.

This script:
1. Converts PI-CAI MHA images to nnU-Net raw data format
2. Binarizes human expert labels (ISUP grades -> binary csPCa)
3. Creates dataset.json for nnU-Net
4. Writes fold-0 split from picai_baseline

The resulting dataset can be trained with:
    nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
    nnUNetv2_train DATASET_ID 3d_fullres 0 -num_gpus 4 -tr nnUNetTrainerFocalDiceLoss

Usage:
    python prepare_nnunet.py [--dataset_id 100]
"""

import argparse
import json
import logging
import os
import shutil
from collections import OrderedDict
from pathlib import Path

import numpy as np
import SimpleITK as sitk

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# nnU-Net environment variables
NNUNET_RAW = Path(os.environ.get("nnUNet_raw", "/workspace/P1-MAR/nnunet_data/nnUNet_raw"))
NNUNET_PREPROCESSED = Path(os.environ.get("nnUNet_preprocessed",
                                           "/workspace/P1-MAR/nnunet_data/nnUNet_preprocessed"))
NNUNET_RESULTS = Path(os.environ.get("nnUNet_results",
                                      "/workspace/P1-MAR/nnunet_data/nnUNet_results"))

# PI-CAI data paths
PICAI_ROOT = Path("/workspace/P1-MAR/data")
IMAGES_DIR = PICAI_ROOT / "images"
LABELS_DIR = PICAI_ROOT / "labels" / "csPCa_lesion_delineations" / "human_expert" / "merged"


def setup_env_vars():
    """Set nnU-Net environment variables if not already set."""
    os.environ.setdefault("nnUNet_raw", str(NNUNET_RAW))
    os.environ.setdefault("nnUNet_preprocessed", str(NNUNET_PREPROCESSED))
    os.environ.setdefault("nnUNet_results", str(NNUNET_RESULTS))

    for d in [NNUNET_RAW, NNUNET_PREPROCESSED, NNUNET_RESULTS]:
        d.mkdir(parents=True, exist_ok=True)


def get_fold0_split():
    """Load official PI-CAI fold-0 split."""
    try:
        from picai_baseline.splits.picai import valid_splits, train_splits
        return {
            "train": train_splits[0]["subject_list"],
            "val": valid_splits[0]["subject_list"],
        }
    except ImportError:
        split_path = Path(__file__).parent / "configs" / "fold0_split.json"
        with open(split_path) as f:
            return json.load(f)


def find_all_studies(images_dir: Path) -> dict:
    """Find all study prefixes and their patient directories."""
    studies = {}
    for patient_dir in sorted(images_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        for f in patient_dir.glob("*.mha"):
            parts = f.stem.rsplit("_", 1)
            if len(parts) == 2 and parts[1] in ("t2w", "hbv", "adc"):
                prefix = parts[0]
                if prefix not in studies:
                    studies[prefix] = patient_dir
    return studies


def resample_to_reference(image: sitk.Image, reference: sitk.Image,
                          is_label: bool = False) -> sitk.Image:
    """Resample image to match reference geometry (spacing, size, origin, direction)."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetTransform(sitk.Transform())
    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0.0)
    return resampler.Execute(image)


def binarize_label(label_path: Path, output_path: Path,
                   reference: sitk.Image = None):
    """Read label, binarize (ISUP >= 1 -> 1), optionally resample to reference, save."""
    img = sitk.ReadImage(str(label_path))
    if reference is not None:
        img = resample_to_reference(img, reference, is_label=True)
    arr = sitk.GetArrayFromImage(img)
    binary = (arr >= 1).astype(np.uint8)
    out_img = sitk.GetImageFromArray(binary)
    out_img.CopyInformation(img if reference is None else reference)
    sitk.WriteImage(out_img, str(output_path))


def prepare_dataset(dataset_id: int = 100):
    """Convert PI-CAI data to nnU-Net v2 raw format."""
    setup_env_vars()

    dataset_name = f"Dataset{dataset_id:03d}_PICAI"
    dataset_dir = NNUNET_RAW / dataset_name
    imagesTr = dataset_dir / "imagesTr"
    labelsTr = dataset_dir / "labelsTr"

    for d in [imagesTr, labelsTr]:
        d.mkdir(parents=True, exist_ok=True)

    # Get fold-0 split
    fold_split = get_fold0_split()
    train_subjects = set(fold_split["train"])
    val_subjects = set(fold_split["val"])
    all_subjects = train_subjects | val_subjects

    # Find all studies
    studies = find_all_studies(IMAGES_DIR)
    logger.info("Found %d studies in images directory", len(studies))

    # nnU-Net channel mapping: 0=T2W, 1=HBV (high b-value DWI), 2=ADC
    modality_map = {"t2w": "0000", "hbv": "0001", "adc": "0002"}

    processed = 0
    skipped = 0

    for study_id, patient_dir in sorted(studies.items()):
        if study_id not in all_subjects:
            skipped += 1
            continue

        # Check all modalities exist
        has_all = True
        for mod in ["t2w", "hbv", "adc"]:
            if not (patient_dir / f"{study_id}_{mod}.mha").is_file():
                has_all = False
                break

        label_path = LABELS_DIR / f"{study_id}.nii.gz"
        if not label_path.is_file():
            has_all = False

        if not has_all:
            logger.warning("Missing files for %s, skipping", study_id)
            skipped += 1
            continue

        # All cases go in imagesTr — nnU-Net uses splits_final.json for train/val
        img_dir = imagesTr
        lbl_dir = labelsTr

        # Resample all modalities to T2W geometry (reference)
        case_id = study_id.replace("_", "")  # nnU-Net doesn't like underscores
        t2w_path = patient_dir / f"{study_id}_t2w.mha"
        t2w_img = sitk.ReadImage(str(t2w_path))

        for mod, ch_id in modality_map.items():
            src = patient_dir / f"{study_id}_{mod}.mha"
            dst = img_dir / f"{case_id}_{ch_id}.nii.gz"
            if not dst.exists():
                mod_img = sitk.ReadImage(str(src))
                if mod != "t2w":
                    mod_img = resample_to_reference(mod_img, t2w_img, is_label=False)
                sitk.WriteImage(mod_img, str(dst))

        # Binarize and resample label to T2W geometry
        lbl_dst = lbl_dir / f"{case_id}.nii.gz"
        if not lbl_dst.exists():
            binarize_label(label_path, lbl_dst, reference=t2w_img)

        processed += 1
        if processed % 100 == 0:
            logger.info("Processed %d studies...", processed)

    logger.info("Processed: %d, Skipped: %d", processed, skipped)

    # Create dataset.json
    dataset_json = {
        "channel_names": {
            "0": "T2W",
            "1": "HBV",
            "2": "ADC",
        },
        "labels": {
            "background": 0,
            "csPCa": 1,
        },
        "numTraining": len([f for f in imagesTr.glob("*_0000.nii.gz")]),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }

    with open(dataset_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)
    logger.info("dataset.json written: %d training cases", dataset_json["numTraining"])

    # Write fold-0 split for nnU-Net v2
    # nnU-Net v2 expects splits_final.json in preprocessed dir
    splits_dir = NNUNET_PREPROCESSED / dataset_name
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_ids = sorted([s.replace("_", "") for s in fold_split["train"]
                        if s in studies])
    val_ids = sorted([s.replace("_", "") for s in fold_split["val"]
                      if s in studies])

    splits = [{"train": train_ids, "val": val_ids}]  # fold 0 only

    with open(splits_dir / "splits_final.json", "w") as f:
        json.dump(splits, f, indent=2)
    logger.info("splits_final.json written: %d train, %d val",
                len(train_ids), len(val_ids))

    # Write env setup script
    env_script = dataset_dir / "setup_env.sh"
    with open(env_script, "w") as f:
        f.write(f"""#!/bin/bash
# nnU-Net v2 environment setup for PI-CAI
export nnUNet_raw="{NNUNET_RAW}"
export nnUNet_preprocessed="{NNUNET_PREPROCESSED}"
export nnUNet_results="{NNUNET_RESULTS}"

echo "nnU-Net v2 environment configured"
echo "  Raw: $nnUNet_raw"
echo "  Preprocessed: $nnUNet_preprocessed"
echo "  Results: $nnUNet_results"
echo ""
echo "Next steps:"
echo "  1. Plan and preprocess:"
echo "     nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity"
echo "  2. Train fold 0 (4 GPUs):"
echo "     nnUNetv2_train {dataset_id} 3d_fullres 0 -num_gpus 4"
echo "  3. Inference on validation set:"
echo "     nnUNetv2_predict -i {NNUNET_RAW}/{dataset_name}/imagesTs -o {NNUNET_RESULTS}/predictions -d {dataset_id} -c 3d_fullres -f 0"
""")
    os.chmod(env_script, 0o755)
    logger.info("Environment setup script: %s", env_script)

    return dataset_dir


def main():
    parser = argparse.ArgumentParser(description="Prepare nnU-Net v2 data for PI-CAI")
    parser.add_argument("--dataset_id", type=int, default=100,
                        help="nnU-Net dataset ID (default: 100)")
    args = parser.parse_args()

    dataset_dir = prepare_dataset(args.dataset_id)
    logger.info("nnU-Net v2 dataset prepared at: %s", dataset_dir)


if __name__ == "__main__":
    main()
