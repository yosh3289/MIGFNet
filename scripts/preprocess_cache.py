"""
Offline preprocessing: preprocess all PI-CAI cases and save as .pt files.

This eliminates the ~80s/batch on-the-fly SimpleITK bottleneck by caching
preprocessed tensors (T2W, HBV, ADC, label) as PyTorch files.

Usage:
    python preprocess_cache.py [--workers 8]
"""

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from src.data.dataset import preprocess_case, PICIADataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def process_one_case(case_info: dict, target_spacing, crop_size, cache_dir: str) -> str:
    """Preprocess and save a single case. Returns study_id or error string."""
    study_id = case_info.get("study_id", case_info["patient_id"])
    out_path = Path(cache_dir) / f"{study_id}.pt"

    if out_path.exists():
        return f"SKIP {study_id}"

    try:
        data = preprocess_case(
            patient_dir=case_info["patient_dir"],
            label_path=case_info["label_path"],
            gland_path=case_info["gland_path"],
            target_spacing=target_spacing,
            crop_size=crop_size,
        )
        tensors = {
            "t2w": torch.from_numpy(data["t2w"]).to(torch.float16),
            "hbv": torch.from_numpy(data["hbv"]).to(torch.float16),
            "adc": torch.from_numpy(data["adc"]).to(torch.float16),
            "label": torch.from_numpy(data["label"]).to(torch.uint8),
            "study_id": study_id,
            "patient_id": case_info["patient_id"],
        }
        torch.save(tensors, out_path)
        return f"OK {study_id}"
    except Exception as e:
        return f"FAIL {study_id}: {e}"


def main():
    parser = argparse.ArgumentParser(description="Preprocess PI-CAI dataset to cache")
    parser.add_argument("--config", type=str, default="configs/adaptive_mpnet.yaml")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    config_path = Path(__file__).parent / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_cfg = config.get("data", {})
    target_spacing = data_cfg.get("target_spacing", [0.5, 0.5, 3.0])
    crop_size = data_cfg.get("crop_size", [128, 128, 32])
    root_dir = data_cfg.get("dataset_root", "/workspace/P1-MAR/data")

    cache_dir = Path(root_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Discover all cases using the dataset's discovery logic
    lesion_labels_dir = data_cfg.get("lesion_labels_dir",
                                     "labels/csPCa_lesion_delineations/human_expert/merged")
    ds = PICIADataset(root_dir, "train", {
        "root_dir": root_dir,
        "target_spacing": target_spacing,
        "crop_size": crop_size,
        "split_seed": 42,
        "train_ratio": 1.0,  # Discover all
        "val_ratio": 0.0,
        "test_ratio": 0.0,
        "lesion_labels_dir": lesion_labels_dir,
    })
    all_cases = ds.all_cases
    logger.info("Total cases to preprocess: %d", len(all_cases))
    logger.info("Cache directory: %s", cache_dir)

    # Check how many already cached
    existing = sum(1 for c in all_cases if (cache_dir / f"{c['patient_id']}.pt").exists())
    if existing > 0:
        logger.info("Already cached: %d/%d", existing, len(all_cases))

    t0 = time.time()
    ok, skip, fail = 0, 0, 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_one_case, case, target_spacing, crop_size, str(cache_dir)): case
            for case in all_cases
        }
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result.startswith("OK"):
                ok += 1
            elif result.startswith("SKIP"):
                skip += 1
            else:
                fail += 1
                logger.warning(result)

            if i % 50 == 0 or i == len(all_cases):
                elapsed = time.time() - t0
                rate = i / elapsed
                eta = (len(all_cases) - i) / rate if rate > 0 else 0
                logger.info(
                    "Progress: %d/%d (%.1f/s) | OK=%d SKIP=%d FAIL=%d | ETA: %.0fs",
                    i, len(all_cases), rate, ok, skip, fail, eta,
                )

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("Preprocessing complete in %.1fs", elapsed)
    logger.info("OK: %d | SKIP: %d | FAIL: %d", ok, skip, fail)

    # Estimate cache size
    cache_files = list(cache_dir.glob("*.pt"))
    total_size = sum(f.stat().st_size for f in cache_files)
    logger.info("Cache size: %.1f MB (%d files)", total_size / 1e6, len(cache_files))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
