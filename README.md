# MIGFNet: Modality-Isolated Gated Fusion for Robust Multi-Modal Prostate MRI Segmentation

[![arXiv](https://img.shields.io/badge/arXiv-2604.10702-b31b1b.svg)](https://arxiv.org/abs/2604.10702)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Clinically significant prostate cancer (csPCa) segmentation from multi-parametric MRI using per-modality feature isolation, gated fusion, and modality-dropout training. Evaluated on the [PI-CAI](https://pi-cai.grand-challenge.org/) public dataset (1500 studies, fold-0 split: 1200 train / 300 val, human expert labels).

This repository contains the official implementation for *Architecture-Agnostic Modality-Isolated Gated Fusion for Robust Multi-Modal Prostate MRI Segmentation*. It covers the MIGF module, cross-backbone integration, and the full ablation study. The companion repository [P1-picai-backbone-bench](https://github.com/yosh3289/P1-picai-backbone-bench) provides the bare-backbone benchmark that selects the starting architectures for this work.

## Key Results

**Best configuration: MIGFNet-nnUNet A2** (Gating + ModDrop, no DeepSup) -- Score **0.7304**, 9.45M params.

All results are 5-seed means (seeds 42, 123, 456, 789, 1024) over 7 robustness scenarios (ideal + 3 missing + 3 artifact modalities), evaluated on 300 fold-0 validation cases with PI-CAI official metrics.

## Architecture

MIGFNet wraps an existing segmentation backbone with three modules:

1. **Per-modality encoders** -- Each MRI channel (T2W, HBV, ADC) is encoded independently before fusion, isolating modality-specific features.
2. **AdaptiveModalGating (G)** -- A lightweight quality estimator produces per-modality scalar weights for additive fusion. When a modality is missing or degraded, its gate weight drops toward zero.
3. **Modality Dropout (M)** -- During training, entire modality channels are randomly zeroed (p=0.3), forcing the model to learn from any subset of available inputs.
4. **Deep Supervision (D)** -- Auxiliary segmentation heads at intermediate decoder levels (optional; harmful for shallow backbones).

The nnUNet-style encoder-decoder uses 3 levels with base_features=32 (9.45M total params for MIGFNet-nnUNet).

## Backbone Benchmark (companion repo)

6 bare backbones x 5 seeds x 7 scenarios = 210 evaluations. See [P1-picai-backbone-bench](https://github.com/yosh3289/P1-picai-backbone-bench) for the standalone backbone selection experiments.

| Model | Params | Score | SD | AUROC | Dice(+) | CaseSpec |
|-------|--------|-------|-----|-------|---------|----------|
| MONAI UNet | 31.80M | **0.7061** | 0.0781 | 0.8610 | 0.4700 | 0.3528 |
| nnUNet | 7.11M | 0.6981 | **0.0262** | 0.8523 | 0.4722 | **0.5602** |
| Conv1D | 2.48M | 0.6881 | 0.0497 | 0.8379 | 0.4276 | 0.4130 |
| Conv3D | 2.54M | 0.6354 | 0.0206 | 0.8075 | 0.4290 | 0.4028 |
| Mamba | 9.82M | 0.6250 | 0.0666 | 0.7813 | 0.4165 | 0.2630 |
| SwinUNETR | 62.19M | 0.5550 | 0.0232 | 0.7518 | 0.3235 | 0.2157 |

**Key finding**: nnUNet offers the best balance of performance (Score 0.698) and stability (SD 0.026). T2W is the most critical modality (missing T2W causes -25% to -40% score drop across all models).

## MIGFNet Cross-Backbone Results

3 MIGFNet variants x 5 seeds x 7 scenarios = 105 evaluations.

All three backbones benefit from the MIGFNet wrapper. Mamba gains the most (+7.9%), largely from specificity recovery.

| Model | Params | Score | SD | AUROC | Dice(+) | CaseSpec | vs Bare |
|-------|--------|-------|-----|-------|---------|----------|---------|
| **MIGFNet-UNet** | 52.60M | **0.7257** | 0.0467 | 0.8657 | **0.4855** | **0.6213** | +2.8% |
| MIGFNet-nnUNet | 9.45M | 0.7228 | 0.0590 | 0.8654 | 0.4816 | 0.5444 | +3.5% |
| MIGFNet-Mamba | 18.67M | 0.6743 | **0.0275** | 0.8391 | 0.4278 | 0.5750 | **+7.9%** |

**Key findings**:
- All 3 backbones benefit from MIGFNet (contradicting earlier single-seed results).
- Mamba gains the most (+7.9% Score, CaseSpec 0.263 to 0.575).
- MIGFNet stabilizes training: Mamba SD 0.060 to 0.028, UNet SD 0.070 to 0.047.
- MIGFNet-nnUNet has the best parameter efficiency: 9.45M params for Score 0.723.

## Module Ablation (MIGFNet-nnUNet)

6 ablation configs x 5 seeds x 7 scenarios = 210 evaluations, 30 training runs.

Which modules matter? A complete 2^3 factorial design (8 configurations) isolates the contribution of Gating (G), Deep Supervision (D), and Modality Dropout (M).

### Module Contribution Matrix (Ideal Scenario, 5-seed mean)

| Config | G | D | M | Score | SD |
|--------|:-:|:-:|:-:|-------|-----|
| **A2 (G+M)** | Y | N | Y | **0.7304** | 0.0437 |
| Full (G+D+M) | Y | Y | Y | 0.7228 | 0.0590 |
| Bare (none) | N | N | N | 0.6981 | 0.0262 |
| A4 G only | Y | N | N | 0.6848 | 0.0385 |
| A6 M only | N | N | Y | 0.6786 | 0.0218 |
| A3 w/o M (G+D) | Y | Y | N | 0.6777 | 0.0455 |
| A1 w/o G (D+M) | N | Y | Y | 0.6698 | 0.0297 |
| A5 D only | N | Y | N | 0.6342 | 0.0314 |

### Marginal Module Effects

Average Score with module ON minus average Score with module OFF (across all configs):

| Module | Marginal Effect | Interpretation |
|--------|:-:|---|
| Gating (G) | **+0.034** | Most valuable single module |
| ModDrop (M) | +0.027 | Critical for robustness |
| DeepSup (D) | -0.022 | Harmful when combined with G+M |

**Key findings**:
- **A2 (Gating + ModDrop, no DeepSup) is the best configuration**, outperforming the full model by +1.1% Score.
- Deep supervision has a negative marginal effect -- it consumes gradient budget in the shallow nnUNet without helping convergence.
- Gating alone (A4) already adds +3.5% over bare nnUNet.
- ModDrop is essential for robustness: A3 (no ModDrop) suffers the largest artifact-scenario drops.

## Deep Supervision Cross-Model Validation

Does the "DeepSup is harmful" finding generalize? Tested on MIGFNet-UNet and MIGFNet-Mamba (5 seeds each).

| Model | Params | Depth | With DS | Without DS | Delta |
|-------|--------|:-----:|---------|------------|:-----:|
| MIGFNet-nnUNet | 9.45M | 3 levels | 0.7228 | **0.7304** | +1.1% |
| MIGFNet-Mamba | 18.67M | 4 levels | 0.6743 | **0.7086** | **+3.4%** |
| MIGFNet-UNet | 52.60M | 5 levels | **0.7257** | 0.6783 | -4.7% |

**Conclusion**: Deep supervision benefit depends on model depth/capacity. Shallow networks (3--4 levels) have limited parameter budget -- auxiliary heads steal gradient from the main task. Deep networks (5 levels) need auxiliary heads to propagate gradients to early layers.

## Data Sources

All data is publicly available:

| Resource | URL |
|----------|-----|
| PI-CAI Images (25 GB, 1500 studies) | https://zenodo.org/records/6624726 |
| PI-CAI Labels (human expert annotations) | https://github.com/DIAGNijmegen/picai_labels |
| PI-CAI Eval (official evaluation metrics) | https://github.com/DIAGNijmegen/picai_eval |

### Download Instructions

```bash
# 1. Clone labels
git clone https://github.com/DIAGNijmegen/picai_labels.git data/labels/

# 2. Download images via Zenodo API
pip install zenodo_get
zenodo_get -d https://zenodo.org/records/6624726 -o data/raw/

# 3. Extract all folds into data/images/
for f in data/raw/picai_public_images_fold*.zip; do
    unzip -q "$f" -d data/images/
done
```

## Project Structure

```
MIGFNet/
├── configs/                      # Experiment configs & fold-0 split
│   ├── adaptive_mpnet.yaml       # Main MIGFNet config
│   ├── ablation_nnunet_A*.yaml   # 6 ablation configs (A1--A6)
│   ├── ablation_*_no_deepsup.yaml # Cross-backbone DeepSup configs
│   └── fold0_split.json          # Official PI-CAI fold-0 (1200/300)
├── src/
│   ├── models/
│   │   ├── adaptive_mpnet.py     # MIGFNet (Gating + ModDrop + DeepSup)
│   │   ├── adaptive_native.py    # MIGFNet variants for UNet/nnUNet/Mamba
│   │   ├── mamba_ssm.py          # Pure PyTorch Mamba SSM reimplementation
│   │   └── baselines.py          # SwinUNETR, Mamba, etc.
│   ├── data/dataset.py           # Multi-modal dataset with modality dropout
│   └── utils/                    # Losses, metrics, GPU monitor
├── scripts/
│   ├── train.py                  # DDP training (MIGFNet)
│   ├── train_baseline.py         # DDP training (bare baselines)
│   ├── train_official_unet*.py   # MONAI 3D UNet training
│   ├── evaluate.py               # picai_eval + 7-scenario robustness
│   ├── preprocess_cache.py       # Offline preprocessing cache
│   ├── run_ampnet_multiseed.sh   # Cross-backbone multi-seed runner
│   ├── run_ablation_nnunet.sh    # Module ablation runner
│   ├── summarize_ampnet.py       # Cross-backbone results aggregation
│   └── summarize_ablation_nnunet.py  # Ablation results aggregation
└── tests/                        # Gradient and smoke tests
```

## Training

All models trained on 4x RTX 5090 (32 GB) with DDP, 300 fixed epochs, fold-0 split.

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW (lr=5e-5, wd=0.01) |
| Scheduler | ReduceOnPlateau (patience=10, factor=0.5) |
| Loss | DiceFocalLoss (alpha=0.9, gamma=2.0) |
| Batch size | 8 per GPU |
| Patch size | 64 x 64 x 16 |
| Mixed precision | Yes (AMP) |
| Modality dropout | p=0.3 |
| Positive oversampling | 5x (approx 65% positive batches) |

### Preprocessing

MRI channels are cached as `.pt` tensors (target spacing 0.5 x 0.5 x 3.0 mm, crop 128 x 128 x 32):

| Channel | Normalization |
|---------|---------------|
| T2W | Z-score |
| HBV (high b-value DWI) | Z-score |
| ADC | Min-max [0, 1] (clipped at 3000) |

## Related Repositories

| Repo | Description |
|------|-------------|
| [P1-picai-backbone-bench](https://github.com/yosh3289/P1-picai-backbone-bench) | 6 bare backbones x 5 seeds benchmark (backbone selection) |

## Evaluation Protocol

- **Metrics**: AUROC, Average Precision (AP), Ranking Score = (AUROC + AP) / 2, Dice (positive cases), Sensitivity, Specificity (voxel-level), Case Sensitivity, Case Specificity.
- **Robustness scenarios** (7 total): ideal, missing T2W/HBV/ADC, artifact T2W/HBV/ADC.
- **Statistical design**: 5 random seeds per config. Tables report mean and SD of Score across seeds under the ideal scenario.
- **Evaluation library**: [picai_eval](https://github.com/DIAGNijmegen/picai_eval) v1.4.13.

## Citation

If you use MIGFNet in your research, please cite:

```bibtex
@article{shu2026migfnet,
  author        = {Shu, Yongbo and Xie, Wenzhao and Yao, Shanhu and Xin, Zirui and Lei, Luo and Chen, Kewen and Luo, Aijing},
  title         = {Architecture-Agnostic Modality-Isolated Gated Fusion for Robust Multi-Modal Prostate MRI Segmentation},
  year          = {2026},
  eprint        = {2604.10702},
  archivePrefix = {arXiv},
  primaryClass  = {eess.IV},
  url           = {https://arxiv.org/abs/2604.10702}
}
```

## License

Released under the MIT License. See [LICENSE](LICENSE).
