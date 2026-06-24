# MIGFNet: Modality-Isolated Gated Fusion for Multi-Modal Prostate MRI Segmentation

[![arXiv](https://img.shields.io/badge/arXiv-2604.10702-b31b1b.svg)](https://arxiv.org/abs/2604.10702)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Clinically significant prostate cancer (csPCa) segmentation from multi-parametric MRI (T2W, HBV, ADC) using per-modality feature isolation, learned gated fusion, and modality-dropout training. Evaluated on the [PI-CAI](https://pi-cai.grand-challenge.org/) public dataset (1500 studies) with cross-cohort validation on [Prostate158](https://github.com/kbressem/prostate158).

This repository is the official implementation for *Backbone-Conditional Behavior of Modality Gating in Multi-Modal Prostate MRI Segmentation: A 5-Fold Cross-Validation and Gate Mechanism Analysis* (arXiv:2604.10702). It covers the Modality-Isolated Gated Fusion (MIGF) module, its integration across nnU-Net / Mamba / U-Net backbones, the 5-fold factorial ablation, and the gate-mechanism analysis.

> **Revision note (v2).** This version supersedes an earlier single-fold preprint. Under rigorous 5-fold cross-validation (3 seeds, 180 trained models) and a direct gate-mechanism analysis, the effect of modality gating is shown to be **backbone-conditional** rather than architecture-agnostic. Conclusions and numbers below reflect this revised analysis.

## Key Findings (5-fold cross-validation)

Modality gating is **backbone-conditional**, not universally beneficial:

- **On nnU-Net (convolutional), gating *reduces* performance.** Adding gating (configs A2/A3/A4) lowers the Ranking Score versus Bare by -0.036 / -0.060 / -0.054 (all p < 0.05, two-sided paired t-test); the best configuration is **A1 (D+M, no gating)** at 0.7483. Marginal gating effect: **-0.037**.
- **On Mamba (state-space), gating *helps*.** The G+M configuration improves the Ranking Score (**+0.024, p = 0.037**) and raises case-level specificity (0.465 -> 0.579).
- **Gate-mechanism analysis explains why.** On nnU-Net the learned gate collapses into a near-static modality prior (across-case SD **0.0033**); on Mamba it retains sample-dependent variation (**0.0365**, ~11x larger, non-overlapping distributions). Replacing per-sample gate outputs with their training-set mean leaves nnU-Net unchanged but degrades Mamba.
- **Modality dropout is the only component beneficial on *both* backbones** (marginal +0.023 / +0.021).
- **Cross-cohort (Prostate158).** Convolutional backbones (nnU-Net, MONAI 3D U-Net) collapse to case-level specificity ~0; only Mamba retains it (**MIGF-Mamba 0.314**, highest of all families). Modality isolation (MIGF) improves the cross-cohort Ranking Score for all three backbones.

### nnU-Net 8-configuration factorial ablation

5 folds x 3 seeds (n = 15 per config), ideal-input scenario; mean Ranking Score.

| Config | G | D | M | Ranking Score | p vs Bare |
|--------|:-:|:-:|:-:|---------------|:---------:|
| **A1 (D+M)** | - | Y | Y | **0.7483** | 0.44 |
| Bare | - | - | - | 0.7363 | - |
| A6 (M) | - | - | Y | 0.7294 | 0.71 |
| Full (G+D+M) | Y | Y | Y | 0.7095 | 0.097 |
| A5 (D) | - | Y | - | 0.7016 | 0.058 |
| A2 (G+M) | Y | - | Y | 0.7004 | **0.012** |
| A4 (G) | Y | - | - | 0.6826 | **0.006** |
| A3 (G+D) | Y | Y | - | 0.6764 | **0.003** |

Marginal effects: gating **-0.037**, modality dropout **+0.023**, deep supervision -0.003.

### Mamba

Bare and A2 use 3 seeds (n = 15); the remaining six configurations use a single seed and are descriptive only.

| Config | Ranking Score | p vs Bare | CaseSpec |
|--------|---------------|:---------:|----------|
| **A2 (G+M)** | **0.6724** | **0.037** | **0.579** |
| Bare | 0.6488 | - | 0.465 |

Marginal effects: gating **+0.005**, modality dropout **+0.021**, deep supervision -0.013.

## Architecture

MIGF wraps an existing segmentation backbone with three optional components:

1. **Per-modality encoders** -- each MRI channel (T2W, HBV, ADC) is encoded by an independent, *bias-free* stream, so a zero-filled (missing) modality yields an identically zero feature map and contributes neither signal nor noise (the *isolation property*).
2. **AdaptiveModalGating (G)** -- a lightweight per-modality estimator produces convex modality weights (softmax) that linearly combine the streams, followed by a small feature-level gate (two 1x1x1 convolutions, SiLU, sigmoid). Whether these gate weights perform sample-dependent routing or collapse to a static prior is the subject of the gate-mechanism analysis -- and is found to depend on the backbone.
3. **Modality Dropout (M)** -- during training, one modality channel is randomly zeroed (p = 0.3), exercising the isolation property.

Optionally, **Deep Supervision (D)** attaches two auxiliary 1x1x1-convolution segmentation heads at the two intermediate decoder levels (loss weights 0.5 and 0.25). The three components are studied in a full 2^3 = 8 factorial. MIGF preserves each backbone's encoder-decoder topology, yielding MIGF-nnU-Net, MIGF-Mamba, and MIGF-U-Net.

## Experimental protocol

- **Datasets.** PI-CAI (n = 1500; 404 csPCa-positive after the 128x128x32 crop), 5 patient-level folds via the `picai_baseline` pipeline; lesion annotations combine the human-expert delineations with the 205 manually annotated AI-positive cases. Prostate158 (n = 158) for inference-only cross-cohort evaluation.
- **Models.** nnU-Net: all 8 configurations x 5 folds x 3 seeds (42/123/789). Mamba: Bare and A2 with 3 seeds, the other six single-seed. **180 trained models.** A MONAI 3D U-Net (Bare and MIGF-A2) is additionally trained for the cross-cohort and efficiency comparisons.
- **Gate-mechanism analysis.** Across-case standard deviation of the entry-stage modality weights, and a counterfactual mean-gate replacement, on the 30 A2 checkpoints (15 nnU-Net, 15 Mamba).
- **Metrics.** `picai_eval` Ranking Score = (lesion AUROC + AP) / 2, plus case-level specificity/sensitivity and voxel Dice.

## Repository structure

```
MIGFNet/
+-- analysis/                     # 5-fold CV & gate-mechanism analysis (this paper)
|   +-- extract_gate_weights.py   # entry-stage alpha weights + across-case SD + counterfactual
|   +-- recompute_stats.py        # paired t-test / Wilcoxon / 95% CI (pure Python, no SciPy)
|   +-- make_fig_gate.py          # gate-mechanism figure
|   +-- make_fig_robustness.py    # robustness figure
|   +-- gate_weights_local.json   # extracted gate-weight summary
|   +-- verified_stats.txt        # frozen statistics used in the paper
|   +-- README.md
+-- configs/                      # experiment configs & fold splits
+-- src/
|   +-- models/                   # MIGF module + nnU-Net / Mamba / U-Net backbones
|   +-- data/dataset.py           # multi-modal dataset with modality dropout
|   +-- utils/                    # losses (incl. deep-supervision weighting), metrics
+-- scripts/                      # training, evaluation, preprocessing, runners
+-- paper.pdf                     # current manuscript
+-- tests/
```

## Data sources

| Resource | URL |
|----------|-----|
| PI-CAI images (1500 studies) | https://zenodo.org/records/6624726 |
| PI-CAI labels | https://github.com/DIAGNijmegen/picai_labels |
| picai_eval | https://github.com/DIAGNijmegen/picai_eval |
| Prostate158 (external) | https://github.com/kbressem/prostate158 |

## Training

300 fixed epochs per model. Training runs were dispatched across up to four NVIDIA RTX 5090 (32 GB) GPUs, one single-GPU job per model (up to four running concurrently).

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW (lr = 5e-5) |
| Loss | DiceFocal (alpha = 0.9, gamma = 2.0) |
| Batch size | 8 (single GPU) |
| Mixed precision | AMP |
| Modality dropout | p = 0.3 |
| Checkpoint selection | best validation Ranking Score |

Preprocessing caches the three channels (target spacing 0.5 x 0.5 x 3.0 mm, centre crop 128x128x32): T2W and HBV Z-scored, ADC min-max scaled to [0, 1].

## Citation

If you use this code, please cite the paper (see `CITATION.cff`):

```bibtex
@article{shu2026migf,
  title   = {Backbone-Conditional Behavior of Modality Gating in Multi-Modal Prostate MRI Segmentation: A 5-Fold Cross-Validation and Gate Mechanism Analysis},
  author  = {Shu, Yongbo and Xie, Wenzhao and Yao, Shanhu and Xin, Zirui and Lei, Luo and Chen, Kewen and Luo, Aijing},
  journal = {arXiv preprint arXiv:2604.10702},
  year    = {2026}
}
```

## License

MIT -- see [LICENSE](LICENSE).
