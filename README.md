# MIGFNet: Modality-Isolated Gated Fusion for Multi-Modal Prostate MRI Segmentation

[![arXiv](https://img.shields.io/badge/arXiv-2604.10702-b31b1b.svg)](https://arxiv.org/abs/2604.10702)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Code for modality-isolated gated fusion (MIGF) in clinically significant prostate cancer (csPCa) lesion segmentation from biparametric MRI (T2W, HBV, ADC). MIGF was evaluated with five-fold cross-validation on the public [PI-CAI](https://pi-cai.grand-challenge.org/) dataset (1500 studies) and on the external [Prostate158](https://github.com/kbressem/prostate158) dataset.

> **Status (October 2026).** This README summarizes the current version of the study, *Learned Modality Gating for Prostate MRI Lesion Segmentation: Five-Fold Component Contrasts and Gate Behavior in Convolutional and State-Space Networks*. Its analysis differs from that of arXiv:2604.10702 v4 (`paper.pdf`), whose conclusions are superseded by the findings below. The repository state that accompanied v4 is kept under the tag `arxiv-v4`.

## Main findings

Seven combinations of gating (G), deep supervision (D) and modality dropout (M) were trained in a convolutional nnU-Net-style network (105 models) and a Mamba state-space network (45 models). Each component's effect was estimated from matched configurations that differ in that component only.

- **Gating effects differed between the two networks.** Gating lowered the Ranking Score of the nnU-Net-style network in all three matched contrasts (-0.025 to -0.039); in the Mamba network the three contrasts were mixed (-0.012 to +0.034).
- **Modality dropout was the most consistently helpful component.** It raised the score in all three nnU-Net-style contrasts (+0.018 to +0.047) and in two of the three Mamba contrasts, most clearly when a sequence was missing.
- **Gate weights.** The entry-stage modality weights were practically constant across cases in the nnU-Net-style models (across-case SD 0.0033 ± 0.0008) and about 11 times more variable in the Mamba models (0.0365 ± 0.0080). Replacing them by their training mean lowered Dice in all 15 Mamba models and changed the nnU-Net-style models little. These tests show that the Mamba models use input-dependent weights; they do not show that the weights track image quality.
- **External cohort.** On Prostate158, case-level specificity was at most 0.007 for the nnU-Net-style configurations and 0.193 to 0.482 for the Mamba configurations. Prostate158 was scored on a different field than PI-CAI (see below), so this difference is not attributed to domain shift, architecture or fusion design.

### Evaluation conditions

- PI-CAI metrics were computed in a 64 × 64 × 16-voxel field centered on the reference lesion (on the volume center for studies without a lesion mask). They describe segmentation and case decisions inside this label-centered field, not label-free detection over the whole gland.
- Prostate158 was evaluated on the cached 128 × 128 × 32 gland-centered grid. This field is inferred from the archived preprocessing and early evaluation scripts; the per-checkpoint evaluation program was not retained.
- Checkpoints were selected by validation Dice on the folds on which results are reported, so all PI-CAI values are development-fold estimates.
- Case-level labels are derived from the cached lesion masks: 404 of the 425 csPCa-positive studies keep a non-empty mask in the cached field, and the other 21 count as negative.
- All analyses are descriptive. Runs of the same fold share validation studies and the training sets overlap, so runs are not independent samples.

## Results

### Configurations (PI-CAI validation folds, ideal input, mean ± SD over runs)

| Network | Config | G | D | M | n | Ranking Score | Dice | CaseSens | CaseSpec |
|---|---|:-:|:-:|:-:|:-:|---|---|---|---|
| nnU-Net-style | A1 | - | Y | Y | 15 | 0.748 ± 0.043 | 0.480 ± 0.031 | 0.951 ± 0.031 | 0.535 ± 0.163 |
| | A2 | Y | - | Y | 15 | 0.700 ± 0.048 | 0.471 ± 0.034 | 0.940 ± 0.037 | 0.456 ± 0.153 |
| | A3 | Y | Y | - | 15 | 0.676 ± 0.067 | 0.460 ± 0.029 | 0.941 ± 0.040 | 0.445 ± 0.197 |
| | A4 | Y | - | - | 15 | 0.683 ± 0.054 | 0.459 ± 0.029 | 0.957 ± 0.030 | 0.401 ± 0.184 |
| | A5 | - | Y | - | 15 | 0.702 ± 0.053 | 0.463 ± 0.030 | 0.944 ± 0.027 | 0.515 ± 0.145 |
| | A6 | - | - | Y | 15 | 0.729 ± 0.046 | 0.480 ± 0.031 | 0.966 ± 0.025 | 0.426 ± 0.225 |
| | Full | Y | Y | Y | 15 | 0.710 ± 0.052 | 0.478 ± 0.036 | 0.954 ± 0.027 | 0.453 ± 0.149 |
| Mamba | A1 | - | Y | Y | 5 | 0.638 ± 0.065 | 0.422 ± 0.039 | 0.959 ± 0.034 | 0.341 ± 0.152 |
| | A2 | Y | - | Y | 15 | 0.672 ± 0.054 | 0.432 ± 0.031 | 0.926 ± 0.033 | 0.579 ± 0.091 |
| | A3 | Y | Y | - | 5 | 0.654 ± 0.060 | 0.420 ± 0.025 | 0.937 ± 0.036 | 0.534 ± 0.148 |
| | A4 | Y | - | - | 5 | 0.622 ± 0.069 | 0.415 ± 0.026 | 0.924 ± 0.031 | 0.503 ± 0.161 |
| | A5 | - | Y | - | 5 | 0.621 ± 0.081 | 0.414 ± 0.036 | 0.944 ± 0.046 | 0.360 ± 0.244 |
| | A6 | - | - | Y | 5 | 0.671 ± 0.054 | 0.430 ± 0.037 | 0.923 ± 0.023 | 0.540 ± 0.100 |
| | Full | Y | Y | Y | 5 | 0.649 ± 0.066 | 0.432 ± 0.037 | 0.910 ± 0.054 | 0.514 ± 0.302 |

Ranking Score = mean of the case-level AUROC and the lesion-level average precision (`picai_eval`). CaseSens/CaseSpec use the any-voxel rule. n = five folds × nominal seeds (Mamba configurations other than A2: seed 42 only).

### Matched contrasts (Ranking Score, ideal input)

Mean difference over matched runs, followed by the number of runs with a positive difference (nnU-Net-style: of 15, plus folds with a positive seed-averaged difference, of 5; Mamba: of 5, seed 42).

| Component | Contrast | nnU-Net-style | Mamba |
|---|---|---|---|
| Gating (G) | A2 - A6 | -0.029 (4/15; 1/5) | -0.012 (2/5) |
| | A3 - A5 | -0.025 (4/15; 0/5) | +0.034 (3/5) |
| | Full - A1 | -0.039 (4/15; 1/5) | +0.010 (3/5) |
| Deep supervision (D) | A1 - A6 | +0.019 (11/15; 4/5) | -0.032 (1/5) |
| | A3 - A4 | -0.006 (8/15; 2/5) | +0.033 (5/5) |
| | Full - A2 | +0.009 (8/15; 2/5) | -0.010 (2/5) |
| Modality dropout (M) | A2 - A4 | +0.018 (11/15; 3/5) | +0.037 (5/5) |
| | A1 - A5 | +0.047 (12/15; 4/5) | +0.018 (3/5) |
| | Full - A3 | +0.033 (11/15; 4/5) | -0.006 (2/5) |

### Entry-stage gate weights of the 30 A2 models

| Quantity | nnU-Net-style (15 models) | Mamba (15 models) |
|---|---|---|
| Mean weight T2W / HBV / ADC | 0.517 / 0.262 / 0.221 | 0.468 / 0.303 / 0.228 |
| Across-case SD of the weights | 0.0033 ± 0.0008 | 0.0365 ± 0.0080 |
| Dice change, weights replaced by training mean (native - replaced) | -0.0003 ± 0.0003 (2/15 positive) | +0.0433 ± 0.0187 (15/15 positive) |
| Dice change, weights replaced by uniform weights | +0.0412 ± 0.0258 (15/15) | +0.0895 ± 0.0372 (15/15) |

## Architecture

MIGF encodes each sequence in its own stream and fuses the streams at every encoder stage with a gating module:

1. **Modality weights.** A scoring network (two-layer perceptron on globally pooled features) gives one score per sequence; a softmax turns the scores into modality weights that combine the streams. Only the entry-stage weights are weights of individual sequences, because later stages receive copies of the previous fused feature. The code calls the scoring network a quality estimator; whether its output tracks image quality was not tested.
2. **Feature gate (G).** The combined feature is rescaled element-wise by a small gate (two 1×1×1 convolutions with SiLU and sigmoid) before an output projection.
3. **Modality dropout (M).** During training, one randomly chosen sequence is set to zero with probability 0.3 for each sample.
4. **Deep supervision (D).** Two auxiliary pointwise segmentation heads at the two intermediate decoder levels, with loss weights 0.5 and 0.25.

Configurations: A1 (D+M), A2 (G+M), A3 (G+D), A4 (G), A5 (D), A6 (M) and Full (G+D+M), trained with the same MIGF trainer. The all-off cell of the 2^3 design was not trained with this trainer, so component effects come only from contrasts among these seven configurations.

- **nnU-Net-style network:** three gated encoder levels (32/64/128 channels), a 256-channel bottleneck, a three-level decoder with skip connections and a separate scoring network per sequence; 7.11 M parameters without gating, 9.45 M with gating. It follows the nnU-Net encoder-decoder design but is a custom network, not the self-configuring nnU-Net framework. Its first-stage streams (bias-free convolution, instance normalization without affine parameters, LeakyReLU) map an all-zero input to an all-zero feature map; a zeroed stream still receives a non-zero softmax weight.
- **Mamba network:** per-sequence patch embedding with squeeze-and-excitation, four gated encoder stages (48/96/192/384 channels) with two Mamba blocks per branch and stage (state dimension 16, convolution width 4, expansion factor 2), a 384-channel bottleneck, three decoder stages and one scoring network shared by the three sequences; 18.18 M parameters without gating, 18.67 M with gating. Its patch embedding and squeeze-and-excitation layers contain biases, so it has no zero-preservation guarantee.

## Experimental protocol

- **Data.** PI-CAI Public Training and Development Dataset: 1500 studies of 1476 patients. Lesion masks combine the 220 expert annotations distributed with the dataset and the expert annotations of the remaining 205 csPCa-positive studies (Pooch et al., in `picai_labels`). The clinical marksheet lists 425 csPCa-positive (ISUP grade group ≥ 2) and 1075 negative studies. Five patient-level folds follow the public PI-CAI cross-validation lists (validation folds: 300, 298, 302, 304 and 296 studies). Prostate158 (158 studies) was used for inference only.
- **Preprocessing.** Resampling to 0.5 × 0.5 × 3.0 mm, cropping or padding to 128 × 128 × 32 voxels around the centroid of the whole-gland segmentation distributed with `picai_labels`, z-score normalization of T2W and HBV, and clipping of ADC to [0, 3000] followed by min-max scaling.
- **Models.** nnU-Net-style: seven configurations × five folds × three nominal seeds (42, 123, 789) = 105 models. Mamba: A2 with three seeds and the other six configurations with seed 42 = 45 models. 150 models in total.
- **Gate analysis.** Entry-stage weights of the 30 A2 models on their validation folds; replacement of the weights of every gating stage by their training mean or by uniform weights.
- **Input scenarios.** Ideal input, one sequence set to zero (no T2W, no HBV, no ADC) and additive Gaussian noise (σ = 0.5) on one normalized sequence.

## Training

| Setting | Value |
|---|---|
| Epochs | 300 |
| Optimizer | AdamW (learning rate 5e-5, weight decay 0.01) |
| Schedule | 10-epoch linear warm-up, then reduce-on-plateau on validation Dice (factor 0.5, patience 10 validation checks, minimum 1e-6) |
| Loss | DiceFocal (focal alpha = 0.9, gamma = 2.0) |
| Batch size | 8 (64 × 64 × 16 patches, centered on a lesion voxel with probability 0.75; studies with a lesion mask oversampled five-fold) |
| Augmentation | random flips, Gaussian noise (sigma = 0.1), simulated motion in one sequence with probability 0.15 |
| Modality dropout | p = 0.3 (configurations with M) |
| Precision / clipping | mixed precision, gradient clipping at 1.0 |
| Checkpoint selection | highest validation Dice, evaluated every five epochs |

Each model was trained as a single-GPU job; the jobs were distributed over four NVIDIA RTX 5090 GPUs.

## Repository structure

```
MIGFNet/
+-- analysis/                     # gate-weight extraction and figure scripts
|   +-- extract_gate_weights.py   # entry-stage weights and across-case SD of the A2 models
|   +-- gate_weights_local.json   # extracted gate-weight summary
|   +-- recompute_stats.py        # statistics of the arXiv v4 analysis (superseded)
|   +-- verified_stats.txt        # statistics of the arXiv v4 analysis (superseded)
|   +-- make_fig_gate.py, make_fig_robustness.py
+-- configs/                      # experiment configurations
+-- src/
|   +-- models/                   # MIGF module and network implementations
|   +-- data/dataset.py           # multi-modal dataset with modality dropout
|   +-- utils/                    # losses (incl. deep-supervision weighting), metrics
+-- scripts/                      # training, evaluation, preprocessing, runners
+-- paper.pdf                     # arXiv v4 manuscript (superseded)
+-- tests/
```

## Data sources

| Resource | URL |
|---|---|
| PI-CAI images (1500 studies) | https://doi.org/10.5281/zenodo.6624726 |
| PI-CAI annotations (incl. Pooch et al. expert annotations) | https://github.com/DIAGNijmegen/picai_labels |
| PI-CAI cross-validation lists | https://github.com/DIAGNijmegen/picai_baseline |
| picai_eval | https://github.com/DIAGNijmegen/picai_eval |
| Prostate158 (external) | https://github.com/kbressem/prostate158 |

## Citation

Until the current version is posted, please cite the arXiv preprint (see `CITATION.cff`):

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
