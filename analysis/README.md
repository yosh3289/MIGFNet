# 5-Fold Cross-Validation and Gate-Mechanism Analysis

This directory holds the analysis code for the paper *"Backbone-Conditional
Behavior of Modality Gating in Multi-Modal Prostate MRI Segmentation: A 5-Fold
Cross-Validation and Gate Mechanism Analysis"* (the gate-mechanism study that
supersedes the earlier single-fold report).

## Experimental protocol

- Datasets: PI-CAI (n=1500, 404 csPCa-positive; 5 patient-level folds from the
  picai_baseline pipeline) for internal CV; Prostate158 (n=158) for cross-cohort,
  inference-only evaluation.
- Factorial ablation: eight module configurations over gating (G), modality
  dropout (M), and deep supervision (D) -- Bare, A1(D+M), A2(G+M), A3(G+D), A4(G),
  A5(D), A6(M), Full(G+D+M) -- on two backbones (nnU-Net and Mamba), under 5-fold
  CV. nnU-Net: all 8 configs x 5 folds x 3 seeds (42/123/789). Mamba: Bare and
  A2 with 3 seeds, the other six configs single-seed. 180 trained models total.
- Gate-mechanism analysis: on the 30 A2 checkpoints (15 nnU-Net, 15 Mamba), the
  per-modality quality weights (alpha) are extracted on each model's validation
  fold; per-sample variability is quantified by the across-case standard
  deviation, and a counterfactual replaces the per-sample gate output with the
  training-set mean.

## Scripts

| File | Purpose |
|------|---------|
| extract_gate_weights.py | Hooks the modality-quality estimator on the 30 A2 checkpoints, collects per-case alpha weights on each validation fold, and writes per-modality mean + across-case SD to gate_weights_local.json. |
| recompute_stats.py | Exact paired statistics from the per-(fold,seed) result JSONs: Student-t p-values via the incomplete beta function, Wilcoxon signed-rank, and 95% CIs (no SciPy dependency). |
| make_fig_gate.py | Builds the gate-mechanism figure: modality prior, across-case SD separation (~11x), and counterfactual delta. |
| make_fig_robustness.py | Builds the seven-scenario robustness figure (Bare vs A2, per backbone). |
| gate_weights_local.json | Extracted gate-weight summary (output of extract_gate_weights.py). |
| verified_stats.txt | Frozen, verified statistics used in the manuscript. |

## Reproducing

Run under an environment with PyTorch (CUDA) and picai_baseline installed (see
the repository requirements.txt). The scripts expect the trained checkpoints
under outputs/checkpoints/ablation_{nnunet,mamba}/A2/foldF_seedS/ and the dataset
cache produced by the preprocessing pipeline; adjust the path variables at the
top of each script to your local layout before running.

```bash
python analysis/extract_gate_weights.py   # -> gate_weights_local.json
python analysis/recompute_stats.py         # paired tests + CIs
python analysis/make_fig_gate.py           # gate-mechanism figure
python analysis/make_fig_robustness.py     # robustness figure
```
