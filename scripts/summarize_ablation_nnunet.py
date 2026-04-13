#!/usr/bin/env python3
"""Summarize AMPNet-nnUNet ablation results (6 configs × 5 seeds + Full + Bare)."""

import json
import os
import numpy as np
from pathlib import Path

SEEDS = [42, 123, 456, 789, 1024]
SCENARIOS = ["ideal", "missing_t2w", "missing_hbv", "missing_adc",
             "artifact_t2w", "artifact_hbv", "artifact_adc"]

# 8-point matrix: 6 ablation + Full + Bare
CONFIGS = [
    # (label, G, D, M, eval_json_pattern)
    ("Full (G+D+M)",   "Y", "Y", "Y", "outputs/results/seed{seed}/adaptive_nnunet_eval.json"),
    ("A1 w/o G (D+M)", "N", "Y", "Y", "outputs/results/ablation_nnunet/A1/seed{seed}/adaptive_nnunet_eval.json"),
    ("A2 w/o D (G+M)", "Y", "N", "Y", "outputs/results/ablation_nnunet/A2/seed{seed}/adaptive_nnunet_eval.json"),
    ("A3 w/o M (G+D)", "Y", "Y", "N", "outputs/results/ablation_nnunet/A3/seed{seed}/adaptive_nnunet_eval.json"),
    ("A4 G only",      "Y", "N", "N", "outputs/results/ablation_nnunet/A4/seed{seed}/adaptive_nnunet_eval.json"),
    ("A5 D only",      "N", "Y", "N", "outputs/results/ablation_nnunet/A5/seed{seed}/adaptive_nnunet_eval.json"),
    ("A6 M only",      "N", "N", "Y", "outputs/results/ablation_nnunet/A6/seed{seed}/adaptive_nnunet_eval.json"),
]

# Bare nnUNet from picai-backbone-bench
BARE_PATTERN = "/workspace/P1-picai-backbone-bench/outputs/results/seed{seed}/nnunet_eval.json"


def load_results(pattern, seeds):
    """Load eval JSONs for all seeds, return dict of scenario -> metric -> [values]."""
    all_data = {}
    missing = 0
    for seed in seeds:
        path = pattern.format(seed=seed)
        if not os.path.exists(path):
            missing += 1
            continue
        with open(path) as f:
            data = json.load(f)
        scenarios = data.get("scenarios", data)
        for sc_name, sc_data in scenarios.items():
            if sc_name not in all_data:
                all_data[sc_name] = {k: [] for k in ["auroc", "ap", "ranking_score",
                                                       "dice", "sensitivity", "specificity",
                                                       "case_sensitivity", "case_specificity"]}
            for k in all_data[sc_name]:
                if k in sc_data:
                    all_data[sc_name][k].append(sc_data[k])
    return all_data, missing


def fmt(vals, fmt_str=".4f"):
    if not vals:
        return "N/A"
    m = np.mean(vals)
    s = np.std(vals)
    return f"{m:{fmt_str}}+/-{s:{fmt_str}}"


def main():
    lines = []
    lines.append("# AMPNet-nnUNet Module Ablation Results\n")
    lines.append("**Model**: adaptive_nnunet (base_features=32)")
    lines.append("**Training**: AdamW lr=5e-5, batch=8x4GPU, 300 epochs, 5 seeds")
    lines.append("**Data**: PI-CAI fold-0 (1200 train / 300 val)\n")
    lines.append("**Modules**: G=Adaptive Gating, D=Deep Supervision, M=Modality Dropout\n")
    lines.append("---\n")

    # Collect all results
    all_results = []  # (label, G, D, M, params, data, missing)
    for label, g, d, m, pattern in CONFIGS:
        data, miss = load_results(pattern, SEEDS)
        params = "9.45M" if g == "Y" else "7.11M"
        all_results.append((label, g, d, m, params, data, miss))

    # Bare
    bare_data, bare_miss = load_results(BARE_PATTERN, SEEDS)
    all_results.append(("Bare (none)", "N", "N", "N", "7.11M", bare_data, bare_miss))

    total = sum(1 for _, _, _, _, _, _, m in all_results if m == 0)
    lines.append(f"Loaded {total}/8 configs with complete data.\n")

    # Table 1: Ideal scenario detailed
    lines.append("## Table 1: Module Contribution Matrix (Ideal Scenario, mean+/-sd)\n")
    lines.append("| Config | G | D | M | Params | AUROC | AP | Score | Dice(+) | Sens(+) | Spec | CaseSens | CaseSpec |")
    lines.append("| :--- | :-: | :-: | :-: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    for label, g, d, m, params, data, miss in all_results:
        if "ideal" not in data:
            lines.append(f"| {label} | {g} | {d} | {m} | {params} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue
        sc = data["ideal"]
        lines.append(f"| {label} | {g} | {d} | {m} | {params} | "
                     f"{fmt(sc['auroc'])} | {fmt(sc['ap'])} | {fmt(sc['ranking_score'])} | "
                     f"{fmt(sc['dice'])} | {fmt(sc['sensitivity'])} | {fmt(sc['specificity'])} | "
                     f"{fmt(sc['case_sensitivity'])} | {fmt(sc['case_specificity'])} |")

    lines.append("\n---\n")

    # Table 2: Robustness (Score across 7 scenarios)
    lines.append("## Table 2: Robustness — Score across 7 Scenarios (mean+/-sd)\n")
    header = "| Config | G | D | M |"
    for sc in SCENARIOS:
        header += f" {sc} |"
    header += " Avg |"
    lines.append(header)
    sep = "| :--- | :-: | :-: | :-: |" + " ---: |" * (len(SCENARIOS) + 1)
    lines.append(sep)

    for label, g, d, m, params, data, miss in all_results:
        row = f"| {label} | {g} | {d} | {m} |"
        avgs = []
        ideal_score = np.mean(data["ideal"]["ranking_score"]) if "ideal" in data else None
        for sc in SCENARIOS:
            if sc not in data or not data[sc]["ranking_score"]:
                row += " N/A |"
                continue
            score_mean = np.mean(data[sc]["ranking_score"])
            score_sd = np.std(data[sc]["ranking_score"])
            avgs.append(score_mean)
            if sc == "ideal":
                row += f" {score_mean:.4f}+/-{score_sd:.4f} |"
            else:
                pct = ((score_mean - ideal_score) / ideal_score * 100) if ideal_score else 0
                row += f" {score_mean:.4f} ({pct:+.0f}%) |"
        avg_score = np.mean(avgs) if avgs else 0
        row += f" {avg_score:.4f} |"
        lines.append(row)

    lines.append("\n---\n")

    # Table 3: Marginal effect of each module
    lines.append("## Table 3: Marginal Module Effect (avg Score across all configs)\n")
    lines.append("Marginal effect = mean(Score when module ON) - mean(Score when module OFF)\n")

    # For each module, compute mean ideal Score when ON vs OFF
    for mod_name, mod_idx in [("G (Gating)", 1), ("D (DeepSup)", 2), ("M (ModDrop)", 3)]:
        on_scores = []
        off_scores = []
        for label, g, d, m, params, data, miss in all_results:
            flags = [g, d, m]
            if "ideal" not in data or not data["ideal"]["ranking_score"]:
                continue
            mean_score = np.mean(data["ideal"]["ranking_score"])
            if flags[mod_idx - 1] == "Y":
                on_scores.append(mean_score)
            else:
                off_scores.append(mean_score)
        if on_scores and off_scores:
            delta = np.mean(on_scores) - np.mean(off_scores)
            lines.append(f"- **{mod_name}**: ON={np.mean(on_scores):.4f} (n={len(on_scores)}), "
                        f"OFF={np.mean(off_scores):.4f} (n={len(off_scores)}), "
                        f"**Delta={delta:+.4f}**")

    lines.append("")

    # Write output
    output_dir = Path("outputs/tables")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ablation_nnunet_results.md"
    content = "\n".join(lines)
    output_path.write_text(content)

    print(content)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
