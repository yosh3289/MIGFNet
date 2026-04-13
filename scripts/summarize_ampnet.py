"""
Summarize AMPNet multi-seed benchmark results.

Tables:
1. Ideal scenario: mean +/- sd across seeds
2. Robustness: Score per 7 scenarios with % change from ideal
3. Per-seed Score transparency table
"""

import json
import io
import sys
from pathlib import Path

import numpy as np

MODELS = ["adaptive_nnunet", "adaptive_unet", "AdaptiveMPNet"]
MODEL_LABELS = {
    "adaptive_nnunet": "AMPNet-nnUNet",
    "adaptive_unet": "AMPNet-UNet",
    "AdaptiveMPNet": "AMPNet-Mamba",
}
SEEDS = [42, 123, 456, 789, 1024]
SCENARIOS = ["ideal", "missing_t2w", "missing_hbv", "missing_adc",
             "artifact_t2w", "artifact_hbv", "artifact_adc"]
SCENARIO_LABELS = {
    "ideal": "Ideal",
    "missing_t2w": "Miss T2W",
    "missing_hbv": "Miss HBV",
    "missing_adc": "Miss ADC",
    "artifact_t2w": "Art T2W",
    "artifact_hbv": "Art HBV",
    "artifact_adc": "Art ADC",
}

results_dir = Path("outputs/results")


def load_results():
    data = {}
    for model in MODELS:
        data[model] = {}
        for seed in SEEDS:
            fpath = results_dir / f"seed{seed}" / f"{model}_eval.json"
            if not fpath.exists():
                continue
            with open(fpath) as f:
                raw = json.load(f)
            data[model][seed] = {}
            for sc_name, sc_metrics in raw["scenarios"].items():
                m = dict(sc_metrics)
                m["score"] = (m.get("auroc", 0) + m.get("ap", 0)) / 2
                if "dice_positive" not in m and "dice" in m:
                    m["dice_positive"] = m["dice"]
                if "sensitivity_positive" not in m and "sensitivity" in m:
                    m["sensitivity_positive"] = m["sensitivity"]
                csens = m.get("case_sensitivity", 0)
                cspec = m.get("case_specificity", 0)
                m["case_sens_spec"] = (csens + cspec) / 2
                data[model][seed][sc_name] = m
    return data


def get_params(model):
    for seed in SEEDS:
        fpath = results_dir / f"seed{seed}" / f"{model}_eval.json"
        if fpath.exists():
            with open(fpath) as f:
                return json.load(f).get("params_M", "?")
    return "?"


def fmt(mean, std):
    return f"{mean:.4f}+/-{std:.4f}"


def get_vals(data, model, scenario, metric):
    seeds = [s for s in SEEDS if s in data[model] and scenario in data[model][s]]
    return [data[model][s][scenario].get(metric, 0) for s in seeds]


def print_ideal_table(data):
    metrics = [
        ("auroc", "AUROC"), ("ap", "AP"), ("score", "Score"),
        ("dice_positive", "Dice(+)"),
        ("sensitivity_positive", "Sens(+)"), ("specificity", "Spec"),
        ("case_sensitivity", "CaseSens"), ("case_specificity", "CaseSpec"),
    ]

    print("## Table 1: Ideal Scenario (mean +/- sd, N=5 seeds)")
    print()
    header = "| Model | Params |"
    sep = "| :--- | ---: |"
    for _, label in metrics:
        header += f" {label} |"
        sep += " ---: |"
    print(header)
    print(sep)

    for model in MODELS:
        label = MODEL_LABELS[model]
        params = get_params(model)
        row = f"| {label} | {params}M |"
        for metric_key, _ in metrics:
            vals = get_vals(data, model, "ideal", metric_key)
            if vals:
                row += f" {fmt(np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else 0)} |"
            else:
                row += " N/A |"
        print(row)
    print()


def print_robustness_table(data):
    print("## Table 2: Robustness -- Score (mean +/- sd)")
    print()

    header = "| Model |"
    sep = "| :--- |"
    for sc in SCENARIOS:
        header += f" {SCENARIO_LABELS[sc]} |"
        sep += " ---: |"
    print(header)
    print(sep)

    for model in MODELS:
        label = MODEL_LABELS[model]
        row = f"| {label} |"
        ideal_vals = get_vals(data, model, "ideal", "score")
        ideal_mean = np.mean(ideal_vals) if ideal_vals else 0

        for sc in SCENARIOS:
            vals = get_vals(data, model, sc, "score")
            if not vals:
                row += " N/A |"
                continue
            mean = np.mean(vals)
            std = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
            if sc == "ideal":
                row += f" {fmt(mean, std)} |"
            else:
                pct = ((mean - ideal_mean) / ideal_mean * 100) if ideal_mean else 0
                row += f" {fmt(mean, std)} ({pct:+.0f}%) |"
        print(row)
    print()


def print_per_seed_table(data):
    print("## Table 3: Per-Seed Score (Ideal)")
    print()
    header = "| Model |"
    sep = "| :--- |"
    for seed in SEEDS:
        header += f" seed={seed} |"
        sep += " ---: |"
    header += " Mean | SD |"
    sep += " ---: | ---: |"
    print(header)
    print(sep)

    for model in MODELS:
        label = MODEL_LABELS[model]
        row = f"| {label} |"
        vals = []
        for seed in SEEDS:
            if seed in data[model] and "ideal" in data[model][seed]:
                v = data[model][seed]["ideal"]["score"]
                vals.append(v)
                row += f" {v:.4f} |"
            else:
                row += " -- |"
        if vals:
            row += f" {np.mean(vals):.4f} | {np.std(vals, ddof=1) if len(vals) > 1 else 0:.4f} |"
        else:
            row += " -- | -- |"
        print(row)
    print()


def main():
    data = load_results()

    total = 0
    missing = 0
    for model in MODELS:
        for seed in SEEDS:
            total += 1
            if seed not in data[model]:
                missing += 1
                print(f"  MISSING: {model} seed={seed}")
    print(f"\nLoaded {total - missing}/{total} results ({missing} missing)\n")

    if missing > 0:
        print("WARNING: Some results are missing. Tables will be partial.\n")

    # Print to stdout and save
    old_stdout = sys.stdout
    sys.stdout = buf = io.StringIO()

    print("# AMPNet Multi-Seed Benchmark Results")
    print()
    print("**Training**: AdamW lr=5e-5, batch=8x4GPU, 300 epochs, seed in {42, 123, 456, 789, 1024}")
    print("**Data**: PI-CAI fold-0 (1200 train / 300 val), human expert labels")
    print("**Evaluation**: picai_eval (AUROC, AP, Score) + voxel/case-level metrics, 7 scenarios")
    print()
    print("---\n")
    print_ideal_table(data)
    print("---\n")
    print_robustness_table(data)
    print("---\n")
    print_per_seed_table(data)

    sys.stdout = old_stdout
    content = buf.getvalue()

    outpath = Path("outputs/tables/ampnet_multiseed_results.md")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        f.write(content)
    print(f"Saved to {outpath}\n")
    print(content)


if __name__ == "__main__":
    main()
