#!/usr/bin/env python3
"""Smoke test for 6 AMPNet-nnUNet ablation configs."""

import torch
import yaml
from src.models.adaptive_native import build_adaptive_native

CONFIGS = [
    ("A1 w/o Gating",    "configs/ablation_nnunet_A1_no_gating.yaml"),
    ("A2 w/o DeepSup",   "configs/ablation_nnunet_A2_no_deepsup.yaml"),
    ("A3 w/o ModDrop",   "configs/ablation_nnunet_A3_no_moddrop.yaml"),
    ("A4 Gating Only",   "configs/ablation_nnunet_A4_gating_only.yaml"),
    ("A5 DeepSup Only",  "configs/ablation_nnunet_A5_deepsup_only.yaml"),
    ("A6 ModDrop Only",  "configs/ablation_nnunet_A6_moddrop_only.yaml"),
]

B, D, H, W = 2, 16, 64, 64
device = "cuda" if torch.cuda.is_available() else "cpu"

results = []
for label, cfg_path in CONFIGS:
    print(f"\n{'='*60}")
    print(f"Testing: {label} ({cfg_path})")
    print(f"{'='*60}")
    try:
        config = yaml.safe_load(open(cfg_path))
        model = build_adaptive_native("adaptive_nnunet", config).to(device)
        nparams = sum(p.numel() for p in model.parameters()) / 1e6

        modalities = [torch.randn(B, 1, D, H, W, device=device) for _ in range(3)]
        label_t = torch.randint(0, 2, (B, 1, D, H, W), device=device).float()

        # Forward
        model.train()
        out = model(modalities)
        if isinstance(out, tuple):
            logits, aux = out
            aux_info = f"{len(aux)} aux heads"
        else:
            logits = out
            aux_info = "no aux (deep_sup off)"

        print(f"  Params: {nparams:.2f}M")
        print(f"  Output: {logits.shape}, {aux_info}")
        print(f"  Gating: {model.use_adaptive_gating}")

        # Backward
        loss = logits.mean()
        loss.backward()
        print(f"  Backward: OK")
        print(f"  Status: PASS")
        results.append((label, nparams, "PASS"))
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback; traceback.print_exc()
        results.append((label, 0, "FAIL"))

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"{'Config':<25} {'Params':>8} {'Status':>8}")
print("-" * 45)
for label, params, status in results:
    print(f"{label:<25} {params:>7.2f}M {status:>8}")

passed = sum(1 for _, _, s in results if s == "PASS")
print(f"\n{passed}/{len(results)} passed")
