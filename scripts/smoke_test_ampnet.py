"""
Smoke test: forward + backward for 3 AMPNet variants.
Verifies shapes, gradient flow, and param counts.

Models:
  adaptive_nnunet  → AMPNet-nnUNet (AdaptiveNNUNet)
  adaptive_unet    → AMPNet-UNet (AdaptiveUNet)
  AdaptiveMPNet    → AMPNet-Mamba (AdaptiveMPNet, backbone=mamba)

Usage: python smoke_test_ampnet.py
"""

import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.models.adaptive_native import build_adaptive_native
from src.models.adaptive_mpnet import build_adaptive_mpnet


def smoke_test():
    config_path = Path(__file__).parent / "configs" / "adaptive_mpnet.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    B = 2
    D, H, W = 16, 64, 64

    models_to_test = [
        ("adaptive_nnunet", "AMPNet-nnUNet"),
        ("adaptive_unet", "AMPNet-UNet"),
        ("AdaptiveMPNet", "AMPNet-Mamba"),
    ]

    results = []
    all_pass = True

    for name, label in models_to_test:
        print(f"\n{'='*60}")
        print(f"Testing: {label} ({name})")
        print(f"{'='*60}")

        try:
            if name == "AdaptiveMPNet":
                model = build_adaptive_mpnet(config).to(device)
            else:
                model = build_adaptive_native(name, config).to(device)
            model.train()

            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Parameters: {param_count/1e6:.2f}M")

            # Create dummy input (3 modalities)
            t2w = torch.randn(B, 1, D, H, W, device=device)
            hbv = torch.randn(B, 1, D, H, W, device=device)
            adc = torch.randn(B, 1, D, H, W, device=device)

            # Forward
            outputs = model([t2w, hbv, adc])

            if isinstance(outputs, (tuple, list)) and len(outputs) == 2:
                logits, aux_list = outputs
                print(f"  Output: logits {logits.shape}, {len(aux_list)} aux heads")
                for i, aux in enumerate(aux_list):
                    print(f"    aux[{i}]: {aux.shape}")
            else:
                logits = outputs
                aux_list = []
                print(f"  Output: logits {logits.shape} (no deep supervision)")

            assert logits.shape[0] == B, f"Batch mismatch: {logits.shape[0]} != {B}"
            assert logits.shape[1] == 2, f"Classes mismatch: {logits.shape[1]} != 2"
            print(f"  Output spatial: {logits.shape[2:]} (input was [{D}, {H}, {W}])")

            # Backward
            loss = logits.mean()
            if aux_list:
                for aux in aux_list:
                    loss = loss + aux.mean() * 0.1
            loss.backward()

            # Check gradients
            has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                          for p in model.parameters() if p.requires_grad)
            assert has_grad, "No gradients!"
            print(f"  Gradients: OK")

            results.append((label, param_count / 1e6, "PASS"))
            print(f"  Status: PASS")

        except Exception as e:
            results.append((label, 0, f"FAIL: {e}"))
            print(f"  Status: FAIL — {e}")
            import traceback
            traceback.print_exc()
            all_pass = False

        # Free memory
        if 'model' in dir():
            del model
        torch.cuda.empty_cache()

    # Summary
    print(f"\n\n{'='*60}")
    print("SMOKE TEST SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Params':>10} {'Status':>10}")
    print("-" * 45)
    for label, params, status in results:
        params_str = f"{params:.2f}M" if params > 0 else "—"
        print(f"{label:<20} {params_str:>10} {status:>10}")
    print(f"{'='*60}")

    if all_pass:
        print("\nAll 3 AMPNet models passed!")
    else:
        print("\nSome models FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    smoke_test()
