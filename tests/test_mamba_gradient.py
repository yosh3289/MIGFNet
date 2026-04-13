"""Gradient flow verification for Mamba parallel scan."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.models.mamba_ssm import _parallel_scan, SelectiveSSM, MambaBlock


def test_parallel_scan_gradient_flow():
    """Verify gradients flow through _parallel_scan to both inputs."""
    a = torch.randn(2, 4, 32, 16, requires_grad=True)
    b = torch.randn(2, 4, 32, 16, requires_grad=True)
    _, h = _parallel_scan(a, b)
    h.sum().backward()
    assert a.grad is not None and a.grad.norm() > 0, "No gradient for a"
    assert b.grad is not None and b.grad.norm() > 0, "No gradient for b"
    assert not a.grad.isnan().any(), "NaN in gradient for a"
    assert not b.grad.isnan().any(), "NaN in gradient for b"


def test_parallel_scan_with_checkpointing():
    """Verify gradients work with gradient checkpointing."""
    a = torch.randn(2, 4, 32, 16, requires_grad=True)
    b = torch.randn(2, 4, 32, 16, requires_grad=True)
    _, h = torch.utils.checkpoint.checkpoint(
        _parallel_scan, a, b, use_reentrant=False)
    h.sum().backward()
    assert a.grad is not None and a.grad.norm() > 0
    assert b.grad is not None and b.grad.norm() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_selective_ssm_all_params_get_gradients():
    """Every parameter in SelectiveSSM should receive a non-zero gradient."""
    device = torch.device("cuda")
    ssm = SelectiveSSM(d_model=48, d_state=16).to(device)
    x = torch.randn(2, 64, 48, device=device, requires_grad=True)
    with torch.amp.autocast("cuda"):
        y = ssm(x)
    y.float().sum().backward()
    for name, p in ssm.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"
        assert p.grad.norm() > 0, f"{name} has zero gradient"
        assert not p.grad.isnan().any(), f"{name} has NaN gradient"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mamba_block_gradient_with_checkpointing():
    """MambaBlock with gradient checkpointing should propagate gradients."""
    device = torch.device("cuda")
    block = MambaBlock(d_model=48, d_state=16).to(device)
    block.train()
    x = torch.randn(2, 64, 48, device=device, requires_grad=True)
    with torch.amp.autocast("cuda"):
        y = block(x)
    y.float().sum().backward()
    assert x.grad is not None and x.grad.norm() > 0
    for name, p in block.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"
