"""
Pure-PyTorch implementation of the Mamba (Selective State Space Model) for
3D medical image segmentation.

This module provides a functionally equivalent replacement for the native
``mamba-ssm`` CUDA package, using only standard PyTorch operations. It is
intended for environments where the CUDA-accelerated Mamba kernels cannot
be installed (e.g. CUDA version mismatch).

Reference
---------
Gu, A. & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with
Selective State Spaces*. arXiv:2312.00752.

Modules
-------
- SelectiveSSM      : Core selective state space model (the "Mamba layer").
- MambaBlock        : SelectiveSSM wrapped with LayerNorm and a residual
                      connection.
- MambaLayer        : A stack of MambaBlock modules.
- PatchEmbed3D      : 3-D patch embedding (volume -> sequence).
- PatchExpand3D     : 3-D patch expansion  (sequence -> volume, for decoder
                      upsampling).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parallel_scan(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Hillis-Steele parallel associative scan for linear recurrences.

    Computes h[t] = a[t] * h[t-1] + b[t] for all t, where h[-1] = 0.

    The associative operation is:
        (a2, b2) . (a1, b1) = (a2*a1, a2*b1 + b2)

    After the scan, a_out[t] = prod(a[0..t]) and b_out[t] = h[t].

    Parameters
    ----------
    a : Tensor, shape ``(..., L, N)``
        Multiplicative (decay) coefficients.
    b : Tensor, shape ``(..., L, N)``
        Additive (input) coefficients.

    Returns
    -------
    a_out, b_out : Tensors with same shape as inputs.
        a_out[t] is the cumulative product of decays.
        b_out[t] is the hidden state h[t] assuming h[-1] = 0.
    """
    L = a.shape[-2]
    num_steps = int(math.ceil(math.log2(max(L, 1))))

    for d in range(num_steps):
        stride = 1 << d
        if stride >= L:
            break
        # Shifted copies: positions < stride get identity (a=1, b=0)
        # Use F.pad instead of in-place slice assignment for clean autograd
        a_prev = F.pad(a[..., :-stride, :], (0, 0, stride, 0), value=1.0)
        b_prev = F.pad(b[..., :-stride, :], (0, 0, stride, 0), value=0.0)
        # Compose: (a, b) . (a_prev, b_prev)
        b = a * b_prev + b
        a = a * a_prev

    return a, b


def _parallel_selective_scan(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Parallel selective scan using associative scan with chunking.

    Uses the Hillis-Steele parallel prefix scan within chunks of
    ``chunk_size`` tokens (O(log chunk_size) parallel steps each),
    and carries hidden state across chunk boundaries sequentially.

    Parameters
    ----------
    x : Tensor, shape ``(B, D, L)``
    dt : Tensor, shape ``(B, D, L)``
    A : Tensor, shape ``(D, N)``
    B : Tensor, shape ``(B, N, L)``
    C : Tensor, shape ``(B, N, L)``
    D : Tensor, shape ``(D,)``
    chunk_size : int
        Tokens per parallel-scan chunk. Default 2048.

    Returns
    -------
    y : Tensor, shape ``(B, D, L)``
    """
    B_batch, D_dim, L = x.shape
    N = A.shape[1]
    device, dtype = x.device, x.dtype

    h = torch.zeros(B_batch, D_dim, N, device=device, dtype=dtype)
    ys = []

    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        cs = end - start

        # Slice chunk  -- keep (B, D/N, cs) layout
        x_c = x[:, :, start:end]       # (B, D, cs)
        dt_c = dt[:, :, start:end]     # (B, D, cs)
        B_c = B[:, :, start:end]       # (B, N, cs)
        C_c = C[:, :, start:end]       # (B, N, cs)

        # Compute per-timestep SSM coefficients
        # a_coeff[t] = exp(dt[t] * A)   decay
        # b_coeff[t] = dt[t] * B[t] * x[t]   input contribution
        # Reshape for (B, D, cs, N)
        dt_4 = dt_c.unsqueeze(-1)                    # (B, D, cs, 1)
        A_4 = A.unsqueeze(0).unsqueeze(2)            # (1, D, 1, N)
        a_coeff = torch.exp(dt_4 * A_4)              # (B, D, cs, N)

        B_4 = B_c.permute(0, 2, 1).unsqueeze(1)     # (B, 1, cs, N)
        x_4 = x_c.unsqueeze(-1)                     # (B, D, cs, 1)
        b_coeff = dt_4 * B_4 * x_4                   # (B, D, cs, N)

        # Parallel associative scan within this chunk (h[-1] = 0)
        a_scan, h_vals = _parallel_scan(a_coeff, b_coeff)  # (B,D,cs,N)

        # Incorporate carry from previous chunks:
        # true_h[t] = a_scan[t] * h_prev + h_vals[t]
        h_vals = a_scan * h.unsqueeze(2) + h_vals    # (B, D, cs, N)

        # Update carry for next chunk
        h = h_vals[:, :, -1]                          # (B, D, N)

        # Output: y[t] = sum_n(C[t,n] * h[t,n]) + D * x[t]
        C_4 = C_c.permute(0, 2, 1).unsqueeze(1)     # (B, 1, cs, N)
        y_c = (h_vals * C_4).sum(dim=-1)             # (B, D, cs)
        y_c = y_c + D.unsqueeze(0).unsqueeze(-1) * x_c

        ys.append(y_c)

    return torch.cat(ys, dim=2)


# ---------------------------------------------------------------------------
# SelectiveSSM
# ---------------------------------------------------------------------------

class SelectiveSSM(nn.Module):
    """Core Selective State Space Model (Mamba block internals).

    This implements the full Mamba selective-scan datapath in pure PyTorch:

    1. Input projection with expansion factor.
    2. Causal depth-wise 1-D convolution.
    3. SSM parameter projection (dt, B, C).
    4. Discretisation and selective scan.
    5. Gated output projection.

    Parameters
    ----------
    d_model : int
        Input / output feature dimension.
    d_state : int, optional
        SSM hidden-state dimension *N*.  Default ``16``.
    d_conv : int, optional
        Causal depthwise-conv kernel width.  Default ``4``.
    expand : int, optional
        Expansion factor for the inner dimension.  Default ``2``.
    chunk_size : int, optional
        Chunk length for the sequential scan.  Default ``256``.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        chunk_size: int = 256,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand
        self.chunk_size = chunk_size

        # --- input projection: d_model -> 2 * d_inner (x_proj and z) ---
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # --- causal depthwise conv on x_proj ---
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,   # causal: we will trim the right side
            groups=self.d_inner,
            bias=True,
        )

        # --- SSM parameter projections from x_proj ---
        # dt projection:  d_inner -> d_inner  (per-channel time-step)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        # B projection:   d_inner -> d_state
        self.B_proj = nn.Linear(self.d_inner, d_state, bias=False)
        # C projection:   d_inner -> d_state
        self.C_proj = nn.Linear(self.d_inner, d_state, bias=False)

        # --- SSM core parameters ---
        # A:  (d_inner, d_state)  -- log-spaced negative initialisation
        log_A = torch.log(
            torch.linspace(1, d_state, d_state, dtype=torch.float32)
        ).unsqueeze(0).expand(self.d_inner, -1).clone()
        self.A_log = nn.Parameter(log_A)  # stored in log-space

        # D (skip connection):  (d_inner,)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # --- output projection: d_inner -> d_model ---
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Initialise dt bias so that softplus(bias) ~ small positive value
        with torch.no_grad():
            # Uniform in [1.0, 2.0] before softplus -> ~[1.3, 2.1]
            self.dt_proj.bias.uniform_(1.0, 2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor, shape ``(B, L, d_model)``
            Input sequence.

        Returns
        -------
        Tensor, shape ``(B, L, d_model)``
            Output sequence.
        """
        B_batch, L, _ = x.shape

        # 1. Input projection ------------------------------------------
        xz = self.in_proj(x)                       # (B, L, 2*d_inner)
        x_proj, z = xz.chunk(2, dim=-1)            # each (B, L, d_inner)

        # 2. Causal depthwise conv + SiLU ------------------------------
        # Conv1d expects (B, C, L)
        x_proj = x_proj.transpose(1, 2).contiguous()   # (B, d_inner, L)
        x_proj = self.conv1d(x_proj)[:, :, :L]         # trim to causal
        x_proj = F.silu(x_proj)                         # (B, d_inner, L)

        # For SSM param projections we need (B, L, d_inner)
        x_proj_bl = x_proj.transpose(1, 2)             # (B, L, d_inner)

        # 3. SSM parameters --------------------------------------------
        dt = self.dt_proj(x_proj_bl)                    # (B, L, d_inner)
        dt = F.softplus(dt)                             # ensure positive
        dt = dt.transpose(1, 2).contiguous()            # (B, d_inner, L)

        B_param = self.B_proj(x_proj_bl)                # (B, L, N)
        B_param = B_param.transpose(1, 2).contiguous()  # (B, N, L)

        C_param = self.C_proj(x_proj_bl)                # (B, L, N)
        C_param = C_param.transpose(1, 2).contiguous()  # (B, N, L)

        # Recover A from log-space (negative values)
        A = -torch.exp(self.A_log)                      # (d_inner, N)

        # 4. Selective scan (parallel) ---------------------------------
        y = _parallel_selective_scan(
            x_proj, dt, A, B_param, C_param, self.D,
            chunk_size=self.chunk_size,
        )  # (B, d_inner, L)

        # 5. Gated output ---------------------------------------------
        y = y.transpose(1, 2)                           # (B, L, d_inner)
        y = y * F.silu(z)                               # gate
        out = self.out_proj(y)                           # (B, L, d_model)
        return out


# ---------------------------------------------------------------------------
# MambaBlock
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    """Single Mamba block: LayerNorm -> SelectiveSSM -> residual.

    Uses gradient checkpointing during training to reduce memory usage
    (the parallel scan creates large intermediate tensors).

    Parameters
    ----------
    d_model : int
        Feature dimension.
    d_state : int, optional
        SSM hidden state size.  Default ``16``.
    d_conv : int, optional
        Causal conv kernel width.  Default ``4``.
    expand : int, optional
        Expansion factor.  Default ``2``.
    dropout : float, optional
        Dropout probability applied after the SSM.  Default ``0.0``.
    chunk_size : int, optional
        Chunk length for the parallel scan.  Default ``2048``.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        chunk_size: int = 2048,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            chunk_size=chunk_size,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def _inner_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ssm(self.norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with pre-norm residual and gradient checkpointing.

        Parameters
        ----------
        x : Tensor, shape ``(B, L, d_model)``

        Returns
        -------
        Tensor, shape ``(B, L, d_model)``
        """
        if self.training and torch.is_grad_enabled():
            return x + self.dropout(
                torch.utils.checkpoint.checkpoint(
                    self._inner_forward, x, use_reentrant=False
                )
            )
        return x + self.dropout(self._inner_forward(x))


# ---------------------------------------------------------------------------
# MambaLayer
# ---------------------------------------------------------------------------

class MambaLayer(nn.Module):
    """Stack of :class:`MambaBlock` modules.

    Parameters
    ----------
    d_model : int
        Feature dimension.
    depth : int
        Number of :class:`MambaBlock` layers.
    d_state : int, optional
        SSM hidden state size.  Default ``16``.
    d_conv : int, optional
        Causal conv kernel width.  Default ``4``.
    expand : int, optional
        Expansion factor.  Default ``2``.
    dropout : float, optional
        Dropout probability for each block.  Default ``0.0``.
    chunk_size : int, optional
        Chunk length for the parallel scan.  Default ``2048``.
    """

    def __init__(
        self,
        d_model: int,
        depth: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        chunk_size: int = 2048,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
                chunk_size=chunk_size,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through all blocks followed by a final LayerNorm.

        Parameters
        ----------
        x : Tensor, shape ``(B, L, d_model)``
        **kwargs : ignored (accepts spatial_shape for interface compat)

        Returns
        -------
        Tensor, shape ``(B, L, d_model)``
        """
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# PatchEmbed3D
# ---------------------------------------------------------------------------

class PatchEmbed3D(nn.Module):
    """3-D patch embedding for volumetric (medical-imaging) data.

    Converts a 5-D volume ``[B, C_in, D, H, W]`` into a sequence
    ``[B, L, embed_dim]`` where ``L = (D // pd) * (H // ph) * (W // pw)``.

    The spatial grid dimensions are stored in ``self.grid_size`` after each
    forward call so that downstream modules (e.g. :class:`PatchExpand3D`)
    can reshape the sequence back to 3-D.

    Parameters
    ----------
    in_channels : int
        Number of input channels (e.g. 1 for CT, 4 for multi-modal MRI).
    embed_dim : int
        Embedding / feature dimension.
    patch_size : tuple of int, optional
        ``(pd, ph, pw)`` -- patch size along depth, height, width.
        Default ``(4, 4, 2)``.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: Union[Tuple[int, int, int], int] = (4, 4, 2),
    ) -> None:
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        # Stored after each forward so decoders can use them
        self.grid_size: Optional[Tuple[int, int, int]] = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """Forward pass.

        Parameters
        ----------
        x : Tensor, shape ``(B, C_in, D, H, W)``

        Returns
        -------
        tokens : Tensor, shape ``(B, L, embed_dim)``
            ``L = Gd * Gh * Gw`` where ``G* = spatial_dim // patch_dim``.
        grid_size : tuple of int
            ``(Gd, Gh, Gw)`` spatial grid dimensions.
        """
        # x: (B, C, D, H, W) -> (B, embed_dim, Gd, Gh, Gw)
        x = self.proj(x)
        B, C, Gd, Gh, Gw = x.shape
        self.grid_size = (Gd, Gh, Gw)
        # Flatten spatial dims -> sequence
        x = x.flatten(2).transpose(1, 2)  # (B, L, embed_dim)
        return x, (Gd, Gh, Gw)


# ---------------------------------------------------------------------------
# PatchExpand3D
# ---------------------------------------------------------------------------

class PatchExpand3D(nn.Module):
    """3-D patch expansion (inverse of :class:`PatchEmbed3D`).

    Reshapes a sequence ``[B, L, in_dim]`` back to a 3-D volume and
    upsamples via ``ConvTranspose3d`` to recover the original spatial
    resolution.

    Parameters
    ----------
    in_dim : int
        Input feature dimension (sequence channel size).
    out_dim : int
        Output feature dimension (volume channel size after expansion).
    patch_size : tuple of int, optional
        ``(pd, ph, pw)`` -- same patch size that was used for embedding.
        Default ``(4, 4, 2)``.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        patch_size: Union[Tuple[int, int, int], int] = (4, 4, 2),
    ) -> None:
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        self.patch_size = patch_size

        self.proj = nn.ConvTranspose3d(
            in_dim,
            out_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        grid_size: Tuple[int, int, int],
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor, shape ``(B, L, in_dim)``
            Sequence representation.
        grid_size : tuple of int
            ``(Gd, Gh, Gw)`` -- the spatial grid dimensions that were
            stored by :class:`PatchEmbed3D` during embedding.

        Returns
        -------
        Tensor, shape ``(B, out_dim, Gd*pd, Gh*ph, Gw*pw)``
            Upsampled 3-D volume.
        """
        Gd, Gh, Gw = grid_size
        B, L, C = x.shape
        assert L == Gd * Gh * Gw, (
            f"Sequence length {L} does not match grid {Gd}x{Gh}x{Gw} = "
            f"{Gd * Gh * Gw}"
        )
        # (B, L, C) -> (B, C, Gd, Gh, Gw)
        x = x.transpose(1, 2).reshape(B, C, Gd, Gh, Gw)
        x = self.proj(x)
        return x
