"""
Adaptive Multi-channel Mamba Fusion Network (Adaptive-MPNet)
for robust multi-modal prostate MRI segmentation.

Key innovations:
1. Independent Tokenizer: Each MRI sequence (T2W, HBV, ADC) is tokenized
   independently via channel-attention patch embedding.
2. Adaptive Modal State Gating: Quality-aware fusion that dynamically
   suppresses noisy/missing modalities and enhances reliable ones.
3. U-shaped Mamba backbone: Linear-complexity 3D volumetric processing.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .mamba_ssm import SelectiveSSM, MambaLayer, PatchEmbed3D


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ModalityTokenizer(nn.Module):
    """Independent patch embedding + channel attention for a single modality.

    Converts a single-channel 3D volume [B, 1, D, H, W] into a sequence of
    feature tokens [B, L, C] where L = (D/pd)*(H/ph)*(W/pw).
    """

    def __init__(self, in_channels: int, embed_dim: int,
                 patch_size: tuple[int, ...] = (4, 4, 2)):
        super().__init__()
        self.patch_embed = PatchEmbed3D(in_channels, embed_dim, patch_size)
        # Channel attention squeeze-excitation on the embedded tokens
        self.se_fc1 = nn.Linear(embed_dim, embed_dim // 4)
        self.se_fc2 = nn.Linear(embed_dim // 4, embed_dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, 1, D, H, W]
        Returns:
            tokens: [B, L, C]
            spatial_shape: tuple (Dp, Hp, Wp) for later reshaping
        """
        tokens, spatial_shape = self.patch_embed(x)  # [B, L, C]
        # Squeeze-excitation channel attention
        gap = tokens.mean(dim=1)  # [B, C]
        attn = torch.sigmoid(self.se_fc2(F.silu(self.se_fc1(gap))))  # [B, C]
        tokens = tokens * attn.unsqueeze(1)
        return tokens, spatial_shape


class ModalityQualityEstimator(nn.Module):
    """Estimates the quality/reliability of each modality's feature tokens.

    Produces a scalar quality score per modality per sample, used by the
    Adaptive Modal State Gating Unit to weight fusion.
    """

    def __init__(self, embed_dim: int, num_modalities: int = 3):
        super().__init__()
        self.quality_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.SiLU(),
            nn.Linear(embed_dim // 2, 1),
        )
        self.num_modalities = num_modalities

    def forward(self, token_list: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            token_list: list of [B, L, C] tensors, one per modality
        Returns:
            weights: [B, num_modalities, 1] softmax-normalized quality weights
        """
        scores = []
        for tokens in token_list:
            # Global average pool over sequence then estimate quality
            gap = tokens.mean(dim=1)  # [B, C]
            score = self.quality_net(gap)  # [B, 1]
            scores.append(score)
        scores = torch.cat(scores, dim=-1)  # [B, num_modalities]
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # [B, M, 1]
        return weights


class SimpleFusion(nn.Module):
    """Simple mean fusion (ablation: replaces AdaptiveModalGating).

    Takes the mean of modality tokens and applies a linear projection.
    No quality estimation, no gating — used for the 'w/o Adaptive Gating' ablation.
    """

    def __init__(self, embed_dim: int, num_modalities: int = 3):
        super().__init__()
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, token_list: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            token_list: list of [B, L, C] tensors (one per modality)
        Returns:
            fused: [B, L, C] fused token sequence
        """
        return self.proj(torch.stack(token_list, dim=1).mean(dim=1))


class Conv1DBlock(nn.Module):
    """Single 1D conv block with depthwise-separable conv + residual (ablation: replaces MambaBlock).

    LayerNorm → depthwise Conv1d(kernel=7) → SiLU → pointwise Conv1d → residual.
    """

    def __init__(self, d_model: int, kernel_size: int = 7, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        # Depthwise conv
        self.dw_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=d_model, bias=True,
        )
        self.act = nn.SiLU()
        # Pointwise conv
        self.pw_conv = nn.Conv1d(d_model, d_model, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, C]
        Returns:
            [B, L, C]
        """
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)  # [B, C, L]
        x = self.dw_conv(x)
        x = self.act(x)
        x = self.pw_conv(x)
        x = x.transpose(1, 2)  # [B, L, C]
        return residual + self.dropout(x)


class Conv1DLayer(nn.Module):
    """Stack of Conv1DBlock modules (ablation: replaces MambaLayer).

    Same interface as MambaLayer: [B, L, C] → [B, L, C].
    """

    def __init__(self, d_model: int, depth: int = 2, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            Conv1DBlock(d_model, dropout=dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class Conv3DBlock(nn.Module):
    """Single 3D conv block with depthwise-separable conv + residual (ablation: preserves 3D spatial structure).

    LayerNorm → reshape to [B,C,D,H,W] → depthwise Conv3d(kernel=3) → SiLU → pointwise Conv3d → flatten → residual.
    """

    def __init__(self, d_model: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dw_conv = nn.Conv3d(
            d_model, d_model, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=d_model, bias=True,
        )
        self.act = nn.SiLU()
        self.pw_conv = nn.Conv3d(d_model, d_model, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, spatial_shape: tuple[int, ...]) -> torch.Tensor:
        """
        Args:
            x: [B, L, C]
            spatial_shape: (D, H, W)
        Returns:
            [B, L, C]
        """
        residual = x
        B, L, C = x.shape
        D, H, W = spatial_shape
        x = self.norm(x)
        x = x.view(B, D, H, W, C).permute(0, 4, 1, 2, 3)  # [B, C, D, H, W]
        x = self.dw_conv(x)
        x = self.act(x)
        x = self.pw_conv(x)
        x = x.permute(0, 2, 3, 4, 1).reshape(B, L, C)  # [B, L, C]
        return residual + self.dropout(x)


class Conv3DLayer(nn.Module):
    """Stack of Conv3DBlock modules (ablation: replaces MambaLayer with 3D spatial processing).

    Same interface as MambaLayer: [B, L, C] → [B, L, C], but requires spatial_shape.
    """

    def __init__(self, d_model: int, depth: int = 2, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            Conv3DBlock(d_model, dropout=dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, spatial_shape: tuple[int, ...] | None = None, **kwargs) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, spatial_shape)
        return self.norm(x)


class AdaptiveModalGating(nn.Module):
    """Adaptive Modal State Gating Unit.

    Fuses multi-modal token streams by:
    1. Estimating quality of each modality's features.
    2. Applying quality-weighted summation.
    3. Passing through a gating projection for refined fusion.
    """

    def __init__(self, embed_dim: int, num_modalities: int = 3):
        super().__init__()
        self.quality_estimator = ModalityQualityEstimator(embed_dim, num_modalities)
        self.gate_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, token_list: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            token_list: list of [B, L, C] tensors (one per modality)
        Returns:
            fused: [B, L, C] fused token sequence
        """
        weights = self.quality_estimator(token_list)  # [B, M, 1]
        # Stack modalities: [B, M, L, C]
        stacked = torch.stack(token_list, dim=1)
        # Weighted sum: [B, L, C]
        fused = (stacked * weights.unsqueeze(2)).sum(dim=1)
        # Gate and project
        gate = self.gate_proj(fused)
        fused = self.out_proj(fused * gate)
        return fused


# ---------------------------------------------------------------------------
# Encoder / Decoder stages
# ---------------------------------------------------------------------------

class EncoderStage(nn.Module):
    """Single encoder stage: per-modality Mamba processing + adaptive fusion + downsampling."""

    def __init__(self, in_dim: int, out_dim: int, num_modalities: int,
                 depth: int, d_state: int, d_conv: int, expand: int,
                 dropout: float = 0.1,
                 use_adaptive_gating: bool = True,
                 backbone: str = "mamba"):
        super().__init__()
        self.num_modalities = num_modalities
        # Per-modality sequence processing layers
        if backbone == "mamba":
            self.modality_mamba = nn.ModuleList([
                MambaLayer(in_dim, depth=depth, d_state=d_state,
                           d_conv=d_conv, expand=expand, dropout=dropout)
                for _ in range(num_modalities)
            ])
        elif backbone == "conv3d":
            self.modality_mamba = nn.ModuleList([
                Conv3DLayer(in_dim, depth=depth, dropout=dropout)
                for _ in range(num_modalities)
            ])
        else:  # conv1d
            self.modality_mamba = nn.ModuleList([
                Conv1DLayer(in_dim, depth=depth, dropout=dropout)
                for _ in range(num_modalities)
            ])
        # Fusion
        if use_adaptive_gating:
            self.fusion = AdaptiveModalGating(in_dim, num_modalities)
        else:
            self.fusion = SimpleFusion(in_dim, num_modalities)
        # Downsample projection (spatial downsampling handled externally)
        self.downsample = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, token_list: list[torch.Tensor], spatial_shape: tuple[int, ...] | None = None):
        """
        Args:
            token_list: list of [B, L, C] per modality
            spatial_shape: (D, H, W) for Conv3D backbone; ignored by Mamba/Conv1D
        Returns:
            fused: [B, L, C_out] fused and downsampled
            skip: [B, L, C] pre-downsample for skip connection
        """
        # Process each modality independently
        processed = []
        for i, tokens in enumerate(token_list):
            processed.append(self.modality_mamba[i](tokens, spatial_shape=spatial_shape))
        # Adaptive fusion
        fused = self.fusion(processed)
        skip = self.norm(fused)
        # Downsample features
        out = self.downsample(skip)
        return out, skip


class SpatialDownsample(nn.Module):
    """Spatial 2x downsampling by merging 2x2x2 neighboring tokens."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.reduction = nn.Linear(in_dim * 8, out_dim)
        self.norm = nn.LayerNorm(in_dim * 8)

    def forward(self, x: torch.Tensor, spatial_shape: tuple[int, ...]):
        """
        Args:
            x: [B, L, C]
            spatial_shape: (D, H, W) current spatial dims
        Returns:
            x: [B, L//8, C_out]
            new_shape: (D//2, H//2, W//2)
        """
        D, H, W = spatial_shape
        B, L, C = x.shape
        x = x.view(B, D, H, W, C)
        # Pad if needed for even dimensions
        pad_d = D % 2
        pad_h = H % 2
        pad_w = W % 2
        if pad_d or pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
            D, H, W = D + pad_d, H + pad_h, W + pad_w
        # Merge 2x2x2 patches
        x = x.view(B, D // 2, 2, H // 2, 2, W // 2, 2, C)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        x = x.view(B, -1, 8 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x, (D // 2, H // 2, W // 2)


class SpatialUpsample(nn.Module):
    """Spatial 2x upsampling by expanding each token to 2x2x2 neighbors."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.expand = nn.Linear(in_dim, out_dim * 8)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, spatial_shape: tuple[int, ...]):
        """
        Args:
            x: [B, L, C]
            spatial_shape: (D, H, W) current spatial dims
        Returns:
            x: [B, L*8, C_out]
            new_shape: (D*2, H*2, W*2)
        """
        D, H, W = spatial_shape
        B, L, C = x.shape
        x = self.expand(x)  # [B, L, C_out*8]
        C_out = x.shape[-1] // 8
        x = x.view(B, D, H, W, 2, 2, 2, C_out)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        x = x.view(B, D * 2, H * 2, W * 2, C_out)
        x = x.view(B, -1, C_out)
        x = self.norm(x)
        return x, (D * 2, H * 2, W * 2)


class DecoderStage(nn.Module):
    """Single decoder stage: upsample + skip fusion + Mamba processing."""

    def __init__(self, in_dim: int, skip_dim: int, out_dim: int,
                 depth: int, d_state: int, d_conv: int, expand: int,
                 dropout: float = 0.1, backbone: str = "mamba"):
        super().__init__()
        self.upsample = SpatialUpsample(in_dim, out_dim)
        self.skip_proj = nn.Linear(skip_dim + out_dim, out_dim)
        if backbone == "mamba":
            self.mamba = MambaLayer(out_dim, depth=depth, d_state=d_state,
                                    d_conv=d_conv, expand=expand, dropout=dropout)
        elif backbone == "conv3d":
            self.mamba = Conv3DLayer(out_dim, depth=depth, dropout=dropout)
        else:  # conv1d
            self.mamba = Conv1DLayer(out_dim, depth=depth, dropout=dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor,
                spatial_shape: tuple[int, ...],
                target_shape: tuple[int, ...] | None = None):
        """
        Args:
            x: [B, L, C_in] from deeper stage
            skip: [B, L_skip, C_skip] skip connection
            spatial_shape: current spatial dims of x
            target_shape: expected output spatial dims (from encoder skip),
                          used to correct for downsample padding mismatches
        Returns:
            x: [B, L_out, C_out]
            out_shape: output spatial dims (target_shape if provided, else upsampled)
        """
        x, new_shape = self.upsample(x, spatial_shape)
        # Match sequence lengths (handle padding differences from downsampling)
        if x.shape[1] != skip.shape[1]:
            min_len = min(x.shape[1], skip.shape[1])
            x = x[:, :min_len]
            skip = skip[:, :min_len]
        x = self.skip_proj(torch.cat([x, skip], dim=-1))
        x = self.mamba(x, spatial_shape=target_shape if target_shape is not None else new_shape)
        x = self.norm(x)
        out_shape = target_shape if target_shape is not None else new_shape
        return x, out_shape


# ---------------------------------------------------------------------------
# Full Adaptive-MPNet
# ---------------------------------------------------------------------------

class AdaptiveMPNet(nn.Module):
    """Adaptive Multi-channel Mamba Fusion Network.

    U-shaped encoder-decoder with:
    - Independent modality tokenizers (no early concatenation)
    - Adaptive quality-aware modal fusion at each encoder stage
    - Mamba SSM backbone for linear-complexity 3D processing
    - Skip connections for fine-grained spatial recovery
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_modalities: int = 3,
        num_classes: int = 2,
        base_features: int = 48,
        num_stages: int = 4,
        patch_size: tuple[int, ...] = (4, 4, 2),
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        depths: tuple[int, ...] | None = None,
        use_adaptive_gating: bool = True,
        backbone: str = "mamba",
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.num_stages = num_stages
        self.patch_size = patch_size
        self.backbone = backbone

        if depths is None:
            depths = tuple([2] * num_stages)

        # Feature dimensions at each stage
        dims = [base_features * (2 ** i) for i in range(num_stages)]
        # dims = [48, 96, 192, 384]

        # --- Independent modality tokenizers ---
        self.tokenizers = nn.ModuleList([
            ModalityTokenizer(in_channels, dims[0], patch_size)
            for _ in range(num_modalities)
        ])

        # --- Encoder stages ---
        self.encoder_stages = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        for i in range(num_stages):
            self.encoder_stages.append(
                EncoderStage(
                    in_dim=dims[i],
                    out_dim=dims[i],
                    num_modalities=num_modalities,
                    depth=depths[i],
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                    use_adaptive_gating=use_adaptive_gating,
                    backbone=backbone,
                )
            )
            if i < num_stages - 1:
                self.downsamplers.append(
                    SpatialDownsample(dims[i], dims[i + 1])
                )

        # --- Bottleneck ---
        if backbone == "mamba":
            self.bottleneck = MambaLayer(
                dims[-1], depth=2, d_state=d_state,
                d_conv=d_conv, expand=expand, dropout=dropout
            )
        elif backbone == "conv3d":
            self.bottleneck = Conv3DLayer(dims[-1], depth=2, dropout=dropout)
        else:  # conv1d
            self.bottleneck = Conv1DLayer(dims[-1], depth=2, dropout=dropout)

        # --- Decoder stages ---
        self.decoder_stages = nn.ModuleList()
        for i in range(num_stages - 1, 0, -1):
            self.decoder_stages.append(
                DecoderStage(
                    in_dim=dims[i],
                    skip_dim=dims[i - 1],
                    out_dim=dims[i - 1],
                    depth=depths[i - 1],
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                    backbone=backbone,
                )
            )

        # --- Segmentation head ---
        self.seg_head = nn.Sequential(
            nn.LayerNorm(dims[0]),
            nn.Linear(dims[0], num_classes),
        )

        # --- Deep supervision auxiliary heads ---
        self.aux_heads = nn.ModuleList()
        for i in range(num_stages - 2):
            aux_dim = dims[num_stages - 2 - i]
            self.aux_heads.append(nn.Linear(aux_dim, num_classes))

        # Bias initialization for rare foreground prior
        self._init_seg_bias()

    def _init_seg_bias(self):
        """Initialize seg head bias so initial foreground prediction ~1%."""
        prior = 0.01
        bias_val = -math.log((1 - prior) / prior)  # ≈ -4.595
        # Main seg head
        head_layer = self.seg_head[-1]
        nn.init.constant_(head_layer.bias, 0)
        head_layer.bias.data[1] = bias_val
        # Auxiliary heads
        for aux in self.aux_heads:
            nn.init.constant_(aux.bias, 0)
            aux.bias.data[1] = bias_val

    def forward(self, modalities: list[torch.Tensor]):
        """
        Args:
            modalities: list of [B, 1, D, H, W] tensors, one per modality
                        (T2W, HBV, ADC). Missing modalities should be zero-filled.
        Returns:
            logits: [B, num_classes, D, H, W] segmentation logits
        """
        assert len(modalities) == self.num_modalities

        # --- Tokenize each modality independently ---
        token_lists = []
        spatial_shape = None
        for i, mod_input in enumerate(modalities):
            tokens, sp = self.tokenizers[i](mod_input)
            token_lists.append(tokens)
            if spatial_shape is None:
                spatial_shape = sp

        # --- Encoder ---
        skips = []
        spatial_shapes = [spatial_shape]
        current_tokens = token_lists  # list of [B, L, C] per modality

        for i, enc_stage in enumerate(self.encoder_stages):
            fused, skip = enc_stage(current_tokens, spatial_shape=spatial_shapes[-1])
            skips.append(skip)

            if i < self.num_stages - 1:
                fused, spatial_shape = self.downsamplers[i](fused, spatial_shapes[-1])
                spatial_shapes.append(spatial_shape)
                # Broadcast fused to all modality streams for next stage
                current_tokens = [fused.clone() for _ in range(self.num_modalities)]
            else:
                # Bottleneck
                fused = self.bottleneck(fused, spatial_shape=spatial_shapes[-1])

        # --- Decoder ---
        x = fused
        current_shape = spatial_shapes[-1]
        aux_outputs = []
        for i, dec_stage in enumerate(self.decoder_stages):
            skip_idx = self.num_stages - 2 - i
            # Pass the skip's original spatial shape to correct padding mismatches
            target_shape = spatial_shapes[skip_idx]
            x, current_shape = dec_stage(x, skips[skip_idx], current_shape,
                                         target_shape)

            # Deep supervision: collect auxiliary outputs (all but last stage)
            if self.training and i < len(self.aux_heads):
                aux_seq = self.aux_heads[i](x)  # [B, L, num_classes]
                B_a = aux_seq.shape[0]
                D_a, H_a, W_a = current_shape
                exp_a = D_a * H_a * W_a
                if aux_seq.shape[1] != exp_a:
                    aux_seq = aux_seq[:, :exp_a]
                aux_vol = aux_seq.view(B_a, D_a, H_a, W_a, -1)
                aux_vol = aux_vol.permute(0, 4, 1, 2, 3).contiguous()
                pd, ph, pw = self.patch_size
                aux_vol = F.interpolate(aux_vol, scale_factor=(pd, ph, pw),
                                        mode='trilinear', align_corners=False)
                aux_outputs.append(aux_vol)

        # --- Segmentation head ---
        logits_seq = self.seg_head(x)  # [B, L, num_classes]

        # Reshape to 3D volume
        B = logits_seq.shape[0]
        D_out, H_out, W_out = spatial_shapes[0]
        expected_len = D_out * H_out * W_out
        if logits_seq.shape[1] != expected_len:
            logits_seq = logits_seq[:, :expected_len]
        logits = logits_seq.view(B, D_out, H_out, W_out, -1)
        logits = logits.permute(0, 4, 1, 2, 3).contiguous()

        # Upsample to original input resolution via patch_size
        pd, ph, pw = self.patch_size
        logits = F.interpolate(
            logits,
            scale_factor=(pd, ph, pw),
            mode='trilinear',
            align_corners=False,
        )

        if self.training and aux_outputs:
            return logits, aux_outputs
        return logits


def build_adaptive_mpnet(config: dict) -> AdaptiveMPNet:
    """Factory function to build AdaptiveMPNet from config dict."""
    model_cfg = config['model']

    # Backbone selection: new "backbone" key takes priority over legacy "use_mamba"
    backbone = model_cfg.get('backbone', None)
    if backbone is None:
        use_mamba_flag = model_cfg.get('use_mamba', True)
        backbone = "mamba" if use_mamba_flag else "conv1d"

    return AdaptiveMPNet(
        in_channels=model_cfg.get('in_channels', 1),
        num_modalities=model_cfg.get('num_modalities', 3),
        num_classes=model_cfg.get('num_classes', 2),
        base_features=model_cfg.get('base_features', 48),
        num_stages=model_cfg.get('num_stages', 4),
        patch_size=tuple(model_cfg.get('patch_size', [4, 4, 2])),
        d_state=model_cfg.get('ssm_d_state', 16),
        d_conv=model_cfg.get('ssm_d_conv', 4),
        expand=model_cfg.get('ssm_expand', 2),
        dropout=model_cfg.get('dropout', 0.1),
        use_adaptive_gating=model_cfg.get('use_adaptive_gating', True),
        backbone=backbone,
    )
