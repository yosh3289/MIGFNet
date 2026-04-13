"""
Baseline models for comparison with Adaptive-MPNet.

Baseline A: 3D nnU-Net style encoder-decoder with concat input
Baseline B: SwinUNETR (via MONAI)
Baseline C: Vanilla U-Mamba with concat input (no adaptive fusion)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba_ssm import MambaLayer, PatchEmbed3D


# ============================================================================
# Common: Multi-modal concat wrapper
# ============================================================================

class ConcatMultiModalWrapper(nn.Module):
    """Wraps a single-input model to accept multi-modal inputs via concatenation.

    Concatenates T2W, HBV, ADC along channel dimension before passing to the
    backbone. This is the traditional (non-adaptive) approach.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, modalities: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            modalities: list of [B, 1, D, H, W] tensors
        Returns:
            logits: [B, num_classes, D, H, W]
        """
        x = torch.cat(modalities, dim=1)  # [B, 3, D, H, W]
        return self.backbone(x)


# ============================================================================
# Baseline A: 3D nnU-Net style (simple ConvNet encoder-decoder)
# ============================================================================

class ConvBlock3D(nn.Module):
    """Double 3D convolution block with instance norm and LeakyReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class NNUNetBaseline(nn.Module):
    """3D nnU-Net style encoder-decoder (Baseline A).

    Simple 4-stage U-Net with 3D convolutions, strided conv downsampling,
    and transposed conv upsampling. Accepts 3-channel concatenated input.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 2,
                 base_features: int = 32):
        super().__init__()
        f = base_features
        # Encoder
        self.enc1 = ConvBlock3D(in_channels, f)
        self.down1 = nn.Conv3d(f, f * 2, 2, stride=2)
        self.enc2 = ConvBlock3D(f * 2, f * 2)
        self.down2 = nn.Conv3d(f * 2, f * 4, 2, stride=2)
        self.enc3 = ConvBlock3D(f * 4, f * 4)
        self.down3 = nn.Conv3d(f * 4, f * 8, 2, stride=2)

        # Bottleneck
        self.bottleneck = ConvBlock3D(f * 8, f * 8)

        # Decoder
        self.up3 = nn.ConvTranspose3d(f * 8, f * 4, 2, stride=2)
        self.dec3 = ConvBlock3D(f * 8, f * 4)
        self.up2 = nn.ConvTranspose3d(f * 4, f * 2, 2, stride=2)
        self.dec2 = ConvBlock3D(f * 4, f * 2)
        self.up1 = nn.ConvTranspose3d(f * 2, f, 2, stride=2)
        self.dec1 = ConvBlock3D(f * 2, f)

        self.seg_head = nn.Conv3d(f, num_classes, 1)

        # Deep supervision auxiliary heads
        self.aux_head3 = nn.Conv3d(f * 4, num_classes, 1)
        self.aux_head2 = nn.Conv3d(f * 2, num_classes, 1)

        # Bias initialization for rare foreground prior
        self._init_seg_bias()

    def _init_seg_bias(self):
        prior = 0.01
        bias_val = -math.log((1 - prior) / prior)
        for head in [self.seg_head, self.aux_head3, self.aux_head2]:
            nn.init.constant_(head.bias, 0)
            head.bias.data[1] = bias_val

    def forward(self, x: torch.Tensor):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        bn = self.bottleneck(self.down3(e3))

        # Decoder with skip connections
        d3 = self.up3(bn)
        if d3.shape != e3.shape:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='trilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        if d2.shape != e2.shape:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='trilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape != e1.shape:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='trilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        logits = self.seg_head(d1)

        if self.training:
            aux3 = self.aux_head3(d3)
            aux2 = self.aux_head2(d2)
            return logits, [aux3, aux2]
        return logits


# ============================================================================
# Baseline B: SwinUNETR (via MONAI)
# ============================================================================

def build_swin_unetr(in_channels: int = 3, num_classes: int = 2,
                     feature_size: int = 48) -> nn.Module:
    """Build SwinUNETR baseline using MONAI.

    Falls back to the nnU-Net baseline if MONAI's SwinUNETR is unavailable.
    """
    try:
        from monai.networks.nets import SwinUNETR
        model = SwinUNETR(
            in_channels=in_channels,
            out_channels=num_classes,
            feature_size=feature_size,
            spatial_dims=3,
            use_checkpoint=False,
        )
        return model
    except Exception as e:
        import logging
        logging.warning("SwinUNETR unavailable (%s), falling back to nnU-Net baseline", e)
        return NNUNetBaseline(in_channels, num_classes)


# ============================================================================
# Baseline C: Vanilla U-Mamba with concat input
# ============================================================================

class VanillaUMamba(nn.Module):
    """Vanilla U-Mamba: Mamba-based U-Net with concatenated multi-modal input.

    Same architecture concept as Adaptive-MPNet but WITHOUT:
    - Independent modality tokenizers
    - Adaptive modal gating
    Instead, simply concatenates all modalities at input.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 2,
                 base_features: int = 48, num_stages: int = 4,
                 patch_size: tuple = (4, 4, 2),
                 d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.num_stages = num_stages
        self.patch_size = patch_size

        dims = [base_features * (2 ** i) for i in range(num_stages)]

        # Single tokenizer for concatenated input
        self.patch_embed = PatchEmbed3D(in_channels, dims[0], patch_size)

        # Encoder Mamba stages
        self.encoder_stages = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        for i in range(num_stages):
            self.encoder_stages.append(
                MambaLayer(dims[i], depth=2, d_state=d_state,
                           d_conv=d_conv, expand=expand, dropout=dropout)
            )
            if i < num_stages - 1:
                self.downsamplers.append(
                    nn.Linear(dims[i] * 8, dims[i + 1])  # patch merging
                )

        # Bottleneck
        self.bottleneck = MambaLayer(
            dims[-1], depth=2, d_state=d_state,
            d_conv=d_conv, expand=expand, dropout=dropout
        )

        # Decoder stages
        self.decoder_stages = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        for i in range(num_stages - 1, 0, -1):
            self.upsamplers.append(nn.Linear(dims[i], dims[i - 1] * 8))
            self.decoder_stages.append(
                MambaLayer(dims[i - 1], depth=2, d_state=d_state,
                           d_conv=d_conv, expand=expand, dropout=dropout)
            )
        self.skip_projs = nn.ModuleList([
            nn.Linear(dims[i - 1] * 2, dims[i - 1])
            for i in range(num_stages - 1, 0, -1)
        ])

        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.LayerNorm(dims[0]),
            nn.Linear(dims[0], num_classes),
        )

        # Deep supervision auxiliary heads
        self.aux_heads = nn.ModuleList()
        for i in range(num_stages - 2):
            aux_dim = dims[num_stages - 2 - i]
            self.aux_heads.append(nn.Linear(aux_dim, num_classes))

        # Bias initialization for rare foreground prior
        self._init_seg_bias()

    def _init_seg_bias(self):
        prior = 0.01
        bias_val = -math.log((1 - prior) / prior)
        head_layer = self.seg_head[-1]
        nn.init.constant_(head_layer.bias, 0)
        head_layer.bias.data[1] = bias_val
        for aux in self.aux_heads:
            nn.init.constant_(aux.bias, 0)
            aux.bias.data[1] = bias_val

    def _downsample(self, x, spatial_shape, stage_idx):
        D, H, W = spatial_shape
        B, L, C = x.shape
        x = x.view(B, D, H, W, C)
        pad_d, pad_h, pad_w = D % 2, H % 2, W % 2
        if pad_d or pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
            D, H, W = D + pad_d, H + pad_h, W + pad_w
        x = x.view(B, D // 2, 2, H // 2, 2, W // 2, 2, C)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        x = x.view(B, -1, 8 * C)
        x = self.downsamplers[stage_idx](x)
        return x, (D // 2, H // 2, W // 2)

    def _upsample(self, x, spatial_shape, stage_idx):
        D, H, W = spatial_shape
        B, L, C = x.shape
        x = self.upsamplers[stage_idx](x)
        C_out = x.shape[-1] // 8
        x = x.view(B, D, H, W, 2, 2, 2, C_out)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        x = x.view(B, -1, C_out)
        return x, (D * 2, H * 2, W * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, D, H, W] concatenated multi-modal input
        Returns:
            logits: [B, num_classes, D, H, W]
        """
        tokens, spatial_shape = self.patch_embed(x)

        # Encoder
        skips = []
        shapes = [spatial_shape]
        for i, enc in enumerate(self.encoder_stages):
            tokens = enc(tokens)
            skips.append(tokens)
            if i < self.num_stages - 1:
                tokens, spatial_shape = self._downsample(tokens, shapes[-1], i)
                shapes.append(spatial_shape)

        # Bottleneck
        tokens = self.bottleneck(tokens)

        # Decoder
        current_shape = shapes[-1]
        aux_outputs = []
        for i, (dec, skip_proj) in enumerate(zip(self.decoder_stages, self.skip_projs)):
            tokens, current_shape = self._upsample(tokens, current_shape, i)
            skip_idx = self.num_stages - 2 - i
            skip = skips[skip_idx]
            if tokens.shape[1] != skip.shape[1]:
                min_len = min(tokens.shape[1], skip.shape[1])
                tokens = tokens[:, :min_len]
                skip = skip[:, :min_len]
            # Correct shape to match skip's original spatial shape (handles
            # padding mismatches from downsampling when a dimension was odd)
            current_shape = shapes[skip_idx]
            tokens = skip_proj(torch.cat([tokens, skip], dim=-1))
            tokens = dec(tokens)

            # Deep supervision: collect auxiliary outputs (all but last stage)
            if self.training and i < len(self.aux_heads):
                aux_seq = self.aux_heads[i](tokens)
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

        # Head
        logits_seq = self.seg_head(tokens)
        B = logits_seq.shape[0]
        D, H, W = shapes[0]
        expected = D * H * W
        if logits_seq.shape[1] != expected:
            logits_seq = logits_seq[:, :expected]
        logits = logits_seq.view(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()

        pd, ph, pw = self.patch_size
        logits = F.interpolate(logits, scale_factor=(pd, ph, pw),
                               mode='trilinear', align_corners=False)

        if self.training and aux_outputs:
            return logits, aux_outputs
        return logits


# ============================================================================
# Factory functions
# ============================================================================

def build_baseline(name: str, config: dict) -> nn.Module:
    """Build a baseline model by name.

    Args:
        name: one of 'nnUNet_concat', 'SwinUNETR', 'UMamba_concat', 'OfficialUNet'
        config: full experiment config dict
    Returns:
        model wrapped with ConcatMultiModalWrapper (accepts list of modalities)
    """
    model_cfg = config.get("model", {})
    num_classes = model_cfg.get("num_classes", 2)
    data_cfg = config.get("data", {})
    # Use patch_crop_size if available (v4+), else fall back to crop_size
    crop_size = data_cfg.get("patch_crop_size", data_cfg.get("crop_size", [128, 128, 32]))
    img_size = (crop_size[2], crop_size[0], crop_size[1])  # D, H, W

    if name == "nnUNet_concat":
        backbone = NNUNetBaseline(in_channels=3, num_classes=num_classes)
    elif name == "SwinUNETR":
        backbone = build_swin_unetr(
            in_channels=3, num_classes=num_classes,
            feature_size=model_cfg.get("base_features", 48),
        )
    elif name == "UMamba_concat":
        backbone = VanillaUMamba(
            in_channels=3, num_classes=num_classes,
            base_features=model_cfg.get("base_features", 48),
            num_stages=model_cfg.get("num_stages", 4),
            patch_size=tuple(model_cfg.get("patch_size", [4, 4, 2])),
            d_state=model_cfg.get("ssm_d_state", 16),
            d_conv=model_cfg.get("ssm_d_conv", 4),
            expand=model_cfg.get("ssm_expand", 2),
            dropout=model_cfg.get("dropout", 0.1),
        )
    elif name == "OfficialUNet":
        from monai.networks.nets import UNet
        backbone = UNet(
            spatial_dims=3,
            in_channels=3,
            out_channels=num_classes,
            channels=(32, 64, 128, 256, 512, 1024),
            strides=((2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)),
            num_res_units=0,
        )
    else:
        raise ValueError(f"Unknown baseline: {name}")

    return ConcatMultiModalWrapper(backbone)
