"""
Native 3D backbone architectures with AdaptiveMPNet's three key modules:
  1. Adaptive Modal State Gating (quality-aware multi-modal fusion)
  2. Deep Supervision (auxiliary heads at intermediate decoder stages)
  3. Modality Dropout (handled in training loop, not model)

Models:
  - AdaptiveNNUNet: nnU-Net style ConvBlock3D encoder-decoder
  - AdaptiveUNet: MONAI UNet-style (larger channel progression)
  - AdaptiveSwinUNETR: SwinUNETR with per-modality stems + adaptive gating

For AdaptiveConv3D: use AdaptiveMPNet(backbone="conv3d") from adaptive_mpnet.py.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Adaptive Modal Gating for 3D feature maps
# ---------------------------------------------------------------------------

class ModalityQualityEstimator3D(nn.Module):
    """Estimates quality score for each modality's 3D feature map."""

    def __init__(self, channels: int, num_modalities: int = 3):
        super().__init__()
        self.quality_nets = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.Linear(channels, channels // 4),
                nn.SiLU(),
                nn.Linear(channels // 4, 1),
            )
            for _ in range(num_modalities)
        ])

    def forward(self, feat_list: list[torch.Tensor]) -> torch.Tensor:
        scores = [net(f) for net, f in zip(self.quality_nets, feat_list)]
        scores = torch.cat(scores, dim=-1)  # [B, M]
        return F.softmax(scores, dim=-1)  # [B, M]


class AdaptiveModalGating3D(nn.Module):
    """Adaptive gating for 3D feature maps [B, C, D, H, W]."""

    def __init__(self, channels: int, num_modalities: int = 3):
        super().__init__()
        self.quality_estimator = ModalityQualityEstimator3D(channels, num_modalities)
        self.gate_conv = nn.Sequential(
            nn.Conv3d(channels, channels, 1),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 1),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Conv3d(channels, channels, 1)

    def forward(self, feat_list: list[torch.Tensor]) -> torch.Tensor:
        weights = self.quality_estimator(feat_list)  # [B, M]
        stacked = torch.stack(feat_list, dim=1)  # [B, M, C, D, H, W]
        w = weights[:, :, None, None, None, None]  # [B, M, 1, 1, 1, 1]
        fused = (stacked * w).sum(dim=1)  # [B, C, D, H, W]
        gate = self.gate_conv(fused)
        return self.out_proj(fused * gate)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

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


class PerModalityEncoderStage(nn.Module):
    """Per-modality conv processing + adaptive gating at one encoder level."""

    def __init__(self, in_ch: int, out_ch: int, num_modalities: int = 3):
        super().__init__()
        self.modality_convs = nn.ModuleList([
            ConvBlock3D(in_ch, out_ch) for _ in range(num_modalities)
        ])
        self.gating = AdaptiveModalGating3D(out_ch, num_modalities)

    def forward(self, feat_list: list[torch.Tensor]) -> torch.Tensor:
        processed = [conv(f) for conv, f in zip(self.modality_convs, feat_list)]
        return self.gating(processed)


# ---------------------------------------------------------------------------
# AdaptiveNNUNet
# ---------------------------------------------------------------------------

class AdaptiveNNUNet(nn.Module):
    """nnU-Net style architecture with per-modality encoders + adaptive gating + deep supervision.

    Same channel progression as NNUNetBaseline: [32, 64, 128, 256].
    Each encoder level has per-modality ConvBlock3D + AdaptiveModalGating3D.
    Shared decoder with skip connections and deep supervision.

    Args:
        use_adaptive_gating: If True (default), use per-modality encoding + adaptive gating.
            If False, concatenate modalities and use simple ConvBlock3D per level.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2,
                 num_modalities: int = 3, base_features: int = 32,
                 use_adaptive_gating: bool = True):
        super().__init__()
        f = base_features
        self.num_modalities = num_modalities
        self.use_adaptive_gating = use_adaptive_gating

        # Encoder stages
        if use_adaptive_gating:
            self.enc1 = PerModalityEncoderStage(in_channels, f, num_modalities)
            self.enc2 = PerModalityEncoderStage(f * 2, f * 2, num_modalities)
            self.enc3 = PerModalityEncoderStage(f * 4, f * 4, num_modalities)
        else:
            # Simple concat: [B, num_modalities*in_ch, D, H, W] → ConvBlock3D
            self.enc1 = ConvBlock3D(in_channels * num_modalities, f)
            self.enc2 = ConvBlock3D(f * 2, f * 2)
            self.enc3 = ConvBlock3D(f * 4, f * 4)

        self.down1 = nn.Conv3d(f, f * 2, 2, stride=2)
        self.down2 = nn.Conv3d(f * 2, f * 4, 2, stride=2)
        self.down3 = nn.Conv3d(f * 4, f * 8, 2, stride=2)

        # Bottleneck (fused, no per-modality)
        self.bottleneck = ConvBlock3D(f * 8, f * 8)

        # Shared decoder
        self.up3 = nn.ConvTranspose3d(f * 8, f * 4, 2, stride=2)
        self.dec3 = ConvBlock3D(f * 8, f * 4)
        self.up2 = nn.ConvTranspose3d(f * 4, f * 2, 2, stride=2)
        self.dec2 = ConvBlock3D(f * 4, f * 2)
        self.up1 = nn.ConvTranspose3d(f * 2, f, 2, stride=2)
        self.dec1 = ConvBlock3D(f * 2, f)

        # Segmentation head
        self.seg_head = nn.Conv3d(f, num_classes, 1)

        # Deep supervision auxiliary heads
        self.aux_head3 = nn.Conv3d(f * 4, num_classes, 1)
        self.aux_head2 = nn.Conv3d(f * 2, num_classes, 1)

        self._init_seg_bias()

    def _init_seg_bias(self):
        prior = 0.01
        bias_val = -math.log((1 - prior) / prior)
        for head in [self.seg_head, self.aux_head3, self.aux_head2]:
            nn.init.constant_(head.bias, 0)
            head.bias.data[1] = bias_val

    def forward(self, modalities: list[torch.Tensor]):
        # modalities: list of [B, 1, D, H, W]
        assert len(modalities) == self.num_modalities

        if self.use_adaptive_gating:
            # Per-modality encoding + adaptive gating
            e1 = self.enc1(modalities)  # [B, f, D, H, W]
            d1_feats = self.down1(e1)
            e2 = self.enc2([d1_feats] * self.num_modalities)
            d2_feats = self.down2(e2)
            e3 = self.enc3([d2_feats] * self.num_modalities)
        else:
            # Simple concat + single encoder
            x = torch.cat(modalities, dim=1)  # [B, 3, D, H, W]
            e1 = self.enc1(x)  # [B, f, D, H, W]
            d1_feats = self.down1(e1)
            e2 = self.enc2(d1_feats)
            d2_feats = self.down2(e2)
            e3 = self.enc3(d2_feats)

        d3_feats = self.down3(e3)

        bn = self.bottleneck(d3_feats)

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


# ---------------------------------------------------------------------------
# AdaptiveUNet
# ---------------------------------------------------------------------------

class AdaptiveUNet(nn.Module):
    """MONAI UNet-style architecture with per-modality encoders + adaptive gating + deep supervision.

    Inspired by MONAI UNet channel progression but with 5 levels (32→512)
    to keep parameter count reasonable with per-modality encoders.
    Uses anisotropic strides to handle D < H,W.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2,
                 num_modalities: int = 3):
        super().__init__()
        self.num_modalities = num_modalities
        channels = [32, 64, 128, 256, 512]
        # Anisotropic strides for D=16, H=64, W=64
        strides = [(2, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)]

        # Per-modality encoder stages
        self.enc_stages = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        for i in range(len(channels)):
            in_ch = in_channels if i == 0 else channels[i - 1]
            self.enc_stages.append(PerModalityEncoderStage(in_ch, channels[i], num_modalities))
            if i < len(channels) - 1:
                s = strides[i]
                self.down_convs.append(nn.Conv3d(channels[i], channels[i], s, stride=s))

        # Decoder stages
        self.up_convs = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        for i in range(len(channels) - 2, -1, -1):
            s = strides[i]
            self.up_convs.append(nn.ConvTranspose3d(channels[i + 1], channels[i], s, stride=s))
            self.dec_stages.append(ConvBlock3D(channels[i] * 2, channels[i]))

        # Seg head
        self.seg_head = nn.Conv3d(channels[0], num_classes, 1)

        # Deep supervision aux heads from early decoder stages (deepest first)
        # Decoder stage i outputs channels[len(channels)-2-i]
        num_aux = min(2, len(channels) - 1)
        self.aux_heads = nn.ModuleList([
            nn.Conv3d(channels[len(channels) - 2 - i], num_classes, 1)
            for i in range(num_aux)
        ])

        self._init_seg_bias()

    def _init_seg_bias(self):
        prior = 0.01
        bias_val = -math.log((1 - prior) / prior)
        nn.init.constant_(self.seg_head.bias, 0)
        self.seg_head.bias.data[1] = bias_val
        for head in self.aux_heads:
            nn.init.constant_(head.bias, 0)
            head.bias.data[1] = bias_val

    def forward(self, modalities: list[torch.Tensor]):
        assert len(modalities) == self.num_modalities

        # Encoder
        skips = []
        current = modalities
        for i, enc in enumerate(self.enc_stages):
            fused = enc(current)
            skips.append(fused)
            if i < len(self.down_convs):
                down = self.down_convs[i](fused)
                current = [down] * self.num_modalities

        # Decoder
        x = skips[-1]
        aux_outputs = []
        for i, (up, dec) in enumerate(zip(self.up_convs, self.dec_stages)):
            skip_idx = len(skips) - 2 - i
            skip = skips[skip_idx]
            x = up(x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
            x = dec(torch.cat([x, skip], dim=1))

            # Deep supervision from early decoder stages
            if self.training and i < len(self.aux_heads):
                aux_outputs.append(self.aux_heads[i](x))

        logits = self.seg_head(x)

        if self.training and aux_outputs:
            return logits, aux_outputs
        return logits


# ---------------------------------------------------------------------------
# AdaptiveSwinUNETR
# ---------------------------------------------------------------------------

class AdaptiveSwinUNETR(nn.Module):
    """SwinUNETR with per-modality stems + adaptive gating + deep supervision.

    Per-modality lightweight stems extract features, adaptive gating fuses them,
    then the fused features are processed by SwinUNETR encoder-decoder.
    Deep supervision heads are added at decoder intermediate stages.
    """

    def __init__(self, num_classes: int = 2, num_modalities: int = 3,
                 feature_size: int = 48, img_size: tuple = (16, 64, 64)):
        super().__init__()
        self.num_modalities = num_modalities

        # Per-modality stems: lightweight conv to extract features
        self.stems = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(1, 16, 3, padding=1, bias=False),
                nn.InstanceNorm3d(16),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Conv3d(16, 32, 3, padding=1, bias=False),
                nn.InstanceNorm3d(32),
                nn.LeakyReLU(0.01, inplace=True),
            )
            for _ in range(num_modalities)
        ])

        # Adaptive gating on stem features
        self.gating = AdaptiveModalGating3D(32, num_modalities)

        # Project gated features to SwinUNETR input channels
        self.proj = nn.Conv3d(32, 3, 1)

        # SwinUNETR backbone
        from monai.networks.nets import SwinUNETR
        try:
            self.swin = SwinUNETR(
                img_size=img_size,
                in_channels=3,
                out_channels=num_classes,
                feature_size=feature_size,
                spatial_dims=3,
                use_checkpoint=False,
            )
        except TypeError:
            # Newer MONAI versions don't accept img_size
            self.swin = SwinUNETR(
                in_channels=3,
                out_channels=num_classes,
                feature_size=feature_size,
                spatial_dims=3,
                use_checkpoint=False,
            )

        # Deep supervision: extract features from SwinUNETR decoder
        # SwinUNETR decoder outputs go through decoder stages with feature_size * {8, 4, 2, 1}
        self.aux_head1 = nn.Conv3d(feature_size * 2, num_classes, 1)
        self.aux_head2 = nn.Conv3d(feature_size * 4, num_classes, 1)

        self._init_seg_bias()

    def _init_seg_bias(self):
        prior = 0.01
        bias_val = -math.log((1 - prior) / prior)
        for head in [self.aux_head1, self.aux_head2]:
            nn.init.constant_(head.bias, 0)
            head.bias.data[1] = bias_val

    def _hook_decoder_features(self):
        """Register hooks to capture intermediate decoder features for deep supervision."""
        self._decoder_feats = {}

        def make_hook(name):
            def hook(module, input, output):
                self._decoder_feats[name] = output
            return hook

        # Hook into SwinUNETR decoder blocks
        # decoder5 outputs feature_size*8, decoder4: feature_size*4, decoder3: feature_size*2
        if hasattr(self.swin, 'decoder4'):
            self.swin.decoder4.register_forward_hook(make_hook('dec4'))
        if hasattr(self.swin, 'decoder3'):
            self.swin.decoder3.register_forward_hook(make_hook('dec3'))

    def forward(self, modalities: list[torch.Tensor]):
        assert len(modalities) == self.num_modalities

        # Per-modality stem processing
        stem_feats = [stem(mod) for stem, mod in zip(self.stems, modalities)]

        # Adaptive gating fusion
        fused = self.gating(stem_feats)  # [B, 32, D, H, W]

        # Project to SwinUNETR input
        x = self.proj(fused)  # [B, 3, D, H, W]

        # Register hooks for deep supervision (only during training)
        if self.training:
            self._decoder_feats = {}
            hooks = []
            if hasattr(self.swin, 'decoder4'):
                hooks.append(self.swin.decoder4.register_forward_hook(
                    lambda m, i, o: self._decoder_feats.update({'dec4': o})))
            if hasattr(self.swin, 'decoder3'):
                hooks.append(self.swin.decoder3.register_forward_hook(
                    lambda m, i, o: self._decoder_feats.update({'dec3': o})))

        logits = self.swin(x)

        if self.training:
            # Remove hooks
            for h in hooks:
                h.remove()

            aux_outputs = []
            if 'dec3' in self._decoder_feats:
                aux_outputs.append(self.aux_head1(self._decoder_feats['dec3']))
            if 'dec4' in self._decoder_feats:
                aux_outputs.append(self.aux_head2(self._decoder_feats['dec4']))
            if aux_outputs:
                return logits, aux_outputs

        return logits


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_adaptive_native(name: str, config: dict) -> nn.Module:
    """Build an adaptive native model by name.

    Args:
        name: one of 'adaptive_nnunet', 'adaptive_unet', 'adaptive_swinunetr'
        config: full experiment config dict
    Returns:
        model
    """
    model_cfg = config.get("model", {})
    num_classes = model_cfg.get("num_classes", 2)
    num_modalities = model_cfg.get("num_modalities", 3)
    data_cfg = config.get("data", {})
    patch_crop = data_cfg.get("patch_crop_size", [64, 64, 16])

    if name == "adaptive_nnunet":
        return AdaptiveNNUNet(
            in_channels=1, num_classes=num_classes,
            num_modalities=num_modalities,
            base_features=model_cfg.get("nnunet_base_features", 32),
            use_adaptive_gating=model_cfg.get("use_adaptive_gating", True),
        )
    elif name == "adaptive_unet":
        return AdaptiveUNet(
            in_channels=1, num_classes=num_classes,
            num_modalities=num_modalities,
        )
    elif name == "adaptive_swinunetr":
        # SwinUNETR needs D >= 32
        img_d = max(patch_crop[2], 32)
        img_size = (img_d, patch_crop[0], patch_crop[1])
        return AdaptiveSwinUNETR(
            num_classes=num_classes,
            num_modalities=num_modalities,
            feature_size=model_cfg.get("swin_feature_size", 48),
            img_size=img_size,
        )
    else:
        raise ValueError(f"Unknown adaptive native model: {name}")
