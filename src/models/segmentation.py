"""Unified segmentation model supporting multiple backbones.

Provides a backbone-agnostic segmentation architecture with FPN decoder
that works with DINOv3, ResNet-50, and other registered backbones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbone_registry import create_backbone, get_backbone_spec


class FPNDecoder(nn.Module):
    """Feature Pyramid Network decoder for semantic segmentation.

    Takes multi-scale features from a backbone and produces a single
    high-resolution feature map through lateral connections and top-down
    pathway.
    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: int = 256,
        output_stride: int = 4,
    ):
        """Initialize FPN decoder.

        Args:
            in_channels: List of input channel counts from backbone (low to high res).
            out_channels: Number of output channels for all FPN levels.
            output_stride: Target output stride (spatial resolution).
        """
        super().__init__()

        self.out_channels = out_channels
        self.output_stride = output_stride

        # Lateral connections (1x1 conv to reduce channels)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels
        ])

        # Smooth convs after addition
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels
        ])

        # Final fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * len(in_channels), out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Apply FPN decoder.

        Args:
            features: List of feature maps from backbone [low_res, ..., high_res].
                      Note: Assumes features are ordered from highest stride to lowest.

        Returns:
            Fused feature map at output_stride resolution.
        """
        # Build FPN from top (coarsest) to bottom (finest)
        laterals = [
            lateral_conv(f)
            for lateral_conv, f in zip(self.lateral_convs, features)
        ]

        # Top-down pathway with lateral connections
        # Start from coarsest level (last in list)
        for i in range(len(laterals) - 1, 0, -1):
            # Upsample coarser level to match finer level
            upsampled = F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            laterals[i - 1] = laterals[i - 1] + upsampled

        # Apply smooth convs
        outputs = [
            smooth_conv(lat)
            for smooth_conv, lat in zip(self.smooth_convs, laterals)
        ]

        # Upsample all to finest resolution and concatenate
        target_size = outputs[0].shape[2:]
        upsampled_outputs = [outputs[0]]
        for out in outputs[1:]:
            upsampled_outputs.append(
                F.interpolate(out, size=target_size, mode="bilinear", align_corners=False)
            )

        # Concatenate and fuse
        fused = torch.cat(upsampled_outputs, dim=1)
        return self.fusion(fused)


class SegmentationHead(nn.Module):
    """Segmentation head with optional dropout."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout: float = 0.1,
    ):
        """Initialize segmentation head.

        Args:
            in_channels: Number of input channels.
            num_classes: Number of output classes.
            dropout: Dropout rate before final conv.
        """
        super().__init__()

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply segmentation head."""
        x = self.dropout(x)
        return self.conv(x)


class SegmentationModel(nn.Module):
    """Backbone-agnostic segmentation model.

    Combines a registered backbone with FPN decoder and segmentation head.
    Supports any backbone that outputs multi-scale features.
    """

    def __init__(
        self,
        backbone_name: str,
        num_classes: int = 2,
        decoder_channels: int = 256,
        head_dropout: float = 0.1,
        pretrained: bool = True,
        **backbone_kwargs,
    ):
        """Initialize segmentation model.

        Args:
            backbone_name: Name of registered backbone.
            num_classes: Number of segmentation classes.
            decoder_channels: Number of channels in FPN decoder.
            head_dropout: Dropout rate for segmentation head.
            pretrained: Whether to load pretrained backbone weights.
            **backbone_kwargs: Additional arguments for backbone creation.
        """
        super().__init__()

        self.backbone_name = backbone_name
        self.num_classes = num_classes

        # Get backbone spec and create backbone
        spec = get_backbone_spec(backbone_name)
        self.backbone = create_backbone(
            backbone_name,
            pretrained=pretrained,
            **backbone_kwargs,
        )

        # Create decoder
        self.decoder = FPNDecoder(
            in_channels=spec.output_channels,
            out_channels=decoder_channels,
            output_stride=min(spec.output_strides),  # Target finest resolution
        )

        # Create head
        self.head = SegmentationHead(
            in_channels=decoder_channels,
            num_classes=num_classes,
            dropout=head_dropout,
        )

        # Store expected input/output info
        self.input_channels = spec.input_channels
        self.output_strides = spec.output_strides

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Logits tensor [B, num_classes, H, W] at input resolution.
        """
        # Extract multi-scale features
        features = self.backbone(x)

        # Decode
        decoded = self.decoder(features)

        # Segment
        logits = self.head(decoded)

        # Upsample to input resolution if needed
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(
                logits,
                size=x.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        return logits

    def get_backbone_params(self):
        """Get backbone parameters for differential learning rates."""
        return self.backbone.parameters()

    def get_decoder_params(self):
        """Get decoder parameters."""
        return list(self.decoder.parameters()) + list(self.head.parameters())


def create_segmentation_model(
    backbone_name: str,
    num_classes: int = 2,
    pretrained: bool = True,
    **kwargs,
) -> SegmentationModel:
    """Factory function for creating segmentation models.

    Args:
        backbone_name: Name of registered backbone ("dinov3_vitl16_sat", "resnet50").
        num_classes: Number of output classes.
        pretrained: Whether to use pretrained backbone.
        **kwargs: Additional model arguments.

    Returns:
        Configured SegmentationModel.
    """
    return SegmentationModel(
        backbone_name=backbone_name,
        num_classes=num_classes,
        pretrained=pretrained,
        **kwargs,
    )
