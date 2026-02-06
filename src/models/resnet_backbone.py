"""ResNet-50 backbone with 6-band input support.

Adapts ImageNet-pretrained ResNet-50 for 6-band satellite imagery
by replacing the first conv layer.
"""

import torch
import torch.nn as nn
from torchvision import models

from src.models.backbone_registry import register_backbone


def adapt_first_conv(model: nn.Module, in_channels: int = 6) -> None:
    """Replace first conv layer for N-channel input.

    Copies RGB weights to first 3 channels and zero-initializes the rest.

    Args:
        model: ResNet model to modify (in-place).
        in_channels: Number of input channels (default: 6 for HLS).
    """
    old_conv = model.conv1

    # Create new conv with more input channels
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )

    # Initialize: copy RGB weights, zero-init extra channels
    with torch.no_grad():
        new_conv.weight.zero_()
        new_conv.weight[:, :3, :, :] = old_conv.weight
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    model.conv1 = new_conv


class ResNetBackbone(nn.Module):
    """ResNet-50 backbone with multi-scale feature extraction.

    Extracts features from ResNet's 4 stages at different resolutions
    for use with FPN-style decoders.
    """

    def __init__(
        self,
        in_channels: int = 6,
        pretrained: bool = True,
    ):
        """Initialize ResNet-50 backbone.

        Args:
            in_channels: Number of input channels.
            pretrained: Whether to load ImageNet pretrained weights.
        """
        super().__init__()

        self.in_channels = in_channels

        # Load ResNet-50
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = models.resnet50(weights=weights)

        # Adapt for 6-band input
        if in_channels != 3:
            adapt_first_conv(resnet, in_channels)

        # Split into stages for multi-scale extraction
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.layer1 = resnet.layer1  # stride 4, 256 channels
        self.layer2 = resnet.layer2  # stride 8, 512 channels
        self.layer3 = resnet.layer3  # stride 16, 1024 channels
        self.layer4 = resnet.layer4  # stride 32, 2048 channels

        self.output_channels = [256, 512, 1024, 2048]
        self.output_strides = [4, 8, 16, 32]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract multi-scale features.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            List of 4 feature maps at strides 4, 8, 16, 32.
        """
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        return [f1, f2, f3, f4]

    def get_output_channels(self) -> list[int]:
        """Get output channel counts for each feature level."""
        return self.output_channels

    def get_output_strides(self) -> list[int]:
        """Get output strides for each feature level."""
        return self.output_strides


# Register with backbone registry
@register_backbone(
    name="resnet50",
    input_channels=6,
    output_channels=[256, 512, 1024, 2048],
    output_strides=[4, 8, 16, 32],
)
def create_resnet50(
    in_channels: int = 6,
    pretrained: bool = True,
    **kwargs,
) -> ResNetBackbone:
    """Create ResNet-50 backbone with ImageNet pretraining.

    Args:
        in_channels: Number of input channels.
        pretrained: Whether to load pretrained weights.

    Returns:
        ResNetBackbone instance.
    """
    return ResNetBackbone(
        in_channels=in_channels,
        pretrained=pretrained,
    )
