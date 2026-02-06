"""DINOv3 backbone with 6-band input support for satellite imagery.

Adapts the DINOv3-ViT-L/16 model (trained on SAT-493M) for 6-band HLS input
by replacing the 3-channel patch embedding with a 6-channel version.
"""

import torch
import torch.nn as nn
from transformers import AutoModel

from src.models.backbone_registry import register_backbone


def adapt_patch_embedding(model: nn.Module, in_channels: int = 6) -> None:
    """Replace patch embedding for N-channel input.

    Copies RGB weights to first 3 channels and zero-initializes the rest.
    This preserves pretrained features while allowing additional bands.

    Args:
        model: DINOv3 model to modify (in-place).
        in_channels: Number of input channels (default: 6 for HLS).
    """
    old_proj = model.embeddings.patch_embeddings

    # Get old projection parameters
    old_weight = old_proj.weight  # [out_channels, 3, kernel_h, kernel_w]
    old_bias = old_proj.bias if old_proj.bias is not None else None

    out_channels = old_weight.shape[0]
    kernel_size = old_weight.shape[2:]

    # Create new projection with more input channels
    new_proj = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=old_proj.stride,
        padding=old_proj.padding,
        bias=old_bias is not None,
    )

    # Initialize: copy RGB weights, zero-init extra channels
    with torch.no_grad():
        new_proj.weight.zero_()
        new_proj.weight[:, :3, :, :] = old_weight
        if old_bias is not None:
            new_proj.bias.copy_(old_bias)

    # Replace in model
    model.embeddings.patch_embeddings = new_proj


class DINOv3Backbone(nn.Module):
    """DINOv3 backbone with multi-scale feature extraction.

    Extracts features at multiple depths from the ViT and reshapes them
    to spatial feature maps for use with FPN-style decoders.

    The model outputs features at 4 scales by selecting intermediate layers
    and reshaping the token sequences back to 2D spatial grids.
    """

    # Layer indices for multi-scale features (ViT-L/16 has 24 layers)
    # hidden_states[0] is embeddings, [1] is layer 0 output, etc.
    # So we want layers 5, 11, 17, 23 -> hidden_states indices 6, 12, 18, 24
    FEATURE_INDICES = [6, 12, 18, 24]  # Evenly spaced through depth

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitl16-pretrain-sat493m",
        in_channels: int = 6,
        pretrained: bool = True,
    ):
        """Initialize DINOv3 backbone.

        Args:
            model_name: HuggingFace model identifier.
            in_channels: Number of input channels.
            pretrained: Whether to load pretrained weights.
        """
        super().__init__()

        self.in_channels = in_channels
        self.model_name = model_name

        # Load model
        if pretrained:
            self.model = AutoModel.from_pretrained(model_name)
        else:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name)
            self.model = AutoModel.from_config(config)

        # Ensure all parameters are trainable (HF loads in eval mode)
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True

        # Adapt for 6-band input
        if in_channels != 3:
            adapt_patch_embedding(self.model, in_channels)

        # Get embedding dimension from model config
        self.embed_dim = self.model.config.hidden_size  # 1024 for ViT-L
        self.patch_size = self.model.config.patch_size  # 16

        # Number of register tokens (DINOv3 uses 4)
        self.num_register_tokens = getattr(self.model.config, "num_register_tokens", 4)

    def _reshape_to_spatial(
        self, tokens: torch.Tensor, h: int, w: int
    ) -> torch.Tensor:
        """Reshape token sequence to spatial feature map.

        Args:
            tokens: Token tensor [B, N, C] including CLS and register tokens.
            h: Height in patches.
            w: Width in patches.

        Returns:
            Spatial feature map [B, C, H, W].
        """
        # Skip CLS token (1) and register tokens (typically 4)
        n_skip = 1 + self.num_register_tokens
        patch_tokens = tokens[:, n_skip:, :]

        # Reshape to spatial
        B, N, C = patch_tokens.shape
        spatial = patch_tokens.reshape(B, h, w, C).permute(0, 3, 1, 2)
        return spatial

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract multi-scale features.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            List of feature maps at 4 scales, all at stride 16 (ViT patch size).
            Each has shape [B, embed_dim, H/16, W/16].
        """
        B, C, H, W = x.shape

        # Compute spatial dimensions in patches
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size

        # Run forward with hidden states output
        outputs = self.model(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        # Select features at target indices and reshape to spatial
        spatial_features = [
            self._reshape_to_spatial(hidden_states[idx], h_patches, w_patches)
            for idx in self.FEATURE_INDICES
        ]

        return spatial_features

    def get_output_channels(self) -> list[int]:
        """Get output channel counts for each feature level."""
        return [self.embed_dim] * len(self.FEATURE_INDICES)

    def get_output_strides(self) -> list[int]:
        """Get output strides for each feature level."""
        # All features are at patch resolution (stride 16)
        return [self.patch_size] * len(self.FEATURE_INDICES)


# Register with backbone registry
@register_backbone(
    name="dinov3_vitl16_sat",
    input_channels=6,
    output_channels=[1024, 1024, 1024, 1024],  # ViT-L embed dim
    output_strides=[16, 16, 16, 16],  # All at patch resolution
)
def create_dinov3_vitl16_sat(
    in_channels: int = 6,
    pretrained: bool = True,
    **kwargs,
) -> DINOv3Backbone:
    """Create DINOv3-ViT-L/16 backbone pretrained on satellite imagery.

    Args:
        in_channels: Number of input channels.
        pretrained: Whether to load pretrained weights.

    Returns:
        DINOv3Backbone instance.
    """
    return DINOv3Backbone(
        model_name="facebook/dinov3-vitl16-pretrain-sat493m",
        in_channels=in_channels,
        pretrained=pretrained,
    )
