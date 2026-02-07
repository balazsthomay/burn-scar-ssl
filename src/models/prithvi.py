"""Prithvi-EO-2.0 model factory for burn scar segmentation."""

from typing import Literal

import torch
from terratorch.models import EncoderDecoderFactory


# Band configuration for HLS Burn Scars
BACKBONE_BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]

# Neck indices for different Prithvi model sizes
NECK_INDICES = {
    "prithvi_eo_v1_100": [2, 5, 8, 11],
    "prithvi_eo_v2_300": [5, 11, 17, 23],
    "prithvi_eo_v2_300_tl": [5, 11, 17, 23],
    "prithvi_eo_v2_600": [7, 15, 23, 31],
    "prithvi_eo_v2_600_tl": [7, 15, 23, 31],
}

BackboneType = Literal[
    "prithvi_eo_v1_100",
    "prithvi_eo_v2_300",
    "prithvi_eo_v2_300_tl",
    "prithvi_eo_v2_600",
    "prithvi_eo_v2_600_tl",
]


def build_prithvi_segmentation_model(
    backbone: BackboneType = "prithvi_eo_v2_300",
    pretrained: bool = True,
    num_classes: int = 2,
    img_size: int = 512,
    decoder_channels: list[int] | None = None,
    head_dropout: float = 0.1,
    drop_path_rate: float = 0.0,
    peft_config: dict | None = None,
) -> torch.nn.Module:
    """Build a Prithvi-based segmentation model.

    Creates a model using TerraTorch's EncoderDecoderFactory with the Prithvi-EO
    backbone, standard necks for ViT-to-spatial conversion, and UNet decoder.

    Args:
        backbone: Prithvi backbone variant to use.
        pretrained: Whether to load pretrained weights.
        num_classes: Number of output segmentation classes.
        img_size: Input image size (height and width).
        decoder_channels: UNet decoder channel configuration.
            Defaults to [512, 256, 128, 64].
        head_dropout: Dropout rate for the segmentation head.
        drop_path_rate: Stochastic depth rate for backbone.
        peft_config: Optional PEFT configuration dict for parameter-efficient
            fine-tuning. Passed directly to EncoderDecoderFactory.build_model().
            Expected keys: "method" (e.g. "LORA"), "replace_qkv" (e.g. "qkv"),
            "peft_config_kwargs" (e.g. {"r": 8, "lora_alpha": 16, ...}).

    Returns:
        Configured segmentation model.
    """
    if decoder_channels is None:
        decoder_channels = [512, 256, 128, 64]

    # Get neck indices for the selected backbone
    if backbone not in NECK_INDICES:
        raise ValueError(
            f"Unknown backbone: {backbone}. "
            f"Supported: {list(NECK_INDICES.keys())}"
        )
    neck_indices = NECK_INDICES[backbone]

    factory = EncoderDecoderFactory()

    model = factory.build_model(
        task="segmentation",
        # Backbone configuration
        backbone=backbone,
        backbone_pretrained=pretrained,
        backbone_num_frames=1,
        backbone_img_size=img_size,
        backbone_bands=BACKBONE_BANDS,
        backbone_drop_path_rate=drop_path_rate,
        # Neck configuration - transforms ViT tokens to spatial format
        necks=[
            {"name": "SelectIndices", "indices": neck_indices},
            {"name": "ReshapeTokensToImage"},
            {"name": "LearnedInterpolateToPyramidal"},
        ],
        # Decoder configuration
        decoder="UNetDecoder",
        decoder_channels=decoder_channels,
        # Head configuration
        head_dropout=head_dropout,
        num_classes=num_classes,
        # PEFT configuration (LoRA/DoRA)
        peft_config=peft_config,
    )

    return model


def get_model_info(model: torch.nn.Module) -> dict:
    """Get information about a model.

    Args:
        model: The model to inspect.

    Returns:
        Dictionary with model information including parameter counts.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": total_params - trainable_params,
        "total_parameters_millions": total_params / 1e6,
        "trainable_parameters_millions": trainable_params / 1e6,
    }


def freeze_backbone(model: torch.nn.Module) -> None:
    """Freeze the backbone parameters of a model.

    This is useful for transfer learning when you want to only train
    the decoder and head.

    Args:
        model: Model with encoder attribute to freeze.
    """
    if hasattr(model, "encoder"):
        for param in model.encoder.parameters():
            param.requires_grad = False
    else:
        raise ValueError("Model does not have an 'encoder' attribute to freeze")


def unfreeze_backbone(model: torch.nn.Module) -> None:
    """Unfreeze the backbone parameters of a model.

    Args:
        model: Model with encoder attribute to unfreeze.
    """
    if hasattr(model, "encoder"):
        for param in model.encoder.parameters():
            param.requires_grad = True
    else:
        raise ValueError("Model does not have an 'encoder' attribute to unfreeze")
