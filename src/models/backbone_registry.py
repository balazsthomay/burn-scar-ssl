"""Backbone registry for segmentation models.

Provides a unified interface for creating different backbone architectures
(Prithvi, DINOv2, ResNet, etc.) with consistent output specifications.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch.nn as nn


@dataclass
class BackboneSpec:
    """Specification for a backbone architecture.

    Attributes:
        name: Unique identifier for the backbone.
        input_channels: Expected number of input channels.
        output_channels: List of channel counts for each output feature level.
        output_strides: List of downsampling factors for each output level.
        factory_fn: Callable that creates the backbone module.
    """

    name: str
    input_channels: int
    output_channels: list[int]
    output_strides: list[int]
    factory_fn: Callable[..., nn.Module]


# Global registry mapping backbone names to their specifications
BACKBONE_REGISTRY: dict[str, BackboneSpec] = {}


def register_backbone(
    name: str,
    input_channels: int,
    output_channels: list[int],
    output_strides: list[int],
) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
    """Decorator to register a backbone factory function.

    Args:
        name: Unique name for the backbone (e.g., "dinov2_base", "resnet50").
        input_channels: Number of input channels the backbone expects.
        output_channels: Channel dimensions for each feature map level.
        output_strides: Downsampling factors for each feature map level.

    Returns:
        Decorator that registers the factory function.

    Example:
        @register_backbone(
            name="resnet50",
            input_channels=6,
            output_channels=[256, 512, 1024, 2048],
            output_strides=[4, 8, 16, 32],
        )
        def create_resnet50(**kwargs) -> nn.Module:
            return ResNetBackbone(**kwargs)
    """

    def decorator(factory_fn: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        spec = BackboneSpec(
            name=name,
            input_channels=input_channels,
            output_channels=output_channels,
            output_strides=output_strides,
            factory_fn=factory_fn,
        )
        BACKBONE_REGISTRY[name] = spec
        return factory_fn

    return decorator


def get_backbone_spec(name: str) -> BackboneSpec:
    """Get the specification for a registered backbone.

    Args:
        name: Backbone name.

    Returns:
        BackboneSpec for the requested backbone.

    Raises:
        KeyError: If backbone is not registered.
    """
    if name not in BACKBONE_REGISTRY:
        available = list(BACKBONE_REGISTRY.keys())
        raise KeyError(f"Unknown backbone: {name}. Available: {available}")
    return BACKBONE_REGISTRY[name]


def create_backbone(name: str, **kwargs) -> nn.Module:
    """Create a backbone instance by name.

    Args:
        name: Registered backbone name.
        **kwargs: Arguments passed to the backbone factory.

    Returns:
        Instantiated backbone module.

    Raises:
        KeyError: If backbone is not registered.
    """
    spec = get_backbone_spec(name)
    return spec.factory_fn(**kwargs)


def list_backbones() -> list[str]:
    """List all registered backbone names.

    Returns:
        List of registered backbone names.
    """
    return list(BACKBONE_REGISTRY.keys())


def backbone_info(name: str) -> dict:
    """Get information about a backbone.

    Args:
        name: Backbone name.

    Returns:
        Dictionary with backbone information.
    """
    spec = get_backbone_spec(name)
    return {
        "name": spec.name,
        "input_channels": spec.input_channels,
        "output_channels": spec.output_channels,
        "output_strides": spec.output_strides,
        "n_feature_levels": len(spec.output_channels),
    }
