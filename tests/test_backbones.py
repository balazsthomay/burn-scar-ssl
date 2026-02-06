"""Tests for backbone registry and backbone implementations."""

import pytest
import torch
import torch.nn as nn

from src.models.backbone_registry import (
    BACKBONE_REGISTRY,
    BackboneSpec,
    backbone_info,
    create_backbone,
    get_backbone_spec,
    list_backbones,
    register_backbone,
)

# Import backbones to register them
import src.models.dinov3_backbone  # noqa: F401
import src.models.resnet_backbone  # noqa: F401


class TestBackboneRegistry:
    """Tests for the backbone registry system."""

    def test_register_backbone_decorator(self):
        """Should register a backbone with the decorator."""
        # Create a temporary registry to avoid polluting global state
        original_registry = BACKBONE_REGISTRY.copy()

        try:
            @register_backbone(
                name="test_backbone",
                input_channels=6,
                output_channels=[64, 128, 256, 512],
                output_strides=[4, 8, 16, 32],
            )
            def create_test_backbone(**kwargs) -> nn.Module:
                return nn.Identity()

            assert "test_backbone" in BACKBONE_REGISTRY
            spec = BACKBONE_REGISTRY["test_backbone"]
            assert spec.name == "test_backbone"
            assert spec.input_channels == 6
            assert spec.output_channels == [64, 128, 256, 512]
            assert spec.output_strides == [4, 8, 16, 32]
        finally:
            # Restore original registry
            BACKBONE_REGISTRY.clear()
            BACKBONE_REGISTRY.update(original_registry)

    def test_get_backbone_spec_unknown(self):
        """Should raise KeyError for unknown backbone."""
        with pytest.raises(KeyError, match="Unknown backbone"):
            get_backbone_spec("nonexistent_backbone")

    def test_create_backbone_unknown(self):
        """Should raise KeyError for unknown backbone."""
        with pytest.raises(KeyError, match="Unknown backbone"):
            create_backbone("nonexistent_backbone")

    def test_list_backbones_returns_list(self):
        """Should return a list of backbone names."""
        result = list_backbones()
        assert isinstance(result, list)

    def test_backbone_info(self):
        """Should return backbone info dict."""
        original_registry = BACKBONE_REGISTRY.copy()

        try:
            @register_backbone(
                name="info_test",
                input_channels=3,
                output_channels=[64, 128],
                output_strides=[4, 8],
            )
            def create_info_test(**kwargs) -> nn.Module:
                return nn.Identity()

            info = backbone_info("info_test")
            assert info["name"] == "info_test"
            assert info["input_channels"] == 3
            assert info["n_feature_levels"] == 2
        finally:
            BACKBONE_REGISTRY.clear()
            BACKBONE_REGISTRY.update(original_registry)


class TestBackboneSpec:
    """Tests for BackboneSpec dataclass."""

    def test_backbone_spec_creation(self):
        """Should create BackboneSpec with all fields."""
        spec = BackboneSpec(
            name="test",
            input_channels=6,
            output_channels=[256, 512],
            output_strides=[8, 16],
            factory_fn=lambda: nn.Identity(),
        )
        assert spec.name == "test"
        assert spec.input_channels == 6
        assert spec.output_channels == [256, 512]
        assert spec.output_strides == [8, 16]
        assert callable(spec.factory_fn)


class DummyBackbone(nn.Module):
    """Simple backbone for testing multi-scale output."""

    def __init__(self, in_channels: int = 6):
        super().__init__()
        self.in_channels = in_channels
        # Simulate 4-level feature pyramid
        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=4, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        f4 = self.conv4(f3)
        return [f1, f2, f3, f4]


class TestBackboneOutputContract:
    """Tests for backbone output contract (multi-scale features)."""

    def test_dummy_backbone_output_shapes(self):
        """Backbone should output list of feature maps at correct scales."""
        backbone = DummyBackbone(in_channels=6)
        x = torch.randn(2, 6, 512, 512)

        features = backbone(x)

        assert len(features) == 4
        # Check each feature level has expected spatial size
        # stride 4: 512/4 = 128
        assert features[0].shape == (2, 64, 128, 128)
        # stride 8: 512/8 = 64
        assert features[1].shape == (2, 128, 64, 64)
        # stride 16: 512/16 = 32
        assert features[2].shape == (2, 256, 32, 32)
        # stride 32: 512/32 = 16
        assert features[3].shape == (2, 512, 16, 16)

    def test_backbone_batch_independence(self):
        """Backbone outputs should be independent across batch."""
        backbone = DummyBackbone(in_channels=6)
        x1 = torch.randn(1, 6, 256, 256)
        x2 = torch.randn(1, 6, 256, 256)
        x_batch = torch.cat([x1, x2], dim=0)

        features_1 = backbone(x1)
        features_batch = backbone(x_batch)

        # First sample in batch should match standalone
        for f1, fb in zip(features_1, features_batch):
            torch.testing.assert_close(f1, fb[:1])


class TestDINOv3Backbone:
    """Tests for DINOv3 backbone implementation."""

    def test_dinov3_registered(self):
        """DINOv3 should be registered in backbone registry."""
        assert "dinov3_vitl16_sat" in BACKBONE_REGISTRY

    def test_dinov3_spec(self):
        """DINOv3 spec should have correct parameters."""
        spec = get_backbone_spec("dinov3_vitl16_sat")
        assert spec.input_channels == 6
        assert spec.output_channels == [1024, 1024, 1024, 1024]
        assert spec.output_strides == [16, 16, 16, 16]

    @pytest.mark.slow
    def test_dinov3_forward_pass(self):
        """DINOv3 should produce correct output shapes."""
        from src.models.dinov3_backbone import DINOv3Backbone

        backbone = DINOv3Backbone(pretrained=True)
        backbone.eval()

        # Use smaller input for faster test
        x = torch.randn(1, 6, 224, 224)

        with torch.no_grad():
            features = backbone(x)

        assert len(features) == 4
        # All features at stride 16: 224/16 = 14
        for f in features:
            assert f.shape == (1, 1024, 14, 14)

    @pytest.mark.slow
    def test_dinov3_6band_adaptation(self):
        """DINOv3 should handle 6-band input correctly."""
        from src.models.dinov3_backbone import DINOv3Backbone

        backbone = DINOv3Backbone(in_channels=6, pretrained=True)

        # Check patch embedding was adapted
        patch_proj = backbone.model.embeddings.patch_embeddings
        assert patch_proj.weight.shape[1] == 6

    @pytest.mark.slow
    def test_dinov3_create_via_registry(self):
        """Should create DINOv3 via registry."""
        backbone = create_backbone("dinov3_vitl16_sat", pretrained=True)
        assert backbone is not None
        assert backbone.in_channels == 6


class TestResNet50Backbone:
    """Tests for ResNet-50 backbone implementation."""

    def test_resnet50_registered(self):
        """ResNet-50 should be registered in backbone registry."""
        assert "resnet50" in BACKBONE_REGISTRY

    def test_resnet50_spec(self):
        """ResNet-50 spec should have correct parameters."""
        spec = get_backbone_spec("resnet50")
        assert spec.input_channels == 6
        assert spec.output_channels == [256, 512, 1024, 2048]
        assert spec.output_strides == [4, 8, 16, 32]

    def test_resnet50_forward_pass(self):
        """ResNet-50 should produce correct output shapes."""
        from src.models.resnet_backbone import ResNetBackbone

        backbone = ResNetBackbone(pretrained=True)
        backbone.eval()

        x = torch.randn(2, 6, 512, 512)

        with torch.no_grad():
            features = backbone(x)

        assert len(features) == 4
        # Check each feature level
        assert features[0].shape == (2, 256, 128, 128)   # stride 4
        assert features[1].shape == (2, 512, 64, 64)    # stride 8
        assert features[2].shape == (2, 1024, 32, 32)   # stride 16
        assert features[3].shape == (2, 2048, 16, 16)   # stride 32

    def test_resnet50_6band_adaptation(self):
        """ResNet-50 should handle 6-band input correctly."""
        from src.models.resnet_backbone import ResNetBackbone

        backbone = ResNetBackbone(in_channels=6, pretrained=True)

        # Check first conv was adapted
        assert backbone.stem[0].weight.shape[1] == 6

    def test_resnet50_create_via_registry(self):
        """Should create ResNet-50 via registry."""
        backbone = create_backbone("resnet50", pretrained=True)
        assert backbone is not None
        assert backbone.in_channels == 6


class TestSegmentationModel:
    """Tests for unified segmentation model."""

    def test_segmentation_resnet50_shapes(self):
        """Segmentation model with ResNet-50 should produce correct shapes."""
        from src.models.segmentation import SegmentationModel

        model = SegmentationModel(
            backbone_name="resnet50",
            num_classes=2,
            pretrained=True,
        )
        model.eval()

        x = torch.randn(2, 6, 256, 256)
        with torch.no_grad():
            logits = model(x)

        # Output should match input spatial dimensions
        assert logits.shape == (2, 2, 256, 256)

    @pytest.mark.slow
    def test_segmentation_dinov3_shapes(self):
        """Segmentation model with DINOv3 should produce correct shapes."""
        from src.models.segmentation import SegmentationModel

        model = SegmentationModel(
            backbone_name="dinov3_vitl16_sat",
            num_classes=2,
            pretrained=True,
        )
        model.eval()

        x = torch.randn(1, 6, 224, 224)
        with torch.no_grad():
            logits = model(x)

        assert logits.shape == (1, 2, 224, 224)

    def test_segmentation_factory(self):
        """Factory function should create model correctly."""
        from src.models.segmentation import create_segmentation_model

        model = create_segmentation_model(
            backbone_name="resnet50",
            num_classes=3,
            pretrained=True,
        )
        assert model.num_classes == 3
        assert model.backbone_name == "resnet50"

    def test_segmentation_param_groups(self):
        """Model should expose separate backbone and decoder params."""
        from src.models.segmentation import SegmentationModel

        model = SegmentationModel(
            backbone_name="resnet50",
            num_classes=2,
            pretrained=True,
        )

        backbone_params = list(model.get_backbone_params())
        decoder_params = list(model.get_decoder_params())

        assert len(backbone_params) > 0
        assert len(decoder_params) > 0

        # Params should be disjoint
        backbone_ids = {id(p) for p in backbone_params}
        decoder_ids = {id(p) for p in decoder_params}
        assert backbone_ids.isdisjoint(decoder_ids)
