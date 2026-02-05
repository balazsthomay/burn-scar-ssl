"""Tests for Prithvi model factory."""

import pytest
import torch

from src.models.prithvi import (
    BACKBONE_BANDS,
    NECK_INDICES,
    build_prithvi_segmentation_model,
    freeze_backbone,
    get_model_info,
    unfreeze_backbone,
)


class TestModelConstants:
    """Tests for model configuration constants."""

    def test_backbone_bands_count(self):
        """Should have 6 bands for HLS data."""
        assert len(BACKBONE_BANDS) == 6

    def test_backbone_bands_names(self):
        """Should have correct band names."""
        expected = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
        assert BACKBONE_BANDS == expected

    def test_neck_indices_has_300m(self):
        """Should have indices for 300M model."""
        assert "prithvi_eo_v2_300" in NECK_INDICES

    def test_neck_indices_300m_values(self):
        """300M model should have correct layer indices."""
        assert NECK_INDICES["prithvi_eo_v2_300"] == [5, 11, 17, 23]

    def test_neck_indices_has_all_variants(self):
        """Should have indices for all supported variants."""
        expected_keys = [
            "prithvi_eo_v1_100",
            "prithvi_eo_v2_300",
            "prithvi_eo_v2_300_tl",
            "prithvi_eo_v2_600",
            "prithvi_eo_v2_600_tl",
        ]
        for key in expected_keys:
            assert key in NECK_INDICES


class TestBuildModel:
    """Tests for model building function."""

    def test_invalid_backbone_raises_error(self):
        """Should raise error for unknown backbone."""
        with pytest.raises(ValueError, match="Unknown backbone"):
            build_prithvi_segmentation_model(backbone="invalid_backbone")

    @pytest.mark.slow
    def test_build_model_default_config(self):
        """Should build model with default configuration."""
        model = build_prithvi_segmentation_model(pretrained=False)
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    @pytest.mark.slow
    def test_build_model_custom_classes(self):
        """Should build model with custom number of classes."""
        model = build_prithvi_segmentation_model(
            pretrained=False,
            num_classes=5,
        )
        assert model is not None

    @pytest.mark.slow
    def test_build_model_custom_decoder_channels(self):
        """Should build model with custom decoder channels."""
        model = build_prithvi_segmentation_model(
            pretrained=False,
            decoder_channels=[256, 128, 64, 32],
        )
        assert model is not None


@pytest.mark.slow
class TestModelForward:
    """Tests for model forward pass."""

    @pytest.fixture
    def model(self):
        """Create a model for testing."""
        return build_prithvi_segmentation_model(
            backbone="prithvi_eo_v2_300",
            pretrained=False,
            num_classes=2,
            img_size=224,  # Smaller for faster tests
        )

    def test_forward_pass_shape(self, model):
        """Forward pass should produce correct output shape."""
        batch_size = 2
        num_bands = 6
        img_size = 224

        x = torch.randn(batch_size, num_bands, img_size, img_size)
        output = model(x)

        # TerraTorch models return a ModelOutput object
        assert hasattr(output, "output")
        assert output.output.shape == (batch_size, 2, img_size, img_size)

    def test_forward_pass_batch_size_1(self, model):
        """Forward pass should work with batch size 1."""
        x = torch.randn(1, 6, 224, 224)
        output = model(x)
        assert output.output.shape == (1, 2, 224, 224)

    def test_forward_pass_dtype(self, model):
        """Output should be float tensor."""
        x = torch.randn(1, 6, 224, 224)
        output = model(x)
        assert output.output.dtype == torch.float32


@pytest.mark.slow
class TestModelInfo:
    """Tests for model information utilities."""

    @pytest.fixture
    def model(self):
        """Create a model for testing."""
        return build_prithvi_segmentation_model(
            pretrained=False,
            img_size=224,
        )

    def test_get_model_info_returns_dict(self, model):
        """Should return dictionary with model info."""
        info = get_model_info(model)
        assert isinstance(info, dict)

    def test_get_model_info_has_required_keys(self, model):
        """Should have all required keys."""
        info = get_model_info(model)
        required_keys = [
            "total_parameters",
            "trainable_parameters",
            "frozen_parameters",
            "total_parameters_millions",
            "trainable_parameters_millions",
        ]
        for key in required_keys:
            assert key in info

    def test_get_model_info_positive_params(self, model):
        """Parameter counts should be positive."""
        info = get_model_info(model)
        assert info["total_parameters"] > 0
        assert info["trainable_parameters"] > 0

    def test_get_model_info_params_add_up(self, model):
        """Trainable + frozen should equal total."""
        info = get_model_info(model)
        assert (
            info["trainable_parameters"] + info["frozen_parameters"]
            == info["total_parameters"]
        )


@pytest.mark.slow
class TestFreezeUnfreeze:
    """Tests for backbone freezing utilities."""

    @pytest.fixture
    def model(self):
        """Create a model for testing."""
        return build_prithvi_segmentation_model(
            pretrained=False,
            img_size=224,
        )

    def test_freeze_backbone_reduces_trainable(self, model):
        """Freezing backbone should reduce trainable parameters."""
        info_before = get_model_info(model)
        freeze_backbone(model)
        info_after = get_model_info(model)

        assert info_after["trainable_parameters"] < info_before["trainable_parameters"]

    def test_freeze_then_unfreeze_restores(self, model):
        """Unfreezing should restore trainable parameters."""
        info_original = get_model_info(model)

        freeze_backbone(model)
        unfreeze_backbone(model)

        info_restored = get_model_info(model)
        assert (
            info_restored["trainable_parameters"] == info_original["trainable_parameters"]
        )

    def test_freeze_backbone_model_still_works(self, model):
        """Model should still work after freezing."""
        freeze_backbone(model)
        x = torch.randn(1, 6, 224, 224)
        output = model(x)
        assert output.output.shape == (1, 2, 224, 224)
