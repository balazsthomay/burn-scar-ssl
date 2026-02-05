"""Tests for training infrastructure."""

from pathlib import Path
import tempfile

import pytest
import torch

from src.training.metrics import SegmentationMetrics, compute_iou
from src.training.trainer import BurnScarSegmentationTask, create_trainer
from src.utils.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    load_config,
    save_config,
)


class TestComputeIoU:
    """Tests for compute_iou function."""

    def test_perfect_prediction(self):
        """Perfect predictions should give IoU of 1.0."""
        predictions = torch.tensor([[[0, 0, 1, 1], [0, 0, 1, 1]]])
        targets = torch.tensor([[[0, 0, 1, 1], [0, 0, 1, 1]]])

        result = compute_iou(predictions, targets)

        assert result["iou_not_burned"] == 1.0
        assert result["iou_burn_scar"] == 1.0
        assert result["mean_iou"] == 1.0

    def test_zero_overlap(self):
        """No overlap should give IoU of 0.0."""
        predictions = torch.tensor([[[1, 1], [1, 1]]])
        targets = torch.tensor([[[0, 0], [0, 0]]])

        result = compute_iou(predictions, targets)

        assert result["iou_not_burned"] == 0.0
        assert result["iou_burn_scar"] == 0.0

    def test_partial_overlap(self):
        """Partial overlap should give intermediate IoU."""
        # 2x2 image, predictions: all 1, targets: half 0, half 1
        predictions = torch.tensor([[[1, 1], [1, 1]]])
        targets = torch.tensor([[[0, 0], [1, 1]]])

        result = compute_iou(predictions, targets)

        # IoU for class 1: intersection=2, union=4, IoU=0.5
        assert result["iou_burn_scar"] == 0.5
        # IoU for class 0: intersection=0, union=2, IoU=0.0
        assert result["iou_not_burned"] == 0.0

    def test_ignore_index(self):
        """Pixels with ignore_index should not affect IoU."""
        predictions = torch.tensor([[[0, 1], [0, 1]]])
        targets = torch.tensor([[[0, -1], [0, 1]]])  # -1 is ignored

        result = compute_iou(predictions, targets, ignore_index=-1)

        # Only 3 valid pixels: (0,0), (1,0), (1,1)
        # Class 0: pred=[0,0], target=[0,0] -> intersection=2, union=2, IoU=1.0
        # Class 1: pred=[1], target=[1] -> intersection=1, union=1, IoU=1.0
        assert result["iou_not_burned"] == 1.0
        assert result["iou_burn_scar"] == 1.0


class TestSegmentationMetrics:
    """Tests for SegmentationMetrics class."""

    def test_init(self):
        """Metrics should initialize correctly."""
        metrics = SegmentationMetrics(num_classes=2)
        assert metrics.num_classes == 2

    def test_update_and_compute(self):
        """Update and compute should work correctly."""
        metrics = SegmentationMetrics(num_classes=2)

        predictions = torch.tensor([[[0, 0, 1, 1], [0, 0, 1, 1]]])
        targets = torch.tensor([[[0, 0, 1, 1], [0, 0, 1, 1]]])

        metrics.update(predictions, targets)
        result = metrics.compute()

        assert "iou_not_burned" in result
        assert "iou_burn_scar" in result
        assert "mean_iou" in result
        assert result["mean_iou"] == pytest.approx(1.0)

    def test_update_with_logits(self):
        """Should handle logits input (B, C, H, W)."""
        metrics = SegmentationMetrics(num_classes=2)

        # Logits where class 0 has higher score
        logits = torch.zeros(1, 2, 2, 2)
        logits[0, 0, :, :] = 1.0  # Class 0 predictions

        targets = torch.zeros(1, 2, 2, dtype=torch.long)  # All class 0

        metrics.update(logits, targets)
        result = metrics.compute()

        assert result["iou_not_burned"] == 1.0

    def test_reset(self):
        """Reset should clear accumulated state."""
        metrics = SegmentationMetrics(num_classes=2)

        predictions = torch.tensor([[[1, 1], [1, 1]]])
        targets = torch.tensor([[[1, 1], [1, 1]]])

        metrics.update(predictions, targets)
        metrics.reset()

        # After reset, computing should work on empty state
        # (torchmetrics returns 0 for empty)
        result = metrics.compute()
        assert "mean_iou" in result


class TestDataConfig:
    """Tests for DataConfig dataclass."""

    def test_default_values(self):
        """Default values should be set correctly."""
        config = DataConfig()
        assert config.dataset_path == "data/hls_burn_scars"
        assert config.batch_size == 8
        assert config.num_workers == 4

    def test_custom_values(self):
        """Custom values should be accepted."""
        config = DataConfig(batch_size=16, num_workers=2)
        assert config.batch_size == 16
        assert config.num_workers == 2


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_default_values(self):
        """Default values should be set correctly."""
        config = ModelConfig()
        assert config.backbone == "prithvi_eo_v2_300"
        assert config.pretrained is True
        assert config.img_size == 512
        assert config.decoder_channels == [512, 256, 128, 64]

    def test_custom_decoder_channels(self):
        """Custom decoder channels should be accepted."""
        config = ModelConfig(decoder_channels=[256, 128, 64, 32])
        assert config.decoder_channels == [256, 128, 64, 32]


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_values(self):
        """Default values should be set correctly."""
        config = TrainingConfig()
        assert config.lr == 1e-4
        assert config.max_epochs == 100
        assert config.loss == "ce"
        assert config.use_wandb is True

    def test_custom_values(self):
        """Custom values should be accepted."""
        config = TrainingConfig(lr=5e-5, loss="focal", use_wandb=False)
        assert config.lr == 5e-5
        assert config.loss == "focal"
        assert config.use_wandb is False


class TestExperimentConfig:
    """Tests for ExperimentConfig dataclass."""

    def test_default_values(self):
        """Default values should be set correctly."""
        config = ExperimentConfig()
        assert config.name == "prithvi-300m-baseline"
        assert config.project == "burn-scar-ssl"
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.training, TrainingConfig)


class TestConfigIO:
    """Tests for config save/load functions."""

    def test_save_and_load(self):
        """Config should be savable and loadable."""
        config = ExperimentConfig(
            name="test-experiment",
            seed=123,
        )

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            config_path = Path(f.name)

        try:
            save_config(config, config_path)
            loaded_config = load_config(config_path)

            assert loaded_config.name == "test-experiment"
            assert loaded_config.seed == 123
            assert loaded_config.data.batch_size == config.data.batch_size
        finally:
            config_path.unlink()


@pytest.mark.slow
class TestBurnScarSegmentationTask:
    """Tests for BurnScarSegmentationTask."""

    def test_init_default(self):
        """Task should initialize with defaults."""
        task = BurnScarSegmentationTask(pretrained=False)
        assert task is not None

    def test_init_with_frozen_backbone(self):
        """Task should initialize with frozen backbone."""
        task = BurnScarSegmentationTask(pretrained=False, freeze_backbone=True)
        assert task is not None


@pytest.mark.slow
class TestCreateTrainer:
    """Tests for create_trainer function."""

    def test_create_trainer_default(self):
        """Should create trainer with defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = create_trainer(
                output_dir=tmpdir,
                use_wandb=False,  # Don't require W&B login
            )
            assert trainer is not None
            assert trainer.max_epochs == 100

    def test_create_trainer_custom_epochs(self):
        """Should create trainer with custom epochs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = create_trainer(
                max_epochs=50,
                output_dir=tmpdir,
                use_wandb=False,
            )
            assert trainer.max_epochs == 50
