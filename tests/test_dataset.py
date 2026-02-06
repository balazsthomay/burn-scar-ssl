"""Tests for HLS Burn Scars dataset module."""

from pathlib import Path

import pytest
import torch

from src.data.dataset import (
    BAND_MEANS,
    BAND_NAMES,
    BAND_STDS,
    HLSBurnScarsDataModule,
    create_datamodule,
)
from src.data.transforms import (
    get_augmented_train_transforms,
    get_test_transforms,
    get_train_transforms,
    get_val_transforms,
)


DATASET_PATH = Path("data/hls_burn_scars")


@pytest.fixture
def dataset_exists() -> bool:
    """Check if dataset has been downloaded."""
    return (DATASET_PATH / "data").is_dir() and (DATASET_PATH / "splits").is_dir()


class TestBandStatistics:
    """Tests for band normalization statistics."""

    def test_band_means_length(self):
        """Band means should have 6 values (one per band)."""
        assert len(BAND_MEANS) == 6

    def test_band_stds_length(self):
        """Band stds should have 6 values (one per band)."""
        assert len(BAND_STDS) == 6

    def test_band_names_length(self):
        """Band names should have 6 values."""
        assert len(BAND_NAMES) == 6

    def test_band_means_positive(self):
        """All band means should be positive (reflectance values)."""
        assert all(m > 0 for m in BAND_MEANS)

    def test_band_stds_positive(self):
        """All band stds should be positive."""
        assert all(s > 0 for s in BAND_STDS)

    def test_band_means_reasonable_range(self):
        """Band means should be in reasonable reflectance range [0, 1]."""
        assert all(0 < m < 1 for m in BAND_MEANS)

    def test_band_stds_reasonable_range(self):
        """Band stds should be in reasonable range."""
        assert all(0 < s < 0.5 for s in BAND_STDS)


class TestTransforms:
    """Tests for transform functions."""

    def test_train_transforms_not_empty(self):
        """Train transforms should return a non-empty list."""
        transforms = get_train_transforms()
        assert len(transforms) > 0

    def test_val_transforms_not_empty(self):
        """Val transforms should return a non-empty list."""
        transforms = get_val_transforms()
        assert len(transforms) > 0

    def test_test_transforms_not_empty(self):
        """Test transforms should return a non-empty list."""
        transforms = get_test_transforms()
        assert len(transforms) > 0

    def test_augmented_transforms_not_empty(self):
        """Augmented transforms should return a non-empty list."""
        transforms = get_augmented_train_transforms()
        assert len(transforms) > 0


class TestHLSBurnScarsDataModule:
    """Tests for HLSBurnScarsDataModule wrapper."""

    def test_init_default_params(self):
        """DataModule should initialize with default parameters."""
        dm = HLSBurnScarsDataModule(dataset_path=DATASET_PATH)
        assert dm.batch_size == 8
        assert dm.num_workers == 4

    def test_init_custom_params(self):
        """DataModule should accept custom parameters."""
        dm = HLSBurnScarsDataModule(
            dataset_path=DATASET_PATH,
            batch_size=16,
            num_workers=2,
        )
        assert dm.batch_size == 16
        assert dm.num_workers == 2

    def test_init_custom_transforms(self):
        """DataModule should accept custom transforms."""
        custom_transforms = get_augmented_train_transforms()
        dm = HLSBurnScarsDataModule(
            dataset_path=DATASET_PATH,
            train_transform=custom_transforms,
        )
        assert dm.train_transform == custom_transforms

    def test_validate_missing_dataset(self):
        """Validation should fail for missing dataset."""
        dm = HLSBurnScarsDataModule(dataset_path="nonexistent/path")
        with pytest.raises(FileNotFoundError):
            dm.build()


@pytest.mark.skipif(
    not (DATASET_PATH / "data").is_dir(),
    reason="Dataset not downloaded. Run scripts/download_data.py first.",
)
class TestDataModuleWithData:
    """Tests that require the actual dataset to be downloaded."""

    def test_build_datamodule(self):
        """DataModule should build successfully with downloaded data."""
        dm = HLSBurnScarsDataModule(dataset_path=DATASET_PATH)
        datamodule = dm.build()
        assert datamodule is not None

    def test_setup_fit(self):
        """DataModule should setup for fit stage."""
        dm = HLSBurnScarsDataModule(dataset_path=DATASET_PATH, batch_size=4)
        datamodule = dm.build()
        datamodule.setup("fit")
        assert datamodule.train_dataset is not None
        assert datamodule.val_dataset is not None

    def test_train_dataloader(self):
        """Train dataloader should return batches."""
        dm = HLSBurnScarsDataModule(
            dataset_path=DATASET_PATH,
            batch_size=2,
            num_workers=0,
        )
        datamodule = dm.build()
        datamodule.setup("fit")

        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))

        assert "image" in batch
        assert "mask" in batch

    def test_batch_image_shape(self):
        """Images should have correct shape (B, C, H, W)."""
        dm = HLSBurnScarsDataModule(
            dataset_path=DATASET_PATH,
            batch_size=2,
            num_workers=0,
        )
        datamodule = dm.build()
        datamodule.setup("fit")

        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))

        image = batch["image"]
        assert image.dim() == 4  # B, C, H, W
        assert image.shape[1] == 6  # 6 bands
        assert image.shape[2] == 512  # Height
        assert image.shape[3] == 512  # Width

    def test_batch_mask_shape(self):
        """Masks should have correct shape (B, H, W)."""
        dm = HLSBurnScarsDataModule(
            dataset_path=DATASET_PATH,
            batch_size=2,
            num_workers=0,
        )
        datamodule = dm.build()
        datamodule.setup("fit")

        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))

        mask = batch["mask"]
        assert mask.dim() == 3  # B, H, W
        assert mask.shape[1] == 512  # Height
        assert mask.shape[2] == 512  # Width

    def test_mask_values(self):
        """Mask should contain valid class values (0, 1, or -1)."""
        dm = HLSBurnScarsDataModule(
            dataset_path=DATASET_PATH,
            batch_size=4,
            num_workers=0,
        )
        datamodule = dm.build()
        datamodule.setup("fit")

        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))

        mask = batch["mask"]
        unique_values = torch.unique(mask)

        # Should only contain -1 (no data), 0 (not burned), 1 (burn scar)
        for val in unique_values:
            assert val.item() in [-1, 0, 1], f"Unexpected mask value: {val.item()}"

    def test_image_dtype(self):
        """Images should be float tensors."""
        dm = HLSBurnScarsDataModule(
            dataset_path=DATASET_PATH,
            batch_size=2,
            num_workers=0,
        )
        datamodule = dm.build()
        datamodule.setup("fit")

        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))

        assert batch["image"].dtype == torch.float32

    def test_create_datamodule_convenience(self):
        """Convenience function should create working datamodule."""
        datamodule = create_datamodule(
            dataset_path=DATASET_PATH,
            batch_size=2,
            num_workers=0,
        )
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))
        assert "image" in batch
        assert "mask" in batch

    def test_val_dataloader(self):
        """Validation dataloader should return batches."""
        dm = HLSBurnScarsDataModule(
            dataset_path=DATASET_PATH,
            batch_size=2,
            num_workers=0,
        )
        datamodule = dm.build()
        datamodule.setup("fit")

        val_loader = datamodule.val_dataloader()
        batch = next(iter(val_loader))

        assert "image" in batch
        assert "mask" in batch

    def test_split_sizes(self):
        """Dataset splits should have expected sizes."""
        dm = HLSBurnScarsDataModule(
            dataset_path=DATASET_PATH,
            batch_size=2,
            num_workers=0,
        )
        datamodule = dm.build()
        datamodule.setup("fit")

        # These are the sizes from the downloaded dataset
        assert len(datamodule.train_dataset) == 524
        assert len(datamodule.val_dataset) == 160

    def test_custom_train_split_file(self):
        """DataModule should support custom train split file."""
        # Use the 10% subset (52 samples)
        dm = HLSBurnScarsDataModule(
            dataset_path=DATASET_PATH,
            batch_size=2,
            num_workers=0,
            train_split_file="train_10pct.txt",
        )
        datamodule = dm.build()
        datamodule.setup("fit")

        assert len(datamodule.train_dataset) == 52
        # Val should remain unchanged
        assert len(datamodule.val_dataset) == 160

    def test_custom_train_split_5pct(self):
        """DataModule should load 5% subset correctly."""
        dm = HLSBurnScarsDataModule(
            dataset_path=DATASET_PATH,
            batch_size=2,
            num_workers=0,
            train_split_file="train_5pct.txt",
        )
        datamodule = dm.build()
        datamodule.setup("fit")

        assert len(datamodule.train_dataset) == 26
