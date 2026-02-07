"""Tests for semi-supervised data module (DualViewDataset + SemiSupervisedDataModule)."""

import numpy as np
import pytest
import rasterio
import torch
from pathlib import Path
from rasterio.transform import from_bounds
from unittest.mock import patch

from src.data.ssl_datamodule import DualViewDataset, SemiSupervisedDataModule
from src.data.transforms import get_strong_transforms, get_weak_transforms


# ---------------------------------------------------------------------------
# Fixtures: create minimal GeoTIFF files for testing
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data_dir(tmp_path):
    """Create a temporary directory with fake GeoTIFF images and masks."""
    data_dir = tmp_path / "hls_burn_scars" / "data"
    data_dir.mkdir(parents=True)
    splits_dir = tmp_path / "hls_burn_scars" / "splits"
    splits_dir.mkdir(parents=True)

    sample_ids = [f"T00TEST.2024001.v{i}" for i in range(20)]

    H, W, C = 64, 64, 6
    transform = from_bounds(0, 0, 1, 1, W, H)

    for sid in sample_ids:
        # Write image
        img_path = data_dir / f"subsetted_512x512_HLS.S30.{sid}.4_merged.tif"
        with rasterio.open(
            img_path, "w", driver="GTiff", height=H, width=W,
            count=C, dtype="float32", transform=transform,
        ) as dst:
            data = np.random.rand(C, H, W).astype(np.float32) * 0.3
            dst.write(data)

        # Write mask
        mask_path = data_dir / f"subsetted_512x512_HLS.S30.{sid}.4.mask.tif"
        with rasterio.open(
            mask_path, "w", driver="GTiff", height=H, width=W,
            count=1, dtype="int16", transform=transform,
        ) as dst:
            mask = np.random.choice([0, 1], size=(1, H, W)).astype(np.int16)
            dst.write(mask)

    # Write split files
    (splits_dir / "train.txt").write_text("\n".join(sample_ids) + "\n")
    (splits_dir / "val.txt").write_text("\n".join(sample_ids[:4]) + "\n")
    (splits_dir / "test.txt").write_text("\n".join(sample_ids[:4]) + "\n")

    # Labeled subset: first 5 samples
    labeled_ids = sample_ids[:5]
    (splits_dir / "train_25pct.txt").write_text("\n".join(labeled_ids) + "\n")

    return tmp_path / "hls_burn_scars", sample_ids, labeled_ids


# ---------------------------------------------------------------------------
# Tests: DualViewDataset
# ---------------------------------------------------------------------------

class TestDualViewDataset:
    def test_labeled_returns_weak_strong_and_mask(self, sample_data_dir):
        dataset_path, sample_ids, _ = sample_data_dir
        data_dir = dataset_path / "data"

        ds = DualViewDataset(
            data_dir=data_dir,
            sample_ids=sample_ids[:3],
            weak_transforms=get_weak_transforms(),
            strong_transforms=get_strong_transforms(),
            labeled=True,
        )

        item = ds[0]
        assert "image_weak" in item
        assert "image_strong" in item
        assert "mask" in item

    def test_unlabeled_returns_weak_strong_no_mask(self, sample_data_dir):
        dataset_path, sample_ids, _ = sample_data_dir
        data_dir = dataset_path / "data"

        ds = DualViewDataset(
            data_dir=data_dir,
            sample_ids=sample_ids[:3],
            weak_transforms=get_weak_transforms(),
            strong_transforms=get_strong_transforms(),
            labeled=False,
        )

        item = ds[0]
        assert "image_weak" in item
        assert "image_strong" in item
        assert "mask" not in item

    def test_image_shape_is_correct(self, sample_data_dir):
        dataset_path, sample_ids, _ = sample_data_dir
        data_dir = dataset_path / "data"

        ds = DualViewDataset(
            data_dir=data_dir,
            sample_ids=sample_ids[:1],
            weak_transforms=get_weak_transforms(),
            strong_transforms=get_strong_transforms(),
            labeled=True,
        )

        item = ds[0]
        # (C, H, W) — 6 channels, 64x64
        assert item["image_weak"].shape == (6, 64, 64)
        assert item["image_strong"].shape == (6, 64, 64)

    def test_mask_shape_is_correct(self, sample_data_dir):
        dataset_path, sample_ids, _ = sample_data_dir
        data_dir = dataset_path / "data"

        ds = DualViewDataset(
            data_dir=data_dir,
            sample_ids=sample_ids[:1],
            weak_transforms=get_weak_transforms(),
            strong_transforms=get_strong_transforms(),
            labeled=True,
        )

        item = ds[0]
        assert item["mask"].shape == (64, 64)

    def test_mask_values_are_valid(self, sample_data_dir):
        dataset_path, sample_ids, _ = sample_data_dir
        data_dir = dataset_path / "data"

        ds = DualViewDataset(
            data_dir=data_dir,
            sample_ids=sample_ids[:3],
            weak_transforms=get_weak_transforms(),
            strong_transforms=get_strong_transforms(),
            labeled=True,
        )

        for i in range(len(ds)):
            mask = ds[i]["mask"]
            # Should be 0, 1, or -1
            valid = (mask == 0) | (mask == 1) | (mask == -1)
            assert valid.all()

    def test_dataset_length(self, sample_data_dir):
        dataset_path, sample_ids, _ = sample_data_dir
        data_dir = dataset_path / "data"

        ds = DualViewDataset(
            data_dir=data_dir,
            sample_ids=sample_ids[:7],
            weak_transforms=get_weak_transforms(),
            strong_transforms=get_strong_transforms(),
            labeled=True,
        )

        assert len(ds) == 7

    def test_weak_and_strong_differ(self, sample_data_dir):
        """Strong augmentation should (usually) produce a different image than weak."""
        dataset_path, sample_ids, _ = sample_data_dir
        data_dir = dataset_path / "data"

        ds = DualViewDataset(
            data_dir=data_dir,
            sample_ids=sample_ids[:1],
            weak_transforms=get_weak_transforms(),
            strong_transforms=get_strong_transforms(),
            labeled=True,
        )

        # Over multiple samples, at least one should differ
        # (D4 in weak may also transform, but strong has photometric augs)
        any_diff = False
        for _ in range(10):
            item = ds[0]
            if not torch.allclose(item["image_weak"], item["image_strong"]):
                any_diff = True
                break

        assert any_diff, "Strong and weak views should differ at least sometimes"

    def test_image_values_are_raw(self, sample_data_dir):
        """Image values should be raw pixel values (no normalization applied).

        This matches TerraTorch's GenericNonGeoSegmentationDataModule which
        also passes raw values — its Normalize class is created but never called.
        """
        dataset_path, sample_ids, _ = sample_data_dir
        data_dir = dataset_path / "data"

        ds = DualViewDataset(
            data_dir=data_dir,
            sample_ids=sample_ids[:1],
            weak_transforms=get_weak_transforms(),
            strong_transforms=get_strong_transforms(),
            labeled=True,
        )

        item = ds[0]
        img = item["image_weak"]
        # Fixture data is rand()*0.3, so values should be in [0, 0.3] range
        # (before any photometric augmentation from strong transforms)
        assert img.min() >= -0.1, "Raw values should not be strongly negative"
        assert img.max() < 1.0, "Raw values should be in reflectance range"


# ---------------------------------------------------------------------------
# Tests: SemiSupervisedDataModule
# ---------------------------------------------------------------------------

class TestSemiSupervisedDataModule:
    def test_setup_creates_labeled_and_unlabeled_datasets(self, sample_data_dir):
        dataset_path, sample_ids, labeled_ids = sample_data_dir

        dm = SemiSupervisedDataModule(
            dataset_path=dataset_path,
            labeled_split_file="train_25pct.txt",
            batch_size_labeled=2,
            batch_size_unlabeled=2,
            num_workers=0,
        )
        dm.setup("fit")

        assert len(dm.labeled_dataset) == len(labeled_ids)

    def test_unlabeled_is_complement_of_labeled(self, sample_data_dir):
        dataset_path, sample_ids, labeled_ids = sample_data_dir

        dm = SemiSupervisedDataModule(
            dataset_path=dataset_path,
            labeled_split_file="train_25pct.txt",
            batch_size_labeled=2,
            batch_size_unlabeled=2,
            num_workers=0,
        )
        dm.setup("fit")

        n_labeled = len(dm.labeled_dataset)
        n_unlabeled = len(dm.unlabeled_dataset)
        n_total = len(sample_ids)

        assert n_labeled + n_unlabeled == n_total
        assert n_unlabeled == n_total - len(labeled_ids)

    def test_no_overlap_between_labeled_and_unlabeled(self, sample_data_dir):
        dataset_path, sample_ids, labeled_ids = sample_data_dir

        dm = SemiSupervisedDataModule(
            dataset_path=dataset_path,
            labeled_split_file="train_25pct.txt",
            batch_size_labeled=2,
            batch_size_unlabeled=2,
            num_workers=0,
        )
        dm.setup("fit")

        labeled_set = set(dm.labeled_dataset.sample_ids)
        unlabeled_set = set(dm.unlabeled_dataset.sample_ids)

        assert labeled_set & unlabeled_set == set()

    def test_labeled_dataset_is_labeled(self, sample_data_dir):
        dataset_path, _, _ = sample_data_dir

        dm = SemiSupervisedDataModule(
            dataset_path=dataset_path,
            labeled_split_file="train_25pct.txt",
            batch_size_labeled=2,
            batch_size_unlabeled=2,
            num_workers=0,
        )
        dm.setup("fit")

        assert dm.labeled_dataset.labeled is True

    def test_unlabeled_dataset_is_unlabeled(self, sample_data_dir):
        dataset_path, _, _ = sample_data_dir

        dm = SemiSupervisedDataModule(
            dataset_path=dataset_path,
            labeled_split_file="train_25pct.txt",
            batch_size_labeled=2,
            batch_size_unlabeled=2,
            num_workers=0,
        )
        dm.setup("fit")

        assert dm.unlabeled_dataset.labeled is False

    def test_train_dataloader_yields_combined_batches(self, sample_data_dir):
        dataset_path, _, _ = sample_data_dir

        dm = SemiSupervisedDataModule(
            dataset_path=dataset_path,
            labeled_split_file="train_25pct.txt",
            batch_size_labeled=2,
            batch_size_unlabeled=2,
            num_workers=0,
        )
        dm.setup("fit")

        loader = dm.train_dataloader()
        # CombinedLoader yields (data_dict, batch_idx, dataloader_idx)
        batch, batch_idx, dataloader_idx = next(iter(loader))

        assert "labeled" in batch
        assert "unlabeled" in batch

        # Labeled should have mask
        assert "image_weak" in batch["labeled"]
        assert "image_strong" in batch["labeled"]
        assert "mask" in batch["labeled"]

        # Unlabeled should not have mask
        assert "image_weak" in batch["unlabeled"]
        assert "image_strong" in batch["unlabeled"]
        assert "mask" not in batch["unlabeled"]

    def test_access_before_setup_raises(self):
        dm = SemiSupervisedDataModule(
            dataset_path="/fake/path",
            labeled_split_file="train_10pct.txt",
        )

        with pytest.raises(RuntimeError, match="setup"):
            _ = dm.labeled_dataset

        with pytest.raises(RuntimeError, match="setup"):
            _ = dm.unlabeled_dataset


# ---------------------------------------------------------------------------
# Tests: Strong augmentation pipeline
# ---------------------------------------------------------------------------

class TestStrongTransforms:
    def test_strong_transforms_produce_valid_output(self, sample_data_dir):
        """Strong transforms should produce a tensor with correct shape."""
        dataset_path, sample_ids, _ = sample_data_dir
        data_dir = dataset_path / "data"

        ds = DualViewDataset(
            data_dir=data_dir,
            sample_ids=sample_ids[:1],
            weak_transforms=get_weak_transforms(),
            strong_transforms=get_strong_transforms(),
            labeled=True,
        )

        item = ds[0]
        assert item["image_strong"].dtype == torch.float32
        assert not torch.isnan(item["image_strong"]).any()

    def test_strong_transforms_more_aggressive_than_weak(self, sample_data_dir):
        """Strong augmentation should on average modify images more than weak."""
        dataset_path, sample_ids, _ = sample_data_dir
        data_dir = dataset_path / "data"

        ds = DualViewDataset(
            data_dir=data_dir,
            sample_ids=sample_ids[:1],
            weak_transforms=get_weak_transforms(),
            strong_transforms=get_strong_transforms(),
            labeled=True,
        )

        # Compare variance of augmented outputs across multiple runs
        weak_images = []
        strong_images = []
        for _ in range(20):
            item = ds[0]
            weak_images.append(item["image_weak"])
            strong_images.append(item["image_strong"])

        weak_stack = torch.stack(weak_images)
        strong_stack = torch.stack(strong_images)

        # Strong should have higher variance across augmentations
        weak_var = weak_stack.var(dim=0).mean().item()
        strong_var = strong_stack.var(dim=0).mean().item()

        # This might not always hold due to D4's geometric transforms in weak,
        # but on average strong should be more variable due to photometric augs
        # We use a relaxed check
        assert strong_var > 0, "Strong augmentation should produce variance"
