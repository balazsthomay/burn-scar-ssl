"""Semi-supervised data module for FixMatch training.

Provides dual-view datasets (weak + strong augmentation of the same image)
and a Lightning DataModule that pairs labeled and unlabeled batches via
CombinedLoader.
"""

from pathlib import Path

import albumentations
import lightning.pytorch as pl
import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import BAND_NAMES, HLSBurnScarsDataModule
from src.data.transforms import get_strong_transforms, get_val_transforms, get_weak_transforms


class DualViewDataset(Dataset):
    """Dataset that returns both weak and strong augmented views of each image.

    Reads GeoTIFFs directly with rasterio and applies both augmentation
    pipelines to the same image. No explicit normalization is applied — this
    matches TerraTorch's GenericNonGeoSegmentationDataModule which also passes
    raw pixel values (its Normalize class is created but never called).

    For labeled samples, also returns the mask. For unlabeled samples,
    the mask is omitted.
    """

    def __init__(
        self,
        data_dir: Path,
        sample_ids: list[str],
        weak_transforms: list,
        strong_transforms: list,
        labeled: bool = True,
    ):
        """Initialize dual-view dataset.

        Args:
            data_dir: Directory containing GeoTIFF data files.
            sample_ids: List of sample identifiers.
            weak_transforms: Albumentations transform list for weak augmentation.
            strong_transforms: Albumentations transform list for strong augmentation.
            labeled: Whether this dataset has masks (True for labeled, False for unlabeled).
        """
        self.data_dir = Path(data_dir)
        self.sample_ids = sample_ids
        self.weak_pipeline = albumentations.Compose(weak_transforms)
        self.strong_pipeline = albumentations.Compose(strong_transforms)
        self.labeled = labeled

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _load_image(self, sample_id: str) -> np.ndarray:
        """Load a multi-band GeoTIFF image.

        No normalization is applied — raw pixel values are passed through,
        matching TerraTorch's data pipeline used for val/test.

        Returns:
            Image array of shape (H, W, C), with no-data replaced by 0.
        """
        img_path = self.data_dir / f"subsetted_512x512_HLS.S30.{sample_id}.4_merged.tif"
        with rasterio.open(img_path) as src:
            # rasterio reads as (C, H, W)
            image = src.read().astype(np.float32)

        # Replace NaN/inf with 0
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        # Transpose to (H, W, C) for albumentations
        image = image.transpose(1, 2, 0)
        return image

    def _load_mask(self, sample_id: str) -> np.ndarray:
        """Load a mask GeoTIFF.

        Returns:
            Mask array of shape (H, W), with no-data replaced by -1.
        """
        mask_path = self.data_dir / f"subsetted_512x512_HLS.S30.{sample_id}.4.mask.tif"
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int64)

        # Replace any value that isn't 0 or 1 with -1 (no-data)
        mask = np.where((mask == 0) | (mask == 1), mask, -1)
        return mask

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample_id = self.sample_ids[idx]
        image = self._load_image(sample_id)

        if self.labeled:
            mask = self._load_mask(sample_id)
            weak = self.weak_pipeline(image=image, mask=mask)
            strong = self.strong_pipeline(image=image, mask=mask)
            mask = torch.from_numpy(weak["mask"]) if isinstance(weak["mask"], np.ndarray) else weak["mask"]
            return {
                "image_weak": weak["image"],
                "image_strong": strong["image"],
                "mask": mask.long(),
            }
        else:
            weak = self.weak_pipeline(image=image)
            strong = self.strong_pipeline(image=image)
            return {
                "image_weak": weak["image"],
                "image_strong": strong["image"],
            }


class SemiSupervisedDataModule(pl.LightningDataModule):
    """Lightning DataModule for semi-supervised segmentation.

    Splits the full training set into labeled and unlabeled subsets
    based on a split file. Uses CombinedLoader to yield paired batches
    (labeled + unlabeled) during training.

    Val/test loaders are standard supervised loaders reused from
    HLSBurnScarsDataModule.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        labeled_split_file: str | Path,
        batch_size_labeled: int = 8,
        batch_size_unlabeled: int = 8,
        num_workers: int = 4,
    ):
        """Initialize the semi-supervised data module.

        Args:
            dataset_path: Path to the hls_burn_scars directory.
            labeled_split_file: Split file for labeled samples (e.g., train_10pct.txt).
            batch_size_labeled: Batch size for labeled data.
            batch_size_unlabeled: Batch size for unlabeled data.
            num_workers: Number of data loading workers.
        """
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.labeled_split_file = Path(labeled_split_file)
        self.batch_size_labeled = batch_size_labeled
        self.batch_size_unlabeled = batch_size_unlabeled
        self.num_workers = num_workers

        self._labeled_dataset: DualViewDataset | None = None
        self._unlabeled_dataset: DualViewDataset | None = None

    def _read_split_file(self, split_file: Path) -> list[str]:
        """Read sample IDs from a split file."""
        return [
            line.strip()
            for line in split_file.read_text().strip().splitlines()
            if line.strip()
        ]

    def _resolve_split_path(self, split_file: Path) -> Path:
        """Resolve a split file path, checking both absolute and relative to splits dir."""
        if split_file.is_absolute() and split_file.exists():
            return split_file
        # Try relative to splits directory
        resolved = self.dataset_path / "splits" / split_file
        if resolved.exists():
            return resolved
        # Try as given
        if split_file.exists():
            return split_file
        raise FileNotFoundError(
            f"Split file not found: {split_file} "
            f"(also checked {resolved})"
        )

    def setup(self, stage: str | None = None) -> None:
        splits_dir = self.dataset_path / "splits"
        data_dir = self.dataset_path / "data"

        # Read full training set and labeled subset
        full_train_file = splits_dir / "train.txt"
        labeled_file = self._resolve_split_path(self.labeled_split_file)

        full_train_ids = self._read_split_file(full_train_file)
        labeled_ids = self._read_split_file(labeled_file)

        # Unlabeled = full training set minus labeled
        labeled_set = set(labeled_ids)
        unlabeled_ids = [sid for sid in full_train_ids if sid not in labeled_set]

        weak_transforms = get_weak_transforms()
        strong_transforms = get_strong_transforms(n_channels=len(BAND_NAMES))

        self._labeled_dataset = DualViewDataset(
            data_dir=data_dir,
            sample_ids=labeled_ids,
            weak_transforms=weak_transforms,
            strong_transforms=strong_transforms,
            labeled=True,
        )

        self._unlabeled_dataset = DualViewDataset(
            data_dir=data_dir,
            sample_ids=unlabeled_ids,
            weak_transforms=weak_transforms,
            strong_transforms=strong_transforms,
            labeled=False,
        )

        # Build val/test via the existing HLSBurnScarsDataModule
        self._supervised_dm = HLSBurnScarsDataModule(
            dataset_path=self.dataset_path,
            batch_size=self.batch_size_labeled,
            num_workers=self.num_workers,
        )
        self._supervised_datamodule = self._supervised_dm.build()
        self._supervised_datamodule.setup(stage)

    @property
    def labeled_dataset(self) -> DualViewDataset:
        if self._labeled_dataset is None:
            raise RuntimeError("Call setup() before accessing datasets")
        return self._labeled_dataset

    @property
    def unlabeled_dataset(self) -> DualViewDataset:
        if self._unlabeled_dataset is None:
            raise RuntimeError("Call setup() before accessing datasets")
        return self._unlabeled_dataset

    def train_dataloader(self):
        from lightning.pytorch.utilities.combined_loader import CombinedLoader

        labeled_loader = DataLoader(
            self.labeled_dataset,
            batch_size=self.batch_size_labeled,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        unlabeled_loader = DataLoader(
            self.unlabeled_dataset,
            batch_size=self.batch_size_unlabeled,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return CombinedLoader(
            {"labeled": labeled_loader, "unlabeled": unlabeled_loader},
            mode="max_size_cycle",
        )

    def val_dataloader(self):
        return self._supervised_datamodule.val_dataloader()

    def test_dataloader(self):
        return self._supervised_datamodule.test_dataloader()
