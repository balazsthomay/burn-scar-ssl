"""HLS Burn Scars dataset module using TerraTorch's GenericNonGeoSegmentationDataModule."""

from pathlib import Path

import albumentations
import albumentations.pytorch
import terratorch.datamodules


# Official normalization statistics for HLS Burn Scars dataset
# Bands: BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2
BAND_MEANS = [
    0.0333497067415863,
    0.0570118552053618,
    0.0588974813200132,
    0.2323245113436119,
    0.1972854853760658,
    0.1194491422518656,
]

BAND_STDS = [
    0.0226913556882377,
    0.0268075602230702,
    0.0400410984436278,
    0.0779173242367269,
    0.0870873883814014,
    0.0724197947743781,
]

BAND_NAMES = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]


class HLSBurnScarsDataModule:
    """Wrapper around TerraTorch's GenericNonGeoSegmentationDataModule for HLS Burn Scars.

    This provides a clean interface for creating the datamodule with sensible defaults
    for the burn scar segmentation task.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        batch_size: int = 8,
        num_workers: int = 4,
        train_transform: list | None = None,
        val_transform: list | None = None,
        test_transform: list | None = None,
    ):
        """Initialize the HLS Burn Scars data module.

        Args:
            dataset_path: Path to the extracted hls_burn_scars directory.
            batch_size: Batch size for dataloaders.
            num_workers: Number of workers for data loading.
            train_transform: Optional custom train transforms (albumentations).
            val_transform: Optional custom validation transforms.
            test_transform: Optional custom test transforms.
        """
        self.dataset_path = Path(dataset_path)
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Default transforms if not provided
        if train_transform is None:
            train_transform = [
                albumentations.D4(),  # Random flips and 90° rotations
                albumentations.pytorch.transforms.ToTensorV2(),
            ]
        if val_transform is None:
            val_transform = [
                albumentations.pytorch.transforms.ToTensorV2(),
            ]
        if test_transform is None:
            test_transform = [
                albumentations.pytorch.transforms.ToTensorV2(),
            ]

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

        self._datamodule: terratorch.datamodules.GenericNonGeoSegmentationDataModule | None = None

    def _validate_dataset_path(self) -> None:
        """Validate that the dataset path contains expected files."""
        data_dir = self.dataset_path / "data"
        splits_dir = self.dataset_path / "splits"

        if not data_dir.is_dir():
            raise FileNotFoundError(
                f"Data directory not found: {data_dir}. "
                "Run scripts/download_data.py first."
            )
        if not splits_dir.is_dir():
            raise FileNotFoundError(
                f"Splits directory not found: {splits_dir}. "
                "Run scripts/download_data.py first."
            )

        # Check split files exist
        for split in ["train", "val", "test"]:
            split_file = splits_dir / f"{split}.txt"
            if not split_file.is_file():
                raise FileNotFoundError(f"Split file not found: {split_file}")

    def build(self) -> terratorch.datamodules.GenericNonGeoSegmentationDataModule:
        """Build and return the TerraTorch datamodule.

        Returns:
            Configured GenericNonGeoSegmentationDataModule.
        """
        self._validate_dataset_path()

        data_dir = self.dataset_path / "data"
        splits_dir = self.dataset_path / "splits"

        self._datamodule = terratorch.datamodules.GenericNonGeoSegmentationDataModule(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            num_classes=2,
            # Data paths
            train_data_root=data_dir,
            train_label_data_root=data_dir,
            val_data_root=data_dir,
            val_label_data_root=data_dir,
            test_data_root=data_dir,
            test_label_data_root=data_dir,
            # Split files
            train_split=splits_dir / "train.txt",
            val_split=splits_dir / "val.txt",
            test_split=splits_dir / "test.txt",
            # File patterns
            img_grep="*_merged.tif",
            label_grep="*.mask.tif",
            # Transforms
            train_transform=self.train_transform,
            val_transform=self.val_transform,
            test_transform=self.test_transform,
            # Normalization
            means=BAND_MEANS,
            stds=BAND_STDS,
            # Handle no-data values
            no_data_replace=0,
            no_label_replace=-1,
        )

        return self._datamodule

    @property
    def datamodule(self) -> terratorch.datamodules.GenericNonGeoSegmentationDataModule:
        """Get the datamodule, building it if necessary."""
        if self._datamodule is None:
            self.build()
        return self._datamodule


def create_datamodule(
    dataset_path: str | Path = "data/hls_burn_scars",
    batch_size: int = 8,
    num_workers: int = 4,
) -> terratorch.datamodules.GenericNonGeoSegmentationDataModule:
    """Create a configured datamodule for HLS Burn Scars.

    Convenience function for quickly setting up the datamodule.

    Args:
        dataset_path: Path to the extracted hls_burn_scars directory.
        batch_size: Batch size for dataloaders.
        num_workers: Number of workers for data loading.

    Returns:
        Configured and ready-to-use datamodule.
    """
    wrapper = HLSBurnScarsDataModule(
        dataset_path=dataset_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    datamodule = wrapper.build()
    datamodule.setup("fit")
    return datamodule
