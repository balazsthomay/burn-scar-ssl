#!/usr/bin/env python
"""Download HLS Burn Scars dataset from Google Drive.

The dataset is hosted on Google Drive and organized for TerraTorch:
- data/ directory with *_merged.tif images and *.mask.tif labels
- splits/ directory with train.txt, val.txt, test.txt
"""

import argparse
import tarfile
from pathlib import Path

import gdown


GDRIVE_FILE_ID = "1yFDNlGqGPxkc9lh9l1O70TuejXAQYYtC"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"


def cleanup_macos_artifacts(dataset_path: Path) -> None:
    """Remove macOS resource fork files (._* prefix) from the dataset."""
    data_dir = dataset_path / "data"
    if not data_dir.is_dir():
        return

    artifacts = list(data_dir.glob("._*"))
    if artifacts:
        print(f"Cleaning up {len(artifacts)} macOS resource fork files...")
        for f in artifacts:
            f.unlink()


def download_hls_burn_scars(output_dir: Path) -> Path:
    """Download and extract the HLS Burn Scars dataset.

    Args:
        output_dir: Directory to save the dataset.

    Returns:
        Path to the extracted dataset directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_path = output_dir / "hls_burn_scars.tar.gz"
    dataset_path = output_dir / "hls_burn_scars"

    # Download if archive doesn't exist
    if not archive_path.is_file():
        print(f"Downloading HLS Burn Scars dataset to {archive_path}...")
        gdown.download(GDRIVE_URL, str(archive_path), quiet=False)
    else:
        print(f"Archive already exists: {archive_path}")

    # Extract if dataset directory doesn't exist
    if not dataset_path.is_dir():
        print(f"Extracting to {dataset_path}...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=output_dir, filter="data")
        print("Extraction complete!")

        # Clean up macOS resource fork files (._* prefix)
        cleanup_macos_artifacts(dataset_path)
    else:
        print(f"Dataset already extracted: {dataset_path}")

    # Verify and print statistics
    verify_dataset(dataset_path)

    return dataset_path


def verify_dataset(dataset_path: Path) -> None:
    """Verify dataset structure and print statistics."""
    print("\n" + "=" * 50)
    print("Dataset Verification")
    print("=" * 50)

    data_dir = dataset_path / "data"
    splits_dir = dataset_path / "splits"

    # Check directories exist
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not splits_dir.is_dir():
        raise FileNotFoundError(f"Splits directory not found: {splits_dir}")

    # Count files (exclude macOS resource fork files starting with ._)
    images = [f for f in data_dir.glob("*_merged.tif") if not f.name.startswith("._")]
    masks = [f for f in data_dir.glob("*.mask.tif") if not f.name.startswith("._")]

    print(f"Data directory: {data_dir}")
    print(f"  Images (*_merged.tif): {len(images)}")
    print(f"  Masks (*.mask.tif): {len(masks)}")

    # Check splits
    print(f"\nSplits directory: {splits_dir}")
    for split_name in ["train", "val", "test"]:
        split_file = splits_dir / f"{split_name}.txt"
        if split_file.is_file():
            with open(split_file) as f:
                count = len([line for line in f if line.strip()])
            print(f"  {split_name}.txt: {count} samples")
        else:
            print(f"  {split_name}.txt: NOT FOUND")

    # Sample image info
    if images:
        import rasterio

        sample_image = images[0]
        with rasterio.open(sample_image) as src:
            print(f"\nSample image: {sample_image.name}")
            print(f"  Shape: {src.height} x {src.width}")
            print(f"  Bands: {src.count}")
            print(f"  Dtype: {src.dtypes[0]}")
            print(f"  CRS: {src.crs}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download HLS Burn Scars dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output directory for dataset (default: data)",
    )
    args = parser.parse_args()

    download_hls_burn_scars(args.output_dir)


if __name__ == "__main__":
    main()
