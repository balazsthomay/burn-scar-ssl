#!/usr/bin/env python3
"""Generate qualitative prediction grids for README figures.

Produces a 6-row x 4-column figure: [RGB | Ground Truth | Prediction | Overlay]
for a diverse set of test images spanning different burn scar coverage fractions.

Usage:
    uv run scripts/generate_figures.py
    uv run scripts/generate_figures.py --onnx outputs/phase5/full_ft/model_fp32.int8.onnx
    uv run scripts/generate_figures.py --output outputs/figures/prediction_grid.png
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from src.data.dataset import BAND_MEANS, BAND_STDS
from src.demo.inference import BurnScarPredictor

DEFAULT_ONNX = "outputs/phase5/full_ft/model_fp32.int8.onnx"
DEFAULT_DATA_DIR = "data/hls_burn_scars/data"
DEFAULT_SPLITS_DIR = "data/hls_burn_scars/splits"
DEFAULT_OUTPUT = "outputs/figures/prediction_grid.png"


def select_diverse_samples(
    data_dir: Path, splits_dir: Path, n_samples: int = 6
) -> list[tuple[str, Path, Path]]:
    """Pick test images spanning a range of burn scar fractions.

    Returns:
        List of (tile_id, image_path, mask_path) tuples sorted by burn fraction.
    """
    test_file = splits_dir / "test.txt"
    if not test_file.exists():
        raise FileNotFoundError(f"Test split not found: {test_file}")

    tile_ids = test_file.read_text().strip().splitlines()

    # Compute burn fraction for each test image
    samples = []
    for tile_id in tile_ids:
        img_matches = [p for p in data_dir.glob(f"*{tile_id}*_merged.tif") if not p.name.startswith("._")]
        mask_matches = [p for p in data_dir.glob(f"*{tile_id}*.mask.tif") if not p.name.startswith("._")]
        if not img_matches or not mask_matches:
            continue

        with rasterio.open(mask_matches[0]) as src:
            mask = src.read(1)
        valid = mask >= 0
        if valid.sum() == 0:
            continue
        burn_frac = (mask[valid] == 1).sum() / valid.sum()
        samples.append((tile_id, img_matches[0], mask_matches[0], burn_frac))

    if not samples:
        raise RuntimeError("No valid test samples found")

    # Sort by burn fraction and pick evenly spaced samples
    samples.sort(key=lambda x: x[3])
    n = len(samples)
    indices = np.linspace(0, n - 1, n_samples, dtype=int)
    selected = [samples[i] for i in indices]

    return [(s[0], s[1], s[2]) for s in selected]


def _image_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert 6-band image to RGB uint8 with percentile stretch."""
    rgb = image[[2, 1, 0]]  # R, G, B
    rgb = np.transpose(rgb, (1, 2, 0))
    for c in range(3):
        band = rgb[:, :, c]
        lo, hi = np.percentile(band, 2), np.percentile(band, 98)
        if hi - lo > 1e-8:
            rgb[:, :, c] = np.clip((band - lo) / (hi - lo), 0, 1)
        else:
            rgb[:, :, c] = 0
    return rgb


def _mask_to_rgba(mask: np.ndarray, color: tuple[int, ...]) -> np.ndarray:
    """Convert binary mask to RGBA float [0,1] with given color for burn pixels."""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4))
    burn = mask == 1
    rgba[burn] = [c / 255.0 for c in color]
    return rgba


def generate_prediction_grid(
    predictor: BurnScarPredictor,
    samples: list[tuple[str, Path, Path]],
    output_path: Path,
) -> None:
    """Generate and save the qualitative prediction grid figure."""
    n_rows = len(samples)
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["RGB", "Ground Truth", "Prediction", "Overlay"]

    for row, (tile_id, img_path, mask_path) in enumerate(samples):
        image = predictor.read_geotiff(img_path)
        gt_mask = predictor.read_mask(mask_path)
        norm = predictor.normalize(image)
        pred_mask, confidence, _ = predictor.predict(norm)

        rgb = _image_to_rgb(image)

        # Column 0: RGB
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_ylabel(tile_id, fontsize=8, rotation=0, ha="right", va="center")

        # Column 1: Ground Truth (cyan overlay on RGB)
        axes[row, 1].imshow(rgb)
        gt_rgba = _mask_to_rgba(gt_mask, (0, 180, 255, 160))
        axes[row, 1].imshow(gt_rgba)

        # Column 2: Prediction (orange overlay on RGB)
        axes[row, 2].imshow(rgb)
        pred_rgba = _mask_to_rgba(pred_mask, (255, 140, 0, 160))
        axes[row, 2].imshow(pred_rgba)

        # Column 3: Overlay (both GT cyan + pred orange)
        axes[row, 3].imshow(rgb)
        axes[row, 3].imshow(gt_rgba * 0.5)
        axes[row, 3].imshow(pred_rgba * 0.5)

        for col in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=11, fontweight="bold")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"Saved prediction grid to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate qualitative prediction figures")
    parser.add_argument("--onnx", default=DEFAULT_ONNX, help="Path to ONNX model")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Path to data directory")
    parser.add_argument("--splits-dir", default=DEFAULT_SPLITS_DIR, help="Path to splits directory")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output figure path")
    parser.add_argument("--n-samples", type=int, default=6, help="Number of sample rows")
    args = parser.parse_args()

    predictor = BurnScarPredictor(args.onnx)
    samples = select_diverse_samples(Path(args.data_dir), Path(args.splits_dir), args.n_samples)
    generate_prediction_grid(predictor, samples, Path(args.output))


if __name__ == "__main__":
    main()
