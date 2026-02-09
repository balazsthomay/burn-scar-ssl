"""ONNX inference and visualization for burn scar segmentation.

Wraps an ORT session with GeoTIFF I/O, normalization, and PNG rendering
so the demo server can produce visual results from raw satellite tiles.
"""

import io
import time
from pathlib import Path

import matplotlib
import numpy as np
import onnxruntime as ort
import rasterio
from PIL import Image

from src.data.dataset import BAND_MEANS, BAND_STDS

# Burn scar overlay colors (RGBA 0-255)
_BURN_PRED_COLOR = (255, 140, 0, 160)  # orange, semi-transparent
_BURN_GT_COLOR = (0, 180, 255, 160)  # cyan, semi-transparent


class BurnScarPredictor:
    """ONNX-backed burn scar predictor with visualization helpers.

    Args:
        onnx_path: Path to an exported ONNX model (FP32 or INT8).
    """

    def __init__(self, onnx_path: str | Path):
        self.onnx_path = Path(onnx_path)
        self.session = ort.InferenceSession(str(self.onnx_path))
        self.input_name = self.session.get_inputs()[0].name

        # Pre-compute normalization arrays shaped for (1, C, 1, 1) broadcasting
        self._means = np.array(BAND_MEANS, dtype=np.float32).reshape(1, -1, 1, 1)
        self._stds = np.array(BAND_STDS, dtype=np.float32).reshape(1, -1, 1, 1)

    def read_geotiff(self, path: str | Path) -> np.ndarray:
        """Read a 6-band HLS GeoTIFF into a float32 array.

        Args:
            path: Path to *_merged.tif file.

        Returns:
            ``[6, H, W]`` float32 array with NaN replaced by 0.
        """
        with rasterio.open(path) as src:
            image = src.read().astype(np.float32)  # (C, H, W)
        image = np.nan_to_num(image, nan=0.0)
        return image

    def read_mask(self, path: str | Path) -> np.ndarray:
        """Read a single-band mask GeoTIFF.

        Args:
            path: Path to *.mask.tif file.

        Returns:
            ``[H, W]`` int32 array (0=background, 1=burn, -1=nodata).
        """
        with rasterio.open(path) as src:
            mask = src.read(1).astype(np.int32)
        return mask

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Apply per-band normalization.

        Args:
            image: ``[6, H, W]`` or ``[B, 6, H, W]`` float32.

        Returns:
            ``[1, 6, H, W]`` float32 normalized array (batch dim added if needed).
        """
        image = np.nan_to_num(image, nan=0.0)
        if image.ndim == 3:
            image = image[np.newaxis]  # add batch dim
        return (image - self._means) / self._stds

    def predict(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Run ONNX inference on a normalized image.

        Args:
            image: ``[1, 6, H, W]`` float32 normalized array.

        Returns:
            Tuple of (mask, confidence, inference_ms):
            - mask: ``[H, W]`` int array with values {0, 1}.
            - confidence: ``[H, W]`` float array in [0, 1] — burn scar probability.
            - inference_ms: Wall-clock inference time in milliseconds.
        """
        t0 = time.perf_counter()
        logits = self.session.run(None, {self.input_name: image})[0]  # (1, 2, H, W)
        inference_ms = (time.perf_counter() - t0) * 1000

        logits = logits[0]  # (2, H, W)

        # Softmax over class dimension
        exp = np.exp(logits - logits.max(axis=0, keepdims=True))
        probs = exp / exp.sum(axis=0, keepdims=True)  # (2, H, W)

        mask = np.argmax(probs, axis=0).astype(np.int32)  # (H, W)
        confidence = probs[1]  # burn scar probability

        return mask, confidence, inference_ms

    def to_rgb_png(self, image: np.ndarray) -> bytes:
        """Render a 6-band image as a false-color RGB PNG.

        Uses bands [2, 1, 0] (RED, GREEN, BLUE) with 2nd/98th percentile stretch.

        Args:
            image: ``[6, H, W]`` float32 array (unnormalized).

        Returns:
            PNG-encoded bytes.
        """
        rgb = image[[2, 1, 0]]  # (3, H, W)
        rgb = np.transpose(rgb, (1, 2, 0))  # (H, W, 3)

        # Percentile stretch per channel
        for c in range(3):
            band = rgb[:, :, c]
            lo = np.percentile(band, 2)
            hi = np.percentile(band, 98)
            if hi - lo > 1e-8:
                rgb[:, :, c] = np.clip((band - lo) / (hi - lo), 0, 1)
            else:
                rgb[:, :, c] = 0

        rgb_uint8 = (rgb * 255).astype(np.uint8)
        return _array_to_png(rgb_uint8, mode="RGB")

    def mask_to_png(self, mask: np.ndarray, *, gt: bool = False) -> bytes:
        """Render a binary mask as an RGBA PNG with transparency.

        Burn pixels get a semi-transparent color; non-burn/nodata pixels are fully
        transparent.

        Args:
            mask: ``[H, W]`` int array (0=background, 1=burn, -1=nodata).
            gt: If True, uses ground-truth color (cyan); else prediction color (orange).

        Returns:
            PNG-encoded bytes.
        """
        h, w = mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        color = _BURN_GT_COLOR if gt else _BURN_PRED_COLOR
        burn = mask == 1
        rgba[burn] = color
        return _array_to_png(rgba, mode="RGBA")

    def confidence_to_png(self, confidence: np.ndarray) -> bytes:
        """Render a confidence map as a colormapped RGBA PNG.

        Uses matplotlib's 'inferno' colormap. Pixels with confidence < 0.01
        are made transparent.

        Args:
            confidence: ``[H, W]`` float array in [0, 1].

        Returns:
            PNG-encoded bytes.
        """
        cmap = matplotlib.colormaps["inferno"]
        colored = cmap(confidence)  # (H, W, 4) float [0, 1]
        rgba = (colored * 255).astype(np.uint8)
        # Make very-low-confidence pixels transparent
        rgba[confidence < 0.01, 3] = 0
        return _array_to_png(rgba, mode="RGBA")

    def overlay_png(
        self, rgb_bytes: bytes, mask_bytes: bytes, alpha: float = 0.5
    ) -> bytes:
        """Composite a mask overlay onto an RGB image.

        Args:
            rgb_bytes: PNG-encoded RGB image.
            mask_bytes: PNG-encoded RGBA mask.
            alpha: Blend factor for the mask layer.

        Returns:
            PNG-encoded composited image.
        """
        rgb_img = Image.open(io.BytesIO(rgb_bytes)).convert("RGBA")
        mask_img = Image.open(io.BytesIO(mask_bytes)).convert("RGBA")

        # Scale mask alpha by the blend factor
        mask_arr = np.array(mask_img)
        mask_arr[:, :, 3] = (mask_arr[:, :, 3].astype(np.float32) * alpha).astype(
            np.uint8
        )
        mask_img = Image.fromarray(mask_arr, "RGBA")

        composite = Image.alpha_composite(rgb_img, mask_img)
        return _image_to_png(composite)


def _array_to_png(arr: np.ndarray, mode: str) -> bytes:
    """Encode a numpy array as PNG bytes."""
    img = Image.fromarray(arr, mode)
    return _image_to_png(img)


def _image_to_png(img: Image.Image) -> bytes:
    """Encode a PIL image as PNG bytes."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
