"""Tests for Phase 6 demo: inference engine and FastAPI endpoints."""

import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import pytest
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Fixtures — fake ONNX model for deterministic testing
# ---------------------------------------------------------------------------


class _PlainModel(torch.nn.Module):
    """Minimal 2-class segmentation model returning raw logit tensors."""

    def __init__(self, num_classes: int = 2, in_channels: int = 6):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _export_plain_onnx(directory: Path, img_size: int = 32) -> Path:
    """Export a _PlainModel to ONNX in the given directory."""
    model = _PlainModel()
    model.eval()
    onnx_path = directory / "test_model.onnx"
    dummy = torch.randn(1, 6, img_size, img_size)
    torch.onnx.export(
        model,
        (dummy,),
        str(onnx_path),
        opset_version=17,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        dynamo=False,
    )
    return onnx_path


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def onnx_path(tmp_dir):
    """Export a small ONNX model and return its path."""
    return _export_plain_onnx(tmp_dir, img_size=32)


@pytest.fixture
def predictor(onnx_path):
    """A BurnScarPredictor backed by a small fake ONNX model."""
    from src.demo.inference import BurnScarPredictor

    return BurnScarPredictor(onnx_path)


# ---------------------------------------------------------------------------
# TestBurnScarPredictor — inference engine
# ---------------------------------------------------------------------------


class TestBurnScarPredictor:
    """Tests for BurnScarPredictor normalization, prediction, and rendering."""

    def test_normalize_shape_dtype(self, predictor):
        """normalize() should produce [1, 6, H, W] float32."""
        image = np.random.rand(6, 32, 32).astype(np.float32)
        out = predictor.normalize(image)
        assert out.shape == (1, 6, 32, 32)
        assert out.dtype == np.float32

    def test_normalize_nan_handling(self, predictor):
        """normalize() should replace NaN with 0 before normalizing."""
        image = np.full((6, 32, 32), np.nan, dtype=np.float32)
        out = predictor.normalize(image)
        assert not np.any(np.isnan(out))

    def test_predict_shapes(self, predictor):
        """predict() should return mask [H,W] int and confidence [H,W] float."""
        image = np.random.rand(6, 32, 32).astype(np.float32)
        norm = predictor.normalize(image)
        mask, confidence, ms = predictor.predict(norm)
        assert mask.shape == (32, 32)
        assert confidence.shape == (32, 32)
        assert mask.dtype in (np.int32, np.int64)
        assert np.issubdtype(confidence.dtype, np.floating)
        assert ms > 0

    def test_predict_mask_binary(self, predictor):
        """Predicted mask should contain only values {0, 1}."""
        image = np.random.rand(6, 32, 32).astype(np.float32)
        norm = predictor.normalize(image)
        mask, _, _ = predictor.predict(norm)
        unique = set(np.unique(mask))
        assert unique <= {0, 1}

    def test_confidence_range(self, predictor):
        """Confidence values should be in [0, 1]."""
        image = np.random.rand(6, 32, 32).astype(np.float32)
        norm = predictor.normalize(image)
        _, confidence, _ = predictor.predict(norm)
        assert confidence.min() >= 0.0
        assert confidence.max() <= 1.0

    def test_to_rgb_png_valid(self, predictor):
        """to_rgb_png should produce a decodable 3-channel PNG."""
        image = np.random.rand(6, 32, 32).astype(np.float32)
        png_bytes = predictor.to_rgb_png(image)
        img = Image.open(io.BytesIO(png_bytes))
        assert img.mode == "RGB"
        assert img.size == (32, 32)

    def test_mask_to_png_transparency(self, predictor):
        """Non-burn pixels should have alpha=0."""
        mask = np.zeros((32, 32), dtype=np.int32)
        mask[10:20, 10:20] = 1  # burn region
        png_bytes = predictor.mask_to_png(mask)
        img = Image.open(io.BytesIO(png_bytes))
        assert img.mode == "RGBA"
        arr = np.array(img)
        # Non-burn pixels should be transparent
        assert arr[0, 0, 3] == 0
        # Burn pixels should be opaque-ish
        assert arr[15, 15, 3] > 0

    def test_mask_to_png_gt_vs_pred_color(self, predictor):
        """Ground truth and prediction masks should use different colors."""
        mask = np.ones((8, 8), dtype=np.int32)
        pred_bytes = predictor.mask_to_png(mask, gt=False)
        gt_bytes = predictor.mask_to_png(mask, gt=True)
        pred_arr = np.array(Image.open(io.BytesIO(pred_bytes)))
        gt_arr = np.array(Image.open(io.BytesIO(gt_bytes)))
        # The RGB channels should differ between gt and pred
        assert not np.array_equal(pred_arr[:, :, :3], gt_arr[:, :, :3])

    def test_confidence_to_png_valid(self, predictor):
        """confidence_to_png should produce a decodable RGBA PNG."""
        confidence = np.random.rand(32, 32).astype(np.float32)
        png_bytes = predictor.confidence_to_png(confidence)
        img = Image.open(io.BytesIO(png_bytes))
        assert img.mode == "RGBA"
        assert img.size == (32, 32)

    def test_overlay_png_valid(self, predictor):
        """overlay_png should composite mask onto RGB producing a valid PNG."""
        image = np.random.rand(6, 32, 32).astype(np.float32)
        mask = np.zeros((32, 32), dtype=np.int32)
        mask[5:15, 5:15] = 1
        rgb_bytes = predictor.to_rgb_png(image)
        mask_bytes = predictor.mask_to_png(mask)
        overlay_bytes = predictor.overlay_png(rgb_bytes, mask_bytes, alpha=0.6)
        img = Image.open(io.BytesIO(overlay_bytes))
        assert img.size == (32, 32)


# ---------------------------------------------------------------------------
# TestDemoAPI — FastAPI endpoints
# ---------------------------------------------------------------------------


@pytest.fixture
def test_geotiff(tmp_dir):
    """Create a minimal 6-band GeoTIFF for upload testing."""
    import rasterio
    from rasterio.transform import from_bounds

    tif_path = tmp_dir / "test_merged.tif"
    data = np.random.rand(6, 32, 32).astype(np.float32)
    transform = from_bounds(0, 0, 1, 1, 32, 32)
    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        height=32,
        width=32,
        count=6,
        dtype="float32",
        transform=transform,
    ) as dst:
        dst.write(data)
    return tif_path


@pytest.fixture
def api_client(onnx_path, tmp_dir):
    """Create a FastAPI TestClient with a fake ONNX model and gallery dir."""
    from fastapi.testclient import TestClient

    from src.demo.app import create_app

    # Create a minimal gallery: one image + mask
    import rasterio
    from rasterio.transform import from_bounds

    gallery_dir = tmp_dir / "gallery"
    gallery_dir.mkdir()
    transform = from_bounds(0, 0, 1, 1, 32, 32)

    image_path = gallery_dir / "subsetted_512x512_HLS.S30.T10SDH.2020248.v1.4_merged.tif"
    data = np.random.rand(6, 32, 32).astype(np.float32)
    with rasterio.open(
        image_path, "w", driver="GTiff",
        height=32, width=32, count=6, dtype="float32", transform=transform,
    ) as dst:
        dst.write(data)

    mask_path = gallery_dir / "subsetted_512x512_HLS.S30.T10SDH.2020248.v1.4.mask.tif"
    mask_data = np.zeros((1, 32, 32), dtype=np.float32)
    mask_data[0, 10:20, 10:20] = 1
    with rasterio.open(
        mask_path, "w", driver="GTiff",
        height=32, width=32, count=1, dtype="float32", transform=transform,
    ) as dst:
        dst.write(mask_data)

    app = create_app(
        onnx_path=str(onnx_path),
        data_dir=str(gallery_dir),
        gallery_ids=["T10SDH.2020248.v1"],
        results_json=None,
    )
    return TestClient(app)


class TestDemoAPI:
    """Tests for the FastAPI demo server endpoints."""

    def test_root_html(self, api_client):
        """GET / should return HTML."""
        resp = api_client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_gallery_list(self, api_client):
        """GET /api/gallery should return a list of image IDs."""
        resp = api_client.get("/api/gallery")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_predict_gallery(self, api_client):
        """GET /api/predict/{image_id} should return prediction result."""
        resp = api_client.get("/api/predict/T10SDH.2020248.v1")
        assert resp.status_code == 200
        data = resp.json()
        assert "rgb" in data
        assert "prediction" in data
        assert "stats" in data
        assert data["stats"]["burn_fraction"] >= 0

    def test_predict_404(self, api_client):
        """GET /api/predict/{bad_id} should return 404."""
        resp = api_client.get("/api/predict/NONEXISTENT.2099999.v1")
        assert resp.status_code == 404

    def test_model_info(self, api_client):
        """GET /api/model-info should return model metadata."""
        resp = api_client.get("/api/model-info")
        assert resp.status_code == 200
        data = resp.json()
        assert "onnx_path" in data

    def test_predict_upload(self, api_client, test_geotiff):
        """POST /api/predict/upload should accept a GeoTIFF and return results."""
        with open(test_geotiff, "rb") as f:
            resp = api_client.post(
                "/api/predict/upload",
                files={"file": ("test.tif", f, "image/tiff")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "rgb" in data
        assert "prediction" in data
        assert data["ground_truth"] is None  # uploads have no GT
