"""FastAPI demo server for burn scar segmentation inference.

Serves an interactive frontend and provides REST endpoints for running
ONNX inference on gallery images or uploaded GeoTIFFs.
"""

import base64
import json
import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.demo.inference import BurnScarPredictor

# Default gallery image IDs — selected for visual diversity across burn scar
# coverage fractions. These are tile IDs from the HLS Burn Scars test split.
DEFAULT_GALLERY_IDS = [
    "T10TGT.2018285.v1",
    "T12SVC.2020285.v1",
    "T13TCH.2020280.v1",
    "T10SEH.2020285.v1",
    "T12SVC.2019280.v1",
    "T10TGS.2018285.v1",
    "T10UGU.2020280.v1",
    "T13TDL.2020280.v1",
]

DEFAULT_ONNX = "outputs/phase5/full_ft/model_fp32.int8.onnx"
DEFAULT_DATA_DIR = "data/hls_burn_scars/data"
DEFAULT_RESULTS_JSON = "outputs/phase5/full_ft/deployment_results.json"

_STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    onnx_path: str = DEFAULT_ONNX,
    data_dir: str = DEFAULT_DATA_DIR,
    gallery_ids: list[str] | None = None,
    results_json: str | None = DEFAULT_RESULTS_JSON,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        onnx_path: Path to the ONNX model file.
        data_dir: Path to directory containing GeoTIFF images and masks.
        gallery_ids: List of tile IDs for the gallery. Defaults to DEFAULT_GALLERY_IDS.
        results_json: Path to deployment_results.json for model info endpoint.

    Returns:
        Configured FastAPI instance.
    """
    app = FastAPI(title="Burn Scar Segmentation Demo")

    predictor = BurnScarPredictor(onnx_path)
    data_path = Path(data_dir)
    ids = gallery_ids or DEFAULT_GALLERY_IDS

    # Load deployment results if available
    deployment_info: dict = {}
    if results_json and Path(results_json).exists():
        deployment_info = json.loads(Path(results_json).read_text())

    def _find_image(tile_id: str) -> Path | None:
        """Find the *_merged.tif file for a given tile ID."""
        pattern = f"*{tile_id}*_merged.tif"
        matches = [p for p in data_path.glob(pattern) if not p.name.startswith("._")]
        return matches[0] if matches else None

    def _find_mask(tile_id: str) -> Path | None:
        """Find the *.mask.tif file for a given tile ID."""
        pattern = f"*{tile_id}*.mask.tif"
        matches = [p for p in data_path.glob(pattern) if not p.name.startswith("._")]
        return matches[0] if matches else None

    def _b64(data: bytes) -> str:
        return base64.b64encode(data).decode("ascii")

    def _run_prediction(image: np.ndarray, mask: np.ndarray | None = None) -> dict:
        """Run inference and build the response dict."""
        norm = predictor.normalize(image)
        pred_mask, confidence, inference_ms = predictor.predict(norm)

        rgb_png = predictor.to_rgb_png(image)
        pred_png = predictor.mask_to_png(pred_mask)
        conf_png = predictor.confidence_to_png(confidence)
        overlay_png = predictor.overlay_png(rgb_png, pred_png, alpha=0.6)

        gt_png = None
        if mask is not None:
            gt_png = _b64(predictor.mask_to_png(mask, gt=True))

        burn_pixels = int((pred_mask == 1).sum())
        total_pixels = int(pred_mask.size)

        return {
            "rgb": _b64(rgb_png),
            "ground_truth": gt_png,
            "prediction": _b64(pred_png),
            "confidence": _b64(conf_png),
            "overlay": _b64(overlay_png),
            "stats": {
                "burn_fraction": burn_pixels / total_pixels if total_pixels > 0 else 0,
                "mean_confidence": float(confidence.mean()),
                "inference_ms": round(inference_ms, 1),
            },
        }

    @app.get("/", response_class=HTMLResponse)
    async def root():
        index = _STATIC_DIR / "index.html"
        if index.exists():
            return HTMLResponse(index.read_text())
        return HTMLResponse("<h1>Burn Scar Demo</h1><p>Frontend not found.</p>")

    @app.get("/api/gallery")
    async def gallery():
        available = []
        for tile_id in ids:
            img_path = _find_image(tile_id)
            if img_path is not None:
                available.append(tile_id)
        return available

    @app.get("/api/predict/{image_id}")
    async def predict_gallery(image_id: str):
        img_path = _find_image(image_id)
        if img_path is None:
            raise HTTPException(404, f"Image not found: {image_id}")

        image = predictor.read_geotiff(img_path)
        mask_path = _find_mask(image_id)
        mask = predictor.read_mask(mask_path) if mask_path else None

        result = _run_prediction(image, mask)
        result["image_id"] = image_id
        return result

    @app.post("/api/predict/upload")
    async def predict_upload(file: UploadFile):
        contents = await file.read()
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            image = predictor.read_geotiff(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        result = _run_prediction(image, mask=None)
        result["image_id"] = file.filename
        return result

    @app.get("/api/model-info")
    async def model_info():
        info = {
            "onnx_path": str(predictor.onnx_path),
            "onnx_size_mb": round(predictor.onnx_path.stat().st_size / (1024 * 1024), 1),
        }
        if deployment_info:
            info["deployment"] = deployment_info
        return info

    # Serve static files (frontend assets)
    if _STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    return app
