"""ONNX export, quantization, and benchmarking for burn scar segmentation models.

Supports two checkpoint types:
- "terratorch": BurnScarSegmentationTask (Phase 2 full fine-tuning)
- "peft": Phase4PEFTModule (Phase 4 LoRA/DoRA)

Both produce a PixelWiseModel (TerraTorch) whose forward() returns a ModelOutput
dataclass. OnnxExportWrapper strips that wrapping so the ONNX graph returns a raw
[B, num_classes, H, W] logit tensor.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn


@dataclass
class BenchmarkResult:
    """Latency and throughput measurements for a single backend/batch_size combo."""

    backend: str
    batch_size: int
    device: str
    precision: str
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    throughput_samples_per_sec: float
    model_size_mb: float
    num_runs: int = 200
    warmup_runs: int = 20
    extra: dict = field(default_factory=dict)


class OnnxExportWrapper(nn.Module):
    """Wraps a TerraTorch PixelWiseModel so it returns raw logit tensors.

    TerraTorch models return a ``ModelOutput`` dataclass with an ``.output``
    attribute. ONNX tracing needs plain tensors, so this wrapper peels off
    the dataclass and returns ``output.output`` directly.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        if hasattr(output, "output"):
            return output.output
        return output


def _extract_logits(output: torch.Tensor) -> torch.Tensor:
    """Extract raw logits from model output, handling ModelOutput wrapper."""
    if hasattr(output, "output"):
        return output.output
    return output


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    checkpoint_type: Literal["terratorch", "peft"],
    map_location: str = "cpu",
) -> nn.Module:
    """Load a trained model from a Lightning checkpoint.

    Args:
        checkpoint_path: Path to the ``.ckpt`` file.
        checkpoint_type: ``"terratorch"`` for BurnScarSegmentationTask,
            ``"peft"`` for Phase4PEFTModule.
        map_location: Device to map checkpoint tensors to.

    Returns:
        The inner segmentation model (not the Lightning wrapper) in eval mode.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if checkpoint_type == "terratorch":
        from src.training.trainer import BurnScarSegmentationTask

        task = BurnScarSegmentationTask.load_from_checkpoint(
            str(checkpoint_path), map_location=map_location
        )
        model = task.model
    elif checkpoint_type == "peft":
        from scripts.run_phase4_peft import Phase4PEFTModule

        task = Phase4PEFTModule.load_from_checkpoint(
            str(checkpoint_path), map_location=map_location
        )
        model = task.model
    else:
        raise ValueError(
            f"Unknown checkpoint_type: {checkpoint_type!r}. "
            "Use 'terratorch' or 'peft'."
        )

    model.eval()
    return model


def export_to_onnx(
    model: nn.Module,
    output_path: str | Path,
    img_size: int = 512,
    num_channels: int = 6,
    opset_version: int = 17,
) -> Path:
    """Export a segmentation model to ONNX format.

    Uses the legacy (non-dynamo) exporter with ``set_exportable(True)`` from
    timm to replace fused attention ops with traceable equivalents.

    Args:
        model: PyTorch segmentation model (raw, not Lightning wrapper).
        output_path: Where to write the ``.onnx`` file.
        img_size: Spatial input size (height = width).
        num_channels: Number of input bands.
        opset_version: ONNX opset version.

    Returns:
        Path to the exported ONNX file.
    """
    from timm.layers.config import set_exportable

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wrapper = OnnxExportWrapper(model)
    wrapper.eval()

    dummy_input = torch.randn(1, num_channels, img_size, img_size)

    set_exportable(True)
    try:
        # Use legacy TorchScript exporter (dynamo=False) — Prithvi ViT has
        # SDPA ops that the dynamo exporter can't handle cleanly.
        torch.onnx.export(
            wrapper,
            (dummy_input,),
            str(output_path),
            opset_version=opset_version,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
            dynamo=False,
        )
    finally:
        set_exportable(False)

    # Validate the exported model
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    return output_path


def validate_onnx_model(
    onnx_path: str | Path,
    pytorch_model: nn.Module,
    img_size: int = 512,
    num_channels: int = 6,
    atol: float = 1e-4,
) -> dict:
    """Validate that ONNX output matches PyTorch output within tolerance.

    Args:
        onnx_path: Path to the ONNX model file.
        pytorch_model: The original PyTorch model for comparison.
        img_size: Spatial input size.
        num_channels: Number of input bands.
        atol: Absolute tolerance for comparison.

    Returns:
        Dict with ``max_diff``, ``mean_diff``, ``shapes_match``, and ``within_tolerance``.
    """
    onnx_path = Path(onnx_path)

    dummy_input = torch.randn(1, num_channels, img_size, img_size)

    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pt_output = _extract_logits(pytorch_model(dummy_input))
    pt_numpy = pt_output.numpy()

    # ORT inference
    session = ort.InferenceSession(str(onnx_path))
    ort_output = session.run(None, {"input": dummy_input.numpy()})[0]

    shapes_match = pt_numpy.shape == ort_output.shape
    max_diff = float(np.max(np.abs(pt_numpy - ort_output)))
    mean_diff = float(np.mean(np.abs(pt_numpy - ort_output)))

    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "shapes_match": shapes_match,
        "within_tolerance": max_diff < atol,
        "pytorch_shape": list(pt_numpy.shape),
        "onnx_shape": list(ort_output.shape),
    }


def quantize_dynamic_int8(onnx_path: str | Path) -> Path:
    """Apply dynamic INT8 quantization to an ONNX model.

    Tries ORT's transformer optimizer first (better node fusion for ViTs).
    Falls back to raw ``quantize_dynamic`` if the optimizer fails.

    Args:
        onnx_path: Path to the FP32 ONNX model.

    Returns:
        Path to the quantized INT8 ONNX model.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    onnx_path = Path(onnx_path)
    quantized_path = onnx_path.with_suffix(".int8.onnx")

    # Try transformer-specific optimization first
    try:
        from onnxruntime.transformers.optimizer import optimize_model

        optimized_path = onnx_path.with_suffix(".optimized.onnx")
        opt_model = optimize_model(
            str(onnx_path),
            model_type="vit",
            opt_level=1,
        )
        opt_model.save_model_to_file(str(optimized_path))
        source_path = optimized_path
    except Exception:
        # Transformer optimizer may fail on non-standard ViT architectures
        source_path = onnx_path

    quantize_dynamic(
        model_input=str(source_path),
        model_output=str(quantized_path),
        weight_type=QuantType.QInt8,
    )

    # Clean up intermediate optimized model if it exists
    optimized_candidate = onnx_path.with_suffix(".optimized.onnx")
    if optimized_candidate.exists() and optimized_candidate != onnx_path:
        optimized_candidate.unlink()

    return quantized_path


def benchmark_model(
    model_or_path: nn.Module | str | Path,
    input_shape: tuple[int, ...],
    backend: Literal["pytorch", "ort"] = "pytorch",
    device: str = "cpu",
    num_runs: int = 200,
    warmup_runs: int = 20,
    precision: str = "fp32",
) -> BenchmarkResult:
    """Benchmark inference latency and throughput.

    Args:
        model_or_path: PyTorch ``nn.Module`` (for ``backend="pytorch"``)
            or path to an ONNX file (for ``backend="ort"``).
        input_shape: Input tensor shape ``(B, C, H, W)``.
        backend: ``"pytorch"`` or ``"ort"``.
        device: ``"cpu"`` or ``"cuda"``.
        num_runs: Number of timed inference runs.
        warmup_runs: Number of warmup runs (not timed).
        precision: Label for the precision (``"fp32"``, ``"int8"``).

    Returns:
        BenchmarkResult with latency percentiles and throughput.
    """
    batch_size = input_shape[0]
    use_cuda = device == "cuda" and torch.cuda.is_available()

    if backend == "pytorch":
        if not isinstance(model_or_path, nn.Module):
            raise TypeError("For pytorch backend, model_or_path must be nn.Module")
        model = model_or_path
        model.eval()

        if use_cuda:
            model = model.cuda()

        dummy = torch.randn(*input_shape)
        if use_cuda:
            dummy = dummy.cuda()

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(dummy)
                if use_cuda:
                    torch.cuda.synchronize()

        # Timed runs
        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                if use_cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(dummy)
                if use_cuda:
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)  # ms

        model_size_mb = sum(
            p.numel() * p.element_size() for p in model.parameters()
        ) / (1024 * 1024)

    elif backend == "ort":
        if isinstance(model_or_path, nn.Module):
            raise TypeError("For ort backend, model_or_path must be a path to ONNX file")

        onnx_path = Path(model_or_path)
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
        session = ort.InferenceSession(str(onnx_path), providers=providers)

        dummy_np = np.random.randn(*input_shape).astype(np.float32)
        input_name = session.get_inputs()[0].name

        # Warmup
        for _ in range(warmup_runs):
            session.run(None, {input_name: dummy_np})

        # Timed runs
        latencies = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            session.run(None, {input_name: dummy_np})
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

        model_size_mb = onnx_path.stat().st_size / (1024 * 1024)

    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    latencies_arr = np.array(latencies)
    p50 = float(np.percentile(latencies_arr, 50))
    p95 = float(np.percentile(latencies_arr, 95))
    p99 = float(np.percentile(latencies_arr, 99))
    throughput = batch_size / (p50 / 1000)  # samples/sec based on median

    return BenchmarkResult(
        backend=backend,
        batch_size=batch_size,
        device=device,
        precision=precision,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        latency_p99_ms=p99,
        throughput_samples_per_sec=throughput,
        model_size_mb=model_size_mb,
        num_runs=num_runs,
        warmup_runs=warmup_runs,
    )


def evaluate_onnx_accuracy(
    onnx_path: str | Path,
    datamodule,
    num_classes: int = 2,
) -> dict:
    """Evaluate ONNX model accuracy on a test set.

    Args:
        onnx_path: Path to the ONNX model file.
        datamodule: A Lightning DataModule (already setup). Must have ``test_dataloader()``.
        num_classes: Number of segmentation classes.

    Returns:
        Dict with per-class IoU and mIoU.
    """
    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name

    # Accumulate intersection and union per class
    intersection_sum = np.zeros(num_classes, dtype=np.float64)
    union_sum = np.zeros(num_classes, dtype=np.float64)

    datamodule.setup("test")
    for batch in datamodule.test_dataloader():
        images = batch["image"].numpy()
        masks = batch["mask"].numpy()

        logits = session.run(None, {input_name: images})[0]
        preds = np.argmax(logits, axis=1)

        valid = masks != -1
        for cls in range(num_classes):
            pred_cls = (preds == cls) & valid
            target_cls = (masks == cls) & valid
            intersection_sum[cls] += (pred_cls & target_cls).sum()
            union_sum[cls] += (pred_cls | target_cls).sum()

    ious = intersection_sum / (union_sum + 1e-8)
    miou = float(ious.mean())

    return {
        "per_class_iou": {f"class_{i}": float(ious[i]) for i in range(num_classes)},
        "miou": miou,
        "iou_burn_scar": float(ious[1]) if num_classes > 1 else float(ious[0]),
    }
