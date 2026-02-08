"""Tests for Phase 5 ONNX export, quantization, and benchmarking."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.deployment.export import (
    BenchmarkResult,
    OnnxExportWrapper,
    benchmark_model,
    export_to_onnx,
    quantize_dynamic_int8,
    validate_onnx_model,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _FakeModelOutput:
    """Mimics TerraTorch's ModelOutput dataclass."""

    def __init__(self, tensor: torch.Tensor):
        self.output = tensor


class _FakeSegModel(torch.nn.Module):
    """Minimal model that returns a ModelOutput-like object, same as TerraTorch."""

    def __init__(self, num_classes: int = 2, in_channels: int = 6):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, num_classes, 1)

    def forward(self, x: torch.Tensor):
        return _FakeModelOutput(self.conv(x))


class _PlainModel(torch.nn.Module):
    """Model that returns raw tensors (no ModelOutput wrapping)."""

    def __init__(self, num_classes: int = 2, in_channels: int = 6):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


@pytest.fixture
def fake_model():
    """A minimal model with ModelOutput wrapping (like TerraTorch)."""
    model = _FakeSegModel(num_classes=2, in_channels=6)
    model.eval()
    return model


@pytest.fixture
def plain_model():
    """A minimal model returning raw tensors."""
    model = _PlainModel(num_classes=2, in_channels=6)
    model.eval()
    return model


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ---------------------------------------------------------------------------
# TestOnnxExportWrapper
# ---------------------------------------------------------------------------


class TestOnnxExportWrapper:
    """Tests for the OnnxExportWrapper that strips ModelOutput."""

    def test_strips_model_output(self, fake_model):
        """Wrapper should return raw tensor from ModelOutput."""
        wrapper = OnnxExportWrapper(fake_model)
        x = torch.randn(1, 6, 32, 32)
        out = wrapper(x)
        assert isinstance(out, torch.Tensor)
        assert not hasattr(out, "output")

    def test_correct_output_shape(self, fake_model):
        """Output shape should be [B, num_classes, H, W]."""
        wrapper = OnnxExportWrapper(fake_model)
        x = torch.randn(2, 6, 64, 64)
        out = wrapper(x)
        assert out.shape == (2, 2, 64, 64)

    def test_passthrough_for_plain_tensor(self, plain_model):
        """If model already returns a tensor, wrapper should pass it through."""
        wrapper = OnnxExportWrapper(plain_model)
        x = torch.randn(1, 6, 32, 32)
        out = wrapper(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, 2, 32, 32)


# ---------------------------------------------------------------------------
# TestExportToOnnx
# ---------------------------------------------------------------------------


class TestExportToOnnx:
    """Tests for ONNX export functionality."""

    def test_produces_onnx_file(self, fake_model, tmp_dir):
        """Export should produce an .onnx file."""
        output_path = tmp_dir / "model.onnx"
        result = export_to_onnx(fake_model, output_path, img_size=32, num_channels=6)
        assert result.exists()
        assert result.suffix == ".onnx"

    def test_valid_onnx_model(self, fake_model, tmp_dir):
        """Exported model should pass onnx.checker validation."""
        import onnx

        output_path = tmp_dir / "model.onnx"
        export_to_onnx(fake_model, output_path, img_size=32, num_channels=6)
        model = onnx.load(str(output_path))
        # This will raise if invalid
        onnx.checker.check_model(model)

    def test_correct_io_names(self, fake_model, tmp_dir):
        """ONNX model should have 'input' and 'logits' as I/O names."""
        import onnx

        output_path = tmp_dir / "model.onnx"
        export_to_onnx(fake_model, output_path, img_size=32, num_channels=6)
        model = onnx.load(str(output_path))
        input_names = [inp.name for inp in model.graph.input]
        output_names = [out.name for out in model.graph.output]
        assert "input" in input_names
        assert "logits" in output_names

    def test_dynamic_batch_axis(self, fake_model, tmp_dir):
        """ONNX model should accept different batch sizes via ORT."""
        import onnxruntime as ort

        output_path = tmp_dir / "model.onnx"
        export_to_onnx(fake_model, output_path, img_size=32, num_channels=6)
        session = ort.InferenceSession(str(output_path))

        # Batch size 1
        out1 = session.run(None, {"input": np.random.randn(1, 6, 32, 32).astype(np.float32)})[0]
        assert out1.shape[0] == 1

        # Batch size 3
        out3 = session.run(None, {"input": np.random.randn(3, 6, 32, 32).astype(np.float32)})[0]
        assert out3.shape[0] == 3


# ---------------------------------------------------------------------------
# TestValidateOnnx
# ---------------------------------------------------------------------------


class TestValidateOnnx:
    """Tests for ONNX vs PyTorch numerical validation."""

    @pytest.fixture
    def exported_model(self, fake_model, tmp_dir):
        """Export a model and return (onnx_path, pytorch_model)."""
        onnx_path = tmp_dir / "model.onnx"
        export_to_onnx(fake_model, onnx_path, img_size=32, num_channels=6)
        return onnx_path, fake_model

    def test_ort_matches_pytorch(self, exported_model):
        """ORT output should match PyTorch output within tolerance."""
        onnx_path, model = exported_model
        result = validate_onnx_model(onnx_path, model, img_size=32, num_channels=6)
        assert result["within_tolerance"]

    def test_shapes_match(self, exported_model):
        """Output shapes should match between PyTorch and ORT."""
        onnx_path, model = exported_model
        result = validate_onnx_model(onnx_path, model, img_size=32, num_channels=6)
        assert result["shapes_match"]

    def test_tolerance_values(self, exported_model):
        """Max and mean diff should be small positive numbers."""
        onnx_path, model = exported_model
        result = validate_onnx_model(onnx_path, model, img_size=32, num_channels=6)
        assert result["max_diff"] >= 0
        assert result["mean_diff"] >= 0
        assert result["max_diff"] < 1e-4


# ---------------------------------------------------------------------------
# TestQuantize
# ---------------------------------------------------------------------------


class TestQuantize:
    """Tests for dynamic INT8 quantization."""

    @pytest.fixture
    def fp32_onnx(self, fake_model, tmp_dir):
        onnx_path = tmp_dir / "model.onnx"
        export_to_onnx(fake_model, onnx_path, img_size=32, num_channels=6)
        return onnx_path

    def test_produces_int8_file(self, fp32_onnx):
        """Quantization should produce an INT8 ONNX file."""
        int8_path = quantize_dynamic_int8(fp32_onnx)
        assert int8_path.exists()
        assert "int8" in int8_path.name

    def test_int8_runs_inference(self, fp32_onnx):
        """INT8 model should produce valid inference results."""
        import onnxruntime as ort

        int8_path = quantize_dynamic_int8(fp32_onnx)
        session = ort.InferenceSession(str(int8_path))
        dummy = np.random.randn(1, 6, 32, 32).astype(np.float32)
        out = session.run(None, {"input": dummy})[0]
        assert out.shape == (1, 2, 32, 32)

    def test_int8_output_close_to_fp32(self, fp32_onnx):
        """INT8 output should be reasonably close to FP32."""
        import onnxruntime as ort

        int8_path = quantize_dynamic_int8(fp32_onnx)

        dummy = np.random.randn(1, 6, 32, 32).astype(np.float32)

        fp32_session = ort.InferenceSession(str(fp32_onnx))
        fp32_out = fp32_session.run(None, {"input": dummy})[0]

        int8_session = ort.InferenceSession(str(int8_path))
        int8_out = int8_session.run(None, {"input": dummy})[0]

        # For a small conv model, quantization should preserve outputs closely
        max_diff = np.max(np.abs(fp32_out - int8_out))
        assert max_diff < 1.0, f"INT8 diverged from FP32 by {max_diff}"


# ---------------------------------------------------------------------------
# TestBenchmark
# ---------------------------------------------------------------------------


class TestBenchmark:
    """Tests for the benchmark_model function."""

    def test_returns_benchmark_result(self, plain_model):
        """Should return a BenchmarkResult dataclass."""
        result = benchmark_model(
            plain_model, (1, 6, 32, 32), "pytorch", "cpu",
            num_runs=5, warmup_runs=2,
        )
        assert isinstance(result, BenchmarkResult)

    def test_positive_latencies(self, plain_model):
        """All latency percentiles should be positive."""
        result = benchmark_model(
            plain_model, (1, 6, 32, 32), "pytorch", "cpu",
            num_runs=5, warmup_runs=2,
        )
        assert result.latency_p50_ms > 0
        assert result.latency_p95_ms > 0
        assert result.latency_p99_ms > 0

    def test_throughput_math(self, plain_model):
        """Throughput should equal batch_size / (p50_ms / 1000)."""
        result = benchmark_model(
            plain_model, (2, 6, 32, 32), "pytorch", "cpu",
            num_runs=5, warmup_runs=2,
        )
        expected = 2 / (result.latency_p50_ms / 1000)
        assert abs(result.throughput_samples_per_sec - expected) < 0.01

    def test_ort_benchmark(self, fake_model, tmp_dir):
        """ORT backend should also return valid BenchmarkResult."""
        onnx_path = tmp_dir / "model.onnx"
        export_to_onnx(fake_model, onnx_path, img_size=32, num_channels=6)

        result = benchmark_model(
            onnx_path, (1, 6, 32, 32), "ort", "cpu",
            num_runs=5, warmup_runs=2,
        )
        assert isinstance(result, BenchmarkResult)
        assert result.backend == "ort"
        assert result.model_size_mb > 0

    def test_pytorch_requires_module(self):
        """pytorch backend should reject non-Module input."""
        with pytest.raises(TypeError, match="nn.Module"):
            benchmark_model("/some/path.onnx", (1, 6, 32, 32), "pytorch", "cpu")

    def test_ort_requires_path(self, plain_model):
        """ort backend should reject nn.Module input."""
        with pytest.raises(TypeError, match="path"):
            benchmark_model(plain_model, (1, 6, 32, 32), "ort", "cpu")


# ---------------------------------------------------------------------------
# TestLoadCheckpoint — uses real Prithvi models, marked slow
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestLoadCheckpoint:
    """Tests for loading checkpoints from different training phases."""

    @pytest.fixture
    def terratorch_checkpoint(self, tmp_dir):
        """Create a temp BurnScarSegmentationTask checkpoint."""
        from src.training.trainer import BurnScarSegmentationTask

        task = BurnScarSegmentationTask(
            backbone="prithvi_eo_v2_300",
            pretrained=False,
            img_size=224,
        )

        import lightning.pytorch as pl

        trainer = pl.Trainer(
            barebones=True,
            accelerator="cpu",
            devices=1,
            max_steps=0,
        )
        trainer.strategy.connect(task)
        ckpt_path = tmp_dir / "terratorch.ckpt"
        trainer.save_checkpoint(str(ckpt_path))
        return ckpt_path

    @pytest.fixture
    def peft_checkpoint(self, tmp_dir):
        """Create a temp Phase4PEFTModule checkpoint."""
        from scripts.run_phase4_peft import Phase4PEFTModule

        peft_config = {
            "method": "LORA",
            "replace_qkv": "qkv",
            "peft_config_kwargs": {
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_linear", "v_linear"],
                "lora_dropout": 0.1,
                "use_dora": False,
            },
        }
        task = Phase4PEFTModule(
            peft_config=peft_config,
            backbone="prithvi_eo_v2_300",
            num_classes=2,
            img_size=224,
        )

        import lightning.pytorch as pl

        trainer = pl.Trainer(
            barebones=True,
            accelerator="cpu",
            devices=1,
            max_steps=0,
        )
        trainer.strategy.connect(task)
        ckpt_path = tmp_dir / "peft.ckpt"
        trainer.save_checkpoint(str(ckpt_path))
        return ckpt_path

    def test_load_terratorch_checkpoint(self, terratorch_checkpoint):
        """Should load a BurnScarSegmentationTask checkpoint and return the model."""
        from src.deployment.export import load_model_from_checkpoint

        model = load_model_from_checkpoint(terratorch_checkpoint, "terratorch")
        assert isinstance(model, torch.nn.Module)
        assert not model.training  # Should be in eval mode

        # Should produce output
        x = torch.randn(1, 6, 224, 224)
        with torch.no_grad():
            out = model(x)
        logits = out.output if hasattr(out, "output") else out
        assert logits.shape == (1, 2, 224, 224)

    def test_load_peft_checkpoint(self, peft_checkpoint):
        """Should load a Phase4PEFTModule checkpoint and return the model."""
        from src.deployment.export import load_model_from_checkpoint

        model = load_model_from_checkpoint(peft_checkpoint, "peft")
        assert isinstance(model, torch.nn.Module)
        assert not model.training

        x = torch.randn(1, 6, 224, 224)
        with torch.no_grad():
            out = model(x)
        logits = out.output if hasattr(out, "output") else out
        assert logits.shape == (1, 2, 224, 224)

    def test_load_nonexistent_checkpoint(self, tmp_dir):
        """Should raise FileNotFoundError for missing checkpoint."""
        from src.deployment.export import load_model_from_checkpoint

        with pytest.raises(FileNotFoundError):
            load_model_from_checkpoint(tmp_dir / "nope.ckpt", "terratorch")

    def test_load_invalid_type(self, terratorch_checkpoint):
        """Should raise ValueError for unknown checkpoint type."""
        from src.deployment.export import load_model_from_checkpoint

        with pytest.raises(ValueError, match="Unknown checkpoint_type"):
            load_model_from_checkpoint(terratorch_checkpoint, "unknown")
