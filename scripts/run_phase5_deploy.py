#!/usr/bin/env python3
"""Run Phase 5 deployment pipeline: ONNX export + quantization + benchmarking.

Takes a single checkpoint, exports to ONNX, quantizes to INT8, benchmarks
latency across backends, and optionally evaluates accuracy on the test set.

Usage:
    uv run scripts/run_phase5_deploy.py \
        --checkpoint outputs/phase5/full_ft/checkpoints/best.ckpt \
        --checkpoint-type terratorch \
        --output-dir outputs/phase5/full_ft

    uv run scripts/run_phase5_deploy.py \
        --checkpoint outputs/phase4/lora/r16/100pct/seed42/checkpoints/best.ckpt \
        --checkpoint-type peft \
        --output-dir outputs/phase5/lora_r16 \
        --skip-benchmark
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import onnxruntime as ort
import torch
import yaml
from rich.console import Console
from rich.table import Table

from src.deployment.export import (
    BenchmarkResult,
    OnnxExportWrapper,
    benchmark_model,
    evaluate_onnx_accuracy,
    export_to_onnx,
    load_model_from_checkpoint,
    quantize_dynamic_int8,
    validate_onnx_model,
)

console = Console()


def run_benchmark_matrix(
    pytorch_model: torch.nn.Module,
    onnx_fp32_path: Path,
    onnx_int8_path: Path | None,
    img_size: int = 512,
    num_channels: int = 6,
    batch_sizes: list[int] | None = None,
    num_runs: int = 200,
    warmup_runs: int = 20,
) -> list[BenchmarkResult]:
    """Run benchmarks across multiple backends, devices, and batch sizes.

    Args:
        pytorch_model: The original PyTorch model.
        onnx_fp32_path: Path to the FP32 ONNX model.
        onnx_int8_path: Path to the INT8 ONNX model (optional).
        img_size: Spatial input size.
        num_channels: Number of input bands.
        batch_sizes: List of batch sizes to test.
        num_runs: Number of timed runs per benchmark.
        warmup_runs: Number of warmup runs.

    Returns:
        List of BenchmarkResult objects.
    """
    if batch_sizes is None:
        batch_sizes = [1, 4, 8]

    results = []
    has_cuda = torch.cuda.is_available()

    wrapped = OnnxExportWrapper(pytorch_model)
    wrapped.eval()

    for bs in batch_sizes:
        shape = (bs, num_channels, img_size, img_size)

        if has_cuda:
            # PyTorch GPU benchmark
            console.print(f"  PyTorch GPU  bs={bs}...", end=" ")
            r = benchmark_model(wrapped, shape, "pytorch", "cuda", num_runs, warmup_runs, "fp32")
            results.append(r)
            console.print(f"P50={r.latency_p50_ms:.1f}ms")

            # ORT GPU only if CUDAExecutionProvider is actually available
            ort_providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in ort_providers:
                console.print(f"  ORT GPU FP32 bs={bs}...", end=" ")
                r = benchmark_model(onnx_fp32_path, shape, "ort", "cuda", num_runs, warmup_runs, "fp32")
                results.append(r)
                console.print(f"P50={r.latency_p50_ms:.1f}ms")
        else:
            # CPU fallback only when no GPU
            console.print(f"  PyTorch CPU  bs={bs}...", end=" ")
            r = benchmark_model(wrapped, shape, "pytorch", "cpu", num_runs, warmup_runs, "fp32")
            results.append(r)
            console.print(f"P50={r.latency_p50_ms:.1f}ms")

            console.print(f"  ORT CPU FP32 bs={bs}...", end=" ")
            r = benchmark_model(onnx_fp32_path, shape, "ort", "cpu", num_runs, warmup_runs, "fp32")
            results.append(r)
            console.print(f"P50={r.latency_p50_ms:.1f}ms")

            if onnx_int8_path and onnx_int8_path.exists():
                console.print(f"  ORT CPU INT8 bs={bs}...", end=" ")
                r = benchmark_model(onnx_int8_path, shape, "ort", "cpu", num_runs, warmup_runs, "int8")
                results.append(r)
                console.print(f"P50={r.latency_p50_ms:.1f}ms")

    return results


def print_benchmark_table(results: list[BenchmarkResult]) -> None:
    """Print a formatted table of benchmark results."""
    table = Table(title="Benchmark Results")
    table.add_column("Backend", style="cyan")
    table.add_column("Device")
    table.add_column("Precision")
    table.add_column("Batch", justify="right")
    table.add_column("P50 (ms)", justify="right")
    table.add_column("P95 (ms)", justify="right")
    table.add_column("P99 (ms)", justify="right")
    table.add_column("Throughput", justify="right")
    table.add_column("Size (MB)", justify="right")

    for r in results:
        table.add_row(
            r.backend,
            r.device,
            r.precision,
            str(r.batch_size),
            f"{r.latency_p50_ms:.1f}",
            f"{r.latency_p95_ms:.1f}",
            f"{r.latency_p99_ms:.1f}",
            f"{r.throughput_samples_per_sec:.1f}",
            f"{r.model_size_mb:.1f}",
        )

    console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 5: ONNX export, quantization, and benchmarking"
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Path to .ckpt file"
    )
    parser.add_argument(
        "--checkpoint-type",
        type=str,
        choices=["terratorch", "peft"],
        required=True,
        help="Checkpoint type: terratorch (Phase 2) or peft (Phase 4)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/phase5"),
        help="Output directory for ONNX files and results",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase5_deploy.yaml"),
        help="Path to deployment config",
    )
    parser.add_argument(
        "--img-size", type=int, default=None, help="Override input image size"
    )
    parser.add_argument(
        "--skip-benchmark", action="store_true", help="Skip latency benchmarking"
    )
    parser.add_argument(
        "--skip-accuracy", action="store_true", help="Skip accuracy evaluation"
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="Disable W&B logging"
    )
    args = parser.parse_args()

    # Load config
    config = {}
    if args.config.exists():
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}

    export_cfg = config.get("export", {})
    benchmark_cfg = config.get("benchmark", {})
    accuracy_cfg = config.get("accuracy", {})

    img_size = args.img_size or export_cfg.get("img_size", 512)
    num_channels = export_cfg.get("num_channels", 6)
    opset_version = export_cfg.get("opset_version", 17)
    batch_sizes = benchmark_cfg.get("batch_sizes", [1, 4, 8])
    num_runs = benchmark_cfg.get("num_runs", 200)
    warmup_runs = benchmark_cfg.get("warmup_runs", 20)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold]Phase 5: ONNX Deployment Pipeline[/bold]")
    console.print(f"  Checkpoint:  {args.checkpoint}")
    console.print(f"  Type:        {args.checkpoint_type}")
    console.print(f"  Image size:  {img_size}")
    console.print(f"  Output:      {output_dir}")

    # --- Stage 1: Load model ---
    console.print(f"\n[bold blue]Stage 1: Loading model...[/bold blue]")
    model = load_model_from_checkpoint(
        args.checkpoint, args.checkpoint_type
    )
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    console.print(f"  Parameters: {param_count:.2f}M")

    # --- Stage 2: Export to ONNX ---
    console.print(f"\n[bold blue]Stage 2: Exporting to ONNX...[/bold blue]")
    onnx_fp32_path = output_dir / "model_fp32.onnx"
    export_to_onnx(
        model, onnx_fp32_path, img_size=img_size,
        num_channels=num_channels, opset_version=opset_version,
    )
    onnx_size_mb = onnx_fp32_path.stat().st_size / (1024 * 1024)
    console.print(f"  Exported: {onnx_fp32_path} ({onnx_size_mb:.1f} MB)")

    # Validate numerical match
    console.print(f"  Validating ONNX vs PyTorch...")
    val_result = validate_onnx_model(
        onnx_fp32_path, model, img_size=img_size, num_channels=num_channels
    )
    console.print(
        f"  Max diff: {val_result['max_diff']:.2e}, "
        f"Mean diff: {val_result['mean_diff']:.2e}, "
        f"Shapes match: {val_result['shapes_match']}"
    )
    if not val_result["within_tolerance"]:
        console.print(
            f"  [yellow]Warning: Max diff {val_result['max_diff']:.2e} exceeds tolerance[/yellow]"
        )

    # --- Stage 3: Quantize ---
    console.print(f"\n[bold blue]Stage 3: Quantizing to INT8...[/bold blue]")
    onnx_int8_path = quantize_dynamic_int8(onnx_fp32_path)
    int8_size_mb = onnx_int8_path.stat().st_size / (1024 * 1024)
    compression = onnx_size_mb / int8_size_mb if int8_size_mb > 0 else 0
    console.print(
        f"  Quantized: {onnx_int8_path} ({int8_size_mb:.1f} MB, "
        f"{compression:.1f}x compression)"
    )

    # --- Stage 4: Benchmark ---
    benchmark_results = []
    if not args.skip_benchmark:
        console.print(f"\n[bold blue]Stage 4: Benchmarking...[/bold blue]")
        benchmark_results = run_benchmark_matrix(
            pytorch_model=model,
            onnx_fp32_path=onnx_fp32_path,
            onnx_int8_path=onnx_int8_path,
            img_size=img_size,
            num_channels=num_channels,
            batch_sizes=batch_sizes,
            num_runs=num_runs,
            warmup_runs=warmup_runs,
        )
        print_benchmark_table(benchmark_results)

    # --- Stage 5: Accuracy ---
    accuracy_results = {}
    if not args.skip_accuracy:
        console.print(f"\n[bold blue]Stage 5: Accuracy evaluation...[/bold blue]")
        dataset_path = accuracy_cfg.get("dataset_path", config.get("data", {}).get("dataset_path", "data/hls_burn_scars"))

        from src.data.dataset import HLSBurnScarsDataModule

        dm = HLSBurnScarsDataModule(
            dataset_path=dataset_path,
            batch_size=accuracy_cfg.get("batch_size", 4),
            num_workers=accuracy_cfg.get("num_workers", 4),
        )
        datamodule = dm.build()

        console.print(f"  Evaluating FP32 ONNX...")
        fp32_acc = evaluate_onnx_accuracy(onnx_fp32_path, datamodule)
        console.print(
            f"  FP32 — mIoU: {fp32_acc['miou']:.4f}, "
            f"Burn scar IoU: {fp32_acc['iou_burn_scar']:.4f}"
        )

        console.print(f"  Evaluating INT8 ONNX...")
        int8_acc = evaluate_onnx_accuracy(onnx_int8_path, datamodule)
        console.print(
            f"  INT8 — mIoU: {int8_acc['miou']:.4f}, "
            f"Burn scar IoU: {int8_acc['iou_burn_scar']:.4f}"
        )

        accuracy_results = {"fp32": fp32_acc, "int8": int8_acc}

    # --- Save results ---
    results = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_type": args.checkpoint_type,
        "img_size": img_size,
        "onnx_fp32_path": str(onnx_fp32_path),
        "onnx_fp32_size_mb": onnx_size_mb,
        "onnx_int8_path": str(onnx_int8_path),
        "onnx_int8_size_mb": int8_size_mb,
        "compression_ratio": compression,
        "validation": val_result,
        "param_count_millions": param_count,
    }

    if benchmark_results:
        results["benchmarks"] = [asdict(r) for r in benchmark_results]

    if accuracy_results:
        results["accuracy"] = accuracy_results

    results_path = output_dir / "deployment_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    console.print(f"\n[green]Done. Results saved to {results_path}[/green]")

    # W&B logging
    if not args.no_wandb:
        try:
            import wandb

            wandb.init(
                project=config.get("project", "burn-scar-ssl"),
                name=f"phase5-{args.checkpoint_type}",
                tags=["phase5", "deployment", args.checkpoint_type],
                config=results,
            )
            wandb.log(results)
            wandb.finish()
        except Exception as e:
            console.print(f"  [yellow]W&B logging failed: {e}[/yellow]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
