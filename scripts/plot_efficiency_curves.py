#!/usr/bin/env python3
"""Plot label efficiency curves from Phase 2 sweep results.

Generates a figure showing IoU vs label fraction for each backbone,
with error bars from multiple seeds.

Usage:
    uv run scripts/plot_efficiency_curves.py
    uv run scripts/plot_efficiency_curves.py --results outputs/phase2/all_results.json
    uv run scripts/plot_efficiency_curves.py --output figures/efficiency_curves.png
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()

# Color scheme for backbones
BACKBONE_COLORS = {
    "prithvi_eo_v2_300": "#2ecc71",  # Green
    "dinov3_vitl16_sat": "#3498db",  # Blue
    "resnet50": "#e74c3c",           # Red
}

BACKBONE_LABELS = {
    "prithvi_eo_v2_300": "Prithvi-EO-2.0-300M",
    "dinov3_vitl16_sat": "DINOv3-ViT-L/16 (SAT)",
    "resnet50": "ResNet-50 (ImageNet)",
}


def load_results(results_path: Path) -> list[dict]:
    """Load results from JSON file."""
    with open(results_path) as f:
        return json.load(f)


def aggregate_results(results: list[dict]) -> dict:
    """Aggregate results by backbone and fraction.

    Returns dict: backbone -> fraction -> {"mean": float, "std": float, "values": list}
    """
    # Group by backbone and fraction
    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        backbone = r["backbone"]
        fraction = r["fraction"]
        iou = r["test_iou_burn"]
        grouped[backbone][fraction].append(iou)

    # Compute mean and std
    aggregated = {}
    for backbone, fractions in grouped.items():
        aggregated[backbone] = {}
        for fraction, values in fractions.items():
            aggregated[backbone][fraction] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "values": values,
            }

    return aggregated


def plot_efficiency_curves(
    aggregated: dict,
    output_path: Path,
    title: str = "Label Efficiency: IoU vs Training Labels",
):
    """Generate label efficiency curve plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for backbone, fractions in sorted(aggregated.items()):
        sorted_fractions = sorted(fractions.keys())
        means = [fractions[f]["mean"] for f in sorted_fractions]
        stds = [fractions[f]["std"] for f in sorted_fractions]
        x = [f * 100 for f in sorted_fractions]  # Convert to percentage

        color = BACKBONE_COLORS.get(backbone, "#888888")
        label = BACKBONE_LABELS.get(backbone, backbone)

        ax.errorbar(
            x,
            means,
            yerr=stds,
            label=label,
            color=color,
            marker="o",
            markersize=8,
            linewidth=2,
            capsize=4,
            capthick=2,
        )

    ax.set_xlabel("Training Labels (%)", fontsize=12)
    ax.set_ylabel("Test IoU (Burn Scar Class)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 1)

    # Add percentage ticks
    ax.set_xticks([5, 10, 25, 50, 100])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    console.print(f"[green]Saved plot to {output_path}[/green]")


def print_results_table(aggregated: dict):
    """Print results as a rich table."""
    table = Table(title="Label Efficiency Results (Test IoU - Burn Scar)")

    # Get all fractions
    all_fractions = set()
    for fractions in aggregated.values():
        all_fractions.update(fractions.keys())
    sorted_fractions = sorted(all_fractions)

    # Add columns
    table.add_column("Backbone", style="bold")
    for f in sorted_fractions:
        table.add_column(f"{int(f*100)}%", justify="right")

    # Add rows
    for backbone in sorted(aggregated.keys()):
        label = BACKBONE_LABELS.get(backbone, backbone)
        row = [label]
        for f in sorted_fractions:
            if f in aggregated[backbone]:
                mean = aggregated[backbone][f]["mean"]
                std = aggregated[backbone][f]["std"]
                row.append(f"{mean:.3f} ± {std:.3f}")
            else:
                row.append("-")
        table.add_row(*row)

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Plot label efficiency curves")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("outputs/phase2/all_results.json"),
        help="Path to results JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/phase2/label_efficiency_curves.png"),
        help="Output path for the plot",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Label Efficiency: IoU vs Training Labels",
        help="Plot title",
    )
    args = parser.parse_args()

    # Check if results exist
    if not args.results.exists():
        console.print(f"[red]Results file not found: {args.results}[/red]")
        console.print("Run scripts/run_phase2_sweep.py first to generate results.")
        return 1

    # Load and aggregate results
    console.print(f"Loading results from {args.results}")
    results = load_results(args.results)
    console.print(f"Found {len(results)} experiment results")

    aggregated = aggregate_results(results)

    # Print table
    print_results_table(aggregated)

    # Generate plot
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_efficiency_curves(aggregated, args.output, args.title)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
