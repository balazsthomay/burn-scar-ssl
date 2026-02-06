#!/usr/bin/env python3
"""Generate stratified training subsets for label efficiency experiments.

This script reads the training split and generates subset split files
at various label fractions (5%, 10%, 25%, 50%) while preserving the
distribution of burn scar prevalence through stratified sampling.

Usage:
    uv run scripts/generate_subsets.py
    uv run scripts/generate_subsets.py --fractions 0.05 0.10 0.25
    uv run scripts/generate_subsets.py --seed 123
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.table import Table

from src.data.subset_generator import (
    compute_all_prevalences,
    generate_stratified_subsets,
)

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Generate stratified training subsets"
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/hls_burn_scars"),
        help="Path to HLS burn scars dataset",
    )
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.25, 0.50],
        help="Label fractions to generate",
    )
    parser.add_argument(
        "--n-strata",
        type=int,
        default=4,
        help="Number of prevalence strata",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # Validate paths
    splits_dir = args.dataset_path / "splits"
    data_dir = args.dataset_path / "data"

    if not splits_dir.exists():
        console.print(f"[red]Splits directory not found: {splits_dir}[/red]")
        console.print("Run scripts/download_data.py first.")
        return 1

    train_split = splits_dir / "train.txt"
    if not train_split.exists():
        console.print(f"[red]Training split not found: {train_split}[/red]")
        return 1

    # Load training samples
    train_samples = train_split.read_text().strip().split("\n")
    console.print(f"[green]Loaded {len(train_samples)} training samples[/green]")

    # Compute prevalences for reporting
    console.print("[blue]Computing burn scar prevalences...[/blue]")
    prevalences = compute_all_prevalences(train_samples, data_dir)

    if not prevalences:
        console.print("[red]No valid samples found![/red]")
        return 1

    prev_values = list(prevalences.values())
    console.print(
        f"Prevalence stats: min={min(prev_values):.3f}, "
        f"max={max(prev_values):.3f}, "
        f"mean={sum(prev_values)/len(prev_values):.3f}"
    )

    # Generate subsets
    console.print(f"\n[blue]Generating subsets with seed={args.seed}...[/blue]")
    subsets = generate_stratified_subsets(
        train_samples=train_samples,
        mask_dir=data_dir,
        fractions=args.fractions,
        n_strata=args.n_strata,
        seed=args.seed,
        output_dir=splits_dir,
    )

    # Display results
    table = Table(title="Generated Subsets")
    table.add_column("Fraction", justify="right")
    table.add_column("Samples", justify="right")
    table.add_column("File", justify="left")
    table.add_column("Prevalence Mean", justify="right")
    table.add_column("Prevalence Std", justify="right")

    for frac in sorted(args.fractions):
        samples = subsets[frac]
        pct = int(frac * 100)
        filename = f"train_{pct}pct.txt"

        # Compute subset prevalence stats
        subset_prev = [prevalences[s] for s in samples if s in prevalences]
        mean_prev = sum(subset_prev) / len(subset_prev) if subset_prev else 0
        std_prev = (
            (sum((p - mean_prev) ** 2 for p in subset_prev) / len(subset_prev)) ** 0.5
            if subset_prev else 0
        )

        table.add_row(
            f"{pct}%",
            str(len(samples)),
            filename,
            f"{mean_prev:.4f}",
            f"{std_prev:.4f}",
        )

    # Add full training set row
    full_mean = sum(prev_values) / len(prev_values)
    full_std = (sum((p - full_mean) ** 2 for p in prev_values) / len(prev_values)) ** 0.5
    table.add_row(
        "100%",
        str(len(train_samples)),
        "train.txt",
        f"{full_mean:.4f}",
        f"{full_std:.4f}",
        style="bold",
    )

    console.print(table)

    # Verify nested structure
    console.print("\n[blue]Verifying nested subset structure...[/blue]")
    sorted_fracs = sorted(args.fractions)
    all_nested = True
    for i in range(len(sorted_fracs) - 1):
        smaller = set(subsets[sorted_fracs[i]])
        larger = set(subsets[sorted_fracs[i + 1]])
        is_nested = smaller.issubset(larger)
        status = "[green]✓[/green]" if is_nested else "[red]✗[/red]"
        console.print(
            f"  {status} {int(sorted_fracs[i]*100)}% ⊂ {int(sorted_fracs[i+1]*100)}%"
        )
        all_nested = all_nested and is_nested

    if all_nested:
        console.print("[green]All subsets properly nested![/green]")
    else:
        console.print("[yellow]Warning: Some subsets are not properly nested[/yellow]")

    console.print(f"\n[green]Subset files written to {splits_dir}[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
