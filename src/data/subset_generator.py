"""Stratified subset generation for label efficiency experiments.

Generates training subsets that preserve the distribution of burn scar prevalence
across different label fractions.
"""

from pathlib import Path

import numpy as np
import rasterio


def get_mask_path_for_sample(sample_id: str, mask_dir: Path) -> Path:
    """Get the mask file path for a given sample ID.

    Args:
        sample_id: Sample identifier (e.g., "T10SEJ.2018185.v1").
        mask_dir: Directory containing mask files.

    Returns:
        Path to the mask file.

    Raises:
        FileNotFoundError: If mask file doesn't exist.
    """
    mask_path = mask_dir / f"subsetted_512x512_HLS.S30.{sample_id}.4.mask.tif"

    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    return mask_path


def compute_sample_prevalence(mask_path: Path) -> float:
    """Calculate burn scar prevalence for a single mask.

    Prevalence is the fraction of pixels that are burn scars (class 1),
    excluding no-data pixels.

    Args:
        mask_path: Path to mask GeoTiff file.

    Returns:
        Burn scar prevalence in [0, 1].

    Raises:
        FileNotFoundError: If mask file doesn't exist.
    """
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    with rasterio.open(mask_path) as src:
        mask = src.read(1)

    # Valid pixels are 0 (not burned) or 1 (burn scar)
    # -1 or other values might indicate no-data
    valid_mask = (mask == 0) | (mask == 1)
    n_valid = valid_mask.sum()

    if n_valid == 0:
        return 0.0

    n_burn = (mask == 1).sum()
    return float(n_burn) / float(n_valid)


def generate_stratified_subsets(
    train_samples: list[str],
    mask_dir: Path,
    fractions: list[float],
    n_strata: int = 4,
    seed: int = 42,
    output_dir: Path | None = None,
) -> dict[float, list[str]]:
    """Generate stratified subsets preserving burn scar prevalence distribution.

    Uses stratified sampling to ensure subsets maintain similar distribution
    of burn scar prevalence as the full training set. Smaller subsets are
    guaranteed to be subsets of larger ones (nested structure).

    Args:
        train_samples: List of sample IDs from training set.
        mask_dir: Directory containing mask files.
        fractions: List of fractions (e.g., [0.05, 0.10, 0.25, 0.50]).
        n_strata: Number of strata (bins) for prevalence.
        seed: Random seed for reproducibility.
        output_dir: If provided, write split files to this directory.

    Returns:
        Dictionary mapping fraction to list of sample IDs.
    """
    rng = np.random.default_rng(seed)

    # Compute prevalence for each sample
    prevalences = []
    valid_samples = []
    for sample_id in train_samples:
        try:
            mask_path = get_mask_path_for_sample(sample_id, mask_dir)
            prev = compute_sample_prevalence(mask_path)
            prevalences.append(prev)
            valid_samples.append(sample_id)
        except FileNotFoundError:
            continue

    prevalences = np.array(prevalences)
    n_samples = len(valid_samples)

    # Assign samples to strata based on prevalence quantiles
    quantiles = np.linspace(0, 1, n_strata + 1)
    thresholds = np.quantile(prevalences, quantiles)
    strata_indices = np.digitize(prevalences, thresholds[1:-1])  # 0 to n_strata-1

    # For each stratum, shuffle indices once (for reproducibility)
    stratum_shuffled_indices: dict[int, list[int]] = {}
    for s in range(n_strata):
        indices = np.where(strata_indices == s)[0]
        rng.shuffle(indices)
        stratum_shuffled_indices[s] = indices.tolist()

    # Build subsets: for each fraction, take the first N samples per stratum
    # This naturally creates nested subsets (smaller is prefix of larger)
    subsets: dict[float, list[str]] = {}

    for frac in fractions:
        target_size = int(frac * n_samples)
        target_per_stratum = target_size // n_strata
        remainder = target_size % n_strata

        current_subset = []

        for s in range(n_strata):
            # Take target_per_stratum samples from this stratum
            # Distribute remainder across first few strata
            n_to_take = target_per_stratum + (1 if s < remainder else 0)
            n_to_take = min(n_to_take, len(stratum_shuffled_indices[s]))

            selected_indices = stratum_shuffled_indices[s][:n_to_take]
            current_subset.extend([valid_samples[i] for i in selected_indices])

        subsets[frac] = current_subset

    # Write split files if output_dir provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for frac, samples in subsets.items():
            pct = int(frac * 100)
            split_file = output_dir / f"train_{pct}pct.txt"
            split_file.write_text("\n".join(samples) + "\n")

    return subsets


def compute_all_prevalences(
    samples: list[str],
    mask_dir: Path,
) -> dict[str, float]:
    """Compute prevalences for all samples.

    Args:
        samples: List of sample IDs.
        mask_dir: Directory containing mask files.

    Returns:
        Dictionary mapping sample ID to prevalence.
    """
    prevalences = {}
    for sample_id in samples:
        try:
            mask_path = get_mask_path_for_sample(sample_id, mask_dir)
            prevalences[sample_id] = compute_sample_prevalence(mask_path)
        except FileNotFoundError:
            continue
    return prevalences
