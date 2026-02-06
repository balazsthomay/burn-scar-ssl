"""Tests for stratified subset generation."""

from pathlib import Path
import tempfile

import numpy as np
import pytest
import rasterio

from src.data.subset_generator import (
    compute_sample_prevalence,
    generate_stratified_subsets,
    get_mask_path_for_sample,
)


@pytest.fixture
def temp_mask_dir():
    """Create temporary directory with synthetic mask files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        # Create synthetic masks with known prevalences
        # Format: subsetted_512x512_HLS.S30.{sample_id}.4.mask.tif
        samples = []
        prevalences = []

        for i, prev in enumerate([0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0] * 10):
            sample_id = f"T{i:02d}ABC.2020{i:03d}.v1"
            samples.append(sample_id)
            prevalences.append(prev)

            mask_path = tmpdir / f"subsetted_512x512_HLS.S30.{sample_id}.4.mask.tif"

            # Create mask with specified prevalence
            mask = np.zeros((512, 512), dtype=np.uint8)
            n_burn_pixels = int(prev * 512 * 512)
            if n_burn_pixels > 0:
                # Set first n_burn_pixels to 1
                flat_mask = mask.flatten()
                flat_mask[:n_burn_pixels] = 1
                mask = flat_mask.reshape(512, 512)

            # Write as GeoTiff
            with rasterio.open(
                mask_path,
                'w',
                driver='GTiff',
                height=512,
                width=512,
                count=1,
                dtype=np.uint8,
            ) as dst:
                dst.write(mask, 1)

        yield tmpdir, samples, prevalences


class TestGetMaskPath:
    """Tests for mask path resolution."""

    def test_get_mask_path_format(self, temp_mask_dir):
        """Should construct correct mask path from sample ID."""
        tmpdir, samples, _ = temp_mask_dir
        sample_id = samples[0]

        mask_path = get_mask_path_for_sample(sample_id, tmpdir)
        assert mask_path.exists()
        assert ".mask.tif" in mask_path.name
        assert sample_id in mask_path.name


class TestComputePrevalence:
    """Tests for burn scar prevalence computation."""

    def test_prevalence_zero(self, temp_mask_dir):
        """Should return 0.0 for mask with no burn scars."""
        tmpdir, samples, _ = temp_mask_dir
        # First sample has 0% prevalence
        sample_id = samples[0]
        mask_path = get_mask_path_for_sample(sample_id, tmpdir)

        prevalence = compute_sample_prevalence(mask_path)
        assert prevalence == pytest.approx(0.0, abs=0.001)

    def test_prevalence_full(self, temp_mask_dir):
        """Should return 1.0 for fully burned mask."""
        tmpdir, samples, _ = temp_mask_dir
        # 8th sample (index 7) has 100% prevalence
        sample_id = samples[7]
        mask_path = get_mask_path_for_sample(sample_id, tmpdir)

        prevalence = compute_sample_prevalence(mask_path)
        assert prevalence == pytest.approx(1.0, abs=0.001)

    def test_prevalence_partial(self, temp_mask_dir):
        """Should return correct prevalence for partial burns."""
        tmpdir, samples, _ = temp_mask_dir
        # Sample with 25% prevalence
        sample_id = samples[3]  # 0.25 prevalence
        mask_path = get_mask_path_for_sample(sample_id, tmpdir)

        prevalence = compute_sample_prevalence(mask_path)
        assert prevalence == pytest.approx(0.25, abs=0.001)

    def test_prevalence_nonexistent_file(self):
        """Should raise error for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            compute_sample_prevalence(Path("/nonexistent/mask.tif"))


class TestGenerateStratifiedSubsets:
    """Tests for stratified subset generation."""

    def test_subset_sizes(self, temp_mask_dir):
        """Subsets should have correct sizes."""
        tmpdir, samples, _ = temp_mask_dir
        fractions = [0.1, 0.25, 0.5]

        subsets = generate_stratified_subsets(
            train_samples=samples,
            mask_dir=tmpdir,
            fractions=fractions,
            n_strata=4,
            seed=42,
        )

        n_total = len(samples)
        for frac in fractions:
            expected_size = int(frac * n_total)
            # Allow ±1 due to rounding
            assert abs(len(subsets[frac]) - expected_size) <= 2

    def test_subset_contains_valid_samples(self, temp_mask_dir):
        """Subset samples should be from original samples."""
        tmpdir, samples, _ = temp_mask_dir

        subsets = generate_stratified_subsets(
            train_samples=samples,
            mask_dir=tmpdir,
            fractions=[0.25],
            n_strata=4,
            seed=42,
        )

        for sample_id in subsets[0.25]:
            assert sample_id in samples

    def test_reproducibility(self, temp_mask_dir):
        """Same seed should produce identical subsets."""
        tmpdir, samples, _ = temp_mask_dir

        subsets1 = generate_stratified_subsets(
            train_samples=samples,
            mask_dir=tmpdir,
            fractions=[0.25],
            n_strata=4,
            seed=42,
        )

        subsets2 = generate_stratified_subsets(
            train_samples=samples,
            mask_dir=tmpdir,
            fractions=[0.25],
            n_strata=4,
            seed=42,
        )

        assert subsets1[0.25] == subsets2[0.25]

    def test_different_seeds_different_results(self, temp_mask_dir):
        """Different seeds should (usually) produce different subsets."""
        tmpdir, samples, _ = temp_mask_dir

        subsets1 = generate_stratified_subsets(
            train_samples=samples,
            mask_dir=tmpdir,
            fractions=[0.5],
            n_strata=4,
            seed=42,
        )

        subsets2 = generate_stratified_subsets(
            train_samples=samples,
            mask_dir=tmpdir,
            fractions=[0.5],
            n_strata=4,
            seed=123,
        )

        # With 50% of 80 samples, very unlikely to be identical
        assert subsets1[0.5] != subsets2[0.5]

    def test_stratification_preserves_distribution(self, temp_mask_dir):
        """Subsets should preserve prevalence distribution across strata."""
        tmpdir, samples, prevalences = temp_mask_dir

        subsets = generate_stratified_subsets(
            train_samples=samples,
            mask_dir=tmpdir,
            fractions=[0.5],
            n_strata=4,
            seed=42,
        )

        # Get prevalences for subset
        subset_samples = subsets[0.5]
        subset_prevalences = []
        for sample_id in subset_samples:
            idx = samples.index(sample_id)
            subset_prevalences.append(prevalences[idx])

        # Check that subset spans similar range as original
        orig_min, orig_max = min(prevalences), max(prevalences)
        sub_min, sub_max = min(subset_prevalences), max(subset_prevalences)

        # Subset should have samples from low and high prevalence
        assert sub_min <= 0.15  # Has low-prevalence samples
        assert sub_max >= 0.85  # Has high-prevalence samples

    def test_no_duplicate_samples(self, temp_mask_dir):
        """Subsets should not contain duplicate samples."""
        tmpdir, samples, _ = temp_mask_dir

        subsets = generate_stratified_subsets(
            train_samples=samples,
            mask_dir=tmpdir,
            fractions=[0.5],
            n_strata=4,
            seed=42,
        )

        subset = subsets[0.5]
        assert len(subset) == len(set(subset))

    def test_nested_subsets(self, temp_mask_dir):
        """Smaller fraction subsets should be subsets of larger ones."""
        tmpdir, samples, _ = temp_mask_dir

        subsets = generate_stratified_subsets(
            train_samples=samples,
            mask_dir=tmpdir,
            fractions=[0.1, 0.25, 0.5],
            n_strata=4,
            seed=42,
        )

        # 10% should be subset of 25%
        assert set(subsets[0.1]).issubset(set(subsets[0.25]))
        # 25% should be subset of 50%
        assert set(subsets[0.25]).issubset(set(subsets[0.5]))


class TestIntegrationWithRealData:
    """Integration tests with actual dataset (skip if not available)."""

    DATASET_PATH = Path("data/hls_burn_scars")

    @pytest.mark.skipif(
        not (DATASET_PATH / "data").is_dir(),
        reason="Dataset not downloaded",
    )
    def test_generate_subsets_real_data(self):
        """Should generate subsets from real training data."""
        train_split = self.DATASET_PATH / "splits" / "train.txt"
        samples = train_split.read_text().strip().split("\n")

        subsets = generate_stratified_subsets(
            train_samples=samples,
            mask_dir=self.DATASET_PATH / "data",
            fractions=[0.05, 0.10, 0.25, 0.50],
            n_strata=4,
            seed=42,
        )

        # Check sizes: 524 total train samples
        assert 24 <= len(subsets[0.05]) <= 28  # ~26
        assert 50 <= len(subsets[0.10]) <= 54  # ~52
        assert 129 <= len(subsets[0.25]) <= 133  # ~131
        assert 260 <= len(subsets[0.50]) <= 264  # ~262
