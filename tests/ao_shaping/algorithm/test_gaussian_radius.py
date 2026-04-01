"""Test second moment radius calculation on wf-less optimization results."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ao_shaping.algorithm.target_func import ImageTargetFunc


DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "wf-less"


def _find_latest_pkl() -> Path | None:
    """Find the most recent .pkl file under data/wf-less/."""
    if not DATA_DIR.exists():
        return None
    pkl_files = sorted(
        DATA_DIR.rglob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return pkl_files[0] if pkl_files else None


LATEST_PKL = _find_latest_pkl()


def _load_pkl_dataframe(pkl_path: Path) -> pd.DataFrame:
    """Load a wf-less pickle file and return its dataframe."""
    return pd.read_pickle(pkl_path, compression="zip")


class TestSecondMomentRadiusOnPklData:
    """Test second moment radius calculation using real wf-less optimization data."""

    @pytest.fixture
    def pkl_path(self) -> Path:
        """Return the path to the latest pkl file, or skip if none exists."""
        if LATEST_PKL is None:
            pytest.skip("No wf-less pkl files found in data/wf-less/")
        return LATEST_PKL

    @pytest.fixture
    def df(self, pkl_path: Path) -> pd.DataFrame:
        """Load the dataframe from the pkl file."""
        return _load_pkl_dataframe(pkl_path)

    def test_pkl_file_loaded(self, pkl_path: Path, df: pd.DataFrame):
        """Verify the pkl file loads correctly and has expected columns."""
        assert df.shape[0] > 0, "Dataframe should not be empty"
        assert "_img" in df.columns, "Dataframe should have '_img' column"
        assert "pib" in df.columns, "Dataframe should have 'pib' column"

    def test_image_shape(self, df: pd.DataFrame):
        """Verify images have expected shape."""
        img = df.iloc[0]["_img"]
        assert img is not None, "First image should not be None"
        assert img.ndim == 2, "Image should be 2D"
        assert img.shape[0] > 0 and img.shape[1] > 0, (
            "Image should have non-zero dimensions"
        )

    def test_second_moment_radius_on_first_image(self, df: pd.DataFrame):
        """Test second moment radius calculation on the first image."""
        img = df.iloc[0]["_img"]
        target = ImageTargetFunc.build_from_init_image(img)
        radius = target.second_moment_radius(img)
        # Second moment radius should always be positive
        assert radius > 0, f"Second moment radius should be positive, got {radius}"
        # Reasonable upper bound for a 200x200 image
        assert radius < 100, f"Second moment radius seems too large: {radius}"

    def test_second_moment_radius_on_pib_best_image(self, df: pd.DataFrame):
        """Test second moment radius on the image with highest PIB."""
        best_pib_idx = df["pib"].idxmax()
        img = df.iloc[best_pib_idx]["_img"]
        target = ImageTargetFunc.build_from_init_image(img)
        radius = target.second_moment_radius(img)
        assert radius > 0, f"Second moment radius should be positive, got {radius}"

    def test_second_moment_radius_on_all_images(self, df: pd.DataFrame):
        """Calculate second moment radius for all images and verify consistency."""
        # Sample every 100th image to keep test fast
        sample_df = df.iloc[::100]
        radii = []
        for _, row in sample_df.iterrows():
            img = row["_img"]
            if img is not None:
                target = ImageTargetFunc.build_from_init_image(img)
                r = target.second_moment_radius(img)
                radii.append(r)

        assert len(radii) > 0, "Should have calculated radii"

        # All radii should be positive
        for r in radii:
            assert r > 0, f"Radius should be positive, got {r}"

    def test_minimum_second_moment_radius_selection(self, df: pd.DataFrame):
        """Test finding the image with minimum second moment radius."""
        # Test on a subset for speed
        sample_df = df.iloc[:500] if len(df) > 500 else df

        radii = []
        for idx, row in sample_df.iterrows():
            img = row["_img"]
            if img is not None:
                target = ImageTargetFunc.build_from_init_image(img)
                r = target.second_moment_radius(img)
                radii.append((idx, r))

        assert len(radii) > 0, "Should have at least one valid radius"

        min_idx, min_radius = min(radii, key=lambda x: x[1])
        assert min_radius > 0, "Minimum radius should be positive"

        # Verify that this is indeed the minimum
        for _, r in radii:
            assert r >= min_radius, f"Found radius {r} less than minimum {min_radius}"

    def test_second_moment_vs_pib_best_comparison(self, df: pd.DataFrame):
        """Compare the PIB-best image vs second-moment-radius-best image."""
        sample_df = df.iloc[:500] if len(df) > 500 else df

        pib_best_idx = sample_df["pib"].idxmax()

        radii = []
        for idx, row in sample_df.iterrows():
            img = row["_img"]
            if img is not None:
                target = ImageTargetFunc.build_from_init_image(img)
                r = target.second_moment_radius(img)
                radii.append((idx, r))

        if len(radii) > 0:
            best_idx, best_radius = min(radii, key=lambda x: x[1])

            # Both indices should be valid
            assert 0 <= best_idx < len(df)
            assert 0 <= pib_best_idx < len(df)

            # The best image should have a reasonable radius
            assert best_radius > 0
            assert best_radius < 100


class TestSecondMomentRadiusFunction:
    """Unit tests for the second moment radius function."""

    def test_second_moment_radius_on_synthetic_gaussian(self):
        """Test radius calculation on a known synthetic Gaussian beam."""
        size = 100
        center = (50, 50)
        true_sigma = 8.0

        # Create synthetic 2D Gaussian
        y, x = np.ogrid[:size, :size]
        img = np.exp(
            -((x - center[0]) ** 2 + (y - center[1]) ** 2) / (2 * true_sigma**2)
        )
        img = (img * 1000).astype(np.uint16)

        target = ImageTargetFunc.build_from_init_image(img)
        radius = target.second_moment_radius(img)

        # For a 2D Gaussian, second moment radius = sqrt(2) * sigma
        expected_r2 = np.sqrt(2) * true_sigma
        assert abs(radius - expected_r2) / expected_r2 < 0.3, (
            f"Second moment radius {radius} should be close to {expected_r2}"
        )

    def test_second_moment_radius_on_uniform_image(self):
        """Test second moment radius on uniform image."""
        img = np.ones((50, 50), dtype=np.uint16) * 100
        target = ImageTargetFunc.build_from_init_image(img)
        radius = target.second_moment_radius(img)
        # Uniform image should give a large radius (spread across entire image)
        assert radius > 0, "Radius should be positive"
        # For a 50x50 uniform image, second moment radius should be substantial
        assert radius > 10, f"Uniform image should give large radius, got {radius}"

    def test_second_moment_radius_on_noise_image(self):
        """Test second moment radius on noise image."""
        np.random.seed(42)
        img = np.random.randint(0, 100, (50, 50), dtype=np.uint16)
        target = ImageTargetFunc.build_from_init_image(img)
        radius = target.second_moment_radius(img)
        # Noise image should still give a positive radius
        assert radius > 0, "Radius should be positive"

    def test_second_moment_radius_different_sigmas(self):
        """Test radius calculation on Gaussians with different widths."""
        size = 100
        center = (50, 50)

        sigmas = [5.0, 10.0, 15.0]
        radii = []

        for sigma in sigmas:
            y, x = np.ogrid[:size, :size]
            img = np.exp(
                -((x - center[0]) ** 2 + (y - center[1]) ** 2) / (2 * sigma**2)
            )
            img = (img * 1000).astype(np.uint16)

            target = ImageTargetFunc.build_from_init_image(img)
            r = target.second_moment_radius(img)
            radii.append(r)

        # All radii should be positive
        for r in radii:
            assert r > 0, f"Radius should be positive, got {r}"

        # Larger sigma should give larger second moment radius
        for i in range(len(radii) - 1):
            assert radii[i] < radii[i + 1], (
                f"Radius should increase with sigma: {radii[i]} vs {radii[i + 1]}"
            )

    def test_second_moment_radius_on_zero_image(self):
        """Test second moment radius on all-zero image."""
        img = np.zeros((50, 50), dtype=np.uint16)
        target = ImageTargetFunc.build_from_init_image(img)
        radius = target.second_moment_radius(img)
        # Zero image should return 0
        assert radius == 0.0, f"Zero image should give radius 0, got {radius}"
