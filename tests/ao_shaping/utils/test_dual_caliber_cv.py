"""Tests for `average_same_exposure` and `compute_cv_dual_caliber` (issue #165 P0, Module 4)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ao_shaping.utils.image.beam_metrics import (
    average_same_exposure,
    compute_cv_dual_caliber,
    compute_square_metrics,
)


class TestAverageSameExposure:
    """Same-exposure multi-frame averaging."""

    def test_mean_of_three_same_exposure_frames(self):
        frames = [
            np.full((4, 5), 1.0, dtype=np.float32),
            np.full((4, 5), 2.0, dtype=np.float64),
            np.arange(20, dtype=np.float64).reshape(4, 5),
        ]
        result = average_same_exposure(frames, [0.05, 0.05, 0.05])
        expected = (1.0 + 2.0 + np.arange(20, dtype=np.float64).reshape(4, 5)) / 3.0
        assert result.dtype == np.float64
        assert result.shape == (4, 5)
        np.testing.assert_allclose(result, expected)

    def test_empty_sequence_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            average_same_exposure([], [])

    def test_mixed_exposures_raise(self):
        """0.05 ms vs 0.04 ms — the bench's real exposure ladder steps."""
        frames = [np.ones((3, 3)), np.ones((3, 3))]
        with pytest.raises(ValueError, match="exposure bins"):
            average_same_exposure(frames, [0.05, 0.04])

    def test_length_mismatch_raises(self):
        frames = [np.ones((3, 3)), np.ones((3, 3))]
        with pytest.raises(ValueError, match="same length"):
            average_same_exposure(frames, [0.05])

    def test_mismatched_shapes_raise(self):
        frames = [np.ones((3, 3)), np.ones((3, 4))]
        with pytest.raises(ValueError, match="shape"):
            average_same_exposure(frames, [0.05, 0.05])

    def test_non_finite_frame_raises(self):
        bad = np.ones((3, 3))
        bad[1, 1] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            average_same_exposure([np.ones((3, 3)), bad], [0.05, 0.05])

    def test_exposures_within_rtol_accepted(self):
        """0.050 ms and 0.0509 ms differ by 1.8 % < rtol=0.02 -> one bin."""
        frames = [np.full((2, 2), 4.0), np.full((2, 2), 6.0)]
        result = average_same_exposure(frames, [0.050, 0.0509])
        np.testing.assert_allclose(result, np.full((2, 2), 5.0))

    def test_non_finite_exposure_raises(self):
        frames = [np.ones((2, 2)), np.ones((2, 2))]
        with pytest.raises(ValueError, match="finite"):
            average_same_exposure(frames, [0.05, float("nan")])


def _ripple_image(h: int = 42, w: int = 42, base: float = 2.0, amp: float = 0.3):
    """Uniform `base` plus a half-period cos ripple across the full width.

    The ripple is maximal at the left/right edges and nearly flat at the
    centre, so the central flat sub-box has a strictly smaller relative
    fluctuation than the full target box (the ordering the dual-caliber
    test asserts).
    """
    xx = np.arange(w, dtype=np.float64)
    u = (xx - (w // 2) + (w - 1) / 2.0) / (w - 1)  # u in [0, 1] across columns
    row = base + amp * np.cos(math.pi * u)
    return np.tile(row, (h, 1))


class TestComputeCvDualCaliber:
    """Dual-caliber CV on the same image."""

    def test_uniform_image_both_zero(self):
        img = np.full((32, 32), 7.5)
        out = compute_cv_dual_caliber(img, 20, (16, 16))
        assert out["cv_target_box"] == pytest.approx(0.0, abs=1e-12)
        assert out["cv_flat_rms"] == pytest.approx(0.0, abs=1e-12)
        assert out["flat_side"] == max(3, 20 // 4)

    def test_ripple_both_positive_and_flat_smaller(self):
        """Sinusoidal ripple: ordering cv_flat_rms < cv_target_box, not a magic number."""
        side = 40
        img = _ripple_image(side + 2, side + 2)
        center = ((side + 2) // 2, (side + 2) // 2)
        out = compute_cv_dual_caliber(img, side, center)
        assert out["cv_target_box"] > 0.0
        assert out["cv_flat_rms"] > 0.0
        assert out["cv_flat_rms"] < out["cv_target_box"]

    def test_flat_side_honoured(self):
        img = _ripple_image(42, 42)
        center = (21, 21)
        small = compute_cv_dual_caliber(img, 40, center, flat_side=5)
        large = compute_cv_dual_caliber(img, 40, center, flat_side=13)
        assert small["flat_side"] == 5
        assert large["flat_side"] == 13
        assert small["cv_flat_rms"] != pytest.approx(large["cv_flat_rms"])
        assert small["cv_flat_rms"] < large["cv_flat_rms"]
        assert (
            small["cv_target_box"] == pytest.approx(large["cv_target_box"])
        ), "the box caliber must not depend on flat_side"

    def test_zero_mean_region_returns_inf(self):
        """Documented choice: degenerate denominator -> float('inf'), no exception."""
        img = np.zeros((32, 32))
        out = compute_cv_dual_caliber(img, 20, (16, 16))
        assert out["cv_target_box"] == float("inf")
        assert out["cv_flat_rms"] == float("inf")

    def test_non_uniform_zero_mean_returns_inf(self):
        """Mean exactly zero (positive and negative lobes): inf, not a division."""
        img = np.tile(np.array([-1.0, 0.0, 1.0]), (32, 11))[:, :33]
        out = compute_cv_dual_caliber(img, 20, (16, 16))
        assert out["cv_target_box"] == float("inf")

    def test_box_outside_image_raises(self):
        img = np.ones((32, 32))
        with pytest.raises(ValueError, match="outside the image"):
            compute_cv_dual_caliber(img, 20, (2, 16))

    def test_non_2d_raises(self):
        with pytest.raises(ValueError, match="2D"):
            compute_cv_dual_caliber(np.ones((8, 8, 3)), 4, (4, 4))

    def test_target_side_below_one_raises(self):
        with pytest.raises(ValueError, match="target_side"):
            compute_cv_dual_caliber(np.ones((32, 32)), 0, (16, 16))

    def test_flat_side_beyond_box_raises(self):
        with pytest.raises(ValueError, match="fit inside"):
            compute_cv_dual_caliber(np.ones((32, 32)), 20, (16, 16), flat_side=25)

    def test_consistency_with_compute_square_metrics(self):
        """cv_target_box and uniformity_cv are the SAME caliber — must agree."""
        rng = np.random.default_rng(20261016)
        img = rng.uniform(1.0, 5.0, size=(64, 64))
        for side, center in [(40, (32, 32)), (21, (30, 35)), (8, (10, 50))]:
            dual = compute_cv_dual_caliber(img, side, center)
            square = compute_square_metrics(img, side, center)
            assert dual["cv_target_box"] == pytest.approx(
                square["uniformity_cv"], rel=1e-12, abs=1e-15
            )
