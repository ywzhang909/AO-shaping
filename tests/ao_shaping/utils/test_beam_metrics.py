"""Regression-anchor tests for :mod:`ao_shaping.utils.beam_metrics`.

Pins the exact current behavior of the pure-NumPy beam-shaping metrics
(compute_metrics, compute_shaping_metrics, compute_square_metrics,
compute_quality_score, measure_bright_span, clamp_side,
measure_spot_diameter_cam, intensity_to_amplitude, normalize_pattern).

These tests are intentionally read-only anchors: they describe what the
source does today so future refactors of callers can be verified against
a fixed baseline.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ao_shaping.utils.image.beam_metrics import (
    clamp_side,
    compute_metrics,
    compute_quality_score,
    compute_shaping_metrics,
    compute_square_metrics,
    intensity_to_amplitude,
    measure_bright_span,
    measure_spot_diameter_cam,
    normalize_pattern,
)


class TestIntensityToAmplitude:
    def test_sqrt_and_normalize_to_peak(self):
        # sqrt([0,1,4,9]) = [0,1,2,3]; peak=3 -> [0, 1/3, 2/3, 1]
        amp = intensity_to_amplitude(np.array([[0.0, 1.0], [4.0, 9.0]]))
        np.testing.assert_allclose(amp, [[0.0, 1 / 3], [2 / 3, 1.0]], rtol=1e-6)
        assert amp.dtype == np.float32

    def test_no_normalize_keeps_raw_sqrt(self):
        amp = intensity_to_amplitude(np.array([[0.0, 1.0], [4.0, 9.0]]), normalize=False)
        np.testing.assert_allclose(amp, [[0.0, 1.0], [2.0, 3.0]], rtol=1e-6)

    def test_zero_intensity_returns_zeros(self):
        amp = intensity_to_amplitude(np.zeros((2, 2)))
        np.testing.assert_array_equal(amp, np.zeros((2, 2), dtype=np.float32))

    def test_negative_values_sqrt_is_nan_free(self):
        # sqrt of negatives is NaN; normalize guard (amax>0) still divides
        with np.errstate(invalid="ignore"):
            amp = intensity_to_amplitude(np.array([[-4.0, 0.0]]))
        assert np.isnan(amp[0, 0])
        assert amp[0, 1] == 0.0


class TestNormalizePattern:
    def test_peak_mode_scales_to_one(self):
        out = normalize_pattern(np.array([[0.0, 2.0], [4.0, 8.0]]))
        np.testing.assert_allclose(out, [[0.0, 0.25], [0.5, 1.0]], rtol=1e-6)
        assert out.dtype == np.float32

    def test_sum_mode_scales_to_unit_energy(self):
        out = normalize_pattern(np.array([[1.0, 2.0], [3.0, 4.0]]), mode="sum")
        np.testing.assert_allclose(out, [[0.1, 0.2], [0.3, 0.4]], rtol=1e-6)

    def test_zero_pattern_unchanged(self):
        out = normalize_pattern(np.zeros((2, 2)))
        np.testing.assert_array_equal(out, np.zeros((2, 2), dtype=np.float32))
        out_sum = normalize_pattern(np.zeros((2, 2)), mode="sum")
        np.testing.assert_array_equal(out_sum, np.zeros((2, 2), dtype=np.float32))

    def test_nan_inf_replaced_by_zero(self):
        out = normalize_pattern(np.array([[np.nan, 1.0], [np.inf, -np.inf]]))
        np.testing.assert_allclose(out, [[0.0, 1.0], [0.0, 0.0]], rtol=1e-6)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown normalize mode"):
            normalize_pattern(np.ones((2, 2)), mode="bogus")


class TestMeasureSpotDiameterCam:
    @staticmethod
    def _uniform_disk(radius: float, cy: float, cx: float, size: int = 41) -> np.ndarray:
        y, x = np.mgrid[0:size, 0:size]
        return ((x - cx) ** 2 + (y - cy) ** 2 <= radius**2).astype(float)

    def test_uniform_disk_90_percent_energy(self):
        # Uniform disk radius 10: the 90%-encircled-energy radius is
        # sqrt(0.9)*10 (area-proportional), diameter = 2*sqrt(0.9)*10.
        img = self._uniform_disk(10.0, 20.0, 20.0)
        d = measure_spot_diameter_cam(img, energy=0.90)
        assert d == pytest.approx(2 * np.sqrt(0.9) * 10, abs=0.1)

    def test_uniform_disk_50_percent_energy(self):
        img = self._uniform_disk(10.0, 20.0, 20.0)
        d = measure_spot_diameter_cam(img, energy=0.50)
        assert d == pytest.approx(2 * np.sqrt(0.5) * 10, abs=0.1)

    def test_offcenter_disk_uses_centroid(self):
        img = self._uniform_disk(10.0, 12.0, 15.0)
        d = measure_spot_diameter_cam(img, energy=0.90)
        assert d == pytest.approx(2 * np.sqrt(0.9) * 10, abs=0.1)


class TestComputeMetrics:
    def test_identical_uniform_patterns(self):
        m = compute_metrics(np.ones((2, 2)), np.ones((2, 2)))
        assert m["mse"] == 0.0
        assert m["correlation"] == 1.0
        assert m["efficiency"] == 1.0

    def test_scale_invariance(self):
        # Both maps are normalized to unit sum, so absolute scale is irrelevant.
        a = compute_metrics(np.ones((2, 2)), np.ones((2, 2)))
        b = compute_metrics(2 * np.ones((2, 2)), 5 * np.ones((2, 2)))
        assert b["mse"] == pytest.approx(a["mse"])
        assert b["correlation"] == pytest.approx(a["correlation"])
        assert b["efficiency"] == pytest.approx(a["efficiency"])

    def test_uniform_vs_zero(self):
        m = compute_metrics(np.ones((2, 2)), np.zeros((2, 2)))
        # measured normalized to 0.25 each; target stays 0
        assert m["mse"] == pytest.approx(0.0625)
        assert m["correlation"] == 0.0  # zero-variance guard -> not allclose
        assert m["efficiency"] == 0.0

    def test_zero_vs_zero_safe_branch(self):
        m = compute_metrics(np.zeros((2, 2)), np.zeros((2, 2)))
        assert m["mse"] == 0.0
        assert m["correlation"] == 1.0  # zero-variance guard -> allclose
        assert m["efficiency"] == 0.0  # total == 0 -> safe branch

    def test_hand_computed_correlation(self):
        # measured=[[1,0],[0,0]], target=[[0,1],[0,0]] (both already unit sum)
        m = compute_metrics(np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([[0.0, 1.0], [0.0, 0.0]]))
        assert m["mse"] == pytest.approx(0.5)
        assert m["correlation"] == pytest.approx(-1 / 3)
        assert m["efficiency"] == 0.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_metrics(np.ones((2, 2)), np.ones((3, 3)))


class TestComputeShapingMetrics:
    def test_uniform_full_mask(self):
        m = compute_shaping_metrics(np.ones((4, 4)), np.ones((4, 4), dtype=bool))
        assert m["uniformity_cv"] == 0.0
        assert m["encircled_energy"] == 1.0
        assert m["peak"] == 1.0
        assert m["in_mask_mean"] == 1.0

    def test_hand_computed_cv(self):
        # in-mask values [1,2,3,4]: mean 2.5, std sqrt(1.25)
        m = compute_shaping_metrics(np.array([[1.0, 2.0], [3.0, 4.0]]), np.ones((2, 2), dtype=bool))
        assert m["uniformity_cv"] == pytest.approx(np.sqrt(1.25) / 2.5)
        assert m["encircled_energy"] == 1.0
        assert m["peak"] == 4.0
        assert m["in_mask_mean"] == 2.5

    def test_partial_mask(self):
        m = compute_shaping_metrics(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[True, False], [False, False]]),
        )
        assert m["uniformity_cv"] == 0.0  # single in-mask value
        assert m["encircled_energy"] == pytest.approx(0.1)
        assert m["peak"] == 4.0
        assert m["in_mask_mean"] == 1.0

    def test_empty_mask_safe_branch(self):
        m = compute_shaping_metrics(np.ones((2, 2)), np.zeros((2, 2), dtype=bool))
        assert m == {
            "uniformity_cv": 0.0,
            "encircled_energy": 0.0,
            "peak": 0.0,
            "in_mask_mean": 0.0,
        }

    def test_zero_total_safe_branch(self):
        m = compute_shaping_metrics(np.zeros((2, 2)), np.ones((2, 2), dtype=bool))
        assert m == {
            "uniformity_cv": 0.0,
            "encircled_energy": 0.0,
            "peak": 0.0,
            "in_mask_mean": 0.0,
        }

    def test_zero_in_mask_mean_guard(self):
        # mask covers only zero pixels -> cv guard returns 0.0 (no NaN)
        m = compute_shaping_metrics(
            np.array([[0.0, 0.0], [0.0, 5.0]]),
            np.array([[True, True], [True, False]]),
        )
        assert m["uniformity_cv"] == 0.0
        assert m["encircled_energy"] == 0.0
        assert m["peak"] == 5.0
        assert m["in_mask_mean"] == 0.0

    def test_mask_larger_than_intensity_clipped_centered(self):
        # 4x4 mask clipped to the 2x2 intensity bounds -> full mask
        m = compute_shaping_metrics(np.ones((2, 2)), np.ones((4, 4), dtype=bool))
        assert m["uniformity_cv"] == 0.0
        assert m["encircled_energy"] == 1.0
        assert m["in_mask_mean"] == 1.0

    def test_mask_smaller_than_intensity_padded_centered(self):
        # 2x2 mask padded to the center of the 4x4 intensity
        m = compute_shaping_metrics(np.ones((4, 4)), np.ones((2, 2), dtype=bool))
        assert m["uniformity_cv"] == 0.0
        assert m["encircled_energy"] == pytest.approx(0.25)
        assert m["in_mask_mean"] == 1.0

    def test_non_2d_input_raises(self):
        with pytest.raises(ValueError, match="2D"):
            compute_shaping_metrics(np.ones(4), np.ones((2, 2), dtype=bool))


class TestComputeSquareMetrics:
    def test_uniform_square(self):
        m = compute_square_metrics(np.ones((10, 10)), target_side=4, center=(5, 5))
        assert m["aspect_ratio"] == 1.0
        assert m["squareness"] == 0.0
        assert m["uniformity_cv"] == 0.0
        assert m["encircled_energy"] == pytest.approx(0.16)  # 4x4 region / 100
        assert m["flatness_factor"] == 1.0
        assert m["intensity_max"] == 1.0
        assert m["intensity_mean"] == 1.0

    def test_rectangular_bright_region_aspect_ratio(self):
        img = np.zeros((10, 10))
        img[4:6, 2:6] = 1.0  # 2 rows x 4 cols bright block
        m = compute_square_metrics(img, target_side=4, center=(5, 5))
        assert m["aspect_ratio"] == 2.0
        assert m["squareness"] == 1.0
        # region rows 3..6, cols 3..6 holds six of the eight bright pixels
        assert m["encircled_energy"] == pytest.approx(0.75)
        assert m["flatness_factor"] == pytest.approx(0.375)
        assert m["intensity_mean"] == pytest.approx(0.08)

    def test_zero_total_safe_branch(self):
        m = compute_square_metrics(np.zeros((5, 5)), target_side=4, center=(2, 2))
        assert m == {
            "aspect_ratio": 1.0,
            "squareness": 0.0,
            "uniformity_cv": 0.0,
            "encircled_energy": 0.0,
            "flatness_factor": 0.0,
            "intensity_max": 0.0,
            "intensity_mean": 0.0,
        }

    def test_hand_computed_region_metrics(self):
        img = np.arange(16, dtype=float).reshape(4, 4)
        m = compute_square_metrics(img, target_side=2, center=(2, 2))
        # bright = >= 7.5 -> rows 2..3 all cols -> 4x2 -> aspect 2.0
        assert m["aspect_ratio"] == 2.0
        assert m["squareness"] == 1.0
        # region rows 1..2, cols 1..2 = [[5,6],[9,10]]
        assert m["uniformity_cv"] == pytest.approx(np.sqrt(4.25) / 7.5)
        assert m["encircled_energy"] == pytest.approx(30 / 120)
        assert m["flatness_factor"] == pytest.approx(7.5 / 15)
        assert m["intensity_max"] == 15.0
        assert m["intensity_mean"] == 7.5

    def test_region_clipped_at_edge(self):
        m = compute_square_metrics(np.ones((10, 10)), target_side=10, center=(0, 0))
        # half=5, region rows 0..4, cols 0..4 -> 25 pixels
        assert m["encircled_energy"] == pytest.approx(0.25)
        assert m["uniformity_cv"] == 0.0
        assert m["flatness_factor"] == 1.0

    def test_center_rounding(self):
        m = compute_square_metrics(np.ones((10, 10)), target_side=4, center=(5.4, 5.6))
        # cx=5, cy=6 -> region rows 4..7, cols 3..6 (4x4)
        assert m["encircled_energy"] == pytest.approx(0.16)

    def test_empty_region_guard(self):
        # total > 0 but the region around the center is all zeros
        img = np.zeros((10, 10))
        img[0, 0] = 1.0
        m = compute_square_metrics(img, target_side=4, center=(5, 5))
        assert m["uniformity_cv"] == 0.0  # region_mean == 0 -> guard
        assert m["encircled_energy"] == 0.0
        assert m["flatness_factor"] == 0.0
        assert m["aspect_ratio"] == 1.0  # single bright pixel -> 1x1


class TestComputeQualityScore:
    def test_perfect_metrics_score_one(self):
        score = compute_quality_score(
            {"aspect_ratio": 1.0, "uniformity_cv": 0.0, "encircled_energy": 1.0}
        )
        assert score == pytest.approx(1.0)

    def test_hand_computed_weights_and_kernels(self):
        # f_ar = exp(-((1.3-1)/0.3)^2) = exp(-1); f_uni = exp(-(0.3/0.3)^2) = exp(-1)
        score = compute_quality_score(
            {"aspect_ratio": 1.3, "uniformity_cv": 0.3, "encircled_energy": 0.5}
        )
        expected = 0.3 * math.exp(-1) + 0.4 * math.exp(-1) + 0.3 * 0.5
        assert score == pytest.approx(expected)

    def test_encircled_energy_clipped_above_one(self):
        score = compute_quality_score(
            {"aspect_ratio": 1.0, "uniformity_cv": 0.0, "encircled_energy": 2.0}
        )
        assert score == pytest.approx(1.0)

    def test_encircled_energy_clipped_below_zero(self):
        score = compute_quality_score(
            {"aspect_ratio": 1.0, "uniformity_cv": 0.0, "encircled_energy": -0.5}
        )
        assert score == pytest.approx(0.7)  # 0.3*1 + 0.4*1 + 0.3*0


class TestMeasureBrightSpan:
    def test_bounding_box_of_bright_region(self):
        img = np.zeros((5, 5))
        img[1:3, 2:4] = 1.0
        assert measure_bright_span(img) == (2, 2)

    def test_single_pixel(self):
        img = np.array([[0.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
        assert measure_bright_span(img) == (1, 1)

    def test_zero_intensity_returns_zero(self):
        assert measure_bright_span(np.zeros((3, 3))) == (0, 0)

    def test_peak_frac_above_one_returns_zero(self):
        assert measure_bright_span(np.ones((3, 3)), peak_frac=2.0) == (0, 0)

    def test_custom_peak_frac(self):
        img = np.array([[0.0, 0.0], [0.0, 4.0]])
        # threshold 0.25*4=1 -> only the 4-pixel is bright
        assert measure_bright_span(img, peak_frac=0.25) == (1, 1)


class TestClampSide:
    def test_within_limit_unchanged(self):
        assert clamp_side(10, 100, 100) == 10

    def test_exceeding_limit_clamped_to_max_side(self):
        assert clamp_side(100, 100, 100) == 84  # 100 - 2*8

    def test_boundary_exact_max_side(self):
        assert clamp_side(84, 100, 100) == 84

    def test_boundary_just_over(self):
        assert clamp_side(85, 100, 100) == 84

    def test_negative_max_side_guarded_to_one(self):
        # min(10,10) - 16 = -6 -> max(1, -6) = 1
        assert clamp_side(5, 10, 10) == 1

    def test_zero_side_guarded_to_one(self):
        assert clamp_side(0, 100, 100) == 1

    def test_small_height_dominates(self):
        assert clamp_side(10, 10, 100) == 1  # min(10,100) - 16 = -6

    def test_custom_margin(self):
        assert clamp_side(10, 100, 100, margin=20) == 10  # max_side = 60
        assert clamp_side(70, 100, 100, margin=20) == 60