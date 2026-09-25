"""Regression-anchor tests for :mod:`ao_shaping.utils.beam_metrics`.

Pins the exact current behavior of the pure-NumPy beam-shaping metrics
(compute_metrics, compute_shaping_metrics, compute_square_metrics,
compute_quality_score, measure_bright_span, clamp_side,
measure_spot_diameter_cam, intensity_to_amplitude, normalize_pattern,
zero_order_center, median_zero_order_center).

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
    median_zero_order_center,
    normalize_pattern,
    smart_zero_order_center,
    zero_order_center,
)


class TestIntensityToAmplitude:
    def test_sqrt_and_normalize_to_peak(self):
        # sqrt([0,1,4,9]) = [0,1,2,3]; peak=3 -> [0, 1/3, 2/3, 1]
        amp = intensity_to_amplitude(np.array([[0.0, 1.0], [4.0, 9.0]]))
        np.testing.assert_allclose(amp, [[0.0, 1 / 3], [2 / 3, 1.0]], rtol=1e-6)
        assert amp.dtype == np.float32

    def test_no_normalize_keeps_raw_sqrt(self):
        amp = intensity_to_amplitude(
            np.array([[0.0, 1.0], [4.0, 9.0]]), normalize=False
        )
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
    def _uniform_disk(
        radius: float, cy: float, cx: float, size: int = 41
    ) -> np.ndarray:
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
        m = compute_metrics(
            np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([[0.0, 1.0], [0.0, 0.0]])
        )
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
        m = compute_shaping_metrics(
            np.array([[1.0, 2.0], [3.0, 4.0]]), np.ones((2, 2), dtype=bool)
        )
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


class TestZeroOrderCenter:
    def test_single_peak_returns_argmax(self):
        # A lone bright pixel: the windowed centroid collapses onto the anchor.
        frame = np.zeros((40, 40))
        frame[20, 25] = 100.0
        assert zero_order_center(frame) == (25, 20)

    def test_refine_false_returns_exact_argmax(self):
        # No refinement -> the global argmax wins even with a secondary lobe.
        frame = np.zeros((40, 40))
        frame[20, 25] = 100.0
        frame[20, 28] = 80.0
        assert zero_order_center(frame, refine=False) == (25, 20)

    def test_all_dark_returns_frame_center(self):
        frame = np.zeros((40, 80))
        assert zero_order_center(frame) == (40, 20)  # (w//2, h//2)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError, match="2D"):
            zero_order_center(np.zeros(10))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="2D"):
            zero_order_center(np.zeros((0, 0)))

    def test_refinement_shifts_toward_mass(self):
        # Asymmetric blob: peak at (20,20), secondary lobe at (20,24). The
        # percentile-20-clipped windowed centroid lands between them.
        frame = np.zeros((40, 40))
        frame[20, 20] = 100.0
        frame[20, 24] = 80.0
        x, y = zero_order_center(frame)
        assert (x, y) == (22, 20)  # (100*20 + 80*24) / 180 = 21.78 -> 22
        assert x > 20  # shifted right of the argmax anchor

    def test_half_win_limits_refinement_window(self):
        # half_win=1 -> 3x3 window around the anchor excludes the lobe, so the
        # centroid stays pinned on the argmax.
        frame = np.zeros((40, 40))
        frame[20, 20] = 100.0
        frame[20, 24] = 80.0
        assert zero_order_center(frame, half_win=1) == (20, 20)

    def test_returns_int_tuple(self):
        frame = np.zeros((40, 40))
        frame[20, 25] = 100.0
        center = zero_order_center(frame)
        assert isinstance(center, tuple)
        assert all(isinstance(v, int) for v in center)


class TestMedianZeroOrderCenter:
    @staticmethod
    def _frame_with_peak(cx: int, cy: int, size: int = 30) -> np.ndarray:
        frame = np.zeros((size, size))
        frame[cy, cx] = 100.0
        return frame

    def test_median_across_frames(self):
        frames = [self._frame_with_peak(cx, 15) for cx in (10, 12, 14)]
        assert median_zero_order_center(frames) == (12, 15)

    def test_median_even_count_exact_mid(self):
        frames = [self._frame_with_peak(cx, 15) for cx in (10, 12)]
        assert median_zero_order_center(frames) == (11, 15)

    def test_median_rounds_mid_value(self):
        # np.median([10, 11]) = 10.5 -> int(round(10.5)) = 10 (banker's).
        frames = [self._frame_with_peak(cx, 15) for cx in (10, 11)]
        assert median_zero_order_center(frames) == (10, 15)

    def test_empty_sequence_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            median_zero_order_center([])

    def test_robust_to_outlier_hot_pixel_frame(self):
        # Three clean frames at (25,20) plus one hot-pixel frame at (0,0):
        # the per-axis median ignores the outlier.
        frames = [self._frame_with_peak(25, 20, size=40) for _ in range(3)]
        hot = np.zeros((40, 40))
        hot[0, 0] = 1000.0
        frames.append(hot)
        assert median_zero_order_center(frames) == (25, 20)

    def test_returns_int_tuple(self):
        center = median_zero_order_center([self._frame_with_peak(10, 15)])
        assert isinstance(center, tuple)
        assert all(isinstance(v, int) for v in center)


class TestSmartZeroOrderCenter:
    @staticmethod
    def _spot_frame(size=64, center=(40, 20), peak=200.0, sigma=3.0) -> np.ndarray:
        yy, xx = np.mgrid[0:size, 0:size]
        img = peak * np.exp(
            -(((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / (2 * sigma**2))
        )
        return img.astype(np.uint8)

    def test_flat_core_uses_wider_window(self):
        """A Gaussian spot has a flat (non-hollow) core: smart_zero_order_center
        should widen the centroid window and still land on the true centre."""
        img = self._spot_frame(center=(30, 25), peak=200.0)
        cx, cy = smart_zero_order_center(img)
        assert abs(cx - 30) <= 3
        assert abs(cy - 25) <= 3

    def test_hollow_core_uses_default_window(self):
        """A donut (hollow core) has a low centre: the flat-core check must NOT
        widen the window, otherwise the centroid drifts to the ring.

        For a hollow-core donut the argmax sits on the ring edge (radius ~12 from
        the geometric centre), and the small-window centroid stays near that ring
        edge rather than drifting toward the ring centroid = (40, 40). The key
        assertion is that the wide-window centroid was NOT used (which would pull
        the result close to (40,40)).
        """
        size = 80
        yy, xx = np.mgrid[0:size, 0:size]
        r = np.sqrt((xx - 40) ** 2 + (yy - 40) ** 2)
        img = np.where((r > 10) & (r < 15), 200.0, 0.0).astype(np.uint8)
        cx, cy = smart_zero_order_center(img)
        dist = np.hypot(cx - 40, cy - 40)
        # Ring is at radius 10–15 from centre → result stays near the ring edge,
        # NOT pulled to the geometric centre (which would mean dist ≈ 0).
        assert 9 <= dist <= 24

    def test_all_dark_returns_frame_centre(self):
        assert smart_zero_order_center(np.zeros((50, 80), dtype=np.uint8)) == (40, 25)

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError, match="2D"):
            smart_zero_order_center(np.zeros((4, 4, 3), dtype=np.uint8))

    def test_returns_int_tuple(self):
        center = smart_zero_order_center(self._spot_frame())
        assert isinstance(center, tuple)
        assert all(isinstance(v, int) for v in center)


class TestClampCenterToFrame:
    def test_near_edge_is_clamped_so_the_roi_fits(self):
        from ao_shaping.utils.image.beam_metrics import clamp_center_to_frame as ccf

        x, y = ccf((5, 5), (200, 200), 100)
        assert (x, y) == (50, 50)
        assert 0 <= x < 200 and 0 <= y < 200

    def test_interior_center_is_unchanged(self):
        from ao_shaping.utils.image.beam_metrics import clamp_center_to_frame as ccf

        assert ccf((125, 125), (512, 512), 200) == (125, 125)

    def test_window_larger_than_frame_stays_in_frame(self):
        from ao_shaping.utils.image.beam_metrics import clamp_center_to_frame as ccf

        x, y = ccf((5, 5), (120, 90), 250)
        assert 0 <= x < 90 and 0 <= y < 120
