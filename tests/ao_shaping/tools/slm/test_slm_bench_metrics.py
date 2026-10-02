"""Offline tests for the pure bench-meetrics kernel.

No hardware, no device objects, no I/O: everything here is numpy on synthetic
frames, so the whole file runs in CI.

The load-bearing class is :class:`TestPrepDivergence`. The bench has **two**
frame preparations that look interchangeable and are not, and a reader who
"cleans them up" to match each other silently breaks an objective. These tests
pin both against the *live* optimizer functions, not against a copy.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.tools.slm.slm_bench_metrics import (
    build_block_pattern,
    camera_pixel_um_from_focal_scale,
    crop_roi,
    exposure_monotonicity,
    finite_clip,
    finite_median_subtract,
    flat_to_flat_floor,
    roi_l2,
    settle_time_s,
    snr_vs_averages,
)


def _pedestal_frame(pedestal: float = 8.0, blob: float = 100.0) -> np.ndarray:
    """A blob on a uniform (non-zero) background, as a real CCD frame has."""
    yy, xx = np.mgrid[0:80, 0:80]
    r2 = (xx - 40.0) ** 2 + (yy - 40.0) ** 2
    return pedestal + blob * np.exp(-r2 / (2 * 3.0**2))


class TestPrepDivergence:
    """The two preps are deliberately different. Do not merge them.

    ``finite_clip`` mirrors ``slm_square_shaping.square_peak_to_background_ratio``
    (clip only) and ``finite_median_subtract`` mirrors
    ``slm_gs_refine._prepare_frame`` (median-subtract then clip). Both are
    correct for their own metric, and swapping them breaks one.
    """

    def test_clip_preserves_the_pedestal(self):
        assert finite_clip(_pedestal_frame(8.0)).mean() == pytest.approx(8.0, abs=1.0)

    def test_median_subtract_removes_the_pedestal(self):
        """The *background* becomes zero; the blob necessarily survives.

        Subtracting the median cannot remove signal, only the offset. The
        background level is what the prep exists to null.
        """
        frame = _pedestal_frame(8.0)
        out = finite_median_subtract(frame)
        background = out[0, 0]
        assert background == pytest.approx(0.0, abs=1e-9)
        # Signal is still there -- median subtraction is not a signal remover.
        assert out[40, 40] == pytest.approx(100.0, rel=0.02)

    def test_clip_matches_the_live_pbr_prep(self):
        """Lock against the real optimizer, not a copy of it."""
        from ao_shaping.optimizer.wfless.slm_square_shaping import (
            square_peak_to_background_ratio,
        )

        frame = _pedestal_frame(3.0)
        centre, side = (40, 40), 20

        # Reproduce PBR on the same data and confirm it equals peak/mean(outside)
        # of the clip-only prep -- i.e. `finite_clip` really is that function's prep.
        prepped = finite_clip(frame)
        h, w = frame.shape
        box = prepped[30:50, 30:50]
        mask = np.ones(frame.shape, dtype=bool)
        mask[30:50, 30:50] = False
        expected = float(box.max()) / float(prepped[mask].mean())
        assert square_peak_to_background_ratio(frame, centre, side) == pytest.approx(
            expected, rel=1e-9
        )

    def test_median_subtract_matches_the_live_prepare_frame(self):
        """Lock against ``slm_gs_refine._prepare_frame`` itself."""
        from ao_shaping.optimizer.wfless.slm_gs_refine import _prepare_frame

        frame = _pedestal_frame(5.0)
        assert np.array_equal(finite_median_subtract(frame), _prepare_frame(frame))

    def test_the_two_preps_are_not_interchangeable(self):
        """The load-bearing test.

        Under median subtraction the PBR denominator collapses (the pedestal *is*
        the median), so the ratio explodes. This is why `finite_clip` exists and
        why the two must not be "unified".
        """
        frame = _pedestal_frame(3.0)
        clip = finite_clip(frame)
        median = finite_median_subtract(frame)
        assert not np.array_equal(clip, median)

        mask = np.ones(frame.shape, dtype=bool)
        mask[30:50, 30:50] = False
        box = slice(30, 50)
        pbr_clip = clip[box, box].max() / clip[mask].mean()
        # Denominator is numerically ~0, so guard the division the way the
        # metric does, then show the magnitude is nonsense.
        bg = float(median[mask].mean())
        pbr_median = float(median[box, box].max()) / max(bg, 1e-30)
        assert pbr_clip == pytest.approx(103.0 / 3.0, rel=0.05)
        assert pbr_median > 1e4, "median subtraction must break the PBR ratio"

    def test_both_treat_non_finite_as_zero(self):
        bad = np.array([[1.0, np.nan], [np.inf, 4.0]])
        for fn in (finite_clip, finite_median_subtract):
            out = fn(bad)
            assert np.isfinite(out).all()
        assert finite_clip(bad)[0, 1] == 0.0
        assert finite_clip(bad)[1, 0] == 0.0

    def test_both_reject_non_2d(self):
        for fn in (finite_clip, finite_median_subtract):
            with pytest.raises(ValueError, match="2D"):
                fn(np.zeros((2, 2, 2)))


class TestCropRoi:
    def test_extracts_the_requested_window(self):
        frame = np.arange(80 * 80, dtype=float).reshape(80, 80)
        assert np.array_equal(crop_roi(frame, (40, 40), 5), frame[35:45, 35:45])

    def test_zero_pads_when_the_centre_is_off_frame(self):
        frame = np.ones((10, 10))
        out = crop_roi(frame, (0, 0), 4)
        assert out.shape == (8, 8)
        assert out[0, 0] == 0.0
        assert out[-1, -1] == 1.0

    def test_roi_l2_is_zero_for_identical_frames(self):
        a = np.arange(50.0).reshape(5, 10)
        assert roi_l2(a, a) == 0.0

    def test_roi_l2_matches_the_definition(self):
        a = np.zeros((4, 4))
        b = np.zeros((4, 4))
        b[1, 1] = 3.0
        assert roi_l2(a, b) == pytest.approx(3.0)


class TestFlatToFlatFloor:
    def test_identical_frames_give_zero(self):
        a = np.ones((8, 8))
        assert flat_to_flat_floor([a, a.copy(), a.copy()])[0] == 0.0

    def test_returns_the_median_and_the_full_series(self):
        frames = [np.full((4, 4), float(i)) for i in range(5)]
        median, series = flat_to_flat_floor(frames)
        # Each step changes all 16 pixels by 1.0 -> L2 = sqrt(16) = 4.0.
        assert series == [4.0, 4.0, 4.0, 4.0]
        assert median == pytest.approx(4.0)

    def test_needs_at_least_two_frames(self):
        with pytest.raises(ValueError, match="two"):
            flat_to_flat_floor([np.zeros((4, 4))])


class TestSettleTime:
    def test_detects_the_settling_time(self):
        times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        deltas = [10.0, 8.0, 3.0, 1.0, 0.9, 0.9, 0.9]
        # tol = 10% of the final value 0.9 => 0.09. t=1.5 sits at 1.0, which is
        # 0.1 away, so the first window wholly inside tolerance starts at 2.0.
        assert settle_time_s(deltas, times) == pytest.approx(2.0)

    def test_returns_none_when_it_never_settles(self):
        times = [0.0, 1.0, 2.0, 3.0]
        assert settle_time_s([10.0, 7.0, 4.0, 1.0], times) is None

    def test_run_length_is_respected(self):
        times = [0.0, 1.0, 2.0, 3.0, 4.0]
        # Only the final TWO samples sit at the settled floor (1.0 +/- 0.1).
        deltas = [9.0, 9.0, 2.0, 1.0, 1.0]
        assert settle_time_s(deltas, times, run=2) == pytest.approx(3.0)
        assert settle_time_s(deltas, times, run=3) is None

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            settle_time_s([1.0, 2.0], [0.0])


class TestSnrVsAverages:
    def test_matches_the_sqrt_k_law(self):
        floor = 4.0
        signal = [10.0, 10.0, 10.0, 10.0]
        out = snr_vs_averages(signal, floor, [1, 2, 4])
        assert out[1] == pytest.approx(10.0 / 4.0)
        assert out[2] == pytest.approx(10.0 / (4.0 / np.sqrt(2)))
        assert out[4] == pytest.approx(10.0 / (4.0 / 2.0))

    def test_averaging_helps_when_noise_is_random(self):
        """K=9 must beat K=1, otherwise the bench floor is drift-dominated."""
        rng = np.random.default_rng(0)
        noisy = list(10.0 + rng.normal(0.0, 4.0, size=60))
        out = snr_vs_averages(noisy, 4.0, [1, 9, 25])
        assert out[9] > out[1]
        assert out[25] > out[9]

    def test_rejects_a_non_positive_floor(self):
        with pytest.raises(ValueError, match="floor"):
            snr_vs_averages([1.0], 0.0, [1])


class TestExposureMonotonicity:
    def test_accepts_a_clean_ramp(self):
        out = exposure_monotonicity([0.4, 0.8, 1.2], [40.0, 80.0, 120.0])
        assert out["verdict"] == "monotonic"
        assert len(out["ratios"]) == 2

    def test_flags_the_historical_non_monotonic_ladder(self):
        """The 2026-10-01 bracket: peak collapsed at 0.8 ms -> below 0.4 ms."""
        out = exposure_monotonicity([0.4, 0.6, 0.8, 1.0], [194.0, 193.0, 80.0, 117.0])
        assert out["verdict"] == "non_monotonic"

    def test_flags_saturation(self):
        out = exposure_monotonicity([0.4, 1.5], [40.0, 242.0], saturation_level=250)
        assert out["saturated"] is False
        out = exposure_monotonicity([0.4, 2.0], [40.0, 255.0], saturation_level=250)
        assert out["saturated"] is True

    def test_requires_matching_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            exposure_monotonicity([0.4, 0.8], [40.0])


class TestBuildBlockPattern:
    def test_tiles_a_coefficient_grid_onto_the_panel(self):
        grid = np.arange(24 * 24, dtype=float).reshape(24, 24)
        phase = build_block_pattern(grid.ravel(), 24, (120, 192))
        assert phase.shape == (120, 192)
        # 1200x1920 panel / 24 grid -> 5x8 px blocks, so block (row, col) starts
        # at (row*5, col*8) and the coefficient repeats across its whole block.
        assert phase[0, 0] == pytest.approx(grid[0, 0])
        assert phase[0, 8] == pytest.approx(grid[0, 1])
        assert phase[5, 0] == pytest.approx(grid[1, 0])
        assert phase[4, 7] == pytest.approx(grid[0, 0]), "inside the same block"
        assert phase[5, 8] == pytest.approx(grid[1, 1])

    def test_block_size_is_derived_from_the_grid_and_panel(self):
        grid = np.zeros(24 * 24)
        phase = build_block_pattern(grid, 24, (120, 192))
        # Every block identical when all coefficients are zero.
        assert np.all(phase == 0.0)

    def test_rejects_a_wrong_length_coefficient_vector(self):
        with pytest.raises(ValueError, match="grid"):
            build_block_pattern(np.zeros(10), 24, (120, 192))

    def test_rejects_a_non_positive_grid(self):
        with pytest.raises(ValueError, match="grid"):
            build_block_pattern(np.zeros(4), 0, (120, 192))


class TestCameraPixelFromFocalScale:
    """Derive the CCD pixel pitch from the bench's own measured focus scale.

    The camera pixel pitch used to be hardcoded to 3.31 um, which is a
    *consequence* of a guessed geometry rather than a measurement. On this bench
    the measured tilt scale (TILT_SHIFT_SCALE = 7400, i.e. a 2*pi ramp over P
    panel px moves the spot TILT_SHIFT_SCALE/P camera px) pins it to 2.25 um.

    This inverts the same relation the fitters already use,
    ``shift_px = focal_scale / period`` with
    ``focal_scale = wavelength * f / (d_slm * p_cam)``, so the two constants can
    never drift apart again.
    """

    def test_measured_scale_recovers_the_measured_pitch(self):
        # 1064 nm, f = 125 mm, d_slm = 8 um, K = 7400 -> 2.247 um.
        got = camera_pixel_um_from_focal_scale(
            wavelength_nm=1064.0, focal_length_m=0.125,
            slm_pixel_um=8.0, focal_scale_px=7400.0,
        )
        assert got == pytest.approx(2.247, rel=1e-3)

    def test_round_trips_against_the_forward_relation(self):
        """The whole point: deriving K and inverting it must be consistent."""
        for p_um in (1.4, 2.2, 3.31, 5.0):
            k = (
                1064e-9 * 0.125 / ((8e-6) * (p_um * 1e-6))
            )  # forward: K = lam*f/(d_slm*p_cam)
            back = camera_pixel_um_from_focal_scale(
                wavelength_nm=1064.0, focal_length_m=0.125,
                slm_pixel_um=8.0, focal_scale_px=k,
            )
            assert back == pytest.approx(p_um, rel=1e-6)

    def test_stale_331_value_is_rejected_as_inconsistent(self):
        """Guards the regression that motivated this: 3.31 um implies K=5023,
        which is 33% away from the measured 7400-7600."""
        k_from_stale = 1064e-9 * 0.125 / (8e-6 * 3.31e-6)
        assert k_from_stale == pytest.approx(5023.0, rel=0.01)
        derived = camera_pixel_um_from_focal_scale(
            wavelength_nm=1064.0, focal_length_m=0.125,
            slm_pixel_um=8.0, focal_scale_px=7400.0,
        )
        assert derived < 2.4, "must not reproduce the stale 3.31 um"

    def test_rejects_degenerate_inputs(self):
        for bad in (
            dict(wavelength_nm=0.0, focal_length_m=0.125, slm_pixel_um=8.0, focal_scale_px=7400.0),
            dict(wavelength_nm=1064.0, focal_length_m=0.0, slm_pixel_um=8.0, focal_scale_px=7400.0),
            dict(wavelength_nm=1064.0, focal_length_m=0.125, slm_pixel_um=0.0, focal_scale_px=7400.0),
            dict(wavelength_nm=1064.0, focal_length_m=0.125, slm_pixel_um=8.0, focal_scale_px=0.0),
        ):
            with pytest.raises(ValueError):
                camera_pixel_um_from_focal_scale(**bad)