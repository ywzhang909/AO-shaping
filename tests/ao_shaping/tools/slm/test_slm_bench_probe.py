"""Offline tests for the shared SLM bench-probe measurement core.

**No hardware.** Every case is pure numpy on synthetic frames, which is the point:
the three pitfalls these functions exist to prevent (raw ``argmax`` on a dim
frame, a stale "flat" reference, a fixed memory slot) are all *measurement* bugs
and have to be locked down without a camera.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.tools.slm.bench_kernels import (
    BEAM_CENTER_PANEL,
    SLM_PANEL_H,
    SLM_PANEL_W,
    core_fraction,
    despike_frame,
    estimate_shift,
    fit_linear_slope,
    measure_spot,
    ramp_panel,
    smooth_frame,
    tilt_shift_px,
    zernike_panel,
)


def _frame_with_spot(
    cx: float, cy: float, sigma: float = 3.0, peak: float = 100.0, shape=(200, 300)
) -> np.ndarray:
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return peak * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)))


class TestSmoothFrame:
    def test_preserves_shape_and_constant_level(self) -> None:
        img = np.full((20, 30), 7.0)
        assert smooth_frame(img, 5).shape == img.shape
        assert float(smooth_frame(img, 5).mean()) == pytest.approx(7.0)

    def test_a_box_blur_alone_cannot_outrun_a_hot_pixel(self) -> None:
        """Documents *why* despiking is a separate, earlier stage.

        A 5000-count defect spread over a 5x5 box still reads 200, which beats a
        dim spot peaking at 20 -- so smoothing alone does not make argmax safe.
        """
        img = _frame_with_spot(150.0, 100.0, peak=20.0)
        img[5, 5] = 5000.0
        # Whichever window the hot pixel lands in, it wins -- and it is nowhere
        # near the real spot.
        assert np.unravel_index(int(np.argmax(img)), img.shape) == (5, 5)
        blurred_arg = np.unravel_index(int(np.argmax(smooth_frame(img, 5))), img.shape)
        assert blurred_arg != (100, 150)
        # Despike first, and the spot wins.
        assert np.unravel_index(
            int(np.argmax(smooth_frame(despike_frame(img, 3), 5))), img.shape
        ) == (100, 150)

    def test_k_below_two_is_a_no_op(self) -> None:
        img = _frame_with_spot(10.0, 10.0)
        assert np.array_equal(smooth_frame(img, 1), img)


class TestDespikeFrame:
    def test_kills_an_isolated_hot_pixel(self) -> None:
        img = _frame_with_spot(150.0, 100.0, peak=20.0)
        img[5, 5] = 5000.0
        out = despike_frame(img, 3)
        assert out[5, 5] < 100.0

    def test_kills_an_isolated_cold_pixel(self) -> None:
        """The pixel is replaced by the local median, which is background here."""
        img = _frame_with_spot(150.0, 100.0, peak=100.0)
        img[5, 5] = -50.0
        assert despike_frame(img, 3)[5, 5] >= 0.0

    def test_leaves_a_real_spot_alone(self) -> None:
        """A 3x3 median at a Gaussian peak samples slightly lower neighbours, so
        the peak erodes by a few percent -- acceptable, since the centroid is what
        matters and a median preserves it exactly."""
        img = _frame_with_spot(150.0, 100.0, peak=100.0)
        out = despike_frame(img, 3)
        assert out[100, 150] == pytest.approx(100.0, rel=0.1)
        assert measure_spot(img).centroid_x == pytest.approx(150.0, abs=0.5)

    def test_k_below_three_is_a_no_op(self) -> None:
        img = _frame_with_spot(10.0, 10.0)
        assert np.array_equal(despike_frame(img, 1), img)


class TestMeasureSpot:
    def test_recovers_a_symmetric_spot(self) -> None:
        m = measure_spot(_frame_with_spot(150.0, 100.0, sigma=3.0, peak=100.0))
        assert m.centroid_x == pytest.approx(150.0, abs=0.5)
        assert m.centroid_y == pytest.approx(100.0, abs=0.5)
        assert m.peak == pytest.approx(100.0, rel=0.02)
        assert m.fwhm_px > 0.0

    def test_finds_the_spot_not_a_noise_spike(self) -> None:
        """A dim frame with one hot pixel must still report the real spot."""
        img = _frame_with_spot(150.0, 100.0, sigma=3.0, peak=22.0)
        img[3, 3] = 400.0
        m = measure_spot(img)
        assert m.centroid_x == pytest.approx(150.0, abs=1.0)
        assert m.centroid_y == pytest.approx(100.0, abs=1.0)

    def test_hollowness_is_near_one_for_a_single_lobe(self) -> None:
        m = measure_spot(_frame_with_spot(150.0, 100.0, sigma=4.0, peak=80.0))
        assert m.hollowness > 0.8

    def test_hollowness_collapses_for_a_ring(self) -> None:
        """A ring puts the centroid in a dark gap -- width cannot see this."""
        yy, xx = np.mgrid[0 : 200, 0 : 300]
        r2 = (xx - 150.0) ** 2 + (yy - 100.0) ** 2
        ring = 100.0 * np.exp(-((np.sqrt(r2) - 30.0) ** 2) / (2 * 3.0**2))
        assert measure_spot(ring).hollowness < 0.6

    def test_empty_frame_does_not_raise(self) -> None:
        m = measure_spot(np.zeros((50, 60)))
        assert m.peak == 0.0
        assert np.isfinite(m.centroid_x)


class TestEstimateShift:
    @staticmethod
    def _pattern(shape=(200, 300), cx=150.0, cy=100.0, sigma=6.0) -> np.ndarray:
        yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
        return np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)))

    def test_zero_shift(self) -> None:
        a = self._pattern()
        dy, dx, peak = estimate_shift(a, a)
        assert abs(dy) < 0.1 and abs(dx) < 0.1
        assert peak > 0.9

    @pytest.mark.parametrize("shift_x,shift_y", [(5, 0), (0, 4), (-3, 7), (12, -9)])
    def test_recovers_a_known_shift(self, shift_x: int, shift_y: int) -> None:
        ref = self._pattern()
        moving = np.roll(np.roll(ref, shift_y, axis=0), shift_x, axis=1)
        dy, dx, peak = estimate_shift(ref, moving)
        assert dx == pytest.approx(float(shift_x), abs=0.3)
        assert dy == pytest.approx(float(shift_y), abs=0.3)
        assert peak > 0.9

    def test_survives_an_asymmetric_multi_lobed_spot(self) -> None:
        """The reason this exists rather than using the centroid.

        A tilt large enough to give signal deforms the spot into unequal lobes.
        The brightest lobe -- and therefore the window a centroid is taken over --
        can change, which is what made hardware centroid readings jump several px
        between repeats of the *same* phase. Phase correlation is indifferent to
        that: it only needs the pattern to move.
        """
        yy, xx = np.mgrid[0 : 200, 0 : 300]
        # Deliberately unequal lobes at unequal separations: the centroid sits far
        # from the brightest lobe, so any peak-locked measure is fragile.
        split = (
            1.0 * np.exp(-(((xx - 128.0) ** 2 + (yy - 100.0) ** 2) / 26.0))
            + 0.45 * np.exp(-(((xx - 176.0) ** 2 + (yy - 100.0) ** 2) / 10.0))
            + 0.30 * np.exp(-(((xx - 150.0) ** 2 + (yy - 82.0) ** 2) / 14.0))
        )
        # Genuinely asymmetric: the intensity centroid is nowhere near mid-span.
        m0 = measure_spot(split)
        assert abs(m0.centroid_x - 150.0) > 2.0, "pattern must not be symmetric"

        moved = np.roll(np.roll(split, 5, axis=0), -6, axis=1)
        dy, dx, peak = estimate_shift(split, moved)
        assert dx == pytest.approx(-6.0, abs=0.3)
        assert dy == pytest.approx(5.0, abs=0.3)
        assert peak > 0.9

    def test_rejects_a_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="shape mismatch"):
            estimate_shift(np.zeros((20, 20)), np.zeros((21, 20)))

    def test_max_shift_rejects_a_wrap(self) -> None:
        ref = self._pattern(shape=(60, 60), cx=30.0, cy=30.0, sigma=3.0)
        moving = np.roll(ref, 25, axis=1)
        dy, dx, _ = estimate_shift(ref, moving, max_shift=10)
        assert np.isnan(dx)
class TestCoreFraction:
    def test_flat_field_concentrates_energy_at_the_centre(self) -> None:
        img = _frame_with_spot(150.0, 100.0, sigma=2.0, peak=100.0, shape=(200, 300))
        assert core_fraction(img, 150.0, 100.0, 40.0) > 0.8

    def test_scattered_field_has_a_lower_core_fraction(self) -> None:
        img = _frame_with_spot(150.0, 100.0, sigma=2.0, peak=100.0, shape=(200, 300))
        wide = _frame_with_spot(150.0, 100.0, sigma=25.0, peak=100.0, shape=(200, 300))
        assert core_fraction(wide, 150.0, 100.0, 40.0) < core_fraction(
            img, 150.0, 100.0, 40.0
        )

    def test_zero_energy_is_zero(self) -> None:
        assert core_fraction(np.zeros((10, 10)), 5, 5, 3.0) == 0.0


class TestZernikePanel:
    def test_shape_and_far_field_is_zero(self) -> None:
        p = zernike_panel({(2, 0): 1.0}, 450, BEAM_CENTER_PANEL)
        assert p.shape == (SLM_PANEL_H, SLM_PANEL_W)
        assert float(p[0, 0]) == 0.0
        assert float(p[-1, -1]) == 0.0

    def test_lands_on_the_pupil_not_the_panel_centre(self) -> None:
        """The pupil is the panel centre here, so move it and check it follows."""
        pupil = (1000, 480)
        p = zernike_panel({(2, 0): 1.0}, 200, pupil)
        # Non-zero within the disc at the pupil...
        assert np.abs(p[pupil[1], pupil[0]]) > 0.0
        # ...and the panel centre is far outside a 200 px disc at (1000, 480).
        assert np.abs(p[SLM_PANEL_H // 2, 0]).sum() == 0.0

    def test_defocus_scales_linearly_with_amplitude(self) -> None:
        one = zernike_panel({(2, 0): 1.0}, 300, BEAM_CENTER_PANEL)
        two = zernike_panel({(2, 0): 2.0}, 300, BEAM_CENTER_PANEL)
        inner = (slice(400, 800), slice(700, 1100))
        assert np.allclose(two[inner], 2.0 * one[inner])

    def test_empty_coefficients_give_a_flat_panel(self) -> None:
        assert not zernike_panel({}, 300, BEAM_CENTER_PANEL).any()

    def test_clips_at_the_panel_edge_instead_of_wrapping(self) -> None:
        p = zernike_panel({(2, 0): 1.0}, 300, (5, 5))
        assert p.shape == (SLM_PANEL_H, SLM_PANEL_W)
        assert np.isfinite(p).all()


class TestRampPanel:
    def test_advances_two_pi_every_period(self) -> None:
        """A sawtooth: 0 at the period boundary, approaching 2*pi just before."""
        period = 40
        p = ramp_panel(period, axis=1)
        assert p.shape == (SLM_PANEL_H, SLM_PANEL_W)
        assert p[0, 0] == pytest.approx(0.0)
        assert p[0, period - 1] == pytest.approx(2 * np.pi * (period - 1) / period)
        assert p[0, period] == pytest.approx(0.0)  # wraps
        # Monotone within one period.
        assert np.all(np.diff(p[0, :period]) > 0)

    def test_axis_selects_the_swept_direction(self) -> None:
        assert not np.allclose(ramp_panel(40, axis=0), ramp_panel(40, axis=1))

    def test_larger_period_means_a_smaller_shift(self) -> None:
        assert tilt_shift_px(480) < tilt_shift_px(120)
        # Measured 2026-09-30: displacements from flat were 62.1, 30.3, 14.7,
        # 6.9, 3.4 px for periods 120, 240, 480, 960, 1920 -- cleanly 1/P.
        assert tilt_shift_px(120) == pytest.approx(62.0, abs=6.0)
        assert tilt_shift_px(1920) == pytest.approx(3.4, abs=0.6)


class TestFitLinearSlope:
    def test_recovers_an_exact_line(self) -> None:
        slope, intercept = fit_linear_slope(
            np.array([1.0, 2.0, 3.0]), np.array([3.0, 5.0, 7.0])
        )
        assert slope == pytest.approx(2.0)
        assert intercept == pytest.approx(1.0)

    def test_degenerate_input_is_nan_not_an_exception(self) -> None:
        slope, intercept = fit_linear_slope(np.array([1.0]), np.array([2.0]))
        assert np.isnan(slope) and np.isnan(intercept)
        slope, _ = fit_linear_slope(np.array([2.0, 2.0]), np.array([1.0, 3.0]))
        assert np.isnan(slope)
