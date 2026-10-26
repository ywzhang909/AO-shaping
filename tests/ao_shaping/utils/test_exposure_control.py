"""Quantitative CCD exposure must be bounded and scale consistently."""

import numpy as np
import pytest

from ao_shaping.utils.image import exposure_control
from ao_shaping.utils.image.exposure_control import (
    DAHENG_EXPOSURE_LIMIT_ENV,
    ExposureCeilingEstimate,
    choose_exposure,
    configured_daheng_exposure_limit,
    estimate_exposure_ceiling_ms,
    exposure_normalized_frame,
    probe_exposure_ceiling_ms,
    resolve_daheng_exposure_limit,
)


def test_exposure_scaling_is_invariant_for_the_same_light() -> None:
    short = np.array([[10, 20], [40, 80]], dtype=np.uint16)
    long = short * 4
    np.testing.assert_allclose(
        exposure_normalized_frame(short, 0.05),
        exposure_normalized_frame(long, 0.2),
    )


def test_auto_exposure_never_crosses_the_configured_ceiling() -> None:
    dim = np.full((4, 4), 100, dtype=np.uint16)
    decision = choose_exposure(dim, 0.02, max_ms=0.4)
    assert decision.next_ms == 0.4
    assert not decision.accepted


def test_saturated_frame_is_rejected() -> None:
    saturated = np.full((4, 4), 65535, dtype=np.uint16)
    decision = choose_exposure(saturated, 0.02, max_ms=0.4)
    assert not decision.accepted
    assert decision.next_ms < 0.02


def test_invalid_auto_exposure_inputs_fail_closed() -> None:
    with pytest.raises(ValueError):
        choose_exposure(np.zeros((2, 2), dtype=np.float32), 0.02, max_ms=0.4)
    with pytest.raises(ValueError):
        choose_exposure(np.zeros((2, 2), dtype=np.uint16), 0.41, max_ms=0.4)


# --- per-session ceiling: env var, else image probe -------------------------


def test_configured_limit_reads_env(monkeypatch) -> None:
    monkeypatch.setenv(DAHENG_EXPOSURE_LIMIT_ENV, "0.6")
    assert configured_daheng_exposure_limit() == 0.6
    monkeypatch.setenv(DAHENG_EXPOSURE_LIMIT_ENV, "0")
    assert configured_daheng_exposure_limit() == 0.0
    monkeypatch.delenv(DAHENG_EXPOSURE_LIMIT_ENV)
    assert configured_daheng_exposure_limit() == 0.0
    monkeypatch.setenv(DAHENG_EXPOSURE_LIMIT_ENV, "not-a-number")
    with pytest.raises(ValueError):
        configured_daheng_exposure_limit()


def test_ceiling_estimate_uses_the_most_conservative_frame() -> None:
    # uint8 full scale 255: peaks 51/102/204 are exactly 0.2/0.4/0.8 of it, so
    # the fitted line through them is peak = 2550 * ms and the fitted
    # saturation point is 0.09 ms.  The clipped 255 frame at 0.16 ms is a
    # measured *bound* of 0.9 * 0.16 = 0.144 ms: it is what caps the fit (not a
    # per-rung estimate that could not raise a minimum), so any fit
    # extrapolating past 0.144 ms is clamped to it.  Here the fitted 0.09 ms
    # sits below the bound, so the bound does not decide this ceiling.
    ladder = [
        (0.02, np.full((4, 4), 51, dtype=np.uint8)),
        (0.04, np.full((4, 4), 102, dtype=np.uint8)),
        (0.08, np.full((4, 4), 204, dtype=np.uint8)),
        (0.16, np.full((4, 4), 255, dtype=np.uint8)),
    ]
    estimate = estimate_exposure_ceiling_ms(ladder)
    assert estimate.last_safe_ms == 0.08
    assert estimate.saturated_ms == 0.16
    assert estimate.ceiling_ms == pytest.approx(0.081)


def test_ceiling_estimate_below_start_when_already_saturated() -> None:
    with pytest.raises(ValueError, match="saturated at the minimum"):
        estimate_exposure_ceiling_ms(
            [(0.02, np.full((2, 2), 255, dtype=np.uint8))]
        )


def test_ceiling_estimate_fails_without_signal() -> None:
    with pytest.raises(ValueError, match="no ladder frame carries signal"):
        estimate_exposure_ceiling_ms([(0.02, np.zeros((2, 2), dtype=np.uint8))])


def test_probe_ladder_stops_at_the_first_saturated_frame() -> None:
    calls: list[float] = []

    def set_exposure(ms: float) -> float:
        calls.append(float(ms))
        return float(ms)

    def shoot() -> np.ndarray:
        peak = int(255 * min(1.0, calls[-1] / 0.16))
        return np.full((4, 4), peak, dtype=np.uint8)

    estimate = probe_exposure_ceiling_ms(set_exposure, shoot, start_ms=0.02)
    assert calls == [0.02, 0.04, 0.08, 0.16]
    assert estimate.saturated_ms == 0.16
    assert estimate.last_safe_ms == 0.08
    assert estimate.ceiling_ms == pytest.approx(0.1296)


def test_probe_respects_the_hardware_cap() -> None:
    calls: list[float] = []

    def set_exposure(ms: float) -> float:
        calls.append(float(ms))
        return float(ms)

    def shoot() -> np.ndarray:
        # A dim but *real* signal: a small blob well above the frame's noise
        # floor, so the frame counts as signal (a lone count on a zero
        # background is indistinguishable from read noise — that is the defect
        # the noise-floor criterion removes).
        frame = np.zeros((4, 4), dtype=np.uint8)
        frame[1:3, 1:3] = 8
        return frame

    estimate = probe_exposure_ceiling_ms(
        set_exposure, shoot, start_ms=0.02, hard_max_ms=0.08
    )
    assert calls == [0.02, 0.04, 0.08]
    # Nothing measured can say where the bench saturates beyond the camera
    # maximum, and nothing above ``hard_max_ms`` is settable, so the ceiling is
    # the cap with the safety discount applied -- not an extrapolation past it
    # that the caller could never use.  This ladder is dim enough that the fit
    # is rejected (8/255 = 3 % is below the 10 % gate), so it is the per-rung
    # minimum that would otherwise run past the cap.
    assert estimate.ceiling_ms == pytest.approx(0.08 * 0.9)
    assert estimate.bound_ms == pytest.approx(0.08)


# --- unified ceiling resolver: explicit > env > image probe -----------------


class _ProbeCamera:
    """Duck-typed camera whose peak rises linearly until ``saturate_at_ms``.

    Only ``reset_exposure_time`` and ``get_numpy_image`` are exercised, matching
    the two attributes :func:`resolve_daheng_exposure_limit` depends on.
    """

    def __init__(self, saturate_at_ms: float = 0.16) -> None:
        self.saturate_at_ms = saturate_at_ms
        self.current_ms = 0.0
        self.calls: list[float] = []

    def reset_exposure_time(self, ms: float) -> float:
        self.current_ms = float(ms)
        self.calls.append(float(ms))
        return float(ms)

    def get_numpy_image(self, n_sample: int = 2) -> np.ndarray:
        if n_sample != 1:
            return np.zeros((4, 4), dtype=np.float32)
        peak = int(255 * min(1.0, self.current_ms / self.saturate_at_ms))
        return np.full((4, 4), peak, dtype=np.uint8)


def test_resolve_prefers_explicit_value_and_never_probes(monkeypatch) -> None:
    monkeypatch.setenv(DAHENG_EXPOSURE_LIMIT_ENV, "0.9")
    camera = _ProbeCamera()
    assert resolve_daheng_exposure_limit(0.6, camera=camera) == 0.6
    assert camera.calls == []  # explicit short-circuits both env and probe


def test_resolve_falls_back_to_env_without_probing(monkeypatch) -> None:
    monkeypatch.setenv(DAHENG_EXPOSURE_LIMIT_ENV, "0.6")
    camera = _ProbeCamera()
    assert resolve_daheng_exposure_limit(0.0, camera=camera) == 0.6
    assert camera.calls == []


def test_resolve_probes_camera_when_nothing_configured(monkeypatch) -> None:
    monkeypatch.delenv(DAHENG_EXPOSURE_LIMIT_ENV, raising=False)
    camera = _ProbeCamera(saturate_at_ms=0.16)
    ceiling = resolve_daheng_exposure_limit(0.0, camera=camera)
    assert camera.calls == [0.02, 0.04, 0.08, 0.16]  # doubling ladder to saturation
    assert ceiling == pytest.approx(0.144 * 0.9, rel=1e-9)
    assert ceiling >= 0.02


def test_resolve_without_camera_or_config_raises(monkeypatch) -> None:
    monkeypatch.delenv(DAHENG_EXPOSURE_LIMIT_ENV, raising=False)
    with pytest.raises(ValueError, match="pass an open camera to probe"):
        resolve_daheng_exposure_limit(0.0, camera=None)


def test_resolve_raises_when_probe_hits_the_ladder_floor(monkeypatch) -> None:
    # Already saturated at the ladder floor: no exposure setting can help.
    monkeypatch.delenv(DAHENG_EXPOSURE_LIMIT_ENV, raising=False)
    camera = _ProbeCamera(saturate_at_ms=0.01)
    with pytest.raises(RuntimeError, match="reduce laser power or add attenuation"):
        resolve_daheng_exposure_limit(0.0, camera=camera)


def test_resolve_rejects_invalid_requested_value() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        resolve_daheng_exposure_limit(-1.0)
    with pytest.raises(ValueError, match="finite and non-negative"):
        resolve_daheng_exposure_limit(float("inf"))


# --- noise floor: the ladder floor is read noise, not light ----------------


def _noise_frame(peak_counts: int = 2, size: int = 64) -> np.ndarray:
    """A frame that carries no light: a zero background with a few hot pixels."""
    frame = np.zeros((size, size), dtype=np.uint8)
    frame[0, 0] = peak_counts
    frame[1, 1] = peak_counts
    frame[2, 2] = peak_counts
    return frame


def _signal_frame(peak_counts: int, size: int = 64) -> np.ndarray:
    """A frame whose brightest pixel carries ``peak_counts`` of real light."""
    frame = np.zeros((size, size), dtype=np.uint8)
    frame[size // 2, size // 2] = peak_counts
    return frame


class _NoiseFloorCamera:
    """Fake camera whose dark end of the ladder is pure read noise.

    Models the bench measured on the experiment system (Daheng MER2,
    2026-10-21): the ladder floor sits at the sensor minimum exposure, where
    the frame carries no light at all — its peak is **2 counts** of read noise
    on a zero background — while the peak grows linearly with the exposure
    above it and clips at ``saturate_at_ms``.
    """

    def __init__(self, *, saturate_at_ms: float = 50.0) -> None:
        self.saturate_at_ms = float(saturate_at_ms)
        self.current_ms = 0.0

    def reset_exposure_time(self, ms: float) -> float:
        self.current_ms = float(ms)
        return float(ms)

    def get_numpy_image(self, n_sample: int = 1, skip_first: bool = True) -> np.ndarray:
        if n_sample != 1:
            raise ValueError("the fake camera serves single frames only")
        signal = int(255.0 * min(1.0, self.current_ms / self.saturate_at_ms))
        if signal <= 0:
            return _noise_frame()
        return _signal_frame(signal)


def _probe_with(camera, **kwargs) -> ExposureCeilingEstimate:
    """Probe the fake camera exactly like the Daheng resolver does."""
    return probe_exposure_ceiling_ms(
        camera.reset_exposure_time,
        lambda: np.asarray(camera.get_numpy_image(n_sample=1)),
        **kwargs,
    )


def test_probe_ignores_the_noise_frames_at_the_bottom_of_the_ladder() -> None:
    """The 2026-10-21 failure: the noise floor dragged the ceiling to 2 ms.

    The bench saturates at 50 ms, but the 0.02 ms frame peaks at 2 counts of
    read noise, so extrapolating from it gives
    ``0.02 * 0.9 / (2/255) = 2.295 ms`` -> a ceiling of 2.0655 ms, ~30x below
    the truth, and the hardware flow then refused legal exposures.
    """
    camera = _NoiseFloorCamera(saturate_at_ms=50.0)
    estimate = _probe_with(camera, start_ms=0.02)
    assert 25.0 <= estimate.ceiling_ms <= 100.0
    assert estimate.ceiling_ms > 10.0  # nowhere near the 2.0655 ms failure


def test_the_noise_floor_is_what_used_to_break_the_ceiling(monkeypatch) -> None:
    """With the noise-floor test switched off the old 2.0655 ms comes back."""
    monkeypatch.setattr(exposure_control, "NOISE_FLOOR_SIGMAS", 0.0)
    camera = _NoiseFloorCamera(saturate_at_ms=50.0)
    estimate = _probe_with(camera, start_ms=0.02)
    assert estimate.ceiling_ms == pytest.approx(2.0655, rel=1e-3)


def test_a_ladder_of_nothing_but_noise_fails_with_an_actionable_error() -> None:
    camera = _NoiseFloorCamera(saturate_at_ms=50.0)
    # Every frame of this ladder is a noise frame: the bench is dark enough
    # that even the top of the ladder carries no light.
    camera.saturate_at_ms = 1e6
    with pytest.raises(ValueError, match="no ladder frame carries signal"):
        _probe_with(camera, start_ms=0.02)


def test_a_clean_signal_ladder_matches_the_analytic_ceiling() -> None:
    """With a linear, pedestal-free response the estimate is the analytic value."""
    ladder = [
        (ms, _signal_frame(int(255.0 * ms / 50.0)))
        for ms in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 50.0)
    ]
    estimate = estimate_exposure_ceiling_ms(ladder)
    # 0.9 (saturation fraction) * 50 ms (true ceiling) * 0.9 (headroom)
    assert estimate.ceiling_ms == pytest.approx(40.5, rel=0.05)
    assert estimate.saturated_ms == 50.0
    assert estimate.last_safe_ms == 32.0


# --- rung selection: fit the ceiling, do not min over the rungs ------------
#
# Measured on the Daheng MER2 in the experiment system on 2026-10-24, one probe
# (``data/debug/bench_settings/exposure_probe/20261024_110551.json``):
# ``ms`` 0.02..0.64 with peaks 13/21/38/74/142/255(clipped).  The per-rung
# extrapolation is 0.353/0.437/0.483/0.496/0.517/0.576 ms, so ``min`` over the
# rungs returned a ceiling of 0.3178 ms even though the three brightest
# unsaturated rungs agree on ~0.52 ms: the 0.02 ms rung's 13 counts are read
# noise (true signal ~8.5 counts), not light.
_MEASURED_LADDER: tuple[tuple[float, int], ...] = (
    (0.02, 13),
    (0.04, 21),
    (0.08, 38),
    (0.16, 74),
    (0.32, 142),
    (0.64, 255),
)


def _measured_ladder() -> list[tuple[float, np.ndarray]]:
    return [(ms, _signal_frame(peak)) for ms, peak in _MEASURED_LADDER]


def test_the_measured_ladder_fits_the_ceiling_instead_of_min_over_rungs() -> None:
    """Replay of the 2026-10-24 probe: the biased low rung must not decide."""
    estimate = estimate_exposure_ceiling_ms(_measured_ladder())
    # The old rule, kept here as the number the fix has to move away from.
    biased = min(estimate.estimates_ms) * 0.9
    assert biased == pytest.approx(0.3178, abs=1e-4)
    assert estimate.ceiling_ms == pytest.approx(0.470, abs=0.02)
    assert estimate.ceiling_ms > biased * 1.3  # a clear margin, not a nudge
    # Only the rungs at 0.08/0.16/0.32 ms carry enough light to be fitted.
    assert estimate.fit_points == 3
    assert estimate.fit_slope is not None and estimate.fit_slope > 0.0
    assert estimate.fit_intercept is not None


def test_a_clipped_step_caps_an_extrapolation_that_runs_past_it() -> None:
    """A clipped frame is measured fact; the fit may not extrapolate past it."""
    # These rungs roll off, so the fit puts 90 % of full scale *after* the step
    # that is already clipped.
    ladder = [
        (0.02, _signal_frame(60)),
        (0.04, _signal_frame(100)),
        (0.08, _signal_frame(130)),
        (0.16, _signal_frame(255)),
    ]
    estimate = estimate_exposure_ceiling_ms(ladder)
    assert estimate.saturated_ms == 0.16
    assert estimate.fit_ms is not None and estimate.fit_ms > 0.16
    # Clipping at 0.16 means 90 % of full scale was reached by 0.9 * 0.16.
    assert estimate.ceiling_ms == pytest.approx(0.16 * 0.9 * 0.9)


def test_a_ladder_with_no_bright_rung_keeps_the_min_over_rungs() -> None:
    """The dim-bench path: nothing reaches 10 % of full scale, so no fit."""
    ladder = [
        (0.02, _signal_frame(5)),
        (0.04, _signal_frame(10)),
        (0.08, _signal_frame(20)),
    ]
    estimate = estimate_exposure_ceiling_ms(ladder)
    assert estimate.fit_points == 0
    assert estimate.fit_ms is None
    assert estimate.ceiling_ms == pytest.approx(min(estimate.estimates_ms) * 0.9)


# --- fit guards: a slope no rung constrains must not become a ceiling ------


def test_a_flat_pedestal_ladder_is_not_fitted_into_a_ceiling_that_does_not_exist() -> None:
    """The top-rung gate: a DC pedestal is not a slope.

    26/27 counts at every exposure.  Every frame passes ``_carries_signal`` (a
    frame whose peak equals its median is admitted as signal) and 26/255 =
    10.2 % squeaks past the 10 % gate, so all four rungs enter the fit.  The
    line through them is 4.35 counts/ms with a 26.2-count intercept — the whole
    "signal" is the pedestal — and extrapolating it would return 42.1 ms, 265x
    the conservative per-rung minimum of 0.159 ms, on a bench that clips long
    before that.
    """
    ladder = [
        (0.02, np.full((8, 8), 26, dtype=np.uint8)),
        (0.04, np.full((8, 8), 27, dtype=np.uint8)),
        (0.08, np.full((8, 8), 26, dtype=np.uint8)),
        (0.16, np.full((8, 8), 27, dtype=np.uint8)),
    ]
    estimate = estimate_exposure_ceiling_ms(ladder)
    assert estimate.fit_ms is None  # rejected: the top rung is only 10.6 % bright
    assert estimate.fit_points == 4  # ...but all four rungs did enter the fit
    assert estimate.ceiling_ms == pytest.approx(min(estimate.estimates_ms) * 0.9)
    assert estimate.ceiling_ms == pytest.approx(0.1589, abs=1e-4)


def test_two_rungs_are_not_enough_to_constrain_a_fit() -> None:
    """The three-rung floor: two points leave no residual to reject a bad line."""
    # The measured counter-example: 30/34 counts at 0.02/0.04 ms fits 0.916 ms
    # against a 0.138 ms per-rung minimum.
    dim = [(0.02, _signal_frame(30)), (0.04, _signal_frame(34))]
    estimate = estimate_exposure_ceiling_ms(dim)
    assert estimate.fit_points == 2
    assert estimate.fit_ms is None
    assert estimate.ceiling_ms == pytest.approx(min(estimate.estimates_ms) * 0.9)
    assert estimate.ceiling_ms == pytest.approx(0.1377, abs=1e-4)

    # Bright enough to clear the top-rung gate, so only the rung count rejects
    # this one — that is what pins the three-rung floor rather than the gate.
    bright = [(0.02, _signal_frame(120)), (0.04, _signal_frame(200))]
    estimate = estimate_exposure_ceiling_ms(bright)
    assert estimate.fit_points == 2
    assert estimate.fit_ms is None
    assert estimate.ceiling_ms == pytest.approx(min(estimate.estimates_ms) * 0.9)


# --- clamping: the measured bound, not the extrapolation, decides ----------


def test_the_first_saturated_step_caps_the_fit_even_when_nothing_clipped() -> None:
    """``first_saturated_ms`` bounds the fit on its own when nothing clipped."""
    # 50/75/125 counts at 0.02/0.04/0.08 ms lie on peak = 25 + 1250 * ms, so the
    # fit puts 90 % of full scale (229.5 counts) at 0.1636 ms.  The 0.16 ms
    # frame is 242/255 = 0.949: at the saturation fraction but *not* clipped, so
    # ``first_clipped_ms`` stays None and only ``first_saturated_ms`` bounds the
    # ceiling.  The extrapolation past 0.16 is still clamped to it.
    ladder = [
        (0.02, _signal_frame(50)),
        (0.04, _signal_frame(75)),
        (0.08, _signal_frame(125)),
        (0.16, _signal_frame(242)),
    ]
    estimate = estimate_exposure_ceiling_ms(ladder)
    assert estimate.fit_ms == pytest.approx(0.1636, abs=1e-4)
    assert estimate.saturated_ms == 0.16
    assert estimate.fit_ms > estimate.saturated_ms
    assert estimate.first_clipped_ms is None  # nothing reached full scale
    assert estimate.bound_ms == pytest.approx(0.16)
    # One discount only: ``first_saturated_ms`` is the exposure at which the
    # saturation level was *measured*, so the bound is 0.16 ms itself and only
    # ``headroom`` applies.  The extra ``saturation_fraction`` factor belongs to
    # the *clipped* bound, where the true saturation point is only bounded.
    assert estimate.ceiling_ms == pytest.approx(0.16 * 0.9)


def test_probe_caps_an_accepted_fit_at_the_hardware_maximum() -> None:
    """A ladder that reaches the camera maximum without saturating is capped."""
    calls: list[float] = []

    def set_exposure(ms: float) -> float:
        calls.append(float(ms))
        return float(ms)

    def shoot() -> np.ndarray:
        return _signal_frame(int(1500.0 * calls[-1]))  # 30/60/120 counts

    estimate = probe_exposure_ceiling_ms(
        set_exposure, shoot, start_ms=0.02, hard_max_ms=0.08
    )
    assert calls == [0.02, 0.04, 0.08]
    assert estimate.saturated_ms is None  # never reached 90 % before the cap
    # The fit is accepted (0.153 ms) and is what the walk saw; the camera simply
    # cannot expose past 0.08 ms, so the ceiling is the cap with the discount.
    assert estimate.fit_ms == pytest.approx(0.153)
    assert estimate.ceiling_ms == pytest.approx(0.08 * 0.9)
    assert estimate.bound_ms == pytest.approx(0.08)


