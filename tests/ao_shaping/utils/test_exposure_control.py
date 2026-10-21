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
    # uint8 full scale 255: peaks 51/102/204 are exactly 0.2/0.4/0.8 of it,
    # so every unsaturated estimate is 0.09 ms; the clipped 255 frame gives
    # 0.144 (its true saturation point is higher, so it cannot raise the min).
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
    assert estimate.ceiling_ms > 0.08


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
