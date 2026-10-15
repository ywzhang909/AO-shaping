"""Quantitative CCD exposure must be bounded and scale consistently."""

import numpy as np
import pytest

from ao_shaping.utils.image.exposure_control import (
    DAHENG_EXPOSURE_LIMIT_ENV,
    choose_exposure,
    configured_daheng_exposure_limit,
    estimate_exposure_ceiling_ms,
    exposure_normalized_frame,
    probe_exposure_ceiling_ms,
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
    assert estimate.ceiling_ms == pytest.approx(0.09 * 0.9, rel=1e-9)


def test_ceiling_estimate_below_start_when_already_saturated() -> None:
    estimate = estimate_exposure_ceiling_ms(
        [(0.02, np.full((2, 2), 255, dtype=np.uint8))]
    )
    assert estimate.ceiling_ms < 0.02
    assert estimate.last_safe_ms == 0.0
    assert estimate.saturated_ms == 0.02


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
    assert estimate.ceiling_ms == pytest.approx(0.144 * 0.9, rel=1e-9)


def test_probe_respects_the_hardware_cap() -> None:
    calls: list[float] = []

    def set_exposure(ms: float) -> float:
        calls.append(float(ms))
        return float(ms)

    def shoot() -> np.ndarray:
        frame = np.zeros((4, 4), dtype=np.uint8)
        frame[2, 2] = 1  # dim but nonzero: never saturates within the cap
        return frame

    estimate = probe_exposure_ceiling_ms(
        set_exposure, shoot, start_ms=0.02, hard_max_ms=0.08
    )
    assert calls == [0.02, 0.04, 0.08]
    assert estimate.ceiling_ms > 0.08
