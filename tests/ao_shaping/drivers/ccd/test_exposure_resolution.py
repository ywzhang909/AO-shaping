"""Offline tests for the live-verified exposure resolution (no hardware).

``resolve_exposure_ms`` uses the saved flat-field frames only as an initial guess
and then rescales against the LIVE peak, because stale flats were measured 12x
dimmer than the bench. A fake camera models ``peak = k * exposure`` so the
convergence, saturation handling and black-frame guard can be checked offline.
"""

from __future__ import annotations

import numpy as np
import pytest

import ao_shaping.drivers.ccd.common as ccd_common
from ao_shaping.drivers.ccd.common import resolve_exposure_ms


@pytest.fixture
def stale_flat(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend the saved flats suggest 191.5 ms (the real stale value)."""
    monkeypatch.setattr(ccd_common, "select_exposure_from_flats", lambda **kwargs: 191.5)


class FakeCamera:
    """Linear dose model: peak = k * exposure (clipped to 255)."""

    def __init__(self, k: float = 10.0, exposure_ms: float = 3.0) -> None:
        self.k = k
        self.exposure_ms = exposure_ms
        self.set_calls: list[float] = []

    def get_numpy_image(self, n_sample: int = 1) -> np.ndarray:
        peak = min(self.k * self.exposure_ms, 255.0)
        return np.full((8, 8), peak, dtype=np.uint8)

    def reset_exposure_time(self, time_ms: float) -> None:  # preferred setter path
        self.exposure_ms = float(time_ms)
        self.set_calls.append(float(time_ms))


def test_converges_to_target(stale_flat: None) -> None:
    cam = FakeCamera(k=10.0)  # target 200 -> 20 ms

    exposure, peak = resolve_exposure_ms(
        cam, target_max=200.0, tolerance=12.0, max_iterations=10, n_sample=1
    )

    assert abs(peak - 200.0) <= 12.0
    assert exposure == pytest.approx(20.0, rel=0.15)


def test_stale_flat_guess_is_corrected_live(stale_flat: None) -> None:
    """A 191.5 ms flat guess would saturate; the live bisection must step down."""
    cam = FakeCamera(k=10.0, exposure_ms=3.0)

    exposure, peak = resolve_exposure_ms(
        cam, target_max=200.0, tolerance=12.0, max_iterations=10, n_sample=1
    )

    assert peak < 250.0
    assert exposure < 100.0


def test_saturation_steps_down(stale_flat: None) -> None:
    cam = FakeCamera(k=40.0)  # target 200 -> 5ms

    exposure, peak = resolve_exposure_ms(
        cam, target_max=200.0, tolerance=12.0, max_iterations=12, n_sample=1
    )

    assert peak < 250.0
    assert exposure == pytest.approx(5.0, rel=0.25)


def test_black_frame_guard_does_not_crank_exposure(stale_flat: None) -> None:
    cam = FakeCamera(k=0.0, exposure_ms=3.0)  # no light at all

    exposure, peak = resolve_exposure_ms(
        cam, target_max=200.0, tolerance=12.0, max_iterations=5, n_sample=1
    )

    assert peak == 0.0
    assert len(cam.set_calls) == 1  # stops instead of escalating
    assert exposure <= 1000.0


def test_tolerance_zero_keeps_single_pass(stale_flat: None) -> None:
    cam = FakeCamera(k=10.0)

    _exposure, peak = resolve_exposure_ms(
        cam, target_max=200.0, tolerance=0.0, max_iterations=4, n_sample=1
    )

    assert len(cam.set_calls) == 1
    assert np.isfinite(peak)
