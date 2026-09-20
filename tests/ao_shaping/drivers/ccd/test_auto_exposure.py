"""Offline tests for the unified driver-level ``auto_exposure`` contract.

Both hardware backends expose the same signature and semantics:
``target_max`` / ``tolerance`` are 0-255 grayscale, and the method returns the
adjusted ``np.ndarray`` image. No hardware is touched: the MiiCam test injects a
fake ``self.cam`` and a fast fake ``reset_exposure_time``; the Daheng test mocks
the SDK handle.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _make_miicam(peak_per_ms: float = 10.0):
    """Construct an unopened MIICamera with a scripted image source."""
    from ao_shaping.drivers.ccd.miicam.driver import MIICamera

    cam = MIICamera(cam_id=0, exposure_time_ms=10.0)
    cam.cam = MagicMock()

    def fake_image(n_sample=1, skip_first=True):
        peak = min(cam.exposure_time_ms * peak_per_ms, 255.0)
        return np.full((4, 4), int(round(peak)), dtype=np.uint8)

    cam.get_numpy_image = fake_image  # type: ignore[method-assign]
    return cam


def _install_fast_reset(cam) -> list[float]:
    """Replace ``reset_exposure_time`` with a fast spy (keeps the contract)."""
    calls: list[float] = []

    def _reset(ms):
        calls.append(float(ms))
        cam.exposure_time_ms = float(ms)
        return float(ms)

    cam.reset_exposure_time = _reset  # type: ignore[method-assign]
    return calls


def test_miicam_auto_exposure_converges_and_returns_ndarray():
    cam = _make_miicam(peak_per_ms=10.0)
    _install_fast_reset(cam)

    result = cam.auto_exposure(target_max=50.0, tolerance=5.0, n_sample=1)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8
    assert np.max(result) == pytest.approx(50, abs=5)
    assert cam.exposure_time_ms == pytest.approx(5.0, rel=0.2)


def test_miicam_auto_exposure_changes_exposure_via_reset_exposure_time():
    cam = _make_miicam(peak_per_ms=10.0)
    calls = _install_fast_reset(cam)

    cam.auto_exposure(target_max=50.0, tolerance=5.0, n_sample=1)

    # MiiCam can only change exposure through reset_exposure_time (Stop->set->start).
    assert calls, "auto_exposure must route exposure changes through reset_exposure_time"


def test_miicam_auto_exposure_stops_at_hardware_boundary_when_unreachable():
    from ao_shaping.drivers.ccd.miicam.driver import MIICamera

    cam = MIICamera(cam_id=0, exposure_time_ms=1.0)
    cam.cam = MagicMock()
    # Peak locked at 80 regardless of exposure -> target 220 is unreachable.
    cam.get_numpy_image = lambda n_sample=1, skip_first=True: np.full(
        (4, 4), 80, dtype=np.uint8
    )
    _install_fast_reset(cam)

    result = cam.auto_exposure(target_max=220.0, tolerance=5.0, n_sample=1)

    assert isinstance(result, np.ndarray)
    assert cam.exposure_time_ms == pytest.approx(cam.max_exposure_ms)


@patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
def test_daheng_auto_exposure_returns_ndarray_not_tuple(mock_gx):
    """Regression: the old implementation returned ``(exp, ratio)`` when it
    converged, leaking a tuple into callers expecting an image."""
    from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

    cam = DahengCamera(cam_id=0, exposure_time_ms=10.0)
    cam.cam = MagicMock()
    cam.cam.ExposureTime.get.return_value = 10000  # 10 ms
    cam._DahengCamera__take_one_shot = MagicMock(  # type: ignore[attr-defined]
        return_value=np.full((4, 4), 50, dtype=np.uint8)
    )
    cam.cam_width, cam.cam_height = 4, 4

    result = cam.auto_exposure(
        target_max=50.0, tolerance=5.0, twice_valid=True, use_sdk_auto=False
    )

    assert isinstance(result, np.ndarray)
    assert int(np.max(result)) == 50


def test_sim_camera_exposes_auto_exposure():
    """pib_sim_eval patches pib's MIICamera with SimCamera, so it must match."""
    from ao_shaping.optimizer.wfless.pib_sim_eval import SimCamera

    cam = SimCamera(cam_id=0)
    img = cam.auto_exposure(target_max=40.0, twice_valid=False)
    assert isinstance(img, np.ndarray)


def test_exposure_range_prefers_backend_get_exposure_range():
    """DahengCamera exposes ``get_exposure_range()``; it must win over the
    MiiCam-style min/max properties *and* over the generic fallback — otherwise
    the reported range is the generic (0.011, 10000) instead of the device's."""

    class _OwnRangeCamera:
        def get_exposure_range(self):
            return (0.02, 1000.0)

        # MiiCam-style properties must NOT win when get_exposure_range exists.
        min_exposure_ms = 0.2
        max_exposure_ms = 350.0

    from ao_shaping.drivers.ccd.common import get_camera_exposure_range

    assert get_camera_exposure_range(_OwnRangeCamera()) == (0.02, 1000.0)


def test_exposure_range_falls_back_to_properties_then_generic():
    from ao_shaping.drivers.ccd.common import get_camera_exposure_range

    class _PropsOnly:
        min_exposure_ms = 0.2
        max_exposure_ms = 350.0

    class _Nothing:
        pass

    assert get_camera_exposure_range(_PropsOnly()) == (0.2, 350.0)
    assert get_camera_exposure_range(_Nothing()) == (0.011, 10_000.0)
