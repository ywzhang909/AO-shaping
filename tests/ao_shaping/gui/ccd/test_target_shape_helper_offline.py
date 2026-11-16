"""Target-shape helper: M4 effective ROI readback and M5 stale-camera close (offline, no hardware)."""

from __future__ import annotations

from ao_shaping.gui.ccd.target_shape_helper import (
    _close_stale_camera,
    _effective_window,
    _store_effective_window,
)


class _FakeCamera:
    def __init__(self, close_error: Exception | None = None) -> None:
        self.closed = False
        self._close_error = close_error

    def close(self) -> None:
        self.closed = True
        if self._close_error is not None:
            raise self._close_error


class _NoCloseCamera:
    pass


def test_m4_effective_window_overrides_request() -> None:
    """Readback clamped by the device wins over the requested geometry."""
    eff_size, eff_center = _effective_window((300, 300), (10.0, 20.0), ((256, 288), (5, 7)))
    assert eff_size == (256, 288)
    assert eff_center == (5.0, 7.0)


def test_m4_none_readback_keeps_requested() -> None:
    """Backends without readback (None) keep the requested geometry."""
    eff_size, eff_center = _effective_window((300, 300), (10.0, 20.0), None)
    assert eff_size == (300, 300)
    assert eff_center == (10.0, 20.0)


def test_m4_store_writes_effective_window_to_session() -> None:
    """The stored session geometry reflects the effective window, not the request."""
    state: dict[str, object] = {}
    eff_size, eff_center = _store_effective_window(
        state, 300, (10.0, 20.0), ((256, 288), (5, 7))
    )
    assert state["ts_window_size"] == 256
    assert state["ts_roi_window"] == ((256, 288), (5.0, 7.0))
    assert eff_size == (256, 288)
    assert eff_center == (5.0, 7.0)


def test_m4_store_none_readback_keeps_requested_in_session() -> None:
    state: dict[str, object] = {}
    _store_effective_window(state, 300, (10.0, 20.0), None)
    assert state["ts_window_size"] == 300
    assert state["ts_roi_window"] == ((300, 300), (10.0, 20.0))


def test_m5_stale_camera_closed_before_discard() -> None:
    cam = _FakeCamera()
    _close_stale_camera(cam, "not connected")
    assert cam.closed is True


def test_m5_close_failure_is_swallowed() -> None:
    """A failing close must not raise; the discard proceeds."""
    cam = _FakeCamera(close_error=RuntimeError("device busy"))
    _close_stale_camera(cam, "not connected")  # must not raise
    assert cam.closed is True


def test_m5_camera_without_close_is_ignored() -> None:
    _close_stale_camera(_NoCloseCamera(), "not connected")  # must not raise


def test_m5_none_camera_is_ignored() -> None:
    _close_stale_camera(None, "not connected")  # must not raise
