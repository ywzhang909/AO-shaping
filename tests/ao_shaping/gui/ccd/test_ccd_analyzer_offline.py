"""CCD analyzer: L1(a) liveness probe and L1(b) single-shot capture (offline, no hardware)."""

from __future__ import annotations

import queue
import threading
import time
from typing import Any

import numpy as np
import pytest

from ao_shaping.gui.ccd import ccd_analyzer
from ao_shaping.gui.ccd.ccd_analyzer import (
    _camera_is_live,
    _run_capture_loop,
)


class _ProbeCamera:
    def __init__(self, result: bool | Exception) -> None:
        self._result = result

    def is_connected(self) -> bool:
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class _LegacyCamera:
    """MiiCam-style: no is_connected, liveness is the cam attribute."""

    def __init__(self, cam: object | None) -> None:
        self.cam = cam


class _FrameCamera:
    def __init__(self, fail: bool = False) -> None:
        self.calls = 0
        self._fail = fail

    def get_numpy_image(self, n_sample: int = 1, skip_first: bool = False) -> np.ndarray:
        self.calls += 1
        if self._fail:
            raise RuntimeError("grab failed")
        return np.zeros((4, 4), dtype=np.uint8)


def test_l1a_probe_false_means_not_live() -> None:
    assert _camera_is_live(_ProbeCamera(False)) is False


def test_l1a_probe_raising_means_not_live() -> None:
    assert _camera_is_live(_ProbeCamera(RuntimeError("usb gone"))) is False


def test_l1a_probe_true_means_live() -> None:
    assert _camera_is_live(_ProbeCamera(True)) is True


def test_l1a_legacy_camera_without_probe_uses_cam_attribute() -> None:
    assert _camera_is_live(_LegacyCamera(object())) is True
    assert _camera_is_live(_LegacyCamera(None)) is False


def test_l1b_single_shot_delivers_one_frame_and_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ccd_analyzer, "_mjpeg_streamer", None)
    cam = _FrameCamera()
    q: queue.Queue[Any] = queue.Queue(maxsize=3)
    stop = threading.Event()
    params = {"n_sample": 1, "skip_first": False, "single_shot": True, "dt": 0.01}
    thread = threading.Thread(target=_run_capture_loop, args=(cam, q, params, stop))
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert cam.calls == 1
    items = []
    while not q.empty():
        items.append(q.get_nowait())
    frames = [i for i in items if isinstance(i, np.ndarray)]
    assert len(frames) == 1


def test_l1b_single_shot_error_is_reported_on_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ccd_analyzer, "_mjpeg_streamer", None)
    cam = _FrameCamera(fail=True)
    q: queue.Queue[Any] = queue.Queue(maxsize=3)
    stop = threading.Event()
    params = {"n_sample": 1, "skip_first": False, "single_shot": True, "dt": 0.01}
    thread = threading.Thread(target=_run_capture_loop, args=(cam, q, params, stop))
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    item = q.get_nowait()
    assert item[0] == "error"
    assert "grab failed" in item[1]


def test_l1b_continuous_loop_keeps_running(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: without single_shot the worker keeps capturing until stopped."""
    monkeypatch.setattr(ccd_analyzer, "_mjpeg_streamer", None)
    cam = _FrameCamera()
    q: queue.Queue[Any] = queue.Queue(maxsize=3)
    stop = threading.Event()
    params = {"n_sample": 1, "skip_first": False, "dt": 0.01}
    thread = threading.Thread(target=_run_capture_loop, args=(cam, q, params, stop))
    thread.start()
    time.sleep(0.15)
    assert thread.is_alive()
    stop.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert cam.calls > 1
