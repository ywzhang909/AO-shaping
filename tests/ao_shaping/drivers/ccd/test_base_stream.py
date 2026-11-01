"""BaseCamera 通用取流层 (``start_stream`` / ``stop_stream`` / ``set_exposure_ms``)。

GUI (``BeamQualityApp``) 只认 ``BaseCamera`` 这一个接口, 因此「任意后端都能推帧」
必须是**基类**的性质, 而不是 MiiCam 的特例:

* ``supports_callback_stream=False`` (Daheng / ffmpeg / 仿真) → 基类开轮询线程,
  反复 ``get_numpy_image(1, skip_first=False)``;
* ``supports_callback_stream=True`` (MiiCam) → 子类覆写 ``start_stream`` 直接进
  SDK 回调模式, 不多开一层线程。

这里用无 SDK 的假子类锁定轮询分支的契约: 真能出帧、能幂等启停、抓帧异常不能打死
线程、``set_exposure_ms`` 默认委托 ``reset_exposure_time``、未打开时报
``CameraError`` 而不是把空句柄送进 SDK。
"""

from __future__ import annotations

import threading
import time
from typing import Callable

import numpy as np
import pytest

from ao_shaping.drivers.ccd.base import BaseCamera, CameraError


class _PollCamera(BaseCamera):
    """无硬件假相机: ``get_numpy_image`` 立刻返回一帧合成的图。"""

    stream_poll_interval_s = 0.005

    def __init__(self, fail_first: bool = False) -> None:
        super().__init__(cam_id=0, exposure_time_ms=1.0)
        self.cam = object()  # 视为 initialize() 已完成
        self.fail_first = fail_first
        self.pull_count = 0
        self.skip_first_flags: list[bool] = []
        self.exposure_requests: list[float] = []
        self.opened = False
        self.closed = False
        self._count_lock = threading.Lock()

    # --- 抽象方法 ---
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass

    def initialize(self) -> None:
        self.cam = object()

    def open(self) -> None:
        self.opened = True
        self.cam = object()

    def close(self) -> None:
        self.closed = True

    def reset_exposure_time(self, time_ms: float) -> float:
        self.exposure_requests.append(float(time_ms))
        return float(time_ms)

    def reset_window(self, center, size):
        return (0, 0), (0, 0)

    def enable_auto_exposure(self, enable: bool = True, mode: str = "once") -> None:
        pass

    def get_auto_exposure_state(self) -> dict:
        return {}

    def set_auto_exposure_range(self, min_ms: float, max_ms: float) -> None:
        pass

    @staticmethod
    def get_cam_list() -> list:
        return []

    def get_numpy_image(self, n_sample: int = 1, skip_first: bool = True) -> np.ndarray:
        with self._count_lock:
            self.pull_count += 1
            n = self.pull_count
            self.skip_first_flags.append(skip_first)
        if self.fail_first and n == 1:
            raise CameraError("synthetic pull failure")
        return np.full((2, 2), n % 256, dtype=np.uint8)


def _wait_until(predicate: Callable[[], bool], timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return predicate()


def test_start_stream_polls_until_stopped() -> None:
    cam = _PollCamera()
    frames: list[np.ndarray] = []
    cam.start_stream(lambda img: frames.append(img))
    thread = cam._stream_thread
    try:
        assert _wait_until(lambda: len(frames) >= 3), "轮询线程必须在 5s 内产出 3 帧"
        assert thread is not None and thread.is_alive()
        # 不丢首帧: skip_first 必须为 False (改曝光后的陈旧帧由驱动自己消费)
        assert set(cam.skip_first_flags) == {False}
    finally:
        cam.stop_stream()

    assert cam._stream_thread is None
    assert cam._stream_stop is None
    assert not thread.is_alive(), "stop_stream 必须等取流线程退出"
    count = len(frames)
    time.sleep(0.05)
    assert len(frames) == count, "stop_stream 之后不得再有回调"


def test_start_stream_is_idempotent() -> None:
    cam = _PollCamera()
    cam.start_stream()
    try:
        thread = cam._stream_thread
        assert thread is not None
        cam.start_stream()  # 第二次调用不得再起一个线程
        assert cam._stream_thread is thread
    finally:
        cam.stop_stream()


def test_stop_stream_without_start_is_safe() -> None:
    cam = _PollCamera()
    cam.stop_stream()
    cam.stop_stream()
    assert cam._stream_thread is None


def test_stream_survives_pull_failure() -> None:
    """单帧抓取异常只该被记录并重试, 不能打死取流线程 (否则界面永久停帧)。"""
    cam = _PollCamera(fail_first=True)
    frames: list[np.ndarray] = []
    cam.start_stream(lambda img: frames.append(img))
    try:
        assert _wait_until(lambda: len(frames) >= 2), "首帧异常后必须继续产帧"
        assert cam.pull_count >= 3
    finally:
        cam.stop_stream()


def test_callback_exception_does_not_kill_stream() -> None:
    cam = _PollCamera()
    seen: list[int] = []

    def _boom(_img: np.ndarray) -> None:
        seen.append(1)
        raise RuntimeError("consumer bug")

    cam.start_stream(_boom)
    try:
        assert _wait_until(lambda: len(seen) >= 3), "回调抛异常不得终止取流"
    finally:
        cam.stop_stream()


def test_start_stream_requires_open_camera() -> None:
    cam = _PollCamera()
    cam.cam = None  # 未 initialize
    with pytest.raises(CameraError):
        cam.start_stream()
    assert cam._stream_thread is None


def test_set_exposure_ms_defaults_to_reset_exposure_time() -> None:
    cam = _PollCamera()
    assert cam.set_exposure_ms(7.5) == pytest.approx(7.5)
    assert cam.exposure_requests == pytest.approx([7.5])


def test_backends_declare_stream_capability() -> None:
    """MiiCam 走原生回调 (覆写 start_stream); Daheng 走基类轮询 (零改动)。"""
    from ao_shaping.drivers.ccd.daheng.driver import DahengCamera
    from ao_shaping.drivers.ccd.miicam.driver import MIICamera

    assert MIICamera.supports_callback_stream is True
    assert DahengCamera.supports_callback_stream is False
    # 覆写必须真的发生: 两个后端在这里一个是子类方法, 一个是基类方法
    assert MIICamera.start_stream is not BaseCamera.start_stream
    assert DahengCamera.start_stream is BaseCamera.start_stream
