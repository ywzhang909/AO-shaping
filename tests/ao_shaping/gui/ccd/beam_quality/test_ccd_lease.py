"""CCD 进程内租约 + ``CameraFrameSource`` 接入契约（无硬件、无窗口）。

覆盖:
* 同键冲突不覆盖旧租约, 释放后可重新获取, 重复释放幂等;
* 不同相机可同时租用, ``current()`` 跟随最近一次获取;
* ``CameraFrameSource.start`` 成功即持有租约, 第二次 start 同一相机抛
  ``RuntimeError`` 且不触碰设备;
* start 中途失败 (open / start_stream) 释放租约, 之后可被重新获取;
* ``stop`` 幂等且释放; 仿真源不占租约。
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from ao_shaping.gui.ccd.beam_quality import ccd_lease
from ao_shaping.gui.ccd.beam_quality.frame_source import (
    CameraFrameSource,
    SimBeamSource,
)


@pytest.fixture(autouse=True)
def _clean_lease():
    """每个用例前后清空进程级租约, 避免用例间泄漏。"""
    ccd_lease.reset_for_tests()
    yield
    ccd_lease.reset_for_tests()


class _FakeBackend:
    """最小相机后端: 起流时同步推一帧, 不碰任何 SDK。"""

    def __init__(self, *, fail_open: bool = False, fail_stream: bool = False) -> None:
        self.fail_open = fail_open
        self.fail_stream = fail_stream
        self.opened = False
        self.closed = False
        self.streaming = False
        self.exposure_ms = 10.0

    def open(self) -> None:
        if self.fail_open:
            raise RuntimeError("open boom")
        self.opened = True

    def close(self) -> None:
        self.closed = True

    def start_stream(self, on_frame) -> None:
        if self.fail_stream:
            raise RuntimeError("stream boom")
        self.streaming = True
        on_frame(np.full((4, 4), 5, dtype=np.uint8))

    def stop_stream(self) -> None:
        self.streaming = False

    def get_exposure_time_ms(self) -> float:
        return self.exposure_ms

    def get_exposure_range(self) -> tuple[float, float]:
        return (0.02, 1000.0)


# ------------------------------------------------------------------ 纯租约
def test_acquire_conflict_keeps_old_lease() -> None:
    assert ccd_lease.try_acquire("Daheng", 0) is True
    assert ccd_lease.current() == ("daheng", 0)

    # "0" 与 0 是同一台相机; 失败方不得覆盖旧租约
    assert ccd_lease.try_acquire("daheng", "0") is False
    assert ccd_lease.current() == ("daheng", 0)
    assert ccd_lease.held_keys() == (("daheng", 0),)


def test_release_is_idempotent_and_allows_reacquire() -> None:
    assert ccd_lease.try_acquire("daheng", 1) is True
    ccd_lease.release("daheng", 1)
    assert ccd_lease.current() is None
    ccd_lease.release("daheng", 1)  # 二次释放无操作
    ccd_lease.release("daheng", 9)  # 释放未持有的键无操作
    assert ccd_lease.current() is None
    assert ccd_lease.try_acquire("daheng", 1) is True


def test_distinct_cameras_can_be_leased_and_current_tracks_latest() -> None:
    assert ccd_lease.try_acquire("daheng", 0) is True
    assert ccd_lease.try_acquire("miicam", 1) is True
    assert ccd_lease.current() == ("miicam", 1)

    ccd_lease.release("miicam", 1)
    assert ccd_lease.current() == ("daheng", 0)
    assert ccd_lease.held_keys() == (("daheng", 0),)


def test_invalid_key_is_rejected_without_registration() -> None:
    with pytest.raises(ValueError):
        ccd_lease.try_acquire("", 0)
    with pytest.raises(ValueError):
        ccd_lease.try_acquire("daheng", "")
    assert ccd_lease.current() is None


# ---------------------------------------------------- CameraFrameSource 接入
def test_source_start_holds_lease_and_conflict_raises() -> None:
    first_backend = _FakeBackend()
    first = CameraFrameSource(camera_type="daheng", cam_id=0, camera=first_backend)
    first.start()
    try:
        assert ccd_lease.current() == ("daheng", 0)

        second_backend = _FakeBackend()
        second = CameraFrameSource(
            camera_type="daheng", cam_id=0, camera=second_backend
        )
        with pytest.raises(RuntimeError, match="已被占用"):
            second.start()
        # 冲突时不触碰设备
        assert second_backend.opened is False
    finally:
        first.stop()

    assert first_backend.closed is True
    assert ccd_lease.current() is None


def test_source_start_failure_releases_lease() -> None:
    broken = CameraFrameSource(
        camera_type="daheng", cam_id=0, camera=_FakeBackend(fail_open=True)
    )
    with pytest.raises(RuntimeError, match="open boom"):
        broken.start()
    assert broken.is_running is False
    assert ccd_lease.current() is None

    stream_broken = CameraFrameSource(
        camera_type="daheng", cam_id=0, camera=_FakeBackend(fail_stream=True)
    )
    with pytest.raises(RuntimeError, match="stream boom"):
        stream_broken.start()
    assert ccd_lease.current() is None

    healthy = CameraFrameSource(camera_type="daheng", cam_id=0, camera=_FakeBackend())
    healthy.start()
    try:
        assert ccd_lease.current() == ("daheng", 0)
    finally:
        healthy.stop()
    assert ccd_lease.current() is None


def test_source_stop_is_idempotent_and_releases() -> None:
    source = CameraFrameSource(camera_type="daheng", cam_id=2, camera=_FakeBackend())
    source.start()
    assert ccd_lease.current() == ("daheng", 2)
    source.stop()
    source.stop()
    assert ccd_lease.current() is None


def test_sim_source_does_not_take_lease() -> None:
    assert ccd_lease.try_acquire("daheng", 0) is True
    sim = SimBeamSource(width=32, height=24, fps=50.0)
    sim.start()
    try:
        assert ccd_lease.current() == ("daheng", 0)
        assert ccd_lease.held_keys() == (("daheng", 0),)
    finally:
        sim.stop()
    assert ccd_lease.current() == ("daheng", 0)
