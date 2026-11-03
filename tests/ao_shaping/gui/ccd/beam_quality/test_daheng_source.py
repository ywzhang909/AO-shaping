"""Daheng 源接入 + 曝光范围契约的离线测试 (无硬件、无 gxipy)。

覆盖:
- ``SOURCE_KINDS`` 把 ``("Daheng", "daheng")`` 放在首位, 且可通过
  ``_source_combo`` 选中 (currentIndex 落到 Daheng 行);
- ``CameraFrameSource`` 的 ``exposure_range_ms`` 在未打开时回退到
  ``(0.011, 10000.0)``, 打开后查询真实设备范围 ``(0.02, 1000.0)``;
- ``get_exposure_range()`` 抛异常时属性不致命 (回退);
- ``SimBeamSource.exposure_range_ms == (0.011, 10000.0)``;
- UI: 仿真源打开后曝光 spin 范围 == 源的 ``exposure_range_ms``; 关闭后回退。
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from ao_shaping.gui.ccd.beam_quality.frame_source import (
    CameraFrameSource,
    SimBeamSource,
)


def _pump_until(qt_app, predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qt_app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    qt_app.processEvents()
    return bool(predicate())


class _DahengFakeBackend:
    """仿 Daheng 的假后端, 通过 camera= 注入 (不碰硬件 / gxipy)。"""

    def __init__(self) -> None:
        self.opened = False
        self.closed = False
        self.streaming = False
        self.exposure_ms = 20.0
        self.exposure_requests: list[float] = []
        self.frames_pushed = 0
        self.last_frame_info = None
        self.pixel_size_um = 2.2

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.closed = True

    def start_stream(self, on_frame) -> None:
        self.streaming = True
        for value in (7, 9):
            self.frames_pushed += 1
            on_frame(np.full((4, 4), value, dtype=np.uint8))

    def stop_stream(self) -> None:
        self.streaming = False

    def get_numpy_image(self, n, skip_first=False):
        return np.full((4, 4), 1, dtype=np.uint8)

    def get_exposure_range(self) -> tuple[float, float]:
        return (0.02, 1000.0)

    def get_exposure_time_ms(self) -> float:
        return self.exposure_ms

    def set_exposure_ms(self, ms: float) -> float:
        self.exposure_requests.append(float(ms))
        self.exposure_ms = float(ms)
        return self.exposure_ms


# ---------------------------------------------------------------- 契约: SOURCE_KINDS
def test_source_kinds_daheng_first() -> None:
    from ao_shaping.gui.ccd.beam_quality.app import SOURCE_KINDS

    assert SOURCE_KINDS[0] == ("Daheng", "daheng")
    kinds = [key for _label, key in SOURCE_KINDS]
    assert "daheng" in kinds


# ---------------------------------------------------------------- CameraFrameSource
def test_camera_frame_source_exposure_range_before_after_start() -> None:
    backend = _DahengFakeBackend()
    source = CameraFrameSource(camera_type="daheng", camera=backend)
    # 未打开: 回退
    assert source.exposure_range_ms == (0.011, 10000.0)

    source.start()
    try:
        assert backend.opened and backend.streaming
        # 打开后查询真实设备范围
        assert source.exposure_range_ms == (0.02, 1000.0)
    finally:
        source.stop()
    assert backend.closed is True


def test_camera_frame_source_exposure_range_fallback_on_raise() -> None:
    class _RaisingBackend(_DahengFakeBackend):
        def get_exposure_range(self) -> tuple[float, float]:
            raise RuntimeError("boom")

    backend = _RaisingBackend()
    source = CameraFrameSource(camera_type="daheng", camera=backend)
    source.start()
    try:
        # 实现若让异常传播, 此处会抛; 契约要求回退
        assert source.exposure_range_ms == (0.011, 10000.0)
    finally:
        source.stop()


def test_sim_beam_source_exposure_range() -> None:
    assert SimBeamSource().exposure_range_ms == (0.011, 10000.0)


# ---------------------------------------------------------------- UI
@pytest.fixture(scope="module")
def qt_app():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_daheng_selectable_via_source_combo(qt_app) -> None:
    from ao_shaping.gui.ccd.beam_quality.app import BeamQualityApp, SOURCE_KINDS

    window = BeamQualityApp()
    daheng_index = next(
        i for i, (_label, key) in enumerate(SOURCE_KINDS) if key == "daheng"
    )
    assert SOURCE_KINDS[daheng_index] == ("Daheng", "daheng")
    window._source_combo.setCurrentIndex(daheng_index)
    assert window._source_combo.currentIndex() == daheng_index
    window.close()
    qt_app.processEvents()


def test_exposure_spin_range_tracks_source(qt_app) -> None:
    from ao_shaping.gui.ccd.beam_quality.app import BeamQualityApp, SOURCE_KINDS

    window = BeamQualityApp()
    kinds = [key for _label, key in SOURCE_KINDS]
    assert "sim" in kinds
    window._source_combo.setCurrentIndex(kinds.index("sim"))
    window.open_source()
    assert _pump_until(qt_app, lambda: window._source is not None), "仿真源未打开"

    source = window._source
    assert (window._exposure_spin.minimum(), window._exposure_spin.maximum()) == (
        float(source.exposure_range_ms[0]),
        float(source.exposure_range_ms[1]),
    )

    window.close_source()
    assert _pump_until(qt_app, lambda: not window._busy), "关闭作业未完成"
    assert (window._exposure_spin.minimum(), window._exposure_spin.maximum()) == (
        0.011,
        10000.0,
    )

    window.close()
    qt_app.processEvents()
