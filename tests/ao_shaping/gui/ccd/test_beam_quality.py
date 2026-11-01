"""光束质量分析 (PySide6) 的离线测试: 算法 / 缓冲 / 帧源 / 界面冒烟。

全部用例都不需要硬件: 分析是纯 numpy/scipy, 帧源用 :class:`SimBeamSource`,
界面在 ``QT_QPA_PLATFORM=offscreen`` 下实例化。
"""

from __future__ import annotations

import csv
import os
import time

# 必须在任何 QApplication 之前设置: CI / 无显示环境下用离屏后端
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from ao_shaping.gui.ccd.beam_quality import frame_source as frame_source_module
from ao_shaping.gui.ccd.beam_quality.analyzer import (
    FWHM_TO_SIGMA,
    analyze_frame,
    bin_2x2,
)
from ao_shaping.gui.ccd.beam_quality.frame_source import SimBeamSource
from ao_shaping.gui.ccd.beam_quality.trend_buffer import (
    CSV_HEADER,
    SCALAR_KEYS,
    TrendBuffer,
)


def _gaussian_frame(
    width: int = 260,
    height: int = 300,
    cx: float = 120.4,
    cy: float = 150.7,
    sigma_x: float = 18.0,
    sigma_y: float = 11.0,
    amplitude: float = 180.0,
    background: float = 5.0,
    angle_deg: float = 0.0,
    noise: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """合成 (可旋转的) 高斯光斑, 供算法用例使用。"""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    dx = xx - cx
    dy = yy - cy
    if angle_deg:
        phi = np.deg2rad(angle_deg)
        u = dx * np.cos(phi) + dy * np.sin(phi)
        v = -dx * np.sin(phi) + dy * np.cos(phi)
    else:
        u, v = dx, dy
    beam = amplitude * np.exp(
        -(u**2) / (2.0 * sigma_x**2) - (v**2) / (2.0 * sigma_y**2)
    )
    frame = background + beam
    if noise:
        frame = frame + np.random.default_rng(seed).normal(0.0, noise, size=frame.shape)
    return np.clip(frame, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------- 算法
def test_analyze_frame_centroid_and_gaussian() -> None:
    """质心误差 < 1 px, 椭圆轴长 < 10%, 高斯 sigma 回收 < 10%。"""
    sigma_x, sigma_y = 18.0, 11.0
    cx, cy = 120.4, 150.7
    img = _gaussian_frame(cx=cx, cy=cy, sigma_x=sigma_x, sigma_y=sigma_y, noise=0.8)

    metrics = analyze_frame(img)

    assert abs(metrics.cx - cx) < 1.0
    assert abs(metrics.cy - cy) < 1.0
    assert metrics.max_value == pytest.approx(185.0, abs=3.0)
    assert metrics.sat_ratio == 0.0
    assert metrics.ellipse is not None

    # 半高阈值下的全宽 = 2.3548 * sigma
    assert metrics.ellipse.fwhm_major == pytest.approx(
        FWHM_TO_SIGMA * sigma_x, rel=0.10
    )
    assert metrics.ellipse.fwhm_minor == pytest.approx(
        FWHM_TO_SIGMA * sigma_y, rel=0.10
    )
    assert abs(metrics.ellipse.angle_deg) < 3.0

    assert metrics.x_fit is not None
    assert metrics.y_fit is not None
    assert metrics.x_fit.sigma == pytest.approx(sigma_x, rel=0.10)
    assert metrics.y_fit.sigma == pytest.approx(sigma_y, rel=0.10)
    assert metrics.x_fit.center == pytest.approx(cx, abs=1.0)
    assert metrics.y_fit.center == pytest.approx(cy, abs=1.0)
    # 剖面长度等于图宽/图高
    assert metrics.x_profile.size == img.shape[1]
    assert metrics.y_profile.size == img.shape[0]


def test_analyze_frame_rotated_ellipse_angle() -> None:
    """旋转 30 度的高斯: 椭圆倾角应≈30 度, 长短轴对应旋转后的 sigma。"""
    angle = 30.0
    sigma_major, sigma_minor = 20.0, 10.0
    img = _gaussian_frame(sigma_x=sigma_major, sigma_y=sigma_minor, angle_deg=angle)

    metrics = analyze_frame(img)

    assert metrics.ellipse is not None
    assert metrics.ellipse.angle_deg == pytest.approx(angle, abs=5.0)
    assert metrics.ellipse.fwhm_major == pytest.approx(
        FWHM_TO_SIGMA * sigma_major, rel=0.10
    )
    assert metrics.ellipse.fwhm_minor == pytest.approx(
        FWHM_TO_SIGMA * sigma_minor, rel=0.10
    )


def test_analyze_frame_blank_and_saturated() -> None:
    """全黑帧不该抛异常; 全饱和帧的饱和占比为 1。"""
    blank = np.zeros((40, 50), dtype=np.uint8)
    metrics = analyze_frame(blank)
    assert metrics.ellipse is None
    assert np.isfinite(metrics.cx) and np.isfinite(metrics.cy)
    assert metrics.x_fit is None or np.isfinite(metrics.x_fit.sigma)

    saturated = np.full((32, 32), 255, dtype=np.uint8)
    assert analyze_frame(saturated).sat_ratio == pytest.approx(1.0)


def test_analyze_frame_scalars_and_rescale() -> None:
    """趋势缓冲要的标量字段齐全; binning 后坐标可换算回原始像素。"""
    img = _gaussian_frame()
    metrics = analyze_frame(img)
    scalars = metrics.as_scalars()
    assert set(scalars) == set(SCALAR_KEYS)

    binned = bin_2x2(img)
    assert binned.shape == (img.shape[0] // 2, img.shape[1] // 2)
    binned_metrics = analyze_frame(binned).rescaled(2.0)
    assert binned_metrics.cx == pytest.approx(metrics.cx, abs=1.5)
    assert binned_metrics.cy == pytest.approx(metrics.cy, abs=1.5)
    assert binned_metrics.pixel_scale == 2.0
    assert binned_metrics.width == img.shape[1]
    assert binned_metrics.ellipse is not None
    assert binned_metrics.ellipse.fwhm_major == pytest.approx(
        metrics.ellipse.fwhm_major, rel=0.10
    )


# ------------------------------------------------------------------ 趋势缓冲
def _fake_scalars(value: float) -> dict[str, float]:
    return {
        "max": value,
        "mean": value / 10.0,
        "std": 1.0,
        "cx": 100.0,
        "cy": 200.0,
        "sat_ratio": 0.0,
    }


def test_trend_buffer_append_pause_clear(tmp_path) -> None:
    buffer = TrendBuffer(maxlen=10)
    assert len(buffer) == 0

    for index in range(3):
        sample = buffer.append(1000.0 + index, _fake_scalars(10.0 * index))
        assert sample is not None
    assert len(buffer) == 3
    assert buffer.t0 == pytest.approx(1000.0)
    assert buffer.latest is not None and buffer.latest.max == pytest.approx(20.0)

    # 暂停: 不记录, 已有数据保留
    buffer.paused = True
    assert buffer.append(1003.0, _fake_scalars(99.0)) is None
    assert len(buffer) == 3
    buffer.paused = False
    assert buffer.append(1004.0, _fake_scalars(40.0)) is not None
    assert len(buffer) == 4

    # 时间窗 (相对最新样本); 1003 那一秒被暂停吞掉, 因此只剩 2.0 与 4.0
    recent = buffer.records(2.0)
    assert [round(buffer.relative_time(s), 1) for s in recent] == [2.0, 4.0]
    assert len(buffer.records(None)) == 4

    # 环形: maxlen 生效
    for index in range(20):
        buffer.append(2000.0 + index, _fake_scalars(float(index)))
    assert len(buffer) == 10

    buffer.clear()
    assert len(buffer) == 0
    assert buffer.t0 is None
    assert buffer.records(30.0) == []


def test_trend_buffer_export_csv(tmp_path) -> None:
    buffer = TrendBuffer()
    for index in range(5):
        buffer.append(time.time() + index * 0.1, _fake_scalars(float(index)))

    target = tmp_path / "nested" / "trend.csv"
    written = buffer.export_csv(target)
    assert written == target and target.exists()

    with target.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert tuple(rows[0]) == CSV_HEADER
    assert len(rows) == 6  # 表头 + 5 条
    assert float(rows[-1][2]) == pytest.approx(4.0)  # max 列
    assert float(rows[-1][1]) == pytest.approx(0.4)  # t_rel_s 列


# -------------------------------------------------------------------- 帧源
def test_sim_source_produces_frames_within_two_seconds() -> None:
    source = SimBeamSource(width=160, height=120, fps=40.0, seed=1)
    assert source.is_running is False
    source.start()
    try:
        frame = None
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and frame is None:
            frame = source.drain_latest()
            time.sleep(0.005)
        assert frame is not None, "2 秒内没有拿到仿真帧"
        img, timestamp = frame
        assert img.shape == (120, 160)
        assert img.dtype == np.uint8
        assert isinstance(timestamp, float)
        assert source.is_running is True
        # 再取几帧以让帧率 EMA 收敛
        for _ in range(50):
            source.drain_latest()
            time.sleep(0.005)
        assert source.fps > 1.0
    finally:
        source.stop()
    assert source.is_running is False
    # 幂等
    source.stop()


def test_miicam_import_is_lazy() -> None:
    """MIICamera 不得在 frame_source 模块导入期被拉进来 (无 SDK 机器也要能导入)。"""
    assert "frame_source" in frame_source_module.__name__
    assert not hasattr(frame_source_module, "MIICamera")
    assert frame_source_module.MIICamFrameSource is not None
    # 构造函数不碰相机: 只记录参数
    source = frame_source_module.MIICamFrameSource(cam_id=3, exposure_ms=5.0)
    assert source.is_running is False
    assert source.exposure_ms == pytest.approx(5.0)
    assert source.cam_id == 3
    assert source.drain_latest() is None
    source.stop()  # 未打开时 stop 必须幂等且不抛


# ---------------------------------------------------------------- 界面冒烟
@pytest.fixture(scope="module")
def qt_app():
    """离屏 QApplication (一个进程只能有一个)。"""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_app_smoke_with_sim_source(qt_app) -> None:
    """离屏实例化主窗口 + 仿真源跑一会儿 + 关闭不崩、不留运行中的源。"""
    from ao_shaping.gui.ccd.beam_quality.app import BeamQualityApp, SOURCE_KINDS

    window = BeamQualityApp()
    window.show()
    kinds = [kind for _label, kind in SOURCE_KINDS]
    window._source_combo.setCurrentIndex(kinds.index("sim"))
    window.open_source()
    # open_source 是异步的 (后台线程执行硬件打开), 泵事件等 done 回调落地
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and window._source is None:
        qt_app.processEvents()
        time.sleep(0.01)
    assert window._source is not None

    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and window._processed == 0:
        qt_app.processEvents()
        time.sleep(0.01)
    assert window._processed > 0, "3 秒内界面没有处理任何一帧"
    assert window._last_metrics is not None

    # 趋势缓冲在同一段时间里也应收到样本 (10 Hz 节流)
    assert len(window._buffer) > 0
    window._trend.update_buffer()
    window._trend.refresh()

    window.close()
    qt_app.processEvents()
    assert window._source is None


def test_apply_exposure_does_not_block_ui_thread(qt_app) -> None:
    """应用曝光必须在后台线程执行 (真机 MiiCam 改曝光 ~10s),
    同步调用会把整个窗口冻住 (2026-10-31 真机实测)。"""
    import threading

    from ao_shaping.gui.ccd.beam_quality.app import BeamQualityApp

    window = BeamQualityApp()
    gate = threading.Event()

    class _SlowSource:
        is_running = True
        fps = 1.0
        exposure_ms = 20.0
        last_frame_info = None

        def set_exposure_ms(self, ms: float) -> float:
            gate.wait(5.0)  # 模拟硬件阻塞
            return ms

    window._source = _SlowSource()  # type: ignore[assignment]
    window._exposure_spin.setValue(5.0)

    t0 = time.monotonic()
    window.apply_exposure()
    assert time.monotonic() - t0 < 0.5, "apply_exposure 必须立即返回"
    assert window._busy

    gate.set()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and window._busy:
        qt_app.processEvents()
        time.sleep(0.01)
    assert not window._busy, "后台任务应在放行后完成"
    assert window._exposure_spin.value() == 5.0
    window._source = None
    window.close()


# ------------------------------------------------- 通用相机帧源 (注入假后端)
class _FakeBackend:
    """假 BaseCamera: 记录 open/start/close, ``start_stream`` 里立刻推 2 帧。"""

    def __init__(self) -> None:
        self.opened = False
        self.closed = False
        self.streaming = False
        self.exposure_ms = 20.0
        self.exposure_requests: list[float] = []
        self.frames_pushed = 0
        self.last_frame_info = None

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

    def set_exposure_ms(self, ms: float) -> float:
        self.exposure_requests.append(float(ms))
        self.exposure_ms = float(ms)
        return self.exposure_ms

    def get_exposure_time_ms(self) -> float:
        return self.exposure_ms


def test_camera_frame_source_uses_injected_backend() -> None:
    """构造不碰硬件; start 走 open→start_stream, 信箱只留最新帧, stop 关闭设备。"""
    from ao_shaping.gui.ccd.beam_quality.frame_source import CameraFrameSource

    backend = _FakeBackend()
    source = CameraFrameSource(camera_type="daheng", cam_id=2, exposure_ms=3.0, camera=backend)
    assert source.is_running is False
    assert backend.opened is False, "构造期不得打开设备"

    forwarded: list[tuple[int, int]] = []
    source.start(on_frame=lambda img, _ts: forwarded.append(img.shape))
    try:
        assert backend.opened and backend.streaming
        assert source.is_running is True
        assert forwarded == [(4, 4), (4, 4)], "on_frame 必须逐帧转发"

        drained = source.drain_latest()
        assert drained is not None
        img, _ts = drained
        assert int(img[0, 0]) == 9, "信箱只保留最新帧"
        assert source.drain_latest() is None

        assert source.set_exposure_ms(5.5) == pytest.approx(5.5)
        assert backend.exposure_requests == pytest.approx([5.5])
        assert source.exposure_ms == pytest.approx(5.5)
    finally:
        source.stop()

    assert source.is_running is False
    assert backend.closed is True
    assert backend.streaming is False
    source.stop()  # 幂等


def test_camera_frame_source_closes_backend_when_stream_start_fails() -> None:
    """起流失败必须先把设备关掉再抛, 否则下一次 open 会撞 REPEAT_OPENED。"""
    from ao_shaping.gui.ccd.beam_quality.frame_source import CameraFrameSource

    class _BadStream(_FakeBackend):
        def start_stream(self, on_frame) -> None:
            raise RuntimeError("no frames")

    backend = _BadStream()
    source = CameraFrameSource(camera_type="daheng", camera=backend)
    with pytest.raises(RuntimeError):
        source.start()
    assert source.is_running is False
    assert backend.closed is True


def test_camera_frame_source_keeps_backward_compatible_alias() -> None:
    from ao_shaping.gui.ccd.beam_quality.frame_source import (
        CameraFrameSource,
        MIICamFrameSource,
    )

    assert MIICamFrameSource is CameraFrameSource
