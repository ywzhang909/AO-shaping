"""⑥ 实时画面: 逐帧 CCD 图 / X-Y 剖面 / 块进度 / 帧进度。

线程边界
--------
本模块**只**在主线程里碰控件。会话跑在 :class:`QThreadPool` 的后台线程里,
帧数据经 :class:`~ao_shaping.gui.slm.transfer_desktop.page._WorkerSignals.frame`
(一个 ``QObject`` 信号) 排队送到 :meth:`LiveView.apply_payload`; 回调本身在
后台线程里只做 ``emit(dict)``, 不 import 本模块、不碰任何 QWidget。

刷新节奏
--------
**每一次 CCD 读出都刷新一次** (负载由 ``live_frame.frame_payload`` 构造), 不再是
每块一张快照。CCD 的读出周期约 0.9 s (``session_plan.READ_SECONDS_PER_FRAME``),
即 ~1.1 帧/秒 —— 这仍然不是视频流。单次刷新的成本靠 :func:`display_preview`
先降采样再归一化, 从整幅的 ~220-337 ms 降到 ~16 ms, 所以逐帧刷新比原来的块级
快照还便宜; 测量线程每帧只多一次 ``ndarray.copy()`` (~1.3 ms, 帧周期的 0.15%)。
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QSize, Slot
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from ao_shaping.gui.ccd.beam_quality.profile_widget import ProfileWidget
from ao_shaping.gui.ccd.beam_quality.video_widget import VideoWidget

__all__ = ["LiveView", "display_frame", "display_preview"]

#: 大靶面相机的显示降采样步长: 4 ⇒ 分析尺寸 1/16, 坐标用 ``rescaled(4)`` 还原。
PREVIEW_STEP = 4
#: 只有最短边达到这个像素数才降采样。仿真源与测试里的小图必须原样分析:
#: 8x10 的帧降采样 4 倍只剩 2x3, 连剖面都画不出来。
PREVIEW_MIN_SIDE = 800


def display_frame(frame: np.ndarray) -> np.ndarray:
    """把任意灰度 raw 帧压成 0..255 的 uint8 供 ``VideoWidget`` 显示。

    SimPibCCD 返回 float 计数帧、大恒返回 uint8/uint16; 直接 ``clip`` 会让
    counts 口径下的一帧几乎全黑。减掉 1% 分位背景再按峰值归一, 只改显示对比度,
    不改剖面形状。
    """
    arr = np.asarray(frame, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    arr = np.clip(arr - float(np.percentile(arr, 1)), 0.0, None)
    peak = float(arr.max())
    if peak > 0:
        arr = arr * (255.0 / peak)
    return arr.astype(np.uint8)


def preview_step(shape) -> int:
    """这一帧该按几倍降采样 (1 = 不降采样)。"""
    if len(shape) < 2:
        return 1
    return PREVIEW_STEP if min(int(shape[0]), int(shape[1])) >= PREVIEW_MIN_SIDE else 1


def display_preview(frame) -> tuple[np.ndarray, int]:
    """把一帧 raw 压成 0..255 uint8, 大靶面先降采样。返回 ``(图像, 步长)``。

    归一化必须在**降采样之后**做: 在整幅 2592x1944 上算 1% 分位 (``np.percentile``)
    一帧就要 42 ms, 是这条路径上最贵的一步。先抽稀再算, 一次刷新从 ~220-337 ms
    (整幅归一化 + 整幅分析) 降到 ~16 ms。

    步长交给调用方喂给 :meth:`FrameMetrics.rescaled`, 因为剖面数组的长度是降采样后
    的长度, 而质心/椭圆/拟合 sigma 需要换算回原始像素。
    """
    arr = np.asarray(frame)
    if arr.ndim == 3:
        arr = arr[..., 0]
    step = preview_step(arr.shape)
    if step > 1:
        arr = arr[::step, ::step]
    return display_frame(arr), step


class LiveView(QWidget):
    """CCD 画面 + X/Y 剖面 + 块进度条 + 帧进度条。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._video = VideoWidget()
        self._video._label.setMinimumSize(QSize(320, 200))
        self._video.setMinimumSize(QSize(320, 200))
        self._profile = ProfileWidget()
        self._profile.setMinimumHeight(140)

        self._info = QLabel("等待第一块采集完成…")
        self._info.setWordWrap(True)

        self._block_bar = QProgressBar()
        self._block_bar.setRange(0, 1)
        self._block_bar.setValue(0)
        self._block_bar.setFormat("块 %v / %m")
        self._block_label = QLabel("当前块 0 / 0")

        self._remaining_bar = QProgressBar()
        self._remaining_bar.setRange(0, 1)
        self._remaining_bar.setValue(0)
        self._remaining_bar.setFormat("距下次刷新还剩 %v iter")

        frames = QHBoxLayout()
        frames.setSpacing(8)
        frames.addWidget(self._video, 1)
        frames.addWidget(self._profile, 1)

        bars = QHBoxLayout()
        bars.setSpacing(10)
        bars.addWidget(self._block_label, 0)
        bars.addWidget(self._block_bar, 1)
        bars.addWidget(self._remaining_bar, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._info)
        layout.addLayout(frames, 1)
        layout.addLayout(bars, 0)

        self._last_payload: dict | None = None

    # ------------------------------------------------------------------ 接口
    @Slot(dict)
    def apply_payload(self, payload: dict) -> None:
        """主线程槽: 一次实时刷新, 更新画面/剖面/两根进度条。"""
        if not isinstance(payload, dict):
            return
        frame = payload.get("frame")
        if frame is None:
            return
        from ao_shaping.gui.ccd.beam_quality.analyzer import analyze_frame

        # 先降采样再归一化/分析 (2592x1944 → 648x486), 步长换回原始像素坐标。
        shown, step = display_preview(frame)
        metrics = analyze_frame(shown)
        if step != 1:
            metrics = metrics.rescaled(step)
        self._video.set_frame(shown, metrics)
        self._profile.update_metrics(metrics)

        blocks_total = int(payload.get("blocks_total", 0) or 0)
        block_index = int(payload.get("block_index", 0) or 0)
        iterations_total = int(payload.get("iterations_total", 0) or 0)
        iteration = int(payload.get("iteration", 0) or 0)
        frames_total = int(payload.get("frames_total", 0) or 0)
        frame_index = int(payload.get("frame_index", 0) or 0)

        # 第一根条: 锚定块的块进度; 采集段没有块, 直接标阶段名。
        if blocks_total > 0:
            self._block_bar.setRange(0, blocks_total)
            self._block_bar.setFormat("块 %v / %m")
            self._block_bar.setValue(min(max(block_index, 0), blocks_total))
            self._block_label.setText(
                f"当前块 {block_index} / {blocks_total}"
                f"（形状 {payload.get('shape', '?')}）"
            )
        else:
            self._block_bar.setRange(0, 1)
            self._block_bar.setFormat("采集段")
            self._block_bar.setValue(1)
            self._block_label.setText(
                f"当前阶段：{'曝光标定' if payload.get('phase') == 'exposure' else '采集'}")

        # 第二根条: 这次采集的第几帧。刷新已经是逐帧的, 所以它不再是倒计时。
        if frames_total > 0:
            self._remaining_bar.setRange(0, frames_total)
            self._remaining_bar.setFormat("帧 %v / %m")
            self._remaining_bar.setValue(min(max(frame_index, 0), frames_total))
        else:
            self._remaining_bar.setRange(0, 1)
            self._remaining_bar.setFormat(f"第 {max(frame_index, 1)} 帧")
            self._remaining_bar.setValue(1)

        detail = []
        if blocks_total > 0:
            detail.append(f"块内 iter {iteration}/{iterations_total}")
        detail.append(f"帧 {frame_index}/{frames_total}" if frames_total > 0
                      else f"第 {max(frame_index, 1)} 帧")
        label = payload.get("label", "") or "实时帧"
        self._info.setText(f"实时画面：{label} · " + " · ".join(detail))
        self._last_payload = payload

    def clear(self) -> None:
        """会话开始前/结束后回到空态 (不改动 ⑤ 进度表)。"""
        self._video.clear()
        self._profile.clear()
        self._block_bar.setRange(0, 1)
        self._block_bar.setValue(0)
        self._remaining_bar.setRange(0, 1)
        self._remaining_bar.setValue(0)
        self._block_label.setText("当前块 0 / 0")
        self._info.setText("等待第一帧 CCD 图像…")
        self._last_payload = None

    # 只读访问器, 供离屏测试/截图自检断言, 不暴露可写控件
    def last_payload(self) -> dict | None:
        return self._last_payload

    def block_bar(self) -> QProgressBar:
        return self._block_bar

    def remaining_bar(self) -> QProgressBar:
        return self._remaining_bar

    def video(self) -> VideoWidget:
        return self._video

    def profile(self) -> ProfileWidget:
        return self._profile

    def message(self) -> str:
        return self._info.text()
