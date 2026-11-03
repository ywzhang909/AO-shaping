"""⑥ 实时画面: 每块最后一帧的 CCD 图 / X-Y 剖面 / 块进度 / 下次刷新倒计时。

线程边界
--------
本模块**只**在主线程里碰控件。会话跑在 :class:`QThreadPool` 的后台线程里,
帧数据经 :class:`~ao_shaping.gui.slm.transfer_desktop.page._WorkerSignals.frame`
(一个 ``QObject`` 信号) 排队送到 :meth:`LiveView.apply_payload`; 回调本身在
后台线程里只做 ``emit(dict)``, 不 import 本模块、不碰任何 QWidget。

刷新节奏
--------
``validate.emit_block_frame`` 每个锚定块只发一次: 该块验收/锚定采集的**最后一帧**。
所以画面是块级快照, 不是逐 iter 视频流 —— 逐 iter 回调会把 60 次迭代 * 全幅帧的
拷贝与 GUI 重绘压进采集关键路径, 而验收口径本来也只按块结算。
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

__all__ = ["LiveView"]


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


class LiveView(QWidget):
    """CCD 画面 + X/Y 剖面 + 块进度条 + 「距下次刷新还剩 N iter」进度条。"""

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
        """主线程槽: 一次块刷新, 更新画面/剖面/两根进度条。"""
        if not isinstance(payload, dict):
            return
        frame = payload.get("frame")
        if frame is None:
            return
        from ao_shaping.gui.ccd.beam_quality.analyzer import analyze_frame

        shown = display_frame(frame)
        metrics = analyze_frame(shown)
        self._video.set_frame(shown, metrics)
        self._profile.update_metrics(metrics)

        blocks_total = max(1, int(payload.get("blocks_total", 1)))
        block_index = int(payload.get("block_index", 0))
        iterations_total = max(1, int(payload.get("iterations_total", 1)))
        remaining = int(payload.get("remaining", 0))
        self._block_bar.setRange(0, blocks_total)
        self._block_bar.setValue(min(block_index, blocks_total))
        self._block_label.setText(
            f"当前块 {block_index} / {blocks_total}（形状 {payload.get('shape', '?')}）"
        )
        self._remaining_bar.setRange(0, iterations_total)
        self._remaining_bar.setValue(min(max(remaining, 0), iterations_total))
        self._info.setText(
            f"实时画面：{payload.get('label', '') or '本块最后一帧'} · "
            f"块内 iter {int(payload.get('iteration', 0))}/{iterations_total} · "
            f"距下次刷新还剩 {remaining} iter"
        )
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
        self._info.setText("等待第一块采集完成…")
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
