"""视频显示部件: 单次构造的灰度 QImage + 缩放 pixmap 上叠加质心/椭圆/信息条。

关键点 (针对"显示延迟/撕裂"):

* 每帧只构造一次 ``QImage`` (直接引用 numpy 缓冲区, 由 ``self._array`` 保活),
  一次 ``scaled()`` 得到目标尺寸的副本, 再 ``QPixmap.fromImage`` —— 不出现
  先画一半再缩放的两步显示;
* 叠加绘制全部发生在缩放后的 pixmap 上 (坐标按缩放比换算), 与显示尺寸无关;
* 保持宽高比, 不做插值 (``FastTransformation``): 像素级画面不该被平滑掉。
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ao_shaping.gui.ccd.beam_quality.analyzer import FrameMetrics

# 叠加层配色 (深色画面上足够醒目)
_CROSS_COLOR = QColor(80, 255, 140)
_ELLIPSE_COLOR = QColor(80, 200, 255)
_INFO_BG = QColor(0, 0, 0, 150)
_INFO_FG = QColor(225, 225, 235)


def to_uint8(img: np.ndarray) -> np.ndarray:
    """把任意灰度数组安全地转成连续 uint8 视图 (供 QImage 直接引用)。"""
    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


class VideoWidget(QWidget):
    """相机画面 + 分析叠加层。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._label = QLabel("无画面")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setMinimumSize(QSize(480, 320))
        self._label.setStyleSheet("background-color: #12121a; color: #8a8a96;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)

        self._array: np.ndarray | None = None
        self._metrics: FrameMetrics | None = None
        self._fps: float = 0.0

    # ------------------------------------------------------------------ 接口
    def set_frame(
        self,
        img: np.ndarray,
        metrics: FrameMetrics | None = None,
        fps: float = 0.0,
    ) -> None:
        """显示一帧并刷新叠加层。"""
        self._array = to_uint8(img)
        self._metrics = metrics
        self._fps = float(fps)
        self._render()

    def clear(self) -> None:
        """清空画面。"""
        self._array = None
        self._metrics = None
        self._label.clear()
        self._label.setText("无画面")

    # ------------------------------------------------------------------ 渲染
    def resizeEvent(self, event) -> None:  # Qt 覆写点 (驼峰命名由 Qt 决定)
        super().resizeEvent(event)
        if self._array is not None:
            self._render()

    def _render(self) -> None:
        arr = self._array
        if arr is None or arr.size == 0:
            return
        height, width = int(arr.shape[0]), int(arr.shape[1])
        image = QImage(
            arr.data,
            width,
            height,
            int(arr.strides[0]),
            QImage.Format.Format_Grayscale8,
        )
        target = self._label.size()
        if target.width() < 2 or target.height() < 2:
            target = QSize(width, height)
        scaled = image.scaled(
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        pixmap = QPixmap.fromImage(scaled)
        painter = QPainter(pixmap)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            scale = scaled.width() / width if width else 1.0
            self._paint_overlay(painter, scale, scaled.width())
        finally:
            painter.end()
        self._label.setPixmap(pixmap)

    def _paint_overlay(self, painter: QPainter, scale: float, out_width: int) -> None:
        metrics = self._metrics
        if metrics is not None:
            painter.setPen(QPen(_CROSS_COLOR, 1.4))
            cx = metrics.cx * scale
            cy = metrics.cy * scale
            arm = 16.0
            painter.drawLine(QPointF(cx - arm, cy), QPointF(cx + arm, cy))
            painter.drawLine(QPointF(cx, cy - arm), QPointF(cx, cy + arm))

            ellipse = metrics.ellipse
            if ellipse is not None and ellipse.semi_major > 0.5:
                painter.setPen(QPen(_ELLIPSE_COLOR, 1.4))
                painter.save()
                painter.translate(ellipse.center[0] * scale, ellipse.center[1] * scale)
                painter.rotate(ellipse.angle_deg)
                a = ellipse.semi_major * scale
                b = ellipse.semi_minor * scale
                painter.drawEllipse(QRectF(-a, -b, 2.0 * a, 2.0 * b))
                painter.restore()

            info = (
                f"fps {self._fps:5.1f}   max {metrics.max_value:5.1f}   "
                f"mean {metrics.mean_value:6.2f}   sat {metrics.sat_ratio * 100:5.2f}%"
            )
        else:
            info = f"fps {self._fps:5.1f}"

        font = QFont()
        font.setPointSize(9)
        painter.setFont(font)
        bar_height = 20
        bar_width = float(max(out_width, 10))
        painter.fillRect(QRectF(0, 0, bar_width, bar_height), _INFO_BG)
        painter.setPen(QPen(_INFO_FG))
        painter.drawText(
            QRectF(6, 0, max(bar_width - 6.0, 10.0), bar_height),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            info,
        )
