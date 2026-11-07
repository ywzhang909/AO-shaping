"""X/Y 过中心剖面 + 一维高斯拟合曲线。

上半部分/左右两个 ``QChart`` 各画一条剖面: 散点是实测数据, 虚线是高斯拟合,
图下方标签给出 sigma、中心与拟合优度 (R^2), 拟合失败时明确写"拟合失败"。
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QScatterSeries, QValueAxis
from PySide6.QtCore import QMargins, QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ao_shaping.gui.ccd.beam_quality.analyzer import (
    FrameMetrics,
    GaussianFit,
    gaussian_1d,
)

# 单条剖面最多画多少个散点 (2688 像素全画没必要, 且拖动会卡)
MAX_POINTS = 1200
# 拟合曲线采样点数
FIT_SAMPLES = 400


class _ProfilePanel(QWidget):
    """一条剖面 (X 或 Y) 的图 + 拟合标签。"""

    def __init__(
        self, title: str, color: QColor, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._data = QScatterSeries()
        self._data.setName("数据")
        self._data.setMarkerSize(4.0)
        self._data.setColor(color)

        self._fit = QLineSeries()
        self._fit.setName("高斯拟合")
        pen = QPen(QColor(255, 170, 60))
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setWidthF(1.6)
        self._fit.setPen(pen)

        self._axis_x = QValueAxis()
        self._axis_x.setTitleText("像素")
        self._axis_x.setLabelFormat("%.0f")
        self._axis_y = QValueAxis()
        self._axis_y.setTitleText("灰度")
        self._axis_y.setLabelFormat("%.1f")

        chart = QChart()
        chart.addSeries(self._data)
        chart.addSeries(self._fit)
        chart.addAxis(self._axis_x, Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(self._axis_y, Qt.AlignmentFlag.AlignLeft)
        self._data.attachAxis(self._axis_x)
        self._data.attachAxis(self._axis_y)
        self._fit.attachAxis(self._axis_x)
        self._fit.attachAxis(self._axis_y)
        chart.setTitle(title)
        chart.legend().setVisible(True)
        chart.setMargins(QMargins(2, 2, 2, 2))
        chart.setAnimationOptions(QChart.AnimationOption.NoAnimation)

        self._view = QChartView(chart)
        self._view.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._view.setMinimumHeight(150)

        self._label = QLabel("σ = --    中心 = --")
        self._label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self._view, 1)
        layout.addWidget(self._label, 0)

    # ------------------------------------------------------------------ 更新
    def update_profile(
        self,
        profile: np.ndarray,
        fit: GaussianFit | None,
        pixel_scale: float = 1.0,
    ) -> None:
        """刷新数据点、拟合曲线与标签。"""
        values = np.asarray(profile, dtype=np.float64)
        count = int(values.size)
        if count == 0:
            self.clear()
            return
        step = max(1, int(np.ceil(count / MAX_POINTS)))
        index = np.arange(0, count, step, dtype=np.float64) * float(pixel_scale)
        picked = values[::step]

        self._data.replace([QPointF(float(x), float(y)) for x, y in zip(index, picked)])

        x_min = float(index[0]) if index.size else 0.0
        x_max = float(index[-1]) if index.size else 1.0
        if x_max <= x_min:
            x_max = x_min + 1.0

        y_max = float(np.max(values))
        y_min = float(np.min(values))
        pad = max((y_max - y_min) * 0.08, 1.0)
        # 灰度非负: 下界钳到 0, 避免背景扣到 0 的剖面把坐标轴拖进负值区
        self._axis_y.setRange(max(y_min - pad, 0.0), y_max + pad)

        if fit is not None and fit.sigma > 0.0:
            grid = np.linspace(x_min, x_max, FIT_SAMPLES)
            curve = gaussian_1d(grid, fit.amplitude, fit.center, fit.sigma, fit.offset)
            self._fit.replace(
                [QPointF(float(x), float(y)) for x, y in zip(grid, curve)]
            )
            self._axis_x.setRange(x_min, x_max)
            # fit 的 center/sigma 已由 FrameMetrics.rescaled() 换算回原始像素
            self._label.setText(
                f"σ = {fit.sigma:.2f} px    "
                f"FWHM = {fit.fwhm:.2f} px    "
                f"中心 = {fit.center:.1f} px    "
                f"R² = {fit.r_squared:.4f}"
            )
        else:
            self._fit.clear()
            self._axis_x.setRange(x_min, x_max)
            self._label.setText("高斯拟合失败 (剖面非单峰或信噪比不足)")

    def clear(self) -> None:
        self._data.clear()
        self._fit.clear()
        self._label.setText("σ = --    中心 = --")


class ProfileWidget(QWidget):
    """X / Y 两条剖面的并列图。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._x_panel = _ProfilePanel("X 剖面", QColor(90, 190, 255))
        self._y_panel = _ProfilePanel("Y 剖面", QColor(120, 230, 150))
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._x_panel)
        layout.addWidget(self._y_panel)

    def update_metrics(self, metrics: FrameMetrics | None) -> None:
        """按最新分析结果刷新两条剖面。"""
        if metrics is None:
            self.clear()
            return
        self._x_panel.update_profile(
            metrics.x_profile, metrics.x_fit, metrics.pixel_scale
        )
        self._y_panel.update_profile(
            metrics.y_profile, metrics.y_fit, metrics.pixel_scale
        )

    def clear(self) -> None:
        self._x_panel.clear()
        self._y_panel.clear()
