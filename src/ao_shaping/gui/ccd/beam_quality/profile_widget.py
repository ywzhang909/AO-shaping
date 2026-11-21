"""X/Y 过中心剖面 + 一维高斯拟合曲线。

上半部分/左右两个 ``QChart`` 各画一条剖面: 散点是实测数据, 虚线是高斯拟合,
图下方是四格读数 (σ / FWHM / 中心 / R²); 拟合失败时四格留 "—", 下面一行写原因。
整个组件顶部有一行"显示高斯拟合曲线"开关, 只切虚线, 不影响散点与读数。
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QScatterSeries, QValueAxis
from PySide6.QtCore import QMargins, QPointF, Qt
from PySide6.QtGui import QColor, QFontDatabase, QPainter, QPen
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ao_shaping.gui.ccd.beam_quality.analyzer import (
    FrameMetrics,
    GaussianFit,
    gaussian_1d,
)

# 单条剖面最多画多少个散点 (2688 像素全画没必要, 且拖动会卡)
MAX_POINTS = 1200
# 拟合曲线采样点数
FIT_SAMPLES = 400

#: 拟合读数格: (键, 格标题)。原先是一整行 "σ = … FWHM = … 中心 = … R² = …",
#: 那条单行文字把面板最小宽度撑到 1536 px (两面板并排后整窗被迫加宽到 2254 px,
#: 在 1560 宽的外壳里会被裁), 现在拆成四格 "小标题 + 数值"。
_FIT_CELLS: tuple[tuple[str, str], ...] = (
    ("sigma", "σ"),
    ("fwhm", "FWHM"),
    ("center", "中心"),
    ("r2", "R²"),
)
_FIT_PLACEHOLDER = "—"
_MUTED_STYLE = "color: #808088;"


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

        self._chart = QChart()
        self._chart.addSeries(self._data)
        self._chart.addSeries(self._fit)
        self._chart.addAxis(self._axis_x, Qt.AlignmentFlag.AlignBottom)
        self._chart.addAxis(self._axis_y, Qt.AlignmentFlag.AlignLeft)
        self._data.attachAxis(self._axis_x)
        self._data.attachAxis(self._axis_y)
        self._fit.attachAxis(self._axis_x)
        self._fit.attachAxis(self._axis_y)
        self._chart.setTitle(title)
        self._chart.legend().setVisible(True)
        self._chart.setMargins(QMargins(2, 2, 2, 2))
        self._chart.setAnimationOptions(QChart.AnimationOption.NoAnimation)

        self._view = QChartView(self._chart)
        self._view.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._view.setMinimumHeight(150)

        # 四格读数: 格标题在上 (说明), 数值在下 (等宽数字, 纵向对齐)
        mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        self._fit_values: dict[str, QLabel] = {}
        metrics_row = QHBoxLayout()
        metrics_row.setContentsMargins(0, 0, 0, 0)
        metrics_row.setSpacing(12)
        for key, caption in _FIT_CELLS:
            cell = QVBoxLayout()
            cell.setContentsMargins(0, 0, 0, 0)
            cell.setSpacing(0)
            cap = QLabel(caption)
            cap.setStyleSheet(_MUTED_STYLE)
            value = QLabel(_FIT_PLACEHOLDER)
            value.setFont(mono)
            value.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            cell.addWidget(cap)
            cell.addWidget(value)
            metrics_row.addLayout(cell)
            self._fit_values[key] = value
        metrics_row.addStretch(1)
        # 拟合失败是状态而不是读数: 单独一行提示, 正常时留空 (高度固定, 不跳动)
        self._note = QLabel("")
        self._note.setWordWrap(True)
        self._note.setStyleSheet("color: #d08a3a;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self._view, 1)
        layout.addLayout(metrics_row, 0)
        layout.addWidget(self._note, 0)

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
            self._fit_values["sigma"].setText(f"{fit.sigma:.2f} px")
            self._fit_values["fwhm"].setText(f"{fit.fwhm:.2f} px")
            self._fit_values["center"].setText(f"{fit.center:.1f} px")
            self._fit_values["r2"].setText(f"{fit.r_squared:.4f}")
            self._note.clear()
        else:
            self._fit.clear()
            self._axis_x.setRange(x_min, x_max)
            for value in self._fit_values.values():
                value.setText(_FIT_PLACEHOLDER)
            self._note.setText("高斯拟合失败 (剖面非单峰或信噪比不足)")

    def clear(self) -> None:
        self._data.clear()
        self._fit.clear()
        for value in self._fit_values.values():
            value.setText(_FIT_PLACEHOLDER)
        self._note.clear()

    def set_fit_visible(self, visible: bool) -> None:
        """只切高斯拟合虚线的可见性。

        散点、坐标轴、四格读数和失败提示都是实测结果, 一律不受影响; 曲线数据
        留在 series 里, 重新显示时直接恢复最近一次拟合, 不用等下一帧。
        """
        visible = bool(visible)
        self._fit.setVisible(visible)
        # 图例项不跟随 series 可见性, 需单独同步, 否则关掉曲线后图例还挂着"高斯拟合"
        for marker in self._chart.legend().markers(self._fit):
            marker.setVisible(visible)

    def fit_visible(self) -> bool:
        """拟合虚线当前是否可见 (与散点是否为空无关)。"""
        return self._fit.isVisible()


class ProfileWidget(QWidget):
    """X / Y 两条剖面的并列图, 顶部一行放"显示高斯拟合曲线"开关。

    开关只控制两条虚线; 散点、坐标轴、σ / FWHM / 中心 / R² 读数与拟合失败
    提示始终可见。默认勾选 (与加开关之前的表现一致)。
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._x_panel = _ProfilePanel("X 剖面", QColor(90, 190, 255))
        self._y_panel = _ProfilePanel("Y 剖面", QColor(120, 230, 150))

        # 显式保留控件对象: 离屏测试和调用方可以直接点它
        self.fit_toggle = QCheckBox("显示高斯拟合曲线")
        self.fit_toggle.setChecked(True)
        self.fit_toggle.setToolTip(
            "虚线是按当前帧剖面的高斯拟合补出来的, 只影响显示; "
            "关掉后 σ / FWHM / R² 仍照常计算并显示"
        )
        self.fit_toggle.toggled.connect(self.set_fit_curve_visible)

        # 顶部一行只占一个控件高度, 两条剖面仍在下面平分宽度;
        # 左侧 6px 让勾选框与下方图表卡片的左边缘对齐
        toggle_row = QHBoxLayout()
        toggle_row.setContentsMargins(6, 0, 6, 0)
        toggle_row.addWidget(self.fit_toggle, 0)
        toggle_row.addStretch(1)

        panels_row = QHBoxLayout()
        panels_row.setContentsMargins(0, 0, 0, 0)
        panels_row.setSpacing(4)
        panels_row.addWidget(self._x_panel)
        panels_row.addWidget(self._y_panel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addLayout(toggle_row, 0)
        layout.addLayout(panels_row, 1)

    def set_fit_curve_visible(self, visible: bool) -> None:
        """显示/隐藏两条高斯拟合虚线 (幂等; 不会重新拟合, 也不清数据)。"""
        visible = bool(visible)
        if self.fit_toggle.isChecked() != visible:
            # 同步勾选框, toggled 信号会再回来应用一次
            self.fit_toggle.setChecked(visible)
        self._x_panel.set_fit_visible(visible)
        self._y_panel.set_fit_visible(visible)

    def fit_curve_visible(self) -> bool:
        """高斯拟合虚线当前是否显示。"""
        return self._x_panel.fit_visible() and self._y_panel.fit_visible()

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
