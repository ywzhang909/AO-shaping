"""光束质量趋势折线图 (主视图): 峰值/均值/饱和占比 + 中心位置 + 椭圆长短半轴。

数据来自 :class:`TrendBuffer`; 本部件只负责画, 不负责记录 —— 记录与暂停状态
由缓冲自己持有, 因此暂停时曲线保持在屏幕上不再增长。

刷新节流由调用方负责 (主窗口 ~5 Hz 调 :meth:`update_buffer` + :meth:`refresh`),
这样 30 fps 的采样不会变成 30 次/秒的重绘。

三个子图从上到下:
1. 主图: 峰值/均值 (左轴, 灰度) + 饱和占比 (右轴, 比例);
2. 光束中心位置: cx / cy (按当前选择的中心算法算出的中心);
3. 椭圆长短半轴: semi_major / semi_minor (无椭圆时为 NaN, 跳过该点)。
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PySide6.QtCore import QMargins, QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ao_shaping.gui.ccd.beam_quality.trend_buffer import TrendBuffer, TrendSample

# (标签, 时间窗秒数); None = 全部
WINDOWS: tuple[tuple[str, float | None], ...] = (
    ("30 s", 30.0),
    ("2 min", 120.0),
    ("10 min", 600.0),
    ("全部", None),
)

_MAX_COLOR = QColor(80, 170, 255)
_MEAN_COLOR = QColor(120, 225, 150)
_SAT_COLOR = QColor(255, 166, 90)
_CX_COLOR = QColor(200, 140, 255)
_CY_COLOR = QColor(100, 210, 235)
_MAJOR_COLOR = QColor(255, 130, 110)
_MINOR_COLOR = QColor(140, 255, 170)


class TrendWidget(QWidget):
    """光束质量随时间变化的折线图 (峰值/均值/饱和 + 中心 + 椭圆半轴)。"""

    def __init__(
        self, buffer: TrendBuffer | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._buffer = buffer if buffer is not None else TrendBuffer()
        self._records: list[TrendSample] = []
        self._window: float | None = WINDOWS[0][1]

        # --- 序列 (主图) ---
        self._max_series = QLineSeries()
        self._max_series.setName("max (峰值)")
        pen = QPen(_MAX_COLOR)
        pen.setWidthF(2.2)
        self._max_series.setPen(pen)

        self._mean_series = QLineSeries()
        self._mean_series.setName("mean (均值)")
        pen = QPen(_MEAN_COLOR)
        pen.setWidthF(1.2)
        self._mean_series.setPen(pen)

        self._sat_series = QLineSeries()
        self._sat_series.setName("饱和占比")
        pen = QPen(_SAT_COLOR)
        pen.setWidthF(1.2)
        pen.setStyle(Qt.PenStyle.DashLine)
        self._sat_series.setPen(pen)

        # --- 序列 (新增子图) ---
        self._cx_series = self._make_series("cx (x 中心)", _CX_COLOR, width=1.2)
        self._cy_series = self._make_series("cy (y 中心)", _CY_COLOR, width=1.2)
        self._major_series = self._make_series("长半轴", _MAJOR_COLOR, width=1.4)
        self._minor_series = self._make_series("短半轴", _MINOR_COLOR, width=1.4)

        self._axis_x = QValueAxis()
        self._axis_x.setTitleText("时间 (s)")
        self._axis_x.setLabelFormat("%.1f")
        self._axis_y = QValueAxis()
        self._axis_y.setTitleText("灰度")
        self._axis_y.setLabelFormat("%.0f")
        self._axis_sat = QValueAxis()
        self._axis_sat.setTitleText("饱和占比")
        self._axis_sat.setLabelFormat("%.2f")
        self._axis_sat.setRange(0.0, 1.0)

        self._chart = QChart()
        for series in (self._max_series, self._mean_series, self._sat_series):
            self._chart.addSeries(series)
        self._chart.addAxis(self._axis_x, Qt.AlignmentFlag.AlignBottom)
        self._chart.addAxis(self._axis_y, Qt.AlignmentFlag.AlignLeft)
        self._chart.addAxis(self._axis_sat, Qt.AlignmentFlag.AlignRight)
        for series in (self._max_series, self._mean_series):
            series.attachAxis(self._axis_x)
            series.attachAxis(self._axis_y)
        self._sat_series.attachAxis(self._axis_x)
        self._sat_series.attachAxis(self._axis_sat)
        self._chart.setTitle("光束峰值趋势")
        self._chart.legend().setVisible(True)
        self._chart.setMargins(QMargins(2, 2, 2, 2))
        self._chart.setAnimationOptions(QChart.AnimationOption.NoAnimation)

        self._view = QChartView(self._chart)
        self._view.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # --- 新增子图: 中心位置 / 椭圆长短半轴 ---
        (
            self._center_view,
            self._center_axis_x,
            self._center_axis_y,
        ) = self._make_panel(
            "光束中心位置趋势", "中心 (px)", [self._cx_series, self._cy_series]
        )
        self._center_view.setMinimumHeight(110)

        (
            self._ellipse_view,
            self._ellipse_axis_x,
            self._ellipse_axis_y,
        ) = self._make_panel(
            "椭圆长短半轴趋势", "半轴 (px)", [self._major_series, self._minor_series]
        )
        self._ellipse_view.setMinimumHeight(110)

        # --- 控件 ---
        self._window_combo = QComboBox()
        for label, seconds in WINDOWS:
            self._window_combo.addItem(label, seconds)
        self._window_combo.setCurrentIndex(0)

        self._pause_btn = QPushButton("暂停记录")
        self._pause_btn.setCheckable(True)
        self._pause_btn.setToolTip("暂停后图表保留历史, 但不再记录新采样")

        self._clear_btn = QPushButton("清空")
        self._export_btn = QPushButton("导出 CSV")

        self._status = QLabel("暂无数据")
        self._status.setWordWrap(True)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.addWidget(QLabel("时间窗"))
        controls.addWidget(self._window_combo)
        controls.addWidget(self._pause_btn)
        controls.addWidget(self._clear_btn)
        controls.addWidget(self._export_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addLayout(controls)
        layout.addWidget(self._view, 3)
        layout.addWidget(self._center_view, 2)
        layout.addWidget(self._ellipse_view, 2)
        layout.addWidget(self._status, 0)

        self._window_combo.currentIndexChanged.connect(self._on_window_changed)
        self._pause_btn.toggled.connect(self._on_pause_toggled)
        self._clear_btn.clicked.connect(self.clear)
        self._export_btn.clicked.connect(lambda: self.export_csv())

        self.refresh()

    # ---------------------------------------------------------------- 构建
    def _make_series(
        self, name: str, color: QColor, *, width: float = 1.2
    ) -> QLineSeries:
        series = QLineSeries()
        series.setName(name)
        pen = QPen(color)
        pen.setWidthF(width)
        series.setPen(pen)
        return series

    def _make_panel(
        self, title: str, y_title: str, series_list: list[QLineSeries]
    ) -> tuple[QChartView, QValueAxis, QValueAxis]:
        """建一个单 y 轴子图 (x = 时间, y = y_title), 返回 (view, axis_x, axis_y)。"""
        chart = QChart()
        for series in series_list:
            chart.addSeries(series)
        axis_x = QValueAxis()
        axis_x.setTitleText("时间 (s)")
        axis_x.setLabelFormat("%.1f")
        axis_y = QValueAxis()
        axis_y.setTitleText(y_title)
        axis_y.setLabelFormat("%.1f")
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        for series in series_list:
            series.attachAxis(axis_x)
            series.attachAxis(axis_y)
        chart.setTitle(title)
        chart.legend().setVisible(True)
        chart.setMargins(QMargins(2, 2, 2, 2))
        chart.setAnimationOptions(QChart.AnimationOption.NoAnimation)
        view = QChartView(chart)
        view.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        return view, axis_x, axis_y

    # ------------------------------------------------------------------ 属性
    @property
    def buffer(self) -> TrendBuffer:
        return self._buffer

    @property
    def window_seconds(self) -> float | None:
        return self._window

    # ------------------------------------------------------------------ 交互
    def _on_window_changed(self, index: int) -> None:
        if 0 <= index < len(WINDOWS):
            self._window = WINDOWS[index][1]
        self.update_buffer()
        self.refresh()

    def _on_pause_toggled(self, checked: bool) -> None:
        self._buffer.paused = checked
        self._pause_btn.setText("继续记录" if checked else "暂停记录")

    def clear(self) -> None:
        """清空缓冲并清屏。"""
        self._buffer.clear()
        self._records = []
        self.refresh()

    def export_csv(self, path: str | Path | None = None) -> Path | None:
        """导出 CSV; 未给路径时弹文件对话框。返回实际写入路径 (取消时 None)。"""
        target = path
        if target is None:
            default = (
                Path("data")
                / "beam_quality"
                / (f"beam_trend_{datetime.now():%Y%m%d_%H%M%S}.csv")
            )
            chosen, _ = QFileDialog.getSaveFileName(
                self, "导出趋势 CSV", str(default), "CSV 文件 (*.csv)"
            )
            if not chosen:
                return None
            target = chosen
        written = self._buffer.export_csv(target)
        self._status.setText(f"已导出 {len(self._buffer)} 条 → {written}")
        return written

    # ------------------------------------------------------------ 数据 → 图
    def update_buffer(self) -> None:
        """从缓冲按当前时间窗取样本 (不重绘)。"""
        self._records = self._buffer.records(self._window)

    def refresh(self) -> None:
        """按已取样本重绘曲线与坐标轴 (调用方负责节流)。"""
        records = self._records
        if not records:
            self._clear_all_series()
            self._axis_x.setRange(0.0, 30.0)
            self._axis_y.setRange(0.0, 255.0)
            self._center_axis_x.setRange(0.0, 30.0)
            self._center_axis_y.setRange(0.0, 1.0)
            self._ellipse_axis_x.setRange(0.0, 30.0)
            self._ellipse_axis_y.setRange(0.0, 1.0)
            self._status.setText("暂无数据")
            return

        t0 = self._buffer.t0 if self._buffer.t0 is not None else records[0].timestamp
        xs = [sample.timestamp - t0 for sample in records]
        self._max_series.replace(
            [QPointF(x, sample.max) for x, sample in zip(xs, records)]
        )
        self._mean_series.replace(
            [QPointF(x, sample.mean) for x, sample in zip(xs, records)]
        )
        self._sat_series.replace(
            [QPointF(x, sample.sat_ratio) for x, sample in zip(xs, records)]
        )

        span = max(xs[-1] - xs[0], 1e-3)
        x_hi = xs[0] + max(span, 1.0)
        self._axis_x.setRange(xs[0], x_hi)
        self._center_axis_x.setRange(xs[0], x_hi)
        self._ellipse_axis_x.setRange(xs[0], x_hi)

        values = [sample.max for sample in records] + [
            sample.mean for sample in records
        ]
        low = min(values)
        high = max(values)
        pad = max((high - low) * 0.1, 1.0)
        self._axis_y.setRange(max(low - pad, 0.0), high + pad)

        sat_peak = max(sample.sat_ratio for sample in records)
        self._axis_sat.setRange(0.0, min(max(sat_peak * 1.3, 0.005), 1.0))

        # 中心位置 (cx/cy 恒有值)
        self._cx_series.replace(
            [QPointF(x, sample.cx) for x, sample in zip(xs, records)]
        )
        self._cy_series.replace(
            [QPointF(x, sample.cy) for x, sample in zip(xs, records)]
        )
        center_vals = [
            sample.cx for sample in records
        ] + [sample.cy for sample in records]
        clow, chigh = min(center_vals), max(center_vals)
        cpad = max((chigh - clow) * 0.1, 1.0)
        # 像素坐标非负: 下界钳到 0
        self._center_axis_y.setRange(max(clow - cpad, 0.0), chigh + cpad)

        # 椭圆长短半轴 (无椭圆 = NaN, 跳过该点)
        self._major_series.replace(
            [
                QPointF(x, sample.semi_major)
                for x, sample in zip(xs, records)
                if not math.isnan(sample.semi_major)
            ]
        )
        self._minor_series.replace(
            [
                QPointF(x, sample.semi_minor)
                for x, sample in zip(xs, records)
                if not math.isnan(sample.semi_minor)
            ]
        )
        ellipse_vals = [
            v
            for sample in records
            for v in (sample.semi_major, sample.semi_minor)
            if not math.isnan(v)
        ]
        if ellipse_vals:
            elow, ehigh = min(ellipse_vals), max(ellipse_vals)
            epad = max((ehigh - elow) * 0.1, 1.0)
            self._ellipse_axis_y.setRange(max(elow - epad, 0.0), ehigh + epad)
        else:
            self._ellipse_axis_y.setRange(0.0, 1.0)

        latest = records[-1]
        self._status.setText(
            f"{len(records)} 采样 | 最新 max {latest.max:.1f} / "
            f"mean {latest.mean:.2f} | 饱和 {latest.sat_ratio * 100:.2f}%"
        )

    def _clear_all_series(self) -> None:
        """清空全部序列 (主图 + 两个子图)。"""
        for series in (
            self._max_series,
            self._mean_series,
            self._sat_series,
            self._cx_series,
            self._cy_series,
            self._major_series,
            self._minor_series,
        ):
            series.clear()
