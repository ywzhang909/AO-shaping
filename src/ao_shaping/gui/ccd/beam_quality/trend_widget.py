"""最大亮度趋势折线图 (主视图): 峰值/均值曲线 + 可选饱和占比。

数据来自 :class:`TrendBuffer`; 本部件只负责画, 不负责记录 —— 记录与暂停状态
由缓冲自己持有, 因此暂停时曲线保持在屏幕上不再增长。

刷新节流由调用方负责 (主窗口 ~5 Hz 调 :meth:`update_buffer` + :meth:`refresh`),
这样 30 fps 的采样不会变成 30 次/秒的重绘。
"""

from __future__ import annotations

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


class TrendWidget(QWidget):
    """光束峰值/均值随时间变化的折线图。"""

    def __init__(
        self, buffer: TrendBuffer | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._buffer = buffer if buffer is not None else TrendBuffer()
        self._records: list[TrendSample] = []
        self._window: float | None = WINDOWS[0][1]

        # --- 序列 ---
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
        layout.addWidget(self._view, 1)
        layout.addWidget(self._status, 0)

        self._window_combo.currentIndexChanged.connect(self._on_window_changed)
        self._pause_btn.toggled.connect(self._on_pause_toggled)
        self._clear_btn.clicked.connect(self.clear)
        self._export_btn.clicked.connect(lambda: self.export_csv())

        self.refresh()

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
            self._max_series.clear()
            self._mean_series.clear()
            self._sat_series.clear()
            self._axis_x.setRange(0.0, 30.0)
            self._axis_y.setRange(0.0, 255.0)
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
        self._axis_x.setRange(xs[0], xs[0] + max(span, 1.0))

        values = [sample.max for sample in records] + [
            sample.mean for sample in records
        ]
        low = min(values)
        high = max(values)
        pad = max((high - low) * 0.1, 1.0)
        self._axis_y.setRange(max(low - pad, 0.0), high + pad)

        sat_peak = max(sample.sat_ratio for sample in records)
        self._axis_sat.setRange(0.0, min(max(sat_peak * 1.3, 0.005), 1.0))

        latest = records[-1]
        self._status.setText(
            f"{len(records)} 采样 | 最新 max {latest.max:.1f} / "
            f"mean {latest.mean:.2f} | 饱和 {latest.sat_ratio * 100:.2f}%"
        )
