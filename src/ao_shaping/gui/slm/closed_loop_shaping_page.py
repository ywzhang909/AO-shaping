"""CCD 闭环目标保持页（默认；逐轮优化保留为可选的旧流程）。

这一页与 :mod:`dynamic_shaping_page` 的区别只有一句话：**相位下一步往哪走，由 CCD
这一帧的实测形状决定，而不是由开环时间轴决定**。每轮严格串行地做
写相位 → 等稳定 → 取新帧 → 帧门禁 → 实测分数 → 决定接受或回滚。

安全约定
--------
* 页面构造时**绝不碰设备**。只有操作员点了「开始」并在确认框里确认后，才在专用
  worker 线程里打开 SLM/CCD；关窗时停线程、等线程、关设备。
* 默认「目标保持（全程闭环）」：先收敛到固定 CCD 目标，再持续校正形状与 CCD 像素
  位置偏差；目标在本次运行中不变，位置参考是固定 CCD 坐标，**不重新居中光斑**。
  「逐轮优化」（每轮实测后由操作员确认接受或拒绝）保留为可选的旧流程。
* 「目标保持」目前只在离线回放里验证过 —— 界面措辞不承诺它已在硬件上验证过拒漂移。
* 帧门禁未通过的帧不允许被接受（核心层保证）；页面上把它标成红色并触发回滚。
* 模型预测分数与实测分数分列显示：预测只是"下一步的猜测"，绝不是接受依据。
* **transfer 在线适配默认关闭**。开启后必须填检查点、含 ``reference_start`` 与
  两轴 ``±ramp`` 的既有标定数据集、以及独立会话输出目录；三件套在占用设备之
  前只读校验，不合格就不开 SLM/CCD。训练/验证门槛由
  :class:`~closed_loop_shaping.AdaptationPolicy` 决定，页面不提供放宽入口。

适配的"在线"是累计样本 → warm-start 批量重拟合 → 独立验证晋级，不是逐帧更新：
单个 ``(phase, frame, exposure)`` 样本解不出几何与辐射标定（见
``tools/slm/transfer/fit.py`` 的 ``load_dataset`` 前置条件）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ao_shaping.gui.slm.closed_loop_shaping import (
    AdaptationPolicy,
    ClosedLoopConfig,
    ClosedLoopRunner,
    LoopEvent,
    LoopState,
    LoopSummary,
    OnlineTransferAdapter,
    analysis_grid,
    describe_gate,
    grid_spacing_px,
    target_field,
)
from ao_shaping.gui.slm.target_lock import (
    CalibrationSample,
    LockConfig,
    calibrate_lock_thresholds,
    calibration_stats_from_frames,
    target_on_frame,
)

__all__ = ["ClosedLoopShapingPage"]

_STATE_TEXT: dict[LoopState, str] = {
    LoopState.IDLE: "空闲",
    LoopState.PREFLIGHT: "自检",
    LoopState.UPDATING: "计算候选",
    LoopState.MEASURING: "实测",
    LoopState.AWAITING_DECISION: "待确认",
    LoopState.PAUSED: "已暂停",
    LoopState.STOPPING: "停止中",
    LoopState.STOPPED: "已停止",
    LoopState.FAULT: "故障",
}

_DECISION_TEXT = {
    "baseline": "起点",
    "auto": "自动接受",
    "accept": "人工接受",
    "reject": "拒绝/回滚",
    "gate_fail": "门禁失败/回滚",
    None: "—",
}


def _spot_of(row: dict[str, Any]) -> np.ndarray | None:
    """取 manifest 行的实测光斑，口径与 ``transfer.fit.spot`` 一致。"""
    if "spot_xy" in row:
        values = row["spot_xy"]
    else:
        spot = row.get("spot")
        if not isinstance(spot, dict) or "centroid_x" not in spot or "centroid_y" not in spot:
            return None
        values = [spot["centroid_x"], spot["centroid_y"]]
    try:
        point = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None
    if point.shape != (2,) or not np.isfinite(point).all():
        return None
    return point


class _RunnerSignals(QObject):
    """worker → GUI 的信号；payload 都是纯数据，不跨线程碰控件。"""

    event = Signal(object)  # LoopEvent
    finished = Signal(object)  # LoopSummary
    failed = Signal(str)


class _RunnerThread(QThread):
    """在专用线程里跑整场闭环；设备句柄只在这个线程里存活。"""

    def __init__(self, runner: ClosedLoopRunner, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.signals = _RunnerSignals()
        self._runner = runner

    @property
    def runner(self) -> ClosedLoopRunner:
        return self._runner

    def run(self) -> None:  # pragma: no cover - 线程体，由离线测试驱动 runner 本身
        try:
            summary = self._runner.run()
        except Exception as exc:  # 线程里不能让异常静默消失
            self.signals.failed.emit(str(exc))
            return
        self.signals.finished.emit(summary)


class _CalibrationSignals(QObject):
    done = Signal(object)  # CalibrationSample
    failed = Signal(str)


class _CalibrationThread(QThread):
    """在专用线程里采集标定帧；设备句柄只在这个线程里存活。

    标定相位刻意用零相位：它给的是"已知、可复现的起点"下的噪声地板，而不是
    某次碰巧调出来的目标形状。标定帧与目标图同形，评分重复性只作为噪声代理。
    """

    def __init__(
        self,
        bench: Any,
        *,
        xy: np.ndarray,
        target: np.ndarray,
        frame_count: int,
        grid_size: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.signals = _CalibrationSignals()
        self._bench = bench
        self._xy = np.asarray(xy, dtype=np.float64)
        self._target = np.asarray(target, dtype=np.float64)
        self._frame_count = frame_count
        self._grid_size = grid_size

    def run(self) -> None:  # pragma: no cover - 线程体，离线测试驱动统计纯函数
        try:
            self._bench.open()
            self._bench.apply_phase(np.zeros((self._grid_size, self._grid_size)))
            captured = [self._bench.capture() for _ in range(self._frame_count)]
            # 硬件帧是整幅 CCD，目标是窗口图；统计要求同形，先按绝对像素铺回整帧。
            target_frame = target_on_frame(
                self._xy, self._target, captured[0].frame.shape
            )
            sample = calibration_stats_from_frames(
                [item.frame for item in captured],
                target=target_frame,
                full_scale=float(captured[0].full_scale),
            )
        except Exception as exc:
            self.signals.failed.emit(str(exc))
            return
        finally:
            try:
                self._bench.close()
            except Exception:  # 关闭失败不该掩盖主结论
                pass
        self.signals.done.emit(sample)


class ClosedLoopShapingPage(QWidget):
    """独立闭环页。所有昂贵/危险的东西都延迟到显式开始之后。"""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        bench_factory: Callable[..., Any] | None = None,
        forward_factory: Callable[[np.ndarray], Any] | None = None,
        adapter_factory: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("CCD 闭环目标保持")
        self._bench_factory = bench_factory
        self._forward_factory = forward_factory
        self._adapter_factory = adapter_factory
        self._thread: _RunnerThread | None = None
        self._closing = False
        self._last_record: Any | None = None
        #: 两种运行方式的轮数上限含义不同（保持要跑很多轮），各记一份，切换时不丢。
        self._round_cap: dict[str, int] = {"hold": 120, "optimize": 8}
        self._mode_shown: str | None = None
        self._running = False
        #: 标定面板状态：建议值一旦输入变化就作废，避免应用到过期阈值。
        self._cal_recommended: LockConfig | None = None
        self._cal_filling = False
        self._cal_thread: _CalibrationThread | None = None

        self._build_ui()
        self._set_controls_running(False)

    # -- 界面 -------------------------------------------------------------- #

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)

        title = QLabel("CCD 闭环目标保持 · 实测驱动")
        title.setStyleSheet("font-size: 15px; font-weight: 600;")
        root.addWidget(title)
        note = QLabel(
            "目标保持（默认）：先收敛到固定 CCD 目标，再持续校正形状与 CCD 像素位置"
            "偏差；目标在本次运行中不变。\n"
            "逐轮优化：每轮实测后停下，由你确认接受或拒绝（旧流程）。"
            "模型预测仅供观察；只有实测改善才会被接受。"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #888;")
        root.addWidget(note)

        status_row = QHBoxLayout()
        self._state_label = QLabel("状态：空闲")
        self._state_label.setObjectName("closedLoopState")
        status_row.addWidget(self._state_label)
        self._decision_hint = QLabel("")
        self._decision_hint.setWordWrap(True)
        status_row.addWidget(self._decision_hint, 1)
        root.addLayout(status_row)

        # 故障横幅：冻结/异常的原因留在页面正文里，不只靠一个弹窗。
        self._fault_banner = QLabel("")
        self._fault_banner.setObjectName("closedLoopFault")
        self._fault_banner.setWordWrap(True)
        self._fault_banner.setStyleSheet(
            "color: #ffffff; background: #c0392b; border-radius: 4px; padding: 6px;"
        )
        self._fault_banner.hide()
        root.addWidget(self._fault_banner)

        root.addWidget(self._build_mode())
        root.addWidget(self._build_target())
        root.addWidget(self._build_params())
        root.addWidget(self._build_calibration())
        root.addWidget(self._build_adaptation())

        # 相位行：运行方式为「目标保持」时显示 收敛中 → 维持中 → 已冻结。
        self._phase_label = QLabel("")
        self._phase_label.setObjectName("closedLoopPhase")
        self._phase_label.setStyleSheet("font-size: 14px; font-weight: 600;")
        root.addWidget(self._phase_label)

        control_row = QHBoxLayout()
        self._start_button = QPushButton("开始闭环…")
        self._start_button.setObjectName("closedLoopStart")
        self._start_button.clicked.connect(self.on_start)
        control_row.addWidget(self._start_button)
        self._accept_button = QPushButton("接受候选")
        self._accept_button.setObjectName("closedLoopAccept")
        self._accept_button.clicked.connect(self.on_accept)
        control_row.addWidget(self._accept_button)
        self._reject_button = QPushButton("拒绝并回滚")
        self._reject_button.setObjectName("closedLoopReject")
        self._reject_button.clicked.connect(self.on_reject)
        control_row.addWidget(self._reject_button)
        self._pause_button = QPushButton("暂停")
        self._pause_button.setObjectName("closedLoopPause")
        self._pause_button.clicked.connect(self.on_pause_resume)
        control_row.addWidget(self._pause_button)
        self._stop_button = QPushButton("停止")
        self._stop_button.setObjectName("closedLoopStop")
        self._stop_button.setStyleSheet("font-weight: 600;")
        self._stop_button.setMinimumWidth(72)
        self._stop_button.clicked.connect(self.on_stop)
        control_row.addWidget(self._stop_button)
        control_row.addStretch(1)
        root.addLayout(control_row)

        self._table = QTableWidget(0, 7)
        self._table.setHorizontalHeaderLabels(
            ["轮次", "门禁", "实测分数", "预测分数", "步长RMS", "窗口占比", "决定"]
        )
        self._table.setObjectName("closedLoopRounds")
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        root.addWidget(self._table, 1)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(500)
        self._log.setObjectName("closedLoopLog")
        root.addWidget(self._log, 1)

        self._append_log("页面就绪，尚未打开任何设备。设置参数后点「开始闭环」。")
        # 模式相关的可见/可用性与轮数上限在这里统一落到默认值（目标保持）。
        self._on_mode_changed()

    def _build_mode(self) -> QGroupBox:
        """运行方式：目标保持（默认，全程闭环）或逐轮优化（旧流程）。"""
        box = QGroupBox("运行方式")
        form = QFormLayout(box)

        self._mode_combo = QComboBox()
        self._mode_combo.setObjectName("closedLoopMode")
        self._mode_combo.addItem("目标保持（全程闭环）", "hold")
        self._mode_combo.addItem("逐轮优化", "optimize")
        self._mode_combo.setCurrentIndex(0)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        form.addRow("模式", self._mode_combo)

        self._hold_tolerance = QDoubleSpinBox()
        self._hold_tolerance.setRange(0.5, 50.0)
        self._hold_tolerance.setSingleStep(0.5)
        self._hold_tolerance.setDecimals(1)
        self._hold_tolerance.setValue(5.0)
        form.addRow("位置容差（CCD px）", self._hold_tolerance)

        self._mode_hint = QLabel("")
        self._mode_hint.setWordWrap(True)
        self._mode_hint.setStyleSheet("color: #888;")
        form.addRow("", self._mode_hint)
        return box

    def _build_target(self) -> QGroupBox:
        """固定 CCD 目标：中心与尺寸在整场运行中不变（开跑后控件冻结）。"""
        box = QGroupBox("固定 CCD 目标（运行中不变）")
        form = QFormLayout(box)

        self._span = QSpinBox()
        self._span.setRange(2, 120)
        self._span.setValue(4)
        form.addRow("分析窗口 span（半宽，步长 2px）", self._span)

        self._centre_x = QDoubleSpinBox()
        self._centre_x.setRange(0.0, 4096.0)
        self._centre_x.setValue(500.0)
        self._centre_y = QDoubleSpinBox()
        self._centre_y.setRange(0.0, 4096.0)
        self._centre_y.setValue(500.0)
        centre_row = QHBoxLayout()
        centre_row.addWidget(QLabel("x"))
        centre_row.addWidget(self._centre_x)
        centre_row.addWidget(QLabel("y"))
        centre_row.addWidget(self._centre_y)
        form.addRow("中心（CCD 像素 x、y）", centre_row)

        self._half_width = QDoubleSpinBox()
        self._half_width.setRange(0.5, 60.0)
        self._half_width.setValue(2.0)
        self._half_height = QDoubleSpinBox()
        self._half_height.setRange(0.5, 60.0)
        self._half_height.setValue(2.0)
        shape_row = QHBoxLayout()
        shape_row.addWidget(QLabel("宽"))
        shape_row.addWidget(self._half_width)
        shape_row.addWidget(QLabel("高"))
        shape_row.addWidget(self._half_height)
        form.addRow("目标矩形半宽/半高（px）", shape_row)
        return box

    def _build_params(self) -> QGroupBox:
        box = QGroupBox("运行参数")
        form = QFormLayout(box)

        self._grid_size = QSpinBox()
        self._grid_size.setRange(8, 512)
        self._grid_size.setValue(128)
        form.addRow("SLM 控制网格边长", self._grid_size)

        self._rounds = QSpinBox()
        self._rounds_cap_label = QLabel("运行轮数上限")
        form.addRow(self._rounds_cap_label, self._rounds)

        self._exposure_ms = QDoubleSpinBox()
        self._exposure_ms.setRange(0.1, 1000.0)
        self._exposure_ms.setValue(20.0)
        form.addRow("曝光（ms）", self._exposure_ms)

        self._auto_accept = QCheckBox("自动接受实测改善（默认关闭，需人工确认）")
        self._auto_accept.setChecked(False)
        form.addRow("", self._auto_accept)
        return box

    # -- 运行方式 ---------------------------------------------------------- #

    def _run_mode(self) -> str:
        """当前运行方式：``hold``（目标保持）或 ``optimize``（逐轮优化）。"""
        data = self._mode_combo.currentData()
        return str(data) if data else "hold"

    def _on_mode_changed(self, *_args: Any) -> None:
        """切换运行方式：轮数上限、确认控件与提示随之收口。"""
        mode = self._run_mode()
        if self._mode_shown is not None and self._mode_shown != mode:
            # 两种模式的轮数上限含义不同，离开前先记住当前值。
            self._round_cap[self._mode_shown] = self._rounds.value()
        self._mode_shown = mode
        if mode == "hold":
            self._rounds_cap_label.setText("运行轮数上限")
            self._rounds.setRange(1, 500)
            # 保持模式的自动接受由 ``candidate_policy="auto"`` 承担, 与这个复选框无关:
            # 复选框在保持模式下隐藏, 且保持为未勾选 —— 切回逐轮优化时不会被悄悄打开。
        else:
            self._rounds_cap_label.setText("最大轮数")
            self._rounds.setRange(1, 200)
        self._rounds.setValue(self._round_cap[mode])
        self._apply_mode_ui()

    def _apply_mode_ui(self) -> None:
        """按运行方式调整可见/可用性；开跑后再由 ``_set_controls_running`` 冻结。"""
        hold = self._run_mode() == "hold"
        # 目标保持在跑完之前不需要逐轮确认；逐轮优化保持旧行为。
        self._accept_button.setVisible(not hold)
        self._reject_button.setVisible(not hold)
        self._auto_accept.setVisible(not hold)
        self._hold_tolerance.setEnabled(hold and not self._running)
        self._phase_label.setVisible(hold)
        if hold:
            self._decision_hint.clear()
            self._mode_hint.setText(
                "全程闭环：先收敛到固定 CCD 目标，再维持形状与 CCD 像素位置；"
                "目标在本次运行中不变。保持模式自动接受实测改善。"
            )
        else:
            self._mode_hint.setText(
                "逐轮优化：每轮算出候选并实测后停下，由你确认接受或拒绝（旧流程）。"
            )

    # -- LockConfig 阈值标定 ------------------------------------------------ #

    def _build_calibration(self) -> QGroupBox:
        """用实测噪声/信号推导本次运行的 LockConfig 阈值。

        默认不启用。标定是显式动作，采集会短时占用设备；推导值与输入值同屏显示，
        操作员核对后才勾选应用。采集相位刻意用零相位作为可复现的噪声地板起点。
        """
        box = QGroupBox("锁定阈值标定（LockConfig · 默认不启用）")
        form = QFormLayout(box)

        def add_spin(
            label: str,
            minimum: float,
            maximum: float,
            value: float,
            decimals: int = 2,
        ) -> QDoubleSpinBox:
            spin = QDoubleSpinBox()
            spin.setRange(minimum, maximum)
            spin.setDecimals(decimals)
            spin.setValue(value)
            form.addRow(label, spin)
            spin.valueChanged.connect(self._on_calibration_input_changed)
            return spin

        self._cal_dark_mean = add_spin("暗场均值（counts）", 0.0, 65535.0, 0.0)
        self._cal_dark_sigma = add_spin("暗场噪声 σ（counts）", 0.0, 65535.0, 1.0)
        self._cal_peak = add_spin("峰值 counts", 0.0, 65535.0, 200.0)
        self._cal_total = add_spin("总 counts", 0.0, 1e9, 10000.0, 1)
        self._cal_full_scale = add_spin("相机满量程 counts", 1.0, 65535.0, 255.0)
        self._cal_saturated = add_spin("饱和像素占比（%）", 0.0, 100.0, 0.0, 3)
        self._cal_centroid_sigma = add_spin("质心重复性 σ（CCD px）", 0.0, 1000.0, 0.5, 3)
        self._cal_score_sigma = add_spin("评分重复性 σ", 0.0, 10.0, 0.01, 4)

        row = QHBoxLayout()
        self._cal_capture_button = QPushButton("采集标定帧…")
        self._cal_capture_button.setObjectName("lockCalibrationCapture")
        self._cal_capture_button.clicked.connect(self.on_capture_calibration)
        row.addWidget(self._cal_capture_button)
        self._cal_calc_button = QPushButton("计算建议阈值")
        self._cal_calc_button.setObjectName("lockCalibrationCalc")
        self._cal_calc_button.clicked.connect(self.on_calibrate)
        row.addWidget(self._cal_calc_button)
        self._cal_reset_button = QPushButton("恢复默认输入")
        self._cal_reset_button.clicked.connect(self.on_reset_calibration)
        row.addWidget(self._cal_reset_button)
        form.addRow("", row)

        self._cal_preview = QLabel("尚未计算建议阈值；默认阈值仅面向 8/12-bit 相机。")
        self._cal_preview.setWordWrap(True)
        self._cal_preview.setObjectName("lockCalibrationPreview")
        form.addRow("推导结果", self._cal_preview)

        self._cal_apply = QCheckBox("应用标定阈值到本次目标保持运行（默认关闭）")
        self._cal_apply.setObjectName("lockCalibrationApply")
        self._cal_apply.setChecked(False)
        form.addRow("", self._cal_apply)
        return box

    def _calibration_target_and_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """用页面当前目标参数构造与 on_start 完全一致的网格和目标图。"""
        centre = (self._centre_x.value(), self._centre_y.value())
        xy = analysis_grid(centre, self._span.value())
        target = target_field(
            xy,
            centre_xy=centre,
            half_width=self._half_width.value(),
            half_height=self._half_height.value(),
        )
        return xy, target

    def _calibration_sample(self) -> CalibrationSample:
        return CalibrationSample(
            dark_mean=self._cal_dark_mean.value(),
            dark_sigma=self._cal_dark_sigma.value(),
            peak_counts=max(self._cal_peak.value(), 1e-6),
            total_counts=max(self._cal_total.value(), 1e-6),
            saturated_fraction=self._cal_saturated.value() / 100.0,
            full_scale=self._cal_full_scale.value(),
            centroid_sigma_px=self._cal_centroid_sigma.value(),
            score_sigma=self._cal_score_sigma.value(),
            pixel_count=1,
        )

    def _on_calibration_input_changed(self, *_args: Any) -> None:
        if self._cal_filling:
            return
        if self._cal_recommended is not None:
            self._cal_recommended = None
            self._refresh_calibration_preview()

    def _refresh_calibration_preview(self) -> None:
        cfg = self._cal_recommended
        if cfg is None:
            self._cal_preview.setText("尚未计算建议阈值；默认阈值仅面向 8/12-bit 相机。")
            return
        self._cal_preview.setText(
            f"饱和阈值 {cfg.saturation_counts:.1f} counts，允许饱和占比 "
            f"{cfg.max_saturated_fraction:.3%}；峰值下限 {cfg.min_peak_counts:.1f}，"
            f"总量下限 {cfg.min_total_counts:.1f}\n"
            f"位置容差 {cfg.max_centroid_error_px:.2f} 网格单位；改进余量 "
            f"{cfg.improvement_epsilon:.4f}，回滚余量 {cfg.rollback_epsilon:.4f}，"
            f"振荡幅度 {cfg.oscillation_amplitude:.4f}"
        )

    def on_calibrate(self, _checked: bool = False, *, xy: np.ndarray | None = None) -> bool:
        """按当前输入算建议阈值并预览；返回是否成功。"""
        if xy is None:
            xy, _ = self._calibration_target_and_grid()
        spacing = grid_spacing_px(xy)
        if spacing is None:
            QMessageBox.warning(self, "无法标定", "分析网格没有正的有限步长。")
            return False
        try:
            cfg = calibrate_lock_thresholds(
                self._calibration_sample(),
                position_tolerance_ccd_px=self._hold_tolerance.value(),
                grid_spacing_px=spacing,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "标定输入无效", str(exc))
            return False
        self._cal_recommended = cfg
        self._refresh_calibration_preview()
        self._append_log(f"LockConfig 建议阈值：{cfg}")
        return True

    def _lock_config_for_run(self, xy: np.ndarray) -> LockConfig | None:
        if not self._cal_apply.isChecked():
            return None
        if self._cal_recommended is None and not self.on_calibrate(xy=xy):
            raise ValueError("已勾选应用标定阈值，但标定输入无效")
        assert self._cal_recommended is not None
        return self._cal_recommended

    def on_reset_calibration(self, *_args: Any) -> None:
        self._cal_filling = True
        try:
            self._cal_dark_mean.setValue(0.0)
            self._cal_dark_sigma.setValue(1.0)
            self._cal_peak.setValue(200.0)
            self._cal_total.setValue(10000.0)
            self._cal_full_scale.setValue(255.0)
            self._cal_saturated.setValue(0.0)
            self._cal_centroid_sigma.setValue(0.5)
            self._cal_score_sigma.setValue(0.01)
        finally:
            self._cal_filling = False
        self._cal_recommended = None
        self._refresh_calibration_preview()
        self._append_log("LockConfig 标定输入已恢复默认。")

    def on_capture_calibration(self, *_args: Any) -> None:
        if self._running or self._cal_thread is not None:
            return
        confirm = QMessageBox.question(
            self,
            "标定将占用设备",
            "采集标定帧会独占 SLM 与 CCD 约数秒，期间请勿启动其他占用同一设备的进程。\n"
            "采集用零相位作为已知起点，不改目标设置；请先把曝光调到峰值不满量程，"
            "否则饱和像素会污染噪声与峰值标定。\n是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm is not QMessageBox.StandardButton.Yes:
            return
        xy, target = self._calibration_target_and_grid()
        thread = _CalibrationThread(
            self._make_bench(),
            xy=xy,
            target=target,
            frame_count=8,
            grid_size=self._grid_size.value(),
            parent=self,
        )
        thread.signals.done.connect(self._on_calibration_done)
        thread.signals.failed.connect(self._on_calibration_failed)
        self._cal_thread = thread
        self._cal_capture_button.setEnabled(False)
        self._append_log("标定：正在采集 8 帧，设备已占用。")
        thread.start()

    def _on_calibration_done(self, sample: CalibrationSample) -> None:
        self._cal_filling = True
        try:
            self._cal_dark_mean.setValue(sample.dark_mean)
            self._cal_dark_sigma.setValue(sample.dark_sigma)
            self._cal_peak.setValue(sample.peak_counts)
            self._cal_total.setValue(sample.total_counts)
            self._cal_full_scale.setValue(sample.full_scale)
            self._cal_saturated.setValue(sample.saturated_fraction * 100.0)
            self._cal_centroid_sigma.setValue(sample.centroid_sigma_px)
            self._cal_score_sigma.setValue(sample.score_sigma)
        finally:
            self._cal_filling = False
        self._clear_cal_thread()
        self._cal_capture_button.setEnabled(True)
        self._append_log("标定帧采集完成，已回填统计量。")
        self.on_calibrate()

    def _on_calibration_failed(self, message: str) -> None:
        self._clear_cal_thread()
        self._cal_capture_button.setEnabled(True)
        self._append_log(f"标定采集失败：{message}")
        QMessageBox.warning(self, "标定采集失败", message)

    def _clear_cal_thread(self) -> None:
        thread = self._cal_thread
        self._cal_thread = None
        if thread is not None:
            thread.deleteLater()

    # -- 故障横幅 ---------------------------------------------------------- #

    def _show_fault(self, reason: str) -> None:
        self._fault_banner.setText(f"已冻结：{reason}")
        self._fault_banner.show()
        self._append_log(f"冻结：{reason}")

    def _clear_fault(self) -> None:
        self._fault_banner.clear()
        self._fault_banner.hide()

    def _build_adaptation(self) -> QGroupBox:
        """transfer 在线适配的显式配置：检查点、会话输出目录、标定数据集。

        默认关闭。三个路径在勾选后就开始校验（只读，不建目录、不复制文件、
        不导入 torch），这样"缺哪一项"在占用设备之前就能看见。
        """
        box = QGroupBox("transfer 在线适配（默认关闭，不降低训练/验证门槛）")
        form = QFormLayout(box)

        self._adapt = QCheckBox("启用 transfer 在线适配")
        self._adapt.setChecked(False)
        self._adapt.toggled.connect(lambda _checked: self._refresh_adapt_status())
        form.addRow("", self._adapt)

        self._checkpoint_edit = QLineEdit()
        self._checkpoint_edit.setPlaceholderText("frozen_parameters.json 或其所在目录")
        self._checkpoint_edit.textChanged.connect(
            lambda _text: self._refresh_adapt_status()
        )
        self._checkpoint_button = QPushButton("浏览…")
        self._checkpoint_button.clicked.connect(self.on_browse_checkpoint)
        checkpoint_row = QHBoxLayout()
        checkpoint_row.addWidget(self._checkpoint_edit, 1)
        checkpoint_row.addWidget(self._checkpoint_button)
        form.addRow("检查点", checkpoint_row)

        self._session_dir_edit = QLineEdit()
        self._session_dir_edit.setPlaceholderText("本场采集与重拟合样本的输出目录")
        self._session_dir_edit.textChanged.connect(
            lambda _text: self._refresh_adapt_status()
        )
        self._session_dir_button = QPushButton("浏览…")
        self._session_dir_button.clicked.connect(self.on_browse_session_dir)
        session_row = QHBoxLayout()
        session_row.addWidget(self._session_dir_edit, 1)
        session_row.addWidget(self._session_dir_button)
        form.addRow("会话输出目录", session_row)

        self._source_dataset_edit = QLineEdit()
        self._source_dataset_edit.setPlaceholderText(
            "既有采集数据集：reference_start + x/y ±ramp 行"
        )
        self._source_dataset_edit.textChanged.connect(
            lambda _text: self._refresh_adapt_status()
        )
        self._source_dataset_button = QPushButton("浏览…")
        self._source_dataset_button.clicked.connect(self.on_browse_source_dataset)
        source_row = QHBoxLayout()
        source_row.addWidget(self._source_dataset_edit, 1)
        source_row.addWidget(self._source_dataset_button)
        form.addRow("标定数据集", source_row)

        self._span.valueChanged.connect(lambda _value: self._refresh_adapt_status())
        self._centre_x.valueChanged.connect(lambda _value: self._refresh_adapt_status())
        self._centre_y.valueChanged.connect(lambda _value: self._refresh_adapt_status())

        self._adapt_hint = QLabel("")
        self._adapt_hint.setWordWrap(True)
        form.addRow("", self._adapt_hint)

        policy = AdaptationPolicy()
        note = QLabel(
            "门槛仍由 AdaptationPolicy 决定：至少 "
            f"{policy.min_train_rows} 个训练行 + {policy.min_validation_rows} 个独立验证行，"
            f"且验证分数须优于在位检查点 {policy.promotion_margin} 才会晋级；"
            "样本不足时保持旧检查点、只报告原因。"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #888;")
        form.addRow("", note)

        self._refresh_adapt_status()
        return box

    # -- 适配路径 ---------------------------------------------------------- #

    def _adaptation_paths(self) -> tuple[str, str, str]:
        return (
            self._checkpoint_edit.text().strip(),
            self._session_dir_edit.text().strip(),
            self._source_dataset_edit.text().strip(),
        )

    def _adaptation_input_error(self) -> str | None:
        """只读校验适配三件套；返回第一条问题，全部合格返回 ``None``。

        门槛（训练/验证行数、验证分数）不在这里，也不因为这里通过而放宽：
        只检查"路径存在、检查点可读、数据集带 reference_start 与 x/y ±ramp"。
        """
        checkpoint_text, session_text, source_text = self._adaptation_paths()

        if not checkpoint_text:
            return "请填写 transfer 检查点（frozen_parameters.json 或其所在目录）。"
        checkpoint = Path(checkpoint_text).expanduser()
        if checkpoint.is_file():
            if checkpoint.suffix.lower() != ".json":
                return f"检查点文件应为 JSON：{checkpoint}"
            try:
                json.loads(checkpoint.read_text(encoding="utf-8"))
            # policy: 这是表单预检；异常文本原样拼进返回的错误串，操作员看得到失败原因，
            # 所以宽泛捕获不会让失败变成静默的通过。
            except Exception as exc:
                return f"检查点 JSON 读取失败：{exc}"
        elif checkpoint.is_dir():
            standalone = checkpoint / "frozen_parameters.json"
            if not standalone.is_file() and not (checkpoint / "config.json").is_file():
                return (
                    f"{checkpoint} 既没有 frozen_parameters.json，"
                    "也没有 config.json 检查点链。"
                )
            if standalone.is_file():
                try:
                    json.loads(standalone.read_text(encoding="utf-8"))
                # policy: 同上一条检查点预检；解析失败原样回报，不存在被吞掉的失败。
                except Exception as exc:
                    return f"frozen_parameters.json 读取失败：{exc}"
        else:
            return f"检查点不存在：{checkpoint}"

        if not session_text:
            return "请填写会话输出目录（本场采集与重拟合样本会写到这里）。"
        session = Path(session_text).expanduser()
        if session.exists() and not session.is_dir():
            return f"会话输出目录已被同名文件占用：{session}"
        if checkpoint.is_dir() and session.resolve() == checkpoint.resolve():
            return "会话输出目录不能与检查点目录相同：晋级检查点会被会话文件混杂。"
        if source_text and session.resolve() == Path(source_text).expanduser().resolve():
            return "会话输出目录不能与标定数据集相同：新样本会写进标定源。"

        if not source_text:
            return "请填写既有标定数据集（需含 reference_start 与 x/y ±ramp 行）。"
        source = Path(source_text).expanduser()
        if not source.is_dir():
            return f"标定数据集目录不存在：{source}"
        manifest = source / "manifest.json"
        if not manifest.is_file():
            return f"{source} 缺 manifest.json，不是采集数据集。"
        try:
            rows = json.loads(manifest.read_text(encoding="utf-8"))
        # policy: 数据集预检；解析失败的消息直接返回给操作员，不静默继续。
        except Exception as exc:
            return f"manifest.json 无法解析：{exc}"
        if not isinstance(rows, list):
            return "manifest.json 顶层必须是按行记录的 JSON 数组。"

        row_dicts = [row for row in rows if isinstance(row, dict)]
        if len(row_dicts) != len(rows):
            return "manifest.json 的每一行都必须是 JSON 对象。"
        labels = {str(row.get("label", "")) for row in row_dicts}
        if "reference_start" not in labels:
            return "标定数据集缺 reference_start 行：适配原点无从确定。"
        reference = next(
            row for row in row_dicts if row.get("label") == "reference_start"
        )
        if _spot_of(reference) is None:
            return "reference_start 行缺实测光斑（spot_xy 或 spot.centroid_x/y）：适配网格没有中心。"
        for axis in ("x", "y"):
            signs_by_period: dict[str, set[str]] = {}
            for label in labels:
                parts = label.split("_")
                if len(parts) == 4 and parts[0] == "ramp" and parts[1] == axis:
                    signs_by_period.setdefault(parts[2], set()).add(parts[3])
            if not any({"1", "-1"} <= found for found in signs_by_period.values()):
                return (
                    f"标定数据集缺 {axis} 轴 ±ramp 成对行"
                    "（同一 period 的 +1 与 -1 各一行）。"
                )

        usable = [
            row
            for row in row_dicts
            if row.get("label") in {"reference_start", "reference_end"}
            or str(row.get("label", "")).startswith(("ramp_", "probe_"))
        ]
        for row in usable:
            name = row.get("file")
            if not name:
                return f"manifest 行 {row.get('label')!r} 缺 file 字段。"
            if not (source / str(name)).is_file():
                return f"标定数据集缺载荷文件：{name}"
        return None

    def _reference_centre_warning(self) -> str | None:
        """分析窗口中心与数据集 reference_start 实测点不同心时给出提醒。

        预测网格由 ``transfer.fit.load_dataset`` 以 reference_start 的实测光斑
        为中心建立；闭环分析窗口却以操作员填的相机坐标为原点。两者不同心时，
        预测与实测量的不是同一块像素 —— 这是只靠"形状对得上"发现不了错位的地方。
        """
        _, _, source_text = self._adaptation_paths()
        if not source_text:
            return None
        try:
            rows = json.loads(
                (Path(source_text).expanduser() / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            reference = next(
                row for row in rows
                if isinstance(row, dict) and row.get("label") == "reference_start"
            )
            spot = _spot_of(reference)
        # policy: 这是「提示型」检查——读不到数据集里的 reference_start 只是没有可提示的
        # 内容，返回 None 让检查静默跳过；它不产出任何实验数值，跳过不会让数字变错。
        except Exception:
            return None
        if spot is None:
            return None
        centre = np.array([self._centre_x.value(), self._centre_y.value()], dtype=float)
        offset = float(np.hypot(*(spot - centre)))
        if offset <= 1.0:
            return None
        return (
            f"注意：光斑中心 ({centre[0]:.0f}, {centre[1]:.0f}) 与数据集 reference_start "
            f"实测点 ({spot[0]:.0f}, {spot[1]:.0f}) 相差 {offset:.1f} px。预测网格以实测点"
            "为中心，分析窗口与它不同心会让预测与实测错位。"
        )

    def _refresh_adapt_status(self) -> None:
        """把适配是否可用、还缺什么，就地显示在配置组里。"""
        if not self._adapt.isChecked():
            self._adapt_hint.setText("未启用：闭环只做实测接受/回滚，不训练 transfer 模型。")
            self._adapt_hint.setStyleSheet("color: #888;")
            return
        if self._adapter_factory is not None:
            self._adapt_hint.setText(
                "已启用：适配器由注入的工厂构造（测试或自定义接入），跳过路径校验。"
            )
            self._adapt_hint.setStyleSheet("color: #888;")
            return
        problem = self._adaptation_input_error()
        if problem:
            self._adapt_hint.setText(f"路径未通过校验：{problem}")
            self._adapt_hint.setStyleSheet("color: #c0392b;")
            return
        hint = (
            f"路径校验通过：分析窗口 span={self._span.value()}（与适配共用），"
            "开始时把标定行复制进会话目录；训练/验证门槛不变。"
        )
        warning = self._reference_centre_warning()
        if warning:
            self._adapt_hint.setText(f"{hint}\n{warning}")
            self._adapt_hint.setStyleSheet("color: #b26a00;")
            return
        self._adapt_hint.setText(hint)
        self._adapt_hint.setStyleSheet("color: #2e7d32;")

    def _make_adapter(self, xy: np.ndarray) -> Any:
        """构造适配器：注入的工厂优先；默认路径构造真实 OnlineTransferAdapter。

        真实构造只复制标定行到会话目录（stdlib 的 JSON/文件操作），torch 仍要
        等到 worker 里第一次 ``refit`` / ``forward`` 才导入。
        """
        if self._adapter_factory is not None:
            return self._adapter_factory(xy)
        checkpoint, session_dir, source_dataset = self._adaptation_paths()
        adapter = OnlineTransferAdapter(
            checkpoint=checkpoint,
            session_dir=session_dir,
            span=self._span.value(),
            source_dataset=source_dataset,
        )
        if not adapter.seeded:
            # 适配是操作员显式启用的：开场标定没接上就不该悄悄降级成"只做实测"。
            raise RuntimeError(adapter.seed_error or "开场标定不可用，适配无法建立")
        return adapter

    # -- 文件选择 ---------------------------------------------------------- #

    def on_browse_checkpoint(self) -> None:
        start = self._checkpoint_edit.text().strip() or str(Path.home())
        chosen, _ = QFileDialog.getOpenFileName(
            self, "选择 transfer 检查点", start, "检查点 (*.json);;所有文件 (*)"
        )
        if chosen:
            self._checkpoint_edit.setText(chosen)

    def on_browse_session_dir(self) -> None:
        start = self._session_dir_edit.text().strip() or str(Path.home())
        chosen = QFileDialog.getExistingDirectory(self, "选择会话输出目录", start)
        if chosen:
            self._session_dir_edit.setText(chosen)

    def on_browse_source_dataset(self) -> None:
        start = self._source_dataset_edit.text().strip() or str(Path.home())
        chosen = QFileDialog.getExistingDirectory(self, "选择既有标定数据集", start)
        if chosen:
            self._source_dataset_edit.setText(chosen)

    # -- 控件动作 ---------------------------------------------------------- #

    def on_start(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        xy = analysis_grid(
            (self._centre_x.value(), self._centre_y.value()), self._span.value()
        )
        target = target_field(
            xy,
            centre_xy=(self._centre_x.value(), self._centre_y.value()),
            half_width=self._half_width.value(),
            half_height=self._half_height.value(),
        )
        start_phase = np.zeros((self._grid_size.value(), self._grid_size.value()))
        try:
            lock_config = self._lock_config_for_run(xy)
        except ValueError as exc:
            self._append_log(f"标定阈值未应用，未占用设备：{exc}")
            QMessageBox.warning(self, "标定输入无效", str(exc))
            return
        mode = self._run_mode()
        hold = mode == "hold"
        config = ClosedLoopConfig(
            xy=xy,
            target=target,
            start_phase=start_phase,
            max_rounds=self._rounds.value(),
            mode=mode,
            hold_target_ccd_px=self._hold_tolerance.value(),
            lock_config=lock_config,
            # 目标保持无人值守，核心层要求 auto；逐轮优化沿用人工确认开关。
            candidate_policy="auto" if hold or self._auto_accept.isChecked() else "confirm",
            adapt=self._adapt.isChecked(),
            exposure_ms=self._exposure_ms.value(),
        )

        # 适配配置在**占用设备之前**校验：缺检查点/会话目录/标定数据集时
        # 直接停在这里，而不是先开 SLM/CCD 再在 worker 里报错。
        if config.adapt and self._adapter_factory is None:
            problem = self._adaptation_input_error()
            if problem:
                self._refresh_adapt_status()
                self._append_log(f"适配配置未通过校验，未占用设备：{problem}")
                QMessageBox.warning(self, "适配配置未通过校验", problem)
                return

        mode_text = "目标保持（全程闭环）" if hold else "逐轮优化"
        cap_text = f"运行轮数上限 {config.max_rounds} 轮" if hold else f"最多 {config.max_rounds} 轮"
        confirm_text = (
            "闭环将独占 SLM 与 CCD，期间请勿启动其他占用同一设备的进程。\n"
            f"固定目标中心（CCD 像素 x、y）= ({self._centre_x.value():.1f}, "
            f"{self._centre_y.value():.1f})，矩形半宽/半高 = "
            f"{self._half_width.value():.1f}/{self._half_height.value():.1f} px；"
            "目标在本次运行中不变。\n"
            f"运行方式：{mode_text}；{cap_text}，曝光 {config.exposure_ms} ms。"
        )
        if hold:
            confirm_text += f"\n位置容差 {config.hold_target_ccd_px:.1f} CCD px（固定像素参考，不重新居中）。"
            confirm_text += (
                "\n锁定阈值：使用标定值。" if lock_config else "\n锁定阈值：默认（未标定）。"
            )
        if config.adapt and self._adapter_factory is None:
            session = self._session_dir_edit.text().strip()
            confirm_text += (
                f"\n已启用 transfer 适配：标定行与会话样本都会写进 {session}。"
            )
        confirm_text += "确认开始？"
        reply = QMessageBox.question(
            self,
            "确认占用设备",
            confirm_text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            self._append_log("已取消：未打开任何设备。")
            return

        try:
            # 先建适配器（只碰文件），再开设备：标定接不上时不该已经占着 SLM/CCD。
            adapter = self._make_adapter(xy) if config.adapt else None
            bench = self._make_bench()
            forward = None
            if self._forward_factory is not None:
                forward = self._forward_factory(xy)
            runner = ClosedLoopRunner(
                bench, config=config, forward=forward, adapter=adapter,
                on_event=self._on_runner_event,
            )
        except Exception as exc:  # 构造失败要给用户可读结论
            self._append_log(f"初始化失败（未开始）：{exc}")
            QMessageBox.warning(self, "无法开始", str(exc))
            return

        self._thread = _RunnerThread(runner, self)
        self._thread.signals.event.connect(self._on_event)
        self._thread.signals.finished.connect(self._on_finished)
        self._thread.signals.failed.connect(self._on_failed)
        self._set_controls_running(True)
        self._clear_rounds()
        self._clear_fault()
        if hold:
            self._phase_label.setText(f"收敛中 · 第 1/{config.max_rounds} 轮")
        self._append_log("已确认设备占用，闭环开始。")
        self._thread.start()

    def _make_bench(self) -> Any:
        if self._bench_factory is not None:
            return self._bench_factory()
        from ao_shaping.gui.slm.closed_loop_bench import HardwareBench

        return HardwareBench(exposure_ms=self._exposure_ms.value())

    def on_accept(self) -> None:
        if self._run_mode() == "hold":
            return
        if self._thread is not None and self._thread.isRunning():
            self._thread.runner.request_accept()

    def on_reject(self) -> None:
        if self._run_mode() == "hold":
            return
        if self._thread is not None and self._thread.isRunning():
            self._thread.runner.request_reject()

    def on_pause_resume(self) -> None:
        if self._thread is None or not self._thread.isRunning():
            return
        runner = self._thread.runner
        if runner.state is LoopState.PAUSED:
            runner.request_resume()
            self._pause_button.setText("暂停")
        else:
            runner.request_pause()
            self._pause_button.setText("继续")

    def on_stop(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            self._thread.runner.request_stop()

    # -- 事件渲染 ---------------------------------------------------------- #

    def _on_runner_event(self, event: LoopEvent) -> None:
        """worker 线程调用：只往 Qt 信号里塞数据。"""
        thread = self._thread
        if thread is not None:
            thread.signals.event.emit(event)

    def _on_event(self, event: LoopEvent) -> None:
        if event.state is not None:
            self._state_label.setText(
                f"状态：{_STATE_TEXT.get(event.state, event.state.value)}"
            )
        if event.kind == "lock":
            self._render_lock(event.payload)
        elif event.kind == "round":
            self._append_round(event.payload)
        elif event.kind == "bench":
            self._append_log(f"设备：{event.payload}")
        elif event.kind == "awaiting_decision":
            if self._run_mode() == "hold":
                # 保持模式无人值守；旧事件即便到达也不该把页面拉回逐轮确认。
                self._append_log(
                    f"第 {event.payload.round_index} 轮到达逐轮确认事件，目标保持下忽略。"
                )
            else:
                self._decision_hint.setText(
                    f"第 {event.payload.round_index} 轮实测分数 "
                    f"{event.payload.measured_score:.4f}，请确认接受或拒绝。"
                )
                self._append_log(f"第 {event.payload.round_index} 轮等待人工确认。")
        elif event.kind == "adaptation":
            self._append_log(
                f"transfer 适配：{event.payload.status} — {event.payload.detail}"
            )
        elif event.kind == "state":
            self._append_log(str(event.payload))

    @staticmethod
    def _payload_value(payload: Any, key: str, default: Any = None) -> Any:
        """lock 事件的 payload 既可能是 dict 也可能是对象；缺键一律走默认值。"""
        if isinstance(payload, dict):
            return payload.get(key, default)
        return getattr(payload, key, default)

    def _render_lock(self, payload: Any) -> None:
        """把目标保持的相位事件画成一行结论：收敛中 → 维持中 → 已冻结。"""
        state = str(self._payload_value(payload, "lock_state", "") or "")
        action = str(self._payload_value(payload, "action", "") or "")
        index = self._payload_value(payload, "round_index")
        cap = self._rounds.value()
        phase = {
            "converging": "收敛中",
            "maintaining": "维持中",
            "frozen": "已冻结",
        }.get(state, "")
        if phase:
            if index is not None and state != "frozen":
                phase = f"{phase} · 第 {index}/{cap} 轮"
            self._phase_label.setText(phase)
        if state == "frozen" or action == "fault":
            reason = str(self._payload_value(payload, "reason", "") or "")
            if not reason:
                error = self._payload_value(payload, "centroid_error_ccd_px")
                try:
                    value = float(error) if error is not None else float("nan")
                except (TypeError, ValueError):
                    value = float("nan")
                if np.isfinite(value):
                    prefix = f"第 {index} 轮" if index is not None else "闭环"
                    reason = f"{prefix}位置偏差 {value:.1f} CCD px"
                else:
                    reason = "闭环进入冻结状态，未给出原因。"
            self._show_fault(reason)

    def _append_round(self, record: Any) -> None:
        self._last_record = record
        row = self._table.rowCount()
        self._table.insertRow(row)
        values = [
            str(record.round_index),
            "通过" if record.gate.ok else describe_gate(record.gate),
            f"{record.measured_score:.4f}" if record.measured_score is not None else "—",
            f"{record.predicted_score:.4f}" if record.predicted_score is not None else "—",
            f"{record.phase_step_rms:.4f}" if record.phase_step_rms is not None else "—",
            f"{record.window_fraction:.4f}" if record.window_fraction is not None else "—",
            _DECISION_TEXT.get(record.decision, str(record.decision)),
        ]
        for col, value in enumerate(values):
            item = QTableWidgetItem(value)
            if not record.gate.ok or record.decision == "gate_fail":
                item.setForeground(Qt.GlobalColor.red)
            self._table.setItem(row, col, item)
        self._table.scrollToBottom()

    def _on_finished(self, summary: LoopSummary) -> None:
        self._state_label.setText(f"状态：{_STATE_TEXT.get(summary.state, summary.state.value)}")
        self._set_controls_running(False)
        self._decision_hint.clear()
        self._phase_label.setText("已结束")
        if summary.state is LoopState.FAULT:
            self._show_fault(summary.reason or "闭环以故障结束。")
        else:
            self._clear_fault()
        self._append_log(f"闭环结束（{summary.state.value}）：{summary.reason}")
        self._clear_thread()

    def _on_failed(self, message: str) -> None:
        self._state_label.setText("状态：故障")
        self._phase_label.setText("已冻结")
        self._set_controls_running(False)
        self._show_fault(message)
        self._append_log(f"闭环异常：{message}")
        QMessageBox.warning(self, "闭环异常", message)
        self._clear_thread()

    def _set_controls_running(self, running: bool) -> None:
        self._running = running
        for widget in (
            self._start_button, self._accept_button, self._reject_button,
            self._pause_button, self._stop_button,
        ):
            widget.setEnabled(running if widget is not self._start_button else not running)
        # 适配路径与开关只在两场闭环之间可改：跑起来后换会话目录会让样本写错地方。
        for widget in (
            self._adapt, self._checkpoint_edit, self._checkpoint_button,
            self._session_dir_edit, self._session_dir_button,
            self._source_dataset_edit, self._source_dataset_button,
        ):
            widget.setEnabled(not running)
        # 目标与运行方式开跑后冻结：目标在本次运行中不允许改变，位置参考不重居中。
        for widget in (
            self._mode_combo, self._span, self._centre_x, self._centre_y,
            self._half_width, self._half_height, self._rounds, self._grid_size,
        ):
            widget.setEnabled(not running)
        # 标定面板开跑后整体冻结：不能和闭环抢设备，也不能中途换阈值口径。
        for widget in (
            self._cal_dark_mean, self._cal_dark_sigma, self._cal_peak, self._cal_total,
            self._cal_full_scale, self._cal_saturated, self._cal_centroid_sigma,
            self._cal_score_sigma, self._cal_calc_button, self._cal_reset_button,
            self._cal_apply,
        ):
            widget.setEnabled(not running)
        self._cal_capture_button.setEnabled(not running and self._cal_thread is None)
        self._pause_button.setText("暂停")
        # 模式相关（位置容差、确认按钮可见性）最后统一收口。
        self._apply_mode_ui()

    def _clear_rounds(self) -> None:
        self._table.setRowCount(0)
        self._decision_hint.clear()

    def _append_log(self, text: str) -> None:
        self._log.appendPlainText(text)

    def _clear_thread(self) -> None:
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.deleteLater()

    def closeEvent(self, event: Any) -> None:  # Qt 命名
        self._closing = True
        thread = self._thread
        if thread is not None and thread.isRunning():
            thread.runner.request_stop()
            thread.wait(5000)
        cal_thread = self._cal_thread
        if cal_thread is not None and cal_thread.isRunning():
            cal_thread.wait(5000)
        super().closeEvent(event)
