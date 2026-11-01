"""光束质量分析主窗口 (PySide6 + QtCharts)。

布局:

* 左: 控制面板 (源选择 / cam_id / 打开关闭 / 曝光 (手动 + 自动) / 多帧平均 / 快照);
* 中: 视频 (上) + X/Y 剖面 (下), 竖向 ``QSplitter``;
* 右: 最大亮度趋势图 (主视图, 时间窗可切换)。

线程与背压: 帧由源在 SDK/仿真线程里推进容量 1 的信箱, UI 线程用 30 Hz
``QTimer`` 取"最新一帧" —— 处理不过来时丢中间帧而不是排队, 因此显示延迟有界。
分析在 UI 线程同步执行 (单帧 ~10 ms; 超过 1080p 的画面先 2x2 binning 再分析,
计算量降为 1/4, 坐标再用 ``FrameMetrics.rescaled(2.0)`` 还原)。
生命周期操作 (打开/关闭/改曝光) 是 MiiCam 真机秒级阻塞 (~10-16s),
一律下放到单线程 ``QThreadPool`` 串行执行, UI 保持响应, 忙碌期间按钮禁用。

自动曝光是同一条通道上的闭环: 每 ``AUTO_EXPOSURE_INTERVAL_MS`` 取最新一帧原始画面
(不是多帧平均 —— 平均窗口里可能还是上一次曝光下的帧) 的一个统计量, 与目标灰度比较
后把曝光往目标推一步; 走的仍是 ``apply_exposure`` 这条后台作业, 因此不会阻塞 UI,
也不会和手动改曝光打架 (自动模式下手动曝光控件禁用)。

多帧平均 (1~10 帧) 只作用于显示与特征计算: 曝光一变就清空窗口, 免得把两个曝光下
的画面平均成一个不代表任何时刻的亮度。
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger
from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Qt, Signal
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from ao_shaping.gui.ccd.beam_quality.analyzer import (
    FrameMetrics,
    analyze_frame,
    bin_2x2,
)
from ao_shaping.gui.ccd.beam_quality.exposure_control import (
    AUTO_STATISTIC_KEYS,
    AUTO_STATISTICS,
    DEFAULT_TARGETS,
    FrameAverage,
    frame_statistic,
    is_converged,
    next_exposure,
)
from ao_shaping.gui.ccd.beam_quality.frame_source import (
    FrameSource,
    CameraFrameSource,
    SimBeamSource,
)
from ao_shaping.gui.ccd.beam_quality.profile_widget import ProfileWidget
from ao_shaping.gui.ccd.beam_quality.trend_buffer import TrendBuffer
from ao_shaping.gui.ccd.beam_quality.trend_widget import TrendWidget
from ao_shaping.gui.ccd.beam_quality.video_widget import VideoWidget, to_uint8

# 源类型: (显示名, 内部键)
SOURCE_KINDS: tuple[tuple[str, str], ...] = (
    ("MiiCam", "miicam"),
    ("仿真源 (离线)", "sim"),
)

ANALYSIS_MAX_WIDTH = 1920
ANALYSIS_MAX_HEIGHT = 1080


class _JobSignals(QObject):
    """QRunnable 无法自带信号, 用伴随 QObject 桥接完成事件。"""

    # (ok, result_or_error); 经 Qt 队列连接回到 UI 线程
    done = Signal(bool, object)


class _Job(QRunnable):
    """把阻塞调用 (相机 open/stop/改曝光) 挪出 UI 线程。

    MiiCam 的硬件操作是真机秒级阻塞 (open ~16s, 改曝光 ~10s),
    同步调在 UI 线程会把整个窗口冻住 (2026-10-31 真机实测)。
    """

    def __init__(self, fn) -> None:
        super().__init__()
        self._fn = fn
        self.signals = _JobSignals()

    def run(self) -> None:
        try:
            self.signals.done.emit(True, self._fn())
        except Exception as exc:  # policy: 结果统一回 UI 线程处理, 工作线程不碰界面
            self.signals.done.emit(False, exc)


class BeamQualityApp(QMainWindow):
    """光束质量分析主窗口。"""

    POLL_INTERVAL_MS = 33  # ~30 Hz 取帧
    TREND_INTERVAL_MS = 200  # ~5 Hz 重绘趋势
    TREND_SAMPLE_INTERVAL_S = 0.1  # 趋势采样 10 Hz (3600 点 ≈ 6 min)
    TREND_MAXLEN = 7200
    AUTO_EXPOSURE_INTERVAL_MS = 1200  # 自动曝光闭环周期 (改一次曝光 ~1s)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("光束质量分析 (MiiCam)")
        self.resize(1480, 900)

        self._source: FrameSource | None = None
        self._buffer = TrendBuffer(maxlen=self.TREND_MAXLEN)
        self._last_frame: np.ndarray | None = None
        #: 最新一帧的采集时刻 (源侧 monotonic 时钟), 自动曝光用它判断反馈是否够新
        self._last_frame_ts = 0.0
        #: 最近一次改曝光完成的时刻; 早于它的帧仍属旧曝光, 不能当闭环反馈
        self._exposure_applied_at = 0.0
        #: 最近一次喂给分析/显示的画面 (多帧平均后的; 1 帧模式下与 _last_frame 同一引用)
        self._last_view: np.ndarray | None = None
        self._averager = FrameAverage(1)
        self._auto_exposure = False
        #: 自动曝光最近一步的说明 (信息栏显示, 状态栏留给硬件操作的即时反馈)
        self._auto_hint = ""
        self._last_metrics: FrameMetrics | None = None
        self._last_append_ts = 0.0
        self._processed = 0
        self._ui_fps = 0.0
        self._ui_fps_count = 0
        self._ui_fps_since = time.monotonic()

        # 阻塞型硬件操作 (open/close/改曝光) 串行下到这个单线程池,
        # UI 线程只收发结果; _busy 期间相关按钮禁用, 防止并发操作同一相机。
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._busy = False
        # 在飞任务的强引用 —— QRunnable autoDelete 后 Python 侧信号对象若被
        # GC, 排队中的 done 信号会被 Qt 随发送者一并丢弃, 任务"永远不完成"。
        self._jobs: set[_Job] = set()

        self._build_ui()
        self._set_controls_enabled(True)

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(self.POLL_INTERVAL_MS)
        self._poll_timer.timeout.connect(self._on_poll)

        self._trend_timer = QTimer(self)
        self._trend_timer.setInterval(self.TREND_INTERVAL_MS)
        self._trend_timer.timeout.connect(self._on_trend_tick)

        self._auto_timer = QTimer(self)
        self._auto_timer.setInterval(self.AUTO_EXPOSURE_INTERVAL_MS)
        self._auto_timer.timeout.connect(self._on_auto_exposure_tick)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        central = QWidget(self)
        outer = QHBoxLayout(central)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        outer.addWidget(self._build_controls_scroll(), 0)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self._video = VideoWidget()
        self._profile = ProfileWidget()
        splitter.addWidget(self._video)
        splitter.addWidget(self._profile)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([560, 340])
        outer.addWidget(splitter, 1)

        self._trend = TrendWidget(self._buffer)
        self._trend.setMinimumWidth(430)
        outer.addWidget(self._trend, 0)

        self.setCentralWidget(central)

        self._stats_label = QLabel("就绪")
        self.statusBar().addPermanentWidget(self._stats_label)
        self.statusBar().showMessage("选择信号源后点击「打开」", 5000)

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(250)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        source_box = QGroupBox("相机源")
        form = QFormLayout(source_box)
        self._source_combo = QComboBox()
        for label, _key in SOURCE_KINDS:
            self._source_combo.addItem(label)
        self._cam_id_spin = QSpinBox()
        self._cam_id_spin.setRange(0, 15)
        self._cam_id_spin.setValue(0)
        form.addRow("类型", self._source_combo)
        form.addRow("cam_id", self._cam_id_spin)
        layout.addWidget(source_box)

        self._open_btn = QPushButton("打开")
        self._close_btn = QPushButton("关闭")
        self._close_btn.setEnabled(False)
        row = QHBoxLayout()
        row.addWidget(self._open_btn)
        row.addWidget(self._close_btn)
        layout.addLayout(row)

        exposure_box = QGroupBox("曝光")
        exposure_form = QFormLayout(exposure_box)
        self._exposure_spin = QDoubleSpinBox()
        self._exposure_spin.setRange(0.011, 10000.0)
        self._exposure_spin.setDecimals(3)
        self._exposure_spin.setSingleStep(1.0)
        self._exposure_spin.setValue(20.0)
        self._exposure_spin.setSuffix(" ms")
        self._apply_btn = QPushButton("应用曝光")
        self._auto_check = QCheckBox("自动曝光")
        self._auto_check.setToolTip("闭环调整曝光时间, 使下面选中的统计量逼近目标灰度")
        self._auto_stat_combo = QComboBox()
        for label, _key in AUTO_STATISTICS:
            self._auto_stat_combo.addItem(label)
        self._auto_target_spin = QDoubleSpinBox()
        self._auto_target_spin.setRange(1.0, 255.0)
        self._auto_target_spin.setDecimals(1)
        self._auto_target_spin.setSingleStep(5.0)
        self._auto_target_spin.setValue(DEFAULT_TARGETS[AUTO_STATISTIC_KEYS[0]])
        exposure_form.addRow("曝光", self._exposure_spin)
        exposure_form.addRow(self._apply_btn)
        exposure_form.addRow(self._auto_check, self._auto_stat_combo)
        exposure_form.addRow("目标灰度", self._auto_target_spin)
        layout.addWidget(exposure_box)

        average_box = QGroupBox("多帧平均")
        average_form = QFormLayout(average_box)
        self._avg_spin = QSpinBox()
        self._avg_spin.setRange(1, 10)
        self._avg_spin.setValue(1)
        self._avg_spin.setSuffix(" 帧")
        self._avg_spin.setToolTip("特征计算前先对最近 N 帧求平均 (1 = 单帧)")
        average_form.addRow("平均帧数", self._avg_spin)
        average_hint = QLabel("1 帧 = 不平均; 帧数越多噪声越低, UI 帧率随之下降")
        average_hint.setWordWrap(True)
        average_hint.setStyleSheet("color: #808088;")
        average_form.addRow(average_hint)
        layout.addWidget(average_box)

        self._snapshot_btn = QPushButton("保存快照 PNG")
        layout.addWidget(self._snapshot_btn)

        info_box = QGroupBox("信息")
        info_layout = QVBoxLayout(info_box)
        self._info_label = QLabel("分辨率: --\n曝光: --\n源帧率: --\n丢弃帧: --")
        self._info_label.setWordWrap(True)
        self._info_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        info_layout.addWidget(self._info_label)
        layout.addWidget(info_box)

        hint = QLabel(
            "分析口径: 半高阈值包围椭圆 (二阶矩)\n>1080p 画面先 2x2 binning 再分析"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #808088;")
        layout.addWidget(hint)
        layout.addStretch(1)

        self._open_btn.clicked.connect(self.open_source)
        self._close_btn.clicked.connect(self.close_source)
        self._apply_btn.clicked.connect(self.apply_exposure)
        self._auto_check.toggled.connect(self._set_auto_exposure)
        self._auto_stat_combo.currentIndexChanged.connect(self._on_auto_stat_changed)
        self._avg_spin.valueChanged.connect(self._on_average_changed)
        self._snapshot_btn.clicked.connect(self.save_snapshot)
        return panel

    def _build_controls_scroll(self) -> QScrollArea:
        """把控制面板装进竖向滚动区 (左栏)。

        # policy: 左栏控制项只增不减 (自动曝光 / 多帧平均 / 信息行), 窗口不够高时
        面板必须能滚动, 否则最下面的 信息 组和口径提示会被窗口底边直接裁掉。
        滚动区宽度 = 面板宽度 + 竖向滚动条宽度: 滚动条出现时内容不被横向挤窄;
        横向滚动条关闭, 无边框且背景透明, 外观与直接把面板放进布局时一致。
        """
        panel = self._build_controls()
        area = QScrollArea()
        area.setWidget(panel)
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        area.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        bar_width = area.style().pixelMetric(
            QStyle.PixelMetric.PM_ScrollBarExtent, None, area
        )
        area.setFixedWidth(panel.minimumWidth() + bar_width)
        # 滚动区与视口都不画自己的底色, 否则会露出 QScrollArea 默认的 Base 色块
        area.setStyleSheet("QScrollArea { background: transparent; }")
        area.viewport().setAutoFillBackground(False)
        return area

    # -------------------------------------------------------------- 源控制
    def _set_controls_enabled(self, enabled: bool) -> None:
        """按"是否忙碌 + 是否有源 + 是否自动曝光"刷新控件可用态。"""
        has_source = self._source is not None
        manual_exposure = enabled and has_source and not self._auto_exposure
        self._open_btn.setEnabled(enabled and not has_source)
        self._close_btn.setEnabled(enabled and has_source)
        self._apply_btn.setEnabled(manual_exposure)
        self._exposure_spin.setEnabled(manual_exposure)
        self._auto_check.setEnabled(enabled and has_source)
        self._auto_stat_combo.setEnabled(enabled and has_source)
        self._auto_target_spin.setEnabled(enabled and has_source)
        self._avg_spin.setEnabled(enabled)
        self._snapshot_btn.setEnabled(enabled)
        self._source_combo.setEnabled(enabled and not has_source)
        self._cam_id_spin.setEnabled(enabled and not has_source)

    def _start_job(self, label: str, fn, on_done) -> bool:
        """把阻塞硬件操作下放到单线程池串行执行; 完成回调在 UI 线程。

        忙碌期间拒绝新任务 (按钮已禁用, 这里是第二道闸)。
        """
        if self._busy:
            self.statusBar().showMessage("正忙, 请稍候…", 3000)
            return False
        self._busy = True
        self._set_controls_enabled(False)
        self.statusBar().showMessage(label)
        job = _Job(fn)
        job.setAutoDelete(False)
        self._jobs.add(job)

        def _finish(ok: bool, payload: object) -> None:
            self._jobs.discard(job)
            self._busy = False
            self._set_controls_enabled(True)
            on_done(ok, payload)

        job.signals.done.connect(_finish)
        self._pool.start(job)
        return True

    def open_source(self) -> None:
        """按当前选择启动帧源 (MiiCam 硬件打开真机 ~16s, 后台线程执行)。"""
        if self._source is not None or self._busy:
            return
        index = self._source_combo.currentIndex()
        kind = SOURCE_KINDS[index][1] if 0 <= index < len(SOURCE_KINDS) else "miicam"
        exposure = float(self._exposure_spin.value())
        cam_id = int(self._cam_id_spin.value())

        def _work() -> FrameSource:
            source: FrameSource
            if kind == "sim":
                source = SimBeamSource(exposure_ms=exposure)
            else:
                source = CameraFrameSource(
                    camera_type=kind, cam_id=cam_id, exposure_ms=exposure
                )
            source.start()
            return source

        def _done(ok: bool, payload: object) -> None:
            if not ok:
                # policy: 设备打不开是可预期状态 (未接相机/被占用), 状态栏报错, 不弹模态框
                logger.opt(exception=payload).error("打开帧源失败")
                self.statusBar().showMessage(f"打开失败: {payload}", 8000)
                return
            source = payload
            self._source = source
            self._last_append_ts = 0.0
            self._last_frame = None
            self._last_frame_ts = 0.0
            self._last_view = None
            self._averager.reset()
            self._poll_timer.start()
            self._trend_timer.start()
            if self._auto_exposure:
                self._auto_timer.start()
            self._set_controls_enabled(True)
            self.statusBar().showMessage(
                f"已启动 {SOURCE_KINDS[index][0]} (曝光 {source.exposure_ms:.3f} ms)",
                5000,
            )

        self._start_job(f"正在打开 {SOURCE_KINDS[index][0]}…", _work, _done)

    def close_source(self) -> None:
        """停止帧源并清画面 (MiiCam 停止真机 ~7-12s, 后台执行); 可重复调用。"""
        if self._busy:
            return
        self._poll_timer.stop()
        self._trend_timer.stop()
        self._auto_timer.stop()
        source = self._source
        self._source = None
        self._last_frame = None
        self._last_frame_ts = 0.0
        self._last_view = None
        self._averager.reset()
        self._video.clear()
        self._profile.clear()
        self._info_label.setText("分辨率: --\n曝光: --\n源帧率: --\n丢弃帧: --")
        self._set_controls_enabled(True)
        if source is None:
            self.statusBar().showMessage("已关闭", 3000)
            return

        def _work() -> None:
            source.stop()

        def _done(ok: bool, payload: object) -> None:
            if not ok:
                # policy: 关停失败只记录, 不能让关窗口抛异常留下僵尸界面
                logger.opt(exception=payload).warning("停止帧源失败")
            self.statusBar().showMessage("已关闭", 3000)

        self._start_job("正在关闭…", _work, _done)

    def apply_exposure(self) -> None:
        """把界面上的曝光值写给当前源 (MiiCam 重启拉流真机 ~10s, 后台执行)。"""
        source = self._source
        if source is None:
            self.statusBar().showMessage("先打开帧源再改曝光", 4000)
            return
        request = float(self._exposure_spin.value())

        def _work() -> float:
            return source.set_exposure_ms(request)

        def _done(ok: bool, payload: object) -> None:
            if not ok:
                # policy: 改曝光失败时设备状态未知, 状态栏报错并提示重新打开, 不弹模态框
                logger.opt(exception=payload).error("设置曝光失败")
                self.statusBar().showMessage(f"设置曝光失败: {payload}", 8000)
                return
            applied = float(payload)
            self._exposure_spin.setValue(applied)
            # 曝光一变, 之前攒的画面就是另一个曝光下的了: 平均窗口清空,
            # 并记下切换时刻 —— 自动曝光只接受此后的新帧当反馈。
            self._exposure_applied_at = time.monotonic()
            self._averager.reset()
            self._last_view = None
            self.statusBar().showMessage(f"曝光已设为 {applied:.3f} ms", 4000)

        self._start_job(f"正在设置曝光 {request:.3f} ms…", _work, _done)

    # ---------------------------------------------------------------- 自动曝光
    def _set_auto_exposure(self, on: bool) -> None:
        """开关自动曝光闭环; 自动模式下手动曝光控件让位给闭环。"""
        self._auto_exposure = bool(on)
        self._auto_hint = ""
        if self._auto_exposure:
            if self._source is not None and not self._busy:
                self._auto_timer.start()
            self.statusBar().showMessage(
                f"自动曝光已开启: {self._auto_stat_combo.currentText()} "
                f"→ {self._auto_target_spin.value():.0f}",
                5000,
            )
        else:
            self._auto_timer.stop()
            self.statusBar().showMessage("自动曝光已关闭", 3000)
        self._set_controls_enabled(not self._busy)

    def _on_auto_stat_changed(self, index: int) -> None:
        """换反馈统计量时把目标灰度切到该统计量的常用值。"""
        if 0 <= index < len(AUTO_STATISTIC_KEYS):
            self._auto_target_spin.setValue(DEFAULT_TARGETS[AUTO_STATISTIC_KEYS[index]])

    def _on_average_changed(self, value: int) -> None:
        """换平均帧数: 重建滚动平均, 之前攒的历史作废。"""
        self._averager = FrameAverage(int(value))
        self._last_view = None

    def _on_auto_exposure_tick(self) -> None:
        """UI 线程: 把曝光往目标推一步 (改曝光本身仍走后台作业)。

        反馈只取**最新一帧原始画面**, 不取多帧平均: 平均窗口里的帧可能来自上一次
        曝光, 拿它们当反馈会让闭环对着旧画面一步步推, 真机上会把曝光一路推到上限
        (2026-11-01 实测: 4 帧平均 + 90 灰度 → 10 s 曝光)。同理, 一张比上次改曝光
        更早拍到的帧也不接受 —— 那还是旧曝光下的画面。

        收敛或已贴到曝光上下限时不再动作 —— 否则每过一个周期就关一次流,
        白白丢掉几帧。
        """
        if not self._auto_exposure or self._busy:
            return
        source = self._source
        frame = self._last_frame
        if source is None or frame is None:
            return
        if self._last_frame_ts <= self._exposure_applied_at:
            # 等一张新曝光下的帧 (长曝光时可能等好几秒, 定时器下次再看)
            return
        index = min(
            max(self._auto_stat_combo.currentIndex(), 0), len(AUTO_STATISTIC_KEYS) - 1
        )
        kind = AUTO_STATISTIC_KEYS[index]
        label = AUTO_STATISTICS[index][0]
        try:
            stat = frame_statistic(frame, kind)
        # policy: 统计量算不出来 (空帧/换分辨率) 不该停掉闭环定时器, 等下一帧即可
        except (TypeError, ValueError):
            return
        target = float(self._auto_target_spin.value())
        wanted = next_exposure(
            stat,
            target,
            float(source.exposure_ms),
            min_ms=self._exposure_spin.minimum(),
            max_ms=self._exposure_spin.maximum(),
        )
        what = f"{label}={stat:.1f} (目标 {target:.0f})"
        if wanted is None:
            state = "已收敛" if is_converged(stat, target) else "已到曝光极限"
            self._auto_hint = f"自动曝光{state}: {what}"
            return
        self._auto_hint = f"自动曝光: {what} → {wanted:.3f} ms"
        self._exposure_spin.setValue(wanted)
        self.apply_exposure()

    def save_snapshot(self) -> None:
        """把最近一帧原始画面存成 PNG。"""
        frame = self._last_frame
        if frame is None:
            self.statusBar().showMessage("还没有画面可保存", 4000)
            return
        default_dir = Path("data") / "snapshots"
        default = default_dir / f"beam_{datetime.now():%Y%m%d_%H%M%S}.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "保存快照", str(default), "PNG 图像 (*.png)"
        )
        if not path:
            return
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        gray = to_uint8(frame)
        height, width = int(gray.shape[0]), int(gray.shape[1])
        image = QImage(
            gray.data,
            width,
            height,
            int(gray.strides[0]),
            QImage.Format.Format_Grayscale8,
        )
        if image.save(str(target), "PNG"):
            self.statusBar().showMessage(f"快照已保存: {target}", 5000)
        else:
            self.statusBar().showMessage(f"保存失败: {target}", 8000)

    # -------------------------------------------------------------- 周期任务
    def _on_poll(self) -> None:
        """UI 线程: 只处理最新一帧 → 分析 → 刷新视频/剖面/趋势缓冲。"""
        source = self._source
        if source is None:
            return
        drained = source.drain_latest()
        if drained is None:
            return
        img, timestamp = drained
        self._last_frame = img
        self._last_frame_ts = float(timestamp)
        # 多帧平均: 1 帧时 push 原样返回 (不含拷贝), N>1 时返回最近 N 帧均值
        self._last_view = self._averager.push(img)

        metrics = self._analyze(self._last_view)
        if metrics is not None:
            self._last_metrics = metrics
            self._video.set_frame(self._last_view, metrics, source.fps)
            self._profile.update_metrics(metrics)
            now = time.time()
            if now - self._last_append_ts >= self.TREND_SAMPLE_INTERVAL_S:
                self._buffer.append(now, metrics)
                self._last_append_ts = now

        self._processed += 1
        self._ui_fps_count += 1
        elapsed = time.monotonic() - self._ui_fps_since
        if elapsed >= 1.0:
            self._ui_fps = self._ui_fps_count / elapsed
            self._ui_fps_count = 0
            self._ui_fps_since = time.monotonic()
            self._update_stats(img)

    def _analyze(self, img: np.ndarray) -> FrameMetrics | None:
        """分析一帧; 超大靶面先 2x2 binning, 再把坐标换算回原始像素。"""
        try:
            arr = np.asarray(img)
            scale = 1.0
            if arr.shape[0] > ANALYSIS_MAX_HEIGHT or arr.shape[1] > ANALYSIS_MAX_WIDTH:
                arr = bin_2x2(arr)
                scale = 2.0
            metrics = analyze_frame(arr)
            return metrics.rescaled(scale) if scale != 1.0 else metrics
        # policy: 分析异常不该杀掉 30 Hz 定时器; 记录并让界面继续显示上一帧的结果
        except Exception:  # policy: 逐帧分析横跨 numpy/scipy, 异常类型不可枚举
            logger.exception("帧分析失败, 跳过该帧")
            return None

    def _on_trend_tick(self) -> None:
        if self._source is None:
            return
        self._trend.update_buffer()
        self._trend.refresh()

    def _update_stats(self, img: np.ndarray) -> None:
        source = self._source
        height, width = int(img.shape[0]), int(img.shape[1])
        exposure = source.exposure_ms if source is not None else float("nan")
        source_fps = source.fps if source is not None else 0.0
        dropped = getattr(source, "dropped_frames", None)
        lines = [
            f"分辨率: {width} x {height}",
            f"曝光: {exposure:.3f} ms",
            f"源帧率: {source_fps:.1f} fps",
            f"UI 处理: {self._ui_fps:.1f} fps",
            f"丢弃帧: {'--' if dropped is None else dropped}",
        ]
        if self._averager.size > 1:
            lines.append(
                f"多帧平均: {self._averager.size} 帧 (已攒 {self._averager.filled})"
            )
        if self._auto_exposure:
            lines.append(self._auto_hint or "自动曝光: 等待画面…")
        self._info_label.setText("\n".join(lines))
        self._stats_label.setText(
            f"已处理 {self._processed} 帧 | UI {self._ui_fps:.1f} fps"
        )

    # ------------------------------------------------------------------ 退出
    def closeEvent(self, event) -> None:  # Qt 覆写点 (驼峰命名由 Qt 决定)
        """关窗: 停定时器 → 停源 (退出 SDK 回调线程/仿真线程) → 再放行。"""
        try:
            self._close_source_for_exit()
        # policy: 退出路径上的任何异常都不能阻止窗口关闭, 否则进程挂住
        except Exception:  # policy: 见上
            logger.exception("窗口关闭时停止帧源出错")
        super().closeEvent(event)

    def _close_source_for_exit(self) -> None:
        self._poll_timer.stop()
        self._trend_timer.stop()
        self._auto_timer.stop()
        # 先等在飞的后台任务 (open/改曝光) 落地, 再停源 —— 否则停源与工作线程
        # 会并发操作同一相机。done 回调排在 UI 事件队列里 (它负责赋值
        # ``self._source``), 干等 waitForDone 会拿不到源, 必须 processEvents 驱动。
        deadline = time.monotonic() + 30.0
        while self._busy and time.monotonic() < deadline:
            QApplication.processEvents()
            time.sleep(0.02)
        source = self._source
        self._source = None
        if source is not None:
            source.stop()


def main(argv: list[str] | None = None) -> int:
    """启动光束质量分析界面 (``python -m ao_shaping.gui.ccd.beam_quality``)。"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(list(sys.argv) if argv is None else list(argv))
    app.setApplicationName("ao-beam-quality")
    window = BeamQualityApp()
    window.show()
    return int(app.exec())
