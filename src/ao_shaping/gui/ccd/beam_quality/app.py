"""光束质量分析主窗口 (PySide6 + QtCharts)。

布局:

* 左: 控制面板 (源选择 / cam_id / 打开关闭 / 曝光 (手动 + 自动) / 多帧平均 / 快照);
* 中: 视频 (上) + X/Y 剖面 (下), 竖向 ``QSplitter``;
* 右: 最大亮度趋势图 (主视图, 时间窗可切换)。

线程与背压: 帧由源在 SDK/仿真线程里推进容量 1 的信箱, UI 线程用 30 Hz
``QTimer`` 取"最新一帧" —— 处理不过来时丢中间帧而不是排队, 因此显示延迟有界。
分析在 UI 线程同步执行 (单帧 ~10 ms; 超过 1080p 的画面先 2x2 binning 再分析,
计算量降为 1/4, 坐标再用 ``FrameMetrics.rescaled(2.0)`` 还原)。
生命周期操作 (打开/关闭/改曝光) 是相机硬件的秒级阻塞 (大恒打开 ~0.6-3.6 s,
MiiCam 打开 ~10-16 s、改曝光 ~10 s), 一律下放到单线程 ``QThreadPool`` 串行执行,
UI 保持响应, 忙碌期间按钮禁用。

自动曝光是同一条通道上的闭环: 每 ``AUTO_EXPOSURE_INTERVAL_MS`` 取最新一帧原始画面
(不是多帧平均 —— 平均窗口里可能还是上一次曝光下的帧) 的一个统计量, 与目标灰度比较
后把曝光往目标推一步; 走的仍是 ``apply_exposure`` 这条后台作业, 因此不会阻塞 UI,
也不会和手动改曝光打架 (自动模式下手动曝光控件禁用)。

多帧平均 (1~10 帧) 只作用于显示与特征计算: 曝光一变就清空窗口, 免得把两个曝光下
的画面平均成一个不代表任何时刻的亮度。

HDR 获取 (「HDR 获取」开关) 同样**只作用于显示预览**: 打开后取流让位给逐档曝光的
多曝光 bracket, 融合 + 色调映射后的预览图带着来源标记回到 :meth:`_on_poll`, 因此不
进多帧平均窗口、不做特征计算、不进趋势缓冲 —— 那几条通路继续吃单曝光原始帧, 测量
口径与开关之前完全一致 (HDR 预览的亮度来自融合, 不是相机在某一个曝光下的读数)。
bracket 与取流争用的是同一个相机句柄, 所以"停流 / 逐档抓帧 / 恢复取流"这套节拍全在
帧源里 (:class:`~ao_shaping.gui.ccd.beam_quality.frame_source.CameraFrameSource`),
UI 线程既不碰相机也不睡眠; 任何一轮失败都由帧源退回单曝光取流, 界面上如实写明。
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger
from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Qt, Signal
from PySide6.QtGui import QIcon, QImage
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
    QLineEdit,
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
    CENTER_METHODS,
    DEFAULT_CENTER_METHOD,
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
    FALLBACK_EXPOSURE_RANGE_MS,
    FrameSource,
    CameraFrameSource,
    SimBeamSource,
)
from ao_shaping.gui.ccd.beam_quality.hdr_preview import (
    DEFAULT_HDR_LADDER_MS,
    default_ladder_text,
    format_ladder_ms,
    ladder_to_text,
    parse_exposure_ladder,
    preview_mode_text,
    validate_exposure_ladder,
)
from ao_shaping.gui.ccd.beam_quality.profile_widget import ProfileWidget
from ao_shaping.gui.ccd.beam_quality.trend_buffer import TrendBuffer
from ao_shaping.gui.ccd.beam_quality.trend_widget import TrendWidget
from ao_shaping.gui.ccd.beam_quality.video_widget import VideoWidget, to_uint8
from ao_shaping.utils.io import paths
from ao_shaping.utils.io.file import ROOT_DIR

# 源类型: (显示名, 内部键)
SOURCE_KINDS: tuple[tuple[str, str], ...] = (
    ("Daheng", "daheng"),
    ("MiiCam", "miicam"),
    ("仿真源 (离线)", "sim"),
)

#: 打开源之前的曝光控件占位值 (打开后按设备真实量程刷新, 关闭后恢复)
DEFAULT_EXPOSURE_MS = 20.0

#: HDR 预览的色调映射口径: 融合的线性 radiance 压回 0..255 显示图所用的方法名。
#: 实现在 gui/slm/transfer_desktop/live_view.py 的 tone_map_hdr (Reinhard 曲线),
#: 帧源在采集线程里惰性调用它; 这里只是把同一个口径写进"预览模式"那行字。
HDR_TONE_MAP = "Reinhard"

#: 「HDR 获取」开关的说明。提到模块级是因为"后端不支持"时要把开关的说明换成别的
#: (见 :meth:`BeamQualityApp._sync_hdr_capability`), 两处共用一份才不会各说各话。
_HDR_CHECK_TOOLTIP = (
    "打开后预览改走多曝光融合: 逐档曝光各抓一帧, 融成一张宽动态图,"
    "再色调映射上屏。\n"
    "只影响显示预览 —— 多帧平均、特征、剖面、趋势、CSV 仍吃单曝光原始帧, "
    "测量口径不变。\n"
    "代价: 一轮 bracket 要几次曝光切换加读出, 预览节拍从相机帧率掉到每轮一次,"
    " 看起来像翻页而不像视频。\n"
    "任何一轮失败 (相机不支持 / 阶梯非法 / 融合抛错) 都自动退回单曝光取流, "
    "并在「预览模式」那行写明原因, 不会把画面弄死。"
)

ANALYSIS_MAX_WIDTH = 1920
ANALYSIS_MAX_HEIGHT = 1080


class _JobSignals(QObject):
    """QRunnable 无法自带信号, 用伴随 QObject 桥接完成事件。"""

    # (ok, result_or_error); 经 Qt 队列连接回到 UI 线程
    done = Signal(bool, object)


class _Job(QRunnable):
    """把阻塞调用 (相机 open/stop/改曝光) 挪出 UI 线程。

    MiiCam 的硬件操作是实验系统秒级阻塞 (open ~16s, 改曝光 ~10s),
    同步调在 UI 线程会把整个窗口冻住 (2026-10-31 实验系统实测)。
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
        self.setWindowTitle("光束质量分析")
        _icon = ROOT_DIR / "camera_rotate.ico"
        if _icon.exists():
            self.setWindowIcon(QIcon(str(_icon)))
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
        #: 中心计算方法键 (见 analyzer.CENTER_METHODS); 变更后重析当前帧
        self._center_method = DEFAULT_CENTER_METHOD
        # ---------------------------------------------- HDR 预览 (只动显示, 不动测量)
        #: 操作员意图; 帧源那边是否真的在跑 bracket 见其 ``hdr_active``, 界面把两者都显示
        self._hdr_enabled = False
        #: 曝光阶梯 (ms, 升序): 文本框改动即刷新, 打开源时随 set_hdr 一并注入
        self._hdr_ladder_ms: tuple[float, ...] = DEFAULT_HDR_LADDER_MS
        #: 最近一次 HDR 失败/回退的原因 ("" = 没有); 预览模式那行如实带上它
        self._last_hdr_error = ""
        #: 屏上这一帧是不是 HDR 预览 (由帧源的来历标记刷新); 分析读数分档靠它
        self._view_is_hdr = False
        #: 已经画到屏上的 HDR 预览帧数: 只在新的一轮到来时才重画一次, 不重复刷新
        self._hdr_seq = 0
        #: 最近一帧 HDR 预览的采集时刻 (源侧时钟); 与 _last_frame_ts 分开, 免得显示
        #: 派生的时间戳混进自动曝光的"反馈够不够新"判断
        self._hdr_frame_ts = 0.0
        #: 当前帧源有没有多曝光采集通路 (打开源时探测); 没源时按"可有可无"占位
        self._hdr_supported = True
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
        self._exposure_spin.setRange(*FALLBACK_EXPOSURE_RANGE_MS)
        self._exposure_spin.setDecimals(3)
        self._exposure_spin.setSingleStep(1.0)
        self._exposure_spin.setValue(DEFAULT_EXPOSURE_MS)
        self._exposure_spin.setSuffix(" ms")
        self._apply_btn = QPushButton("应用曝光")
        # ------------------------------------------- HDR 预览 (和曝光控件是同一个决定)
        self._hdr_check = QCheckBox("HDR 获取")
        self._hdr_check.setToolTip(_HDR_CHECK_TOOLTIP)
        self._hdr_check.toggled.connect(self._on_hdr_toggled)
        self._hdr_ladder_edit = QLineEdit(default_ladder_text())
        self._hdr_ladder_edit.setPlaceholderText(default_ladder_text())
        self._hdr_ladder_edit.setToolTip(
            "HDR 逐档曝光时长 (ms): 逗号/空格/分号分隔, 至少 2 档、最多 10 档, "
            "自动升序去重, 最后一档作融合参考。\n"
            f"默认 {default_ladder_text()} ms; 只有「HDR 获取」打开时可改。"
        )
        self._hdr_ladder_edit.setEnabled(False)
        self._hdr_ladder_edit.editingFinished.connect(self._on_hdr_ladder_edited)
        self._hdr_retake_btn = QPushButton("重拍一轮")
        self._hdr_retake_btn.setToolTip(
            "按当前阶梯立刻再跑一轮 bracket (不等节拍), 改完阶梯用它马上看到新效果。\n"
            "这一轮仍然占住相机: 期间取流是停着的, 失败原因写在「预览模式」那行。"
        )
        self._hdr_retake_btn.setEnabled(False)
        self._hdr_retake_btn.clicked.connect(self._on_hdr_retake)
        ladder_row = QHBoxLayout()
        ladder_row.addWidget(self._hdr_ladder_edit)
        ladder_row.addWidget(self._hdr_retake_btn)
        self._hdr_hint = QLabel(
            "打开后取流让位给逐档曝光的 bracket, 预览节拍变慢; 测量仍用单曝光帧。"
        )
        self._hdr_hint.setWordWrap(True)
        self._hdr_hint.setStyleSheet("color: #808088;")
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
        self._preview_mode_label = QLabel(preview_mode_text(False))
        self._preview_mode_label.setWordWrap(True)
        self._preview_mode_label.setToolTip(
            "屏上这一帧是哪一路来的: 单曝光原始帧, 还是 HDR 融合预览 "
            f"(色调映射 {HDR_TONE_MAP})。融合失败会自动退回单曝光并在这里说明原因。"
        )
        exposure_form.addRow("曝光", self._exposure_spin)
        exposure_form.addRow(self._apply_btn)
        exposure_form.addRow(self._hdr_check)
        exposure_form.addRow("曝光阶梯", ladder_row)
        exposure_form.addRow(self._hdr_hint)
        exposure_form.addRow(self._auto_check, self._auto_stat_combo)
        exposure_form.addRow("目标灰度", self._auto_target_spin)
        exposure_form.addRow(self._preview_mode_label)
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

        calc_box = QGroupBox("中心计算")
        calc_form = QFormLayout(calc_box)
        self._center_method_combo = QComboBox()
        for key, label in CENTER_METHODS:
            self._center_method_combo.addItem(label, userData=key)
        self._center_method_combo.setCurrentIndex(
            next(
                i
                for i, (key, _label) in enumerate(CENTER_METHODS)
                if key == DEFAULT_CENTER_METHOD
            )
        )
        self._center_method_combo.setToolTip(
            "过中心剖面与十字准线所用的 (cx, cy) 按所选口径计算"
        )
        self._center_method_combo.currentIndexChanged.connect(
            self._on_center_method_changed
        )
        calc_form.addRow("方法", self._center_method_combo)
        layout.addWidget(calc_box)

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
        # 「HDR 获取」本身: 要有源、后端支持、不忙; 自动曝光是另一条同通道的闭环,
        # 它会不停改曝光, 与逐档 bracket 抢同一个曝光旋钮, 所以二者互斥。
        hdr_switch = enabled and has_source and self._hdr_supported and not self._auto_exposure
        # 阶梯与补拍是 HDR 的从属控件: 关着时整组灰掉, 免得像在生效
        hdr_live = hdr_switch and self._hdr_enabled
        self._open_btn.setEnabled(enabled and not has_source)
        self._close_btn.setEnabled(enabled and has_source)
        self._apply_btn.setEnabled(manual_exposure)
        self._exposure_spin.setEnabled(manual_exposure)
        self._hdr_check.setEnabled(hdr_switch)
        self._hdr_ladder_edit.setEnabled(hdr_live)
        self._hdr_retake_btn.setEnabled(hdr_live)
        self._auto_check.setEnabled(enabled and has_source)
        self._auto_stat_combo.setEnabled(enabled and has_source)
        self._auto_target_spin.setEnabled(enabled and has_source)
        self._avg_spin.setEnabled(enabled)
        self._snapshot_btn.setEnabled(enabled)
        self._source_combo.setEnabled(enabled and not has_source)
        self._cam_id_spin.setEnabled(enabled and not has_source)

    def _sync_hdr_check(self) -> None:
        """把「HDR 获取」的勾选刷成 :attr:`_hdr_enabled` (程序回滚时不发信号)。

        信号一响就会再去动相机, 而回滚的起因往往正是"动相机这件事本身", 必须断掉。
        """
        was = self._hdr_check.blockSignals(True)
        self._hdr_check.setChecked(self._hdr_enabled)
        self._hdr_check.blockSignals(was)

    def _sync_hdr_capability(self, source: FrameSource) -> None:
        """打开源后探测该后端有没有多曝光采集通路, 并把开关状态如实反映出来。

        帧源侧的约定是 ``hdr_supported`` / ``set_hdr`` 成对存在 (仿真源与真相机源都
        有); 换成没有这对接口的源就退回"单曝光取流", 开关禁用并说明原因 ——
        而不是默默显示一个不会生效的开关。
        """
        supported = callable(getattr(source, "set_hdr", None)) and bool(
            getattr(source, "hdr_supported", False)
        )
        self._hdr_supported = supported
        if not supported:
            if self._hdr_enabled:
                self._hdr_enabled = False
                self._sync_hdr_check()
                self.statusBar().showMessage("该帧源不支持 HDR 预览, 已退回单曝光", 6000)
            self._hdr_check.setToolTip(
                "当前帧源没有多曝光采集通路 (没有 set_hdr / capture_hdr), "
                "预览只能走单曝光取流。"
            )
        else:
            self._hdr_check.setToolTip(_HDR_CHECK_TOOLTIP)

    def _sync_exposure_range(self, source: FrameSource) -> None:
        """按源的设备真实量程刷新曝光控件, 并把值同步到当前生效曝光。"""
        lo, hi = source.exposure_range_ms
        self._exposure_spin.setRange(float(lo), float(hi))
        self._exposure_spin.setValue(float(source.exposure_ms))
        self._check_ladder_within_range()

    def _check_ladder_within_range(self) -> None:
        """按曝光控件当前量程校一遍已存的阶梯; 超量程就退回默认阶梯。

        量程是打开源后才读得到的设备特征, 而阶梯可能是上次留下的 (或默认那组),
        所以这里补一次校验: 带着一档超量程的阶梯开跑, 源侧整轮都会被拒, 操作员
        只会看见"预览没动静", 那比现在退回默认值难查得多。
        """
        if not self._hdr_ladder_ms:
            return
        try:
            self._hdr_ladder_ms = validate_exposure_ladder(
                self._hdr_ladder_ms, max_ms=float(self._exposure_spin.maximum())
            )
        except ValueError as exc:
            self._last_hdr_error = str(exc)
            self._hdr_ladder_ms = DEFAULT_HDR_LADDER_MS
            self._hdr_ladder_edit.setText(default_ladder_text())
            self.statusBar().showMessage(
                f"{exc}; 曝光阶梯已退回默认 {default_ladder_text()} ms", 8000
            )
            self._push_hdr_state_to_source()

    def _reset_exposure_range(self) -> None:
        """源关闭后把曝光控件退回占位量程与默认值。"""
        self._exposure_spin.setRange(*FALLBACK_EXPOSURE_RANGE_MS)
        self._exposure_spin.setValue(DEFAULT_EXPOSURE_MS)

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
        """按当前选择启动帧源 (相机硬件打开秒级阻塞, 后台线程执行)。"""
        if self._source is not None or self._busy:
            return
        index = self._source_combo.currentIndex()
        kind = SOURCE_KINDS[index][1] if 0 <= index < len(SOURCE_KINDS) else "daheng"
        exposure = float(self._exposure_spin.value())
        cam_id = int(self._cam_id_spin.value())
        # HDR 意图在 UI 线程读干净, 后台线程照着开 (设备操作一律不在 UI 线程发起)
        hdr = bool(self._hdr_enabled and self._hdr_supported)
        ladder = tuple(self._hdr_ladder_ms)

        def _work() -> FrameSource:
            source: FrameSource
            if kind == "sim":
                source = SimBeamSource(exposure_ms=exposure)
            else:
                source = CameraFrameSource(
                    camera_type=kind, cam_id=cam_id, exposure_ms=exposure
                )
            setter = getattr(source, "set_hdr", None)
            if hdr and callable(setter):
                # 先交底再 start(): 这样取流一次也不会起来, 相机从 open 起就归 bracket
                # 独占 —— 省掉"先起流再停流"那一趟 MiiCam 的采集管线重建 (秒级)。
                # 阶梯不合法由源侧记进 last_hdr_error 并保持单曝光, open 不会因 HDR 失败。
                setter(True, ladder)
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
            # HDR 的显示侧状态是"这一帧的来历", 换源之后没有来历可言, 全部归零
            self._view_is_hdr = False
            self._hdr_seq = 0
            self._last_hdr_error = ""
            self._poll_timer.start()
            self._trend_timer.start()
            if self._auto_exposure:
                self._auto_timer.start()
            self._set_controls_enabled(True)
            # 后端有没有多曝光采集通路, 打开以后才知道; 不支持就如实退回单曝光
            self._sync_hdr_capability(source)
            # open 时注入的 set_hdr 也可能自己失败 (约定是不抛, 源那边记下原因并保持
            # 单曝光): 先把帧源的事实捞回来, 开关才不会亮着而 bracket 根本没在跑
            self._sync_source_hdr_state(source)
            self._refresh_preview_mode()
            # 打开后把曝光控件换成该设备的真实量程 (占位量程只用于打开前)
            self._sync_exposure_range(source)
            self.statusBar().showMessage(
                f"已启动 {SOURCE_KINDS[index][0]} (曝光 {source.exposure_ms:.3f} ms)",
                5000,
            )

        self._start_job(f"正在打开 {SOURCE_KINDS[index][0]}…", _work, _done)

    def close_source(self) -> None:
        """停止帧源并清画面 (停相机秒级阻塞, 后台执行); 可重复调用。"""
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
        # HDR 的显示侧状态跟着画面一起清; 开关与阶梯本身保留 (下次打开照原样开)
        self._view_is_hdr = False
        self._hdr_seq = 0
        self._last_hdr_error = ""
        self._preview_mode_label.setText(preview_mode_text(False))
        self._info_label.setText("分辨率: --\n曝光: --\n源帧率: --\n丢弃帧: --")
        self._set_controls_enabled(True)
        # 源已摘掉, 曝光控件退回打开前的占位量程与默认值
        self._reset_exposure_range()
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
        """把界面上的曝光值写给当前源 (MiiCam 重启拉流实验系统 ~10s, 后台执行)。"""
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

    # ---------------------------------------------------------------- HDR 预览
    def _on_hdr_toggled(self, on: bool) -> None:
        """「HDR 获取」开关: 阶梯先过一遍校验, 再把选择交给帧源在后台切换。

        校验放在 UI 线程 (纯算数, 不碰设备): 阶梯不合法就不该去动相机, 直接把开关
        回弹并说明原因。切换本身 (停取流 / 起节拍 / 恢复取流) 是硬件操作, 一律走
        后台作业, UI 线程不在这里等相机。
        """
        if self._busy:
            return  # 硬件操作在飞 (按钮此时也是禁用的, 这里是第二道闸)
        ladder = self._hdr_ladder_ms
        if on:
            if not self._hdr_supported:
                self._rollback_hdr_switch("HDR 预览开不起来", "当前帧源没有多曝光采集通路")
                return
            try:
                ladder = validate_exposure_ladder(
                    parse_exposure_ladder(self._hdr_ladder_edit.text()),
                    max_ms=float(self._exposure_spin.maximum()),
                )
            except ValueError as exc:
                self._rollback_hdr_switch("HDR 预览开不起来", str(exc))
                return
            self._hdr_ladder_ms = ladder
            self._sync_ladder_edit()
        self._hdr_enabled = bool(on)
        self._last_hdr_error = ""
        self._push_hdr_state_to_source()
        if on:
            self.statusBar().showMessage(
                f"HDR 预览已开启: 阶梯 {format_ladder_ms(ladder)} ms, "
                "取流让位给逐档 bracket",
                5000,
            )
        else:
            self.statusBar().showMessage("HDR 预览已关闭, 恢复单曝光取流", 4000)

    def _on_hdr_ladder_edited(self) -> None:
        """阶梯文本改完: 合法就立刻交给帧源, 不合法就退回现用值并说明。"""
        try:
            ladder = validate_exposure_ladder(
                parse_exposure_ladder(self._hdr_ladder_edit.text()),
                max_ms=float(self._exposure_spin.maximum()),
            )
        except ValueError as exc:
            self._last_hdr_error = str(exc)
            self._sync_ladder_edit()
            self._refresh_preview_mode()
            self.statusBar().showMessage(f"曝光阶梯不生效: {exc}", 8000)
            return
        self._hdr_ladder_ms = ladder
        self._last_hdr_error = ""
        self._sync_ladder_edit()
        if self._hdr_enabled:
            self._push_hdr_state_to_source()
            self.statusBar().showMessage(
                f"曝光阶梯已设为 {format_ladder_ms(ladder)} ms, 下一轮 bracket 起用",
                5000,
            )
        else:
            self._refresh_preview_mode()
            self.statusBar().showMessage(
                f"曝光阶梯已记下 {format_ladder_ms(ladder)} ms (HDR 未开, 暂不生效)",
                5000,
            )

    def _on_hdr_retake(self) -> None:
        """手动补一轮 bracket: 让帧源的节拍线程立刻重采 (仍然不碰 UI 线程)。

        这里不起第二个采集者 —— 相机只有一个句柄。回路在跑就等于补拍已发出;
        回路没跑 (源没开 / 后端不支持 / 上一轮失败退回) 时如实说"没发出去"。
        """
        source = self._source
        if source is None or self._busy:
            return
        if not self._hdr_enabled:
            self.statusBar().showMessage("先打开「HDR 获取」再补拍", 5000)
            return
        runner = getattr(source, "run_one_hdr_cycle", None)
        if not callable(runner):
            self.statusBar().showMessage("该帧源不支持 HDR 补拍", 5000)
            return

        def _work() -> bool:
            return bool(runner())

        def _done(ok: bool, payload: object) -> None:
            if not ok:
                # policy: 补拍只是多要一张预览图, 失败只说明, 不打断预览也不弹模态
                logger.opt(exception=payload).warning("HDR 补拍失败")
                self.statusBar().showMessage(f"HDR 补拍失败: {payload}", 6000)
            elif not payload:
                self.statusBar().showMessage(
                    "HDR 回路当前没在跑, 补拍没有发出去 (见预览模式那行的原因)", 6000
                )
            else:
                self.statusBar().showMessage("HDR 补拍已发出, 按当前阶梯重画一轮", 4000)
            self._sync_source_hdr_state(source)
            self._refresh_preview_mode()

        self._start_job("HDR 补拍一轮…", _work, _done)

    def _push_hdr_state_to_source(self) -> None:
        """把开关/阶梯的意图交到帧源 (后台作业); 没有源时只记意图, 打开时随 start 交底。"""
        source = self._source
        if source is None:
            self._refresh_preview_mode()
            return
        setter = getattr(source, "set_hdr", None)
        if not callable(setter):
            self._rollback_hdr_switch("切换失败", "当前帧源没有 HDR 通路")
            return
        enabled = bool(self._hdr_enabled)
        ladder = tuple(self._hdr_ladder_ms)

        def _work() -> None:
            setter(enabled, ladder)

        def _done(ok: bool, payload: object) -> None:
            if not ok:
                # policy: 切 HDR 失败不能让画面死掉: 回滚开关、说明原因、取流继续跑
                logger.opt(exception=payload).warning("切换 HDR 预览失败")
                self._rollback_hdr_switch("HDR 切换失败", str(payload))
                return
            self._sync_source_hdr_state(source)
            self._refresh_preview_mode()

        self._start_job("正在切换 HDR 预览…", _work, _done)

    def _sync_source_hdr_state(self, source: FrameSource | None = None) -> None:
        """把帧源那边的 HDR 事实捞回界面: 开关回弹、失败原因、模式行。

        帧源的约定是"失败不抛, 自己退回单曝光并记下原因"。界面照单全收: 那边说没在
        跑 HDR, 这里的开关就必须是灭的 —— 亮着而屏上是单曝光画面, 比报错更难查。
        """
        if source is None:
            source = self._source
        if source is None:
            return
        reason = str(getattr(source, "last_hdr_error", "") or "")
        live = bool(getattr(source, "hdr_enabled", self._hdr_enabled))
        if reason or live or not self._last_hdr_error:
            # 源那边有话就以它为准; 源说 HDR 正在跑, 上一轮的降级说明也就过期了。
            # 只有一种情况例外: 源没话说、HDR 也没在跑、而界面自己记着原因 (自动曝光
            # 接管 / 阶梯超量程那种让位), 这时清空就等于把"谁按灭了开关"这句话丢掉。
            self._last_hdr_error = reason
        if live != self._hdr_enabled:
            self._hdr_enabled = live
            self._sync_hdr_check()
            self._set_controls_enabled(not self._busy)
            if not live and reason:
                # 源自己退回单曝光时, 开关灭了必须当场说一句 (与开启成功那句对称),
                # 否则操作员只会再点一次同一个勾, 或者更糟 —— 以为屏上还是 HDR 画面。
                self.statusBar().showMessage(f"HDR 预览已退回单曝光: {reason}", 8000)

    def _rollback_hdr_switch(self, prefix: str, reason: str) -> None:
        """开关回弹 + 写清原因: 没成功的选择不能留下"看起来已经开"的状态。"""
        self._hdr_enabled = False
        self._last_hdr_error = reason
        self._sync_hdr_check()
        self._set_controls_enabled(not self._busy)
        self._refresh_preview_mode()
        self.statusBar().showMessage(f"{prefix}: {reason}", 8000)

    def _sync_ladder_edit(self) -> None:
        """把文本框刷成现用阶梯 (升序去重后的写法), 免得屏幕上的和跑的不是一回事。"""
        was = self._hdr_ladder_edit.blockSignals(True)
        self._hdr_ladder_edit.setText(ladder_to_text(self._hdr_ladder_ms))
        self._hdr_ladder_edit.blockSignals(was)

    def _refresh_preview_mode(self) -> None:
        """刷新"预览模式"一行: 现在按下的选择、阶梯、色调映射、有没有降级。"""
        source = self._source
        active = bool(getattr(source, "hdr_active", False)) if source is not None else False
        ladder = self._hdr_ladder_ms
        note = ""
        if self._last_hdr_error:
            note = f"已退回单曝光: {self._last_hdr_error}"
        elif self._hdr_enabled and source is not None and not active:
            note = "HDR 回路未运行"
        elif self._hdr_enabled and source is None:
            note = "未打开帧源"
        self._preview_mode_label.setText(
            preview_mode_text(
                self._hdr_enabled,
                ladder,
                tone_map=HDR_TONE_MAP if self._hdr_enabled else "",
                reference_ms=ladder[-1] if self._hdr_enabled and ladder else None,
                stale=note,
            )
        )

    def _show_hdr_preview(self, img: np.ndarray, timestamp: float, fps: float) -> None:
        """HDR 预览只上屏: 不写 _last_frame, 不进平均窗口、趋势缓冲与处理计数。

        融合 + Reinhard 映射后的 0..255 显示图已经不是任何一个曝光下的原始计数:
        拿它去存快照/进 CSV 就是假数据。剖面与中心读数还是画的 (操作员要能对着
        亮环调 ABA), 但读数在信息栏标成"显示派生", 且 _last_metrics 保持不动。
        """
        self._hdr_seq += 1
        self._hdr_frame_ts = float(timestamp)
        self._view_is_hdr = True
        self._last_view = img
        metrics = self._analyze(img)
        if metrics is None:
            return
        self._video.set_frame(img, metrics, fps)
        self._profile.update_metrics(metrics)

    # ---------------------------------------------------------------- 自动曝光
    def _set_auto_exposure(self, on: bool) -> None:
        """开关自动曝光闭环; 自动模式下手动曝光控件让位给闭环。"""
        self._auto_exposure = bool(on)
        self._auto_hint = ""
        yielded = ""
        if self._auto_exposure and self._hdr_enabled:
            # 闭环按目标灰度不停地改曝光, bracket 每一档也在改曝光: 同一条通道上
            # 两个写入者只会互相踩 (闭环会把 bracket 的中间档当成"太暗"往上推)。
            # 自动曝光是带反馈的闭环, 让它优先, HDR 预览退回并由开关如实回弹。
            self._hdr_enabled = False
            self._last_hdr_error = "自动曝光已接管曝光, HDR 预览让位"
            self._sync_hdr_check()
            self._push_hdr_state_to_source()
            # 让位这件事得跟着"闭环已开启"那一句一起说: 单独 show 一次会被下一句覆盖,
            # 操作员最后只看得见开关自己灭了, 不知道为什么。
            yielded = " · 二者都要改曝光, 只能开一个, 已关闭 HDR 预览"
        if self._auto_exposure:
            if self._source is not None and not self._busy:
                self._auto_timer.start()
            self.statusBar().showMessage(
                f"自动曝光已开启: {self._auto_stat_combo.currentText()} "
                f"→ {self._auto_target_spin.value():.0f}{yielded}",
                8000,
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
        曝光, 拿它们当反馈会让闭环对着旧画面一步步推, 实验系统上会把曝光一路推到上限
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
        """把最近一帧单曝光原始画面存成 PNG (HDR 预览的显示图不存)。"""
        if self._view_is_hdr:
            # 取流在 HDR 期间是停着的: 屏上是融合 + 色调映射后的 0..255 显示图,
            # 不是任何一个曝光下的原始计数; 手里那帧原始画面又是开 HDR 之前的旧帧.
            # 两种存法都会往 snapshots/ 里放一张和文件名口径不符的图, 所以都拒绝,
            # 并把话说清楚怎么存 (关 HDR 等一帧取流画面), 而不是默默存旧帧。
            self.statusBar().showMessage(
                "屏上是 HDR 预览显示图, 不是单曝光原始帧, 不存快照: "
                "关闭 HDR 获取并等一帧取流画面后再存",
                8000,
            )
            return
        frame = self._last_frame
        if frame is None:
            self.statusBar().showMessage("还没有画面可保存", 4000)
            return
        default_dir = paths.data_subdir("snapshots")
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
        # 帧的来历由帧源标好: HDR 预览是融合 + 色调映射后的显示图, 不是任何一个曝光
        # 下的原始计数, 因此只上屏, 不进原始帧、平均窗口、趋势缓冲与处理计数。
        kind = str(getattr(source, "last_frame_kind", "stream"))
        fps = source.fps
        if kind == "hdr":
            self._show_hdr_preview(img, float(timestamp), fps)
        else:
            self._view_is_hdr = False
            self._last_frame = img
            self._last_frame_ts = float(timestamp)
            # 多帧平均: 1 帧时 push 原样返回 (不含拷贝), N>1 时返回最近 N 帧均值
            self._last_view = self._averager.push(img)

            metrics = self._analyze(self._last_view)
            if metrics is not None:
                self._last_metrics = metrics
                self._video.set_frame(self._last_view, metrics, fps)
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
            if self._hdr_enabled:
                # 每秒和帧源对一次账: bracket 在后台某一轮失败时, 源那边自己退回单曝光.
                # 界面不能继续亮着开关 —— 亮着而屏上是单曝光画面比报错更难查。
                # 这里只读两个属性, 不碰设备, 1 Hz 足够。
                self._sync_source_hdr_state(source)
                self._refresh_preview_mode()
            self._update_stats(img)

    def _analyze(self, img: np.ndarray) -> FrameMetrics | None:
        """分析一帧; 超大靶面先 2x2 binning, 再把坐标换算回原始像素。"""
        try:
            arr = np.asarray(img)
            scale = 1.0
            if arr.shape[0] > ANALYSIS_MAX_HEIGHT or arr.shape[1] > ANALYSIS_MAX_WIDTH:
                arr = bin_2x2(arr)
                scale = 2.0
            metrics = analyze_frame(arr, center_method=self._center_method)
            return metrics.rescaled(scale) if scale != 1.0 else metrics
        # policy: 分析异常不该杀掉 30 Hz 定时器; 记录并让界面继续显示上一帧的结果
        except Exception:  # policy: 逐帧分析横跨 numpy/scipy, 异常类型不可枚举
            logger.exception("帧分析失败, 跳过该帧")
            return None

    def _on_center_method_changed(self, index: int) -> None:
        """中心计算方法变更: 记下键并立即用最新一帧重析, 刷新准线/剖面。"""
        key = self._center_method_combo.itemData(index)
        if key is None:
            return
        self._center_method = str(key)
        if self._last_view is None or self._source is None:
            return
        metrics = self._analyze(self._last_view)
        if metrics is None:
            return
        self._last_metrics = metrics
        self._video.set_frame(self._last_view, metrics, self._source.fps)
        self._profile.update_metrics(metrics)

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
        ]
        if self._hdr_enabled:
            # HDR 开着就要说清两件事: 用的是哪一版阶梯, 以及屏上这一帧是融合预览
            # 还是原始帧 (预览还没回来时画面仍是单曝光, 读数口径也跟着不同)。
            ladder = self._hdr_ladder_ms
            note = f"已画 {self._hdr_seq} 帧" if self._hdr_seq else "等待第一帧…"
            if self._last_hdr_error:
                note = f"{note} · 已退回单曝光: {self._last_hdr_error}"
            lines.append(f"曝光阶梯: {format_ladder_ms(ladder)} ms ({note})")
            lines.append(
                f"读数: HDR 预览显示派生 · 第 {self._hdr_seq} 帧"
                if self._view_is_hdr
                else "读数: 单曝光原始帧"
            )
        lines += [
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
        stats = f"已处理 {self._processed} 帧 | UI {self._ui_fps:.1f} fps"
        if self._hdr_seq:
            # HDR 预览帧不计进"已处理": 那是显示节拍, 不是被测量过的帧
            stats += f" | HDR 预览 {self._hdr_seq} 帧"
        self._stats_label.setText(stats)

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
