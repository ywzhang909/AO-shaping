"""SLM 动态整形页：把「生成一张 → 下发一次」改成「沿时间轴分步演化」。

交互设计
--------
桌面壳里的 SLM 页此前只有「选图案 → 生成预览 → 下发」：一次点击就把面板从
当前图案换成目标图案，中间没有过程，操作员也看不到远场是怎么过去的。用户
要的是**动态整形**：远场随时间从当前/起始形状逐步变化到目标形状，过渡时长
可控、可暂停续播、可中止，并且失败时能说清停在哪一帧。

注意「逐步变化」指的是相位分多个中间步逼近目标；远场强度是相位的傅里叶变换，
不是灰度码的线性函数，所以界面不承诺光斑按直线均匀变形。

边界
----
本页是**开环**：相位只沿灰度时间轴下发。右侧 CCD 面板（若启用）只并行观测，
不回灌闭环、也不保持目标；需要按 CCD 实测保持固定目标请去「目标保持」页
（页头的按钮会发出 :attr:`DynamicShapingPage.request_target_hold`，由外壳切页）。

本页把这件事组织成一条时间轴：

+-----------------+-----------------------------------------------------------+
| ① 目标形状       | 图案族 + 参数（沿用 pattern_specs 的单一真源）              |
| ② 起点           | 平场 / 自定义起始参数 / 本页上一帧                          |
| ③ 过渡节奏       | 时长、帧率、每步停留、缓动曲线；实时给出帧数与实际时长        |
| ④ 光学观测来源   | 与静态页同一套快照，写进下发元数据                          |
| ⑤ 设备           | SLM 编号与波长；整段过渡只 open/close 一次                  |
+-----------------+-----------------------------------------------------------+
| 右侧上：远场示意  | 起始 / 中途 / 目标三帧，同一强度归一化，标注「示意」         |
| 右侧中：实时帧    | 真正写入面板的那一帧（后台缩略回传），逐帧变化              |
| 右侧下：走带      | 开始 / 暂停·继续 / 停止 + 进度条 + 已用/剩余 + 状态横幅     |
+-----------------+-----------------------------------------------------------+

状态机与反馈
------------
``idle → connecting → running ⇄ paused → finished / stopped / failed``。
「暂停」保持当前帧、把暂停时长从时间轴里扣掉；「继续」从同一进度接续；
「停止」不自动回零（LCOS 是保持型显示），横幅明说「面板停在过渡中间」。
设备异常 → 红色横幅给出「第几帧失败 + 面板保留哪一帧 + 怎么重试」。

线程与设备边界
--------------
相位生成、远场示意、下发全部离开主线程；本页不打开任何设备，直到用户点
「开始过渡」并在确认框里点「是」。测试与离屏构建都不触碰硬件。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger
from PySide6.QtCore import (
    QObject,
    QRunnable,
    QSize,
    Qt,
    QThread,
    QThreadPool,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QIcon,
    QImage,
    QPainter,
    QPalette,
    QPen,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ao_shaping.gui.slm.dynamic_shaping import (
    EASING_LABELS,
    MAX_DWELL_MS,
    DynamicShapingWorker,
    TransitionEstimate,
    TransitionPlan,
    blend_gray,
    ease,
    estimate_transition,
    far_field_images,
)
from ao_shaping.gui.slm.pattern_specs import (
    PATTERN_TYPES,
    ParamSpec,
    generate_phase_radians,
    specs_for,
)
from ao_shaping.gui.slm.static_shaping_page import OpticalPathSnapshot, StaticShapingPage
from ao_shaping.utils.io.file import ROOT_DIR
from ao_shaping.utils.slm.phase_display import phase_to_slm_grayscale
from ao_shaping.utils.wavefront.pattern_helper import PatternHelper

__all__ = ["DynamicShapingPage", "ShapingPage", "build_plan_snapshot"]

#: 未连接时的面板几何（与静态页 / Streamlit helper 的回退一致）。
_DEFAULT_PANEL_RES: tuple[int, int] = (1920, 1200)
_DEFAULT_BITS: int = 10
#: 参数变化后等这么久再重算估算与远场示意，避免每转一格滑杆就重算一次相位。
_DEBOUNCE_MS = 260
#: 左侧表单宽度上限（px）。再宽会把右侧时间轴挤没。
_FORM_WIDTH = 430

#: 状态横幅的语义色。文字色在暗/亮主题下都可读，左侧色条承担状态区分。
_STATUS_COLORS: dict[str, str] = {
    "idle": "#6b7280",
    "running": "#2f6fbe",
    "paused": "#a86a00",
    "done": "#1f7a4d",
    "error": "#b00020",
}

#: CCD 同步监测可选的相机后端（与 ``ao_shaping.drivers.ccd.common`` 同名）。
_CCD_CAMERA_LABELS: tuple[tuple[str, str], ...] = (
    ("daheng", "大恒 (daheng)"),
    ("miicam", "MiiCam (miicam)"),
    ("sim", "仿真 (sim)"),
)

#: 采样策略：只决定「什么时候采」，不承诺每张 CCD 图对应一帧 SLM。
_CCD_STRATEGIES: tuple[tuple[str, str], ...] = (
    ("every", "每 N 次下发采一帧 (every)"),
    ("periodic", "按时间间隔采样 (periodic)"),
    ("at-end", "过渡结束后采一帧 (at-end)"),
)

#: 下发结束后等采样线程自行收尾的时间；超时就只请求退出线程，不强杀。
_CCD_SHUTDOWN_GRACE_MS = 8000


@dataclass
class PlanSnapshot:
    """一次离线规划的结果：端点相位 + 端点灰度 + 速率估算 + 三帧远场示意。

    ``revision`` 与页面的参数代次绑定：后台任务晚到时若代次不符就作废，
    避免把旧参数的相位/示意贴到新表单上。
    """

    revision: int
    target_phase: np.ndarray
    start_phase: np.ndarray | None
    start_gray: np.ndarray
    target_gray: np.ndarray
    estimate: TransitionEstimate
    far_fields: list[np.ndarray]


def build_plan_snapshot(
    *,
    helper: PatternHelper,
    pattern_key: str,
    target_params: dict[str, Any],
    start_mode: str,
    start_params: dict[str, Any] | None,
    plan: TransitionPlan,
    previous_gray: np.ndarray | None,
    revision: int,
) -> PlanSnapshot:
    """纯函数：生成端点相位、离线灰度、速率估算与远场示意。

    远场示意与真实下发共用「起点灰度 / 目标灰度」的混合结果，唯一差别是目标
    灰度在离线时只做纯数学编码（不含设备的矫正 / LUT）；界面会标注「示意」。
    """
    target_phase = generate_phase_radians(helper, pattern_key, target_params)
    target_gray = phase_to_slm_grayscale(target_phase)

    start_phase: np.ndarray | None
    if start_mode == "flat":
        start_phase = np.zeros_like(target_phase)
    elif start_mode == "phase":
        if not start_params:
            raise ValueError("未提供起始参数")
        start_phase = generate_phase_radians(helper, pattern_key, start_params)
    else:
        start_phase = None
        if previous_gray is None:
            raise ValueError("本页还没有可续接的上一帧")

    if start_phase is not None:
        start_gray = phase_to_slm_grayscale(start_phase)
    else:
        assert previous_gray is not None
        start_gray = np.asarray(previous_gray, dtype=np.uint16)

    if start_gray.shape != target_gray.shape:
        raise ValueError(
            f"起点 {start_gray.shape} 与目标 {target_gray.shape} 面板尺寸不一致"
        )

    estimate = estimate_transition(plan, start_gray, target_gray)
    mid_gray = blend_gray(start_gray, target_gray, ease(0.5, plan.easing))
    far_fields = far_field_images([start_gray, mid_gray, target_gray])
    return PlanSnapshot(
        revision=revision,
        target_phase=target_phase,
        start_phase=start_phase,
        start_gray=start_gray,
        target_gray=target_gray,
        estimate=estimate,
        far_fields=far_fields,
    )


class _ComputeSignals(QObject):
    ready = Signal(object)
    failed = Signal(int, str)


class _ComputeTask(QRunnable):
    """后台算一次规划（相位生成 + 远场 FFT）。"""

    def __init__(self, signals: _ComputeSignals, kwargs: dict[str, Any]) -> None:
        super().__init__()
        self.signals = signals
        self._kwargs = kwargs

    def run(self) -> None:
        revision = int(self._kwargs["revision"])
        try:
            snapshot = build_plan_snapshot(**self._kwargs)
        except Exception as exc:  # 参数/设备无关的纯计算失败也要变成界面结论
            logger.warning("动态整形规划失败 (rev {}): {}", revision, exc)
            self.signals.failed.emit(revision, str(exc))
            return
        self.signals.ready.emit(snapshot)


class _ParamForm(QWidget):
    """按 :func:`specs_for` 生成一列表单控件；值变化时发 ``changed``。"""

    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._form = QFormLayout(self)
        self._form.setContentsMargins(0, 0, 0, 0)
        self._form.setSpacing(6)
        self._widgets: dict[str, QWidget] = {}
        self._values: dict[str, Any] = {}

    def set_pattern(self, key: str) -> None:
        """重建控件（切图案族时参数集合不同）。"""
        while self._form.count():
            item = self._form.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._widgets.clear()
        self._values.clear()
        for spec in specs_for(key):
            widget = self._build_widget(spec)
            self._widgets[spec.name] = widget
            self._form.addRow(spec.label, widget)

    def _build_widget(self, spec: ParamSpec) -> QWidget:
        if spec.kind == "select":
            combo = QComboBox()
            for option in spec.options:
                combo.addItem(option, option)
            if spec.default in spec.options:
                combo.setCurrentIndex(spec.options.index(spec.default))
            self._values[spec.name] = combo.currentData()
            combo.currentIndexChanged.connect(
                lambda _i, name=spec.name: self._on_select(name)
            )
            return combo
        if spec.kind in ("int", "seed"):
            spin = QSpinBox()
            if spec.min is not None:
                spin.setMinimum(int(spec.min))
            if spec.max is not None:
                spin.setMaximum(int(spec.max))
            spin.setValue(int(spec.default) if spec.default is not None else -1)
            self._values[spec.name] = spin.value()
            spin.valueChanged.connect(lambda value, name=spec.name: self._on_int(name, value))
            return spin
        fspin = QDoubleSpinBox()
        if spec.min is not None:
            fspin.setMinimum(float(spec.min))
        if spec.max is not None:
            fspin.setMaximum(float(spec.max))
        if spec.step is not None:
            fspin.setSingleStep(float(spec.step))
        if spec.default is not None:
            fspin.setValue(float(spec.default))
        fspin.setDecimals(6)
        self._values[spec.name] = fspin.value()
        fspin.valueChanged.connect(
            lambda value, name=spec.name: self._on_float(name, value)
        )
        return fspin

    def _on_select(self, name: str) -> None:
        widget = self._widgets[name]
        assert isinstance(widget, QComboBox)
        self._values[name] = widget.currentData()
        self.changed.emit()

    def _on_int(self, name: str, value: int) -> None:
        self._values[name] = value
        self.changed.emit()

    def _on_float(self, name: str, value: float) -> None:
        self._values[name] = value
        self.changed.emit()

    def values(self) -> dict[str, Any]:
        """当前取值；``seed`` 的 -1 占位还原成 ``None``。"""
        out: dict[str, Any] = {}
        key = self._pattern_key
        specs = {spec.name: spec for spec in specs_for(key)} if key else {}
        for name, value in self._values.items():
            spec = specs.get(name)
            if spec is not None and spec.kind == "seed" and value == -1:
                value = None
            out[name] = value
        return out

    _pattern_key: str = ""

    def set_key(self, key: str) -> None:
        self._pattern_key = key
        self.set_pattern(key)


class _ImagePanel(QFrame):
    """等比缩放显示一张图；无图时画居中的空状态说明。

    用自绘而不是 ``QLabel.setPixmap``：切帧时图幅比例可能变化（不同帧的
    缩略图尺寸一致，但远场面板会随窗口变化），自绘能始终居中且不变形。
    """

    def __init__(self, empty_text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image: QImage | None = None
        self._empty_text = empty_text
        self.setMinimumHeight(150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFrameShape(QFrame.Shape.NoFrame)

    def set_image(self, image: QImage | None) -> None:
        self._image = image
        self.update()

    def set_empty_text(self, text: str) -> None:
        self._empty_text = text
        self.update()

    @property
    def has_image(self) -> bool:
        return self._image is not None

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(0, 0, -1, -1)
        painter.setPen(QPen(self.palette().color(QPalette.ColorRole.Mid), 1))
        painter.setBrush(self.palette().color(QPalette.ColorRole.Base))
        painter.drawRoundedRect(rect, 7, 7)
        if self._image is None:
            painter.setPen(self.palette().color(QPalette.ColorRole.PlaceholderText))
            painter.drawText(
                rect.adjusted(18, 18, -18, -18),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                self._empty_text,
            )
            return
        target = self.size() - QSize(14, 14)
        scaled = self._image.scaled(
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawImage(x, y, scaled)


class DynamicShapingPage(QWidget):
    """动态整形页（时间轴式交互）。所有设备操作都在确认后由后台线程执行。"""

    #: 操作员想从这一页去「目标保持」页。页面自身不切页，由外壳接线切换。
    request_target_hold = Signal()

    _START_LABELS: dict[str, str] = {
        "flat": "平场（波前平坦）",
        "phase": "自定义起始参数",
        "previous": "本页上一帧",
    }

    def __init__(
        self,
        parent: QWidget | None = None,
        optical_path: OpticalPathSnapshot | None = None,
    ) -> None:
        super().__init__(parent)
        icon = ROOT_DIR / "camera_rotate.ico"
        if icon.exists():
            self.setWindowIcon(QIcon(str(icon)))

        self._helper = PatternHelper(_DEFAULT_PANEL_RES, bits=_DEFAULT_BITS)
        self._optical_path: OpticalPathSnapshot = optical_path or OpticalPathSnapshot(
            source="model_forward"
        )
        self._revision = 0
        self._snapshot: PlanSnapshot | None = None
        self._last_gray: np.ndarray | None = None
        self._worker: DynamicShapingWorker | None = None
        self._state = "idle"
        self._run_frames: tuple[int, int] = (0, 0)
        self._syncing_duration = False
        self._closing = False
        # CCD 同步监测（可选）：下发归 SLM 线程，取样归另一个线程，互不阻塞。
        self._ccd_monitor: Any = None
        self._ccd_sampler: QObject | None = None
        self._ccd_thread: QThread | None = None
        self._ccd_dispatch: Any = None
        self._ccd_frames = 0

        # 计算/示意用单线程池：同一时刻只算一份，避免快速拖参数时堆任务。
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._compute_signals = _ComputeSignals()
        self._compute_signals.ready.connect(self._on_computed)
        self._compute_signals.failed.connect(self._on_compute_failed)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(_DEBOUNCE_MS)
        self._debounce.timeout.connect(self._launch_compute)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 14)
        root.setSpacing(12)
        root.addWidget(self._build_header())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_form())
        splitter.addWidget(self._build_stage())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 900])
        root.addWidget(splitter, 1)

        self._on_pattern_changed()
        self._apply_state("idle")
        self._refresh_ccd_controls()

    # ------------------------------------------------------------------ 构建

    def _build_header(self) -> QWidget:
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        title = QLabel("SLM 动态整形")
        font = title.font()
        font.setPointSize(font.pointSize() + 5)
        font.setWeight(QFont.Weight.DemiBold)
        title.setFont(font)
        subtitle = QLabel(
            "相位分多个中间步逐步逼近目标，远场随之渐变（非线性） · 可暂停、续播与中止"
        )
        subtitle.setWordWrap(True)
        subtitle.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        # 开环说明：这一页只走 SLM 灰度时间轴，右侧 CCD 面板只观测、不回灌。
        # 想去闭环保持目标，用这里的按钮去「目标保持」页（由外壳接线切页）。
        callout = QFrame()
        callout.setObjectName("openLoopCallout")
        callout.setFrameShape(QFrame.Shape.StyledPanel)
        callout_layout = QHBoxLayout(callout)
        callout_layout.setContentsMargins(10, 8, 10, 8)
        callout_layout.setSpacing(8)
        callout_note = QLabel(
            "本页是开环：相位只按 SLM 灰度时间轴逐帧下发；CCD 面板（若启用）只并行"
            "观测，不回灌闭环，也不保持目标。需要按 CCD 实测保持固定目标，请用"
            "「目标保持」页。"
        )
        callout_note.setWordWrap(True)
        callout_note.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        callout_layout.addWidget(callout_note, 1)
        self._target_hold_button = QPushButton("打开目标保持页")
        self._target_hold_button.setObjectName("openTargetHold")
        self._target_hold_button.clicked.connect(
            lambda: self.request_target_hold.emit()
        )
        callout_layout.addWidget(self._target_hold_button)
        layout.addWidget(callout)
        return host

    def _section(self, title: str) -> tuple[QGroupBox, QVBoxLayout]:
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(8)
        return box, layout

    def _build_form(self) -> QWidget:
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # ① 目标形状 --------------------------------------------------------
        target_box, target_layout = self._section("① 目标形状")
        self._pattern_combo = QComboBox()
        for key, spec in PATTERN_TYPES.items():
            self._pattern_combo.addItem(spec.label, key)
        self._pattern_combo.currentIndexChanged.connect(self._on_pattern_changed)
        self._pattern_desc = QLabel()
        self._pattern_desc.setWordWrap(True)
        self._pattern_desc.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        self._target_form = _ParamForm()
        self._target_form.changed.connect(self._on_params_changed)
        target_layout.addWidget(self._pattern_combo)
        target_layout.addWidget(self._pattern_desc)
        target_layout.addWidget(self._target_form)
        layout.addWidget(target_box)

        # ② 起点 ------------------------------------------------------------
        start_box, start_layout = self._section("② 起点")
        self._start_combo = QComboBox()
        for key, label in self._START_LABELS.items():
            self._start_combo.addItem(label, key)
        self._start_combo.currentIndexChanged.connect(self._on_start_mode_changed)
        self._start_hint = QLabel()
        self._start_hint.setWordWrap(True)
        self._start_hint.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        self._start_form = _ParamForm()
        self._start_form.changed.connect(self._on_params_changed)
        self._start_form.setVisible(False)
        start_layout.addWidget(self._start_combo)
        start_layout.addWidget(self._start_hint)
        start_layout.addWidget(self._start_form)
        layout.addWidget(start_box)

        # ③ 过渡节奏 --------------------------------------------------------
        pace_box, pace_layout = self._section("③ 过渡节奏")
        pace_form = QFormLayout()
        pace_form.setSpacing(6)

        duration_host = QWidget()
        duration_layout = QHBoxLayout(duration_host)
        duration_layout.setContentsMargins(0, 0, 0, 0)
        duration_layout.setSpacing(8)
        self._duration_slider = QSlider(Qt.Orientation.Horizontal)
        self._duration_slider.setRange(5, 600)  # 0.1 s 步进：0.5 … 60.0 s
        self._duration_slider.setValue(60)
        self._duration_spin = QDoubleSpinBox()
        self._duration_spin.setRange(0.5, 120.0)
        self._duration_spin.setSingleStep(0.5)
        self._duration_spin.setDecimals(1)
        self._duration_spin.setSuffix(" s")
        self._duration_spin.setValue(6.0)
        self._duration_slider.valueChanged.connect(self._on_duration_slider)
        self._duration_spin.valueChanged.connect(self._on_duration_spin)
        duration_layout.addWidget(self._duration_slider, 1)
        duration_layout.addWidget(self._duration_spin)
        pace_form.addRow("过渡时长", duration_host)

        self._fps_spin = QSpinBox()
        self._fps_spin.setRange(1, 30)
        self._fps_spin.setValue(10)
        self._fps_spin.setSuffix(" 帧/秒")
        self._fps_spin.valueChanged.connect(self._on_params_changed)
        pace_form.addRow("帧率", self._fps_spin)

        self._dwell_spin = QSpinBox()
        self._dwell_spin.setRange(0, int(MAX_DWELL_MS))
        self._dwell_spin.setSingleStep(10)
        self._dwell_spin.setSuffix(" ms")
        self._dwell_spin.setSpecialValueText("自动（按像素翻转时间）")
        self._dwell_spin.setToolTip(
            "每写一帧后停留多久再写下一帧。0 = 按该帧最大灰度变化自动估算 LCOS "
            "像素翻转时间（上限 200 ms）。设定值低于翻转时间时会被抬到翻转时间"
            "——像素没翻稳就被下一帧覆盖，中间帧会糊掉。"
        )
        self._dwell_spin.valueChanged.connect(self._on_params_changed)
        pace_form.addRow("每步停留", self._dwell_spin)

        self._easing_combo = QComboBox()
        for key, label in EASING_LABELS.items():
            self._easing_combo.addItem(label, key)
        self._easing_combo.currentIndexChanged.connect(self._on_params_changed)
        pace_form.addRow("变化曲线", self._easing_combo)
        pace_layout.addLayout(pace_form)

        self._estimate_label = QLabel("正在规划帧序列…")
        self._estimate_label.setWordWrap(True)
        fixed = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        fixed.setPointSize(max(8, fixed.pointSize() - 0.5))
        self._estimate_label.setFont(fixed)
        self._estimate_label.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        pace_layout.addWidget(self._estimate_label)
        layout.addWidget(pace_box)

        # ④ 光学观测来源 ----------------------------------------------------
        opt_box, opt_layout = self._section("④ 光学观测来源")
        self._optical_combo = QComboBox()
        self._optical_combo.addItem("○ 模型前向（未观测，按理想光路）", "model_forward")
        self._optical_combo.addItem("○ 未知（几何未标定，用内置回退）", "unknown")
        self._optical_combo.addItem("● 已标定光路（本会话实测）", "calibrated")
        index = self._optical_combo.findData(self._optical_path.source)
        self._optical_combo.setCurrentIndex(max(0, index))
        self._optical_combo.currentIndexChanged.connect(self._refresh_optical_hint)
        self._optical_hint = QLabel()
        self._optical_hint.setWordWrap(True)
        opt_layout.addWidget(self._optical_combo)
        opt_layout.addWidget(self._optical_hint)
        self._refresh_optical_hint()
        layout.addWidget(opt_box)

        # ⑤ 设备 ------------------------------------------------------------
        device_box, device_layout = self._section("⑤ 设备")
        device_form = QFormLayout()
        device_form.setSpacing(6)
        self._slm_spin = QSpinBox()
        self._slm_spin.setRange(1, 8)
        self._slm_spin.setValue(1)
        self._slm_spin.setToolTip("目标 SLM 编号（与驱动 Santec(slm_number=…) 一致）")
        self._wavelength_spin = QSpinBox()
        self._wavelength_spin.setRange(1000, 1100)
        self._wavelength_spin.setValue(1064)
        self._wavelength_spin.setSuffix(" nm")
        self._wavelength_spin.setToolTip(
            "用于相位→灰度换算的波长；2π 灰度由设备读数决定，不在这里设置"
        )
        device_form.addRow("SLM 编号", self._slm_spin)
        device_form.addRow("波长", self._wavelength_spin)
        device_layout.addLayout(device_form)
        device_hint = QLabel(
            "整段过渡只打开一次设备：单句柄连续写帧，结束后关闭。"
            "不会逐帧 open/close（那会让驱动板假死）。"
        )
        device_hint.setWordWrap(True)
        device_hint.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        device_layout.addWidget(device_hint)
        self._slm_spin.valueChanged.connect(self._on_params_changed)
        self._wavelength_spin.valueChanged.connect(self._on_params_changed)
        layout.addWidget(device_box)

        # ⑥ CCD 同步监测（可选） ---------------------------------------------
        layout.addWidget(self._build_sync_ccd())

        # 动作 --------------------------------------------------------------
        actions = QHBoxLayout()
        self._preview_btn = QPushButton("预览过渡")
        self._preview_btn.setToolTip("算三帧远场示意与速率估算；不打开 SLM")
        self._preview_btn.clicked.connect(self._launch_compute)
        self._start_btn = QPushButton("开始过渡")
        self._start_btn.setObjectName("primaryAction")
        self._start_btn.setToolTip("确认后逐帧写入 SLM；可随时暂停或停止")
        self._start_btn.clicked.connect(self._on_start_clicked)
        actions.addWidget(self._preview_btn)
        actions.addWidget(self._start_btn)
        layout.addLayout(actions)

        safety = QLabel(
            "开始后每一帧都会写入所选 SLM；「暂停」保持当前帧，「停止」停在过渡中间，"
            "都不会自动回零。"
        )
        safety.setWordWrap(True)
        safety.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        layout.addWidget(safety)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(host)
        scroll.setMinimumWidth(320)
        scroll.setMaximumWidth(_FORM_WIDTH)
        return scroll

    def _build_sync_ccd(self) -> QGroupBox:
        """⑥ CCD 同步监测（可选）：与「下发」并行的一路观测。

        默认关闭 —— 同一台相机同一时刻只能被一个页面占用，开页就抢相机会让
        别的页面打不开设备。开启后相机由独立的采样线程打开与释放，界面线程与
        SLM 下发线程都不碰相机 IO；相机被占用时只在这里报错，不影响纯下发。
        """
        box, layout = self._section("⑥ CCD 同步监测（可选）")
        self._ccd_enabled = QCheckBox("启用 CCD 同步监测（与 SLM 下发并行）")
        self._ccd_enabled.setChecked(False)
        self._ccd_enabled.setToolTip(
            "默认关闭，不打开任何相机。开启后 CCD 采样跑在单独线程里，"
            "与 SLM 逐帧下发互不阻塞；相机被别的程序占用时只报错，下发照常。"
        )
        layout.addWidget(self._ccd_enabled)

        self._ccd_camera = QComboBox()
        for code, label in _CCD_CAMERA_LABELS:
            self._ccd_camera.addItem(label, code)

        self._ccd_cam_id = QSpinBox()
        self._ccd_cam_id.setRange(0, 15)

        self._ccd_exposure = QDoubleSpinBox()
        self._ccd_exposure.setRange(0.01, 1000.0)
        self._ccd_exposure.setDecimals(3)
        self._ccd_exposure.setSingleStep(1.0)
        self._ccd_exposure.setValue(5.0)
        self._ccd_exposure.setSuffix(" ms")
        self._ccd_exposure.setToolTip(
            "采样线程打开相机后设的曝光。太短只剩读出噪声，太长会把光斑打过曝。"
        )

        device_form = QFormLayout()
        device_form.setSpacing(6)
        device_form.addRow("相机类型", self._ccd_camera)
        device_form.addRow("cam_id", self._ccd_cam_id)
        device_form.addRow("曝光", self._ccd_exposure)
        layout.addLayout(device_form)

        self._ccd_strategy = QComboBox()
        for code, label in _CCD_STRATEGIES:
            self._ccd_strategy.addItem(label, code)

        self._ccd_every_n = QSpinBox()
        self._ccd_every_n.setRange(1, 1000)
        self._ccd_every_n.setValue(1)
        self._ccd_every_n.setPrefix("每 ")
        self._ccd_every_n.setSuffix(" 次下发")

        self._ccd_interval = QDoubleSpinBox()
        self._ccd_interval.setRange(0.02, 60.0)
        self._ccd_interval.setDecimals(2)
        self._ccd_interval.setSingleStep(0.1)
        self._ccd_interval.setValue(0.10)
        self._ccd_interval.setSuffix(" s")

        strategy_form = QFormLayout()
        strategy_form.setSpacing(6)
        strategy_form.addRow("采样策略", self._ccd_strategy)
        strategy_form.addRow("每 N 次", self._ccd_every_n)
        strategy_form.addRow("周期", self._ccd_interval)
        layout.addLayout(strategy_form)

        self._ccd_hint = QLabel(
            "策略只决定“什么时候采”：相机没就绪或正被别的程序占用时，采样会顺延"
            "或直接报错 —— 不是每个 SLM 帧都配一张 CCD 图，采样也不改变下发的"
            "帧数与速度。"
        )
        self._ccd_hint.setWordWrap(True)
        self._ccd_hint.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        layout.addWidget(self._ccd_hint)

        self._ccd_enabled.toggled.connect(self._refresh_ccd_controls)
        self._ccd_strategy.currentIndexChanged.connect(self._refresh_ccd_controls)
        return box

    def _build_stage(self) -> QWidget:
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        far_row = QHBoxLayout()
        far_row.setSpacing(10)
        self._far_panels: list[tuple[QLabel, _ImagePanel]] = []
        for title in ("起始 · 0%", "中途 · 50%", "目标 · 100%"):
            cell = QWidget()
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(4)
            caption = QLabel(title)
            caption_font = caption.font()
            caption_font.setPointSize(max(8, caption_font.pointSize() - 0.5))
            caption_font.setWeight(QFont.Weight.DemiBold)
            caption.setFont(caption_font)
            panel = _ImagePanel("远场示意将在这里显示")
            panel.setMinimumHeight(148)
            cell_layout.addWidget(caption)
            cell_layout.addWidget(panel, 1)
            far_row.addWidget(cell, 1)
            self._far_panels.append((caption, panel))
        layout.addLayout(far_row)

        stage_label = QLabel("写入面板的当前帧")
        stage_font = stage_label.font()
        stage_font.setPointSize(max(8, stage_font.pointSize() - 0.5))
        stage_font.setWeight(QFont.Weight.DemiBold)
        stage_label.setFont(stage_font)
        layout.addWidget(stage_label)

        self._live_panel = _ImagePanel(
            "尚无过渡帧\n\n"
            "① 选目标形状 → ② 定起点 → ③ 调时长与帧率\n"
            "先点「预览过渡」看三帧远场示意；点「开始过渡」后，"
            "这里逐帧显示真正写入面板的画面。"
        )
        self._live_panel.setMinimumHeight(300)
        layout.addWidget(self._live_panel, 1)

        # CCD 同步监测的观测窗（默认隐藏，勾选左栏后才出现） ------------------
        self._ccd_box = QGroupBox("CCD 同步监测")
        self._ccd_box.setVisible(False)
        ccd_layout = QHBoxLayout(self._ccd_box)
        ccd_layout.setContentsMargins(10, 8, 10, 10)
        ccd_layout.setSpacing(10)

        self._ccd_preview = _ImagePanel(
            "CCD 同步监测未启用\n\n"
            "勾选左栏「⑥ CCD 同步监测」后，这里显示采样线程回传的画面；"
            "不启用时不会打开任何相机。"
        )
        self._ccd_preview.setMinimumHeight(132)
        ccd_layout.addWidget(self._ccd_preview, 2)

        ccd_side = QVBoxLayout()
        ccd_side.setContentsMargins(0, 0, 0, 0)
        ccd_side.setSpacing(4)
        self._ccd_status = QLabel()
        self._ccd_status.setWordWrap(True)
        self._ccd_stat = QLabel()
        stat_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        stat_font.setPointSize(max(8, stat_font.pointSize() - 0.5))
        self._ccd_stat.setFont(stat_font)
        self._ccd_stat.setWordWrap(True)
        self._ccd_stat.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        ccd_side.addWidget(self._ccd_status)
        ccd_side.addWidget(self._ccd_stat)
        ccd_side.addStretch(1)
        ccd_layout.addLayout(ccd_side, 1)

        layout.addWidget(self._ccd_box)
        self._set_ccd_status("idle", "○ 未启用 —— 本次只下发 SLM，不开相机。")
        self._render_ccd_stat(None)

        # 走带 (transport) ---------------------------------------------------
        transport = QFrame()
        transport.setFrameShape(QFrame.Shape.StyledPanel)
        transport_layout = QVBoxLayout(transport)
        transport_layout.setContentsMargins(12, 10, 12, 10)
        transport_layout.setSpacing(8)

        status_row = QHBoxLayout()
        self._status = QLabel()
        self._status.setWordWrap(True)
        status_row.addWidget(self._status, 1)
        self._frame_label = QLabel("—")
        self._frame_label.setFont(
            QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        )
        self._frame_label.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        status_row.addWidget(self._frame_label)
        transport_layout.addLayout(status_row)

        self._bar = QProgressBar()
        self._bar.setRange(0, 1)
        self._bar.setValue(0)
        self._bar.setFormat("第 %v / %m 帧")
        self._bar.setTextVisible(True)
        transport_layout.addWidget(self._bar)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        self._pause_btn = QPushButton("暂停")
        self._pause_btn.clicked.connect(self._on_pause_resume)
        self._stop_btn = QPushButton("停止")
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        self._elapsed_label = QLabel("已用 — / 剩余 —")
        self._elapsed_label.setFont(
            QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        )
        self._elapsed_label.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        controls.addWidget(self._pause_btn)
        controls.addWidget(self._stop_btn)
        controls.addStretch(1)
        controls.addWidget(self._elapsed_label)
        transport_layout.addLayout(controls)
        layout.addWidget(transport)
        return host

    # ------------------------------------------------------------------ 参数

    @property
    def _pattern_key(self) -> str:
        data = self._pattern_combo.currentData()
        assert isinstance(data, str)
        return data

    @Slot()
    def _on_pattern_changed(self) -> None:
        key = self._pattern_key
        self._target_form.set_key(key)
        self._start_form.set_key(key)
        spec = PATTERN_TYPES[key]
        self._pattern_desc.setText(
            f"{spec.label}，按下方参数生成。动态过渡只改变中间过程：相位逐帧向"
            "这组参数生成的终点逼近；远场随之变化，但不是线性变形。"
        )
        self._on_params_changed()

    @Slot()
    def _on_params_changed(self) -> None:
        if self._state in ("running", "paused", "connecting"):
            return
        self._revision += 1
        if self._state in ("finished", "stopped", "failed"):
            self._apply_state("idle")
        self._estimate_label.setText("参数已变化，正在重算帧序列…")
        self._debounce.start()

    @Slot()
    def _on_start_mode_changed(self) -> None:
        mode = self._start_combo.currentData()
        custom = mode == "phase"
        self._start_form.setVisible(custom)
        if mode == "flat":
            hint = "从波前平坦（校正后平场）开始：目标形状会从零级光斑中逐渐长出。"
        elif mode == "phase":
            hint = (
                "起始与目标用同一图案族、不同参数：适合光斑由小变大、"
                "光栅周期连续变化这类演化。"
            )
        else:
            hint = "以本页最近一次过渡的最后一帧为起点，可连续接第二段过渡。"
        self._start_hint.setText(hint)
        self._on_params_changed()

    def _on_duration_slider(self, value: int) -> None:
        if self._syncing_duration:
            return
        self._syncing_duration = True
        self._duration_spin.setValue(value / 10.0)
        self._syncing_duration = False
        self._on_params_changed()

    def _on_duration_spin(self, value: float) -> None:
        if self._syncing_duration:
            return
        self._syncing_duration = True
        self._duration_slider.setValue(int(round(value * 10)))
        self._syncing_duration = False
        self._on_params_changed()

    def _current_plan(self) -> TransitionPlan:
        easing = self._easing_combo.currentData()
        assert isinstance(easing, str)
        return TransitionPlan(
            duration_s=float(self._duration_spin.value()),
            fps=int(self._fps_spin.value()),
            easing=easing,
            dwell_ms=float(self._dwell_spin.value()),
        )

    def _compute_kwargs(self) -> dict[str, Any]:
        start_mode = self._start_combo.currentData()
        assert isinstance(start_mode, str)
        return {
            "helper": self._helper,
            "pattern_key": self._pattern_key,
            "target_params": self._target_form.values(),
            "start_mode": start_mode,
            "start_params": self._start_form.values() if start_mode == "phase" else None,
            "plan": self._current_plan(),
            "previous_gray": self._last_gray,
            "revision": self._revision,
        }

    @Slot()
    def _launch_compute(self) -> None:
        self._debounce.stop()
        if self._state in ("running", "paused", "connecting"):
            return
        try:
            kwargs = self._compute_kwargs()
            self._current_plan()  # 提前暴露时长/帧率越界，不必等后台任务
        except ValueError as exc:
            self._estimate_label.setText(f"参数无效：{exc}")
            self._start_btn.setEnabled(False)
            return
        self._preview_btn.setEnabled(False)
        self._pool.start(_ComputeTask(self._compute_signals, kwargs))

    @Slot(object)
    def _on_computed(self, snapshot: PlanSnapshot) -> None:
        self._preview_btn.setEnabled(True)
        if snapshot.revision != self._revision:
            return  # 参数已变，旧结果作废
        self._snapshot = snapshot
        self._render_far_fields(snapshot)
        self._render_estimate(snapshot.estimate)
        self._start_btn.setEnabled(self._state in ("idle", "finished", "stopped", "failed"))

    @Slot(int, str)
    def _on_compute_failed(self, revision: int, message: str) -> None:
        self._preview_btn.setEnabled(True)
        if revision != self._revision:
            return
        self._estimate_label.setText(f"无法规划：{message}")
        self._start_btn.setEnabled(False)

    def _render_estimate(self, estimate: TransitionEstimate) -> None:
        lines = [
            f"{estimate.frames} 帧 · 名义间隔 {estimate.interval_s * 1000:.0f} ms",
            f"预计 {estimate.expected_duration_s:.1f} s · "
            f"单帧最大变化 {estimate.max_step_gray:.0f} / 1023 码",
            f"每步停留 {estimate.dwell_s * 1000:.1f} ms"
            "（含像素翻转等待）",
        ]
        if estimate.dwell_clamped:
            lines.append(
                f"⚠ 设定的停留短于翻转时间 {estimate.flip_wait_s * 1000:.0f} ms，"
                "已按翻转时间执行。"
            )
        if estimate.limited_by_panel:
            lines.append(
                "⚠ 每步停留比设定帧间隔更慢，实际过渡会比设定时长更长。"
            )
        lines.append("相位逐帧逼近目标；远场变化不是线性变形。")
        self._estimate_label.setText("\n".join(lines))

    def _render_far_fields(self, snapshot: PlanSnapshot) -> None:
        for (_caption, panel), rgb in zip(self._far_panels, snapshot.far_fields):
            panel.set_image(_rgb_to_qimage(rgb))
        if not snapshot.far_fields:
            for _caption, panel in self._far_panels:
                panel.set_image(None)

    def _refresh_optical_hint(self) -> None:
        source = self._optical_combo.itemData(self._optical_combo.currentIndex())
        if source == "calibrated":
            self._optical_hint.setText(
                "● 本次过渡的元数据按「已标定光路」记录（光路快照随会话注入）。"
            )
            self._optical_hint.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        else:
            self._optical_hint.setText(
                "○ 光路假设未经 CCD 验证：会写进元数据，结果不代表实测光路。"
            )
            self._optical_hint.setForegroundRole(QPalette.ColorRole.PlaceholderText)

    # ------------------------------------------------------- CCD 监测（可选）

    def _refresh_ccd_controls(self, *_args: object) -> None:
        """按「启用 + 策略 + 当前状态」决定左栏控件能不能点、右窗出不出现。"""
        enabled = bool(self._ccd_enabled.isChecked())
        busy = self._state in ("running", "paused", "connecting")
        strategy = str(self._ccd_strategy.currentData() or "every")
        for widget in (
            self._ccd_camera,
            self._ccd_cam_id,
            self._ccd_exposure,
            self._ccd_strategy,
        ):
            widget.setEnabled(enabled and not busy)
        self._ccd_every_n.setEnabled(enabled and not busy and strategy == "every")
        self._ccd_interval.setEnabled(enabled and not busy and strategy == "periodic")
        self._ccd_hint.setEnabled(enabled)
        box = getattr(self, "_ccd_box", None)
        if box is not None:
            box.setVisible(enabled)
        if not enabled:
            self._set_ccd_status("idle", "○ 未启用 —— 本次只下发 SLM，不开相机。")
            self._render_ccd_stat(None)

    def _set_ccd_status(self, kind: str, text: str) -> None:
        """CCD 一侧的状态行：失败只红这一行，下发横幅与面板不受影响。"""
        label = getattr(self, "_ccd_status", None)
        if label is None:
            return
        if kind == "idle":
            label.setStyleSheet("")
            label.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        else:
            label.setStyleSheet(f"color: {_STATUS_COLORS.get(kind, _STATUS_COLORS['idle'])};")
        label.setText(text)

    def _render_ccd_stat(self, payload: object | None) -> None:
        """统计行只报手里的数：峰值 / FWHM / 质心，缺哪项就写「—」。"""
        label = getattr(self, "_ccd_stat", None)
        if label is None:
            return
        stats = payload if isinstance(payload, dict) else {}
        parts = [f"帧 {self._ccd_frames}"]
        peak = _ccd_first(stats, ("peak", "max", "peak_counts"))
        parts.append(f"峰值 {_ccd_scalar(peak, 0) if peak is not None else '—'}")
        fwhm = _ccd_first(stats, ("fwhm_px", "fwhm", "width_px"))
        if fwhm is not None:
            parts.append(f"FWHM {_ccd_scalar(fwhm, 1)} px")
        centroid = _ccd_first(stats, ("centroid_px", "centroid", "center"))
        text = "—"
        if centroid is not None:
            try:
                text = (
                    f"({_ccd_scalar(centroid[0], 0)}, "
                    f"{_ccd_scalar(centroid[1], 0)})"
                )
            except (TypeError, IndexError, KeyError):
                text = "—"
        parts.append(f"质心 {text}")
        label.setText(" · ".join(parts))

    @Slot(object)
    def _on_ccd_frame(self, payload: object) -> None:
        """采样线程回传的一帧：dict 里取图与统计，非 dict 就当灰度图。"""
        stats: object = payload
        frame: np.ndarray | None = None
        if isinstance(payload, dict):
            for key in ("frame", "image", "gray"):
                candidate = payload.get(key)
                if isinstance(candidate, np.ndarray):
                    frame = candidate
                    break
        elif isinstance(payload, np.ndarray):
            frame = payload
            stats = None
        if frame is not None:
            self._ccd_preview.set_image(_gray_to_qimage(_to_uint8_image(frame)))
        self._ccd_frames += 1
        self._render_ccd_stat(stats)
        self._set_ccd_status(
            "running",
            f"● 采样中 —— 已收到 {self._ccd_frames} 帧（与下发并行运行）",
        )

    @Slot(str)
    def _on_ccd_status(self, text: str) -> None:
        self._set_ccd_status("running", f"● {text}")

    @Slot(object)
    def _on_ccd_summary(self, summary: object) -> None:
        status = ""
        if isinstance(summary, dict):
            status = str(summary.get("status") or "").strip()
        # 采样失败时 failed 先红过一次；这里按 summary 的状态别把红盖回成功色。
        kind = "error" if status == "failed" else "done"
        mark = "⚠" if kind == "error" else "■"
        self._set_ccd_status(kind, f"{mark} 采样结束 —— {_ccd_summary_text(summary)}")
        self._shutdown_ccd_thread()

    @Slot(str)
    def _on_ccd_failed(self, message: str) -> None:
        self._set_ccd_status(
            "error", f"⚠ CCD 采样失败：{message}（SLM 下发不受影响）"
        )
        self._shutdown_ccd_thread()

    def _prepare_ccd_dispatch(self) -> object | None:
        """下发前先建好快照：第一条进度就不丢。未启用或模块缺失返回 None。"""
        self._ccd_dispatch = None
        self._ccd_monitor = None
        self._ccd_frames = 0
        if not self._ccd_enabled.isChecked():
            return None
        try:
            from ao_shaping.gui.slm import sync_ccd_monitor as monitor
        except Exception as exc:  # 模块还没装/写坏：退化成纯下发，界面必须能用
            self._set_ccd_status(
                "error", f"⚠ CCD 同步监测不可用：{exc}（本次只下发 SLM）"
            )
            return None
        try:
            dispatch = monitor.DispatchSnapshot()
        except (TypeError, AttributeError) as exc:
            # 接口不匹配只可能是模块版本对不上：退化成纯下发，界面必须能用
            self._set_ccd_status(
                "error", f"⚠ 无法创建下发快照：{exc}（本次只下发 SLM）"
            )
            return None
        self._ccd_monitor = monitor
        self._ccd_dispatch = dispatch
        self._ccd_preview.set_image(None)
        self._ccd_preview.set_empty_text("等待第一张 CCD 帧…")
        self._render_ccd_stat(None)
        return dispatch

    @Slot(int, int, float)
    def _on_ccd_dispatch_progress(self, sent: int, total: int, elapsed: float) -> None:
        del total  # 采样快照只关心「第几帧 / 用了多久」
        dispatch = self._ccd_dispatch
        if dispatch is None:
            return
        try:
            dispatch.update(int(sent), float(elapsed))
        except Exception as exc:  # 观测失败不该影响下发
            logger.warning("CCD 下发快照更新失败: {}", exc)

    def _start_ccd_sampler(self, dispatch: object | None) -> None:
        """把采样器挪进自己的线程再启动；相机开在那条线程里。"""
        monitor = self._ccd_monitor
        if dispatch is None or monitor is None:
            return
        try:
            sampler = monitor.CCDSyncSampler(
                camera_type=str(self._ccd_camera.currentData()),
                cam_id=int(self._ccd_cam_id.value()),
                exposure_ms=float(self._ccd_exposure.value()),
                dispatch_state=dispatch,
                strategy=str(self._ccd_strategy.currentData()),
                every_n=max(1, int(self._ccd_every_n.value())),
                interval_s=float(self._ccd_interval.value()),
            )
        except (TypeError, ValueError) as exc:
            # 采样器构造只做参数校验；参数非法不该拖住 SLM 下发
            self._set_ccd_status(
                "error", f"⚠ 无法创建 CCD 采样器：{exc}（SLM 下发照常）"
            )
            return

        for name, slot in (
            ("frame_ready", self._on_ccd_frame),
            ("status_changed", self._on_ccd_status),
            ("finished_summary", self._on_ccd_summary),
            ("failed", self._on_ccd_failed),
        ):
            signal = getattr(sampler, name, None)
            if signal is not None:
                signal.connect(slot)

        # fixer 的实现本身就是 QThread：直接 start，不要再套一层线程；
        # 若换成普通 QObject 采样器，再按 moveToThread 的老路走。
        if isinstance(sampler, QThread):
            thread = sampler
            thread.finished.connect(self._on_ccd_thread_finished)
        else:
            thread = QThread(self)
            sampler.moveToThread(thread)
            entry = getattr(sampler, "start", None) or getattr(sampler, "run", None)
            if callable(entry):
                thread.started.connect(entry)
            thread.finished.connect(sampler.deleteLater)
            thread.finished.connect(self._on_ccd_thread_finished)
        self._ccd_sampler = sampler
        self._ccd_thread = thread
        self._set_ccd_status(
            "running", "● 启动 CCD 采样线程 —— 相机在该线程打开，不占界面线程"
        )
        thread.start()

    def _mark_ccd_dispatch_done(self) -> None:
        """下发结束（完成/停止/失败/意外退出）都要通知采样器收尾。"""
        dispatch = self._ccd_dispatch
        if dispatch is None:
            return
        try:
            dispatch.mark_done()
        # policy: 收尾通知是尽力而为 —— 这里抛错会挡住关窗/结论回填，采样器另有看门狗兜底
        except Exception as exc:
            logger.warning("CCD 下发快照 mark_done 失败: {}", exc)

    def _schedule_ccd_watchdog(self) -> None:
        if self._ccd_thread is None:
            return
        QTimer.singleShot(_CCD_SHUTDOWN_GRACE_MS, self._on_ccd_watchdog)

    def _on_ccd_watchdog(self) -> None:
        thread = self._ccd_thread
        if thread is None or self._closing or not thread.isRunning():
            return
        self._set_ccd_status(
            "error",
            "⚠ 采样线程一直没收尾 —— 已请求退出；相机可能仍被别的程序占着，"
            "关窗时会再等一次。",
        )
        self._request_ccd_stop()
        thread.quit()

    def _request_ccd_stop(self) -> None:
        """优先走采样器自己的停止口；它不认 quit()（自写 run 的 QThread）。"""
        sampler = self._ccd_sampler
        stop = getattr(sampler, "request_stop", None) if sampler is not None else None
        if callable(stop):
            try:
                stop()
                return
            # policy: 停止口是可选协议 —— 采样器实现可能没有/抛出，关窗路径不能因此中断
            except Exception as exc:
                logger.warning("CCD 采样器停止请求失败: {}", exc)
        thread = self._ccd_thread
        if thread is not None and thread.isRunning():
            thread.requestInterruption()

    def _shutdown_ccd_thread(self, wait_ms: int = 0) -> None:
        thread = self._ccd_thread
        if thread is None:
            return
        if thread.isRunning():
            self._request_ccd_stop()
            thread.quit()
            if wait_ms and not thread.wait(wait_ms):
                logger.warning("CCD 采样线程未在 {} ms 内退出", wait_ms)

    @Slot()
    def _on_ccd_thread_finished(self) -> None:
        thread = self._ccd_thread
        self._ccd_thread = None
        self._ccd_sampler = None
        if thread is not None:
            thread.deleteLater()

    # ------------------------------------------------------------------ 运行

    @Slot()
    def _on_start_clicked(self) -> None:
        if self._state in ("running", "paused", "connecting"):
            return
        try:
            plan = self._current_plan()
        except ValueError as exc:
            QMessageBox.warning(self, "无法开始过渡", str(exc))
            return
        # 参数可能刚变、后台还没算完：同步补算一次，保证下发的相位与表单一致。
        try:
            snapshot = build_plan_snapshot(**self._compute_kwargs())
        except Exception as exc:  # 生成失败（参数/形状）不进入设备路径
            QMessageBox.warning(self, "无法开始过渡", f"相位生成失败：{exc}")
            return
        self._revision = max(self._revision, snapshot.revision)
        self._snapshot = snapshot
        self._render_far_fields(snapshot)
        self._render_estimate(snapshot.estimate)

        start_mode = self._start_combo.currentData()
        if start_mode == "previous" and self._last_gray is None:
            QMessageBox.warning(
                self,
                "无法开始过渡",
                "「本页上一帧」还没有内容：请先完成一次过渡，或改选平场/自定义起点。",
            )
            return

        slm_number = self._slm_spin.value()
        source = self._optical_combo.itemData(self._optical_combo.currentIndex())
        start_label = self._START_LABELS[str(start_mode)]
        warning = ""
        if source in ("model_forward", "unknown"):
            warning = "\n\n⚠ 本次基于模型前向假设，未经 CCD 验证。"
        reply = QMessageBox.question(
            self,
            "开始动态过渡",
            f"即将向 SLM {slm_number} 连续下发 {snapshot.estimate.frames} 帧，"
            f"用时约 {snapshot.estimate.expected_duration_s:.1f} s。\n"
            f"远场：{start_label} → {PATTERN_TYPES[self._pattern_key].label}\n\n"
            f"每步停留约 {snapshot.estimate.dwell_s * 1000:.1f} ms"
            "（不低于像素翻转时间）；暂停保持当前帧，停止停在过渡中间。"
            "确认光路与相机安全后开始。" + warning,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        worker = DynamicShapingWorker(
            slm_number=slm_number,
            wavelength=int(self._wavelength_spin.value()),
            target_phase=snapshot.target_phase,
            plan=plan,
            start_mode=str(start_mode),
            start_phase=snapshot.start_phase,
            start_gray=snapshot.start_gray if start_mode == "previous" else None,
            pattern_type=self._pattern_key,
            pattern_label=PATTERN_TYPES[self._pattern_key].label,
            params=self._target_form.values(),
            start_params=self._start_form.values() if start_mode == "phase" else None,
            optical_path={
                "source": source,
                "calibrated_at": self._optical_path.calibrated_at,
                "note": self._optical_path.note,
            },
        )
        worker.opened.connect(self._on_worker_opened)
        worker.progress.connect(self._on_progress)
        worker.preview.connect(self._on_frame_preview)
        worker.paused.connect(self._on_paused)
        worker.finished_ok.connect(self._on_finished)
        worker.stopped.connect(self._on_stopped)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker

        # CCD 同步监测（可选）：快照必须先建好，第一条进度才不丢；采样线程在
        # worker 启动之后再开，相机打开失败也不影响这一轮下发。
        dispatch = self._prepare_ccd_dispatch()
        if dispatch is not None:
            worker.progress.connect(self._on_ccd_dispatch_progress)

        self._run_frames = (0, snapshot.estimate.frames)
        self._bar.setRange(0, snapshot.estimate.frames)
        self._bar.setValue(0)
        self._frame_label.setText(f"0 / {snapshot.estimate.frames}")
        self._elapsed_label.setText(f"已用 0.0 s / 预计 {snapshot.estimate.expected_duration_s:.1f} s")
        self._live_panel.set_image(None)
        self._live_panel.set_empty_text("正在连接 SLM 并准备首帧…")
        self._apply_state("connecting")
        worker.start()
        self._start_ccd_sampler(dispatch)

    @Slot()
    def _on_worker_opened(self) -> None:
        self._apply_state("running")
        self._set_status(
            "running",
            "过渡中 —— 单句柄逐帧写入（轮换内存槽），相位分中间步逼近目标。",
        )

    def _on_progress(self, sent: int, total: int, elapsed: float) -> None:
        self._bar.setRange(0, max(1, total))
        self._bar.setValue(sent)
        self._run_frames = (sent, total)
        self._frame_label.setText(f"{sent} / {total}")
        # 只报「已用 / 名义时长」：面板翻转慢时实际会更久，报「剩余」会先归零。
        nominal = self._current_plan().duration_s
        self._elapsed_label.setText(f"已用 {elapsed:.1f} s / 名义 {nominal:.1f} s")

    @Slot(object)
    def _on_frame_preview(self, thumb: np.ndarray) -> None:
        self._live_panel.set_image(_gray_to_qimage(np.asarray(thumb)))

    @Slot(bool)
    def _on_paused(self, paused: bool) -> None:
        if paused:
            self._apply_state("paused")
            sent, total = self._run_frames
            self._set_status(
                "paused",
                f"已暂停 —— 面板停在当前帧（第 {sent} / {total} 帧），"
                "点「继续」从同一进度接续。",
            )
        else:
            self._apply_state("running")
            self._set_status("running", "已继续 —— 从暂停处接着演化。")

    def _on_finished(self, sent: int, elapsed: float) -> None:
        self._last_gray = self._snapshot.target_gray.copy() if self._snapshot else None
        self._bar.setValue(sent)
        self._frame_label.setText(f"{sent} / {sent}")
        self._elapsed_label.setText(f"已用 {elapsed:.1f} s")
        self._set_status(
            "done", f"过渡完成 —— {sent} 帧，用时 {elapsed:.1f} s；面板已停在目标形状。"
        )
        self._mark_ccd_dispatch_done()
        self._apply_state("finished")

    def _on_stopped(self, sent: int, elapsed: float) -> None:
        self._bar.setValue(sent)
        self._frame_label.setText(f"{sent} / {self._run_frames[1]}")
        self._elapsed_label.setText(f"已用 {elapsed:.1f} s")
        self._set_status(
            "paused",
            f"已停止 —— 面板停在过渡中的第 {sent} 帧（未到目标）。"
            "可改参数重新开始，或把起点选为「本页上一帧」从该处接续。",
        )
        self._apply_state("stopped")
        self._refresh_previous_available()
        self._mark_ccd_dispatch_done()

    def _on_failed(self, sent: int, message: str) -> None:
        self._bar.setValue(sent)
        self._frame_label.setText(f"{sent} / {self._run_frames[1]}")
        self._set_status("error", f"过渡中断：{message}")
        self._mark_ccd_dispatch_done()
        self._apply_state("failed")

    @Slot()
    def _on_worker_finished(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()
        # 线程收完了：无论它发没发结论，采样器都该收尾；再给一个宽限观察窗。
        self._mark_ccd_dispatch_done()
        self._schedule_ccd_watchdog()
        if self._state in ("running", "paused", "connecting"):
            # 线程结束但没有发出结论信号：按中断处理，绝不假装完成。
            self._set_status("error", "过渡线程意外结束，未发出结果；面板停在最后一帧。")
            self._apply_state("failed")

    @Slot()
    def _on_pause_resume(self) -> None:
        worker = self._worker
        if worker is None:
            return
        if self._state == "running":
            worker.request_pause()
        elif self._state == "paused":
            worker.request_resume()

    @Slot()
    def _on_stop_clicked(self) -> None:
        worker = self._worker
        if worker is not None and self._state in ("running", "paused", "connecting"):
            self._stop_btn.setEnabled(False)
            self._set_status("paused", "正在停止 —— 写完当前帧后关闭设备…")
            worker.request_stop()

    # ------------------------------------------------------------------ 状态

    def _apply_state(self, state: str) -> None:
        """按状态启用/禁用控件。运行中锁表单，防止界面与面板不一致。"""
        self._state = state
        busy = state in ("running", "paused", "connecting")
        idle_ok = state in ("idle", "finished", "stopped", "failed")
        self._start_btn.setEnabled(idle_ok and self._snapshot is not None)
        self._preview_btn.setEnabled(idle_ok)
        self._pause_btn.setEnabled(state in ("running", "paused"))
        self._pause_btn.setText("继续" if state == "paused" else "暂停")
        self._stop_btn.setEnabled(state in ("running", "paused", "connecting"))
        for widget in (
            self._pattern_combo,
            self._target_form,
            self._start_combo,
            self._start_form,
            self._duration_slider,
            self._duration_spin,
            self._fps_spin,
            self._dwell_spin,
            self._easing_combo,
            self._optical_combo,
            self._slm_spin,
            self._wavelength_spin,
        ):
            widget.setEnabled(not busy)

        self._refresh_ccd_controls()

        if state == "idle":
            self._set_status("idle", "就绪 —— 设定目标与过渡节奏后点「预览过渡」。")
        elif state == "connecting":
            self._set_status("running", "正在连接 SLM 并准备首帧…")
        elif state == "finished":
            self._live_panel.set_empty_text("过渡已完成")

    def _set_status(self, kind: str, text: str) -> None:
        """状态横幅：左侧色条表示状态，底色是同色的极浅底（暗/亮主题都可读）。"""
        color = _STATUS_COLORS.get(kind, _STATUS_COLORS["idle"])
        base = QColor(color)
        # QColor 的 HexArgb 带 alpha=255，直接当背景会盖掉文字；改用极浅的
        # 同色底：把颜色亮度拉高、饱和压低。
        tint = QColor.fromHsv(
            base.hue(), max(0, int(base.saturation() * 0.25)), 250, 70
        )
        self._status.setStyleSheet(
            "QLabel {"
            f" border-left: 4px solid {color};"
            " padding: 7px 10px; border-radius: 4px;"
            f" background: rgba({tint.red()}, {tint.green()}, {tint.blue()}, 70);"
            " }"
        )
        self._status.setText(text)

    # ------------------------------------------------------------------ 生命周期

    def closeEvent(self, event) -> None:
        """关窗即停：不让过渡线程在后台继续写面板。"""
        self._closing = True
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.request_stop()
            worker.wait(3000)
        # 采样器不拿到 mark_done 会一直等下发状态；关窗时再给它一次有界等待。
        self._mark_ccd_dispatch_done()
        self._shutdown_ccd_thread(wait_ms=3000)
        self._debounce.stop()
        super().closeEvent(event)


class ShapingPage(QWidget):
    """导航页 2 的容器：「动态过渡」为主，「单帧下发」保留原静态能力。

    把两者并成一张页而不是再加一个导航项：用户要的是「整形」这一件事的两种
    时间尺度 —— 连续演化与一次定格。默认落在动态页，静态页仍可一键切换。
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        optical_path: OpticalPathSnapshot | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        self.dynamic_page = DynamicShapingPage(optical_path=optical_path)
        self.static_page = StaticShapingPage(optical_path=optical_path)
        tabs.addTab(self.dynamic_page, "动态过渡")
        tabs.addTab(self.static_page, "单帧下发")
        tabs.setCurrentIndex(0)
        layout.addWidget(tabs)
        self.tabs = tabs

    def closeEvent(self, event) -> None:
        self.dynamic_page.close()
        super().closeEvent(event)


def _ccd_first(stats: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """按候选键名取第一个非 None 的统计量（采样器的键名可能因版本而异）。"""
    for key in keys:
        value = stats.get(key)
        if value is not None:
            return value
    return None


def _ccd_scalar(value: object, digits: int) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _ccd_summary_text(summary: object) -> str:
    """采样器收尾对象 → 一行中文；不认识的类型就给截断的字符串。"""
    if isinstance(summary, dict):
        # 真采样器报 sample_count；其它实现可能报 frames / count，一并认。
        count = _ccd_first(
            summary, ("sample_count", "frames", "count", "n_frames", "frames_sampled")
        )
        skipped = _ccd_first(summary, ("skipped", "n_skipped", "missed"))
        duration = _ccd_first(summary, ("duration_s", "elapsed_s"))
        error = summary.get("error")
        status = str(summary.get("status") or "").strip()
        parts: list[str] = []
        head = {
            "finished": "正常收尾",
            "stopped": "已手动停止",
            "failed": "失败中止",
        }.get(status)
        if head:
            parts.append(head)
        if count is not None:
            parts.append(f"共 {_ccd_scalar(count, 0)} 帧")
        if duration is not None:
            parts.append(f"用时 {_ccd_scalar(duration, 1)} s")
        if skipped is not None:
            parts.append(f"跳过 {_ccd_scalar(skipped, 0)} 次触发")
        if error:
            parts.append(f"原因：{error}")
        if parts:
            return "，".join(parts)
        return "已收尾"
    text = str(summary).strip()
    return text[:160] if text else "已收尾"


def _to_uint8_image(gray: np.ndarray) -> np.ndarray:
    """任意位深的灰度 → uint8（按本帧最大值归一，只用于预览）。"""
    arr = np.asarray(gray)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.dtype == np.uint8:
        return arr
    data = np.nan_to_num(arr.astype(np.float64))
    peak = float(data.max()) if data.size else 0.0
    if peak <= 0:
        return np.zeros(data.shape, dtype=np.uint8)
    return np.clip(data / peak * 255.0, 0, 255).astype(np.uint8)


def _gray_to_qimage(gray: np.ndarray) -> QImage:
    """uint8 灰度 → QImage（拷贝，避免底层数组被回收）。"""
    arr = np.ascontiguousarray(gray, dtype=np.uint8)
    h, w = arr.shape
    return QImage(arr.data, w, h, int(arr.strides[0]), QImage.Format.Format_Grayscale8).copy()


def _rgb_to_qimage(rgb: np.ndarray) -> QImage:
    """RGB uint8 → QImage（拷贝）。"""
    arr = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, _ = arr.shape
    return QImage(arr.data, w, h, int(arr.strides[0]), QImage.Format.Format_RGB888).copy()
