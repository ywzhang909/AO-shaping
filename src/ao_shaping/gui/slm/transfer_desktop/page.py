"""迁移学习闭环整形的 PySide6 桌面页。

页面形状 = 操作顺序: ① 设备 → ② 目标 → ③ 校验 → ④ 运行 → ⑤ 进度。①–④ 排成左侧
560 px 的可滚动窄列, ⑤ 固定在右侧 —— 会话动辄十几分钟, 操作员要一边盯阶段表一边回头
改参数, 不该为了看进度把表单滚走。

P1 只暴露必设参数
-----------------
表单只给相机类型、cam_id、SLM 编号、目标形状、目标边长、载波周期。其余字段 (探针数、
拟合迭代、锚定块数、验收口径、检查点路径……) 一律用 :class:`TransferFormValues` 的
默认值透传 —— 它们默认值已镜像 Streamlit 的 ``TRANSFER_RECIPE``, 少开一个控件就少一处
两套默认值漂移的机会。

线程: ``run_session`` 同步、动辄几十分钟, 所以会话不在 GUI 线程跑。本页自持一个
``maxThreadCount=1`` 的 :class:`QThreadPool` (同一台设备上并发跑两场只会互相踩设备),
由 :class:`_SessionTask` 在后台线程里**才** import 并调用它 —— 模块级 import 会把 torch
与整条管线拖进 GUI 冷启动。后台线程只经信号回主线程 (阶段 → ``stage``, 结束 →
``finished`` / ``failed``), 绝不碰控件: 跨线程改控件是 Qt 里最常见的崩法。

进度: ``on_stage`` 只说「阶段名 + 三态」, 说不出「这一阶段测出了什么」; 明细只能从会话
目录的 ``journal.json`` 与各阶段产物里读回来, :func:`collect_progress` 已把这层读法封成
「产物缺失也绝不抛异常」的纯函数。信号保证及时, 1 s 的 ``QTimer`` 保证内容。

校验: 控件一变就重跑 :func:`validate_form` (它内部惰性 import torch, 首次约数秒)。有
error 就禁用「开始运行」并把第一条写进 tooltip, warning 只提示。校验结论只来自 params
层, 页面不另写一套判据。

自检: 「设备自检」按钮显式触发一次只读硬件枚举 (相机列表 + 所选 SLM 探针), 结论落在
① 设备区。它**不**挂在 ``_revalidate`` 上 —— 校验是纯参数判断, 每改一个数字就跑一遍
硬件枚举既慢又会让「通过」变成噪音; 反过来, 设备参数改过之后旧结论只会标成「已过期」,
不会自动重跑。自检本身走 :class:`HardwareCheckTask` (后台线程), 与运行中的会话互斥。

读取零级: 「读取零级」按钮同样只在点击时发生, 走 :class:`_ZeroOrderTask` (后台线程,
相机 SDK 不在 UI 线程里调), 同样与自检 / 运行互斥。「目标边长」的控件上限因此不再
写死 400 px, 而是由实测 0 级到最近帧边的距离决定 (:mod:`~ao_shaping.gui.slm.transfer_desktop.zero_order`);
采样窗半跨随之按边长派生 (``params.effective_window_span``), 所以 ``side=900`` 这类
大目标能通过校验 —— 前提确实是 0 级离帧边够远。

关窗: ``run_session`` 没有取消接口, 所以关窗**不假装能停** —— 停轮询, 最多等
:data:`_CLOSE_GRACE_S` 秒让在飞作业落地 (期间泵事件队列, 让完成信号能送达), 超时就把
窗口关掉; 会话本身仍在后台线程里按计划跑完并写进会话目录。

配色: 状态色 (红 = 阻断 / 失败, 黄 = 提醒, 绿 = 通过 / 完成, 蓝 = 进行中) 按当前调色
板的明暗取两档, 并跟随 ``PaletteChange`` 重刷 —— 写死浅色字会在亮色主题里变成白底白字。
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from html import escape
from pathlib import Path
from typing import Any

from loguru import logger
from PySide6.QtCore import QEvent, QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QColor, QFont, QPalette
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ao_shaping.gui.slm.transfer_desktop import params
from ao_shaping.gui.slm.transfer_desktop.check_task import HardwareCheckTask
from ao_shaping.gui.slm.transfer_desktop.live_view import LiveView
from ao_shaping.gui.slm.transfer_progress import (
    MISSING,
    STATUS_LABELS,
    collect_progress,
)

__all__ = ["TransferPage"]

#: 表单默认值。单一口径取自 :mod:`params` (那边用字面量钉住了 Streamlit 的同名默认值)。
_DEFAULTS = params.TransferFormValues()

#: 进度轮询间隔 (ms)。``journal.json`` 同步落盘, 1 s 足够跟上分钟级的阶段。
_POLL_MS = 1000

#: 左侧表单列宽度 (px)。够放下「标签 + 控件」, 又不至于把右侧阶段表挤没。
_FORM_WIDTH = 560

#: 关窗时等在飞会话落地的上限 (s)。会话没有取消接口, 无限等会把宿主窗口冻住。
_CLOSE_GRACE_S = 5.0

#: 「目标边长」的保守上限 (px)。没有量过 0 级位置时只能给这个值 —— 采样窗默认
#: 半跨 220 px 扣掉两侧留白后正好 424 px, 400 是留了余量的保守档。
#: 点「读取零级」后按实测 0 级到帧边的距离放宽 (物理上限), 见 ``zero_order``。
_SIDE_MAX_CONSERVATIVE_PX = 400.0

#: 相机类型的中文说明。代码值与 ``CAMERA_CHOICES`` 一一对应。
_CAMERA_LABELS: dict[str, str] = {
    "daheng": "大恒（已验证）",
    "miicam": "MiiCam（仅有接口）",
    "sim": "仿真源（仅有接口）",
}

#: 状态色: (亮色主题, 暗色主题) 各一档 —— 写死浅色字会在亮色主题里变成「白底白字」。
_STATUS_COLORS: dict[str, tuple[str, str]] = {
    "error": ("#b3261e", "#ff8a80"),
    "failed": ("#b3261e", "#ff8a80"),
    "warn": ("#8a5a00", "#ffcc66"),
    "ok": ("#1a7f37", "#7ee0a1"),
    "complete": ("#1a7f37", "#7ee0a1"),
    "running": ("#0b5cad", "#7cbcff"),
    "pending": ("#5f6368", "#9aa0a6"),
    "skipped": ("#5f6368", "#9aa0a6"),
}

#: ``CheckItem.level`` → 状态色档位。``fail`` 必须显式映射到 ``error``:
#: ``_STATUS_COLORS`` 里没有 ``fail`` 这一档, 直接查表会静默退成 ``pending`` 的灰。
_CHECK_LEVELS: dict[str, str] = {"ok": "ok", "warn": "warn", "fail": "error"}


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def _palette_colors(palette: QPalette) -> dict[str, str]:
    """按调色板明暗挑一档状态色。"""
    dark = palette.color(QPalette.ColorRole.Window).lightness() < 128
    return {kind: pair[1 if dark else 0] for kind, pair in _STATUS_COLORS.items()}


def _rich(text: str) -> str:
    """纯文本结论 → 富文本片段 (转义, 保留结论里的换行)。"""
    return escape(text).replace("\n", "<br>")


def _duration_text(seconds: float | None) -> str:
    """阶段耗时 → 表格文字。``None`` 是「没测到」, 不是 0 s。"""
    return MISSING if seconds is None else f"{seconds:.1f} s"


def _stage_summary(stage: Mapping[str, Any]) -> str:
    """阶段表末列: 失败给错误, 否则把关键指标压成一行。"""
    error = stage.get("error")
    if error:
        return " ".join(str(error).split())
    metrics = stage.get("metrics") or ()
    return " · ".join(f"{metric['label']} {metric['text']}" for metric in metrics)


def _section(title: str) -> tuple[QGroupBox, QVBoxLayout]:
    """带序号的分组框 (序号就是操作顺序) + 内边距已设好的竖排布局。"""
    box = QGroupBox(title)
    font = box.font()
    font.setWeight(QFont.Weight.DemiBold)
    box.setFont(font)
    layout = QVBoxLayout(box)
    layout.setContentsMargins(10, 8, 10, 10)
    layout.setSpacing(8)
    return box, layout


def _bold(widget: QWidget) -> QWidget:
    """把一个控件刷成强调字重 (返回同一个控件, 便于链式使用)。"""
    font = widget.font()
    font.setWeight(QFont.Weight.DemiBold)
    widget.setFont(font)
    return widget


# ---------------------------------------------------------------------------
# 后台会话
# ---------------------------------------------------------------------------


class _WorkerSignals(QObject):
    """后台会话 → 主线程的单向通道 (``QRunnable`` 不是 ``QObject``, 发不出信号)。"""

    stage = Signal(str, str)
    #: 每块一帧的实时画面 (``dict`` 负载, 含 ``frame`` 数组与整数进度)。
    frame = Signal(dict)
    finished = Signal(str)
    failed = Signal(str)


class _SessionTask(QRunnable):
    """在后台线程跑一整场迁移学习会话。"""

    def __init__(self, values: params.TransferFormValues, out: Path) -> None:
        super().__init__()
        self._values = values
        self._out = out
        self.signals = _WorkerSignals()

    def run(self) -> None:
        """线程池入口 (在后台线程执行)。"""
        # 只在后台线程 import: 模块级 import 会拖起 torch 与整条管线。
        from ao_shaping.tools.slm.transfer.workflow import run_session

        try:
            session = params.build_session(self._values, out=self._out)
            run_session(session, on_stage=self._on_stage, on_frame=self._on_frame)
        # policy: 后台线程的任何失败都必须变成界面上的结论; 这里只发信号、不吞细节,
        # 也不让异常逃出 QRunnable (Qt 会直接终止线程, 界面就永远停在「运行中」)。
        except Exception as exc:
            logger.warning("迁移学习会话失败: {}", exc)
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.signals.finished.emit(str(self._out))

    def _on_stage(self, name: str, status: str) -> None:
        """``run_session`` 的阶段回调 (后台线程): 只发信号, 不碰控件。"""
        self.signals.stage.emit(name, status)

    def _on_frame(self, payload: dict) -> None:
        """``run_session`` 的帧回调 (后台线程): 只发信号, 不碰控件。"""
        self.signals.frame.emit(payload)


class _ZeroSignals(QObject):
    """「读取零级」后台任务 → 主线程 (``QRunnable`` 自己发不出信号)。"""

    #: ``ZeroOrderReading`` 或异常实例 —— 全暗 / 贴边也是要上屏的结论。
    done = Signal(object)


class _ZeroOrderTask(QRunnable):
    """在后台线程开一次相机读 0 级 (**只读**, 不碰 SLM)。

    打开相机、改曝光、抓帧都是 SDK 调用, 快则几十毫秒、卡住则秒级; 落在 UI
    线程会把窗口冻住。与 :class:`~ao_shaping.gui.slm.transfer_desktop.check_task.HardwareCheckTask`
    同一套做法: 工作线程只发信号。
    """

    def __init__(self, camera: str, cam_id: int) -> None:
        super().__init__()
        self._camera = camera
        self._cam_id = cam_id
        self.signals = _ZeroSignals()

    def run(self) -> None:
        """线程池入口 (在后台线程执行)。"""
        # 惰性 import: ``zero_order`` 会拉起相机 SDK, 页面冷启动不该先付这笔
        # 代价; 也是测试 patch ``zero_order.read_zero_order`` 的接缝。
        from ao_shaping.gui.slm.transfer_desktop.zero_order import read_zero_order

        try:
            reading = read_zero_order(self._camera, self._cam_id)
        # policy: 工作线程不碰界面; 任何失败都变成一条回到主线程的结论
        except Exception as exc:
            logger.opt(exception=exc).warning("读取零级失败")
            self.signals.done.emit(exc)
        else:
            self.signals.done.emit(reading)


#: 在飞会话的强引用。页面先关、线程后完时任务对象不能被回收, 否则后台线程还在
#: 跑的对象会先消失。平时由页面的 ``_task`` 持有, 这里只兜关窗那一段。
_LIVE_TASKS: set[_SessionTask] = set()


# ---------------------------------------------------------------------------
# 页面
# ---------------------------------------------------------------------------


class TransferPage(QWidget):
    """迁移学习闭环整形的桌面页 (设备 → 目标 → 校验 → 运行 → 进度)。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)

        self._task: _SessionTask | None = None
        self._running = False
        self._session_dir: Path | None = None
        # 自检只在点按钮时发生。``_check_task`` 是强引用 (与 ``_task`` 同理, 掉引用
        # 会让在飞任务被回收); 结论存成纯数据, 换主题时靠它重上色。
        self._checking = False
        self._check_task: HardwareCheckTask | None = None
        self._check_key: tuple[str, int, int] | None = None
        self._check_heading: tuple[str, str] | None = None
        self._check_rows: list[tuple[str, str, str]] = []
        # 「读取零级」同理只在点按钮时发生; ``_zero_task`` 是强引用, 掉引用会让
        # 在飞任务被回收 (信号永远送不到)。``_zero_reading`` 同时是互斥标志。
        self._zero_reading = False
        self._zero_task: _ZeroOrderTask | None = None
        self._zero_limit: int = int(_SIDE_MAX_CONSERVATIVE_PX)
        #: ② 目标区常驻说明的 (色档, 正文)；``None`` = 还没读过零级。
        self._zero_state: tuple[str, str] | None = None
        self._last_errors: list[str] = []
        self._last_warnings: list[str] = []
        self._status_kind = "pending"
        self._status_message = "尚未运行。参数自检通过后即可启动。"
        self._colors: dict[str, str] = _palette_colors(self.palette())

        self._poll = QTimer(self)
        self._poll.setInterval(_POLL_MS)
        self._poll.timeout.connect(self._refresh_progress)

        self._build()
        self._apply_theme()
        # 首次校验会 import torch (数秒), 但按钮状态必须一开始就是真的:
        # 同步校验, 免得「能点」被读成「参数没问题」。
        self._revalidate()

    # ------------------------------------------------------------------ 布局
    def _build(self) -> None:
        """搭出「标题 + 左表单 / 右进度」两栏。"""
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(12)
        root.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setSpacing(16)
        body.addWidget(self._build_form(), 0)
        body.addWidget(self._build_right(), 1)
        root.addLayout(body, 1)

    def _build_right(self) -> QWidget:
        """右栏: ⑥ 实时画面 (CCD / 剖面 / 块进度) + ⑤ 阶段进度表。"""
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        box, live_layout = _section("⑥ 实时画面（每块最后一帧）")
        self._live_view = LiveView()
        live_layout.addWidget(self._live_view)
        layout.addWidget(box, 1)
        layout.addWidget(self._build_progress(), 1)
        return host

    def _build_header(self) -> QWidget:
        """标题与一句话说明。"""
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        title = QLabel("迁移学习闭环整形")
        font = title.font()
        font.setPointSize(font.pointSize() + 5)
        font.setWeight(QFont.Weight.DemiBold)
        title.setFont(font)

        subtitle = QLabel(
            "物理预测模型 · 探针采集 · CCD 锚定验收：一次会话跑完采集、迁移、整形与审计"
        )
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        return host

    def _build_form(self) -> QWidget:
        """左侧可滚动表单: ① 设备 → ② 目标 → ③ 校验 → ④ 运行。"""
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(self._build_device())
        layout.addWidget(self._build_target())
        layout.addWidget(self._build_checks())
        layout.addWidget(self._build_run())
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(host)
        scroll.setFixedWidth(_FORM_WIDTH)
        return scroll

    def _build_device(self) -> QGroupBox:
        """① 设备: 往哪台硬件下发。"""
        box, layout = _section("① 设备")

        self._camera = self._combo(
            [(kind, _CAMERA_LABELS.get(kind, kind)) for kind in params.CAMERA_CHOICES],
            _DEFAULTS.camera,
        )
        self._cam_id = self._spin(_DEFAULTS.cam_id, 0, 15)
        self._slm_number = self._spin(_DEFAULTS.slm_number, 1, 4)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)
        form.addRow("相机类型", self._camera)
        form.addRow("CCD cam_id", self._cam_id)
        form.addRow("SLM 编号", self._slm_number)
        layout.addLayout(form)

        # 自检按钮: 只有点它才会碰硬件 (枚举相机 + 探针 SLM)。默认可用 —— 参数
        # 校验没过时, 恰恰更需要知道硬件是不是好的。
        self._check_btn = QPushButton("设备自检")
        self._check_btn.setToolTip(
            "只读枚举相机并探测所选 SLM（会开关一次 SLM，不改显示内容）"
        )
        self._check_btn.clicked.connect(self._run_check)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self._check_btn)
        row.addStretch(1)
        layout.addLayout(row)

        self._check_view = QLabel()
        self._check_view.setTextFormat(Qt.TextFormat.RichText)
        self._check_view.setWordWrap(True)
        self._check_view.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._check_view.setVisible(False)
        layout.addWidget(self._check_view)
        return box

    def _build_target(self) -> QGroupBox:
        """② 目标: 跑什么形状、多大、什么载波周期。"""
        box, layout = _section("② 目标与载波")

        self._shapes: dict[str, QCheckBox] = {}
        shape_host = QWidget()
        shape_row = QHBoxLayout(shape_host)
        shape_row.setContentsMargins(0, 0, 0, 0)
        shape_row.setSpacing(12)
        for name in params.SHAPE_CHOICES:
            check = self._check(
                params.SHAPE_LABELS.get(name, name), name in _DEFAULTS.shapes
            )
            self._shapes[name] = check
            shape_row.addWidget(check)
        shape_row.addStretch(1)

        self._side = self._dspin(
            _DEFAULTS.side, 8.0, _SIDE_MAX_CONSERVATIVE_PX, step=5.0
        )
        self._period = self._dspin(_DEFAULTS.period, 4.0, 1024.0, step=1.0)

        # A–B–B–A 配对验收: 默认关 (只按模型损失接受, 跳过 CCD 实测配对与方向核对)。
        self._abba = self._check("A–B–B–A 配对验收（CCD 实测）", _DEFAULTS.abba)
        self._abba.setToolTip(
            "开 = 每块测 A1/B1/B2/A2, 由实测指标判定候选是否接受; "
            "关（默认）= 保留 CCD 锚定反馈, 跳过 B1/B2/A2 与方向核对, "
            "候选只按模型损失接受, 摘要写 acceptance=model_only（等价 CLI --no-abba）"
        )
        # 「读取零级」: 让目标边长的上限来自实测光轴 (0 级到最近帧边的距离),
        # 而不是写死的保守值。做法与自检一致 —— 点了才碰硬件, 且只读。
        self._zero_btn = QPushButton("读取零级")
        self._zero_btn.setToolTip(
            "只读：开一次相机找 0 级，由 0 级到帧边的距离算出目标边长上限。"
            "不改光路、不改 SLM 显示"
        )
        self._zero_btn.clicked.connect(self._read_zero_order)
        side_row = QHBoxLayout()
        side_row.setContentsMargins(0, 0, 0, 0)
        side_row.setSpacing(8)
        side_row.addWidget(self._side, 1)
        side_row.addWidget(self._zero_btn, 0)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)
        form.addRow("目标形状", shape_host)
        form.addRow("目标边长（CCD px）", side_row)
        form.addRow("载波周期（面板 px）", self._period)
        form.addRow("配对验收", self._abba)
        layout.addLayout(form)

        # 常驻说明: 上限从哪来、现在是多少 —— 读不到时也要把「仍是保守值」讲清楚。
        self._zero_view = QLabel()
        self._zero_view.setTextFormat(Qt.TextFormat.RichText)
        self._zero_view.setWordWrap(True)
        self._zero_view.setVisible(True)
        hint_font = self._zero_view.font()
        hint_font.setPointSize(max(hint_font.pointSize() - 1, 7))
        self._zero_view.setFont(hint_font)
        layout.addWidget(self._zero_view)
        self._render_zero()
        return box

    def _build_checks(self) -> QGroupBox:
        """③ 校验: 红字阻断, 黄字提醒。"""
        box, layout = _section("③ 校验")
        self._checks = QLabel("正在校验参数…")
        self._checks.setTextFormat(Qt.TextFormat.RichText)
        self._checks.setWordWrap(True)
        self._checks.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self._checks.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._checks.setMinimumHeight(96)
        layout.addWidget(self._checks)
        return box

    def _build_run(self) -> QGroupBox:
        """④ 运行: 一个按钮 + 状态 + 会话目录。"""
        box, layout = _section("④ 运行")

        self._run_btn = _bold(QPushButton("开始运行"))
        self._run_btn.setMinimumHeight(38)
        # 首次校验通过前不允许启动 —— 默认禁用, 免得「能点」被读成「参数没问题」。
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._start_run)
        layout.addWidget(self._run_btn)

        self._status = QLabel()
        self._status.setTextFormat(Qt.TextFormat.RichText)
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        planned = params.plan_session_dir()
        self._session_label = QLabel(f"会话目录（规划）: {planned}")
        self._session_label.setWordWrap(True)
        self._session_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self._session_label)
        return box

    def _build_progress(self) -> QGroupBox:
        """⑤ 进度: 当前阶段 + 汇总 + 进度条 + 阶段表。"""
        box, layout = _section("⑤ 进度")

        head = QHBoxLayout()
        head.setSpacing(10)
        self._current = _bold(QLabel("尚无阶段计划"))
        self._current.setWordWrap(True)
        self._counts = QLabel("")
        head.addWidget(self._current, 1)
        head.addWidget(self._counts, 0, Qt.AlignmentFlag.AlignRight)
        self._replay_btn = QPushButton("归档回放…")
        self._replay_btn.setToolTip(
            "离线查看已落盘的每块 CCD 帧与 SLM 相位（只读，不碰硬件）"
        )
        self._replay_btn.clicked.connect(self._open_replay)
        head.addWidget(self._replay_btn, 0)
        layout.addLayout(head)

        self._bar = QProgressBar()
        self._bar.setRange(0, 1)
        self._bar.setFormat("已结束 %v / 计划 %m 阶段")
        layout.addWidget(self._bar)

        layout.addWidget(self._build_table(), 1)
        return box

    def _build_table(self) -> QTableWidget:
        """阶段表: 序号 / 阶段 / 状态 / 耗时 / 说明。"""
        table = QTableWidget(0, 5)
        table.setHorizontalHeaderLabels(["#", "阶段", "状态", "耗时", "说明"])
        table.verticalHeader().setVisible(False)
        table.verticalHeader().setDefaultSectionSize(26)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        table.setAlternatingRowColors(True)
        table.setWordWrap(False)
        table.setShowGrid(False)

        header = table.horizontalHeader()
        for column, width in ((0, 42), (2, 76), (3, 72)):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
            header.resizeSection(column, width)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        header.resizeSection(1, 220)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)

        self._table = table
        return table

    # -------------------------------------------------------------- 控件工厂
    # 每个工厂顺手把「改了 → 重算校验」接上: 新加控件时不可能忘记接线。
    def _spin(self, value: int, low: int, high: int, *, step: int = 1) -> QSpinBox:
        """整数框 (范围与 Streamlit 页逐项一致)。"""
        widget = QSpinBox()
        widget.setRange(low, high)
        widget.setSingleStep(step)
        widget.setValue(value)
        # 不跟键: 输入 "1" → "10" → "100" 的中途不该每一下都报警。
        widget.setKeyboardTracking(False)
        widget.valueChanged.connect(self._revalidate)
        return widget

    def _dspin(
        self, value: float, low: float, high: float, *, step: float = 1.0
    ) -> QDoubleSpinBox:
        """浮点框 (目标边长 / 载波周期), 一位小数。"""
        widget = QDoubleSpinBox()
        widget.setRange(low, high)
        widget.setDecimals(1)
        widget.setSingleStep(step)
        widget.setValue(value)
        widget.setKeyboardTracking(False)
        widget.valueChanged.connect(self._revalidate)
        return widget

    def _combo(self, options: Sequence[tuple[str, str]], value: str) -> QComboBox:
        """代码值 + 中文说明的下拉框 (``data`` 存代码, 显示存说明)。"""
        widget = QComboBox()
        for code, label in options:
            widget.addItem(label, code)
        widget.setCurrentIndex(max(widget.findData(value), 0))
        widget.currentIndexChanged.connect(self._revalidate)
        return widget

    def _check(self, text: str, checked: bool) -> QCheckBox:
        """复选框。"""
        widget = QCheckBox(text)
        widget.setChecked(checked)
        widget.toggled.connect(self._revalidate)
        return widget

    # -------------------------------------------------------------- 公开接口
    def form_values(self) -> params.TransferFormValues:
        """控件当前值 → 纯表单快照 (交给 params 层校验与建会话)。"""
        shapes = tuple(
            name for name in params.SHAPE_CHOICES if self._shapes[name].isChecked()
        )
        return params.TransferFormValues(
            shapes=shapes,
            side=float(self._side.value()),
            period=float(self._period.value()),
            camera=str(self._camera.currentData()),
            cam_id=int(self._cam_id.value()),
            slm_number=int(self._slm_number.value()),
            abba=bool(self._abba.isChecked()),
        )

    def current_problems(self) -> tuple[list[str], list[str]]:
        """最近一次校验的 ``(errors, warnings)`` (副本, 调用方随便改)。"""
        return list(self._last_errors), list(self._last_warnings)

    def run_button(self) -> QPushButton:
        """「开始运行」按钮 (自动化验证按它判断可启动性)。"""
        return self._run_btn

    def check_button(self) -> QPushButton:
        """「设备自检」按钮 (自动化验证按它判断自检是否可触发 / 已结束)。"""
        return self._check_btn

    def zero_order_button(self) -> QPushButton:
        """「读取零级」按钮 (自动化验证按它判断读数是否可触发 / 已结束)。"""
        return self._zero_btn

    def check_report(self) -> str:
        """最近一次自检的纯文本结论 (没自检过为空串; 供自动化断言)。"""
        if self._check_heading is None:
            return ""
        lines = [self._check_heading[1]]
        lines.extend(f"{label}：{detail}" for _, label, detail in self._check_rows)
        if self._check_stale():
            lines.append("设备参数已改，请重新点「设备自检」")
        return "\n".join(lines)

    def set_camera(self, kind: str) -> None:
        """程序方式选相机后端, 联动与手工点选相同 (未知后端直接报错)。"""
        index = self._camera.findData(str(kind))
        if index < 0:
            choices = ", ".join(params.CAMERA_CHOICES)
            raise ValueError(f"未知相机类型 {kind!r}（可选 {choices}）")
        self._camera.setCurrentIndex(index)
        self._revalidate()

    def set_period(self, value: float) -> None:
        """程序方式填载波周期 (面板 px), 联动与手工输入相同。"""
        self._period.setValue(float(value))
        self._revalidate()

    def set_shapes(self, shapes: Iterable[str]) -> None:
        """程序方式勾选目标形状; 一次改完只校验一次。"""
        chosen = {str(name) for name in shapes}
        for name, check in self._shapes.items():
            check.blockSignals(True)
            check.setChecked(name in chosen)
            check.blockSignals(False)
        self._revalidate()

    # ------------------------------------------------------------------ 校验
    def _revalidate(self) -> None:
        """改动即校验: 有阻塞就禁用启动, 提醒只上色不拦。

        只做参数判断 —— 硬件自检是「设备自检」按钮的事, 不在这里触发 (否则每敲
        一个数字都枚举一次设备)。这里顺带重画自检区, 只是把旧结论标成过期。
        """
        errors, warnings = params.validate_form(self.form_values())
        self._last_errors = errors
        self._last_warnings = warnings
        self._render_checks()
        self._run_btn.setEnabled(
            not errors
            and not self._running
            and not self._checking
            and not self._zero_reading
        )
        if errors:
            reason = errors[0]
        elif self._running:
            reason = "会话运行中"
        elif self._checking:
            reason = "正在自检设备，稍候"
        elif self._zero_reading:
            reason = "正在读取 0 级，稍候"
        else:
            reason = "启动整场迁移学习会话"
        self._run_btn.setToolTip(reason)
        self._render_check()

    def _render_checks(self) -> None:
        """把最近一次校验结论画成红 / 黄两段 (无结论 = 绿字通过)。"""
        colors = self._colors
        parts: list[str] = []
        for texts, kind, tail, mark in (
            (self._last_errors, "error", "项阻塞 · 修好才能启动", "■"),
            (self._last_warnings, "warn", "项提醒 · 可运行，但请确认", "!"),
        ):
            if texts:
                parts.append(
                    f'<p style="margin:0 0 4px 0;color:{colors[kind]}">'
                    f"<b>{len(texts)} {tail}</b></p>"
                )
                parts.extend(
                    f'<p style="margin:0 0 6px 0;color:{colors[kind]}">'
                    f"{mark} {_rich(text)}</p>"
                    for text in texts
                )
        if not parts:
            parts.append(
                f'<p style="margin:0;color:{colors["ok"]}"><b>参数自检通过</b></p>'
            )
            parts.append(
                f'<p style="margin:4px 0 0 0;color:{self._muted()}">'
                "载波周期与目标尺寸放得进采样窗，探针数是 6 的倍数。</p>"
            )
        self._checks.setText("".join(parts))

    def _set_form_enabled(self, enabled: bool) -> None:
        """运行中锁住必设参数 —— 本次会话用的是启动那一刻的快照。"""
        for widget in (
            self._camera,
            self._cam_id,
            self._slm_number,
            self._side,
            self._period,
            *self._shapes.values(),
        ):
            widget.setEnabled(enabled)
        # 自检同样要锁: 会话正占着 SLM 时再探一次, 只会跟它对抢设备。
        self._check_btn.setEnabled(enabled and not self._checking)
        # 读零级要开相机: 会话跑起来后相机归它用, 关闭时也顺手复位。
        self._zero_btn.setEnabled(
            enabled and not self._checking and not self._zero_reading
        )

    # ------------------------------------------------------------------ 自检
    def _device_key(self) -> tuple[str, int, int]:
        """自检唯一关心的三个字段 (其余表单值与硬件无关)。"""
        values = self.form_values()
        return values.camera, values.cam_id, values.slm_number

    def _check_stale(self) -> bool:
        """结论是不是对着另一组设备参数跑的 —— 改过设备字段就算过期。"""
        if self._check_key is None:
            return False
        return self._device_key() != self._check_key

    def _run_check(self) -> None:
        """「设备自检」按钮: 显式触发一次只读硬件枚举, 交给后台线程。"""
        if self._checking or self._running or self._zero_reading:
            # 按钮已禁用, 这里是第二道闸 (程序方式触发 clicked 时也拦住)。
            return
        key = self._device_key()
        self._checking = True
        self._check_key = key
        self._check_heading = ("pending", "正在自检…（枚举相机 / 探测 SLM）")
        self._check_rows = []
        self._check_btn.setEnabled(False)  # 一次只跑一份; 也避免连点排两个任务
        self._zero_btn.setEnabled(False)  # 自检正占着 SLM, 此刻读零级只会抢设备
        self._render_check()
        self._revalidate()

        task = HardwareCheckTask(*key)
        # 与 ``_SessionTask`` 一样不交给 Qt 自动回收: 回收早于信号送达就丢掉结论。
        task.setAutoDelete(False)
        self._check_task = task
        task.signals.done.connect(self._on_check_done)
        self._pool.start(task)

    def _on_check_done(self, payload: object) -> None:
        """自检结论回到 UI 线程: 逐项上屏, 再把两个按钮的决定权交回校验。"""
        self._checking = False
        self._check_task = None
        if isinstance(payload, BaseException):
            # 自检脚本自己炸了 (SDK 缺失、进程被占用……) 也是一种必须说清的结论。
            self._check_heading = ("error", "自检未完成：检查本身出错")
            self._check_rows = [
                ("error", type(payload).__name__, str(payload) or "（无消息）")
            ]
        else:
            ok = bool(getattr(payload, "ok", False))
            summary = str(getattr(payload, "summary", lambda: "")())
            self._check_heading = (
                "ok" if ok else "error",
                summary or "硬件自检完成",
            )
            self._check_rows = [
                (
                    _CHECK_LEVELS.get(str(getattr(item, "level", "")), "pending"),
                    str(getattr(item, "label", "")),
                    str(getattr(item, "detail", "")),
                )
                for item in getattr(payload, "items", ())
            ]
        self._check_btn.setEnabled(not self._running)
        self._zero_btn.setEnabled(not self._running and not self._zero_reading)
        self._render_check()
        self._revalidate()

    def _render_check(self) -> None:
        """自检区: 结论 + 逐项明细, 颜色按当前主题取; 过期只是加一句提醒。"""
        if self._check_heading is None:
            self._check_view.clear()
            self._check_view.setVisible(False)
            return
        self._check_view.setVisible(True)
        kind, text = self._check_heading
        color = self._colors.get(kind, self._colors["pending"])
        parts = [
            f'<p style="margin:0 0 2px 0"><b style="color:{color}">'
            f"{escape(text)}</b></p>"
        ]
        for level, label, detail in self._check_rows:
            item_color = self._colors.get(level, self._colors["pending"])
            parts.append(
                f'<p style="margin:0"><span style="color:{item_color}">■ '
                f"{escape(label)}</span>{escape('：' + detail)}</p>"
            )
        if self._check_stale():
            parts.append(
                '<p style="margin:2px 0 0 0"><span style="color:'
                f'{self._colors["warn"]}">设备参数已改，请重新点「设备自检」</span></p>'
            )
        self._check_view.setText("".join(parts))

    # -------------------------------------------------------------- 读取零级
    def _read_zero_order(self) -> None:
        """「读取零级」按钮: 只读地开一次相机量 0 级, 交给后台线程。

        与自检 / 运行同一套互斥: 自检正开着 SLM、会话正占着相机时再开一次,
        只会跟它们对抢设备。
        """
        if self._zero_reading or self._checking or self._running:
            # 按钮已禁用, 这里是第二道闸 (程序方式触发 clicked 时也拦住)。
            return
        values = self.form_values()
        self._zero_reading = True
        self._zero_btn.setEnabled(False)  # 一次只跑一份; 也避免连点排两个任务
        self._set_run_state("running", "正在读取 0 级…（开关一次相机）")
        self._revalidate()

        task = _ZeroOrderTask(values.camera, int(values.cam_id))
        # 与 ``_SessionTask`` 一样不交给 Qt 自动回收: 回收早于信号送达就丢掉结论。
        task.setAutoDelete(False)
        self._zero_task = task
        task.signals.done.connect(self._on_zero_done)
        self._pool.start(task)

    def _on_zero_done(self, payload: object) -> None:
        """读数回到 UI 线程: 按实测上限放宽控件, 再把按钮的决定权交回校验。"""
        self._zero_reading = False
        self._zero_task = None
        if isinstance(payload, BaseException):
            # 全暗 / 0 级贴边都是操作性结论: 原因原样上屏, 上限保持原值。
            self._zero_state = (
                "error",
                f"读取零级失败：{payload}。"
                f"目标边长上限仍为 {self._zero_limit} px（保守值）。",
            )
            self._set_run_state("failed", f"读取零级失败：{payload}")
        else:
            max_side = int(getattr(payload, "max_side_px", 0))
            summary = str(getattr(payload, "summary", lambda: "")())
            if max_side > 0:
                self._zero_limit = max_side
                # 上限就是物理依据: 0 级到最近帧边的距离折成的边长。
                self._side.setMaximum(float(max_side))
                if self._side.value() > max_side:
                    self._side.setValue(float(max_side))
            center = tuple(getattr(payload, "center", (0, 0)))
            shape = tuple(getattr(payload, "frame_shape", (0, 0)))
            self._zero_state = (
                "ok",
                f"已读零级：目标边长上限 {max_side} px"
                f"（0 级 ({center[0]}, {center[1]})，帧 {shape[1]}×{shape[0]}）；"
                "采样窗半跨按边长自动派生。",
            )
            self._set_run_state("complete", summary or "读取零级完成")
        self._zero_btn.setEnabled(not self._running and not self._checking)
        self._render_zero()
        self._revalidate()

    def _render_zero(self) -> None:
        """② 目标的常驻说明: 上限从哪来、现在是多少 (跟随主题换色)。"""
        if self._zero_state is None:
            text = (
                f"未读取零级：目标边长上限 {self._zero_limit} px（保守值）。"
                "点「读取零级」按实际光轴放宽。"
            )
            color = self._muted()
        else:
            kind, text = self._zero_state
            color = self._colors.get(kind, self._muted())
        self._zero_view.setText(f'<span style="color:{color}">{escape(text)}</span>')

    # ------------------------------------------------------------------ 运行
    def _start_run(self) -> None:
        """启动会话: 目录在这一刻确定, 之后改表单不影响本次运行。"""
        if self._running or self._checking or self._zero_reading:
            # 自检正开着 SLM、读零级正开着相机, 此时开跑两边会抢设备
            # (按钮已禁用, 这是第二道闸)。
            return
        values = self.form_values()
        errors, _ = params.validate_form(values)
        if errors:
            # 按钮已禁用, 这里是第二道闸 (程序方式触发 clicked 时也拦住)。
            self._last_errors = errors
            self._render_checks()
            self._run_btn.setToolTip(errors[0])
            return

        out = params.plan_session_dir()
        self._session_dir = out
        self._table.setRowCount(0)
        self._bar.setRange(0, 1)
        self._bar.setValue(0)
        self._current.setText("等待第一个阶段…")
        self._counts.setText("")
        self._session_label.setText(f"会话目录: {out}")
        self._live_view.clear()

        self._running = True
        self._set_form_enabled(False)
        self._run_btn.setEnabled(False)
        self._run_btn.setToolTip("会话运行中")
        self._set_run_state("running", "运行中：本次会话的参数已锁定，改表单不影响它。")

        task = _SessionTask(values, out)
        # 不交给 Qt 自动回收: 回收早于信号送达就会丢掉最后一条结论。
        task.setAutoDelete(False)
        task.signals.stage.connect(self._on_stage)
        task.signals.frame.connect(self._live_view.apply_payload)
        task.signals.finished.connect(self._on_finished)
        task.signals.failed.connect(self._on_failed)
        self._task = task
        _LIVE_TASKS.add(task)
        self._poll.start()
        self._pool.start(task)

    def _on_stage(self, name: str, status: str) -> None:
        """阶段回调 (已排队回主线程): 让这一次刷新提前发生。"""
        logger.debug("迁移学习阶段 {} → {}", name, status)
        self._refresh_progress()

    def _on_finished(self, out: str) -> None:
        """会话跑完: 停轮询, 把最终产物再读一遍, 放回可编辑状态。"""
        self._settle("complete", f"会话完成：{out}")

    def _on_failed(self, message: str) -> None:
        """会话失败: 停轮询, 把 journal 里已写下的阶段留在表上。"""
        self._settle("failed", f"会话失败：{message}")

    def _settle(self, kind: str, message: str) -> None:
        """运行结束后的共同收尾 (完成 / 失败 / 停止都走这里复位)。"""
        self._poll.stop()
        self._running = False
        task, self._task = self._task, None
        if task is not None:
            _LIVE_TASKS.discard(task)
        self._refresh_progress()
        self._set_form_enabled(True)
        self._set_run_state(kind, message)
        self._revalidate()

    # ------------------------------------------------------------------ 进度
    def _open_replay(self) -> None:
        """打开归档回放弹窗: 扫描整个 debug 根下所有历史 pkl (只读, 无硬件)。"""
        from ao_shaping.gui.slm.transfer_desktop.archive_replay import (
            DEBUG_ROOT,
            open_archive_replay,
        )

        open_archive_replay(self, DEBUG_ROOT, preselect=self._session_dir)

    def _refresh_progress(self) -> None:
        """读会话目录, 刷新阶段表 / 进度条 / 当前阶段 / 汇总。"""
        view = collect_progress(self._session_dir)
        stages = view["stages"]
        self._sync_table(stages)

        counts = view["counts"]
        total = max(int(counts.get("total", 0)), 1)
        settled = sum(
            int(counts.get(key, 0)) for key in ("complete", "failed", "skipped")
        )
        self._bar.setRange(0, total)
        self._bar.setValue(min(settled, total))

        current = view["current"]
        if current is not None:
            status = STATUS_LABELS.get(str(current["status"]), str(current["status"]))
            self._current.setText(f"当前：{current['label']} · {status}")
        elif stages:
            self._current.setText("尚未开始")
        else:
            self._current.setText("尚无阶段计划")

        self._counts.setText(
            " · ".join(
                f"{STATUS_LABELS[key]} {counts.get(key, 0)}"
                for key in ("complete", "running", "pending", "skipped", "failed")
            )
        )

    def _sync_table(self, stages: Sequence[Mapping[str, Any]]) -> None:
        """按计划顺序重画阶段表 (十几个阶段, 重建比就地更新简单且无歧义)。"""
        table = self._table
        table.setRowCount(len(stages))
        for row, stage in enumerate(stages):
            name = str(stage["name"])
            status = str(stage["status"])
            cells = (
                str(stage["index"]),
                params.STAGE_LABELS.get(name, str(stage["label"])),
                STATUS_LABELS.get(status, status),
                _duration_text(stage["duration_s"]),
                _stage_summary(stage),
            )
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column != 4:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if column == 2:
                    item.setForeground(
                        QColor(self._colors.get(status, self._colors["pending"]))
                    )
                table.setItem(row, column, item)

    # ------------------------------------------------------------------ 外观
    def _muted(self) -> str:
        """次级文字色 (跟随调色板的占位文字色, 不写死灰值)。"""
        return self.palette().color(QPalette.ColorRole.PlaceholderText).name()

    def _set_run_state(self, kind: str, message: str) -> None:
        """运行状态行: 记下原文, 换主题时按新配色重画。"""
        self._status_kind = kind
        self._status_message = message
        color = self._colors.get(kind, self._colors["pending"])
        self._status.setText(f'<span style="color:{color}">{escape(message)}</span>')

    def _apply_theme(self) -> None:
        """按当前调色板重刷状态色 (亮 / 暗主题各一档)。"""
        self._colors = _palette_colors(self.palette())
        self._render_checks()
        self._render_check()
        self._render_zero()
        self._set_run_state(self._status_kind, self._status_message)

    def changeEvent(self, event: QEvent) -> None:
        """系统主题切换 → 重算状态色 (Qt 覆写点, 驼峰命名由 Qt 决定)。"""
        if event.type() == QEvent.Type.PaletteChange:
            self._apply_theme()
        super().changeEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        """关窗: 停轮询, 有限度地等在飞会话落地 —— ``run_session`` 没有取消接口。"""
        self._poll.stop()
        if self._running or self._zero_reading:
            # 完成信号排在 UI 事件队列里, 干等会拿不到; 必须泵事件驱动回调。
            # 读零级也要一起等: 相机还没关就退池, 下次打开会撞上占用。
            deadline = time.monotonic() + _CLOSE_GRACE_S
            while (self._running or self._zero_reading) and time.monotonic() < deadline:
                QApplication.processEvents()
                time.sleep(0.02)
        super().closeEvent(event)
