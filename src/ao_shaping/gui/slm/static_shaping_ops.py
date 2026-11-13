"""SLM 静态整形页的**设备侧**（枚举 / 连接 / 断开 / 波长 / 自检 / 下发）。

为什么单独成模块
----------------
:mod:`ao_shaping.gui.slm.static_shaping_page` 是那一页的**界面**：控件树、布局、
预览上屏、参数表重建。而这一模块是它的**设备侧**：所有会碰到 SLM 硬件的调用、
所有后台线程任务（:class:`_PreviewTask` / :class:`_DeviceTask` /
:class:`_SendTask`），以及把驱动异常翻译成操作员能读的中文结论的
:class:`_OpError`。

拆分只搬代码、不改行为：

* 设备仍然**只 open 一次** —— 句柄常驻在页面侧 ``self._slm``，下发复用它，
  「断开」/ 重连前 / 页面 ``closeEvent`` 才 close；
* 下发仍然**先发后读**：``create_phase_from_array(弧度)`` → ``display_data(灰度)``，
  **不二次平移**（shift 只在 ``create_phase_from_array`` 里应用一次，与
  :func:`helper_tab._display_phase` 一致），发完再 ``get_displayed_phase()`` 把
  **实际发出去的灰度**读回来（读回是 best-effort：读不到就用刚写进去的灰度顶上，
  状态行注明来源）；
* 没有现成相位时仍然**先生成再发**（同一个后台线程里完成）；
* 驱动异常仍然收窄到 :func:`_device_errors` 那几类，不会被宽 ``except`` 连同真正
  的缺陷一起吞掉。

依赖方向
--------
本模块**不** import 页面模块 —— 页面 import 本模块，反向不成立，因此没有循环
导入。页面需要的名字（:class:`_OpError`、:func:`_send_phase`、
:func:`_probe_devices` 等）由页面模块显式再导出，旧的 import 路径保持不变。
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from loguru import logger
from PySide6.QtCore import QObject, QRunnable, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QWidget,
)

from ao_shaping.gui.slm.pattern_specs import PATTERN_TYPES, generate_phase_radians
from ao_shaping.utils.wavefront.pattern_helper import PatternHelper

__all__ = [
    "_DEFAULT_PANEL_RES",
    "_DEFAULT_WAVELENGTH",
    "_PROBE_SLM_NUMBERS",
    "_TONES",
    "_WAVELENGTH_MAX",
    "_WAVELENGTH_MIN",
    "_DeviceControlMixin",
    "_DeviceTask",
    "_OpError",
    "_PreviewTask",
    "_SendTask",
    "_WorkerSignals",
    "_apply_wavelength",
    "_close_handle",
    "_connect_device",
    "_device_errors",
    "_disconnect_device",
    "_probe_devices",
    "_selfcheck_phase",
    "_send_phase",
]

#: 未连接时的默认面板几何（与 Streamlit ``helper_tab`` 的回退一致）。桌面页没有
#: 设备连接后的 session_state，预览一律在缺省面板几何上生成，尺寸由此封顶。
_DEFAULT_PANEL_RES: tuple[int, int] = (1920, 1200)

#: 枚举设备时扫描的 SLM 编号。Santec 控制器最多支持 8 路，但本机只接了 1–4；
#: 扫更多路只会让「枚举」变慢（每个编号最多等 3 s 探测超时）。
_PROBE_SLM_NUMBERS: tuple[int, ...] = (1, 2, 3, 4)

#: 波长可填范围 (nm)。驱动侧硬校验是 450–1600，这里放宽一圈，让操作员能先把
#: 想要的波长填进去，再由驱动给出权威的报错文案。
_WAVELENGTH_MIN: int = 400
_WAVELENGTH_MAX: int = 1700
#: 默认工作波长 (nm) —— 本机主力 SLM 是 1064 nm 那台。
_DEFAULT_WAVELENGTH: int = 1064
#: 状态行的语气色：空闲=中性灰，在飞=琥珀，成功=绿，失败=朱红。
_TONES: dict[str, str] = {
    "idle": "#8a94a6",
    "busy": "#e8a33d",
    "ok": "#2fa84f",
    "bad": "#e4572e",
}


class _OpError(Exception):
    """带面向操作员的中文结论的操作失败（直接显示在状态行里）。"""


_DEVICE_ERRORS: tuple[type[BaseException], ...] | None = None


def _device_errors() -> tuple[type[BaseException], ...]:
    """设备路径可能抛出的异常类型（惰性 import 驱动，页面 import 不碰 SDK）。

    收窄到这几类而不是 ``Exception``：驱动抛 :class:`SantecError`（不是
    ``RuntimeError`` 的子类），设备层还会冒 ``OSError`` / ``RuntimeError``。
    ``ValueError`` / ``TypeError`` 覆盖「参数写错」这类编程错误 —— 它们要变成
    界面上的可读结论，而不是被一个宽 except 连同真正的缺陷一起吞掉。
    """
    global _DEVICE_ERRORS
    if _DEVICE_ERRORS is None:
        from ao_shaping.drivers.slm.santec import SantecError

        _DEVICE_ERRORS = (SantecError, OSError, RuntimeError, ValueError, TypeError)
    return _DEVICE_ERRORS


# ---------------------------------------------------------------------------
# 后台任务
# ---------------------------------------------------------------------------


class _WorkerSignals(QObject):
    """后台线程 → 主线程的单向通道（``QRunnable`` 不是 ``QObject``，发不出信号）。

    信号对象在**主线程**创建（任务对象也在主线程构造），因此后台线程 emit 的
    信号会排成队列事件，主线程 ``processEvents`` 后才拿到 —— 控件只在主线程被动。
    """

    #: 生成预览完成：``(phase: np.ndarray, gray: np.ndarray, label: str)``。
    preview_ready = Signal(object, object, str)
    #: 生成本身抛异常：``(message: str)``。
    preview_failed = Signal(str)
    #: 设备操作完成：``(op: str, payload: dict)``。
    device_done = Signal(str, object)
    #: 设备操作失败：``(op: str, message: str)``（message 已是可读结论）。
    device_failed = Signal(str, str)
    #: 下发完成：``(payload: dict)``（``phase`` / ``gray`` / ``source`` 三项）。
    sent = Signal(object)


class _PreviewTask(QRunnable):
    """在后台线程生成一张相位 + 转灰度，经信号送回主线程。

    同时产出「弧度相位」（用于下发）与「uint16 灰度」（用于预览上屏）—— 与
    Streamlit 的 helper_tab 一致：预览显示 gray，下发用 phase。两者不在界面上
    二次换算，避免双重 mod-2π 误差。
    """

    def __init__(
        self,
        pattern_type: str,
        params: dict[str, Any],
        helper: PatternHelper,
    ) -> None:
        super().__init__()
        self._pattern_type = pattern_type
        self._params = params
        self._helper = helper
        self.signals = _WorkerSignals()

    def run(self) -> None:
        """线程池入口（后台线程执行）。"""
        try:
            phase = generate_phase_radians(
                self._helper, self._pattern_type, self._params
            )
            if self._pattern_type == "turbulence":
                logger.info("湍流相位屏已按当前 r0/L0 重新初始化解偶生成。")
            gray = self._helper.to_uint16(phase)
            label = PATTERN_TYPES[self._pattern_type].label
            self.signals.preview_ready.emit(phase, gray, label)
        except Exception as e:  # 后台不吞细节：异常必须变成界面结论
            logger.exception("Pattern helper preview failed: {}", e)
            self.signals.preview_failed.emit(str(e))


class _DeviceTask(QRunnable):
    """通用后台设备任务：跑一个无参可调用对象，结果/错误经信号送回主线程。

    调用体自己把设备异常翻译成 :class:`_OpError`（带可读结论），本类只负责
    「跑 + 送回」：主线程因此永远不会阻塞在设备调用上。
    """

    def __init__(self, op: str, work: Callable[[], dict[str, Any]]) -> None:
        super().__init__()
        self._op = op
        self._work = work
        self.signals = _WorkerSignals()

    def run(self) -> None:
        """线程池入口（后台线程执行）。"""
        try:
            payload = self._work()
        except _OpError as e:
            self.signals.device_failed.emit(self._op, str(e))
            return
        except _device_errors() as e:
            logger.exception("设备操作 {} 失败: {}", self._op, e)
            self.signals.device_failed.emit(self._op, f"{self._op} 失败：{e}")
            return
        self.signals.device_done.emit(self._op, payload)


# ---------------------------------------------------------------------------
# 设备动作（后台线程里跑的那部分）
# ---------------------------------------------------------------------------


def _close_handle(handle: Any) -> bool:
    """关闭设备句柄。失败只记日志 —— 页面不能因为设备不配合就退不出来。"""
    try:
        handle.close()
    except _device_errors() as e:
        logger.warning("关闭 SLM 句柄失败（已忽略）: {}", e)
        return False
    return True


def _probe_devices(slm_numbers: tuple[int, ...]) -> dict[str, Any]:
    """枚举 SLM：逐个编号跑 :func:`probe_slm`。

    :func:`probe_slm` 自带 3 s 线程超时（Santec SDK 历史上会在快速开关后挂住），
    返回 ``None`` 表示该编号没有设备，返回带 ``in_use`` 的字典表示被占用。
    """
    from ao_shaping.drivers.slm.santec.probe import probe_slm

    devices: list[dict[str, Any]] = []
    for number in slm_numbers:
        device = probe_slm(number)
        if device:
            devices.append(dict(device))
    return {"devices": devices}


def _connect_device(
    slm_number: int, old_handle: Any, wavelength: int = _DEFAULT_WAVELENGTH
) -> dict[str, Any]:
    """连接一台 SLM：先放掉旧句柄，再 ``Santec(...).open()`` 并把句柄交回主线程。

    句柄对象本身交给主线程存进 ``self._slm``，之后所有下发都复用它 ——
    这是本页「只开一次设备」的关键。

    ``wavelength`` 是操作员为这台设备选定的**工作波长**（nm）：连接本身不改
    面板的波长设置（那是「应用波长」按钮的事），但它随连接结论一起回给页面，
    让「连接」与「波长」两处状态说的是同一件事。
    """
    from ao_shaping.drivers.slm.santec import Santec

    if old_handle is not None:
        # 重连前先把旧句柄放掉 —— USB 句柄独占，不关就开不了新的。
        _close_handle(old_handle)

    try:
        slm = Santec(slm_number=slm_number, video_mode=0)
        slm.open()
    except _device_errors() as e:
        raise _OpError(
            f"连接 SLM {slm_number} 失败：{e} —— 检查驱动板供电与 USB 连接后重试。"
        ) from e

    try:
        serial = getattr(slm, "serial_number", None)
    except _device_errors() as e:
        logger.warning("读取 SLM 序列号失败: {}", e)
        serial = None
    width, height = getattr(slm, "Panel_Res", _DEFAULT_PANEL_RES)
    message = (
        f"SLM {slm_number} · {serial if serial else '序列号未知'} · "
        f"{width}×{height} · 已连接"
    )
    return {"handle": slm, "message": message, "wavelength": wavelength}


def _disconnect_device(handle: Any, slm_number: int) -> dict[str, Any]:
    """断开 SLM：``close()`` 常驻句柄。close 失败只记日志，不向外抛。"""
    if _close_handle(handle):
        message = f"已断开 SLM {slm_number} —— 需要再下发请重新「连接」。"
    else:
        message = (
            f"断开 SLM {slm_number} 未完全成功（句柄已作废）—— "
            "重连前请确认设备已被释放。"
        )
    return {"message": message}


def _apply_wavelength(handle: Any, slm_number: int, wavelength: int) -> dict[str, Any]:
    """给 SLM 设置工作波长（驱动可能要几十秒，所以必须放后台）。"""
    try:
        handle.set_wavelength(wavelength)
    except _device_errors() as e:
        raise _OpError(f"设置波长失败：{e} —— 检查波长范围与设备连接后重试。") from e
    return {"message": f"SLM {slm_number} · 波长已设为 {wavelength} nm"}


def _selfcheck_phase(handle: Any, slm_number: int) -> dict[str, Any]:
    """「实验前相位自检」：读当前灰度 + 读回当前显示的相位。"""
    try:
        grayscale = handle.get_current_grayscale()
    except _device_errors() as e:
        raise _OpError(f"相位自检失败：{e} —— 检查设备连接后重试。") from e

    try:
        _phase, source = handle.get_displayed_phase()
    except _device_errors() as e:
        logger.warning("读回 SLM {} 显示相位失败: {}", slm_number, e)
        source = f"读回失败（{e}）"
    return {"message": f"SLM {slm_number} · 当前灰度 {grayscale} · 显示相位：{source}"}


def _send_phase(
    handle: Any,
    phase_radians: np.ndarray | None,
    pattern_type: str,
    params: dict[str, Any],
    helper: PatternHelper,
) -> dict[str, Any]:
    """把相位下发到句柄，并把**实际显示的灰度**读回来。

    与 Streamlit 的 :func:`helper_tab._display_phase` 同源：
    ``create_phase_from_array(弧度)`` → ``display_data(灰度)``，**不二次平移**
    （shift 只在 ``create_phase_from_array`` 里应用一次）。

    若调用方没有现成的相位（还没点过「生成预览」），这里**先在同一个后台线程
    里生成再发** —— 操作员不必先点一次「生成预览」。

    读回是 best-effort：读不到就用刚写进去的灰度顶上，``source`` 里注明来源。

    Returns:
        ``{"phase": 弧度相位, "gray": 实际显示的灰度, "source": 灰度来源说明}``。
    """
    phase = phase_radians
    if phase is None:
        phase = generate_phase_radians(helper, pattern_type, params)
    phase_gray = handle.create_phase_from_array(phase)
    handle.display_data(phase_gray)

    read_gray: np.ndarray | None = None
    source = "读回失败（显示的是刚写入的灰度）"
    try:
        read_gray, source = handle.get_displayed_phase()
    except _device_errors() as e:
        logger.warning("读回显示相位失败: {}", e)
    shown = read_gray if read_gray is not None else phase_gray
    return {
        "phase": np.asarray(phase),
        "gray": np.asarray(shown),
        "source": source,
    }


class _SendTask(QRunnable):
    """在后台线程把相位下发到**常驻句柄**，并把实际发出的灰度读回。

    真正的下发在 :func:`_send_phase`（与 Streamlit 同源的那条链路）里；本类只
    负责把它丢进线程并把结果/失败经信号送回主线程 —— 主线程因此永远不会阻塞在
    设备调用上。

    句柄在**点击那一刻**由主线程快照进来：任务排队期间操作员若点了「断开」，
    这个快照仍指向已关闭的句柄，设备调用会失败并给出可读结论，而不是把新连接
    的设备误当成目标。
    """

    def __init__(
        self,
        slm: Any,
        phase_radians: np.ndarray | None,
        pattern_type: str,
        params: dict[str, Any],
        helper: PatternHelper,
    ) -> None:
        super().__init__()
        self._slm = slm
        self._phase_radians = phase_radians
        self._pattern_type = pattern_type
        self._params = params
        self._helper = helper
        self.signals = _WorkerSignals()

    def run(self) -> None:
        """线程池入口（后台线程执行）。"""
        try:
            payload = _send_phase(
                self._slm,
                self._phase_radians,
                self._pattern_type,
                self._params,
                self._helper,
            )
        except _device_errors() as e:
            logger.exception("相位下发失败: {}", e)
            self.signals.device_failed.emit(
                "send", f"下发失败：{e} —— 检查设备连接与波长设置后重试。"
            )
            return
        self.signals.sent.emit(payload)


class _DeviceControlMixin:
    """设备控制区的控件树与动作处理（由页面类继承）。

    这里只有「按操作员的动作发后台任务 + 把结论写回状态行」这一层，真正的设备
    调用仍是模块级的 :func:`_probe_devices` / :func:`_connect_device` 等函数。宿主
    类必须提供页面侧成员（:meth:`_section`、``self._pool``、``self._slm`` 等）——
    由 :class:`~ao_shaping.gui.slm.static_shaping_page.StaticShapingPage` 提供；
    本模块**不**反向 import 页面，因此没有循环导入。
    """

    # ------------------------------------------------------------------ 设备控制

    def _build_device_section(self) -> QGroupBox:
        """建「③ 设备控制」分组（层级参考 Streamlit 侧栏的设备区）。

        信息层级：设备下拉（哪台）→ 连接/断开（状态切换）→ 波长 → 相位自检 →
        一行状态行（当前设备身份 + 下一步提示）。
        """
        box, layout = self._section("③ 设备控制")
        top_row = QHBoxLayout()
        self._probe_btn = QPushButton("枚举设备 / 刷新")
        self._probe_btn.setToolTip("扫描 SLM 1–4，列出可用与被占用的设备")
        self._probe_btn.clicked.connect(self._on_probe)
        top_row.addWidget(self._probe_btn)
        self._connect_btn = QPushButton("连接")
        self._connect_btn.clicked.connect(self._on_connect)
        top_row.addWidget(self._connect_btn)
        self._disconnect_btn = QPushButton("断开")
        self._disconnect_btn.clicked.connect(self._on_disconnect)
        top_row.addWidget(self._disconnect_btn)
        layout.addLayout(top_row)

        form = QFormLayout()
        self._device_combo = QComboBox()
        self._device_combo.currentIndexChanged.connect(self._on_device_selected)
        form.addRow("设备", self._device_combo)

        wavelength_row = QWidget()
        wavelength_layout = QHBoxLayout(wavelength_row)
        wavelength_layout.setContentsMargins(0, 0, 0, 0)
        self._wavelength_spin = QSpinBox()
        self._wavelength_spin.setRange(_WAVELENGTH_MIN, _WAVELENGTH_MAX)
        self._wavelength_spin.setValue(_DEFAULT_WAVELENGTH)
        self._wavelength_spin.setSuffix(" nm")
        wavelength_layout.addWidget(self._wavelength_spin)
        self._wavelength_btn = QPushButton("应用波长")
        self._wavelength_btn.clicked.connect(self._on_apply_wavelength)
        wavelength_layout.addWidget(self._wavelength_btn)
        form.addRow("波长", wavelength_row)

        self._selfcheck_btn = QPushButton("相位自检")
        self._selfcheck_btn.setToolTip(
            "读回当前灰度与显示中的相位 —— 实验前先确认面板在线"
        )
        self._selfcheck_btn.clicked.connect(self._on_selfcheck)
        form.addRow("实验自检", self._selfcheck_btn)
        layout.addLayout(form)

        self._device_status = QLabel(
            "尚未探测设备 —— 点「枚举设备 / 刷新」查看可用的 SLM。"
        )
        self._device_status.setWordWrap(True)
        layout.addWidget(self._device_status)
        self._set_status(self._device_status.text(), "idle")
        return box

    def _set_status(self, text: str, tone: str) -> None:
        """写状态行并按语气着色（空闲 / 在飞 / 成功 / 失败）。"""
        self._device_status.setText(text)
        self._device_status.setStyleSheet(f"color: {_TONES.get(tone, _TONES['idle'])};")

    def _sync_device_buttons(self) -> None:
        """按「是否连接 / 是否有设备操作在飞」统一刷新按钮可用性与提示。"""
        connected = self._slm is not None
        busy = self._device_op is not None
        has_device = self._device_combo.count() > 0
        self._probe_btn.setEnabled(not busy)
        self._connect_btn.setEnabled(not busy and not connected and has_device)
        self._disconnect_btn.setEnabled(not busy and connected)
        self._wavelength_spin.setEnabled(not busy and connected)
        self._wavelength_btn.setEnabled(not busy and connected)
        self._selfcheck_btn.setEnabled(not busy and connected)
        self._send_btn.setEnabled(not busy and connected)
        self._send_btn.setToolTip(
            "先连接设备再下发 —— 在「③ 设备控制」里选设备并点「连接」。"
            if not connected
            else "把当前相位下发到已连接的 SLM，并把读回的灰度显示到右侧。"
        )

    def _set_device_busy(self, op: str | None) -> None:
        """标记/清除后台设备操作，并同步按钮状态。"""
        self._device_op = op
        self._sync_device_buttons()

    def _start_device_task(self, op: str, work: Callable[[], dict[str, Any]]) -> None:
        """把一个设备动作丢进后台线程池，结果经信号回主线程。"""
        self._set_device_busy(op)
        task = _DeviceTask(op, work)
        task.signals.device_done.connect(self._on_device_done)
        task.signals.device_failed.connect(self._on_device_failed)
        self._pool.start(task)

    # ------------------------------------------------------------------ 设备动作

    @Slot()
    def _on_device_selected(self) -> None:
        """换了下拉里的设备：连接按钮随之可用/禁用。"""
        self._sync_device_buttons()

    @Slot()
    def _on_connect(self) -> None:
        """点「连接」：后台 open 设备，句柄常驻 ``self._slm``。"""
        data = self._device_combo.currentData()
        if not isinstance(data, dict):
            self._set_status(
                "没有选中的设备 —— 先点「枚举设备 / 刷新」再选一台 SLM。", "bad"
            )
            return
        slm_number = int(data.get("slm_number", 0) or 0)
        if slm_number <= 0:
            self._set_status(
                "选中的设备没有有效编号 —— 重新点「枚举设备 / 刷新」。", "bad"
            )
            return
        if data.get("in_use"):
            self._set_status(
                f"SLM {slm_number} 已被别的程序占用 —— 请先在占用它的窗口断开，再回来连接。",
                "bad",
            )
            return
        self._connect_btn.setText("正在连接…")
        self._set_status(f"正在连接 SLM {slm_number} ……", "busy")
        # 句柄在点击那一刻快照：排队期间的操作员动作不会改写这次连接的目标。
        old_handle = self._slm
        wavelength = self._wavelength_spin.value()
        self._start_device_task(
            "connect", lambda: _connect_device(slm_number, old_handle, wavelength)
        )

    @Slot()
    def _on_disconnect(self) -> None:
        """点「断开」：后台 close 常驻句柄，主线程清空句柄。"""
        if self._slm is None:
            self._set_status("当前没有已连接的设备 —— 无需断开。", "idle")
            return
        self._set_status(f"正在断开 SLM {self._slm_number} ……", "busy")
        handle = self._slm
        slm_number = self._slm_number or 0
        self._start_device_task(
            "disconnect", lambda: _disconnect_device(handle, slm_number)
        )

    @Slot()
    def _on_apply_wavelength(self) -> None:
        """点「应用波长」：后台写工作波长（驱动可能要几十秒）。"""
        if self._slm is None:
            self._set_status("没有已连接的设备 —— 先点「连接」再设波长。", "bad")
            return
        wavelength = self._wavelength_spin.value()
        self._set_status(f"正在设置波长 {wavelength} nm ……", "busy")
        handle = self._slm
        slm_number = self._slm_number or 0
        self._start_device_task(
            "wavelength",
            lambda: _apply_wavelength(handle, slm_number, wavelength),
        )

    @Slot()
    def _on_selfcheck(self) -> None:
        """点「相位自检」：后台读当前灰度 + 读回显示中的相位。"""
        if self._slm is None:
            self._set_status("没有已连接的设备 —— 先点「连接」再做相位自检。", "bad")
            return
        self._set_status("正在做相位自检 ……", "busy")
        handle = self._slm
        slm_number = self._slm_number or 0
        self._start_device_task(
            "selfcheck", lambda: _selfcheck_phase(handle, slm_number)
        )

    # ------------------------------------------------------------------ 结果分派

    @Slot(str, object)
    def _on_device_done(self, op: str, payload: object) -> None:
        """后台设备操作完成：分派到各操作的结果处理。"""
        self._set_device_busy(None)
        self._connect_btn.setText("连接")
        self._send_btn.setText("下发到 SLM")
        data = payload if isinstance(payload, dict) else {}
        if op == "probe":
            self._apply_devices(data.get("devices", []))
        elif op == "connect":
            handle = data.get("handle")
            self._slm = handle
            self._slm_number = int(getattr(handle, "slm_number", 0) or 0)
            self._set_status(str(data.get("message", "已连接 SLM")), "ok")
        elif op == "disconnect":
            self._slm = None
            self._slm_number = None
            self._set_status(str(data.get("message", "已断开 SLM")), "idle")
        elif op == "send":
            self._apply_sent(data)
        else:
            self._set_status(str(data.get("message", "操作完成")), "ok")
        self._sync_device_buttons()

    @Slot(str, str)
    def _on_device_failed(self, op: str, message: str) -> None:
        """后台设备操作失败：把可读结论写进状态行，并复位按钮。"""
        self._set_device_busy(None)
        self._connect_btn.setText("连接")
        self._send_btn.setText("下发到 SLM")
        if op == "connect":
            self._slm = None
            self._slm_number = None
        self._set_status(message, "bad")
        self._sync_device_buttons()

    def _apply_devices(self, devices: object) -> None:
        """枚举结果填下拉 + 一行状态（空结果点名下一步）。"""
        items = list(devices) if isinstance(devices, list) else []
        self._device_combo.clear()
        for device in items:
            if not isinstance(device, dict):
                continue
            number = device.get("slm_number")
            serial = device.get("serial")
            if device.get("in_use"):
                text = f"SLM {number} · 占用中"
            elif serial:
                text = f"SLM {number} · SN {serial}"
            else:
                text = f"SLM {number} · 序列号未知"
            self._device_combo.addItem(text, dict(device))
        if not self._device_combo.count():
            self._set_status(
                "未发现可用 SLM —— 检查驱动板/线缆后点「枚举设备 / 刷新」。", "bad"
            )
        else:
            self._set_status(
                f"发现 {self._device_combo.count()} 台 SLM —— 选一台后点「连接」。",
                "ok",
            )
        self._sync_device_buttons()
