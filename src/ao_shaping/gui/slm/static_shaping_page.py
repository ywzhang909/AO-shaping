"""SLM 静态整形页：把 Streamlit 的 PatternHelper 交互搬进 PySide6 桌面控制台。

Streamlit 侧 (:mod:`ao_shaping.gui.slm.helper_tab`) 把 :class:`PatternHelper`
的相位生成暴露成「选图案类型 → 调参数 → 生成预览 → 下发到 SLM」。本页把它搬进
:class:`~ao_shaping.gui.ccd.beam_quality.shell.DesktopShellApp`，并在 2026-11-12
补上 :mod:`ao_shaping.gui.slm.multi_slm_controller` 的整套设备控制（枚举 / 连接 /
断开 / 波长 / 相位自检），下发成功后**把读回的相位直接显示在预览区**。

本模块保留图案表单、预览和页面生命周期，作为原有 import 路径的 facade。
后台任务、设备操作与设备控制区放在 :mod:`ao_shaping.gui.slm.static_shaping_ops`；
该模块不反向导入页面，设备辅助函数仍从本模块显式再导出。

三条硬契约
----------
1. **单一真源**：图案类型注册表 :data:`PATTERN_TYPES` 与参数规格不重写 ——
   桌面页与 Streamlit 共用 :mod:`~ao_shaping.gui.slm.pattern_specs`，不会漂移。
2. **线程边界**：造网格 / 跑相屏模拟 / 设备 open / 写面板都是秒级阻塞，一律丢给
   本页自持的**单线程** :class:`QThreadPool`（同一台设备并发只会互相踩）。后台
   线程只经 Qt 信号把结论送回主线程 —— **主线程绝不阻塞在设备调用上**。
3. **句柄只 open 一次**：SLM 的 USB 句柄独占，背靠背 open/close 会让驱动板假死。
   连接后句柄常驻 ``self._slm``；「断开」/ 重连前 / ``closeEvent`` 才 close
   并清空；下发复用它，不再每次 open/close。close 失败只记日志。

下发路径与 Streamlit 完全同源：相位（弧度）→ ``slm.create_phase_from_array``
（内部完成 弧度→灰度 + shift + 矫正 + LUT）→ ``slm.display_data``。**不二次施加
平移** —— shift 只在 ``create_phase_from_array`` 里应用一次。发完再
``get_displayed_phase()`` 把**实际发出去的灰度**读回来上屏（读回是 best-effort：
读不到就用刚写进去的灰度顶上，状态行注明来源）。没有相位也能直接发：点「下发到
SLM」时若还没生成过相位，同一个后台任务里**先生成再发**。
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import Qt, QThreadPool, Slot
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ao_shaping.gui.slm.pattern_specs import (
    PATTERN_TYPES,
    ParamSpec,
    specs_for,
)
from ao_shaping.gui.slm.static_shaping_ops import (
    _DEFAULT_PANEL_RES,
    _PROBE_SLM_NUMBERS,
    _DeviceControlMixin,
    _DeviceTask,
    _OpError,
    _PreviewTask,
    _SendTask,
    _WorkerSignals,
    _apply_wavelength,
    _close_handle,
    _connect_device,
    _disconnect_device,
    _probe_devices,
    _selfcheck_phase,
    _send_phase,
)
from ao_shaping.utils.io.file import ROOT_DIR
from ao_shaping.utils.wavefront.pattern_helper import PatternHelper

#: 设备侧的名字在这里**显式再导出**：旧的 import 路径
#: ``from ao_shaping.gui.slm.static_shaping_page import _send_phase`` 等必须继续
#: 可用（页面与测试都按这个路径取）。列进 ``__all__`` 既声明了这份公共表面，
#: 也让 ruff 知道这些 import 不是没用（否则 F401 会把它们删掉）。
__all__ = [
    "StaticShapingPage",
    "make_preview_pixmap",
    "pattern_to_gray",
    "_DeviceTask",
    "_OpError",
    "_PreviewTask",
    "_SendTask",
    "_WorkerSignals",
    "_apply_wavelength",
    "_connect_device",
    "_disconnect_device",
    "_probe_devices",
    "_selfcheck_phase",
    "_send_phase",
]

#: 面板位深（灰度 0..1023）。
_DEFAULT_BITS: int = 10
#: 预览显示时最长边缩到这个像素量级，避免把整张 1920 宽位图塞进小预览框。
_PREVIEW_MAX_SIDE: int = 480
#: 页面级样式：让分组框、按钮与输入控件在同一套圆角/描边语言里说话。
_PAGE_QSS = """
QGroupBox { font-weight: 600; border: 1px solid palette(mid); border-radius: 8px;
            margin-top: 14px; padding-top: 10px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QPushButton { padding: 6px 14px; border-radius: 6px; }
QPushButton:disabled { color: palette(mid); }
QComboBox, QSpinBox, QDoubleSpinBox { padding: 3px 6px; border-radius: 5px; }
"""


# ---------------------------------------------------------------------------
# 预览工具
# ---------------------------------------------------------------------------


def pattern_to_gray(phase_radians: np.ndarray, bits: int = 10) -> np.ndarray:
    """把弧度相位归一化成 ``0..2**bits-1`` 的灰度（与 Streamlit 预览同一数学）。"""
    return PatternHelper(_DEFAULT_PANEL_RES, bits=bits).to_uint16(phase_radians)


def make_preview_pixmap(gray: np.ndarray) -> QPixmap:
    """把 uint16 灰度图缩放到预览框可显示尺寸，转成 :class:`QPixmap`。

    长边压到 :data:`_PREVIEW_MAX_SIDE`，两个方向都保持原长宽比，观感与 Streamlit
    的自适应显示一致。
    """
    gray = np.asarray(gray)
    if gray.size == 0:
        return QPixmap()
    height, width = gray.shape
    max_side = max(height, width)
    scale = (_PREVIEW_MAX_SIDE / max_side) if max_side > _PREVIEW_MAX_SIDE else 1.0
    tgt_w = max(1, int(round(width * scale)))
    tgt_h = max(1, int(round(height * scale)))
    # uint16 灰度 → QImage：每个像素 16bit，用 Format_Grayscale16 可以直接显示。
    data = gray.astype(np.uint16).copy()
    image = QImage(
        data.tobytes(),
        width,
        height,
        width * 2,
        QImage.Format.Format_Grayscale16,
    ).scaled(
        tgt_w,
        tgt_h,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    return QPixmap.fromImage(image)


# ---------------------------------------------------------------------------
# 页面
# ---------------------------------------------------------------------------


class StaticShapingPage(_DeviceControlMixin, QWidget):
    """SLM 静态整形页：PatternHelper 面板 + 设备控制 + 预览 + 下发。

    左侧是可滚动的表单列（图案类型 + 参数 + 设备控制 + 两个动作按钮），右侧固定
    一块预览区。生成与所有设备调用都在后台线程池跑，界面不卡。设备控制那一半的
    控件树与动作处理由 :class:`~ao_shaping.gui.slm.static_shaping_ops._DeviceControlMixin`
    提供，本类只管图案/预览/下发的界面。
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(_PAGE_QSS)
        _icon = ROOT_DIR / "camera_rotate.ico"
        if _icon.exists():
            self.setWindowIcon(QIcon(str(_icon)))
        self._helper = PatternHelper(_DEFAULT_PANEL_RES, bits=_DEFAULT_BITS)
        #: 上一张成功生成的弧度相位（供「下发到 SLM」复用；None = 还没生成过）。
        self._last_phase: np.ndarray | None = None
        #: 预览区当前显示的灰度（测试据此核对「显示的是哪一张图」）。
        self._preview_gray: np.ndarray | None = None
        #: 生成预览代次：每次参数/图案类型变化 +1，作废在飞的旧生成结果 ——
        #: 否则生成途中改参数，旧相位仍会上屏并被「下发到 SLM」误发。
        self._gen_token = 0
        #: 在飞的「生成预览」任务数：只有最后一个任务负责恢复按钮，避免旧任务的
        #: 结果抢在更新任务前面把按钮重新点亮（单线程池里任务也可能排队）。
        self._pending_generates = 0
        #: **常驻** SLM 句柄（``Santec`` 实例）。连接成功后填入，下发复用它 ——
        #: 每次下发都 open/close 会拖慢冷启动，还会让驱动板假死。
        self._slm: Any | None = None
        #: 当前连接的 SLM 编号（未连接时为 None）。
        self._slm_number: int | None = None
        #: 正在跑的后台设备操作名（None = 空闲）。用于禁用按钮 + 改文案。
        self._device_op: str | None = None
        #: 是否已经自动探测过一次（首次显示页时触发，之后不再自动探）。
        self._probed_once = False

        # 单线程池：同设备下发不能并发，与 transfer_desktop 同一纪律。
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---- 左列：可滚动表单 ----
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(360)
        form_host = QWidget()
        form_layout = QVBoxLayout(form_host)
        form_layout.setContentsMargins(14, 14, 14, 14)
        form_layout.setSpacing(12)

        # ① 图案类型
        type_box, type_layout = self._section("① 图案类型")
        self._pattern_combo = QComboBox()
        for key in PATTERN_TYPES:
            self._pattern_combo.addItem(PATTERN_TYPES[key].label, key)
        self._pattern_combo.currentIndexChanged.connect(self._on_pattern_changed)
        type_layout.addWidget(self._pattern_combo)
        self._type_desc = QLabel()
        self._type_desc.setWordWrap(True)
        type_layout.addWidget(self._type_desc)
        form_layout.addWidget(type_box)

        # ② 参数
        self._param_box, self._param_layout = self._section("② 参数")
        self._param_form = QFormLayout()
        self._param_layout.addLayout(self._param_form)
        form_layout.addWidget(self._param_box)

        # ③ 设备控制
        form_layout.addWidget(self._build_device_section())

        # 动作按钮
        buttons = QHBoxLayout()
        self._preview_btn = QPushButton("生成预览")
        self._preview_btn.clicked.connect(self._on_generate)
        self._send_btn = QPushButton("下发到 SLM")
        self._send_btn.clicked.connect(self._on_send)
        buttons.addWidget(self._preview_btn)
        buttons.addWidget(self._send_btn)
        form_layout.addLayout(buttons)

        form_layout.addStretch(1)
        scroll.setWidget(form_host)
        root.addWidget(scroll)

        # ---- 右列：预览 ----
        preview_host = QWidget()
        preview_layout = QVBoxLayout(preview_host)
        preview_layout.setContentsMargins(14, 14, 14, 14)
        self._preview = QLabel(
            "尚未生成预览 —— 请先选图案类型、调参数，再点「生成预览」。"
        )
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumSize(500, 420)
        preview_layout.addWidget(self._preview, 1)
        root.addWidget(preview_host, 1)

        self._on_pattern_changed()
        self._sync_device_buttons()

    def _section(self, title: str) -> tuple[QGroupBox, QVBoxLayout]:
        """表单分组框 + 竖排布局（与 transfer_desktop 同风格）。"""
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(8)
        return box, layout

    # ------------------------------------------------------------------ 设备动作

    @Slot()
    def _on_probe(self) -> None:
        """点「枚举设备 / 刷新」：后台扫描 SLM 1–4，结果回主线程填下拉。

        留在页面这一侧（而不是设备控制 mixin 里），因为它按**本模块的全局名**取
        :func:`_probe_devices` —— 测试与离线环境用
        ``static_shaping_page._probe_devices`` 替换真实探测，这个路径不能变。
        """
        self._set_status("正在枚举 SLM（1–4）……", "busy")
        self._start_device_task("probe", lambda: _probe_devices(_PROBE_SLM_NUMBERS))

    # ------------------------------------------------------------------ 参数

    @property
    def _current_key(self) -> str:
        data = self._pattern_combo.currentData()
        assert isinstance(data, str)
        return data

    def _on_pattern_changed(self) -> None:
        """切图案类型：重建参数控件并刷新一句话说明（空状态契约的入口）。"""
        key = self._current_key
        spec = PATTERN_TYPES[key]
        self._type_desc.setText(f"{spec.label} —— {self._describe(spec)}")
        # 清掉旧参数控件，按新规格组重建。
        while self._param_form.rowCount() > 0:
            self._param_form.removeRow(0)
        self._param_widgets: dict[str, QWidget] = {}
        self._param_values: dict[str, Any] = {}
        for p in specs_for(key):
            widget = self._build_param_widget(p)
            self._param_widgets[p.name] = widget
            self._param_values[p.name] = p.default
            self._param_form.addRow(p.label, widget)
        # 切类型后旧预览不再可信，指回操作顺序。
        self._clear_preview()

    def _describe(self, spec) -> str:
        """一句话说明当前图案类型（改用操作员可读的话，含湍流的两步语义）。"""
        if spec.requires_init:
            return "先按 r0/L0 初始化随机湍流屏，再生成一张相位屏 —— 每次点击重新解偶生成。"
        return (
            f"{spec.label}，按下方参数生成。"
            "可以直接点「下发到 SLM」（会先自动生成再发送），"
            "也可以先点「生成预览」离线看效果。"
        )

    def _build_param_widget(self, spec: ParamSpec) -> QWidget:
        """按参数规格建一个控件（int / float / select / seed 四种）。"""
        if spec.kind == "select":
            combo = QComboBox()
            for opt in spec.options:
                combo.addItem(opt, opt)
            if spec.default in spec.options:
                combo.setCurrentIndex(spec.options.index(spec.default))
            combo.currentIndexChanged.connect(
                lambda _i, name=spec.name: self._on_param_changed(name)
            )
            return combo
        # int / seed 共用整型步进；float 用浮点步进。seed 的 None 默认用 -1 占位。
        if spec.kind in ("int", "seed"):
            spin = QSpinBox()
            if spec.min is not None:
                spin.setMinimum(int(spec.min))
            if spec.max is not None:
                spin.setMaximum(int(spec.max))
            spin.setValue(int(spec.default) if spec.default is not None else -1)
            spin.valueChanged.connect(
                lambda value, name=spec.name: self._on_param_changed_int(name, value)
            )
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
        fspin.setDecimals(9)
        fspin.valueChanged.connect(
            lambda value, name=spec.name: self._on_param_changed_float(name, value)
        )
        return fspin

    def _on_param_changed(self, name: str) -> None:
        combo = self._param_widgets[name]
        assert isinstance(combo, QComboBox)
        self._param_values[name] = combo.currentData()
        self._clear_preview()

    def _on_param_changed_int(self, name: str, value: int) -> None:
        self._param_values[name] = value
        self._clear_preview()

    def _on_param_changed_float(self, name: str, value: float) -> None:
        self._param_values[name] = value
        self._clear_preview()

    # ------------------------------------------------------------------ 动作

    def _collect_params(self) -> dict[str, Any]:
        """从当前控件取值（seed 的 -1 占位还原成 None）。"""
        out: dict[str, Any] = {}
        specs = {s.name: s for s in specs_for(self._current_key)}
        for name in self._param_widgets:
            value = self._param_values[name]
            if specs[name].kind == "seed" and value == -1:
                value = None
            out[name] = value
        return out

    @Slot()
    def _on_generate(self) -> None:
        """点「生成预览」：后台生成，主线程上屏。"""
        self._preview_btn.setEnabled(False)
        # 先撤掉上一张图再写状态文字：QLabel 上 pixmap 优先于 text，不 clear 的话
        # 重生成时「正在生成预览……」会被旧图盖住，按钮反馈等于没有。
        self._preview.clear()
        self._preview.setText("正在生成预览……")
        token = self._gen_token
        self._pending_generates += 1
        task = _PreviewTask(self._current_key, self._collect_params(), self._helper)
        task.signals.preview_ready.connect(
            lambda phase, gray, label, t=token: self._on_preview_ready(
                phase, gray, label, t
            )
        )
        task.signals.preview_failed.connect(self._on_preview_failed)
        self._pool.start(task)

    @Slot(object, object, str, int)
    def _on_preview_ready(
        self, phase: np.ndarray, gray: np.ndarray, label: str, token: int
    ) -> None:
        """后台生成结果回主线程：作废过期结果，只在最后一个任务结束时报按钮。"""
        self._pending_generates -= 1
        if self._pending_generates <= 0:
            self._pending_generates = 0
            self._preview_btn.setEnabled(True)
        if token != self._gen_token:
            # 生成期间参数/图案类型已变：本次结果对应旧参数，作废不上屏（空状态
            # 已由 _clear_preview 设好）。_last_phase 保持 None，下发按钮无图可发。
            return
        self._apply_preview(phase, gray, label)

    @Slot(object, object, str)
    def _apply_preview(self, phase: np.ndarray, gray: np.ndarray, label: str) -> None:
        """把后台生成的灰度图上屏，并缓存弧度相位供「下发到 SLM」复用。

        这里**不**恢复「生成预览」按钮 —— 按钮只由最后一个结束的任务恢复，否则
        排队中的新任务会被提前点亮。
        """
        self._preview_gray = np.asarray(gray).copy()
        self._preview.setPixmap(make_preview_pixmap(gray))
        self._last_phase = np.asarray(phase, dtype=np.float64).copy()
        self._preview.setToolTip(
            f"图案「{label}」预览，尺寸 {gray.shape[1]}×{gray.shape[0]}"
        )

    @Slot(str)
    def _on_preview_failed(self, message: str) -> None:
        self._pending_generates -= 1
        if self._pending_generates <= 0:
            self._pending_generates = 0
            self._preview_btn.setEnabled(True)
        self._preview.setText("生成相位失败 —— 请检查参数后重试。")
        QMessageBox.warning(self, "生成预览", message)

    @Slot()
    def _on_send(self) -> None:
        """点「下发到 SLM」：走常驻句柄；没有相位就先生成再发。"""
        if self._slm is None:
            self._set_status(
                "没有已连接的设备 —— 先在「③ 设备控制」里选设备并点「连接」。", "bad"
            )
            return
        self._send_btn.setText("下发中…")
        self._set_device_busy("send")
        # 句柄、相位、图案参数在点击那一刻快照：排队期间的操作员动作不会改写这次
        # 下发的目标（换了图案或断了设备，发的仍是操作员点下去那一刻的内容）。
        task = _DeviceTask(
            "send",
            lambda: _send_phase(
                self._slm,
                self._last_phase,
                self._current_key,
                self._collect_params(),
                self._helper,
            ),
        )
        task.signals.device_done.connect(self._on_device_done)
        task.signals.device_failed.connect(self._on_device_failed)
        self._pool.start(task)

    @Slot(object)
    def _apply_sent(self, payload: object) -> None:
        """下发完成：把**实际发出去的灰度**直接显示到预览区。

        读回失败时这里显示的是刚写入面板的灰度 —— 状态行明确写出「读回失败」，
        不假装读回成功。
        """
        data = payload if isinstance(payload, dict) else {}
        gray = data.get("gray")
        phase = data.get("phase")
        source = str(data.get("source", ""))
        self._set_device_busy(None)
        self._send_btn.setText("下发到 SLM")
        if isinstance(gray, np.ndarray):
            self._preview_gray = np.asarray(gray).copy()
            self._preview.setPixmap(make_preview_pixmap(self._preview_gray))
            self._preview.setToolTip(
                f"已下发的相位（灰度 {gray.shape[1]}×{gray.shape[0]}），来源：{source}"
            )
        if isinstance(phase, np.ndarray):
            self._last_phase = np.asarray(phase, dtype=np.float64).copy()
        self._set_status(f"已下发 SLM {self._slm_number} · {source}", "ok")
        self._sync_device_buttons()

    def _clear_preview(self) -> None:
        """参数一变，旧预览就不可信：退回空状态并指回下一步。

        同时把生成代次 +1，作废仍在后台跑的旧「生成预览」任务 —— 它的结果即便
        回来也不会被上屏或缓存，避免把与当前表单不符的相位发给 SLM。
        """
        self._gen_token += 1
        self._last_phase = None
        self._preview_gray = None
        self._preview.clear()
        self._preview.setText(
            "尚未生成预览 —— 请先选图案类型、调参数，再点「生成预览」。"
        )

    # ------------------------------------------------------------------ 收尾

    def showEvent(self, event) -> None:
        """首次显示这一页时自动枚举一次设备（之后不再自动探）。

        设备只在**用户动作或首次显示页**时才碰 —— 构造函数里绝不碰设备，这样
        无 SDK 的 CI 也能建出页面。
        """
        super().showEvent(event)
        if not self._probed_once:
            self._probed_once = True
            self._on_probe()

    def closeEvent(self, event) -> None:
        """关页：把常驻的 SLM 句柄放掉，别让设备一直被桌面页占着。"""
        # 先等后台设备调用收尾（最多 5 s），再 close —— 否则「下发」还在写面板时
        # 就把设备关了。
        self._pool.clear()
        self._pool.waitForDone(5000)
        slm, self._slm = self._slm, None
        self._slm_number = None
        if slm is not None:
            _close_handle(slm)
        super().closeEvent(event)


if (
    __name__ == "__main__"
):  # 允许 python -m ao_shaping.gui.slm.static_shaping_page 独立预览
    from PySide6.QtWidgets import QApplication

    app = QApplication([])
    w = StaticShapingPage()
    w.resize(1080, 720)
    w.show()
    app.exec()
