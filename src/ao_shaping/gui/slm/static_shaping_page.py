"""SLM 静态整形页：把 Streamlit 的 PatternHelper 交互搬进 PySide6 桌面控制台。

为什么要有这一页
----------------
Streamlit 侧 (:mod:`ao_shaping.gui.slm.helper_tab`) 把 :class:`PatternHelper`
的相位生成直接暴露成「选图案类型 → 调参数 → 生成预览 → 下发到 SLM」。操作员在
同一台实验机上有两套前端：Streamlit 网页版负责带设备表格交互的复杂流程，而桌面
控制台 :class:`~ao_shaping.gui.ccd.beam_quality.shell.DesktopShellApp` 只放了光束
质量与迁移学习整形。用户要求把 PatternHelper 这一套「静态整形」交互也搬进桌面
控制台，好让不跑 Streamlit 时也能直接给一组 SLM 下发一张静态相位图。

交互映射（Streamlit → Qt，一一对应）
------------------------------------
+-------------------------------------+--------------------------------------------+
| Streamlit 步骤                       | Qt 控件 / 动作                              |
+=====================================+============================================+
| ``st.selectbox("图案类型", ...)``     | 左侧 :class:`QComboBox`（图案类型）          |
+-------------------------------------+--------------------------------------------+
| ``st.number_input / selectbox``      | 左侧 :class:`QFormLayout` 参数区（按规格组）  |
| （每个参数）                          |                                            |
+-------------------------------------+--------------------------------------------+
| 「生成预览」按钮 → ``to_uint16``       | 「生成预览」:class:`QPushButton` → 后台生成 + |
| ``st.image``                         | :class:`QLabel` 显示归一化的灰度图           |
+-------------------------------------+--------------------------------------------+
| 「下发到 SLM」按钮 →                  | 「下发到 SLM」按钮 → 后台线程                 |
| ``create_phase_from_array``          | open 设备 + ``create_phase_from_array``     |
| ``display_data``                     | + ``display_data``                          |
+-------------------------------------+--------------------------------------------+

周期 ①–③ 的单一真源
--------------------
图案类型注册表 :data:`PATTERN_TYPES` 与参数规格 :data:`PARAM_SPECS` 不在这里
重写 —— 桌面页与 Streamlit 共用 :mod:`~ao_shaping.gui.slm.pattern_specs`（该模块
不 import Streamlit）。前端只有「怎么渲染控件」不同，图案集合与参数语义完全一致，
两套前端不会各写一套而漂移。

线程边界
--------
``PatternHelper`` 在**全面板分辨率**下造坐标网格、湍流屏要跑相屏模拟、下发时
open/写 SLM 都是秒级阻塞 —— 这些一律丢给本页自持的单线程
:class:`QThreadPool`（同一台设备上并发只会互相踩）。后台线程只经 Qt 信号把
「生成结果 / 下发结论」送回主线程，绝不碰控件。预览图按 :meth:`PatternHelper.to_uint16`
归一化（与 Streamlit 的 ``st.image`` 同一条 math），所以两种前端看到的灰度和形状一致。

下发路径与 Streamlit 完全同源：相位（弧度）→ ``slm.create_phase_from_array``
（内部完成 弧度→灰度 + shift + 矫正 + LUT）→ ``slm.display_data``。**不二次施加
平移** —— shift 只在 ``create_phase_from_array`` 里应用一次，与 helper_tab 的
:func:`_display_phase` 一致。
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal, Slot
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
    ParamSpec,
    PATTERN_TYPES,
    generate_phase_radians,
    specs_for,
)
from ao_shaping.utils.io.file import ROOT_DIR
from ao_shaping.utils.wavefront.pattern_helper import PatternHelper

__all__ = ["StaticShapingPage", "make_preview_pixmap", "pattern_to_gray"]

#: 未连接时的默认面板几何（与 Streamlit ``helper_tab`` 的回退一致）。桌面页没有
#: 设备连接后的 session_state，预览一律在缺省面板几何上生成，尺寸由此封顶（避免
#: 全分辨率 1920×1200 的预览图在窗体里显示得太小 / 生成太慢时还有小图兜底）。
_DEFAULT_PANEL_RES: tuple[int, int] = (1920, 1200)
_DEFAULT_BITS: int = 10
#: 预览显示时最长边缩到这个像素量级，避免把整张 1920 宽位图塞进小预览框。
_PREVIEW_MAX_SIDE: int = 480


def pattern_to_gray(phase_radians: np.ndarray, bits: int = 10) -> np.ndarray:
    """把弧度相位归一化成 ``0..2**bits-1`` 的灰度（与 Streamlit 预览同一数学）。

    复用 :meth:`PatternHelper.to_uint16`（内部走
    ``phase_display.phase_to_slm_grayscale`` 的 ``mod 2π → 灰度``），保证桌面预览
    与 Streamlit ``st.image`` 看到的是同一张图。
    """
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
# 后台任务
# ---------------------------------------------------------------------------


class _WorkerSignals(QObject):
    """后台线程 → 主线程的单向通道（``QRunnable`` 不是 ``QObject``，发不出信号）。"""

    #: 生成预览完成：``(phase: np.ndarray, gray: np.ndarray, label: str)``。
    preview_ready = Signal(object, object, str)
    #: 生成本身抛异常：``(message: str)``。
    preview_failed = Signal(str)
    #: 下发完成 / 失败（含设备侧错误提示）。
    sent = Signal(str)
    send_failed = Signal(str)


class _PreviewTask(QRunnable):
    """在后台线程生成一张相位 + 转灰度，经信号送回主线程。

    同时产出「弧度相位」（用于下发）与「uint16 灰度」（用于预览上屏）—— 与
    Streamlit 的 helper_tab 一致：生成时分别得到 phase 与 ``to_uint16`` 的 gray，
    预览显示 gray，下发用 phase。两者不在界面上二次换算，避免双重 mod-2π 误差。
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


class _SendTask(QRunnable):
    """在后台线程把相位下发到指定编号的 SLM。"""

    def __init__(
        self,
        slm_number: int,
        phase_radians: np.ndarray,
        wavelength: int,
    ) -> None:
        super().__init__()
        self._slm_number = slm_number
        self._phase_radians = phase_radians
        self._wavelength = wavelength
        self.signals = _WorkerSignals()

    def run(self) -> None:
        """线程池入口（后台线程执行）。

        驱动 import 放在这里：桌面控制台冷启动不该被 SLM SDK 拖慢；只有真的要点
        「下发到 SLM」才拉起驱动。
        """
        from ao_shaping.drivers.slm.santec import Santec

        try:
            # 与 helper_tab 的下发路径一致；open / write 是阻塞设备操作。
            with Santec(
                slm_number=self._slm_number,
                wavelength=self._wavelength,
                video_mode=0,
            ) as slm:
                slm.set_phase_overlay(None)
                phase_gray = slm.create_phase_from_array(self._phase_radians)
                slm.display_data(phase_gray)
            self.signals.sent.emit(f"相位已下发到 SLM {self._slm_number} 并显示。")
        except Exception as e:  # 设备错误必须变成界面可读结论
            logger.exception(
                "Pattern helper display failed for SLM {}: {}", self._slm_number, e
            )
            self.signals.send_failed.emit(
                f"下发失败: {e} —— 请先确认 SLM {self._slm_number} 已连接。"
            )


# ---------------------------------------------------------------------------
# 页面
# ---------------------------------------------------------------------------


class StaticShapingPage(QWidget):
    """SLM 静态整形页：PatternHelper 面板 + 预览 + 下发。

    左侧是可滚动的表单列（图案类型 + 参数 + 两个动作按钮），右侧固定一块预览区。
    生成与下发都在后台线程池跑，界面不卡。
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        _icon = ROOT_DIR / "camera_rotate.ico"
        if _icon.exists():
            self.setWindowIcon(QIcon(str(_icon)))
        self._helper = PatternHelper(_DEFAULT_PANEL_RES, bits=_DEFAULT_BITS)
        self._last_phase: np.ndarray | None = None

        #: 生成预览代次：每次参数/图案类型变化 +1，作废在飞的旧生成结果（见
        #: :meth:`_clear_preview`）—— 否则生成途中改参数，旧相位仍会上屏并被「下发
        #: 到 SLM」误发，把与当前表单不符的图写进面板。
        self._gen_token = 0
        #: 在飞的「生成预览」任务数：只有最后一个任务负责恢复按钮，避免旧任务的结果
        #: 抢在更新任务前面把按钮重新点亮（单线程池里任务也可能排队）。
        self._pending_generates = 0

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

        # ③ SLM 编号
        slm_box, slm_layout = self._section("③ 目标 SLM")
        self._slm_form = QFormLayout()
        self._slm_spin = QSpinBox()
        self._slm_spin.setRange(1, 8)
        self._slm_spin.setValue(1)
        self._slm_form.addRow("SLM 编号", self._slm_spin)
        slm_layout.addLayout(self._slm_form)
        form_layout.addWidget(slm_box)

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

    def _section(self, title: str) -> tuple[QGroupBox, QVBoxLayout]:
        """表单分组框 + 竖排布局（与 transfer_desktop 同风格）。"""
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(8)
        return box, layout

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
            return "先按 r0/L0 初始化随机湍流屏，再生成一张相位屏 —— 每次点击重新解耦生成。"
        return f"{spec.label}，按下方参数生成。生成后点「生成预览」看效果，点「下发到 SLM」写进面板。"

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
        # float
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
        for name, widget in self._param_widgets.items():
            key = self._current_key
            specs = {s.name: s for s in specs_for(key)}
            spec = specs[name]
            value = self._param_values[name]
            if spec.kind == "seed" and value == -1:
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

        注意：这里**不**恢复「生成预览」按钮 —— 按钮只由最后一个结束的任务恢复
        (:meth:`_on_preview_ready` / :meth:`_on_preview_failed`)，否则排队中的新任务
        会被提前点亮。
        """
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
        """点「下发到 SLM」：需要先有一份已缓存的弧度相位。"""
        if self._last_phase is None:
            QMessageBox.information(
                self, "下发到 SLM", "请先点「生成预览」得到一张相位图，再下发。"
            )
            return
        slm_number = self._slm_spin.value()
        self._send_btn.setEnabled(False)
        self._send_btn.setText("下发中…")
        task = _SendTask(
            slm_number,
            self._last_phase,
            wavelength=1064,
        )
        task.signals.sent.connect(self._on_sent)
        task.signals.send_failed.connect(self._on_send_failed)
        self._pool.start(task)

    @Slot(str)
    def _on_sent(self, message: str) -> None:
        self._send_btn.setEnabled(True)
        self._send_btn.setText("下发到 SLM")
        QMessageBox.information(self, "下发到 SLM", message)

    @Slot(str)
    def _on_send_failed(self, message: str) -> None:
        self._send_btn.setEnabled(True)
        self._send_btn.setText("下发到 SLM")
        QMessageBox.critical(self, "下发到 SLM", message)

    def _clear_preview(self) -> None:
        """参数一变，旧预览就不可信：退回空状态并指回下一步。

        同时把生成代次 +1，作废仍在后台跑的旧「生成预览」任务 —— 它的结果即便
        回来也不会被上屏或缓存，避免把与当前表单不符的相位发给 SLM。
        """
        self._gen_token += 1
        self._last_phase = None
        self._preview.clear()
        self._preview.setText(
            "尚未生成预览 —— 请先选图案类型、调参数，再点「生成预览」。"
        )


if (
    __name__ == "__main__"
):  # 允许 python -m ao_shaping.gui.slm.static_shaping_page 独立预览
    from PySide6.QtWidgets import QApplication

    app = QApplication([])
    w = StaticShapingPage()
    w.resize(1080, 720)
    w.show()
    raise SystemExit(app.exec())
