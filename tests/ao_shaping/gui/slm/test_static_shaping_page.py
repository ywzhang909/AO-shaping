"""SLM 静态整形页的离屏契约。

钉住这一页的关键保证:

* 页面能建出来、能渲染（空状态文案点名下一步，而不是空白面板）；
* 图案规格与 Streamlit ``helper_tab`` 共享同一份（同一对象）—— 两套前端不会
  各写一套图案而漂移；
* 生成的图案 shape/范围符合预期（纯生成 round-trip）；
* 下发路径走后台线程并带正确参数（mock 驱动，不真发设备）。
"""

from __future__ import annotations

import os

# 必须在任何 QApplication 之前设置: CI / 无显示环境走离屏后端。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from ao_shaping.gui.slm import helper_tab as tab
from ao_shaping.gui.slm.pattern_specs import (
    PARAM_SPECS,
    PATTERN_TYPES,
    generate_phase_radians,
)
from ao_shaping.gui.slm.static_shaping_page import (
    _PREVIEW_MAX_SIDE,
    StaticShapingPage,
    make_preview_pixmap,
)
from ao_shaping.utils.wavefront.pattern_helper import PatternHelper


@pytest.fixture(scope="module")
def app() -> QApplication:
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    return instance


@pytest.fixture()
def page(app: QApplication):
    widget = StaticShapingPage()
    widget.show()
    app.processEvents()
    yield widget
    widget.close()
    app.processEvents()


def test_page_builds_and_renders(app, page) -> None:
    """页面能建能渲染，且空状态可见。"""
    assert page.isVisible()
    # 空状态文案点名下一步：不是空白面板。
    assert "生成预览" in page._preview.text()


def test_spec_table_shared_with_streamlit_tab() -> None:
    """两种前端用同一份图案/参数规格对象，不是复制。"""
    assert tab.PATTERN_TYPES is PATTERN_TYPES
    assert tab.PARAM_SPECS is PARAM_SPECS


def test_generated_pattern_shape_and_range() -> None:
    """生成一个图案，shape 与位深范围符合约定。

    用缺省面板几何 (240, 160) 造 helper：shape 应为 (height, width)，灰度落在
    ``0..2**bits-1``。选棋盘格，离线可复现。
    """
    helper = PatternHelper((240, 160), bits=10)
    phase = generate_phase_radians(helper, "checkerboard", {"period": 40})
    assert phase.shape == (160, 240)
    assert phase.dtype == np.float64
    gray = helper.to_uint16(phase)
    assert gray.dtype == np.uint16
    assert gray.min() >= 0 and gray.max() <= 2**10 - 1


def test_make_preview_pixmap_yields_pixmap() -> None:
    """灰度转预览图：任意尺寸都能压到预览框显示。"""
    gray = np.arange(240 * 160, dtype=np.uint16).reshape(160, 240) % 1024
    pix = make_preview_pixmap(gray)
    assert not pix.isNull()
    # 长边被压到 _PREVIEW_MAX_SIDE 以内。
    assert max(pix.width(), pix.height()) <= _PREVIEW_MAX_SIDE


class _FakeHandle:
    """最小 SLM 句柄：记录下发链路、返回可预期的读回灰度（不碰真设备）。"""

    def __init__(
        self,
        *,
        readback_gray: np.ndarray | None = None,
        readback_source: str = "缓存槽",
        readback_fail: bool = False,
    ) -> None:
        self.slm_number = 1
        self.calls: list[tuple[str, np.ndarray]] = []
        self._readback_gray = readback_gray
        self._readback_source = readback_source
        self._readback_fail = readback_fail

    def create_phase_from_array(self, phase: np.ndarray) -> np.ndarray:
        self.calls.append(("create_phase_from_array", np.array(phase, copy=True)))
        return np.full(phase.shape, 511, dtype=np.uint16)

    def display_data(self, gray: np.ndarray) -> None:
        self.calls.append(("display_data", np.array(gray, copy=True)))

    def get_displayed_phase(self) -> tuple[np.ndarray, str]:
        if self._readback_fail:
            raise RuntimeError("读回失败")
        gray = (
            self._readback_gray
            if self._readback_gray is not None
            else np.full((2, 2), 7, dtype=np.uint16)
        )
        return gray, self._readback_source

    def close(self) -> None:
        self.calls.append(("close", np.zeros(1, dtype=np.uint8)))


def test_send_phase_wires_driver_without_double_shift() -> None:
    """下发走 ``create_phase_from_array(弧度)`` → ``display_data(灰度)``，不二次平移。

    传入的相位应以**同样的值**进入 ``create_phase_from_array``；若页面在下发前
    又 shift 一次，这里抓到的就不是原数组。
    """
    from ao_shaping.gui.slm.static_shaping_page import _send_phase

    helper = PatternHelper((240, 160), bits=10)
    handle = _FakeHandle()
    phase = np.linspace(0.0, 2 * np.pi, 12 * 16, dtype=np.float64).reshape(12, 16)

    out = _send_phase(handle, phase, "checkerboard", {}, helper)

    assert [c[0] for c in handle.calls] == ["create_phase_from_array", "display_data"]
    assert np.array_equal(handle.calls[0][1], phase)  # 未二次平移
    assert handle.calls[1][1].dtype == np.uint16  # 下发的是灰度
    assert out["source"] == "缓存槽"
    assert isinstance(out["gray"], np.ndarray)
    assert isinstance(out["phase"], np.ndarray)
    assert np.array_equal(out["phase"], phase)


def test_send_phase_generates_first_when_no_phase() -> None:
    """没有现成相位（没点过「生成预览」）时，下发任务自己先生成再发。"""
    from ao_shaping.gui.slm.static_shaping_page import _send_phase

    helper = PatternHelper((240, 160), bits=10)
    handle = _FakeHandle()

    out = _send_phase(handle, None, "checkerboard", {"period": 40}, helper)

    assert [c[0] for c in handle.calls] == ["create_phase_from_array", "display_data"]
    # 缺省面板 (240,160) → 相位 shape (高 160, 宽 240)。
    assert handle.calls[0][1].shape == (160, 240)
    assert out["phase"].shape == (160, 240)


def test_send_phase_falls_back_when_readback_fails() -> None:
    """读回失败：用刚写入的灰度顶上，并在来源里注明（best-effort）。"""
    from ao_shaping.gui.slm.static_shaping_page import _send_phase

    helper = PatternHelper((240, 160), bits=10)
    handle = _FakeHandle(readback_fail=True)
    phase = np.zeros((5, 7), dtype=np.float64)

    out = _send_phase(handle, phase, "checkerboard", {}, helper)

    # 读回失败 → 显示的是 create_phase_from_array 刚返回的 511 灰度。
    assert out["gray"].shape == (5, 7)
    assert int(out["gray"].max()) == 511
    assert "读回失败" in out["source"]


def test_generate_preview_sends_phase_and_gray_through_signals(app, page) -> None:
    """生成预览走后台任务，经信号把 phase + gray 送回主线程。

    直接构造 _PreviewTask 并连接信号，断言两端 payload 类型正确。
    """
    from ao_shaping.gui.slm.static_shaping_page import _PreviewTask

    helper = PatternHelper((240, 160), bits=10)
    received: dict[str, object] = {}

    def _on_ready(phase, gray, label):
        received["phase"] = phase
        received["gray"] = gray
        received["label"] = label

    task = _PreviewTask("checkerboard", {"period": 40}, helper)
    task.signals.preview_ready.connect(_on_ready)
    # 直接同步跑 run()：信号是直接连接，回调在同一线程触发。
    task.run()

    assert received["label"] == "棋盘格"
    assert received["phase"].shape == (160, 240)
    assert received["gray"].dtype == np.uint16


def test_stale_preview_result_is_discarded(app, page) -> None:
    """生成途中改参数：过期结果不上屏、不进 ``_last_phase``，按钮仍恢复。

    直接驱动 :meth:`_on_preview_ready` 的 token 守卫，不依赖后台线程时序：参数
    改动会经 :meth:`_clear_preview` 把代次 +1，旧代次的结果必须作废 —— 否则下发
    会把与当前表单不符的旧相位写进 SLM。
    """
    from ao_shaping.gui.slm.static_shaping_page import _PREVIEW_MAX_SIDE

    phase = np.full((8, 8), 1.0)
    gray = np.full((8, 8), 511, dtype=np.uint16)
    empty_text = "尚未生成预览 —— 请先选图案类型、调参数，再点「生成预览」。"

    # 模拟一次参数改动：代次 +1、清空预览、_last_phase 置 None。
    page._clear_preview()
    current = page._gen_token

    # 过期（参数改动前）的结果回来 → 必须被丢弃。
    page._pending_generates = 1
    page._on_preview_ready(phase, gray, "棋盘格", current - 1)
    assert page._last_phase is None
    assert page._preview.pixmap().isNull()
    assert page._preview.text() == empty_text
    assert page._preview_btn.isEnabled()

    # 当前代次的结果 → 正常上屏并缓存相位。
    page._pending_generates = 1
    page._on_preview_ready(phase, gray, "棋盘格", current)
    assert page._last_phase is not None
    assert page._preview.pixmap() is not None and not page._preview.pixmap().isNull()
    assert max(page._preview.pixmap().width(), page._preview.pixmap().height()) <= (
        _PREVIEW_MAX_SIDE
    )


def test_queued_generate_does_not_prematurely_enable_button(app, page) -> None:
    """排队中的新生成未结束前，按钮保持禁用，不被旧任务提前点亮。

    单线程池里任务会排队：旧任务先结束（结果仍有效、照常上屏），但只要有新任务
    还排在后面，按钮就得继续禁用 —— 否则用户会以为可以再点一次而叠请求。
    """
    phase = np.full((8, 8), 1.0)
    gray = np.full((8, 8), 511, dtype=np.uint16)
    page._clear_preview()
    token = page._gen_token

    page._pending_generates = 2
    page._preview_btn.setEnabled(False)
    page._on_preview_ready(phase, gray, "棋盘格", token)
    assert not page._preview_btn.isEnabled()  # 新任务还在排队, 不提前点亮
    assert page._last_phase is not None  # 结果仍有效, 照常上屏

    # 最后一个任务结束 → 才恢复按钮。
    page._on_preview_ready(phase, gray, "棋盘格", token)
    assert page._preview_btn.isEnabled()


def test_regeneration_clears_old_preview_for_status(app, page, monkeypatch) -> None:
    """重生成时先撤旧图：状态文字不被上一张预览图盖住。

    QLabel 上 pixmap 优先于 text，不先 ``clear`` 的话「正在生成预览……」会被旧图
    盖住，按钮反馈形同虚设。这里拦下线程池 ``start``，不真跑生成。
    """
    from PySide6.QtGui import QPixmap

    started: list[object] = []
    monkeypatch.setattr(page._pool, "start", lambda task: started.append(task))

    page._preview.setPixmap(QPixmap(12, 12))
    page._on_generate()

    assert page._preview.pixmap().isNull()
    assert page._preview.text() == "正在生成预览……"
    assert not page._preview_btn.isEnabled()
    assert len(started) == 1


# ---------------------------------------------------------------------------
# 设备生命周期（全部用假的驱动 / 句柄，绝不碰真设备）
# ---------------------------------------------------------------------------


def test_probe_devices_collects_only_available(monkeypatch) -> None:
    """枚举：只收有返回的编号，``None``（无设备）被跳过。"""
    from ao_shaping.drivers.slm.santec import probe as probe_mod
    from ao_shaping.gui.slm.static_shaping_page import _probe_devices

    def fake_probe(number: int):
        return {"slm_number": number, "serial": f"SN{number}"} if number == 2 else None

    monkeypatch.setattr(probe_mod, "probe_slm", fake_probe)
    out = _probe_devices((1, 2, 3))
    assert out["devices"] == [{"slm_number": 2, "serial": "SN2"}]


def test_connect_device_closes_old_handle_first(monkeypatch) -> None:
    """重连前先放掉旧句柄，再把新句柄交回（USB 句柄独占）。"""
    from ao_shaping.drivers.slm import santec as santec_mod
    from ao_shaping.gui.slm.static_shaping_page import _connect_device

    events: list[str] = []

    class FakeSantec:
        def __init__(self, slm_number: int = 1, video_mode: int = 0):
            self.slm_number = slm_number
            self.Panel_Res = (1920, 1080)
            self.wavelength = 1064
            self.serial_number = "SN1"

        def open(self) -> None:
            events.append("open")

        def close(self) -> None:
            events.append("new-close")

    class OldHandle:
        def close(self) -> None:
            events.append("old-close")

    monkeypatch.setattr(santec_mod, "Santec", FakeSantec)
    out = _connect_device(1, OldHandle(), 1064)

    assert events == ["old-close", "open"]  # 旧句柄先关，再开新句柄
    assert out["handle"].slm_number == 1
    assert out["wavelength"] == 1064
    assert "SN1" in out["message"]


def test_connect_failure_wrapped_as_operator_error(monkeypatch) -> None:
    """连接失败 → 面向操作员的中文结论（``_OpError``），不是裸驱动异常。"""
    from ao_shaping.drivers.slm import santec as santec_mod
    from ao_shaping.gui.slm.static_shaping_page import _OpError, _connect_device

    class BoomSantec:
        def __init__(self, *a, **k):
            pass

        def open(self) -> None:
            raise RuntimeError("no device")

    monkeypatch.setattr(santec_mod, "Santec", BoomSantec)
    with pytest.raises(_OpError, match="连接 SLM 3 失败"):
        _connect_device(3, None, 1064)


def test_disconnect_closes_once_and_swallows_failure() -> None:
    """断开只 ``close()`` 一次；close 抛错也只记日志，绝不把页面拖崩。"""
    from ao_shaping.gui.slm.static_shaping_page import _disconnect_device

    class Handle:
        def __init__(self, fail: bool = False) -> None:
            self.calls = 0
            self._fail = fail

        def close(self) -> None:
            self.calls += 1
            if self._fail:
                raise RuntimeError("close boom")

    ok = Handle()
    msg = _disconnect_device(ok, 1)["message"]
    assert ok.calls == 1
    assert "已断开" in msg

    bad = Handle(fail=True)
    msg2 = _disconnect_device(bad, 1)["message"]  # 不抛
    assert bad.calls == 1
    assert "未完全成功" in msg2


def test_apply_wavelength_sets_and_wraps_errors() -> None:
    """设波长：成功写入；驱动抛错时包成 ``_OpError``。"""
    from ao_shaping.gui.slm.static_shaping_page import _OpError, _apply_wavelength

    class Handle:
        value = None

        def set_wavelength(self, nm: int) -> None:
            self.value = nm

    good = Handle()
    _apply_wavelength(good, 1, 1064)
    assert good.value == 1064

    class Bad:
        def set_wavelength(self, nm: int) -> None:
            raise ValueError("out of range")

    with pytest.raises(_OpError, match="设置波长失败"):
        _apply_wavelength(Bad(), 1, 2000)


def test_selfcheck_reports_readback_source() -> None:
    """相位自检：读当前灰度 + 读回显示相位，来源写进状态文案。"""
    from ao_shaping.gui.slm.static_shaping_page import _selfcheck_phase

    class Handle:
        def get_current_grayscale(self) -> int:
            return 42

        def get_displayed_phase(self) -> tuple[np.ndarray, str]:
            return np.zeros((2, 2), dtype=np.uint16), "缓存槽 #3"

    out = _selfcheck_phase(Handle(), 1)
    assert "42" in out["message"]
    assert "缓存槽 #3" in out["message"]


# ---------------------------------------------------------------------------
# 页面状态（自建未 show 的页面 —— 避免 showEvent 触发后台探测的时序干扰）
# ---------------------------------------------------------------------------


def test_constructor_does_not_touch_device(monkeypatch) -> None:
    """构造页面绝不探测 / 打开设备（无 SDK 的 CI 也能建页）。"""
    import ao_shaping.gui.slm.static_shaping_page as page_mod

    calls: list[tuple[int, ...]] = []
    monkeypatch.setattr(
        page_mod, "_probe_devices", lambda nums: calls.append(nums) or {"devices": []}
    )

    page = page_mod.StaticShapingPage()
    try:
        assert calls == []  # 构造期间未探测
        assert page._slm is None
        assert page._probed_once is False
    finally:
        page.close()


def test_show_probes_only_once(app, monkeypatch) -> None:
    """首次 show 自动枚举一次；再次 show 不再探（避免反复打扰设备）。"""
    import ao_shaping.gui.slm.static_shaping_page as page_mod

    calls: list[tuple[int, ...]] = []
    monkeypatch.setattr(
        page_mod, "_probe_devices", lambda nums: calls.append(nums) or {"devices": []}
    )

    page = page_mod.StaticShapingPage()
    # 同步执行池任务：结果在本线程直接回主线程，免去线程时序。
    monkeypatch.setattr(page._pool, "start", lambda task: task.run())
    try:
        page.show()
        app.processEvents()
        assert len(calls) == 1
        page.hide()
        page.show()
        app.processEvents()
        assert len(calls) == 1  # 第二次 show 不重复探测
    finally:
        page.close()


def test_send_button_state_follows_connection(app) -> None:
    """下发按钮：未连接禁用，已连接且空闲可用，有操作在飞再禁用。"""
    page = StaticShapingPage()
    try:
        page._slm = None
        page._slm_number = None
        page._device_op = None
        page._sync_device_buttons()
        assert not page._send_btn.isEnabled()

        page._slm = _FakeHandle()
        page._slm_number = 1
        page._sync_device_buttons()
        assert page._send_btn.isEnabled()

        page._device_op = "send"
        page._sync_device_buttons()
        assert not page._send_btn.isEnabled()

        page._slm = None  # 别让 closeEvent 去关 fake 句柄
    finally:
        page.close()


def test_apply_sent_shows_readback_gray(app) -> None:
    """下发完成：预览区显示**读回**的灰度，并缓存相位。"""
    page = StaticShapingPage()
    try:
        gray = np.full((4, 6), 300, dtype=np.uint16)
        phase = np.full((4, 6), 1.0)
        page._slm = _FakeHandle()
        page._slm_number = 1
        page._apply_sent({"gray": gray, "source": "缓存槽", "phase": phase})

        assert page._preview_gray is not None
        assert np.array_equal(page._preview_gray, gray)
        assert not page._preview.pixmap().isNull()
        assert "缓存槽" in page._device_status.text()
        assert page._last_phase is not None
        assert np.array_equal(page._last_phase, phase)
        page._slm = None
    finally:
        page.close()
