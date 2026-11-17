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
    OpticalPathSnapshot,
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


def test_send_task_invokes_driver_with_expected_args(monkeypatch, page) -> None:
    """下发任务带正确参数调用驱动（mock 掉 Santec，不真发设备）。

    断言 open 后的链路：``create_phase_from_array(弧度)`` → ``display_data(灰度)``，
    且不下发时不会多调一次 shift —— 平移只在 create_phase_from_array 里发生一次。
    """
    from ao_shaping.drivers.slm import santec as santec_mod

    calls: list[str] = []

    class FakeSantec:
        def __init__(self, slm_number=1, wavelength=None, video_mode=0):
            self.slm_number = slm_number

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def set_phase_overlay(self, value):
            calls.append(f"set_phase_overlay({value})")

        def create_phase_from_array(self, phase):
            calls.append(f"create_phase_from_array(shape={phase.shape})")
            return np.full(phase.shape, 511, dtype=np.uint16)

        def display_data(self, gray):
            calls.append(f"display_data(shape={gray.shape})")

    monkeypatch.setattr(santec_mod, "Santec", FakeSantec)

    from ao_shaping.gui.slm.static_shaping_page import _SendTask

    phase = np.zeros((120, 192), dtype=np.float64)
    task = _SendTask(slm_number=2, phase_radians=phase, wavelength=1064)
    # 直接跑 run()（不走线程池），断言参数接线正确。
    task.run()

    assert calls == [
        "set_phase_overlay(None)",
        "create_phase_from_array(shape=(120, 192))",
        "display_data(shape=(120, 192))",
    ]


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


def test_send_task_records_optical_path_metadata(
    app, page, monkeypatch, tmp_path
) -> None:
    """_SendTask 下发成功后把光路认知快照与相位摘要落盘为 JSON。

    用临时目录替换 ROOT_DIR，断言 JSON 里 source/observed/note 与传入快照一致，
    且 phase_sha256 能对账下发相位。
    """
    import json as _json
    from pathlib import Path as _Path

    from ao_shaping.drivers.slm import santec as santec_mod
    from ao_shaping.gui.slm import static_shaping_page as sp

    # 不真开设备：mock Santec 的上下文与调用链。
    class FakeSantec:
        def __init__(self, slm_number=1, wavelength=None, video_mode=0):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def set_phase_overlay(self, value):
            pass

        def create_phase_from_array(self, phase):
            return np.full(phase.shape, 511, dtype=np.uint16)

        def display_data(self, gray):
            pass

    monkeypatch.setattr(santec_mod, "Santec", FakeSantec)

    # 把落盘目标重定向到 tmp_path，避免污染仓库 data/。
    monkeypatch.setattr(sp, "ROOT_DIR", _Path(tmp_path))

    captured: dict[str, object] = {}
    original = sp._record_send_metadata

    def _spy(slm_number, pattern_type, phase, optical_path):
        captured["slm_number"] = slm_number
        captured["pattern_type"] = pattern_type
        captured["phase"] = phase
        captured["optical_path"] = optical_path
        original(slm_number, pattern_type, phase, optical_path)

    monkeypatch.setattr(sp, "_record_send_metadata", _spy)

    phase = np.zeros((120, 192), dtype=np.float64)
    snapshot = OpticalPathSnapshot(
        source="model_forward",
        note="测试：未观测",
    )
    task = sp._SendTask(
        slm_number=2,
        phase_radians=phase,
        wavelength=1064,
        optical_path=snapshot,
        pattern_type="checkerboard",
    )
    task.run()

    # _spy 被调用且参数接线正确。
    assert captured["slm_number"] == 2
    assert captured["pattern_type"] == "checkerboard"
    assert captured["optical_path"] is snapshot

    # 落盘文件存在且内容与快照一致。
    out_dir = _Path(tmp_path) / "data" / "debug" / "static_shaping_send"
    files = list(out_dir.glob("*_slm2_checkerboard.json"))
    assert len(files) == 1, f"预期一份元数据 JSON，实得 {files}"
    record = _json.loads(files[0].read_text(encoding="utf-8"))
    assert record["slm_number"] == 2
    assert record["pattern_type"] == "checkerboard"
    assert record["optical_path"]["source"] == "model_forward"
    assert record["optical_path"]["observed"] is False
    assert record["optical_path"]["note"] == "测试：未观测"
    assert record["phase_shape"] == [120, 192]
    # sha256 能复现下发相位。
    import hashlib as _hashlib

    expected_sha = _hashlib.sha256(phase.tobytes()).hexdigest()
    assert record["phase_sha256"] == expected_sha


def test_send_task_defaults_to_model_forward_when_no_snapshot(app, page) -> None:
    """不传 optical_path 时，_SendTask 默认构造 model_forward 快照。"""
    from ao_shaping.gui.slm import static_shaping_page as sp

    phase = np.zeros((120, 192), dtype=np.float64)
    task = sp._SendTask(slm_number=1, phase_radians=phase, wavelength=1064)
    assert task._optical_path.source == "model_forward"
    assert task._optical_path.observed is False


def test_page_injected_snapshot_sets_combo_default(app) -> None:
    """注入 calibrated 快照时，光学观测来源 combo 默认选中已标定。"""
    snapshot = OpticalPathSnapshot(
        source="calibrated",
        calibrated_at="2026-11-17T00:00:00",
        pupil_center=(889.0, 590.0),
        observed=True,
        note="已实测",
    )
    widget = StaticShapingPage(optical_path=snapshot)
    try:
        assert widget._optical_path is snapshot
        index = widget._optical_combo.currentIndex()
        assert widget._optical_combo.itemData(index) == "calibrated"
    finally:
        widget.close()
        app.processEvents()


def test_refresh_optical_hint_disables_calibrated_when_unobserved(app, page) -> None:
    """未观测快照下，「已标定」项被禁用且提示声明未验证。"""
    # 默认 page 是 model_forward。
    page._optical_combo.setCurrentIndex(
        page._optical_combo.findData("model_forward")
    )
    page._refresh_optical_hint()
    calibrated_item = page._optical_combo.findData("calibrated")
    assert not page._optical_combo.model().item(calibrated_item).isEnabled()
    assert "未验证" in page._optical_hint.text()


def test_on_send_model_forward_asks_confirmation(app, page, monkeypatch) -> None:
    """model_forward 模式下点下发：弹确认框，拒绝则不启动任务。"""
    from PySide6.QtWidgets import QMessageBox

    from ao_shaping.gui.slm import static_shaping_page as sp

    page._last_phase = np.zeros((120, 192), dtype=np.float64)
    page._optical_combo.setCurrentIndex(
        page._optical_combo.findData("model_forward")
    )
    started: list[object] = []
    monkeypatch.setattr(page._pool, "start", lambda task: started.append(task))
    # 模拟用户点「否」。
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **kw: QMessageBox.StandardButton.No)
    )

    page._on_send()

    assert started == [], "拒绝后不应启动下发任务"
    assert page._send_btn.isEnabled(), "拒绝后按钮应恢复可用"


def test_on_send_calibrated_skips_confirmation(app, page, monkeypatch) -> None:
    """calibrated 模式下点下发：不弹确认框，直接启动任务。"""
    from PySide6.QtWidgets import QMessageBox

    page._last_phase = np.zeros((120, 192), dtype=np.float64)
    page._optical_combo.setCurrentIndex(
        page._optical_combo.findData("calibrated")
    )
    # 允许「已标定」项被选中：临时放开禁用。
    calibrated_item = page._optical_combo.findData("calibrated")
    page._optical_combo.model().item(calibrated_item).setEnabled(True)

    started: list[object] = []
    monkeypatch.setattr(page._pool, "start", lambda task: started.append(task))
    # 若弹了确认框，这个哨兵会被调用；断言不被调用。
    asked: list[bool] = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(lambda *a, **kw: asked.append(True) or QMessageBox.StandardButton.No),
    )

    page._on_send()

    assert asked == [], "已标定模式下不应弹确认框"
    assert len(started) == 1
