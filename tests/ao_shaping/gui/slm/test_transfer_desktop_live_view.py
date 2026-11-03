"""⑥ 实时画面的离屏契约: 后台帧信号 → 主线程上屏。

钉住线程边界与刷新语义:

* ``_SessionTask._on_frame`` 只 ``emit``, 不碰控件 (QRunnable 不是 QObject);
* ``frame`` 信号经事件队列排队到 ``LiveView.apply_payload``, 跨线程不直接改 UI;
* 一帧上屏会同时更新 CCD 图、剖面、块进度条与「距下次刷新还剩 N iter」;
* ⑤ 阶段进度面板仍然存在 (实时画面是新增列, 不是替换);
* 开跑前清空实时画面, 避免上一场会话的残帧。
"""

from __future__ import annotations

import os
import threading

# 必须在任何 QApplication 之前设置: CI / 无显示环境走离屏后端。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

from ao_shaping.gui.slm.transfer_desktop import page as page_mod
from ao_shaping.gui.slm.transfer_desktop.live_view import LiveView, display_frame


@pytest.fixture(scope="module")
def app() -> QApplication:
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    return instance


@pytest.fixture()
def page(app: QApplication, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(page_mod.params, "validate_form", lambda values: ([], []))
    widget = page_mod.TransferPage()
    widget.show()
    app.processEvents()
    yield widget
    widget.close()
    app.processEvents()


def _payload(**overrides) -> dict:
    yy, xx = np.mgrid[0:40, 0:48]
    frame = (2000 * np.exp(-(((xx - 24) ** 2) / 60.0 + ((yy - 20) ** 2) / 40.0))).astype(
        np.float64
    )
    payload = dict(
        frame=frame,
        xy=None,
        shape="triangle",
        block=1,
        block_index=2,
        blocks_total=4,
        iteration=12,
        iterations_total=12,
        remaining=12,
        label="验收采集",
    )
    payload.update(overrides)
    return payload


def test_display_frame_normalises_arbitrary_raw_frames():
    """float 计数帧不能 clip 成全黑; uint16 / 3 通道也都要能显示。"""
    float_frame = np.linspace(0, 4321.0, 64, dtype=np.float64).reshape(8, 8)
    shown = display_frame(float_frame)
    assert shown.dtype == np.uint8
    assert shown.max() == 255
    assert display_frame(np.zeros((0, 0))).shape == (1, 1)
    assert display_frame(np.full((4, 4), 7.0)).max() == 0
    rgb = np.zeros((4, 4, 3), dtype=np.uint16)
    rgb[..., 0] = 500
    assert display_frame(rgb).ndim == 2
    nan_frame = np.array([[np.nan, np.inf], [-np.inf, 10.0]])
    assert display_frame(nan_frame).max() == 255


def test_apply_payload_updates_frame_profiles_and_both_bars(app: QApplication):
    view = LiveView()
    view.show()
    app.processEvents()
    view.apply_payload(_payload())
    app.processEvents()

    assert view._video._array is not None
    # X/Y 两条剖面都被填了数据点 (ProfileWidget 不自存 metrics)。
    assert view._profile._x_panel._data.count() > 0
    assert view._profile._y_panel._data.count() > 0
    assert (view._block_bar.value(), view._block_bar.maximum()) == (2, 4)
    assert "2 / 4" in view._block_label.text()
    assert "triangle" in view._block_label.text()
    assert (view._remaining_bar.value(), view._remaining_bar.maximum()) == (12, 12)
    assert "距下次刷新还剩 12 iter" in view._info.text()
    view.close()


def test_bad_or_empty_payloads_do_not_touch_the_view(app: QApplication):
    view = LiveView()
    for bad in (None, "not-a-dict", {}, {"frame": None}):
        view.apply_payload(bad)  # type: ignore[arg-type]
    assert view._last_payload is None
    assert view._video._array is None
    view.close()


def test_last_block_reports_zero_remaining(app: QApplication):
    view = LiveView()
    view.apply_payload(_payload(block_index=4, remaining=0))
    assert view._remaining_bar.value() == 0
    assert "距下次刷新还剩 0 iter" in view._info.text()
    view.close()


def test_page_exposes_live_view_and_keeps_the_progress_panel(page):
    assert isinstance(page._live_view, LiveView)
    # ⑤ 进度面板与它的阶段表都还在, 实时画面是右栏新增的一块。
    assert page._table.columnCount() == 5
    assert page._bar.format().startswith("已结束")


def test_session_task_on_frame_only_emits(page):
    """后台回调经信号入队, 主线程泵完事件才上屏 —— 不是直接改控件。

    必须从**真的后台线程** emit: 同线程 emit 会被 Qt 优化成 DirectConnection,
    那样测到的「没上屏」只是错觉, 而不是队列在起作用。
    """
    task = page_mod._SessionTask(page.form_values(), page_mod.params.plan_session_dir())
    task.signals.frame.connect(page._live_view.apply_payload)
    assert page._live_view._last_payload is None

    worker = threading.Thread(target=task._on_frame, args=(_payload(),))
    worker.start()
    worker.join(timeout=10)
    assert not worker.is_alive()
    # 后台线程已经 emit 完毕, 但主线程还没泵事件 → 控件必须仍然是空态。
    assert page._live_view._last_payload is None
    for _ in range(10):
        QCoreApplication.processEvents()
    assert page._live_view._last_payload is not None
    assert page._live_view._block_bar.value() == 2


def test_start_run_clears_the_previous_frame(page, monkeypatch: pytest.MonkeyPatch):
    page._live_view.apply_payload(_payload())
    assert page._live_view._last_payload is not None
    # 不真正开会话: 钉住的只是「开跑前清屏」这一步。
    monkeypatch.setattr(page._pool, "start", lambda task: None)
    monkeypatch.setattr(page._poll, "start", lambda *args: None)
    page._start_run()
    assert page._live_view._last_payload is None
    assert page._live_view._video._array is None
    assert page._live_view._block_bar.value() == 0
    page._poll.stop()
