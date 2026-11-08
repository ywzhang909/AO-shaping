"""迁移学习闭环页「读取零级」按钮的离屏交互契约。

要求来自操作层: 「目标边长」的上限不该是写死的 400 px —— 唯一物理依据是 0 级
到最近帧边的距离。但读数是**只读诊断**, 必须满足三条:

* 只在点按钮时发生, 且相机 SDK 调用**绝不**落在 UI 线程 (开相机可能卡住);
* 成功后按实测上限放宽控件, 并让采样窗半跨随边长派生 (于是大目标能通过校验);
* 失败 (全暗 / 0 级贴边) 原样显示中文原因, 上限保持保守值, 按钮复位;
* 运行中 / 自检中点击不产生任何读取 —— 设备此刻归别人用。
"""

from __future__ import annotations

import os
import threading
import time

# 必须在任何 QApplication 之前设置: CI / 无显示环境走离屏后端。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from ao_shaping.gui.slm.transfer_desktop import page as page_mod
from ao_shaping.gui.slm.transfer_desktop import zero_order as zo

_READING = zo.ZeroOrderReading(
    center=(1075, 939),
    max_side_px=1858,
    exposure_ms=0.5,
    peak=180.0,
    frame_shape=(2048, 2048),
)


@pytest.fixture(scope="module")
def app() -> QApplication:
    """离屏 QApplication (一个进程只能有一个)。"""
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    return instance


@pytest.fixture()
def page(app: QApplication, monkeypatch: pytest.MonkeyPatch):
    """离屏页面; 校验换成常量, 免得每次实例化都 import torch 跑一遍 preflight。"""
    monkeypatch.setattr(page_mod.params, "validate_form", lambda values: ([], []))
    widget = page_mod.TransferPage(auto_detect_ccd=False)
    widget.show()
    app.processEvents()
    yield widget
    widget.close()
    app.processEvents()


def _wait_zero_done(app: QApplication, widget, timeout_s: float = 10.0) -> None:
    """等后台读数落地 (Qt 队列信号必须靠事件循环送达)。"""
    deadline = time.monotonic() + timeout_s
    while not widget.zero_order_button().isEnabled() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    assert widget.zero_order_button().isEnabled(), "读取零级没有在超时内结束"


def test_the_summary_is_one_chinese_line_with_the_measured_numbers() -> None:
    text = _READING.summary()

    assert "0 级 (1075, 939)" in text
    assert "0.50 ms" in text
    assert "1858 px" in text
    assert "2048×2048" in text


def test_clicking_reads_in_a_worker_thread_and_widens_the_limit(
    page, app, monkeypatch
) -> None:
    """读数在后台线程里做 (UI 线程一次硬件调用都不许有), 成功后放宽上限。"""
    seen: list[tuple[str, int, threading.Thread]] = []

    def _fake(camera: str, cam_id: int) -> zo.ZeroOrderReading:
        seen.append((camera, cam_id, threading.current_thread()))
        return _READING

    monkeypatch.setattr(zo, "read_zero_order", _fake)
    assert page._side.maximum() == 400.0

    page.zero_order_button().click()
    _wait_zero_done(app, page)

    assert len(seen) == 1
    camera, cam_id, thread = seen[0]
    assert (camera, cam_id) == ("daheng", 0)
    # 关键: 相机是在工作线程里开的, 不在 UI 线程同步调用硬件。
    assert thread is not threading.main_thread()

    assert page._side.maximum() == 1858.0
    # 矩形长边同样按实测上限放宽 —— 此前只更新 _side, _side_long 仍锁保守值,
    # 矩形长边敲不进探测允许的值 (用户反馈「被限制在 350/400」)。
    assert page._side_long.maximum() == 1858.0
    assert "1858" in page._status.text()
    assert "1858" in page._zero_view.text()


def test_a_current_value_above_the_limit_is_pulled_down(page, app, monkeypatch) -> None:
    """读到比现有目标更小的上限时, 当前值也必须降下来 (否则校验永远不过)。"""
    small = zo.ZeroOrderReading(
        center=(600, 600), max_side_px=300, exposure_ms=0.5, peak=150.0,
        frame_shape=(2048, 2048),
    )
    monkeypatch.setattr(zo, "read_zero_order", lambda *args: small)
    page._side.setValue(400.0)
    page._side_long.setValue(400.0)

    page.zero_order_button().click()
    _wait_zero_done(app, page)

    assert page._side.maximum() == 300.0
    assert page._side.value() == 300.0
    # 长边超出新上限时同样被夹回 (矩形两维都不能超过帧可容纳极限)。
    assert page._side_long.maximum() == 300.0
    assert page._side_long.value() == 300.0


def test_a_failed_read_keeps_the_conservative_limit_and_says_why(
    page, app, monkeypatch
) -> None:
    def _boom(camera: str, cam_id: int) -> zo.ZeroOrderReading:
        raise ValueError("相机全暗：0 级没有进画面")

    monkeypatch.setattr(zo, "read_zero_order", _boom)

    page.zero_order_button().click()
    _wait_zero_done(app, page)

    assert page._side.maximum() == 400.0
    assert "400" in page._zero_view.text()
    assert "相机全暗" in page._zero_view.text()
    assert "相机全暗" in page._status.text()
    assert page.zero_order_button().isEnabled()


def test_no_read_while_a_run_or_a_self_check_is_in_flight(
    page, app, monkeypatch
) -> None:
    calls: list[tuple] = []
    monkeypatch.setattr(
        zo, "read_zero_order", lambda *args: calls.append(args) or _READING
    )

    page._running = True
    page.zero_order_button().click()
    page._read_zero_order()  # 第二道闸: 程序方式触发也要拦住
    page._running = False

    page._checking = True
    page.zero_order_button().click()
    page._read_zero_order()
    page._checking = False

    QTest.qWait(60)
    app.processEvents()

    assert calls == []
    assert page._side.maximum() == 400.0
