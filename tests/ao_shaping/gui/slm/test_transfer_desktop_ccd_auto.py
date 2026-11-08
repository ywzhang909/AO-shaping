"""迁移学习闭环页「加载时自动探测在线相机」的离屏契约。

要求来自现场: 桌面上常连一台相机, 操作员不该先猜该选 Daheng 还是 MiiCam。于是
页面加载时后台跑一次**只枚举、不开相机**的探测 (:class:`CcdDetectTask`), 把 ① 设备区
的相机类型 / cam_id 摆到在线的那台上, 并把一行中文结论写进中性提示行。

这里钉住四件事 (全部离线, 相机枚举一律 monkeypatch, 不碰真 SDK):

* :func:`select_ccd` 的三种选型 (优先后端有机器 / 换到另一后端 / 都空) 与两种退化
  (枚举抛异常、cam_id 越界);
* 枚举异常被吞进 ``detail``, 绝不从 ``select_ccd`` 里抛出来;
* :class:`CcdDetectTask` 的信号发出 ``CcdChoice`` 或异常实例;
* 页面接线: 结论写控件、失败只写提示、``auto_detect_ccd=False`` 一次都不枚举。
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

from ao_shaping.gui.slm.transfer_desktop import check_task
from ao_shaping.gui.slm.transfer_desktop import hardware_check as hc
from ao_shaping.gui.slm.transfer_desktop import page as page_mod


@pytest.fixture(scope="module")
def app() -> QApplication:
    """离屏 QApplication (一个进程只能有一个)。"""
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    return instance


def _patch_lists(
    monkeypatch: pytest.MonkeyPatch, *, daheng: list, miicam: list
) -> None:
    """把两个后端的枚举换成常量 (SDK 一次也不碰)。"""
    monkeypatch.setattr(
        hc.DahengCamera, "get_cam_list", staticmethod(lambda: list(daheng))
    )
    monkeypatch.setattr(
        hc.MIICamera, "get_cam_list", staticmethod(lambda: list(miicam))
    )


# --------------------------------------------------------------- select_ccd
def test_select_ccd_keeps_preferred_backend(monkeypatch) -> None:
    _patch_lists(monkeypatch, daheng=[{"index": 0}], miicam=[{"index": 0}])

    choice = hc.select_ccd("daheng", 0)

    assert (choice.kind, choice.cam_id, choice.source) == ("daheng", 0, "preferred")
    assert (choice.daheng_count, choice.miicam_count) == (1, 1)
    assert "Daheng 1 台" in choice.detail
    assert "已选 daheng cam_id=0" in choice.detail


def test_select_ccd_switches_when_preferred_is_empty(monkeypatch) -> None:
    _patch_lists(monkeypatch, daheng=[], miicam=[{"index": 0}])

    choice = hc.select_ccd("daheng", 0)

    assert (choice.kind, choice.cam_id, choice.source) == ("miicam", 0, "switched")
    assert (choice.daheng_count, choice.miicam_count) == (0, 1)
    assert "Daheng 0 台 / MiiCam 1 台" in choice.detail
    assert "已改用 miicam cam_id=0" in choice.detail


def test_select_ccd_keeps_defaults_when_nothing_is_online(monkeypatch) -> None:
    _patch_lists(monkeypatch, daheng=[], miicam=[])

    choice = hc.select_ccd("daheng", 3)

    assert (choice.kind, choice.cam_id, choice.source) == ("daheng", 3, "unavailable")
    assert "未发现在线相机" in choice.detail
    assert "保留 daheng cam_id=3" in choice.detail


def test_select_ccd_swallows_enumeration_failure(monkeypatch) -> None:
    def _boom() -> list:
        raise RuntimeError("gxipy 初始化失败")

    monkeypatch.setattr(hc.DahengCamera, "get_cam_list", staticmethod(_boom))
    monkeypatch.setattr(
        hc.MIICamera, "get_cam_list", staticmethod(lambda: [{"index": 0}])
    )

    choice = hc.select_ccd("daheng", 0)  # 不抛

    assert (choice.kind, choice.source) == ("miicam", "switched")
    assert choice.daheng_count == 0
    assert "Daheng 枚举失败" in choice.detail
    assert "RuntimeError" in choice.detail
    assert "\n" not in choice.detail


def test_select_ccd_falls_back_to_first_camera_on_bad_cam_id(monkeypatch) -> None:
    _patch_lists(monkeypatch, daheng=[{"index": 0}, {"index": 1}], miicam=[])

    choice = hc.select_ccd("daheng", 7)

    assert choice.kind == "daheng"
    assert choice.cam_id == 0
    assert "已选 daheng cam_id=0" in choice.detail


# ----------------------------------------------------------- CcdDetectTask
def test_ccd_detect_task_emits_choice(monkeypatch) -> None:
    expected = hc.CcdChoice(
        kind="miicam",
        cam_id=0,
        source="switched",
        daheng_count=0,
        miicam_count=1,
        detail="自动探测：Daheng 0 台 / MiiCam 1 台，已改用 miicam cam_id=0",
    )
    monkeypatch.setattr(hc, "select_ccd", lambda preferred, cam_id: expected)

    task = check_task.CcdDetectTask("daheng", 0)
    seen: list[object] = []
    task.signals.done.connect(seen.append)

    task.run()  # 同步调用: 不靠线程池, 也不需要事件循环

    assert seen == [expected]


def test_ccd_detect_task_emits_exception_instead_of_raising(monkeypatch) -> None:
    def _boom(preferred: str, cam_id: int) -> object:
        raise RuntimeError("SDK 不可用")

    monkeypatch.setattr(hc, "select_ccd", _boom)

    task = check_task.CcdDetectTask("daheng", 0)
    seen: list[object] = []
    task.signals.done.connect(seen.append)

    task.run()

    assert len(seen) == 1
    assert isinstance(seen[0], RuntimeError)


# ------------------------------------------------------------------ 页面接线
def _make_page(
    app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    *,
    auto_detect_ccd: bool = True,
):
    """离屏页面; 参数校验换成常量, 免得为一次探测先 import torch。"""
    monkeypatch.setattr(page_mod.params, "validate_form", lambda values: ([], []))
    widget = page_mod.TransferPage(auto_detect_ccd=auto_detect_ccd)
    widget.show()
    app.processEvents()
    return widget


def _wait_probe(app: QApplication, widget, timeout_s: float = 10.0) -> None:
    """等加载时那次后台探测落地 (Qt 队列信号必须靠事件循环送达)。"""
    deadline = time.monotonic() + timeout_s
    while widget._ccd_task is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    assert widget._ccd_task is None, "相机自动探测没有在超时内结束"


def test_page_applies_detected_camera_and_shows_detail(app, monkeypatch) -> None:
    choice = hc.CcdChoice(
        kind="miicam",
        cam_id=1,
        source="switched",
        daheng_count=0,
        miicam_count=1,
        detail="自动探测：Daheng 0 台 / MiiCam 1 台，已改用 miicam cam_id=1",
    )
    seen: list[tuple[str, int]] = []
    monkeypatch.setattr(
        hc,
        "select_ccd",
        lambda preferred, cam_id: seen.append((preferred, cam_id)) or choice,
    )

    widget = _make_page(app, monkeypatch)
    try:
        _wait_probe(app, widget)
        assert seen == [("daheng", 0)]  # 探测用的是页面默认值
        assert widget._camera.currentData() == "miicam"
        assert widget._cam_id.value() == 1
        summary = widget.ccd_probe_summary()
        assert summary == choice.detail
        assert "MiiCam 1 台" in summary
        assert "miicam cam_id=1" in summary
    finally:
        widget.close()
        app.processEvents()


def test_page_degrades_when_detection_fails(app, monkeypatch) -> None:
    def _boom(preferred: str, cam_id: int) -> object:
        raise RuntimeError("SDK 不可用")

    monkeypatch.setattr(hc, "select_ccd", _boom)

    widget = _make_page(app, monkeypatch)
    try:
        _wait_probe(app, widget)  # 不抛
        summary = widget.ccd_probe_summary()
        assert "自动探测失败" in summary
        assert "RuntimeError" in summary
        assert "保留 daheng cam_id=0" in summary
        assert widget._camera.currentData() == "daheng"
        assert widget._cam_id.value() == 0
    finally:
        widget.close()
        app.processEvents()


def test_page_does_not_overwrite_operator_choice(app, monkeypatch) -> None:
    release = threading.Event()
    choice = hc.CcdChoice(
        kind="miicam",
        cam_id=0,
        source="switched",
        daheng_count=0,
        miicam_count=1,
        detail="自动探测：已改用 miicam cam_id=0",
    )

    def _slow(preferred: str, cam_id: int) -> object:
        release.wait(10.0)
        return choice

    monkeypatch.setattr(hc, "select_ccd", _slow)

    widget = _make_page(app, monkeypatch)
    try:
        widget.set_camera("sim")  # 操作员自己先选了
        release.set()
        _wait_probe(app, widget)
        assert widget._camera.currentData() == "sim"
        assert widget.ccd_probe_summary() == choice.detail
    finally:
        release.set()
        widget.close()
        app.processEvents()


def test_page_without_auto_detect_never_enumerates(app, monkeypatch) -> None:
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        hc,
        "select_ccd",
        lambda preferred, cam_id: calls.append((preferred, cam_id)),
    )

    widget = _make_page(app, monkeypatch, auto_detect_ccd=False)
    try:
        assert widget._ccd_task is None
        assert widget.ccd_probe_summary() == ""
        assert widget._probe_label.text() == ""
        QTest.qWait(50)
        assert calls == []
    finally:
        widget.close()
        app.processEvents()