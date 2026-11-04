"""迁移学习闭环页「设备自检」按钮的离屏交互契约。

要求来自操作层: 自检要**一键**, 但**绝不**跟着控件改动跑 —— 表单每变一次就枚举
一遍 SDK, 既会把窗口卡在硬件调用上, 也会让「通过」变成噪音。这里钉住:

* 只改表单 (含相机 / cam_id / SLM 编号) 不产生任何自检调用;
* 点按钮 → 后台跑完 → 结论逐项上屏、按钮复位;
* 结论失败时结论行写明未通过;
* 设备参数改过 → 旧结论只标「请重新自检」, 不自动重跑;
* 自检与运行互斥; 参数校验没过也仍然可以自检 (那正是最需要看硬件的时候)。
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

from ao_shaping.gui.slm.transfer_desktop import hardware_check as hc
from ao_shaping.gui.slm.transfer_desktop import page as page_mod
from ao_shaping.gui.slm.transfer_desktop.hardware_check import (
    CheckItem,
    HardwareReport,
)

_ALL_OK = HardwareReport(
    items=(CheckItem("Daheng 枚举", "ok", "枚举到 1 台大恒相机"),)
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
    widget = page_mod.TransferPage()
    widget.show()
    app.processEvents()
    yield widget
    widget.close()
    app.processEvents()


def _wait_check_done(app: QApplication, widget, timeout_s: float = 10.0) -> None:
    """等后台自检落地 (Qt 队列信号必须靠事件循环送达)。"""
    deadline = time.monotonic() + timeout_s
    while not widget.check_button().isEnabled() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    assert widget.check_button().isEnabled(), "自检没有在超时内结束"


def test_form_edits_never_start_a_self_check(page, monkeypatch) -> None:
    calls: list[tuple] = []
    monkeypatch.setattr(
        hc, "check_hardware", lambda *args: calls.append(args) or _ALL_OK
    )

    page.set_period(96.0)
    page.set_shapes(("square", "ring"))
    page.set_camera("daheng")
    # 设备字段是自检的输入: 改它们同样不能触发自检。
    page._slm_number.setValue(2)
    page._cam_id.setValue(1)
    QTest.qWait(60)

    assert calls == []
    assert page.check_report() == ""


def test_repeating_the_same_check_is_reused_not_rerun(page, app, monkeypatch) -> None:
    """同一设备再次点自检: 直接复用已有结论, 不重跑硬件, 也不阻塞。"""
    calls: list[tuple[str, int, int]] = []

    def _fake(camera: str, cam_id: int, slm_number: int) -> HardwareReport:
        calls.append((camera, cam_id, slm_number))
        return _ALL_OK

    monkeypatch.setattr(hc, "check_hardware", _fake)

    page.check_button().click()
    _wait_check_done(app, page)
    assert len(calls) == 1

    # 第二次点同一设备: 复用, 不新增硬件调用, 按钮保持可用。
    page.check_button().click()
    app.processEvents()
    assert page.check_button().isEnabled()
    assert len(calls) == 1
    assert "复用上次自检" in page.check_report()

    # 换设备后不再复用, 重新跑。
    page._slm_number.setValue(2)
    page.check_button().click()
    _wait_check_done(app, page)
    assert len(calls) == 2
    assert "复用上次自检" not in page.check_report()


def test_clicking_the_button_runs_one_check_and_renders_it(page, app, monkeypatch) -> None:
    calls: list[tuple[str, int, int]] = []

    def _fake(camera: str, cam_id: int, slm_number: int) -> HardwareReport:
        calls.append((camera, cam_id, slm_number))
        return HardwareReport(
            items=(
                CheckItem("Daheng 枚举", "ok", "枚举到 1 台大恒相机"),
                CheckItem("曝光安全上限", "warn", "曝光安全上限未配置", required=False),
            )
        )

    monkeypatch.setattr(hc, "check_hardware", _fake)

    page.check_button().click()
    _wait_check_done(app, page)

    assert calls == [("daheng", 0, 1)]
    report = page.check_report()
    assert "硬件自检通过" in report
    assert "Daheng 枚举：枚举到 1 台大恒相机" in report
    assert "曝光安全上限：曝光安全上限未配置" in report


def test_failing_report_says_so_and_edits_only_mark_it_stale(
    page, app, monkeypatch
) -> None:
    monkeypatch.setattr(
        hc,
        "check_hardware",
        lambda *args: HardwareReport(
            items=(CheckItem("SLM 探测", "fail", "未发现 SLM #1"),)
        ),
    )

    page.check_button().click()
    _wait_check_done(app, page)
    assert "未通过" in page.check_report()
    assert "未发现 SLM #1" in page.check_report()

    page._slm_number.setValue(2)
    report = page.check_report()
    assert "请重新点「设备自检」" in report
    # 只是标注过期: 结论与「未通过」都还在, 也没有重跑。
    assert "未发现 SLM #1" in report


# ------------------------------------------------------- boundary 自动增大周期回填
def test_boundary_failure_auto_bumps_period_and_backfills(page, app) -> None:
    """boundary 失败后自动**增大**倾斜相位周期并回填输入栏 (只回填, 不自动重跑)。"""
    page._tilt.setChecked(True)
    page.set_period(64.0)
    before = page._period.value()
    page._on_failed("ValueError: target intersects camera window boundary; adjust period/side")
    assert page._period.value() > before
    assert page._task is None  # 会话已收尾
    assert "已自动增大倾斜相位周期" in page._status_message


def test_boundary_failure_at_period_cap_does_not_change(page, app) -> None:
    """周期已到控件上限时, boundary 失败不改值 (避免死循环)。"""
    page._tilt.setChecked(True)
    cap = page._period.maximum()
    page.set_period(cap)
    page._on_failed("ValueError: target intersects camera window boundary; adjust period/side")
    assert page._period.value() == cap


def test_non_boundary_failure_does_not_bump_period(page, app) -> None:
    """无关失败 (如设备枚举) 不回填倾斜相位周期。"""
    page._tilt.setChecked(True)
    page.set_period(64.0)
    page._on_failed("RuntimeError: no daheng camera found")
    assert page._period.value() == 64.0


def test_check_and_run_are_mutually_exclusive(page, app, monkeypatch) -> None:
    release = threading.Event()

    def _slow(*args) -> HardwareReport:
        release.wait(10.0)
        return _ALL_OK

    monkeypatch.setattr(hc, "check_hardware", _slow)
    try:
        page.check_button().click()
        app.processEvents()
        assert not page.check_button().isEnabled()
        assert not page.run_button().isEnabled()

        page.check_button().click()  # 连点必须无效 (按钮禁用 + _checking 兜底)
        page._start_run()  # 直接调用也要被第二道闸拦住
        assert page.check_report().startswith("正在自检")
    finally:
        release.set()
    _wait_check_done(app, page)

    assert page.check_button().isEnabled()
    assert page.run_button().isEnabled()


def test_check_is_available_even_when_parameters_block_the_run(
    page, monkeypatch
) -> None:
    monkeypatch.setattr(
        page_mod.params, "validate_form", lambda values: (["至少选择一个目标形状"], [])
    )
    page.set_shapes(())
    assert not page.run_button().isEnabled()
    # 参数不合法不该挡住硬件诊断 —— 那时恰恰最需要知道设备在不在。
    assert page.check_button().isEnabled()
