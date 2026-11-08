"""迁移学习闭环页 G1（黄字逐条「已确认」）的离屏契约。

今天的 ③ 校验有两条通道: 红字阻塞启动, 黄字只上色。G1 把黄字也变成一道闸 ——
每条提醒配一个勾选框, 全部勾上「已确认」之前 ④ 的启动键一直禁用。这里钉住:

* 没有提醒时确认区整块隐藏, 启动键可用;
* 每条提醒一个勾选框, 标签以「已确认 ·」开头, 长文截断但 tooltip 留全文;
* 勾满才放行, 取消一条立刻回到禁用;
* 改任何参数都会作废确认 (重建勾选行), 提醒仍在;
* 有红字时根本不显示勾选框, 启动键 tooltip 是第一条红字;
* 程序方式点 ``_start_run`` 也拦得住 (不建目录、不进运行态);
* 程序方式 ``set_all_acks(False)`` 能重新拦上; 运行中确认区随表单一起置灰。

离屏跑法: ``QT_QPA_PLATFORM=offscreen``; 页面一律 ``auto_detect_ccd=False``。
"""

from __future__ import annotations

import os

# 必须在任何 QApplication 之前设置: CI / 无显示环境走离屏后端。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from ao_shaping.gui.slm.transfer_desktop import page as page_mod

_WARN_A = "目标边长接近采样窗边缘\n    → 建议不超过 380 px"
_WARN_B = "倾斜相位已开启\n    → 确认周期与目标一致"
_LONG_WARNING = (
    "这是一条很长的提醒，用来检查勾选标签会不会把左侧 560 px 的表单列撑宽"
    "\n    → 完整内容只应出现在 tooltip 里"
)


@pytest.fixture(scope="module")
def app() -> QApplication:
    """离屏 QApplication (一个进程只能有一个)。"""
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    return instance


@pytest.fixture()
def make_page(app: QApplication, monkeypatch: pytest.MonkeyPatch):
    """按测试给定的 ``(errors, warnings)`` 造页面; 关闭相机自动探测。"""
    created: list[page_mod.TransferPage] = []

    def _make(
        errors: list[str] | None = None,
        warnings: list[str] | None = None,
    ) -> page_mod.TransferPage:
        canned_errors = list(errors or ())
        canned_warnings = list(warnings or ())
        monkeypatch.setattr(
            page_mod.params,
            "validate_form",
            lambda values: (list(canned_errors), list(canned_warnings)),
        )
        widget = page_mod.TransferPage(auto_detect_ccd=False)
        widget.show()
        app.processEvents()
        created.append(widget)
        return widget

    yield _make
    for widget in created:
        widget.close()
    app.processEvents()


# --- 无提醒 -----------------------------------------------------------------


def test_no_warnings_hides_the_ack_box_and_leaves_the_button_enabled(
    make_page,
) -> None:
    page = make_page()

    assert not page._ack_box.isVisible()
    assert page.pending_acks() == []
    assert page.run_button().isEnabled()
    assert page.run_button().toolTip() == "启动整场迁移学习会话"


# --- 有提醒 -----------------------------------------------------------------


def test_each_warning_gets_a_checkbox_and_blocks_the_button(make_page) -> None:
    page = make_page(warnings=[_WARN_A, _WARN_B])

    checks = page._ack_checks
    assert page._ack_box.isVisible()
    assert len(checks) == 2
    assert all(check.text().startswith("已确认 · ") for check in checks)
    assert checks[0].toolTip() == _WARN_A
    assert "→" in checks[0].toolTip()
    assert page._ack_hint.text() == "改任何参数都会作废这些确认，需重新勾选"
    assert page.pending_acks() == [_WARN_A, _WARN_B]
    assert not page.run_button().isEnabled()
    assert "勾选" in page.run_button().toolTip()
    # ③ 的黄字尾句也说明「要逐条确认才放行」。
    assert "逐条勾选「已确认」后才能启动" in page._checks.text()


def test_a_long_warning_is_truncated_but_the_tooltip_keeps_it_whole(make_page) -> None:
    page = make_page(warnings=[_LONG_WARNING])

    check = page._ack_checks[0]
    assert check.text().endswith("…")
    assert len(check.text()) <= 34
    assert check.toolTip() == _LONG_WARNING


def test_ticking_every_warning_releases_the_button(make_page) -> None:
    page = make_page(warnings=[_WARN_A, _WARN_B])

    page._ack_checks[0].setChecked(True)
    assert not page.run_button().isEnabled()
    page._ack_checks[1].setChecked(True)

    assert page.run_button().isEnabled()
    assert page.pending_acks() == []

    page._ack_checks[1].setChecked(False)
    assert not page.run_button().isEnabled()
    assert page.pending_acks() == [_WARN_B]


def test_changing_a_parameter_invalidates_the_acks(make_page) -> None:
    page = make_page(warnings=[_WARN_A])
    page.set_all_acks(True)
    assert page.run_button().isEnabled()

    page.set_shapes(["star"])

    assert not page.run_button().isEnabled()
    assert len(page._ack_checks) == 1
    assert not page._ack_checks[0].isChecked()
    assert page.pending_acks() == [_WARN_A]


# --- 红字优先 ---------------------------------------------------------------


def test_an_error_hides_the_checkboxes_and_shows_the_first_error(make_page) -> None:
    page = make_page(errors=["目标边长超出采样窗"], warnings=[_WARN_A])

    assert page._ack_checks == []
    assert not page._ack_box.isVisible()
    assert not page.run_button().isEnabled()
    assert page.run_button().toolTip() == "目标边长超出采样窗"
    assert "项阻塞 · 修好才能启动" in page._checks.text()
    # 确认区整块隐藏时, 黄字段的尾句不能还说「逐条勾选后才能启动」——
    # 那是在指一堆不存在的勾选框。
    assert "逐条勾选「已确认」后才能启动" not in page._checks.text()
    assert "先修阻塞项，之后逐条勾选「已确认」" in page._checks.text()


def test_an_error_alone_also_blocks_without_checkboxes(make_page) -> None:
    page = make_page(errors=["探针数不是 6 的倍数"])

    assert page._ack_checks == []
    assert not page._ack_box.isVisible()
    assert not page.run_button().isEnabled()
    assert page.run_button().toolTip() == "探针数不是 6 的倍数"


# --- 第二道闸 / 程序接口 ----------------------------------------------------


def test_a_programmatic_start_still_refuses_an_unacked_warning(make_page) -> None:
    page = make_page(warnings=[_WARN_A])

    page._start_run()

    assert not page.is_running()
    assert page.session_dir() is None
    assert not page.run_button().isEnabled()


def test_set_all_acks_false_reblocks(make_page) -> None:
    page = make_page(warnings=[_WARN_A])
    page.set_all_acks(True)
    assert page.run_button().isEnabled()

    page.set_all_acks(False)

    assert not page.run_button().isEnabled()
    assert page.pending_acks() == [_WARN_A]


def test_the_ack_box_greys_out_with_the_rest_of_the_form(make_page) -> None:
    page = make_page(warnings=[_WARN_A])
    assert page._ack_box.isEnabled()

    page._set_form_enabled(False)

    assert not page._ack_box.isEnabled()


def test_a_theme_repaint_keeps_the_acks(make_page) -> None:
    """系统明暗切换不是「改了参数」: 换主题只重上色, 不该作废已勾的确认。"""
    page = make_page(warnings=[_WARN_A])
    page.set_all_acks(True)
    assert page.run_button().isEnabled()
    before = list(page._ack_checks)

    page._apply_theme()

    assert page._ack_checks == before  # 同一批控件, 没有重建
    assert page.pending_acks() == []
    assert page.run_button().isEnabled()
