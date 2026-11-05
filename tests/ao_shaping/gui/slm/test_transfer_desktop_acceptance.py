"""迁移学习闭环页「验收阈值循环」的离屏交互契约。

验收阈值循环 = CCD 与模型预测比对、不达标则重训 (两轮之间可停)。页面端只做三件事:

* 一个开关 + 两个阈值浮点框 (相关性下界 / 相对 L2 上界), 关 → 禁用置灰并显示
  算法建议; 开 → 可编辑, 阈值原样进表单快照, 进而随 :func:`build_session` 进会话;
* 「停止 (两轮间生效)」是个软停止: 只置位一个 standard-lib flag, 让跑在两轮之间
  的会话优雅收尾; 没开循环 / 没在跑时按钮禁用;
* 结算时若会话开过验收循环, 状态行补一行「验收阈值: 通过 / 未达标…」。

这里不该也不需要碰任何硬件: 全部离屏、参数校验换成常量、会话 tasks 用桩。
"""

from __future__ import annotations

import json
import os
import types

# 必须在任何 QApplication 之前设置: CI / 无显示环境走离屏后端。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from ao_shaping.gui.slm.transfer_desktop import page as page_mod
from ao_shaping.gui.slm.transfer_desktop import params


class _StubSignal:
    def connect(self, *args, **kwargs):
        pass


class _StubTask(page_mod.QRunnable):
    """接住 ``_start_run`` 建的任务, 不真正开会话 (不碰硬件)。"""

    def __init__(self, values, out, should_stop=None):
        super().__init__()
        self.values = values
        self.out = out
        self.should_stop = should_stop
        self.signals = types.SimpleNamespace(
            stage=_StubSignal(),
            frame=_StubSignal(),
            finished=_StubSignal(),
            failed=_StubSignal(),
        )

    def run(self):
        pass


@pytest.fixture(scope="module")
def app() -> QApplication:
    """离屏 QApplication (一个进程只能有一个)。"""
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    return instance


@pytest.fixture()
def page(app: QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path):
    """离屏页面; 校验换成常量, tasks 换成桩, 免得拉进 torch / 硬件。"""
    monkeypatch.setattr(page_mod.params, "validate_form", lambda values: ([], []))
    monkeypatch.setattr(page_mod.params, "plan_session_dir", lambda: tmp_path / "s")
    monkeypatch.setattr(page_mod, "_SessionTask", _StubTask)
    widget = page_mod.TransferPage()
    app.processEvents()
    yield widget
    widget.close()
    app.processEvents()


# --- 验收阈值的控件与默认值 ----------------------------------------------


def test_acceptance_controls_default_off_with_algorithm_suggested_bounds(page) -> None:
    """默认关 = 当前单轮行为; 连着的两个浮点框给算法建议值。"""
    assert page._accept.isChecked() is False
    # 关 → 上下界禁用置灰, 常驻提示读的是 params 常量。
    assert page._accept_lower.isEnabled() is False
    assert page._accept_upper.isEnabled() is False
    assert page._accept_lower.value() == pytest.approx(0.85)
    assert page._accept_upper.value() == pytest.approx(0.25)
    assert "0.85" in page._accept_hint.text()
    assert "0.25" in page._accept_hint.text()


def test_enabling_acceptance_enables_the_bounds(page) -> None:
    """开循环 → 上下界可编辑; 关回来 → 又禁用。"""
    page._accept.setChecked(True)
    assert page._accept_lower.isEnabled()
    assert page._accept_upper.isEnabled()

    page._accept.setChecked(False)
    assert not page._accept_lower.isEnabled()
    assert not page._accept_upper.isEnabled()


def test_form_values_carry_the_acceptance_snapshot(page) -> None:
    """控件当前值原样进表单快照 (供 params 层建会话用)。"""
    page._accept.setChecked(True)
    page._accept_lower.setValue(0.9)
    page._accept_upper.setValue(0.2)

    values = page.form_values()

    assert values.acceptance_enabled is True
    assert values.acceptance_metric == "quality"
    assert values.acceptance_lower == pytest.approx(0.9)
    assert values.acceptance_upper == pytest.approx(0.2)
    assert values.acceptance_max_rounds == 8  # 页面没有这控件, 用默认安全带


def test_acceptance_defaults_match_params_layer(page) -> None:
    """页面控件默认值就是 params 层的默认值 (单一口径, 不另立一套)。"""
    values = params.TransferFormValues()
    assert page._accept.isChecked() == bool(values.acceptance_enabled)
    assert page._accept_lower.value() == pytest.approx(values.acceptance_lower)
    assert page._accept_upper.value() == pytest.approx(values.acceptance_upper)


# --- 停止按钮 -------------------------------------------------------------


def test_stop_button_is_disabled_until_a_running_session(page) -> None:
    assert page.stop_button().isEnabled() is False


def test_starting_with_acceptance_armed_enables_stop_and_flag(page, monkeypatch) -> None:
    """开着验收循环启动 → 停止可用; 点击置位 flag 并传给任务, 随后禁用。"""
    captured = {}

    class _CapTask(_StubTask):
        def __init__(self, values, out, should_stop=None):
            super().__init__(values, out, should_stop)
            captured["should_stop"] = should_stop

    monkeypatch.setattr(page_mod, "_SessionTask", _CapTask)
    monkeypatch.setattr(page._pool, "start", lambda task: None)
    monkeypatch.setattr(page._poll, "start", lambda *args: None)
    page._accept.setChecked(True)
    page._start_run()

    assert page.stop_button().isEnabled() is True
    assert page.acceptance_stop_is_set() is False
    assert captured.get("should_stop") is not None

    page.stop_button().click()
    assert page.acceptance_stop_is_set() is True
    assert page.stop_button().isEnabled() is False  # 防连点

    page._on_failed("cleanup")


def test_starting_with_acceptance_off_keeps_stop_disabled(page, monkeypatch) -> None:
    """没开验收循环 → 即使会话在跑, 停止按钮也不可用 (跨轮无得可停)。"""
    monkeypatch.setattr(page._pool, "start", lambda task: None)
    monkeypatch.setattr(page._poll, "start", lambda *args: None)
    page._start_run()
    assert page.stop_button().isEnabled() is False
    page._on_failed("cleanup")


def test_new_run_resets_the_stop_flag(page, monkeypatch) -> None:
    """上一次点了停止 → 新一轮启动时 flag 复位。"""
    page._acceptance_stop.set()
    monkeypatch.setattr(page._pool, "start", lambda task: None)
    monkeypatch.setattr(page._poll, "start", lambda *args: None)
    page._start_run()
    assert page.acceptance_stop_is_set() is False
    page._on_failed("cleanup")


def test_settle_disables_stop_and_clears_flag(page, monkeypatch) -> None:
    """收尾 (完成 / 失败 / 停止都走) 一律禁用停止按钮。"""
    captured = {}

    class _CapTask(_StubTask):
        def __init__(self, values, out, should_stop=None):
            super().__init__(values, out, should_stop)
            captured["should_stop"] = should_stop

    monkeypatch.setattr(page_mod, "_SessionTask", _CapTask)
    monkeypatch.setattr(page._pool, "start", lambda task: None)
    monkeypatch.setattr(page._poll, "start", lambda *args: None)
    page._accept.setChecked(True)
    page._start_run()
    assert page.stop_button().isEnabled() is True

    page._on_failed("cleanup")
    assert page.stop_button().isEnabled() is False


# --- 验收阈值结论的显示 --------------------------------------------------


def _write_looped_session(tmp_path, accepted_rounds):
    """造一个有唯一 round 记录、会判「通过」的会话, 供显示测试。"""
    session_dir = tmp_path / "s"
    session_dir.mkdir(exist_ok=True)
    (session_dir / "session.json").write_text(
        '{"resolved": {"acceptance": "looped"}}', encoding="utf-8"
    )
    (session_dir / "journal.json").write_text("[]", encoding="utf-8")
    rounds = [
        {"round": i, "measured_correlation": 0.9, "measured_relative_l2": 0.2}
        for i in range(accepted_rounds)
    ]
    (session_dir / "rounds.json").write_text(json.dumps(rounds), encoding="utf-8")
    return session_dir


def test_verdict_line_skips_non_looped_sessions(tmp_path) -> None:
    session_dir = tmp_path / "s"
    session_dir.mkdir()
    (session_dir / "session.json").write_text(
        '{"resolved": {"acceptance": "model_only"}}', encoding="utf-8"
    )
    assert page_mod.acceptance_verdict_line(session_dir) == ""


def test_verdict_line_shows_passed_for_looped_accepted(tmp_path) -> None:
    session_dir = _write_looped_session(tmp_path, accepted_rounds=2)
    line = page_mod.acceptance_verdict_line(session_dir)
    assert "验收阈值" in line
    assert "通过" in line
    assert "2 轮" in line


def test_verdict_line_never_raises_on_missing_dir(tmp_path) -> None:
    assert page_mod.acceptance_verdict_line(tmp_path / "nope") == ""


def test_verdict_line_never_raises_on_partial_artifacts(tmp_path) -> None:
    session_dir = tmp_path / "s"
    session_dir.mkdir()
    (session_dir / "session.json").write_text(
        '{"resolved": {"acceptance": "looped"}}', encoding="utf-8"
    )
    # 没有 rounds.json / fit 产物 → 不报错, 返回空串 (没值可显示)。
    assert page_mod.acceptance_verdict_line(session_dir) == ""