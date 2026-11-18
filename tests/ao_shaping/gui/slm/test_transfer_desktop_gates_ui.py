"""#242 剩下三道安全门 (G2 / G2b / P4) 的页面接线与纯函数契约。

G3 (逐块饱和) 的行级契约在 ``test_transfer_desktop_campaign_blocks_ui.py`` 里,
这里只钉住:

* G2 红停行的阈值措辞来自会话自己的 ``acceptance_lower`` (不重述 0.85),
  产物缺失 / 相关性缺失时降级成一句"没有可用相关性", 绝不抛;
* G2 只往状态行追加红字, **不** 锁死「开始运行」;
* G2b 沿 ``zero_drift_verdict`` 的三档: 20 px 内安静, 20–40 px 黄警, 40 px 以上硬;
* P4 的四个按钮只写 / 删 ``campaign_control`` 约定的标记, 不另造状态文件。

离屏跑法: ``QT_QPA_PLATFORM=offscreen``; 页面一律 ``auto_detect_ccd=False``。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from ao_shaping.gui.slm.transfer_desktop import gates
from ao_shaping.gui.slm.transfer_desktop import page as page_mod
from ao_shaping.tools.slm.transfer import campaign_control


# ---------------------------------------------------------------------------
# 合成会话产物
# ---------------------------------------------------------------------------


def _session(
    root: Path, *, lower: float = 0.85, correlation: float | None = 0.72
) -> Path:
    """一个「开过验收阈值循环、最后一行相关性低于下限」的会话目录。"""
    root.mkdir(parents=True, exist_ok=True)
    (root / "session.json").write_text(
        json.dumps(
            {
                "settings": {"acceptance_lower": lower},
                "resolved": {"acceptance": "looped"},
            }
        ),
        encoding="utf-8",
    )
    record: dict[str, object] = {"measured_relative_l2": 0.10}
    if correlation is not None:
        record["measured_correlation"] = correlation
    (root / "rounds.json").write_text(json.dumps([record]), encoding="utf-8")
    return root


def _probe(root: Path, current: list[float], reference: list[float]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "optical_probe.json").write_text(
        json.dumps(
            {
                "current": {"zero_xy": current},
                "reference": {"zero_xy": reference},
            }
        ),
        encoding="utf-8",
    )
    return root


@pytest.fixture(scope="module")
def app() -> QApplication:
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    return instance


@pytest.fixture()
def make_page(app: QApplication, monkeypatch: pytest.MonkeyPatch):
    created: list[page_mod.TransferPage] = []
    monkeypatch.setattr(page_mod.params, "validate_form", lambda values: ([], []))

    def _make() -> page_mod.TransferPage:
        widget = page_mod.TransferPage(auto_detect_ccd=False)
        widget.show()
        app.processEvents()
        created.append(widget)
        return widget

    yield _make
    for widget in created:
        widget.close()
        widget.deleteLater()
    app.processEvents()


# ---------------------------------------------------------------------------
# G2: 相关性红停
# ---------------------------------------------------------------------------


def test_g2_line_uses_the_session_threshold(tmp_path: Path) -> None:
    session = _session(tmp_path / "s", lower=0.85, correlation=0.72)

    line = gates.correlation_red_stop_line(session)

    assert "红停（G2）" in line
    # 阈值与实测值都来自会话自己的产物, 页面不另写 0.85。
    assert "0.720" in line
    assert "0.85" in line


def test_g2_line_follows_a_relaxed_threshold(tmp_path: Path) -> None:
    # 同一个 0.72 的相关性: 会话把下限放到 0.60 就不该再红停。
    session = _session(tmp_path / "s", lower=0.60, correlation=0.72)

    assert gates.correlation_red_stop_line(session) == ""


def test_g2_line_when_correlation_is_missing(tmp_path: Path) -> None:
    session = _session(tmp_path / "s", correlation=None)

    line = gates.correlation_red_stop_line(session)

    assert "没有可用的验证相关性" in line


def test_g2_line_is_quiet_without_a_session_dir(tmp_path: Path) -> None:
    assert gates.correlation_red_stop_line(tmp_path / "missing") == ""


# ---------------------------------------------------------------------------
# G2b: 零级漂移三档
# ---------------------------------------------------------------------------


def test_zero_drift_20px_is_still_reuse(tmp_path: Path) -> None:
    root = _probe(tmp_path / "s", [120.0, 100.0], [100.0, 100.0])

    assert gates.zero_drift_warning(root) == ("", "")


def test_zero_drift_30px_warns(tmp_path: Path) -> None:
    root = _probe(tmp_path / "s", [130.0, 100.0], [100.0, 100.0])

    level, text = gates.zero_drift_warning(root)

    assert level == "warn"
    assert "超黄警" in text


def test_zero_drift_50px_hits_the_hard_limit(tmp_path: Path) -> None:
    root = _probe(tmp_path / "s", [150.0, 100.0], [100.0, 100.0])

    level, text = gates.zero_drift_warning(root)

    assert level == "hard"
    assert "硬上限" in text


def test_zero_drift_without_reference_is_quiet(tmp_path: Path) -> None:
    root = tmp_path / "s"
    root.mkdir()
    (root / "optical_probe.json").write_text(
        json.dumps({"current": {"zero_xy": [130.0, 100.0]}}), encoding="utf-8"
    )

    assert gates.zero_drift_warning(root) == ("", "")


# ---------------------------------------------------------------------------
# P4: 标记往返 (纯函数)
# ---------------------------------------------------------------------------


def test_confirm_gate_round_trip(tmp_path: Path) -> None:
    assert gates.stage_gate_state(tmp_path).confirm_pending is False

    gates.arm_confirm_gate(tmp_path)
    assert gates.stage_gate_state(tmp_path).confirm_pending is True

    assert gates.disarm_confirm_gate(tmp_path) is True
    assert gates.stage_gate_state(tmp_path).confirm_pending is False


def test_pause_request_round_trip(tmp_path: Path) -> None:
    gates.request_pause(tmp_path)
    assert gates.stage_gate_state(tmp_path).pause_requested is True

    assert gates.clear_pause_request(tmp_path) is True
    assert gates.stage_gate_state(tmp_path).pause_requested is False


def test_answers_write_campaign_control_markers(tmp_path: Path) -> None:
    gates.answer_confirm_gate(tmp_path, confirmed=True)
    assert (tmp_path / campaign_control.CONFIRM_RESUME_FILE).is_file()

    gates.answer_confirm_gate(tmp_path, confirmed=False)
    assert (tmp_path / campaign_control.CONFIRM_REJECT_FILE).is_file()


# ---------------------------------------------------------------------------
# G2 / G2b 的页面落点
# ---------------------------------------------------------------------------


def test_g2_red_line_is_appended_to_the_status(make_page, tmp_path: Path) -> None:
    session = _session(tmp_path / "s", lower=0.85, correlation=0.72)
    page = make_page()

    page._on_finished(str(session))

    status = page._status.text()
    assert page._status_kind == "complete"
    assert "红停（G2）" in status
    assert page._colors["error"] in status


def test_g2_red_line_does_not_lock_the_run_button(make_page, tmp_path: Path) -> None:
    session = _session(tmp_path / "s", lower=0.85, correlation=0.72)
    page = make_page()
    before = page._run_btn.isEnabled()

    page._on_finished(str(session))

    # G2 红停说的是「后续形状别再跑」, 不是「这个页面不许再跑」: 修正后重跑是正当操作。
    assert page._run_btn.isEnabled() is before


def test_zero_drift_warning_shows_in_the_progress_panel(
    make_page, tmp_path: Path
) -> None:
    page = make_page()
    page._session_dir = _probe(tmp_path / "s", [130.0, 100.0], [100.0, 100.0])

    page._refresh_progress()

    assert not page._drift_warning.isHidden()
    html = page._drift_warning.text()
    assert "G2b" in html
    assert "超黄警" in html
    assert page._colors["warn"] in html


def test_zero_drift_within_the_warn_line_keeps_the_label_hidden(
    make_page, tmp_path: Path
) -> None:
    page = make_page()
    page._session_dir = _probe(tmp_path / "s", [110.0, 100.0], [100.0, 100.0])

    page._refresh_progress()

    assert page._drift_warning.isHidden()


# ---------------------------------------------------------------------------
# P4 的页面落点
# ---------------------------------------------------------------------------


def test_gate_controls_are_disabled_without_a_campaign_dir(make_page) -> None:
    page = make_page()

    page.reload_campaign_progress()

    for widget in (page._confirm_toggle, page._confirm_yes, page._confirm_no, page._pause_btn):
        assert not widget.isEnabled()


def test_confirm_toggle_writes_and_clears_the_marker(make_page, tmp_path: Path) -> None:
    page = make_page()
    page.set_campaign_dir(tmp_path)
    assert page._confirm_toggle.isEnabled()

    page._confirm_toggle.click()

    assert campaign_control.stage_confirm_requested(tmp_path)
    assert page._confirm_toggle.text() == page_mod._CONFIRM_ON_TEXT
    assert page._confirm_yes.isEnabled()

    page._confirm_toggle.click()

    assert not campaign_control.stage_confirm_requested(tmp_path)
    assert page._confirm_toggle.text() == page_mod._CONFIRM_OFF_TEXT
    assert not page._confirm_yes.isEnabled()


def test_pause_button_toggles_the_request(make_page, tmp_path: Path) -> None:
    page = make_page()
    page.set_campaign_dir(tmp_path)

    page._pause_btn.click()

    assert campaign_control.stage_stop_requested(tmp_path)
    assert page._pause_btn.text() == page_mod._PAUSE_CLEAR_TEXT

    page._pause_btn.click()

    assert not campaign_control.stage_stop_requested(tmp_path)
    assert page._pause_btn.text() == page_mod._PAUSE_REQUEST_TEXT


def test_answer_buttons_write_the_two_reply_markers(make_page, tmp_path: Path) -> None:
    page = make_page()
    page.set_campaign_dir(tmp_path)
    page._confirm_toggle.click()

    page._confirm_yes.click()
    assert (tmp_path / campaign_control.CONFIRM_RESUME_FILE).is_file()

    page._confirm_no.click()
    assert (tmp_path / campaign_control.CONFIRM_REJECT_FILE).is_file()

    # 挂起标记不归页面删: 那是 runner 在段边界读取后的职责。
    assert campaign_control.stage_confirm_requested(tmp_path)
