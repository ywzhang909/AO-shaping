"""段间暂停闸：请求文件的存在性语义 (免设备)。"""

from __future__ import annotations

import json

from ao_shaping.tools.slm.transfer.campaign_control import (
    STOP_AFTER_STAGE_FILE,
    clear_stage_stop,
    request_stage_stop,
    stage_stop_requested,
)


def test_no_request_means_no_pause(tmp_path):
    assert stage_stop_requested(tmp_path) is False
    assert clear_stage_stop(tmp_path) is False


def test_a_request_is_a_single_file_in_the_campaign_root(tmp_path):
    out = tmp_path / "campaign" / "run_01"
    path = request_stage_stop(out, reason="想先看一眼这一段", requested_by="operator")

    assert path == out / STOP_AFTER_STAGE_FILE
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["reason"] == "想先看一眼这一段"
    assert payload["requested_by"] == "operator"
    assert payload["timestamp_utc"]
    assert stage_stop_requested(out) is True


def test_clearing_is_idempotent(tmp_path):
    out = tmp_path / "campaign"
    request_stage_stop(out, reason="once")

    assert clear_stage_stop(out) is True
    assert stage_stop_requested(out) is False
    assert clear_stage_stop(out) is False


def test_the_request_only_looks_at_existence(tmp_path):
    """内容坏了也不该让暂停请求凭空失效 —— 读侧只看文件在不在。"""
    out = tmp_path / "campaign"
    out.mkdir()
    (out / STOP_AFTER_STAGE_FILE).write_text("{ 坏掉的 json", encoding="utf-8")

    assert stage_stop_requested(out) is True
    assert clear_stage_stop(out) is True
    assert stage_stop_requested(out) is False


def test_a_repeated_request_stays_one_request(tmp_path):
    out = tmp_path / "campaign"
    request_stage_stop(out, reason="第一次")
    request_stage_stop(out, reason="第二次")

    payload = json.loads((out / STOP_AFTER_STAGE_FILE).read_text(encoding="utf-8"))
    assert payload["reason"] == "第二次"
    assert len(list(out.iterdir())) == 1


def test_the_marker_name_keeps_the_parallel_fit_convention():
    assert STOP_AFTER_STAGE_FILE == "stop_after_stage.request"


# ---------------------------------------------------- 确认后继续闸 (#243 前置四)

from ao_shaping.tools.slm.transfer.campaign_control import (  # noqa: E402
    CONFIRM_AND_CONTINUE_FILE,
    CONFIRM_REJECT_FILE,
    CONFIRM_RESUME_FILE,
    await_stage_confirmation,
    clear_stage_confirm,
    request_stage_confirm,
    stage_confirm_requested,
)


class _Clock:
    """注入时钟: 每次读表前进 step 秒, sleep 只推进不等待。"""

    def __init__(self, step=10.0):
        self.now = 0.0
        self.step = step

    def monotonic(self):
        return self.now

    def sleep(self, _s):
        self.now += self.step


def test_confirm_gate_off_by_default(tmp_path):
    assert stage_confirm_requested(tmp_path) is False


def test_confirm_request_resume_and_clear_round_trip(tmp_path):
    path = request_stage_confirm(tmp_path, reason="看一眼成本表")
    assert path == tmp_path / CONFIRM_AND_CONTINUE_FILE
    assert stage_confirm_requested(tmp_path) is True
    (tmp_path / CONFIRM_RESUME_FILE).write_text("{}", encoding="utf-8")
    clock = _Clock()
    assert await_stage_confirmation(tmp_path, monotonic=clock.monotonic,
                                    sleep=clock.sleep) == "confirmed"
    assert clear_stage_confirm(tmp_path) is True
    assert not (tmp_path / CONFIRM_AND_CONTINUE_FILE).exists()
    assert not (tmp_path / CONFIRM_RESUME_FILE).exists()
    assert clear_stage_confirm(tmp_path) is False


def test_confirm_rejection_beats_confirmation(tmp_path):
    request_stage_confirm(tmp_path)
    (tmp_path / CONFIRM_RESUME_FILE).write_text("{}", encoding="utf-8")
    (tmp_path / CONFIRM_REJECT_FILE).write_text("{}", encoding="utf-8")
    clock = _Clock()
    assert await_stage_confirmation(tmp_path, monotonic=clock.monotonic,
                                    sleep=clock.sleep) == "rejected"


def test_confirm_timeout_is_fail_closed_rejection(tmp_path):
    request_stage_confirm(tmp_path)
    clock = _Clock(step=60.0)
    assert await_stage_confirmation(tmp_path, timeout_s=120.0,
                                    monotonic=clock.monotonic,
                                    sleep=clock.sleep) == "timed_out"


class _Journal:
    def __init__(self):
        self.entries = []

    def record(self, key, status, **extra):
        self.entries.append(dict(stage=key, status=status, **extra))


def test_confirm_at_boundary_passthrough_without_request(tmp_path):
    from ao_shaping.tools.slm.transfer.session_record import confirm_at_boundary
    journal = _Journal()
    assert confirm_at_boundary(journal, "shape_a/stage_1", tmp_path) is False
    assert journal.entries == []


def test_confirm_at_boundary_confirms_and_continues(tmp_path):
    from ao_shaping.tools.slm.transfer.session_record import confirm_at_boundary
    request_stage_confirm(tmp_path)
    (tmp_path / CONFIRM_RESUME_FILE).write_text("{}", encoding="utf-8")
    journal = _Journal()
    assert confirm_at_boundary(journal, "shape_a/stage_1", tmp_path) is False
    assert journal.entries[0]["status"] == "confirm_confirmed"
    assert not stage_confirm_requested(tmp_path)


def test_confirm_at_boundary_reject_stops_the_campaign(tmp_path):
    from ao_shaping.tools.slm.transfer.session_record import confirm_at_boundary
    request_stage_confirm(tmp_path)
    (tmp_path / CONFIRM_REJECT_FILE).write_text("{}", encoding="utf-8")
    journal = _Journal()
    assert confirm_at_boundary(journal, "shape_a/stage_1", tmp_path) is True
    assert journal.entries[0]["status"] == "confirm_rejected"


def test_confirm_at_boundary_timeout_stops_like_rejection(tmp_path):
    import ao_shaping.tools.slm.transfer.session_record as session_record
    request_stage_confirm(tmp_path)
    journal = _Journal()
    assert session_record.confirm_at_boundary(
        journal, "shape_a/stage_1", tmp_path, timeout_s=0.0) is True
    assert journal.entries[0]["status"] == "confirm_timed_out"
