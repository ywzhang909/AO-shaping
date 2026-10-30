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
