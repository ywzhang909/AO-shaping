"""离线测试：可微优化控制器页的「一键光路快速探测」。

这个按钮一次要给三份实测结果（四态 + 建议曝光写回 + 证据目录），并且
**无设备也能跑测试**：采集器用注入的假证据 JSON（``render_optical_probe`` 的
``probe_readings`` 接受路径），四态判定不在这页重算（单一来源是
``tools/slm/panel/probe_policy.decide``）。

锁住四件事（都是实际踩点定的口径）：
1. 三份结果都渲染出来，四态各有自己的措辞（不是一句「失败」）。
2. 写回走 ``bench_settings.record_exposure_probe`` + ``queue_pending``，不直接
   改 widget 会话状态（那样会丢取样时刻、会和 widget 生命周期打架）。
3. 没有可用窗口时不写回、只把「这张表就是结论」说清楚。
4. 无设备 / 忙碌时按钮 disabled 且给出原因，不抛异常。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from ao_shaping.gui.slm import diff_opt_controller as ctl
from ao_shaping.tools.slm.panel import probe_policy

# AppTest.from_file 相对路径按调用文件解析，所以从本测试位置绝对拼入口。
APP = str(
    Path(__file__).resolve().parents[4]
    / "src"
    / "ao_shaping"
    / "gui"
    / "slm"
    / "diff_opt_controller.py"
)

# 探测按钮的 widget key（页面用 f"{prefix or 'bench'}_optical_probe"）。
PROBE_BUTTON_KEY = "slm1" + "_optical_probe"
NO_DEVICE_BUTTON_KEY = "bench" + "_optical_probe"

# 只渲染探测区，绕开侧栏设备枚举 —— 注入路径测试的最小入口。
PROBE_SCRIPT = """
import streamlit as st
from ao_shaping.gui.slm.diff_opt_controller import render_optical_probe

render_optical_probe(1, "slm1", st.session_state.get("probe_path"))
"""


def _payload(
    state: str,
    *,
    exposure_ms: float | None = 0.35,
    reasons: list[str] | None = None,
    advice: list[str] | None = None,
) -> dict[str, Any]:
    """构造一份 ``optical_path_probe/1`` 证据（字段与脚本落盘口径一致）。"""
    payload: dict[str, Any] = {
        "schema": ctl._PROBE_SCHEMA,
        "created_utc": "2026-10-29T22:00:00Z",
        "verdict": {
            "state": state,
            "reasons": reasons if reasons is not None else [f"{state} 的实测依据"],
            "advice": advice if advice is not None else [f"{state} 的建议动作"],
            "exit_code": 0 if state == probe_policy.STATE_OK else 2,
        },
        "ladder_summary": {
            "noise_floor": {
                "threshold_counts": 12.5,
                "median_counts": 3.0,
                "sigma_counts": 2.1,
                "peak_counts": 160.0,
                "exposure_ms": 0.2,
            },
            "best_usable_peak_counts": 168.0,
            "usable_window_ms": [0.05, 0.2],
            "recommended_exposure": (
                None
                if exposure_ms is None
                else {
                    "exposure_ms": exposure_ms,
                    "peak_counts": 168.0,
                    "in_target_band": True,
                    "target_band_counts": [100.0, 200.0],
                }
            ),
        },
    }
    return payload


def _write_evidence(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "optical_path_probe.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _probe_page(
    evidence: Path | None = None,
    *,
    connected: bool = True,
    busy: bool = False,
) -> AppTest:
    at = AppTest.from_string(PROBE_SCRIPT, default_timeout=90)
    if connected:
        at.session_state["slm1"] = object()
        at.session_state["slm1_connected"] = True
    if busy:
        at.session_state["bench_busy"] = True
    if evidence is not None:
        at.session_state["probe_path"] = str(evidence)
    return at


def _texts(at: AppTest) -> str:
    """把所有可见文案拼成一个串，方便断言（markdown / caption / 各种提示框）。"""
    chunks: list[str] = []
    for group in (at.markdown, at.caption, at.success, at.warning, at.error, at.info):
        chunks.extend(str(item.value) for item in group)
    chunks.extend(str(item.value) for item in at.code)
    return "\n".join(chunks)


# --------------------------------------------------------------------------- #
# 1. 三份结果 + 四态措辞
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("state", "headline"),
    [
        (probe_policy.STATE_OK, "光路正常"),
        (probe_policy.STATE_LOW_SNR, "信噪比不足"),
        (probe_policy.STATE_WEAK_PATH, "光路被衰减"),
        (probe_policy.STATE_NO_RESPONSE, "面板没在调制"),
    ],
)
def test_each_state_renders_its_own_verdict(
    tmp_path: Path, state: str, headline: str
) -> None:
    at = _probe_page(_write_evidence(tmp_path, _payload(state)))
    at.run()
    at.button(key=PROBE_BUTTON_KEY).click().run()
    assert not at.exception
    text = _texts(at)
    # ① 四态（状态名 + 该态专属措辞）
    assert "① 光路四态" in text
    assert f"`{state}`" in text
    assert headline in text
    # ② 建议曝光、③ 证据目录都在
    assert "② 建议曝光" in text
    assert "③ 证据落盘" in text


def test_no_response_uses_the_error_box_while_others_use_warning(tmp_path: Path) -> None:
    silent = _probe_page(_write_evidence(tmp_path, _payload(probe_policy.STATE_NO_RESPONSE)))
    silent.run()
    silent.button(key=PROBE_BUTTON_KEY).click().run()
    assert any("面板没在调制" in item.value for item in silent.error)

    low = _probe_page(_write_evidence(tmp_path, _payload(probe_policy.STATE_LOW_SNR)))
    low.run()
    low.button(key=PROBE_BUTTON_KEY).click().run()
    assert any("信噪比不足" in item.value for item in low.warning)


# --------------------------------------------------------------------------- #
# 2. 写回：record_exposure_probe + queue_pending（不是直接改 widget）
# --------------------------------------------------------------------------- #
def test_click_writes_exposure_through_the_detected_table(tmp_path: Path) -> None:
    evidence = _write_evidence(tmp_path, _payload(probe_policy.STATE_OK, exposure_ms=0.35))
    at = _probe_page(evidence)
    at.run()

    at.button(key=PROBE_BUTTON_KEY).click().run()
    assert not at.exception

    outcome = at.session_state["diff_opt_optical_probe"]
    assert outcome["written_ms"] == pytest.approx(0.35)
    assert outcome["evidence"] == str(evidence)

    # 取样时刻 + 探测值表（写回的唯一留痕点）。
    detected = at.session_state["bench_detected"]
    assert detected["exposure_ms"] == pytest.approx(0.35)
    assert "probed_at" in detected
    # widget 值排队等下一次 flush，而不是当场改 widget key。
    assert at.session_state["bench_pending"]["bench_exposure_ms"] == pytest.approx(0.35)

    text = _texts(at)
    assert "已写回" in text
    # 证据落盘目录显示出来。
    assert str(evidence) in text


def test_no_usable_window_does_not_write_back(tmp_path: Path) -> None:
    evidence = _write_evidence(
        tmp_path, _payload(probe_policy.STATE_LOW_SNR, exposure_ms=None)
    )
    at = _probe_page(evidence)
    at.run()

    at.button(key=PROBE_BUTTON_KEY).click().run()
    assert not at.exception

    outcome = at.session_state["diff_opt_optical_probe"]
    assert outcome["written_ms"] is None
    assert "bench_detected" not in at.session_state
    text = _texts(at)
    assert "不写回" in text
    # 没有可用窗口时「这张表本身就是结论」要说清楚。
    assert "这张表本身就是结论" in text


# --------------------------------------------------------------------------- #
# 3. 无设备 / 忙碌：按钮 disabled + 原因，不抛异常
# --------------------------------------------------------------------------- #
def test_no_device_shows_disabled_probe_button_with_reason() -> None:
    """真页面路径：没连接 SLM 时按钮仍画出，但 disabled 且说明缺什么。"""
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception

    button = at.button(key=NO_DEVICE_BUTTON_KEY)
    assert button.disabled is True
    assert any("未连接 SLM" in item.value for item in at.caption)


def test_busy_bench_shows_disabled_probe_button_with_reason() -> None:
    at = _probe_page(connected=True, busy=True)
    at.run()
    assert not at.exception

    button = at.button(key=PROBE_BUTTON_KEY)
    assert button.disabled is True
    assert any("占用" in item.value for item in at.caption)


def test_connected_idle_bench_enables_the_button() -> None:
    at = _probe_page(connected=True)
    at.run()
    assert not at.exception

    button = at.button(key=PROBE_BUTTON_KEY)
    assert button.disabled is False
    # 还没按过：明确说「尚无探测结果」，而不是空白。
    assert any("尚无探测结果" in item.value for item in at.caption)


# --------------------------------------------------------------------------- #
# 4. 形状校验：判定不在这页重算，坏的证据要报 «不可用» 而非乱取值
# --------------------------------------------------------------------------- #
def test_payload_guard_rejects_wrong_schema() -> None:
    with pytest.raises(ctl.ProbeUnavailable):
        ctl._check_probe_payload(
            {"schema": "something/else", "verdict": {}, "ladder_summary": {}},
            "test",
        )


def test_payload_guard_rejects_unknown_state() -> None:
    bad = _payload(probe_policy.STATE_OK)
    bad["verdict"]["state"] = "maybe"
    with pytest.raises(ctl.ProbeUnavailable):
        ctl._check_probe_payload(bad, "test")


def test_missing_evidence_renders_as_actionable_error(tmp_path: Path) -> None:
    at = _probe_page(tmp_path / "does-not-exist.json")
    at.run()

    at.button(key=PROBE_BUTTON_KEY).click().run()
    assert not at.exception

    outcome = at.session_state["diff_opt_optical_probe"]
    assert outcome["error"]
    assert outcome["payload"] is None
    assert any("读不到探测证据" in item.value for item in at.error)
