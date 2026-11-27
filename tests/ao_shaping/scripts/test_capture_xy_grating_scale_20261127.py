"""独立 x/y 光栅标度采集的离线契约测试（不碰设备、不跑 --execute）。

锁定三件事：冻结函数确实是复用而非重写；合成 ±1 位移可算出 pitch；
无信号/饱和/缺级一律 fail-closed 且 plan 确定性。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

import scripts.slm.capture_s1_mapping_disambiguation_20261124 as s1
import scripts.slm.capture_xy_grating_scale_20261127 as capture
import scripts.slm.issue247_axis_map_solve_20261121 as axis_map
import scripts.slm.issue247_geometry_capture_20261120 as geometry
from scripts.slm.judge_s1_criteria_offline_20261127 import FROZEN_2_COL_WINDOW_PX

ZERO = (110.0, 150.0)
PITCH = 95.0


def _plan_args(out: Path) -> argparse.Namespace:
    return argparse.Namespace(
        out=str(out),
        exposure_ms=capture.DEFAULT_EXPOSURE_MS,
        wait_until_idle=capture.DEFAULT_WAIT_IDLE_S,
        slm_number=capture.DEFAULT_SLM_NUMBER,
        cam_type=capture.DEFAULT_CAM_TYPE,
        cam_id=capture.DEFAULT_CAM_ID,
        frames=capture.DEFAULT_FRAMES,
        settle_s=capture.DEFAULT_SETTLE_S,
        execute=False,
        self_test=False,
    )


def test_frozen_functions_and_patterns_are_reused() -> None:
    assert capture.flat_signal_measurement is s1.flat_signal_measurement
    assert capture.find_spots is axis_map.find_spots
    assert capture.grating_phase is geometry.grating_phase
    assert capture.phase_to_gray is geometry.phase_to_gray
    assert set(capture.PATTERN_BY_ID) == set(capture.CAPTURE_IDS)
    assert len(capture.CAPTURE_IDS) == 3
    for pattern_id in (capture.X_GRATING_ID, capture.Y_GRATING_ID):
        spec = capture.PATTERN_BY_ID[pattern_id]
        assert int(spec["two_pi_gray_scale"]) == 993
        assert int(spec["period_px"]) == 128
    assert capture.PATTERN_BY_ID[capture.FLAT_ID]["kind"] == "flat"
    for pattern_id in capture.CAPTURE_IDS:
        spec = capture.PATTERN_BY_ID[pattern_id]
        record = next(
            item
            for item in capture.protocol_patterns()
            if item["id"] == pattern_id
        )
        phase = capture.build_phase(spec, capture.DEFAULT_PANEL_SHAPE)
        gray = capture.materialize_gray(spec, capture.DEFAULT_PANEL_SHAPE)
        assert capture.digest(phase) == record["phase_sha256"]
        assert capture.digest(gray) == record["gray_sha256"]


@pytest.mark.parametrize(
    ("axis", "pattern_id", "axis_name", "component"),
    [
        (1, capture.X_GRATING_ID, "col", "dcol_px"),
        (0, capture.Y_GRATING_ID, "row", "drow_px"),
    ],
)
def test_synthetic_plus_minus_displacement(
    axis: int, pattern_id: str, axis_name: str, component: str
) -> None:
    frame = capture._synthetic_grating_frame(ZERO, PITCH, axis)
    gate, _ = capture.frame_signal(frame, float(capture.FULL_SCALE_REQUIRED))
    assert gate["has_signal"] is True
    pair = capture.measure_grating_pair(frame, ZERO, pattern_id)
    assert pair["found"] is True
    assert pair["reason"] is None
    assert pair["dominant_ccd_axis"] == axis_name
    assert abs(float(pair["pitch_px"]) - PITCH) < 1.0
    assert abs(float(pair["separation_px"]) - 2.0 * PITCH) < 2.0
    displacements = sorted(float(order[component]) for order in pair["orders"])
    assert abs(displacements[0] + PITCH) < 1.0
    assert abs(displacements[1] - PITCH) < 1.0
    assert len(pair["spots"]) >= 3
    if pattern_id == capture.X_GRATING_ID:
        relation = capture.frozen_window_relation(pair, pattern_id)
        assert relation["applicable"] is True
        assert relation["col_window_px"] == list(FROZEN_2_COL_WINDOW_PX)
        assert relation["criterion_2_evaluated"] is False
        assert relation["descriptive_within_window"] is True  # 95 px inside frozen 92-97
    else:
        relation = capture.frozen_window_relation(pair, pattern_id)
        assert relation["applicable"] is False
        assert relation["criterion_2_evaluated"] is False


def test_no_signal_saturation_and_missing_orders_fail_closed() -> None:
    blank = np.zeros((61, 61), dtype=np.uint16)
    gate, measured = capture.frame_signal(blank, float(capture.FULL_SCALE_REQUIRED))
    assert gate["has_signal"] is False
    assert gate["raw_peak_counts"] == 0
    assert capture.zero_order_from_flat(measured, gate) is None

    saturated = np.full((61, 61), 4000, dtype=np.uint16)
    sat_gate, _ = capture.frame_signal(saturated, float(capture.FULL_SCALE_REQUIRED))
    assert sat_gate["saturated"] is True
    assert sat_gate["has_signal"] is False

    no_spots = capture.measure_grating_pair(blank, ZERO, capture.X_GRATING_ID)
    assert no_spots["found"] is False
    assert no_spots["reason"] == "no_spots"
    assert no_spots["pitch_px"] is None
    assert no_spots["orders"] == []
    assert no_spots["zero_order_px"] is None

    one_spot = capture._synthetic_grating_frame(ZERO, PITCH, 1, with_orders=False)
    missing = capture.measure_grating_pair(one_spot, ZERO, capture.X_GRATING_ID)
    assert missing["found"] is False
    assert missing["pitch_px"] is None
    assert missing["orders"] == []

    relation = capture.frozen_window_relation(missing, capture.X_GRATING_ID)
    assert relation["descriptive_within_window"] is None
    assert relation["criterion_2_evaluated"] is False


def test_plan_is_deterministic_and_does_not_touch_hardware(tmp_path: Path) -> None:
    args = _plan_args(tmp_path / "planned")
    plan_a = capture.build_plan(args)
    plan_b = capture.build_plan(args)
    assert json.dumps(plan_a, sort_keys=True) == json.dumps(plan_b, sort_keys=True)
    assert "timestamp" not in json.dumps(plan_a)
    assert plan_a["capture_order"] == list(capture.CAPTURE_IDS)
    assert [record["id"] for record in plan_a["patterns"]] == list(capture.CAPTURE_IDS)
    assert plan_a["frames_per_pattern"] == 1
    assert plan_a["focus"]["compensation_applied"] is False
    assert plan_a["frozen_window"]["criterion_2_evaluated"] is False
    assert plan_a["signal_gate"]["reused"] is True
    assert "scale1024" not in json.dumps(plan_a)
    assert "blaze2d" not in json.dumps(plan_a)
    assert "ladder" not in json.dumps(plan_a)


def test_cli_plan_mode_never_creates_output_or_opens_devices(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "planned"
    assert capture.main(["--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert json.loads(printed)["schema"] == capture.PLAN_SCHEMA
    assert not out.exists()

    out.mkdir()
    with pytest.raises(SystemExit) as exc:
        capture.main(["--out", str(out), "--execute"])
    assert exc.value.code == 2


def test_script_self_test_entry_point(capsys: pytest.CaptureFixture[str]) -> None:
    assert capture.self_test() == 0
    assert json.loads(capsys.readouterr().out)["self_test"] == "passed"
