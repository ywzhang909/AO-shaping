# -*- coding: utf-8 -*-
"""离线门禁: S1f 随机抖动归因采集脚本 (2026-11-27).

断言:
- 计划模式只读: 不建目录、不取锁、不 import 驱动;
- 结构: 5 段 / 96 帧 / 28 次写屏, 变体 A 写 1 次采 24 帧, 变体 B 写 24 次各采 1 帧,
  flat 头/中/尾各写 1 次采 16 帧, pattern_id 命名与 arm 映射;
- H 灰度字节经 ``ascontiguousarray + tobytes(order="A")`` 的 sha256 等于
  run_05_postfix 的 ``7276d9da...``;
- plan.json 顶层字段与 ``arm_layout`` 结构。

不打开设备、不取锁、不建输出目录。冻结脚本 / 冻结测试的 blob 不在本文件范围内。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = importlib.import_module(
    "scripts.slm.s1f_jitter_attribution_capture_20261127"
)
REPO_ROOT = Path(__file__).resolve().parents[3]

# 独立转写自预注册 §3.1/§3.2, 不从 SCRIPT 导入.
PANEL_SHAPE = (1200, 1920)
NATIVE_CCD_SHAPE = (1944, 2592)
ENCODING_MAX_GRAY = 1023
GRATING_PERIOD_PX = 128
BIT_DEPTH = 12
FULL_SCALE = 4095
ARM_A_FRAMES = 24
ARM_B_WRITES = 24
FLAT_FRAMES = 16
SETTLE_S = 0.8
DEFAULT_EXPOSURE_MS = 0.05
SATURATION_FRACTION = 0.9
DEFAULT_WAIT_IDLE_S = 1800.0
DEFAULT_SLM_NUMBER = 1
DEFAULT_CAM_ID = 0
FROZEN_H_PHYSICAL_SHA256 = (
    "7276d9da3b2a5ea536d56ea944adcfdcb38c98ad1dd01c1433d1b2219ac8218e"
)

ARM_A_IDS = [f"s1f_a{i:02d}" for i in range(1, ARM_A_FRAMES + 1)]
ARM_B_IDS = [f"s1f_b{i:02d}" for i in range(1, ARM_B_WRITES + 1)]

EXPECTED_SEGMENTS = [
    ("flat_head", "flat_drift", "flat", 1, FLAT_FRAMES),
    ("arm_a", "same_write", "H", 1, ARM_A_FRAMES),
    ("flat_mid", "flat_drift", "flat", 1, FLAT_FRAMES),
    ("arm_b", "repeat_write", "H", ARM_B_WRITES, ARM_B_WRITES),
    ("flat_tail", "flat_drift", "flat", 1, FLAT_FRAMES),
]
EXPECTED_PLAN_KEYS = {
    "schema",
    "task",
    "design_source",
    "arm_layout",
    "n_frames_expected",
    "n_writes_expected",
    "frozen_h_physical_sha256",
    "expected_digests",
    "constants",
    "encoding",
    "capture_settings",
    "display_topology",
    "frames",
    "phase_builder_source",
}
EXPECTED_LAYOUT_KEYS = {
    "segment",
    "id",
    "arm",
    "array",
    "writes",
    "frames",
    "pattern_ids",
}


def test_module_constants_match_prereg() -> None:
    assert SCRIPT.PANEL_SHAPE == PANEL_SHAPE
    assert SCRIPT.NATIVE_CCD_SHAPE == NATIVE_CCD_SHAPE
    assert SCRIPT.ENCODING_MAX_GRAY == ENCODING_MAX_GRAY
    assert SCRIPT.GRATING_PERIOD_PX == GRATING_PERIOD_PX
    assert SCRIPT.BIT_DEPTH == BIT_DEPTH
    assert SCRIPT.FULL_SCALE == FULL_SCALE
    assert SCRIPT.ARM_A_FRAMES == ARM_A_FRAMES
    assert SCRIPT.ARM_B_WRITES == ARM_B_WRITES
    assert SCRIPT.FLAT_FRAMES == FLAT_FRAMES
    assert SCRIPT.SETTLE_S == SETTLE_S
    assert SCRIPT.DEFAULT_EXPOSURE_MS == DEFAULT_EXPOSURE_MS
    assert SCRIPT.SATURATION_FRACTION == SATURATION_FRACTION
    assert SCRIPT.DEFAULT_WAIT_IDLE_S == DEFAULT_WAIT_IDLE_S
    assert SCRIPT.DEFAULT_SLM_NUMBER == DEFAULT_SLM_NUMBER
    assert SCRIPT.DEFAULT_CAM_ID == DEFAULT_CAM_ID
    assert SCRIPT.FROZEN_H_PHYSICAL_SHA256 == FROZEN_H_PHYSICAL_SHA256


def test_arm_layout_segment_order_and_arm_mapping() -> None:
    layout = SCRIPT.build_arm_layout()
    assert len(layout) == 5
    observed = [
        (
            segment["id"],
            segment["arm"],
            segment["array"],
            segment["writes"],
            segment["frames"],
        )
        for segment in layout
    ]
    assert observed == EXPECTED_SEGMENTS
    assert [segment["segment"] for segment in layout] == [1, 2, 3, 4, 5]
    for segment in layout:
        assert set(segment) == EXPECTED_LAYOUT_KEYS
        assert len(segment["pattern_ids"]) == segment["frames"]


def test_arm_pattern_id_naming_and_ranges() -> None:
    layout = {segment["id"]: segment for segment in SCRIPT.build_arm_layout()}

    assert layout["flat_head"]["pattern_ids"] == ["flat_head"] * FLAT_FRAMES
    assert layout["flat_mid"]["pattern_ids"] == ["flat_mid"] * FLAT_FRAMES
    assert layout["flat_tail"]["pattern_ids"] == ["flat_tail"] * FLAT_FRAMES
    assert layout["arm_a"]["pattern_ids"] == ARM_A_IDS
    assert layout["arm_b"]["pattern_ids"] == ARM_B_IDS

    # flat 头/中/尾各写 1 次; 变体 A 只写 1 次; 变体 B 每帧写 1 次.
    assert layout["flat_head"]["writes"] == 1
    assert layout["flat_mid"]["writes"] == 1
    assert layout["flat_tail"]["writes"] == 1
    assert layout["arm_a"]["writes"] == 1
    assert layout["arm_b"]["writes"] == ARM_B_WRITES == len(ARM_B_IDS)


def test_frame_schedule_totals_and_write_indices() -> None:
    frames = SCRIPT.build_frame_plan()
    assert len(frames) == FLAT_FRAMES * 3 + ARM_A_FRAMES + ARM_B_WRITES == 96
    assert [entry["index"] for entry in frames] == list(range(96))

    head = frames[:16]
    arm_a = frames[16:40]
    mid = frames[40:56]
    arm_b = frames[56:80]
    tail = frames[80:]

    assert {entry["pattern_id"] for entry in head} == {"flat_head"}
    assert [entry["pattern_id"] for entry in arm_a] == ARM_A_IDS
    assert {entry["pattern_id"] for entry in mid} == {"flat_mid"}
    assert [entry["pattern_id"] for entry in arm_b] == ARM_B_IDS
    assert {entry["pattern_id"] for entry in tail} == {"flat_tail"}

    assert {entry["arm"] for entry in head} == {"flat_drift"}
    assert {entry["arm"] for entry in arm_a} == {"same_write"}
    assert {entry["arm"] for entry in mid} == {"flat_drift"}
    assert {entry["arm"] for entry in arm_b} == {"repeat_write"}
    assert {entry["arm"] for entry in tail} == {"flat_drift"}

    assert {entry["array"] for entry in head} == {"flat"}
    assert {entry["array"] for entry in arm_a} == {"H"}
    assert {entry["array"] for entry in arm_b} == {"H"}

    # 变体 A 的 24 帧共享同一次写屏; 变体 B 每帧各自一次写屏.
    assert {entry["write_index"] for entry in arm_a} == {1}
    assert [entry["write_index"] for entry in arm_b] == list(range(3, 27))
    assert {entry["write_index"] for entry in head} == {0}
    assert {entry["write_index"] for entry in mid} == {2}
    assert {entry["write_index"] for entry in tail} == {27}

    assert sum(1 for entry in frames if entry["write_before"]) == 28
    assert [entry["write_before"] for entry in arm_a] == [True] + [False] * 23
    assert all(entry["write_before"] for entry in arm_b)
    assert [entry["frame_in_arm"] for entry in arm_a] == list(range(24))
    assert [entry["frame_in_arm"] for entry in arm_b] == list(range(24))


def test_build_plan_top_level_and_arm_layout_structure() -> None:
    plan = SCRIPT.build_plan()
    assert set(plan) == EXPECTED_PLAN_KEYS
    assert plan["schema"] == "issue247-s1f-jitter-attribution-capture/v1"
    assert plan["n_frames_expected"] == 96
    assert plan["n_writes_expected"] == 28
    assert plan["frozen_h_physical_sha256"] == FROZEN_H_PHYSICAL_SHA256
    assert plan["display_topology"]["status"] == "not_probed"
    assert plan["capture_settings"]["laser_operated"] is False
    assert plan["capture_settings"]["dark_frames"] is False
    assert len(plan["frames"]) == 96

    for segment in plan["arm_layout"]:
        assert set(segment) == EXPECTED_LAYOUT_KEYS
        assert segment["frames"] == len(segment["pattern_ids"])

    h_digests = plan["expected_digests"]["H"]
    assert h_digests["gray_sha256"] == FROZEN_H_PHYSICAL_SHA256


def test_h_gray_bytes_match_run_05_postfix() -> None:
    gray = SCRIPT.build_gray("H")
    assert gray.dtype == np.uint16
    assert gray.shape == PANEL_SHAPE
    observed = hashlib.sha256(
        np.ascontiguousarray(gray).tobytes(order="A")
    ).hexdigest()
    assert observed == FROZEN_H_PHYSICAL_SHA256
    assert SCRIPT.physical_digest(gray) == FROZEN_H_PHYSICAL_SHA256
    SCRIPT.assert_frozen_h_bytes(gray)

    drifted = np.ascontiguousarray(gray.copy())
    drifted[0, 0] = np.uint16(int(drifted[0, 0]) + 1)
    with pytest.raises(RuntimeError):
        SCRIPT.assert_frozen_h_bytes(drifted)


def test_plan_mode_is_read_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _forbidden_lock(wait_s: float) -> None:
        raise AssertionError("plan mode must not acquire the device lock")

    monkeypatch.setattr(SCRIPT, "acquire_device_lock", _forbidden_lock)
    monkeypatch.chdir(tmp_path)

    assert SCRIPT.main([]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["n_frames_expected"] == 96
    assert list(tmp_path.iterdir()) == []
    assert "capture_journal.jsonl" not in capsys.readouterr().out


def test_plan_mode_imports_no_driver() -> None:
    code = "\n".join(
        [
            "import contextlib, importlib, io, json, sys",
            (
                "m = importlib.import_module("
                "'scripts.slm.s1f_jitter_attribution_capture_20261127')"
            ),
            "buffer = io.StringIO()",
            "with contextlib.redirect_stdout(buffer):",
            "    rc = m.main([])",
            "assert rc == 0, rc",
            "assert json.loads(buffer.getvalue())['n_frames_expected'] == 96",
            "print(json.dumps(sorted(k for k in sys.modules "
            "if k.startswith('ao_shaping.drivers'))))",
        ]
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    loaded = json.loads(completed.stdout.strip().splitlines()[-1])
    assert loaded == []


def test_execute_without_out_errors_before_lock() -> None:
    with pytest.raises(SystemExit) as excinfo:
        SCRIPT.main(["--execute"])
    assert excinfo.value.code == 2


def test_execute_lock_failure_returns_2_and_creates_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(SCRIPT, "acquire_device_lock", lambda wait_s: None)
    run_dir = tmp_path / "run_01"
    args = argparse.Namespace(
        out=run_dir,
        execute=True,
        exposure_ms=DEFAULT_EXPOSURE_MS,
        wait_until_idle=0.0,
        slm_number=DEFAULT_SLM_NUMBER,
        cam_id=DEFAULT_CAM_ID,
    )
    assert SCRIPT.execute(args, SCRIPT.build_plan()) == 2
    assert not run_dir.exists()

# ---------------------------------------------------------------------------
# §9 数据质量预检（非归因判据）
# ---------------------------------------------------------------------------
LIGHT_MIN_PEAK_CONTRAST = 50.0


def test_light_preflight_rejects_no_light_session() -> None:
    frame = np.full(NATIVE_CCD_SHAPE, 15, dtype=np.uint16)
    frame[1, 460] = 41  # run_01 实测: 边缘噪声峰
    assert SCRIPT.LIGHT_MIN_PEAK_CONTRAST == LIGHT_MIN_PEAK_CONTRAST
    assert SCRIPT.light_contrast(frame) == pytest.approx(26.0)
    with pytest.raises(RuntimeError, match="no usable light"):
        SCRIPT.assert_light_level(frame)


def test_light_preflight_accepts_healthy_frame() -> None:
    frame = np.full(NATIVE_CCD_SHAPE, 15, dtype=np.uint16)
    frame[1105, 1227] = 2524  # S1e run_05_postfix flat 实测
    info = SCRIPT.assert_light_level(frame)
    assert info["passed"] is True
    assert info["contrast_px"] == pytest.approx(2509.0)
