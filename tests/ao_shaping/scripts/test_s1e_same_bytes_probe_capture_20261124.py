# -*- coding: utf-8 -*-
"""Offline tests for the preregistered S1e same-bytes probe (§12.16).

Every assertion runs on the plan, pure helpers or materialized arrays: no device
is opened, no lock is taken and no output directory is created.  ``--execute``
is only checked through argparse's early rejection, never by calling execute().
"""

from __future__ import annotations

import argparse
import ast
import ctypes
import importlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

SCRIPT = importlib.import_module(
    "scripts.slm.s1e_same_bytes_probe_capture_20261124"
)
# Independently transcribed from §12.16.2 / §12.16.7, not imported from SCRIPT.
FROZEN_ORDER: tuple[str, ...] = (
    "flat",
    "s1e_a1",
    "s1e_a2",
    "s1e_b1",
    "s1e_a3",
    "s1e_b2",
    "s1e_a4",
    "flat_mid",
    "s1e_a5",
    "s1e_b3",
    "s1e_a6",
    "flat_post",
)
FROZEN_ARRAY_BY_ID = {
    "flat": "flat",
    "s1e_a1": "H",
    "s1e_a2": "H",
    "s1e_b1": "B",
    "s1e_a3": "H",
    "s1e_b2": "B",
    "s1e_a4": "H",
    "flat_mid": "flat",
    "s1e_a5": "H",
    "s1e_b3": "B",
    "s1e_a6": "H",
    "flat_post": "flat",
}
FROZEN_PREDECESSOR = {
    "s1e_a1": "flat",
    "s1e_a2": "H",
    "s1e_b1": "H",
    "s1e_a3": "B",
    "s1e_b2": "H",
    "s1e_a4": "B",
    "s1e_a5": "flat_mid",
    "s1e_b3": "H",
    "s1e_a6": "B",
}
FROZEN_DIGESTS = {
    "flat": (
        "11770da8db5cd320c64cb51890b2a12b4776b87c8a591e138eb722864b419d67",
        "cb228af4065a8be95006792a8199172366d0b63a4d39ce365f9591246861db16",
        "cb228af4065a8be95006792a8199172366d0b63a4d39ce365f9591246861db16",
    ),
    "H": (
        "d8f16e03d98e79491a2f714fc2f42ffff9e0c5f27861979aba4de8193c84ec0b",
        "7276d9da3b2a5ea536d56ea944adcfdcb38c98ad1dd01c1433d1b2219ac8218e",
        "b98be9a6696cd94dbb3fa966e22a7f86e52f52b16ae252d21f7aa07af4079c3a",
    ),
    "B": (
        "382100b02c8b07a68e995c70295e8c2ab34abf6cd05e5af8ec163067ff0eda3f",
        "c8d937cedb294339fa385f9ffa65cebd0ff3471ff62009cef42d7637bf590a73",
        "c8d937cedb294339fa385f9ffa65cebd0ff3471ff62009cef42d7637bf590a73",
    ),
}


def _args(**overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "out": None,
        "execute": False,
        "wait_until_idle": 1800.0,
        "exposure_ms": 0.087,
        "settle_s": 0.8,
        "frames_per_upload": 2,
        "slm_number": 1,
        "cam_id": 0,
        "wavelength_nm": 1064.0,
        "saturation_fraction": 0.9,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _physical_bytes(array: np.ndarray) -> bytes:
    """Exactly what driver.py:2660 ``ctypes.memmove`` copies to the panel."""
    buffer = (ctypes.c_ushort * array.size)()
    ctypes.memmove(buffer, array.ctypes.data, array.nbytes)
    return bytes(buffer)


def test_frozen_order_arrays_and_predecessors() -> None:
    assert SCRIPT.FROZEN_ORDER == FROZEN_ORDER
    assert SCRIPT.ARRAY_BY_ID == FROZEN_ARRAY_BY_ID
    assert SCRIPT.PREDECESSOR_BY_ID == FROZEN_PREDECESSOR
    assert len(set(FROZEN_ORDER)) == 12
    assert [i for i, pid in enumerate(FROZEN_ORDER) if SCRIPT.ARRAY_BY_ID[pid] == "H"] == [
        1, 2, 4, 6, 8, 10,
    ]
    assert [SCRIPT.PREDECESSOR_BY_ID[pid] for pid in ("s1e_a1", "s1e_a5", "s1e_a6")] == [
        "flat", "flat_mid", "B",
    ]


def test_pattern_records_carry_frozen_logical_and_physical_digests() -> None:
    records = SCRIPT.build_pattern_records()
    assert [record["id"] for record in records] == list(FROZEN_ORDER)
    for order, record in enumerate(records):
        array_id = record["array"]
        phase, gray, physical = FROZEN_DIGESTS[array_id]
        assert record["order"] == order
        assert record["array"] == array_id
        assert record["predecessor"] == FROZEN_PREDECESSOR.get(record["id"])
        assert record["phase_sha256"] == phase
        assert record["gray_sha256"] == gray
        assert record["gray_layout"]["physical_sha256"] == physical
        assert record["n_frames"] == 2


def test_generated_arrays_match_frozen_digests_and_rebuild_bitwise() -> None:
    for pattern_id, array_id in FROZEN_ARRAY_BY_ID.items():
        phase1 = SCRIPT.build_phase(pattern_id)
        gray1 = SCRIPT.build_gray(pattern_id)
        phase2 = SCRIPT.build_phase(pattern_id)
        gray2 = SCRIPT.build_gray(pattern_id)
        expected_phase, expected_gray, expected_physical = FROZEN_DIGESTS[array_id]
        assert np.array_equal(phase1, phase2)
        assert np.array_equal(gray1, gray2)
        assert SCRIPT.digest(phase1) == expected_phase
        assert SCRIPT.digest(gray1) == expected_gray
        assert SCRIPT.physical_digest(gray1) == expected_physical


def test_physical_digest_is_the_memmove_byte_stream_not_the_logical_one() -> None:
    h_gray = SCRIPT.build_gray("s1e_a1")
    b_gray = SCRIPT.build_gray("s1e_b1")
    assert _physical_bytes(h_gray) == h_gray.tobytes(order="A")
    assert _physical_bytes(b_gray) == b_gray.tobytes(order="A")
    # §12.16.7: H is F-contiguous, so logical C-order bytes differ from the
    # physical stream actually handed to the SDK.
    assert SCRIPT.digest(h_gray) != SCRIPT.physical_digest(h_gray)
    assert SCRIPT.digest(b_gray) == SCRIPT.physical_digest(b_gray)


def test_layout_record_exposes_strides_and_contiguity() -> None:
    h_layout = SCRIPT.layout_record(SCRIPT.build_gray("s1e_a1"))
    b_layout = SCRIPT.layout_record(SCRIPT.build_gray("s1e_b1"))
    assert h_layout["shape_rows_cols"] == [1200, 1920]
    assert h_layout["dtype"] == "uint16"
    assert h_layout["strides"] == [2, 2400]
    assert h_layout["c_contiguous"] is False
    assert h_layout["f_contiguous"] is True
    assert b_layout["strides"] == [3840, 2]
    assert b_layout["c_contiguous"] is True
    assert b_layout["f_contiguous"] is False
    for layout in (h_layout, b_layout):
        assert len(layout["physical_sha256"]) == 64


def test_frame_plan_is_24_frames_two_per_upload() -> None:
    frames = SCRIPT.build_frame_plan(2)
    assert len(frames) == 24
    assert [frame["index"] for frame in frames] == list(range(24))
    for upload_index, pattern_id in enumerate(FROZEN_ORDER):
        pair = frames[2 * upload_index : 2 * upload_index + 2]
        assert [frame["pattern_id"] for frame in pair] == [pattern_id] * 2
        assert [frame["upload_index"] for frame in pair] == [upload_index] * 2
        assert [frame["frame_in_upload"] for frame in pair] == [0, 1]
        assert [frame["filename"] for frame in pair] == [
            f"{2 * upload_index:04d}_{pattern_id}.npy",
            f"{2 * upload_index + 1:04d}_{pattern_id}.npy",
        ]


def test_plan_freezes_device_settings_and_criteria() -> None:
    plan = SCRIPT.build_plan(_args())
    assert plan["schema"] == SCRIPT.PLAN_SCHEMA
    assert plan["capture_order"] == list(FROZEN_ORDER)
    assert plan["uploads"] == 12
    assert plan["frames_per_upload"] == 2
    assert plan["n_frames_expected"] == 24
    assert plan["expected_digests"] == {
        array_id: {
            "phase_sha256": FROZEN_DIGESTS[array_id][0],
            "gray_sha256": FROZEN_DIGESTS[array_id][1],
            "physical_sha256": FROZEN_DIGESTS[array_id][2],
        }
        for array_id in ("flat", "H", "B")
    }
    settings = plan["capture_settings"]
    assert settings["slm_number"] == 1
    assert settings["cam_id"] == 0
    assert settings["wavelength_nm"] == 1064.0
    assert settings["video_mode"] == 0
    assert settings["exposure_ms_requested"] == 0.087
    assert settings["settle_s"] == 0.8
    assert settings["bit_depth"] == 12
    assert settings["full_scale"] == 4095
    assert settings["encoding_max_gray"] == 1023
    assert settings["focus_d_m1"] == 0.0
    assert settings["saturation_threshold"] == pytest.approx(0.9 * 4095)
    assert plan["focus_compensation"]["compensation_applied"] is False


def test_frozen_criteria_are_not_loosened() -> None:
    assert SCRIPT.FROZEN_CRITERIA == {
        "intra_upload_max_dist_px": 1.5,
        "D1": {"min_effective_uploads": 5, "max_pairwise_dist_px": 1.5},
        "D2": {
            "axis_ratio_min": 1.5,
            "col_dist_px": [90.0, 102.0],
            "row_dist_px": [52.0, 64.0],
        },
        "D4": {"weak_peak_fraction": 0.10},
        "D6": {"zero_exception_groups_required": True},
    }


def test_pure_verdict_helpers() -> None:
    assert SCRIPT.pairwise_max_distance([]) == 0.0
    assert SCRIPT.pairwise_max_distance([(0.0, 0.0), (3.0, 4.0)]) == 5.0
    assert SCRIPT.d1_verdict(1.5, 6) == "D-pass"
    assert SCRIPT.d1_verdict(1.51, 6) == "D-fail"
    assert SCRIPT.d1_verdict(0.5, 4) == "inconclusive"
    assert SCRIPT.axis_type(2.0, 95.6) == "col-type"
    assert SCRIPT.axis_type(57.0, 2.0) == "row-type"
    assert SCRIPT.axis_type(40.0, 40.0) == "other"
    assert SCRIPT.axis_type(2.0, 70.0) == "other"  # outside frozen D2 window
    assert SCRIPT.d6_zero_exception({1: {"col-type"}, 2: {"col-type"}}) is True
    assert SCRIPT.d6_zero_exception({1: {"col-type", "row-type"}}) is False
    assert SCRIPT.d6_zero_exception({}) is False


def test_plan_only_cli_creates_no_directory_and_touches_no_driver(
    tmp_path: Path,
) -> None:
    script = Path(SCRIPT.__file__).resolve()
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["schema"] == SCRIPT.PLAN_SCHEMA
    assert plan["capture_order"] == list(FROZEN_ORDER)
    assert plan["display_topology"]["status"] == "not_probed"
    assert list(tmp_path.iterdir()) == []


def test_execute_without_out_is_rejected_before_any_device_code(tmp_path: Path) -> None:
    script = Path(SCRIPT.__file__).resolve()
    result = subprocess.run(
        [sys.executable, str(script), "--execute"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--out" in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_build_summary_keeps_one_record_per_upload_and_frame() -> None:
    plan = SCRIPT.build_plan(_args())
    frames = SCRIPT.build_frame_plan(2)
    frame_records = [
        {
            "index": frame["index"],
            "upload_index": frame["upload_index"],
            "pattern_id": frame["pattern_id"],
            "filename": frame["filename"],
            "frame_sha256": "ab" * 32,
            "peak": 1000 + frame["index"],
            "median": 10.0,
            "saturated_count": 0,
            "utc": "2026-11-24T00:00:00Z",
        }
        for frame in frames
    ]
    upload_records = [
        {"pattern_id": pattern_id, "array": SCRIPT.ARRAY_BY_ID[pattern_id]}
        for pattern_id in FROZEN_ORDER
    ]
    summary = SCRIPT.build_summary(
        plan,
        run={
            "run_id": "run_01",
            "capture_dir": "run_01",
            "started_at_utc": "2026-11-24T00:00:00Z",
            "status": "ok",
            "error": None,
        },
        device={
            "exposure_ms": 0.087,
            "bit_depth": 12,
            "full_scale": 4095,
            "settle_s": 0.8,
            "slm_number": 1,
            "drive_board_id": "dummy",
            "wavelength_nm": 1064.0,
            "camera": {"cam_type": "daheng", "cam_id": 0},
        },
        release={
            "closed_normally": True,
            "elapsed_s": 1.0,
        },
        frame_records=frame_records,
        upload_records=upload_records,
    )
    assert summary["schema"] == SCRIPT.SUMMARY_SCHEMA
    assert summary["status"] == "ok"
    assert summary["n_frames"] == 24
    assert summary["n_uploads"] == 12
    assert summary["expected_frames"] == 24
    assert summary["closed_normally"] is True
    assert [pattern["id"] for pattern in summary["patterns"]] == list(FROZEN_ORDER)
    assert all(len(pattern["frames"]) == 2 for pattern in summary["patterns"])
    assert all(pattern["upload"] is not None for pattern in summary["patterns"])


def test_source_has_no_array_reorientation_or_laser_control() -> None:
    source = Path(SCRIPT.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "create_phase_from_array",
        "np.rot90",
        "np.flip",
        "np.transpose",
        ".transpose(",
        "cv2.resize",
        "np.resize",
        "power_cycle",
        "set_laser",
        "laser.on",
    ):
        assert forbidden not in source, forbidden


def test_driver_imports_are_confined_to_execute() -> None:
    tree = ast.parse(Path(SCRIPT.__file__).read_text(encoding="utf-8"))
    top_level_driver_imports: list[str] = []
    for node in tree.body:
        candidates: list[ast.AST] = []
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            candidates.append(node)
        for sub in candidates:
            text = ast.unparse(sub)
            if "ao_shaping.drivers" in text:
                top_level_driver_imports.append(text)
    assert top_level_driver_imports == []
    execute_fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "execute"
    )
    driver_imports = [
        ast.unparse(node)
        for node in ast.walk(execute_fn)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and "ao_shaping.drivers" in ast.unparse(node)
    ]
    assert any("create_camera" in text for text in driver_imports)
    assert any("Santec" in text for text in driver_imports)


def test_journal_source_carries_slot_and_layout_fields() -> None:
    source = Path(SCRIPT.__file__).read_text(encoding="utf-8")
    for field in (
        '"video_mode": 0',
        '"memory_number": memory_number',
        '"current_memory_slot_before_display"',
        '"displayed_slot_safe"',
        '"c_contiguous"',
        '"f_contiguous"',
        '"physical_sha256"',
        '"event": "upload"',
        '"event": "frame_saved"',
    ):
        assert field in source, field
    assert source.count("slm.open()") == 1
    assert source.count("slm.close()") == 1


def test_display_topology_probe_reports_unavailable_off_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(SCRIPT.sys, "platform", "linux")
    topology = SCRIPT.probe_display_topology()
    assert topology["status"] == "unavailable"
    assert "linux" in topology["reason"]
    assert topology.get("displays") is None


@pytest.mark.skipif(sys.platform != "win32", reason="Windows display enumeration")
def test_display_topology_probe_is_readable_or_explicitly_unavailable() -> None:
    topology = SCRIPT.probe_display_topology()
    assert topology["status"] in {"ok", "unavailable"}
    if topology["status"] == "ok":
        assert topology["displays"]
        for display in topology["displays"]:
            assert re.fullmatch(r"\d+", str(display["width_px"]))
            assert display["orientation_code"] in (0, 1, 2, 3)
    else:
        assert topology["reason"]
