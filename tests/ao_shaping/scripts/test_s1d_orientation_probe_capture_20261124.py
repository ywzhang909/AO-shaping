# -*- coding: utf-8 -*-
"""Offline tests for the preregistered S1d orientation-probe capture (§12.14).

No hardware is touched and no output directory is created: every assertion runs
on the offline plan, the materialized phase/gray arrays and the pure summary
builder only.  The script is never run with ``--execute`` here.
"""

from __future__ import annotations

import argparse
import fnmatch
import importlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT = importlib.import_module(
    "scripts.slm.s1d_orientation_probe_capture_20261124"
)
GEOMETRY = importlib.import_module(
    "scripts.slm.issue247_geometry_capture_20261120"
)
CONTENT = importlib.import_module(
    "scripts.slm.content_axis_2x2_capture_20261124"
)
SOLVER = importlib.import_module("scripts.slm.issue247_axis_map_solve_20261121")

SHAPE = SCRIPT.DEFAULT_PANEL_SHAPE

FROZEN_ORDER: tuple[str, ...] = (
    "flat",
    "grating_x_p128",
    "grating_y_p128",
    "grating_x_p256",
    "grating_y_p256",
    "ladder_x",
    "ladder_y",
    "blaze2d_px128_py192_sx+1_sy+1",
    "blaze2d_px128_py192_sx+1_sy-1",
    "blaze2d_px192_py128_sx+1_sy+1",
    "blaze2d_px192_py128_sx+1_sy-1",
    "content_a128",
    "content_a256",
    "flat_post",
)

FROZEN_BLAZE2D: tuple[str, ...] = (
    "blaze2d_px128_py192_sx+1_sy+1",
    "blaze2d_px128_py192_sx+1_sy-1",
    "blaze2d_px192_py128_sx+1_sy+1",
    "blaze2d_px192_py128_sx+1_sy-1",
)


def _spec(pattern_id: str) -> dict[str, Any]:
    for spec in SCRIPT.PATTERN_SPECS:
        if spec["pattern_id"] == pattern_id:
            return spec
    raise AssertionError(f"pattern_id {pattern_id!r} not in frozen pattern specs")


def _args(**overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "out": None,
        "execute": False,
        "wait_until_idle": 1800.0,
        "exposure_ms": 0.087,
        "settle_s": 0.8,
        "frames_per_pattern": 2,
        "slm_number": 1,
        "cam_id": 0,
        "wavelength_nm": 1064.0,
        "saturation_fraction": 0.9,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_frozen_pattern_order_and_ids() -> None:
    assert SCRIPT.CAPTURE_ORDER == FROZEN_ORDER
    assert len(SCRIPT.PATTERN_SPECS) == 14
    assert [spec["pattern_id"] for spec in SCRIPT.PATTERN_SPECS] == list(
        FROZEN_ORDER
    )
    # No duplicate ids: the solver keys frames by pattern id.
    assert len(set(SCRIPT.CAPTURE_ORDER)) == 14


def test_frozen_hardware_and_encoding_constants() -> None:
    assert SHAPE == (1200, 1920)
    assert SCRIPT.ENCODING_MAX_GRAY == 1023
    assert SCRIPT.DEFAULT_FRAMES_PER_PATTERN == 2
    assert SCRIPT.DEFAULT_EXPOSURE_MS == 0.087
    assert SCRIPT.DEFAULT_SETTLE_S == 0.8
    assert SCRIPT.DEFAULT_WAIT_IDLE_S == 1800.0
    assert SCRIPT.DEFAULT_SLM_NUMBER == 1
    assert SCRIPT.DEFAULT_CAM_ID == 0
    assert SCRIPT.DEFAULT_WAVELENGTH_NM == 1064.0
    assert SCRIPT.DEFAULT_SATURATION_FRACTION == 0.9
    assert SCRIPT.BIT_DEPTH_REQUIRED == 12
    assert SCRIPT.FULL_SCALE_REQUIRED == 4095
    assert SCRIPT.FOCUS_D_M1 == 0.0
    assert SCRIPT.NATIVE_CCD_SHAPE == (1944, 2592)
    assert SCRIPT.FLAT_FIRST_MIN_PEAK == 100
    for spec in SCRIPT.PATTERN_SPECS:
        assert spec["encoding_max_gray"] == 1023


def test_builder_arguments_match_direct_invocation() -> None:
    """Every pattern must be bit-identical to the frozen builder call."""
    for spec in SCRIPT.PATTERN_SPECS:
        kind = spec["kind"]
        if kind == "flat":
            expected = GEOMETRY.flat_phase(SHAPE)
        elif kind == "grating":
            expected = GEOMETRY.grating_phase(
                SHAPE, period_px=spec["period_px"], axis=spec["axis"]
            )
        elif kind == "ladder":
            expected = GEOMETRY.ladder_phase(
                SHAPE, axis=spec["axis"], sign=spec["sign"]
            )
        elif kind == "blaze2d":
            expected = GEOMETRY.blaze2d_phase(
                SHAPE,
                period_x_px=spec["period_x_px"],
                period_y_px=spec["period_y_px"],
                sign_x=spec["sign_x"],
                sign_y=spec["sign_y"],
            )
        elif kind == "content":
            expected = CONTENT.make_phase(
                spec["content"], spec["period_px"], SHAPE, spec["d_m1"]
            )
        else:  # pragma: no cover - guards the frozen kind vocabulary
            raise AssertionError(f"unexpected kind {kind!r}")
        ours = SCRIPT.build_phase(spec, SHAPE)
        assert ours.shape == SHAPE
        assert np.array_equal(ours, expected), spec["pattern_id"]
        assert SCRIPT.digest(ours) == SCRIPT.digest(expected)


def test_frozen_grating_axes_and_periods() -> None:
    assert _spec("grating_x_p128")["axis"] == 1
    assert _spec("grating_x_p128")["period_px"] == 128
    assert _spec("grating_y_p128")["axis"] == 0
    assert _spec("grating_y_p128")["period_px"] == 128
    assert _spec("grating_x_p256")["axis"] == 1
    assert _spec("grating_x_p256")["period_px"] == 256
    assert _spec("grating_y_p256")["axis"] == 0
    assert _spec("grating_y_p256")["period_px"] == 256
    assert _spec("ladder_x")["axis"] == 1
    assert _spec("ladder_x")["sign"] == 1
    assert _spec("ladder_y")["axis"] == 0
    assert _spec("ladder_y")["sign"] == 1
    for pattern_id, px, py, sx, sy in (
        (FROZEN_BLAZE2D[0], 128.0, 192.0, 1, 1),
        (FROZEN_BLAZE2D[1], 128.0, 192.0, 1, -1),
        (FROZEN_BLAZE2D[2], 192.0, 128.0, 1, 1),
        (FROZEN_BLAZE2D[3], 192.0, 128.0, 1, -1),
    ):
        spec = _spec(pattern_id)
        assert spec["kind"] == "blaze2d"
        assert spec["period_x_px"] == px
        assert spec["period_y_px"] == py
        assert spec["sign_x"] == sx
        assert spec["sign_y"] == sy


def test_content_a_phases_bit_identical_to_make_phase() -> None:
    for period in (128, 256):
        spec = _spec(f"content_a{period}")
        ours = SCRIPT.build_phase(spec, SHAPE)
        theirs = CONTENT.make_phase("A", period, SHAPE, 0.0)
        assert np.array_equal(ours, theirs)
        assert SCRIPT.digest(ours) == SCRIPT.digest(theirs)


def test_flat_patterns_are_zero_phase_and_zero_gray() -> None:
    for pattern_id in ("flat", "flat_post"):
        spec = _spec(pattern_id)
        phase = SCRIPT.build_phase(spec, SHAPE)
        assert np.count_nonzero(phase) == 0
        gray = SCRIPT.build_gray(spec, SHAPE)
        assert np.count_nonzero(gray) == 0


def test_grating_gray_is_binary_at_encoding_1023() -> None:
    for pattern_id in (
        "grating_x_p128",
        "grating_y_p128",
        "grating_x_p256",
        "grating_y_p256",
    ):
        gray = SCRIPT.build_gray(_spec(pattern_id), SHAPE)
        assert gray.dtype == np.uint16
        assert gray.shape == SHAPE
        assert set(int(v) for v in np.unique(gray)) == {0, 512}


def test_gray_dtype_and_range_for_all_patterns() -> None:
    for spec in SCRIPT.PATTERN_SPECS:
        gray = SCRIPT.build_gray(spec, SHAPE)
        assert gray.shape == SHAPE
        assert gray.dtype == np.uint16
        assert int(gray.max()) <= SCRIPT.ENCODING_MAX_GRAY
        assert int(gray.min()) >= 0


def test_frame_plan_is_28_frames_in_frozen_order() -> None:
    frames = SCRIPT.build_frame_plan(2)
    assert len(frames) == 28
    assert [frame["index"] for frame in frames] == list(range(28))
    for order, pattern_id in enumerate(FROZEN_ORDER):
        pair = frames[2 * order : 2 * order + 2]
        assert [frame["pattern_id"] for frame in pair] == [pattern_id] * 2
        assert [frame["frame_in_pattern"] for frame in pair] == [0, 1]
        assert [frame["order"] for frame in pair] == [order, order]
        for frame in pair:
            assert frame["filename"] == SCRIPT.frame_filename(
                frame["index"], pattern_id
            )


def test_frame_filename_scheme_matches_solver_glob() -> None:
    assert SCRIPT.frame_filename(0, "flat") == "0000_flat.npy"
    assert SCRIPT.frame_filename(27, "flat_post") == "0027_flat_post.npy"
    for frame in SCRIPT.build_frame_plan(2):
        assert re.fullmatch(
            r"\d{4}_[A-Za-z0-9_+\-]+\.npy", frame["filename"]
        ), frame["filename"]
        # The frozen solver loads frames with ``frames.glob(f"*_{pattern}.npy")``.
        assert fnmatch.fnmatch(
            frame["filename"], f"*_{frame['pattern_id']}.npy"
        )


def test_plan_pattern_records_carry_digests_and_frame_counts() -> None:
    plan = SCRIPT.build_plan(_args())
    assert plan["schema"] == SCRIPT.PLAN_SCHEMA
    assert plan["capture_order"] == list(FROZEN_ORDER)
    assert plan["n_patterns"] == 14
    assert plan["n_frames_expected"] == 28
    assert plan["frames_per_pattern"] == 2
    assert len(plan["patterns"]) == 14
    assert len(plan["frames"]) == 28
    for order, record in enumerate(plan["patterns"]):
        spec = _spec(record["id"])
        assert record["order"] == order
        assert record["kind"] == spec["kind"]
        assert record["encoding_max_gray"] == 1023
        assert record["n_frames"] == 2
        assert record["panel_shape_rows_cols"] == [1200, 1920]
        phase = SCRIPT.build_phase(spec, SHAPE)
        gray = SCRIPT.phase_to_gray(phase, 1023)
        assert record["phase_sha256"] == SCRIPT.digest(phase)
        assert record["gray_sha256"] == SCRIPT.digest(gray)
        assert len(record["phase_sha256"]) == 64
        assert len(record["gray_sha256"]) == 64
    settings = plan["capture_settings"]
    assert settings["exposure_ms_requested"] == 0.087
    assert settings["settle_s"] == 0.8
    assert settings["focus_d_m1"] == 0.0
    assert settings["bit_depth"] == 12
    assert settings["full_scale"] == 4095
    assert settings["saturation_threshold"] == 0.9 * 4095
    assert settings["slm_number"] == 1
    assert settings["cam_id"] == 0
    assert settings["wavelength_nm"] == 1064.0


def test_plan_and_summary_expose_the_four_blaze2d_designs() -> None:
    assert SCRIPT.BLAZE2D_PATTERN_IDS == FROZEN_BLAZE2D
    plan = SCRIPT.build_plan(_args())
    assert plan["blaze2d_designs"] == list(FROZEN_BLAZE2D)
    solver_ids = {pid for pid, _, _, _ in SOLVER.PROBES}
    assert set(FROZEN_BLAZE2D) <= solver_ids
    assert set(FROZEN_BLAZE2D) == set(SOLVER.BLAZE2D_FALLBACK_IDS)
    # Every non-content probe id must be known to the frozen solver.
    for pattern_id in FROZEN_ORDER:
        if not pattern_id.startswith(("content_", "flat")):
            assert pattern_id in solver_ids


def _synthetic_frame(index: int, pattern_id: str) -> dict[str, Any]:
    return {
        "index": index,
        "pattern_id": pattern_id,
        "filename": SCRIPT.frame_filename(index, pattern_id),
        "frame_sha256": "ab" * 32,
        "peak": 1234,
        "median": 17.5,
        "saturated_count": 0,
        "utc": "2026-11-24T00:00:00+00:00",
        "exposure_ms": 0.087,
        "bit_depth": 12,
        "full_scale": 4095,
    }


def test_summary_builder_matches_solver_contract() -> None:
    plan = SCRIPT.build_plan(_args())
    frame_records: list[dict[str, Any]] = []
    per_pattern: dict[str, list[dict[str, Any]]] = {}
    index = 0
    for pattern_id in FROZEN_ORDER:
        per_pattern[pattern_id] = []
        for _ in range(2):
            row = _synthetic_frame(index, pattern_id)
            frame_records.append(row)
            per_pattern[pattern_id].append(row)
            index += 1
    release = {
        "slm_closed": True,
        "camera_closed": True,
        "lock_released": True,
        "closed_normally": True,
        "error": None,
        "elapsed_s": 1.0,
    }
    summary = SCRIPT.build_summary(
        plan,
        run={
            "run_id": "s1d_test",
            "capture_dir": "D:/nowhere/s1d_test",
            "started_at_utc": "2026-11-24T00:00:00+00:00",
            "status": "ok",
            "error": None,
        },
        device={
            "exposure_ms": 0.087,
            "bit_depth": 12,
            "full_scale": 4095,
            "settle_s": 0.8,
            "slm_number": 1,
            "drive_board_id": "board-0",
            "wavelength_nm": 1064.0,
            "camera": {"cam_type": "daheng", "cam_id": 0, "bit_depth": 12},
        },
        release=release,
        frame_records=frame_records,
        per_pattern_frames=per_pattern,
    )
    assert summary["schema"] == SCRIPT.SUMMARY_SCHEMA
    assert summary["run"]["run_id"] == "s1d_test"
    assert summary["status"] == "ok"
    assert summary["n_frames"] == 28
    assert summary["expected_frames"] == 28
    assert summary["blaze2d_designs"] == list(FROZEN_BLAZE2D)
    assert summary["exposure_ms"] == 0.087
    assert summary["bit_depth"] == 12
    assert summary["full_scale"] == 4095
    assert summary["settle_s"] == 0.8
    assert summary["focus_d_m1"] == 0.0
    assert summary["slm_number"] == 1
    assert summary["wavelength_nm"] == 1064.0
    assert summary["camera"]["cam_id"] == 0
    assert summary["closed_normally"] is True
    assert [p["id"] for p in summary["patterns"]] == list(FROZEN_ORDER)
    for pattern in summary["patterns"]:
        assert pattern["n_frames"] == 2
        assert pattern["n_frames_planned"] == 2
        assert len(pattern["phase_sha256"]) == 64
        assert len(pattern["gray_sha256"]) == 64
        for frame in pattern["frames"]:
            assert set(frame) == {
                "index",
                "filename",
                "frame_sha256",
                "peak",
                "median",
                "saturated_count",
                "utc",
            }
            assert fnmatch.fnmatch(
                frame["filename"], f"*_{pattern['id']}.npy"
            )
    # Round-trips as strict JSON (no NaN/Inf), as written by write_json.
    json.dumps(summary, allow_nan=False)
    # The frozen solver discovers the design list at the top level.
    assert SOLVER._collect_blaze2d_ids(summary) == list(FROZEN_BLAZE2D)


def test_plan_only_cli_returns_zero_and_creates_no_dir(tmp_path: Path) -> None:
    out_dir = tmp_path / "never_created"
    result = subprocess.run(
        [
            sys.executable,
            str(Path(SCRIPT.__file__)),
            "--out",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        cwd=Path(SCRIPT.__file__).resolve().parents[2],
    )
    assert result.returncode == 0, result.stderr
    assert f'"schema": "{SCRIPT.PLAN_SCHEMA}"' in result.stdout
    plan = json.loads(result.stdout)
    assert plan["capture_order"] == list(FROZEN_ORDER)
    assert plan["n_frames_expected"] == 28
    assert not out_dir.exists()
    assert not (out_dir / "frames").exists()


def test_execute_without_out_is_rejected_before_any_device_use() -> None:
    result = subprocess.run(
        [sys.executable, str(Path(SCRIPT.__file__)), "--execute"],
        capture_output=True,
        text=True,
        cwd=Path(SCRIPT.__file__).resolve().parents[2],
    )
    assert result.returncode == 2
    assert "--out is required with --execute" in result.stderr
