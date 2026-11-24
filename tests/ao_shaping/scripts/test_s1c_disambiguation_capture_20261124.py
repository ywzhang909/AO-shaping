# -*- coding: utf-8 -*-
"""Offline tests for the preregistered S1c minimal-disambiguation capture script.

No hardware is touched: every assertion runs on the offline plan / phase /
gray arrays only.
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT = importlib.import_module(
    "scripts.slm.s1c_disambiguation_capture_20261124"
)
CONTENT = importlib.import_module(
    "scripts.slm.content_axis_2x2_capture_20261124"
)

SHAPE = SCRIPT.DEFAULT_PANEL_SHAPE


def _pattern_by_id(patterns: tuple[dict[str, object], ...], frame_id: str) -> dict:
    for pattern in patterns:
        if pattern["frame_id"] == frame_id:
            return pattern
    raise AssertionError(f"frame_id {frame_id!r} not in frozen frame specs")


def _gray_for(frame_id: str) -> np.ndarray:
    spec = _pattern_by_id(SCRIPT.FRAME_SPECS, frame_id)
    phase = SCRIPT.build_phase(spec, SHAPE)
    return SCRIPT.phase_to_gray(phase, int(spec["encoding_max_gray"]))


def test_frozen_frame_order_and_ids() -> None:
    assert SCRIPT.CAPTURE_ORDER == (
        "flat_pre",
        "gx128_e1023",
        "gx128_e993",
        "gy128_e1023",
        "blaze2d_px128_py192_pp",
        "ladder_x",
        "gx128_e1023_repeat",
        "flat_post",
    )
    assert len(SCRIPT.FRAME_SPECS) == 8
    assert [spec["frame_id"] for spec in SCRIPT.FRAME_SPECS] == list(
        SCRIPT.CAPTURE_ORDER
    )


def test_encoding_scales_per_frame() -> None:
    expected = {
        "flat_pre": SCRIPT.ENCODING_1023,
        "gx128_e1023": SCRIPT.ENCODING_1023,
        "gx128_e993": SCRIPT.ENCODING_993,
        "gy128_e1023": SCRIPT.ENCODING_1023,
        "blaze2d_px128_py192_pp": SCRIPT.ENCODING_1023,
        "ladder_x": SCRIPT.ENCODING_1023,
        "gx128_e1023_repeat": SCRIPT.ENCODING_1023,
        "flat_post": SCRIPT.ENCODING_1023,
    }
    for spec in SCRIPT.FRAME_SPECS:
        assert int(spec["encoding_max_gray"]) == expected[spec["frame_id"]]


def test_gx128_gray_unique_values() -> None:
    gray_1023 = _gray_for("gx128_e1023")
    gray_993 = _gray_for("gx128_e993")
    assert set(int(v) for v in np.unique(gray_1023)) == {0, 512}
    assert set(int(v) for v in np.unique(gray_993)) == {0, 497}
    assert gray_1023.dtype == np.uint16
    assert gray_993.dtype == np.uint16


def test_frame2_and_frame7_gray_arrays_byte_identical() -> None:
    gray_2 = _gray_for("gx128_e1023")
    gray_7 = _gray_for("gx128_e1023_repeat")
    assert np.array_equal(gray_2, gray_7)
    assert SCRIPT.digest(gray_2) == SCRIPT.digest(gray_7)


def test_gx128_phase_bit_identical_to_content_axis_A() -> None:
    ours = SCRIPT.build_phase(
        next(s for s in SCRIPT.FRAME_SPECS if s["frame_id"] == "gx128_e1023"),
        SHAPE,
    )
    theirs = CONTENT.make_phase("A", 128, SHAPE, d_m1=0.0)
    assert np.array_equal(ours, theirs)
    assert SCRIPT.digest(ours) == SCRIPT.digest(theirs)


def test_flat_frames_all_zero() -> None:
    for frame_id in ("flat_pre", "flat_post"):
        spec = next(s for s in SCRIPT.FRAME_SPECS if s["frame_id"] == frame_id)
        phase = SCRIPT.build_phase(spec, SHAPE)
        assert np.count_nonzero(phase) == 0
        gray = SCRIPT.phase_to_gray(phase, int(spec["encoding_max_gray"]))
        assert np.count_nonzero(gray) == 0


def test_blaze2d_and_ladder_parameters_are_frozen() -> None:
    blaze = next(
        s for s in SCRIPT.FRAME_SPECS if s["frame_id"] == "blaze2d_px128_py192_pp"
    )
    assert blaze["period_x_px"] == 128.0
    assert blaze["period_y_px"] == 192.0
    assert blaze["sign_x"] == 1
    assert blaze["sign_y"] == 1
    assert blaze["kind"] == "blaze2d"

    ladder = next(s for s in SCRIPT.FRAME_SPECS if s["frame_id"] == "ladder_x")
    assert ladder["axis"] == 1
    assert ladder["sign"] == 1
    assert ladder["kind"] == "ladder"

    # Materialized arrays match the frozen spec.
    phase = SCRIPT.build_phase(blaze, SHAPE)
    assert phase.shape == SHAPE
    assert phase.dtype == np.float64


def test_plan_patterns_digests_match_frames() -> None:
    args = argparse.Namespace(
        exposure_ms=0.087,
        settle_s=0.8,
        saturation_fraction=0.9,
        slm_number=1,
        cam_id=0,
        wavelength_nm=1064.0,
        wait_until_idle=1.0,
        out=None,
    )
    plan = SCRIPT.build_plan(args)
    patterns = plan["frames"]
    assert len(patterns) == 8
    for pattern in patterns:
        spec = _pattern_by_id(SCRIPT.FRAME_SPECS, pattern["frame_id"])
        phase = SCRIPT.build_phase(spec, SHAPE)
        gray = SCRIPT.phase_to_gray(phase, int(spec["encoding_max_gray"]))
        assert pattern["phase_sha256"] == SCRIPT.digest(phase)
        assert pattern["gray_sha256"] == SCRIPT.digest(gray)
        assert len(pattern["phase_sha256"]) == 64
        assert len(pattern["gray_sha256"]) == 64


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
    assert '"schema": "issue247-s1c-disambiguation-capture/v1"' in result.stdout
    assert not out_dir.exists()
