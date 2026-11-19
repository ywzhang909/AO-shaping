"""Issue #247 离线分析：先估旋转再对齐，再做零级/目标判读。

合成一组 "#247-like" 采集：面板 +x/+y 斜坡让目标分别沿 CCD 上旋转后的
两轴移动。分析必须先从未旋转帧量出两矢量、估出修正角（不是写死 90°），
再旋转帧；轴对不可用时保留原始帧并记录原因。
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from ao_shaping.utils.image.rotation import (
    ROTATION_CONVENTION,
    rotate_image_about_center,
    rotated_canvas_shape,
)

ISSUE247 = importlib.import_module("scripts.slm.issue247_zero_order_stability_20261117")

SHAPE = (128, 160)
ZERO_XY = (0.5 * (SHAPE[1] - 1), 0.5 * (SHAPE[0] - 1))
SIGMA = 2.0
MAG = 14.0
DELTA_DEG = 27.0
ZERO_AMP = 800.0
TARGET_AMP = 600.0


def _spots_frame(spots: list[tuple[float, float, float]]) -> np.ndarray:
    yy, xx = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]
    frame = np.zeros(SHAPE, dtype=np.float64)
    for cx, cy, amp in spots:
        frame += amp * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * SIGMA**2)))
    return np.rint(frame).astype(np.uint16)


def _session(
    root: Path,
    *,
    delta_deg: float = DELTA_DEG,
    vectors: tuple[tuple[float, float], tuple[float, float]] | None = None,
    drop: str | None = None,
) -> tuple[Path, list[Path]]:
    if vectors is None:
        c, s = np.cos(np.deg2rad(delta_deg)), np.sin(np.deg2rad(delta_deg))
        vx = (MAG * c, MAG * s)
        vy = (-MAG * s, MAG * c)
    else:
        vx, vy = vectors

    zx, zy = ZERO_XY
    zero = (zx, zy, ZERO_AMP)

    def shifted(vec: tuple[float, float], sign: float) -> tuple[float, float, float]:
        return (zx + sign * vec[0], zy + sign * vec[1], TARGET_AMP)

    entries = [
        ("flat_pre", "flat", [zero]),
        ("flat_mid", "flat", [zero]),
        ("flat_post", "flat", [zero]),
        ("tilt_x_plus", "tilt", [zero, shifted(vx, +1.0)]),
        ("tilt_x_minus", "tilt", [zero, shifted(vx, -1.0)]),
        ("tilt_y_plus", "tilt", [zero, shifted(vy, +1.0)]),
        ("tilt_y_minus", "tilt", [zero, shifted(vy, -1.0)]),
    ]
    entries = [entry for entry in entries if entry[0] != drop]

    manifest: list[dict] = []
    files: list[Path] = []
    for idx, (label, kind, spots) in enumerate(entries):
        frame = _spots_frame(spots)
        name = f"{idx:02d}_{label}.npz"
        path = root / name
        np.savez(path, raw_frames=frame[np.newaxis, ...])
        files.append(path)
        manifest.append(
            {
                "label": label,
                "kind": kind,
                "file": name,
                "exposure_ms_readback": [1.0],
                "peak": int(frame.max()),
                "saturated_pixels": 0,
            }
        )
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root, files


def test_rotated_panel_axes_are_estimated_and_aligned(tmp_path: Path) -> None:
    session, files = _session(tmp_path)
    before = {path.name: path.read_bytes() for path in files}

    res = ISSUE247.analyse(session, ("ellipse",))

    # 原始采集不得被改写。
    assert {path.name: path.read_bytes() for path in files} == before

    align = res["image_alignment"]
    assert align["available"] is True
    assert align["frames_rotated"] is True
    assert align["analysis_frame"] == "panel_aligned_rotated"
    assert align["convention"] == ROTATION_CONVENTION
    assert align["lobe_threshold_frac"] == pytest.approx(0.02)
    assert align["correction_deg"] == pytest.approx(-DELTA_DEG, abs=0.5)
    for axis in ("panel_x", "panel_y"):
        assert align[f"{axis}_lobes_mid_xy"] == pytest.approx(ZERO_XY, abs=0.25)
    assert align["panel_x_delta_xy"] == pytest.approx(
        [MAG * np.cos(np.deg2rad(DELTA_DEG)), MAG * np.sin(np.deg2rad(DELTA_DEG))],
        abs=0.2,
    )
    assert align["aligned_shape"] == list(
        rotated_canvas_shape(SHAPE, align["correction_deg"])
    )
    assert align["aligned_shape"] != list(SHAPE)

    # 旋转后两轴位移应落在各自数组轴上。
    xinfo = res["tilt"]["tilt_x_plus/tilt_x_minus"]
    assert xinfo["displacement_along_axis_px"] == pytest.approx(MAG, abs=0.6)
    assert xinfo["displacement_cross_axis_px"] == pytest.approx(0.0, abs=0.6)
    assert xinfo["target_lobes_midpoint_xy"] == pytest.approx(
        [0.5 * (align["aligned_shape"][1] - 1), 0.5 * (align["aligned_shape"][0] - 1)],
        abs=0.3,
    )
    yinfo = res["tilt"]["tilt_y_plus/tilt_y_minus"]
    assert yinfo["displacement_along_axis_px"] == pytest.approx(MAG, abs=0.6)
    assert yinfo["displacement_cross_axis_px"] == pytest.approx(0.0, abs=0.6)


def test_pair_displacement_is_rotation_invariant_and_midpoint_is_stationary() -> None:
    c, s = np.cos(np.deg2rad(DELTA_DEG)), np.sin(np.deg2rad(DELTA_DEG))
    vx = (MAG * c, MAG * s)
    zx, zy = ZERO_XY
    plus = _spots_frame([(zx, zy, ZERO_AMP), (zx + vx[0], zy + vx[1], TARGET_AMP)])
    minus = _spots_frame([(zx, zy, ZERO_AMP), (zx - vx[0], zy - vx[1], TARGET_AMP)])

    raw = ISSUE247._pair_displacement(plus.astype(float), minus.astype(float))
    assert raw is not None
    theta = np.deg2rad(-DELTA_DEG)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    expected = rotation @ np.array([raw["dx"], raw["dy"]])

    output_shape = rotated_canvas_shape(SHAPE, -DELTA_DEG)
    plus_aligned = rotate_image_about_center(
        plus, -DELTA_DEG, output_shape=output_shape, order=1
    )
    minus_aligned = rotate_image_about_center(
        minus, -DELTA_DEG, output_shape=output_shape, order=1
    )
    aligned = ISSUE247._pair_displacement(plus_aligned, minus_aligned)
    assert aligned is not None
    assert np.array([aligned["dx"], aligned["dy"]]) == pytest.approx(expected, abs=0.35)
    assert aligned["mid"] == pytest.approx(
        [0.5 * (output_shape[1] - 1), 0.5 * (output_shape[0] - 1)], abs=0.25
    )


def test_pair_displacement_ignores_uniform_frame_background_offset() -> None:
    zx, zy = ZERO_XY
    plus = _spots_frame([(zx, zy, ZERO_AMP), (zx + MAG, zy, TARGET_AMP)]).astype(float)
    minus = _spots_frame([(zx, zy, ZERO_AMP), (zx - MAG, zy, TARGET_AMP)]).astype(float)

    baseline = ISSUE247._pair_displacement(plus, minus)
    offset = ISSUE247._pair_displacement(plus + 137.0, minus + 137.0)

    assert baseline is not None and offset is not None
    assert [offset["dx"], offset["dy"], offset["sep"]] == pytest.approx(
        [baseline["dx"], baseline["dy"], baseline["sep"]], abs=1e-10
    )


def test_missing_axis_pair_keeps_raw_frames_with_reason(tmp_path: Path) -> None:
    session, _ = _session(tmp_path, drop="tilt_y_minus")
    res = ISSUE247.analyse(session, ("ellipse",))

    align = res["image_alignment"]
    assert align["available"] is False
    assert align["frames_rotated"] is False
    assert align["analysis_frame"] == "raw_unrotated"
    assert align["correction_deg"] is None
    assert align["source_shape"] == list(SHAPE)
    assert align["aligned_shape"] == list(SHAPE)
    assert "panel_y" in align["reason"]
    assert "missing" in align["reason"]
    # 其余分析仍然完成（x 轴对可用）。
    assert "tilt_x_plus/tilt_x_minus" in res["tilt"]


def test_degenerate_axis_pair_keeps_raw_frames_with_reason(tmp_path: Path) -> None:
    session, _ = _session(tmp_path, vectors=((12.0, 0.0), (24.0, 0.0)))
    res = ISSUE247.analyse(session, ("ellipse",))

    align = res["image_alignment"]
    assert align["available"] is False
    assert align["frames_rotated"] is False
    assert align["analysis_frame"] == "raw_unrotated"
    assert align["correction_deg"] is None
    assert "degenerate" in align["reason"]
