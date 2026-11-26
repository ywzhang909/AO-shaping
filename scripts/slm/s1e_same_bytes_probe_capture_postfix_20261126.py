# -*- coding: utf-8 -*-
"""S1e same-bytes probe capture -- POST-FIX round (§12.16.5 gate, 2026-11-26).

This is the frozen §12.16.2 / §12.16.3-7 plan with exactly ONE change: the
continuity guard installed at ``driver.py::_write_to_memory`` (§12.16.7) makes
``ctypes.memmove`` copy the logical C-order byte stream.  So this journal's
``physical_sha256`` records the bytes ACTUALLY delivered to the panel --
``np.ascontiguousarray(array).tobytes()`` -- which for H is 7276d9da... instead
of the pre-fix raw F-order buffer b98be9a6....  The pre-guard buffer is still
recorded as ``pre_guard_buffer_sha256`` for provenance.

Everything else is frozen and unchanged: 12 uploads x 2 frames = 24 frames in
ONE SLM #1 open/close session (Daheng cam 0, full-frame Mono12, 0.087 ms,
settle 0.8 s, ``focus_d_m1=0.0``), the same H/B/flat builders, the same order
and predecessors, and the same D1/D2/D3 criteria in the frozen judge.  The
laser is never operated.

Without ``--execute`` the module only builds and prints a deterministic plan:
it creates no directory, imports no driver and never takes the bench lock.

Design source: docs/dev/hw_window_20261123_attribution_prereg.md §12.16;
post-fix round scope: docs/dev/s1e_postfix_prereg_20261126.md.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "src", ROOT / "libs", ROOT, Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts.slm.issue247_geometry_capture_20261120 import (
    flat_phase,
    grating_phase,
    phase_to_gray,
)
from scripts.slm.s1d_orientation_probe_capture_20261124 import (
    acquire_device_lock,
    camera_identity,
    frame_centroid,
    frame_filename,
    require_drive_board_entry,
    utc_now,
    write_json,
)

PLAN_SCHEMA = "issue247-s1e-same-bytes-probe-capture/v1"
SUMMARY_SCHEMA = "issue247-s1e-same-bytes-probe-summary/v1"
JOURNAL_SCHEMA = "issue247-s1e-same-bytes-probe-journal/v1"
RELEASE_SCHEMA = "issue247-s1e-same-bytes-probe-released/v1"

DEFAULT_PANEL_SHAPE = (1200, 1920)  # rows, columns
NATIVE_CCD_SHAPE = (1944, 2592)
BIT_DEPTH_REQUIRED = 12
FULL_SCALE_REQUIRED = (1 << BIT_DEPTH_REQUIRED) - 1  # 4095
ENCODING_MAX_GRAY = 1023
GRATING_PERIOD_PX = 128

DEFAULT_UPLOADS = 12
DEFAULT_FRAMES_PER_UPLOAD = 2
DEFAULT_EXPOSURE_MS = 0.087
DEFAULT_SETTLE_S = 0.8
DEFAULT_WAIT_IDLE_S = 1800.0
DEFAULT_SLM_NUMBER = 1
DEFAULT_CAM_ID = 0
DEFAULT_WAVELENGTH_NM = 1064.0
DEFAULT_SATURATION_FRACTION = 0.9
FLAT_FIRST_MIN_PEAK = 100
FOCUS_D_M1 = 0.0

# Logical digests = the frozen §12.16.2 arrays (ascontiguousarray bytes).
H_PHASE_SHA256 = "d8f16e03d98e79491a2f714fc2f42ffff9e0c5f27861979aba4de8193c84ec0b"
H_GRAY_SHA256 = "7276d9da3b2a5ea536d56ea944adcfdcb38c98ad1dd01c1433d1b2219ac8218e"
B_PHASE_SHA256 = "382100b02c8b07a68e995c70295e8c2ab34abf6cd05e5af8ec163067ff0eda3f"
B_GRAY_SHA256 = "c8d937cedb294339fa385f9ffa65cebd0ff3471ff62009cef42d7637bf590a73"
# Post-guard PHYSICAL digests = the byte stream ctypes.memmove really sends
# AFTER the §12.16.5 continuity guard (np.ascontiguousarray before memmove).
# For H this is now the logical C-order stream == gray SHA; pre-fix it was the
# F-order raw buffer b98be9a6... (recorded separately as pre_guard_buffer).
H_PHYSICAL_SHA256 = "7276d9da3b2a5ea536d56ea944adcfdcb38c98ad1dd01c1433d1b2219ac8218e"
H_PRE_GUARD_BUFFER_SHA256 = (
    "b98be9a6696cd94dbb3fa966e22a7f86e52f52b16ae252d21f7aa07af4079c3a"
)
B_PHYSICAL_SHA256 = "c8d937cedb294339fa385f9ffa65cebd0ff3471ff62009cef42d7637bf590a73"
B_PRE_GUARD_BUFFER_SHA256 = (
    "c8d937cedb294339fa385f9ffa65cebd0ff3471ff62009cef42d7637bf590a73"
)
FLAT_PHASE_SHA256 = "11770da8db5cd320c64cb51890b2a12b4776b87c8a591e138eb722864b419d67"
FLAT_GRAY_SHA256 = "cb228af4065a8be95006792a8199172366d0b63a4d39ce365f9591246861db16"
FLAT_PHYSICAL_SHA256 = "cb228af4065a8be95006792a8199172366d0b63a4d39ce365f9591246861db16"
FLAT_PRE_GUARD_BUFFER_SHA256 = (
    "cb228af4065a8be95006792a8199172366d0b63a4d39ce365f9591246861db16"
)

#: §12.16.2 -- order, array and predecessor are all frozen.
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
ARRAY_BY_ID: dict[str, str] = {
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
PREDECESSOR_BY_ID: dict[str, str] = {
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

#: §12.16.3/§12.16.6 -- frozen judgment thresholds, never edited after capture.
FROZEN_CRITERIA: dict[str, Any] = {
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


# ---------------------------------------------------------------------------
# Offline materialization: no devices, no output directory
# ---------------------------------------------------------------------------
def build_phase(pattern_id: str, shape: tuple[int, int] = DEFAULT_PANEL_SHAPE) -> np.ndarray:
    """Raw phase from the frozen builder; no phase formula lives here."""
    array_id = ARRAY_BY_ID[pattern_id]
    if array_id == "flat":
        return flat_phase(shape)
    if array_id == "H":
        return grating_phase(shape, period_px=GRATING_PERIOD_PX, axis=1)
    if array_id == "B":
        return grating_phase(shape, period_px=GRATING_PERIOD_PX, axis=0)
    raise ValueError(f"unknown array id {array_id!r} for {pattern_id!r}")


def build_gray(pattern_id: str, shape: tuple[int, int] = DEFAULT_PANEL_SHAPE) -> np.ndarray:
    """Phase -> uint16 panel gray through the single existing wrap point."""
    return phase_to_gray(build_phase(pattern_id, shape), ENCODING_MAX_GRAY)


def digest(array: np.ndarray) -> str:
    """Logical digest used by §12.16.2 (C-order, matches S1b/S1c/S1d plans)."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def physical_digest(array: np.ndarray) -> str:
    """Digest of byte stream ACTUALLY sent post-guard (§12.16.5).

    The driver's ``np.ascontiguousarray`` before ``ctypes.memmove`` means the
    delivered stream is the C-order logical bytes.  Recorded as
    ``physical_sha256`` so the judge's sent-array grouping stays meaningful.
    """
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def pre_guard_buffer_digest(array: np.ndarray) -> str:
    """Raw in-memory layout bytes (pre-fix F-order buffer for H), for provenance.

    This is what ``ctypes.memmove`` copied BEFORE the continuity guard; keeping
    it makes the fix auditable without altering the delivered-byte field.
    """
    return hashlib.sha256(array.tobytes(order="A")).hexdigest()


def layout_record(array: np.ndarray) -> dict[str, Any]:
    """Layout fields required by §12.16.7 before any device is touched."""
    return {
        "shape_rows_cols": list(int(v) for v in array.shape),
        "dtype": str(array.dtype),
        "strides": list(int(v) for v in array.strides),
        "c_contiguous": bool(array.flags.c_contiguous),
        "f_contiguous": bool(array.flags.f_contiguous),
        "physical_sha256": physical_digest(array),
        "pre_guard_buffer_sha256": pre_guard_buffer_digest(array),
    }


def expected_digests(array_id: str) -> dict[str, str]:
    """Frozen expected logical + physical digests for flat / H / B (post-fix)."""
    values = {
        "flat": (
            FLAT_PHASE_SHA256, FLAT_GRAY_SHA256, FLAT_PHYSICAL_SHA256,
            FLAT_PRE_GUARD_BUFFER_SHA256,
        ),
        "H": (
            H_PHASE_SHA256, H_GRAY_SHA256, H_PHYSICAL_SHA256,
            H_PRE_GUARD_BUFFER_SHA256,
        ),
        "B": (
            B_PHASE_SHA256, B_GRAY_SHA256, B_PHYSICAL_SHA256,
            B_PRE_GUARD_BUFFER_SHA256,
        ),
    }
    phase, gray, physical, pre_guard = values[array_id]
    return {
        "phase_sha256": phase,
        "gray_sha256": gray,
        "physical_sha256": physical,
        "pre_guard_buffer_sha256": pre_guard,
    }


def build_pattern_records(
    shape: tuple[int, int] = DEFAULT_PANEL_SHAPE,
    frames_per_upload: int = DEFAULT_FRAMES_PER_UPLOAD,
) -> list[dict[str, Any]]:
    """Per-upload plan records with frozen order, predecessor and digests."""
    records: list[dict[str, Any]] = []
    for order, pattern_id in enumerate(FROZEN_ORDER):
        array_id = ARRAY_BY_ID[pattern_id]
        phase = build_phase(pattern_id, shape)
        gray = build_gray(pattern_id, shape)
        record: dict[str, Any] = {
            "order": order,
            "id": pattern_id,
            "array": array_id,
            "predecessor": PREDECESSOR_BY_ID.get(pattern_id),
            "phase_sha256": digest(phase),
            "gray_sha256": digest(gray),
            "n_frames": int(frames_per_upload),
            "panel_shape_rows_cols": [int(shape[0]), int(shape[1])],
            "gray_layout": layout_record(gray),
        }
        records.append(record)
    return records


def build_frame_plan(
    frames_per_upload: int = DEFAULT_FRAMES_PER_UPLOAD,
) -> list[dict[str, Any]]:
    """Global frame order: two consecutive frames after each single upload."""
    plan: list[dict[str, Any]] = []
    index = 0
    for upload_index, pattern_id in enumerate(FROZEN_ORDER):
        for frame_in_upload in range(int(frames_per_upload)):
            plan.append(
                {
                    "index": index,
                    "upload_index": upload_index,
                    "pattern_id": pattern_id,
                    "frame_in_upload": frame_in_upload,
                    "filename": frame_filename(index, pattern_id),
                }
            )
            index += 1
    return plan


def _frozen_settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "exposure_ms_requested": float(args.exposure_ms),
        "settle_s": float(args.settle_s),
        "saturation_fraction": float(args.saturation_fraction),
        "saturation_threshold": float(
            args.saturation_fraction * FULL_SCALE_REQUIRED
        ),
        "frames_per_upload": int(args.frames_per_upload),
        "uploads": int(DEFAULT_UPLOADS),
        "slm_number": int(args.slm_number),
        "cam_id": int(args.cam_id),
        "wavelength_nm": float(args.wavelength_nm),
        "focus_d_m1": FOCUS_D_M1,
        "bit_depth": BIT_DEPTH_REQUIRED,
        "full_scale": FULL_SCALE_REQUIRED,
        "native_ccd_shape_rows_cols": list(NATIVE_CCD_SHAPE),
        "panel_shape_rows_cols": list(DEFAULT_PANEL_SHAPE),
        "encoding_max_gray": ENCODING_MAX_GRAY,
        "grating_period_px": GRATING_PERIOD_PX,
        "video_mode": 0,
    }


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Deterministic plan; ``display_topology`` is filled at execute time."""
    frames_per_upload = int(args.frames_per_upload)
    return {
        "schema": PLAN_SCHEMA,
        "task": (
            "preregistered S1e same-bytes probe: 12 uploads x "
            f"{frames_per_upload} frame(s), single session"
        ),
        "design_source": (
            "docs/dev/hw_window_20261123_attribution_prereg.md §12.16; "
            "post-fix round: docs/dev/s1e_postfix_prereg_20261126.md"
        ),
        "capture_order": list(FROZEN_ORDER),
        "array_by_id": dict(ARRAY_BY_ID),
        "predecessor_by_id": dict(PREDECESSOR_BY_ID),
        "uploads": int(DEFAULT_UPLOADS),
        "frames_per_upload": frames_per_upload,
        "n_frames_expected": int(DEFAULT_UPLOADS) * frames_per_upload,
        "encoding": {
            "scales_used": [ENCODING_MAX_GRAY],
            "encoding_max_gray": ENCODING_MAX_GRAY,
            "source": (
                "scripts.slm.issue247_geometry_capture_20261120.phase_to_gray "
                "(G_REF=1023)"
            ),
        },
        "focus_compensation": {
            "focus_d_m1": FOCUS_D_M1,
            "compensation_applied": False,
        },
        "expected_digests": {
            "flat": expected_digests("flat"),
            "H": expected_digests("H"),
            "B": expected_digests("B"),
        },
        "criteria": FROZEN_CRITERIA,
        "display_topology": {
            "status": "not_probed",
            "reason": "display topology is queried at execute time only",
        },
        "capture_settings": _frozen_settings(args),
        "patterns": build_pattern_records(DEFAULT_PANEL_SHAPE, frames_per_upload),
        "frames": build_frame_plan(frames_per_upload),
        "phase_builder_source": (
            "scripts.slm.issue247_geometry_capture_20261120 "
            "(flat_phase / grating_phase / phase_to_gray)"
        ),
    }


# ---------------------------------------------------------------------------
# Frozen judgment helpers (pure, locked before capture; executed afterwards)
# ---------------------------------------------------------------------------
def pairwise_max_distance(points: list[tuple[float, float]]) -> float:
    """Maximum Euclidean distance between any two spot centers."""
    if len(points) < 2:
        return 0.0
    return max(
        float(np.hypot(r1 - r2, c1 - c2))
        for i, (r1, c1) in enumerate(points)
        for r2, c2 in points[i + 1:]
    )


def axis_type(dr: float, dc: float) -> str:
    """D2 col/row/other classification from the frozen distance windows."""
    spec = FROZEN_CRITERIA["D2"]
    ratio = float(spec["axis_ratio_min"])
    dist = float(np.hypot(dr, dc))
    col_lo, col_hi = (float(v) for v in spec["col_dist_px"])
    row_lo, row_hi = (float(v) for v in spec["row_dist_px"])
    if abs(dc) > ratio * abs(dr) and col_lo <= dist <= col_hi:
        return "col-type"
    if abs(dr) > ratio * abs(dc) and row_lo <= dist <= row_hi:
        return "row-type"
    return "other"


def d1_verdict(max_dist_px: float, n_effective: int) -> str:
    """D1 same-array determinism verdict from frozen tolerances."""
    spec = FROZEN_CRITERIA["D1"]
    if n_effective < int(spec["min_effective_uploads"]):
        return "inconclusive"
    if max_dist_px <= float(spec["max_pairwise_dist_px"]):
        return "D-pass"
    return "D-fail"


def d6_zero_exception(groups: dict[int, set[str]]) -> bool:
    """True when every slot group has exactly one axis type (D6 prerequisite)."""
    return bool(groups) and all(len(types) == 1 for types in groups.values())


# ---------------------------------------------------------------------------
# Display topology probe: unavailable must be written, never inferred
# ---------------------------------------------------------------------------
def probe_display_topology() -> dict[str, Any]:
    """Best-effort Windows display enumeration; never claims unreadable state."""
    if sys.platform != "win32":
        return {"status": "unavailable", "reason": f"platform {sys.platform!r}"}
    try:
        from ctypes import wintypes

        class _DisplayPosition(ctypes.Structure):
            # DEVMODEW dmPosition/dmDisplayOrientation/dmDisplayFixedOutput.
            _fields_ = (
                ("x", wintypes.LONG),
                ("y", wintypes.LONG),
                ("orientation", wintypes.DWORD),
                ("fixed_output", wintypes.DWORD),
            )

        class _DisplayUnion(ctypes.Union):
            _fields_ = (
                ("printer", ctypes.c_short * 8),
                ("display", _DisplayPosition),
            )

        class _DevMode(ctypes.Structure):
            _fields_ = (
                ("dm_device_name", wintypes.WCHAR * 32),
                ("dm_spec_version", wintypes.WORD),
                ("dm_driver_version", wintypes.WORD),
                ("dm_size", wintypes.WORD),
                ("dm_driver_extra", wintypes.WORD),
                ("dm_fields", wintypes.DWORD),
                ("dm_union", _DisplayUnion),
                ("dm_color", ctypes.c_short),
                ("dm_duplex", ctypes.c_short),
                ("dm_y_resolution", ctypes.c_short),
                ("dm_tt_option", ctypes.c_short),
                ("dm_collate", ctypes.c_short),
                ("dm_form_name", wintypes.WCHAR * 32),
                ("dm_log_pixels", wintypes.WORD),
                ("dm_bits_per_pel", wintypes.DWORD),
                ("dm_pels_width", wintypes.DWORD),
                ("dm_pels_height", wintypes.DWORD),
                ("dm_display_flags", wintypes.DWORD),
                ("dm_display_frequency", wintypes.DWORD),
                ("dm_icm_method", wintypes.DWORD),
                ("dm_icm_intent", wintypes.DWORD),
                ("dm_media_type", wintypes.DWORD),
                ("dm_dither_type", wintypes.DWORD),
                ("dm_reserved1", wintypes.DWORD),
                ("dm_reserved2", wintypes.DWORD),
                ("dm_panning_width", wintypes.DWORD),
                ("dm_panning_height", wintypes.DWORD),
            )

        class _DisplayDevice(ctypes.Structure):
            _fields_ = (
                ("cb", wintypes.DWORD),
                ("device_name", wintypes.WCHAR * 32),
                ("device_string", wintypes.WCHAR * 128),
                ("state_flags", wintypes.DWORD),
            )

        user32 = ctypes.windll.user32
        displays: list[dict[str, Any]] = []
        device = _DisplayDevice()
        device.cb = ctypes.sizeof(device)
        index = 0
        while user32.EnumDisplayDevicesW(None, index, ctypes.byref(device), 0):
            name = str(device.device_name)
            mode = _DevMode()
            mode.dm_size = ctypes.sizeof(mode)
            if user32.EnumDisplaySettingsW(name, -1, ctypes.byref(mode)):
                displays.append(
                    {
                        "device_name": name,
                        "device_string": str(device.device_string),
                        "width_px": int(mode.dm_pels_width),
                        "height_px": int(mode.dm_pels_height),
                        "orientation_code": int(mode.dm_union.display.orientation),
                        "log_pixels": int(mode.dm_log_pixels),
                    }
                )
            index += 1
        if not displays:
            return {
                "status": "unavailable",
                "reason": "EnumDisplaySettingsW returned no readable display mode",
            }
        return {"status": "ok", "displays": displays}
    except Exception as exc:  # non-interactive session / missing user32
        return {
            "status": "unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
        }


def slot_snapshot(slm: Any) -> dict[str, Any]:
    """Read-only slot state for the §12.16.6 journal fields."""
    try:
        displayed = slm._read_displayed_slot_safe()
    except Exception:
        displayed = None
    try:
        counter = int(slm._current_memory_slot)
    except (AttributeError, TypeError, ValueError):
        counter = None
    return {
        "current_memory_slot_before_display": counter,
        "displayed_slot_safe": None if displayed is None else int(displayed),
    }


def _check_frozen_arrays(records: list[dict[str, Any]]) -> None:
    """Abort before opening devices if a frozen array drifted."""
    for record in records:
        expected = expected_digests(record["array"])
        if record["phase_sha256"] != expected["phase_sha256"]:
            raise RuntimeError(f"{record['id']}: phase SHA256 drifted")
        if record["gray_sha256"] != expected["gray_sha256"]:
            raise RuntimeError(f"{record['id']}: gray SHA256 drifted")
        if record["gray_layout"]["physical_sha256"] != expected["physical_sha256"]:
            raise RuntimeError(f"{record['id']}: post-guard byte stream drifted")
        pre_guard = record["gray_layout"].get("pre_guard_buffer_sha256")
        if pre_guard != expected["pre_guard_buffer_sha256"]:
            raise RuntimeError(f"{record['id']}: pre-guard buffer drifted")


def build_summary(
    plan: dict[str, Any],
    *,
    run: dict[str, Any],
    device: dict[str, Any],
    release: dict[str, Any],
    frame_records: list[dict[str, Any]],
    upload_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Pure, offline-testable summary body (one record per upload and frame)."""
    patterns = []
    for record in plan["patterns"]:
        frames = [
            {
                "index": frame["index"],
                "upload_index": frame["upload_index"],
                "filename": frame["filename"],
                "frame_sha256": frame["frame_sha256"],
                "peak": frame["peak"],
                "median": frame["median"],
                "saturated_count": frame["saturated_count"],
                "utc": frame["utc"],
            }
            for frame in frame_records
            if frame["pattern_id"] == record["id"]
        ]
        upload = next(
            (u for u in upload_records if u["pattern_id"] == record["id"]),
            None,
        )
        patterns.append(
            {
                **record,
                "n_frames_planned": int(record["n_frames"]),
                "n_frames": len(frames),
                "frames": frames,
                "upload": upload,
            }
        )
    return {
        "schema": SUMMARY_SCHEMA,
        "run": {
            "run_id": run["run_id"],
            "capture_dir": run["capture_dir"],
            "started_at_utc": run["started_at_utc"],
            "finished_at_utc": utc_now(),
            "status": run["status"],
            "design_source": (
                "docs/dev/hw_window_20261123_attribution_prereg.md §12.16; "
                "post-fix round: docs/dev/s1e_postfix_prereg_20261126.md"
            ),
        },
        "status": run["status"],
        "n_uploads": len(upload_records),
        "expected_uploads": int(plan["uploads"]),
        "n_frames": len(frame_records),
        "expected_frames": int(plan["n_frames_expected"]),
        "frames_per_upload": int(plan["frames_per_upload"]),
        "uploads": upload_records,
        "patterns": patterns,
        "criteria": dict(FROZEN_CRITERIA),
        "slm_number": device["slm_number"],
        "slm_drive_board_id": device["drive_board_id"],
        "wavelength_nm": device["wavelength_nm"],
        "exposure_ms": device["exposure_ms"],
        "bit_depth": device["bit_depth"],
        "full_scale": device["full_scale"],
        "settle_s": device["settle_s"],
        "camera": device["camera"],
        "encoding_max_gray": ENCODING_MAX_GRAY,
        "focus_d_m1": FOCUS_D_M1,
        "display_topology": plan["display_topology"],
        "closed_normally": release["closed_normally"],
        "elapsed_s": release["elapsed_s"],
        "timestamp_utc": utc_now(),
        "error": run["error"],
    }


# ---------------------------------------------------------------------------
# Device session (only reached with --execute)
# ---------------------------------------------------------------------------
def execute(args: argparse.Namespace, plan: dict[str, Any]) -> int:
    out: Path = args.out
    lock = acquire_device_lock(float(args.wait_until_idle))
    if lock is None:
        logger.error(
            "could not acquire bench device lock within {}s", args.wait_until_idle
        )
        return 2

    status = "error"
    error: str | None = None
    exit_code = 1
    created_out = False
    slm: Any = None
    cam: Any = None
    drive_board_id: str | None = None
    frame_records: list[dict[str, Any]] = []
    upload_records: list[dict[str, Any]] = []
    release: dict[str, Any] = {
        "schema": RELEASE_SCHEMA,
        "slm_closed": False,
        "camera_closed": False,
        "lock_released": False,
        "closed_normally": False,
        "error": None,
    }
    started = time.monotonic()
    started_utc = utc_now()
    exposure_ms: float | None = None
    actual_bits: int | None = None
    full_scale: float | None = None

    try:
        # Materialize and verify EVERY gray array before any device is touched.
        prepared: list[tuple[dict[str, Any], np.ndarray, np.ndarray]] = [
            (
                record,
                build_phase(str(record["id"])),
                build_gray(str(record["id"])),
            )
            for record in plan["patterns"]
        ]
        _check_frozen_arrays(plan["patterns"])

        out.mkdir(parents=True, exist_ok=False)
        created_out = True
        (out / "frames").mkdir()
        plan["display_topology"] = probe_display_topology()
        write_json(out / "plan.json", plan)
        journal_path = out / "capture_journal.jsonl"
        journal_path.touch()

        from ao_shaping.drivers.ccd.common import (
            create_camera,
            get_camera_exposure_ms,
            set_camera_exposure_ms,
        )
        from ao_shaping.drivers.slm.santec import Santec

        # Exactly ONE SLM open and ONE close for the whole arm.
        slm = Santec(slm_number=int(args.slm_number), video_mode=0)
        slm.open()
        drive_board_id = require_drive_board_entry(slm)
        actual_panel_shape = (int(slm.Panel_Res[1]), int(slm.Panel_Res[0]))
        if actual_panel_shape != DEFAULT_PANEL_SHAPE:
            raise RuntimeError(
                f"SLM panel shape {actual_panel_shape} != frozen {DEFAULT_PANEL_SHAPE}"
            )
        slm.set_phase_overlay(None)

        cam = create_camera(
            "daheng",
            int(args.cam_id),
            exposure_time_ms=float(args.exposure_ms),
            bit_depth=BIT_DEPTH_REQUIRED,
        )
        cam.open()
        actual_bits = getattr(cam, "bit_depth", None)
        if actual_bits != BIT_DEPTH_REQUIRED:
            raise RuntimeError(
                f"camera bit-depth readback must be 12, got {actual_bits!r}"
            )
        set_camera_exposure_ms(cam, float(args.exposure_ms))
        exposure_ms = float(get_camera_exposure_ms(cam))
        full_scale = float(
            cam.effective_full_scale(np.zeros((1,), dtype=np.uint16))
        )
        if full_scale != float(FULL_SCALE_REQUIRED):
            raise RuntimeError(
                f"camera full scale {full_scale} != {FULL_SCALE_REQUIRED}"
            )
        saturation_threshold = float(args.saturation_fraction) * full_scale
        settle_s = float(args.settle_s)
        frames_per_upload = int(args.frames_per_upload)

        with journal_path.open("a", encoding="utf-8") as journal:
            for record, phase, gray in prepared:
                pattern_id = str(record["id"])
                array_id = str(record["array"])
                before = slot_snapshot(slm)
                memory_number = int(slm.display_data(gray))
                after = slot_snapshot(slm)
                sent_layout = layout_record(gray)
                expected = expected_digests(array_id)
                if sent_layout["physical_sha256"] != expected["physical_sha256"]:
                    raise RuntimeError(
                        f"{pattern_id}: post-guard byte stream drift at upload time"
                    )
                if (
                    sent_layout["pre_guard_buffer_sha256"]
                    != expected["pre_guard_buffer_sha256"]
                ):
                    raise RuntimeError(
                        f"{pattern_id}: pre-guard buffer drift at upload time"
                    )
                upload = {
                    "schema": JOURNAL_SCHEMA,
                    "event": "upload",
                    "upload_index": int(record["order"]),
                    "pattern_id": pattern_id,
                    "array": array_id,
                    "predecessor": record["predecessor"],
                    "video_mode": 0,
                    "memory_number": memory_number,
                    "slots_before": before,
                    "slots_after": {
                        "current_memory_slot_after_display": (
                            after["current_memory_slot_before_display"]
                        ),
                        "displayed_slot_safe": after["displayed_slot_safe"],
                    },
                    "phase_sha256": digest(phase),
                    "gray_sha256": digest(gray),
                    "sent_array": sent_layout,
                    "utc": utc_now(),
                }
                upload_records.append(upload)
                journal.write(json.dumps(upload, ensure_ascii=False, allow_nan=False) + "\n")
                journal.flush()

                time.sleep(settle_s)
                for frame_in_upload in range(frames_per_upload):
                    frame = np.asarray(cam.get_numpy_image())
                    if frame.shape != NATIVE_CCD_SHAPE:
                        raise RuntimeError(
                            f"frame shape {frame.shape} != {NATIVE_CCD_SHAPE}"
                        )
                    peak = int(np.max(frame))
                    median = float(np.median(frame))
                    saturated_count = int(
                        np.count_nonzero(frame >= saturation_threshold)
                    )
                    timestamp = utc_now()
                    index = len(frame_records)
                    filename = frame_filename(index, pattern_id)
                    raw_path = out / "frames" / filename
                    np.save(raw_path, frame.astype(np.uint16, copy=False))
                    row = {
                        "schema": JOURNAL_SCHEMA,
                        "event": "frame_saved",
                        "index": index,
                        "upload_index": int(record["order"]),
                        "pattern_id": pattern_id,
                        "array": array_id,
                        "predecessor": record["predecessor"],
                        "frame_in_upload": frame_in_upload,
                        "filename": filename,
                        "frame_sha256": digest(frame),
                        "peak": peak,
                        "median": median,
                        "saturated": saturated_count > 0,
                        "saturated_count": saturated_count,
                        "centroid": frame_centroid(frame),
                        "utc": timestamp,
                        "phase_sha256": digest(phase),
                        "gray_sha256": digest(gray),
                        "physical_sha256": sent_layout["physical_sha256"],
                        "memory_number": memory_number,
                        "current_memory_slot": after[
                            "current_memory_slot_before_display"
                        ],
                        "displayed_slot_safe": after["displayed_slot_safe"],
                        "exposure_ms": exposure_ms,
                        "bit_depth": int(actual_bits),
                        "full_scale": int(full_scale),
                        "settle_s": settle_s,
                        "focus_d_m1": FOCUS_D_M1,
                    }
                    frame_records.append(row)
                    journal.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    journal.flush()

                    # Frozen stop conditions; never auto-reopen or retry.
                    if index == 0 and peak < FLAT_FIRST_MIN_PEAK:
                        raise RuntimeError(
                            f"flat first-frame 0th peak {peak} < "
                            f"{FLAT_FIRST_MIN_PEAK} (no light / invalid optical path)"
                        )
                    if saturated_count > 0:
                        raise RuntimeError(
                            f"frame {filename!r} saturated: {saturated_count} "
                            f"pixels >= {saturation_threshold:.0f}"
                        )

        status = "ok"
        exit_code = 0
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.error("S1e capture aborted: {}", error)
    finally:
        if slm is not None:
            try:
                slm.close()
                release["slm_closed"] = True
            except Exception as exc:
                release["error"] = f"slm.close: {type(exc).__name__}: {exc}"
        if cam is not None:
            try:
                cam.close()
                release["camera_closed"] = True
            except Exception as exc:
                release["error"] = (
                    f"{release['error']}; camera.close: {type(exc).__name__}: {exc}"
                )
        if lock is not None:
            try:
                lock.release()
                release["lock_released"] = True
            except Exception as exc:
                release["error"] = (
                    f"{release['error']}; lock.release: {type(exc).__name__}: {exc}"
                )
        release["closed_normally"] = bool(
            release["slm_closed"]
            and release["camera_closed"]
            and release["lock_released"]
            and release["error"] is None
        )
        release["elapsed_s"] = time.monotonic() - started
        release["released_at_utc"] = utc_now()
        if created_out:
            write_json(out / "released.json", release)
            summary = build_summary(
                plan,
                run={
                    "run_id": out.name,
                    "capture_dir": str(out),
                    "started_at_utc": started_utc,
                    "status": status,
                    "error": error,
                },
                device={
                    "exposure_ms": exposure_ms,
                    "bit_depth": actual_bits,
                    "full_scale": full_scale,
                    "settle_s": float(args.settle_s),
                    "slm_number": int(args.slm_number),
                    "drive_board_id": drive_board_id,
                    "wavelength_nm": float(args.wavelength_nm),
                    "camera": None if cam is None else camera_identity(cam, args),
                },
                release=release,
                frame_records=frame_records,
                upload_records=upload_records,
            )
            write_json(out / "summary.json", summary)
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "S1e same-bytes probe capture; offline plan by default (§12.16)"
        )
    )
    parser.add_argument(
        "--out", type=Path, help="fresh output directory (required with --execute)"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="open devices and acquire the frozen 24 frames",
    )
    parser.add_argument(
        "--wait-until-idle",
        type=float,
        default=DEFAULT_WAIT_IDLE_S,
        help="seconds to wait for the bench device lock (default 1800)",
    )
    parser.add_argument("--exposure-ms", type=float, default=DEFAULT_EXPOSURE_MS)
    parser.add_argument("--settle-s", type=float, default=DEFAULT_SETTLE_S)
    parser.add_argument(
        "--frames-per-upload",
        type=int,
        default=DEFAULT_FRAMES_PER_UPLOAD,
        help="frames captured per upload (frozen plan: 2)",
    )
    parser.add_argument("--slm-number", type=int, default=DEFAULT_SLM_NUMBER)
    parser.add_argument("--cam-id", type=int, default=DEFAULT_CAM_ID)
    parser.add_argument(
        "--wavelength-nm", type=float, default=DEFAULT_WAVELENGTH_NM
    )
    parser.add_argument(
        "--saturation-fraction",
        type=float,
        default=DEFAULT_SATURATION_FRACTION,
    )
    args = parser.parse_args(argv)

    if args.execute and args.out is None:
        parser.error("--out is required with --execute")
    if args.exposure_ms <= 0:
        parser.error("--exposure-ms must be positive")
    if args.settle_s <= 0:
        parser.error("--settle-s must be positive")
    if args.frames_per_upload < 1:
        parser.error("--frames-per-upload must be >= 1")
    if args.wait_until_idle < 0:
        parser.error("--wait-until-idle must be non-negative")
    if not 0.0 < args.saturation_fraction <= 1.0:
        parser.error("--saturation-fraction must be in (0, 1]")

    plan = build_plan(args)
    if args.execute:
        return execute(args, plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
