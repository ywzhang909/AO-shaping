# -*- coding: utf-8 -*-
"""S1d orientation-probe capture (preregistered §12.14, frozen before device use).

The frozen plan is 14 patterns x 2 frames = 28 frames in one SLM open/close
session on SLM #1 (1064 nm), Daheng camera 0, full-frame Mono12 (full scale
4095), no Fresnel compensation (``focus_d_m1 = 0.0``); the script never operates
the laser.  Without ``--execute`` this module only builds and prints a
deterministic plan: it does not create the output directory, does not import
device drivers and does not touch the bench lock.

Every phase array is produced by an existing builder (never reimplemented here):

* ``scripts.slm.issue247_geometry_capture_20261120`` -- ``flat_phase``,
  ``grating_phase``, ``ladder_phase``, ``blaze2d_phase`` and the single
  ``phase_to_gray`` wrap point;
* ``scripts.slm.content_axis_2x2_capture_20261124`` -- ``make_phase`` (the
  S1b content A pattern).

The output layout is the solver-compatible A1 one: ``frames/{idx:04d}_{pid}.npy``
(raw uint16, one file per frame) plus ``summary.json``, which
``scripts.slm.issue247_axis_map_solve_20261121`` reads directly.

Design source: docs/dev/hw_window_20261123_attribution_prereg.md §12.14.
"""

from __future__ import annotations

import argparse
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

# Phase math is owned by the two builder modules (single homes).  This file
# never reimplements a phase formula, and never edits those modules.
from scripts.slm.content_axis_2x2_capture_20261124 import make_phase
from scripts.slm.issue247_geometry_capture_20261120 import (
    blaze2d_phase,
    flat_phase,
    grating_phase,
    ladder_phase,
    phase_to_gray,
)

PLAN_SCHEMA = "issue247-s1d-orientation-probe-capture/v1"
SUMMARY_SCHEMA = "issue247-s1d-orientation-probe-summary/v1"
JOURNAL_SCHEMA = "issue247-s1d-orientation-probe-journal/v1"
RELEASE_SCHEMA = "issue247-s1d-orientation-probe-released/v1"

DEFAULT_PANEL_SHAPE = (1200, 1920)  # rows, columns
NATIVE_CCD_SHAPE = (1944, 2592)  # full native Daheng frame, never windowed
BIT_DEPTH_REQUIRED = 12
FULL_SCALE_REQUIRED = (1 << BIT_DEPTH_REQUIRED) - 1  # 4095

#: Frozen rad->gray encoding scale for every pattern in §12.14.
ENCODING_MAX_GRAY = 1023

DEFAULT_FRAMES_PER_PATTERN = 2
DEFAULT_EXPOSURE_MS = 0.087
DEFAULT_SETTLE_S = 0.8
DEFAULT_WAIT_IDLE_S = 1800.0
DEFAULT_SLM_NUMBER = 1
DEFAULT_CAM_ID = 0
DEFAULT_WAVELENGTH_NM = 1064.0
DEFAULT_SATURATION_FRACTION = 0.9

#: The first ``flat`` frame's 0th peak below this means no light / invalid path.
FLAT_FIRST_MIN_PEAK = 100
LOCK_POLL_S = 1.0
#: No Fresnel compensation is stacked, matching S1b D=0 and A1.
FOCUS_D_M1 = 0.0

CONTENT_ID = "A"
BLAZE2D_SIGNS: tuple[tuple[float, float, int, int], ...] = (
    (128.0, 192.0, 1, 1),
    (128.0, 192.0, 1, -1),
    (192.0, 128.0, 1, 1),
    (192.0, 128.0, 1, -1),
)


# ---------------------------------------------------------------------------
# Frozen pattern plan (§12.14.1) -- order matters
# ---------------------------------------------------------------------------
def _flat_spec(pattern_id: str) -> dict[str, Any]:
    return {
        "pattern_id": pattern_id,
        "kind": "flat",
        "encoding_max_gray": ENCODING_MAX_GRAY,
    }


def _grating_spec(
    pattern_id: str, *, axis: int, period_px: int
) -> dict[str, Any]:
    return {
        "pattern_id": pattern_id,
        "kind": "grating",
        "encoding_max_gray": ENCODING_MAX_GRAY,
        "axis": int(axis),
        "period_px": int(period_px),
    }


def _ladder_spec(pattern_id: str, *, axis: int, sign: int = 1) -> dict[str, Any]:
    return {
        "pattern_id": pattern_id,
        "kind": "ladder",
        "encoding_max_gray": ENCODING_MAX_GRAY,
        "axis": int(axis),
        "sign": int(sign),
    }


def _blaze2d_spec(
    pattern_id: str,
    *,
    period_x_px: float,
    period_y_px: float,
    sign_x: int,
    sign_y: int,
) -> dict[str, Any]:
    return {
        "pattern_id": pattern_id,
        "kind": "blaze2d",
        "encoding_max_gray": ENCODING_MAX_GRAY,
        "period_x_px": float(period_x_px),
        "period_y_px": float(period_y_px),
        "sign_x": int(sign_x),
        "sign_y": int(sign_y),
    }


def _content_spec(pattern_id: str, *, period_px: int) -> dict[str, Any]:
    return {
        "pattern_id": pattern_id,
        "kind": "content",
        "encoding_max_gray": ENCODING_MAX_GRAY,
        "content": CONTENT_ID,
        "period_px": int(period_px),
        "d_m1": FOCUS_D_M1,
    }


def _blaze2d_id(period_x: float, period_y: float, sx: int, sy: int) -> str:
    return (
        f"blaze2d_px{int(period_x)}_py{int(period_y)}"
        f"_sx{sx:+d}_sy{sy:+d}"
    )


PATTERN_SPECS: tuple[dict[str, Any], ...] = (
    _flat_spec("flat"),
    _grating_spec("grating_x_p128", axis=1, period_px=128),
    _grating_spec("grating_y_p128", axis=0, period_px=128),
    _grating_spec("grating_x_p256", axis=1, period_px=256),
    _grating_spec("grating_y_p256", axis=0, period_px=256),
    _ladder_spec("ladder_x", axis=1, sign=1),
    _ladder_spec("ladder_y", axis=0, sign=1),
    *(
        _blaze2d_spec(
            _blaze2d_id(px, py, sx, sy),
            period_x_px=px,
            period_y_px=py,
            sign_x=sx,
            sign_y=sy,
        )
        for px, py, sx, sy in BLAZE2D_SIGNS
    ),
    _content_spec("content_a128", period_px=128),
    _content_spec("content_a256", period_px=256),
    _flat_spec("flat_post"),
)

CAPTURE_ORDER: tuple[str, ...] = tuple(
    spec["pattern_id"] for spec in PATTERN_SPECS
)

#: The four 2x2-fit designs, listed for the frozen solver's ``blaze2d_designs``.
BLAZE2D_PATTERN_IDS: tuple[str, ...] = tuple(
    spec["pattern_id"] for spec in PATTERN_SPECS if spec["kind"] == "blaze2d"
)


# ---------------------------------------------------------------------------
# Offline materialization (no devices, no mounts)
# ---------------------------------------------------------------------------
def build_phase(spec: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    """Materialize the raw phase array for one frozen pattern spec."""
    kind = spec["kind"]
    if kind == "flat":
        return flat_phase(shape)
    if kind == "grating":
        return grating_phase(
            shape, period_px=int(spec["period_px"]), axis=int(spec["axis"])
        )
    if kind == "ladder":
        return ladder_phase(
            shape, axis=int(spec["axis"]), sign=int(spec["sign"])
        )
    if kind == "blaze2d":
        return blaze2d_phase(
            shape,
            period_x_px=float(spec["period_x_px"]),
            period_y_px=float(spec["period_y_px"]),
            sign_x=int(spec["sign_x"]),
            sign_y=int(spec["sign_y"]),
        )
    if kind == "content":
        return make_phase(
            str(spec["content"]),
            int(spec["period_px"]),
            shape,
            float(spec["d_m1"]),
        )
    raise ValueError(f"unknown pattern kind {kind!r}")


def build_gray(spec: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    """Phase -> uint16 panel gray through the single wrap point."""
    return phase_to_gray(build_phase(spec, shape), int(spec["encoding_max_gray"]))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _gray_unique_record(gray: np.ndarray) -> dict[str, Any]:
    unique = sorted(int(v) for v in np.unique(gray))
    if len(unique) <= 64:
        return {"gray_unique": unique}
    return {
        "gray_unique_count": len(unique),
        "gray_unique_sample": unique[:64],
    }


def frame_filename(index: int, pattern_id: str) -> str:
    """Solver-compatible frame name: ``{idx:04d}_{pattern_id}.npy``."""
    return f"{int(index):04d}_{pattern_id}.npy"


def build_pattern_records(
    shape: tuple[int, int] = DEFAULT_PANEL_SHAPE,
    frames_per_pattern: int = DEFAULT_FRAMES_PER_PATTERN,
) -> list[dict[str, Any]]:
    """Per-pattern plan records with the frozen args and phase/gray digests."""
    records: list[dict[str, Any]] = []
    for order, spec in enumerate(PATTERN_SPECS):
        phase = build_phase(spec, shape)
        gray = phase_to_gray(phase, int(spec["encoding_max_gray"]))
        record: dict[str, Any] = {
            "order": order,
            "id": spec["pattern_id"],
            "kind": spec["kind"],
            "encoding_max_gray": int(spec["encoding_max_gray"]),
            "phase_sha256": digest(phase),
            "gray_sha256": digest(gray),
            "n_frames": int(frames_per_pattern),
            "panel_shape_rows_cols": [int(shape[0]), int(shape[1])],
            **_gray_unique_record(gray),
        }
        if spec["kind"] == "grating":
            record["axis"] = int(spec["axis"])
            record["period_px"] = int(spec["period_px"])
        elif spec["kind"] == "ladder":
            record["axis"] = int(spec["axis"])
            record["sign"] = int(spec["sign"])
        elif spec["kind"] == "blaze2d":
            record["period_x_px"] = float(spec["period_x_px"])
            record["period_y_px"] = float(spec["period_y_px"])
            record["sign_x"] = int(spec["sign_x"])
            record["sign_y"] = int(spec["sign_y"])
        elif spec["kind"] == "content":
            record["content"] = str(spec["content"])
            record["period_px"] = int(spec["period_px"])
            record["d_m1"] = float(spec["d_m1"])
        records.append(record)
    return records


def build_frame_plan(
    frames_per_pattern: int = DEFAULT_FRAMES_PER_PATTERN,
) -> list[dict[str, Any]]:
    """Global frame order: ``frames_per_pattern`` consecutive frames per pattern."""
    plan: list[dict[str, Any]] = []
    index = 0
    for order, spec in enumerate(PATTERN_SPECS):
        for frame_in_pattern in range(int(frames_per_pattern)):
            plan.append(
                {
                    "index": index,
                    "order": order,
                    "pattern_id": spec["pattern_id"],
                    "frame_in_pattern": frame_in_pattern,
                    "filename": frame_filename(index, spec["pattern_id"]),
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
        "frames_per_pattern": int(args.frames_per_pattern),
        "slm_number": int(args.slm_number),
        "cam_id": int(args.cam_id),
        "wavelength_nm": float(args.wavelength_nm),
        "focus_d_m1": FOCUS_D_M1,
        "bit_depth": BIT_DEPTH_REQUIRED,
        "full_scale": FULL_SCALE_REQUIRED,
        "panel_shape_rows_cols": [
            int(DEFAULT_PANEL_SHAPE[0]),
            int(DEFAULT_PANEL_SHAPE[1]),
        ],
        "encoding_max_gray": ENCODING_MAX_GRAY,
    }


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    frames_per_pattern = int(args.frames_per_pattern)
    return {
        "schema": PLAN_SCHEMA,
        "task": (
            "preregistered S1d orientation probe: 14 patterns x "
            f"{frames_per_pattern} frame(s), single session"
        ),
        "design_source": (
            "docs/dev/hw_window_20261123_attribution_prereg.md §12.14"
        ),
        "capture_order": list(CAPTURE_ORDER),
        "frames_per_pattern": frames_per_pattern,
        "n_patterns": len(CAPTURE_ORDER),
        "n_frames_expected": len(CAPTURE_ORDER) * frames_per_pattern,
        "encoding": {
            "scales_used": [ENCODING_MAX_GRAY],
            "encoding_max_gray": ENCODING_MAX_GRAY,
            "formula": (
                "gray = mod(rint(mod(phase, 2pi) * max_grayscale / (2pi)), "
                "max_grayscale + 1)"
            ),
            "source": (
                "scripts.slm.issue247_geometry_capture_20261120.phase_to_gray "
                "(G_REF=1023)"
            ),
        },
        "focus_compensation": {
            "focus_d_m1": FOCUS_D_M1,
            "compensation_applied": False,
        },
        "capture_settings": _frozen_settings(args),
        "blaze2d_designs": list(BLAZE2D_PATTERN_IDS),
        "patterns": build_pattern_records(
            DEFAULT_PANEL_SHAPE, frames_per_pattern
        ),
        "frames": build_frame_plan(frames_per_pattern),
        "phase_builder_source": (
            "scripts.slm.issue247_geometry_capture_20261120 "
            "(flat_phase / grating_phase / ladder_phase / blaze2d_phase / "
            "phase_to_gray) + "
            "scripts.slm.content_axis_2x2_capture_20261124.make_phase"
        ),
        "generated_at_utc": utc_now(),
    }


# ---------------------------------------------------------------------------
# Hardware execution (drivers imported only here, after lock + fresh --out)
# ---------------------------------------------------------------------------
def acquire_device_lock(wait_s: float) -> Any | None:
    """Acquire ``DeviceLock.default()``, polling for up to ``wait_s`` seconds."""
    from ao_shaping.tools.kernels.device_mutex import DeviceLock

    lock = DeviceLock.default()
    logged_holder = False
    deadline = time.monotonic() + max(0.0, float(wait_s))
    while True:
        if lock.acquire(timeout=0.0):
            if logged_holder:
                logger.info("bench device lock acquired after waiting")
            return lock
        if not logged_holder:
            holder = lock.holder()
            logger.warning(
                "bench device lock held by {}",
                holder.describe() if holder else "unknown",
            )
            logged_holder = True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        time.sleep(min(LOCK_POLL_S, remaining))


def require_drive_board_entry(slm: Any) -> str | None:
    """Acquisition-entry drive-board guard from its single home.

    Duck-typed: simulation panels and test doubles lack
    ``require_drive_board()`` and the shared helper treats that as a no-op.
    """
    from ao_shaping.drivers.slm.factory import require_drive_board_entry as _guard

    return _guard(slm)


def frame_centroid(frame: np.ndarray) -> dict[str, float | None]:
    """Descriptive intensity-weighted centroid of the full frame."""
    arr = frame.astype(np.float64)
    total = float(arr.sum())
    if total <= 0:
        return {"row": None, "col": None}  # type: ignore[dict-item]
    height, width = arr.shape
    rows = np.arange(height, dtype=np.float64)[:, None]
    cols = np.arange(width, dtype=np.float64)[None, :]
    return {
        "row": float((rows * arr).sum() / total),
        "col": float((cols * arr).sum() / total),
    }


def camera_identity(cam: Any, args: argparse.Namespace) -> dict[str, Any]:
    serial = getattr(cam, "serial_number", None) or getattr(cam, "sn", None)
    return {
        "cam_type": "daheng",
        "cam_id": int(args.cam_id),
        "serial": None if serial is None else str(serial),
        "bit_depth": getattr(cam, "bit_depth", None),
    }


def build_summary(
    plan: dict[str, Any],
    *,
    run: dict[str, Any],
    device: dict[str, Any],
    release: dict[str, Any],
    frame_records: list[dict[str, Any]],
    per_pattern_frames: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Assemble the solver-facing ``summary.json`` body (pure, offline-testable).

    Only ``frames/`` and ``summary.json`` are read by the frozen solver, so this
    is the whole machine-readable contract of the capture.
    """
    patterns = [
        {
            **record,
            "n_frames_planned": int(record["n_frames"]),
            "n_frames": len(per_pattern_frames.get(record["id"], [])),
            "frames": [
                {
                    "index": frame["index"],
                    "filename": frame["filename"],
                    "frame_sha256": frame["frame_sha256"],
                    "peak": frame["peak"],
                    "median": frame["median"],
                    "saturated_count": frame["saturated_count"],
                    "utc": frame["utc"],
                }
                for frame in per_pattern_frames.get(record["id"], [])
            ],
        }
        for record in plan["patterns"]
    ]
    return {
        "schema": SUMMARY_SCHEMA,
        "run": {
            "run_id": run["run_id"],
            "capture_dir": run["capture_dir"],
            "started_at_utc": run["started_at_utc"],
            "finished_at_utc": utc_now(),
            "status": run["status"],
            "design_source": (
                "docs/dev/hw_window_20261123_attribution_prereg.md §12.14"
            ),
        },
        "status": run["status"],
        "n_frames": len(frame_records),
        "expected_frames": int(plan["n_frames_expected"]),
        "frames_per_pattern": int(plan["frames_per_pattern"]),
        "blaze2d_designs": list(BLAZE2D_PATTERN_IDS),
        "patterns": patterns,
        "exposure_ms": device["exposure_ms"],
        "bit_depth": device["bit_depth"],
        "full_scale": device["full_scale"],
        "settle_s": device["settle_s"],
        "focus_d_m1": FOCUS_D_M1,
        "slm_number": device["slm_number"],
        "slm_drive_board_id": device["drive_board_id"],
        "wavelength_nm": device["wavelength_nm"],
        "camera": device["camera"],
        "encoding_max_gray": ENCODING_MAX_GRAY,
        "closed_normally": release["closed_normally"],
        "elapsed_s": release["elapsed_s"],
        "timestamp_utc": utc_now(),
        "error": run["error"],
    }


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
    per_pattern_frames: dict[str, list[dict[str, Any]]] = {
        pattern_id: [] for pattern_id in CAPTURE_ORDER
    }
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
    #: Filled once the camera answers; stay ``None`` if the run aborts earlier.
    exposure_ms: float | None = None
    actual_bits: int | None = None
    full_scale: float | None = None

    try:
        #: Materialize EVERY pattern's gray array before any device is touched.
        prepared: list[tuple[dict[str, Any], np.ndarray]] = [
            (spec, build_gray(spec, DEFAULT_PANEL_SHAPE))
            for spec in PATTERN_SPECS
        ]

        out.mkdir(parents=True, exist_ok=False)
        created_out = True
        (out / "frames").mkdir()
        write_json(out / "plan.json", plan)
        journal_path = out / "capture_journal.jsonl"
        journal_path.touch()

        from ao_shaping.drivers.ccd.common import (
            create_camera,
            get_camera_exposure_ms,
            set_camera_exposure_ms,
        )
        from ao_shaping.drivers.slm.santec import Santec

        # Exactly ONE SLM open and ONE close for the whole run.
        slm = Santec(slm_number=int(args.slm_number), video_mode=0)
        slm.open()
        drive_board_id = require_drive_board_entry(slm)
        actual_panel_shape = (int(slm.Panel_Res[1]), int(slm.Panel_Res[0]))
        if actual_panel_shape != DEFAULT_PANEL_SHAPE:
            raise RuntimeError(
                f"SLM panel shape {actual_panel_shape} != frozen "
                f"{DEFAULT_PANEL_SHAPE}"
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
        frames_per_pattern = int(args.frames_per_pattern)

        global_index = 0
        with journal_path.open("a", encoding="utf-8") as journal:
            for spec, gray in prepared:
                pattern_id = str(spec["pattern_id"])
                for frame_in_pattern in range(frames_per_pattern):
                    slm.display_data(gray)
                    time.sleep(settle_s)
                    frame = np.asarray(cam.get_numpy_image(1, True)).copy()
                    if frame.shape != NATIVE_CCD_SHAPE:
                        raise RuntimeError(
                            f"expected full-frame {NATIVE_CCD_SHAPE}, "
                            f"got {frame.shape}"
                        )
                    if frame.dtype.kind not in "ui":
                        raise RuntimeError(
                            f"expected integer CCD data, got dtype {frame.dtype}"
                        )
                    peak = int(np.max(frame))
                    median = float(np.median(frame))
                    saturated_count = int(
                        np.count_nonzero(frame >= saturation_threshold)
                    )
                    timestamp = utc_now()
                    filename = frame_filename(global_index, pattern_id)
                    raw_path = out / "frames" / filename
                    np.save(raw_path, frame.astype(np.uint16, copy=False))
                    row = {
                        "index": global_index,
                        "order": int(
                            PATTERN_SPECS.index(spec)
                        ),
                        "pattern_id": pattern_id,
                        "frame_in_pattern": frame_in_pattern,
                        "filename": filename,
                        "frame_sha256": digest(frame),
                        "peak": peak,
                        "median": median,
                        "saturated": saturated_count > 0,
                        "saturated_count": saturated_count,
                        "centroid": frame_centroid(frame),
                        "utc": timestamp,
                        "phase_sha256": next(
                            record["phase_sha256"]
                            for record in plan["patterns"]
                            if record["id"] == pattern_id
                        ),
                        "gray_sha256": next(
                            record["gray_sha256"]
                            for record in plan["patterns"]
                            if record["id"] == pattern_id
                        ),
                        "slm_number": int(args.slm_number),
                        "wavelength_nm": float(args.wavelength_nm),
                        "encoding_max_gray": int(spec["encoding_max_gray"]),
                        "exposure_ms": exposure_ms,
                        "bit_depth": int(actual_bits),
                        "full_scale": int(full_scale),
                        "settle_s": settle_s,
                        "focus_d_m1": FOCUS_D_M1,
                    }
                    frame_records.append(row)
                    per_pattern_frames[pattern_id].append(row)
                    journal.write(
                        json.dumps(
                            {
                                "schema": JOURNAL_SCHEMA,
                                "event": "frame_saved",
                                **row,
                            },
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    journal.flush()

                    # Frozen stop conditions (all lead to the single release
                    # path; no automatic reopen/retry).
                    if (
                        pattern_id == "flat"
                        and global_index == 0
                        and peak < FLAT_FIRST_MIN_PEAK
                    ):
                        raise RuntimeError(
                            f"flat first-frame 0th peak {peak} < "
                            f"{FLAT_FIRST_MIN_PEAK} "
                            "(no light / invalid optical path)"
                        )
                    if saturated_count > 0:
                        raise RuntimeError(
                            f"frame {filename!r} saturated: {saturated_count} "
                            f"pixels >= {saturation_threshold:.0f}"
                        )
                    global_index += 1

        status = "ok"
        exit_code = 0
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("S1d orientation-probe capture failed")
    finally:
        # Single release path, in the frozen order: flatten the SLM, close the
        # camera, close the SLM, release the bench lock.  No reopen/retry.
        if slm is not None:
            try:
                slm.set_grayscale(0)
            except Exception as exc:
                logger.warning("failed to set SLM gray 0 on release: {}", exc)
        if cam is not None:
            try:
                cam.close()
                release["camera_closed"] = True
            except Exception as exc:
                logger.warning("failed to close camera: {}", exc)
        if slm is not None:
            try:
                slm.close()
                release["slm_closed"] = True
            except Exception as exc:
                logger.warning("failed to close SLM: {}", exc)
        try:
            lock.release()
            release["lock_released"] = True
        except Exception as exc:
            logger.warning("failed to release bench lock: {}", exc)
        release["error"] = error
        release["closed_normally"] = bool(
            release["slm_closed"] is True
            and release["camera_closed"] is True
            and release["lock_released"] is True
            and error is None
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
                per_pattern_frames=per_pattern_frames,
            )
            write_json(out / "summary.json", summary)
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "S1d orientation-probe capture; offline plan by default "
            "(§12.14)"
        )
    )
    parser.add_argument(
        "--out", type=Path, help="fresh output directory (required with --execute)"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="open devices and acquire the frozen 28 frames",
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
        "--frames-per-pattern",
        type=int,
        default=DEFAULT_FRAMES_PER_PATTERN,
        help="frames captured per pattern (frozen plan: 2)",
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
        help="fraction of full scale above which a frame counts as saturated",
    )
    args = parser.parse_args(argv)

    if args.execute and args.out is None:
        parser.error("--out is required with --execute")
    if args.exposure_ms <= 0:
        parser.error("--exposure-ms must be positive")
    if args.settle_s <= 0:
        parser.error("--settle-s must be positive")
    if args.frames_per_pattern < 1:
        parser.error("--frames-per-pattern must be >= 1")
    if args.wait_until_idle < 0:
        parser.error("--wait-until-idle must be non-negative")
    if not 0.0 < args.saturation_fraction <= 1.0:
        parser.error("--saturation-fraction must be in (0, 1]")

    plan = build_plan(args)
    if not args.execute:
        print(json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    return execute(args, plan)


if __name__ == "__main__":
    raise SystemExit(main())
