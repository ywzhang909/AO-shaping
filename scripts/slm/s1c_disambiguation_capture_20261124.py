# -*- coding: utf-8 -*-
"""S1c minimal disambiguation capture (preregistered §12.12, frozen before device use).

The frozen plan is 8 frames in one SLM open/close session on SLM #1 (1064 nm),
Daheng camera 0, full-frame Mono12 (full scale 4095), no Fresnel compensation
(``focus_d_m1 = 0.0``), and the script never operates the laser.  Without
``--execute`` this module only builds and prints a deterministic plan; device
drivers are imported only inside ``execute`` after the bench lock is acquired
and the fresh output directory exists.  All phase math comes verbatim from
``scripts.slm.issue247_geometry_capture_20261120`` (never reimplemented here).

Design source: docs/dev/hw_window_20261123_attribution_prereg.md §12.12.2.
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

# Phase math is owned by the issue247 geometry-capture module (the single
# builder home).  This file never reimplements a phase formula.
from scripts.slm.issue247_geometry_capture_20261120 import (
    blaze2d_phase,
    flat_phase,
    grating_phase,
    ladder_phase,
    phase_to_gray,
)

PLAN_SCHEMA = "issue247-s1c-disambiguation-capture/v1"
SUMMARY_SCHEMA = "issue247-s1c-disambiguation-summary/v1"
JOURNAL_SCHEMA = "issue247-s1c-disambiguation-journal/v1"
RELEASE_SCHEMA = "issue247-s1c-disambiguation-released/v1"

DEFAULT_PANEL_SHAPE = (1200, 1920)  # rows, columns
NATIVE_CCD_SHAPE = (1944, 2592)  # full native Daheng frame, never windowed
BIT_DEPTH_REQUIRED = 12
FULL_SCALE_REQUIRED = (1 << BIT_DEPTH_REQUIRED) - 1  # 4095

# The two rad->gray ENCODING scales used by the frozen plan.  1023 is the
# repo-wide default encoding reference (``issue247 ... G_REF``); 993 is the
# physical 2π device reading at 1064 nm used here as a second ENCODING scale
# for the SAME phase array (§12.12.0: it is not the rad->gray default).
ENCODING_1023 = 1023
ENCODING_993 = 993

GRATING_PERIOD_PX = 128
BLAZE_PERIOD_X_PX = 128.0
BLAZE_PERIOD_Y_PX = 192.0
BLAZE_SIGN_X = 1
BLAZE_SIGN_Y = 1
LADDER_AXIS = 1  # panel x
LADDER_SIGN = 1

DEFAULT_EXPOSURE_MS = 0.087
DEFAULT_SETTLE_S = 0.8
DEFAULT_WAIT_IDLE_S = 1800.0
DEFAULT_SLM_NUMBER = 1
DEFAULT_CAM_ID = 0
DEFAULT_WAVELENGTH_NM = 1064.0
DEFAULT_SATURATION_FRACTION = 0.9

#: ``flat_pre`` 0th peak below this means no light / invalid optical path.
FLAT_PRE_MIN_PEAK = 100
LOCK_POLL_S = 1.0
#: No Fresnel compensation is stacked, matching S1b D=0 and A1.
FOCUS_D_M1 = 0.0


# ---------------------------------------------------------------------------
# Frozen frame plan
# ---------------------------------------------------------------------------
def _grating_spec(
    frame_id: str, encoding_max_gray: int, axis: int, period_px: int
) -> dict[str, Any]:
    return {
        "frame_id": frame_id,
        "kind": "grating",
        "encoding_max_gray": int(encoding_max_gray),
        "axis": int(axis),
        "period_px": int(period_px),
    }


#: The frozen 8-frame plan.  Order matters; frame #7 reuses the exact same
#: phase array (and therefore gray array) as #2.
FRAME_SPECS: tuple[dict[str, Any], ...] = (
    {"frame_id": "flat_pre", "kind": "flat", "encoding_max_gray": ENCODING_1023},
    _grating_spec("gx128_e1023", ENCODING_1023, axis=1, period_px=GRATING_PERIOD_PX),
    _grating_spec("gx128_e993", ENCODING_993, axis=1, period_px=GRATING_PERIOD_PX),
    _grating_spec("gy128_e1023", ENCODING_1023, axis=0, period_px=GRATING_PERIOD_PX),
    {
        "frame_id": "blaze2d_px128_py192_pp",
        "kind": "blaze2d",
        "encoding_max_gray": ENCODING_1023,
        "period_x_px": BLAZE_PERIOD_X_PX,
        "period_y_px": BLAZE_PERIOD_Y_PX,
        "sign_x": BLAZE_SIGN_X,
        "sign_y": BLAZE_SIGN_Y,
    },
    {
        "frame_id": "ladder_x",
        "kind": "ladder",
        "encoding_max_gray": ENCODING_1023,
        "axis": LADDER_AXIS,
        "sign": LADDER_SIGN,
    },
    _grating_spec(
        "gx128_e1023_repeat", ENCODING_1023, axis=1, period_px=GRATING_PERIOD_PX
    ),
    {"frame_id": "flat_post", "kind": "flat", "encoding_max_gray": ENCODING_1023},
)

CAPTURE_ORDER: tuple[str, ...] = tuple(spec["frame_id"] for spec in FRAME_SPECS)


def build_phase(spec: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    """Materialize the phase array for one frozen frame spec (raw radians)."""
    kind = spec["kind"]
    if kind == "flat":
        return flat_phase(shape)
    if kind == "grating":
        return grating_phase(
            shape, period_px=int(spec["period_px"]), axis=int(spec["axis"])
        )
    if kind == "blaze2d":
        return blaze2d_phase(
            shape,
            period_x_px=float(spec["period_x_px"]),
            period_y_px=float(spec["period_y_px"]),
            sign_x=int(spec["sign_x"]),
            sign_y=int(spec["sign_y"]),
        )
    if kind == "ladder":
        return ladder_phase(shape, axis=int(spec["axis"]), sign=int(spec["sign"]))
    raise ValueError(f"unknown frame kind {kind!r}")


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


def build_patterns(
    shape: tuple[int, int] = DEFAULT_PANEL_SHAPE,
) -> list[dict[str, Any]]:
    """Materialize every frozen frame's phase/gray digests offline."""
    patterns: list[dict[str, Any]] = []
    for index, spec in enumerate(FRAME_SPECS):
        phase = build_phase(spec, shape)
        gray = phase_to_gray(phase, int(spec["encoding_max_gray"]))
        record: dict[str, Any] = {
            "index": index,
            "frame_id": spec["frame_id"],
            "kind": spec["kind"],
            "encoding_max_gray": int(spec["encoding_max_gray"]),
            "phase_sha256": digest(phase),
            "gray_sha256": digest(gray),
            "panel_shape_rows_cols": [int(shape[0]), int(shape[1])],
            **_gray_unique_record(gray),
        }
        if spec["kind"] == "grating":
            record["axis"] = int(spec["axis"])
            record["period_px"] = int(spec["period_px"])
        elif spec["kind"] == "blaze2d":
            record["period_x_px"] = float(spec["period_x_px"])
            record["period_y_px"] = float(spec["period_y_px"])
            record["sign_x"] = int(spec["sign_x"])
            record["sign_y"] = int(spec["sign_y"])
        elif spec["kind"] == "ladder":
            record["axis"] = int(spec["axis"])
            record["sign"] = int(spec["sign"])
        patterns.append(record)
    return patterns


def _frozen_settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "exposure_ms_requested": float(args.exposure_ms),
        "settle_s": float(args.settle_s),
        "saturation_fraction": float(args.saturation_fraction),
        "saturation_threshold": float(
            args.saturation_fraction * FULL_SCALE_REQUIRED
        ),
        "bit_depth_required": BIT_DEPTH_REQUIRED,
        "full_scale_required": FULL_SCALE_REQUIRED,
        "slm_number": int(args.slm_number),
        "slm_video_mode": 0,
        "cam_type": "daheng",
        "cam_id": int(args.cam_id),
        "wavelength_nm": float(args.wavelength_nm),
        "focus_d_m1": FOCUS_D_M1,
        "panel_shape_rows_cols": [int(DEFAULT_PANEL_SHAPE[0]), int(DEFAULT_PANEL_SHAPE[1])],
        "native_ccd_shape_rows_cols": [
            int(NATIVE_CCD_SHAPE[0]),
            int(NATIVE_CCD_SHAPE[1]),
        ],
        "laser_operated": False,
    }


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": PLAN_SCHEMA,
        "task": "preregistered S1c minimal disambiguation: 8 frames, single session",
        "design_source": "docs/dev/hw_window_20261123_attribution_prereg.md §12.12",
        "capture_order": list(CAPTURE_ORDER),
        "n_frames_expected": len(CAPTURE_ORDER),
        "encoding": {
            "scales_used": [ENCODING_1023, ENCODING_993],
            "default_encoding_max_gray": ENCODING_1023,
            "formula": (
                "gray = mod(rint(mod(phase, 2pi) * max_grayscale / (2pi)), "
                "max_grayscale + 1)"
            ),
            "source": (
                "scripts.slm.issue247_geometry_capture_20261120.phase_to_gray "
                "(G_REF=1023 default; per-frame encoding_max_gray)"
            ),
            "physical_two_pi_reading_note": (
                "993 is the physical 2π device reading at 1064 nm, used here as "
                "a second ENCODING scale for the same phase array (§12.12.0); it "
                "is not the rad->gray encoding default."
            ),
        },
        "focus_compensation": {
            "focus_d_m1": FOCUS_D_M1,
            "compensation_applied": False,
            "fresnel_phase_sha256": digest(
                np.zeros(DEFAULT_PANEL_SHAPE, dtype=np.float64)
            ),
        },
        "capture_settings": _frozen_settings(args),
        "phase_builder_source": (
            "scripts.slm.issue247_geometry_capture_20261120 "
            "(flat_phase / grating_phase / blaze2d_phase / ladder_phase / "
            "phase_to_gray)"
        ),
        "frames": build_patterns(DEFAULT_PANEL_SHAPE),
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
                "bench device lock held by {}", holder.describe() if holder else "unknown"
            )
            logged_holder = True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        time.sleep(min(LOCK_POLL_S, remaining))


def require_drive_board_entry(slm: Any) -> str | None:
    """Acquisition-entry drive-board guard, imported lazily from its single home.

    Duck-typed like the content-axis script: simulation panels and test doubles
    lack ``require_drive_board()`` and the shared helper treats that as a no-op.
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
    frame_records: list[dict[str, Any]] = []
    release: dict[str, Any] = {
        "schema": RELEASE_SCHEMA,
        "slm_closed": False,
        "camera_closed": False,
        "lock_released": False,
        "closed_normally": False,
        "error": None,
    }
    started = time.monotonic()

    #: Materialize every frozen phase/gray array before any device is touched.
    arrays: list[tuple[dict[str, Any], np.ndarray, np.ndarray]] = []
    try:
        out.mkdir(parents=True, exist_ok=False)
        created_out = True
        (out / "raw").mkdir()
        write_json(out / "plan.json", plan)
        journal_path = out / "capture_journal.jsonl"
        journal_path.touch()

        for spec in FRAME_SPECS:
            phase = build_phase(spec, DEFAULT_PANEL_SHAPE)
            gray = phase_to_gray(phase, int(spec["encoding_max_gray"]))
            arrays.append((spec, phase, gray))

        from ao_shaping.drivers.ccd.common import (
            create_camera,
            get_camera_exposure_ms,
            set_camera_exposure_ms,
        )
        from ao_shaping.drivers.slm.santec import Santec

        # Exactly ONE SLM open and ONE close for the whole run.
        slm = Santec(slm_number=int(args.slm_number), video_mode=0)
        slm.open()
        require_drive_board_entry(slm)
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
            raise RuntimeError(f"camera full scale {full_scale} != {FULL_SCALE_REQUIRED}")
        saturation_threshold = float(args.saturation_fraction) * full_scale
        wavelength_nm = float(args.wavelength_nm)
        settle_s = float(args.settle_s)

        with journal_path.open("a", encoding="utf-8") as journal:
            for index, (spec, phase, gray) in enumerate(arrays):
                frame_id = spec["frame_id"]
                slm.display_data(gray)
                time.sleep(settle_s)
                frame = np.asarray(cam.get_numpy_image(1, True)).copy()
                if frame.shape != NATIVE_CCD_SHAPE:
                    raise RuntimeError(
                        f"expected full-frame {NATIVE_CCD_SHAPE}, got {frame.shape}"
                    )
                if frame.dtype.kind not in "ui":
                    raise RuntimeError(
                        f"expected integer CCD data, got dtype {frame.dtype}"
                    )
                peak = int(np.max(frame))
                median = float(np.median(frame))
                saturated_count = int(np.count_nonzero(frame >= saturation_threshold))
                saturated = saturated_count > 0
                timestamp = utc_now()
                raw_path = out / "raw" / f"{frame_id}.npz"
                metadata_path = out / "raw" / f"{frame_id}.json"
                np.savez_compressed(raw_path, frame=frame)
                metadata = {
                    "schema": PLAN_SCHEMA,
                    "frame_id": frame_id,
                    "index": index,
                    "encoding_max_gray": int(spec["encoding_max_gray"]),
                    "kind": spec["kind"],
                    "exposure_ms": exposure_ms,
                    "bit_depth": int(actual_bits),
                    "full_scale": int(full_scale),
                    "saturation_threshold": float(saturation_threshold),
                    "saturated": saturated,
                    "saturated_count": saturated_count,
                    "peak": peak,
                    "median": median,
                    "centroid": frame_centroid(frame),
                    "timestamp_utc": timestamp,
                    "phase_sha256": digest(phase),
                    "gray_sha256": digest(gray),
                    "frame_sha256": digest(frame),
                    "slm_number": int(args.slm_number),
                    "wavelength_nm": wavelength_nm,
                    "camera_identity": camera_identity(cam, args),
                    "settle_s": settle_s,
                    "focus_d_m1": FOCUS_D_M1,
                    "fresnel_phase_sha256": plan["focus_compensation"][
                        "fresnel_phase_sha256"
                    ],
                }
                write_json(metadata_path, metadata)
                row = {
                    **metadata,
                    "raw_file": str(raw_path.relative_to(out)),
                    "metadata_file": str(metadata_path.relative_to(out)),
                }
                frame_records.append(row)
                journal.write(
                    json.dumps(
                        {"schema": JOURNAL_SCHEMA, "event": "frame_saved", **row},
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
                journal.flush()

                # Frozen stop conditions (all lead to the single release path;
                # no automatic reopen/retry).
                if frame_id == "flat_pre" and peak < FLAT_PRE_MIN_PEAK:
                    raise RuntimeError(
                        f"flat_pre 0th peak {peak} < {FLAT_PRE_MIN_PEAK} "
                        "(no light / invalid optical path)"
                    )
                if saturated:
                    raise RuntimeError(
                        f"frame {frame_id!r} saturated: {saturated_count} pixels "
                        f">= {saturation_threshold:.0f}"
                    )

        status = "ok"
        exit_code = 0
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("S1c disambiguation capture failed")
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
            summary = {
                "schema": SUMMARY_SCHEMA,
                "status": status,
                "n_frames": len(frame_records),
                "expected_frames": len(CAPTURE_ORDER),
                "frames": frame_records,
                "peaks": {row["frame_id"]: row["peak"] for row in frame_records},
                "saturation_count": sum(
                    bool(row["saturated"]) for row in frame_records
                ),
                "closed_normally": release["closed_normally"],
                "focus_d_m1": FOCUS_D_M1,
                "slm_number": int(args.slm_number),
                "wavelength_nm": float(args.wavelength_nm),
                "encoding_scales_used": [ENCODING_1023, ENCODING_993],
                "elapsed_s": release["elapsed_s"],
                "timestamp_utc": utc_now(),
                "error": error,
            }
            write_json(out / "summary.json", summary)
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="S1c minimal disambiguation capture; offline plan by default"
    )
    parser.add_argument(
        "--out", type=Path, help="fresh output directory (required with --execute)"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="open devices and acquire the eight frozen frames",
    )
    parser.add_argument(
        "--wait-until-idle",
        type=float,
        default=DEFAULT_WAIT_IDLE_S,
        help="seconds to wait for the bench device lock (default 1800)",
    )
    parser.add_argument(
        "--exposure-ms", type=float, default=DEFAULT_EXPOSURE_MS
    )
    parser.add_argument("--settle-s", type=float, default=DEFAULT_SETTLE_S)
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
    if args.wait_until_idle < 0:
        parser.error("--wait-until-idle must be non-negative")
    if not 0.0 < args.saturation_fraction <= 1.0:
        parser.error("--saturation-fraction must be in (0, 1]")
    if not np.isfinite(args.wavelength_nm) or args.wavelength_nm <= 0:
        parser.error("--wavelength-nm must be positive and finite")
    plan = build_plan(args)
    sys.stdout.write(
        json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    if not args.execute:
        logger.info("plan only; no output directory created and no devices opened")
        return 0
    return execute(args, plan)


if __name__ == "__main__":
    raise SystemExit(main())
