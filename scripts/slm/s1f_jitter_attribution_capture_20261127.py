# -*- coding: utf-8 -*-
"""S1f random-jitter attribution probe -- capture (§3.2, 2026-11-27).

Three arms in ONE SLM #1 open/close session quantify where the residual
~1.6 px jitter comes from:

* arm A (``same_write``): H written ONCE, then 24 frames -- readout/extraction
  noise floor;
* arm B (``repeat_write``): the SAME H bytes rewritten 24 times, 1 frame each
  -- write-screen randomness on top of the floor;
* arm C (``flat_drift``): flat written once at head / mid / tail, 16 frames
  each -- slow optical drift.

Arm A/B H gray bytes must equal ``run_05_postfix`` ``7276d9da...`` (logical
C-order stream actually delivered post-guard) or the run aborts before the
first write.  No laser, no dark.  Frozen builders and shared helpers are
imported, never re-implemented.

Without ``--execute`` the module only builds and prints a deterministic plan:
it creates no directory, imports no driver and never takes the bench lock.

Design source: docs/dev/s1f_jitter_attribution_prereg_20261127.md §2/§3.1/§3.2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
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

SCHEMA = "issue247-s1f-jitter-attribution-capture/v1"
SUMMARY_SCHEMA = "issue247-s1f-jitter-attribution-summary/v1"
RELEASE_SCHEMA = "issue247-s1d-orientation-probe-released/v1"

#: §3.1 -- session constants, frozen from S1e.
PANEL_SHAPE = (1200, 1920)  # rows, columns
NATIVE_CCD_SHAPE = (1944, 2592)
ENCODING_MAX_GRAY = 1023
GRATING_PERIOD_PX = 128
BIT_DEPTH = 12
FULL_SCALE = (1 << BIT_DEPTH) - 1  # 4095 -- literally 4095, never edited
ARM_A_FRAMES = 24
ARM_B_WRITES = 24
FLAT_FRAMES = 16
SETTLE_S = 0.8
DEFAULT_EXPOSURE_MS = 0.05
SATURATION_FRACTION = 0.9
DEFAULT_WAIT_IDLE_S = 1800.0
DEFAULT_SLM_NUMBER = 1
DEFAULT_CAM_ID = 0

#: run_05_postfix journal -- H gray bytes actually delivered to the panel.
FROZEN_H_PHYSICAL_SHA256 = (
    "7276d9da3b2a5ea536d56ea944adcfdcb38c98ad1dd01c1433d1b2219ac8218e"
)

ARM_FLAT = "flat_drift"
ARM_A = "same_write"
ARM_B = "repeat_write"

_FLAT_HEAD = "flat_head"
_FLAT_MID = "flat_mid"
_FLAT_TAIL = "flat_tail"
_ARM_A_IDS: tuple[str, ...] = tuple(f"s1f_a{i:02d}" for i in range(1, ARM_A_FRAMES + 1))
_ARM_B_IDS: tuple[str, ...] = tuple(f"s1f_b{i:02d}" for i in range(1, ARM_B_WRITES + 1))


# ---------------------------------------------------------------------------
# Offline materialization: frozen builders only, no devices, no output dir
# ---------------------------------------------------------------------------
def build_phase(
    array_id: str, shape: tuple[int, int] = PANEL_SHAPE
) -> np.ndarray:
    """Raw phase from the frozen builder; no phase formula lives here."""
    if array_id == "flat":
        return flat_phase(shape)
    if array_id == "H":
        return grating_phase(shape, period_px=GRATING_PERIOD_PX, axis=1)
    raise ValueError(f"unknown array id {array_id!r}")


def build_gray(
    array_id: str, shape: tuple[int, int] = PANEL_SHAPE
) -> np.ndarray:
    """Phase -> uint16 panel gray through the single existing wrap point."""
    return phase_to_gray(build_phase(array_id, shape), ENCODING_MAX_GRAY)


def digest(array: np.ndarray) -> str:
    """Logical digest of the C-order byte stream (S1b/S1c/S1d convention)."""
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes(order="A")
    ).hexdigest()


def physical_digest(array: np.ndarray) -> str:
    """Digest of bytes ACTUALLY sent post-guard (§3.2 byte assertion)."""
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes(order="A")
    ).hexdigest()


def assert_frozen_h_bytes(gray: np.ndarray) -> None:
    """Abort unless the H gray bytes equal the run_05_postfix stream."""
    got = physical_digest(gray)
    if got != FROZEN_H_PHYSICAL_SHA256:
        raise RuntimeError(
            f"H gray byte stream drifted: {got} != {FROZEN_H_PHYSICAL_SHA256}"
        )


def build_arm_layout() -> list[dict[str, Any]]:
    """§3.2 segment table; ``pattern_ids`` is one entry per captured frame."""
    return [
        {
            "segment": 1,
            "id": "flat_head",
            "arm": ARM_FLAT,
            "array": "flat",
            "writes": 1,
            "frames": FLAT_FRAMES,
            "pattern_ids": [_FLAT_HEAD] * FLAT_FRAMES,
        },
        {
            "segment": 2,
            "id": "arm_a",
            "arm": ARM_A,
            "array": "H",
            "writes": 1,
            "frames": ARM_A_FRAMES,
            "pattern_ids": list(_ARM_A_IDS),
        },
        {
            "segment": 3,
            "id": "flat_mid",
            "arm": ARM_FLAT,
            "array": "flat",
            "writes": 1,
            "frames": FLAT_FRAMES,
            "pattern_ids": [_FLAT_MID] * FLAT_FRAMES,
        },
        {
            "segment": 4,
            "id": "arm_b",
            "arm": ARM_B,
            "array": "H",
            "writes": ARM_B_WRITES,
            "frames": ARM_B_WRITES,
            "pattern_ids": list(_ARM_B_IDS),
        },
        {
            "segment": 5,
            "id": "flat_tail",
            "arm": ARM_FLAT,
            "array": "flat",
            "writes": 1,
            "frames": FLAT_FRAMES,
            "pattern_ids": [_FLAT_TAIL] * FLAT_FRAMES,
        },
    ]


def build_frame_plan() -> list[dict[str, Any]]:
    """Deterministic 96-frame schedule; ``write_before`` marks a fresh upload."""
    frames: list[dict[str, Any]] = []
    write_index = 0
    for segment in build_arm_layout():
        pattern_ids = segment["pattern_ids"]
        writes = int(segment["writes"])
        if writes == 1:
            for frame_in_arm, pattern_id in enumerate(pattern_ids):
                frames.append(
                    {
                        "index": len(frames),
                        "pattern_id": pattern_id,
                        "arm": segment["arm"],
                        "array": segment["array"],
                        "segment": segment["id"],
                        "write_index": write_index,
                        "write_before": frame_in_arm == 0,
                        "frame_in_arm": frame_in_arm,
                    }
                )
            write_index += 1
        elif writes == len(pattern_ids):
            for frame_in_arm, pattern_id in enumerate(pattern_ids):
                frames.append(
                    {
                        "index": len(frames),
                        "pattern_id": pattern_id,
                        "arm": segment["arm"],
                        "array": segment["array"],
                        "segment": segment["id"],
                        "write_index": write_index,
                        "write_before": True,
                        "frame_in_arm": frame_in_arm,
                    }
                )
                write_index += 1
        else:
            raise ValueError(
                f"segment {segment['id']!r}: writes {writes} does not match "
                f"frames {len(pattern_ids)}"
            )
    return frames


def build_plan() -> dict[str, Any]:
    """Deterministic plan; no timestamps, no filesystem, no device access."""
    frames = build_frame_plan()
    return {
        "schema": SCHEMA,
        "task": (
            "preregistered S1f random-jitter attribution: 5 segments / "
            "96 frames in one SLM/CCD session"
        ),
        "design_source": (
            "docs/dev/s1f_jitter_attribution_prereg_20261127.md §2/§3.1/§3.2"
        ),
        "arm_layout": build_arm_layout(),
        "n_frames_expected": len(frames),
        "n_writes_expected": sum(1 for entry in frames if entry["write_before"]),
        "frozen_h_physical_sha256": FROZEN_H_PHYSICAL_SHA256,
        "expected_digests": {
            "flat": {
                "phase_sha256": digest(build_phase("flat")),
                "gray_sha256": digest(build_gray("flat")),
            },
            "H": {
                "phase_sha256": digest(build_phase("H")),
                "gray_sha256": physical_digest(build_gray("H")),
            },
        },
        "constants": {
            "panel_shape_rows_cols": list(PANEL_SHAPE),
            "native_ccd_shape_rows_cols": list(NATIVE_CCD_SHAPE),
            "encoding_max_gray": ENCODING_MAX_GRAY,
            "grating_period_px": GRATING_PERIOD_PX,
            "bit_depth": BIT_DEPTH,
            "full_scale": FULL_SCALE,
            "arm_a_frames": ARM_A_FRAMES,
            "arm_b_writes": ARM_B_WRITES,
            "flat_frames": FLAT_FRAMES,
            "settle_s": SETTLE_S,
            "default_exposure_ms": DEFAULT_EXPOSURE_MS,
            "saturation_fraction": SATURATION_FRACTION,
        },
        "encoding": {
            "encoding_max_gray": ENCODING_MAX_GRAY,
            "source": (
                "scripts.slm.issue247_geometry_capture_20261120.phase_to_gray"
            ),
        },
        "capture_settings": {
            "slm_number": DEFAULT_SLM_NUMBER,
            "video_mode": 0,
            "cam_type": "daheng",
            "cam_id": DEFAULT_CAM_ID,
            "bit_depth": BIT_DEPTH,
            "full_scale": FULL_SCALE,
            "settle_s": SETTLE_S,
            "default_exposure_ms": DEFAULT_EXPOSURE_MS,
            "saturation_fraction": SATURATION_FRACTION,
            "laser_operated": False,
            "dark_frames": False,
        },
        "display_topology": {
            "status": "not_probed",
            "reason": "display topology is queried at execute time only",
        },
        "frames": frames,
        "phase_builder_source": (
            "scripts.slm.issue247_geometry_capture_20261120 "
            "(flat_phase / grating_phase / phase_to_gray)"
        ),
    }


# ---------------------------------------------------------------------------
# Device session (only reached with --execute)
# ---------------------------------------------------------------------------
def execute(args: argparse.Namespace, plan: dict[str, Any]) -> int:
    out: Path = args.out
    lock = acquire_device_lock(float(args.wait_until_idle))
    if lock is None:
        logger.error(
            "could not acquire bench device lock within {}s",
            args.wait_until_idle,
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
    exposure_ms: float | None = None
    actual_bits: int | None = None
    full_scale: float | None = None
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

    try:
        # Materialize and verify EVERY byte class before any device is touched.
        phase_by_array = {"flat": build_phase("flat"), "H": build_phase("H")}
        gray_by_array = {"flat": build_gray("flat"), "H": build_gray("H")}
        assert_frozen_h_bytes(gray_by_array["H"])

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

        # Exactly ONE SLM open and ONE close for the whole session.
        slm = Santec(slm_number=int(args.slm_number), video_mode=0)
        slm.open()
        drive_board_id = require_drive_board_entry(slm)
        actual_panel_shape = (int(slm.Panel_Res[1]), int(slm.Panel_Res[0]))
        if actual_panel_shape != PANEL_SHAPE:
            raise RuntimeError(
                f"SLM panel shape {actual_panel_shape} != frozen {PANEL_SHAPE}"
            )
        slm.set_phase_overlay(None)

        cam = create_camera(
            "daheng",
            int(args.cam_id),
            exposure_time_ms=float(args.exposure_ms),
            bit_depth=BIT_DEPTH,
        )
        cam.open()
        actual_bits = getattr(cam, "bit_depth", None)
        if actual_bits != BIT_DEPTH:
            raise RuntimeError(
                f"camera bit-depth readback must be {BIT_DEPTH}, "
                f"got {actual_bits!r}"
            )
        set_camera_exposure_ms(cam, float(args.exposure_ms))
        exposure_ms = float(get_camera_exposure_ms(cam))
        full_scale = float(
            cam.effective_full_scale(np.zeros((1,), dtype=np.uint16))
        )
        if full_scale != float(FULL_SCALE):
            raise RuntimeError(
                f"camera full scale {full_scale} != {FULL_SCALE}"
            )
        saturation_threshold = SATURATION_FRACTION * full_scale

        with journal_path.open("a", encoding="utf-8") as journal:
            for entry in plan["frames"]:
                array_id = str(entry["array"])
                arm = str(entry["arm"])
                pattern_id = str(entry["pattern_id"])
                write_index = int(entry["write_index"])
                gray = gray_by_array[array_id]
                if entry["write_before"]:
                    if array_id == "H":
                        assert_frozen_h_bytes(gray)
                    memory_number = int(slm.display_data(gray))
                    upload = {
                        "schema": SCHEMA,
                        "event": "upload",
                        "write_index": write_index,
                        "arm": arm,
                        "pattern_id": pattern_id,
                        "array": array_id,
                        "memory_number": memory_number,
                        "phase_sha256": digest(phase_by_array[array_id]),
                        "gray_sha256": digest(gray),
                        "physical_sha256": physical_digest(gray),
                        "utc": utc_now(),
                    }
                    upload_records.append(upload)
                    journal.write(
                        json.dumps(
                            upload, ensure_ascii=False, allow_nan=False
                        )
                        + "\n"
                    )
                    journal.flush()
                    time.sleep(SETTLE_S)

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
                index = len(frame_records)
                filename = frame_filename(index, pattern_id)
                np.save(
                    out / "frames" / filename,
                    frame.astype(np.uint16, copy=False),
                )
                row = {
                    "schema": SCHEMA,
                    "event": "frame_saved",
                    "index": index,
                    "pattern_id": pattern_id,
                    "arm": arm,
                    "frame_in_arm": int(entry["frame_in_arm"]),
                    "write_index": write_index,
                    "filename": filename,
                    "frame_sha256": digest(frame),
                    "peak": peak,
                    "median": median,
                    "saturated": saturated_count > 0,
                    "saturated_count": saturated_count,
                    "centroid": frame_centroid(frame),
                    "utc": utc_now(),
                    "exposure_ms": exposure_ms,
                    "bit_depth": int(actual_bits),
                    "full_scale": int(full_scale),
                }
                frame_records.append(row)
                journal.write(
                    json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                )
                journal.flush()

                # Frozen stop condition; never auto-reopen or retry.
                if saturated_count > 0:
                    raise RuntimeError(
                        f"frame {filename!r} saturated: {saturated_count} "
                        f"pixels >= {saturation_threshold:.0f}"
                    )

        status = "ok"
        exit_code = 0
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.error("S1f capture aborted: {}", error)
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
                    f"{release['error']}; camera.close: "
                    f"{type(exc).__name__}: {exc}"
                )
        if lock is not None:
            try:
                lock.release()
                release["lock_released"] = True
            except Exception as exc:
                release["error"] = (
                    f"{release['error']}; lock.release: "
                    f"{type(exc).__name__}: {exc}"
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
            summary = {
                "schema": SUMMARY_SCHEMA,
                "status": status,
                "run_id": out.name,
                "capture_dir": str(out),
                "started_at_utc": started_utc,
                "finished_at_utc": utc_now(),
                "design_source": plan["design_source"],
                "closed_normally": release["closed_normally"],
                "elapsed_s": release["elapsed_s"],
                "error": error,
                "error_release": release["error"],
                "drive_board_id": drive_board_id,
                "slm_number": int(args.slm_number),
                "camera": None if cam is None else camera_identity(cam, args),
                "exposure_ms": exposure_ms,
                "bit_depth": actual_bits,
                "full_scale": full_scale,
                "n_frames_saved": len(frame_records),
                "n_uploads": len(upload_records),
            }
            write_json(out / "summary.json", summary)
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "S1f random-jitter attribution capture; offline plan by default"
        )
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="open devices and acquire the 96-frame attribution probe",
    )
    parser.add_argument(
        "--exposure-ms",
        type=float,
        default=DEFAULT_EXPOSURE_MS,
        help="camera exposure in ms (frozen default 0.05)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="fresh output directory (required with --execute)",
    )
    parser.add_argument(
        "--wait-until-idle",
        type=float,
        default=DEFAULT_WAIT_IDLE_S,
        help="seconds to wait for the bench device lock (default 1800)",
    )
    parser.add_argument("--slm-number", type=int, default=DEFAULT_SLM_NUMBER)
    parser.add_argument("--cam-id", type=int, default=DEFAULT_CAM_ID)
    args = parser.parse_args(argv)

    if args.execute and args.out is None:
        parser.error("--out is required with --execute")
    if args.exposure_ms <= 0:
        parser.error("--exposure-ms must be positive")

    plan = build_plan()
    if not args.execute:
        print(json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    return execute(args, plan)


if __name__ == "__main__":
    raise SystemExit(main())
