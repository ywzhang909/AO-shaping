# -*- coding: utf-8 -*-
"""Offline-plannable capture for the preregistered S1b content × fx design.

The frozen order is A128, A256, B128, B256. Without ``--execute`` this module
only builds and prints a deterministic plan; device drivers are imported only
inside ``execute`` after the bench lock has been acquired.
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


def resolve_max_grayscale() -> int:
    """2π grayscale resolved at runtime from its single source.

    ``utils/slm/gray_scale`` owns the mapping (config ``[slm] two_pi_grayscale``,
    default 1023). It must never be re-declared here, otherwise a configured
    scale would silently not apply to this capture. Wrapped in a function so the
    import stays lazy and the value is re-read per call.
    """
    from ao_shaping.utils.slm.gray_scale import resolve_max_grayscale as _resolve

    return int(_resolve())


PLAN_SCHEMA = "issue247-content-axis-2x2-capture/v1"
SUMMARY_SCHEMA = "issue247-content-axis-2x2-summary/v1"
JOURNAL_SCHEMA = "issue247-content-axis-2x2-journal/v1"
DEFAULT_PANEL_SHAPE = (1200, 1920)  # rows, columns
NATIVE_CCD_SHAPE = (1944, 2592)
PERIODS_PX = (128, 256)
CAPTURE_ORDER = ("A128", "A256", "B128", "B256")
DEFAULT_EXPOSURE_MS = 0.087
DEFAULT_WAIT_IDLE_S = 600.0
DEFAULT_SLM_NUMBER = 1
DEFAULT_CAM_TYPE = "daheng"
DEFAULT_CAM_ID = 0
BIT_DEPTH_REQUIRED = 12
FULL_SCALE_REQUIRED = (1 << BIT_DEPTH_REQUIRED) - 1
# The 2π encoding scale is NOT a constant here: it is resolved per run from
# ``ao_shaping.utils.slm.gray_scale`` (config ``[slm] two_pi_grayscale``).
SATURATION_FRACTION = 0.9
#: Audit fields copied verbatim from each plan pattern into frame/summary records.
ROW0_STEP_KEYS = (
    "row0_step_gray_unique",
    "row0_step_gray_mod",
    "row0_step_radians",
    "other_rows_step_gray_unique",
    "gray_modulus",
)
LOCK_POLL_S = 1.0
#: SLM #1 (1064 nm) — the Fresnel compensation leaf's default too.
FOCUS_WAVELENGTH_NM = 1064.0
#: Scale used to turn the prereg's dioptre D into the leaf's ``focal_mm``.
#: Kept identical to ``focus_probe._measure_point`` (no sign flip).
DIOPTRE_TO_FOCAL_MM = 1000.0
# Frozen from the issue247 axis-map convention; expected displacement = K * N.
K_COL = 6.4
K_ROW = 6.2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def bench_geometry() -> dict[str, Any]:
    """Panel-space Fresnel geometry, read from the single bench constants leaf."""
    from ao_shaping.tools.kernels.bench_kernels import (
        BEAM_CENTER_PANEL,
        BEAM_RADIUS_PANEL,
        SLM_PITCH_M,
    )

    return {
        "pupil_center_px": [
            int(BEAM_CENTER_PANEL[0]),
            int(BEAM_CENTER_PANEL[1]),
        ],
        "radius_px": int(BEAM_RADIUS_PANEL),
        "pixel_um": float(SLM_PITCH_M) * 1e6,
        "wavelength_nm": FOCUS_WAVELENGTH_NM,
    }


def focal_mm_from_diopter(d_m1: float) -> float | None:
    """``1000 / D`` exactly as ``focus_probe._measure_point``; 0 means no lens."""
    d = float(d_m1)
    if d == 0.0:
        return None
    focal_mm = DIOPTRE_TO_FOCAL_MM / d
    if not np.isfinite(focal_mm):
        raise ValueError(f"focus dioptre produces non-finite focal_mm: {d!r}")
    return focal_mm


def compensation_phase(
    d_m1: float, shape: tuple[int, int] = DEFAULT_PANEL_SHAPE
) -> np.ndarray | None:
    """Fresnel panel phase for dioptre ``d_m1``; ``None`` when no compensation.

    The sign is not flipped here: ``focal_mm = 1000 / D`` is the repo-wide
    conversion, and which branch refocuses is what S0-R1 measures.
    """
    focal_mm = focal_mm_from_diopter(d_m1)
    if focal_mm is None:
        return None
    from ao_shaping.tools.kernels.bench_kernels import fresnel_panel

    geometry = bench_geometry()
    return fresnel_panel(
        focal_mm,
        radius=int(geometry["radius_px"]),
        pupil_center=(
            int(geometry["pupil_center_px"][0]),
            int(geometry["pupil_center_px"][1]),
        ),
        panel_shape=(int(shape[0]), int(shape[1])),
        wavelength_nm=float(geometry["wavelength_nm"]),
        pixel_um=float(geometry["pixel_um"]),
    )


def focus_record(
    d_m1: float, shape: tuple[int, int] = DEFAULT_PANEL_SHAPE
) -> dict[str, Any]:
    """Stable schema for the plan/summary: present even when D = 0."""
    focal_mm = focal_mm_from_diopter(d_m1)
    compensation = compensation_phase(d_m1, shape)
    geometry = bench_geometry()
    applied = compensation is not None
    return {
        "focus_d_m1": float(d_m1),
        "focal_mm": None if focal_mm is None else float(focal_mm),
        "compensation_applied": applied,
        "fresnel_phase_sha256": digest(
            compensation
            if compensation is not None
            else np.zeros((int(shape[0]), int(shape[1])), dtype=np.float64)
        ),
        "pupil_center": geometry["pupil_center_px"],
        "radius": geometry["radius_px"],
        "pixel_um": geometry["pixel_um"],
        "wavelength_nm": geometry["wavelength_nm"],
        "panel_shape_rows_cols": [int(shape[0]), int(shape[1])],
        "source": (
            "ao_shaping.tools.kernels.bench_kernels.fresnel_panel; "
            "focal_mm = 1000 / focus_d_m1 (no sign flip)"
        ),
    }


def make_phase(
    content: str,
    period_px: int,
    shape: tuple[int, int],
    d_m1: float = 0.0,
) -> np.ndarray:
    """Create binary fx grating; B differs from A only on panel row zero.

    ``d_m1`` adds the Fresnel compensation phase to the whole panel before the
    single wrap point, so B - A stays exactly pi on row 0 for any D.
    """
    rows, cols = int(shape[0]), int(shape[1])
    if content not in ("A", "B"):
        raise ValueError(f"unknown content {content!r}")
    if period_px not in PERIODS_PX:
        raise ValueError(f"period must be one of {PERIODS_PX}")
    high = (np.arange(cols) % period_px) < (period_px // 2)
    line = np.where(high, np.pi, 0.0)
    phase = np.broadcast_to(line[None, :], (rows, cols)).copy()
    if content == "B":
        phase[0, :] = np.mod(phase[0, :] + np.pi, 2.0 * np.pi)
    compensation = compensation_phase(d_m1, shape)
    if compensation is None:
        return phase.astype(np.float64, copy=False)
    return np.mod(phase + compensation, 2.0 * np.pi).astype(np.float64, copy=False)


def encoding_record(max_grayscale: int) -> dict[str, Any]:
    """Record the phase → gray scale used, so a capture can be re-derived."""
    resolved = int(max_grayscale)
    return {
        "max_grayscale": resolved,
        "two_pi_radians": float(2.0 * np.pi),
        "gray_formula": (
            "gray = mod(rint(mod(phase, 2pi) * max_grayscale / (2pi)), "
            "max_grayscale + 1)"
        ),
        "source": "ao_shaping.utils.slm.gray_scale.resolve_max_grayscale",
    }


def phase_to_gray(phase: np.ndarray, max_grayscale: int) -> np.ndarray:
    """Encode wrapped phase to gray at the caller-supplied 2π scale."""
    scale = int(max_grayscale)
    if scale < 1:
        raise ValueError(f"max_grayscale must be >= 1; got {max_grayscale!r}")
    wrapped = np.mod(np.asarray(phase, dtype=np.float64), 2.0 * np.pi)
    return np.mod(np.rint(wrapped * scale / (2.0 * np.pi)), scale + 1).astype(np.uint16)


def row0_step_record(
    gray_a: np.ndarray, gray_b: np.ndarray, max_grayscale: int
) -> dict[str, Any]:
    """Post-quantisation evidence for content B's row-0 pi step.

    The pi step is exact in radians but not in gray: ``pi`` maps to
    ``max_grayscale / 2``, which is a half-integer for odd scales and rounds to
    the nearest even level. The step is therefore recorded as the single
    wrapped ``B - A`` gray difference, with every unique value kept so a
    non-uniform row is auditable instead of silently averaged.
    """
    scale = int(max_grayscale)
    modulus = scale + 1
    delta = np.mod(gray_b.astype(np.int64) - gray_a.astype(np.int64), modulus)
    row0 = np.unique(delta[0])
    below = np.unique(delta[1:])
    single = int(row0.size) == 1
    step_gray = int(row0[0]) if single else None
    record: dict[str, Any] = {
        "row0_step_gray_unique": [int(value) for value in row0],
        "row0_step_gray_mod": step_gray,
        "row0_step_radians": (
            None if not single else float(step_gray * 2.0 * np.pi / scale)
        ),
        "row0_step_deg": (None if not single else float(step_gray * 360.0 / scale)),
        "other_rows_step_gray_unique": [int(value) for value in below],
        "gray_modulus": modulus,
    }
    if not single:
        logger.warning(
            "row-0 gray step is not one value {} (max_grayscale={}); criterion "
            "unchanged, needs a human decision",
            record["row0_step_gray_unique"],
            scale,
        )
    nonzero_below = [value for value in record["other_rows_step_gray_unique"] if value]
    if nonzero_below:
        logger.warning(
            "rows below row 0 differ between content A and B {}; content B must "
            "modify row 0 only",
            nonzero_below,
        )
    return record


def build_patterns(
    shape: tuple[int, int] = DEFAULT_PANEL_SHAPE,
    d_m1: float = 0.0,
    max_grayscale: int | None = None,
) -> list[dict[str, Any]]:
    if max_grayscale is None:
        max_grayscale = resolve_max_grayscale()
    scale = int(max_grayscale)
    phases: dict[str, np.ndarray] = {}
    grays: dict[str, np.ndarray] = {}
    for pattern_id in CAPTURE_ORDER:
        phase = make_phase(pattern_id[0], int(pattern_id[1:]), shape, d_m1)
        phases[pattern_id] = phase
        grays[pattern_id] = phase_to_gray(phase, scale)
    patterns: list[dict[str, Any]] = []
    for pattern_id in CAPTURE_ORDER:
        content = pattern_id[0]
        period = int(pattern_id[1:])
        phase = phases[pattern_id]
        gray = grays[pattern_id]
        n = float(shape[1]) / period
        patterns.append(
            {
                "id": pattern_id,
                "content": content,
                "period_px": period,
                "axis": "panel_x",
                "frames": 1,
                "phase_sha256": digest(phase),
                "gray_sha256": digest(gray),
                "expected_displacement_px": {
                    "column": K_COL * n,
                    "row": K_ROW * n,
                },
                "expected_primary_axis_if_content_correlated": "column"
                if content == "A"
                else "row",
                "period_count_N": n,
                **row0_step_record(grays[f"A{period}"], grays[f"B{period}"], scale),
            }
        )
    return patterns


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    d_m1 = float(getattr(args, "focus_d_m1", 0.0))
    max_grayscale = resolve_max_grayscale()
    return {
        "schema": PLAN_SCHEMA,
        "task": "preregistered S1b content A/B × fx period 128/256 factorial",
        "design_source": "docs/dev/hw_window_20261123_attribution_prereg.md §2.1",
        "capture_order": list(CAPTURE_ORDER),
        "n_frames_expected": len(CAPTURE_ORDER),
        "frames_per_cell": 1,
        "panel_shape_expected_rows_cols": list(DEFAULT_PANEL_SHAPE),
        "periods_px": list(PERIODS_PX),
        "encoding": encoding_record(max_grayscale),
        "content_definition": {
            "A": "full-frame binary fx grating; every panel row identical",
            "B": "A plus exactly pi phase added modulo 2pi on panel row 0 only",
        },
        "axis_map_constants": {
            "source": "issue247 axis-map convention; frozen at plan time",
            "K_col": K_COL,
            "K_row": K_ROW,
            "displacement_rule": "K_axis * N; N = panel_columns / period_px",
        },
        "focus_compensation": focus_record(d_m1),
        "exposure_ms_requested": float(args.exposure_ms),
        "bit_depth_required": int(args.bit_depth),
        "full_scale_required": (1 << int(args.bit_depth)) - 1,
        "slm_number": int(args.slm_number),
        "cam_type": str(args.cam_type),
        "cam_id": str(args.cam_id),
        "patterns": build_patterns(DEFAULT_PANEL_SHAPE, d_m1, max_grayscale),
        "generated_at_utc": utc_now(),
    }


def run_self_test() -> int:
    scale = resolve_max_grayscale()
    patterns = build_patterns(DEFAULT_PANEL_SHAPE, 0.0, scale)
    assert [item["id"] for item in patterns] == list(CAPTURE_ORDER)
    assert len(patterns) == 4
    for period in PERIODS_PX:
        for d_m1 in (0.0, 0.6):
            phase_a = make_phase("A", period, DEFAULT_PANEL_SHAPE, d_m1)
            phase_b = make_phase("B", period, DEFAULT_PANEL_SHAPE, d_m1)
            delta = np.mod(phase_b[0] - phase_a[0], 2.0 * np.pi)
            assert np.allclose(delta, np.pi, rtol=0.0, atol=1e-9), (period, d_m1)
            assert np.array_equal(phase_b[1:, :], phase_a[1:, :]), (period, d_m1)
            gray_a = phase_to_gray(phase_a, scale)
            gray_b = phase_to_gray(phase_b, scale)
            step = row0_step_record(gray_a, gray_b, scale)
            assert step["row0_step_gray_unique"] == [step["row0_step_gray_mod"]], (
                period,
                d_m1,
                step,
            )
            assert step["other_rows_step_gray_unique"] == [0], (period, d_m1, step)
            assert abs(step["row0_step_gray_mod"] - scale / 2.0) <= 1.0, (
                period,
                d_m1,
                step,
            )
        phase_a = make_phase("A", period, DEFAULT_PANEL_SHAPE)
        assert np.array_equal(phase_a, np.broadcast_to(phase_a[0], phase_a.shape))
    assert all(
        len(p["phase_sha256"]) == 64 and len(p["gray_sha256"]) == 64 for p in patterns
    )
    zero = focus_record(0.0)
    assert zero["compensation_applied"] is False and zero["focal_mm"] is None
    assert zero["fresnel_phase_sha256"] == digest(
        np.zeros(DEFAULT_PANEL_SHAPE, dtype=np.float64)
    )
    focused = focus_record(0.6)
    assert focused["compensation_applied"] is True
    assert abs(focused["focal_mm"] - DIOPTRE_TO_FOCAL_MM / 0.6) < 1e-9
    logger.info("S1b offline self-test passed: 4 frozen cells, row-0 pi-step verified")
    return 0


def require_drive_board_entry(slm: Any) -> str | None:
    """Acquisition-entry drive-board guard, imported lazily from its single home.

    ``utils``-style duck typing is intentional: simulation panels and test
    doubles lack ``require_drive_board()`` and the shared helper treats that as
    a no-op. A direct ``Santec.require_drive_board()`` call would break those
    paths, so this wrapper is the only hook this script uses.
    """
    from ao_shaping.drivers.slm.factory import require_drive_board_entry as _guard

    return _guard(slm)


def open_slm_guarded(slm: Any) -> str | None:
    """Open the SLM and assert its drive board answers before any phase write."""
    slm.open()
    return require_drive_board_entry(slm)


def acquire_device_lock(wait_s: float) -> Any | None:
    from ao_shaping.tools.kernels.device_mutex import DeviceLock

    lock = DeviceLock.default()
    deadline = time.monotonic() + max(0.0, float(wait_s))
    while True:
        if lock.acquire(timeout=0.0):
            return lock
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        time.sleep(min(LOCK_POLL_S, remaining))


def execute(args: argparse.Namespace, plan: dict[str, Any]) -> int:
    out: Path = args.out
    lock = acquire_device_lock(args.wait_until_idle)
    if lock is None:
        logger.error(
            "could not acquire bench device lock within {}s", args.wait_until_idle
        )
        return 2

    slm: Any = None
    cam: Any = None
    output_created = False
    frame_records: list[dict[str, Any]] = []
    status = "error"
    error: str | None = None
    exit_code = 1
    release: dict[str, Any] = {
        "schema": "issue247-content-axis-2x2-released/v1",
        "slm_closed": False,
        "camera_closed": False,
        "lock_released": False,
        "timestamp_utc": utc_now(),
        "error": None,
    }
    started = time.monotonic()
    focus = dict(plan["focus_compensation"])
    d_m1 = float(focus["focus_d_m1"])
    if not focus["compensation_applied"]:
        logger.warning("离焦补偿未叠加，结果不满足预注册 §2.1 通用规则")
    try:
        out.mkdir(parents=True, exist_ok=False)
        output_created = True
        (out / "raw").mkdir()
        write_json(out / "plan.json", plan)
        journal_path = out / "capture_journal.jsonl"
        journal_path.touch()

        # All four phase and gray arrays are materialized before device capture.
        max_grayscale = int(plan["encoding"]["max_grayscale"])
        patterns = build_patterns(DEFAULT_PANEL_SHAPE, d_m1, max_grayscale)
        arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for item in patterns:
            phase = make_phase(
                item["content"], item["period_px"], DEFAULT_PANEL_SHAPE, d_m1
            )
            arrays[item["id"]] = (phase, phase_to_gray(phase, max_grayscale))

        from ao_shaping.drivers.ccd.common import (
            create_camera,
            get_camera_exposure_ms,
            set_camera_exposure_ms,
        )
        from ao_shaping.drivers.slm.santec import Santec

        slm = Santec(slm_number=args.slm_number, video_mode=0)
        open_slm_guarded(slm)
        actual_panel_shape = (int(slm.Panel_Res[1]), int(slm.Panel_Res[0]))
        if actual_panel_shape != DEFAULT_PANEL_SHAPE:
            raise RuntimeError(
                f"SLM panel shape {actual_panel_shape} != frozen {DEFAULT_PANEL_SHAPE}"
            )
        slm.set_phase_overlay(None)

        cam = create_camera(
            args.cam_type,
            args.cam_id,
            exposure_time_ms=float(args.exposure_ms),
            bit_depth=int(args.bit_depth),
        )
        cam.open()
        actual_bits = getattr(cam, "bit_depth", None)
        if (
            actual_bits != BIT_DEPTH_REQUIRED
            or int(args.bit_depth) != BIT_DEPTH_REQUIRED
        ):
            raise RuntimeError(
                f"camera bit-depth readback/request must be 12, got {actual_bits!r}/{args.bit_depth}"
            )
        set_camera_exposure_ms(cam, float(args.exposure_ms))
        exposure_ms = float(get_camera_exposure_ms(cam))
        full_scale = float(cam.effective_full_scale(np.zeros((1,), dtype=np.uint16)))
        if full_scale != float(FULL_SCALE_REQUIRED):
            raise RuntimeError(
                f"camera full scale {full_scale} != {FULL_SCALE_REQUIRED}"
            )

        with journal_path.open("a", encoding="utf-8") as journal:
            for index, item in enumerate(patterns):
                pattern_id = item["id"]
                phase, gray = arrays[pattern_id]
                slm.display_data(gray)
                time.sleep(0.8)
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
                saturated = bool(np.any(frame >= SATURATION_FRACTION * full_scale))
                timestamp = utc_now()
                raw_path = out / "raw" / f"{pattern_id}.npz"
                metadata_path = out / "raw" / f"{pattern_id}.json"
                np.savez_compressed(raw_path, frame=frame)
                metadata = {
                    "pattern_id": pattern_id,
                    "exposure_ms": exposure_ms,
                    "actual_bit_depth": int(actual_bits),
                    "full_scale": int(full_scale),
                    "peak": peak,
                    "saturated": saturated,
                    "timestamp_utc": timestamp,
                    "frame_sha256": digest(frame),
                    "phase_sha256": digest(phase),
                    "gray_sha256": digest(gray),
                    "focus_d_m1": d_m1,
                    "fresnel_phase_sha256": focus["fresnel_phase_sha256"],
                    **{key: item[key] for key in ROW0_STEP_KEYS},
                }
                write_json(metadata_path, metadata)
                row = {
                    "index": index,
                    **metadata,
                    "raw_file": str(raw_path.relative_to(out)),
                    "metadata_file": str(metadata_path.relative_to(out)),
                }
                frame_records.append(row)
                journal.write(
                    json.dumps(
                        {"schema": JOURNAL_SCHEMA, "event": "frame_saved", **row},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                journal.flush()
        status = "ok"
        exit_code = 0
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("content-axis capture failed")
    finally:
        if cam is not None:
            try:
                cam.close()
                release["camera_closed"] = True
            except Exception as exc:
                logger.warning("failed to close camera: {}", exc)
        if slm is not None:
            try:
                slm.set_grayscale(0)
            except Exception as exc:
                logger.warning("failed to set SLM gray 0 on release: {}", exc)
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
        release["closed_normally"] = all(
            release[key] is True
            for key in ("slm_closed", "camera_closed", "lock_released")
        )
        release["elapsed_s"] = time.monotonic() - started
        release["released_at_utc"] = utc_now()
        if output_created:
            write_json(out / "released.json", release)
            summary = {
                "schema": SUMMARY_SCHEMA,
                "status": status,
                "n_frames": len(frame_records),
                "expected_frames": len(CAPTURE_ORDER),
                "frames": frame_records,
                "saturation_count": sum(
                    bool(row["saturated"]) for row in frame_records
                ),
                "focus_compensation": focus,
                "encoding": dict(plan["encoding"]),
                "row0_step_by_pattern": {
                    item["id"]: {key: item[key] for key in ROW0_STEP_KEYS}
                    for item in plan["patterns"]
                },
                "elapsed_s": release["elapsed_s"],
                "timestamp_utc": utc_now(),
                "error": error,
            }
            write_json(out / "summary.json", summary)
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="S1b 2x2 content-axis capture; offline plan by default"
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="fresh output directory (required for normal planning/capture)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="open devices and acquire the four frozen frames",
    )
    parser.add_argument(
        "--self-test", action="store_true", help="run offline invariants only"
    )
    parser.add_argument("--exposure-ms", type=float, default=DEFAULT_EXPOSURE_MS)
    parser.add_argument(
        "--focus-d-m1",
        type=float,
        default=0.0,
        help=(
            "defocus compensation dioptre D (m^-1) measured in THIS window "
            "(S0-R1); 0 = no Fresnel compensation stacked"
        ),
    )
    parser.add_argument("--bit-depth", type=int, default=BIT_DEPTH_REQUIRED)
    parser.add_argument("--wait-until-idle", type=float, default=DEFAULT_WAIT_IDLE_S)
    parser.add_argument("--slm-number", type=int, default=DEFAULT_SLM_NUMBER)
    parser.add_argument("--cam-type", default=DEFAULT_CAM_TYPE)
    parser.add_argument("--cam-id", default=DEFAULT_CAM_ID)
    args = parser.parse_args(argv)
    if args.self_test:
        return run_self_test()
    if args.out is None:
        parser.error("--out is required unless --self-test is used")
    if args.exposure_ms <= 0:
        parser.error("--exposure-ms must be positive")
    if args.bit_depth != BIT_DEPTH_REQUIRED:
        parser.error(f"--bit-depth must be {BIT_DEPTH_REQUIRED}")
    if args.wait_until_idle < 0:
        parser.error("--wait-until-idle must be non-negative")
    if not np.isfinite(args.focus_d_m1):
        parser.error("--focus-d-m1 must be finite")
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
