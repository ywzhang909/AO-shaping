# -*- coding: utf-8 -*-
"""A1 0 级对齐探针：单图样平场，反复测当前 0 级质心（离线可规划）。

为什么需要它（2026-11-26 实测）
-------------------------------
§12.11.5 六帧采集（``scripts/slm/capture_s1_mapping_disambiguation_20261124.py``）
的冻结判据①要求 ``flat_scale993`` 复现 A1 0 级参考 (row 1105.249, col
1227.556)，偏移超过 5 px 即判「本次会话与 A1 不同束流、六帧全部作废」。11-26
实验系统采集实测质心 (1018.9, 1273.0)，偏移 [-86.4, +45.5] px，判据①证伪，六帧作废。
根因是光路与 A1 不同束流（物理对齐），不是脚本 bug。重跑六帧代价很高，因此需要
一个轻量探针：操作员在物理调光路时反复执行它，只看当前 0 级质心与 5 px 容差，
确认进入容差后再跑完整六帧重采。

口径（禁止重写公式）
---------------------
* 质心：直接从 S1 脚本导入 ``_centroid_px``（baseline-subtracted centroid）。
  S1 的 ``flat_centroid_offset_px`` 用的就是它，本探针必须同口径，否则与判据①
  不可比。
* 相位/灰度：``issue247_geometry_capture_20261120.flat_phase`` + ``phase_to_gray``；
  平场 2π 灰度 scale = 993（设备读数），``gray_modulus`` = 994。本文件不自造相位。
* 冻结参考常量只在本文件内定义，不改 S1 脚本。

范围声明：本探针**只判读 0 级质心**（S1 判据①）。它不替代六帧判据②-⑥，也不写
任何六帧汇总；六帧是否重采仍由操作员按冻结协议决定。

不做的事：不操作激光、不叠加离焦补偿（``flat_phase`` 无 Fresnel 面板）。没有
``--execute`` 时只打印确定性 plan；设备驱动只在 ``execute`` 里、拿到台面锁且新输出
目录就绪之后才惰性导入。

退出码（``--execute``）：0 = 测到且 within_tolerance；3 = 测到但超出 5 px 容差
（测量本身有效，只是还没对齐）；2 = 等不到台面锁；1 = 其它错误。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "src", ROOT / "libs", ROOT, Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# The S1 module owns the centroid convention used by criterion ①.  Import it
# (and its phase builder) rather than restating either formula here.
from scripts.slm.capture_s1_mapping_disambiguation_20261124 import (
    _centroid_px,
    utc_now,
)
from scripts.slm.issue247_geometry_capture_20261120 import flat_phase, phase_to_gray

PLAN_SCHEMA = "issue247-a1-zero-order-alignment-probe-plan/v1"
JOURNAL_SCHEMA = "issue247-a1-zero-order-alignment-probe-journal/v1"
MEASUREMENT_SCHEMA = "issue247-a1-zero-order-alignment-probe-measurement/v1"
RELEASE_SCHEMA = "issue247-a1-zero-order-alignment-probe-released/v1"

DEFAULT_PANEL_SHAPE = (1200, 1920)  # rows, cols (driver Panel_Res is cols, rows)
NATIVE_CCD_SHAPE = (1944, 2592)
BIT_DEPTH_REQUIRED = 12
FULL_SCALE_REQUIRED = (1 << BIT_DEPTH_REQUIRED) - 1
DEFAULT_EXPOSURE_MS = 0.087
DEFAULT_WAIT_IDLE_S = 600.0
DEFAULT_SLM_NUMBER = 1
DEFAULT_CAM_TYPE = "daheng"
DEFAULT_CAM_ID = 0
DEFAULT_FRAMES = 3
DEFAULT_SETTLE_S = 0.8
PREFLIGHT_DISCARD_FRAMES = 3
PREFLIGHT_MEASURE_FRAMES = 1
SATURATION_FRACTION = 0.9
LOCK_POLL_S = 1.0

#: 平场 2π 灰度（设备读数 993 @1064 nm）；wrap 模数为 scale + 1。
TWO_PI_GRAY_SCALE_DEVICE = 993
GRAY_MODULUS = TWO_PI_GRAY_SCALE_DEVICE + 1

#: A1 平场 0 级参考（issue247 A1 帧质心，用户冻结，与 S1 判据①同值）。
ZERO_ORDER_REFERENCE_PX = {"row": 1105.249, "col": 1227.556}
TOLERANCE_PX = 5.0
REFERENCE_SOURCE = (
    "issue247 A1 平场 0 级质心（用户冻结）；与 §12.11.5 六帧判据①同一参考点"
)
SCOPE_NOTE = (
    "本探针只判读 0 级质心是否进入 5 px 容差（S1 判据①），不替代六帧判据②-⑥，"
    "也不产出六帧汇总；六帧是否重采由操作员按冻结协议决定。"
)
CRITERION_TEXT = (
    "flat_scale993 的 0 级质心相对 A1 参考 (row 1105.249, col 1227.556) 的 "
    "欧氏距离 dist = hypot(drow, dcol) <= 5.0 px 判 within_tolerance；"
    "dist > 5.0 px 表示当前束流与 A1 不同，六帧仍会作废。"
)
#: 2026-11-26 实测质心，写进 plan 作为复现判据①证伪的记录（仅供操作员参照）。
OBSERVED_20261126_CENTROID_PX = {"row": 1018.9, "col": 1273.0}


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _journal(handle: Any, event: str, **fields: Any) -> None:
    handle.write(
        json.dumps(
            {
                "schema": JOURNAL_SCHEMA,
                "event": event,
                "timestamp_utc": utc_now(),
                **fields,
            },
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )
        + "\n"
    )
    handle.flush()


def materialize_flat_gray(shape: tuple[int, int] = DEFAULT_PANEL_SHAPE) -> np.ndarray:
    """Flat (zero-radian) phase encoded at the device 2π gray scale."""
    phase = flat_phase((int(shape[0]), int(shape[1])))
    return phase_to_gray(phase, TWO_PI_GRAY_SCALE_DEVICE)


@lru_cache(maxsize=4)
def _flat_pattern_cached(shape: tuple[int, int]) -> dict[str, Any]:
    """Hash the flat pattern once per panel shape; keep only the record."""
    shape = (int(shape[0]), int(shape[1]))
    gray = materialize_flat_gray(shape)
    unique = np.unique(gray)
    record: dict[str, Any] = {
        "id": "flat_scale993",
        "a1_pattern_id": "flat",
        "kind": "flat",
        "role": "a1_zero_order_reference_reproduction",
        "two_pi_gray_scale": int(TWO_PI_GRAY_SCALE_DEVICE),
        "gray_modulus": int(GRAY_MODULUS),
        "phase_sha256": digest(flat_phase(shape)),
        "gray_sha256": digest(gray),
        "gray_max": int(gray.max()) if gray.size else 0,
        "gray_min": int(gray.min()) if gray.size else 0,
        "gray_unique_count": int(unique.size),
        "panel_shape_rows_cols": [shape[0], shape[1]],
        "focus_d_m1": 0.0,
        "compensation_applied": False,
    }
    return record


def flat_pattern(shape: tuple[int, int] = DEFAULT_PANEL_SHAPE) -> dict[str, Any]:
    """Copy of the frozen single-pattern record (callers may mutate safely)."""
    return dict(_flat_pattern_cached((int(shape[0]), int(shape[1]))))


def offset_from_reference(centroid_px: list[float] | tuple[float, float]) -> dict[str, Any]:
    """Pure verdict: offset of one centroid from the frozen A1 reference.

    ``dist_px <= TOLERANCE_PX`` is within tolerance (boundary inclusive), matching
    criterion ① ("偏移超过 5 px" fails).
    """
    row = float(centroid_px[0])
    col = float(centroid_px[1])
    drow = row - float(ZERO_ORDER_REFERENCE_PX["row"])
    dcol = col - float(ZERO_ORDER_REFERENCE_PX["col"])
    dist = float(np.hypot(drow, dcol))
    return {
        "drow_px": drow,
        "dcol_px": dcol,
        "dist_px": dist,
        "within_tolerance": bool(dist <= TOLERANCE_PX),
    }


def mean_centroid(centroids: list[list[float]]) -> list[float]:
    """Arithmetic mean of per-frame [row, col] centroids."""
    if not centroids:
        raise ValueError("mean_centroid requires at least one frame")
    return [float(value) for value in np.asarray(centroids, dtype=np.float64).mean(axis=0)]


def counter_criteria() -> dict[str, Any]:
    """Frozen criterion ① restated for the single-pattern probe."""
    return {
        "zero_order_reference_px": dict(ZERO_ORDER_REFERENCE_PX),
        "reference_source": REFERENCE_SOURCE,
        "tolerance_px": TOLERANCE_PX,
        "boundary": "dist_px <= 5.0 within; dist_px > 5.0 fails criterion ①",
        "criterion_text": CRITERION_TEXT,
        "scope_note": SCOPE_NOTE,
        "observed_20261126_centroid_px": dict(OBSERVED_20261126_CENTROID_PX),
        "observed_20261126_offset_px": [
            OBSERVED_20261126_CENTROID_PX["row"] - ZERO_ORDER_REFERENCE_PX["row"],
            OBSERVED_20261126_CENTROID_PX["col"] - ZERO_ORDER_REFERENCE_PX["col"],
        ],
        "observed_20261126_within_tolerance": False,
    }


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Deterministic, hardware-free plan (frozen reference + pattern + devices)."""
    frames = int(args.frames)
    if frames < 1:
        raise ValueError("--frames must be >= 1")
    exposure = float(args.exposure_ms)
    wait_idle = float(args.wait_until_idle)
    settle_s = float(args.settle_s)
    if settle_s < 0.0:
        raise ValueError("--settle-s must be >= 0")

    pattern = flat_pattern(DEFAULT_PANEL_SHAPE)
    return {
        "schema": PLAN_SCHEMA,
        "generated_at_utc": utc_now(),
        "purpose": "在物理调光路时反复测量当前 0 级质心，判断是否进入判据①的 5 px 容差",
        "counter_criteria": counter_criteria(),
        "pattern": pattern,
        "n_frames": frames,
        "frames_note": "连续采 N 帧同一平场图样，逐帧质心取均值后再判读",
        "settle_s": settle_s,
        "exposure_ms_requested": exposure,
        "bit_depth_required": BIT_DEPTH_REQUIRED,
        "full_scale_required": FULL_SCALE_REQUIRED,
        "panel_shape_expected_rows_cols": [
            DEFAULT_PANEL_SHAPE[0],
            DEFAULT_PANEL_SHAPE[1],
        ],
        "native_ccd_shape_rows_cols": [NATIVE_CCD_SHAPE[0], NATIVE_CCD_SHAPE[1]],
        "preflight": {
            "discard_frames": PREFLIGHT_DISCARD_FRAMES,
            "measure_frames": PREFLIGHT_MEASURE_FRAMES,
            "abort_on_saturation": True,
            "saturation_fraction": SATURATION_FRACTION,
        },
        "acquisition": {
            "slm_number": int(args.slm_number),
            "cam_type": str(args.cam_type),
            "cam_id": int(args.cam_id),
            "exposure_ms": exposure,
            "wait_until_idle_s": wait_idle,
            "settle_s": settle_s,
            "n_frames": frames,
        },
        "panel_grayscale_config_untouched": True,
        "centroid_source": (
            "scripts.slm.capture_s1_mapping_disambiguation_20261124._centroid_px"
        ),
        "exit_codes": {
            "0": "measured, within_tolerance",
            "3": "measured, outside 5 px tolerance",
            "2": "bench lock not acquired",
            "1": "error",
        },
        "note": (
            "平场 phase 为全零、2π 灰度 scale=993（gray_modulus=994），不改面板 LUT；"
            + SCOPE_NOTE
        ),
    }


def _panel_shape_from(slm: Any) -> tuple[int, int]:
    """(rows, cols) from the driver's Panel_Res, which is (cols, rows)."""
    panel = getattr(slm, "Panel_Res", None)
    if not panel or len(panel) != 2:
        logger.warning(
            "SLM Panel_Res missing/odd ({!r}); assuming frozen {}",
            panel,
            DEFAULT_PANEL_SHAPE,
        )
        return DEFAULT_PANEL_SHAPE
    return int(panel[1]), int(panel[0])


def run_preflight(cam: Any, full_scale: float, journal: Any) -> dict[str, Any]:
    """Discard N frames then measure one; caller aborts before science frames."""
    for _ in range(PREFLIGHT_DISCARD_FRAMES):
        cam.get_numpy_image(1, True)
    frame = np.asarray(cam.get_numpy_image(1, True)).copy()
    peak = int(frame.max()) if frame.size else 0
    threshold = SATURATION_FRACTION * full_scale
    saturated = bool(peak >= threshold)
    record: dict[str, Any] = {
        "discard_frames": PREFLIGHT_DISCARD_FRAMES,
        "measure_frames": PREFLIGHT_MEASURE_FRAMES,
        "peak_counts": peak,
        "saturation_threshold_counts": threshold,
        "saturation_fraction": SATURATION_FRACTION,
        "saturated": saturated,
    }
    _journal(journal, "preflight", **record)
    return record


def execute(args: argparse.Namespace, plan: dict[str, Any]) -> int:
    """Acquire bench lock, open devices once, measure N flat frames."""
    out: Path = args.out
    lock = acquire_device_lock(float(args.wait_until_idle))
    if lock is None:
        logger.error(
            "could not acquire bench device lock within {}s", args.wait_until_idle
        )
        return 2

    slm: Any = None
    cam: Any = None
    journal: Any = None
    output_created = False
    frame_records: list[dict[str, Any]] = []
    centroids: list[list[float]] = []
    mean: list[float] | None = None
    verdict: dict[str, Any] | None = None
    preflight_record: dict[str, Any] | None = None
    status = "error"
    error: str | None = None
    exit_code = 1
    release: dict[str, Any] = {
        "schema": RELEASE_SCHEMA,
        "slm_closed": False,
        "camera_closed": False,
        "lock_released": False,
        "acquired_at_utc": utc_now(),
        "finished_at_utc": None,
        "released_at_utc": None,
        "elapsed_s": None,
        "status": status,
        "error": None,
    }
    started = time.monotonic()
    focus = {
        "focus_d_m1": 0.0,
        "compensation_applied": False,
        "focal_mm": None,
        "note": "平场探针不叠加离焦补偿（与 S1 flat_scale993 一致）",
    }
    panel_shape = DEFAULT_PANEL_SHAPE
    bit_depth = BIT_DEPTH_REQUIRED
    exposure_actual: float | None = None
    full_scale = float(FULL_SCALE_REQUIRED)
    try:
        out.mkdir(parents=True, exist_ok=False)
        (out / "raw").mkdir()
        output_created = True
        write_json(out / "plan.json", plan)
        journal = (out / "capture_journal.jsonl").open("a", encoding="utf-8")
        _journal(
            journal,
            "lock_acquired",
            wait_until_idle_s=float(args.wait_until_idle),
        )

        gray = materialize_flat_gray(DEFAULT_PANEL_SHAPE)
        if digest(gray) != plan["pattern"]["gray_sha256"]:
            raise RuntimeError("flat gray integrity check failed")

        from ao_shaping.drivers.ccd.common import (
            create_camera,
            get_camera_exposure_ms,
            set_camera_exposure_ms,
        )
        from ao_shaping.drivers.slm.santec import Santec

        slm = Santec(slm_number=int(args.slm_number), video_mode=0)
        open_slm_guarded(slm)
        panel_shape = _panel_shape_from(slm)
        if panel_shape != DEFAULT_PANEL_SHAPE:
            raise RuntimeError(
                f"expected panel {DEFAULT_PANEL_SHAPE}, got {panel_shape}"
            )
        slm.set_phase_overlay(None)

        cam = create_camera(
            str(args.cam_type),
            int(args.cam_id),
            exposure_time_ms=float(args.exposure_ms),
            bit_depth=BIT_DEPTH_REQUIRED,
        )
        cam.open()
        bit_depth = int(getattr(cam, "bit_depth", BIT_DEPTH_REQUIRED))
        if bit_depth != BIT_DEPTH_REQUIRED:
            raise RuntimeError(
                f"camera bit-depth must be {BIT_DEPTH_REQUIRED}, got {bit_depth!r}"
            )
        set_camera_exposure_ms(cam, float(args.exposure_ms))
        exposure_actual = float(get_camera_exposure_ms(cam))
        full_scale = float(
            cam.effective_full_scale(np.zeros((1,), dtype=np.uint16))
        )
        if full_scale != float(FULL_SCALE_REQUIRED):
            raise RuntimeError(f"camera full scale {full_scale} != {FULL_SCALE_REQUIRED}")

        _journal(
            journal,
            "devices_ready",
            slm_number=int(args.slm_number),
            cam_type=str(args.cam_type),
            cam_id=int(args.cam_id),
            exposure_ms_requested=float(args.exposure_ms),
            exposure_ms_actual=exposure_actual,
            bit_depth=bit_depth,
            full_scale=int(full_scale),
            panel_shape_rows_cols=list(panel_shape),
        )
        preflight_record = run_preflight(cam, full_scale, journal)
        if preflight_record["saturated"]:
            raise RuntimeError(
                "preflight saturated: peak "
                f"{preflight_record['peak_counts']} >= "
                f"{preflight_record['saturation_threshold_counts']}；"
                "先降低曝光/光强再测"
            )

        slm.display_data(gray)
        for index in range(int(plan["n_frames"])):
            time.sleep(float(args.settle_s))
            frame = np.asarray(cam.get_numpy_image(1, True)).copy()
            if frame.shape != NATIVE_CCD_SHAPE:
                raise RuntimeError(
                    f"expected full-frame {NATIVE_CCD_SHAPE}, got {frame.shape}"
                )
            if frame.dtype.kind not in "ui":
                raise RuntimeError(
                    f"expected integer CCD data, got dtype {frame.dtype}"
                )
            saturated = bool(frame.max() >= SATURATION_FRACTION * full_scale)
            centroid = _centroid_px(frame)
            centroids.append(centroid)
            raw_path = out / "raw" / f"{index:02d}_flat.npy"
            np.save(raw_path, frame)
            row: dict[str, Any] = {
                "index": index,
                "pattern_id": plan["pattern"]["id"],
                "centroid_px": centroid,
                "offset_from_reference": offset_from_reference(centroid),
                "frame_sha256": digest(frame),
                "frame_shape": list(frame.shape),
                "frame_max_counts": int(frame.max()),
                "frame_mean_counts": float(frame.mean()),
                "saturated": saturated,
                "bit_depth": bit_depth,
                "full_scale": int(full_scale),
                "exposure_ms_requested": float(args.exposure_ms),
                "exposure_ms_actual": exposure_actual,
                "panel_shape_rows_cols": list(panel_shape),
                "raw_file": raw_path.relative_to(out).as_posix(),
                "saved_at_utc": utc_now(),
            }
            frame_records.append(row)
            _journal(journal, "frame", **row)
            _journal(
                journal,
                "centroid",
                index=index,
                centroid_row_px=centroid[0],
                centroid_col_px=centroid[1],
                drow_px=row["offset_from_reference"]["drow_px"],
                dcol_px=row["offset_from_reference"]["dcol_px"],
                dist_px=row["offset_from_reference"]["dist_px"],
                within_tolerance=row["offset_from_reference"]["within_tolerance"],
            )

        mean = mean_centroid(centroids)
        verdict = offset_from_reference(mean)
        _journal(
            journal,
            "verdict",
            mean_centroid_px=mean,
            tolerance_px=TOLERANCE_PX,
            **verdict,
        )
        status = "ok"
        exit_code = 0 if verdict["within_tolerance"] else 3
        logger.info(
            "A1 0 级质心 mean=({:.3f}, {:.3f}) dist={:.3f}px within={}",
            mean[0],
            mean[1],
            verdict["dist_px"],
            verdict["within_tolerance"],
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("A1 0 级对齐探针采集失败")
    finally:
        if journal is not None:
            try:
                _journal(
                    journal,
                    "release",
                    slm_closed=slm is not None,
                    camera_closed=cam is not None,
                    error=error,
                    status=status,
                )
            except Exception as exc:
                logger.warning("failed to write release journal event: {}", exc)
            try:
                journal.close()
            except Exception as exc:
                logger.warning("failed to close journal: {}", exc)
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
        release["status"] = status
        release["finished_at_utc"] = utc_now()
        release["released_at_utc"] = utc_now()
        release["elapsed_s"] = time.monotonic() - started
        release["closed_normally"] = all(
            release[key] is True for key in ("slm_closed", "camera_closed", "lock_released")
        ) and status == "ok"
        if output_created:
            write_json(out / "released.json", release)
            measurement: dict[str, Any] = {
                "schema": MEASUREMENT_SCHEMA,
                "status": status,
                "pattern": dict(plan["pattern"]),
                "counter_criteria": dict(plan["counter_criteria"]),
                "frames": frame_records,
                "n_frames_requested": int(plan["n_frames"]),
                "n_frames_used": len(frame_records),
                "mean_centroid_px": mean,
                "mean_offset_from_reference": verdict,
                "within_tolerance": (
                    None if verdict is None else bool(verdict["within_tolerance"])
                ),
                "any_frame_saturated": any(
                    bool(record["saturated"]) for record in frame_records
                ),
                "camera": {
                    "cam_type": str(args.cam_type),
                    "cam_id": int(args.cam_id),
                    "exposure_ms_requested": float(args.exposure_ms),
                    "exposure_ms_actual": exposure_actual,
                    "bit_depth": bit_depth,
                    "full_scale": int(full_scale),
                },
                "panel_shape_rows_cols": list(panel_shape),
                "focus_compensation": focus,
                "preflight": preflight_record,
                "acquisition": dict(plan["acquisition"]),
                "exit_code": exit_code,
                "elapsed_s": release["elapsed_s"],
                "timestamp_utc": utc_now(),
                "error": error,
            }
            write_json(out / "alignment_measurement.json", measurement)
    return exit_code


def require_drive_board_entry(slm: Any) -> str | None:
    """Acquisition-entry drive-board guard, imported lazily from its single home."""
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


def self_test() -> int:
    """Offline invariant check; no hardware, no directories created."""
    pattern = flat_pattern(DEFAULT_PANEL_SHAPE)
    assert pattern["id"] == "flat_scale993"
    assert pattern["two_pi_gray_scale"] == 993
    assert pattern["gray_modulus"] == 994
    assert pattern["panel_shape_rows_cols"] == [1200, 1920]
    assert pattern["focus_d_m1"] == 0.0 and pattern["compensation_applied"] is False
    gray = materialize_flat_gray(DEFAULT_PANEL_SHAPE)
    assert gray.shape == DEFAULT_PANEL_SHAPE and gray.dtype == np.uint16
    assert int(gray.max()) == 0  # flat phase -> gray 0 everywhere
    assert digest(gray) == pattern["gray_sha256"]
    assert pattern["gray_unique_count"] == 1
    assert len(pattern["phase_sha256"]) == 64 and len(pattern["gray_sha256"]) == 64

    # Centroid convention must be S1's function itself, not a copy.
    import scripts.slm.capture_s1_mapping_disambiguation_20261124 as s1

    assert _centroid_px is s1._centroid_px

    zero = offset_from_reference(
        [ZERO_ORDER_REFERENCE_PX["row"], ZERO_ORDER_REFERENCE_PX["col"]]
    )
    assert zero["within_tolerance"] is True and zero["dist_px"] == 0.0
    boundary = offset_from_reference(
        [ZERO_ORDER_REFERENCE_PX["row"] + TOLERANCE_PX, ZERO_ORDER_REFERENCE_PX["col"]]
    )
    assert boundary["within_tolerance"] is True
    outside = offset_from_reference(
        [
            ZERO_ORDER_REFERENCE_PX["row"] + TOLERANCE_PX + 0.01,
            ZERO_ORDER_REFERENCE_PX["col"],
        ]
    )
    assert outside["within_tolerance"] is False
    observed = offset_from_reference(
        [OBSERVED_20261126_CENTROID_PX["row"], OBSERVED_20261126_CENTROID_PX["col"]]
    )
    assert observed["within_tolerance"] is False
    assert 97.5 < observed["dist_px"] < 97.7

    # Synthetic single peak: centroid lands on the peak centre.
    frame = np.zeros((61, 61), dtype=np.uint16)
    frame[29:32, 40:43] = 500
    assert _centroid_px(frame) == [30.0, 41.0]

    logger.info("A1 0 级对齐探针离线自检通过（单图样、同口径质心、5 px 容差）")
    print(
        json.dumps(
            {
                "self_test": "passed",
                "pattern": "flat_scale993",
                "n_frames_default": DEFAULT_FRAMES,
                "tolerance_px": TOLERANCE_PX,
            },
            ensure_ascii=False,
        )
    )
    return 0


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="A1 0 级对齐探针（单图样平场）；无 --execute 只打印 plan"
    )
    parser.add_argument(
        "--out", type=Path, help="全新输出目录（离线规划与采集都要）"
    )
    parser.add_argument(
        "--execute", action="store_true", help="打开设备并采集（需台面锁）"
    )
    parser.add_argument("--self-test", action="store_true", help="离线自检")
    parser.add_argument(
        "--exposure-ms", type=float, default=DEFAULT_EXPOSURE_MS, help="相机曝光 (ms)"
    )
    parser.add_argument(
        "--wait-until-idle",
        type=float,
        default=DEFAULT_WAIT_IDLE_S,
        help="等待台面锁的最长秒数",
    )
    parser.add_argument(
        "--slm-number", type=int, default=DEFAULT_SLM_NUMBER, help="SLM 编号"
    )
    parser.add_argument("--cam-type", default=DEFAULT_CAM_TYPE, help="相机类型")
    parser.add_argument("--cam-id", type=int, default=DEFAULT_CAM_ID, help="相机 id")
    parser.add_argument(
        "--frames",
        type=_positive_int,
        default=DEFAULT_FRAMES,
        help="连续采集的平场帧数（逐帧质心取均值）",
    )
    parser.add_argument(
        "--settle-s",
        type=float,
        default=DEFAULT_SETTLE_S,
        help="每帧前等待 SLM 稳定的秒数",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    if args.out is None:
        parser.error("--out 是必需参数（离线规划与采集都要写出目录）")

    if args.execute and args.out.exists():
        parser.error(f"--execute 要求全新输出目录，{args.out} 已存在")

    if args.settle_s < 0.0:
        parser.error("--settle-s 必须 >= 0")
    if args.exposure_ms <= 0.0:
        parser.error("--exposure-ms 必须 > 0")

    plan = build_plan(args)
    if not args.execute:
        print(json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    return execute(args, plan)


if __name__ == "__main__":
    raise SystemExit(main())
