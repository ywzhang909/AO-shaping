#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立 x/y 光栅标度采集：只采 flat + x 光栅 + y 光栅（离线可规划）。

为什么需要它
------------
S1 六帧协议（``scripts/slm/capture_s1_mapping_disambiguation_20261124.py``）的
判据②用 grating_x_p128_scale993 的 ±1 级列向位移窗描述面板标度。重采六帧代价高，
本脚本只取其中三张冻结图样（scale=993、period=128 px），量出本会话的 ±1 级
pitch/位移，并把它与冻结窗做**数值对照**；不改判据②，不重判六帧，不声称算法收益。

冻结口径（只复用，不复制、不改写）
----------------------------------
* 图样：S1 ``PATTERN_SPECS`` / ``protocol_patterns()`` 中的三条记录；
  相位用同一宿主的 ``grating_phase`` / ``flat_phase``，灰度 wrap 唯一入口是
  ``phase_to_gray``；2π 灰度 993、period 128 px 全部来自 S1 常量。
* 斑点：``issue247_axis_map_solve_20261121.find_spots`` 与判读脚本的
  ``merge_nearby_spots``；0 级匹配半径、离轴最小距离、判据②冻结窗全部从判读脚本
  导入，本文件不自设参与裁定的数值阈值。
* raw 无信号门：S1 v2 的 ``flat_signal_measurement()["has_signal"]``；饱和门用 S1
  的 ``SATURATION_FRACTION``。该函数的质心只用于 flat 0 级；光栅帧只取 raw 门控
  量，不拿它的全帧掩膜质心冒充 ±1 级位置。

fail-closed
-----------
峰值过低 / 无信号 / 饱和 / 找不到 ±1 级时：状态与退出码为非成功
（4=no_signal, 5=saturated, 6=no_orders），缺失的质心与 pitch 一律写 null，
绝不输出几何中心或重复斑当作测量值。

硬件纪律：没有 ``--execute`` 时只打印确定性 plan；台面锁、SLM/CCD 驱动只在
``execute`` 内惰性导入/获取，finally 保证释放并写 released.json 与 journal。
不操作激光，不叠加离焦补偿（无 Fresnel 面板），不写共享 report/provenance 文件。
"""

from __future__ import annotations

import argparse
import json
import subprocess
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

from scripts.slm.capture_s1_mapping_disambiguation_20261124 import (
    PATTERN_SPECS,
    SATURATION_FRACTION,
    build_phase,
    digest,
    flat_signal_measurement,
    protocol_patterns,
    utc_now,
    write_json,
)
from scripts.slm.issue247_axis_map_solve_20261121 import find_spots
from scripts.slm.issue247_geometry_capture_20261120 import (
    flat_phase,
    grating_phase,
    phase_to_gray,
)
from scripts.slm.judge_s1_criteria_offline_20261127 import (
    FROZEN_2_COL_WINDOW_PX,
    FROZEN_2_ROW_RESIDUAL_MAX_PX,
    OFF_AXIS_MIN_DIST_PX,
    ZERO_SPOT_MATCH_RADIUS_PX,
    merge_nearby_spots,
)

PLAN_SCHEMA = "xy-grating-scale-capture-plan/v1"
MEASUREMENT_SCHEMA = "xy-grating-scale-capture-measurement/v1"
JOURNAL_SCHEMA = "xy-grating-scale-capture-journal/v1"
RELEASE_SCHEMA = "xy-grating-scale-capture-released/v1"

DEFAULT_PANEL_SHAPE = (1200, 1920)  # rows, cols (driver Panel_Res is cols, rows)
NATIVE_CCD_SHAPE = (1944, 2592)
BIT_DEPTH_REQUIRED = 12
FULL_SCALE_REQUIRED = (1 << BIT_DEPTH_REQUIRED) - 1
DEFAULT_EXPOSURE_MS = 0.087
DEFAULT_WAIT_IDLE_S = 600.0
DEFAULT_SLM_NUMBER = 1
DEFAULT_CAM_TYPE = "daheng"
DEFAULT_CAM_ID = 0
DEFAULT_FRAMES = 1
DEFAULT_SETTLE_S = 0.8
PREFLIGHT_DISCARD_FRAMES = 3
PREFLIGHT_MEASURE_FRAMES = 1
LOCK_POLL_S = 1.0

#: 本脚本的三帧；S1 六帧其余图样（scale1024 / blaze2d / ladder）一律不采。
FLAT_ID = "flat_scale993"
X_GRATING_ID = "grating_x_p128_scale993"
Y_GRATING_ID = "grating_y_p128_scale993"
CAPTURE_IDS: tuple[str, str, str] = (FLAT_ID, X_GRATING_ID, Y_GRATING_ID)
PATTERN_BY_ID: dict[str, dict[str, Any]] = {
    str(spec["id"]): dict(spec)
    for spec in PATTERN_SPECS
    if spec["id"] in CAPTURE_IDS
}

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NO_LOCK = 2
EXIT_NO_SIGNAL = 4
EXIT_SATURATED = 5
EXIT_NO_ORDERS = 6

SIGNAL_GATE_SOURCE = (
    "scripts.slm.capture_s1_mapping_disambiguation_20261124.flat_signal_measurement"
)
SPOT_SOURCE = "scripts.slm.issue247_axis_map_solve_20261121.find_spots"
PHASE_SOURCE = "scripts.slm.issue247_geometry_capture_20261120.grating_phase"
GRAY_SOURCE = "scripts.slm.issue247_geometry_capture_20261120.phase_to_gray"
FROZEN_WINDOW_SOURCE = "scripts.slm.judge_s1_criteria_offline_20261127"
SCOPE_NOTE = (
    "只陈述本次实测 pitch/位移与冻结窗的数值关系；不重判 S1 判据①/②，"
    "不重采 S1 六帧，不修改任何冻结阈值，不声称算法收益。"
)


def _git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


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


def materialize_gray(spec: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    """Frozen phase -> uint16 panel gray; only the S1 wrap point is used."""
    shape = (int(shape[0]), int(shape[1]))
    if spec["kind"] == "grating":
        phase = grating_phase(
            shape, period_px=int(spec["period_px"]), axis=int(spec["axis"])
        )
    elif spec["kind"] == "flat":
        phase = flat_phase(shape)
    else:
        raise ValueError(f"unexpected pattern kind {spec['kind']!r}")
    return phase_to_gray(phase, int(spec["two_pi_gray_scale"]))


def raw_files(frames: int) -> dict[str, list[str]]:
    """Deterministic raw-file layout for the three patterns."""
    return {
        pattern_id: [
            f"{index:02d}_{pattern_id}_f{frame_index:02d}.npy"
            for frame_index in range(int(frames))
        ]
        for index, pattern_id in enumerate(CAPTURE_IDS)
    }


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Deterministic plan: no clock, no hardware, no volatile commit field."""
    patterns = [dict(item) for item in protocol_patterns() if item["id"] in CAPTURE_IDS]
    if [item["id"] for item in patterns] != list(CAPTURE_IDS):
        raise RuntimeError("frozen three-pattern snapshot does not match capture order")
    frames = int(args.frames)
    return {
        "schema": PLAN_SCHEMA,
        "out": Path(args.out).as_posix(),
        "capture_order": list(CAPTURE_IDS),
        "patterns": patterns,
        "frames_per_pattern": frames,
        "raw_files": raw_files(frames),
        "acquisition": {
            "exposure_ms": float(args.exposure_ms),
            "wait_until_idle_s": float(args.wait_until_idle),
            "slm_number": int(args.slm_number),
            "cam_type": str(args.cam_type),
            "cam_id": int(args.cam_id),
            "settle_s": float(args.settle_s),
            "bit_depth_required": BIT_DEPTH_REQUIRED,
            "full_scale_required": FULL_SCALE_REQUIRED,
            "native_ccd_shape": list(NATIVE_CCD_SHAPE),
            "preflight_discard_frames": PREFLIGHT_DISCARD_FRAMES,
            "preflight_measure_frames": PREFLIGHT_MEASURE_FRAMES,
        },
        "focus": {
            "focus_d_m1": 0.0,
            "compensation_applied": False,
            "fresnel_overlay": False,
        },
        "signal_gate": {
            "source": SIGNAL_GATE_SOURCE,
            "reused": True,
            "usage": "raw has_signal/peak gate only; grating centroid is NOT taken",
            "saturation_source": "scripts.slm.capture_s1_mapping_disambiguation_20261124.SATURATION_FRACTION",
            "saturation_fraction": float(SATURATION_FRACTION),
        },
        "spot_detection": {
            "find_spots": SPOT_SOURCE,
            "merge_nearby_spots": FROZEN_WINDOW_SOURCE + ".merge_nearby_spots",
            "zero_match_radius_source": FROZEN_WINDOW_SOURCE + ".ZERO_SPOT_MATCH_RADIUS_PX",
            "zero_match_radius_px": float(ZERO_SPOT_MATCH_RADIUS_PX),
            "off_axis_min_dist_source": FROZEN_WINDOW_SOURCE + ".OFF_AXIS_MIN_DIST_PX",
            "off_axis_min_dist_px": float(OFF_AXIS_MIN_DIST_PX),
            "pair_rule": (
                " brightest spot on each side of the flat-matched 0th order along "
                "the CCD axis whose opposite-side pair has the larger peak sum"
            ),
        },
        "phase_and_gray": {"phase": PHASE_SOURCE, "gray": GRAY_SOURCE},
        "frozen_window": {
            "source": FROZEN_WINDOW_SOURCE,
            "col_window_px": list(FROZEN_2_COL_WINDOW_PX),
            "row_residual_max_px": float(FROZEN_2_ROW_RESIDUAL_MAX_PX),
            "descriptive_only": True,
            "criterion_2_evaluated": False,
        },
        "scope_note": SCOPE_NOTE,
        "exit_codes": {
            "ok": EXIT_OK,
            "error": EXIT_ERROR,
            "no_lock": EXIT_NO_LOCK,
            "no_signal": EXIT_NO_SIGNAL,
            "saturated": EXIT_SATURATED,
            "no_orders": EXIT_NO_ORDERS,
        },
    }


def frame_signal(frame: np.ndarray, full_scale: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """Raw-only gate from S1 v2; returns (serializable record, S1 measurement)."""
    measured = flat_signal_measurement(frame)
    peak = int(frame.max()) if frame.size else 0
    threshold = float(SATURATION_FRACTION) * float(full_scale)
    saturated = bool(peak >= threshold)
    record: dict[str, Any] = {
        "raw_peak_counts": peak,
        "raw_peak_snr": float(measured["peak_snr"]),
        "raw_median_counts": float(measured["median_counts"]),
        "raw_sigma_counts": float(measured["sigma_counts"]),
        "signal_pixels": int(measured["signal_pixels"]),
        "signal_weight_counts": float(measured["signal_weight_counts"]),
        "saturation_threshold_counts": threshold,
        "saturated": saturated,
        "has_signal": bool(measured["has_signal"]) and not saturated,
        "gate_source": SIGNAL_GATE_SOURCE,
    }
    return record, measured


def zero_order_from_flat(
    measured: dict[str, Any], gate: dict[str, Any]
) -> list[float] | None:
    """Only a real signal-masked centroid may become the 0th-order reference."""
    if not gate["has_signal"]:
        return None
    if bool(measured["centroid_is_fallback"]):
        return None
    value = measured["centroid_px"]
    if value is None:
        return None
    return [float(value[0]), float(value[1])]


def _describe_spots(spots: list[dict[str, float]]) -> list[dict[str, float]]:
    return [
        {
            "row": float(spot["row"]),
            "col": float(spot["col"]),
            "peak": float(spot["peak"]),
        }
        for spot in spots
    ]


def measure_grating_pair(
    frame: np.ndarray, zero_px: tuple[float, float] | list[float], pattern_id: str
) -> dict[str, Any]:
    """Structural +/-1 pair from frozen spot functions; found=False means no result.

    No new numeric criterion is introduced here: detection uses ``find_spots``'
    own threshold, 0th matching and off-axis exclusion reuse the frozen radii from
    the S1 judge module, and the CCD axis is chosen by the opposite-side pair with
    the larger peak sum.  Every failure returns ``pitch_px=None`` and ``orders=[]``.
    """
    spots = merge_nearby_spots(find_spots(frame))
    result: dict[str, Any] = {
        "pattern_id": pattern_id,
        "found": False,
        "reason": None,
        "pitch_px": None,
        "separation_px": None,
        "dominant_ccd_axis": None,
        "zero_order_px": None,
        "zero_match_dist_px": None,
        "orders": [],
        "spots": _describe_spots(spots),
    }
    if not spots:
        result["reason"] = "no_spots"
        return result

    zero_row, zero_col = float(zero_px[0]), float(zero_px[1])
    origin = min(
        spots,
        key=lambda spot: float(
            np.hypot(float(spot["row"]) - zero_row, float(spot["col"]) - zero_col)
        ),
    )
    match_dist = float(
        np.hypot(float(origin["row"]) - zero_row, float(origin["col"]) - zero_col)
    )
    result["zero_match_dist_px"] = match_dist
    if match_dist > float(ZERO_SPOT_MATCH_RADIUS_PX):
        result["reason"] = "zero_order_not_matched"
        return result

    result["zero_order_px"] = [float(origin["row"]), float(origin["col"])]
    candidates = [
        spot
        for spot in spots
        if spot is not origin
        and float(
            np.hypot(
                float(spot["row"]) - float(origin["row"]),
                float(spot["col"]) - float(origin["col"]),
            )
        )
        >= float(OFF_AXIS_MIN_DIST_PX)
    ]
    if len(candidates) < 2:
        result["reason"] = "no_opposite_pair"
        return result

    best: dict[str, Any] | None = None
    for axis_name in ("row", "col"):
        plus = max(
            (spot for spot in candidates if float(spot[axis_name]) > float(origin[axis_name])),
            key=lambda spot: float(spot["peak"]),
            default=None,
        )
        minus = max(
            (spot for spot in candidates if float(spot[axis_name]) < float(origin[axis_name])),
            key=lambda spot: float(spot["peak"]),
            default=None,
        )
        if plus is None or minus is None:
            continue
        score = float(plus["peak"]) + float(minus["peak"])
        pitch = 0.5 * (
            abs(float(plus[axis_name]) - float(origin[axis_name]))
            + abs(float(minus[axis_name]) - float(origin[axis_name]))
        )
        candidate = {"axis": axis_name, "plus": plus, "minus": minus, "score": score, "pitch": pitch}
        if best is None or (score, pitch) > (best["score"], best["pitch"]):
            best = candidate
    if best is None:
        result["reason"] = "no_opposite_pair"
        return result

    plus, minus = best["plus"], best["minus"]
    orders: list[dict[str, float]] = []
    for label, spot in (("+1", plus), ("-1", minus)):
        orders.append(
            {
                "order": label,
                "row": float(spot["row"]),
                "col": float(spot["col"]),
                "peak": float(spot["peak"]),
                "drow_px": float(spot["row"]) - float(origin["row"]),
                "dcol_px": float(spot["col"]) - float(origin["col"]),
            }
        )
    result.update(
        {
            "found": True,
            "pitch_px": float(best["pitch"]),
            "separation_px": float(
                abs(float(plus[best["axis"]]) - float(minus[best["axis"]]))
            ),
            "dominant_ccd_axis": str(best["axis"]),
            "orders": orders,
        }
    )
    return result


def frozen_window_relation(pair: dict[str, Any] | None, pattern_id: str) -> dict[str, Any]:
    """Descriptive relation only; never a judgement of criterion ②."""
    relation: dict[str, Any] = {
        "source": FROZEN_WINDOW_SOURCE,
        "pattern_id": pattern_id,
        "criterion_2_evaluated": False,
        "descriptive_only": True,
        "applicable": pattern_id == X_GRATING_ID,
    }
    if pattern_id != X_GRATING_ID:
        relation["note"] = "冻结列向窗只针对 x 光栅；本图样只陈述实测 pitch/位移。"
        return relation
    relation.update(
        {
            "col_window_px": list(FROZEN_2_COL_WINDOW_PX),
            "row_residual_max_px": float(FROZEN_2_ROW_RESIDUAL_MAX_PX),
            "observed_dcol_px": [],
            "observed_abs_dcol_px": [],
            "observed_drow_px": [],
        }
    )
    if pair is None or not pair["found"]:
        relation["descriptive_within_window"] = None
        relation["note"] = "未测到 ±1 级，不给冻结窗对照结论。"
        return relation
    dc = [float(order["dcol_px"]) for order in pair["orders"]]
    dr = [float(order["drow_px"]) for order in pair["orders"]]
    relation["observed_dcol_px"] = dc
    relation["observed_abs_dcol_px"] = [abs(value) for value in dc]
    relation["observed_drow_px"] = dr
    relation["descriptive_within_window"] = all(
        float(FROZEN_2_COL_WINDOW_PX[0]) <= abs(value) <= float(FROZEN_2_COL_WINDOW_PX[1])
        for value in dc
    )
    relation["note"] = "仅陈述实测位移与冻结窗的数值关系，不重判判据②、不改窗。"
    return relation


def make_report(
    pitch: dict[str, float | None],
    relations: dict[str, dict[str, Any]],
    status: str,
) -> dict[str, Any]:
    statements = [
        "本次只采集 flat + x 光栅 + y 光栅三张冻结图样（scale=993, period=128 px）。",
        (
            "x 光栅实测 pitch = "
            + ("null（未测到）" if pitch[X_GRATING_ID] is None else f"{pitch[X_GRATING_ID]:.3f} px")
            + "；"
        ),
        (
            "y 光栅实测 pitch = "
            + ("null（未测到）" if pitch[Y_GRATING_ID] is None else f"{pitch[Y_GRATING_ID]:.3f} px")
            + "；"
        ),
    ]
    x_relation = relations[X_GRATING_ID]
    relations_text = (
        "x 光栅 ±1 级列向位移 "
        + repr(x_relation["observed_dcol_px"])
        + " px（绝对值 "
        + repr(x_relation["observed_abs_dcol_px"])
        + " px），冻结窗 "
        + repr(x_relation["col_window_px"])
        + " px；descriptive_within_window="
        + repr(x_relation["descriptive_within_window"])
        + "。"
        if x_relation["applicable"]
        else "冻结列向窗不适用于本次 y 光栅，仅报告实测 pitch/位移。"
    )
    statements.append(relations_text)
    statements.append(SCOPE_NOTE)
    return {
        "status": status,
        "statements": statements,
        "criterion_2_evaluated": False,
        "algorithm_benefit_claim": False,
    }


def _panel_shape_from(slm: Any) -> tuple[int, int]:
    """(rows, cols) from the driver's Panel_Res, which is (cols, rows)."""
    panel = getattr(slm, "Panel_Res", None)
    if not panel or len(panel) != 2:
        logger.warning("SLM Panel_Res missing/odd ({!r}); assuming frozen {}", panel, DEFAULT_PANEL_SHAPE)
        return DEFAULT_PANEL_SHAPE
    return int(panel[1]), int(panel[0])


def run_preflight(cam: Any, full_scale: float, journal: Any) -> dict[str, Any]:
    """Discard N frames then measure one; caller aborts before science frames."""
    for _ in range(PREFLIGHT_DISCARD_FRAMES):
        cam.get_numpy_image(1, True)
    frame = np.asarray(cam.get_numpy_image(1, True)).copy()
    peak = int(frame.max()) if frame.size else 0
    threshold = float(SATURATION_FRACTION) * float(full_scale)
    record: dict[str, Any] = {
        "discard_frames": PREFLIGHT_DISCARD_FRAMES,
        "measure_frames": PREFLIGHT_MEASURE_FRAMES,
        "peak_counts": peak,
        "saturation_threshold_counts": threshold,
        "saturation_fraction": float(SATURATION_FRACTION),
        "saturated": bool(peak >= threshold),
    }
    _journal(journal, "preflight", **record)
    return record


def require_drive_board_entry(slm: Any) -> str | None:
    """Acquisition-entry drive-board guard, imported lazily from its single home."""
    from ao_shaping.drivers.slm.factory import require_drive_board_entry as guard

    return guard(slm)


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
    """Acquire lock, open both devices once, capture the three frozen patterns."""
    out: Path = args.out
    lock = acquire_device_lock(float(args.wait_until_idle))
    if lock is None:
        logger.error("could not acquire bench device lock within {}s", args.wait_until_idle)
        return EXIT_NO_LOCK

    slm: Any = None
    cam: Any = None
    journal: Any = None
    frame_records: list[dict[str, Any]] = []
    preflight: dict[str, Any] | None = None
    drive_board: str | None = None
    status = "error"
    error: str | None = None
    exit_code = EXIT_ERROR
    bit_depth = BIT_DEPTH_REQUIRED
    full_scale = float(FULL_SCALE_REQUIRED)
    exposure_actual: float | None = None
    panel_shape = DEFAULT_PANEL_SHAPE
    zero_mean: list[float] | None = None
    pitch: dict[str, float | None] = {}
    relations: dict[str, dict[str, Any]] = {}
    started = time.monotonic()
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
    try:
        if out.exists():
            raise RuntimeError(f"--execute 要求全新输出目录，{out} 已存在")
        out.mkdir(parents=True, exist_ok=False)
        (out / "raw").mkdir()
        write_json(out / "plan.json", plan)
        journal = (out / "capture_journal.jsonl").open("a", encoding="utf-8")
        _journal(journal, "lock_acquired", wait_until_idle_s=float(args.wait_until_idle))

        for pattern_id in CAPTURE_IDS:
            spec = PATTERN_BY_ID[pattern_id]
            record = next(item for item in plan["patterns"] if item["id"] == pattern_id)
            gray = materialize_gray(spec, DEFAULT_PANEL_SHAPE)
            if digest(gray) != record["gray_sha256"]:
                raise RuntimeError(f"frozen gray integrity check failed: {pattern_id}")
            if digest(build_phase(spec, DEFAULT_PANEL_SHAPE)) != record["phase_sha256"]:
                raise RuntimeError(f"frozen phase integrity check failed: {pattern_id}")

        from ao_shaping.drivers.ccd.common import (
            create_camera,
            get_camera_exposure_ms,
            set_camera_exposure_ms,
        )
        from ao_shaping.drivers.slm.santec import Santec

        slm = Santec(slm_number=int(args.slm_number), video_mode=0)
        slm.open()
        drive_board = require_drive_board_entry(slm)
        panel_shape = _panel_shape_from(slm)
        if panel_shape != DEFAULT_PANEL_SHAPE:
            raise RuntimeError(f"expected panel {DEFAULT_PANEL_SHAPE}, got {panel_shape}")
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
            raise RuntimeError(f"camera bit-depth must be {BIT_DEPTH_REQUIRED}, got {bit_depth!r}")
        set_camera_exposure_ms(cam, float(args.exposure_ms))
        exposure_actual = float(get_camera_exposure_ms(cam))
        full_scale = float(cam.effective_full_scale(np.zeros((1,), dtype=np.uint16)))
        if full_scale != float(FULL_SCALE_REQUIRED):
            raise RuntimeError(f"camera full scale {full_scale} != {FULL_SCALE_REQUIRED}")
        _journal(
            journal,
            "devices_ready",
            slm_number=int(args.slm_number),
            drive_board_id=drive_board,
            panel_shape_rows_cols=list(panel_shape),
            cam_type=str(args.cam_type),
            cam_id=int(args.cam_id),
            exposure_ms_actual=exposure_actual,
            bit_depth=bit_depth,
            full_scale=int(full_scale),
        )

        preflight = run_preflight(cam, full_scale, journal)
        if preflight["saturated"]:
            raise RuntimeError("preflight saturated; reduce exposure/beam before capture")

        zero_px: list[float] | None = None
        for index, pattern_id in enumerate(CAPTURE_IDS):
            gray = materialize_gray(PATTERN_BY_ID[pattern_id], panel_shape)
            slm.display_data(gray)
            for frame_index in range(int(args.frames)):
                time.sleep(float(args.settle_s))
                frame = np.asarray(cam.get_numpy_image(1, True)).copy()
                if frame.shape != NATIVE_CCD_SHAPE:
                    raise RuntimeError(f"expected full-frame {NATIVE_CCD_SHAPE}, got {frame.shape}")
                if frame.dtype.kind not in "ui":
                    raise RuntimeError(f"expected integer CCD data, got dtype {frame.dtype}")

                gate, measured = frame_signal(frame, full_scale)
                raw_name = raw_files(int(args.frames))[pattern_id][frame_index]
                raw_path = out / "raw" / raw_name
                np.save(raw_path, frame)
                row: dict[str, Any] = {
                    "pattern_id": pattern_id,
                    "frame_index": frame_index,
                    "raw_file": raw_path.relative_to(out).as_posix(),
                    "exposure_ms_actual": exposure_actual,
                    "bit_depth": bit_depth,
                    "full_scale": int(full_scale),
                    **gate,
                    "spots": [],
                    "zero_order_px": None,
                    "orders": [],
                    "pitch_px": None,
                    "pair": None,
                    "status": "ok",
                    "saved_at_utc": utc_now(),
                }
                if gate["saturated"]:
                    row["status"] = "saturated"
                elif not gate["has_signal"]:
                    row["status"] = "no_signal"
                elif pattern_id == FLAT_ID:
                    zero_px = zero_order_from_flat(measured, gate)
                    row["zero_order_px"] = zero_px
                    row["spots"] = _describe_spots(merge_nearby_spots(find_spots(frame)))
                    if zero_px is None:
                        row["status"] = "no_signal"
                else:
                    row["spots"] = _describe_spots(merge_nearby_spots(find_spots(frame)))
                    if zero_px is None:
                        row["status"] = "no_zero_reference"
                    else:
                        pair = measure_grating_pair(frame, zero_px, pattern_id)
                        row["pair"] = pair
                        row["orders"] = pair["orders"]
                        row["zero_order_px"] = pair["zero_order_px"]
                        row["pitch_px"] = pair["pitch_px"]
                        if not pair["found"]:
                            row["status"] = "no_orders"
                frame_records.append(row)
                _journal(journal, "frame", **row)

        statuses = {row["status"] for row in frame_records}
        if "saturated" in statuses:
            status, exit_code = "saturated", EXIT_SATURATED
        elif "no_signal" in statuses:
            status, exit_code = "no_signal", EXIT_NO_SIGNAL
        elif statuses & {"no_orders", "no_zero_reference"}:
            status, exit_code = "no_orders", EXIT_NO_ORDERS
        else:
            status, exit_code = "ok", EXIT_OK

        zeros = [
            row["zero_order_px"]
            for row in frame_records
            if row["pattern_id"] == FLAT_ID and row["zero_order_px"] is not None
        ]
        zero_mean = (
            [float(value) for value in np.asarray(zeros, dtype=np.float64).mean(axis=0)]
            if zeros
            else None
        )
        for pattern_id in (X_GRATING_ID, Y_GRATING_ID):
            values = [
                float(row["pitch_px"])
                for row in frame_records
                if row["pattern_id"] == pattern_id and row["pitch_px"] is not None
            ]
            pitch[pattern_id] = float(np.median(values)) if values else None
            pair_row = next(
                (row["pair"] for row in frame_records if row["pattern_id"] == pattern_id and row["pair"]),
                None,
            )
            relations[pattern_id] = frozen_window_relation(pair_row, pattern_id)

        logger.info(
            "xy grating scale: x={} y={} status={}",
            pitch[X_GRATING_ID],
            pitch[Y_GRATING_ID],
            status,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("x/y 光栅标度采集失败")
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
            (release["slm_closed"], release["camera_closed"], release["lock_released"])
        )
        if out.exists() and (out / "plan.json").exists():
            write_json(out / "released.json", release)

            occupancy = {
                "wait_until_idle_s": float(args.wait_until_idle),
                "acquired_at_utc": release["acquired_at_utc"],
                "released_at_utc": release["released_at_utc"],
                "elapsed_s": release["elapsed_s"],
                "slm_closed": release["slm_closed"],
                "camera_closed": release["camera_closed"],
                "lock_released": release["lock_released"],
                "closed_normally": release["closed_normally"],
            }
            measurement: dict[str, Any] = {
                "schema": MEASUREMENT_SCHEMA,
                "status": status,
                "exit_code": exit_code,
                "capture_order": list(CAPTURE_IDS),
                "frames": frame_records,
                "n_frames_requested": len(CAPTURE_IDS) * int(args.frames),
                "n_frames_recorded": len(frame_records),
                "zero_order_px": zero_mean,
                "pitch_px": pitch,
                "displacement_px": {
                    pattern_id: [
                        order
                        for row in frame_records
                        if row["pattern_id"] == pattern_id
                        for order in row["orders"]
                    ]
                    for pattern_id in (X_GRATING_ID, Y_GRATING_ID)
                },
                "frozen_window_relation": relations,
                "report": make_report(pitch, relations, status),
                "signal_gate": dict(plan["signal_gate"]),
                "spot_detection": dict(plan["spot_detection"]),
                "camera": {
                    "cam_type": str(args.cam_type),
                    "cam_id": int(args.cam_id),
                    "exposure_ms_requested": float(args.exposure_ms),
                    "exposure_ms_actual": exposure_actual,
                    "bit_depth": bit_depth,
                    "full_scale": int(full_scale),
                },
                "slm": {
                    "slm_number": int(args.slm_number),
                    "drive_board_id": drive_board,
                    "panel_shape_rows_cols": list(panel_shape),
                },
                "drive_board_id": drive_board,
                "focus": dict(plan["focus"]),
                "preflight": preflight,
                "commit": _git_head(),
                "occupancy": occupancy,
                "timestamp_utc": utc_now(),
                "error": error,
            }
            write_json(out / "measurement.json", measurement)
    return exit_code


def _synthetic_grating_frame(
    zero: tuple[float, float], pitch: float, axis: int, *, with_orders: bool = True
) -> np.ndarray:
    """Gaussian spots: 0th at ``zero``, optionally +/-1 at ``pitch`` on ``axis``."""
    rows, cols = 240, 320
    frame = np.full((rows, cols), 10.0, dtype=np.float64)
    rr, cc = np.mgrid[0:rows, 0:cols]
    spots = [(zero[0], zero[1])]
    if with_orders:
        spots.extend(
            [
                (
                    zero[0] + (pitch if axis == 0 else 0.0),
                    zero[1] + (pitch if axis == 1 else 0.0),
                ),
                (
                    zero[0] - (pitch if axis == 0 else 0.0),
                    zero[1] - (pitch if axis == 1 else 0.0),
                ),
            ]
        )
    for row, col in spots:
        frame += 1500.0 * np.exp(-((rr - row) ** 2 + (cc - col) ** 2) / (2.0 * 3.0**2))
    return frame


def self_test() -> int:
    """Offline invariants: synthetic +/-1, fail-closed, deterministic plan."""
    import scripts.slm.capture_s1_mapping_disambiguation_20261124 as s1
    import scripts.slm.issue247_axis_map_solve_20261121 as axis_map
    import scripts.slm.issue247_geometry_capture_20261120 as geometry

    assert flat_signal_measurement is s1.flat_signal_measurement
    assert find_spots is axis_map.find_spots
    assert phase_to_gray is geometry.phase_to_gray
    assert grating_phase is geometry.grating_phase

    args = argparse.Namespace(
        out=str(ROOT / "data/debug/xy_grating_scale_offline_plan"),
        exposure_ms=DEFAULT_EXPOSURE_MS,
        wait_until_idle=DEFAULT_WAIT_IDLE_S,
        slm_number=DEFAULT_SLM_NUMBER,
        cam_type=DEFAULT_CAM_TYPE,
        cam_id=DEFAULT_CAM_ID,
        frames=DEFAULT_FRAMES,
        settle_s=DEFAULT_SETTLE_S,
        execute=False,
        self_test=False,
    )
    plan_a = build_plan(args)
    plan_b = build_plan(args)
    assert json.dumps(plan_a, sort_keys=True) == json.dumps(plan_b, sort_keys=True)
    assert plan_a["capture_order"] == list(CAPTURE_IDS)
    assert [record["id"] for record in plan_a["patterns"]] == list(CAPTURE_IDS)
    assert plan_a["frames_per_pattern"] == DEFAULT_FRAMES
    assert plan_a["frozen_window"]["criterion_2_evaluated"] is False
    assert "timestamp" not in json.dumps(plan_a)

    for pattern_id in CAPTURE_IDS:
        spec = PATTERN_BY_ID[pattern_id]
        phase = build_phase(spec, DEFAULT_PANEL_SHAPE)
        gray = materialize_gray(spec, DEFAULT_PANEL_SHAPE)
        record = next(item for item in plan_a["patterns"] if item["id"] == pattern_id)
        assert digest(phase) == record["phase_sha256"]
        assert digest(gray) == record["gray_sha256"]
        assert int(spec["two_pi_gray_scale"]) == 993
        if pattern_id != FLAT_ID:
            assert int(spec["period_px"]) == 128
    assert set(PATTERN_BY_ID) == set(CAPTURE_IDS)

    # Synthetic +/-1 orders: pitch is the mean one-sided displacement.
    zero = (110.0, 150.0)
    for axis, pattern_id, axis_name in (
        (1, X_GRATING_ID, "col"),
        (0, Y_GRATING_ID, "row"),
    ):
        frame = _synthetic_grating_frame(zero, 95.0, axis)
        gate, _ = frame_signal(frame, float(FULL_SCALE_REQUIRED))
        assert gate["has_signal"] is True
        pair = measure_grating_pair(frame, zero, pattern_id)
        assert pair["found"] is True, pair
        assert pair["dominant_ccd_axis"] == axis_name
        assert abs(float(pair["pitch_px"]) - 95.0) < 1.0
        assert abs(float(pair["separation_px"]) - 190.0) < 2.0
        displacements = sorted(
            float(order["drow_px" if axis == 0 else "dcol_px"])
            for order in pair["orders"]
        )
        assert abs(displacements[0] + 95.0) < 1.0
        assert abs(displacements[1] - 95.0) < 1.0

    # No signal / saturation / missing orders: never fabricate a centroid or pitch.
    blank = np.zeros((61, 61), dtype=np.uint16)
    gate, measured = frame_signal(blank, float(FULL_SCALE_REQUIRED))
    assert gate["has_signal"] is False and gate["raw_peak_counts"] == 0
    assert zero_order_from_flat(measured, gate) is None
    saturated = np.full((61, 61), 4000, dtype=np.uint16)
    sat_gate, _ = frame_signal(saturated, float(FULL_SCALE_REQUIRED))
    assert sat_gate["saturated"] is True and sat_gate["has_signal"] is False
    invalid_pair = measure_grating_pair(blank, zero, X_GRATING_ID)
    assert invalid_pair["found"] is False
    assert invalid_pair["pitch_px"] is None and invalid_pair["orders"] == []
    one_spot = _synthetic_grating_frame(zero, 95.0, 1, with_orders=False)
    pair_missing = measure_grating_pair(one_spot, zero, X_GRATING_ID)
    assert pair_missing["found"] is False
    assert pair_missing["pitch_px"] is None

    relation = frozen_window_relation(measure_grating_pair(_synthetic_grating_frame(zero, 95.0, 1), zero, X_GRATING_ID), X_GRATING_ID)
    assert relation["criterion_2_evaluated"] is False
    assert relation["applicable"] is True
    assert relation["observed_dcol_px"] != []
    y_relation = frozen_window_relation(None, Y_GRATING_ID)
    assert y_relation["applicable"] is False

    print(json.dumps({"self_test": "passed", "capture_order": list(CAPTURE_IDS)}, ensure_ascii=False))
    return 0


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="独立 x/y 光栅标度采集（flat + x 光栅 + y 光栅）；无 --execute 只打印 plan"
    )
    parser.add_argument("--out", type=Path, help="全新输出目录（离线规划与采集都要）")
    parser.add_argument("--execute", action="store_true", help="打开设备并采集（需台面锁）")
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
    parser.add_argument("--slm-number", type=int, default=DEFAULT_SLM_NUMBER, help="SLM 编号")
    parser.add_argument("--cam-type", default=DEFAULT_CAM_TYPE, help="相机类型")
    parser.add_argument("--cam-id", type=int, default=DEFAULT_CAM_ID, help="相机 id")
    parser.add_argument(
        "--frames", type=_positive_int, default=DEFAULT_FRAMES, help="每张图样的帧数（>=1）"
    )
    parser.add_argument(
        "--settle-s", type=float, default=DEFAULT_SETTLE_S, help="每帧前等待 SLM 稳定的秒数"
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
