# -*- coding: utf-8 -*-
"""§12.11.5 六帧映射消歧采集（离线可规划，硬件默认不打开）。

本脚本实现预注册 §12.11.5 的六帧消歧协议：在 SLM #1（1064 nm）上一次
open/close 会话里采集 6 帧（每图样 1 帧），用于判定面板当前生效的 2π 灰度
编码到底是设备读数 993 还是面板满量程 1024，并同时给出轴归属（x/y）、
blaze2d 单斑与 ladder_x 锚点。帧序、每图样帧数、两π灰度、相位/灰度哈希全部
由用户冻结，均不是 CLI 旋钮。

不做的事（与预注册一致）：不操作激光、不叠加离焦补偿（``focus_d_m1 = 0.0``，
无 Fresnel 面板）、不修改面板灰度 LUT（两种 scale 都是软件编码，写入面板前
不改 ``set_grayscale``）。没有 ``--execute`` 时只构建并打印确定性的 plan；
设备驱动只在 ``execute`` 里、拿到台面锁且新输出目录就绪之后才惰性导入。

所有相位数学来自 ``scripts.slm.issue247_geometry_capture_20261120``（单一构造
宿主），本文件从不重写相位公式；两π灰度→灰度的唯一 wrap 点是
``issue247.phase_to_gray``。

设计来源：预注册 §12.11.5。该节原文档随并发车道删除而不可恢复，判据文本
与设计来源说明见 ``PREREG_CRITERIA_VERBATIM`` / ``DESIGN_SOURCE_NOTE``。
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

# Phase math is owned by the issue247 geometry-capture module (the single
# builder home).  This file never reimplements a phase formula.
from scripts.slm.issue247_geometry_capture_20261120 import (
    blaze2d_phase,
    flat_phase,
    grating_phase,
    ladder_phase,
    phase_to_gray,
)

PLAN_SCHEMA = "issue247-s1-mapping-disambiguation-capture/v1"
SUMMARY_SCHEMA = "issue247-s1-mapping-disambiguation-summary/v1"
JOURNAL_SCHEMA = "issue247-s1-mapping-disambiguation-journal/v1"
RELEASE_SCHEMA = "issue247-s1-mapping-disambiguation-released/v1"

DEFAULT_PANEL_SHAPE = (1200, 1920)  # rows, cols (driver Panel_Res is cols, rows)
NATIVE_CCD_SHAPE = (1944, 2592)
BIT_DEPTH_REQUIRED = 12
FULL_SCALE_REQUIRED = (1 << BIT_DEPTH_REQUIRED) - 1
DEFAULT_EXPOSURE_MS = 0.087
DEFAULT_WAIT_IDLE_S = 600.0
DEFAULT_SLM_NUMBER = 1
DEFAULT_CAM_TYPE = "daheng"
DEFAULT_CAM_ID = 0
DEFAULT_SETTLE_S = 0.8
PREFLIGHT_DISCARD_FRAMES = 3
PREFLIGHT_MEASURE_FRAMES = 1
SATURATION_FRACTION = 0.9
LOCK_POLL_S = 1.0
PREREG_SECTION = "§12.11.5"
DESIGN_SOURCE = "issue247 跨会话复现预注册 §12.11.5"
DESIGN_SOURCE_NOTE = (
    "§12.11.5 原文档文件已随并发车道删除而不可恢复（全 refs / 悬空 commit / "
    "悬空 blob 均无命中）；本脚本的判据文本与设计来源说明按测试规格 "
    "(tests/ao_shaping/scripts/test_capture_s1_mapping_disambiguation_20261124.py)"
    " 与 §12.11 同族冻结计划逐条复原，非从原 doc 摘录。任何判据变更须人工复核。"
)

#: 候选 2π 编码：设备典型读数 993 (@1064 nm) 与面板满量程 1024（10-bit）。
SCALE_2PI_DEVICE_993 = 993
SCALE_2PI_PANEL_1024 = 1024
GRATING_PERIOD_PX = 128
BLAZE2D_PERIOD_X_PX = 128
BLAZE2D_PERIOD_Y_PX = 192
#: A1 平场 0 级参考（issue247 A1 帧质心，用户冻结）。
ZERO_ORDER_REFERENCE_PX = {"row": 1105.249, "col": 1227.556}
ZERO_ORDER_REFERENCE_TOLERANCE_PX = 5.0

#: 判据逐字文本（用户冻结，非 CLI 旋钮）。含 0 级参考复现、x 光栅 ±1 级
#: 位移预测窗、scale 对能量判别、ladder 阶距、blaze2d 单斑与整体饱和 abort。
PREREG_CRITERIA_VERBATIM = (
    "§12.11.5 六帧判据（用户冻结，逐字）："
    "① flat_scale993 复现 A1 0 级参考 (row 1105.249, col 1227.556)，偏移超过 "
    "5 px 即判本次会话与 A1 不同束流，六帧全部作废；"
    "② grating_x_p128 的 ±1 级相对 0 级在列向的位移须落在 56–59 px 预测窗内，"
    "行向残差 ≤3 px；"
    "③ 同一 base 的 scale993 与 scale1024 两帧相位必须逐字节相同、只有灰度不同："
    "若 scale1024 帧的 ±1 级能量比 scale993 低 >5%，判面板实际 2π=993；反向则判 "
    "1024，落入 ±5% 内判不可判定（不得改 LUT）；"
    "④ ladder_x 各阶相对 0 级间距须为 95 px 的整数倍（±2 px），用于核 axis 归属；"
    "⑤ blaze2d_px128_py192_sx+1_sy+1 必须只出现单一离轴斑（无 ±1 级梯），"
    "位置须与 p_x=128 / p_y=192 的预测一致，轴交换或符号翻转都会把它送到别处；"
    "⑥ 任一帧饱和（≥90% 满量程）或 preflight 饱和即整体 abort，重拍不算重试。"
)

#: 每个帧的冻结规格：id / a1 基图样 id / kind / 几何 / 2π 编码 / 角色。
#: ``frames`` 恒为 1（每图样单帧），是用户冻结值，不是 CLI 旋钮。
PATTERN_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "flat_scale993",
        "a1_pattern_id": "flat",
        "kind": "flat",
        "two_pi_gray_scale": SCALE_2PI_DEVICE_993,
        "role": "zero_order_reference",
    },
    {
        "id": "grating_x_p128_scale993",
        "a1_pattern_id": "grating_x_p128",
        "kind": "grating",
        "axis": 1,
        "period_px": GRATING_PERIOD_PX,
        "two_pi_gray_scale": SCALE_2PI_DEVICE_993,
        "role": "scale_pair_993",
    },
    {
        "id": "grating_x_p128_scale1024",
        "a1_pattern_id": "grating_x_p128",
        "kind": "grating",
        "axis": 1,
        "period_px": GRATING_PERIOD_PX,
        "two_pi_gray_scale": SCALE_2PI_PANEL_1024,
        "role": "scale_pair_1024",
    },
    {
        "id": "grating_y_p128_scale993",
        "a1_pattern_id": "grating_y_p128",
        "kind": "grating",
        "axis": 0,
        "period_px": GRATING_PERIOD_PX,
        "two_pi_gray_scale": SCALE_2PI_DEVICE_993,
        "role": "axis_y_control",
    },
    {
        "id": "blaze2d_px128_py192_sx+1_sy+1_scale993",
        "a1_pattern_id": "blaze2d_px128_py192_sx+1_sy+1",
        "kind": "blaze2d",
        "period_x_px": BLAZE2D_PERIOD_X_PX,
        "period_y_px": BLAZE2D_PERIOD_Y_PX,
        "sign_x": 1,
        "sign_y": 1,
        "two_pi_gray_scale": SCALE_2PI_DEVICE_993,
        "role": "single_spot_axis_sign",
    },
    {
        "id": "ladder_x_scale993",
        "a1_pattern_id": "ladder_x",
        "kind": "ladder",
        "axis": 1,
        "sign": 1,
        "two_pi_gray_scale": SCALE_2PI_DEVICE_993,
        "role": "ladder_anchor",
    },
)
CAPTURE_ORDER: tuple[str, ...] = tuple(spec["id"] for spec in PATTERN_SPECS)
_SPEC_BY_ID: dict[str, dict[str, Any]] = {spec["id"]: spec for spec in PATTERN_SPECS}
FRAMES_PER_PATTERN = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_phase(spec: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    """Materialize the phase array for one frozen pattern spec (raw radians)."""
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
    raise ValueError(f"unknown pattern kind {kind!r}")


@lru_cache(maxsize=4)
def _protocol_patterns_cached(shape: tuple[int, int]) -> tuple[dict[str, Any], ...]:
    """Build the six frozen patterns once per panel shape (immutable snapshot).

    Returns a tuple of plain dicts (no numpy arrays) so callers may cache and
    share; the arrays are hashed then released.  Only the record is kept.
    """
    shape = (int(shape[0]), int(shape[1]))
    records: list[dict[str, Any]] = []
    for index, spec in enumerate(PATTERN_SPECS):
        phase = build_phase(spec, shape)
        scale = int(spec["two_pi_gray_scale"])
        gray = phase_to_gray(phase, scale)
        unique = np.unique(gray)
        record: dict[str, Any] = {
            "id": spec["id"],
            "a1_pattern_id": spec["a1_pattern_id"],
            "kind": spec["kind"],
            "role": spec["role"],
            "index": index,
            "frames": FRAMES_PER_PATTERN,
            "two_pi_gray_scale": scale,
            "gray_modulus": scale + 1,
            "phase_sha256": digest(phase),
            "gray_sha256": digest(gray),
            "gray_max": int(gray.max()),
            "gray_min": int(gray.min()),
            "gray_unique_count": int(unique.size),
            "gray_unique_head": [int(v) for v in unique[:8]],
            "panel_shape_rows_cols": [shape[0], shape[1]],
        }
        for key in ("axis", "period_px", "sign_x", "sign_y", "sign"):
            if key in spec:
                record[key] = spec[key]
        records.append(record)
    return tuple(records)


def protocol_patterns() -> list[dict[str, Any]]:
    """The six frozen protocol patterns (independent copies of the cache)."""
    return [dict(item) for item in _protocol_patterns_cached(DEFAULT_PANEL_SHAPE)]


def focus_record(d_m1: float = 0.0) -> dict[str, Any]:
    """Frozen focus compensation: §12.11.5 has none (no Fresnel overlay)."""
    if float(d_m1) != 0.0:
        raise ValueError(
            "§12.11.5 冻结 focus_d_m1=0.0；本脚本不接受离焦补偿（无 Fresnel 面板）"
        )
    phase = np.zeros(DEFAULT_PANEL_SHAPE, dtype=np.float64)
    return {
        "focus_d_m1": 0.0,
        "compensation_applied": False,
        "focal_mm": None,
        "fresnel_phase_sha256": digest(phase),
        "wavelength_nm": 1064.0,
        "note": "§12.11.5 冻结无补偿：不叠加瞳面二次相位，光路几何保持 A1 基准。",
    }


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Build the deterministic, frozen plan (never touches hardware)."""
    exposure = float(args.exposure_ms)
    wait_idle = float(args.wait_until_idle)
    slm_number = int(args.slm_number)
    cam_type = str(args.cam_type)
    cam_id = int(args.cam_id)

    patterns = protocol_patterns()
    if tuple(item["id"] for item in patterns) != CAPTURE_ORDER:
        raise RuntimeError("pattern order drifted from the frozen CAPTURE_ORDER")

    focus = focus_record(0.0)
    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "generated_at_utc": utc_now(),
        "design_source": DESIGN_SOURCE,
        "design_source_note": DESIGN_SOURCE_NOTE,
        "prereg_section": PREREG_SECTION,
        "prereg_criteria_verbatim": PREREG_CRITERIA_VERBATIM,
        "capture_order": list(CAPTURE_ORDER),
        "n_frames_expected": len(CAPTURE_ORDER),
        "frames_per_pattern": FRAMES_PER_PATTERN,
        "patterns": patterns,
        "focus_compensation": focus,
        "exposure_ms_requested": exposure,
        "bit_depth_required": BIT_DEPTH_REQUIRED,
        "full_scale_required": FULL_SCALE_REQUIRED,
        "panel_shape_expected_rows_cols": [DEFAULT_PANEL_SHAPE[0], DEFAULT_PANEL_SHAPE[1]],
        "settle_s": DEFAULT_SETTLE_S,
        "wait_until_idle_s": wait_idle,
        "acquisition": {
            "slm_number": slm_number,
            "cam_type": cam_type,
            "cam_id": cam_id,
            "exposure_ms": exposure,
            "wait_until_idle_s": wait_idle,
        },
        "preflight": {
            "discard_frames": PREFLIGHT_DISCARD_FRAMES,
            "measure_frames": PREFLIGHT_MEASURE_FRAMES,
            "abort_on_saturation": True,
            "saturation_fraction": SATURATION_FRACTION,
        },
        "a1_zero_order_reference": {
            "row": ZERO_ORDER_REFERENCE_PX["row"],
            "col": ZERO_ORDER_REFERENCE_PX["col"],
            "source": "issue247 A1 平场质心（冻结）",
            "tolerance_px": ZERO_ORDER_REFERENCE_TOLERANCE_PX,
        },
        "saturation_fraction": SATURATION_FRACTION,
        "panel_grayscale_config_untouched": True,
        "note": (
            "两种 scale（993/1024）均为软件编码，写入面板前不改 set_grayscale；"
            "判据见 prereg_criteria_verbatim。"
        ),
    }
    return plan


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


def _panel_shape_from(slm: Any) -> tuple[int, int]:
    """(rows, cols) from the driver's Panel_Res, which is (cols, rows)."""
    panel = getattr(slm, "Panel_Res", None)
    if not panel or len(panel) != 2:
        logger.warning(
            "SLM Panel_Res missing/odd ({!r}); assuming frozen {}", panel,
            DEFAULT_PANEL_SHAPE,
        )
        return DEFAULT_PANEL_SHAPE
    return int(panel[1]), int(panel[0])


def _centroid_px(frame: np.ndarray) -> list[float]:
    """Baseline-subtracted centroid; uniform/blank frame -> geometric centre."""
    rows, cols = int(frame.shape[0]), int(frame.shape[1])
    data = frame.astype(np.float64)
    weights = data - np.median(data)
    weights[weights < 0.0] = 0.0
    total = float(weights.sum())
    if total <= 0.0:
        return [(rows - 1) / 2.0, (cols - 1) / 2.0]
    row_idx = np.arange(rows, dtype=np.float64)
    col_idx = np.arange(cols, dtype=np.float64)
    row = float((weights.sum(axis=1) * row_idx).sum() / total)
    col = float((weights.sum(axis=0) * col_idx).sum() / total)
    return [row, col]


def run_preflight(cam: Any, full_scale: float, journal: Any) -> dict[str, Any]:
    """Discard N frames then measure one; returns the frozen preflight record."""
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
    if saturated:
        logger.warning(
            "preflight saturated: peak {} >= {:.1f} (fraction {:.2f} of full "
            "scale {}); aborting per §12.11.5, no science frame",
            peak, threshold, SATURATION_FRACTION, full_scale,
        )
    return record


def execute(args: argparse.Namespace, plan: dict[str, Any]) -> int:
    """Acquire bench lock, open devices once, capture the six frozen frames."""
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
    preflight_record: dict[str, Any] | None = None
    flat_centroid: list[float] | None = None
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
    try:
        out.mkdir(parents=True, exist_ok=False)
        (out / "raw").mkdir()
        output_created = True
        write_json(out / "plan.json", plan)
        journal_path = out / "capture_journal.jsonl"

        focus = dict(plan["focus_compensation"])
        d_m1 = float(focus["focus_d_m1"])
        if not focus["compensation_applied"]:
            logger.warning("§12.11.5 无离焦补偿（冻结），结果按 A1 基准判读")

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

        with journal_path.open("a", encoding="utf-8") as journal:
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
                    f"{preflight_record['saturation_threshold_counts']} "
                    "§12.11.5 冻结判据：整体 abort，禁止重拍"
                )

            for item in plan["patterns"]:
                index = int(item["index"])
                phase = build_phase(_SPEC_BY_ID[item["id"]], DEFAULT_PANEL_SHAPE)
                gray = phase_to_gray(phase, int(item["two_pi_gray_scale"]))
                if digest(gray) != item["gray_sha256"]:
                    raise RuntimeError(
                        f"gray integrity check failed for {item['id']}"
                    )
                slm.display_data(gray)
                time.sleep(DEFAULT_SETTLE_S)
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
                raw_path = out / "raw" / f"{index:02d}_{item['id']}.npy"
                np.save(raw_path, frame)
                if item["id"] == "flat_scale993":
                    flat_centroid = _centroid_px(frame)
                row: dict[str, Any] = {
                    "index": index,
                    "pattern_id": item["id"],
                    "a1_pattern_id": item["a1_pattern_id"],
                    "role": item["role"],
                    "two_pi_gray_scale": int(item["two_pi_gray_scale"]),
                    "gray_modulus": int(item["gray_modulus"]),
                    "phase_sha256": item["phase_sha256"],
                    "gray_sha256": item["gray_sha256"],
                    "gray_max": int(item["gray_max"]),
                    "focus_d_m1": d_m1,
                    "compensation_applied": bool(focus["compensation_applied"]),
                    "focal_mm": focus["focal_mm"],
                    "bit_depth": bit_depth,
                    "full_scale": int(full_scale),
                    "panel_shape_rows_cols": list(panel_shape),
                    "exposure_ms_requested": float(args.exposure_ms),
                    "exposure_ms_actual": exposure_actual,
                    "frame_sha256": digest(frame),
                    "frame_shape": list(frame.shape),
                    "frame_max_counts": int(frame.max()),
                    "frame_mean_counts": float(frame.mean()),
                    "saturated": saturated,
                    "raw_file": str(raw_path.relative_to(out)),
                    "saved_at_utc": utc_now(),
                }
                frame_records.append(row)
                _journal(journal, "frame_saved", **row)
            _journal(
                journal,
                "capture_finished",
                status="ok",
                n_frames=len(frame_records),
            )
        status = "ok"
        exit_code = 0
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("§12.11.5 六帧采集失败")
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
        release["status"] = status
        release["elapsed_s"] = time.monotonic() - started
        release["finished_at_utc"] = utc_now()
        release["released_at_utc"] = utc_now()
        release["closed_normally"] = all(
            release[key] is True for key in ("slm_closed", "camera_closed", "lock_released")
        ) and status == "ok"
        if output_created:
            write_json(out / "released.json", release)
            offset = None
            if flat_centroid is not None:
                offset = [
                    flat_centroid[0] - ZERO_ORDER_REFERENCE_PX["row"],
                    flat_centroid[1] - ZERO_ORDER_REFERENCE_PX["col"],
                ]
            summary: dict[str, Any] = {
                "schema": SUMMARY_SCHEMA,
                "status": status,
                "n_frames": len(frame_records),
                "expected_frames": len(CAPTURE_ORDER),
                "frames": frame_records,
                "saturation_count": sum(bool(r["saturated"]) for r in frame_records),
                "same_session_flat_centroid_px": flat_centroid,
                "flat_centroid_offset_px": offset,
                "preflight": preflight_record,
                "focus_compensation": focus,
                "acquisition": dict(plan["acquisition"]),
                "prereg_section": PREREG_SECTION,
                "elapsed_s": release["elapsed_s"],
                "timestamp_utc": utc_now(),
                "error": error,
            }
            write_json(out / "summary.json", summary)
    return exit_code


def require_drive_board_entry(slm: Any) -> str | None:
    """Acquisition-entry drive-board guard, imported lazily from its single home.

    ``utils``-style duck typing is intentional: simulation panels and test
    doubles lack ``require_drive_board()`` and the shared helper treats that as
    a no-op.  A direct ``Santec.require_drive_board()`` call would break those
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


def self_test() -> int:
    """Offline invariant check; no hardware, no directories created."""
    patterns = protocol_patterns()
    assert [item["id"] for item in patterns] == list(CAPTURE_ORDER)
    assert len(patterns) == 6
    assert all(item["frames"] == 1 for item in patterns)
    assert [item["index"] for item in patterns] == list(range(6))
    scales = [item["two_pi_gray_scale"] for item in patterns]
    assert scales == [993, 993, 1024, 993, 993, 993]
    assert all(
        item["gray_modulus"] == item["two_pi_gray_scale"] + 1 for item in patterns
    )
    by_id = {item["id"]: item for item in patterns}
    x993 = by_id["grating_x_p128_scale993"]
    x1024 = by_id["grating_x_p128_scale1024"]
    assert x993["phase_sha256"] == x1024["phase_sha256"]
    assert x993["gray_sha256"] != x1024["gray_sha256"]
    assert x993["gray_max"] != x1024["gray_max"]
    assert by_id["grating_x_p128_scale993"]["gray_unique_count"] == 2
    assert by_id["grating_x_p128_scale1024"]["gray_unique_count"] == 2
    assert all(len(item["phase_sha256"]) == 64 for item in patterns)
    assert all(len(item["gray_sha256"]) == 64 for item in patterns)
    focus = focus_record(0.0)
    assert focus["compensation_applied"] is False and focus["focal_mm"] is None
    assert focus["fresnel_phase_sha256"] == digest(
        np.zeros(DEFAULT_PANEL_SHAPE, dtype=np.float64)
    )
    # grating phases are binary 0/pi; ladder spans the panel; blaze2d is a ramp.
    gx = build_phase(_SPEC_BY_ID["grating_x_p128_scale993"], DEFAULT_PANEL_SHAPE)
    assert set(np.unique(gx)).issubset({0.0, float(np.pi)})
    logger.info(
        "S1m §12.11.5 offline self-test passed: 6 frozen frames, "
        "scale pair shares phase and differs only in gray"
    )
    print(json.dumps({"self_test": "passed", "n_frames": 6}, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="§12.11.5 六帧映射消歧采集；无 --execute 只打印冻结 plan"
    )
    parser.add_argument(
        "--out", type=Path, help="全新输出目录（离线规划与采集都要）"
    )
    parser.add_argument(
        "--execute", action="store_true", help="打开设备并采集（需台面锁）"
    )
    parser.add_argument("--self-test", action="store_true", help="离线自检")
    parser.add_argument(
        "--exposure-ms", type=float, default=DEFAULT_EXPOSURE_MS,
        help="相机曝光 (ms)"
    )
    parser.add_argument(
        "--wait-until-idle", type=float, default=DEFAULT_WAIT_IDLE_S,
        help="等待台面锁的最长秒数"
    )
    parser.add_argument(
        "--slm-number", type=int, default=DEFAULT_SLM_NUMBER, help="SLM 编号"
    )
    parser.add_argument("--cam-type", default=DEFAULT_CAM_TYPE, help="相机类型")
    parser.add_argument("--cam-id", type=int, default=DEFAULT_CAM_ID, help="相机 id")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    if args.out is None:
        parser.error("--out 是必需参数（离线规划与采集都要写出目录）")

    if args.execute and args.out.exists():
        parser.error(f"--execute 要求全新输出目录，{args.out} 已存在")

    plan = build_plan(args)
    if not args.execute:
        print(json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    return execute(args, plan)


if __name__ == "__main__":
    raise SystemExit(main())
