#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§12.11.5 S1 判据②–⑥ 的纯离线判读（只读 2026-11-26 六帧）。

本脚本不操作 SLM / CCD / 激光，不申请台面锁，不联网；只读
``data/debug/slm_transfer/s1_verify_20261126_retry/`` 下的六帧与清单，
按冻结判据逐字文本（``PREREG_CRITERIA_VERBATIM``）给出
「已证实有效 / 已证实无效 / 证据不足」三档结论。判据① 不重判。

阈值、预测窗、能量窗口半径均来自冻结文本或显式写在 ``FROZEN_*`` 常量中，
不是 CLI 旋钮。判据⑤ 的冻结文本没有数值容差，因此只给残差与定性匹配，
判「证据不足（缺冻结容差）」，不自设阈值。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "src", ROOT / "libs", ROOT, Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts.slm.capture_s1_mapping_disambiguation_20261124 import (
    PREREG_CRITERIA_VERBATIM,
    flat_signal_measurement,
)
from scripts.slm.issue247_axis_map_solve_20261121 import find_spots

SCHEMA = "s1-offline-judge/v1"
DEFAULT_INPUT_DIR = ROOT / "data/debug/slm_transfer/s1_verify_20261126_retry/raw"
DEFAULT_OUTPUT_DIR = ROOT / "report/shape_trials/s1_offline_judge_20261127"
AXIS_STATUS_PATH = ROOT / "data/debug/hw_window_20261121b/issue247_axis/axis_mapping_status.json"

FRAME_FILES: dict[str, str] = {
    "flat_scale993": "00_flat_scale993.npy",
    "grating_x_p128_scale993": "01_grating_x_p128_scale993.npy",
    "grating_x_p128_scale1024": "02_grating_x_p128_scale1024.npy",
    "grating_y_p128_scale993": "03_grating_y_p128_scale993.npy",
    "blaze2d_px128_py192_sx+1_sy+1_scale993": "04_blaze2d_px128_py192_sx+1_sy+1_scale993.npy",
    "ladder_x_scale993": "05_ladder_x_scale993.npy",
}

#: 判据② 列向位移窗。原冻结窗 56–59 px 取自 y 光栅列向标度（≈59.93 px），
#: 与 x 光栅列向实测 93.22 / 95.09 px（11-26）及 11-21 轴解 95.16 px 冲突；
#: 经用户 2026-11-27 授权改窗，改为对称覆盖 x 光栅 ±1 级列向位移实测区间的
#: (92.0, 97.0) px。行向残差 ≤3 px 为冻结条件，保持不变。
FROZEN_2_COL_WINDOW_PX = (92.0, 97.0)
FROZEN_2_ROW_RESIDUAL_MAX_PX = 3.0
WINDOW_2_REVISION = (
    "判据② 列向窗由冻结 56–59 px 改为 92–97 px（用户 2026-11-27 授权改窗）："
    "原窗取自 y 光栅列向标度，与 x 光栅列向实测 93.22/95.09 px 冲突；"
    "新窗对称覆盖 x 光栅 ±1 级列向位移实测区间，行向残差 ≤3 px 保持冻结不变。"
)
FROZEN_3_REL_BAND = 0.05
FROZEN_4_STEP_PX = 95.0
FROZEN_4_TOL_PX = 2.0
#: ladder_x_scale993 的图样由 frozen ladder_phase() 构造，阶次固定为 (1, 3)，
#: 故预期间距只有 1×95=95 px 与 3×95=285 px，不是 1、2 阶。
FROZEN_LADDER_ORDERS: tuple[int, ...] = (1, 3)
LADDER_ORDER_SOURCE = (
    "scripts/slm/capture_s1_mapping_disambiguation_20261124.py PATTERN_SPECS[ladder_x_scale993] "
    "→ scripts/slm/issue247_geometry_capture_20261120.py ladder_phase() / LADDER_ORDERS=(1, 3) (L143–148)"
)
FROZEN_6_SATURATION_FRACTION = 0.9
#: 非冻结但显式声明的测量口径（跨半径一致，不为单个半径挑结论）。
SPOT_MERGE_RADIUS_PX = 3.0
ENERGY_RADII_PX = (5, 8, 12, 16)
ZERO_SPOT_MATCH_RADIUS_PX = 12.0
OFF_AXIS_MIN_DIST_PX = 20.0
EDGE_ARTIFACT_ROW_MAX = 100
NEAR_AXIS_ROW_TOL_PX = 30.0
LADDER_MIN_PEAK_FRAC = 0.02
BLAZE_SEARCH_RADIUS_PX = 400.0
BLAZE_A_MATRIX: tuple[tuple[float, float], tuple[float, float]] = (
    (26.2532, -7673.4447),
    (7705.9084, -22.0052),
)
BLAZE_PERIOD_X_PX = 128.0
BLAZE_PERIOD_Y_PX = 192.0
BLAZE_A1_HISTORICAL_MEASURED = (-59.1629, 40.8922)
BLAZE_A1_HISTORICAL_FIT_RESIDUAL_PX = 1.1335

Verdict = Literal["已证实有效", "已证实无效", "证据不足"]
VALID: Verdict = "已证实有效"
INVALID: Verdict = "已证实无效"
INSUFFICIENT: Verdict = "证据不足"

SPOT_MERGE_RULE = (
    "find_spots() 在平顶多像素上会给出相邻重复峰：按 peak 降序贪心保留，"
    "与任一已保留斑的 |Δrow|≤3 px 且 |Δcol|≤3 px 的峰判为同一斑并丢弃，"
    "保留 peak 最高者作为该斑代表。"
)
ENERGY_RULE = (
    "±1 能量口径（冻结文本未给）：对每个 ±1 斑中心取方形窗口 "
    "row/col ∈ [c-R, c+R]（R=5/8/12/16 px 全覆盖），baseline=该帧全图 median，"
    "对 max(pixel-baseline, 0) 求和；两帧共用 scale993 斑中心，"
    "另用各帧自身斑中心作交叉核对；比值 ratio=sum1024/sum993。"
)


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def merge_nearby_spots(
    spots: list[dict[str, float]], radius_px: float = SPOT_MERGE_RADIUS_PX
) -> list[dict[str, float]]:
    """按 peak 降序合并空间邻近的重复峰；规则见 ``SPOT_MERGE_RULE``。"""
    kept: list[dict[str, float]] = []
    for spot in sorted(spots, key=lambda s: -float(s["peak"])):
        duplicate = any(
            abs(float(spot["row"]) - float(other["row"])) <= radius_px
            and abs(float(spot["col"]) - float(other["col"])) <= radius_px
            for other in kept
        )
        if not duplicate:
            kept.append(spot)
    return kept


def analyse_frame(frame: np.ndarray, zero: tuple[float, float], key: str) -> dict[str, Any]:
    """复用冻结 find_spots()，给出原始/合并斑表、0 级斑与离轴斑。"""
    raw = find_spots(frame)
    merged = merge_nearby_spots(raw)
    zero_candidates = [
        spot for spot in merged
        if float(np.hypot(spot["row"] - zero[0], spot["col"] - zero[1]))
        <= ZERO_SPOT_MATCH_RADIUS_PX
    ]
    zero_spot = max(zero_candidates, key=lambda s: float(s["peak"])) if zero_candidates else None
    off_axis: list[dict[str, float]] = []
    for spot in merged:
        if spot is zero_spot:
            continue
        if int(spot["row"]) < EDGE_ARTIFACT_ROW_MAX:
            continue  # 冻结 find_spots() 注明的上边缘杂散峰
        if float(np.hypot(spot["row"] - zero[0], spot["col"] - zero[1])) < OFF_AXIS_MIN_DIST_PX:
            continue
        off_axis.append(spot)

    def describe(spot: dict[str, float]) -> dict[str, float]:
        return {
            "row": _round(spot["row"]),
            "col": _round(spot["col"]),
            "peak": _round(spot["peak"], 2),
            "dr": _round(spot["row"] - zero[0]),
            "dc": _round(spot["col"] - zero[1]),
        }

    return {
        "pattern_id": key,
        "raw_spot_count": len(raw),
        "merged_spot_count": len(merged),
        "zero_order_spot": describe(zero_spot) if zero_spot else None,
        "off_axis_spots": [describe(spot) for spot in off_axis],
        "max_counts": float(frame.max()) if frame.size else 0.0,
    }


def _near_axis_pair(
    info: dict[str, Any], zero: tuple[float, float]
) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    """近轴 ±1 识别：|dr|≤30 px 的离轴斑中，列正向/负向各取 peak 最高者。"""
    near = [
        spot for spot in info["off_axis_spots"]
        if abs(float(spot["row"]) - zero[0]) <= NEAR_AXIS_ROW_TOL_PX
    ]
    plus = [s for s in near if float(s["dc"]) > 0]
    minus = [s for s in near if float(s["dc"]) < 0]
    best_plus = max(plus, key=lambda s: float(s["peak"])) if plus else None
    best_minus = max(minus, key=lambda s: float(s["peak"])) if minus else None
    return best_plus, best_minus


def judge_2(info: dict[str, Any], zero: tuple[float, float]) -> dict[str, Any]:
    """判据②：grating_x_p128 的 ±1 级列向窗 92–97 px（改窗 v2）、行向残差 ≤3 px。"""
    plus, minus = _near_axis_pair(info, zero)
    if plus is None or minus is None:
        return {
            "verdict": INSUFFICIENT,
            "evidence": {"reason": "近轴 ±1 斑未成对检出，无法计算位移"},
        }
    lo, hi = FROZEN_2_COL_WINDOW_PX
    checks: list[dict[str, Any]] = []
    for label, spot in (("+1", plus), ("-1", minus)):
        dc, dr = float(spot["dc"]), float(spot["dr"])
        col_ok = lo <= abs(dc) <= hi
        row_ok = abs(dr) <= FROZEN_2_ROW_RESIDUAL_MAX_PX
        checks.append({
            "order": label,
            "row_px": spot["row"],
            "col_px": spot["col"],
            "dc_px": _round(dc),
            "abs_dc_px": _round(abs(dc)),
            "dr_residual_px": _round(dr),
            "col_in_window": col_ok,
            "row_residual_ok": row_ok,
        })
    ok = all(item["col_in_window"] and item["row_residual_ok"] for item in checks)
    return {
        "verdict": VALID if ok else INVALID,
        "evidence": {
            "criterion_text": "② grating_x_p128 的 ±1 级相对 0 级列向位移 92–97 px（改窗 v2）、行向残差 ≤3 px",
            "window_revision": WINDOW_2_REVISION,
            "thresholds_px": {"col_min": lo, "col_max": hi, "row_residual_max": FROZEN_2_ROW_RESIDUAL_MAX_PX},
            "formula": "dc = spot.col - zero.col; dr = spot.row - zero.row; 判 |dc|∈[92,97] 且 |dr|≤3",
            "zero_order_px": [_round(zero[0]), _round(zero[1])],
            "orders": checks,
        },
    }


def _window_energy(frame: np.ndarray, row: float, col: float, radius: int, baseline: float) -> float:
    r0, c0 = int(round(row)), int(round(col))
    window = frame[r0 - radius:r0 + radius + 1, c0 - radius:c0 + radius + 1].astype(np.float64)
    return float(np.maximum(window - baseline, 0.0).sum())


def judge_3(
    info_993: dict[str, Any],
    info_1024: dict[str, Any],
    frames: dict[str, np.ndarray],
    summary: dict[str, Any],
    zero: tuple[float, float],
) -> dict[str, Any]:
    """判据③：相位哈希必须相同、灰度必须不同；±1 能量跨半径判 993/1024。"""
    by_id = {
        str(item.get("pattern_id")): item
        for item in summary.get("frames", [])
        if isinstance(item, dict)
    }
    f993 = by_id.get("grating_x_p128_scale993", {})
    f1024 = by_id.get("grating_x_p128_scale1024", {})
    phase_993 = str(f993.get("phase_sha256", ""))
    phase_1024 = str(f1024.get("phase_sha256", ""))
    gray_993 = str(f993.get("gray_sha256", ""))
    gray_1024 = str(f1024.get("gray_sha256", ""))
    phase_same = bool(phase_993 and phase_993 == phase_1024)
    gray_different = bool(gray_993 and gray_1024 and gray_993 != gray_1024)

    hash_evidence = {
        "phase_sha256_993": phase_993,
        "phase_sha256_1024": phase_1024,
        "phase_byte_identical": phase_same,
        "gray_sha256_993": gray_993,
        "gray_sha256_1024": gray_1024,
        "gray_different": gray_different,
        "source": "summary.json frames[].phase_sha256 / gray_sha256（脚本预注册哈希）",
    }

    plus_993, minus_993 = _near_axis_pair(info_993, zero)
    plus_1024, minus_1024 = _near_axis_pair(info_1024, zero)
    if plus_993 is None or minus_993 is None or plus_1024 is None or minus_1024 is None:
        return {
            "verdict": INSUFFICIENT,
            "evidence": {
                "hash_check": hash_evidence,
                "reason": "两帧近轴 ±1 斑未成对检出，无法计算能量比",
            },
        }

    key_993 = "grating_x_p128_scale993"
    key_1024 = "grating_x_p128_scale1024"
    array_993 = frames[key_993]
    array_1024 = frames[key_1024]
    baseline_993 = float(np.median(array_993))
    baseline_1024 = float(np.median(array_1024))

    def energy(frame: np.ndarray, baseline: float, pair: tuple[dict[str, float] | None, ...], radius: int) -> float:
        return sum(
            _window_energy(frame, float(spot["row"]), float(spot["col"]), radius, baseline)
            for spot in pair if spot is not None
        )

    rows: list[dict[str, Any]] = []
    for radius in ENERGY_RADII_PX:
        for mode in ("common_993_centers", "own_centers"):
            pair_a = (plus_993, minus_993)
            pair_b = (plus_993, minus_993) if mode == "common_993_centers" else (plus_1024, minus_1024)
            sum_993 = energy(array_993, baseline_993, pair_a, radius)
            sum_1024 = energy(array_1024, baseline_1024, pair_b, radius)
            ratio = float(sum_1024 / sum_993) if sum_993 > 0 else float("nan")
            if not np.isfinite(ratio):
                branch = "不可判定"
            elif ratio < 1.0 - FROZEN_3_REL_BAND:
                branch = "993"
            elif ratio > 1.0 + FROZEN_3_REL_BAND:
                branch = "1024"
            else:
                branch = "不可判定"
            rows.append({
                "radius_px": radius,
                "center_mode": mode,
                "sum_993": _round(sum_993, 2),
                "sum_1024": _round(sum_1024, 2),
                "ratio_1024_over_993": _round(ratio, 6),
                "relative_change_pct": _round((ratio - 1.0) * 100.0, 3),
                "branch": branch,
            })

    branches = {str(row["branch"]) for row in rows}
    if not phase_same:
        verdict: Verdict = INVALID
        resolved = "不可判定（前置条件不成立）"
    elif not gray_different:
        verdict = INSUFFICIENT
        resolved = "不可判定（灰度哈希相同）"
    elif branches == {"993"}:
        verdict, resolved = VALID, "993"
    elif branches == {"1024"}:
        verdict, resolved = VALID, "1024"
    else:
        verdict, resolved = INSUFFICIENT, "不可判定（能量比跨窗口不一致或落入 ±5% 内）"

    return {
        "verdict": verdict,
        "evidence": {
            "criterion_text": (
                "③ 相位逐字节相同、只有灰度不同；1024 的 ±1 能量比 993 低 >5% 判 993，"
                "反向判 1024，±5% 内判不可判定"
            ),
            "hash_check": hash_evidence,
            "window_rule": ENERGY_RULE,
            "baseline_counts": {"993": _round(baseline_993, 2), "1024": _round(baseline_1024, 2)},
            "threshold": {"relative_band": FROZEN_3_REL_BAND, "low_branch": "< 0.95", "high_branch": "> 1.05"},
            "pairs": {
                "993_plus": plus_993, "993_minus": minus_993,
                "1024_plus": plus_1024, "1024_minus": minus_1024,
            },
            "energy_rows": rows,
            "resolved_two_pi_gray": resolved,
        },
    }


def judge_4(info: dict[str, Any]) -> dict[str, Any]:
    """判据④：ladder_x 各阶对齐冻结阶次 (1,3) 的 95/285 px，容差 ±2 px。

    归属规则（非冻结阈值，仅证据口径）：主轴按离轴斑最大位移选 row/col；
    按 peak 降序把每个斑归到“距其最近的尚未占用的冻结阶次目标”，
    每个阶次只占用一个斑；剩余斑不参与主阶裁定，记为“未归属/疑似伪斑”。
    """
    expected_orders = [
        {"order": order, "expected_px": _round(order * FROZEN_4_STEP_PX, 1)}
        for order in FROZEN_LADDER_ORDERS
    ]
    base_evidence: dict[str, Any] = {
        "criterion_text": "④ ladder_x 各阶相对 0 级间距须为 95 px 整数倍（±2 px）",
        "order_source": LADDER_ORDER_SOURCE,
        "expected_orders": expected_orders,
        "expected_note": "冻结阶次 (1,3)：预期主阶只有 95 px 与 285 px，不含 2 阶 190 px",
        "thresholds_px": {"step": FROZEN_4_STEP_PX, "tolerance": FROZEN_4_TOL_PX},
        "formula": "step=|spot.dr|（主轴为 row）或 |spot.dc|；残差=|step-95×冻结阶次| ≤2 px",
        "assignment_rule": (
            "按 peak 降序把每个离轴斑归到距其最近且尚未占用的冻结阶次目标（95/285 px）；"
            "每个阶次只占用一个斑，剩余斑列为未归属/疑似伪斑，不参与主阶裁定"
        ),
        "spot_merge_rule": SPOT_MERGE_RULE,
    }
    off_axis = info["off_axis_spots"]
    if not off_axis:
        return {
            "verdict": INSUFFICIENT,
            "evidence": {**base_evidence, "reason": "ladder_x 未检出离轴阶斑"},
        }

    max_dr = max(abs(float(s["dr"])) for s in off_axis)
    max_dc = max(abs(float(s["dc"])) for s in off_axis)
    axis = "row" if max_dr >= max_dc else "col"
    key = "dr" if axis == "row" else "dc"
    candidates = sorted(
        [
            {
                "row_px": spot["row"], "col_px": spot["col"], "peak": spot["peak"],
                "step_px": _round(abs(float(spot[key]))),
            }
            for spot in off_axis
        ],
        key=lambda item: -float(item["peak"]),
    )

    assigned: dict[int, dict[str, Any]] = {}
    assigned_indices: set[int] = set()
    for index, candidate in enumerate(candidates):
        step = float(candidate["step_px"])
        nearest_order = min(
            FROZEN_LADDER_ORDERS,
            key=lambda order: abs(step - order * FROZEN_4_STEP_PX),
        )
        if nearest_order in assigned:
            continue
        target = _round(nearest_order * FROZEN_4_STEP_PX, 1)
        residual = abs(step - target)
        assigned[nearest_order] = {
            **candidate,
            "axis": axis,
            "order": nearest_order,
            "target_px": target,
            "residual_px": _round(residual),
            "within_tolerance": bool(residual <= FROZEN_4_TOL_PX),
        }
        assigned_indices.add(index)

    main_orders: list[dict[str, Any]] = []
    for order in FROZEN_LADDER_ORDERS:
        if order in assigned:
            main_orders.append(assigned[order])
        else:
            main_orders.append({
                "order": order,
                "expected_px": _round(order * FROZEN_4_STEP_PX, 1),
                "status": "冻结阶次未检出对应斑",
                "within_tolerance": False,
            })
    unassigned = [
        {
            **candidate,
            "status": "未归属/疑似伪斑",
            "note": "冻结阶次仅 (1,3)，对应主阶斑已由更高 peak 斑占用；该斑不参与主阶裁定",
        }
        for index, candidate in enumerate(candidates)
        if index not in assigned_indices
    ]
    missing = [order for order in FROZEN_LADDER_ORDERS if order not in assigned]
    verdict: Verdict = (
        INSUFFICIENT if missing
        else (VALID if all(bool(item["within_tolerance"]) for item in main_orders) else INVALID)
    )
    return {
        "verdict": verdict,
        "evidence": {
            **base_evidence,
            "dominant_axis": axis,
            "main_orders": main_orders,
            "unassigned_spots": unassigned,
            "missing_orders": list(missing),
            "reason": (
                "主阶斑已按冻结阶次 (1,3) 归属；残差均超 ±2 px"
                if not missing and verdict == INVALID else None
            ),
        },
    }


def analyze_5_difference(
    measured: tuple[float, float],
    predicted: tuple[float, float],
    zero: tuple[float, float],
    axis_status: dict[str, Any],
) -> dict[str, Any]:
    """判据⑤ 差异来源分解（不设容差，只给量化归因，不改裁定）。"""
    dr, dc = measured
    pred_dr, pred_dc = predicted
    resid_dr = dr - pred_dr
    resid_dc = dc - pred_dc
    resid_norm = float(np.hypot(resid_dr, resid_dc))
    vs_history = float(np.hypot(dr - BLAZE_A1_HISTORICAL_MEASURED[0], dc - BLAZE_A1_HISTORICAL_MEASURED[1]))
    blaze = axis_status.get("blaze2d", {})
    fit_rms = float(blaze.get("residual_rms_px", float("nan")))
    fit_max = float(blaze.get("max_point_residual_px", float("nan")))
    ref = axis_status.get("zero_order_reference_px", {})
    ref_row = float(ref.get("row", float("nan")))
    ref_col = float(ref.get("col", float("nan")))
    origin_dr = zero[0] - ref_row
    origin_dc = zero[1] - ref_col
    origin_norm = float(np.hypot(origin_dr, origin_dc))
    resid_angle = float(np.degrees(np.arctan2(resid_dc, resid_dr)))
    spot_angle = float(np.degrees(np.arctan2(dc, dr)))
    return {
        "residual_to_prediction_px": _round(resid_norm),
        "residual_vector_dr_dc": [_round(resid_dr), _round(resid_dc)],
        "residual_to_a1_historical_px": _round(vs_history),
        "model_fit_residual_rms_px": _round(fit_rms) if np.isfinite(fit_rms) else None,
        "model_fit_residual_max_px": _round(fit_max) if np.isfinite(fit_max) else None,
        "zero_order_origin_shift_px": _round(origin_norm),
        "zero_order_origin_shift_dr_dc": [_round(origin_dr), _round(origin_dc)],
        "spot_direction_deg": _round(spot_angle),
        "residual_direction_deg": _round(resid_angle),
        "sources": [
            "A 矩阵拟合残差：11-21 blaze2d 族自身 RMS 约 %.3f px / 最大 %.4f px，当前 %.4f px 残差约为其数倍，超出模型自身噪声"
            % (fit_rms, fit_max, resid_norm),
            "会话间漂移：当前实测距同图样 11-21 A1 实测 %.4f px，大于模型拟合残差，提示 11-21→11-26 存在离轴映射漂移"
            % vs_history,
            "0 级原点偏移：v2 掩膜质心相对 axis_status 参考仅 %.4f px，不足以解释 %.4f px 残差"
            % (origin_norm, resid_norm),
            "残差方向：残差向量方向 %.1f° 与斑点方向 %.1f° 不共线，非纯径向标度误差，更像旋转/平移/放大复合漂移"
            % (resid_angle, spot_angle),
        ],
    }


def judge_5(info: dict[str, Any], zero: tuple[float, float], axis_status: dict[str, Any]) -> dict[str, Any]:
    """判据⑤：单斑与预测位置；冻结文本无数值容差，位置一致性只给残差。"""
    a11, a12 = BLAZE_A_MATRIX[0]
    a21, a22 = BLAZE_A_MATRIX[1]
    fy = 1.0 / BLAZE_PERIOD_Y_PX
    fx = 1.0 / BLAZE_PERIOD_X_PX
    predicted = (a11 * fy + a12 * fx, a21 * fy + a22 * fx)
    off_axis = [
        s for s in info["off_axis_spots"]
        if float(np.hypot(float(s["dr"]), float(s["dc"]))) <= BLAZE_SEARCH_RADIUS_PX
    ]
    evidence: dict[str, Any] = {
        "criterion_text": (
            "⑤ blaze2d_px128_py192_sx+1_sy+1 必须只出现单一离轴斑（无 ±1 级梯），"
            "位置须与 p_x=128/p_y=192 的预测一致"
        ),
        "model": {
            "source": str(AXIS_STATUS_PATH),
            "field": "blaze2d.A_matrix",
            "A_matrix": [list(BLAZE_A_MATRIX[0]), list(BLAZE_A_MATRIX[1])],
            "formula": "(dr,dc) = A @ (fy,fx); fy=1/192, fx=1/128",
            "predicted_dr_dc": [_round(predicted[0]), _round(predicted[1])],
            "a1_historical_measured_dr_dc": list(BLAZE_A1_HISTORICAL_MEASURED),
            "a1_historical_fit_residual_px": BLAZE_A1_HISTORICAL_FIT_RESIDUAL_PX,
        },
        "frozen_numeric_tolerance_px": None,
        "spot_merge_rule": SPOT_MERGE_RULE,
        "off_axis_count": len(off_axis),
        "off_axis_spots": off_axis,
    }
    if len(off_axis) != 1:
        evidence["reason"] = "离轴斑数 ≠ 1，违反冻结的『只出现单一离轴斑』"
        return {"verdict": INVALID, "evidence": evidence}

    spot = off_axis[0]
    dr, dc = float(spot["dr"]), float(spot["dc"])
    residual = float(np.hypot(dr - predicted[0], dc - predicted[1]))
    vs_history = float(np.hypot(dr - BLAZE_A1_HISTORICAL_MEASURED[0], dc - BLAZE_A1_HISTORICAL_MEASURED[1]))
    sign_match = bool(np.sign(dr) == np.sign(predicted[0]) and np.sign(dc) == np.sign(predicted[1]))
    evidence.update({
        "measured_dr_dc": [_round(dr), _round(dc)],
        "single_off_axis_spot": True,
        "no_plus_minus_ladder": True,
        "sign_quadrant_match": sign_match,
        "residual_to_prediction_px": _round(residual),
        "residual_to_a1_historical_spot_px": _round(vs_history),
        "difference_source_analysis": analyze_5_difference(
            (dr, dc), predicted, zero, axis_status
        ),
        "reason": (
            "单斑与象限/符号一致，但冻结文本没有给位置数值容差；"
            "按预注册纪律不自设阈值，判证据不足（缺冻结容差）。"
            "差异来源已做量化归因（见 difference_source_analysis），不改裁定。"
        ),
    })
    return {"verdict": INSUFFICIENT, "evidence": evidence}


def judge_6(
    frames: dict[str, np.ndarray],
    summary: dict[str, Any],
    journal: list[dict[str, Any]],
) -> dict[str, Any]:
    """判据⑥：任一帧或 preflight 饱和（≥90% 满量程）即整体 abort。"""
    records = {
        str(item.get("pattern_id")): item
        for item in summary.get("frames", [])
        if isinstance(item, dict)
    }
    per_frame: list[dict[str, Any]] = []
    aborted = False
    for key, array in frames.items():
        record = records.get(key, {})
        full_scale = float(record.get("full_scale", 0.0))
        threshold = FROZEN_6_SATURATION_FRACTION * full_scale
        observed = float(array.max()) if array.size else 0.0
        saturated = bool(threshold > 0 and observed >= threshold)
        reported = bool(record.get("saturated", False))
        aborted = aborted or saturated
        per_frame.append({
            "pattern_id": key,
            "full_scale": _round(full_scale, 1),
            "threshold_counts": _round(threshold, 2),
            "observed_max_counts": _round(observed, 1),
            "summary_max_counts": record.get("frame_max_counts"),
            "summary_saturated": reported,
            "saturated_from_raw": saturated,
        })

    preflight = next((item for item in journal if item.get("event") == "preflight"), {})
    pre_peak = float(preflight.get("peak_counts", 0.0))
    pre_threshold = float(preflight.get("saturation_threshold_counts", 0.0))
    pre_saturated = bool(preflight.get("saturated", False)) or bool(pre_threshold > 0 and pre_peak >= pre_threshold)
    aborted = aborted or pre_saturated

    return {
        "verdict": INVALID if aborted else VALID,
        "evidence": {
            "criterion_text": "⑥ 任一帧饱和（≥90% 满量程）或 preflight 饱和即整体 abort，重拍不算重试",
            "threshold_fraction": FROZEN_6_SATURATION_FRACTION,
            "formula": "saturated = observed_max_counts >= 0.9 * full_scale",
            "per_frame": per_frame,
            "preflight": {
                "peak_counts": _round(pre_peak, 1),
                "saturation_threshold_counts": _round(pre_threshold, 2),
                "saturated": pre_saturated,
                "source": "capture_journal.jsonl event=preflight",
            },
            "abort_triggered": aborted,
        },
    }


def _load_inputs(input_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    parent = input_dir.parent
    summary = json.loads((parent / "summary.json").read_text(encoding="utf-8"))
    journal = [
        json.loads(line)
        for line in (parent / "capture_journal.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    axis_status = json.loads(AXIS_STATUS_PATH.read_text(encoding="utf-8"))
    frames: dict[str, np.ndarray] = {}
    for key, filename in FRAME_FILES.items():
        frames[key] = np.load(input_dir / filename, allow_pickle=False)
    return frames, summary, journal, axis_status


def run_judgment(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    """六帧判读并写 JSON/MD 报告；只读输入，只写输出新目录。"""
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    frames, summary, journal, axis_status = _load_inputs(input_dir)
    flat_measurement = flat_signal_measurement(frames["flat_scale993"])
    zero = (float(flat_measurement["centroid_px"][0]), float(flat_measurement["centroid_px"][1]))
    if not bool(flat_measurement["has_signal"]):
        raise RuntimeError("flat 帧 has_signal=False：0 级质心是几何中心，不得用于判据②–⑥")

    info = {key: analyse_frame(array, zero, key) for key, array in frames.items()}
    criteria = {
        "2": judge_2(info["grating_x_p128_scale993"], zero),
        "3": judge_3(
            info["grating_x_p128_scale993"], info["grating_x_p128_scale1024"], frames, summary, zero
        ),
        "4": judge_4(info["ladder_x_scale993"]),
        "5": judge_5(info["blaze2d_px128_py192_sx+1_sy+1_scale993"], zero, axis_status),
        "6": judge_6(frames, summary, journal),
    }
    invalid = [k for k, v in criteria.items() if v["verdict"] == INVALID]
    insufficient = [k for k, v in criteria.items() if v["verdict"] == INSUFFICIENT]
    overall: Verdict = INVALID if invalid else (INSUFFICIENT if insufficient else VALID)

    a_matrix = axis_status.get("measured_principal_spots", {}).get(
        "blaze2d_px128_py192_sx+1_sy+1", {}
    )
    provenance = {
        "raw_dir": str(input_dir),
        "summary_path": str(input_dir.parent / "summary.json"),
        "journal_path": str(input_dir.parent / "capture_journal.jsonl"),
        "axis_status_path": str(AXIS_STATUS_PATH),
        "axis_status_a_matrix_present": bool(a_matrix),
        "frame_sha256": {
            filename: _sha256_file(input_dir / filename) for filename in FRAME_FILES.values()
        },
        "git_head": _git_head(),
    }
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": provenance["git_head"],
        "scope": "§12.11.5 判据②–⑥ 纯离线判读；判据① 不重判",
        "prereg_criteria_verbatim": PREREG_CRITERIA_VERBATIM,
        "frozen_constants": {
            "criterion_2": {
                "col_window_px": list(FROZEN_2_COL_WINDOW_PX),
                "row_residual_max_px": FROZEN_2_ROW_RESIDUAL_MAX_PX,
                "window_revision": WINDOW_2_REVISION,
            },
            "criterion_3": {"relative_band": FROZEN_3_REL_BAND},
            "criterion_4": {
                "step_px": FROZEN_4_STEP_PX,
                "tolerance_px": FROZEN_4_TOL_PX,
                "frozen_orders": list(FROZEN_LADDER_ORDERS),
                "expected_spacings_px": [order * FROZEN_4_STEP_PX for order in FROZEN_LADDER_ORDERS],
                "order_source": LADDER_ORDER_SOURCE,
            },
            "criterion_6": {"saturation_fraction": FROZEN_6_SATURATION_FRACTION},
            "spot_merge_rule": SPOT_MERGE_RULE,
            "energy_window_rule": ENERGY_RULE,
        },
        "functions_used": [
            "scripts.slm.capture_s1_mapping_disambiguation_20261124.PREREG_CRITERIA_VERBATIM",
            "scripts.slm.capture_s1_mapping_disambiguation_20261124.flat_signal_measurement",
            "scripts.slm.issue247_axis_map_solve_20261121.find_spots",
        ],
        "provenance": provenance,
        "zero_order": {
            "source": "flat_scale993 via flat_signal_measurement() v2 噪声掩膜质心",
            "centroid_px": [_round(zero[0]), _round(zero[1])],
            **{key: flat_measurement[key] for key in (
                "has_signal", "centroid_is_fallback", "peak_counts", "peak_snr",
                "signal_pixels", "signal_weight_counts", "signal_threshold_counts",
            )},
        },
        "frames": info,
        "criteria": criteria,
        "overall": {
            "verdict": overall,
            "invalid_criteria": invalid,
            "insufficient_criteria": insufficient,
            "note": (
                "判据② 列向窗已按用户 2026-11-27 授权改为 92–97 px（原 56–59 px 取自 y 光栅标度）；"
                "④ 有越容差证据；③ 跨全部窗口一致指向 1024；"
                "⑤ 单斑与符号匹配但缺冻结容差，残差只做差异来源归因、不改裁定；⑥ 无饱和。"
            ),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "judgment.json"
    md_path = output_dir / "judgment.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    logger.info("judgment written: {} / {}", json_path, md_path)
    logger.info("overall={} invalid={} insufficient={}", overall, invalid, insufficient)
    return report


def _one_line_criterion(index: str, item: dict[str, Any]) -> str:
    verdict = str(item["verdict"])
    if index == "2":
        orders = item.get("evidence", {}).get("orders", [])
        detail = "; ".join(
            f"{o['order']}: |dc|={o['abs_dc_px']}px (窗 92–97), |dr|={abs(o['dr_residual_px'])}px (≤3)"
            for o in orders
        ) or "未成对检出 ±1 斑"
    elif index == "3":
        ev = item.get("evidence", {})
        branches = sorted({str(row["branch"]) for row in ev.get("energy_rows", [])})
        detail = (
            f"相位哈希相同={ev.get('hash_check', {}).get('phase_byte_identical')}, "
            f"灰度不同={ev.get('hash_check', {}).get('gray_different')}, "
            f"ratio 跨 8 组窗口分支={branches}, 结论={ev.get('resolved_two_pi_gray')}"
        )
    elif index == "4":
        ev = item.get("evidence", {})
        orders = ev.get("main_orders", [])
        detail = "; ".join(
            (
                f"order {o['order']}: step={o['step_px']}px→{o['target_px']}px, "
                f"残差 {o['residual_px']}px (>2)"
            )
            if "step_px" in o else f"order {o['order']}: {o.get('status')}"
            for o in orders
        ) or "未检出离轴阶斑"
        unassigned = ev.get("unassigned_spots", [])
        if unassigned:
            detail += "；" + "; ".join(
                f"step={s['step_px']}px/peak={s['peak']} 未归属/疑似伪斑（不参与主阶裁定）"
                for s in unassigned
            )
    elif index == "5":
        ev = item.get("evidence", {})
        diff = ev.get("difference_source_analysis", {})
        detail = (
            f"单斑={ev.get('single_off_axis_spot')}, 符号匹配={ev.get('sign_quadrant_match')}, "
            f"预测 dr/dc={ev.get('model', {}).get('predicted_dr_dc')}, "
            f"实测 dr/dc={ev.get('measured_dr_dc')}, 残差={ev.get('residual_to_prediction_px')}px，"
            f"冻结容差缺失；残差归因={diff.get('residual_vector_dr_dc')}px "
            f"(模型拟合 RMS {diff.get('model_fit_residual_rms_px')}px / "
            f"11-21→11-26 漂移 {diff.get('residual_to_a1_historical_px')}px)"
        )
    else:
        ev = item.get("evidence", {})
        detail = (
            f"帧最大={[p['observed_max_counts'] for p in ev.get('per_frame', [])]}, "
            f"preflight peak={ev.get('preflight', {}).get('peak_counts')}, "
            f"阈值=0.9×4095=3685.5 counts, abort={ev.get('abort_triggered')}"
        )
    return f"- **{index}：{verdict}** — {detail}"


def render_markdown(report: dict[str, Any]) -> str:
    """中文简报：逐条一行结论 + 证据表 + 下一步。"""
    criteria = report["criteria"]
    zero = report["zero_order"]["centroid_px"]
    lines = [
        "# §12.11.5 S1 判据②–⑥ 离线判读（2026-11-27）",
        "",
        f"- 总体裁定：**{report['overall']['verdict']}**；无效判据 {report['overall']['invalid_criteria']}，证据不足判据 {report['overall']['insufficient_criteria']}。",
        f"- 0 级（判据① 不重判，仅作②–⑥ 位移原点）：flat 帧 v2 噪声掩膜质心 ({zero[0]}, {zero[1]}) px，has_signal={report['zero_order']['has_signal']}。",
        f"- 数据：`{report['provenance']['raw_dir']}`（只读）；git HEAD `{report['git_head']}`。",
        "",
        "## 逐条裁定",
        "",
    ]
    for index in ("2", "3", "4", "5", "6"):
        lines.append(_one_line_criterion(index, criteria[index]))
    lines.extend(["", "## 证据表", "", "| 判据 | 裁定 | 关键量化证据 |", "| --- | --- | --- |"])
    for index in ("2", "3", "4", "5", "6"):
        lines.append(f"| {index} | {criteria[index]['verdict']} | {_one_line_criterion(index, criteria[index]).split(' — ', 1)[1].strip()} |")
    lines.extend([
        "",
        "## 斑点与口径",
        "",
        "| 帧 | 原始峰数 | 合并后斑数 | 0 级斑 (row, col) | 主要离轴斑 (dr, dc, peak) |",
        "| --- | --- | --- | --- | --- |",
    ])
    for key, item in report["frames"].items():
        zero_spot = item.get("zero_order_spot")
        ztxt = "—" if not zero_spot else f"({zero_spot['row']}, {zero_spot['col']})"
        off = item["off_axis_spots"][:4]
        otxt = "; ".join(f"({s['dr']}, {s['dc']}, {s['peak']})" for s in off) or "—"
        lines.append(f"| {key} | {item['raw_spot_count']} | {item['merged_spot_count']} | {ztxt} | {otxt} |")
    lines.extend([
        "",
        f"- 合并规则：{SPOT_MERGE_RULE}",
        f"- 能量口径：{ENERGY_RULE}",
        f"- 判据④ 阶次来源：{LADDER_ORDER_SOURCE}；冻结阶次 (1,3) 对应预期 95/285 px，弱斑列未归属/疑似伪斑且不参与主阶裁定。",
        "- 判据⑤ 预测模型：`blaze2d.A_matrix`（见 provenance），`(dr,dc)=A@(1/192,1/128)`；冻结文本未给数值容差，未按自设阈值判过或判死。",
        "",
    ])
    diff = criteria.get("5", {}).get("evidence", {}).get("difference_source_analysis")
    if diff:
        lines.extend([
            "## 判据⑤ 差异来源归因（不设容差、不改裁定）",
            "",
            f"- 实测 vs 预测：残差向量 (dr,dc)=`{diff['residual_vector_dr_dc']}` px，模长 **{diff['residual_to_prediction_px']} px**；方向与斑点方向 {diff['residual_direction_deg']}° / {diff['spot_direction_deg']}°。",
            f"- 模型自身拟合残差 RMS={diff['model_fit_residual_rms_px']} px、最大={diff['model_fit_residual_max_px']} px（11-21 标定）；当前残差超出该噪声水平。",
            f"- 0 级原点偏移（v2 质心 vs axis_status 参考）={diff['zero_order_origin_shift_px']} px `{diff['zero_order_origin_shift_dr_dc']}`，不足以单独解释差异。",
            f"- 同图样 11-21→11-26 会话间实测差={diff['residual_to_a1_historical_px']} px，提示存在跨会话离轴映射漂移。",
            "",
            "来源分解（定性，非容差判定）：",
            "",
        ])
        for source in diff.get("sources", []):
            lines.append(f"- {source}")
        lines.append("")
    orders_2 = criteria["2"].get("evidence", {}).get("orders", [])
    row_fail_2 = [
        f"{o['order']} 行残差 {abs(o['dr_residual_px'])}px（>{FROZEN_2_ROW_RESIDUAL_MAX_PX}px）"
        for o in orders_2 if not o["row_residual_ok"]
    ]
    col_fail_2 = [o["order"] for o in orders_2 if not o["col_in_window"]]
    row2_text = (
        "行残差超限：" + "、".join(row_fail_2) if row_fail_2
        else ("列向越窗：" + "、".join(col_fail_2) if col_fail_2 else "列向与行残差均在冻结范围内")
    )
    lines.extend([
        "## 下一步",
        "",
        f"- ② 改窗（{FROZEN_2_COL_WINDOW_PX[0]:g}–{FROZEN_2_COL_WINDOW_PX[1]:g} px）后：{row2_text}；"
        "④ 相对冻结阶次 (1,3) 的 95/285 px 阶距失配。二者说明本组六帧未按冻结预测复现，"
        "不能据此宣称面板编码或轴归属已被证实；③ 的跨窗口一致结果仅说明本次 scale1024 帧 ±1 能量更高。",
        "- ⑤ 需要用户补充冻结数值容差后才能定论；在此之前不得把定性匹配写成判据通过。",
        "- 所有结论均为离线证据判读，不涉及任何算法收益或实测改进声明。",
        "",
    ])
    return "\n".join(lines)


def _synthetic_spot(shape: tuple[int, int], row: float, col: float, peak: float, sigma: float = 2.0) -> np.ndarray:
    """纯合成单斑帧：零中位数 ±1 背景噪声 + 高斯斑，不碰文件与设备。"""
    rng = np.random.default_rng(20261127)
    frame = rng.choice(np.array([-1.0, 0.0, 1.0]), size=shape)
    rows, cols = np.indices(shape)
    frame = frame + peak * np.exp(-((rows - row) ** 2 + (cols - col) ** 2) / (2.0 * sigma ** 2))
    return frame


def self_test() -> dict[str, Any]:
    """纯合成数组自检：不读文件、不访问设备，输出一行 JSON。"""
    logger.remove()
    checks: list[dict[str, str]] = []

    flat = _synthetic_spot((400, 400), 200.0, 200.0, 1000.0)
    measured = flat_signal_measurement(flat)
    assert bool(measured["has_signal"]), "合成 flat 帧必须 has_signal=True"
    assert abs(float(measured["centroid_px"][0]) - 200.0) < 2.0
    assert abs(float(measured["centroid_px"][1]) - 200.0) < 2.0
    checks.append({"name": "flat_signal_measurement", "result": "passed"})

    duplicates = [
        {"row": 100.0, "col": 100.0, "peak": 10.0},
        {"row": 100.5, "col": 100.2, "peak": 9.0},
        {"row": 120.0, "col": 120.0, "peak": 8.0},
    ]
    merged = merge_nearby_spots(duplicates)
    assert len(merged) == 2 and merged[0]["peak"] == 10.0
    checks.append({"name": "merge_nearby_spots", "result": "passed"})

    zero = (200.0, 200.0)
    frame_2_ok = _synthetic_spot((400, 400), 200.0, 200.0, 1000.0)
    for col in (200.0 - 95.0, 200.0 + 95.0):
        frame_2_ok += _synthetic_spot((400, 400), 200.0, col, 400.0)
    info_2_ok = analyse_frame(frame_2_ok, zero, "syn_grating_x_ok")
    r2_ok = judge_2(info_2_ok, zero)
    assert r2_ok["verdict"] == VALID, r2_ok
    frame_2_bad = _synthetic_spot((400, 400), 200.0, 200.0, 1000.0)
    for col in (200.0 - 57.0, 200.0 + 57.0):
        frame_2_bad += _synthetic_spot((400, 400), 200.0, col, 400.0)
    r2_bad = judge_2(analyse_frame(frame_2_bad, zero, "syn_grating_x_bad"), zero)
    assert r2_bad["verdict"] == INVALID, r2_bad
    checks.append({"name": "criterion_2_pass_and_fail", "result": "passed"})

    def scale_frame(amplitude: float) -> np.ndarray:
        frame = _synthetic_spot((400, 400), 200.0, 200.0, 1000.0)
        for col in (143.0, 257.0):
            frame += _synthetic_spot((400, 400), 200.0, col, amplitude)
        return frame

    def fake_summary(amp_993: float, amp_1024: float, phase_same: bool = True) -> dict[str, Any]:
        return {"frames": [
            {
                "pattern_id": "grating_x_p128_scale993", "phase_sha256": "aa" if phase_same else "zz",
                "gray_sha256": "g993", "full_scale": 4095, "frame_max_counts": amp_993,
            },
            {
                "pattern_id": "grating_x_p128_scale1024", "phase_sha256": "aa",
                "gray_sha256": "g1024", "full_scale": 4095, "frame_max_counts": amp_1024,
            },
        ]}

    r3_high = judge_3(
        analyse_frame(scale_frame(400.0), zero, "syn993_high"),
        analyse_frame(scale_frame(450.0), zero, "syn1024_high"),
        {"grating_x_p128_scale993": scale_frame(400.0), "grating_x_p128_scale1024": scale_frame(450.0)},
        fake_summary(400.0, 450.0),
        zero,
    )
    assert r3_high["verdict"] == VALID and r3_high["evidence"]["resolved_two_pi_gray"] == "1024", r3_high
    r3_low = judge_3(
        analyse_frame(scale_frame(400.0), zero, "syn993_low"),
        analyse_frame(scale_frame(360.0), zero, "syn1024_low"),
        {"grating_x_p128_scale993": scale_frame(400.0), "grating_x_p128_scale1024": scale_frame(360.0)},
        fake_summary(400.0, 360.0),
        zero,
    )
    assert r3_low["verdict"] == VALID and r3_low["evidence"]["resolved_two_pi_gray"] == "993", r3_low
    r3_tie = judge_3(
        analyse_frame(scale_frame(400.0), zero, "syn993_tie"),
        analyse_frame(scale_frame(405.0), zero, "syn1024_tie"),
        {"grating_x_p128_scale993": scale_frame(400.0), "grating_x_p128_scale1024": scale_frame(405.0)},
        fake_summary(400.0, 405.0),
        zero,
    )
    assert r3_tie["verdict"] == INSUFFICIENT, r3_tie
    checks.append({"name": "criterion_3_three_branches", "result": "passed"})

    def ladder_frame(rows_offset: tuple[float, ...]) -> np.ndarray:
        frame = _synthetic_spot((600, 600), 400.0, 300.0, 1000.0)
        for offset in rows_offset:
            frame += _synthetic_spot((600, 600), 400.0 + offset, 300.0, 300.0)
        return frame

    zero_ladder = (400.0, 300.0)
    r4_ok = judge_4(analyse_frame(ladder_frame((-95.0, -285.0)), zero_ladder, "syn_ladder_ok"))
    assert r4_ok["verdict"] == VALID, r4_ok
    r4_bad = judge_4(analyse_frame(ladder_frame((-88.0, -232.0)), zero_ladder, "syn_ladder_bad"))
    assert r4_bad["verdict"] == INVALID, r4_bad
    assert [o["order"] for o in r4_bad["evidence"]["main_orders"]] == [1, 3]
    assert abs(float(r4_bad["evidence"]["main_orders"][0]["residual_px"]) - 7.0) < 2.0
    assert abs(float(r4_bad["evidence"]["main_orders"][1]["residual_px"]) - 53.0) < 2.0
    r4_extra_frame = ladder_frame((-95.0, -285.0))
    r4_extra_frame += _synthetic_spot((600, 600), 400.0 - 140.0, 300.0, 100.0)
    r4_extra = judge_4(analyse_frame(r4_extra_frame, zero_ladder, "syn_ladder_extra"))
    assert r4_extra["verdict"] == VALID, r4_extra
    assert len(r4_extra["evidence"]["unassigned_spots"]) == 1, r4_extra
    checks.append({"name": "criterion_4_frozen_orders_1_3_and_unassigned", "result": "passed"})

    a11, a12 = BLAZE_A_MATRIX[0]
    a21, a22 = BLAZE_A_MATRIX[1]
    pred = (a11 / BLAZE_PERIOD_Y_PX + a12 / BLAZE_PERIOD_X_PX,
            a21 / BLAZE_PERIOD_Y_PX + a22 / BLAZE_PERIOD_X_PX)
    blaze_single = _synthetic_spot((500, 500), 300.0, 250.0, 1000.0)
    blaze_single += _synthetic_spot((500, 500), 300.0 + pred[0], 250.0 + pred[1], 600.0)
    zero_blaze = (300.0, 250.0)
    r5 = judge_5(
        analyse_frame(blaze_single, zero_blaze, "syn_blaze_single"), zero_blaze, {}
    )
    assert r5["verdict"] == INSUFFICIENT and r5["evidence"]["off_axis_count"] == 1, r5
    blaze_multi = blaze_single + _synthetic_spot((500, 500), 300.0 - 100.0, 250.0, 400.0)
    r5_multi = judge_5(
        analyse_frame(blaze_multi, zero_blaze, "syn_blaze_multi"), zero_blaze, {}
    )
    assert r5_multi["verdict"] == INVALID, r5_multi
    checks.append({"name": "criterion_5_no_frozen_tolerance", "result": "passed"})

    syn_summary = {"frames": [
        {"pattern_id": key, "full_scale": 4095, "saturated": False, "frame_max_counts": 1000}
        for key in FRAME_FILES
    ]}
    safe_frame = np.zeros((10, 10), dtype=np.float64)
    hot_frame = np.full((10, 10), 4000.0, dtype=np.float64)
    syn_journal = [{"event": "preflight", "peak_counts": 1000.0, "saturation_threshold_counts": 3685.5, "saturated": False}]
    all_safe = {key: safe_frame for key in FRAME_FILES}
    assert judge_6(all_safe, syn_summary, syn_journal)["verdict"] == VALID
    one_hot = dict(all_safe)
    one_hot["flat_scale993"] = hot_frame
    assert judge_6(one_hot, syn_summary, syn_journal)["verdict"] == INVALID
    checks.append({"name": "criterion_6_abort_gate", "result": "passed"})

    return {
        "schema": "s1-offline-judge-self-test/v1",
        "status": "passed",
        "checks": checks,
        "n_checks": len(checks),
        "exit_code": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="§12.11.5 S1 判据②–⑥ 纯离线判读")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="六帧 raw 目录（只读）")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="报告输出目录（新建）")
    parser.add_argument("--self-test", action="store_true", help="纯合成数组自检，输出一行 JSON 后退出")
    args = parser.parse_args(argv)

    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False, separators=(",", ":")))
        return 0

    report = run_judgment(args.input_dir, args.output_dir)
    logger.info("done: overall={}", report["overall"]["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
