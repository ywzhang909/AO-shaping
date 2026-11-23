"""Issue #247 波形判别实验的离线分析器（2026-11-22）。

输入是一个采集目录：``plan.json``（必需）、``frames/*.npy``（每图样 4 帧，
命名沿用采集脚本的 ``{frame_index:04d}_{pattern_id}.npy``）、可选 ``summary.json``。
全部计算都是确定性的离线处理，**不打开相机/SLM、不读实验系统状态、不拟合映射**。

算法（与 ``docs/dev/issue247_waveform_diagnostic_plan_20261121.md`` 对齐）：

1. 每个图样对 4 帧取均值；``flat_pre/flat_mid/flat_post``（以及旧 ``kind=flat``）
   用连通域求 0 级质心。
2. 以 0 级峰为心、半径 ``max(9, 3*sigma_est)`` 的环带中位数作为局部背景，
   统一从全帧扣除；饱和判据是“满量程的 ``saturation_fraction`` 以上像素
   成片出现”（单个高斯峰顶不会触发）。
3. flat 质心按图样顺序线性插值，作为每个图样的参考 0 级；两两 flat 质心
   位移超过 ``drift_max_px`` 即判 ``invalid_drift``。
4. 峰定位：排除 0 级邻域（半径 ``max(15, 2*lobe_radius)``）后取局部极大。
   二值 grating **同时报告 ±1 两侧**；blaze/tilt/blaze_patch 报单峰，
   若出现同量级第二峰则标 ``multi_peak`` 并保留。
5. 位移 ``(dr, dc)`` 相对时间插值参考；角度按计划 §1 约定
   ``angle = degrees(atan2(dr, dc))``，从 +CCD 列转向 +CCD 行。
6. 门槛：波形依赖门、轴区分门、标度门、符号门（阈值可由 ``plan.json`` 的
   ``analysis_thresholds`` 覆盖），外加由 §2/H4 导出的 patch 位置门。
7. 结论只取计划 §4.2 的六个标签之一，逐门槛写出 pass/fail 与实测值。

产物（写入采集目录，默认拒绝覆盖）：``waveform_probe_analysis.json`` 与
``WAVEFORM_PROBE_REPORT.md``。**绝不修改** ``issue247_axis_map*.json`` 或
``slm_geometry_calib.npz``：即使全部门槛通过，本工具也不自动改映射。

``--self-test`` 只用内存里的合成帧（高斯 0 级 + 已知位移的 lobe），构造
“全部一致 / 二值 grating 转 90° / flat 漂移 5 px”三种场景核对 verdict，
不读真帧、不碰设备。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import label, maximum_filter

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_JSON_NAME = "waveform_probe_analysis.json"
REPORT_MD_NAME = "WAVEFORM_PROBE_REPORT.md"

#: 饱和判据：满量程阈值以上的像素达到这么多才算“成片饱和”。单个高斯峰顶
#: 过阈值通常只有 1-5 个像素；真正削顶的饱和区是一片平台。
SATURATION_PLATEAU_PIXELS = 9

#: 计划 §4.2 的默认阈值。采集后不得事后修改；要覆盖必须写进 ``plan.json``
#: 的 ``analysis_thresholds``，并会原样快照进分析 JSON。
DEFAULT_THRESHOLDS: dict[str, float] = {
    "saturation_fraction": 0.9,
    "drift_max_px": 2.0,
    "waveform_independent_max_deg": 2.0,
    "waveform_dependent_min_deg": 5.0,
    "axis_separation_expected_deg": 88.544,
    "axis_separation_tol_deg": 2.0,
    "tilt_axis_separation_min_deg": 80.0,
    "scale_ratio_expected": 2.0,
    "scale_ratio_tol": 0.10,
    "sign_angle_expected_deg": 180.0,
    "sign_angle_tol_deg": 2.0,
    "patch_vs_blaze_max_deg": 5.0,
}

#: 六个合法结论标签（计划 §4.2）。多一个少一个都算实现错误。
VERDICTS = (
    "waveform_independent_map_ok",
    "binary_grating_specific",
    "tilt_specific",
    "patch_position_specific",
    "map_not_reproduced",
    "invalid_drift_or_saturation",
)

VERDICT_ZH = {
    "waveform_independent_map_ok": "波形无关且映射复现",
    "binary_grating_specific": "仅二值 grating 族异常",
    "tilt_specific": "仅 tilt 族异常",
    "patch_position_specific": "局部 blaze patch 方向随位置异常",
    "map_not_reproduced": "映射未被复现",
    "invalid_drift_or_saturation": "漂移或饱和导致窗口无效",
}

FULL_SCALE_KEYS = (
    "full_scale",
    "full_scale_required",
    "full_scale_counts",
    "effective_full_scale",
)


# ---------------------------------------------------------------------------
# 输入结构与 plan 解析
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PatternSpec:
    """plan.json 里一个图样在分析侧的稳定视图。"""

    pattern_id: str
    family: str  # flat | blaze | grating | tilt | blaze_patch | blaze2d | ladder | unknown
    axis: str | None  # x | y
    period_px: float | None
    side: int | None  # +1 / -1（无符号要求的族为 None）
    order_index: int
    predicted_lobe_radius_px: float | None


@dataclass
class Capture:
    """一次采集的离线内容（自检时用内存帧构造同样的结构）。"""

    plan: dict[str, Any]
    catalog: list[PatternSpec]
    means: dict[str, np.ndarray]
    frame_stats: dict[str, dict[str, Any]]
    full_scale: float
    assumed_full_scale: bool
    full_scale_source: str
    summary: dict[str, Any]
    missing: list[str]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} 不是 JSON 对象")
    return payload


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def classify_family(pattern_id: str, raw_kind: object) -> str:
    """判定图样族。

    先看 id 前缀：本计划里 ``blaze_x_plus`` 的 ``kind`` 故意写成 ``tilt``
    （同一个构造函数，plan 用 ``identical_builder_to`` 明说），若只信 kind
    会把 blaze 族并进 tilt 族，波形依赖门就失去对照。
    """
    low = pattern_id.lower()
    if low == "flat" or low.startswith("flat_"):
        return "flat"
    if low.startswith("blaze2d"):
        return "blaze2d"
    if low.startswith("blaze_patch") or low.startswith("patch_"):
        return "blaze_patch"
    if low.startswith("blaze"):
        return "blaze"
    if low.startswith("grating"):
        return "grating"
    if low.startswith("tilt"):
        return "tilt"
    if low.startswith("ladder"):
        return "ladder"
    kind = str(raw_kind or "").strip().lower()
    aliases = {
        "reference": "flat",
        "ramp": "blaze",
        "patch": "blaze_patch",
        "grating2d": "grating",
    }
    return aliases.get(kind, kind or "unknown")


def _infer_axis(pattern_id: str, raw_axis: object) -> str | None:
    if isinstance(raw_axis, str) and raw_axis.lower() in ("x", "y"):
        return raw_axis.lower()
    if _is_number(raw_axis):
        if float(raw_axis) == 1.0:
            return "x"
        if float(raw_axis) == 0.0:
            return "y"
    match = re.search(r"_(x|y)(?:_|$)", pattern_id)
    return match.group(1) if match else None


def _infer_side(pattern_id: str, raw_side: object) -> int | None:
    if _is_number(raw_side):
        value = float(raw_side)
        if value > 0:
            return 1
        if value < 0:
            return -1
    if "_plus" in pattern_id:
        return 1
    if "_minus" in pattern_id:
        return -1
    return None


def _infer_period(pattern_id: str, raw_period: object) -> float | None:
    if _is_number(raw_period):
        value = float(raw_period)
        if value > 0:
            return value
    match = re.search(r"_p(\d+)(?:_|$)", pattern_id)
    return float(match.group(1)) if match else None


def build_specs(plan: dict[str, Any]) -> list[PatternSpec]:
    """把 ``plan["patterns"]`` 转成分析用的 :class:`PatternSpec` 列表。

    未知 id 不硬失败（分析器要能回放旧目录），但族/轴缺失会在门槛里显示
    ``skipped`` 而不是被悄悄当成通过。
    """
    patterns = plan.get("patterns")
    if not isinstance(patterns, list) or not patterns:
        raise ValueError("plan.json 缺少非空 patterns 列表")
    specs: list[PatternSpec] = []
    for index, entry in enumerate(patterns):
        if not isinstance(entry, dict):
            raise ValueError(f"plan.patterns[{index}] 不是对象")
        pattern_id = str(entry.get("id") or "").strip()
        if not pattern_id:
            raise ValueError(f"plan.patterns[{index}] 缺少 id")
        raw_radius = entry.get("predicted_lobe_radius_px")
        radius = float(raw_radius) if _is_number(raw_radius) and raw_radius > 0 else None
        specs.append(
            PatternSpec(
                pattern_id=pattern_id,
                family=classify_family(pattern_id, entry.get("kind")),
                axis=_infer_axis(pattern_id, entry.get("axis")),
                period_px=_infer_period(pattern_id, entry.get("period_px")),
                side=_infer_side(pattern_id, entry.get("sign") if "sign" in entry else entry.get("side")),
                order_index=index,
                predicted_lobe_radius_px=radius,
            )
        )
    return specs


def resolve_thresholds(plan: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    """默认阈值 + plan 覆盖；返回 (生效阈值, 被覆盖的键)。"""
    thresholds = dict(DEFAULT_THRESHOLDS)
    overrides: dict[str, float] = {}
    raw = plan.get("analysis_thresholds")
    if not isinstance(raw, dict):
        raw = plan.get("waveform_probe_thresholds")
    if isinstance(raw, dict):
        for key, value in raw.items():
            if key in thresholds and _is_number(value):
                thresholds[key] = float(value)
                overrides[key] = float(value)
    top_level_fraction = plan.get("saturation_fraction")
    if _is_number(top_level_fraction) and "saturation_fraction" not in overrides:
        thresholds["saturation_fraction"] = float(top_level_fraction)
        overrides["saturation_fraction"] = float(top_level_fraction)
    return thresholds, overrides


# ---------------------------------------------------------------------------
# 帧读取与满量程
# ---------------------------------------------------------------------------
def discover_frames(frames_dir: Path, pattern_id: str) -> list[Path]:
    """按采集脚本的命名找帧，兼容几种历史写法。"""
    for pattern in (f"*_{pattern_id}.npy", f"{pattern_id}_*.npy", f"{pattern_id}.npy"):
        hits = sorted(frames_dir.glob(pattern))
        if hits:
            return [path for path in hits if path.is_file()]
    subdir = frames_dir / pattern_id
    if subdir.is_dir():
        return sorted(path for path in subdir.glob("*.npy") if path.is_file())
    return []


def _nested_lookup(name: str, mapping: object) -> tuple[float, str] | None:
    if not isinstance(mapping, dict):
        return None
    for key in FULL_SCALE_KEYS:
        value = mapping.get(key)
        if _is_number(value) and float(value) > 0:
            return float(value), f"{name}:{key}"
    return None


def resolve_full_scale(
    plan: dict[str, Any], summary: dict[str, Any]
) -> tuple[float | None, str]:
    """优先用 summary/plan 里声明的满量程；都没有时返回 ``(None, "")``。"""
    for name, mapping in (
        ("summary", summary),
        ("summary.declared_constants", summary.get("declared_constants")),
        ("plan", plan),
        ("plan.declared_constants", plan.get("declared_constants")),
    ):
        found = _nested_lookup(name, mapping)
        if found is not None:
            return found
    # 退路：plan 同时给了饱和阈值与比例时反推，避免多写一份满量程字段。
    fraction = plan.get("saturation_fraction")
    threshold = plan.get("saturation_threshold_counts")
    if _is_number(fraction) and _is_number(threshold) and float(fraction) > 0:
        return float(threshold) / float(fraction), "plan:saturation_threshold_counts/fraction"
    return None, ""


def _scan_frame_max(paths: list[Path]) -> float:
    best = 0.0
    for path in paths:
        array = np.load(path, mmap_mode="r")
        if array.size:
            best = max(best, float(np.asarray(array).max()))
    return best


def load_capture(
    capture_dir: Path,
    plan: dict[str, Any],
    summary: dict[str, Any],
    thresholds: dict[str, float],
) -> Capture:
    """读取采集目录：每图样取 4 帧均值，同时记录每帧的饱和统计。"""
    frames_dir = capture_dir / "frames"
    if not frames_dir.is_dir():
        raise ValueError(f"{frames_dir} 不存在")
    specs = build_specs(plan)
    located = {spec.pattern_id: discover_frames(frames_dir, spec.pattern_id) for spec in specs}

    full_scale, source = resolve_full_scale(plan, summary)
    assumed = full_scale is None
    if full_scale is None:
        all_paths = [path for paths in located.values() for path in paths]
        full_scale = _scan_frame_max(all_paths) or 1.0
        source = "inferred_from_frame_max"

    fraction = float(thresholds["saturation_fraction"])
    threshold_counts = fraction * float(full_scale)
    means: dict[str, np.ndarray] = {}
    frame_stats: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for spec in specs:
        files = located[spec.pattern_id]
        if not files:
            missing.append(spec.pattern_id)
            continue
        accumulator: np.ndarray | None = None
        maxima: list[float] = []
        counts: list[int] = []
        for path in files:
            raw = np.load(path).astype(np.float64)
            accumulator = raw if accumulator is None else accumulator + raw
            maxima.append(float(raw.max()))
            counts.append(int(np.count_nonzero(raw >= threshold_counts)))
        assert accumulator is not None
        means[spec.pattern_id] = accumulator / float(len(files))
        saturated_frames = [i for i, count in enumerate(counts) if count >= SATURATION_PLATEAU_PIXELS]
        # 满量程是推断出来的时，阈值只能取到帧最大值附近，亮斑自身就会“成片”过门限；
        # 此时饱和判据不可用，不触发饱和，只在 JSON/备注里标明。
        reliable = not assumed
        frame_stats[spec.pattern_id] = {
            "n_frames": len(files),
            "frames": [path.name for path in files],
            "frame_max_counts": maxima,
            "max_counts": max(maxima),
            "pixels_at_or_above_threshold": max(counts),
            "saturated_frame_indices": saturated_frames if reliable else [],
            "saturated": bool(saturated_frames) and reliable,
            "saturation_reliable": reliable,
        }

    if not means:
        raise ValueError(f"{frames_dir} 下没有可读的帧")
    return Capture(
        plan=plan,
        catalog=specs,
        means=means,
        frame_stats=frame_stats,
        full_scale=float(full_scale),
        assumed_full_scale=assumed,
        full_scale_source=source,
        summary=summary,
        missing=missing,
    )


# ---------------------------------------------------------------------------
# 图像处理：0 级、背景、峰
# ---------------------------------------------------------------------------
def _argmax_center(frame: np.ndarray) -> tuple[float, float]:
    row, col = np.unravel_index(int(np.argmax(frame)), frame.shape)
    return float(row), float(col)


def _core_sigma(frame: np.ndarray, center: tuple[float, float], win: float = 40.0) -> float:
    """0 级核心的强度加权二阶矩半径（px），用于确定背景环内径。"""
    height, width = frame.shape
    r0 = max(0, int(np.floor(center[0] - win)))
    r1 = min(height, int(np.ceil(center[0] + win)) + 1)
    c0 = max(0, int(np.floor(center[1] - win)))
    c1 = min(width, int(np.ceil(center[1] + win)) + 1)
    patch = np.maximum(frame[r0:r1, c0:c1], 0.0)
    total = float(patch.sum())
    if total <= 0:
        return 0.0
    yy = np.arange(r0, r1)[:, None] - center[0]
    xx = np.arange(c0, c1)[None, :] - center[1]
    return float(np.sqrt(float((patch * (yy * yy + xx * xx)).sum()) / total))


def _ring_inner_radius(shape: tuple[int, int], sigma: float) -> float:
    limit = 0.30 * float(min(shape))
    return float(min(max(9.0, 3.0 * sigma), max(9.0, limit)))


def _ring_background(
    frame: np.ndarray, center: tuple[float, float], inner: float
) -> tuple[float, float, float]:
    """环带中位数作局部背景，返回 (背景, 内径, 外径)。"""
    height, width = frame.shape
    outer = min(inner + max(20.0, inner), 0.45 * float(min(height, width)))
    if outer <= inner:
        outer = inner + 1.0
    r0 = max(0, int(np.floor(center[0] - outer)))
    r1 = min(height, int(np.ceil(center[0] + outer)) + 1)
    c0 = max(0, int(np.floor(center[1] - outer)))
    c1 = min(width, int(np.ceil(center[1] + outer)) + 1)
    patch = frame[r0:r1, c0:c1]
    yy = np.arange(r0, r1)[:, None] - center[0]
    xx = np.arange(c0, c1)[None, :] - center[1]
    radius = np.hypot(yy, xx)
    values = patch[(radius >= inner) & (radius <= outer)]
    background = float(np.median(values)) if values.size else 0.0
    return background, float(inner), float(outer)


def _flat_centroid(work: np.ndarray, hint: tuple[float, float]) -> tuple[float, float]:
    """用连通域确定 0 级光斑并求强度质心；连通域断裂时退回 argmax。"""
    peak = float(work.max())
    if peak <= 0:
        return hint
    height, width = work.shape
    labels, count = label(work >= 0.25 * peak)
    if count == 0:
        return _argmax_center(work)
    row = int(min(max(round(hint[0]), 0), height - 1))
    col = int(min(max(round(hint[1]), 0), width - 1))
    selected = int(labels[row, col])
    if selected == 0:
        flat_index = int(np.argmax(work))
        selected = int(labels[flat_index // width, flat_index % width])
    if selected == 0:
        return _argmax_center(work)
    rows, cols = np.nonzero(labels == selected)
    weights = work[rows, cols]
    total = float(weights.sum())
    if total <= 0:
        return _argmax_center(work)
    return (
        float((rows * weights).sum() / total),
        float((cols * weights).sum() / total),
    )


def _subpixel_peak(work: np.ndarray, row: int, col: int, radius: int = 6) -> tuple[float, float]:
    height, width = work.shape
    r0 = max(0, row - radius)
    r1 = min(height, row + radius + 1)
    c0 = max(0, col - radius)
    c1 = min(width, col + radius + 1)
    patch = work[r0:r1, c0:c1]
    weights = np.where(patch > 0, patch, 0.0)
    total = float(weights.sum())
    if total <= 0:
        return float(row), float(col)
    yy, xx = np.indices(patch.shape)
    return (
        r0 + float((yy * weights).sum() / total),
        c0 + float((xx * weights).sum() / total),
    )


def _find_peaks(
    work: np.ndarray, *, min_sep: int, thresh_frac: float = 0.03
) -> list[dict[str, float]]:
    """局部极大（含亚像素质心），按峰值强度降序。"""
    peak_value = float(work.max())
    if peak_value <= 0:
        return []
    size = max(3, int(min_sep))
    local_max = maximum_filter(work, size=size, mode="nearest")
    mask = (work == local_max) & (work > thresh_frac * peak_value)
    raw: list[dict[str, float]] = []
    for row, col in np.argwhere(mask):
        if row < 100 and col < 100:
            # 本实验系统左上有固定杂散光；与既有离线脚本保持同一屏蔽口径。
            continue
        center_row, center_col = _subpixel_peak(work, int(row), int(col))
        raw.append(
            {"row": center_row, "col": center_col, "peak": float(work[row, col])}
        )
    raw.sort(key=lambda item: -item["peak"])
    kept: list[dict[str, float]] = []
    for candidate in raw:
        if all(
            np.hypot(candidate["row"] - other["row"], candidate["col"] - other["col"])
            >= size / 2.0
            for other in kept
        ):
            kept.append(candidate)
    return kept


# ---------------------------------------------------------------------------
# 角度工具（plan §1：从 +CCD 列转向 +CCD 行）
# ---------------------------------------------------------------------------
def angle_deg(dr: float, dc: float) -> float:
    """CCD 方向角（度）：0°≈+列，90°≈+行。"""
    return float(np.degrees(np.arctan2(float(dr), float(dc))))


def axial_mean_deg(angles: list[float]) -> float | None:
    """轴线平均（模 180°）：把 ± 两侧折成一条轴，返回 (-90, 90]。"""
    values = np.asarray([a for a in angles if a is not None], dtype=np.float64)
    if values.size == 0:
        return None
    doubled = np.radians(2.0 * values)
    sine = float(np.mean(np.sin(doubled)))
    cosine = float(np.mean(np.cos(doubled)))
    if abs(sine) < 1e-12 and abs(cosine) < 1e-12:
        return None
    return float(np.degrees(0.5 * np.arctan2(sine, cosine)))


def axial_diff_deg(first: float | None, second: float | None) -> float | None:
    """两条轴线的无向夹角，范围 [0, 90]。"""
    if first is None or second is None:
        return None
    return float(abs(((float(first) - float(second) + 90.0) % 180.0) - 90.0))


def angle_between_deg(
    first: tuple[float, float], second: tuple[float, float]
) -> float:
    """两个有向位移的夹角，范围 [0, 180]。"""
    dot = first[0] * second[0] + first[1] * second[1]
    cross = first[0] * second[1] - first[1] * second[0]
    return float(abs(np.degrees(np.arctan2(cross, dot))))


# ---------------------------------------------------------------------------
# 单图样测量
# ---------------------------------------------------------------------------
def _measure_pattern(
    spec: PatternSpec,
    frame: np.ndarray,
    reference: tuple[float, float],
    flat_sigma: float,
    stats: dict[str, Any],
) -> dict[str, Any]:
    """测量一个图样的 0 级背景、lobe 峰与相对位移。"""
    sigma = float(flat_sigma) if np.isfinite(flat_sigma) else 0.0
    inner = _ring_inner_radius(frame.shape, sigma)
    background, inner, outer = _ring_background(frame, reference, inner)
    work = np.maximum(frame - background, 0.0)

    lobe_radius = spec.predicted_lobe_radius_px
    if lobe_radius is None:
        # 0 级核心与 probe lobe 共用同一套光学口径，用 flat 核心 sigma 作代理；
        # 上限 24 px 防止大离焦的 0 级光晕把 ±1 级也一并排除。
        lobe_radius = float(min(max(sigma, 4.0), 24.0))
    core_exclude = max(15.0, 2.0 * lobe_radius)

    record: dict[str, Any] = {
        "pattern_id": spec.pattern_id,
        "family": spec.family,
        "axis": spec.axis,
        "period_px": spec.period_px,
        "side": spec.side,
        "order_index": spec.order_index,
        "zero_order": {"row": float(reference[0]), "col": float(reference[1])},
        "background_counts": background,
        "ring_radius_px": inner,
        "ring_outer_px": outer,
        "lobe_radius_px": float(lobe_radius),
        "core_exclude_px": float(core_exclude),
        "n_frames": stats["n_frames"],
        "saturated": stats["saturated"],
        "max_counts": stats["max_counts"],
        "peaks": [],
        "lobes": [],
        "principal": None,
        "axis_angle_deg": None,
        "multi_peak": False,
        "counter_lobe_missing": False,
        "counter_lobe_weak": False,
    }

    if spec.family == "flat":
        # 平场本身就是参考，不含探针 lobe。
        return record

    min_sep = max(5, int(round(lobe_radius)))
    found = _find_peaks(work, min_sep=min_sep)
    peaks: list[dict[str, float]] = []
    for spot in found:
        dr = float(spot["row"] - reference[0])
        dc = float(spot["col"] - reference[1])
        distance = float(np.hypot(dr, dc))
        if distance <= core_exclude:
            continue
        peaks.append(
            {
                "dr": dr,
                "dc": dc,
                "dist": distance,
                "peak": float(spot["peak"]),
                "angle_deg": angle_deg(dr, dc),
            }
        )
    record["peak_count"] = len(peaks)
    record["peaks"] = peaks[:8]

    if not peaks:
        record["counter_lobe_missing"] = spec.family == "grating"
        return record

    best = peaks[0]
    if spec.family == "grating":
        counter = None
        for candidate in peaks[1:]:
            separation = angle_between_deg(
                (best["dr"], best["dc"]), (candidate["dr"], candidate["dc"])
            )
            if 150.0 <= separation <= 210.0:
                counter = candidate
                break
        if counter is None:
            record["counter_lobe_missing"] = True
            if len(peaks) > 1:
                counter = peaks[1]
        if counter is not None:
            record["counter_lobe_weak"] = counter["peak"] < 0.2 * best["peak"]
            record["lobes"] = [best, counter]
            magnitudes = [lobe["dist"] for lobe in (best, counter)]
            angles = [lobe["angle_deg"] for lobe in (best, counter)]
        else:
            record["lobes"] = [best]
            magnitudes = [best["dist"]]
            angles = [best["angle_deg"]]
        record["axis_angle_deg"] = axial_mean_deg(angles)
        record["principal"] = {
            "dr": best["dr"],
            "dc": best["dc"],
            "angle_deg": best["angle_deg"],
            "magnitude": float(np.mean(magnitudes)),
            "peak": best["peak"],
        }
    else:
        if len(peaks) > 1 and peaks[1]["peak"] >= 0.5 * best["peak"]:
            record["multi_peak"] = True
        record["lobes"] = [best]
        record["axis_angle_deg"] = best["angle_deg"]
        record["principal"] = {
            "dr": best["dr"],
            "dc": best["dc"],
            "angle_deg": best["angle_deg"],
            "magnitude": best["dist"],
            "peak": best["peak"],
        }
    return record


# ---------------------------------------------------------------------------
# 门槛
# ---------------------------------------------------------------------------
def _spec_ids(
    catalog: list[PatternSpec],
    family: str,
    axis: str | None = None,
    side: int | None = None,
) -> list[str]:
    return [
        spec.pattern_id
        for spec in catalog
        if spec.family == family
        and (axis is None or spec.axis == axis)
        and (side is None or spec.side == side)
    ]


def _grating_id(catalog: list[PatternSpec], axis: str, period: float) -> str | None:
    candidates = [
        spec
        for spec in catalog
        if spec.family == "grating" and spec.axis == axis and spec.period_px
    ]
    if not candidates:
        return None
    chosen = min(candidates, key=lambda spec: abs(float(spec.period_px) - period))
    return chosen.pattern_id


def _pattern_angles(records: dict[str, dict[str, Any]], ids: list[str]) -> list[float]:
    """收集一组图样的有向角度；grating 用两侧 lobe，其余用 principal。"""
    angles: list[float] = []
    for pattern_id in ids:
        record = records.get(pattern_id)
        if not record:
            continue
        if record["family"] == "grating" and record["lobes"]:
            angles.extend(float(lobe["angle_deg"]) for lobe in record["lobes"])
        elif record.get("principal"):
            angles.append(float(record["principal"]["angle_deg"]))
    return angles


def _vector(records: dict[str, dict[str, Any]], pattern_id: str) -> tuple[float, float] | None:
    record = records.get(pattern_id)
    if not record or not record.get("principal"):
        return None
    principal = record["principal"]
    return float(principal["dr"]), float(principal["dc"])


def _combine(values: list[bool | None], expected: int | None = None) -> bool | None:
    """合并子项判定。

    ``expected`` 是该门槛本应覆盖的子项数；只要有子项因为缺帧/缺族没被
    评估，整体就记 ``skipped``（``None``），**不能**拿其余子项的通过值
    凑出一个干净的 pass —— 缺口必须在结论里可见。
    """
    evaluated = [value for value in values if value is not None]
    if not evaluated:
        return None
    if expected is not None and len(evaluated) < expected:
        return None
    return all(evaluated)


def _status_text(value: bool | None) -> str:
    if value is True:
        return "pass"
    if value is False:
        return "fail"
    return "skipped"


def _build_gates(
    catalog: list[PatternSpec],
    records: dict[str, dict[str, Any]],
    drift: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """按计划 §4.2 构造全部门槛；每个门槛都带实测值与 pass/fail。"""
    independent_max = thresholds["waveform_independent_max_deg"]
    dependent_min = thresholds["waveform_dependent_min_deg"]

    # ---- 波形依赖门：同轴、同频率的二值 grating vs 连续 blaze ----
    waveform_axes: dict[str, Any] = {}
    for axis in ("x", "y"):
        blaze_ids = _spec_ids(catalog, "blaze", axis)
        blaze_period = 128.0
        for pattern_id in blaze_ids:
            record = records.get(pattern_id)
            if record and record.get("period_px"):
                blaze_period = float(record["period_px"])
                break
        grating_id = _grating_id(catalog, axis, blaze_period)
        grating_ids = [grating_id] if grating_id else []
        blaze_axis = axial_mean_deg(_pattern_angles(records, blaze_ids))
        grating_axis = axial_mean_deg(_pattern_angles(records, grating_ids))
        difference = axial_diff_deg(grating_axis, blaze_axis)
        if difference is None:
            status = "skipped"
            passed: bool | None = None
        elif difference <= independent_max:
            status = "independent"
            passed = True
        elif difference >= dependent_min:
            status = "dependent"
            passed = False
        else:
            status = "ambiguous"
            passed = False
        waveform_axes[axis] = {
            "status": status,
            "pass": passed,
            "measured_diff_deg": difference,
            "grating_axis_deg": grating_axis,
            "blaze_axis_deg": blaze_axis,
            "grating_ids": grating_ids,
            "blaze_ids": blaze_ids,
        }
    waveform_gate = {
        "pass": _combine(
            [waveform_axes[axis]["pass"] for axis in ("x", "y")], expected=2
        ),
        "axes": waveform_axes,
        "independent_max_deg": independent_max,
        "dependent_min_deg": dependent_min,
    }

    # ---- 轴区分门 ----
    blaze_x = axial_mean_deg(_pattern_angles(records, _spec_ids(catalog, "blaze", "x")))
    blaze_y = axial_mean_deg(_pattern_angles(records, _spec_ids(catalog, "blaze", "y")))
    blaze_diff = axial_diff_deg(blaze_x, blaze_y)
    expected = thresholds["axis_separation_expected_deg"]
    tolerance = thresholds["axis_separation_tol_deg"]
    blaze_pass = (
        None if blaze_diff is None else abs(blaze_diff - expected) <= tolerance
    )

    tilt_x = axial_mean_deg(_pattern_angles(records, _spec_ids(catalog, "tilt", "x")))
    tilt_y = axial_mean_deg(_pattern_angles(records, _spec_ids(catalog, "tilt", "y")))
    tilt_diff = axial_diff_deg(tilt_x, tilt_y)
    tilt_min = thresholds["tilt_axis_separation_min_deg"]
    tilt_pass = None if tilt_diff is None else tilt_diff > tilt_min

    axis_expected = (1 if _spec_ids(catalog, "blaze") else 0) + (
        1 if _spec_ids(catalog, "tilt") else 0
    )
    axis_gate = {
        "pass": _combine([blaze_pass, tilt_pass], expected=axis_expected),
        "blaze_pass": blaze_pass,
        "tilt_pass": tilt_pass,
        "blaze": {
            "measured_diff_deg": blaze_diff,
            "axis_x_deg": blaze_x,
            "axis_y_deg": blaze_y,
            "expected_deg": expected,
            "tol_deg": tolerance,
            "pass": blaze_pass,
        },
        "tilt": {
            "measured_diff_deg": tilt_diff,
            "axis_x_deg": tilt_x,
            "axis_y_deg": tilt_y,
            "min_deg": tilt_min,
            "pass": tilt_pass,
        },
    }

    # ---- 标度门：grating p128 / p256 ----
    ratio_expected = thresholds["scale_ratio_expected"]
    ratio_tolerance = thresholds["scale_ratio_tol"]
    scale_axes: dict[str, Any] = {}
    for axis in ("x", "y"):
        id_128 = _grating_id(catalog, axis, 128.0)
        id_256 = _grating_id(catalog, axis, 256.0)
        mag_128 = None
        mag_256 = None
        if id_128 and records.get(id_128, {}).get("principal"):
            mag_128 = float(records[id_128]["principal"]["magnitude"])
        if id_256 and records.get(id_256, {}).get("principal"):
            mag_256 = float(records[id_256]["principal"]["magnitude"])
        ratio = (mag_128 / mag_256) if mag_128 is not None and mag_256 else None
        passed = (
            None
            if ratio is None
            else abs(ratio - ratio_expected) <= ratio_tolerance
        )
        scale_axes[axis] = {
            "pass": passed,
            "measured_ratio": ratio,
            "magnitude_p128_px": mag_128,
            "magnitude_p256_px": mag_256,
            "p128_id": id_128,
            "p256_id": id_256,
            "expected": ratio_expected,
            "tol": ratio_tolerance,
        }
    scale_expected = sum(
        1
        for axis in ("x", "y")
        if _grating_id(catalog, axis, 128.0) and _grating_id(catalog, axis, 256.0)
    )
    scale_gate = {
        "pass": _combine(
            [scale_axes[axis]["pass"] for axis in ("x", "y")], expected=scale_expected
        ),
        "axes": scale_axes,
    }

    # ---- 符号门：每轴 + 与 - 的夹角应 180° ----
    sign_expected = thresholds["sign_angle_expected_deg"]
    sign_tolerance = thresholds["sign_angle_tol_deg"]
    sign_pairs: list[dict[str, Any]] = []
    for family in ("blaze", "tilt", "blaze_patch"):
        for axis in ("x", "y"):
            plus_ids = _spec_ids(catalog, family, axis, 1)
            minus_ids = _spec_ids(catalog, family, axis, -1)
            if not plus_ids or not minus_ids:
                continue
            plus_vector = _vector(records, plus_ids[0])
            minus_vector = _vector(records, minus_ids[0])
            if plus_vector is None or minus_vector is None:
                continue
            measured = angle_between_deg(plus_vector, minus_vector)
            passed = abs(measured - sign_expected) <= sign_tolerance
            sign_pairs.append(
                {
                    "family": family,
                    "axis": axis,
                    "plus_id": plus_ids[0],
                    "minus_id": minus_ids[0],
                    "measured_angle_deg": measured,
                    "expected_deg": sign_expected,
                    "tol_deg": sign_tolerance,
                    "pass": passed,
                }
            )
    sign_expected_pairs = sum(
        1
        for family in ("blaze", "tilt", "blaze_patch")
        for axis in ("x", "y")
        if _spec_ids(catalog, family, axis, 1) and _spec_ids(catalog, family, axis, -1)
    )
    sign_gate = {
        "pass": _combine(
            [pair["pass"] for pair in sign_pairs], expected=sign_expected_pairs
        ),
        "pairs": sign_pairs,
    }

    # ---- patch 位置门（§2 H4 / §4.2 的 patch_position_specific 证据）----
    patch_max = thresholds["patch_vs_blaze_max_deg"]
    patch_pairs: list[dict[str, Any]] = []
    for axis in ("x", "y"):
        patch_ids = _spec_ids(catalog, "blaze_patch", axis)
        blaze_ids = _spec_ids(catalog, "blaze", axis)
        if not patch_ids or not blaze_ids:
            continue
        patch_axis = axial_mean_deg(_pattern_angles(records, patch_ids))
        blaze_axis = axial_mean_deg(_pattern_angles(records, blaze_ids))
        difference = axial_diff_deg(patch_axis, blaze_axis)
        if difference is None:
            continue
        patch_pairs.append(
            {
                "axis": axis,
                "patch_ids": patch_ids,
                "blaze_ids": blaze_ids,
                "measured_diff_deg": difference,
                "max_deg": patch_max,
                "pass": difference <= patch_max,
            }
        )
    patch_expected = sum(
        1
        for axis in ("x", "y")
        if _spec_ids(catalog, "blaze_patch", axis) and _spec_ids(catalog, "blaze", axis)
    )
    patch_gate = {
        "pass": _combine(
            [pair["pass"] for pair in patch_pairs], expected=patch_expected
        ),
        "pairs": patch_pairs,
    }

    drift_gate = {
        "pass": drift["pass"],
        "measured_max_px": drift["max_drift_px"],
        "threshold_px": thresholds["drift_max_px"],
        "pairwise_px": drift["pairwise_px"],
    }
    saturated = sorted(
        pattern_id
        for pattern_id, record in records.items()
        if record.get("saturated") is True
    )
    saturation_gate = {
        "pass": not saturated,
        "saturated_patterns": saturated,
        "plateau_pixels": SATURATION_PLATEAU_PIXELS,
    }
    return {
        "drift_gate": drift_gate,
        "saturation_gate": saturation_gate,
        "waveform_gate": waveform_gate,
        "axis_gate": axis_gate,
        "scale_gate": scale_gate,
        "sign_gate": sign_gate,
        "patch_gate": patch_gate,
    }


def _decide_verdict(gates: dict[str, Any]) -> str:
    """按 §4.2 的六个标签落结论；顺序即优先级，失败细节仍在每个 gate 里。"""
    if gates["drift_gate"]["pass"] is False or gates["saturation_gate"]["pass"] is False:
        return "invalid_drift_or_saturation"
    if gates["patch_gate"]["pass"] is False:
        return "patch_position_specific"

    blaze_ok = gates["axis_gate"]["blaze_pass"] is True
    waveform_dependent = any(
        gates["waveform_gate"]["axes"][axis]["status"] == "dependent"
        for axis in ("x", "y")
    )
    if waveform_dependent and blaze_ok:
        return "binary_grating_specific"
    if gates["axis_gate"]["tilt_pass"] is False:
        return "tilt_specific"

    core_ok = (
        gates["waveform_gate"]["pass"] is True
        and blaze_ok
        and gates["scale_gate"]["pass"] is True
        and gates["sign_gate"]["pass"] is True
    )
    if core_ok:
        return "waveform_independent_map_ok"
    return "map_not_reproduced"


# ---------------------------------------------------------------------------
# 主分析
# ---------------------------------------------------------------------------
def _reference_at(
    order_index: int, flat_points: list[tuple[int, tuple[float, float], str]]
) -> tuple[float, float]:
    """flat 质心按图样顺序线性插值；窗口两端外退化为最近参考。"""
    ordered = sorted(flat_points, key=lambda item: item[0])
    before = [item for item in ordered if item[0] <= order_index]
    after = [item for item in ordered if item[0] >= order_index]
    if before and after and before[-1][0] != after[0][0]:
        index_before, point_before, _ = before[-1]
        index_after, point_after, _ = after[0]
        span = float(index_after - index_before)
        fraction = (order_index - index_before) / span
        return (
            point_before[0] + fraction * (point_after[0] - point_before[0]),
            point_before[1] + fraction * (point_after[1] - point_before[1]),
        )
    nearest = before[-1] if before else after[0]
    return nearest[1]


def analyze(capture: Capture, thresholds: dict[str, float]) -> dict[str, Any]:
    """跑完整分析，返回可 JSON 序列化的结果（不写盘）。"""
    specs = capture.catalog
    means = capture.means
    notes: list[str] = []

    flat_specs = [spec for spec in specs if spec.family == "flat" and spec.pattern_id in means]
    if not flat_specs:
        raise ValueError("采集目录没有可用的 flat 参考帧（flat_pre/flat_mid/flat_post）")

    flat_points: list[tuple[int, tuple[float, float], str]] = []
    sigmas: list[float] = []
    for spec in flat_specs:
        frame = means[spec.pattern_id]
        coarse = _argmax_center(frame)
        sigma = _core_sigma(frame, coarse)
        inner = _ring_inner_radius(frame.shape, sigma)
        background, _, _ = _ring_background(frame, coarse, inner)
        centroid = _flat_centroid(np.maximum(frame - background, 0.0), coarse)
        flat_points.append((spec.order_index, centroid, spec.pattern_id))
        sigmas.append(sigma)
    flat_sigma = float(np.median(sigmas)) if sigmas else 0.0

    pairwise: dict[str, float] = {}
    for first in range(len(flat_points)):
        for second in range(first + 1, len(flat_points)):
            id_first = flat_points[first][2]
            id_second = flat_points[second][2]
            point_first = flat_points[first][1]
            point_second = flat_points[second][1]
            pairwise[f"{id_first}->{id_second}"] = float(
                np.hypot(
                    point_second[0] - point_first[0],
                    point_second[1] - point_first[1],
                )
            )
    max_drift = max(pairwise.values()) if pairwise else 0.0
    drift_pass: bool | None
    if len(flat_points) < 2:
        drift_pass = None
        notes.append("可用 flat 参考少于 2 个，漂移门未评估。")
    else:
        drift_pass = max_drift <= float(thresholds["drift_max_px"])
    drift = {
        "pass": drift_pass,
        "max_drift_px": max_drift,
        "pairwise_px": pairwise,
    }

    records: dict[str, dict[str, Any]] = {}
    for spec in specs:
        if spec.pattern_id not in means:
            continue
        if spec.family == "flat":
            own = next(
                (point for index, point, pid in flat_points if pid == spec.pattern_id),
                _argmax_center(means[spec.pattern_id]),
            )
            reference = own
        else:
            reference = _reference_at(spec.order_index, flat_points)
        records[spec.pattern_id] = _measure_pattern(
            spec,
            means[spec.pattern_id],
            reference,
            flat_sigma,
            capture.frame_stats[spec.pattern_id],
        )
        if records[spec.pattern_id]["saturated"]:
            notes.append(f"{spec.pattern_id} 检测到成片饱和（见 saturation_gate）。")

    if capture.assumed_full_scale:
        notes.append(
            "满量程缺失，按帧最大值推断（assumed_full_scale=true）；"
            "饱和判据因此只作参考。"
        )
    for pattern_id in capture.missing:
        notes.append(f"缺少图样帧：{pattern_id}（相关门槛记为 skipped）。")
    for pattern_id, record in records.items():
        if record["counter_lobe_missing"]:
            notes.append(f"{pattern_id} 未找到与主峰近似反向的第二个 ±1 峰。")
        if record["counter_lobe_weak"]:
            notes.append(f"{pattern_id} 的第二侧峰弱于主峰的 20%，可能未完整成像。")
        if record["multi_peak"]:
            notes.append(f"{pattern_id} 出现同量级第二峰（multi_peak）。")

    gates = _build_gates(capture.catalog, records, drift, thresholds)
    verdict = _decide_verdict(gates)
    if verdict not in VERDICTS:
        raise ValueError(f"内部错误：非法 verdict {verdict!r}")
    failed = [
        name
        for name, gate in gates.items()
        if isinstance(gate, dict) and gate.get("pass") is False
    ]

    return {
        "schema": "issue247_waveform_probe_analysis/1",
        "script": "scripts/slm/issue247_waveform_probe_solve_20261122.py",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "capture_dir": _as_repo_str(capture.plan.get("_capture_dir")) if capture.plan.get("_capture_dir") else None,
        "full_scale": {
            "value": capture.full_scale,
            "assumed_full_scale": capture.assumed_full_scale,
            "source": capture.full_scale_source,
            "saturation_threshold_counts": float(thresholds["saturation_fraction"])
            * capture.full_scale,
        },
        "thresholds": dict(thresholds),
        "flat_reference": {
            "sigma_est_px": flat_sigma,
            "centroids": {
                pattern_id: {"row": point[0], "col": point[1]}
                for _, point, pattern_id in flat_points
            },
            "pairwise_drift_px": pairwise,
            "max_drift_px": max_drift,
            "invalid_drift": drift_pass is False,
        },
        "lobe_radius_px": float(min(max(flat_sigma, 4.0), 24.0)),
        "patterns": records,
        "gates": gates,
        "verdict": verdict,
        "failed_gates": failed,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# 输出：JSON + 中文 Markdown
# ---------------------------------------------------------------------------
def _as_repo_str(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _fmt(value: object, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "pass" if value else "fail"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _report_gate_rows(gates: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    """(门槛, 状态, 实测, 期望) —— 报告表格用，覆盖每个子判据。"""
    rows: list[tuple[str, str, str, str]] = []
    drift = gates["drift_gate"]
    rows.append(
        (
            "drift_gate（flat 质心漂移）",
            _status_text(drift["pass"]),
            f"{_fmt(drift['measured_max_px'])} px",
            f"≤ {_fmt(drift['threshold_px'])} px",
        )
    )
    saturation = gates["saturation_gate"]
    rows.append(
        (
            "saturation_gate（成片饱和）",
            _status_text(saturation["pass"]),
            "无" if not saturation["saturated_patterns"] else ", ".join(saturation["saturated_patterns"]),
            "0 个饱和图样",
        )
    )
    for axis in ("x", "y"):
        item = gates["waveform_gate"]["axes"][axis]
        rows.append(
            (
                f"waveform_gate[{axis}]（grating vs blaze 主方向）",
                item["status"],
                f"{_fmt(item['measured_diff_deg'])}°",
                "≤2° 波形无关 / ≥5° 波形依赖",
            )
        )
    blaze = gates["axis_gate"]["blaze"]
    rows.append(
        (
            "axis_gate.blaze（两轴方向差）",
            _status_text(blaze["pass"]),
            f"{_fmt(blaze['measured_diff_deg'])}°",
            f"{_fmt(blaze['expected_deg'], 3)}° ± {_fmt(blaze['tol_deg'])}°",
        )
    )
    tilt = gates["axis_gate"]["tilt"]
    rows.append(
        (
            "axis_gate.tilt（两轴方向差）",
            _status_text(tilt["pass"]),
            f"{_fmt(tilt['measured_diff_deg'])}°",
            f"> {_fmt(tilt['min_deg'])}°",
        )
    )
    for axis in ("x", "y"):
        item = gates["scale_gate"]["axes"][axis]
        rows.append(
            (
                f"scale_gate[{axis}]（p128/p256 位移比）",
                _status_text(item["pass"]),
                _fmt(item["measured_ratio"]),
                f"{_fmt(item['expected'])} ± {_fmt(item['tol'])}",
            )
        )
    for pair in gates["sign_gate"]["pairs"]:
        rows.append(
            (
                f"sign_gate[{pair['family']}_{pair['axis']}]（+ 与 − 夹角）",
                _status_text(pair["pass"]),
                f"{_fmt(pair['measured_angle_deg'])}°",
                f"{_fmt(pair['expected_deg'])}° ± {_fmt(pair['tol_deg'])}°",
            )
        )
    for pair in gates["patch_gate"]["pairs"]:
        rows.append(
            (
                f"patch_gate[{pair['axis']}]（patch vs blaze 主方向）",
                _status_text(pair["pass"]),
                f"{_fmt(pair['measured_diff_deg'])}°",
                f"≤ {_fmt(pair['max_deg'])}°",
            )
        )
    return rows


def build_report(analysis: dict[str, Any], catalog: list[PatternSpec]) -> str:
    """生成中文 Markdown 报告。"""
    lines: list[str] = []
    verdict = analysis["verdict"]
    lines.append(f"# #247 波形判别实验离线分析报告（{analysis['generated_at_utc']}）")
    lines.append("")
    lines.append("## 摘要")
    lines.append("")
    lines.append(f"- 结论：`{verdict}`（{VERDICT_ZH.get(verdict, '未知')}）")
    lines.append(f"- 采集目录：`{analysis.get('capture_dir') or '（未记录）'}`")
    flat = analysis["flat_reference"]
    lines.append(
        f"- flat 参考漂移：最大 {_fmt(flat['max_drift_px'])} px"
        f"（门限 {_fmt(analysis['thresholds']['drift_max_px'])} px）"
        f" → {_status_text(analysis['gates']['drift_gate']['pass'])}"
    )
    saturation = analysis["gates"]["saturation_gate"]
    lines.append(
        "- 饱和图样："
        + ("无" if not saturation["saturated_patterns"] else ", ".join(saturation["saturated_patterns"]))
        + f" → {_status_text(saturation['pass'])}"
    )
    full_scale = analysis["full_scale"]
    lines.append(
        f"- 满量程：{_fmt(full_scale['value'], 0)} counts"
        f"（来源 {full_scale['source']}"
        + ("，assumed_full_scale=true" if full_scale["assumed_full_scale"] else "")
        + "）"
    )
    failed = analysis["failed_gates"]
    lines.append("- 未通过门槛：" + ("无" if not failed else ", ".join(failed)))
    lines.append("")

    lines.append("## 逐图样测量")
    lines.append("")
    lines.append(
        "| ID | 族 | 轴 | 周期(px) | dr(px) | dc(px) | 角度(°) | |位移|(px) | 峰数 | 饱和 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for spec in catalog:
        record = analysis["patterns"].get(spec.pattern_id)
        if record is None:
            lines.append(
                f"| `{spec.pattern_id}` | {spec.family} | {spec.axis or '—'} | "
                f"{_fmt(spec.period_px, 0)} | — | — | — | — | 0 | 缺帧 |"
            )
            continue
        principal = record.get("principal")
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                spec.pattern_id,
                spec.family,
                spec.axis or "—",
                _fmt(spec.period_px, 0),
                _fmt(principal["dr"] if principal else None),
                _fmt(principal["dc"] if principal else None),
                _fmt(principal["angle_deg"] if principal else None),
                _fmt(principal["magnitude"] if principal else None),
                record.get("peak_count", 0),
                "是" if record["saturated"] else "否",
            )
        )
    lines.append("")

    grating_records = [
        record for record in analysis["patterns"].values() if record["family"] == "grating"
    ]
    if grating_records:
        lines.append("### 二值 grating 的 ±1 两侧峰")
        lines.append("")
        lines.append("| ID | 侧 | dr(px) | dc(px) | 角度(°) | |位移|(px) | 峰值(counts) |")
        lines.append("|---|---|---|---|---|---|---|")
        for record in grating_records:
            for index, lobe in enumerate(record["lobes"]):
                lines.append(
                    "| `{}` | {} | {} | {} | {} | {} | {} |".format(
                        record["pattern_id"],
                        "A(主峰)" if index == 0 else "B(反向)",
                        _fmt(lobe["dr"]),
                        _fmt(lobe["dc"]),
                        _fmt(lobe["angle_deg"]),
                        _fmt(lobe["dist"]),
                        _fmt(lobe["peak"], 1),
                    )
                )
            if not record["lobes"]:
                lines.append(
                    f"| `{record['pattern_id']}` | 未找到 | — | — | — | — | — |"
                )
        lines.append("")

    lines.append("## 逐门槛判定")
    lines.append("")
    lines.append("| 门槛 | 状态 | 实测 | 期望 |")
    lines.append("|---|---|---|---|")
    for name, status, measured, expected in _report_gate_rows(analysis["gates"]):
        lines.append(f"| {name} | {status} | {measured} | {expected} |")
    lines.append("")

    lines.append("## 结论")
    lines.append("")
    lines.append(f"- `{verdict}`：{VERDICT_ZH.get(verdict, '未知')}。")
    if failed:
        lines.append("- 失败门槛逐项见上表；未通过的门槛不会被其它门槛抵消。")
    if verdict == "waveform_independent_map_ok":
        lines.append(
            "- 波形与面板轴的对应关系在既定阈值内一致；该结果仅是 #247 的验证材料，"
            "仍不自动冻结映射。"
        )
    elif verdict == "binary_grating_specific":
        lines.append(
            "- 异常只出现在二值 grating 族：应排查灰度/LUT/选峰口径（H1），不改物理预测模型。"
        )
    elif verdict == "tilt_specific":
        lines.append(
            "- 异常只出现在 tilt 族：应排查 tilt builder、period clamp 与灰度编码（H3）。"
        )
    elif verdict == "patch_position_specific":
        lines.append(
            "- 局部 blaze patch 与全幅 blaze 方向不一致：应排查 patch 位置线性相位（H4）。"
        )
    elif verdict == "map_not_reproduced":
        lines.append(
            "- 连续 blaze 自身也未复现既定映射：先查光路漂移与 phase→gray→SLM 写入链，"
            "暂停任何写回。"
        )
    else:
        lines.append("- 窗口漂移或饱和使判据失效；不据此下任何方向结论，需重采。")
    lines.append("")
    lines.append("## 边界")
    lines.append("")
    lines.append("本结果不自动改映射、不写 slm_geometry_calib.npz。")
    lines.append("")
    lines.append("未修改 issue247_axis_map*.json；分析产物只有本文件与同名 JSON。")
    lines.append("")
    if analysis["notes"]:
        lines.append("## 备注")
        lines.append("")
        for note in analysis["notes"]:
            lines.append(f"- {note}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 自检（合成帧，无设备）
# ---------------------------------------------------------------------------
def _synthetic_frame(
    shape: tuple[int, int],
    center: tuple[float, float],
    lobes: list[tuple[float, float, float]],
    sigma: float = 7.0,
    zero_order_amplitude: float = 1200.0,
    background: float = 3.0,
) -> np.ndarray:
    rows = np.arange(shape[0])[:, None]
    cols = np.arange(shape[1])[None, :]
    image = background + zero_order_amplitude * np.exp(
        -((rows - center[0]) ** 2 + (cols - center[1]) ** 2) / (2.0 * sigma**2)
    )
    for dr, dc, amplitude in lobes:
        image = image + amplitude * np.exp(
            -((rows - (center[0] + dr)) ** 2 + (cols - (center[1] + dc)) ** 2)
            / (2.0 * sigma**2)
        )
    return image.astype(np.float32)


def build_self_test_capture(scene: str) -> Capture:
    """构造一个内存采集，供 ``--self-test`` 与单元测试共用。

    场景：``consistent``（全部一致）、``grating_rotated``（二值 grating 转 90°）、
    ``flat_drift``（flat_post 漂移 5 px）。
    """
    if scene not in ("consistent", "grating_rotated", "flat_drift"):
        raise ValueError(f"未知自检场景 {scene!r}")
    shape = (360, 360)
    center = (180.0, 180.0)
    unit_x = (0.0, 1.0)  # panel-x -> +CCD 列
    unit_y = (1.0, 0.0)  # panel-y -> +CCD 行
    rotated = scene == "grating_rotated"
    grating_x_dir = unit_y if rotated else unit_x
    grating_y_dir = unit_x if rotated else unit_y
    full = 96.0
    half = 48.0

    def lobe(direction: tuple[float, float], distance: float, amplitude: float) -> tuple[float, float, float]:
        return (direction[0] * distance, direction[1] * distance, amplitude)

    layout: list[tuple[str, str, int | None, int | None, float | None, list[tuple[float, float, float]]]] = [
        ("flat_pre", "flat", None, None, None, []),
        ("blaze_x_plus", "tilt", 1, 1, 128.0, [lobe(unit_x, full, 420.0)]),
        ("blaze_x_minus", "tilt", 1, -1, 128.0, [lobe(unit_x, -full, 420.0)]),
        ("blaze_y_plus", "tilt", 0, 1, 128.0, [lobe(unit_y, full, 420.0)]),
        ("blaze_y_minus", "tilt", 0, -1, 128.0, [lobe(unit_y, -full, 420.0)]),
        (
            "grating_x_p128",
            "grating",
            1,
            None,
            128,
            [lobe(grating_x_dir, full, 400.0), lobe(grating_x_dir, -full, 400.0)],
        ),
        (
            "grating_y_p128",
            "grating",
            0,
            None,
            128,
            [lobe(grating_y_dir, full, 400.0), lobe(grating_y_dir, -full, 400.0)],
        ),
        ("tilt_x_plus", "tilt", 1, 1, 128.0, [lobe(unit_x, full, 420.0)]),
        ("tilt_y_plus", "tilt", 0, 1, 120.0, [lobe(unit_y, full, 420.0)]),
        ("flat_mid", "flat", None, None, None, []),
        (
            "grating_x_p256",
            "grating",
            1,
            None,
            256,
            [lobe(grating_x_dir, half, 400.0), lobe(grating_x_dir, -half, 400.0)],
        ),
        (
            "grating_y_p256",
            "grating",
            0,
            None,
            256,
            [lobe(grating_y_dir, half, 400.0), lobe(grating_y_dir, -half, 400.0)],
        ),
        ("blaze_patch_x", "blaze_patch", 1, None, 96.0, [lobe(unit_x, full, 220.0)]),
        ("blaze_patch_y", "blaze_patch", 0, None, 96.0, [lobe(unit_y, full, 220.0)]),
        ("flat_post", "flat", None, None, None, []),
    ]

    patterns: list[dict[str, Any]] = []
    means: dict[str, np.ndarray] = {}
    frame_stats: dict[str, dict[str, Any]] = {}
    for pattern_id, kind, axis, sign, period, lobes in layout:
        shift = (0.0, 5.0) if (scene == "flat_drift" and pattern_id == "flat_post") else (0.0, 0.0)
        frame = _synthetic_frame(shape, (center[0] + shift[0], center[1] + shift[1]), lobes)
        entry: dict[str, Any] = {"id": pattern_id, "kind": kind, "frames": 4}
        if axis is not None:
            entry["axis"] = int(axis)
        if sign is not None:
            entry["sign"] = int(sign)
        if period is not None:
            entry["period_px"] = float(period)
        patterns.append(entry)
        means[pattern_id] = frame
        frame_stats[pattern_id] = {
            "n_frames": 4,
            "frames": [f"{index:04d}_{pattern_id}.npy" for index in range(4)],
            "frame_max_counts": [float(frame.max())] * 4,
            "max_counts": float(frame.max()),
            "pixels_at_or_above_threshold": 0,
            "saturated_frame_indices": [],
            "saturated": False,
        }

    plan = {"patterns": patterns, "analysis_thresholds": {}}
    return Capture(
        plan=plan,
        catalog=build_specs(plan),
        means=means,
        frame_stats=frame_stats,
        full_scale=4095.0,
        assumed_full_scale=False,
        full_scale_source="self_test:synthetic",
        summary={},
        missing=[],
    )


def run_self_test() -> int:
    """三种合成场景核对 verdict 与门槛逻辑；不读真帧、不碰设备。"""
    expected = {
        "consistent": "waveform_independent_map_ok",
        "grating_rotated": "binary_grating_specific",
        "flat_drift": "invalid_drift_or_saturation",
    }
    failures: list[str] = []
    for scene, wanted in expected.items():
        capture = build_self_test_capture(scene)
        analysis = analyze(capture, dict(DEFAULT_THRESHOLDS))
        verdict = analysis["verdict"]
        ok = verdict == wanted
        print(
            f"[self-test] scene={scene:16s} verdict={verdict:32s} "
            f"expected={wanted:32s} {'OK' if ok else 'FAIL'}"
        )
        if not ok:
            failures.append(f"{scene}: got {verdict}, expected {wanted}")
    if failures:
        for item in failures:
            print(f"[self-test] FAILED {item}", file=__import__("sys").stderr)
        return 1
    print("[self-test] all invariants passed")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Issue #247 波形判别实验离线分析：读 plan.json + frames/*.npy，"
            "判 waveform/axis/scale/sign 门槛并写出 JSON 与中文报告。"
        )
    )
    parser.add_argument(
        "--dir",
        dest="capture_dir",
        type=Path,
        default=None,
        help="采集目录（含 plan.json、frames/、可选 summary.json）",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="用合成帧自检三种场景，不读真帧、不碰设备",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖采集目录里已存在的分析产物",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()
    if args.capture_dir is None:
        parser.error("--dir 是必需参数（或改用 --self-test）")

    capture_dir = Path(args.capture_dir)
    plan_path = capture_dir / "plan.json"
    if not plan_path.is_file():
        print(f"!! 缺少 {plan_path}", file=__import__("sys").stderr)
        return 2
    out_json = capture_dir / ANALYSIS_JSON_NAME
    out_report = capture_dir / REPORT_MD_NAME
    if not args.overwrite and (out_json.exists() or out_report.exists()):
        print(
            f"!! 分析产物已存在（{out_json.name} / {out_report.name}）；"
            "确认要重跑请加 --overwrite",
            file=__import__("sys").stderr,
        )
        return 3

    try:
        plan = _read_json(plan_path)
        summary_path = capture_dir / "summary.json"
        summary = _read_json(summary_path) if summary_path.is_file() else {}
        plan["_capture_dir"] = capture_dir
        thresholds, overrides = resolve_thresholds(plan)
        capture = load_capture(capture_dir, plan, summary, thresholds)
        analysis = analyze(capture, thresholds)
    except (OSError, ValueError, KeyError) as exc:
        print(f"!! 分析失败：{exc}", file=__import__("sys").stderr)
        return 2

    analysis["threshold_overrides"] = overrides
    try:
        write_json(out_json, analysis)
        out_report.write_text(
            build_report(analysis, capture.catalog) + "\n", encoding="utf-8", newline="\n"
        )
    except OSError as exc:
        print(f"!! 写产物失败：{exc}", file=__import__("sys").stderr)
        return 2

    print(f"verdict: {analysis['verdict']}")
    print(f"-> {out_json}")
    print(f"-> {out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
