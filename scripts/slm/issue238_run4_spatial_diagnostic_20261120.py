r"""Issue #238 run4 离线空间诊断（第一阶段：清单核验 + 固定 ROI 描述量）。

用途（2026-11-20）：对
``data/experiments/issue238_binary_phase_20261119/run4_period_sweep/period32``、
``period48`` 的原始 Mono12 全幅帧做**只读**离线审计与描述性测量，服务于
issue #238 的「离线有效性判断」。本脚本不打开 SLM / CCD / 激光，不 import 任何
硬件模块，不碰 ``DeviceLock``，也不写回原始 run 目录。

本脚本是**第一阶段**：只做清单/元数据核验与预注册的固定 ROI 描述量
（归一化径向轮廓、孔径捕获比例、局部/全局背景、逐灰度块漂移）。§6.6 的
M0/M1 嵌套诊断模型**没有实现**，因此报告显式携带

    stage = "phase1_only"
    "algorithm_benefit_verdict": null
    "verdict_emitted": false

并拒绝输出任何算法收益 / 机制归因 / LUT 或 2π 灰度更新结论（见
``stage_declaration``）。原因：period32 与 period48 各自只有一个顺序 sweep，
period 与采集时段/漂移完全共线；两周期点不足以拟合周期函数；``flat`` 是
照明下的零灰度帧而非遮光暗场。把这些描述量当作机制证据会越界。

冻结规则（实现前写死，不因结果调整）
------------------------------------
- 0 级中心：由 ``group="reference"`` 的 flat 帧按固定规则（高斯平滑 → 峰值 →
  15×15 质心）估计，**整段所有帧共用**这一个中心。
- ±1 级：取**预注册**的探针灰度 ``PROBE_GRAYS = (448, 480, 512)`` 的均值帧，
  在屏蔽 0 级 blank 盘与边界后取最亮平滑像素为 +1，按点对称给出 −1；
  定位一次、整段冻结，**不逐帧搜索、不按结果挑灰度或改 ROI**。
- 孔径：固定圆盘 r=30 px（与 ``issue238_binary_fit.py`` 一致）、r=60 px、
  r=120 px；背景环 r∈[40,70]、r∈[75,105] 只作诊断；均在定位后一次性计算，
  所有帧、所有灰度共用。
- 背景扣除口径冻结为**每帧唯一的主台阶 = 全幅远场背景中位数**
  （屏蔽 ±3 级圆盘后按 stride 采样），孔径积分与归一化轮廓共用它；局部环会
  被该级翼部与相邻级次抬高，只用于报告 ``*_altbg`` 敏感性对照。
- 归一化径向轮廓（2 px bin，至 120 px）与行/列边缘轮廓（限制在 r<=120 圆盘内、
  2 px bin、半宽 128 px）都以 ``net_total`` 归一化，各 bin 分数之和为 1。
- 每灰度 20 帧只在**块内**聚合（中位数/IQR），不当 20 个独立硬件重复；
  周期比较的不确定度用**连续灰度块** block bootstrap，绝不按帧随机拆分。

安全/边界
---------
- ``--out`` 必填，且拒绝落在原始 run 目录、任一输入子树或仓库 ``data/`` 下；
  推荐写到 ``$env:TEMP\opencode\...``。
- NPZ 一律 ``allow_pickle=False`` 读取；逐帧流式处理，不缓存整段帧。
- 输入一律只读；本脚本只写一个 JSON（先写同目录 ``.tmp`` 再原子替换）。

用法（Windows PowerShell，项目 Python 3.12）
---------------------------------------------
::

    .venv\\Scripts\\python.exe scripts/slm/issue238_run4_spatial_diagnostic_20261120.py --self-test

    .venv\\Scripts\\python.exe scripts/slm/issue238_run4_spatial_diagnostic_20261120.py `
        --out "$env:TEMP\\opencode\\issue238_run4_joint_diagnostic_20261120\\period32_48_phase1.json"

退出码：0 = 报告已写且关键完整性检查通过；1 = 报告已写但存在完整性失败/未完成帧；
2 = 输入不可用或 ``--out`` 被拒绝。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "issue238-run4-spatial-diagnostic/v1"
STAGE = "phase1_only"

#: 预注册探针灰度（plan.binary_sweep 内，步进 32 的档位；不按测量结果挑选）。
PROBE_GRAYS: tuple[int, ...] = (448, 480, 512)

_ORDER_NAMES: tuple[str, ...] = ("zero", "plus1", "minus1")
_LABEL_RE = re.compile(r"^(?P<prefix>\d{4})_(?P<base>.+?)(?:_r(?P<rep>\d{2}))?$")
_MANIFEST_FILES: tuple[str, ...] = (
    "plan.json",
    "summary.json",
    "frames.jsonl",
    "optical_path_record.json",
    "device_log.json",
    "release.json",
)


# ---------------------------------------------------------------------------
# 配置（所有 ROI / 阈值 / 定位规则在此冻结，CLI 只能改有限的几个）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Cfg:
    """冻结的测量配置。默认值即 run4 正式口径。"""

    formal_exposure_ms: float = 0.08
    aperture_r_small: float = 30.0
    aperture_r_large: float = 60.0
    aperture_r_total: float = 120.0
    bg_ring_small: tuple[float, float] = (40.0, 70.0)
    bg_ring_large: tuple[float, float] = (75.0, 105.0)
    profile_max_r: float = 120.0
    profile_step: float = 2.0
    marginal_half: float = 128.0
    marginal_step: float = 2.0
    zero_blank_r: float = 120.0
    border: int = 80
    smooth_sigma: float = 3.0
    probe_grays: tuple[int, ...] = PROBE_GRAYS
    frames_per_level: int = 20
    global_bg_stride: int = 4
    global_bg_exclude_r: float = 45.0
    global_bg_order_k: int = 3
    block_levels: int = 8
    bootstrap: int = 400
    seed: int = 238
    hash_frames: str = "all"

    @property
    def patch_half(self) -> int:
        """单个衍射级局部窗的半宽（px），覆盖 profile 与最大背景环。"""
        return int(math.ceil(max(self.profile_max_r, self.bg_ring_large[1],
                                 self.bg_ring_small[1], self.marginal_half))) + 2

    @property
    def n_profile_bins(self) -> int:
        return int(round(self.profile_max_r / self.profile_step))

    @property
    def n_marginal_bins(self) -> int:
        return int(round(2.0 * self.marginal_half / self.marginal_step))


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """返回文件 SHA256；文件不存在时返回空串（缺失由检查项单独报告）。"""
    try:
        with path.open("rb") as fh:
            digest = hashlib.sha256()
            while True:
                block = fh.read(chunk)
                if not block:
                    break
                digest.update(block)
            return digest.hexdigest()
    except FileNotFoundError:
        return ""


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            text = line.strip()
            if not text:
                continue
            obj = json.loads(text)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no} 不是 JSON object")
            rows.append(obj)
    return rows


def load_frame_npz(path: Path) -> dict[str, Any]:
    """安全读取一帧 NPZ（``allow_pickle=False``），返回标量与 uint16 帧。"""
    with np.load(path, allow_pickle=False) as npz:
        if "frame" not in npz.files:
            raise ValueError(f"{path}: NPZ 缺少 'frame'")
        frame = npz["frame"]
        if frame.dtype != np.uint16:
            raise ValueError(f"{path}: frame dtype={frame.dtype}，期望 uint16")
        meta: dict[str, Any] = {"frame": frame}
        for key in ("kind", "gray", "period_px", "exposure_ms", "timestamp_utc",
                    "meta_json"):
            if key in npz.files:
                value = npz[key]
                meta[key] = value.item() if value.shape == () else value
        return meta


def to_jsonable(obj: Any, nd: int = 6) -> Any:
    """递归转成 JSON 安全对象：numpy 标量/数组、Path、tuple，NaN/Inf → null。"""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, nd) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v, nd) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist(), nd)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return round(val, nd) if math.isfinite(val) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _median_iqr(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"median": None, "iqr": [None, None], "n": 0}
    q25, q50, q75 = np.percentile(arr, [25.0, 50.0, 75.0])
    return {"median": float(q50), "iqr": [float(q25), float(q75)], "n": int(arr.size)}


class OutRefused(RuntimeError):
    """``--out`` 落在受保护输入/原始数据树内。"""


def guard_out(out: str | Path, *, protected: Sequence[Path]) -> Path:
    """校验 ``--out``：必须是 .json，且不在任何受保护输入/原始数据树内。"""
    path = Path(out).expanduser()
    if path.suffix.lower() != ".json":
        raise OutRefused(f"--out 必须是 .json 文件，收到 {path}")
    resolved = path.resolve()
    for root in protected:
        root_resolved = Path(root).resolve()
        if resolved == root_resolved or resolved.is_relative_to(root_resolved):
            raise OutRefused(
                f"--out {resolved} 落在受保护输入/原始数据树 {root_resolved} 内；"
                "请显式指定 Temp 输出路径（例如 "
                "$env:TEMP\\opencode\\issue238_run4_joint_diagnostic_20261120\\*.json）"
            )
    return resolved


# ---------------------------------------------------------------------------
# 清单 / 元数据核验（只读 JSONL + plan/summary，不读图像）
# ---------------------------------------------------------------------------
def _ck(checks: list[dict[str, Any]], cid: str, ok: bool | None,
        detail: Any = None) -> None:
    checks.append({"id": cid, "ok": ok, "detail": detail})


def verify_manifest(plan: dict[str, Any], summary: dict[str, Any],
                    rows: list[dict[str, Any]], period_px: int, cfg: Cfg,
                    period_dir: Path) -> dict[str, Any]:
    """核验 frame manifests：帧数、标签、灰度/周期、曝光、饱和、采集顺序。

    只读 JSON 元数据；逐帧 NPZ 标量/shape 的一致性在流式阶段另行累计。
    """
    checks: list[dict[str, Any]] = []
    frames_per_level = int(plan.get("frames_per_level", cfg.frames_per_level))
    sweep_plan: list[int] = [int(g) for g in plan.get("binary_sweep", [])]
    patterns = plan.get("patterns", [])
    pattern_by_index = {int(p["index"]): p for p in patterns if "index" in p}

    # 1) 输入文件齐备
    missing = [name for name in _MANIFEST_FILES if not (period_dir / name).is_file()]
    _ck(checks, "files_present", not missing,
        {"missing": missing, "frame_dir_exists": (period_dir / "frames").is_dir()})

    # 2) plan schema / mode
    schema = plan.get("schema")
    mode = plan.get("mode")
    _ck(checks, "plan_schema", schema == "issue238-gray-phase-probes/v2-binary"
        and mode == "binary", {"schema": schema, "mode": mode})

    # 3) plan 周期与占空
    plan_period = plan.get("binary_period_px")
    duty = plan.get("binary_duty")
    _ck(checks, "plan_period_and_duty",
        plan_period == period_px and float(duty) == 0.5,
        {"plan_period_px": plan_period, "expected_period_px": period_px,
         "duty": duty})

    # 4) plan 计数自洽
    counts = {
        "n_sweep": plan.get("n_sweep"),
        "len_binary_sweep": len(sweep_plan),
        "n_patterns": plan.get("n_patterns"),
        "len_patterns": len(patterns),
        "n_frames_expected": plan.get("n_frames_expected"),
        "sum_pattern_frames": sum(int(p.get("frames", 0)) for p in patterns),
        "frames_per_level": frames_per_level,
    }
    _ck(checks, "plan_counts",
        counts["n_sweep"] == counts["len_binary_sweep"]
        and counts["n_patterns"] == counts["len_patterns"]
        and counts["n_frames_expected"] == counts["sum_pattern_frames"]
        and counts["frames_per_level"] == cfg.frames_per_level, counts)

    # 5) summary 状态
    _ck(checks, "summary_status",
        summary.get("status") == "ok"
        and int(summary.get("frame_count", -1)) == len(rows)
        and int(summary.get("n_patterns", -1)) == len(patterns),
        {"status": summary.get("status"), "frame_count": summary.get("frame_count"),
         "rows": len(rows), "n_patterns": summary.get("n_patterns")})

    # 6) 帧行数
    expected_rows = int(plan.get("n_frames_expected", -1))
    _ck(checks, "frame_count", len(rows) == expected_rows,
        {"rows": len(rows), "expected": expected_rows})

    # 7) 行索引连续
    indices = [int(r.get("index", -1)) for r in rows]
    _ck(checks, "frame_index_sequential",
        indices == list(range(len(rows))),
        {"n": len(indices),
         "first_bad": next((i for i, v in enumerate(indices) if v != i), None)})

    # 8) 必需字段
    required = ("index", "label", "kind", "group", "gray", "period_px",
                "exposure_ms", "frame_file", "frame_shape", "frame_dtype",
                "full_scale", "timestamp_utc", "monotonic_ns", "peak",
                "saturated_pixels")
    missing_fields = sorted({key for r in rows for key in required if key not in r})
    _ck(checks, "row_fields", not missing_fields, {"missing": missing_fields})

    # 9) 标签 ↔ plan pattern 映射
    #    sweep/reference/drift 行标签形如 ``0002_binary_r00``；
    #    bracket 行标签形如 ``bracket_00``，映射到 group="bracket" 的唯一 pattern。
    bracket_patterns = [p for p in patterns if p.get("group") == "bracket"]
    bracket_index = (int(bracket_patterns[0]["index"])
                     if len(bracket_patterns) == 1 else None)

    def resolve_pattern(row: dict[str, Any]) -> tuple[int | None, dict[str, Any] | None]:
        if row.get("group") == "bracket":
            if bracket_index is None:
                return None, None
            return bracket_index, pattern_by_index.get(bracket_index)
        match = _LABEL_RE.match(str(row.get("label", "")))
        if not match:
            return None, None
        pidx = int(match.group("prefix"))
        return pidx, pattern_by_index.get(pidx)

    map_errors: list[dict[str, Any]] = []
    for row in rows:
        label = str(row.get("label", ""))
        if row.get("group") == "bracket":
            if not re.match(r"^bracket_\d{2}$", label):
                map_errors.append({"label": label, "why": "bracket_label_format"})
            if bracket_index is None:
                map_errors.append({"label": label, "why": "bracket_pattern_ambiguous"})
                continue
            pattern = pattern_by_index.get(bracket_index)
        else:
            match = _LABEL_RE.match(label)
            if not match:
                map_errors.append({"label": label, "why": "label_pattern"})
                continue
            pidx = int(match.group("prefix"))
            pattern = pattern_by_index.get(pidx)
            if pattern is None:
                map_errors.append({"label": label, "why": "pattern_index_missing"})
                continue
            base = match.group("base")
            pattern_base = str(pattern.get("label", "")).split("_", 1)[-1]
            if base != pattern_base:
                map_errors.append({"label": label, "why": "label_base",
                                   "pattern_label": pattern.get("label")})
        assert pattern is not None
        if row.get("kind") != pattern.get("kind"):
            map_errors.append({"label": label, "why": "kind",
                               "row": row.get("kind"),
                               "pattern": pattern.get("kind")})
        if row.get("group") != pattern.get("group"):
            map_errors.append({"label": label, "why": "group",
                               "row": row.get("group"),
                               "pattern": pattern.get("group")})
        if row.get("kind") == "binary":
            if int(row.get("gray", -1)) != int(pattern.get("gray", -2)):
                map_errors.append({"label": label, "why": "gray",
                                   "row": row.get("gray"),
                                   "pattern": pattern.get("gray")})
            if int(row.get("period_px", -1)) != period_px:
                map_errors.append({"label": label, "why": "period",
                                   "row": row.get("period_px")})
            params = row.get("pattern_params") or {}
            if float(params.get("duty", -1.0)) != float(duty):
                map_errors.append({"label": label, "why": "duty",
                                   "row": params.get("duty")})
        else:
            if int(row.get("gray", -1)) != int(pattern.get("gray", -2)):
                map_errors.append({"label": label, "why": "flat_gray",
                                   "row": row.get("gray"),
                                   "pattern": pattern.get("gray")})
            if int(row.get("period_px", -1)) != 0:
                map_errors.append({"label": label, "why": "flat_period_nonzero",
                                   "row": row.get("period_px")})
    _ck(checks, "row_label_mapping", not map_errors,
        {"n_errors": len(map_errors), "examples": map_errors[:8]})

    # 10) 每个 plan pattern 的帧数
    per_prefix: dict[int, int] = {}
    for row in rows:
        pidx, _ = resolve_pattern(row)
        if pidx is not None:
            per_prefix[pidx] = per_prefix.get(pidx, 0) + 1
    frame_count_errors = [
        {"pattern_index": int(p["index"]), "label": p.get("label"),
         "expected": int(p.get("frames", 0)),
         "found": per_prefix.get(int(p["index"]), 0)}
        for p in patterns
        if per_prefix.get(int(p["index"]), 0) != int(p.get("frames", 0))
    ]
    _ck(checks, "pattern_frame_counts", not frame_count_errors,
        {"n_errors": len(frame_count_errors), "examples": frame_count_errors[:8]})

    # 11) sweep 灰度集合与每灰度帧数
    sweep_rows = [r for r in rows if r.get("kind") == "binary"
                  and r.get("group") == "sweep"]
    gray_counts: dict[int, int] = {}
    for row in sweep_rows:
        gray = int(row.get("gray", -1))
        gray_counts[gray] = gray_counts.get(gray, 0) + 1
    expected_counts = {int(g): frames_per_level for g in sweep_plan}
    level_errors = [
        {"gray": gray, "expected": expected_counts.get(gray),
         "found": gray_counts.get(gray)}
        for gray in sorted(set(expected_counts) | set(gray_counts))
        if gray_counts.get(gray) != expected_counts.get(gray)
    ]
    _ck(checks, "sweep_level_counts", not level_errors,
        {"n_levels": len(gray_counts), "expected_levels": len(expected_counts),
         "n_errors": len(level_errors), "examples": level_errors[:8]})

    # 12) sweep 采集顺序：按 plan 灰度块连续、块内 r00..r19
    expected_sequence: list[tuple[int, int]] = [
        (int(gray), rep)
        for gray in sweep_plan
        for rep in range(frames_per_level)
    ]
    actual_sequence: list[tuple[int, int]] = []
    seq_errors: list[dict[str, Any]] = []
    for row in sweep_rows:
        match = _LABEL_RE.match(str(row.get("label", "")))
        rep = int(match.group("rep")) if match and match.group("rep") else -1
        actual_sequence.append((int(row.get("gray", -1)), rep))
    if len(actual_sequence) != len(expected_sequence):
        seq_errors.append({"why": "length", "actual": len(actual_sequence),
                           "expected": len(expected_sequence)})
    else:
        for pos, (actual, expected) in enumerate(zip(actual_sequence,
                                                     expected_sequence)):
            if actual != expected:
                seq_errors.append({"pos": pos, "actual": list(actual),
                                   "expected": list(expected)})
                if len(seq_errors) >= 8:
                    break
    _ck(checks, "sweep_gray_sequence", not seq_errors,
        {"n_errors": len(seq_errors), "examples": seq_errors[:8]})

    # 13) 正式曝光：非 bracket 行必须等于 formal；bracket 行必须在 plan 档位内
    bracket_exposures = [float(x) for x in plan.get("exposure_bracket_ms", [])]
    exp_errors: list[dict[str, Any]] = []
    for row in rows:
        value = float(row.get("exposure_ms", float("nan")))
        if row.get("group") == "bracket":
            if not any(abs(value - x) < 1e-9 for x in bracket_exposures):
                exp_errors.append({"label": row.get("label"),
                                   "why": "bracket_not_in_plan",
                                   "exposure_ms": value})
        elif abs(value - cfg.formal_exposure_ms) > 1e-9:
            exp_errors.append({"label": row.get("label"),
                               "why": "not_formal_exposure",
                               "exposure_ms": value})
    chosen = (summary.get("exposure_bracket") or {}).get("chosen_readback_ms")
    if chosen is not None and abs(float(chosen) - cfg.formal_exposure_ms) > 1e-9:
        exp_errors.append({"why": "summary_chosen_mismatch",
                           "chosen_readback_ms": chosen})
    _ck(checks, "exposure_formal", not exp_errors,
        {"formal_ms": cfg.formal_exposure_ms,
         "bracket_plan_ms": bracket_exposures, "chosen": chosen,
         "n_errors": len(exp_errors), "examples": exp_errors[:8]})

    # 14) shape / dtype / full scale / 位深
    native = plan.get("native_ccd_shape")
    full_scale = plan.get("full_scale_required")
    dtype_errors = [
        {"label": r.get("label"), "frame_shape": r.get("frame_shape"),
         "frame_dtype": r.get("frame_dtype"), "full_scale": r.get("full_scale")}
        for r in rows
        if list(r.get("frame_shape", [])) != list(native or [])
        or r.get("frame_dtype") != "uint16"
        or full_scale is not None and float(r.get("full_scale", -1)) != float(full_scale)
    ]
    _ck(checks, "shape_dtype_manifest",
        not dtype_errors and plan.get("bit_depth_required") == 12,
        {"native_ccd_shape": native, "full_scale_required": full_scale,
         "bit_depth_required": plan.get("bit_depth_required"),
         "n_errors": len(dtype_errors), "examples": dtype_errors[:5]})

    # 15) 饱和：正式数据帧必须 0 饱和；饱和只允许出现在 bracket（并已被排除）
    sat_sweep = [{"label": r.get("label"), "saturated_pixels": r.get("saturated_pixels")}
                 for r in sweep_rows if int(r.get("saturated_pixels", 0)) != 0]
    sat_other = [{"label": r.get("label"), "group": r.get("group"),
                  "saturated_pixels": r.get("saturated_pixels")}
                 for r in rows if r.get("group") != "bracket"
                 and int(r.get("saturated_pixels", 0)) != 0]
    sat_total = sum(1 for r in rows if int(r.get("saturated_pixels", 0)) != 0)
    _ck(checks, "saturation_manifest", not sat_sweep and not sat_other
        and int(summary.get("saturated_frames", -1)) == sat_total,
        {"sweep_saturated": sat_sweep[:5], "non_bracket_saturated": sat_other[:5],
         "summary_saturated_frames": summary.get("saturated_frames"),
         "rows_saturated": sat_total})

    # 16) 时间戳单调 + 灰度块时长
    mono = [int(r.get("monotonic_ns", 0)) for r in rows]
    mono_ok = all(mono[i + 1] > mono[i] for i in range(len(mono) - 1))
    ts_ok = all(str(rows[i]["timestamp_utc"]) <= str(rows[i + 1]["timestamp_utc"])
                for i in range(len(rows) - 1))
    block_spans: list[dict[str, Any]] = []
    if sweep_rows:
        by_gray: dict[int, list[dict[str, Any]]] = {}
        for row in sweep_rows:
            by_gray.setdefault(int(row.get("gray", -1)), []).append(row)
        for gray in sweep_plan:
            block = by_gray.get(int(gray), [])
            if not block:
                continue
            block_spans.append({
                "gray": int(gray),
                "t_start_utc": block[0]["timestamp_utc"],
                "t_end_utc": block[-1]["timestamp_utc"],
                "duration_s": (int(block[-1]["monotonic_ns"])
                               - int(block[0]["monotonic_ns"])) / 1e9,
            })
    starts_monotonic = all(
        block_spans[i + 1]["t_start_utc"] >= block_spans[i]["t_end_utc"]
        for i in range(len(block_spans) - 1)
    )
    _ck(checks, "timestamps_monotonic", mono_ok and ts_ok and starts_monotonic,
        {"monotonic_ns_strict": mono_ok, "timestamp_utc_nondecreasing": ts_ok,
         "level_blocks_sequential": starts_monotonic,
         "n_level_blocks": len(block_spans)})

    # 17) 帧文件存在且唯一
    frame_refs = [str(r.get("frame_file", "")) for r in rows]
    missing_files = [ref for ref in frame_refs
                     if not ref or not (period_dir / ref).is_file()]
    dupes = sorted({ref for ref in frame_refs if frame_refs.count(ref) > 1})
    _ck(checks, "frame_files_exist", not missing_files and not dupes,
        {"n_missing": len(missing_files), "missing_examples": missing_files[:5],
         "duplicates": dupes[:5]})

    # 18) bracket 与 drift/reference 结构
    bracket_rows = [r for r in rows if r.get("group") == "bracket"]
    drift_rows = [r for r in rows if r.get("group") == "drift"]
    ref_rows = [r for r in rows if r.get("group") == "reference"]
    bracket_check: dict[str, Any] = {
        "n_bracket_rows": len(bracket_rows),
        "exposures": [r.get("exposure_ms") for r in bracket_rows],
        "peaks": [r.get("peak") for r in bracket_rows],
        "saturated": [r.get("saturated_pixels") for r in bracket_rows],
        "n_drift_rows": len(drift_rows),
        "n_reference_rows": len(ref_rows),
    }
    bracket_ok: bool | None
    if bracket_rows:
        peaks = [float(r.get("peak", float("nan"))) for r in bracket_rows]
        bracket_ok = (
            [float(r.get("exposure_ms", float("nan"))) for r in bracket_rows]
            == bracket_exposures
            and all(peaks[i + 1] >= peaks[i] for i in range(len(peaks) - 1))
            and len(ref_rows) == 1 and len(drift_rows) >= 1
        )
    else:
        bracket_ok = None
    _ck(checks, "bracket_consistency", bracket_ok, bracket_check)

    failed = [c["id"] for c in checks if c["ok"] is False]
    return {
        "checks": checks,
        "failed_ids": failed,
        "ok": not failed,
        "not_applicable": [c["id"] for c in checks if c["ok"] is None],
        "level_blocks": block_spans,
    }


# ---------------------------------------------------------------------------
# 几何：0 级中心 + 固定 ±1 级偏移（整段一次，之后冻结）
# ---------------------------------------------------------------------------
def locate_zero(frame: np.ndarray, cfg: Cfg) -> tuple[float, float, dict[str, Any]]:
    """按固定规则在 reference flat 上定位 0 级：高斯平滑峰值 + 15×15 质心。"""
    img = frame.astype(np.float64)
    sm = ndimage.gaussian_filter(img, cfg.smooth_sigma)
    valid = np.ones(sm.shape, dtype=bool)
    valid[: cfg.border, :] = False
    valid[-cfg.border:, :] = False
    valid[:, : cfg.border] = False
    valid[:, -cfg.border:] = False
    if not valid.any():
        raise ValueError("图像小于 border 掩膜")
    peak_flat = int(np.argmax(np.where(valid, sm, -np.inf)))
    py, px = np.unravel_index(peak_flat, sm.shape)
    half = 7
    y0, y1 = max(0, py - half), min(sm.shape[0], py + half + 1)
    x0, x1 = max(0, px - half), min(sm.shape[1], px + half + 1)
    win = sm[y0:y1, x0:x1]
    weights = np.clip(win - float(np.median(win)), 0.0, None)
    if weights.sum() <= 0:
        cy, cx = float(py), float(px)
    else:
        gy, gx = np.mgrid[y0:y1, x0:x1]
        cy = float((weights * gy).sum() / weights.sum())
        cx = float((weights * gx).sum() / weights.sum())
    meta = {
        "method": "gaussian_smooth(argmax)+centroid15",
        "smooth_sigma_px": cfg.smooth_sigma,
        "border_px": cfg.border,
        "integer_peak_row_col": [int(py), int(px)],
        "smoothed_peak_value": float(sm[py, px]),
        "raw_peak_value": float(img[py, px]),
        "n_pixels": int(img.size),
    }
    return cy, cx, meta


def locate_orders(probe_images: Sequence[np.ndarray], zero: tuple[float, float],
                  cfg: Cfg) -> dict[str, Any]:
    """用预注册探针均值帧定位 +1/−1 级；点对称 + 屏蔽 0 级盘与边界。"""
    if not probe_images:
        raise ValueError("没有可用的探针灰度帧")
    stack = np.mean([img.astype(np.float64) for img in probe_images], axis=0)
    sm = ndimage.gaussian_filter(stack, cfg.smooth_sigma)
    cy, cx = zero
    gy, gx = np.mgrid[0:sm.shape[0], 0:sm.shape[1]]
    r2 = (gy - cy) ** 2 + (gx - cx) ** 2
    valid = r2 >= cfg.zero_blank_r ** 2
    valid[: cfg.border, :] = False
    valid[-cfg.border:, :] = False
    valid[:, : cfg.border] = False
    valid[:, -cfg.border:] = False
    if not valid.any():
        raise ValueError("屏蔽 0 级后无可用像素")
    flat_peak = int(np.argmax(np.where(valid, sm, -np.inf)))
    py, px = np.unravel_index(flat_peak, sm.shape)
    dy, dx = float(py) - cy, float(px) - cx
    spacing = float(math.hypot(dy, dx))
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("±1 级与 0 级重合，定位失败")
    my, mx = 2.0 * cy - py, 2.0 * cx - px
    iy, ix = int(round(my)), int(round(mx))
    if not (0 <= iy < sm.shape[0] and 0 <= ix < sm.shape[1]):
        raise ValueError("镜像 −1 级落在图像外")
    plus_value = float(sm[py, px])
    minus_value = float(sm[iy, ix])
    ratio = plus_value / minus_value if minus_value > 0 else float("inf")
    unit = (dy / spacing, dx / spacing)
    tilt_deg = float(math.degrees(math.atan2(abs(unit[1]), abs(unit[0]))))
    return {
        "probe_grays": [int(g) for g in cfg.probe_grays],
        "n_probe_images": len(probe_images),
        "zero_center_row_col": [float(cy), float(cx)],
        "plus1_center_row_col": [float(py), float(px)],
        "minus1_center_row_col": [my, mx],
        "offset_row_col": [dy, dx],
        "spacing_px": spacing,
        "axis_unit_row_col": [unit[0], unit[1]],
        "tilt_from_row_axis_deg": tilt_deg,
        "plus_over_minus_peak_ratio": ratio,
        "plus_peak_smoothed": plus_value,
        "minus_peak_smoothed": minus_value,
        "search_mask": {
            "zero_blank_r_px": cfg.zero_blank_r,
            "border_px": cfg.border,
            "rule": "mask zero-order disk + border, take brightest smoothed pixel",
        },
        "symmetry_check": {
            "ratio_in_0.25_2.0": bool(0.25 <= ratio <= 2.0),
            "spacing_gt_blank": bool(spacing > cfg.zero_blank_r),
            "patch_fits": bool(
                cfg.border >= cfg.patch_half
                and min(cy, cx, sm.shape[0] - cy, sm.shape[1] - cx)
                >= cfg.patch_half - 2
            ),
        },
    }


# ---------------------------------------------------------------------------
# 固定 ROI 计划（每段/每个衍射级只构建一次，逐帧复用）
# ---------------------------------------------------------------------------
@dataclass
class OrderPlan:
    name: str
    center: tuple[float, float]
    y0: int
    y1: int
    x0: int
    x1: int
    dy: np.ndarray
    dx: np.ndarray
    r2: np.ndarray
    m_small: np.ndarray
    m_large: np.ndarray
    m_total: np.ndarray
    ring_small: np.ndarray
    ring_large: np.ndarray
    bin_idx: np.ndarray
    bin_edges: np.ndarray
    r_int: np.ndarray
    row_idx: np.ndarray
    col_idx: np.ndarray
    marg_edges: np.ndarray
    n_small: int
    n_large: int
    n_total: int


def prepare_order(name: str, center: tuple[float, float], shape: tuple[int, int],
                  cfg: Cfg) -> OrderPlan:
    """为一个衍射级构建冻结的局部窗、圆盘掩膜、背景环和 bin 索引。"""
    cy, cx = float(center[0]), float(center[1])
    half = cfg.patch_half
    y0 = int(math.floor(cy)) - half
    x0 = int(math.floor(cx)) - half
    y1, x1 = y0 + 2 * half + 1, x0 + 2 * half + 1
    if y0 < 0 or x0 < 0 or y1 > shape[0] or x1 > shape[1]:
        raise ValueError(f"{name} 的 {2 * half + 1}px 局部窗越界: "
                         f"y[{y0},{y1}) x[{x0},{x1}) shape={shape}")
    gy, gx = np.mgrid[y0:y1, x0:x1]
    dy = (gy - cy).astype(np.float64)
    dx = (gx - cx).astype(np.float64)
    r2 = dy ** 2 + dx ** 2
    r = np.sqrt(r2)
    m_small = r2 <= cfg.aperture_r_small ** 2
    m_large = r2 <= cfg.aperture_r_large ** 2
    m_total = r2 <= cfg.aperture_r_total ** 2
    ring_small = (r >= cfg.bg_ring_small[0]) & (r < cfg.bg_ring_small[1])
    ring_large = (r >= cfg.bg_ring_large[0]) & (r < cfg.bg_ring_large[1])

    n_bins = cfg.n_profile_bins
    bin_edges = np.arange(n_bins + 1, dtype=np.float64) * cfg.profile_step
    bin_idx = np.floor(r / cfg.profile_step).astype(np.int64)
    bin_idx = np.where(m_total, np.clip(bin_idx, 0, n_bins - 1), -1)
    r_int = np.clip(np.floor(r).astype(np.int64), 0,
                    int(round(cfg.aperture_r_total)))

    n_marg = cfg.n_marginal_bins
    marg_edges = -cfg.marginal_half + np.arange(n_marg + 1) * cfg.marginal_step
    row_idx = np.floor((dy + cfg.marginal_half) / cfg.marginal_step).astype(np.int64)
    col_idx = np.floor((dx + cfg.marginal_half) / cfg.marginal_step).astype(np.int64)
    marg_keep = m_total & (np.abs(dy) < cfg.marginal_half) & (np.abs(dx) < cfg.marginal_half)
    row_idx = np.where(marg_keep, np.clip(row_idx, 0, n_marg - 1), -1)
    col_idx = np.where(marg_keep, np.clip(col_idx, 0, n_marg - 1), -1)
    return OrderPlan(
        name=name, center=(cy, cx), y0=y0, y1=y1, x0=x0, x1=x1,
        dy=dy, dx=dx, r2=r2, m_small=m_small, m_large=m_large, m_total=m_total,
        ring_small=ring_small, ring_large=ring_large, bin_idx=bin_idx,
        bin_edges=bin_edges, r_int=r_int, row_idx=row_idx, col_idx=col_idx,
        marg_edges=marg_edges,
        n_small=int(m_small.sum()), n_large=int(m_large.sum()),
        n_total=int(m_total.sum()),
    )


def prepare_orders(geometry: dict[str, Any], shape: tuple[int, int],
                   cfg: Cfg) -> dict[str, OrderPlan]:
    centers = {
        "zero": tuple(geometry["zero_center_row_col"]),
        "plus1": tuple(geometry["plus1_center_row_col"]),
        "minus1": tuple(geometry["minus1_center_row_col"]),
    }
    return {name: prepare_order(name, center, shape, cfg)
            for name, center in centers.items()}


# ---------------------------------------------------------------------------
# 单帧测光（固定 ROI；主背景 = 每帧全幅远场中位数）
# ---------------------------------------------------------------------------
def measure_order(patch: np.ndarray, plan: OrderPlan, cfg: Cfg,
                  bg_total: float) -> dict[str, Any]:
    """在一个冻结的局部窗内测量孔径积分、背景、归一化轮廓与二阶矩。

    ``bg_total`` 是同一帧的全幅远场背景中位数（见 :func:`global_background`），
    作为唯一的主背景扣除量：局部环 R∈[40,70] 与 [75,105] 会被该级自身翼部与
    相邻级次抬高（见 ``bg_ring_small`` / ``bg_ring_large``），只作诊断与
    敏感性对照（``*_altbg``），不能外推到 r<=``aperture_r_total`` 的整幅积分。
    轮廓用同一 ``bg_total`` 校正并以 ``net_total`` 归一化，因此各 bin 分数之和
    为 1，且与 ``capture_small`` / ``capture_large`` 的分母一致。
    """
    bg_small = float(np.median(patch[plan.ring_small]))
    bg_large = float(np.median(patch[plan.ring_large]))
    ring_small_vals = patch[plan.ring_small]
    bg_sigma = 1.4826 * float(np.median(np.abs(ring_small_vals - bg_small)))

    bg_pedestal = float(bg_total)
    raw_small = float(patch[plan.m_small].sum())
    raw_large = float(patch[plan.m_large].sum())
    raw_total = float(patch[plan.m_total].sum())
    net_small = raw_small - bg_pedestal * plan.n_small
    net_small_altbg = raw_small - bg_large * plan.n_small
    net_large = raw_large - bg_pedestal * plan.n_large
    net_large_altbg = raw_large - bg_large * plan.n_large
    net_total = raw_total - bg_pedestal * plan.n_total
    net_total_altbg = raw_total - bg_large * plan.n_total
    safe_total = net_total if abs(net_total) > 1e-9 else float("nan")

    corrected = patch - bg_pedestal
    flat_idx = plan.bin_idx.ravel()
    flat_vals = corrected.ravel()
    ok = flat_idx >= 0
    energy = np.bincount(flat_idx[ok], weights=flat_vals[ok],
                         minlength=cfg.n_profile_bins)
    counts = np.bincount(flat_idx[ok], minlength=cfg.n_profile_bins)
    radial_mean = energy / np.maximum(counts, 1)
    fraction = energy / safe_total if math.isfinite(safe_total) else np.full_like(
        energy, np.nan)

    # 由 1 px 半径 bin 的累计能量求 r50 / r90（避免 2 px bin 插值偏差）。
    # 必须先把净能量截断到非负：未截断的 (patch - bg) 含负值，累计曲线不再单调，
    # np.interp 会返回无意义值（曾出现 period48 的 -1 级 r50=93 px > 120 px 孔径、
    # 且远大于同帧 rms_radius=25.6 px 的自相矛盾结果）。
    e1 = np.bincount(plan.r_int.ravel(),
                     weights=np.clip(flat_vals, 0.0, None),
                     minlength=int(round(cfg.aperture_r_total)) + 1)
    cum = np.cumsum(e1)
    if cum[-1] > 1e-9:
        cum_frac = cum / cum[-1]
        radii = np.arange(e1.size, dtype=np.float64)
        r50 = float(np.interp(0.5, cum_frac, radii))
        r90 = float(np.interp(0.9, cum_frac, radii))
    else:
        r50 = r90 = float("nan")

    w = np.where(plan.m_large, np.clip(corrected, 0.0, None), 0.0)
    w_sum = float(w.sum())
    if w_sum > 1e-9:
        rms_radius = float(math.sqrt(float((w * plan.r2).sum()) / w_sum))
        sig_row = float(math.sqrt(float((w * plan.dy ** 2).sum()) / w_sum))
        sig_col = float(math.sqrt(float((w * plan.dx ** 2).sum()) / w_sum))
        cov = float((w * plan.dy * plan.dx).sum()) / w_sum
        corr = cov / (sig_row * sig_col) if sig_row > 0 and sig_col > 0 else float("nan")
        cen_row = float((w * plan.dy).sum()) / w_sum
        cen_col = float((w * plan.dx).sum()) / w_sum
    else:
        rms_radius = sig_row = sig_col = corr = cen_row = cen_col = float("nan")

    row_ok = plan.row_idx.ravel() >= 0
    row_marg = np.bincount(plan.row_idx.ravel()[row_ok],
                           weights=flat_vals[row_ok],
                           minlength=cfg.n_marginal_bins)
    col_ok = plan.col_idx.ravel() >= 0
    col_marg = np.bincount(plan.col_idx.ravel()[col_ok],
                           weights=flat_vals[col_ok],
                           minlength=cfg.n_marginal_bins)
    if math.isfinite(safe_total):
        row_marg = row_marg / safe_total
        col_marg = col_marg / safe_total

    peak_flat = int(np.argmax(patch))
    peak_y, peak_x = np.unravel_index(peak_flat, patch.shape)
    return {
        "net_small": net_small,
        "net_small_altbg": net_small_altbg,
        "net_large": net_large,
        "net_large_altbg": net_large_altbg,
        "net_total": net_total,
        "net_total_altbg": net_total_altbg,
        "capture_small": net_small / safe_total if math.isfinite(safe_total) else float("nan"),
        "capture_large": net_large / safe_total if math.isfinite(safe_total) else float("nan"),
        "capture_small_altbg": (net_small_altbg / safe_total
                                if math.isfinite(safe_total) else float("nan")),
        "capture_large_altbg": (net_large_altbg / safe_total
                                if math.isfinite(safe_total) else float("nan")),
        "bg_pedestal": bg_pedestal,
        "bg_ring_small": bg_small,
        "bg_ring_large": bg_large,
        "bg_sigma": bg_sigma,
        "rms_radius_px": rms_radius,
        "r50_px": r50,
        "r90_px": r90,
        "sigma_row_px": sig_row,
        "sigma_col_px": sig_col,
        "corr_row_col": corr,
        "centroid_shift_row_px": cen_row,
        "centroid_shift_col_px": cen_col,
        "peak_value": float(patch.ravel()[peak_flat]),
        "peak_offset_row_col_px": [int(peak_y) - cfg.patch_half,
                                   int(peak_x) - cfg.patch_half],
        "radial_mean": radial_mean,
        "radial_energy_fraction": fraction,
        "row_marginal": row_marg,
        "col_marginal": col_marg,
        "n_pixels_small": plan.n_small,
        "n_pixels_large": plan.n_large,
        "n_pixels_total": plan.n_total,
    }


def global_background(frame: np.ndarray, geometry: dict[str, Any],
                      cfg: Cfg) -> dict[str, float]:
    """按固定 stride 采样、屏蔽 ±k 级圆盘后的全幅背景统计（描述量）。"""
    step = cfg.global_bg_stride
    sub = frame[::step, ::step].astype(np.float64)
    ys = np.arange(0, frame.shape[0], step, dtype=np.float64)
    xs = np.arange(0, frame.shape[1], step, dtype=np.float64)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    cy, cx = geometry["zero_center_row_col"]
    off_y, off_x = geometry["offset_row_col"]
    mask = np.ones(sub.shape, dtype=bool)
    for k in range(-cfg.global_bg_order_k, cfg.global_bg_order_k + 1):
        d2 = (gy - (cy + k * off_y)) ** 2 + (gx - (cx + k * off_x)) ** 2
        mask &= d2 >= cfg.global_bg_exclude_r ** 2
    values = sub[mask]
    if values.size == 0:
        return {"median": float("nan"), "sigma": float("nan"), "n": 0}
    med = float(np.median(values))
    sigma = 1.4826 * float(np.median(np.abs(values - med)))
    return {"median": med, "sigma": sigma, "n": int(values.size)}


def measure_frame(frame: np.ndarray, orders: dict[str, OrderPlan],
                  geometry: dict[str, Any], cfg: Cfg,
                  names: Sequence[str] = _ORDER_NAMES) -> dict[str, Any]:
    """测量一帧内指定衍射级的固定 ROI 描述量。"""
    out: dict[str, Any] = {"orders": {}}
    bg = global_background(frame, geometry, cfg)
    bg_total = bg["median"]
    for name in names:
        plan = orders[name]
        patch = frame[plan.y0:plan.y1, plan.x0:plan.x1].astype(np.float64)
        out["orders"][name] = measure_order(patch, plan, cfg, bg_total)
    out["global_bg"] = bg
    out["frame_max"] = float(frame.max())
    return out


# ---------------------------------------------------------------------------
# 逐灰度聚合（块内 20 帧只做描述统计，不当作独立重复）
# ---------------------------------------------------------------------------
_PROFILE_KEYS: tuple[str, ...] = (
    "net_small", "net_small_altbg", "net_large", "net_large_altbg",
    "net_total", "net_total_altbg",
    "capture_small", "capture_large", "capture_small_altbg",
    "capture_large_altbg", "bg_pedestal",
    "bg_ring_small", "bg_ring_large", "bg_sigma",
    "rms_radius_px", "r50_px", "r90_px",
    "sigma_row_px", "sigma_col_px", "corr_row_col",
    "centroid_shift_row_px", "centroid_shift_col_px", "peak_value",
)


def aggregate_level(gray: int, records: list[dict[str, Any]], cfg: Cfg) -> dict[str, Any]:
    """把一个灰度块内的逐帧记录聚合为中位数/IQR（描述量，非独立重复）。"""
    entry: dict[str, Any] = {
        "gray": int(gray),
        "n_frames": len(records),
        "index_span": [records[0].get("row_index"), records[-1].get("row_index")],
        "label_span": [records[0].get("label"), records[-1].get("label")],
        "t_start_utc": records[0].get("timestamp_utc"),
        "t_end_utc": records[-1].get("timestamp_utc"),
        "duration_s": (records[-1]["monotonic_ns"] - records[0]["monotonic_ns"]) / 1e9,
        "global_bg": _median_iqr(np.array([r["global_bg"]["median"] for r in records])),
        "frame_max": _median_iqr(np.array([r["frame_max"] for r in records])),
        "orders": {},
    }
    for name in _ORDER_NAMES:
        per_order = [r["orders"][name] for r in records]
        agg: dict[str, Any] = {}
        for key in _PROFILE_KEYS:
            agg[key] = _median_iqr(np.array([o[key] for o in per_order]))
        for key in ("radial_mean", "radial_energy_fraction", "row_marginal",
                    "col_marginal"):
            stack = np.vstack([np.asarray(o[key], dtype=float) for o in per_order])
            agg[key] = np.nanmedian(stack, axis=0)
        agg["n_pixels_small"] = per_order[0]["n_pixels_small"]
        agg["n_pixels_large"] = per_order[0]["n_pixels_large"]
        agg["n_pixels_total"] = per_order[0]["n_pixels_total"]
        # 块内漂移（0 级总能量；描述性，不用于判定）
        entry["orders"][name] = agg
    zero_total = np.array([r["orders"]["zero"]["net_total"] for r in records])
    if zero_total.size >= 2 and abs(np.median(zero_total)) > 1e-9:
        entry["zero_net_total_drift_pct"] = float(
            100.0 * (zero_total[-1] - zero_total[0]) / abs(np.median(zero_total)))
        t = np.arange(zero_total.size, dtype=np.float64)
        slope = float(np.polyfit(t, zero_total, 1)[0])
        entry["zero_net_total_slope_per_frame"] = slope
    else:
        entry["zero_net_total_drift_pct"] = float("nan")
        entry["zero_net_total_slope_per_frame"] = float("nan")
    plus = np.array([r["orders"]["plus1"]["net_small"] for r in records])
    minus = np.array([r["orders"]["minus1"]["net_small"] for r in records])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = plus / minus
    entry["plus_over_minus_net_small"] = _median_iqr(ratio)
    return entry


# ---------------------------------------------------------------------------
# 段级分析（流式）
# ---------------------------------------------------------------------------
def _frame_check(acc: dict[str, dict[str, Any]], key: str, ok: bool,
                 example: dict[str, Any] | None = None) -> None:
    entry = acc.setdefault(key, {"count": 0, "ok": 0, "examples": []})
    entry["count"] += 1
    if ok:
        entry["ok"] += 1
    elif len(entry["examples"]) < 5 and example is not None:
        entry["examples"].append(example)


def _frame_checks_ok(acc: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "checks": acc,
        "failed_keys": [k for k, v in acc.items() if v["ok"] != v["count"]],
        "ok": all(v["ok"] == v["count"] for v in acc.values()),
    }


def analyze_period(period_dir: Path, period_px: int, cfg: Cfg) -> dict[str, Any]:
    """流式分析一个 period 段：清单核验 → 定位 → 逐帧固定 ROI 测光 → 聚合。"""
    t_start = time.perf_counter()
    plan = _load_json(period_dir / "plan.json")
    summary = _load_json(period_dir / "summary.json")
    rows = _read_jsonl(period_dir / "frames.jsonl")
    integrity = verify_manifest(plan, summary, rows, period_px, cfg, period_dir)

    hashes: dict[str, str] = {}
    for name in _MANIFEST_FILES:
        path = period_dir / name
        if path.is_file():
            hashes[name] = sha256_file(path)

    result: dict[str, Any] = {
        "period_px": int(period_px),
        "period_dir": str(period_dir),
        "n_rows": len(rows),
        "manifest_sha256": hashes,
        "integrity": integrity,
        "geometry": {"ok": False, "reason": "not attempted"},
        "n_analyzed": 0,
        "n_expected_analysis_frames": 0,
        "frames_failed": [],
        "frame_checks": {},
        "levels": [],
        "flats": {},
        "timing_s": None,
    }
    if not integrity["ok"]:
        result["geometry"]["reason"] = "manifest integrity failed; geometry not run"
        result["timing_s"] = time.perf_counter() - t_start
        return result

    ref_rows = [r for r in rows if r.get("group") == "reference"
                and r.get("kind") == "flat"]
    probe_rows = [r for r in rows if r.get("kind") == "binary"
                  and int(r.get("gray", -1)) in cfg.probe_grays]
    if not ref_rows:
        result["geometry"]["reason"] = "no reference flat row"
        result["timing_s"] = time.perf_counter() - t_start
        return result
    if not probe_rows:
        result["geometry"]["reason"] = "no pre-registered probe-gray frames"
        result["timing_s"] = time.perf_counter() - t_start
        return result

    ref_meta = load_frame_npz(period_dir / ref_rows[0]["frame_file"])
    shape = tuple(int(v) for v in ref_meta["frame"].shape)
    cy, cx, zero_meta = locate_zero(ref_meta["frame"], cfg)
    probe_images = []
    for row in probe_rows:
        probe_images.append(load_frame_npz(period_dir / row["frame_file"])["frame"])
    try:
        geometry = locate_orders(probe_images, (cy, cx), cfg)
        orders = prepare_orders(geometry, shape, cfg)
    except ValueError as exc:
        result["geometry"] = {"ok": False, "reason": str(exc),
                              "zero": zero_meta,
                              "zero_center_row_col": [cy, cx]}
        result["timing_s"] = time.perf_counter() - t_start
        return result
    geometry["ok"] = True
    geometry["zero_estimation"] = zero_meta
    geometry["probe_frame_labels"] = [r.get("label") for r in probe_rows]
    geometry["roi_definition"] = {
        "aperture_r_small_px": cfg.aperture_r_small,
        "aperture_r_large_px": cfg.aperture_r_large,
        "aperture_r_total_px": cfg.aperture_r_total,
        "bg_ring_small_px": list(cfg.bg_ring_small),
        "bg_ring_large_px": list(cfg.bg_ring_large),
        "profile_max_r_px": cfg.profile_max_r,
        "profile_step_px": cfg.profile_step,
        "marginal_half_px": cfg.marginal_half,
        "marginal_step_px": cfg.marginal_step,
        "marginal_rule": ("row/col 边缘轮廓限制在固定 r<=aperture_r_total 圆盘内，"
                          "并按 net_total 归一化"),
        "bg_pedestal_rule": ("每帧唯一背景 = 全幅远场背景中位数（屏蔽 ±global_bg_order_k "
                             "级圆盘，stride 采样）；孔径积分与轮廓共用同一扣除量；"
                             "局部环仅作 *_altbg 敏感性对照"),
        "patch_half_px": cfg.patch_half,
        "frozen_before_measurement": True,
        "tuned_on_results": False,
    }
    result["geometry"] = geometry
    result["image_shape"] = list(shape)

    # 参考 flat / 漂移 flat：只测 0 级（无光栅，±1 级无物理意义）
    flat_stats: dict[str, Any] = {}
    ref_stats = measure_frame(ref_meta["frame"], orders, geometry, cfg,
                              names=("zero",))
    flat_stats["reference"] = {
        "label": ref_rows[0].get("label"),
        "timestamp_utc": ref_rows[0].get("timestamp_utc"),
        "zero": ref_stats["orders"]["zero"],
        "global_bg": ref_stats["global_bg"],
    }
    drift_rows = [r for r in rows if r.get("group") == "drift"]
    drift_entries = []
    for row in drift_rows:
        drift_meta = load_frame_npz(period_dir / row["frame_file"])
        drift_stats = measure_frame(drift_meta["frame"], orders, geometry, cfg,
                                    names=("zero",))
        drift_entries.append({
            "label": row.get("label"),
            "timestamp_utc": row.get("timestamp_utc"),
            "zero_net_small": drift_stats["orders"]["zero"]["net_small"],
            "zero_net_total": drift_stats["orders"]["zero"]["net_total"],
            "zero_bg_ring_small": drift_stats["orders"]["zero"]["bg_ring_small"],
            "global_bg": drift_stats["global_bg"],
        })
    flat_stats["drift"] = drift_entries
    ref_total = ref_stats["orders"]["zero"]["net_total"]
    if drift_entries and abs(ref_total) > 1e-9:
        ratios = [e["zero_net_total"] / ref_total for e in drift_entries]
        flat_stats["drift_over_reference_net_total"] = _median_iqr(np.array(ratios))
    else:
        flat_stats["drift_over_reference_net_total"] = None
    result["flats"] = flat_stats

    # 流式逐帧测光
    sweep_rows = [r for r in rows if r.get("kind") == "binary"
                  and r.get("group") == "sweep"]
    result["n_expected_analysis_frames"] = len(sweep_rows)
    per_gray: dict[int, list[dict[str, Any]]] = {}
    frame_hashes: dict[str, str] = {}
    checks_acc: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    hash_policy = cfg.hash_frames
    gray_order = [int(g) for g in plan.get("binary_sweep", [])]
    hash_targets = set()
    if hash_policy == "all":
        hash_targets = {str(r["frame_file"]) for r in sweep_rows}
    elif hash_policy == "sample":
        for gray in gray_order:
            block = [r for r in sweep_rows if int(r.get("gray", -1)) == gray]
            for take in ([block[0], block[-1]] if block else []):
                hash_targets.add(str(take["frame_file"]))

    for position, row in enumerate(sweep_rows):
        if len(sweep_rows) > 50 and position % 100 == 0:
            print(f"[period{period_px}] {position}/{len(sweep_rows)} 帧",
                  file=sys.stderr, flush=True)
        rel = str(row.get("frame_file"))
        path = period_dir / rel
        if hash_policy != "none" and rel in hash_targets:
            frame_hashes[rel] = sha256_file(path)
        try:
            meta = load_frame_npz(path)
        except Exception as exc:
            failures.append({"index": row.get("index"), "label": row.get("label"),
                             "frame_file": rel, "reason": f"load_error: {exc}"})
            continue
        frame = meta["frame"]
        tol = 1e-9
        _frame_check(checks_acc, "frame_shape",
                     list(frame.shape) == list(row.get("frame_shape", [])),
                     {"label": row.get("label"), "npz": list(frame.shape),
                      "row": row.get("frame_shape")})
        _frame_check(checks_acc, "npz_scalars",
                     int(meta.get("gray", -1)) == int(row.get("gray", -2))
                     and abs(float(meta.get("period_px", -1.0))
                             - float(row.get("period_px", -2.0))) < tol
                     and abs(float(meta.get("exposure_ms", -1.0))
                             - float(row.get("exposure_ms", -2.0))) < tol
                     and str(meta.get("kind", "")) == str(row.get("kind"))
                     and str(meta.get("timestamp_utc", ""))
                     == str(row.get("timestamp_utc")),
                     {"label": row.get("label"),
                      "npz_gray": meta.get("gray"), "row_gray": row.get("gray"),
                      "npz_period": meta.get("period_px"),
                      "row_period": row.get("period_px")})
        meta_json: dict[str, Any] = {}
        try:
            meta_json = json.loads(str(meta.get("meta_json", "{}")))
        except (TypeError, ValueError):
            meta_json = {}
        _frame_check(checks_acc, "meta_json",
                     int(meta_json.get("index", -1)) == int(row.get("index", -2))
                     and str(meta_json.get("label")) == str(row.get("label")),
                     {"label": row.get("label"), "npz_index": meta_json.get("index")})
        rec_max = float(frame.max())
        _frame_check(checks_acc, "peak_recomputed",
                     abs(rec_max - float(row.get("peak", -1.0))) <= 0.5,
                     {"label": row.get("label"), "npz_max": rec_max,
                      "row_peak": row.get("peak")})
        threshold = float(row.get("saturation_threshold", 3685.5))
        rec_sat = int((frame >= threshold).sum())
        _frame_check(checks_acc, "saturation_recomputed",
                     rec_sat == int(row.get("saturated_pixels", -1)),
                     {"label": row.get("label"), "npz_saturated": rec_sat,
                      "row_saturated": row.get("saturated_pixels")})
        _frame_check(checks_acc, "panel_gray_max",
                     int(row.get("panel_gray_max", -1))
                     == (int(row.get("gray", -1)) if row.get("kind") == "binary"
                         else 0),
                     {"label": row.get("label"),
                      "panel_gray_max": row.get("panel_gray_max"),
                      "gray": row.get("gray")})

        measured = measure_frame(frame, orders, geometry, cfg)
        measured.update({
            "row_index": int(row.get("index", -1)),
            "label": row.get("label"),
            "timestamp_utc": row.get("timestamp_utc"),
            "monotonic_ns": int(row.get("monotonic_ns", 0)),
        })
        per_gray.setdefault(int(row.get("gray", -1)), []).append(measured)
        result["n_analyzed"] += 1

    for gray in gray_order:
        records = per_gray.get(int(gray), [])
        if records:
            result["levels"].append(aggregate_level(int(gray), records, cfg))
    result["frames_failed"] = failures
    result["frame_checks"] = _frame_checks_ok(checks_acc)
    result["frame_sha256"] = {
        "policy": hash_policy,
        "n_hashed": len(frame_hashes),
        "sha256": frame_hashes,
    }
    result["timing_s"] = time.perf_counter() - t_start
    result["analysis_complete"] = (
        len(failures) == 0
        and result["n_analyzed"] == len(sweep_rows)
        and len(result["levels"]) == len(gray_order)
        and result["frame_checks"]["ok"]
    )
    return result


# ---------------------------------------------------------------------------
# 两段描述性比较（block bootstrap，灰度块为单位；不输出 verdict）
# ---------------------------------------------------------------------------
def _block_bootstrap(values: np.ndarray, cfg: Cfg) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    if n == 0:
        return {"median": None, "ci95": [None, None], "n_levels": 0,
                "n_blocks": 0, "block_levels": cfg.block_levels,
                "bootstrap": 0}
    blocks = [arr[i:i + cfg.block_levels] for i in range(0, n, cfg.block_levels)]
    rng = np.random.default_rng(cfg.seed)
    meds = np.empty(cfg.bootstrap, dtype=float)
    for i in range(cfg.bootstrap):
        pick = rng.integers(0, len(blocks), len(blocks))
        sample = np.concatenate([blocks[j] for j in pick])
        meds[i] = float(np.nanmedian(sample))
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return {
        "median": float(np.nanmedian(arr)),
        "ci95": [float(lo), float(hi)],
        "n_levels": n,
        "n_blocks": len(blocks),
        "block_levels": cfg.block_levels,
        "bootstrap": cfg.bootstrap,
        "seed": cfg.seed,
        "unit": "consecutive gray blocks (frames never split at random)",
    }


def _levels_by_gray(levels: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(level["gray"]): level for level in levels}


def compare_periods(results: dict[str, dict[str, Any]], cfg: Cfg) -> dict[str, Any]:
    """period32 vs period48 的固定 ROI 描述量差异（仅描述，不归因机制）。"""
    keys = [k for k in ("32", "48") if k in results]
    if len(keys) != 2:
        return {"available": False, "reason": "需要 period32 与 period48 两段"}
    a, b = results["32"], results["48"]
    la, lb = _levels_by_gray(a.get("levels", [])), _levels_by_gray(b.get("levels", []))
    common = sorted(set(la) & set(lb))
    if len(common) < 2:
        return {"available": False, "reason": "共同灰度太少",
                "n_common_gray": len(common), "verdict_emitted": False}
    out: dict[str, Any] = {
        "available": True,
        "n_common_gray": len(common),
        "period_windows": {
            "32": [a.get("levels")[0].get("t_start_utc") if a.get("levels") else None,
                   a.get("levels")[-1].get("t_end_utc") if a.get("levels") else None],
            "48": [b.get("levels")[0].get("t_start_utc") if b.get("levels") else None,
                   b.get("levels")[-1].get("t_end_utc") if b.get("levels") else None],
        },
        "metrics": {},
        "descriptive_only": True,
        "verdict_emitted": False,
        "confound": "period 与采集时段/慢漂移完全共线：两段各只有一个顺序 sweep。",
    }
    for name in _ORDER_NAMES:
        for metric, getter in (
            ("capture_small", lambda lv, n=name: lv["orders"][n]["capture_small"]["median"]),
            ("capture_large", lambda lv, n=name: lv["orders"][n]["capture_large"]["median"]),
            ("rms_radius_px", lambda lv, n=name: lv["orders"][n]["rms_radius_px"]["median"]),
            ("r50_px", lambda lv, n=name: lv["orders"][n]["r50_px"]["median"]),
        ):
            va = np.array([getter(la[g]) for g in common], dtype=float)
            vb = np.array([getter(lb[g]) for g in common], dtype=float)
            diff = va - vb
            valid = np.isfinite(diff)
            if not valid.any():
                out["metrics"][f"{name}.{metric}"] = {"available": False}
                continue
            out["metrics"][f"{name}.{metric}"] = {
                "period32": _block_bootstrap(va[valid], cfg),
                "period48": _block_bootstrap(vb[valid], cfg),
                "difference_32_minus_48": _block_bootstrap(diff[valid], cfg),
                "fraction_levels_32_gt_48": float(np.mean(diff[valid] > 0)),
            }
    # 归一化径向轮廓的 L1 距离（逐灰度；0..2）
    for name in _ORDER_NAMES:
        dist = []
        for gray in common:
            pa = np.asarray(la[gray]["orders"][name]["radial_energy_fraction"],
                            dtype=float)
            pb = np.asarray(lb[gray]["orders"][name]["radial_energy_fraction"],
                            dtype=float)
            if pa.shape == pb.shape and np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
                dist.append(float(np.abs(pa - pb).sum()))
        out["metrics"][f"{name}.profile_l1_32_vs_48"] = (
            _block_bootstrap(np.array(dist), cfg) if dist else {"available": False})
    out["notes"] = [
        "差异只描述固定 ROI 下的形状/捕获比例变化，不能归因于 PSF、像素模糊、"
        "静态分量或串扰。",
        "block bootstrap 以连续灰度块为单位；20 帧/灰度仅作为块内重复测量，"
        "未按帧随机拆分。",
    ]
    return out


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def stage_declaration() -> dict[str, Any]:
    """显式声明本脚本只是第一阶段，并拒绝输出任何算法收益 verdict。"""
    return {
        "stage": STAGE,
        "phase": 1,
        "title": "清单核验 + 固定 ROI 描述性测量",
        "algorithm_benefit_verdict": None,
        "verdict_emitted": False,
        "verdicts_withheld": [
            "algorithm_benefit",
            "mechanism_attribution",
            "lut_change",
            "two_pi_gray_update",
            "crosstalk_calibration",
            "period_function_fit",
        ],
        "m0_m1": {
            "implemented": False,
            "blocking_reasons": [
                "period 与采集时段/慢漂移完全共线：period32、period48 各只有一个"
                "顺序 sweep，无交叉设计，period 与 time-block 不可独立解混。",
                "两个周期点不足以约束任何周期函数；M1 的 period 相关宽度/低维轮廓项"
                "会与共享项退化。",
                "flat 是照明下的零灰度帧（非遮光暗场），无法把加性背景与静态分量"
                "分开；光路记录缺相机 gain / pixel pitch 等常数。",
                "本脚本未实现 M0/M1 的共享轮廓/乘性漂移/加性静态项联合拟合，"
                "因此不能报告留块预测改善。",
            ],
            "would_require": [
                "period 与 time-block 交叉或重复采集（每周期至少两个独立时段）。",
                "遮光暗场 + 稳定的参考功率监测，以分离加性背景与乘性漂移。",
                "预先冻结的 M0/M1 参数化与按连续时间/灰度块的留块验证方案。",
            ],
        },
        "what_this_report_can_say": [
            "run4 两个 period 段的 frame manifest / NPZ 标量 / 曝光 / 饱和 / 采集顺序"
            "是否与 plan 和 summary 一致。",
            "在预注册固定 ROI 下的归一化径向轮廓、孔径捕获比例、局部/全局背景与"
            "块内漂移的描述统计。",
            "period32 与 period48 在这些描述量上的差异及其灰度块 bootstrap 区间。",
        ],
        "what_it_cannot_say": [
            "任何算法收益 / 校正效果 / 模型优劣的结论。",
            "把轮廓宽度解释为绝对 PSF 或像素模糊，把低频项解释为静态偏置或串扰。",
            "据此更新 LUT、冻结 2π 灰度或宣称串扰已定标。",
        ],
    }


def build_report(args: argparse.Namespace, cfg: Cfg,
                 results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    return {
        "schema": SCHEMA,
        "stage": STAGE,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": {
            "path": str(script_path),
            "relative_path": str(script_path.relative_to(ROOT))
            if script_path.is_relative_to(ROOT) else str(script_path),
            "sha256": sha256_file(script_path),
            "argv": [str(x) for x in sys.argv],
            "cwd": str(Path.cwd()),
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "stage_declaration": stage_declaration(),
        "config": {
            "formal_exposure_ms": cfg.formal_exposure_ms,
            "aperture_r_small_px": cfg.aperture_r_small,
            "aperture_r_large_px": cfg.aperture_r_large,
            "aperture_r_total_px": cfg.aperture_r_total,
            "bg_ring_small_px": list(cfg.bg_ring_small),
            "bg_ring_large_px": list(cfg.bg_ring_large),
            "profile_max_r_px": cfg.profile_max_r,
            "profile_step_px": cfg.profile_step,
            "marginal_half_px": cfg.marginal_half,
            "marginal_step_px": cfg.marginal_step,
            "zero_blank_r_px": cfg.zero_blank_r,
            "border_px": cfg.border,
            "smooth_sigma_px": cfg.smooth_sigma,
            "probe_grays": list(cfg.probe_grays),
            "frames_per_level": cfg.frames_per_level,
            "global_bg_stride": cfg.global_bg_stride,
            "global_bg_exclude_r_px": cfg.global_bg_exclude_r,
            "global_bg_order_k": cfg.global_bg_order_k,
            "block_levels": cfg.block_levels,
            "bootstrap": cfg.bootstrap,
            "seed": cfg.seed,
            "hash_frames": cfg.hash_frames,
            "npz_loading": "np.load(..., allow_pickle=False)",
            "frozen_roi": True,
            "roi_tuned_on_results": False,
            "frames_dropped": False,
            "gray_levels_selected": False,
        },
        "inputs": {
            "run_root": str(Path(args.run_root).resolve()),
            "periods": [int(p) for p in args.periods],
        },
        "periods": results,
        "period_comparison": compare_periods(results, cfg),
        "limitations": [
            "period 与采集时段/漂移共线；仅有 2 个周期点。",
            "flat 为照明零灰度帧，不是遮光暗场。",
            "光路记录不完整（缺相机 gain、pixel pitch 等常数）。",
            "主背景用全幅远场中位数（单一台阶），无法消除视场梯度/晕影；"
            "局部环受该级翼部与相邻级次抬高，只作 alt-bg 敏感性对照。",
            "20 帧/灰度是块内重复测量，不是 20 个独立硬件重复。",
        ],
        "not_conclusions": [
            "不把 period 差异归因于任何物理机制。",
            "不更新或建议 LUT / 2π 灰度。",
            "不给出串扰定标结论。",
            "不宣称任何算法收益。",
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_periods(text: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in text.replace(";", ",").split(",") if part)
    if not values:
        raise argparse.ArgumentTypeError("--periods 不能为空")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Issue #238 run4 离线空间诊断（第一阶段：清单核验 + 固定 ROI 描述量）")
    parser.add_argument("--self-test", action="store_true",
                        help="只用临时合成数组跑核心函数自测，不扫描真实数据")
    parser.add_argument("--run-root", default=str(
        ROOT / "data" / "experiments" / "issue238_binary_phase_20261119"
        / "run4_period_sweep"))
    parser.add_argument("--periods", type=_parse_periods, default=(32, 48))
    parser.add_argument("--out", default=None,
                        help="JSON 输出路径（必填；拒绝写入原始 run/输入子树）")
    parser.add_argument("--formal-exposure-ms", type=float,
                        default=Cfg.formal_exposure_ms)
    parser.add_argument("--aperture-r-small", type=float, default=Cfg.aperture_r_small)
    parser.add_argument("--aperture-r-large", type=float, default=Cfg.aperture_r_large)
    parser.add_argument("--aperture-r-total", type=float, default=Cfg.aperture_r_total)
    parser.add_argument("--hash-frames", choices=("all", "sample", "none"),
                        default=Cfg.hash_frames)
    parser.add_argument("--bootstrap", type=int, default=Cfg.bootstrap)
    parser.add_argument("--block-levels", type=int, default=Cfg.block_levels)
    parser.add_argument("--seed", type=int, default=Cfg.seed)
    return parser.parse_args(argv)


def _cfg_from_args(args: argparse.Namespace) -> Cfg:
    return Cfg(
        formal_exposure_ms=float(args.formal_exposure_ms),
        aperture_r_small=float(args.aperture_r_small),
        aperture_r_large=float(args.aperture_r_large),
        aperture_r_total=float(args.aperture_r_total),
        hash_frames=str(args.hash_frames),
        bootstrap=int(args.bootstrap),
        block_levels=int(args.block_levels),
        seed=int(args.seed),
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    text = json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False,
                      allow_nan=False)
    tmp.write_text(text + "\n", encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# 自测（临时合成数组；绝不扫描真实数据）
# ---------------------------------------------------------------------------
def _synthetic_cfg() -> Cfg:
    return Cfg(
        formal_exposure_ms=0.08,
        aperture_r_small=10.0,
        aperture_r_large=20.0,
        aperture_r_total=40.0,
        bg_ring_small=(14.0, 22.0),
        bg_ring_large=(24.0, 34.0),
        profile_max_r=40.0,
        profile_step=2.0,
        marginal_half=32.0,
        marginal_step=2.0,
        zero_blank_r=45.0,
        border=20,
        smooth_sigma=2.0,
        probe_grays=(400,),
        frames_per_level=3,
        global_bg_stride=2,
        global_bg_exclude_r=12.0,
        global_bg_order_k=2,
        block_levels=1,
        bootstrap=50,
        seed=7,
    )


def _synthetic_frame(shape: tuple[int, int], zero: tuple[float, float],
                     spacing: float, gray: int, rng: np.random.Generator) -> np.ndarray:
    """合成 0 级 + 沿行方向 ±spacing 的 ±1 级 + 常数背景 + 噪声。"""
    gy, gx = np.mgrid[0:shape[0], 0:shape[1]]
    cy, cx = zero
    img = np.full(shape, 20.0)
    img += 800.0 * np.exp(-((gy - cy) ** 2 + (gx - cx) ** 2) / (2 * 6.0 ** 2))
    amp = 60.0 + 120.0 * math.sin(math.pi * gray / 1000.0) ** 2
    for sign in (-1, 1):
        cy1 = cy + sign * spacing
        img += amp * np.exp(-((gy - cy1) ** 2 + (gx - cx) ** 2) / (2 * 3.0 ** 2))
    img += rng.normal(0.0, 2.0, size=shape)
    return np.clip(np.rint(img), 0, 4095).astype(np.uint16)


def _synthetic_run(root: Path, cfg: Cfg) -> tuple[Path, dict[str, Any]]:
    """构造一个最小但自洽的 run 目录（plan/summary/frames.jsonl/NPZ）。"""
    period_px = 32
    period_dir = root / f"period{period_px}"
    (period_dir / "frames").mkdir(parents=True)
    shape = (400, 520)
    zero = (200.0, 300.0)
    spacing = 80.0
    grays = [0, 400, 1000]
    n_frames = cfg.frames_per_level
    rng = np.random.default_rng(cfg.seed)

    patterns: list[dict[str, Any]] = [{
        "index": 0, "label": "0000_flat", "kind": "flat", "group": "reference",
        "gray": 0, "period_px": 0, "frames": 1, "params": {},
        "panel_gray_max": 0, "clamp_warning": None,
        "panel_gray_sha256": "0" * 64, "warnings": [],
    }]
    for k, gray in enumerate(grays):
        patterns.append({
            "index": 1 + k, "label": f"{1 + k:04d}_binary", "kind": "binary",
            "group": "sweep", "gray": gray, "period_px": period_px, "duty": 0.5,
            "axis": 1, "frames": n_frames,
            "params": {"note": "synthetic", "duty": 0.5},
            "panel_gray_max": gray, "clamp_warning": None,
            "panel_gray_sha256": f"{k:064x}", "warnings": [],
        })
    patterns.append({
        "index": 1 + len(grays), "label": f"{1 + len(grays):04d}_flat",
        "kind": "flat", "group": "drift", "gray": 0, "period_px": 0,
        "frames": 2, "params": {"note": "synthetic drift"},
        "panel_gray_max": 0, "clamp_warning": None,
        "panel_gray_sha256": "0" * 64, "warnings": [],
    })

    plan = {
        "schema": "issue238-gray-phase-probes/v2-binary",
        "mode": "binary",
        "task": "self-test",
        "g_ref": 1023,
        "two_pi_gray_assumed_issue238": 993,
        "two_pi_gray_true": "self-test",
        "binary_period_px": period_px,
        "binary_duty": 0.5,
        "binary_sweep": grays,
        "n_sweep": len(grays),
        "frames_per_level": n_frames,
        "dense_region": {"lo": 864, "hi": 1023, "step": 8},
        "exposure_bracket_ms": [0.08],
        "n_patterns": len(patterns),
        "n_frames_expected": sum(int(p["frames"]) for p in patterns),
        "native_ccd_shape": list(shape),
        "panel_shape_default": [1200, 1920],
        "bit_depth_required": 12,
        "full_scale_required": 4095,
        "timestamp_utc": "2026-11-20T00:00:00+00:00",
        "patterns": patterns,
    }
    (period_dir / "plan.json").write_text(
        json.dumps(plan, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    mono = 1_000_000_000
    t0 = 1_766_000_000.0

    def add_row(index: int, label: str, kind: str, group: str, gray: int,
                period: int, pattern: dict[str, Any]) -> None:
        nonlocal mono, t0
        frame = _synthetic_frame(shape, zero, spacing, gray, rng)
        rel = f"frames/{index:04d}_{label}.npz"
        np.savez(period_dir / rel, frame=frame, kind=kind, gray=gray,
                 period_px=float(period), exposure_ms=cfg.formal_exposure_ms,
                 timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%S+00:00",
                                             time.gmtime(t0)),
                 meta_json=json.dumps({"index": index, "label": label,
                                       "gray": gray}))
        rows.append({
            "index": index, "label": label, "kind": kind, "group": group,
            "gray": gray, "period_px": period, "pattern_params": {"duty": 0.5},
            "panel_gray_max": gray if kind == "binary" else 0,
            "panel_gray_sha256": pattern["panel_gray_sha256"],
            "warnings": [], "exposure_ms": cfg.formal_exposure_ms,
            "frame_file": rel, "frame_shape": list(shape), "frame_dtype": "uint16",
            "full_scale": 4095.0,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00",
                                           time.gmtime(t0)),
            "monotonic_ns": mono, "peak": float(frame.max()),
            "saturated_pixels": 0, "saturation_threshold": 3685.5,
        })
        mono += 1_400_000_000
        t0 += 1.4

    index = 0
    add_row(index, "0000_flat_r00", "flat", "reference", 0, 0, patterns[0])
    index += 1
    for k, gray in enumerate(grays):
        pattern = patterns[1 + k]
        for rep in range(n_frames):
            add_row(index, f"{pattern['index']:04d}_binary_r{rep:02d}", "binary",
                    "sweep", gray, period_px, pattern)
            index += 1
    drift = patterns[-1]
    for rep in range(2):
        add_row(index, f"{drift['index']:04d}_flat_r{rep:02d}", "flat", "drift",
                0, 0, drift)
        index += 1
    with (period_dir / "frames.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    summary = {
        "schema": plan["schema"], "mode": "binary", "status": "ok",
        "frame_count": len(rows), "n_patterns": len(patterns),
        "exposure_bracket": {
            "n_passes": [1], "n_steps": 1, "n_usable": 1,
            "chosen_readback_ms": cfg.formal_exposure_ms,
            "rows": [{"pass": 1, "requested_ms": cfg.formal_exposure_ms,
                      "readback_ms": cfg.formal_exposure_ms,
                      "peak": float(rows[0]["peak"]), "saturated_pixels": 0,
                      "usable": True}],
            "analysis_policy": "only unsaturated bracket steps are usable",
        },
        "bit_depth_assertion": {"status": "ok", "requested": 12, "readback": 12},
        "g_ref": 1023, "two_pi_gray_assumed_issue238": 993,
        "device_two_pi_gray_readback": 993,
        "two_pi_gray_true": "self-test", "saturated_frames": 0,
        "clamp_warnings": [], "elapsed_s": 1.0,
        "timestamp_utc": "2026-11-20T00:00:00+00:00",
        "monotonic_ns": mono,
    }
    (period_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    (period_dir / "optical_path_record.json").write_text(
        json.dumps({"schema": "ao-optical-path-record/v1", "run_id": "self-test"}),
        encoding="utf-8")
    (period_dir / "device_log.json").write_text(
        json.dumps({"schema": "self-test"}), encoding="utf-8")
    (period_dir / "release.json").write_text(
        json.dumps({"slm_flat_gray0": "ok", "error": None}), encoding="utf-8")
    return period_dir, {"rows": rows, "plan": plan, "summary": summary}


def run_self_test() -> int:
    """核心函数与端到端（临时合成目录）自测，不读取任何真实数据。"""
    failures: list[str] = []
    cfg = _synthetic_cfg()

    def check(name: str, condition: bool, detail: Any = None) -> None:
        if condition:
            print(f"  ok   {name}")
        else:
            print(f"  FAIL {name} :: {detail}")
            failures.append(name)

    print("[self-test] 1/6 合成帧几何与测光")
    shape = (400, 520)
    truth_zero = (200.0, 300.0)
    frame = _synthetic_frame(shape, truth_zero, 80.0, 400,
                             np.random.default_rng(11))
    cy, cx, zero_meta = locate_zero(frame, cfg)
    check("zero center < 2 px", math.hypot(cy - truth_zero[0], cx - truth_zero[1]) < 2.0,
          (cy, cx, zero_meta))
    geometry = locate_orders([frame, frame], (cy, cx), cfg)
    check("order spacing ~80 px", abs(geometry["spacing_px"] - 80.0) < 1.5,
          geometry["spacing_px"])
    orders = prepare_orders(geometry, shape, cfg)
    measured = measure_frame(frame, orders, geometry, cfg)
    zero_meas = measured["orders"]["zero"]
    check("capture_small in (0.5, 1.0]", 0.5 < zero_meas["capture_small"] <= 1.0,
          zero_meas["capture_small"])
    check("capture_small <= capture_large",
          zero_meas["capture_small"] <= zero_meas["capture_large"] + 1e-12,
          (zero_meas["capture_small"], zero_meas["capture_large"]))
    frac = np.asarray(zero_meas["radial_energy_fraction"], dtype=float)
    check("radial fractions finite & sum ~1",
          bool(np.all(np.isfinite(frac))) and abs(float(frac.sum()) - 1.0) < 1e-6,
          float(frac.sum()))
    row_marg = np.asarray(zero_meas["row_marginal"], dtype=float)
    col_marg = np.asarray(zero_meas["col_marginal"], dtype=float)
    # 自测 cfg 的 marginal_half(32) < aperture_r_total(40)，角落像素按设计被
    # |dx|,|dy|<=marginal_half 截掉，故允许 <1 的少量损失；正式口径
    # marginal_half=128 >= aperture_r_total=120，和严格为 1。
    check("marginal profiles limited to aperture disk & sum ~1",
          len(row_marg) == cfg.n_marginal_bins
          and 0.999 <= float(row_marg.sum()) <= 1.0 + 1e-9
          and 0.999 <= float(col_marg.sum()) <= 1.0 + 1e-9,
          (len(row_marg), float(row_marg.sum()), float(col_marg.sum())))
    check("rms radius finite and < 40",
          math.isfinite(zero_meas["rms_radius_px"])
          and zero_meas["rms_radius_px"] < 40.0,
          zero_meas["rms_radius_px"])
    # r50/r90 由截断非负的累计能量得到，必须单调且落在固定孔径内；未截断时
    # 累计曲线非单调，interp 曾给出 93 px > 孔径这种自相矛盾的值。
    check("r50 <= r90 <= aperture radius",
          math.isfinite(zero_meas["r50_px"])
          and math.isfinite(zero_meas["r90_px"])
          and 0.0 < zero_meas["r50_px"] <= zero_meas["r90_px"]
          <= float(cfg.aperture_r_total) + 1e-9,
          (zero_meas["r50_px"], zero_meas["r90_px"], cfg.aperture_r_total))

    print("[self-test] 2/6 清单核验（正例与篡改）")
    with tempfile.TemporaryDirectory(prefix="issue238_selftest_") as tmp:
        root = Path(tmp)
        period_dir, fixture = _synthetic_run(root, cfg)
        good = verify_manifest(fixture["plan"], fixture["summary"],
                               fixture["rows"], 32, cfg, period_dir)
        check("manifest positive case ok", good["ok"], good["failed_ids"])

        tampered = [dict(r) for r in fixture["rows"]]
        for row in tampered:
            if row["kind"] == "binary":
                row["exposure_ms"] = 0.16
        bad = verify_manifest(fixture["plan"], fixture["summary"], tampered, 32,
                              cfg, period_dir)
        check("tampered exposure detected", "exposure_formal" in bad["failed_ids"],
              bad["failed_ids"])

        swapped = [dict(r) for r in fixture["rows"]]
        binary = [r for r in swapped if r["kind"] == "binary"]
        # 跨灰度块交换（块内交换不会改变序列，属预期）
        binary[0]["gray"], binary[3]["gray"] = binary[3]["gray"], binary[0]["gray"]
        bad_seq = verify_manifest(fixture["plan"], fixture["summary"], swapped, 32,
                                  cfg, period_dir)
        check("swapped acquisition order detected",
              "sweep_gray_sequence" in bad_seq["failed_ids"] or
              "row_label_mapping" in bad_seq["failed_ids"],
              bad_seq["failed_ids"])

        bad_len = verify_manifest(fixture["plan"], fixture["summary"],
                                  fixture["rows"][:-1], 32, cfg, period_dir)
        check("row-count mismatch detected", "frame_count" in bad_len["failed_ids"],
              bad_len["failed_ids"])

        print("[self-test] 3/6 端到端（临时目录，流式）")
        result = analyze_period(period_dir, 32, cfg)
        check("period analysis complete", bool(result.get("analysis_complete")),
              {"failed": result.get("frames_failed"),
               "checks": result.get("frame_checks", {}).get("failed_keys"),
               "geometry": result.get("geometry", {}).get("reason")})
        check("3 gray levels aggregated", len(result["levels"]) == 3,
              len(result["levels"]))
        check("9 sweep frames analyzed", result["n_analyzed"] == 9,
              result["n_analyzed"])
        check("all frame checks pass",
              bool(result["frame_checks"]["ok"]),
              result["frame_checks"].get("failed_keys"))

        print("[self-test] 4/6 报告与 verdict 抑制")
        report = build_report(
            argparse.Namespace(run_root=str(root), periods=(32,)),
            cfg, {"32": result})
        check("stage is phase1_only", report["stage"] == STAGE, report["stage"])
        check("algorithm_benefit_verdict is null",
              report["stage_declaration"]["algorithm_benefit_verdict"] is None)
        check("verdict not emitted",
              report["stage_declaration"]["verdict_emitted"] is False)
        check("M0/M1 marked not implemented",
              report["stage_declaration"]["m0_m1"]["implemented"] is False)
        encoded = json.dumps(to_jsonable(report), allow_nan=False)
        check("report JSON-serialisable (no NaN)", len(encoded) > 1000, len(encoded))

        print("[self-test] 5/6 --out 守卫")
        for bad_out in (period_dir / "report.json",
                        ROOT / "data" / "tmp" / "report.json",
                        root / "not_json.txt"):
            try:
                guard_out(bad_out, protected=[root, ROOT / "data"])
                check(f"guard rejects {bad_out}", False, "not refused")
            except OutRefused:
                check(f"guard rejects {str(bad_out)[-24:]}", True)
        safe = guard_out(Path(tempfile.gettempdir()) / "issue238_selftest_out.json",
                         protected=[root, ROOT / "data"])
        check("guard allows temp output",
              safe == (Path(tempfile.gettempdir())
                       / "issue238_selftest_out.json").resolve(), safe)

        print("[self-test] 6/6 NPZ 安全加载")
        object_path = root / "object.npz"
        np.savez(object_path, frame=np.array([{"a": 1}], dtype=object))
        try:
            load_frame_npz(object_path)
            check("object npz refused", False, "loaded with allow_pickle=False?")
        except ValueError as exc:
            check("object npz refused", "allow_pickle" in str(exc)
                  or "dtype" in str(exc) or "unsupported" in str(exc), str(exc))
        good_path = period_dir / fixture["rows"][0]["frame_file"]
        loaded = load_frame_npz(good_path)
        check("uint16 npz loads", loaded["frame"].dtype == np.uint16,
              loaded["frame"].dtype)

    print()
    if failures:
        print(f"[self-test] FAILED: {len(failures)} 项 -> {failures}")
        return 1
    print("[self-test] PASSED：几何/测光/清单/端到端/守卫/NPZ 安全加载全部通过")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test()

    cfg = _cfg_from_args(args)
    run_root = Path(args.run_root).expanduser().resolve()
    if not run_root.is_dir():
        print(f"输入 run 目录不存在: {run_root}", file=sys.stderr)
        return 2
    period_dirs = {str(p): run_root / f"period{p}" for p in args.periods}
    missing = [str(d) for d in period_dirs.values() if not d.is_dir()]
    if missing:
        print(f"缺少 period 子目录: {missing}", file=sys.stderr)
        return 2

    protected: list[Path] = [run_root, ROOT / "data", *period_dirs.values()]
    if args.out is None:
        print("--out 必填：请显式指定 Temp 下的 JSON 路径", file=sys.stderr)
        return 2
    try:
        out_path = guard_out(args.out, protected=protected)
    except OutRefused as exc:
        print(str(exc), file=sys.stderr)
        return 2

    results: dict[str, dict[str, Any]] = {}
    for key, period_dir in period_dirs.items():
        print(f"[issue238] 分析 {period_dir}", file=sys.stderr)
        try:
            results[key] = analyze_period(period_dir, int(key), cfg)
        except Exception as exc:
            print(f"分析 {period_dir} 失败: {exc}", file=sys.stderr)
            return 2

    report = build_report(args, cfg, results)
    _write_json_atomic(out_path, report)
    print(f"[issue238] 报告已写: {out_path}")

    all_ok = True
    for key, result in results.items():
        integrity = result.get("integrity", {})
        complete = bool(result.get("analysis_complete"))
        if not integrity.get("ok") or not complete:
            all_ok = False
        print(f"  period{key}: n_analyzed={result.get('n_analyzed')}"
              f" levels={len(result.get('levels', []))}"
              f" manifest_ok={integrity.get('ok')}"
              f" failed_checks={integrity.get('failed_ids')}"
              f" complete={complete}")
    print("  stage=phase1_only, algorithm_benefit_verdict=null "
          "(M0/M1 未实现；见报告 stage_declaration)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
