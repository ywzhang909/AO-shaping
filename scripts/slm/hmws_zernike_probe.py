# -*- coding: utf-8 -*-
"""HMWS 五点 Zernike 硬件探针 (带离线 Sim2f 自检)。

每次对指定 Noll 模式收 5 帧焦面图 (``a/m/p/mm/pp``), 把探针偏置 **加进该模式的
Zernike 系数** ``a_inject + offset``, 其中 offset 依次为 ``0/-b/+b/-2b/+2b``。
焦面峰值经暗场扣除后送进 :mod:`ao_shaping.utils.image.zernike_probe` 的五点估计,
报告 recovered - inject 差分与多个 inject 下的斜率 (线性度), 不要求绝对零像差归零。

``--dry-run`` 在 64×64 的 Sim2f 数字孪生上走同一套"生成相位 → 收帧 → 取峰 → 估计 →
写产物"路径, 不导入任何 SLM/CCD SDK。真实采集由设备锁 + ExitStack 保护:
锁 → SLM → CCD 依次进入, LIFO 关闭, 锁持有到五帧组全部采完。

没有 ``--execute`` 之类的隐藏开关: 本脚本默认会碰硬件, 离线请显式加 ``--dry-run``。
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "src", ROOT / "libs", ROOT, Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ao_shaping.tools.kernels.bench_kernels import fresnel_panel
from ao_shaping.tools.slm.panel.exposure_adapt import (
    adapt_exposure,
    robust_peak,
)
from ao_shaping.utils.image.exposure_control import frame_noise_floor
from ao_shaping.utils.image.zernike_probe import (
    FRAME_ORDER,
    estimate_zernike_probe,
    probe_peak_intensity,
)
from ao_shaping.utils.wavefront.pattern_helper import PatternHelper
from ao_shaping.utils.wavefront.zernike_calc import noll_to_nm

__all__ = [
    "DEFAULT_BIT_DEPTH",
    "DRY_RUN_FULL_SCALE",
    "DRY_RUN_GRID",
    "PROBE_SELF_CHECK_BIAS",
    "PROBE_SELF_CHECK_EXPECTED",
    "PROBE_SELF_CHECK_INJECT",
    "PROBE_SELF_CHECK_NOLL",
    "PROBE_SELF_CHECK_TOLERANCE",
    "QUALITY_PEAK_HIGH",
    "QUALITY_PEAK_LOW",
    "SCHEMA",
    "assess_group_quality",
    "build_parser",
    "build_probe_phases",
    "compute_linearity",
    "dry_run_self_check",
    "main",
    "noll_to_mode",
    "probe_offsets",
    "read_camera_bit_depth",
    "rescale_dark_frame",
    "resolve_dark_frame",
    "verify_camera_bit_depth",
]

SCHEMA = "hmws-zernike-probe/v1"
DEFAULT_OUT = "data/hmws_zernike_probe"
DEFAULT_N_MAX = 6

#: 请求相机输出位深 (bit)。满量程与质量门/暗帧口径都建立在此值上, 开流后还要
#: 按设备回读值核对, 不允许请求 12 却静默落在 8-bit / 255。
DEFAULT_BIT_DEPTH = 12

#: 焦面峰值质量门限 (相对满量程)。曝光阶梯的目标/接受带与此一致。
EXPOSURE_TARGET = 0.45
EXPOSURE_ACCEPT = (0.25, 0.70)
QUALITY_PEAK_HIGH = 0.70
QUALITY_PEAK_LOW = 0.05

#: Sim2f 输出是 [0, ~0.72] 的 float 强度, 没有相机位深。显式选一个模拟满量程
#: 让默认五帧峰值落在 ~0.48 FS (既非欠曝也非饱和), 不按 dtype 猜。
DRY_RUN_GRID = 64
DRY_RUN_FULL_SCALE = 1.5
DRY_RUN_RADIUS = 28.0

#: 固定自检点: Noll 4 / inject 0.2 / bias 0.3 在 64×64 Sim2f 上应得 ≈0.195。
PROBE_SELF_CHECK_NOLL = 4
PROBE_SELF_CHECK_INJECT = 0.2
PROBE_SELF_CHECK_BIAS = 0.3
PROBE_SELF_CHECK_EXPECTED = 0.195
PROBE_SELF_CHECK_TOLERANCE = 0.05


def build_parser() -> argparse.ArgumentParser:
    """构造 CLI。``--noll`` / ``--inject-rad`` 均可一次给多个值。"""
    parser = argparse.ArgumentParser(
        description="HMWS 五点 Zernike 探针 (焦面峰值; 支持 --dry-run 离线自检)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--noll", type=int, nargs="+", default=[4], help="Noll 模式索引 (可多个)"
    )
    parser.add_argument(
        "--inject-rad",
        type=float,
        nargs="+",
        default=[0.0],
        help="注入的 Zernike 系数 (rad, 可多个)",
    )
    parser.add_argument(
        "--bias-rad", type=float, default=0.3, help="五点探针偏置 b (rad)"
    )
    parser.add_argument(
        "--radius", type=float, default=300.0, help="Zernike 孔径半径 (panel 像素)"
    )
    parser.add_argument(
        "--focal-mm", type=float, default=None, help="可选焦面补偿焦距 (mm); 默认关闭"
    )
    parser.add_argument(
        "--focus-json",
        type=Path,
        default=None,
        help="从该 JSON 的 best_focal_mm 读焦距 (与 --focal-mm 二选一)",
    )
    parser.add_argument("--slm-number", type=int, default=1, help="SLM 编号")
    parser.add_argument(
        "--slm-wavelength", type=int, default=1064, help="SLM 工作波长 (nm)"
    )
    parser.add_argument("--cam-type", default="daheng", help="相机后端类型")
    parser.add_argument("--cam-id", default=0, help="相机编号")
    parser.add_argument(
        "--bit-depth",
        type=int,
        default=DEFAULT_BIT_DEPTH,
        help="请求相机输出位深 (bit); 打开后按设备回读值校验",
    )
    parser.add_argument("--exposure-ms", type=float, default=0.05, help="请求曝光 (ms)")
    parser.add_argument(
        "--auto-exposure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="用曝光阶梯把峰值调到 0.25–0.70 满量程",
    )
    parser.add_argument(
        "--frames", type=int, default=3, help="每帧采集的样本帧数 (get_numpy_image)"
    )
    parser.add_argument(
        "--settle-s", type=float, default=0.5, help="每次改面板后的稳定等待 (s)"
    )
    parser.add_argument(
        "--subtract-dark",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="严格匹配当前曝光的暗帧并扣除 (无匹配则报告 false)",
    )
    parser.add_argument(
        "--wait-until-idle",
        type=float,
        default=600.0,
        help="等设备锁的最长等待 (s)",
    )
    parser.add_argument(
        "--out", type=Path, default=Path(DEFAULT_OUT), help="输出根目录"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Sim2f 数字孪生自检 (不碰任何设备)"
    )
    parser.add_argument(
        "--save-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="每个 (noll, inject) 保存五帧 .npz",
    )
    return parser


def _jsonable_params(args: argparse.Namespace) -> dict[str, Any]:
    """把 argparse 的值写成可 JSON 序列化的参数快照。"""
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def _git_short_sha() -> str | None:
    """当前检出的短 sha; 不是 git 仓库 / 没有 git 时返回 None。"""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() or None


def resolve_focal_mm(args: argparse.Namespace) -> float | None:
    """解析可选焦面补偿: 命令行优先, 其次 focus JSON 的 best_focal_mm。"""
    if args.focal_mm is not None:
        return float(args.focal_mm)
    if args.focus_json is None:
        return None
    path = Path(args.focus_json)
    if not path.is_file():
        raise FileNotFoundError(f"--focus-json 不存在: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("best_focal_mm")
    return None if value is None else float(value)


def noll_to_mode(noll: int) -> tuple[int, int]:
    """Noll → (n, m); 唯一真源是 :mod:`zernike_calc`, 本脚本不写查找表。"""
    return tuple(int(v) for v in noll_to_nm(int(noll)))


def probe_offsets(bias_rad: float) -> dict[str, float]:
    """五帧的探针偏置 (rad): ``a=0, m=-b, p=+b, mm=-2b, pp=+2b``。"""
    b = float(bias_rad)
    return {"a": 0.0, "m": -b, "p": b, "mm": -2.0 * b, "pp": 2.0 * b}


def build_probe_phases(
    helper: PatternHelper,
    noll: int,
    inject_rad: float,
    bias_rad: float,
    radius: float,
    n_max: int = DEFAULT_N_MAX,
    *,
    focal_phase: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """生成一组的五帧相位; 偏置进入模式系数, ``bias_rad`` 参数不传给 helper。"""
    nm = noll_to_mode(noll)
    offsets = probe_offsets(bias_rad)
    phases: dict[str, np.ndarray] = {}
    for key in FRAME_ORDER:
        coefficient = float(inject_rad) + offsets[key]
        phase = np.asarray(
            helper.generate_probe_phase(
                coefficients={nm: coefficient}, radius=float(radius), n_max=int(n_max)
            ),
            dtype=np.float64,
        )
        if focal_phase is not None:
            if focal_phase.shape != phase.shape:
                raise ValueError(
                    f"焦面补偿相位 {focal_phase.shape} 与图案 {phase.shape} 不一致"
                )
            phase = phase + focal_phase
        phases[key] = np.asarray(phase, dtype=np.float64)
    return phases


def _frame_stats(frame: np.ndarray, full_scale: float) -> dict[str, Any]:
    """一帧的峰值 / 满量程比例 / 饱和像素数 / 自身噪声底。"""
    arr = np.asarray(frame)
    peak = probe_peak_intensity(arr)
    noise = frame_noise_floor(arr)
    clipped = int(np.count_nonzero(arr >= float(full_scale)))
    return {
        "peak": float(peak),
        "peak_fraction": float(peak / float(full_scale)),
        "clipped_px": clipped,
        "noise_threshold_counts": float(noise.threshold_counts),
    }


def assess_group_quality(
    raw_frames: Mapping[str, np.ndarray],
    processed_frames: Mapping[str, np.ndarray],
    full_scale: float,
    *,
    exposure_converged: bool = True,
    valid: bool = True,
) -> dict[str, Any]:
    """纯函数质量门: 饱和 / 过曝 / 欠曝 / 曝光不收敛 / 估计退化。"""
    problems: list[str] = []
    per_key: dict[str, dict[str, Any]] = {}
    for key in FRAME_ORDER:
        raw = _frame_stats(raw_frames[key], full_scale)
        processed_peak = probe_peak_intensity(processed_frames[key])
        local: list[str] = []
        if raw["clipped_px"] > 0:
            local.append(f"{key}: clipped_px={raw['clipped_px']}")
        if raw["peak_fraction"] > QUALITY_PEAK_HIGH:
            local.append(
                f"{key}: peak={raw['peak_fraction']:.3f} FS > {QUALITY_PEAK_HIGH}"
            )
        if (
            raw["peak_fraction"] < QUALITY_PEAK_LOW
            or raw["peak"] <= raw["noise_threshold_counts"]
        ):
            local.append(
                f"{key}: 欠曝 (peak={raw['peak']:.3g}, "
                f"fraction={raw['peak_fraction']:.4g}, "
                f"noise_thr={raw['noise_threshold_counts']:.3g})"
            )
        if (
            processed_peak <= raw["noise_threshold_counts"]
            and raw["peak_fraction"] >= QUALITY_PEAK_LOW
        ):
            local.append(f"{key}: 扣暗后峰值 {processed_peak:.3g} 落入读出噪声")
        problems.extend(local)
        per_key[key] = {
            **raw,
            "processed_peak": float(processed_peak),
            "problems": local,
        }
    if not exposure_converged:
        problems.append("曝光阶梯未收敛到接受带")
    if not valid:
        problems.append("五点分母退化 / 估计无效")
    return {
        "qualified": not problems,
        "problems": problems,
        "per_frame": per_key,
    }


def compute_linearity(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """按 Noll 分组对 (inject, recovered) 做线性拟合; 不足两点则斜率留空。"""
    grouped: dict[int, list[tuple[float, float]]] = {}
    for record in records:
        if not record.get("valid"):
            continue
        recovered = record.get("recovered_rad")
        if recovered is None or not np.isfinite(float(recovered)):
            continue
        grouped.setdefault(int(record["noll"]), []).append(
            (float(record["inject_rad"]), float(recovered))
        )
    out: list[dict[str, Any]] = []
    for noll in sorted(grouped):
        points = sorted(grouped[noll])
        x = np.asarray([p[0] for p in points], dtype=np.float64)
        y = np.asarray([p[1] for p in points], dtype=np.float64)
        item: dict[str, Any] = {
            "noll": noll,
            "n_points": len(points),
            "slope": None,
            "intercept": None,
            "r2": None,
        }
        if len(points) >= 2 and np.ptp(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            residual = y - (slope * x + intercept)
            total = float(np.sum((y - np.mean(y)) ** 2))
            item.update(
                {
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "r2": (
                        None if total <= 0 else float(1.0 - np.sum(residual**2) / total)
                    ),
                }
            )
        out.append(item)
    return out


def _build_records(
    groups: Mapping[tuple[int, float], Mapping[str, np.ndarray]],
    processed: Mapping[tuple[int, float], Mapping[str, np.ndarray]],
    raw: Mapping[tuple[int, float], Mapping[str, np.ndarray]],
    quality: Mapping[tuple[int, float], Mapping[str, Any]],
    dark_subtracted: bool,
    frame_files: Mapping[tuple[int, float], str],
    bias_rad: float,
) -> list[dict[str, Any]]:
    """把五帧组送进共享估计器, 拼出每条 (noll, inject) 的记录。"""
    keys = list(groups)
    results = estimate_zernike_probe(
        [groups[k] for k in keys], [k[0] for k in keys], bias_rad
    )
    records: list[dict[str, Any]] = []
    for (noll, inject), result in zip(keys, results):
        q = quality[(noll, inject)]
        peaks = {
            key: probe_peak_intensity(groups[(noll, inject)][key])
            for key in FRAME_ORDER
        }
        stats = q["per_frame"]
        records.append(
            {
                "noll": int(noll),
                "n_m": list(noll_to_mode(noll)),
                "inject_rad": float(inject),
                "recovered_rad": float(result.coefficient_rad),
                "delta_rad": float(result.coefficient_rad - inject),
                "contrast": float(result.contrast),
                "valid": bool(result.valid),
                "dark_subtracted": bool(dark_subtracted),
                "frame_file": frame_files[(noll, inject)],
                "peak_intensities": {k: float(v) for k, v in peaks.items()},
                "peak_fractions": {
                    k: float(stats[k]["peak_fraction"]) for k in FRAME_ORDER
                },
                "clipped_pixels": {k: int(stats[k]["clipped_px"]) for k in FRAME_ORDER},
                "quality": {
                    "qualified": bool(q["qualified"]),
                    "problems": q["problems"],
                },
            }
        )
    # 只有 open 过的对象才会进这里; 保留 processed/raw 引用供调用方审计。
    assert processed and raw
    return records


def _save_group(
    run_dir: Path,
    noll: int,
    inject: float,
    processed: Mapping[str, np.ndarray],
    raw: Mapping[str, np.ndarray],
    meta: Mapping[str, Any],
) -> str:
    """写一个五帧 .npz: a/m/p/mm/pp 为扣暗帧, raw_* 为原始帧。"""
    filename = f"noll{noll:02d}_inject{inject:+.3f}.npz"
    arrays: dict[str, np.ndarray] = {}
    for key in FRAME_ORDER:
        arrays[key] = np.asarray(processed[key], dtype=np.float32)
        arrays[f"raw_{key}"] = np.asarray(raw[key])
    arrays["meta"] = np.array(
        json.dumps(dict(meta), ensure_ascii=False, sort_keys=True)
    )
    np.savez_compressed(run_dir / filename, **arrays)
    return filename


def dry_run_self_check(radius: float) -> dict[str, Any]:
    """固定 Noll 4 / inject 0.2 / bias 0.3 的 Sim2f 恢复自检。"""
    from ml.sim2f.config import Sim2fConfig
    from ml.sim2f.optics import Sim2f

    sim = Sim2f(
        Sim2fConfig(ccd_px=DRY_RUN_GRID, grid=DRY_RUN_GRID, exposure_ms=0.0),
        device="cpu",
    )
    helper = PatternHelper(resolution=(DRY_RUN_GRID, DRY_RUN_GRID))
    phases = build_probe_phases(
        helper,
        PROBE_SELF_CHECK_NOLL,
        PROBE_SELF_CHECK_INJECT,
        PROBE_SELF_CHECK_BIAS,
        radius,
        n_max=DEFAULT_N_MAX,
    )
    frames = {
        key: sim.forward(np.asarray(phase, dtype=np.float32)[None, :, :])[0, 0].numpy()
        for key, phase in phases.items()
    }
    result = estimate_zernike_probe(
        frames, [PROBE_SELF_CHECK_NOLL], PROBE_SELF_CHECK_BIAS
    )[0]
    error = abs(float(result.coefficient_rad) - PROBE_SELF_CHECK_EXPECTED)
    return {
        "performed": True,
        "noll": PROBE_SELF_CHECK_NOLL,
        "inject_rad": PROBE_SELF_CHECK_INJECT,
        "bias_rad": PROBE_SELF_CHECK_BIAS,
        "expected_rad": PROBE_SELF_CHECK_EXPECTED,
        "recovered_rad": float(result.coefficient_rad),
        "abs_error_rad": float(error),
        "tolerance_rad": PROBE_SELF_CHECK_TOLERANCE,
        "passed": bool(error <= PROBE_SELF_CHECK_TOLERANCE),
    }


def run_dry_run(
    args: argparse.Namespace, run_dir: Path, focal_mm: float | None
) -> dict[str, Any]:
    """离线孪生: 与硬件路径共用 build → 取峰 → estimate → 写产物。"""
    from ml.sim2f.config import Sim2fConfig
    from ml.sim2f.optics import Sim2f

    if focal_mm is not None:
        logger.warning("dry-run 不施加焦面补偿 (Sim2f 始终在焦): focal_mm={}", focal_mm)
    sim = Sim2f(
        Sim2fConfig(ccd_px=DRY_RUN_GRID, grid=DRY_RUN_GRID, exposure_ms=0.0),
        device="cpu",
    )
    helper = PatternHelper(resolution=(DRY_RUN_GRID, DRY_RUN_GRID))
    radius = min(float(args.radius), DRY_RUN_RADIUS)
    if radius != float(args.radius):
        logger.info("dry-run 孔径钳到 64×64 网格: {} → {} px", args.radius, radius)

    groups: dict[tuple[int, float], dict[str, np.ndarray]] = {}
    processed: dict[tuple[int, float], dict[str, np.ndarray]] = {}
    raw: dict[tuple[int, float], dict[str, np.ndarray]] = {}
    quality: dict[tuple[int, float], dict[str, Any]] = {}
    frame_files: dict[tuple[int, float], str] = {}
    for noll in args.noll:
        for inject in args.inject_rad:
            key = (int(noll), float(inject))
            phases = build_probe_phases(
                helper, key[0], key[1], float(args.bias_rad), radius, DEFAULT_N_MAX
            )
            frames = {
                k: sim.forward(np.asarray(p, dtype=np.float32)[None, :, :])[
                    0, 0
                ].numpy()
                for k, p in phases.items()
            }
            groups[key] = frames
            processed[key] = {
                k: np.asarray(v, dtype=np.float32) for k, v in frames.items()
            }
            raw[key] = dict(processed[key])
            quality[key] = assess_group_quality(
                raw[key], processed[key], DRY_RUN_FULL_SCALE
            )
            if args.save_frames:
                frame_files[key] = _save_group(
                    run_dir,
                    key[0],
                    key[1],
                    processed[key],
                    raw[key],
                    {
                        "noll": key[0],
                        "inject_rad": key[1],
                        "bias_rad": float(args.bias_rad),
                        "radius_px": radius,
                        "simulated": True,
                    },
                )
            else:
                frame_files[key] = ""

    records = _build_records(
        groups,
        processed,
        raw,
        quality,
        False,
        frame_files,
        float(args.bias_rad),
    )
    self_check = dry_run_self_check(radius)
    problems = (
        [
            f"dry-run 自检失败: recovered={self_check['recovered_rad']:.4f} "
            f"expected={self_check['expected_rad']:.4f}"
        ]
        if not self_check["passed"]
        else []
    )
    for record in records:
        problems.extend(
            f"noll={record['noll']} inject={record['inject_rad']}: {p}"
            for p in record["quality"]["problems"]
        )
    return {
        "records": records,
        "linearity": compute_linearity(records),
        "quality": {
            "qualified": not problems,
            "problems": problems,
            "thresholds": {
                "peak_fraction_low": QUALITY_PEAK_LOW,
                "peak_fraction_high": QUALITY_PEAK_HIGH,
            },
        },
        "config": {
            "schema": SCHEMA,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "simulated": True,
            "params": _jsonable_params(args),
            "git_commit": _git_short_sha(),
            "focal_mm": focal_mm,
            "focal_compensated": False,
            "radius_used_px": radius,
            "n_max": DEFAULT_N_MAX,
            "full_scale": DRY_RUN_FULL_SCALE,
            "dark": {
                "requested": bool(args.subtract_dark),
                "applied": False,
                "matched": False,
                "reason": "Sim2f 无相机/无暗帧库; 帧已是理想强度",
            },
            "exposure": {
                "mode": "simulated",
                "requested_ms": float(args.exposure_ms),
                "start_ms": None,
                "settled_ms": None,
                "effective_ms": None,
                "peak_fraction": None,
                "status": "not_applicable",
                "converged": True,
            },
            "device": {"simulated": True, "grid": DRY_RUN_GRID},
            "self_check": self_check,
        },
    }


def _clamp(value: float, lo: float, hi: float) -> float:
    """把请求值钳到设备量程。"""
    return float(min(max(float(value), lo), hi))


def _positive_float(value: Any) -> float | None:
    """转成正有限浮点数; 非法返回 None。"""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) and out > 0 else None


def _coerce_bit_depth(value: Any) -> int | None:
    """把设备回读值 (整数 / 数字串 / 枚举名) 归一化为位深; 认不出返回 None。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        bits = int(value)
    elif isinstance(value, (float, np.floating)):
        if not float(value).is_integer():
            return None
        bits = int(value)
    elif isinstance(value, str):
        digits = "".join(ch for ch in value if ch.isdigit())
        bits = int(digits) if digits else -1
    else:
        for attr in ("name", "value"):
            inner = getattr(value, attr, None)
            if inner is not None and inner is not value:
                return _coerce_bit_depth(inner)
        return None
    return bits if 0 < bits <= 32 else None


def read_camera_bit_depth(camera: Any, *, cam_type: str = "") -> tuple[int | None, str]:
    """读相机**生效**位深: 设备回读入口优先, 否则退回驱动 ``bit_depth`` 属性。

    Returns:
        ``(位深或 None, 来源)``; 来源 ``device`` = 设备回读, ``driver`` = 驱动
        属性兜底 (已告警), ``none`` = 后端不报告位深 (调用方跳过校验)。
    """
    getter = getattr(camera, "get_bit_depth", None)
    if callable(getter):
        try:
            bits = _coerce_bit_depth(getter())
            if bits is not None:
                return bits, "device"
        except Exception as exc:  # 回读入口存在但设备/SDK 报错
            logger.warning("相机 get_bit_depth() 回读失败: {!r}", exc)
    if str(cam_type).lower() == "daheng":
        pixel_format = getattr(getattr(camera, "cam", None), "PixelFormat", None)
        read_format = getattr(pixel_format, "get", None)
        if callable(read_format):
            try:
                from ao_shaping.drivers.ccd.daheng._bit_depth import (
                    bit_depth_of_format,
                )

                bits = bit_depth_of_format(read_format())
                if bits is not None:
                    return bits, "device"
            except Exception as exc:
                logger.warning("相机 PixelFormat 回读失败: {!r}", exc)
    bits = _coerce_bit_depth(getattr(camera, "bit_depth", None))
    if bits is not None:
        logger.warning(
            "相机 {} 无设备位深回读入口; 回退驱动 bit_depth 属性 ({} bit)",
            type(camera).__name__,
            bits,
        )
        return bits, "driver"
    logger.warning("相机 {} 无位深回读入口, 跳过位深校验", type(camera).__name__)
    return None, "none"


def verify_camera_bit_depth(
    camera: Any, requested: int, *, cam_type: str = ""
) -> tuple[int | None, str]:
    """按设备回读核对位深; 不符即抛错, 绝不带着错量纲的满量程继续采集。"""
    requested_bits = int(requested)
    actual, source = read_camera_bit_depth(camera, cam_type=cam_type)
    if actual is None:
        logger.warning(
            "相机未报告位深, 跳过 --bit-depth={} 校验; 满量程按帧 dtype 兜底",
            requested_bits,
        )
        return None, source
    if actual != requested_bits:
        supported = getattr(camera, "supported_bit_depths", None)
        raise RuntimeError(
            f"相机位深与请求不符: 请求 {requested_bits} bit, "
            f"设备回读 {actual} bit (来源={source}). "
            f"满量程按回读为 {(1 << actual) - 1}, 本轮判据按请求为 "
            f"{(1 << requested_bits) - 1}, 量纲不一致, 拒绝继续. "
            f"supported_bit_depths={supported}; "
            "位深须在开流前写好 (拉流中 PixelFormat 写入不生效)."
        )
    return actual, source


def rescale_dark_frame(
    mean: np.ndarray,
    *,
    dark_bit_depth: int | None,
    camera_full_scale: float,
    dark_full_scale: float | None = None,
) -> np.ndarray | None:
    """把暗帧线性缩放到相机满量程 (四舍五入); 无有效满量程返回 None。

    暗帧位深优先于显式 ``full_scale``: 前者是捕获时写进 manifest 的设备真值。
    """
    bits = _coerce_bit_depth(dark_bit_depth)
    fs_dark = float((1 << bits) - 1) if bits is not None else None
    if fs_dark is None:
        fs_dark = _positive_float(dark_full_scale)
    fs_camera = _positive_float(camera_full_scale)
    arr = np.asarray(mean)
    if (
        fs_dark is None
        or fs_camera is None
        or arr.ndim == 0
        or not np.issubdtype(arr.dtype, np.number)
    ):
        return None
    scaled = np.floor(arr.astype(np.float64) * (fs_camera / fs_dark) + 0.5)
    return scaled.astype(np.float32)


def resolve_dark_frame(
    dark: Any | None,
    *,
    requested: bool,
    exposure_ms: float,
    camera_bit_depth: int | None,
    camera_full_scale: float,
) -> tuple[Any | None, dict[str, Any]]:
    """核对暗帧位深契约: 不一致则缩放后扣暗, 无法缩放则跳过扣暗。"""
    info: dict[str, Any] = {
        "requested": bool(requested),
        "applied": False,
        "matched": dark is not None,
        "exposure_ms": float(exposure_ms),
        "entry": None if dark is None else dark.entry.to_dict(),
        "bit_depth_mismatch": False,
        "rescaled": False,
    }
    if not requested:
        info["reason"] = "未请求"
        return None, info
    if dark is None:
        info["reason"] = "无匹配暗帧, 原始帧直接参与估计"
        return None, info
    bits_camera = _coerce_bit_depth(camera_bit_depth)
    if bits_camera is None:
        info["reason"] = "相机位深未知, 无法核对暗帧位深契约; 跳过扣暗"
        logger.warning(info["reason"])
        return None, info
    entry = dark.entry
    bits_dark = _coerce_bit_depth(getattr(entry, "bit_depth", None))
    if bits_dark == bits_camera:
        info["applied"] = True
        info["reason"] = "严格匹配"
        return dark, info

    info["bit_depth_mismatch"] = True
    dark_fs = getattr(entry, "full_scale", None)
    scaled_mean = rescale_dark_frame(
        dark.mean,
        dark_bit_depth=bits_dark,
        camera_full_scale=camera_full_scale,
        dark_full_scale=dark_fs,
    )
    if scaled_mean is None:
        info["reason"] = (
            f"暗帧位深 {bits_dark} 与相机 {bits_camera} 不一致, "
            "且无有效暗帧满量程可缩放; 跳过扣暗"
        )
        logger.warning(info["reason"])
        return None, info
    scaled_std = None
    std = getattr(dark, "std", None)
    if std is not None:
        scaled_std = rescale_dark_frame(
            std,
            dark_bit_depth=bits_dark,
            camera_full_scale=camera_full_scale,
            dark_full_scale=dark_fs,
        )
    try:
        effective = replace(
            dark,
            mean=scaled_mean,
            std=scaled_std,
            full_scale=_positive_float(camera_full_scale),
        )
    except TypeError as exc:
        info["reason"] = f"暗帧对象不可复制 ({exc!r}); 跳过扣暗"
        logger.warning(info["reason"])
        return None, info
    info["applied"] = True
    info["rescaled"] = True
    info["reason"] = (
        f"暗帧位深 {bits_dark} 与相机 {bits_camera} 不一致, "
        "已线性缩放到相机满量程后扣暗"
    )
    logger.warning(info["reason"])
    return effective, info


def run_hardware(
    args: argparse.Namespace, run_dir: Path, focal_mm: float | None
) -> dict[str, Any]:
    """实验系统采集: 锁 → SLM → CCD 依次进入 ExitStack, 逆序关闭, 最后放锁。"""
    from ao_shaping.drivers.ccd.common import (
        create_camera,
        full_scale,
        get_camera_exposure_ms,
        get_camera_exposure_range,
        set_camera_exposure_ms,
    )
    from ao_shaping.drivers.slm.factory import require_drive_board_entry
    from ao_shaping.drivers.slm.santec import Santec, SlotRotator
    from ao_shaping.tools.kernels.device_mutex import DeviceLock, wait_until_idle
    from ao_shaping.utils.slm.phase_display import display_phase, phase_to_slm_grayscale

    verdict = wait_until_idle(
        float(args.wait_until_idle),
        slm_number=int(args.slm_number),
        cam_id=int(args.cam_id),
        cam_type=str(args.cam_type),
        exposure_ms=float(args.exposure_ms),
        own_paths=[Path(__file__).resolve()],
    )
    if not verdict.ok:
        raise RuntimeError(
            f"设备不可用 ({verdict.status}/{verdict.kind}): {verdict.headline()} "
            f"— {verdict.remedy or verdict.reason}"
        )
    lock = verdict.lock if verdict.lock is not None else DeviceLock.default()
    if not lock.is_locked():
        lock.acquire(timeout=0.0)
    if not lock.is_locked():
        raise RuntimeError("设备锁未能拿到, 拒绝开跑")

    with ExitStack() as stack:
        stack.callback(lock.release)
        slm = stack.enter_context(
            Santec(
                slm_number=int(args.slm_number),
                video_mode=0,
                wavelength=int(args.slm_wavelength),
            )
        )
        require_drive_board_entry(slm)
        camera = stack.enter_context(
            create_camera(
                str(args.cam_type),
                cam_id=args.cam_id,
                exposure_time_ms=float(args.exposure_ms),
                bit_depth=int(args.bit_depth),
            )
        )
        actual_bit_depth, _bit_depth_source = verify_camera_bit_depth(
            camera, int(args.bit_depth), cam_type=str(args.cam_type)
        )
        panel_shape = (int(slm.Panel_Res[1]), int(slm.Panel_Res[0]))
        helper = PatternHelper(resolution=(panel_shape[1], panel_shape[0]))
        rotator = SlotRotator(slm)
        focal_phase = (
            fresnel_panel(
                focal_mm,
                panel_shape=panel_shape,
                wavelength_nm=float(args.slm_wavelength),
            )
            if focal_mm is not None
            else None
        )

        fs = float(full_scale(np.zeros((1, 1), dtype=np.uint16), camera=camera))
        lo, hi = get_camera_exposure_range(camera)
        start_ms = _clamp(float(args.exposure_ms), float(lo), float(hi))
        set_camera_exposure_ms(camera, start_ms)

        def capture() -> np.ndarray:
            frame = camera.get_numpy_image(n_sample=int(args.frames), skip_first=True)
            if frame is None:
                raise RuntimeError("CCD 返回空帧")
            return np.asarray(frame)

        def show_flat() -> None:
            gray = int(
                np.asarray(
                    phase_to_slm_grayscale(np.zeros((1, 1), dtype=np.float64), slm=slm)
                ).flat[0]
            )
            flat = np.full((panel_shape[0], panel_shape[1]), gray, dtype=np.uint16)
            rotator.display_data(slm, flat, float(args.settle_s))

        def show_probe(phase: np.ndarray) -> None:
            display_phase(slm, phase, float(args.settle_s))

        # 曝光预检: 平场 + 全部五帧图案都测一遍, 用最亮的那个定曝光, 防止 ±2b 饱和。
        candidates: list[tuple[str, np.ndarray | None]] = [("flat", None)]
        for noll in args.noll:
            for inject in args.inject_rad:
                phases = build_probe_phases(
                    helper,
                    int(noll),
                    float(inject),
                    float(args.bias_rad),
                    float(args.radius),
                    DEFAULT_N_MAX,
                    focal_phase=focal_phase,
                )
                candidates.extend(
                    (f"noll{noll}_inj{inject:+g}_{k}", p) for k, p in phases.items()
                )
        candidate_peaks: list[dict[str, Any]] = []
        brightest: tuple[str, np.ndarray | None] = candidates[0]
        for label, phase in candidates:
            if phase is None:
                show_flat()
            else:
                show_probe(phase)
            peak = float(robust_peak(capture()))
            candidate_peaks.append(
                {"label": label, "peak_counts": peak, "peak_fraction": peak / fs}
            )
            if peak > next(
                item["peak_counts"]
                for item in candidate_peaks
                if item["label"] == brightest[0]
            ):
                brightest = (label, phase)
        if brightest[1] is None:
            show_flat()
        else:
            show_probe(brightest[1])

        setting = None
        if args.auto_exposure:
            setting = adapt_exposure(
                set_exposure_ms=lambda ms: set_camera_exposure_ms(camera, ms),
                read_frame=capture,
                start_ms=start_ms,
                floor_ms=float(lo),
                ceiling_ms=float(hi),
                full_scale=fs,
                target=EXPOSURE_TARGET,
                accept=EXPOSURE_ACCEPT,
            )
            settled_ms = float(setting.exposure_ms)
        else:
            settled_ms = float(set_camera_exposure_ms(camera, start_ms))
        effective_ms = float(get_camera_exposure_ms(camera))
        exposure_info: dict[str, Any] = {
            "mode": "ladder" if setting is not None else "fixed",
            "requested_ms": float(args.exposure_ms),
            "start_ms": float(start_ms),
            "settled_ms": settled_ms,
            "effective_ms": effective_ms,
            "range_ms": [float(lo), float(hi)],
            "full_scale": fs,
            "peak_fraction": None if setting is None else float(setting.fraction),
            "status": "not_applicable" if setting is None else setting.status,
            "reads": 0 if setting is None else int(setting.reads),
            "converged": True if setting is None else setting.status == "ok",
            "brightest_candidate": brightest[0],
            "candidate_peaks": candidate_peaks,
        }
        logger.info(
            "曝光确定: {} ms (有效回读 {} ms), {}",
            settled_ms,
            effective_ms,
            exposure_info["status"],
        )

        dark_raw = camera.get_dark_frame(effective_ms) if args.subtract_dark else None
        if args.subtract_dark and dark_raw is None:
            logger.warning(
                "曝光 {} ms 没有严格匹配的暗帧; 本轮 dark_subtracted=false, 不伪造暗场",
                effective_ms,
            )
        dark, dark_info = resolve_dark_frame(
            dark_raw,
            requested=bool(args.subtract_dark),
            exposure_ms=effective_ms,
            camera_bit_depth=actual_bit_depth,
            camera_full_scale=fs,
        )

        groups: dict[tuple[int, float], dict[str, np.ndarray]] = {}
        processed: dict[tuple[int, float], dict[str, np.ndarray]] = {}
        raw: dict[tuple[int, float], dict[str, np.ndarray]] = {}
        quality: dict[tuple[int, float], dict[str, Any]] = {}
        frame_files: dict[tuple[int, float], str] = {}
        for noll in args.noll:
            for inject in args.inject_rad:
                key = (int(noll), float(inject))
                phases = build_probe_phases(
                    helper,
                    key[0],
                    key[1],
                    float(args.bias_rad),
                    float(args.radius),
                    DEFAULT_N_MAX,
                    focal_phase=focal_phase,
                )
                group: dict[str, np.ndarray] = {}
                raw_group: dict[str, np.ndarray] = {}
                for frame_key in FRAME_ORDER:
                    show_probe(phases[frame_key])
                    frame = capture()
                    raw_group[frame_key] = frame
                    group[frame_key] = (
                        np.asarray(dark.subtract(frame), dtype=np.float32)
                        if dark is not None
                        else np.asarray(frame, dtype=np.float32)
                    )
                groups[key] = group
                processed[key] = group
                raw[key] = raw_group
                estimate = estimate_zernike_probe(
                    [group], [key[0]], float(args.bias_rad)
                )[0]
                quality[key] = assess_group_quality(
                    raw_group,
                    group,
                    fs,
                    exposure_converged=exposure_info["converged"],
                    valid=bool(estimate.valid),
                )
                if args.save_frames:
                    frame_files[key] = _save_group(
                        run_dir,
                        key[0],
                        key[1],
                        group,
                        raw_group,
                        {
                            "noll": key[0],
                            "inject_rad": key[1],
                            "bias_rad": float(args.bias_rad),
                            "exposure_ms": effective_ms,
                            "full_scale": fs,
                            "dark_subtracted": dark is not None,
                            "focal_mm": focal_mm,
                        },
                    )
                else:
                    frame_files[key] = ""

        records = _build_records(
            groups,
            processed,
            raw,
            quality,
            dark is not None,
            frame_files,
            float(args.bias_rad),
        )
        device_info: dict[str, Any] = {
            "slm": {
                "number": int(args.slm_number),
                "wavelength_nm": int(args.slm_wavelength),
                "panel_shape": list(panel_shape),
                "serial": getattr(slm, "serial_number", None),
                "max_gray": getattr(slm, "_max_gray", None),
            },
            "camera": {
                "type": str(args.cam_type),
                "id": args.cam_id,
                "serial": camera.serial_number,
                "bit_depth": getattr(camera, "bit_depth", None),
                "exposure_range_ms": [float(lo), float(hi)],
            },
            "lock": {
                "status": verdict.status,
                "kind": verdict.kind,
                "waited_s": float(verdict.waited_s),
                "held_until_acquisition_end": True,
            },
        }
        try:
            device_info["slm"]["wavelength_info"] = list(slm.get_wavelength_info())
        except Exception as exc:
            device_info["slm"]["wavelength_info_error"] = str(exc)

    problems: list[str] = []
    if not exposure_info["converged"]:
        problems.append("曝光阶梯未收敛到接受带")
    for record in records:
        problems.extend(
            f"noll={record['noll']} inject={record['inject_rad']}: {p}"
            for p in record["quality"]["problems"]
        )
    return {
        "records": records,
        "linearity": compute_linearity(records),
        "quality": {
            "qualified": not problems,
            "problems": problems,
            "thresholds": {
                "peak_fraction_low": QUALITY_PEAK_LOW,
                "peak_fraction_high": QUALITY_PEAK_HIGH,
            },
        },
        "config": {
            "schema": SCHEMA,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "simulated": False,
            "params": _jsonable_params(args),
            "git_commit": _git_short_sha(),
            "focal_mm": focal_mm,
            "focal_compensated": focal_mm is not None,
            "n_max": DEFAULT_N_MAX,
            "full_scale": fs,
            "exposure": exposure_info,
            "dark": dark_info,
            "device": device_info,
            "self_check": {"performed": False, "reason": "仅 dry-run 执行"},
        },
    }


def write_outputs(run_dir: Path, payload: Mapping[str, Any]) -> None:
    """写 summary.json / summary.csv / probe_config.json (只创建新文件)。"""
    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at_utc": payload["config"]["generated_at_utc"],
        "simulated": payload["config"]["simulated"],
        "quality": payload["quality"],
        "linearity": payload["linearity"],
        "records": payload["records"],
    }
    with (run_dir / "summary.json").open("x", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    with (run_dir / "probe_config.json").open("x", encoding="utf-8") as fh:
        json.dump(payload["config"], fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")

    columns = [
        "noll",
        "n_m",
        "inject_rad",
        "recovered_rad",
        "delta_rad",
        "contrast",
        "valid",
        "qualified",
        "dark_subtracted",
        "frame_file",
        "peak_fraction_a",
        "peak_fraction_m",
        "peak_fraction_p",
        "peak_fraction_mm",
        "peak_fraction_pp",
    ]
    with (run_dir / "summary.csv").open("x", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for record in payload["records"]:
            row = {key: record.get(key) for key in columns}
            row["n_m"] = f"{record['n_m'][0]},{record['n_m'][1]}"
            row["qualified"] = record["quality"]["qualified"]
            for key in FRAME_ORDER:
                row[f"peak_fraction_{key}"] = record["peak_fractions"][key]
            writer.writerow(row)


def utc_stamp() -> str:
    """UTC 运行目录时间戳 (测试可 monkeypatch 以固定同秒碰撞场景)。"""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def main(argv: list[str] | None = None) -> int:
    """入口: 返回 0=合格, 1=产物已写但质量门未过, 3=采集异常。"""
    args = build_parser().parse_args(argv)
    if float(args.bias_rad) <= 0:
        raise SystemExit(f"--bias-rad 必须 > 0, 得到 {args.bias_rad}")
    if not args.noll or not args.inject_rad:
        raise SystemExit("--noll / --inject-rad 不能为空")
    if int(args.frames) < 1:
        raise SystemExit("--frames 必须 >= 1")

    focal_mm = resolve_focal_mm(args)
    stamp = utc_stamp()
    run_dir = Path(args.out) / stamp
    if run_dir.exists():
        raise SystemExit(f"输出目录已存在, 拒绝覆盖: {run_dir}")
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise SystemExit(f"无法创建输出目录 {run_dir}: {exc}") from exc
    logger.info("输出目录: {}", run_dir)

    try:
        payload = (
            run_dry_run(args, run_dir, focal_mm)
            if args.dry_run
            else run_hardware(args, run_dir, focal_mm)
        )
        write_outputs(run_dir, payload)
    except Exception as exc:
        logger.exception("HMWS 探针失败 (产物目录 {}): {}", run_dir, exc)
        return 3

    logger.info(
        "完成: {} 条记录, qualified={}",
        len(payload["records"]),
        payload["quality"]["qualified"],
    )
    return 0 if payload["quality"]["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
