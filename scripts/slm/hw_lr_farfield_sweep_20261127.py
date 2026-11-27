"""硬件实验脚本: 不同 lr 下远场光斑随可微优化的迭代演化 (默认 dry-run)。

要验证什么
----------
同一初始相位、同一目标、同一 seed, 只改 Adam 的 ``lr``, 看**真实远场**在迭代
过程中怎么走:

* 哪一档 lr 在物理系统上真的把能量收进目标 (而不是只在模型里收进去);
* 收敛速度 (多少迭代后峰值/FWHM/质心不再变) 与 lr 的关系;
* lr 过大时是否出现模型损失下降而实测变差的「模型-硬件分岔」。

与 :mod:`scripts.slm.hw_differentiable_phase_opt` 的关系
--------------------------------------------------------
**设备与相位管线完全复用那份脚本**, 不重写、不修改它::

    ModelSpec / build_model_spec / build_source_amplitude / build_target
    phase_model_to_panel      (模型相位 → 面板相位)
    guarded_device            (打开/关闭都套看门狗)
    DifferentiableBeamOptimizer  (仓库唯一的可微整形公开 API)

几何 / 目标 / 曝光 / seed 一律从存档 run 的 ``run_config.json`` 的 ``params``
继承 (:meth:`ModelSpec` 口径由 ``model_pupil_samples`` 等决定), 初始相位直接取
存档的 ``phase_initial.npy``, 焦面补偿直接复用存档 ``focus`` 的结果 ——
**本脚本不做 focus sweep**, 因此与历史 lr=0.05 那次 run 同口径。

🔴 **默认 dry-run; ``--execute`` 才会动设备; 上机前需用户授权。**

不给 ``--execute`` 时本脚本:

* 不 import 任何驱动 (``ao_shaping.drivers.*`` / ``santec_slm`` / ``thorlabs``),
  也不实例化任何硬件对象; 设备 import 全部锁在 :func:`run_execute` 内;
* 只读存档 (``run_config.json`` / ``phase_initial.npy``) 与纯 numpy 模型对象,
  打印完整 plan (lr 网格、每步动作、采集次数、耗时区间、输出布局、曝光策略);
* 收尾断言 ``ao_shaping.drivers`` 没出现在 ``sys.modules`` 里, 违反就 exit 3。

退出码: 0 = 正常; 1 = 未预期异常; 2 = 设备挂死/超时 (``guarded`` 硬退出);
3 = dry-run 违约; 8 = 采集被中止 (饱和超限 / 全黑帧); 9 = 参数或输入不合法。

输出布局 (与渲染脚本约定, 勿改)::

    <out>/run_config.json          # 完整参数 + source_run + 每档 lr 的执行状态
    <out>/journal.jsonl            # 逐帧记录 (逐行 flush)
    <out>/flat/iter_XXXX.npz       # 平相位参考帧 (lr=nan)
    <out>/lr_0p02/iter_0000.npz    # 键: frame / phase_model / exposure_ms / lr / iter / loss

``frame`` 是**原始** CCD 帧的第一帧 (2D, 原始 dtype; 不做暗减、不做归一化);
``--frames`` > 1 时全部原始帧另存为 ``frames`` 键 (3D), ``frame`` 保持 2D ——
渲染脚本按 2D 读取 ``frame``, 两边的键名口径因此一致。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from loguru import logger

from ao_shaping.algorithm.signal_processing.differentiable_beam import (
    DifferentiableBeamOptimizer,
)
from ao_shaping.tools.kernels.bench_kernels import (
    SLM_PANEL_H,
    SLM_PANEL_W,
    fresnel_panel,
)
from ao_shaping.utils.image.frame_analysis import measure_spot
from ao_shaping.utils.slm.bench_constants import SLM_PITCH_M
from scripts.slm.hw_differentiable_phase_opt import (
    ModelSpec,
    build_model_spec,
    build_source_amplitude,
    build_target,
    phase_model_to_panel,
)

ROOT = Path(__file__).resolve().parents[2]
PANEL_SHAPE: tuple[int, int] = (SLM_PANEL_H, SLM_PANEL_W)

DEFAULT_SOURCE_RUN = "data/debug/differentiable_opt/20261116_114424"

#: 退出码
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_DEVICE_HANG = 2
EXIT_DRYRUN_VIOLATION = 3
EXIT_ABORTED = 8
EXIT_BAD_INPUT = 9

#: 饱和像素比例超过它就中止 (0.1 %)
ABORT_SATURATION_FRACTION = 1.0e-3
#: 有饱和 (但低于中止线) 时, 每次把曝光 ×0.5 重采, 最多重采这么多次
MAX_SATURATION_RETRIES = 3
#: 曝光下限 (ms): 再低就全是读出噪声
MIN_EXPOSURE_MS = 0.02
EXPOSURE_DECREASE_FACTOR = 0.5

# 耗时估算用的单次代价 (秒)。只用于 plan 里的区间, 不参与任何判据。
DISPLAY_OVERHEAD_S = 0.20
READOUT_S_PER_FRAME = 0.35
STATS_S_PER_FRAME = 0.40
NPZ_SAVE_S_PER_CAPTURE = 1.00
MODEL_STEP_S = 0.08
TIMING_LOW_FACTOR = 0.7
TIMING_HIGH_FACTOR = 1.6


class RunAborted(Exception):
    """采集必须中止 (饱和超限 / 全黑帧)。带落盘用的细节。"""

    def __init__(self, reason: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = dict(detail or {})


@dataclass
class ModelInputs:
    """从存档 + CLI 装配出来的、**不含硬件**的实验输入。"""

    source_run: Path
    source_config: dict[str, Any]
    params: dict[str, Any]
    spec: ModelSpec
    amplitude: np.ndarray
    target: np.ndarray
    target_info: dict[str, Any]
    init_phase: np.ndarray
    compensation: np.ndarray
    k_mean: float
    focus_info: dict[str, Any]
    geometry: dict[str, Any]
    focal_mm: float
    exposure_ms: float
    seed: int
    wavelength_nm: int
    slm_number: int
    cam_type: str
    cam_id: int
    pupil_center: tuple[float, float]
    pupil_radius: float


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行 (默认 dry-run)。"""
    parser = argparse.ArgumentParser(
        description="不同 lr 下远场光斑随可微优化迭代的硬件实测 (默认 dry-run)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-run",
        default=DEFAULT_SOURCE_RUN,
        help="存档 run 目录 (继承 params/phase_initial/focus)",
    )
    parser.add_argument(
        "--lrs",
        type=float,
        nargs="+",
        default=[0.02, 0.05, 0.08, 0.12],
        help="要扫的 Adam 学习率 (每档冷启动一个独立优化器)",
    )
    parser.add_argument("--iters", type=int, default=60, help="每档 lr 的迭代数")
    parser.add_argument(
        "--capture-every",
        type=int,
        default=1,
        help="每 N 个迭代采集一次 (中间迭代只 update() 不上屏)",
    )
    parser.add_argument("--frames", type=int, default=2, help="每次采集的帧数")
    parser.add_argument("--discard", type=int, default=2, help="每次采集前丢弃的帧数")
    parser.add_argument(
        "--settle-s", type=float, default=0.15, help="上屏后的稳定等待 (秒)"
    )
    parser.add_argument(
        "--exposure-ms",
        type=float,
        default=None,
        help="起始曝光 (默认继承源 run; 饱和时只降不升)",
    )
    parser.add_argument(
        "--flat-every",
        type=int,
        default=20,
        help="每 N 个迭代插一帧平相位参考 (0 = 只保留每组首尾)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出目录 (默认 data/debug/differentiable_opt/lr_farfield_hw_<UTC>)",
    )
    parser.add_argument("--device", default="cuda", help="torch 设备")
    parser.add_argument(
        "--seed", type=int, default=None, help="优化器 seed (默认继承源 run)"
    )
    parser.add_argument(
        "--defocus-diopter",
        type=float,
        default=None,
        help="覆盖焦面补偿的 1/f (m^-1); 0 = 不加补偿相位",
    )
    parser.add_argument(
        "--k-mean",
        type=float,
        default=None,
        help="覆盖远场标度 K (CCD 像元/面板像元); 默认继承源 run 的 geometry",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="只打印 plan (默认行为)"
    )
    parser.add_argument(
        "--execute", action="store_true", help="真正打开设备并采集 (需用户授权)"
    )
    args = parser.parse_args(argv)

    if args.dry_run and args.execute:
        parser.error("--dry-run 与 --execute 互斥")
    if args.iters < 1:
        parser.error("--iters 必须 >= 1")
    if args.capture_every < 1:
        parser.error("--capture-every 必须 >= 1")
    if args.frames < 1:
        parser.error("--frames 必须 >= 1")
    if args.discard < 0:
        parser.error("--discard 必须 >= 0")
    if args.settle_s < 0.0:
        parser.error("--settle-s 必须 >= 0")
    if args.flat_every < 0:
        parser.error("--flat-every 必须 >= 0")
    if not args.lrs or any(lr <= 0.0 for lr in args.lrs):
        parser.error("--lrs 里每个 lr 都必须 > 0")
    if args.exposure_ms is not None and args.exposure_ms <= 0.0:
        parser.error("--exposure-ms 必须 > 0")
    args.lrs = _unique_lrs(args.lrs)
    return args


def _unique_lrs(lrs: Sequence[float]) -> list[float]:
    """保序去重 (重复 lr 会撞同一个输出目录)。"""
    out: list[float] = []
    for lr in lrs:
        if not any(abs(lr - seen) < 1e-12 for seen in out):
            out.append(float(lr))
    if len(out) != len(lrs):
        logger.warning("--lrs 有重复档位, 已去重为 {}", out)
    return out


def lr_dir_name(lr: float) -> str:
    """``lr_0p02`` / ``lr_0p05`` / ``lr_0p12`` (小数点换 ``p``)。"""
    return ("lr_" + f"{float(lr):g}").replace(".", "p")


def resolve_out_dir(args: argparse.Namespace) -> Path:
    """解析输出目录; 已存在就报错 (绝不复用历史产物)。"""
    if args.out is not None:
        out = Path(args.out)
        if not out.is_absolute():
            out = Path.cwd() / out
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out = ROOT / "data" / "debug" / "differentiable_opt" / (
            f"lr_farfield_hw_{stamp}"
        )
    if out.exists():
        raise SystemExit(f"输出目录已存在, 绝不复用: {out}")
    return out


# ---------------------------------------------------------------------------
# 输入装配 (无硬件)
# ---------------------------------------------------------------------------
def load_model_inputs(args: argparse.Namespace) -> ModelInputs:
    """读存档 + 装配模型对象; 缺文件/口径不一致就带着说明退出。"""
    source = Path(args.source_run)
    if not source.is_absolute():
        source = ROOT / source
    config_path = source / "run_config.json"
    phase_path = source / "phase_initial.npy"
    for path in (config_path, phase_path):
        if not path.is_file():
            raise SystemExit(f"源 run 缺少文件: {path}")

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    params = dict(raw.get("params") or {})
    if not params:
        raise SystemExit(f"{config_path} 里没有 params")

    pupil_center = (
        float(params.get("pupil_center", (960.0, 600.0))[0]),
        float(params.get("pupil_center", (960.0, 600.0))[1]),
    )
    pupil_radius = float(params.get("pupil_radius", 480.0))
    n_pupil = int(params.get("model_pupil_samples", 256))
    padding = int(params.get("model_padding", 4))
    w0_px = float(params.get("beam_w0_px", 260.0))
    wavelength_nm = int(params.get("slm_wavelength", 1064))

    spec = build_model_spec(
        pupil_radius=pupil_radius,
        pupil_center=pupil_center,
        n_pupil=n_pupil,
        padding=padding,
        w0_px=w0_px,
    )
    amplitude = build_source_amplitude(spec)

    geometry = dict(raw.get("geometry") or {})
    k_mean = _resolve_k_mean(args, geometry)
    target_kind = str(params.get("target", "square"))
    target, target_info = build_target(
        spec,
        target_kind,
        float(params.get("target_side_px", 60.0)),
        k_mean,
        delta_spot_px=float(params.get("delta_spot_px", 15.0)),
    )

    init_phase = np.load(phase_path)
    if init_phase.shape != (spec.m_grid, spec.m_grid):
        raise SystemExit(
            f"{phase_path} 形状 {init_phase.shape} 与模型网格 "
            f"({spec.m_grid}, {spec.m_grid}) 不一致"
        )
    if target.shape != (spec.m_grid, spec.m_grid):
        raise SystemExit(f"目标形状 {target.shape} 与模型网格不一致")

    focus_info = dict(raw.get("focus") or {})
    focal_mm = _resolve_focal_mm(raw, focus_info, args.defocus_diopter)
    compensation = fresnel_panel(
        focal_mm,
        radius=int(round(pupil_radius)),
        pupil_center=(int(round(pupil_center[0])), int(round(pupil_center[1]))),
        panel_shape=PANEL_SHAPE,
        wavelength_nm=float(wavelength_nm),
        pixel_um=SLM_PITCH_M * 1e6,
    )

    exposure_ms = (
        float(args.exposure_ms)
        if args.exposure_ms is not None
        else float(params.get("exposure_ms", 2.0))
    )
    seed = int(args.seed) if args.seed is not None else int(params.get("seed", 0))

    return ModelInputs(
        source_run=source,
        source_config=raw,
        params=params,
        spec=spec,
        amplitude=amplitude,
        target=target,
        target_info=target_info,
        init_phase=np.asarray(init_phase, dtype=np.float64),
        compensation=np.asarray(compensation, dtype=np.float64),
        k_mean=k_mean,
        focus_info=focus_info,
        geometry=geometry,
        focal_mm=focal_mm,
        exposure_ms=exposure_ms,
        seed=seed,
        wavelength_nm=wavelength_nm,
        slm_number=int(params.get("slm_number", 1)),
        cam_type=str(params.get("cam_type", "daheng")),
        cam_id=int(params.get("cam_id", 0)),
        pupil_center=pupil_center,
        pupil_radius=pupil_radius,
    )


def _resolve_k_mean(args: argparse.Namespace, geometry: dict[str, Any]) -> float:
    """远场标度 K: CLI > 存档 k_mean > 存档 kx/ky 均值。"""
    if args.k_mean is not None:
        k_mean = float(args.k_mean)
    elif geometry.get("k_mean") is not None:
        k_mean = float(geometry["k_mean"])
    elif geometry.get("kx") is not None and geometry.get("ky") is not None:
        k_mean = 0.5 * (float(geometry["kx"]) + float(geometry["ky"]))
    else:
        raise SystemExit(
            "源 run 的 geometry 里没有 k_mean / (kx, ky), 请用 --k-mean 显式给出"
        )
    if not math.isfinite(k_mean) or k_mean <= 0.0:
        raise SystemExit(f"远场标度 K 不合法: {k_mean}")
    return k_mean


def _resolve_focal_mm(
    raw: dict[str, Any], focus_info: dict[str, Any], defocus_diopter: float | None
) -> float:
    """焦面补偿焦距 (mm): 复用存档 focus, ``--defocus-diopter`` 可覆盖。"""
    focal_mm = math.inf
    stored = raw.get("focus_compensation_focal_mm")
    if stored is None:
        stored = focus_info.get("best_focal_mm")
    if stored is not None:
        focal_mm = float(stored)
    if defocus_diopter is not None:
        diopter = float(defocus_diopter)
        focal_mm = math.inf if diopter == 0.0 else 1000.0 / diopter
    return focal_mm


# ---------------------------------------------------------------------------
# plan (纯计算, 不动设备)
# ---------------------------------------------------------------------------
def captured_iterations(iters: int, capture_every: int) -> list[int]:
    """会被采集的迭代下标 (含最后一迭代, 保证终态一定有一帧)。"""
    marks = {it for it in range(iters) if it % capture_every == 0}
    marks.add(iters - 1)
    return sorted(marks)


def flat_marks_for(iters: int, flat_every: int) -> list[int]:
    """平相位参考的插入位置: 组首 (0)、每 N 迭代、组尾 (iters)。"""
    marks = {0, iters}
    if flat_every > 0:
        mark = flat_every
        while mark < iters:
            marks.add(mark)
            mark += flat_every
    return sorted(marks)


def build_plan(
    args: argparse.Namespace, inputs: ModelInputs, out_dir: Path
) -> dict[str, Any]:
    """列出 lr 网格、采集次数与耗时区间。"""
    captured = captured_iterations(int(args.iters), int(args.capture_every))
    flats = flat_marks_for(int(args.iters), int(args.flat_every))
    n_events_per_group = len(captured) + len(flats)
    per_event_s = (
        DISPLAY_OVERHEAD_S
        + float(args.settle_s)
        + (int(args.discard) + int(args.frames)) * READOUT_S_PER_FRAME
        + int(args.frames) * STATS_S_PER_FRAME
        + NPZ_SAVE_S_PER_CAPTURE
    )
    per_group_s = n_events_per_group * per_event_s + int(args.iters) * MODEL_STEP_S
    total_s = per_group_s * len(args.lrs)

    return {
        "dry_run": not args.execute,
        "source_run": str(inputs.source_run),
        "out_dir": str(out_dir),
        "lrs": [float(lr) for lr in args.lrs],
        "lr_dirs": [lr_dir_name(lr) for lr in args.lrs],
        "iters": int(args.iters),
        "capture_every": int(args.capture_every),
        "captured_iters": captured,
        "flat_marks": flats,
        "frames": int(args.frames),
        "discard": int(args.discard),
        "settle_s": float(args.settle_s),
        "flat_every": int(args.flat_every),
        "exposure_ms_start": float(inputs.exposure_ms),
        "exposure_policy": (
            "起始曝光继承源 run; 出现饱和像素则 ×0.5 重采 (只降不升, 下限 "
            f"{MIN_EXPOSURE_MS} ms); 饱和比例 >= {ABORT_SATURATION_FRACTION:.2%} "
            "或全黑帧 => 中止"
        ),
        "events_per_group": n_events_per_group,
        "total_capture_events": n_events_per_group * len(args.lrs),
        "total_frames": n_events_per_group * len(args.lrs) * int(args.frames),
        "timing": {
            "per_event_s": per_event_s,
            "per_group_s": per_group_s,
            "total_s": total_s,
            "low_s": total_s * TIMING_LOW_FACTOR,
            "high_s": total_s * TIMING_HIGH_FACTOR,
        },
        "model": {
            "n_pupil": inputs.spec.n_pupil,
            "m_grid": inputs.spec.m_grid,
            "padding": inputs.spec.padding,
            "dp_panel_px": inputs.spec.dp,
            "pupil_center_panel_xy": list(inputs.pupil_center),
            "pupil_radius_panel_px": inputs.pupil_radius,
            "w0_panel_px": inputs.spec.w0_px,
            "w0_samples": inputs.spec.w0_samples,
            "k_mean_ccd_px_per_panel_px": inputs.k_mean,
            "far_field_sample_ccd_px": inputs.spec.far_field_sample_ccd_px(
                inputs.k_mean
            ),
            "far_field_span_ccd_px": inputs.spec.far_field_span_ccd_px(
                inputs.k_mean
            ),
            "target": inputs.target_info,
            "init_phase": str(inputs.source_run / "phase_initial.npy"),
            "seed": inputs.seed,
            "device": str(args.device),
        },
        "focus": {
            "source": str(inputs.source_run / "run_config.json"),
            "focal_mm": inputs.focal_mm,
            "diopter_m1": (0.0 if math.isinf(inputs.focal_mm) else 1000.0
                           / inputs.focal_mm),
            "best_fwhm_px_at_source": inputs.focus_info.get("best_fwhm_px"),
            "note": "复用存档 focus, 不重新做 focus sweep",
        },
        "steps": [
            "① 打开 SLM#{} ({}nm, video_mode=0) 与 {} CCD(id={}), 回读曝光".format(
                inputs.slm_number,
                inputs.wavelength_nm,
                inputs.cam_type,
                inputs.cam_id,
            ),
            "② 上屏「平相位 + 焦面补偿」采一组 flat 参考帧 (组首/组尾/每 N 迭代)",
            "③ 每档 lr 用同一 phase_initial.npy 冷启动一个独立优化器 (同 seed)",
            "④ 迭代: opt.update() → 模型相位 → phase_model_to_panel + "
            "compensation → 灰度 → 换槽上屏 (含 stable wait)",
            "⑤ 采集: settle → 丢 discard 帧 → 采 frames 帧 → 逐帧写 journal "
            "→ 原始帧 + 模型相位存 npz",
            "⑥ 饱和: 有饱和像素则曝光 ×0.5 重采; >= 0.1% 或全黑 → 中止并落盘原因",
            "⑦ 收尾: 恢复平相位 → 关闭设备; run_config.json 写每档 lr 状态",
        ],
    }


def _fmt_minutes(seconds: float) -> str:
    return f"{seconds / 60.0:.1f} min ({seconds:.0f} s)"


def print_plan(plan: dict[str, Any], inputs: ModelInputs) -> None:
    """把 plan 打印成人能核对的清单。"""
    bar = "─" * 72
    logger.info(bar)
    logger.info("DRY-RUN: 只打印计划, 不 import 驱动、不打开任何设备")
    logger.info(bar)
    logger.info("源 run : {}", plan["source_run"])
    logger.info("输出   : {}", plan["out_dir"])
    logger.info(
        "lr 网格: {}  → 目录 {}",
        ", ".join(f"{lr:g}" for lr in plan["lrs"]),
        ", ".join(plan["lr_dirs"]),
    )
    logger.info(
        "每档迭代: {} (每 {} 迭代采集一次; 采集点 {} 个, 含最后一迭代)",
        plan["iters"],
        plan["capture_every"],
        len(plan["captured_iters"]),
    )
    captured = plan["captured_iters"]
    preview = captured[:8] + (["..."] if len(captured) > 8 else [])
    logger.info("采集迭代: {}", preview)
    logger.info(
        "平相位参考位置: {} (组首 0 与组尾 {} 必拍)",
        plan["flat_marks"],
        plan["iters"],
    )
    logger.info(
        "采集参数: frames={} discard={} settle={}s",
        plan["frames"],
        plan["discard"],
        plan["settle_s"],
    )
    logger.info("曝光    : 起始 {} ms; {}", plan["exposure_ms_start"],
                plan["exposure_policy"])
    logger.info(bar)
    logger.info(
        "模型    : n_pupil={} m_grid={} padding={} dp={} 面板px, "
        "远场采样={:.3f} CCDpx, 覆盖={:.0f} CCDpx",
        plan["model"]["n_pupil"],
        plan["model"]["m_grid"],
        plan["model"]["padding"],
        plan["model"]["dp_panel_px"],
        plan["model"]["far_field_sample_ccd_px"],
        plan["model"]["far_field_span_ccd_px"],
    )
    logger.info(
        "目标    : kind={} 边长≈{:.1f} CCDpx ({} 样本) on {} 网格",
        inputs.target_info.get("kind"),
        inputs.target_info.get("side_ccd_px_effective")
        or inputs.target_info.get("spot_ccd_px_effective")
        or float("nan"),
        inputs.target_info.get("side_samples")
        or inputs.target_info.get("spot_samples"),
        plan["model"]["m_grid"],
    )
    logger.info(
        "焦面补偿: 1/f={:+.4f} m^-1 (f={} mm), 复用存档, 不做 focus sweep",
        plan["focus"]["diopter_m1"],
        "inf" if math.isinf(plan["focus"]["focal_mm"])
        else f"{plan['focus']['focal_mm']:.1f}",
    )
    logger.info(
        "初始相位: {} ({} x {}, 同一份给所有 lr)",
        plan["model"]["init_phase"],
        inputs.spec.m_grid,
        inputs.spec.m_grid,
    )
    logger.info(bar)
    for step in plan["steps"]:
        logger.info(step)
    logger.info(bar)
    logger.info(
        "预计采集 : {} 组 × {} 次 = {} 次采集, 共 {} 帧 (单次 ≈ {:.2f} s 固定代价)",
        len(plan["lrs"]),
        plan["events_per_group"],
        plan["total_capture_events"],
        plan["total_frames"],
        plan["timing"]["per_event_s"],
    )
    logger.info(
        "预计耗时 : {} – {} [中心值 {}]",
        _fmt_minutes(plan["timing"]["low_s"]),
        _fmt_minutes(plan["timing"]["high_s"]),
        _fmt_minutes(plan["timing"]["total_s"]),
    )
    logger.info("           (单档 ≈ {}; 按 settle/读出/npz 落盘/优化步的实测区间折算)",
                _fmt_minutes(plan["timing"]["per_group_s"]))
    logger.info(bar)
    logger.info("输出布局:")
    logger.info("  {}/run_config.json", "<out>")
    logger.info("  {}/journal.jsonl   (逐行 flush)", "<out>")
    logger.info("  {}/flat/iter_XXXX.npz", "<out>")
    logger.info("  {}/lr_0p02|0p05|0p08|0p12/iter_XXXX.npz", "<out>")
    logger.info(
        "  npz 键: frame(首帧原始 CCD, 2D, 不暗减不归一化) / frames(frames>1 时的"
        "全部原始帧) / phase_model(float32) / exposure_ms / lr / iter / loss"
    )
    logger.info(
        "  journal.jsonl 每帧一行: lr/iter/kind/flat/exposure_ms/peak/saturated/"
        "counts/centroid/fwhm/loss (flat 行的 lr 为 null)"
    )
    logger.info(bar)
    logger.info("上机前提 (--execute): ")
    logger.info("  · SLM 与 CCD 的 USB 句柄独占 —— 开跑前确认没有别的会话在跑硬件 sweep")
    logger.info("  · 曝光继承 {} ms; 若饱和只降不升, 下限 {} ms",
                plan["exposure_ms_start"], MIN_EXPOSURE_MS)
    logger.info("  · 光路必须已在焦: 本脚本不做 focus sweep, 直接复用存档的焦面补偿")
    logger.info("  · 焦面补偿会转向 (等价偏心透镜), 零级位置随之漂移; "
                "质心判读请以当次 flat 为参考")
    logger.info(bar)


def assert_no_drivers() -> None:
    """dry-run 的硬保证: 收尾断言 ``ao_shaping.drivers`` 没被 import。"""
    leaked = sorted(name for name in sys.modules if name.startswith("ao_shaping.drivers"))
    if leaked:
        sys.stderr.write(
            "DRY-RUN 违约: 计划打印过程中导入了驱动包:\n  " + "\n  ".join(leaked) + "\n"
        )
        sys.stderr.flush()
        raise SystemExit(EXIT_DRYRUN_VIOLATION)
    logger.info(
        "dry-run 保证通过: sys.modules 里没有任何 ao_shaping.drivers.* "
        "(未 import、未 open 任何设备)"
    )


# ---------------------------------------------------------------------------
# 采集侧的小工具
# ---------------------------------------------------------------------------
class Journal:
    """``journal.jsonl``: 一行一帧, 每行都 flush。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = path.open("a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self._handle.write(
            json.dumps(record, ensure_ascii=False, default=str) + "\n"
        )
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def cheap_frame_stats(frame: np.ndarray, full_scale: float) -> dict[str, Any]:
    """峰值 / 总 counts / 饱和像素数 (不做平滑, 便宜)。"""
    arr = np.asarray(frame)
    total = float(arr.sum())
    sat_pixels = (
        int(np.count_nonzero(arr >= full_scale)) if full_scale > 0.0 else 0
    )
    n_pixels = int(arr.size) if arr.size else 1
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "peak": float(arr.max()) if arr.size else 0.0,
        "counts": total,
        "saturated": sat_pixels > 0,
        "saturated_pixels": sat_pixels,
        "saturation_fraction": sat_pixels / float(n_pixels),
    }


def spot_metrics(frame: np.ndarray) -> dict[str, float]:
    """质心与等效 FWHM (复用 frame_analysis 的共享实现)。"""
    measurement = measure_spot(np.asarray(frame, dtype=np.float64))
    return {
        "centroid_x": float(measurement.centroid_x),
        "centroid_y": float(measurement.centroid_y),
        "fwhm": float(measurement.fwhm_px),
        "smoothed_peak": float(measurement.peak),
    }


class HardwareRunner:
    """一个 lr 扫描会话: 上屏、采帧、落盘。设备对象由外部打开后传入。"""

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        inputs: ModelInputs,
        plan: dict[str, Any],
        out_dir: Path,
        slm: Any,
        cam: Any,
        rotator: Any,
        phase_to_gray: Any,
        journal: Journal,
    ) -> None:
        self.args = args
        self.inputs = inputs
        self.plan = plan
        self.out_dir = out_dir
        self.slm = slm
        self.cam = cam
        self.rotator = rotator
        self.phase_to_gray = phase_to_gray
        self.journal = journal
        self.captured = set(plan["captured_iters"])
        self.flats = set(plan["flat_marks"])
        self.iters = int(plan["iters"])
        self.group_captures = 0
        self.last_iter = 0

    # -- 设备动作 ----------------------------------------------------------
    def set_exposure(self, ms: float) -> float:
        """写曝光并回读生效值 (只降不升由调用方保证)。"""
        effective = float(self.cam.reset_exposure_time(float(ms)) or ms)
        if effective <= 0.0:
            effective = float(ms)
        return effective

    def effective_exposure(self) -> float:
        value = float(self.cam.get_exposure_time_ms())
        return value if value > 0.0 else 0.0

    def display(self, panel_phase: np.ndarray) -> None:
        """面板弧度相位 → 灰度 → 换槽上屏 (含 stable wait)。"""
        gray = self.phase_to_gray(panel_phase, slm=self.slm)
        self.rotator.display_data(self.slm, gray, float(self.args.settle_s))

    def display_model_phase(self, phase_model: np.ndarray) -> None:
        panel = (
            phase_model_to_panel(phase_model, self.inputs.spec, PANEL_SHAPE)
            + self.inputs.compensation
        )
        self.display(panel)

    def restore_flat(self) -> None:
        """恢复平相位 (仍带焦面补偿), 让 SLM 不留在实验图案上。"""
        self.display(self.inputs.compensation)
        logger.info("已恢复平相位 (面板 = 焦面补偿)")

    # -- 采集 --------------------------------------------------------------
    def grab_frames(self) -> tuple[list[np.ndarray], list[dict[str, Any]], float]:
        """settle 后丢 discard 帧再采 frames 帧; 饱和则降曝光重采。"""
        frames = int(self.args.frames)
        discard = int(self.args.discard)
        for attempt in range(MAX_SATURATION_RETRIES + 1):
            for _ in range(discard):
                self.cam.get_numpy_image(n_sample=1)
            raw = [
                np.asarray(self.cam.get_numpy_image(n_sample=1))
                for _ in range(frames)
            ]
            full_scale = float(self.cam.effective_full_scale(raw[0]))
            stats = [cheap_frame_stats(frame, full_scale) for frame in raw]
            sat_frac = max(item["saturation_fraction"] for item in stats)
            if max(item["counts"] for item in stats) <= 0.0:
                raise RunAborted(
                    "全黑帧: 总 counts = 0 (光没到相机或曝光过低)",
                    {"sat_frac": sat_frac, "full_scale": full_scale},
                )
            if sat_frac >= ABORT_SATURATION_FRACTION:
                raise RunAborted(
                    f"饱和像素比例 {sat_frac:.4%} >= "
                    f"{ABORT_SATURATION_FRACTION:.2%}",
                    {"sat_frac": sat_frac, "full_scale": full_scale},
                )
            if sat_frac > 0.0 and attempt < MAX_SATURATION_RETRIES:
                current = self.effective_exposure()
                target = max(current * EXPOSURE_DECREASE_FACTOR, MIN_EXPOSURE_MS)
                if target < current:
                    effective = self.set_exposure(target)
                    logger.warning(
                        "检测到饱和 ({:.4%}), 曝光降至 {} ms (只降不升)",
                        sat_frac,
                        effective,
                    )
                    continue
            if sat_frac > 0.0:
                logger.warning(
                    "仍有 {:.4%} 饱和像素 (低于中止线, 曝光已在 {:.3f} ms), 继续采集",
                    sat_frac,
                    self.effective_exposure(),
                )
            return raw, stats, full_scale
        return raw, stats, full_scale

    def capture(
        self,
        *,
        kind: str,
        lr: float,
        iter_index: int,
        phase_model: np.ndarray,
        loss: float,
        out_path: Path,
    ) -> None:
        """上屏 → 采帧 → 存原始帧 + 模型相位 → 逐帧写 journal。"""
        self.display_model_phase(phase_model)
        raw, stats, _ = self.grab_frames()
        exposure_ms = self.effective_exposure()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        stack = np.stack([np.asarray(frame) for frame in raw])
        payload: dict[str, Any] = {
            "frame": stack[0],
            "phase_model": np.asarray(phase_model, dtype=np.float32),
            "exposure_ms": np.float64(exposure_ms),
            "lr": np.float64(lr),
            "iter": np.int64(iter_index),
            "loss": np.float64(loss),
        }
        if stack.shape[0] > 1:
            # 全部原始帧保留在 frames 里; frame 保持 2D, 与渲染脚本的读取口径一致。
            payload["frames"] = stack
        np.savez_compressed(out_path, **payload)
        timestamp = _utc_now()
        for index, (frame, cheap) in enumerate(zip(raw, stats)):
            metrics = spot_metrics(frame)
            self.journal.write(
                {
                    "timestamp_utc": timestamp,
                    "kind": kind,
                    "flat": kind.startswith("flat"),
                    "lr": None if math.isnan(lr) else float(lr),
                    "iter": int(iter_index),
                    "frame_index": int(index),
                    "exposure_ms": float(exposure_ms),
                    "file": str(out_path.relative_to(self.out_dir)),
                    "loss": None if math.isnan(loss) else float(loss),
                    **cheap,
                    **metrics,
                }
            )
        self.group_captures += 1
        logger.debug("{} iter={} lr={:g}: 已落盘 {}", kind, iter_index, lr, out_path.name)

    def capture_flat(self, mark: int, kind: str) -> None:
        zeros = np.zeros_like(self.inputs.init_phase)
        self.capture(
            kind=kind,
            lr=float("nan"),
            iter_index=mark,
            phase_model=zeros,
            loss=float("nan"),
            out_path=self.out_dir / "flat" / f"iter_{mark:04d}.npz",
        )

    # -- 每组 --------------------------------------------------------------
    def make_optimizer(self, lr: float) -> DifferentiableBeamOptimizer:
        return DifferentiableBeamOptimizer(
            self.inputs.target,
            source_amplitude=self.inputs.amplitude,
            lr=float(lr),
            device=str(self.args.device),
            seed=int(self.inputs.seed),
            init_phase=self.inputs.init_phase,
        )

    def run_group(self, lr: float) -> None:
        """一档 lr: 冷启动优化器, 按采集计划上屏 + 采帧。"""
        self.group_captures = 0
        group_dir = self.out_dir / lr_dir_name(lr)
        group_dir.mkdir(parents=True)
        logger.info("=== lr={:g} → {} ===", lr, group_dir.name)
        optimizer = self.make_optimizer(lr)
        if 0 in self.flats:
            self.capture_flat(0, "flat_head")
        for it in range(self.iters):
            self.last_iter = it
            optimizer.update()
            phase = np.asarray(optimizer.current_phase, dtype=np.float64).copy()
            loss_raw = optimizer.last_loss
            loss = float("nan") if loss_raw is None else float(loss_raw)
            if it in self.captured:
                self.capture(
                    kind="optimize",
                    lr=lr,
                    iter_index=it,
                    phase_model=phase,
                    loss=loss,
                    out_path=group_dir / f"iter_{it:04d}.npz",
                )
                logger.info(
                    "lr={:g} iter={:03d}/{:03d} loss={:.3e} (已采集 {})",
                    lr,
                    it,
                    self.iters,
                    loss,
                    self.group_captures,
                )
            if (it + 1) in self.flats and (it + 1) < self.iters:
                self.capture_flat(it + 1, "flat_iter")
        if self.iters in self.flats:
            self.capture_flat(self.iters, "flat_tail")


# ---------------------------------------------------------------------------
# --execute: 真开设备
# ---------------------------------------------------------------------------
def run_execute(args: argparse.Namespace, inputs: ModelInputs, plan: dict[str, Any]) -> int:
    """真正开设备, 按 run_config 里的 lr 网格逐档跑。返回退出码。"""
    # ---- 设备 import 全部锁在 --execute 分支内 -----------------------------
    from ao_shaping.drivers.ccd.common import create_camera
    from ao_shaping.drivers.slm.santec import Santec, SlotRotator
    from ao_shaping.utils.image.exposure_control import (
        configured_daheng_exposure_limit,
    )
    from ao_shaping.utils.slm.phase_display import phase_to_slm_grayscale
    from scripts.slm.hw_differentiable_phase_opt import guarded_device

    out_dir = Path(plan["out_dir"])
    try:
        out_dir.mkdir(parents=True)
    except FileExistsError:
        logger.error("输出目录已存在, 绝不复用: {}", out_dir)
        return EXIT_BAD_INPUT
    (out_dir / "flat").mkdir()
    journal = Journal(out_dir / "journal.jsonl")

    lr_dirs = [lr_dir_name(lr) for lr in plan["lrs"]]
    status: dict[str, dict[str, Any]] = {
        name: {"state": "pending", "reason": "", "captures": 0} for name in lr_dirs
    }
    devices_info: dict[str, Any] = {}

    def write_config(aborted: dict[str, Any] | None = None) -> None:
        config = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "script": "scripts/slm/hw_lr_farfield_sweep_20261127.py",
            "source_run": str(inputs.source_run),
            "lrs": [float(lr) for lr in plan["lrs"]],
            "iters": int(plan["iters"]),
            "params": args.__dict__ | {"out": str(out_dir)},
            "effective_inputs": {
                "exposure_ms": inputs.exposure_ms,
                "seed": inputs.seed,
                "slm_number": inputs.slm_number,
                "slm_wavelength": inputs.wavelength_nm,
                "cam_type": inputs.cam_type,
                "cam_id": inputs.cam_id,
                "pupil_center": list(inputs.pupil_center),
                "pupil_radius": inputs.pupil_radius,
                "k_mean": inputs.k_mean,
                "focal_mm": "inf" if math.isinf(inputs.focal_mm) else inputs.focal_mm,
                "target": inputs.target_info,
            },
            "plan": plan,
            "devices": devices_info,
            "lr_status": {name: dict(value) for name, value in status.items()},
            "aborted": aborted,
        }
        (out_dir / "run_config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    write_config()

    ceiling = configured_daheng_exposure_limit()
    start_exposure = float(inputs.exposure_ms)
    if ceiling:
        start_exposure = min(start_exposure, float(ceiling))
    logger.info("起始曝光 {} ms (驱动保护上限 {})", start_exposure, ceiling)

    slm_factory = partial(
        Santec,
        slm_number=inputs.slm_number,
        video_mode=0,
        wavelength=inputs.wavelength_nm,
    )
    cam_factory = partial(
        create_camera,
        inputs.cam_type,
        inputs.cam_id,
        exposure_time_ms=start_exposure,
    )

    exit_code = EXIT_OK
    try:
        with guarded_device("SLM", slm_factory) as slm:
            devices_info["slm_serial"] = str(slm.get_serial_number())
            logger.info("SLM #{} 序列号: {}", inputs.slm_number, slm.get_serial_number())
            with guarded_device("CCD", cam_factory) as cam:
                devices_info["ccd_sn"] = getattr(cam, "sn", None)
                devices_info["bit_depth"] = getattr(cam, "bit_depth", None)
                devices_info["panel_shape"] = list(PANEL_SHAPE)
                logger.info("CCD 序列号: {}, 位深: {}",
                            devices_info["ccd_sn"], devices_info["bit_depth"])
                rotator = SlotRotator(slm)
                runner = HardwareRunner(
                    args=args,
                    inputs=inputs,
                    plan=plan,
                    out_dir=out_dir,
                    slm=slm,
                    cam=cam,
                    rotator=rotator,
                    phase_to_gray=phase_to_slm_grayscale,
                    journal=journal,
                )
                runner.set_exposure(start_exposure)
                write_config()

                current = lr_dirs[0] if lr_dirs else ""
                try:
                    for lr, name in zip(plan["lrs"], lr_dirs):
                        current = name
                        status[name]["state"] = "running"
                        write_config()
                        runner.run_group(float(lr))
                        status[name]["state"] = "completed"
                        status[name]["captures"] = runner.group_captures
                        status[name]["reason"] = ""
                        logger.info(
                            "=== lr={:g} 完成: {} 次采集 ===",
                            lr,
                            runner.group_captures,
                        )
                        write_config()
                except RunAborted as exc:
                    status[current]["state"] = "aborted"
                    status[current]["reason"] = exc.reason
                    status[current]["captures"] = runner.group_captures
                    for name in lr_dirs:
                        if status[name]["state"] == "pending":
                            status[name]["state"] = "skipped"
                            status[name]["reason"] = "上游 lr 组中止"
                    write_config(
                        {
                            "reason": exc.reason,
                            "detail": exc.detail,
                            "lr_dir": current,
                            "iter": runner.last_iter,
                        }
                    )
                    logger.error("采集中止 ({}): {}", current, exc.reason)
                    exit_code = EXIT_ABORTED
                finally:
                    runner.restore_flat()
    finally:
        journal.close()
        write_config()
        logger.info("run_config 与 journal 已落盘: {}", out_dir)

    return exit_code


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    """入口: 默认 dry-run。"""
    args = parse_args(argv)
    inputs = load_model_inputs(args)
    out_dir = resolve_out_dir(args)
    plan = build_plan(args, inputs, out_dir)
    print_plan(plan, inputs)

    if not args.execute:
        assert_no_drivers()
        return EXIT_OK
    return run_execute(args, inputs, plan)


if __name__ == "__main__":
    raise SystemExit(main())
