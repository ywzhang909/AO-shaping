#!/usr/bin/env python3
"""硬件物理预测模型保真度阶梯 (单次开关设备)。

把一组**相位图样**依次上屏到 SLM #1, 用 CCD 抓远场, 对每张图样算一次
**物理预测模型预测** (``fftshift(fft2(A·e^{iφ}))``, 复用
``hw_differentiable_phase_opt.model_far_field`` L3294-3300), 再报告「预测 vs
实测」的一致性 (信号区 Spearman 为主判据, 另附覆盖掩膜内 Pearson、径向剖面
相关、FWHM 比、峰值偏移)。目的是**验证 suspected missing defocus term**: 如果
物理预测模型缺了一项离焦 (菲涅尔) 相位, 那么带离焦的图样 (``defocus_z4_*``) 的一致性
会系统性偏低, 而 ``--predict-from runtime`` (把焦面补偿项加进预测) 会把它拉回来。

判据口径: 远场是 sinc²/Airy 型, 覆盖区里 **>99.9%** 的格在 CCD 上只是背景+读出
噪声, 它们的**秩**是随机的 —— 全域 Spearman 会被噪声底主宰 (dry-run 自检只有
0.002~0.07, 本该 ≈1)。所以主判据是 ``spearman_signal``: 只在「预测强度 ≥
``--signal-floor`` × 预测峰值」的格上算秩相关 (门限取自**预测**, 不取实测, 否则
硬件多出来的弥散光会自己把判据放宽)。取值依据与自检数据见
``_hw_fidelity_ladder_sections.DEFAULT_SIGNAL_FLOOR``。

🔴 **硬件纪律** (见 docs/dev/hardware_discipline.md):

* 整个阶梯**只开/关一次** SLM + 相机 —— 背靠背开关 SLM 会触发驱动冻结
  (memory: ``ReadSU=-1000``)。设备在 :func:`run_execute` 里用**嵌套**
  ``guarded_device`` 打开 (SLM 外层、CCD 内层), ``finally`` 释放, 全程写
  ``device_log.json`` (镜像 ``absolute_fidelity_probe_20261116.py`` 口径)。
* 上屏一律走 ``phase_to_slm_grayscale(..., slm=slm)`` + ``SlotRotator`` 换槽,
  **不**调 ``SLM_Ctrl_WriteWL``。
* ``--wait-until-idle`` 复用 ``run_hw_sweep_groups.wait_until_idle`` 的公平
  仲裁 (先看进程再探针, 只按 pid 排除自己)。

🔴 **离线可演练**: ``--dry-run`` (默认, 无 ``--execute``) **不打开任何设备** ——
用合成相位 + 替身实测帧 (``absolute_fidelity_12bit_fixed_20261116`` 的存档帧)
跑完整条 metrics/figures 流水线, 收尾用 ``assert_no_drivers`` 把「没碰驱动」
变成可执行保证 (违约 exit 3)。硬件会话上跑的代码就是这里演练过的代码。

用法::

    # 离线演练 (现在就能跑, 设备离线也安全)
    python scripts/slm/hw_forward_model_fidelity_ladder.py --dry-run

    # 下一次设备会话: 真开设备跑阶梯 (只开/关一次)
    python scripts/slm/hw_forward_model_fidelity_ladder.py --execute \
        --wait-until-idle 600 --focus-compensate

纯数值逻辑 (图样构建、重采样、指标、绘图、落盘) 全部在
``_hw_fidelity_ladder_sections.py`` (零驱动导入); 本文件只放 CLI / 设备编排 /
计划。复用清单与 file:line 见该模块的模块级 docstring。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# 路径自举: 让 ``import _hw_fidelity_ladder_sections`` 与 ``scripts.*`` 可用。
# sections 模块自身还会把 ROOT/src/scripts 补进 sys.path (它的 module-level)。
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_ROOT = _HERE.parents[1]
for _p in (str(_ROOT), str(_ROOT / "src"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 路径自举之后才能 import (E402 已在 scripts/* 的 per-file-ignores 里)。
import _hw_fidelity_ladder_sections as sec
from _hw_fidelity_ladder_sections import (
    BUILTIN_PATTERN_IDS,
    SYNTH_FRAME_SHAPE,
    clock,
    sha256_file,
)

# ---------------------------------------------------------------------------
# 退出码 (与 hw_differentiable_phase_opt.py L190-197 同口径)
# ---------------------------------------------------------------------------
EXIT_OK = 0
EXIT_DEVICE_HANG = 2
EXIT_DRYRUN_VIOLATION = 3
EXIT_PATH_BLOCKED = 4
EXIT_NOT_IN_FOCUS = 5
EXIT_MODEL_FAILED = 6
EXIT_BENCH_BUSY = 7

# ---------------------------------------------------------------------------
# 默认路径
# ---------------------------------------------------------------------------
DEFAULT_GEOMETRY_NPZ = _ROOT / "data" / "slm_geometry_calib.npz"
DEFAULT_OUT_ROOT = _ROOT / "data" / "debug" / "differentiable_opt"
DEFAULT_STANDIN_DIR = (
    _ROOT
    / "data"
    / "debug"
    / "slm_transfer"
    / "absolute_fidelity_12bit_fixed_20261116"
)
#: 只在 ``--execute`` 真上硬件时才把关键产物复制进报告目录 (现在只放占位
#: README, **不**伪造结果)。
REPORT_ARTIFACT_DIR = (
    _ROOT
    / "report"
    / "slm"
    / "differentiable_opt"
    / "artifacts"
    / "fidelity_ladder_20261116"
)
#: 复制进报告目录的关键产物 (存在才复制; 相对运行目录)。
REPORT_ARTIFACT_FILES = (
    "metrics.json",
    "metrics.csv",
    "ladder_summary.md",
    "plan.json",
    "device_log.json",
    "figures/fidlad_radial_profiles.png",
    "figures/fidlad_correlation_bars.png",
)


def _fmt(v: Any, spec: str = ".3f") -> str:
    """日志用的小数格式化 (None / 非有限 → 占位)。"""
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(f):
        return "nan"
    return format(f, spec)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "硬件物理预测模型保真度阶梯: 一组相位图样依次上屏 → CCD 抓远场 → "
            "逐图样算物理预测模型预测 → 报告预测 vs 实测一致性 (验证缺失离焦项)。"
            "默认 dry-run (不开设备); --execute 才真开 SLM+CCD (只开/关一次)。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- 运行模式 ---
    p.add_argument(
        "--execute",
        action="store_true",
        help="真开设备跑阶梯 (SLM #--slm-number + CCD)。默认关 = dry-run。",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="显式声明离线演练 (与缺省 --execute 等价; 仅用于把意图写清楚)。",
    )
    p.add_argument(
        "--wait-until-idle",
        type=float,
        default=0.0,
        help="开跑前等设备空闲的预算 (秒, 默认 0=不等)。复用 "
        "run_hw_sweep_groups.wait_until_idle 的公平仲裁, 只按 pid 排除自己。",
    )
    p.add_argument(
        "--out",
        default=None,
        help="运行目录 (默认 data/debug/differentiable_opt/fidelity_ladder_<时间戳>)。",
    )

    # --- 设备 ---
    p.add_argument("--slm-number", type=int, default=1, help="SLM 编号 (默认 1)。")
    p.add_argument(
        "--slm-wavelength", type=int, default=1064, help="SLM 波长 nm (默认 1064)。"
    )
    p.add_argument("--cam-type", default="daheng", help="相机类型 (默认 daheng)。")
    p.add_argument("--cam-id", type=int, default=0, help="相机索引 (默认 0)。")

    # --- 采集 ---
    p.add_argument(
        "--exposure-ms", type=float, default=2.0, help="起始曝光 ms (默认 2.0)。"
    )
    p.add_argument(
        "--exposure-ceiling-ms",
        type=float,
        default=200.0,
        help="自动曝光上限 ms (默认 200; 与驱动保护上限取小)。",
    )
    p.add_argument("--frames", type=int, default=3, help="平均帧数 (默认 3)。")
    p.add_argument("--discard", type=int, default=2, help="丢弃帧数 (默认 2)。")
    p.add_argument(
        "--settle-s", type=float, default=0.5, help="上屏后稳定等待 s (默认 0.5)。"
    )

    # --- 几何 / 模型网格 ---
    p.add_argument(
        "--geometry-npz",
        default=str(DEFAULT_GEOMETRY_NPZ),
        help="几何标定 npz (默认 data/slm_geometry_calib.npz)。--execute 下给 "
        "'live' 则现场用 SLMCCDCalibrator 实测。",
    )
    p.add_argument(
        "--n-pupil",
        type=int,
        default=None,
        help="瞳面采样数 N (默认: dry-run=128 对齐替身控制网格, execute=256)。",
    )
    p.add_argument(
        "--model-padding",
        type=int,
        default=4,
        help="模型网格零填充倍数 M = padding·N (默认 4)。",
    )
    p.add_argument(
        "--pupil-center",
        default="960,600",
        help='瞳面中心 (面板像元 "x,y", 默认 "960,600")。',
    )
    p.add_argument(
        "--pupil-radius",
        type=float,
        default=480.0,
        help="瞳面半径 面板像元 (默认 480)。",
    )
    p.add_argument(
        "--beam-w0-px",
        type=float,
        default=260.0,
        help="照明高斯腰径 面板像元 (默认 260)。",
    )
    p.add_argument(
        "--target",
        default="square",
        choices=("square", "delta"),
        help="gs_init 图样的目标远场形状 (默认 square)。",
    )
    p.add_argument(
        "--target-side-px",
        type=float,
        default=60.0,
        help="方形目标边长 CCD px (默认 60)。",
    )
    p.add_argument(
        "--delta-spot-px",
        type=float,
        default=15.0,
        help="delta 目标亮斑直径 CCD px (默认 15)。",
    )

    # --- 图样阶梯 ---
    p.add_argument(
        "--patterns",
        default="all",
        help="图样列表 (逗号分隔): 内置名 / 'all' / 'name=path.npy' / 裸 'path.npy' "
        "/ 'dir:路径'。内置: " + ", ".join(BUILTIN_PATTERN_IDS) + "。",
    )
    p.add_argument(
        "--custom-phase-dir",
        default=None,
        help="自定义相位目录 (其下 *.npy 各成一个图样; 等价于追加一个 dir: token)。",
    )
    p.add_argument(
        "--optimized-phase",
        default=None,
        help="optimized_final 图样的相位 npy 路径 (默认在 --optimized-search-root 下"
        "按修改时间取最新的 phase_model.npy / phase_final.npy)。",
    )
    p.add_argument(
        "--optimized-search-root",
        default=str(DEFAULT_OUT_ROOT),
        help="optimized_final 相位的搜索根目录 (默认 data/debug/differentiable_opt)。",
    )
    p.add_argument(
        "--zernike-amp",
        type=float,
        default=1.0,
        help="Zernike 图样 (defocus/tilt) 的口径内 RMS 振幅 rad (默认 1.0)。",
    )
    p.add_argument(
        "--gs-iters",
        type=int,
        default=50,
        help="gs_init 的 GS 交替投影迭代数 (默认 50)。",
    )
    p.add_argument("--seed", type=int, default=0, help="随机种子 (默认 0)。")
    p.add_argument(
        "--resample-foreign-phase",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="历史优化产物 (phase_model.npy) 的网格口径与本次 --n-pupil 不一致时, "
        "是否按「中心锚定最近邻」把它换采样到本次网格 (默认开; 前提是两边 "
        "--model-padding / --pupil-radius 同默认值)。关掉则跳过该图样并在 "
        "ladder_summary.md 写明原因。",
    )

    # --- 预测口径 ---
    p.add_argument(
        "--signal-floor",
        type=float,
        default=sec.DEFAULT_SIGNAL_FLOOR,
        help="spearman_signal (主判据) 的信号门限, 占覆盖区内预测峰值的比例 "
        f"(默认 {sec.DEFAULT_SIGNAL_FLOOR:g})。远场是 sinc²/Airy 型, 覆盖区里 "
        ">99.9%% 的格在 CCD 上只是背景+读出噪声, 它们的秩是随机的; 不设门限的 "
        "spearman 会被噪声底主宰 (dry-run 自检只有 0.002~0.07, 本该 ≈1)。取值"
        "依据见 _hw_fidelity_ladder_sections.DEFAULT_SIGNAL_FLOOR。",
    )
    p.add_argument(
        "--predict-from",
        default="model",
        choices=("model", "runtime"),
        help="前向预测口径: 'model' = 裸 fft2(A·e^{iφ}) (当前物理预测模型); "
        "'runtime' = 在此基础上叠加焦面补偿 (离焦/菲涅尔) 项 —— 直接检验"
        "「缺失离焦项」假设。dry-run 无设备 ⇒ 无补偿 ⇒ 两者相同 (summary 注明)。",
    )
    p.add_argument(
        "--focus-compensate",
        action="store_true",
        help="--execute 下先做焦面扫描, 把最佳焦面的菲涅尔补偿叠到每张图样上 "
        "(与 hw_differentiable_phase_opt 同口径)。dry-run 忽略。",
    )
    p.add_argument(
        "--focus-points", type=int, default=13, help="焦面粗扫点数 (默认 13)。"
    )
    p.add_argument(
        "--focus-diopter-range",
        type=float,
        default=6.0,
        help="焦面粗扫 1/f 半宽 m^-1 (默认 6)。",
    )
    p.add_argument(
        "--focus-refine", type=int, default=7, help="焦面细扫点数 (默认 7)。"
    )
    p.add_argument(
        "--focus-fwhm-max",
        type=float,
        default=40.0,
        help="在焦判定 FWHM 门限 CCD px (默认 40); 超过 ⇒ exit 5 拒绝测量。",
    )

    # --- dry-run 替身 ---
    p.add_argument(
        "--standin-dir",
        default=str(DEFAULT_STANDIN_DIR),
        help="dry-run 替身实测帧目录 (默认 absolute_fidelity_12bit_fixed_20261116)。",
    )
    return p


def _parse_centre(text: str) -> tuple[float, float]:
    """解析 ``--pupil-center "x,y"`` (面板像元)。"""
    try:
        x_str, y_str = str(text).split(",")
        return (float(x_str), float(y_str))
    except Exception as exc:  # 宽捕获: 任何解析失败都变成明确的 CLI 报错
        raise SystemExit(
            f'--pupil-center 需要两个数 "x,y" (面板像元), 给的是 {text!r}'
        ) from exc


# ---------------------------------------------------------------------------
# 公共准备 (离线, dry-run 与 execute 共用)
# ---------------------------------------------------------------------------
class Ladder:
    """一次阶梯运行的离线公共件 (模型网格 + 图样 + 源振幅)。"""

    def __init__(
        self,
        *,
        opt: Any,
        spec: Any,
        amplitude: np.ndarray,
        target: np.ndarray,
        target_info: dict[str, Any],
        patterns: list[sec.Pattern],
        skipped: list[dict[str, str]],
        panel_shape: tuple[int, int],
        pupil_center: tuple[float, float],
        pupil_radius: float,
    ) -> None:
        self.opt = opt
        self.spec = spec
        self.amplitude = amplitude
        self.target = target
        self.target_info = target_info
        self.patterns = patterns
        self.skipped = skipped
        self.panel_shape = panel_shape
        self.pupil_center = pupil_center
        self.pupil_radius = pupil_radius


def prepare_ladder(args: argparse.Namespace, opt: Any, *, n_pupil: int) -> Ladder:
    """建模型网格 + 源振幅 + 目标 + 图样阶梯 (纯离线, 不碰设备)。"""
    panel_shape = (int(opt.SLM_PANEL_H), int(opt.SLM_PANEL_W))
    pupil_center = _parse_centre(args.pupil_center)
    pupil_radius = float(args.pupil_radius)

    spec = opt.build_model_spec(
        pupil_radius=pupil_radius,
        pupil_center=pupil_center,
        n_pupil=int(n_pupil),
        padding=int(args.model_padding),
        w0_px=float(args.beam_w0_px),
    )
    amplitude = opt.build_source_amplitude(spec)
    # target 的远场标度依赖 k_mean; 几何可能在 execute 现场才实测, 这里先用
    # 占位 k_mean=1.0 建一个, 拿到真几何后由 rebuild_target 重建。
    target, target_info = opt.build_target(
        spec, str(args.target), float(args.target_side_px), 1.0, float(args.delta_spot_px)
    )
    tokens = sec.parse_pattern_tokens(args.patterns, args.custom_phase_dir)
    patterns, skipped = sec.build_patterns(
        tokens=tokens,
        opt_mod=opt,
        spec=spec,
        amplitude=amplitude,
        target=target,
        zernike_amp=float(args.zernike_amp),
        gs_iters=int(args.gs_iters),
        seed=int(args.seed),
        optimized_phase=args.optimized_phase,
        optimized_search_root=Path(args.optimized_search_root),
        panel_shape=panel_shape,
        allow_resample=bool(args.resample_foreign_phase),
    )
    return Ladder(
        opt=opt,
        spec=spec,
        amplitude=amplitude,
        target=target,
        target_info=target_info,
        patterns=patterns,
        skipped=skipped,
        panel_shape=panel_shape,
        pupil_center=pupil_center,
        pupil_radius=pupil_radius,
    )


def rebuild_target(ladder: Ladder, args: argparse.Namespace, k_mean: float) -> None:
    """拿到真几何 (k_mean) 后重建目标远场 (delta 的 spot 尺寸依赖远场标度)。"""
    target, target_info = ladder.opt.build_target(
        ladder.spec,
        str(args.target),
        float(args.target_side_px),
        float(k_mean),
        float(args.delta_spot_px),
    )
    ladder.target = target
    ladder.target_info = target_info


# ---------------------------------------------------------------------------
# 逐图样数值处理 (dry-run 与 execute 共用)
# ---------------------------------------------------------------------------
def process_pattern(
    *,
    ladder: Ladder,
    pattern: sec.Pattern,
    meas_frame: np.ndarray,
    centre_xy: tuple[float, float],
    k_mean: float,
    comp_model: np.ndarray | None,
    predict_from: str,
    signal_floor: float = sec.DEFAULT_SIGNAL_FLOOR,
) -> tuple[dict[str, Any], np.ndarray]:
    """单图样: 预测 → 重采样实测 → 指标。返回 ``(row, pred)``。

    ``comp_model`` 是焦面补偿在模型网格上的相位 (M×M); ``predict_from='runtime'``
    时把它加进预测相位 —— 这正是「缺失离焦项」假设的对照: model 不带、runtime 带。
    dry-run 无设备 ⇒ ``comp_model is None`` ⇒ runtime ≡ model。
    """
    opt = ladder.opt
    spec = ladder.spec
    block = np.asarray(pattern.phase_block, dtype=np.float64)
    if not np.all(np.isfinite(block)):
        bad = int(np.count_nonzero(~np.isfinite(block)))
        raise ValueError(
            f"图样 {pattern.pattern_id} 的相位块含 {bad} 个非有限值 "
            "(NaN/Inf); 前向预测会整片变 NaN, 拒绝继续"
        )
    embedded = sec.embed_block(spec, block)
    if predict_from == "runtime" and comp_model is not None:
        pred = opt.model_far_field(ladder.amplitude, embedded + comp_model)
    else:
        pred = opt.model_far_field(ladder.amplitude, embedded)

    meas_grid, cov = sec.resample_with_coverage(opt, meas_frame, centre_xy, k_mean, spec)
    metrics = sec.pattern_metrics(
        opt,
        pred=pred,
        meas=meas_grid,
        coverage=cov,
        k_mean=k_mean,
        spec=spec,
        signal_floor=float(signal_floor),
    )
    stats = sec.phase_stats(np.asarray(pattern.phase_block, dtype=np.float64))
    row: dict[str, Any] = {
        "pattern_id": pattern.pattern_id,
        "source": pattern.source,
        "note": pattern.note,
        "standin_id": pattern.standin_id,
        "predict_from": predict_from,
        "phase_rms_rad": stats["rms_rad"],
        "phase_ptv_rad": stats["ptv_rad"],
        **metrics,
    }
    return row, np.asarray(pred, dtype=np.float64)


def save_pattern_artifacts(
    out_dir: Path,
    opt: Any,
    pattern: sec.Pattern,
    *,
    pred: np.ndarray,
    meas_frame: np.ndarray | None,
    raw_frames: list[np.ndarray] | None = None,
) -> None:
    """落盘单图样产物: 相位 npy、预测 npy、(可选) 平均帧 npy + 原始帧 npz。"""
    phase_dir = out_dir / "phase"
    pred_dir = out_dir / "pred"
    frame_dir = out_dir / "frames"
    phase_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.save(phase_dir / f"{pattern.pattern_id}.npy", np.asarray(pattern.phase_block))
    np.save(pred_dir / f"{pattern.pattern_id}.npy", np.asarray(pred, dtype=np.float32))
    if meas_frame is not None:
        frame_dir.mkdir(parents=True, exist_ok=True)
        np.save(
            frame_dir / f"{pattern.pattern_id}.npy",
            np.asarray(meas_frame, dtype=np.float32),
        )
        # 平均帧另存压缩 npz (与 opt.save_ccd_frame 同口径, 便于报告复制)。
        opt.save_ccd_frame(frame_dir / f"{pattern.pattern_id}_avg.npz", meas_frame)
        if raw_frames:
            np.savez_compressed(
                frame_dir / f"{pattern.pattern_id}_raw.npz",
                frames=np.stack(
                    [np.asarray(f, dtype=np.float32) for f in raw_frames], axis=0
                ),
            )


# ---------------------------------------------------------------------------
# 计划 / 产物
# ---------------------------------------------------------------------------
def _pattern_meta(ladder: Ladder) -> list[dict[str, Any]]:
    """plan.json 里的逐图样元信息 (离线可写)。"""
    meta: list[dict[str, Any]] = []
    for pat in ladder.patterns:
        st = sec.phase_stats(np.asarray(pat.phase_block, dtype=np.float64))
        meta.append(
            {
                "pattern_id": pat.pattern_id,
                "source": pat.source,
                "note": pat.note,
                "standin_id": pat.standin_id,
                "phase_shape": [int(st["shape_rows"]), int(st["shape_cols"])],
                "phase_rms_rad": st["rms_rad"],
                "phase_ptv_rad": st["ptv_rad"],
            }
        )
    return meta


def build_plan(
    args: argparse.Namespace,
    ladder: Ladder,
    *,
    mode: str,
    geometry: dict[str, Any],
    centre_xy: tuple[float, float],
    centre_source: str,
    frame_shape: tuple[int, int],
    standin_dir: Path | None,
    slm_serial: str | None = None,
    cam_sn: str | None = None,
    focus_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """把一次运行的全部决定写进 ``plan.json``。

    键结构同时满足 :func:`_hw_fidelity_ladder_sections.write_summary_md` 的读取
    口径 (``geometry.path/k_mean/rotation_deg``、``model.n_pupil/m_grid/dp_panel_px``、
    顶层 ``centre_xy``、``predict_from``、``timestamp_utc``)。
    """
    spec = ladder.spec
    k_mean = float(geometry.get("k_mean", float("nan")))
    return {
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "purpose": (
            "物理预测模型保真度阶梯: 预测 vs 实测一致性, 验证 suspected missing defocus term"
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "mode": mode,
        "predict_from": str(args.predict_from),
        "signal_floor": float(args.signal_floor),
        "focus_compensate": bool(args.focus_compensate and mode == "execute"),
        "centre_xy": [float(centre_xy[0]), float(centre_xy[1])],
        "centre_source": centre_source,
        "devices": {
            "slm_number": int(args.slm_number),
            "slm_wavelength_nm": int(args.slm_wavelength),
            "cam_type": str(args.cam_type),
            "cam_id": int(args.cam_id),
            "slm_serial": slm_serial,
            "ccd_sn": cam_sn,
        },
        "acquisition": {
            "exposure_ms": float(args.exposure_ms),
            "exposure_ceiling_ms": float(args.exposure_ceiling_ms),
            "frames": int(args.frames),
            "discard": int(args.discard),
            "settle_s": float(args.settle_s),
        },
        "geometry": geometry,
        "model": {
            "n_pupil": int(spec.n_pupil),
            "m_grid": int(spec.m_grid),
            "padding": int(spec.padding),
            "dp_panel_px": float(spec.dp),
            "span_panel_px": float(spec.span),
            "pupil_radius": float(spec.pupil_radius),
            "pupil_center": [float(spec.pupil_center[0]), float(spec.pupil_center[1])],
            "beam_w0_px": float(spec.w0_px),
            "panel_shape": [int(ladder.panel_shape[0]), int(ladder.panel_shape[1])],
            "k_mean": k_mean,
            "far_field_sample_ccd_px": float(spec.far_field_sample_ccd_px(k_mean)),
            "target": ladder.target_info,
        },
        "frame_shape": [int(frame_shape[0]), int(frame_shape[1])],
        "standin": (
            {
                "dir": str(standin_dir),
                "map": dict(sec.STANDIN_MAP),
                "note": "dry-run 替身帧来源; 缺替身的图样走合成帧",
            }
            if standin_dir is not None
            else None
        ),
        "focus": focus_info,
        "patterns": _pattern_meta(ladder),
        "skipped": ladder.skipped,
    }


def finalize_outputs(
    out_dir: Path,
    ladder: Ladder,
    rows: list[dict[str, Any]],
    *,
    mode: str,
    plan: dict[str, Any],
    occupancy: dict[str, Any] | None = None,
) -> None:
    """写 metrics / ladder_summary / 图 / 源码自拷贝 (dry-run 与 execute 共用)。"""
    opt = ladder.opt
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    sec.write_metrics(out_dir, opt, rows)
    radial_entries = [
        (
            r["pattern_id"],
            np.asarray(r.get("radial_r", []), dtype=np.float64),
            np.asarray(r.get("radial_pred", []), dtype=np.float64),
            np.asarray(r.get("radial_meas", []), dtype=np.float64),
        )
        for r in rows
    ]
    sec.plot_radial_profiles(
        radial_entries, out_dir / "figures" / "fidlad_radial_profiles.png"
    )
    sec.plot_correlation_bars(rows, out_dir / "figures" / "fidlad_correlation_bars.png")
    sec.write_summary_md(
        out_dir,
        mode=mode,
        plan=plan,
        rows=rows,
        skipped=ladder.skipped,
        occupancy=occupancy,
    )
    sec.copy_sources(
        out_dir,
        [Path(__file__).resolve(), _HERE / "_hw_fidelity_ladder_sections.py"],
    )
    logger.info(
        "[{}] 产物已写入 {} (metrics.json/csv, ladder_summary.md, figures/fidlad_*)",
        mode,
        out_dir,
    )


# ---------------------------------------------------------------------------
# dry-run: 全程离线, 不打开任何设备
# ---------------------------------------------------------------------------
def run_dry_run(args: argparse.Namespace, opt: Any, ladder: Ladder, out_dir: Path) -> int:
    """离线演练: 合成相位 + 替身/合成实测帧 → 完整 metrics/figures 流水线。"""
    standin_dir = Path(args.standin_dir)
    geometry = sec.load_geometry_npz(Path(args.geometry_npz))
    k_mean = float(geometry["k_mean"])
    rebuild_target(ladder, args, k_mean)

    # 重采样中心: 优先替身运行的同帧定心 spot_xy (替身帧的光斑就在那),
    # 缺则退回几何标定的零级中心。
    standin_spot = sec.load_standin_spot_xy(standin_dir)
    if standin_spot is not None:
        centre_xy = (float(standin_spot[0]), float(standin_spot[1]))
        centre_source = f"替身 plan.json:spot_xy ({standin_dir.name})"
    else:
        centre_xy = (float(geometry["centre_xy"][0]), float(geometry["centre_xy"][1]))
        centre_source = f"几何标定 npz centre ({Path(args.geometry_npz).name})"
    frame_shape = SYNTH_FRAME_SHAPE

    for sub in ("phase", "pred", "frames", "figures"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    geometry_payload = dict(geometry)
    geometry_payload["source"] = f"存档 {args.geometry_npz} (dry-run)"
    geometry_payload["measured_this_run"] = False
    plan = build_plan(
        args,
        ladder,
        mode="dry-run",
        geometry=geometry_payload,
        centre_xy=centre_xy,
        centre_source=centre_source,
        frame_shape=frame_shape,
        standin_dir=standin_dir,
    )
    (out_dir / "plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    logger.info(
        "[dry-run] {} 个图样 · N={} M={} k_mean={:.1f} 中心=({:.0f},{:.0f}) [{}] · 替身 {}",
        len(ladder.patterns),
        ladder.spec.n_pupil,
        ladder.spec.m_grid,
        k_mean,
        centre_xy[0],
        centre_xy[1],
        centre_source,
        standin_dir.name,
    )

    rows: list[dict[str, Any]] = []
    total = len(ladder.patterns)
    for index, pattern in enumerate(ladder.patterns, start=1):
        # 实测帧: 有替身用替身 (真实存档帧), 否则用预测合成一帧 (演练流水线)。
        standin_frame = None
        standin_path = None
        if pattern.standin_id:
            standin_frame, standin_path = sec.load_standin_frame(
                standin_dir, pattern.standin_id
            )
        if standin_frame is not None:
            meas_frame = standin_frame
            measured_source = f"替身帧 {standin_path.name if standin_path else ''}".strip()
        else:
            embedded = sec.embed_block(ladder.spec, pattern.phase_block)
            pred_for_synth = opt.model_far_field(ladder.amplitude, embedded)
            meas_frame = sec.synthetic_ccd_frame(
                pred_for_synth,
                frame_shape=frame_shape,
                centre_xy=centre_xy,
                k_mean=k_mean,
                spec=ladder.spec,
                seed=int(args.seed) + index,
            )
            measured_source = "合成帧 (无替身, 由模型预测生成)"

        row, pred = process_pattern(
            ladder=ladder,
            pattern=pattern,
            meas_frame=meas_frame,
            centre_xy=centre_xy,
            k_mean=k_mean,
            comp_model=None,
            predict_from=str(args.predict_from),
            signal_floor=float(args.signal_floor),
        )
        row["measured_source"] = measured_source
        rows.append(row)
        save_pattern_artifacts(out_dir, opt, pattern, pred=pred, meas_frame=meas_frame)
        logger.info(
            "[dry-run {}/{}] {}: spearman(sig)={} pearson(sig)={} "
            "spearman(cov)={} fwhm比={} [{}]",
            index,
            total,
            pattern.pattern_id,
            _fmt(row.get("spearman_signal")),
            _fmt(row.get("pearson_signal")),
            _fmt(row.get("spearman_coverage")),
            _fmt(row.get("fwhm_ratio")),
            measured_source,
        )

    finalize_outputs(out_dir, ladder, rows, mode="dry-run", plan=plan, occupancy=None)

    # 🔴 dry-run 硬保证: 收尾断言没 import 任何驱动 (违约 exit 3)。
    opt.assert_no_drivers()
    logger.info("[dry-run] 完成: 未打开任何设备 (no --execute); 产物 → {}", out_dir)
    return EXIT_OK


# ---------------------------------------------------------------------------
# execute: 真开设备 (只开/关一次 SLM + CCD)
# ---------------------------------------------------------------------------
def run_execute(args: argparse.Namespace, opt: Any, ladder: Ladder, out_dir: Path) -> int:
    """设备已就绪后跑阶梯。**只开/关一次** SLM + CCD (嵌套 guarded_device)。"""
    # ---- 设备 import 全部锁在 --execute 分支内 (dry-run 永不触发) -------------
    from ao_shaping.drivers.ccd.common import create_camera
    from ao_shaping.drivers.slm.santec import Santec, SlotRotator
    from ao_shaping.tools.slm.calibration.calibration import SLMCCDCalibrator
    from ao_shaping.utils.image.exposure_control import configured_daheng_exposure_limit
    from ao_shaping.utils.slm.phase_display import phase_to_slm_grayscale

    opt.setup_cjk_font()
    for sub in ("phase", "pred", "frames", "figures"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    # ---- P0.5: 等设备空出来 (公平仲裁, 不抢) --------------------------------
    wait_s = float(getattr(args, "wait_until_idle", 0.0) or 0.0)
    if wait_s > 0.0:
        from scripts.slm.run_hw_sweep_groups import wait_until_idle

        logger.info("[occupancy] 等待设备空闲 (预算 {:.0f}s) ...", wait_s)
        code, message = wait_until_idle(
            wait_s,
            slm_number=int(args.slm_number),
            cam_id=int(args.cam_id),
            cam_type=str(args.cam_type),
            exposure_ms=float(args.exposure_ms),
            # ⚠️ 只按 pid 排除自己, 不传 own_paths (会把别的会话误判成自己)。
            own_pids=(os.getpid(),),
        )
        if code != 0:
            logger.error("[occupancy] 设备未等到空闲: {}", message)
            return EXIT_BENCH_BUSY
        logger.info("[occupancy] 设备空闲, 开始采集")

    panel_shape = ladder.panel_shape
    pupil_center = ladder.pupil_center
    pupil_radius = ladder.pupil_radius

    device_log: list[dict[str, Any]] = []
    start_utc = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    logger.info(
        "[occupancy] START 占用 SLM #{} + CCD {}:{} (单次开/关, 阶梯 {} 个图样)",
        args.slm_number,
        args.cam_type,
        args.cam_id,
        len(ladder.patterns),
    )
    occupancy_t0 = time.monotonic()

    def slm_factory() -> Any:
        return Santec(
            slm_number=int(args.slm_number),
            video_mode=0,
            wavelength=int(args.slm_wavelength),
        )

    def cam_factory() -> Any:
        return create_camera(
            args.cam_type, int(args.cam_id), exposure_time_ms=float(args.exposure_ms)
        )

    exit_code = EXIT_DEVICE_HANG
    slm_opens = slm_closes = cam_opens = cam_closes = 0
    try:
        # 🔴 只开/关一次: SLM 外层, CCD 内层, 整段阶梯都在这个 with 里。
        with opt.guarded_device("SLM", slm_factory) as slm:
            slm_opens += 1
            slm_serial = opt.guarded(
                "SLM get_serial_number", slm.get_serial_number, timeout_s=30.0
            )
            device_log.append(
                dict(
                    action="open",
                    device=f"slm#{args.slm_number}",
                    mode="memory (video_mode=0)",
                    serial=slm_serial,
                    **clock(),
                )
            )
            logger.info(
                "[occupancy] SLM #{} 已打开 (序列号 {})", args.slm_number, slm_serial
            )
            with opt.guarded_device("CCD", cam_factory) as cam:
                cam_opens += 1
                cam_sn = getattr(cam, "sn", None)
                device_log.append(
                    dict(
                        action="open",
                        device=f"camera:{args.cam_type}",
                        cam_id=int(args.cam_id),
                        sn=cam_sn,
                        **clock(),
                    )
                )
                logger.info(
                    "[occupancy] CCD {}:{} 已打开 (SN {})",
                    args.cam_type,
                    args.cam_id,
                    cam_sn,
                )

                rotator = SlotRotator(slm)

                def display_phase(phase_rad: np.ndarray) -> int:
                    """面板形状原始弧度相位 → 驱动管线 (灰度) → 换槽显示。"""
                    gray = phase_to_slm_grayscale(phase_rad, slm=slm)
                    return rotator.display_data(slm, gray, float(args.settle_s))

                def read_frame() -> np.ndarray:
                    return np.asarray(cam.get_numpy_image(n_sample=1))

                def set_exposure_ms(ms: float) -> float:
                    return float(cam.reset_exposure_time(float(ms)) or ms)

                device_log.append(
                    dict(action="ready", exposure_ms=float(args.exposure_ms), **clock())
                )

                # 曝光上限: 驱动保护上限与脚本上限取小。
                ceiling = float(configured_daheng_exposure_limit() or 0.0)
                ceiling = (
                    min(ceiling, float(args.exposure_ceiling_ms))
                    if ceiling > 0.0
                    else float(args.exposure_ceiling_ms)
                )

                # ---- 焦面补偿 (可选门禁) ------------------------------------
                compensation: np.ndarray | None = None
                focus_info: dict[str, Any] | None = None
                if args.focus_compensate:
                    logger.info("[focus] 焦面扫描 (在焦判定 + 菲涅尔补偿)")
                    focus_result = opt.sweep_focus(
                        apply_phase=display_phase,
                        read_frame=read_frame,
                        set_exposure_ms=set_exposure_ms,
                        panel_shape=panel_shape,
                        radius_px=pupil_radius,
                        centre_px=pupil_center,
                        coarse_points=int(args.focus_points),
                        diopter_range=float(args.focus_diopter_range),
                        refine_points=int(args.focus_refine),
                        start_exposure_ms=float(args.exposure_ms),
                        floor_ms=opt.FLOOR_EXPOSURE_MS,
                        ceiling_ms=ceiling,
                        frames=int(args.frames),
                        discard=int(args.discard),
                        wavelength_nm=float(args.slm_wavelength),
                        pixel_um=opt.SLM_PITCH_M * 1e6,
                    )
                    device_log.append(
                        dict(
                            action="focus_sweep",
                            best_diopter_m1=focus_result.best_diopter_m1,
                            best_focal_mm=focus_result.best_focal_mm,
                            path_blocked=bool(focus_result.path_blocked),
                            **clock(),
                        )
                    )
                    gate = opt.judge_focus_gate(
                        focus_result, float(args.focus_fwhm_max)
                    )
                    focus_info = {
                        "best_diopter_m1": focus_result.best_diopter_m1,
                        "best_focal_mm": focus_result.best_focal_mm,
                        "path_blocked": bool(focus_result.path_blocked),
                        "blockage_reason": focus_result.blockage_reason,
                        "verdict": gate["verdict"],
                    }
                    if gate["code"] is not None:
                        logger.error("[focus] 拒绝测量: {}", gate["reason"])
                        _write_device_log(out_dir, device_log)
                        return int(gate["code"])
                    compensation = opt.fresnel_panel(
                        float(focus_result.best_focal_mm),
                        radius=int(round(pupil_radius)),
                        pupil_center=(
                            int(round(pupil_center[0])),
                            int(round(pupil_center[1])),
                        ),
                        panel_shape=panel_shape,
                        wavelength_nm=float(args.slm_wavelength),
                        pixel_um=opt.SLM_PITCH_M * 1e6,
                    )
                    opt.plot_focus_sweep(
                        focus_result,
                        out_dir / "figures" / "fidlad_focus_sweep.png",
                        fwhm_max=float(args.focus_fwhm_max),
                    )
                    logger.info(
                        "[focus] 最佳焦面 {} mm ({} m^-1), 已生成菲涅尔补偿",
                        focus_result.best_focal_mm,
                        focus_result.best_diopter_m1,
                    )

                # ---- 几何 (存档 npz 或现场实测) -----------------------------
                geom_arg = str(args.geometry_npz or "").strip().lower()
                if geom_arg in ("live", "", "none"):
                    logger.info("[geometry] 现场实测 (SLMCCDCalibrator)")
                    calibrator = SLMCCDCalibrator(
                        slm, cam, settle_s=float(args.settle_s)
                    )
                    if compensation is not None:
                        slm.set_phase_overlay(compensation)
                    try:
                        raw_calib = opt.guarded(
                            "SLMCCDCalibrator.calibrate",
                            calibrator.calibrate,
                            timeout_s=300.0,
                        )
                    finally:
                        if compensation is not None:
                            slm.set_phase_overlay(None)
                    geometry = {
                        "source": "现场实测 (SLMCCDCalibrator.calibrate)",
                        "measured_this_run": True,
                        "center": np.asarray(raw_calib["center"]).tolist(),
                        "kx": float(raw_calib["Kx"]),
                        "ky": float(raw_calib["Ky"]),
                        "rotation_deg": float(raw_calib["rotation_deg"]),
                        "mirror": bool(raw_calib["mirror"]),
                        "fwhm0": float(raw_calib["fwhm0"]),
                    }
                else:
                    geometry = sec.load_geometry_npz(Path(args.geometry_npz))
                    geometry["source"] = f"存档 {args.geometry_npz}"
                    geometry["measured_this_run"] = False
                    # load_geometry_npz 给 centre_xy=(cx,cy); center 约定是 (cy,cx)。
                    geometry["center"] = [
                        float(geometry["centre_xy"][1]),
                        float(geometry["centre_xy"][0]),
                    ]
                k_mean = 0.5 * (float(geometry["kx"]) + float(geometry["ky"]))
                geometry["k_mean"] = k_mean
                centre_cy, centre_cx = (float(v) for v in geometry["center"])
                centre_xy = (centre_cx, centre_cy)
                device_log.append(
                    dict(
                        action="geometry",
                        source=geometry["source"],
                        k_mean=k_mean,
                        centre_xy=list(centre_xy),
                        **clock(),
                    )
                )
                rebuild_target(ladder, args, k_mean)
                logger.info(
                    "[geometry] {} Kx={:.1f} Ky={:.1f} K={:.1f} 零级=({:.0f},{:.0f})",
                    geometry["source"],
                    geometry["kx"],
                    geometry["ky"],
                    k_mean,
                    centre_cx,
                    centre_cy,
                )

                # 焦面补偿在模型网格上的相位 (runtime 预测用)。
                comp_model: np.ndarray | None = None
                if compensation is not None:
                    comp_model = sec.embed_block(
                        ladder.spec, sec.panel_to_block(compensation, ladder.spec)
                    )

                # ---- 逐图样: 上屏 → 抓帧 → 预测 → 指标 ----------------------
                rows: list[dict[str, Any]] = []
                frame_shape: tuple[int, int] = SYNTH_FRAME_SHAPE
                total = len(ladder.patterns)
                for index, pattern in enumerate(ladder.patterns, start=1):
                    embedded = sec.embed_block(ladder.spec, pattern.phase_block)
                    panel = opt.phase_model_to_panel(
                        embedded, ladder.spec, panel_shape
                    )
                    if compensation is not None:
                        panel = panel + compensation
                    display_phase(panel)
                    device_log.append(
                        dict(
                            action="display",
                            pattern_id=pattern.pattern_id,
                            index=index,
                            **clock(),
                        )
                    )
                    time.sleep(max(0.0, float(args.settle_s)))

                    setting = opt.guarded(
                        f"自动曝光 ({pattern.pattern_id})",
                        lambda: opt.adapt_exposure(
                            set_exposure_ms,
                            read_frame,
                            start_ms=float(args.exposure_ms),
                            floor_ms=opt.FLOOR_EXPOSURE_MS,
                            ceiling_ms=ceiling,
                            full_scale=255.0,
                            discard=int(args.discard),
                        ),
                        timeout_s=opt.ACQUIRE_TIMEOUT_S,
                    )
                    frame, raw = opt.guarded(
                        f"CCD 取帧 ({pattern.pattern_id})",
                        lambda: opt.capture_frames(
                            cam, frames=int(args.frames), discard=int(args.discard)
                        ),
                        timeout_s=opt.ACQUIRE_TIMEOUT_S,
                    )
                    frame_shape = (int(frame.shape[0]), int(frame.shape[1]))
                    device_log.append(
                        dict(
                            action="capture",
                            pattern_id=pattern.pattern_id,
                            index=index,
                            exposure_ms=float(setting.exposure_ms),
                            frames=int(args.frames),
                            peak=float(np.max(frame)),
                            **clock(),
                        )
                    )
                    spot = opt.assess_spot(frame)
                    logger.info(
                        "[{}/{}] {}: 曝光 {} ms ({}) 峰值 {:.1f} counts usable={}",
                        index,
                        total,
                        pattern.pattern_id,
                        setting.exposure_ms,
                        setting.status,
                        float(np.max(frame)),
                        spot.usable,
                    )

                    row, pred = process_pattern(
                        ladder=ladder,
                        pattern=pattern,
                        meas_frame=frame,
                        centre_xy=centre_xy,
                        k_mean=k_mean,
                        comp_model=comp_model,
                        predict_from=str(args.predict_from),
                        signal_floor=float(args.signal_floor),
                    )
                    row["exposure_ms"] = float(setting.exposure_ms)
                    row["spot_usable"] = bool(spot.usable)
                    row["measured_source"] = "CCD 实测"
                    rows.append(row)
                    save_pattern_artifacts(
                        out_dir, opt, pattern, pred=pred, meas_frame=frame, raw_frames=raw
                    )

                # ---- 收尾产物 ----
                plan = build_plan(
                    args,
                    ladder,
                    mode="execute",
                    geometry=geometry,
                    centre_xy=centre_xy,
                    centre_source=geometry["source"],
                    frame_shape=frame_shape,
                    standin_dir=None,
                    slm_serial=slm_serial,
                    cam_sn=cam_sn,
                    focus_info=focus_info,
                )
                (out_dir / "plan.json").write_text(
                    json.dumps(plan, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8",
                )
                end_utc = datetime.now(timezone.utc).isoformat(timespec="microseconds")
                occupancy = {
                    "devices": f"SLM #{args.slm_number} + CCD {args.cam_type}:{args.cam_id}",
                    "start_utc": start_utc,
                    "end_utc": end_utc,
                    "closed_normally": True,
                    "slm_opens": slm_opens,
                    "slm_closes": slm_closes,
                    "cam_opens": cam_opens,
                    "cam_closes": cam_closes,
                }
                finalize_outputs(
                    out_dir, ladder, rows, mode="execute", plan=plan, occupancy=occupancy
                )
                _write_device_log(out_dir, device_log)
                _copy_to_report(out_dir)
                exit_code = EXIT_OK
    finally:
        # guarded_device 的 __exit__ 已经关掉设备; 这里只记数 + 落 device_log。
        slm_closes = slm_opens
        cam_closes = cam_opens
        device_log.append(dict(action="release", device="devices", **clock()))
        _write_device_log(out_dir, device_log)
        elapsed = time.monotonic() - occupancy_t0
        logger.info(
            "[occupancy] END 释放 SLM #{} + CCD {}:{} (占用 {:.1f}s, device_log {} 条)",
            args.slm_number,
            args.cam_type,
            args.cam_id,
            elapsed,
            len(device_log),
        )
    return exit_code


def _write_device_log(out_dir: Path, device_log: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "device_log.json").write_text(
        json.dumps(device_log, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _copy_to_report(out_dir: Path) -> None:
    """只在 --execute 真上硬件后, 把关键产物复制进报告目录 (不伪造结果)。"""
    import shutil

    REPORT_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for rel in REPORT_ARTIFACT_FILES:
        src = out_dir / rel
        if src.is_file():
            dst = REPORT_ARTIFACT_DIR / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(rel)
    logger.info(
        "[report] 已复制 {} 个关键产物 → {}", len(copied), REPORT_ARTIFACT_DIR
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.execute and args.dry_run:
        raise SystemExit("--execute 与 --dry-run 互斥")

    opt = sec.load_opt_module()
    opt.setup_cjk_font()

    # n_pupil: 显式优先; 否则 dry-run=128 (对齐替身控制网格), execute=256。
    if args.n_pupil is not None:
        n_pupil = int(args.n_pupil)
    else:
        n_pupil = 128 if not args.execute else int(opt.DEFAULT_MODEL_PUPIL_SAMPLES)

    if args.out:
        out_dir = Path(args.out)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = DEFAULT_OUT_ROOT / f"fidelity_ladder_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "运行目录: {} (mode={}, N={})",
        out_dir,
        "execute" if args.execute else "dry-run",
        n_pupil,
    )

    ladder = prepare_ladder(args, opt, n_pupil=n_pupil)
    if not ladder.patterns:
        logger.error("没有可跑的图样 (全部被跳过): {}", ladder.skipped)
        return EXIT_MODEL_FAILED

    if args.execute:
        return run_execute(args, opt, ladder, out_dir)
    return run_dry_run(args, opt, ladder, out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
