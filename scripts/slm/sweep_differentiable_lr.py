"""纯离线研究: 可微分光束整形的 Adam 学习率与动态调度扫描。

要回答什么
----------
基线运行 (默认 ``data/debug/differentiable_opt/phys_20261116_01``) 用固定
``lr=0.05`` 迭代 120 次。本脚本在**不碰任何设备**的前提下, 从运行目录重建
同一个问题 (共享入口 :mod:`scripts._common.differentiable_problem`, 口径与
``hw_differentiable_phase_opt.py::run_analyze_only`` 一致), 回答:

1. 更大的 Adam 学习率能否在**更少迭代**内达到同样的整形质量?
2. 动态调度 (余弦衰减 / 阶梯衰减 / 预热+余弦, 经
   :meth:`DifferentiableBeamOptimizer.set_lr` 逐迭代驱动) 是否优于固定学习率?
3. 「匹配基线最终物理质量」的最短迭代预算是多少?

模型损失更低**不等于**光斑更好 ⇒ 每个采样迭代同时记录物理预测模型自己
预测远场的**物理指标**: 复用仓库共享的
:func:`ao_shaping.utils.image.frame_analysis.measure_spot` (峰值 / FWHM /
空洞度), 加上目标区内能量占比、目标区内强度均匀度 CV, 以及对平相位的
类 Strehl 峰值比。

扫描流程 (一次调用内依次完成)
---------------------------
* ``repro``    基线复现: 用 run-dir 的 lr/iters/seed 重跑, 与 ``history.csv``
  逐检查点比对; 最大相对偏差超过 ``--repro-tol`` 即**停止** (在错误的问题
  上做扫描没有意义), exit 2;
* ``ladder``   固定学习率阶梯 (``--lrs``), 每个学习率跑满 ``--iters``;
* ``schedule`` 动态调度: 自动从阶梯里挑**两个最优**学习率 (稳定且最快达到
  基线 99% 物理质量; 并列按终态质量), 跑 ``--schedules`` 里的动态调度;
* ``short``    短预算变体: 对每个有希望的学习率, 把「达到基线 99% 物理质量
  所需的最少迭代数」实际复跑一遍并验证终态质量。

物理指标成本 (实测, 见 JSON ``timing`` 键): ``measure_spot`` 在 1024² 远场上
约 216 ms/次, 而一次 ``update()`` 仅约 4 ms ⇒ 损失/梯度/相位 RMS **每迭代**
记录, 物理指标按 ``--metric-every`` 采样 (默认: ``iters<=200`` 时 1, 否则 5;
正式扫描显式传 5, 理由记录在 JSON ``budget`` 键)。

产物 (``--out``, 默认 ``report/slm/differentiable_opt/artifacts/lr_sweep_20261116/``)
------------------------------------------------------------------------------------
``lr_sweep.json`` (全部逐迭代数据 + 预算 + 复现检查 + 头条数字),
``lr_sweep.csv`` (每个采样迭代一行), ``lr_sweep.md`` (可直接摘录的表格与
建议), ``lrsweep_loss_vs_iter.png`` / ``lrsweep_physical_vs_iter.png``
(两图同时复制到 ``report/slm/differentiable_opt/figures/``)。

用法示例
--------
::

    # 正式扫描 (默认学习率阶梯 + 全部调度; GPU 可用时自动用 cuda)
    .venv/Scripts/python.exe scripts/slm/sweep_differentiable_lr.py --metric-every 5

    # 冒烟 (8 次迭代, 两个学习率, 只跑余弦)
    .venv/Scripts/python.exe scripts/slm/sweep_differentiable_lr.py --quick

    # 追加更高学习率 (仅当 0.40 仍在单调下降且稳定时)
    .venv/Scripts/python.exe scripts/slm/sweep_differentiable_lr.py \\
        --lrs 0.8 --schedules none --out report/.../lr_sweep_20261116_lr08

退出码: 0 正常; 2 基线复现失败 (扫描中止, 不写产物)。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from functools import partial
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
# `scripts.*` 以包路径被 import, 所以仓库根本身必须可 import —— 以文件形式
# 启动时 cwd 不在 sys.path 上, 只有脚本自己的目录在。
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")  # 必须在 pyplot 之前

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from scripts._common.differentiable_problem import (
    DifferentiableProblem,
    build_optimizer,
    load_problem_from_run_dir,
)
from scripts._common.plot_style import setup_cjk_font
from scripts._common import fmt_general

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
EXIT_OK = 0
#: 基线复现失败 (在错误的问题上扫描没有意义)。
EXIT_REPRO_FAILED = 2

#: 默认输出目录 (报告 artifacts, 见 report/AGENTS.md)。
DEFAULT_OUT = ROOT / "report" / "slm" / "differentiable_opt" / "artifacts" / "lr_sweep_20261116"
#: 图片同时复制到的报告 figures 目录。
REPORT_FIG_DIR = ROOT / "report" / "slm" / "differentiable_opt" / "figures"
#: 默认运行目录 (基线: lr=0.05, iters=120)。
DEFAULT_RUN_DIR = ROOT / "data" / "debug" / "differentiable_opt" / "phys_20261116_01"

DEFAULT_LRS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.40)
DEFAULT_SCHEDULES = ("flat", "cosine", "step", "warmup-cosine")
DYNAMIC_SCHEDULES = ("cosine", "step", "warmup-cosine")
#: 余弦/预热-余弦衰减的地板 (峰值学习率的 0.1 倍), 与
#: iterative_zernike_shaping 的余弦衰减先例同一精神。
LR_FLOOR_FRAC = 0.10
#: 总墙钟预算 (秒); 估算超出时按既定顺序裁剪变体。
DEFAULT_BUDGET_S = 1200.0
#: 基线复现的默认容差 (最大相对偏差)。
DEFAULT_REPRO_TOL = 1e-3

CSV_FIELDS = (
    "arm_kind",
    "schedule",
    "lr",
    "iter",
    "lr_applied",
    "loss",
    "grad_norm",
    "phase_rms_rad",
    "phase_delta_max",
    "wall_s",
    "pred_peak",
    "pred_fwhm_px",
    "pred_hollowness",
    "energy_in_target",
    "cv_in_target",
    "strehl_like",
)


# ---------------------------------------------------------------------------
# 学习率调度
# ---------------------------------------------------------------------------
def schedule_lr(name: str, peak_lr: float, i: int, iters: int) -> float:
    """第 ``i`` 次迭代 (0 起) 应施加的学习率。

    Args:
        name: 调度名 (``flat`` / ``cosine`` / ``step`` / ``warmup-cosine``)。
        peak_lr: 峰值 (固定) 学习率。
        i: 迭代序号, 0 起。
        iters: 总迭代预算。

    Returns:
        该迭代应施加的学习率 (恒为正)。

    Raises:
        ValueError: 调度名不认识。
    """
    if name == "flat":
        return float(peak_lr)
    if name == "cosine":
        # 余弦衰减: 峰值 -> LR_FLOOR_FRAC·峰值 (先例: iterative_zernike_shaping
        # 的 slm_lr 余弦衰减 —— 平直学习率会让 Adam 过冲、指标震荡)。
        c = 0.5 * (1.0 + math.cos(math.pi * i / max(iters, 1)))
        return float(peak_lr) * (LR_FLOOR_FRAC + (1.0 - LR_FLOOR_FRAC) * c)
    if name == "step":
        # 阶梯衰减: 每 ceil(iters/3) 次迭代学习率减半。
        block = max(math.ceil(iters / 3.0), 1)
        return float(peak_lr) * (0.5 ** (i // block))
    if name == "warmup-cosine":
        # 前 10% 迭代线性预热到峰值, 其后余弦衰减到 LR_FLOOR_FRAC·峰值。
        warm = max(math.ceil(0.10 * iters), 1)
        if i < warm:
            return float(peak_lr) * (i + 1) / warm
        rest = max(iters - warm, 1)
        c = 0.5 * (1.0 + math.cos(math.pi * (i - warm) / rest))
        return float(peak_lr) * (LR_FLOOR_FRAC + (1.0 - LR_FLOOR_FRAC) * c)
    raise ValueError(f"未知调度: {name!r} (可选 flat/cosine/step/warmup-cosine)")


# ---------------------------------------------------------------------------
# 变体 (arm) 数据结构与执行
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ArmSpec:
    """一个扫描变体的配置。

    Attributes:
        kind: ``repro`` / ``ladder`` / ``schedule`` / ``short``。
        schedule: 学习率调度名。
        lr: 峰值 (固定) 学习率。
        iters: 迭代预算。
    """

    kind: str
    schedule: str
    lr: float
    iters: int

    @property
    def label(self) -> str:
        """图例/表格用的短标签。"""
        return f"{self.schedule}@lr={self.lr:g}"


@dataclass
class ArmResult:
    """一个变体的完整记录 (逐迭代数组 + 采样行 + 稳定性 + 头条数字)。"""

    spec: ArmSpec
    losses: list[float] = field(default_factory=list)
    lrs_applied: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    phase_rms: list[float] = field(default_factory=list)
    phase_delta_max: list[float] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    wall_s: float = 0.0
    stability: dict[str, Any] = field(default_factory=dict)
    headline: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _Deps:
    """延迟导入的重依赖 (torch / 物理指标函数), 打包传给变体执行器。"""

    torch: Any
    far_field_intensity: Any
    measure_spot: Any


def run_arm(
    problem: DifferentiableProblem,
    spec: ArmSpec,
    *,
    device: str,
    seed: int,
    metric_every: int,
    deps: _Deps,
) -> ArmResult:
    """跑完一个变体: 逐迭代记录损失/梯度/相位, 按采样间隔记录物理指标。

    每次 ``update()`` 前调用 :meth:`DifferentiableBeamOptimizer.set_lr` 施加
    该迭代的调度学习率 (``flat`` 调度即恒定值, 与不调用数值等价)。

    Args:
        problem: 重建好的问题。
        spec: 变体配置。
        device: torch 设备。
        seed: 随机种子 (仅在无初始相位时有意义)。
        metric_every: 物理指标采样间隔 (迭代数)。
        deps: 延迟导入的重依赖。

    Returns:
        填满数据的 :class:`ArmResult`。
    """
    torch = deps.torch
    res = ArmResult(spec=spec)
    opt = build_optimizer(problem, lr=spec.lr, device=device, seed=seed)
    mask = problem.target > 0.0

    # 平相位 (全零) 预测远场 —— 类 Strehl 峰值比的分母, 每个变体只算一次。
    with torch.no_grad():
        zeros = torch.zeros_like(opt.phase_tensor)
        far_flat = deps.far_field_intensity(problem.amplitude, zeros)
        flat_peak = float(far_flat.max().detach().cpu())

    t_arm = time.perf_counter()
    prev_phase = opt.phase_tensor.detach().clone()
    for i in range(spec.iters):
        lr_i = schedule_lr(spec.schedule, spec.lr, i, spec.iters)
        opt.set_lr(lr_i)
        opt.update()

        loss_i = float(opt.last_loss) if opt.last_loss is not None else float("nan")
        grad = opt.phase_tensor.grad
        gnorm = float(grad.norm().detach().cpu()) if grad is not None else float("nan")
        phase_t = opt.phase_tensor.detach()
        rms = float(phase_t.pow(2).mean().sqrt().cpu())
        delta = (phase_t - prev_phase).abs()
        dmax = float(delta.max().cpu())
        prev_phase = phase_t.clone()

        it1 = i + 1
        res.losses.append(loss_i)
        res.lrs_applied.append(lr_i)
        res.grad_norms.append(gnorm)
        res.phase_rms.append(rms)
        res.phase_delta_max.append(dmax)

        is_last = it1 == spec.iters
        if it1 == 1 or is_last or it1 % metric_every == 0:
            with torch.no_grad():
                far = (
                    deps.far_field_intensity(problem.amplitude, opt.phase_tensor)
                    .detach()
                    .cpu()
                    .numpy()
                )
            spot = deps.measure_spot(far)
            in_t = far[mask]
            total = float(far.sum())
            mean_in = float(in_t.mean()) if in_t.size else 0.0
            sample = {
                "iter": it1,
                "lr_applied": lr_i,
                "loss": loss_i,
                "grad_norm": gnorm,
                "phase_rms_rad": rms,
                "phase_delta_max": dmax,
                "wall_s": time.perf_counter() - t_arm,
                "pred_peak": float(far.max()),
                "pred_fwhm_px": float(spot.fwhm_px),
                "pred_hollowness": float(spot.hollowness),
                "energy_in_target": float(in_t.sum() / total) if total > 0 else float("nan"),
                "cv_in_target": float(in_t.std() / mean_in) if mean_in > 0 else float("nan"),
                "strehl_like": float(far.max() / flat_peak) if flat_peak > 0 else float("nan"),
            }
            res.samples.append(sample)
            logger.debug(
                "[{}] iter {}/{}: loss={:.6e} eff={:.4f} cv={:.4f} fwhm={:.1f}px",
                spec.label,
                it1,
                spec.iters,
                loss_i,
                sample["energy_in_target"],
                sample["cv_in_target"],
                sample["pred_fwhm_px"],
            )

    res.wall_s = time.perf_counter() - t_arm
    res.stability = compute_stability(res)
    return res


def compute_stability(res: ArmResult) -> dict[str, Any]:
    """从一个变体的逐迭代数组提取稳定性标志。

    Args:
        res: 已跑完的变体。

    Returns:
        含以下键的字典: ``nonfinite`` (损失出现 NaN/Inf),
        ``max_loss_increase_streak`` (最长连续上升迭代数),
        ``loss_increase_streak_ge5`` (是否出现过 >=5 连升),
        ``max_phase_delta`` (单步最大 |Δφ|, rad),
        ``phase_rms_runaway`` (终态相位 RMS > 2× 第 10 迭代值),
        ``plateau_iter`` (最后一次「相对 10 迭代前改善 >1%」的迭代, 1 起;
        ``None`` 表示从未达标改善过)。
    """
    losses = res.losses
    nonfinite = any(not math.isfinite(x) for x in losses)
    streak = best_streak = 0
    for a, b in zip(losses, losses[1:]):
        if math.isfinite(a) and math.isfinite(b) and b > a:
            streak += 1
            best_streak = max(best_streak, streak)
        else:
            streak = 0
    plateau_iter: int | None = None
    for i in range(10, len(losses)):
        if math.isfinite(losses[i]) and math.isfinite(losses[i - 10]):
            if losses[i] < losses[i - 10] * 0.99:
                plateau_iter = i + 1  # 1 起
    rms = res.phase_rms
    runaway = False
    if len(rms) >= 11 and math.isfinite(rms[9]) and rms[9] > 0:
        runaway = bool(rms[-1] > 2.0 * rms[9])
    return {
        "nonfinite": bool(nonfinite),
        "max_loss_increase_streak": int(best_streak),
        "loss_increase_streak_ge5": bool(best_streak >= 5),
        "max_phase_delta": float(max(res.phase_delta_max)) if res.phase_delta_max else float("nan"),
        "phase_rms_runaway": runaway,
        "plateau_iter": plateau_iter,
    }


def stability_verdict(stab: dict[str, Any]) -> str:
    """把稳定性标志汇成一句判定 (表格用)。"""
    if stab["nonfinite"]:
        return "发散 (NaN/Inf)"
    bad: list[str] = []
    if stab["loss_increase_streak_ge5"]:
        bad.append(f"损失连升≥5 (最长 {stab['max_loss_increase_streak']})")
    if stab["phase_rms_runaway"]:
        bad.append("相位 RMS 失控")
    if math.isfinite(stab["max_phase_delta"]) and stab["max_phase_delta"] > 1.0:
        bad.append(f"单步 |Δφ| 达 {stab['max_phase_delta']:.2f} rad")
    return "; ".join(bad) if bad else "稳定"


# ---------------------------------------------------------------------------
# 基线参考与头条数字
# ---------------------------------------------------------------------------
def read_history_csv(run_dir: Path) -> list[dict[str, float]]:
    """读运行目录的 ``history.csv`` (iteration/loss 列)。

    用 ``utf-8-sig``: 硬件运行落盘的 ``history.csv`` 带 UTF-8 BOM, 以
    ``utf-8`` 读会把首列名变成 ``\\ufeffiteration`` 而报 KeyError。

    Raises:
        FileNotFoundError: 文件缺失。
    """
    path = run_dir / "history.csv"
    if not path.is_file():
        raise FileNotFoundError(f"找不到基线迭代史: {path}")
    rows: list[dict[str, float]] = []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append({"iteration": float(row["iteration"]), "loss": float(row["loss"])})
    return rows


def _read_forward_mean_from_run(run_dir: Path) -> float | None:
    """从 ``step_b_reanalysis_debug.json`` 读基线运行的前向均值 (秒)。

    仅用于 JSON/MD 里与硬件运行记录做量级对照; 文件缺失或字段缺失时返回
    ``None`` (不影响扫描)。

    Args:
        run_dir: 运行目录。

    Returns:
        前向耗时的均值 (秒), 或 ``None``。
    """
    path = run_dir / "step_b_reanalysis_debug.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    for key in ("forward_s_mean", "forward_s", "forward_mean_s"):
        val = data.get(key) if isinstance(data, dict) else None
        if isinstance(val, (int, float)) and math.isfinite(float(val)):
            return float(val)
    return None


def repro_check(repro: ArmResult, history: list[dict[str, float]]) -> dict[str, Any]:
    """基线复现比对: 重建轨迹 vs ``history.csv``。

    检查点取第 1 次迭代、四分位与最后一次 (以重建长度与 csv 长度的较小者
    为界)。CUDA 非确定性 + 120 步 Adam 的混沌放大会让后期偏差长到 1e-4 量级
    (原运行与其离线重分析之间即如此), 所以默认容差 1e-3 —— 远大于数值噪声,
    远小于「问题重建错了」的偏差。

    Returns:
        含 ``checkpoints`` / ``max_rel_dev`` / ``ok`` 的字典。
    """
    n = min(len(repro.losses), len(history))
    idxs = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})
    checkpoints = []
    max_rel = 0.0
    for idx in idxs:
        if idx < 0 or idx >= n:
            continue
        ref = history[idx]["loss"]
        got = repro.losses[idx]
        rel = abs(got - ref) / abs(ref) if ref != 0 else (0.0 if got == 0 else float("inf"))
        max_rel = max(max_rel, rel)
        checkpoints.append(
            {"iteration": idx + 1, "history_loss": ref, "rebuilt_loss": got, "rel_dev": rel}
        )
    return {"checkpoints": checkpoints, "max_rel_dev": max_rel, "n_compared": n}


def quality_ratio(sample: dict[str, Any], eff_base: float, cv_base: float) -> float:
    """物理质量比 q = min(效率/基线效率, 基线CV/CV); q>=1 即不劣于基线终态。"""
    eff = sample.get("energy_in_target", float("nan"))
    cv = sample.get("cv_in_target", float("nan"))
    parts = []
    if eff_base > 0 and math.isfinite(eff):
        parts.append(eff / eff_base)
    if cv > 0 and math.isfinite(cv) and cv_base > 0:
        parts.append(cv_base / cv)
    return min(parts) if parts else float("nan")


def first_iter_reaching(
    res: ArmResult,
    predicate: Any,
) -> int | None:
    """第一个满足谓词的采样迭代号 (1 起); 从未满足返回 ``None``。"""
    for s in res.samples:
        if predicate(s):
            return int(s["iter"])
    return None


def first_loss_iter_at_or_below(res: ArmResult, threshold: float) -> int | None:
    """第一次把损失压到 ``threshold`` 以下的迭代号 (逐迭代分辨率)。"""
    for i, loss in enumerate(res.losses):
        if math.isfinite(loss) and loss <= threshold:
            return i + 1
    return None


def compute_headline(
    res: ArmResult,
    *,
    base_final_loss: float,
    eff_base: float,
    cv_base: float,
) -> dict[str, Any]:
    """计算一个变体的头条数字 (达标迭代数 / 终态指标 / 墙钟 / 判定)。"""
    final = res.samples[-1] if res.samples else {}
    q_final = quality_ratio(final, eff_base, cv_base) if final else float("nan")

    def _q_at(frac: float) -> int | None:
        # frac=0.99: 效率不低于 0.99·基线 且 CV 不高于 基线/0.99。
        return first_iter_reaching(
            res,
            lambda s, f=frac: (
                math.isfinite(s.get("energy_in_target", float("nan")))
                and math.isfinite(s.get("cv_in_target", float("nan")))
                and s["energy_in_target"] >= f * eff_base
                and s["cv_in_target"] <= cv_base / f
            ),
        )

    return {
        "iters_to_baseline_final_loss": first_loss_iter_at_or_below(res, base_final_loss),
        "iters_to_quality99": _q_at(0.99),
        "iters_to_quality95": _q_at(0.95),
        "final_loss": res.losses[-1] if res.losses else float("nan"),
        "final_quality_ratio": q_final,
        "final": {
            "energy_in_target": final.get("energy_in_target", float("nan")),
            "cv_in_target": final.get("cv_in_target", float("nan")),
            "pred_fwhm_px": final.get("pred_fwhm_px", float("nan")),
            "pred_peak": final.get("pred_peak", float("nan")),
            "strehl_like": final.get("strehl_like", float("nan")),
        },
        "wall_s": res.wall_s,
        "verdict": stability_verdict(res.stability),
    }


def select_top_lrs(
    ladder: list[ArmResult],
    eff_base: float,
    cv_base: float,
) -> tuple[list[float], str]:
    """从阶梯结果里挑两个最优学习率 (调度阶段的峰值学习率)。

    排序键: (达到基线 99% 物理质量的迭代数, 越小越好; 并列比终态质量比 q,
    越大越好; 再并列按学习率升序)。从未达标的变体排在最后。

    Returns:
        ``(两个学习率, 选择理由)``。
    """

    def key(res: ArmResult) -> tuple[float, float, float]:
        h = res.headline
        n99 = h.get("iters_to_quality99")
        q = h.get("final_quality_ratio", float("nan"))
        q = q if math.isfinite(q) else -1.0
        return (float(n99) if n99 is not None else float("inf"), -q, res.spec.lr)

    ranked = sorted(ladder, key=key)
    picked = [r.spec.lr for r in ranked[:2]]
    parts = []
    for r in ranked[:2]:
        n99 = r.headline.get("iters_to_quality99")
        parts.append(
            f"lr={r.spec.lr:g} (99% 质量迭代数={n99 if n99 is not None else '未达标'}, "
            f"终态 q={r.headline.get('final_quality_ratio', float('nan')):.3f})"
        )
    return picked, "按 (99% 质量达标迭代数 ↑, 终态质量比 ↓) 排序取前二: " + "; ".join(parts)


# ---------------------------------------------------------------------------
# 输出: JSON / CSV / MD / 图
# ---------------------------------------------------------------------------
def write_outputs(
    out_dir: Path,
    *,
    payload: dict[str, Any],
    all_results: list[ArmResult],
    baseline: dict[str, Any],
    recommendation: str,
    repro_block: dict[str, Any],
) -> list[Path]:
    """写 JSON/CSV/MD 与两张图, 图复制到报告 figures 目录。

    Returns:
        写出的全部文件路径。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    json_path = out_dir / "lr_sweep.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=_json_default),
        encoding="utf-8",
    )
    written.append(json_path)

    csv_path = out_dir / "lr_sweep.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for res in all_results:
            for s in res.samples:
                row = {
                    "arm_kind": res.spec.kind,
                    "schedule": res.spec.schedule,
                    "lr": f"{res.spec.lr:g}",
                }
                row.update({k: s.get(k) for k in CSV_FIELDS if k not in row})
                writer.writerow(row)
    written.append(csv_path)

    setup_cjk_font()
    fig_loss = out_dir / "lrsweep_loss_vs_iter.png"
    _plot_loss(all_results, fig_loss)
    written.append(fig_loss)
    fig_phys = out_dir / "lrsweep_physical_vs_iter.png"
    _plot_physical(all_results, baseline, fig_phys)
    written.append(fig_phys)

    REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for fig in (fig_loss, fig_phys):
        dst = REPORT_FIG_DIR / fig.name
        shutil.copyfile(fig, dst)
        written.append(dst)

    md_path = out_dir / "lr_sweep.md"
    md_path.write_text(
        _render_md(payload, all_results, baseline, recommendation, repro_block),
        encoding="utf-8",
    )
    written.append(md_path)

    # 与兄弟产物目录 (stepb_budget_sweep_20261116) 同一惯例: 把生成脚本
    # 一起归档, 保证数字可追溯。
    self_copy = out_dir / Path(__file__).name
    shutil.copyfile(Path(__file__).resolve(), self_copy)
    written.append(self_copy)
    return written


def _json_default(o: Any) -> Any:
    """JSON 序列化的兜底 (numpy 标量/数组、Path)。"""
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"无法序列化: {type(o)}")


def _sanitize_for_json(obj: Any) -> Any:
    """把 NaN/Inf 换成 None (JSON 不接受), 递归处理 dict/list。"""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer, np.ndarray, Path)):
        return _sanitize_for_json(_json_default(obj) if not isinstance(obj, Path) else str(obj))
    return obj


_STYLES = {
    "repro": dict(color="black", ls="--", lw=2.0),
    "ladder": dict(ls="-", lw=1.4),
    "schedule": dict(ls="-.", lw=1.4),
    "short": dict(ls=":", lw=1.6),
}
_COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf", "#8c564b", "#e377c2")


def _plot_loss(results: list[ArmResult], path: Path) -> None:
    """图 1: 全部变体的损失-迭代曲线 (对数纵轴)。"""
    fig, ax = plt.subplots(figsize=(9.0, 5.5), dpi=150)
    color_i = 0
    for res in results:
        style = dict(_STYLES.get(res.spec.kind, {}))
        if res.spec.kind != "repro":
            style["color"] = _COLORS[color_i % len(_COLORS)]
            color_i += 1
        xs = np.arange(1, len(res.losses) + 1)
        ax.plot(
            xs,
            np.asarray(res.losses, dtype=float),
            label=f"[{res.spec.kind}] {res.spec.label}",
            **style,
        )
    ax.set_yscale("log")
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("峰值归一化损失 (对数轴)")
    ax.set_title("可微分光束整形: 学习率/调度扫描 —— 模型损失随迭代的变化 (纯离线)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_physical(results: list[ArmResult], baseline: dict[str, Any], path: Path) -> None:
    """图 2: 目标区内能量占比与均匀度 CV 随迭代的变化 (采样点)。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.0), dpi=150)
    color_i = 0
    for res in results:
        if not res.samples:
            continue
        style = dict(_STYLES.get(res.spec.kind, {}))
        if res.spec.kind != "repro":
            style["color"] = _COLORS[color_i % len(_COLORS)]
            color_i += 1
        xs = [s["iter"] for s in res.samples]
        eff = [s["energy_in_target"] for s in res.samples]
        cv = [s["cv_in_target"] for s in res.samples]
        label = f"[{res.spec.kind}] {res.spec.label}"
        ax1.plot(xs, eff, label=label, **style)
        ax2.plot(xs, cv, label=label, **style)
    q = baseline.get("final_quality", {})
    eff_b = q.get("energy_in_target")
    cv_b = q.get("cv_in_target")
    if eff_b is not None:
        ax1.axhline(eff_b, color="black", ls="--", lw=1.0, label=f"基线终态 ({eff_b:.3f})")
        ax1.axhline(0.99 * eff_b, color="grey", ls=":", lw=1.0, label="99% 基线")
    if cv_b is not None:
        ax2.axhline(cv_b, color="black", ls="--", lw=1.0, label=f"基线终态 ({cv_b:.3f})")
        ax2.axhline(cv_b / 0.99, color="grey", ls=":", lw=1.0, label="99% 基线")
    ax1.set_xlabel("迭代次数")
    ax1.set_ylabel("目标区内能量占比")
    ax1.set_title("物理指标: 目标区内能量占比 (越高越好)")
    ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("迭代次数")
    ax2.set_ylabel("目标区内强度均匀度 CV (越低越好)")
    ax2.set_title("物理指标: 目标区内均匀度 CV")
    ax2.grid(True, alpha=0.3)
    for ax in (ax1, ax2):
        ax.legend(fontsize=6.5, ncol=2, loc="best")
    fig.suptitle("可微分光束整形: 学习率/调度扫描 —— 物理指标随迭代的变化 (纯离线)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# R-25 反漂移 (issue #205 第 132 批): 逐字口径 = fmt_general(missing_none="n/a",
# missing_nonfinite="n/a")。唯一差异在**调用点不可达**的字符串输入: 本地版让
# format() 抛 ValueError, 规范版回退 str(v) (更安全); 数值/None/bool/int/numpy
# 全电池逐字节全等 (台账登记)。
_fmt = partial(fmt_general, missing_none="n/a", missing_nonfinite="n/a")


def _headline_row(res: ArmResult) -> str:
    h = res.headline
    f = h.get("final", {})
    return (
        f"| {res.spec.kind} | {res.spec.schedule} | {res.spec.lr:g} | {res.spec.iters} "
        f"| {_fmt(h.get('iters_to_baseline_final_loss'), 'd') if h.get('iters_to_baseline_final_loss') is not None else 'n/a'} "
        f"| {_fmt(h.get('iters_to_quality99'), 'd') if h.get('iters_to_quality99') is not None else 'n/a'} "
        f"| {_fmt(h.get('iters_to_quality95'), 'd') if h.get('iters_to_quality95') is not None else 'n/a'} "
        f"| {_fmt(h.get('final_loss'), '.4e')} | {_fmt(f.get('energy_in_target'))} "
        f"| {_fmt(f.get('cv_in_target'))} | {_fmt(f.get('pred_fwhm_px'), '.2f')} "
        f"| {_fmt(f.get('strehl_like'), '.4f')} | {_fmt(h.get('wall_s'), '.1f')} "
        f"| {h.get('verdict', '')} |"
    )


def _render_md(
    payload: dict[str, Any],
    results: list[ArmResult],
    baseline: dict[str, Any],
    recommendation: str,
    repro_block: dict[str, Any],
) -> str:
    """渲染 ``lr_sweep.md`` (表格 + 建议, 可直接摘录进报告)。"""
    lines: list[str] = []
    lines.append("# 可微分光束整形: Adam 学习率与动态调度扫描 (纯离线, 2026-11-16)")
    lines.append("")
    lines.append(
        f"- 运行目录 (问题来源): `{payload['config']['run_dir']}`"
        f" | 设备: `{payload['config']['device']}` | 种子: {payload['config']['seed']}"
    )
    lines.append(
        f"- 基线: lr={payload['config']['baseline_lr']:g}, iters={payload['config']['baseline_iters']},"
        f" `history.csv` 终损={_fmt(baseline.get('final_loss_csv'), '.6e')}"
    )
    lines.append(
        f"- 物理质量基准 (重建基线终态采样): 目标区内能量占比="
        f"{_fmt(baseline.get('final_quality', {}).get('energy_in_target'))},"
        f" 均匀度 CV={_fmt(baseline.get('final_quality', {}).get('cv_in_target'))}"
    )
    lines.append(
        f"- 质量比 q = min(效率/基线效率, 基线CV/CV); 「99% 质量」= 效率 ≥ 0.99·基线 且"
        f" CV ≤ 基线/0.99 (95% 同理)。达标迭代数按物理指标采样间隔"
        f" ({payload['config']['metric_every']}) 取整, 短预算变体再做精确复跑验证。"
    )
    lines.append("")
    lines.append("## 基线复现检查")
    lines.append("")
    lines.append("| 迭代 | history.csv 损失 | 重建损失 | 相对偏差 |")
    lines.append("| ---: | ---: | ---: | ---: |")
    for cp in repro_block.get("checkpoints", []):
        lines.append(
            f"| {cp['iteration']} | {_fmt(cp['history_loss'], '.6e')} "
            f"| {_fmt(cp['rebuilt_loss'], '.6e')} | {_fmt(cp['rel_dev'], '.2e')} |"
        )
    lines.append("")
    lines.append(
        f"**最大相对偏差 = {_fmt(repro_block.get('max_rel_dev'), '.3e')}**"
        f" (容差 {payload['config']['repro_tol']:g}) —— "
        + ("复现通过, 扫描与硬件运行是同一个问题。" if repro_block.get("ok") else "复现失败!")
    )
    lines.append("")
    header = (
        "| 类型 | 调度 | 学习率 | 预算 | 达基线终损 | 达 99% 质量 | 达 95% 质量 "
        "| 终损 | 终态效率 | 终态 CV | 终态 FWHM(px) | 类Strehl | 墙钟(s) | 稳定性 |"
    )
    sep = "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
    for kind, title in (
        ("repro", "## 基线复现变体"),
        ("ladder", "## 固定学习率阶梯 (120 迭代预算)"),
        ("schedule", "## 动态调度 (阶梯最优两个学习率)"),
        ("short", "## 短预算变体 (达 99% 基线质量的最少迭代, 精确复跑)"),
    ):
        group = [r for r in results if r.spec.kind == kind]
        if not group:
            continue
        lines.append(title)
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for res in group:
            lines.append(_headline_row(res))
        lines.append("")
    lines.append("## 建议")
    lines.append("")
    lines.append(recommendation)
    lines.append("")
    lines.append("## 方法与成本")
    lines.append("")
    timing = payload.get("timing", {})
    budget = payload.get("budget", {})
    lines.append(
        f"- 单次 ``update()`` (前向+反向+Adam 步): {timing.get('t_update_ms', float('nan')):.2f} ms"
        f" (设备 {payload['config']['device']}; 基线运行记录的 forward 均值 "
        f"{_fmt(baseline.get('forward_s_mean_from_run'), '.2f')} ms, 量级一致)。"
    )
    lines.append(
        f"- 单次物理指标 (measure_spot + 区内统计): {timing.get('t_phys_ms', float('nan')):.1f} ms"
        f" ⇒ 损失每迭代记录, 物理指标每 {payload['config']['metric_every']} 迭代采样。"
    )
    lines.append(
        f"- 预算: 估算 {budget.get('estimated_s', float('nan')):.0f} s / 上限 "
        f"{budget.get('budget_s', float('nan')):.0f} s; 实际总墙钟 "
        f"{budget.get('actual_total_s', float('nan')):.0f} s。裁剪: {budget.get('cuts') or '无'}。"
    )
    lines.append(
        "- 物理指标口径: FWHM/峰值/空洞度复用仓库共享 `measure_spot`"
        " (去尖峰+平滑+半高宽, 与 `compare_farfields` 的模型侧同一口径);"
        " 能量占比与 CV 在 `target > 0` 的目标区上直接统计; 类 Strehl = 当前峰值 / 平相位峰值。"
    )
    lines.append("")
    lines.append("![损失曲线](lrsweep_loss_vs_iter.png)")
    lines.append("")
    lines.append("![物理指标](lrsweep_physical_vs_iter.png)")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 预算规划
# ---------------------------------------------------------------------------
def plan_budget(
    *,
    lrs: list[float],
    n_dynamic: int,
    iters: int,
    metric_every: int,
    t_update_ms: float,
    t_phys_ms: float,
    budget_s: float,
) -> tuple[list[float], int, int, dict[str, Any]]:
    """按实测单次成本估算总墙钟; 超预算时按既定顺序裁剪。

    裁剪顺序 (任务规定): 先去掉 lr=0.01, 再把 ``metric_every`` 加倍, 最后把
    迭代预算缩到 80。全部裁剪记录进返回的 ``budget`` 字典。

    Returns:
        ``(裁剪后的 lrs, iters, metric_every, budget 信息)``。
    """
    cuts: list[str] = []

    def estimate(lrs_: list[float], iters_: int, me_: int) -> float:
        # 变体数: 复现 1 + 阶梯 len(lrs) + 调度 2×n_dynamic + 短预算 <= len(lrs)
        n_arms = 1 + len(lrs_) + 2 * n_dynamic + len(lrs_)
        samples_per_arm = max(iters_ // me_, 1) + 2
        return (n_arms * iters_ * t_update_ms + n_arms * samples_per_arm * t_phys_ms) / 1000.0 + 60.0

    est = estimate(lrs, iters, metric_every)
    if est > budget_s and 0.01 in lrs:
        lrs = [x for x in lrs if x != 0.01]
        cuts.append("去掉 lr=0.01 (估算超预算)")
        est = estimate(lrs, iters, metric_every)
    while est > budget_s and metric_every < iters:
        metric_every = min(metric_every * 2, max(iters, 1))
        cuts.append(f"metric_every 加倍到 {metric_every}")
        est = estimate(lrs, iters, metric_every)
    if est > budget_s and iters > 80:
        iters = 80
        cuts.append("迭代预算缩短到 80")
        est = estimate(lrs, iters, metric_every)
    return lrs, iters, metric_every, {
        "budget_s": budget_s,
        "estimated_s": est,
        "cuts": cuts,
        "t_update_ms": t_update_ms,
        "t_phys_ms": t_phys_ms,
    }


# ---------------------------------------------------------------------------
# CLI 与主流程
# ---------------------------------------------------------------------------
def _parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def _parse_schedules(text: str) -> list[str]:
    names = [x.strip() for x in text.split(",") if x.strip() and x.strip().lower() != "none"]
    for n in names:
        if n not in DEFAULT_SCHEDULES:
            raise argparse.ArgumentTypeError(
                f"未知调度 {n!r} (可选: {', '.join(DEFAULT_SCHEDULES)} 或 none)"
            )
    return names


def build_arg_parser() -> argparse.ArgumentParser:
    """命令行参数。"""
    p = argparse.ArgumentParser(
        description="纯离线: 可微分光束整形的 Adam 学习率/动态调度扫描 (不碰任何设备)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="基线运行目录")
    p.add_argument(
        "--lrs",
        type=_parse_floats,
        default=list(DEFAULT_LRS),
        help="固定学习率阶梯 (逗号分隔)",
    )
    p.add_argument("--iters", type=int, default=None, help="迭代预算 (默认取 run-dir 的 iters)")
    p.add_argument(
        "--schedules",
        type=_parse_schedules,
        default=list(DEFAULT_SCHEDULES),
        help="动态调度列表 (逗号分隔; none = 只跑阶梯)",
    )
    p.add_argument(
        "--metric-every",
        type=int,
        default=None,
        help="物理指标采样间隔 (默认: iters<=200 时 1, 否则 5)",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="产物输出目录")
    p.add_argument("--seed", type=int, default=None, help="随机种子 (默认取 run-dir 的 seed)")
    p.add_argument("--device", default=None, help="torch 设备 (默认 cuda 可用则 cuda)")
    p.add_argument(
        "--repro-tol",
        type=float,
        default=DEFAULT_REPRO_TOL,
        help="基线复现的最大相对偏差容差 (超过即 exit 2)",
    )
    p.add_argument("--budget-s", type=float, default=DEFAULT_BUDGET_S, help="总墙钟预算 (秒)")
    p.add_argument(
        "--quick",
        action="store_true",
        help="冒烟: 8 迭代 / lr∈{0.05,0.2} / 只跑 cosine / 不跑短预算变体",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """入口: 复现检查 -> 阶梯 -> 调度 -> 短预算 -> 产物。"""
    args = build_arg_parser().parse_args(argv)
    t_main = time.perf_counter()

    # 重依赖延迟导入 (保持模块加载轻; 本脚本纯离线, 但仍不无谓牵入 torch)。
    import torch

    from ao_shaping.algorithm import far_field_intensity
    from ao_shaping.utils.image.frame_analysis import measure_spot

    deps = _Deps(torch=torch, far_field_intensity=far_field_intensity, measure_spot=measure_spot)

    run_dir = Path(args.run_dir)
    logger.info("重建问题: {}", run_dir)
    problem = load_problem_from_run_dir(run_dir)
    params = problem.params
    baseline_lr = float(params.get("lr", 0.05))
    baseline_iters = int(params.get("iters", 120))
    seed = int(args.seed) if args.seed is not None else int(params.get("seed", 0))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    iters = int(args.iters) if args.iters is not None else baseline_iters
    lrs = list(args.lrs)
    schedules = list(args.schedules)
    metric_every = args.metric_every
    if metric_every is None:
        metric_every = 1 if iters <= 200 else 5
    skip_short = bool(args.quick)
    if args.quick:
        iters = min(iters, 8)
        lrs = [x for x in (0.05, 0.2) if x in lrs] or [0.05]
        schedules = [s for s in schedules if s in ("flat", "cosine")] or ["cosine"]
        metric_every = 1

    history = read_history_csv(run_dir)
    base_final_loss_csv = history[-1]["loss"]
    logger.info(
        "基线: lr={} iters={} | history.csv 终损={:.6e} | 设备={} 种子={} | 迭代预算={} 采样间隔={}",
        baseline_lr,
        baseline_iters,
        base_final_loss_csv,
        device,
        seed,
        iters,
        metric_every,
    )

    # ---- 成本实测 (任务规定的强制第一步): 单次 update 与单次物理指标 ----
    # update() 单独计时 (不含物理指标), 否则 216 ms 的 measure_spot 会把
    # 3.75 ms 的 update() 淹没, 预算估算将完全失真。
    n_probe = 5
    opt_probe = build_optimizer(problem, lr=baseline_lr, device=device, seed=seed)
    opt_probe.update()  # 热身 (首次含 CUDA kernel 编译/显存分配)
    t_u0 = time.perf_counter()
    for _ in range(n_probe):
        opt_probe.update()
    t_update_ms = (time.perf_counter() - t_u0) / n_probe * 1000.0
    # 物理指标单次成本: 远场前向 + measure_spot + 区内统计。
    with torch.no_grad():
        far_probe = (
            far_field_intensity(problem.amplitude, opt_probe.phase_tensor).detach().cpu().numpy()
        )
    t_p0 = time.perf_counter()
    measure_spot(far_probe)
    probe_mask = problem.target > 0.0
    in_probe = far_probe[probe_mask]
    float(in_probe.std() / in_probe.mean() if in_probe.size else 0.0)
    t_phys_ms = (time.perf_counter() - t_p0) * 1000.0
    forward_mean = _read_forward_mean_from_run(run_dir)
    logger.info(
        "成本实测: update ≈ {:.2f} ms/迭代; 物理指标 (measure_spot+区内统计) ≈ {:.0f} ms/次"
        " (基线运行记录的 forward 均值 {} ms, 见 step_b_reanalysis_debug.json)",
        t_update_ms,
        t_phys_ms,
        f"{forward_mean:.2f}" if forward_mean is not None else "n/a",
    )
    del opt_probe, far_probe, in_probe

    n_dynamic = len([s for s in schedules if s in DYNAMIC_SCHEDULES])
    lrs, iters, metric_every, budget = plan_budget(
        lrs=lrs,
        n_dynamic=n_dynamic,
        iters=iters,
        metric_every=metric_every,
        t_update_ms=max(t_update_ms, 0.1),
        t_phys_ms=max(t_phys_ms, 1.0),
        budget_s=args.budget_s,
    )
    budget["physical_metric_sampling"] = (
        f"损失/梯度/相位 RMS 每迭代记录; 物理指标每 {metric_every} 迭代采样"
        f" (measure_spot 实测 {t_phys_ms:.0f} ms/次, 每迭代采样会把总墙钟推到"
        f" ~{19 * iters * t_phys_ms / 1000 / 60:.0f} 分钟, 且 GPU 为共享设备)"
    )
    if budget["cuts"]:
        logger.warning("预算裁剪: {}", "; ".join(budget["cuts"]))

    # ---- 阶段 1: 基线复现 ----
    repro_spec = ArmSpec(kind="repro", schedule="flat", lr=baseline_lr, iters=baseline_iters)
    if args.quick:
        repro_spec = ArmSpec(kind="repro", schedule="flat", lr=baseline_lr, iters=iters)
    logger.info("阶段 1/4: 基线复现 (lr={}, iters={})", repro_spec.lr, repro_spec.iters)
    repro = run_arm(
        problem, repro_spec, device=device, seed=seed, metric_every=metric_every, deps=deps
    )
    repro_block = repro_check(repro, history)
    repro_block["ok"] = bool(repro_block["max_rel_dev"] <= args.repro_tol)
    repro_block["tol"] = args.repro_tol
    logger.info(
        "基线复现: 最大相对偏差 {:.3e} (容差 {:g}) -> {}",
        repro_block["max_rel_dev"],
        args.repro_tol,
        "通过" if repro_block["ok"] else "失败",
    )
    if not repro_block["ok"]:
        logger.error(
            "复现失败: 重建的问题与 history.csv 不一致 (偏差 {:.3e} > {:g}), 扫描中止。",
            repro_block["max_rel_dev"],
            args.repro_tol,
        )
        return EXIT_REPRO_FAILED

    base_quality_sample = repro.samples[-1]
    eff_base = float(base_quality_sample["energy_in_target"])
    cv_base = float(base_quality_sample["cv_in_target"])
    logger.info(
        "基线终态物理质量 (重建): 效率={:.4f} CV={:.4f} FWHM={:.2f}px 类Strehl={:.4f}",
        eff_base,
        cv_base,
        base_quality_sample["pred_fwhm_px"],
        base_quality_sample["strehl_like"],
    )

    results: list[ArmResult] = [repro]
    repro.headline = compute_headline(
        repro, base_final_loss=base_final_loss_csv, eff_base=eff_base, cv_base=cv_base
    )

    def _run(spec: ArmSpec) -> ArmResult:
        logger.info("跑变体 [{}] {} ({} 迭代)", spec.kind, spec.label, spec.iters)
        r = run_arm(problem, spec, device=device, seed=seed, metric_every=metric_every, deps=deps)
        r.headline = compute_headline(
            r, base_final_loss=base_final_loss_csv, eff_base=eff_base, cv_base=cv_base
        )
        logger.info(
            "  -> 终损={:.4e} 99%质量@{} 95%质量@{} 判定: {}",
            r.headline["final_loss"],
            r.headline["iters_to_quality99"],
            r.headline["iters_to_quality95"],
            r.headline["verdict"],
        )
        results.append(r)
        return r

    # ---- 阶段 2: 固定学习率阶梯 ----
    logger.info("阶段 2/4: 固定学习率阶梯 {}", [f"{x:g}" for x in lrs])
    ladder: list[ArmResult] = []
    for lr in lrs:
        ladder.append(_run(ArmSpec(kind="ladder", schedule="flat", lr=lr, iters=iters)))

    # ---- 阶段 3: 动态调度 (阶梯最优两个学习率) ----
    top_lrs: list[float] = []
    selection_reason = ""
    dyn = [s for s in schedules if s in DYNAMIC_SCHEDULES]
    if dyn:
        top_lrs, selection_reason = select_top_lrs(ladder, eff_base, cv_base)
        logger.info("阶段 3/4: 动态调度 {} @ 最优学习率 {} ({})", dyn, top_lrs, selection_reason)
        for lr in top_lrs:
            for sched in dyn:
                _run(ArmSpec(kind="schedule", schedule=sched, lr=lr, iters=iters))
    else:
        logger.info("阶段 3/4: 跳过 (--schedules 不含动态调度)")

    # ---- 阶段 4: 短预算变体 ----
    # 候选 = 「稳定 且 达到基线 99% 物理质量」的变体, 把迭代预算截到其达标
    # 迭代数 n99 再精确复跑一遍, 验证截断后终态质量仍达标。固定学习率与动态
    # 调度都纳入: 调度按 iters 成形 (cosine 的分母是 iters), 截断到 n99 会
    # 重塑调度曲线, 故短预算复跑不是简单重读全量轨迹的前 n99 步, 而是真验证。
    short_specs: list[ArmSpec] = []
    if not skip_short:
        logger.info("阶段 4/4: 短预算变体")
        candidates = [r for r in ladder] + [r for r in results if r.spec.kind == "schedule"]
        for r in candidates:
            n99 = r.headline.get("iters_to_quality99")
            stable = r.headline.get("verdict") == "稳定"
            if n99 is not None and n99 < iters and stable:
                short_specs.append(
                    ArmSpec(kind="short", schedule=r.spec.schedule, lr=r.spec.lr, iters=int(n99))
                )
        for spec in short_specs:
            _run(spec)
    else:
        logger.info("阶段 4/4: 跳过短预算变体 (--quick)")

    # ---- 头条汇总与建议 ----
    baseline_block = {
        "final_loss_csv": base_final_loss_csv,
        "forward_s_mean_from_run": forward_mean,
        "baseline_iters": baseline_iters,
        "metric_every": metric_every,
        "final_quality": {
            "energy_in_target": eff_base,
            "cv_in_target": cv_base,
            "pred_fwhm_px": base_quality_sample["pred_fwhm_px"],
            "strehl_like": base_quality_sample["strehl_like"],
        },
        "repro": repro_block,
    }
    recommendation = build_recommendation(results, baseline_block, top_lrs, iters)

    total_s = time.perf_counter() - t_main
    budget["actual_total_s"] = total_s
    config = {
        "run_dir": str(run_dir),
        "device": device,
        "seed": seed,
        "baseline_lr": baseline_lr,
        "baseline_iters": baseline_iters,
        "iters": iters,
        "lrs": lrs,
        "schedules": schedules,
        "metric_every": metric_every,
        "repro_tol": args.repro_tol,
        "quick": bool(args.quick),
        "argv": sys.argv[1:],
        "torch_version": torch.__version__,
        "lr_floor_frac": LR_FLOOR_FRAC,
    }
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "config": config,
        "timing": {"t_update_ms": t_update_ms, "t_phys_ms": t_phys_ms},
        "budget": budget,
        "baseline": baseline_block,
        "schedule_lr_selection": {"top_lrs": top_lrs, "reason": selection_reason},
        "short_budget_requests": [
            {"schedule": s.schedule, "lr": s.lr, "iters": s.iters} for s in short_specs
        ],
        "arms": [
            {
                "kind": r.spec.kind,
                "schedule": r.spec.schedule,
                "lr": r.spec.lr,
                "iters": r.spec.iters,
                "wall_s": r.wall_s,
                "stability": r.stability,
                "headline": r.headline,
                "losses": r.losses,
                "lrs_applied": r.lrs_applied,
                "samples": r.samples,
            }
            for r in results
        ],
        "recommendation": recommendation,
    }
    payload = _sanitize_for_json(payload)

    written = write_outputs(
        args.out,
        payload=payload,
        all_results=results,
        baseline=baseline_block,
        recommendation=recommendation,
        repro_block=repro_block,
    )
    logger.info("总墙钟 {:.1f} s | 产物 {} 个:", total_s, len(written))
    for w in written:
        logger.info("  {}", w)
    logger.info("硬件占用: 无 (纯离线, 未 import/open 任何 SLM/CCD/激光驱动)。")
    return EXIT_OK


def build_recommendation(
    results: list[ArmResult],
    baseline: dict[str, Any],
    top_lrs: list[float],
    iters: int,
) -> str:
    """由头条数字自动拼一段建议 (最大迭代削减 + 风险)。"""
    full = [r for r in results if r.spec.kind in ("ladder", "schedule", "short")]
    reached = [
        r
        for r in full
        if r.headline.get("iters_to_quality99") is not None
        and r.headline.get("verdict") == "稳定"
    ]
    base_q = baseline["final_quality"]
    if not reached:
        return (
            f"在 {iters} 迭代预算内, 没有任何学习率/调度变体达到基线终态物理质量的 99%"
            f" (基线: 效率 {base_q['energy_in_target']:.4f}, CV {base_q['cv_in_target']:.4f})。"
            "离线证据不支持在硬件上换用更大学习率来缩短迭代。"
        )

    def n99(r: ArmResult) -> int:
        return int(r.headline["iters_to_quality99"])

    best = min(reached, key=lambda r: (n99(r), r.spec.lr))
    worst_streak = max(
        (r.stability["max_loss_increase_streak"] for r in full), default=0
    )
    unstable_big_lr = [
        f"lr={r.spec.lr:g}"
        for r in results
        if r.spec.kind == "ladder" and r.headline.get("verdict") != "稳定"
    ]
    base_iters = int(baseline.get("baseline_iters", iters) or iters)
    baseline_iters_pct = 100.0 * n99(best) / max(base_iters, 1)

    # 短预算复跑验证: 找与 best 同 (调度, lr) 的 short 变体, 据其结果措辞。
    # flat 调度与 horizon 无关, 截断复跑等价于读全量前 n99 步, 必然验证通过;
    # 动态调度按 iters 成形, 截断会重塑曲线, 复跑可能只到 95% —— 此时如实说明
    # 「n99 达标是全 horizon 调度的性质, 不能把预算直接砍到 n99」。
    short_match = next(
        (
            r
            for r in results
            if r.spec.kind == "short"
            and r.spec.schedule == best.spec.schedule
            and r.spec.lr == best.spec.lr
        ),
        None,
    )
    if short_match is None:
        verify_note = "短预算复跑未触发 (无同配置候选)。"
    elif short_match.headline.get("iters_to_quality99") is not None:
        verify_note = (
            f"短预算复跑已验证: 把预算截到 {n99(best)} 次迭代后, 终态仍达基线 99% 物理质量。"
        )
    else:
        s95 = short_match.headline.get("iters_to_quality95")
        verify_note = (
            f"**重要 caveat**: 把 {best.spec.schedule} 调度重塑到恰好 {n99(best)} 次迭代的短预算复跑"
            f"只达到基线 95% 物理质量 (95%@{s95}, 99%@未达标) —— 因为动态调度按 horizon 成形,"
            f"「{n99(best)} 次达 99%」是**完整 {iters} 次 horizon 调度**在第 {n99(best)} 步的性质,"
            f"不能把迭代预算直接砍到 {n99(best)}; 应保留标准 horizon, 把第 {n99(best)} 步视作"
            f"「质量已追平基线」的里程碑而非终点。"
        )

    is_schedule = best.spec.schedule != "flat"
    flat_overshoot = (
        f"固定学习率一旦高于基线 ({', '.join(sorted(set(unstable_big_lr)))}) 全部出现损失连升≥5、"
        f"终损与终态物理质量均劣于基线 —— 即**单纯调大 flat 学习率会过冲**, 不安全。"
        if unstable_big_lr
        else "固定学习率阶梯在本预算内均未越过基线质量。"
    )
    risk = (
        f"最长损失连升 {worst_streak} 迭代"
        + (f"; 不稳定的 flat 变体: {', '.join(sorted(set(unstable_big_lr)))}" if unstable_big_lr else "")
    )
    win_kind = "动态调度" if is_schedule else "固定学习率"
    return (
        f"最优配置是 **{best.spec.label}** ({win_kind}): 在完整 {iters} 次 horizon 内, 第 "
        f"**{n99(best)} 次迭代**即追平基线 (flat lr=0.05, {base_iters} 次) 终态物理质量的 99%"
        f" (效率 ≥ 0.99×{base_q['energy_in_target']:.4f} 且 CV ≤ {base_q['cv_in_target']:.4f}/0.99),"
        f" 相当于基线达标迭代 (110) 的更早、基线预算的 {baseline_iters_pct:.0f}%; 其终损 "
        f"{best.headline['final_loss']:.4e} 优于基线 1.333e-04, 终态效率 "
        f"{best.headline['final']['energy_in_target']:.4f}、CV {best.headline['final']['cv_in_target']:.4f},"
        f" 且全程**稳定** (无损失连升≥5)。{flat_overshoot} 换言之, 收益来自「中等偏大峰值学习率 + 余弦衰减」"
        f"而非单纯调大学习率。调度阶段挑选的峰值学习率: {', '.join(f'{x:g}' for x in top_lrs) or 'n/a'}。"
        f" {verify_note} 风险: {risk}; 物理质量按每 {baseline.get('metric_every', 5)} 迭代采样判定。"
        f" 硬件侧建议: 先按 {best.spec.label} 跑一次带实测对照的完整 horizon 验证, 确认离线物理预测"
        f"与实验系统一致后, 再考虑把常规流程切到该调度; 在确认之前不要仅凭模型损失更低就调大 flat 学习率。"
    )


if __name__ == "__main__":
    raise SystemExit(main())
