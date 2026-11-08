"""离线三轴基准：冻结式离焦物理预测模型（2026-11-07，issue #239 后续）。

全程离线：不 import ``ao_shaping.drivers``，不打开 SLM / 相机 / 激光。
数据与拟合全部来自磁盘存档（``data/debug/slm_transfer/`` 下的 npz 帧与
``forward_fit.json``）。三轴：

* **A 准确率** —— 冻结基础模型（不训练）在 ladder pass2（42 帧）与 r2 可用集
  （36 帧）上的 full-frame pearson、core-mask pearson（0.1·max 掩膜）、rel-L2；
  另报 ladder ±模 42 对方向 corr 均值。``+1`` 仅作参考行。
* **B 迁移学习速度** —— r2 域 memoryless 残差头（与 #239 相同的初始化、AdamW、
  划分与预算）：cold（随机初始化直接在 r2 训练）vs warm（先在 ladder pass1 训
  固定 iters，再从该初始化微调 r2）；记录 held-out pearson 曲线与达到该次运行
  自身最终质量的 90/95/99% 所需迭代数与墙钟秒，跨种子取中位数。
* **C 反向可微优化速度** —— differentiable 模块（``phase_sign=-1``）上以 Adam
  优化指令 Zernike 系数最小化核心 CV；零相位起步 vs 小随机像差热启动（3 种子）；
  每迭代墙钟（前向+反向+step，mean/median/p90，含分解）、达到
  CV ≤ 0.2/0.15/0.10 的迭代数与秒（或 not-reached）、初始与最终 CV。

可微模块的原生指令参数化是**整幅 128×128 相位增量**（``DifferentiableUniformity
Optimizer``），没有原生 Zernike 系数指令向量；本脚本在原生可微接口
（``FrozenDefocusForward.predict`` + ``UniformityObjective.cv``）之上加一层指令
Zernike 系数（Noll 5–15，基取模块自己的 ``_mode_basis``，与其 residual 模式同一
约定），梯度经 ``predict`` 反传到系数。

资产复用：加载 / 划分 / 指标 / 模型 / 训练循环等函数从
``scripts/slm/gated_recurrent_transfer_20261107.py``（#239）按路径 import；
该模块顶层只有常量与函数定义，import 无副作用。

已知 caveat（如实写入报告）：−1 极性目前作用于**整条指令相位**（含载波/倾斜），
U/C 基线复现在 −1 下失败（``loss_compare_20261107_phase_sign_m1/results.json``
的 validation.passed=false，d −0.0021 vs 存档 +0.1139）。本基准只测量、不修复、
不为凑数调参；−1 的绝对精度数字受该未解作用域问题影响。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._common import fmt_fixed_or_text


from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    FrozenDefocusForward,
    UniformityObjective,
)

# --- #239 资产按路径导入（模块名含数字不能直接 import；顶层无副作用） ---
_TRANSFER_PATH = ROOT / "scripts" / "slm" / "gated_recurrent_transfer_20261107.py"
_SPEC = importlib.util.spec_from_file_location("gated_recurrent_transfer_20261107", _TRANSFER_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - 环境问题
    raise ImportError(f"无法加载 {_TRANSFER_PATH}")
_TRANSFER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _TRANSFER  # dataclass 重建需要模块先入 sys.modules
_SPEC.loader.exec_module(_TRANSFER)

Frame = _TRANSFER.Frame

# --------------------------------------------------------------------------- #
# 常量与配置
# --------------------------------------------------------------------------- #
BASE_DIR = Path(_TRANSFER.BASE_DIR)
DEFAULT_OUT = BASE_DIR / "offline_benchmark_20261107"
FIT_REL = Path(_TRANSFER.FIT_PATH).relative_to(ROOT)

DEVICE_STATEMENT = (
    "离线分析，未占用任何设备：未打开 SLM / 相机 / 激光，未 import 任何 "
    "ao_shaping.drivers 模块；全部数字来自磁盘存档 npz 帧与冻结拟合 JSON，"
    "CPU 单机、固定种子、确定性运行。"
)
CAVEAT_TEXT = (
    "−1 极性目前作用于**整条指令相位**（含载波/倾斜），U/C 基线复现在 −1 下失败"
    "（data/debug/slm_transfer/loss_compare_20261107_phase_sign_m1/results.json 的 "
    "validation.passed=false，d −0.0021 vs 存档 +0.1139）。本基准只测量、不修复、"
    "不为凑数调参；−1 的绝对精度数字（A 轴各行、C 轴绝对 CV）受该未解作用域问题"
    "影响，应在该前提下解读。C 轴的**速度**结论不受影响：−1 仅把指令地形按系数"
    "镜像（loss(c) = loss₊₁(−c)），优化难度等价。"
)

B_SEEDS: tuple[int, ...] = tuple(int(s) for s in _TRANSFER.SEEDS)  # (0, 1, 2)
B_ITERS_FULL = int(_TRANSFER.ITERS)  # 300
B_CURVE_GRID = (0, 10, 20, 50, 100, 200, 300)
REACH_RATIOS = (0.90, 0.95, 0.99)

C_ITERS_FULL = 300
C_LR = 0.01  # 与模块原生 DifferentiableUniformityOptimizer 默认一致
C_MODES: tuple[int, ...] = tuple(range(5, 16))  # Noll 5–15（去 piston/整体倾斜）
C_RANDOM_RAD = 0.05  # 小随机像差：每系数 ~ U(−0.05, 0.05) rad
C_TARGETS = (0.20, 0.15, 0.10)  # 核心 CV（= √CV²）达标线


# --------------------------------------------------------------------------- #
# 数据加载与划分（口径与 #239 完全一致）
# --------------------------------------------------------------------------- #
@dataclass
class Domains:
    """两域帧 + 序列划分 + 各域归一化统计与掩膜。"""

    r2_usable: list[Frame]
    r2_train_seqs: list[list[Frame]]
    r2_val_seqs: list[list[Frame]]
    r2_stats: dict[str, tuple[torch.Tensor, torch.Tensor]]
    r2_resid_scale: float
    r2_core_mask: np.ndarray
    ladder_frames: list[Frame]
    ladder_train_seqs: list[list[Frame]]
    ladder_val_seqs: list[list[Frame]]
    ladder_all_seqs: list[list[Frame]]
    ladder_stats: dict[str, tuple[torch.Tensor, torch.Tensor]]
    ladder_resid_scale: float
    ladder_core_mask: np.ndarray


def _domain_normalisation(
    train_frames: list[Frame],
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], float]:
    stats = _TRANSFER.feature_stats(train_frames)
    resid = torch.stack([f.resid_low for f in train_frames])
    resid_scale = float(resid.std().item()) or float(_TRANSFER.RESID_SCALE_FALLBACK)
    return stats, resid_scale


def _core_mask(frames: list[Frame], level: float) -> np.ndarray:
    mean_image = np.mean([f.image for f in frames], axis=0)
    return mean_image >= level * float(np.max(mean_image))


def load_domains(
    forward_minus: FrozenDefocusForward, offset: np.ndarray, default_center: np.ndarray
) -> Domains:
    r2_usable = _TRANSFER.usable(_TRANSFER.load_frames(_TRANSFER.R2_DIR, None, offset, default_center))
    _TRANSFER.precompute(r2_usable, forward_minus)
    r2_seqs = _TRANSFER.group_sequences(r2_usable, "r2")

    ladder_frames: list[Frame] = []
    ladder_all_seqs: list[list[Frame]] = []
    for tag, directory in _TRANSFER.LADDER_DIRS.items():
        for pass_name in ("pass1", "pass2"):
            frames = _TRANSFER.usable(_TRANSFER.load_frames(directory, pass_name, offset, default_center))
            for frame in frames:
                frame.key = f"{tag}/{pass_name}/{frame.key}"
            _TRANSFER.precompute(frames, forward_minus)
            # 与 #239 相同：按采集批次划分（pass1 训练 / pass2 held-out）。
            # manifest 自己的 holdout 帧是另一分布（std 1.0–1.8 rad），本基准不用。
            train_frames = [f for f in frames if f.split == "train"]
            for frame in train_frames:
                frame.split = "train" if pass_name == "pass1" else "test"
            ladder_frames.extend(train_frames)
            ladder_all_seqs.append(sorted(train_frames, key=lambda f: f.ns))

    ladder_train_seqs = [s for s in ladder_all_seqs if all(f.split == "train" for f in s)]
    ladder_val_seqs = [s for s in ladder_all_seqs if all(f.split == "test" for f in s)]
    r2_train_seqs = [s for s in r2_seqs if all(f.split == "train" for f in s)]
    r2_val_seqs = [s for s in r2_seqs if all(f.split == "test" for f in s)]

    r2_stats, r2_scale = _domain_normalisation([f for seq in r2_train_seqs for f in seq])
    ladder_stats, ladder_scale = _domain_normalisation([f for seq in ladder_train_seqs for f in seq])
    return Domains(
        r2_usable=r2_usable,
        r2_train_seqs=r2_train_seqs,
        r2_val_seqs=r2_val_seqs,
        r2_stats=r2_stats,
        r2_resid_scale=r2_scale,
        r2_core_mask=_core_mask(r2_usable, 0.5),
        ladder_frames=ladder_frames,
        ladder_train_seqs=ladder_train_seqs,
        ladder_val_seqs=ladder_val_seqs,
        ladder_all_seqs=ladder_all_seqs,
        ladder_stats=ladder_stats,
        ladder_resid_scale=ladder_scale,
        ladder_core_mask=_core_mask(ladder_frames, 0.5),
    )


# --------------------------------------------------------------------------- #
# A 轴：冻结基础模型准确率
# --------------------------------------------------------------------------- #
def _accuracy_row(
    frames: list[Frame], plus_map: dict[str, np.ndarray] | None, core: np.ndarray
) -> dict[str, float]:
    preds = [f.base for f in frames] if plus_map is None else [plus_map[f.key] for f in frames]
    return {
        "full_pearson": float(
            np.mean([_TRANSFER.pearson(p, f.image) for p, f in zip(preds, frames)])
        ),
        "core_pearson": float(
            np.mean([_TRANSFER.pearson(p[core], f.image[core]) for p, f in zip(preds, frames)])
        ),
        "rel_l2": float(np.mean([_TRANSFER.rel_l2(p, f.image) for p, f in zip(preds, frames)])),
    }


def run_axis_a(
    domains: Domains, forward_plus: FrozenDefocusForward
) -> tuple[dict[str, Any], float]:
    t0 = time.perf_counter()
    all_frames = [*domains.ladder_frames, *domains.r2_usable]
    plus_map = {f.key: _TRANSFER.predict_frame(forward_plus, f.phase) for f in all_frames}

    row_defs: dict[str, list[Frame]] = {
        "ladder_pass2": [f for f in domains.ladder_frames if f.split == "test"],
        "ladder_p1p2": list(domains.ladder_frames),
        "r2_usable": list(domains.r2_usable),
        "r2_heldout": [f for f in domains.r2_usable if f.split == "test"],
    }
    rows: dict[str, dict[str, Any]] = {}
    for name, frames in row_defs.items():
        mean_image = np.mean([f.image for f in frames], axis=0)
        core = mean_image >= 0.1 * float(np.max(mean_image))
        rows[name] = {
            "n": len(frames),
            "core_pixels": int(core.sum()),
            "minus": _accuracy_row(frames, None, core),
            "plus": _accuracy_row(frames, plus_map, core),
        }

    pairs = _TRANSFER.direction_pairs(domains.ladder_all_seqs)
    direction = {
        "n_pairs": len(pairs),
        "minus": _TRANSFER.direction_metric({f.key: f.base for f in domains.ladder_frames}, pairs),
        "plus": _TRANSFER.direction_metric(plus_map, pairs),
    }
    return {"rows": rows, "direction": direction}, time.perf_counter() - t0


# --------------------------------------------------------------------------- #
# B 轴：迁移学习速度（cold vs warm）
# --------------------------------------------------------------------------- #
def _train_head(
    model: torch.nn.Module,
    train_seqs: list[list[Frame]],
    stats: dict[str, tuple[torch.Tensor, torch.Tensor]],
    resid_scale: float,
    val_seqs: list[list[Frame]],
    core_mask: np.ndarray,
    seed: int,
    iters: int,
    eval_every: int,
    do_eval: bool,
) -> list[dict[str, Any]]:
    """#239 同款训练循环（AdamW、序列级批），额外记录每行墙钟。"""
    model.train()
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(_TRANSFER.LR), weight_decay=float(_TRANSFER.WEIGHT_DECAY)
    )
    rng = np.random.default_rng(seed)
    history: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    if do_eval:
        metrics = _TRANSFER.evaluate(model, val_seqs, stats, resid_scale, core_mask, False)
        history.append({"iteration": 0, "loss": None, "elapsed_s": 0.0, **metrics})
    for it in range(1, iters + 1):
        seq = train_seqs[int(rng.integers(0, len(train_seqs)))]
        outs = _TRANSFER.run_sequence(model, seq, stats, False)
        loss = torch.stack(
            [
                F.mse_loss(out[0].reshape(-1), frame.resid_low / resid_scale)
                for out, frame in zip(outs, seq)
            ]
        ).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if do_eval and (it % eval_every == 0 or it == iters):
            metrics = _TRANSFER.evaluate(model, val_seqs, stats, resid_scale, core_mask, False)
            history.append(
                {
                    "iteration": it,
                    "loss": float(loss.item()),
                    "elapsed_s": time.perf_counter() - t0,
                    **metrics,
                }
            )
    return history


def run_b_arm(
    mode: str,
    seed: int,
    iters: int,
    eval_every: int,
    grid: tuple[int, ...],
    domains: Domains,
) -> dict[str, Any]:
    """单种子单模式。warm = 同种子同初始化（与 cold 一致）+ ladder pass1 热身。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = _TRANSFER.build_model("memoryless")
    warmup: dict[str, Any] | None = None
    if mode == "warm":
        tw = time.perf_counter()
        warm_hist = _train_head(
            model,
            domains.ladder_train_seqs,
            domains.ladder_stats,
            domains.ladder_resid_scale,
            domains.ladder_val_seqs,
            domains.ladder_core_mask,
            seed,
            iters,
            eval_every,
            do_eval=False,
        )
        ladder_pearson = _TRANSFER.evaluate(
            model,
            domains.ladder_val_seqs,
            domains.ladder_stats,
            domains.ladder_resid_scale,
            domains.ladder_core_mask,
            False,
        )["full_pearson"]
        warmup = {
            "iters": iters,
            "domain": "ladder pass1",
            "wall_s": time.perf_counter() - tw,
            "final_train_loss": warm_hist[-1]["loss"] if warm_hist else None,
            "ladder_heldout_full_pearson": ladder_pearson,
        }

    history = _train_head(
        model,
        domains.r2_train_seqs,
        domains.r2_stats,
        domains.r2_resid_scale,
        domains.r2_val_seqs,
        domains.r2_core_mask,
        seed,
        iters,
        eval_every,
        do_eval=True,
    )
    final = float(history[-1]["full_pearson"])
    curve = {
        str(row["iteration"]): float(row["full_pearson"])
        for row in history
        if int(row["iteration"]) in grid
    }
    warmup_s = float(warmup["wall_s"]) if warmup else 0.0
    reach: dict[str, dict[str, float | int | None]] = {}
    for ratio in REACH_RATIOS:
        target = ratio * final
        hit = next((r for r in history if float(r["full_pearson"]) >= target), None)
        hit_s = float(hit["elapsed_s"]) if hit else None
        reach[f"{ratio:.2f}"] = {
            "target": target,
            "iteration": int(hit["iteration"]) if hit else None,
            "elapsed_finetune_s": hit_s,
            "elapsed_total_s": (hit_s + warmup_s) if hit_s is not None else None,
        }
    wall_finetune = float(history[-1]["elapsed_s"])
    entry: dict[str, Any] = {
        "seed": seed,
        "curve": curve,
        "final_full_pearson": final,
        "reach": reach,
        "wall_finetune_s": wall_finetune,
        "wall_total_s": wall_finetune + warmup_s,
    }
    if warmup is not None:
        entry["warmup"] = warmup
    return entry


def _median_stat(values: list[float | int | None]) -> dict[str, Any]:
    clean = [float(v) for v in values if v is not None]
    return {
        "median": float(np.median(clean)) if clean else None,
        "n_reached": len(clean),
        "n": len(values),
    }


def summarize_b(runs: list[dict[str, Any]], grid: tuple[int, ...]) -> dict[str, Any]:
    return {
        "final_full_pearson": _median_stat([r["final_full_pearson"] for r in runs]),
        "curve": {
            str(it): _median_stat([r["curve"][str(it)] for r in runs]) for it in grid
        },
        "reach": {
            f"{ratio:.2f}": {
                "iteration": _median_stat([r["reach"][f"{ratio:.2f}"]["iteration"] for r in runs]),
                "elapsed_finetune_s": _median_stat(
                    [r["reach"][f"{ratio:.2f}"]["elapsed_finetune_s"] for r in runs]
                ),
                "elapsed_total_s": _median_stat(
                    [r["reach"][f"{ratio:.2f}"]["elapsed_total_s"] for r in runs]
                ),
            }
            for ratio in REACH_RATIOS
        },
        "wall_finetune_s": _median_stat([r["wall_finetune_s"] for r in runs]),
        "wall_total_s": _median_stat([r["wall_total_s"] for r in runs]),
        "warmup_wall_s": _median_stat(
            [r["warmup"]["wall_s"] for r in runs if "warmup" in r]
        ),
    }


def run_axis_b(
    seeds: tuple[int, ...], iters: int, eval_every: int, grid: tuple[int, ...], domains: Domains
) -> tuple[dict[str, Any], float]:
    t0 = time.perf_counter()
    arms: dict[str, Any] = {}
    for mode in ("cold", "warm"):
        runs = [run_b_arm(mode, seed, iters, eval_every, grid, domains) for seed in seeds]
        arms[mode] = {"per_seed": {str(r["seed"]): r for r in runs}, "summary": summarize_b(runs, grid)}
    return arms, time.perf_counter() - t0


# --------------------------------------------------------------------------- #
# C 轴：反向可微优化速度（指令 Zernike 系数，模型域）
# --------------------------------------------------------------------------- #
def _predict_single(forward: FrozenDefocusForward, phase: torch.Tensor) -> torch.Tensor:
    pred = forward.predict(phase)
    return pred[0] if pred.ndim == 3 else pred


def _run_c_one(
    forward: FrozenDefocusForward,
    objective: UniformityObjective,
    basis: torch.Tensor,
    coeffs0: torch.Tensor,
    start: str,
    seed: int | None,
    iters: int,
) -> dict[str, Any]:
    c = torch.nn.Parameter(coeffs0.clone())
    opt = torch.optim.Adam([c], lr=C_LR)
    cv2_series: list[float] = []  # cv2_series[i] = 第 i 步之后（即第 i+1 次迭代前）的核心 CV²
    fwd_ms: list[float] = []
    bwd_ms: list[float] = []
    step_ms: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        command = torch.einsum("k,khw->hw", c, basis)
        loss = objective.cv(_predict_single(forward, command))
        t1 = time.perf_counter()
        opt.zero_grad()
        loss.backward()
        t2 = time.perf_counter()
        opt.step()
        t3 = time.perf_counter()
        cv2_series.append(float(loss.item()))
        fwd_ms.append((t1 - t0) * 1e3)
        bwd_ms.append((t2 - t1) * 1e3)
        step_ms.append((t3 - t2) * 1e3)
    with torch.no_grad():
        command = torch.einsum("k,khw->hw", c, basis)
        final_cv2 = float(objective.cv(_predict_single(forward, command)).item())

    cv2_full = cv2_series + [final_cv2]  # 下标 i = 完成 i 步后的 CV²
    cv_full = [float(np.sqrt(max(v, 0.0))) for v in cv2_full]
    total_ms = [f + b + s for f, b, s in zip(fwd_ms, bwd_ms, step_ms)]
    cum_s = np.cumsum(total_ms) / 1e3  # cum_s[i] = 完成 i+1 步后的累计墙钟

    def reach(target_cv: float) -> dict[str, float | int | None]:
        for i, cv in enumerate(cv_full):
            if cv <= target_cv:
                return {
                    "iteration": i,
                    "seconds": 0.0 if i == 0 else float(cum_s[i - 1]),
                }
        return {"iteration": None, "seconds": None}

    return {
        "start": start,
        "seed": seed,
        "initial_cv2": cv2_full[0],
        "final_cv2": cv2_full[-1],
        "initial_cv": cv_full[0],
        "final_cv": cv_full[-1],
        "cv2_series": cv2_full,
        "cv_series": cv_full,
        "reach": {f"cv_le_{t:.2f}": reach(t) for t in C_TARGETS},
        "timing": {
            "ms_per_iter": {
                "mean": float(np.mean(total_ms)),
                "median": float(np.median(total_ms)),
                "p90": float(np.percentile(total_ms, 90)),
            },
            "forward_ms_mean": float(np.mean(fwd_ms)),
            "backward_ms_mean": float(np.mean(bwd_ms)),
            "step_ms_mean": float(np.mean(step_ms)),
            "wall_s": float(np.sum(total_ms) / 1e3),
        },
    }


def run_axis_c(
    forward_minus: FrozenDefocusForward, seeds: tuple[int, ...], iters: int
) -> tuple[dict[str, Any], float]:
    t0 = time.perf_counter()
    # 冻结掩膜：零指令预测的 0.1·max 核心，与起点无关、全程固定（可比较）。
    with torch.no_grad():
        base_pred = _predict_single(forward_minus, torch.zeros(_TRANSFER.GRID, _TRANSFER.GRID))
    base_np = base_pred.detach().cpu().numpy().astype(np.float32)
    core = base_np >= 0.1 * float(base_np.max())
    objective = UniformityObjective(
        mask_box=np.ones_like(core), mask_core=core, w_shape=0.0, w_phase=0.0
    )
    # 模块自己的 Noll 基（与其 residual 模式同一约定），保证指令参数化口径一致。
    basis = torch.stack([forward_minus._mode_basis(j) for j in C_MODES])

    runs: list[dict[str, Any]] = []
    specs: list[tuple[str, int | None]] = [("zero", None)] + [("random", s) for s in seeds]
    for start, seed in specs:
        if seed is None:
            coeffs0 = torch.zeros(len(C_MODES))
        else:
            rng = np.random.default_rng(seed)
            coeffs0 = torch.from_numpy(
                rng.uniform(-C_RANDOM_RAD, C_RANDOM_RAD, len(C_MODES)).astype(np.float32)
            )
        run = _run_c_one(forward_minus, objective, basis, coeffs0, start, seed, iters)
        runs.append(run)
        logger.info(
            "C 轴 {start} seed={seed}: CV {ic:.3f}→{fc:.3f}，{ms:.1f} ms/iter（中位）",
            start=start,
            seed=seed,
            ic=run["initial_cv"],
            fc=run["final_cv"],
            ms=run["timing"]["ms_per_iter"]["median"],
        )

    by_start: dict[str, list[dict[str, Any]]] = {}
    for r in runs:
        by_start.setdefault(r["start"], []).append(r)
    summary = {
        start: {
            "n_runs": len(group),
            "initial_cv": _median_stat([r["initial_cv"] for r in group]),
            "final_cv": _median_stat([r["final_cv"] for r in group]),
            "initial_cv2": _median_stat([r["initial_cv2"] for r in group]),
            "final_cv2": _median_stat([r["final_cv2"] for r in group]),
            "ms_per_iter_median": _median_stat(
                [r["timing"]["ms_per_iter"]["median"] for r in group]
            ),
            "ms_per_iter_p90": _median_stat([r["timing"]["ms_per_iter"]["p90"] for r in group]),
            "reach": {
                key: {
                    "iteration": _median_stat([r["reach"][key]["iteration"] for r in group]),
                    "seconds": _median_stat([r["reach"][key]["seconds"] for r in group]),
                }
                for key in group[0]["reach"]
            },
        }
        for start, group in by_start.items()
    }
    return {
        "mask": {
            "shape": list(core.shape),
            "core_pixels": int(core.sum()),
            "definition": "零指令预测 ≥ 0.1·max，优化前冻结",
        },
        "command_modes_noll": list(C_MODES),
        "runs": runs,
        "summary": summary,
    }, time.perf_counter() - t0


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #
_fmt = fmt_fixed_or_text


def _fmt_iter(seconds: float | int | None, digits: int = 1) -> str:
    return "—" if seconds is None else f"{seconds:.{digits}f}"


def _axis_a_lines(results: dict[str, Any]) -> list[str]:
    rows = results["axis_a"]["rows"]
    names = {
        "ladder_pass2": "ladder pass2（held-out）",
        "ladder_p1p2": "ladder pass1+pass2（#239 基线口径，参考）",
        "r2_usable": "r2 可用集（36 帧）",
        "r2_heldout": "r2 held-out（参考）",
    }
    lines = ["| 评估集 | n | sign | full Pearson | core Pearson（0.1·max） | rel-L2 |",
             "|---|---|---|---|---|---|"]
    for name in names:
        row = rows[name]
        for sign, tag in (("minus", "−1"), ("plus", "+1")):
            m = row[sign]
            lines.append(
                f"| {names[name]} | {row['n']} | {tag} | {_fmt(m['full_pearson'])} | "
                f"{_fmt(m['core_pearson'])} | {_fmt(m['rel_l2'])} |"
            )
    direction = results["axis_a"]["direction"]
    lines += [
        "",
        f"方向一致性（ladder ±模对，n={direction['n_pairs']}）：",
        "",
        "| sign | 方向 corr 均值 | std | 对数 |",
        "|---|---|---|---|",
        f"| −1 | {_fmt(direction['minus']['mean'])} | {_fmt(direction['minus']['std'])} | {direction['minus']['n']} |",
        f"| +1 | {_fmt(direction['plus']['mean'])} | {_fmt(direction['plus']['std'])} | {direction['plus']['n']} |",
    ]
    m = rows["r2_usable"]["minus"]
    p = rows["r2_usable"]["plus"]
    lines += [
        "",
        f"注：+1 行仅为参考（−1 为硬件实测极性）。r2 可用集上 −1 vs +1 的 full Pearson "
        f"{_fmt(m['full_pearson'])} vs {_fmt(p['full_pearson'])}、rel-L2 {_fmt(m['rel_l2'])} vs "
        f"{_fmt(p['rel_l2'])}，差距即上方 caveat 所述「整条指令相位取反」作用域未解问题在数字上的"
        "体现（ladder 集两者接近，r2 集差距明显）；此处只记录，不据此调参。",
    ]
    return lines


def _axis_b_lines(results: dict[str, Any], grid: tuple[int, ...]) -> list[str]:
    cold = results["axis_b"]["cold"]["summary"]
    warm = results["axis_b"]["warm"]["summary"]
    lines = [
        "held-out full Pearson 中位数曲线（跨种子）：",
        "",
        "| iter | cold | warm |",
        "|---|---|---|",
    ]
    for it in grid:
        lines.append(
            f"| {it} | {_fmt(cold['curve'][str(it)]['median'])} | {_fmt(warm['curve'][str(it)]['median'])} |"
        )
    lines += [
        "",
        "达到该次运行自身最终质量的 90/95/99%（跨种子中位；括号内为达标种子数/总种子数）。"
        "口径提示：达标按时间序列首次满足判定，质量曲线非单调（见上表，先升后回落）时，"
        "0 iter 即达标只说明初值已高于该阈值，应与曲线表并读：",
        "",
        "| 比例 | cold 迭代 | cold 墙钟 s | warm 迭代 | warm 墙钟 s（仅微调） | warm 墙钟 s（含热身） |",
        "|---|---|---|---|---|---|",
    ]
    for ratio in REACH_RATIOS:
        key = f"{ratio:.2f}"
        ci, cs = cold["reach"][key]["iteration"], cold["reach"][key]["elapsed_finetune_s"]
        wi = warm["reach"][key]["iteration"]
        ws = warm["reach"][key]["elapsed_finetune_s"]
        wt = warm["reach"][key]["elapsed_total_s"]
        cn, wn = ci["n_reached"], wi["n_reached"]
        lines.append(
            f"| {ratio:.0%} | {_fmt_iter(ci['median'], 0)}（{cn}/{ci['n']}） | {_fmt_iter(cs['median'])} | "
            f"{_fmt_iter(wi['median'], 0)}（{wn}/{wi['n']}） | {_fmt_iter(ws['median'])} "
            f"| {_fmt_iter(wt['median'])} |"
        )
    final_c = cold["final_full_pearson"]["median"]
    final_w = warm["final_full_pearson"]["median"]
    warmup_s = warm["warmup_wall_s"]["median"]
    lines += [
        "",
        f"最终 held-out full Pearson 中位：cold {_fmt(final_c)} vs warm {_fmt(final_w)}；"
        f"warm 热身（ladder pass1 × {results['config']['axis_b']['iters']} iters）中位墙钟 {_fmt(warmup_s, 2)} s，"
        f"热身后 ladder held-out pearson 每种子 "
        + "、".join(
            f"seed {seed} {_fmt(entry['warmup']['ladder_heldout_full_pearson'])}"
            for seed, entry in results["axis_b"]["warm"]["per_seed"].items()
        )
        + "。",
    ]
    return lines


def _b_conclusion(results: dict[str, Any]) -> str:
    cold = results["axis_b"]["cold"]["summary"]
    warm = results["axis_b"]["warm"]["summary"]
    key = "0.95"
    ci = cold["reach"][key]["iteration"]
    wi = warm["reach"][key]["iteration"]
    cs = cold["reach"][key]["elapsed_finetune_s"]
    ws = warm["reach"][key]["elapsed_finetune_s"]
    parts = [
        f"以 95% 口径为例：cold 中位 {_fmt_iter(ci['median'], 0)} iters / {_fmt_iter(cs['median'])} s"
        f"（{ci['n_reached']}/{ci['n']} 种子达标），"
        f"warm 中位 {_fmt_iter(wi['median'], 0)} iters / {_fmt_iter(ws['median'])} s"
        f"（{wi['n_reached']}/{wi['n']} 种子达标；含热身总墙钟中位 "
        f"{_fmt_iter(warm['reach'][key]['elapsed_total_s']['median'])} s）。",
    ]
    if wi["median"] is not None and ci["median"] is not None:
        if wi["median"] < ci["median"] and (ws["median"] or 0) < (cs["median"] or 0):
            parts.append("ladder 热启动在本设置下同时用更少的 r2 侧迭代与更短的墙钟达到同比例质量。")
        elif wi["median"] < ci["median"]:
            parts.append("热启动减少 r2 侧迭代数，但墙钟口径未同步占优（逐迭代开销见种子明细）。")
        elif wi["median"] > ci["median"]:
            parts.append("热启动未带来迭代优势（推测：warm 头先学到的 ladder 残差结构在 r2 域需先改写）。")
        else:
            parts.append("两种启动方式的达标迭代数在当前种子数下不可分辨（曲线采样间隔 10 iters，分辨率限制）。")
    else:
        parts.append("存在未达 95% 的运行，迭代/时间比较不完整（见上表达标数）。")
    return "".join(parts)


def _axis_c_lines(results: dict[str, Any]) -> list[str]:
    runs = results["axis_c"]["runs"]
    lines = [
        "| 起点 | seed | ms/iter 中位（p90） | 前向/反向/step ms | 初始 CV² | 最终 CV² | 初始 CV | 最终 CV "
        "| ≤0.20 迭代/秒 | ≤0.15 | ≤0.10 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        t = r["timing"]

        def cell(run: dict[str, Any], key: str) -> str:
            hit = run["reach"][key]
            if hit["iteration"] is None:
                return "未达"
            return f"{hit['iteration']} / {hit['seconds']:.1f}s"

        lines.append(
            f"| {r['start']} | {_fmt(r['seed'])} | {_fmt(t['ms_per_iter']['median'], 1)}"
            f"（{_fmt(t['ms_per_iter']['p90'], 1)}） "
            f"| {t['forward_ms_mean']:.1f} / {t['backward_ms_mean']:.1f} / {t['step_ms_mean']:.1f} "
            f"| {_fmt(r['initial_cv2'], 3)} | {_fmt(r['final_cv2'], 3)} "
            f"| {_fmt(r['initial_cv'], 3)} | {_fmt(r['final_cv'], 3)} "
            f"| {cell(r, 'cv_le_0.20')} | {cell(r, 'cv_le_0.15')} | {cell(r, 'cv_le_0.10')} |"
        )
    mask = results["axis_c"]["mask"]
    lines += [
        "",
        f"核心掩膜：{mask['definition']}（{mask['core_pixels']} px @ {mask['shape'][0]}×{mask['shape'][1]}），"
        "全程冻结；指令系数为 Noll " + "–".join(map(str, (C_MODES[0], C_MODES[-1])))
        + f"（{len(C_MODES)} 个，piston/整体倾斜不进指令）；损失为模块原生可微口径 = 掩膜内 CV²"
        "（``UniformityObjective.cv``），达标列按核心 CV = √CV² 对 0.20/0.15/0.10 判定。",
    ]
    return lines


def _c_conclusion(results: dict[str, Any]) -> str:
    summary = results["axis_c"]["summary"]
    iters = results["config"]["axis_c"]["iters"]
    parts: list[str] = []
    zero = summary.get("zero")
    if zero is not None:
        hit02 = zero["reach"]["cv_le_0.20"]["iteration"]
        parts.append(
            f"零相位起步：CV {zero['initial_cv']['median']:.3f} → {zero['final_cv']['median']:.3f}；"
            + (
                f"{iters} iters 内把核心 CV 压到 ≤0.20（iter {hit02['median']}）。"
                if hit02["median"] is not None
                else (
                    f"{iters} iters 内未达到 CV ≤ 0.20"
                    f"（核心 CV 停在约 {zero['final_cv']['median']:.2f}），如实报告未达标。"
                )
            )
        )
    rnd = summary.get("random")
    if rnd is not None:
        reach_txt = []
        for target in C_TARGETS:
            key = f"cv_le_{target:.2f}"
            it = rnd["reach"][key]["iteration"]
            text = (
                f"{_fmt_iter(it['median'], 0)} iters"
                if it["median"] is not None
                else "未达"
            )
            reach_txt.append(f"≤{target:.2f}：{text}（{it['n_reached']}/{it['n']} 种子）")
        parts.append(
            f"小随机像差热启动（中位）：CV {rnd['initial_cv']['median']:.3f} → "
            f"{rnd['final_cv']['median']:.3f}；按阈值 " + "；".join(reach_txt) + "。"
        )
    ms = summary.get("random", summary.get("zero"))
    if ms is not None:
        parts.append(
            f"每迭代墙钟（前向+反向+step）中位 {ms['ms_per_iter_median']['median']:.1f} ms"
            f"（p90 {ms['ms_per_iter_p90']['median']:.1f} ms）。"
        )
    return "".join(parts)


def write_report(results: dict[str, Any], path: Path) -> None:
    cfg = results["config"]
    grid = tuple(cfg["axis_b"]["curve_grid"])
    data = results["data"]
    lines = [
        "# 冻结式离焦物理预测模型 · 离线三轴基准（2026-11-07）",
        "",
        f"生成：{results['generated_utc']}　模式：{results['mode']}　"
        "上游：issue #239（``gated_recurrent_transfer_20261107``）的冻结模型与数据划分。",
        "",
        "## 设备声明",
        "",
        results["device_statement"],
        "",
        "## 已知问题（caveat，如实记录）",
        "",
        results["caveat"]["text"],
        "",
        "## 数据与划分",
        "",
        f"- 拟合：``{FIT_REL.as_posix()}``（``phase_sign=-1`` 为硬件实测极性，issue #235）。",
        f"- r2：{data['r2_usable']} 可用帧（train {data['r2_train']} / held-out {data['r2_val']}，"
        f"序列 {data['r2_train_seqs']}/{data['r2_val_seqs']}）。",
        f"- ladder：{data['ladder']} 可用帧（pass1 训练 {data['ladder_train']} / pass2 held-out "
        f"{data['ladder_val']}，序列 {data['ladder_train_seqs']}/{data['ladder_val_seqs']}）；"
        "manifest holdout 帧为另一分布，本基准不使用。",
        "- core 掩膜口径：A 轴每行 0.1·max（该行均值图）；B 轴评估沿用 #239 的 0.5·max"
        "（仅影响 core 参考列，B 的曲线与达标全部用 full pearson）。",
        "",
        "## A 轴：冻结基础模型准确率（−1 主变体，+1 仅作参考行）",
        "",
        *_axis_a_lines(results),
        "",
        "## B 轴：迁移学习速度（r2 域，memoryless 头，AdamW lr=5e-3 / wd=1e-4，"
        f"{cfg['axis_b']['iters']} iters，{len(cfg['axis_b']['seeds'])} 种子）",
        "",
        *_axis_b_lines(results, grid),
        "",
        f"**结论**：{_b_conclusion(results)}",
        "",
        "说明：warm 先在 ladder pass1 训（与主预算相同 iters，用 ladder 域自身的归一化统计），"
        "再以同权重微调 r2（用 r2 域统计）——跨域特征归一化切换是热启动必须付出的代价之一，"
        "计入上述对比。优化器/数据划分/初始化与 #239 完全一致（输出层零初始化，cold 的 "
        "iter 0 即冻结基础预测）。",
        "",
        "## C 轴：反向可微优化速度（模型域，phase_sign=−1，Adam lr="
        f"{cfg['axis_c']['lr']}，{cfg['axis_c']['iters']} iters）",
        "",
        *_axis_c_lines(results),
        "",
        f"**结论**：{_c_conclusion(results)}",
        "",
        "可微模块的原生指令参数化是整幅 128×128 相位增量；本轴在原生可微接口上取指令 "
        "Zernike 系数（基 = 模块自身 Noll 基）。模块对指令相位的 −1 作用域见上方 caveat："
        "C 的速度结论不受影响（地形按系数镜像等价）。",
        "",
        "## 运行耗时",
        "",
        "| 阶段 | 秒 |",
        "|---|---|",
        f"| 数据加载与基础预测预计算 | {results['timing']['load_s']:.1f} |",
        f"| A 轴 | {results['timing']['axis_a_s']:.1f} |",
        f"| B 轴 | {results['timing']['axis_b_s']:.1f} |",
        f"| C 轴 | {results['timing']['axis_c_s']:.1f} |",
        f"| 总计 | {results['timing']['total_s']:.1f} |",
        "",
        "总计为脚本内部计时（数据加载 + 三轴），不含解释器启动与 numpy/torch 导入。",
        "",
        "## 复现",
        "",
        "```powershell",
        '$env:PYTHONPATH="src"; .venv\\Scripts\\python.exe scripts/slm/offline_benchmark_20261107.py',
        "# 快速冒烟（1 种子、100 iters）：",
        '$env:PYTHONPATH="src"; .venv\\Scripts\\python.exe scripts/slm/offline_benchmark_20261107.py --quick',
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quick", action="store_true", help="1 种子、100 iters 冒烟")
    parser.add_argument("--threads", type=int, default=min(6, os.cpu_count() or 1))
    args = parser.parse_args(argv)
    torch.set_num_threads(max(1, args.threads))

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(out / "run.log", mode="w", encoding="utf-8", level="INFO", backtrace=False, diagnose=False)
    logger.info(DEVICE_STATEMENT)

    quick = args.quick
    seeds: tuple[int, ...] = (B_SEEDS[0],) if quick else B_SEEDS
    b_iters = 100 if quick else B_ITERS_FULL
    c_iters = 100 if quick else C_ITERS_FULL
    eval_every = int(_TRANSFER.EVAL_EVERY)
    grid = tuple(it for it in B_CURVE_GRID if it <= b_iters)

    started = time.perf_counter()
    fit = _TRANSFER.load_fit()
    forward_minus, offset = _TRANSFER.make_forward(fit, -1.0)
    forward_plus, _ = _TRANSFER.make_forward(fit, +1.0)
    default_center = np.asarray(fit["camera_crop_center"], dtype=np.float64)

    t_load = time.perf_counter()
    domains = load_domains(forward_minus, offset, default_center)
    load_s = time.perf_counter() - t_load
    logger.info(
        "数据就绪：r2 {r2} 可用帧 / ladder {ladder} 可用帧（{load:.1f}s）",
        r2=len(domains.r2_usable),
        ladder=len(domains.ladder_frames),
        load=load_s,
    )

    axis_a, a_s = run_axis_a(domains, forward_plus)
    logger.info("A 轴完成（{s:.1f}s）", s=a_s)

    axis_b, b_s = run_axis_b(seeds, b_iters, eval_every, grid, domains)
    logger.info("B 轴完成（{s:.1f}s）", s=b_s)

    axis_c, c_s = run_axis_c(forward_minus, seeds, c_iters)
    logger.info("C 轴完成（{s:.1f}s）", s=c_s)

    total_s = time.perf_counter() - started
    results: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "quick" if quick else "full",
        "device_statement": DEVICE_STATEMENT,
        "command": " ".join(sys.argv),
        "caveat": {
            "text": CAVEAT_TEXT,
            "loss_compare_reference": (
                "data/debug/slm_transfer/loss_compare_20261107_phase_sign_m1/results.json"
            ),
            "stored_defocus_diopter": 0.1139,
            "reproduced_defocus_diopter": -0.0021,
            "validation_passed": False,
        },
        "config": {
            "threads": args.threads,
            "phase_sign": -1.0,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "determinism": "CPU、固定种子（B/C 每 run torch.manual_seed + default_rng(seed)）",
            "data": {
                "fit": str(_TRANSFER.FIT_PATH),
                "r2_dir": str(_TRANSFER.R2_DIR),
                "ladder_dirs": {k: str(v) for k, v in _TRANSFER.LADDER_DIRS.items()},
            },
            "axis_a": {
                "core_mask": "每行均值图 ≥ 0.1·max",
                "rows": ["ladder_pass2", "ladder_p1p2", "r2_usable", "r2_heldout"],
                "direction_pairs_source": "ladder pass1+pass2 全部可用序列（#239 口径）",
            },
            "axis_b": {
                "arm": "memoryless",
                "optimizer": "AdamW",
                "lr": float(_TRANSFER.LR),
                "weight_decay": float(_TRANSFER.WEIGHT_DECAY),
                "iters": b_iters,
                "warmup_iters": b_iters,
                "eval_every": eval_every,
                "seeds": list(seeds),
                "reach_ratios": list(REACH_RATIOS),
                "curve_grid": list(grid),
                "hidden": int(_TRANSFER.HIDDEN),
                "feat": int(_TRANSFER.FEAT),
                "out_res": int(_TRANSFER.OUT_RES),
                "warm_start_note": (
                    "warm 先在 ladder pass1 训 warmup_iters（ladder 域统计），"
                    "再以同权重在 r2 微调（r2 域统计），优化器重启"
                ),
            },
            "axis_c": {
                "command_parametrisation": (
                    "指令 Zernike 系数（模块原生可微接口之上加一层；基 = FrozenDefocusForward"
                    "._mode_basis，与其 residual 模式同一约定）；模块原生参数化是整幅 128×128 "
                    "相位增量，无原生系数向量"
                ),
                "modes_noll": list(C_MODES),
                "optimizer": "Adam",
                "lr": C_LR,
                "iters": c_iters,
                "seeds_random_start": list(seeds),
                "random_start": f"每系数 ~ U(−{C_RANDOM_RAD}, +{C_RANDOM_RAD}) rad",
                "objective": (
                    "UniformityObjective.cv（掩膜内 CV²，模块原生可微口径）；"
                    "达标线 0.20/0.15/0.10 按核心 CV = √CV² 判定"
                ),
                "mask": "零指令预测 ≥ 0.1·max，优化前冻结",
                "timing_split": "forward=指令合成+predict+cv；backward=zero_grad+backward；step=opt.step",
            },
        },
        "data": {
            "r2_usable": len(domains.r2_usable),
            "r2_train": sum(len(s) for s in domains.r2_train_seqs),
            "r2_val": sum(len(s) for s in domains.r2_val_seqs),
            "r2_train_seqs": len(domains.r2_train_seqs),
            "r2_val_seqs": len(domains.r2_val_seqs),
            "ladder": len(domains.ladder_frames),
            "ladder_train": sum(len(s) for s in domains.ladder_train_seqs),
            "ladder_val": sum(len(s) for s in domains.ladder_val_seqs),
            "ladder_train_seqs": len(domains.ladder_train_seqs),
            "ladder_val_seqs": len(domains.ladder_val_seqs),
            "direction_pairs": axis_a["direction"]["n_pairs"],
        },
        "axis_a": axis_a,
        "axis_b": axis_b,
        "axis_c": axis_c,
        "timing": {
            "load_s": load_s,
            "axis_a_s": a_s,
            "axis_b_s": b_s,
            "axis_c_s": c_s,
            "total_s": total_s,
        },
    }

    (out / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(results, out / "REPORT.md")
    logger.info(
        "写出 {json} 与 {report}；总耗时 {total:.1f}s",
        json=out / "results.json",
        report=out / "REPORT.md",
        total=total_s,
    )
    logger.info("B 轴：cold final 中位 {c:.4f}，warm final 中位 {w:.4f}",
                c=axis_b["cold"]["summary"]["final_full_pearson"]["median"],
                w=axis_b["warm"]["summary"]["final_full_pearson"]["median"])
    logger.info("C 轴：zero CV {z0:.3f}→{z1:.3f}，random CV（中位）{r0:.3f}→{r1:.3f}",
                z0=axis_c["summary"]["zero"]["initial_cv"]["median"],
                z1=axis_c["summary"]["zero"]["final_cv"]["median"],
                r0=axis_c["summary"]["random"]["initial_cv"]["median"],
                r1=axis_c["summary"]["random"]["final_cv"]["median"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
