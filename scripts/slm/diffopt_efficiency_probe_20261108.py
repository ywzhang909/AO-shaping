"""可微优化效率探针（离线、CPU；不占用任何设备）。

问题来源：``offline_benchmark_20261107.py`` 的 C 轴在 300 iters 内核心 CV 只从
0.648 走到 0.339（目标 0.20），每步 15–17 ms。本探针把「每步成本」「收敛要多少步」
「参数化地板」三件事拆开量化，给出可复算的改进候选，而不是凭印象调参。

五组探针：

* **cost** —— 前向/反向/Adam step 分解；``fft_size`` 512→256 的代价与精度
  （对基线强度的 rel-L2、核心掩膜 IoU）；线程数 1/2/4/6/12 的墙钟；批量起点吞吐
  （一次前向跑 B 个起点）。``grid_size`` 不可单方面缩小（checkpoint 内残差权重绑定）。
* **convergence** —— Adam 与 SGD(momentum) 各扫学习率 0.01/0.03/0.1/0.3，4 个起点，
  300 iters，记录 CV 曲线与达到 CV≤0.20/0.15/0.10 的迭代数/秒数。
* **lbfgs** —— 11 个参数上 LBFGS(strong_wolfe) 的学习率与线搜索行为。
* **floor** —— 最好设置跑到 1000 iters，并把模式集从 Noll 5–15 扩到 5–25 / 5–35，
  回答「是优化器慢还是 11 个模式到不了 0.20」。
* **schedule** —— 粗网格(fft 256)先跑 N1 步再切 512 精修，与全程 512 等墙钟预算对比。

设备纪律：全程只读磁盘上的冻结 checkpoint 与代码，CPU，固定种子；未打开 SLM / 相机 /
激光，未 import ``ao_shaping.drivers``。产物写 ``data/debug/``（不入库）。

⚠ 冻结 checkpoint 路径不设独立默认值：本脚本复用
``offline_benchmark_20261107._TRANSFER.load_fit()`` 的同一路径常量，避免出现第二份
"历史光路配置"；换光路后必须先更新基准脚本的 checkpoint，再跑本探针。
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    DifferentiableUniformityOptimizer,
    FrozenDefocusForward,
    UniformityObjective,
)

DEVICE_STATEMENT = (
    "离线分析，未占用任何设备：未打开 SLM / 相机 / 激光，未 import 任何 "
    "ao_shaping.drivers 模块；全部数字来自磁盘上的冻结 checkpoint 与纯 CPU 前向/反向，"
    "固定种子、确定性运行。"
)

DEFAULT_OUT = ROOT / "data" / "debug" / "slm_transfer" / "diffopt_efficiency_20261108"
CURVE_AT = (0, 1, 2, 5, 10, 20, 50, 100, 150, 200, 300, 500, 1000)
TARGETS = (0.20, 0.15, 0.10)


def _load_bench():
    """按路径导入基准脚本，复用它的 checkpoint 加载与 C 轴参数化（同一口径）。"""
    path = ROOT / "scripts" / "slm" / "offline_benchmark_20261107.py"
    spec = importlib.util.spec_from_file_location("offline_benchmark_20261107", path)
    if spec is None or spec.loader is None:  # pragma: no cover - 环境问题
        raise ImportError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# 构造：冻结前向 + 掩膜 + 模式基
# --------------------------------------------------------------------------- #
def build_forward(bench, *, fft_size: int | None = None):
    fit = bench._TRANSFER.load_fit()
    forward, _offset = bench._TRANSFER.make_forward(fit, -1.0)
    if fft_size is None:
        return forward
    new = dataclasses.replace(forward.params, fft_size=int(fft_size))
    return FrozenDefocusForward(new, forward.camera_xy, device="cpu")


def predict_one(forward: FrozenDefocusForward, phase: torch.Tensor) -> torch.Tensor:
    pred = forward.predict(phase)
    return pred[0] if pred.ndim == 3 else pred


def build_objective(forward: FrozenDefocusForward, bench) -> tuple[UniformityObjective, np.ndarray]:
    with torch.no_grad():
        base = predict_one(forward, torch.zeros(int(bench._TRANSFER.GRID), int(bench._TRANSFER.GRID)))
    base_np = base.detach().cpu().numpy().astype(np.float32)
    core = base_np >= 0.1 * float(base_np.max())
    objective = UniformityObjective(
        mask_box=np.ones_like(core), mask_core=core, w_shape=0.0, w_phase=0.0
    )
    return objective, core


def build_basis(forward: FrozenDefocusForward, modes) -> torch.Tensor:
    return torch.stack([forward._mode_basis(int(j)) for j in modes])


def command_of(forward: FrozenDefocusForward, basis: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    size = int(forward.params.grid_size)
    return torch.einsum("k,khw->hw", coeffs, basis).reshape(size, size)


# --------------------------------------------------------------------------- #
# cost：每步成本分解
# --------------------------------------------------------------------------- #
def warm_up(
    forward: FrozenDefocusForward,
    objective: UniformityObjective,
    basis: torch.Tensor,
    *,
    iters: int = 120,
    lr: float = 0.01,
) -> None:
    """把 CPU/分配器带到稳态：第一段计时会撞到 boost 时钟，不预热会把 1 次抖动写成结论。"""
    c = torch.nn.Parameter(torch.zeros(int(basis.shape[0])))
    opt = torch.optim.Adam([c], lr=lr)
    for _ in range(iters):
        loss = objective.cv(predict_one(forward, command_of(forward, basis, c)))
        opt.zero_grad()
        loss.backward()
        opt.step()


def measure_cost(
    forward: FrozenDefocusForward,
    objective: UniformityObjective,
    basis: torch.Tensor,
    coeffs0: torch.Tensor,
    *,
    iters: int = 30,
    lr: float = 0.01,
) -> dict[str, Any]:
    c = torch.nn.Parameter(coeffs0.clone())
    opt = torch.optim.Adam([c], lr=lr)
    fwd, bwd, step = [], [], []
    for _ in range(iters):
        t0 = time.perf_counter()
        loss = objective.cv(predict_one(forward, command_of(forward, basis, c)))
        t1 = time.perf_counter()
        opt.zero_grad()
        loss.backward()
        t2 = time.perf_counter()
        opt.step()
        t3 = time.perf_counter()
        fwd.append((t1 - t0) * 1e3)
        bwd.append((t2 - t1) * 1e3)
        step.append((t3 - t2) * 1e3)
    total = [f + b + s for f, b, s in zip(fwd, bwd, step)]
    return {
        "forward_ms": float(np.median(fwd)),
        "backward_ms": float(np.median(bwd)),
        "step_ms": float(np.median(step)),
        "total_ms": float(np.median(total)),
        "total_p90_ms": float(np.percentile(total, 90)),
    }


def measure_batch_cost(    forward: FrozenDefocusForward,
    objective: UniformityObjective,
    basis: torch.Tensor,
    coeffs: torch.Tensor,
    *,
    iters: int = 30,
    lr: float = 0.01,
) -> dict[str, Any]:
    """批量起点：一次前向同时跑 B 个起点，测每步墙钟与「每起点」摊销。"""
    c = torch.nn.Parameter(coeffs.clone())
    opt = torch.optim.Adam([c], lr=lr)
    total = []
    for _ in range(iters):
        t0 = time.perf_counter()
        size = int(forward.params.grid_size)
        phase = torch.einsum("bk,khw->bhw", c, basis).reshape(int(coeffs.shape[0]), size, size)
        pred = forward.predict(phase)
        loss = sum(objective.cv(pred[i]) for i in range(pred.shape[0])) / pred.shape[0]
        opt.zero_grad()
        loss.backward()
        opt.step()
        total.append((time.perf_counter() - t0) * 1e3)
    b = int(coeffs.shape[0])
    med = float(np.median(total))
    return {"batch": b, "step_total_ms": med, "per_start_ms": med / b}


# --------------------------------------------------------------------------- #
# 收敛：曲线 + 达标迭代
# --------------------------------------------------------------------------- #
def optimize(
    forward: FrozenDefocusForward,
    objective: UniformityObjective,
    basis: torch.Tensor,
    coeffs0: torch.Tensor,
    *,
    lr: float,
    iters: int,
    optimizer: str = "adam",
) -> dict[str, Any]:
    c = torch.nn.Parameter(coeffs0.clone())

    def closure_loss() -> torch.Tensor:
        # LBFGS 线搜索会多次调用闭包；闭包必须**自己**算梯度（LBFGS 不替你 backward），
        # 且每次清零。两者缺一，step() 会直接返回初值、系数一动不动。
        opt.zero_grad()
        loss = objective.cv(predict_one(forward, command_of(forward, basis, c)))
        loss.backward()
        return loss

    cv2_series: list[float] = []
    elapsed: list[float] = []
    grad_norm: list[float] = []

    if optimizer == "lbfgs":
        opt = torch.optim.LBFGS(
            [c], lr=lr, max_iter=1, history_size=20, line_search_fn="strong_wolfe"
        )
        t0 = time.perf_counter()
        for _ in range(iters):
            loss_val = opt.step(closure_loss)
            cv2_series.append(float(loss_val.detach()))
            elapsed.append(time.perf_counter() - t0)
            grad_norm.append(float(c.grad.norm()) if c.grad is not None else float("nan"))
    else:
        if optimizer == "sgd":
            opt = torch.optim.SGD([c], lr=lr, momentum=0.9)
        else:
            opt = torch.optim.Adam([c], lr=lr)
        t0 = time.perf_counter()
        for _ in range(iters):
            opt.zero_grad()
            loss = objective.cv(predict_one(forward, command_of(forward, basis, c)))
            loss.backward()
            grad_norm.append(float(c.grad.norm()))
            opt.step()
            cv2_series.append(float(loss.detach()))
            elapsed.append(time.perf_counter() - t0)

    # 末尾补一次真实评估，让 cv[k] 严格表示「k 步之后」；步进循环里记的是步前值。
    with torch.no_grad():
        cv2_series.append(
            float(objective.cv(predict_one(forward, command_of(forward, basis, c))))
        )
    elapsed.append(elapsed[-1] if elapsed else 0.0)
    cv2 = np.asarray(cv2_series, dtype=np.float64)
    cv = np.sqrt(np.maximum(cv2, 0.0))
    curve = {str(k): float(cv[min(k, len(cv) - 1)]) for k in CURVE_AT}
    reach = {}
    for target in TARGETS:
        hit = next(
            ((i, s) for i, s in enumerate(zip(cv.tolist(), elapsed)) if s[0] <= target), None
        )
        reach[str(target)] = (
            {"iteration": int(hit[0]), "seconds": float(hit[1][1])}
            if hit
            else {"iteration": None, "seconds": None}
        )
    return {
        "optimizer": optimizer,
        "lr": float(lr),
        "iters": int(iters),
        "initial_cv": float(cv[0]),
        "final_cv": float(cv[-1]),
        "best_cv": float(cv.min()),
        "curve": curve,
        "reach": reach,
        "seconds_total": float(elapsed[-1]),
        "coeffs_final": [float(v) for v in c.detach().tolist()],
        "grad_norm_first": float(grad_norm[0]),
        "grad_norm_mid": float(grad_norm[len(grad_norm) // 2]),
        "grad_norm_last": float(grad_norm[-1]),
    }


def starts(n_modes: int, seed_count: int = 3, random_rad: float = 0.05):
    out = [("zero", None, torch.zeros(n_modes))]
    for seed in range(seed_count):
        rng = np.random.default_rng(seed)
        coeffs = rng.uniform(-random_rad, random_rad, n_modes)
        out.append((f"random{seed}", seed, torch.from_numpy(coeffs.astype(np.float32))))
    return out


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    finals = np.asarray([r["final_cv"] for r in rows], dtype=np.float64)
    bests = np.asarray([r["best_cv"] for r in rows], dtype=np.float64)
    secs = np.asarray([r["seconds_total"] for r in rows], dtype=np.float64)
    summary: dict[str, Any] = {
        "n": len(rows),
        "median_final_cv": float(np.median(finals)),
        "median_best_cv": float(np.median(bests)),
        "best_final_cv": float(bests.min()),
        "median_seconds": float(np.median(secs)),
    }
    for target in TARGETS:
        hits = [r["reach"][str(target)]["iteration"] for r in rows]
        hits = [h for h in hits if h is not None]
        secs_hit = [r["reach"][str(target)]["seconds"] for r in rows]
        secs_hit = [s for s in secs_hit if s is not None]
        summary[f"reach_{target}_count"] = len(hits)
        summary[f"reach_{target}_median_iteration"] = float(np.median(hits)) if hits else None
        summary[f"reach_{target}_median_seconds"] = float(np.median(secs_hit)) if secs_hit else None
    return summary


# --------------------------------------------------------------------------- #
# schedule：粗网格 → 精修
# --------------------------------------------------------------------------- #
def run_schedule(
    bench,
    *,
    lr: float,
    optimizer: str,
    coarse_fft: int,
    coarse_iters: int,
    fine_fft: int,
    fine_iters: int,
    seed_count: int,
) -> dict[str, Any]:
    """先在粗网格优化 ``coarse_iters`` 步，再切到细网格续 ``fine_iters`` 步。"""
    coarse = build_forward(bench, fft_size=coarse_fft)
    fine = build_forward(bench, fft_size=fine_fft)
    c_obj, _ = build_objective(coarse, bench)
    f_obj, _ = build_objective(fine, bench)
    modes = tuple(int(j) for j in bench.C_MODES)
    c_basis = build_basis(coarse, modes)
    f_basis = build_basis(fine, modes)

    rows = []
    for name, seed, coeffs0 in starts(len(modes), seed_count, float(bench.C_RANDOM_RAD)):
        t0 = time.perf_counter()
        first = optimize(coarse, c_obj, c_basis, coeffs0, lr=lr, iters=coarse_iters, optimizer=optimizer)
        carry = torch.tensor(first["coeffs_final"], dtype=torch.float32)
        second = optimize(fine, f_obj, f_basis, carry, lr=lr, iters=fine_iters, optimizer=optimizer)
        wall = time.perf_counter() - t0
        rows.append(
            {
                "start": name,
                "seed": seed,
                "coarse_initial_cv": first["initial_cv"],
                "coarse_final_cv": first["final_cv"],
                "fine_final_cv": second["final_cv"],
                "wall_s": wall,
                "schedule": f"fft{coarse_fft}x{coarse_iters}->fft{fine_fft}x{fine_iters}",
            }
        )
    return {
        "coarse_fft": coarse_fft,
        "fine_fft": fine_fft,
        "coarse_iters": coarse_iters,
        "fine_iters": fine_iters,
        "rows": rows,
        "summary": _summarize(
            [
                {
                    **r,
                    "final_cv": r["fine_final_cv"],
                    "best_cv": r["fine_final_cv"],
                    "seconds_total": r["wall_s"],
                    "reach": {str(t): {"iteration": None, "seconds": None} for t in TARGETS},
                }
                for r in rows
            ]
        ),
    }


def run_deployed(
    bench,
    *,
    lrs: tuple[float, ...],
    iters: int,
    cap: float = 0.1,
) -> dict[str, Any]:
    """在**模块原生** ``DifferentiableUniformityOptimizer`` 上复测学习率。

    ``optimize()`` 是裸 Adam（与基准 C 轴同口径）；原生类多了三步：
    piston/倾斜去除、增量 RMS ≤ ``cap`` 投影、``accept()`` 推进锚点。
    学习率结论必须在这条路径上复核——裸 Adam 上「把 lr 调大就好」的结论
    到这条路径上可能被 cap 削平。

    两种 ``accept_mode`` 对应两种真实语义，必须分开报：

    - ``cumulative``：从不 ``accept()``。``_project_increment()`` 把
      ``delta - accepted``（此时 = 全部 delta）缩到 RMS ≤ cap，于是**总**相位
      偏移被永久钉在 cap 上，优化器只能走 cap 球面内的最优。这是「纯模型侧、
      没有实测确认」的可行域上界。
    - ``per_step``：每步 ``accept()``，锚点跟着走，cap 只限制**单步增量**。
      这是「实测每步都变好」的乐观上界；硬件上 ``accept()`` 由
      ``measured_cv < best_cv`` 门控（``shape_transition_timing_20261106.py:563``），
      因此真实闭环落在两者之间。
    """
    forward = build_forward(bench)
    objective, _core = build_objective(forward, bench)
    size = int(forward.params.grid_size)
    seed = np.zeros((size, size), dtype=np.float32)
    out: dict[str, Any] = {"cap": float(cap), "iters": int(iters), "start": "zero", "arms": {}}
    for accept_mode in ("cumulative", "per_step"):
        rows: dict[str, Any] = {}
        for lr in lrs:
            opt = DifferentiableUniformityOptimizer(
                forward, objective, seed, lr=lr, step_rms_cap=cap, device="cpu"
            )
            t0 = time.perf_counter()
            cvs: list[float] = []
            rms: list[float] = []
            rms_before: list[float] = []
            worse: list[bool] = []
            total_rms: list[float] = []
            for _ in range(iters):
                opt.update()
                rec = opt.loss_history[-1]
                l_cv = float(rec.get("l_cv", rec["loss"]))
                cvs.append(float(np.sqrt(max(l_cv, 0.0))))
                rms.append(float(rec.get("increment_rms", float("nan"))))
                rms_before.append(float(rec.get("increment_rms_before_projection", float("nan"))))
                # candidate - anchor：正值 = 候选更差（``_record`` 的原始符号）。
                worse.append(float(rec.get("candidate_model_loss_change", float("nan"))) > 0.0)
                if accept_mode == "per_step":
                    opt.accept()
                total_rms.append(float(opt.weighted_rms(opt.delta_phase)))
            wall = time.perf_counter() - t0
            cv = np.asarray(cvs, dtype=np.float64)
            rms_a = np.asarray(rms, dtype=np.float64)
            before = np.asarray(rms_before, dtype=np.float64)
            total = np.asarray(total_rms, dtype=np.float64)
            curve = {str(k): float(cv[min(k, len(cv) - 1)]) for k in CURVE_AT}
            reach = {}
            for target in TARGETS:
                hit = next((i for i, v in enumerate(cv.tolist()) if v <= target), None)
                reach[str(target)] = {"iteration": hit}
            rows[f"lr{lr:g}"] = {
                "lr": float(lr),
                "accept_mode": accept_mode,
                "cap": float(cap),
                "iters": int(iters),
                "initial_cv": float(cv[0]),
                "final_cv": float(cv[-1]),
                "best_cv": float(cv.min()),
                "curve": curve,
                "reach": reach,
                "wall_s": float(wall),
                "ms_per_step": float(wall * 1e3 / max(iters, 1)),
                "increment_rms_median": float(np.nanmedian(rms_a)),
                "increment_rms_before_projection_median": float(np.nanmedian(before)),
                "cap_binding_fraction": float(np.mean(rms_a >= 0.999 * cap)),
                "candidate_worse_fraction": float(np.mean(np.asarray(worse, dtype=bool))),
                "held_phase_rms_end": float(total[-1]),
            }
            logger.info(
                "原生优化器 {mode} lr={lr:g}（cap {cap:g}，{it} 步）：CV {i:.3f}→{f:.3f}，"
                "增量 RMS 中位 {r:.3f}（投影前 {rb:.3f}），撞 cap {cb:.0%}，"
                "未锁定相位 RMS {h:.3f}，{ms:.1f} ms/step",
                mode=accept_mode, lr=lr, cap=cap, it=iters, i=cv[0], f=cv[-1],
                r=float(np.nanmedian(rms_a)), rb=float(np.nanmedian(before)),
                cb=float(np.mean(rms_a >= 0.999 * cap)), h=float(total[-1]),
                ms=rows[f"lr{lr:g}"]["ms_per_step"],
            )
        out["arms"][accept_mode] = {
            "rows": rows,
            "ranking": sorted(
                ({"key": k, "final_cv": v["final_cv"]} for k, v in rows.items()),
                key=lambda r: r["final_cv"],
            ),
        }
    return out


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--threads", type=int, default=min(6, __import__("os").cpu_count() or 1))
    parser.add_argument("--quick", action="store_true", help="少起点/少学习率，冒烟用")
    args = parser.parse_args(argv)

    torch.set_num_threads(max(1, args.threads))
    torch.manual_seed(0)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.info(DEVICE_STATEMENT)

    bench = _load_bench()
    modes = tuple(int(j) for j in bench.C_MODES)
    base_lr = float(bench.C_LR)
    lrs = (base_lr,) if args.quick else (base_lr, 0.03, 0.1, 0.3)
    seed_count = 1 if args.quick else 3
    iters = 120 if args.quick else 300
    floor_iters = 400 if args.quick else 1000

    t_all = time.perf_counter()
    results: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device_statement": DEVICE_STATEMENT,
        "command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
        "config": {
            "threads": args.threads,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "modes_noll": list(modes),
            "base_lr": base_lr,
            "iters": iters,
            "floor_iters": floor_iters,
            "random_rad": float(bench.C_RANDOM_RAD),
            "quick": bool(args.quick),
        },
    }

    # ---------- 1. 基线：复现 C 轴 ----------
    forward = build_forward(bench)
    objective, core = build_objective(forward, bench)
    basis = build_basis(forward, modes)
    size = int(forward.params.grid_size)
    results["baseline"] = {
        "fft_size": int(forward.params.fft_size),
        "grid_size": size,
        "pixel_samples": int(forward.params.pixel_samples),
        "core_pixels": int(core.sum()),
        "image_shape": list(core.shape),
        "zero_cv": float(np.sqrt(objective.cv(predict_one(forward, torch.zeros(size, size))).item())),
    }
    base_cost = measure_cost(forward, objective, basis, torch.zeros(len(modes)))
    results["baseline"]["cost"] = base_cost
    logger.info(
        "基线（预热前首测）：{ms:.1f} ms/iter（前向 {f:.1f} / 反向 {b:.1f} / step {s:.1f}）",
        ms=base_cost["total_ms"], f=base_cost["forward_ms"],
        b=base_cost["backward_ms"], s=base_cost["step_ms"],
    )

    # 计时稳定性：第一段计时会撞到 boost 时钟，必须先把噪声地板量出来，
    # 否则「5.1 ms/iter」和「13.1 ms/iter」都是同一台机器。
    warm_up(forward, objective, basis, iters=200 if not args.quick else 60)
    repeats = []
    for k in range(5):
        row = measure_cost(forward, objective, basis, torch.zeros(len(modes)), iters=50)
        repeats.append(row)
        logger.info("计时重复 {k}: {ms:.1f} ms/iter", k=k, ms=row["total_ms"])
    totals = np.asarray([r["total_ms"] for r in repeats], dtype=np.float64)
    results["timing_stability"] = {
        "note": "预热 200 步后连续 5 次测 50 步的每步墙钟，作为所有计时结论的噪声地板",
        "repeats": repeats,
        "median_ms": float(np.median(totals)),
        "min_ms": float(totals.min()),
        "max_ms": float(totals.max()),
        "spread_ratio": float(totals.max() / max(totals.min(), 1e-9)),
    }
    base_cost = repeats[len(repeats) // 2]
    results["baseline"]["cost_sustained"] = base_cost
    logger.info(
        "基线（预热后稳态，{n} 次中位）：{ms:.1f} ms/iter（前向 {f:.1f} / 反向 {b:.1f} / step {s:.1f}），"
        "5 次极差 {lo:.1f}–{hi:.1f}",
        n=len(repeats), ms=base_cost["total_ms"], f=base_cost["forward_ms"],
        b=base_cost["backward_ms"], s=base_cost["step_ms"],
        lo=results["timing_stability"]["min_ms"], hi=results["timing_stability"]["max_ms"],
    )
    base_img = predict_one(forward, torch.zeros(size, size)).detach().numpy().astype(np.float32)

    # ---------- 2. cost 变体：fft_size ----------
    variants: dict[str, Any] = {}
    for name, fft_size in (("fft256", 256),):
        try:
            variant = build_forward(bench, fft_size=fft_size)
            v_size = int(variant.params.grid_size)
            v_img = predict_one(variant, torch.zeros(v_size, v_size)).detach().numpy().astype(np.float32)
            denom = float(np.linalg.norm(base_img)) or 1.0
            rel = float(np.linalg.norm(v_img - base_img) / denom)
            with torch.no_grad():
                v_core = v_img >= 0.1 * float(v_img.max())
            inter = float(np.logical_and(v_core, core).sum())
            union = float(np.logical_or(v_core, core).sum()) or 1.0
            v_obj, _ = build_objective(variant, bench)
            v_basis = build_basis(variant, modes)
            cost = measure_cost(variant, v_obj, v_basis, torch.zeros(len(modes)))
            variants[name] = {
                "fft_size": int(variant.params.fft_size),
                "cost": cost,
                "rel_l2_vs_baseline_image": rel,
                "core_mask_iou_vs_baseline": inter / union,
                "zero_cv": float(np.sqrt(v_obj.cv(predict_one(variant, torch.zeros(v_size, v_size))).item())),
            }
            logger.info(
                "{name}: {ms:.1f} ms/iter，图 rel-L2 {rel:.4f}，核心掩膜 IoU {iou:.3f}",
                name=name, ms=cost["total_ms"], rel=rel, iou=inter / union,
            )
        except Exception as exc:  # pragma: no cover - 变体不可用时如实记录
            variants[name] = {"error": f"{type(exc).__name__}: {exc}"}
            logger.warning("{name} 不可用：{err}", name=name, err=exc)
    variants["grid64"] = {
        "error": "checkpoint 内幅度残差权重按 grid_size=128 存储，缩小网格触发 shape 不匹配；"
        "降网格需重训 checkpoint，不作为本轮候选"
    }
    results["cost_variants"] = variants

    # ---------- 2.5 原生优化器：学习率在 cap + 伪接受下的真实表现 ----------
    results["deployed"] = run_deployed(
        bench,
        lrs=(base_lr,) if args.quick else (base_lr, 0.1, 0.3),
        iters=100 if args.quick else 300,
    )

    # ---------- 3. 线程数 ----------
    thread_rows = {}
    for n in (1, 2, 4, 6, 12):
        torch.set_num_threads(n)
        thread_rows[str(n)] = measure_cost(
            forward, objective, basis, torch.zeros(len(modes)), iters=15
        )
    torch.set_num_threads(max(1, args.threads))
    results["threads"] = thread_rows
    logger.info("线程：{k}", k={k: round(v["total_ms"], 1) for k, v in thread_rows.items()})

    # ---------- 4. 批量起点 ----------
    batch_n = 2 if args.quick else 8
    rng = np.random.default_rng(0)
    coeffs_batch = torch.from_numpy(
        rng.uniform(
            -float(bench.C_RANDOM_RAD), float(bench.C_RANDOM_RAD), (batch_n, len(modes))
        ).astype(np.float32)
    )
    results["batch"] = measure_batch_cost(forward, objective, basis, coeffs_batch)
    logger.info(
        "批量 {b} 起点：{t:.1f} ms/step，每起点 {p:.1f} ms",
        b=batch_n, t=results["batch"]["step_total_ms"], p=results["batch"]["per_start_ms"],
    )

    # ---------- 5. 收敛：优化器 × 学习率 × 起点 ----------
    conv: dict[str, Any] = {}
    for opt_name in ("adam", "sgd"):
        for lr in lrs:
            rows = []
            for name, seed, coeffs0 in starts(len(modes), seed_count, float(bench.C_RANDOM_RAD)):
                row = optimize(forward, objective, basis, coeffs0, lr=lr, iters=iters, optimizer=opt_name)
                row.update({"start": name, "seed": seed})
                rows.append(row)
                logger.info(
                    "{o} lr={lr} {start}: CV {i:.3f}→{f:.3f}（{s:.1f}s）",
                    o=opt_name, lr=lr, start=name, i=row["initial_cv"],
                    f=row["final_cv"], s=row["seconds_total"],
                )
            key = f"{opt_name}_lr{lr:g}"
            conv[key] = _summarize(rows)
            conv[key]["runs"] = rows
    results["convergence"] = conv

    ranked = [
        (k, v["median_final_cv"]) for k, v in conv.items() if v.get("median_final_cv") is not None
    ]
    ranked.sort(key=lambda kv: kv[1])
    results["best"] = {
        "key": ranked[0][0],
        "median_final_cv": ranked[0][1],
        "ranking": [{"key": k, "median_final_cv": v} for k, v in ranked],
    }
    best_key = ranked[0][0]
    best_opt = "sgd" if best_key.startswith("sgd") else "adam"
    best_lr = float(best_key.split("_lr")[1])
    results["best_adam"] = {"key": best_key, "optimizer": best_opt, "lr": best_lr}

    # ---------- 6. LBFGS ----------
    lbfgs_rows = {}
    for lr in ((1.0,) if args.quick else (1.0, 0.1, 0.01)):
        rows = []
        for start, seed, coeffs0 in starts(len(modes), 1, float(bench.C_RANDOM_RAD)):
            row = optimize(
                forward, objective, basis, coeffs0, lr=lr,
                iters=20 if args.quick else 40, optimizer="lbfgs",
            )
            row.update({"start": start, "seed": seed})
            rows.append(row)
        lbfgs_rows[f"lr{lr:g}"] = {"lr": lr, "runs": rows, "summary": _summarize(rows)}
        logger.info(
            "LBFGS lr={lr}: {k}", lr=lr,
            k={r["start"]: (round(r["initial_cv"], 3), round(r["final_cv"], 3)) for r in rows},
        )
    results["lbfgs"] = lbfgs_rows

    # ---------- 7. floor：模式集 × 最好优化器 ----------
    mode_sets = {"5-15": tuple(range(5, 16)), "5-25": tuple(range(5, 26)), "5-35": tuple(range(5, 36))}
    floor_rows: dict[str, Any] = {}
    for label, noll in mode_sets.items():
        m_basis = build_basis(forward, noll)
        rows = []
        for start, seed, coeffs0 in starts(len(noll), 1, float(bench.C_RANDOM_RAD)):
            row = optimize(
                forward, objective, m_basis, coeffs0, lr=best_lr, iters=floor_iters, optimizer=best_opt
            )
            row.update({"start": start, "seed": seed})
            rows.append(row)
        floor_rows[label] = {"modes": list(noll), "summary": _summarize(rows), "runs": rows}
        logger.info(
            "floor 模式 {l}（{n} 个，{o} lr={lr:g}，{it} iters）: 最好 {b:.3f} / 结束 {f:.3f}",
            l=label, n=len(noll), o=best_opt, lr=best_lr, it=floor_iters,
            b=rows[0]["best_cv"], f=rows[0]["final_cv"],
        )
    results["floor"] = {
        "optimizer": best_opt,
        "lr": best_lr,
        "iters": floor_iters,
        "mode_sets": floor_rows,
    }

    # ---------- 8. schedule：粗网格 → 精修 ----------
    fine_iters = 100 if args.quick else 200
    coarse_iters = fine_iters // 2
    results["schedule"] = {
        "arm": run_schedule(
            bench, lr=best_lr, optimizer=best_opt,
            coarse_fft=256, coarse_iters=coarse_iters,
            fine_fft=512, fine_iters=fine_iters,
            seed_count=1 if args.quick else 2,
        ),
        "fine_iters": fine_iters,
        "coarse_iters": coarse_iters,
    }
    logger.info(
        "schedule: {k}",
        k=[
            (r["start"], round(r["fine_final_cv"], 3), round(r["wall_s"], 1))
            for r in results["schedule"]["arm"]["rows"]
        ],
    )
    # 对照：全程 512 同预算
    ctrl_rows = []
    for start, seed, coeffs0 in starts(len(modes), 1 if args.quick else 2, float(bench.C_RANDOM_RAD)):
        t0 = time.perf_counter()
        row = optimize(
            forward, objective, basis, coeffs0, lr=best_lr,
            iters=coarse_iters + fine_iters, optimizer=best_opt,
        )
        row.update({"start": start, "seed": seed, "wall_s": time.perf_counter() - t0})
        ctrl_rows.append(row)
    results["schedule"]["control_fft512"] = {
        "iters": coarse_iters + fine_iters,
        "rows": ctrl_rows,
        "summary": _summarize(ctrl_rows),
    }
    logger.info(
        "control fft512 {it} iters: {k}",
        it=coarse_iters + fine_iters,
        k=[(r["start"], round(r["final_cv"], 3), round(r["wall_s"], 1)) for r in ctrl_rows],
    )

    results["elapsed_s"] = time.perf_counter() - t_all
    (out / "results.json").write_text(
        json.dumps(_jsonable(results), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("写出 {p}（{s:.1f}s）", p=out / "results.json", s=results["elapsed_s"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
