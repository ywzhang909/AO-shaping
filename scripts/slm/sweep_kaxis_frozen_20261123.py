"""纯离线筛选: 冻结问题 phys_20261116_01 上可微整形加速方案的速度/质量公平比较.

要回答什么
----------
在**冻结的离线问题** ``data/debug/differentiable_opt/phys_20261116_01`` 上, 公平比较
可微整形加速方案 (候选数 K, 以及 L-BFGS / 余弦调度两个参照变体) 的速度与整形质量。
**不碰任何设备**: :func:`scripts._common.differentiable_problem.load_problem_from_run_dir`
是纯模型侧重建 (``run_config.json`` 里 ``execute=true`` 只是历史运行记录), 不会执行采集。

固定不变 (公平比较的公共底盘)
------------------------------
物理预测模型 ``fft`` (夫琅禾费, 与基线入口一致)、网格 ``problem.target.shape`` (1024²)、
square 目标、GS 初相位 (``phase_initial.npy``, 50 迭代)、Adam 基线 ``lr=0.05``、
``iterations=120``、种子约定见变体表。**只变化**: ``n_candidates`` ∈ {1,4,8} (K 轴),
或参照变体的优化器/调度。

口径 (与 ``sweep_differentiable_lr.py::run_arm`` 完全一致, 逐行对齐)
-------------------------------------------------------------------
``mask = problem.target > 0.0``; ``far = far_field_intensity(amplitude, phase)``;
``energy_in_target = far[mask].sum()/far.sum()``;
``cv_in_target = far[mask].std()/far[mask].mean()``;
``strehl_like = far.max()/flat_peak`` (flat_peak 为平相位远场峰值);
``pred_fwhm_px = measure_spot(far).fwhm_px``。

变体 (按固定协议冻结)
--------------------
* ``gate_repro``    基线复现门: ``build_optimizer(lr=0.05, seed=0)`` 跑 120 次
  ``update()``, 终态物理质量与 ``lr_sweep_20261116/lr_sweep.json`` 基线比较,
  最大相对偏差 > ``--repro-tol`` (默认 0.001) 即 exit 2 —— 在错误问题上做筛选没有意义。
* K 轴 (``train_beam_shaping``, adam, lr=0.05, 无调度): K=1 确定性 (克隆显式初相位),
  跑 seed=0 一次; K∈{4,8} 每个候选叠加 ``0.1*randn``, 跑 ``--seeds`` (默认 0,1,2) 三次。
  内部基线 = K=1。
* 参照变体 (与 K 轴分开报告): ``lbfgs`` (K 自动回退 1, lr=1.0, seed=0),
  ``cosine`` (adam, lr=0.1, ``lr_schedule="cosine"``, seed=0)。

注: gate 变体与 K 轴变体用的是**不同损失** (基线为峰值归一化 MSE, K 轴为
``w_uniformity=0.4 + w_efficiency=0.6`` 总损失), 因此两者的终态**损失**不可跨口径比较;
本脚本对**所有变体**用同一物理质量口径从终态相位计算 energy/cv/fwhm/strehl, 故物理质量可比较,
每迭代墙钟成本也与目标无关, 速度可比较。

产物
----
``--out-debug`` (默认 ``data/debug/differentiable_opt/kaxis_screening_20261123/``):
``screening.json`` (全部逐迭代损失 + 采样物理质量 + 汇总) 与 ``screening.csv`` (每采样一行)。
``--out-artifact`` (默认 ``report/slm/differentiable_opt/artifacts/kaxis_screening_20261123/``):
``kaxis_screening.json`` (头条数字 + 逐变体汇总, 与 debug JSON 分开的轻量摘要) 与
可选 ``kaxis_cv_vs_iter_20261123.png`` (``--plot``)。

用法示例
--------
::

    python scripts/slm/sweep_kaxis_frozen_20261123.py --plot
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
for _entry in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from scripts._common.differentiable_problem import (
    DifferentiableProblem,
    build_optimizer,
    load_problem_from_run_dir,
)

# ---------------------------------------------------------------------------
# 冻结协议的默认值
# ---------------------------------------------------------------------------
DEFAULT_RUN_DIR = ROOT / "data/debug/differentiable_opt/phys_20261116_01"
DEFAULT_BASELINE_JSON = (
    ROOT / "report/slm/differentiable_opt/artifacts/lr_sweep_20261116/lr_sweep.json"
)
DEFAULT_DEBUG_DIR = ROOT / "data/debug/differentiable_opt/kaxis_screening_20261123"
DEFAULT_ARTIFACT_DIR = (
    ROOT / "report/slm/differentiable_opt/artifacts/kaxis_screening_20261123"
)
REPRO_TOL_DEFAULT = 0.001
BASELINE_ITERS = 120
BASELINE_LR = 0.05
METRIC_EVERY = 5


@dataclass
class ArmSpec:
    """一个筛选变体的配置 (与实现参数分开: 算法变化由 ``kind`` / ``k`` 表达)."""

    label: str
    kind: str  # "gate" | "kaxis" | "ref"
    k: int = 1
    seed: int | None = None
    lr: float = BASELINE_LR
    schedule: str | None = None
    optimizer: str = "adam"
    iters: int = BASELINE_ITERS
    samples: list[dict[str, Any]] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    wall_s: float = float("nan")
    converged: bool = False
    final_loss: float = float("nan")
    final_quality: dict[str, float] = field(default_factory=dict)


def _quality(
    deps: dict[str, Any],
    problem: DifferentiableProblem,
    mask: np.ndarray,
    flat_peak: float,
    phase: Any,
) -> dict[str, float]:
    """从终态/采样相位算物理质量 (口径 = sweep_differentiable_lr.run_arm)."""
    far = deps["far_intensity"](problem.amplitude, phase).detach().cpu().numpy()
    spot = deps["measure_spot"](far)
    in_t = far[mask]
    total = float(far.sum())
    mean_in = float(in_t.mean()) if in_t.size else 0.0
    return {
        "pred_peak": float(far.max()),
        "pred_fwhm_px": float(spot.fwhm_px),
        "pred_hollowness": float(spot.hollowness),
        "energy_in_target": float(in_t.sum() / total) if total > 0 else float("nan"),
        "cv_in_target": float(in_t.std() / mean_in) if mean_in > 0 else float("nan"),
        "strehl_like": float(far.max() / flat_peak) if flat_peak > 0 else float("nan"),
    }


def _flat_peak(deps: dict[str, Any], problem: DifferentiableProblem, device: str) -> float:
    """平相位 (全零) 预测远场峰值 —— 类 Strehl 的分母, 每个变体只算一次."""
    torch = deps["torch"]
    dev = torch.device(device)
    zeros = torch.zeros(problem.target.shape, dtype=torch.float32, device=dev)
    with torch.no_grad():
        far = deps["far_intensity"](problem.amplitude, zeros)
    return float(far.max().detach().cpu())


def run_gate_arm(
    deps: dict[str, Any],
    problem: DifferentiableProblem,
    device: str,
    spec: ArmSpec,
) -> ArmSpec:
    """基线复现门: 用 build_optimizer 复现 lr=0.05/120 基线 (与 run_arm 同口径)."""
    torch = deps["torch"]
    opt = build_optimizer(problem, lr=spec.lr, device=device, seed=spec.seed)
    mask = problem.target > 0.0
    flat_peak = _flat_peak(deps, problem, device)

    t0 = time.perf_counter()
    for i in range(spec.iters):
        opt.update()
        loss_i = float(opt.last_loss) if opt.last_loss is not None else float("nan")
        spec.losses.append(loss_i)
        it1 = i + 1
        if it1 == 1 or it1 == spec.iters or it1 % METRIC_EVERY == 0:
            with torch.no_grad():
                q = _quality(deps, problem, mask, flat_peak, opt.phase_tensor)
            q["iter"] = it1
            q["wall_s"] = time.perf_counter() - t0
            spec.samples.append(q)
    spec.wall_s = time.perf_counter() - t0
    spec.converged = len(spec.losses) >= spec.iters
    spec.final_loss = spec.losses[-1] if spec.losses else float("nan")
    spec.final_quality = dict(spec.samples[-1]) if spec.samples else {}
    return spec


def run_tbs_arm(
    deps: dict[str, Any],
    problem: DifferentiableProblem,
    device: str,
    spec: ArmSpec,
) -> ArmSpec:
    """train_beam_shaping 变体: K 轴 (adam) 与参照变体 (lbfgs / cosine) 共用."""
    torch = deps["torch"]
    mask = problem.target > 0.0
    flat_peak = _flat_peak(deps, problem, device)

    state: dict[str, Any] = {"count": 0, "samples": [], "t0": time.perf_counter()}

    def _cb(phase: Any) -> None:
        state["count"] += 1
        c = state["count"]
        if c == 1 or c == spec.iters or c % METRIC_EVERY == 0:
            with torch.no_grad():
                q = _quality(deps, problem, mask, flat_peak, phase)
            q["iter"] = c
            q["wall_s"] = time.perf_counter() - state["t0"]
            state["samples"].append(q)

    from ao_shaping.algorithm.signal_processing.differentiable_shaping import (
        train_beam_shaping,
    )

    result = train_beam_shaping(
        problem.target,
        problem.target.shape,
        source_amplitude=problem.amplitude,
        initial_phase=problem.init_phase,
        propagation="fft",
        optimizer=spec.optimizer,
        iterations=spec.iters,
        lr=spec.lr,
        lr_schedule=spec.schedule,
        device=device,
        seed=spec.seed,
        n_candidates=spec.k,
        phase_callback=_cb,
    )
    spec.wall_s = time.perf_counter() - state["t0"]
    spec.converged = bool(result.converged)
    spec.losses = list(result.loss_history)
    spec.final_loss = spec.losses[-1] if spec.losses else float("nan")
    spec.samples = state["samples"]
    phase_t = torch.from_numpy(np.asarray(result.phase, dtype=np.float32)).to(
        torch.device(device)
    )
    with torch.no_grad():
        spec.final_quality = _quality(deps, problem, mask, flat_peak, phase_t)
    return spec


def _max_rel_dev(got: dict[str, float], ref: dict[str, float]) -> tuple[float, dict[str, float]]:
    """物理质量键的相对偏差 (|got-ref|/|ref|), 返回 (最大值, 每键偏差)."""
    devs: dict[str, float] = {}
    for key in ("energy_in_target", "cv_in_target", "pred_fwhm_px", "strehl_like"):
        g = got.get(key)
        r = ref.get(key)
        if g is None or r is None or r == 0:
            continue
        devs[key] = abs(float(g) - float(r)) / abs(float(r))
    return (max(devs.values()) if devs else float("nan"), devs)


def _load_baseline(path: Path) -> dict[str, Any]:
    """从 lr_sweep_20261116/lr_sweep.json 提取基线摘要 (final_quality + 复现参考).

    结构: 顶层 ``baseline`` 字典 (final_loss_csv / final_quality / repro),
    复现变体在 ``arms`` 里 ``kind="repro"`` (headline.iters_to_baseline_final_loss)。
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    b = data.get("baseline")
    if b and b.get("final_quality"):
        headline = {}
        for item in data.get("arms", []):
            if isinstance(item, dict) and item.get("kind") == "repro":
                headline = item.get("headline") or {}
                b = dict(b)
                b.setdefault("wall_s", item.get("wall_s"))
                if headline.get("iters_to_baseline_final_loss") is not None:
                    b["iters_to_baseline_final_loss"] = headline["iters_to_baseline_final_loss"]
                break
        return b
    raise ValueError(f"baseline 摘要未找到: {path}")


def _write_debug(debug_dir: Path, payload: dict[str, Any], csv_rows: list[list[str]]) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / "screening.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (debug_dir / "screening.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerows(csv_rows)


def _maybe_plot(
    artifact_dir: Path,
    arms: list[ArmSpec],
    iters: int,
    k1_final_cv: float,
) -> None:
    """cv_in_target 随迭代的曲线 (对数 x 轴), 供报告摘录."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from scripts._common.plot_style import setup_cjk_font
    except Exception as exc:  # pragma: no cover - 绘图是可选增强
        logger.warning("绘图不可用, 跳过 PNG: {}", exc)
        return
    setup_cjk_font()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    palette = {"k1_seed0": "#1f77b4", "k4_seed0": "#d62728", "k8_seed0": "#2ca02c"}
    for arm in arms:
        if not arm.samples:
            continue
        xs = [s["iter"] for s in arm.samples]
        ys = [s["cv_in_target"] for s in arm.samples]
        color = palette.get(arm.label, "#7f7f7f")
        ax.plot(xs, ys, marker="o", ms=3, lw=1.2, color=color, label=arm.label)
    ax.axhline(k1_final_cv, color="#444444", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("迭代数")
    ax.set_ylabel("目标区内强度 CV")
    ax.set_title("冻结问题 phys_20261116_01: 候选数 K 轴质量曲线")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = artifact_dir / "kaxis_cv_vs_iter_20261123.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("图已写入 {}", out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--baseline-json", default=str(DEFAULT_BASELINE_JSON))
    parser.add_argument("--out-debug", default=str(DEFAULT_DEBUG_DIR))
    parser.add_argument("--out-artifact", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--ks", default="1,4,8", help="候选数列表")
    parser.add_argument("--seeds", default="0,1,2", help="K>1 的种子列表")
    parser.add_argument("--iters", type=int, default=BASELINE_ITERS)
    parser.add_argument("--lr", type=float, default=BASELINE_LR)
    parser.add_argument("--device", default=None, help="torch 设备; 默认 cuda→cpu")
    parser.add_argument("--repro-tol", type=float, default=REPRO_TOL_DEFAULT)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)

    # -- 延迟重依赖 ---------------------------------------------------
    try:
        import torch

        from ao_shaping.algorithm import far_field_intensity
        from ao_shaping.utils.image.frame_analysis import measure_spot
    except Exception as exc:  # pragma: no cover - 环境缺依赖
        logger.error("重依赖导入失败 (需要 torch + ml extra): {}", exc)
        return 3

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    deps = {
        "torch": torch,
        "far_intensity": far_field_intensity,
        "measure_spot": measure_spot,
    }

    run_dir = Path(args.run_dir)
    problem = load_problem_from_run_dir(run_dir)
    grid = list(problem.target.shape)
    logger.info("问题重建: run={} grid={} lr={} iters={} seed={}", run_dir.name, grid,
                problem.params.get("lr"), problem.params.get("iters"), problem.params.get("seed"))

    baseline = _load_baseline(Path(args.baseline_json))
    baseline_q = baseline.get("final_quality") or baseline
    baseline_metrics = {
        "final_loss": baseline.get("final_loss_csv") or baseline.get("final_loss"),
        "energy_in_target": baseline_q.get("energy_in_target"),
        "cv_in_target": baseline_q.get("cv_in_target"),
        "pred_fwhm_px": baseline_q.get("pred_fwhm_px"),
        "strehl_like": baseline_q.get("strehl_like"),
        "wall_s": baseline.get("wall_s"),
        "iters_to_baseline_final_loss": baseline.get("iters_to_baseline_final_loss"),
    }

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()] or [0]

    # -- 组装变体表 (固定协议) ------------------------------------------
    arms: list[ArmSpec] = []
    arms.append(ArmSpec(label="gate_repro", kind="gate", seed=0, lr=args.lr,
                        iters=args.iters))
    for k in ks:
        run_seeds = [0] if k == 1 else seeds
        for s in run_seeds:
            arms.append(ArmSpec(label=f"k{k}_seed{s}", kind="kaxis", k=k, seed=s,
                                lr=args.lr, iters=args.iters))
    arms.append(ArmSpec(label="ref_lbfgs", kind="ref", k=1, seed=0, lr=1.0,
                        optimizer="lbfgs", iters=args.iters))
    arms.append(ArmSpec(label="ref_cosine", kind="ref", k=1, seed=0, lr=0.1,
                        schedule="cosine", optimizer="adam", iters=args.iters))

    for spec in arms:
        t0 = time.perf_counter()
        if spec.kind == "gate":
            run_gate_arm(deps, problem, device, spec)
        else:
            run_tbs_arm(deps, problem, device, spec)
        spec.wall_s = time.perf_counter() - t0 if spec.kind == "gate" else spec.wall_s
        per_iter = spec.wall_s / max(1, spec.iters)
        logger.info(
            "变体 {:<18} k={} seed={} {} wall={:.2f}s per_iter={:.3f}s final_loss={:.6e} "
            "cv={:.4f} energy={:.4f} fwhm={:.1f}px",
            spec.label, spec.k, spec.seed, spec.optimizer, spec.wall_s, per_iter,
            spec.final_loss, spec.final_quality.get("cv_in_target", float("nan")),
            spec.final_quality.get("energy_in_target", float("nan")),
            spec.final_quality.get("pred_fwhm_px", float("nan")),
        )

    # -- 门: gate 变体 vs lr_sweep 基线 ---------------------------------
    gate = next(a for a in arms if a.kind == "gate")
    max_dev, devs = _max_rel_dev(gate.final_quality, baseline_q)
    gate_pass = max_dev <= args.repro_tol
    logger.info("gate 复现: max_rel_dev={:.6f} (tol={}) -> {}", max_dev,
                args.repro_tol, "PASS" if gate_pass else "FAIL")

    # -- K=1 终态 CV 作为 K 轴内部基线阈值 -----------------------------
    k1 = next((a for a in arms if a.label == "k1_seed0"), None)
    k1_final_cv = k1.final_quality.get("cv_in_target", float("nan")) if k1 else float("nan")

    # -- 汇总行 --------------------------------------------------------
    summary_rows = []
    for spec in arms:
        iters_to_k1 = None
        if math.isfinite(k1_final_cv) and spec.kind in ("kaxis", "ref"):
            for s in spec.samples:
                if s.get("cv_in_target", float("nan")) <= k1_final_cv * 1.001:
                    iters_to_k1 = s["iter"]
                    break
        summary_rows.append({
            "arm": spec.label,
            "kind": spec.kind,
            "k": spec.k,
            "seed": spec.seed,
            "optimizer": spec.optimizer,
            "lr": spec.lr,
            "schedule": spec.schedule,
            "iters": spec.iters,
            "converged": spec.converged,
            "wall_s": round(spec.wall_s, 4),
            "per_iter_s": round(spec.wall_s / max(1, spec.iters), 5),
            "final_loss": spec.final_loss,
            "final_cv": spec.final_quality.get("cv_in_target", float("nan")),
            "final_energy": spec.final_quality.get("energy_in_target", float("nan")),
            "final_fwhm_px": spec.final_quality.get("pred_fwhm_px", float("nan")),
            "final_strehl": spec.final_quality.get("strehl_like", float("nan")),
            "iters_to_k1_cv": iters_to_k1,
        })

    payload = {
        "title": "冻结问题 phys_20261116_01 可微整形加速方案速度/质量公平筛选",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "problem": {"run_dir": str(run_dir), "grid": grid,
                    "params": {k: problem.params.get(k) for k in (
                        "lr", "iters", "seed", "target", "target_side_px",
                        "model_pupil_samples", "model_padding", "init_phase")}},
        "baseline": baseline_metrics,
        "gate": {
            "label": gate.label,
            "final": gate.final_quality,
            "final_loss": gate.final_loss,
            "wall_s": gate.wall_s,
            "max_rel_dev": max_dev,
            "per_key_rel_dev": devs,
            "repro_tol": args.repro_tol,
            "pass": gate_pass,
        },
        "k1_final_cv": k1_final_cv,
        "arms": [a.__dict__ | {"final_quality": a.final_quality} for a in arms],
        "summary": summary_rows,
    }

    debug_dir = Path(args.out_debug)
    csv_rows = [["arm", "kind", "k", "seed", "iter", "wall_s", "loss", "energy",
                 "cv", "fwhm_px", "strehl"]]
    for spec in arms:
        sample_map = {s["iter"]: s for s in spec.samples}
        for i, loss in enumerate(spec.losses, start=1):
            s = sample_map.get(i, {})
            csv_rows.append([spec.label, spec.kind, str(spec.k), str(spec.seed),
                             str(i), f"{s.get('wall_s', spec.wall_s):.4f}",
                             f"{loss:.6e}", f"{s.get('energy_in_target', float('nan')):.6f}",
                             f"{s.get('cv_in_target', float('nan')):.6f}",
                             f"{s.get('pred_fwhm_px', float('nan')):.4f}",
                             f"{s.get('strehl_like', float('nan')):.6f}"])
    _write_debug(debug_dir, payload, csv_rows)

    artifact_dir = Path(args.out_artifact)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "kaxis_screening.json").write_text(
        json.dumps({
            "title": payload["title"],
            "timestamp_utc": payload["timestamp_utc"],
            "device": device,
            "gate": payload["gate"],
            "k1_final_cv": k1_final_cv,
            "summary": summary_rows,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.plot:
        _maybe_plot(artifact_dir, arms, args.iters, k1_final_cv)

    logger.info("debug 产物: {}", debug_dir / "screening.json")
    logger.info("artifact 产物: {}", artifact_dir / "kaxis_screening.json")
    return 0 if gate_pass else 2


if __name__ == "__main__":
    sys.exit(main())
