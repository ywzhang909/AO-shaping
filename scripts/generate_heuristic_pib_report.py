"""Generate heuristic PIB benchmark report.

Benchmarks all 7 heuristic optimizers in ``ao_shaping.algorithm`` on the PIB
(power-in-bucket) optimization problem, offline (pure numpy, no hardware),
using the synthetic landscape from
``ao_shaping.optimizer.wfless.pib_sim_eval.SimLandscape``. Emits PIB iteration
curves + before/after spot images + summary into ``docs/heuristic_pib/``.

Usage:
    $env:PYTHONPATH = "src;libs"
    python scripts/generate_heuristic_pib_report.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# sys.path bootstrap (repo rule): allow running without PYTHONPATH set
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "libs"))

# ---------------------------------------------------------------------------
# Matplotlib must be configured to Agg BEFORE importing pyplot (repo-wide rule)
# ---------------------------------------------------------------------------
import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from loguru import logger  # noqa: E402

from ao_shaping.algorithm.heuristic.heuristic_base import (
    HeuristicOptimizer,
    OptimizerType,
)  # noqa: E402
from ao_shaping.optimizer.wfless.pib_sim_eval import SimLandscape  # noqa: E402

# ---------------------------------------------------------------------------
# Global matplotlib conventions (repo rule)
# ---------------------------------------------------------------------------
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUT_DIR = ROOT / "docs" / "heuristic_pib"

DIM = 4
BOUNDS = (-12.0, 12.0)
SEED = 42
INIT_X = np.zeros(DIM, dtype=np.float64)
GLOBAL_CENTER = np.array([-4.8, -4.2, -4.5, -4.0], dtype=np.float64)
LOCAL_CENTER = np.array([2.5, 3.2, 2.2, 2.8], dtype=np.float64)

# (OptimizerType, create() kwargs, evals per iteration)
# NOTE: the factory maps `n_iterations` -> `n_generations` for GA, so the
# spec's "n_generations=300" is passed as n_iterations=300.
ALGORITHMS: list[tuple[OptimizerType, dict[str, int], int]] = [
    (OptimizerType.GA, {"n_iterations": 300, "pop_size": 30}, 30),
    (OptimizerType.PSO, {"n_iterations": 400, "n_particles": 30}, 30),
    (OptimizerType.SA, {"n_iterations": 6000}, 1),
    (OptimizerType.HILL_CLIMBING, {"n_iterations": 4000}, 1),
    (OptimizerType.RANDOM_SEARCH, {"n_iterations": 6000}, 1),
    (OptimizerType.CROSS_ENTROPY, {"n_iterations": 300, "pop_size": 30}, 30),
    (OptimizerType.DIFFERENTIAL_EVOLUTION, {"n_iterations": 300, "pop_size": 30}, 30),
]


def run_all_algorithms(landscape: SimLandscape) -> dict[str, dict]:
    """Run all 7 heuristic optimizers on the PIB landscape.

    The device can load only ONE phase pattern at a time, so a single
    fitness evaluation (one candidate phase written + measured) counts as
    one iteration step. The reported convergence curve is the best-so-far
    PIB **per device load** (exactly one entry per fitness call), not the
    per-inner-iteration history of the optimizer.
    """
    results: dict[str, dict] = {}
    for opt_type, kwargs, evals_per_iter in ALGORITHMS:
        name = opt_type.name
        n_iterations = kwargs["n_iterations"]
        n_evals_budget = evals_per_iter * n_iterations
        logger.info(
            "Running {} (n_iterations={}, budget {} loads)...",
            name,
            n_iterations,
            n_evals_budget,
        )
        t0 = time.perf_counter()

        # Per-evaluation tracking: one fitness call == one device load.
        eval_pibs: list[float] = []

        def fitness(v: np.ndarray) -> float:
            pib = float(landscape.score(v)[0])
            eval_pibs.append(pib)
            return -pib

        opt = HeuristicOptimizer.create(
            opt_type, dim=DIM, bounds=BOUNDS, seed=SEED, **kwargs
        )
        best_x, best_f = opt.optimize(fitness, init_x=INIT_X)
        elapsed = time.perf_counter() - t0
        best_x = np.asarray(best_x, dtype=np.float64)
        final_pib = -float(best_f)

        # best-so-far PIB per device load (monotone curve).
        eval_curve = np.empty(len(eval_pibs), dtype=np.float64)
        running_best = -np.inf
        for i, pib in enumerate(eval_pibs):
            if pib > running_best:
                running_best = pib
            eval_curve[i] = running_best

        logger.info(
            "{} finished in {:.2f}s | final PIB = {:.4f} | best_x = {} | {} loads",
            name,
            elapsed,
            final_pib,
            np.round(best_x, 3),
            len(eval_pibs),
        )
        results[name] = {
            "best_x": best_x,
            "final_pib": final_pib,
            "eval_curve": eval_curve.tolist(),
            "n_evals": len(eval_pibs),
            "elapsed_s": elapsed,
        }
    return results


def plot_pib_curves(results: dict[str, dict], out_dir: Path) -> None:
    """Plot PIB curves vs device-load count for all algorithms.

    x-axis = device loads (one phase pattern loaded + measured = one load),
    not the optimizer's internal iteration count. A dot + `load N` label marks
    the first load crossing the 0.9 PIB threshold; the legend shows the first
    load reaching final max PIB (`max@N`). Horizontal dashed lines mark 0.5
    and 0.9 PIB thresholds.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for idx, (name, res) in enumerate(results.items()):
        pib_curve = np.asarray(res["eval_curve"], dtype=np.float64)
        loads = np.arange(1, len(pib_curve) + 1)
        loads_max = iters_to_threshold(pib_curve, float(pib_curve.max()))
        loads_09 = iters_to_threshold(pib_curve, 0.9)
        label = f"{name} (max@{loads_max})" if loads_max is not None else name
        ax.plot(loads, pib_curve, label=label, color=colors[idx], lw=1.5)
        if loads_09 is not None:
            ax.scatter(
                [loads_09],
                [pib_curve[loads_09 - 1]],
                color=colors[idx],
                s=30,
                zorder=5,
                edgecolors="black",
                linewidths=0.5,
            )
            ax.annotate(
                f"load {loads_09}",
                (loads_09, pib_curve[loads_09 - 1]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
                color=colors[idx],
                fontweight="bold",
            )
    ax.axhline(0.5, color="red", ls=":", lw=1.0, alpha=0.6)
    ax.axhline(0.9, color="green", ls=":", lw=1.0, alpha=0.6)
    ax.text(
        1.0,
        0.51,
        "PIB 0.5",
        color="red",
        fontsize=8,
        va="bottom",
        transform=ax.get_yaxis_transform(),
    )
    ax.text(
        1.0,
        0.91,
        "PIB 0.9",
        color="green",
        fontsize=8,
        va="bottom",
        transform=ax.get_yaxis_transform(),
    )
    ax.set_xscale("log")
    ax.set_xlabel("Device loads (一次相位加载 = 一步)")
    ax.set_ylabel("PIB")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(
        "PIB 收敛曲线 (横轴 = 设备加载次数) — 标记点 = 首次达到 0.9 PIB (load N), legend max@N = 首次达到最终 PIB"
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pib_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved {}", out_dir / "pib_curves.png")


def plot_convergence_speed(results: dict[str, dict], out_dir: Path) -> None:
    """Plot grouped bars (log y) of first device-load counts to PIB 0.5/0.9/max."""
    names = sorted(results, key=lambda n: results[n]["final_pib"], reverse=True)
    groups = {"≥0.5": 0.5, "≥0.9": 0.9, "max": None}
    series: dict[str, list[int | None]] = {k: [] for k in groups}
    for n in names:
        curve = np.asarray(results[n]["eval_curve"], dtype=np.float64)
        for key, thr in groups.items():
            series[key].append(
                iters_to_threshold(curve, float(curve.max()) if thr is None else thr)
            )

    x = np.arange(len(names))
    width = 0.27
    colors = {"≥0.5": "#8fbf8f", "≥0.9": "#4c8fbf", "max": "#c44e52"}
    fig, ax = plt.subplots(figsize=(11, 6))
    for offset, (key, vals) in enumerate(series.items()):
        heights = [v if v is not None else 0 for v in vals]
        bars = ax.bar(
            x + (offset - 1) * width,
            heights,
            width,
            label=key,
            color=colors[key],
        )
        for bar, v in zip(bars, vals):
            if v is not None:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    str(v),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    ax.set_yscale("log")
    ax.set_ylim(
        0.8,
        2.0 * max(max(v for v in vals if v is not None) for vals in series.values()),
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Device loads to reach PIB (log)")
    ax.set_title("收敛速度 (设备加载次数, log 轴): 首次达到 PIB 0.5 / 0.9 / 最大值")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "convergence_speed.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved {}", out_dir / "convergence_speed.png")


def plot_spot_before_after(
    results: dict[str, dict], landscape: SimLandscape, out_dir: Path
) -> None:
    """Plot before/after spot images for all algorithms (2x4 grid, one blank)."""
    init_img = landscape.render(INIT_X)
    best_imgs = {name: landscape.render(res["best_x"]) for name, res in results.items()}
    vmax = max(
        float(init_img.max()),
        *(float(img.max()) for img in best_imgs.values()),
    )

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    for i, (name, res) in enumerate(results.items()):
        ax = axes[i]
        combined = np.hstack([init_img, best_imgs[name]])
        ax.imshow(combined, cmap="hot", vmin=0.0, vmax=vmax)
        ax.set_title(f"{name}\nPIB={res['final_pib']:.3f}", fontsize=9)
        ax.axis("off")
    axes[7].axis("off")
    fig.suptitle("PIB 优化前后光斑对比 (左: 初始, 右: 最优)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "spot_before_after.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved {}", out_dir / "spot_before_after.png")


def plot_summary_bars(results: dict[str, dict], out_dir: Path) -> None:
    """Plot horizontal bar chart of final PIB per algorithm."""
    names = sorted(results, key=lambda n: results[n]["final_pib"], reverse=True)
    pibs = [results[n]["final_pib"] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(names, pibs, color="steelblue")
    for bar, pib in zip(bars, pibs):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{pib:.3f}",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("Final PIB")
    ax.set_xlim(0.0, 1.05)
    ax.set_title("PIB 优化结果对比 (SimLandscape, dim=4)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved {}", out_dir / "summary_bars.png")


def save_summary_csv(results: dict[str, dict], init_pib: float, out_dir: Path) -> None:
    """Save summary CSV with per-algorithm results."""
    rows = []
    for name, res in results.items():
        rows.append(
            {
                "algorithm": name,
                "final_pib": res["final_pib"],
                "init_pib": init_pib,
                "best_x_0": res["best_x"][0],
                "best_x_1": res["best_x"][1],
                "best_x_2": res["best_x"][2],
                "best_x_3": res["best_x"][3],
                "n_loads": res["n_evals"],
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)
    logger.info("Saved {}", out_dir / "summary.csv")


def interpret_basin(best_x: np.ndarray, final_pib: float) -> str:
    """Classify which basin the optimizer found."""
    dist_global = float(np.linalg.norm(best_x - GLOBAL_CENTER))
    dist_local = float(np.linalg.norm(best_x - LOCAL_CENTER))
    if final_pib >= 0.9 and dist_global <= 2.5:
        return "found global basin"
    if 0.6 <= final_pib < 0.9 and dist_local <= 2.5:
        return "found local basin"
    if final_pib >= 0.9:
        return "high PIB (near global)"
    if final_pib >= 0.6:
        return "moderate PIB (near local)"
    return "did not converge"


def iters_to_threshold(pib_curve: np.ndarray, threshold: float) -> int | None:
    """First iteration (1-based) where PIB >= threshold, else None."""
    reached = np.flatnonzero(pib_curve >= threshold - 1e-12)
    return int(reached[0]) + 1 if reached.size else None


def format_iters(v: int | None) -> str:
    """Format an iteration count, or a dash when the threshold was never reached."""
    return str(v) if v is not None else "—"


def write_report(results: dict[str, dict], init_pib: float, out_dir: Path) -> None:
    """Write markdown report with results table and interpretation."""
    lines = [
        "# Heuristic PIB Benchmark Report",
        "",
        "Benchmark of all 7 heuristic optimizers in `ao_shaping.algorithm` on the "
        "PIB (power-in-bucket) optimization problem, offline (pure numpy, no hardware), "
        "using the synthetic landscape from "
        "`ao_shaping.optimizer.wfless.pib_sim_eval.SimLandscape`.",
        "",
        f"- **dim**: {DIM}, **bounds**: {BOUNDS}, **seed**: {SEED}",
        f"- **init_x**: {INIT_X.tolist()} (zero voltage), **init PIB**: {init_pib:.4f}",
        f"- **global center**: {GLOBAL_CENTER.tolist()} (PIB = 1.0)",
        f"- **local center**: {LOCAL_CENTER.tolist()} (PIB ~ 0.9)",
        "",
        "## Results",
        "",
        "| Algorithm | Final PIB | Improvement | Loads to max | Loads ≥ 0.9 | "
        "Loads ≥ 0.5 | n_loads | Interpretation |",
        "|-----------|-----------|-------------|--------------|-------------|"
        "-------------|---------|----------------|",
    ]
    for name, res in sorted(
        results.items(), key=lambda kv: kv[1]["final_pib"], reverse=True
    ):
        improvement = res["final_pib"] - init_pib
        interp = interpret_basin(res["best_x"], res["final_pib"])
        pib_curve = np.asarray(res["eval_curve"], dtype=np.float64)
        loads_max = iters_to_threshold(pib_curve, float(pib_curve.max()))
        loads_09 = iters_to_threshold(pib_curve, 0.9)
        loads_05 = iters_to_threshold(pib_curve, 0.5)
        lines.append(
            f"| {name} | {res['final_pib']:.4f} | {improvement:+.4f} | "
            f"{format_iters(loads_max)} | {format_iters(loads_09)} | "
            f"{format_iters(loads_05)} | {res['n_evals']} | {interp} |"
        )
    lines += [
        "",
        "## Algorithm Principles",
        "",
        "All 7 algorithms optimize the scalar fitness function "
        "`fitness(v) = -PIB(v)` (minimize negative PIB = maximize PIB) over "
        "the 4-D coefficient space `v`, with bounds (-12, 12). Their working "
        "principles:",
        "",
        "| Algorithm | Principle |",
        "|-----------|-----------|",
        "| **GA** (Genetic Algorithm) | Population-based evolutionary search: maintains a population of candidate solutions, applies selection (tournament), crossover and mutation each generation, keeps elite individuals, and iterates toward the optimum. |",
        "| **PSO** (Particle Swarm) | Swarm intelligence: each particle moves through the search space with a velocity updated toward its own personal best (`c1`) and the swarm's global best (`c2`) positions. |",
        "| **SA** (Simulated Annealing) | Probabilistic local search: starts at a high temperature and accepts *worse* moves with probability `exp(-Δf/T)`, gradually cooling so the search first escapes local minima then fine-tunes. |",
        "| **HC** (Hill Climbing) | Greedy local search: perturbs the current best candidate, accepts only strictly improving moves — fast but can get stuck in local minima. |",
        "| **RS** (Random Search) | Uniformly samples candidate points across the whole bounded space and keeps the best seen — a no-gradient baseline. |",
        "| **CEM** (Cross-Entropy) | Model-based search: samples candidates from a parameterized Gaussian, keeps elite samples, and re-fits the Gaussian mean/covariance to the elites each iteration. |",
        "| **DE** (Differential Evolution) | Population-based differential mutation: builds candidate vectors by adding scaled differences of other population members to a base vector, then crosses and greedily selects. |",
        "",
        "## Interpretation",
        "",
        "Each algorithm's `best_x` is compared against the global center "
        f"{GLOBAL_CENTER.tolist()} and the local center {LOCAL_CENTER.tolist()}.",
        "PIB >= ~0.9 with `best_x` near the global center = found the global basin; "
        "PIB ~0.6-0.9 = found the local basin; below that = did not converge.",
        "",
        "**Convergence speed** (`Iters to max` / `Iters ≥ 0.9` / `Iters ≥ 0.5`): "
        "the first iteration at which the PIB curve reaches its final maximum, "
        "crosses 0.9, and crosses 0.5, respectively (convergence history is "
        "best-so-far, so the curves are monotone). '—' means the threshold was "
        "never reached within the iteration budget.",
        "",
        "This report is generated offline by `scripts/generate_heuristic_pib_report.py` "
        "and can be regenerated at any time without hardware.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved {}", out_dir / "report.md")


def main() -> None:
    """Run the full benchmark and write all outputs."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    landscape = SimLandscape()
    init_pib = float(landscape.score(INIT_X)[0])
    logger.info("SimLandscape ready | init PIB = {:.4f}", init_pib)

    results = run_all_algorithms(landscape)
    plot_pib_curves(results, OUT_DIR)
    plot_convergence_speed(results, OUT_DIR)
    plot_spot_before_after(results, landscape, OUT_DIR)
    plot_summary_bars(results, OUT_DIR)
    save_summary_csv(results, init_pib, OUT_DIR)
    write_report(results, init_pib, OUT_DIR)
    logger.info("All outputs written to {}", OUT_DIR)


if __name__ == "__main__":
    main()
