"""Generate Strehl benchmark report (7 heuristics + SPGD).

Benchmarks the 7 heuristic optimizers in ``ao_shaping.algorithm`` **and** SPGD
(stochastic parallel gradient descent) on a common offline Strehl objective:
correcting atmospheric turbulence in the physical simulation
``TraditionalAOSystem`` (DM influence functions + Fourier focal plane, 64-D
voltage space, bounds (-1, 1)). The device can only hold one phase pattern at a
time, so 1 device load = 1 phase write+measure = 1 fitness evaluation = 1
iteration step, matching the semantics of the heuristic PIB benchmark.

The turbulence screen is fixed by ``StrehlLandscape(seed=SEED)``, so every
algorithm optimises the *same* landscape and results are reproducible.

Emits Strehl iteration curves + before/after spot images + summary into
``docs/strehl_benchmark/``, and appends a cross-benchmark comparison against the
PIB benchmark (``docs/heuristic_pib/summary.csv``, dim=4 synthetic) when that
summary is present.

Usage:
    $env:PYTHONPATH = "src;libs"
    python scripts/generate_strehl_benchmark_report.py [--n-grid 256]
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
from ao_shaping.optimizer.spgd import spgd_gradient  # noqa: E402
from ao_shaping.optimizer.wfless.strehl_sim_eval import StrehlLandscape  # noqa: E402

# ---------------------------------------------------------------------------
# Global matplotlib conventions (repo rule)
# ---------------------------------------------------------------------------
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUT_DIR = ROOT / "docs" / "strehl_benchmark"

SEED = 42
N_GRID = 256
DIM = 64  # dm_actuators=8 -> 8**2; must match StrehlLandscape defaults
BOUNDS = (-1.0, 1.0)
INIT_X = np.zeros(DIM, dtype=np.float64)

# SPGD fixed-gain parameters (smoke-tested on n_grid=128: reaches Strehl 1.0
# within ~1000 loads; the EMA momentum mirrors sim_spgd's default behaviour).
SPGD_STEPS = 4000  # 2 device loads per step -> 8000 loads
SPGD_GAMMA = 2.0
SPGD_DELTA = 0.1
SPGD_BETA1 = 0.9

# (OptimizerType, create() kwargs, evals per iteration) — same spec as the
# heuristic PIB benchmark, so load budgets match across benchmarks.
ALGORITHMS: list[tuple[OptimizerType, dict[str, int], int]] = [
    (OptimizerType.GA, {"n_iterations": 300, "pop_size": 30}, 30),
    (OptimizerType.PSO, {"n_iterations": 400, "n_particles": 30}, 30),
    (OptimizerType.SA, {"n_iterations": 6000}, 1),
    (OptimizerType.HILL_CLIMBING, {"n_iterations": 4000}, 1),
    (OptimizerType.RANDOM_SEARCH, {"n_iterations": 6000}, 1),
    (OptimizerType.CROSS_ENTROPY, {"n_iterations": 300, "pop_size": 30}, 30),
    (OptimizerType.DIFFERENTIAL_EVOLUTION, {"n_iterations": 300, "pop_size": 30}, 30),
]


def run_all_algorithms(landscape: StrehlLandscape) -> dict[str, dict]:
    """Run all 7 heuristic optimizers on the Strehl landscape.

    One fitness evaluation == one device load (one candidate phase written +
    measured), matching the device-load semantics of the PIB benchmark. The
    reported convergence curve is the best-so-far Strehl **per device load**.
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
        eval_strehls: list[float] = []

        def fitness(v: np.ndarray) -> float:
            strehl = float(landscape.score(v)[0])
            eval_strehls.append(strehl)
            return -strehl

        opt = HeuristicOptimizer.create(
            opt_type, dim=DIM, bounds=BOUNDS, seed=SEED, **kwargs
        )
        best_x, best_f = opt.optimize(fitness, init_x=INIT_X)
        elapsed = time.perf_counter() - t0
        best_x = np.asarray(best_x, dtype=np.float64)
        final_strehl = -float(best_f)

        # best-so-far Strehl per device load (monotone curve).
        eval_curve = np.empty(len(eval_strehls), dtype=np.float64)
        running_best = -np.inf
        for i, s in enumerate(eval_strehls):
            if s > running_best:
                running_best = s
            eval_curve[i] = running_best

        logger.info(
            "{} finished in {:.2f}s | final Strehl = {:.4f} | {} loads",
            name,
            elapsed,
            final_strehl,
            len(eval_strehls),
        )
        results[name] = {
            "best_x": best_x,
            "final_strehl": final_strehl,
            "eval_curve": eval_curve.tolist(),
            "n_evals": len(eval_strehls),
            "elapsed_s": elapsed,
        }
    return results


def run_spgd(landscape: StrehlLandscape) -> dict:
    """Run classic SPGD (2 device loads per step) with fixed gain + EMA momentum.

    Each step perturbs the current voltage vector by a random sign pattern
    ``disturb_i = ±delta``, measures Strehl at ``v + disturb`` and
    ``v - disturb`` (2 device loads), forms the ascent estimate through the
    canonical ``spgd_gradient(pos, neg, disturb, maximize=True)`` helper (the
    only sign-safe way to build it — see ``optimizer/spgd.py``), and walks
    ``v <- v - gamma * momentum`` in the descent frame, i.e. upward in Strehl.

    ``best_x`` is the actually-loaded voltage of the best measured Strehl, so
    ``landscape.render(best_x)`` shows a real device configuration.
    """
    rng = np.random.default_rng(SEED)
    dim = landscape.dim
    lo, hi = landscape.bounds
    param = landscape.init_v.copy()
    momentum = np.zeros(dim, dtype=np.float64)
    eval_strehls: list[float] = []
    best_s = -np.inf
    best_x = param.copy()

    def measure(v: np.ndarray) -> float:
        nonlocal best_s, best_x
        s = float(landscape.score(v)[0])
        eval_strehls.append(s)
        if s > best_s:
            best_s = s
            best_x = v.copy()
        return s

    logger.info(
        "Running SPGD (steps={}, gamma={}, delta={}, beta1={}) budget {} loads...",
        SPGD_STEPS,
        SPGD_GAMMA,
        SPGD_DELTA,
        SPGD_BETA1,
        2 * SPGD_STEPS,
    )
    t0 = time.perf_counter()
    for _ in range(SPGD_STEPS):
        disturb = (rng.binomial(1, 0.5, dim) * 2 - 1) * SPGD_DELTA
        s_pos = measure(np.clip(param + disturb, lo, hi))
        s_neg = measure(np.clip(param - disturb, lo, hi))
        grad = spgd_gradient(s_pos, s_neg, disturb, maximize=True)
        momentum = SPGD_BETA1 * momentum + (1.0 - SPGD_BETA1) * grad
        param = np.clip(param - SPGD_GAMMA * momentum, lo, hi)
    elapsed = time.perf_counter() - t0

    eval_curve = np.empty(len(eval_strehls), dtype=np.float64)
    running_best = -np.inf
    for i, s in enumerate(eval_strehls):
        running_best = max(running_best, s)
        eval_curve[i] = running_best

    logger.info(
        "SPGD finished in {:.2f}s | final Strehl = {:.4f} | {} loads",
        elapsed,
        best_s,
        len(eval_strehls),
    )
    return {
        "best_x": best_x,
        "final_strehl": best_s,
        "eval_curve": eval_curve.tolist(),
        "n_evals": len(eval_strehls),
        "elapsed_s": elapsed,
    }


def plot_strehl_curves(results: dict[str, dict], out_dir: Path) -> None:
    """Plot Strehl curves vs device-load count for all algorithms.

    x-axis = device loads (one phase pattern loaded + measured = one load),
    not the optimizer's internal iteration count. A dot + `load N` label marks
    the first load crossing the 0.9 Strehl threshold; the legend shows the
    first load reaching final max Strehl (`max@N`). Horizontal dashed lines
    mark 0.5 and 0.9 Strehl thresholds.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for idx, (name, res) in enumerate(results.items()):
        curve = np.asarray(res["eval_curve"], dtype=np.float64)
        loads = np.arange(1, len(curve) + 1)
        loads_max = iters_to_threshold(curve, float(curve.max()))
        loads_09 = iters_to_threshold(curve, 0.9)
        label = f"{name} (max@{loads_max})" if loads_max is not None else name
        ax.plot(loads, curve, label=label, color=colors[idx], lw=1.5)
        if loads_09 is not None:
            ax.scatter(
                [loads_09],
                [curve[loads_09 - 1]],
                color=colors[idx],
                s=30,
                zorder=5,
                edgecolors="black",
                linewidths=0.5,
            )
            ax.annotate(
                f"load {loads_09}",
                (loads_09, curve[loads_09 - 1]),
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
        "Strehl 0.5",
        color="red",
        fontsize=8,
        va="bottom",
        transform=ax.get_yaxis_transform(),
    )
    ax.text(
        1.0,
        0.91,
        "Strehl 0.9",
        color="green",
        fontsize=8,
        va="bottom",
        transform=ax.get_yaxis_transform(),
    )
    ax.set_xscale("log")
    ax.set_xlabel("Device loads (一次相位加载 = 一步)")
    ax.set_ylabel("Strehl")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(
        "Strehl 收敛曲线 (横轴 = 设备加载次数) — 标记点 = 首次达到 0.9 Strehl (load N), legend max@N = 首次达到最终 Strehl"
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "strehl_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved {}", out_dir / "strehl_curves.png")


def plot_convergence_speed(results: dict[str, dict], out_dir: Path) -> None:
    """Plot grouped bars (log y) of first device-load counts to Strehl 0.5/0.9/max."""
    names = sorted(results, key=lambda n: results[n]["final_strehl"], reverse=True)
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
    ax.set_ylabel("Device loads to reach Strehl (log)")
    ax.set_title("收敛速度 (设备加载次数, log 轴): 首次达到 Strehl 0.5 / 0.9 / 最大值")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "convergence_speed.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved {}", out_dir / "convergence_speed.png")


def plot_spot_before_after(
    results: dict[str, dict], landscape: StrehlLandscape, out_dir: Path
) -> None:
    """Plot before/after spot images for all 8 algorithms (2x4 grid, full)."""
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
        ax.set_title(f"{name}\nStrehl={res['final_strehl']:.3f}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Strehl 优化前后光斑对比 (左: 初始, 右: 最优)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "spot_before_after.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved {}", out_dir / "spot_before_after.png")


def plot_summary_bars(results: dict[str, dict], out_dir: Path) -> None:
    """Plot horizontal bar chart of final Strehl per algorithm."""
    names = sorted(results, key=lambda n: results[n]["final_strehl"], reverse=True)
    strehls = [results[n]["final_strehl"] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(names, strehls, color="steelblue")
    for bar, s in zip(bars, strehls):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{s:.3f}",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("Final Strehl")
    ax.set_xlim(0.0, 1.05)
    ax.set_title("Strehl 优化结果对比 (TraditionalAOSystem, dim=64)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved {}", out_dir / "summary_bars.png")


def save_summary_csv(
    results: dict[str, dict], init_strehl: float, out_dir: Path
) -> None:
    """Save summary CSV with per-algorithm results (best_x omitted: 64-D)."""
    rows = []
    for name, res in results.items():
        rows.append(
            {
                "algorithm": name,
                "final_strehl": res["final_strehl"],
                "init_strehl": init_strehl,
                "n_loads": res["n_evals"],
                "elapsed_s": round(res["elapsed_s"], 3),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)
    logger.info("Saved {}", out_dir / "summary.csv")


def interpret_strehl(final_strehl: float) -> str:
    """Classify the Strehl level reached (no basin structure on this landscape)."""
    if final_strehl >= 0.9:
        return "near-diffraction-limited"
    if final_strehl >= 0.7:
        return "good correction"
    if final_strehl >= 0.5:
        return "partial correction"
    return "poor correction"


def iters_to_threshold(curve: np.ndarray, threshold: float) -> int | None:
    """First iteration (1-based) where Strehl >= threshold, else None."""
    reached = np.flatnonzero(curve >= threshold - 1e-12)
    return int(reached[0]) + 1 if reached.size else None


def format_iters(v: int | None) -> str:
    """Format an iteration count, or a dash when the threshold was never reached."""
    return str(v) if v is not None else "—"


def load_pib_summary() -> pd.DataFrame | None:
    """Load the PIB benchmark summary for the cross-benchmark comparison."""
    pib_csv = ROOT / "docs" / "heuristic_pib" / "summary.csv"
    if not pib_csv.exists():
        return None
    try:
        df = pd.read_csv(pib_csv)
        cols = ["algorithm", "final_pib", "n_loads"]
        if not all(c in df.columns for c in cols):
            logger.warning("PIB summary {} misses columns {}", pib_csv, cols)
            return None
        return df.loc[:, cols]
    except Exception as exc:  # noqa: BLE001 - report-only, never abort the run
        logger.warning("Could not read PIB summary {}: {}", pib_csv, exc)
        return None


def write_report(results: dict[str, dict], init_strehl: float, out_dir: Path) -> None:
    """Write markdown report with results table, principles and comparison."""
    lines = [
        "# Strehl Benchmark Report (7 Heuristics + SPGD)",
        "",
        "Benchmark of the 7 heuristic optimizers in `ao_shaping.algorithm` "
        "**plus SPGD** on a common offline **Strehl** objective: correcting "
        "atmospheric turbulence in the physical simulation "
        "`TraditionalAOSystem` (DM influence functions + Fourier focal plane), "
        "offline (pure numpy, no hardware), landscape from "
        "`ao_shaping.optimizer.wfless.strehl_sim_eval.StrehlLandscape`.",
        "",
        "- **dim**: 64 (DM voltage space, dm_actuators=8), "
        f"**bounds**: {BOUNDS}, **seed**: {SEED}",
        f"- **init_x**: zeros(64) (zero voltage / flat DM), "
        f"**init Strehl**: {init_strehl:.4f}",
        f"- **n_grid**: {N_GRID} (turbulence screen fixed by the landscape seed)",
        f"- **SPGD**: fixed gain gamma={SPGD_GAMMA}, delta={SPGD_DELTA}, "
        f"momentum beta1={SPGD_BETA1}, {SPGD_STEPS} steps x 2 loads = "
        f"{2 * SPGD_STEPS} loads",
        "",
        "## Results",
        "",
        "| Algorithm | Final Strehl | Improvement | Loads to max | Loads ≥ 0.9 | "
        "Loads ≥ 0.5 | n_loads | Interpretation |",
        "|-----------|--------------|-------------|--------------|-------------|"
        "-------------|---------|----------------|",
    ]
    for name, res in sorted(
        results.items(), key=lambda kv: kv[1]["final_strehl"], reverse=True
    ):
        improvement = res["final_strehl"] - init_strehl
        interp = interpret_strehl(res["final_strehl"])
        curve = np.asarray(res["eval_curve"], dtype=np.float64)
        loads_max = iters_to_threshold(curve, float(curve.max()))
        loads_09 = iters_to_threshold(curve, 0.9)
        loads_05 = iters_to_threshold(curve, 0.5)
        lines.append(
            f"| {name} | {res['final_strehl']:.4f} | {improvement:+.4f} | "
            f"{format_iters(loads_max)} | {format_iters(loads_09)} | "
            f"{format_iters(loads_05)} | {res['n_evals']} | {interp} |"
        )
    lines += [
        "",
        "## Algorithm Principles",
        "",
        "All 8 algorithms optimize the scalar fitness "
        "`fitness(v) = -Strehl(v)` (minimize negative Strehl = maximize Strehl) "
        f"over the 64-D DM voltage space `v`, with bounds {BOUNDS}. Their "
        "working principles:",
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
        "| **SPGD** (Stochastic Parallel Gradient Descent) | Perturbation-based gradient ascent: at each step perturbs the current voltage vector by a random sign pattern `±delta`, measures Strehl at `v+disturb` and `v-disturb` (2 device loads), estimates the ascent direction `(J_pos - J_neg) * disturb`, and walks `v <- v - gamma * momentum` with an EMA momentum (`beta1`) over the descent-frame gradient — it uses gradient information, so it scales to the full 64-D space. |",
        "",
        "## Interpretation",
        "",
        "Strehl levels reached (best-so-far across all device loads): "
        "≥ 0.9 = near-diffraction-limited focus; ≥ 0.7 = good correction; "
        "≥ 0.5 = partial correction; < 0.5 = poor correction. Note that the "
        "flat-DM baseline already sits at a high Strehl (the turbulence is "
        "mild at this seed), so the meaningful question is how quickly each "
        "algorithm closes the remaining gap.",
        "",
        "**Convergence speed** (`Loads to max` / `Loads ≥ 0.9` / `Loads ≥ 0.5`): "
        "the first device load at which the best-so-far Strehl curve reaches its "
        "final maximum, crosses 0.9, and crosses 0.5, respectively. '—' means "
        "the threshold was never reached within the load budget.",
        "",
        "## Comparison with the PIB Benchmark",
        "",
        "The heuristic PIB benchmark (`docs/heuristic_pib/report.md`, generated "
        "by `scripts/generate_heuristic_pib_report.py`) optimizes a **synthetic "
        "dim=4** landscape; this benchmark optimizes the **physical dim=64** "
        "simulation, so rankings are valid *within* each benchmark only. SPGD "
        "participates only here (it is not a `HeuristicOptimizer` algorithm), "
        "which is the gap the original heuristic benchmark left open.",
        "",
    ]

    pib_df = load_pib_summary()
    if pib_df is not None:
        lines += [
            "| Algorithm | PIB (dim=4) final | PIB loads | Strehl (dim=64) final | Strehl loads |",
            "|-----------|-------------------|-----------|------------------------|--------------|",
        ]
        pib_final_map = dict(
            zip(pib_df["algorithm"].astype(str), pib_df["final_pib"].astype(float))
        )
        pib_loads_map = dict(
            zip(pib_df["algorithm"].astype(str), pib_df["n_loads"].astype(int))
        )
        for name in results:
            pib_final = pib_final_map.get(name)
            pib_loads = pib_loads_map.get(name)
            pib_final_txt = f"{pib_final:.4f}" if pib_final is not None else "—"
            pib_loads_txt = str(pib_loads) if pib_loads is not None else "—"
            lines.append(
                f"| {name} | {pib_final_txt} | {pib_loads_txt} | "
                f"{results[name]['final_strehl']:.4f} | {results[name]['n_evals']} |"
            )
        lines += [
            "",
            "SPGD has no PIB-benchmark row: that benchmark only covers `HeuristicOptimizer` algorithms.",
        ]
    else:
        lines += [
            "_`docs/heuristic_pib/summary.csv` not found — cross-benchmark "
            "comparison skipped._",
        ]

    lines += [
        "",
        "This report is generated offline by `scripts/generate_strehl_benchmark_report.py` "
        "and can be regenerated at any time without hardware.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved {}", out_dir / "report.md")


def main(n_grid: int = N_GRID) -> None:
    """Run the full benchmark and write all outputs."""
    if n_grid != 256:
        logger.warning(
            "n_grid={} != 256 — numbers are NOT comparable to the default run", n_grid
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    landscape = StrehlLandscape(seed=SEED, n_grid=n_grid)
    logger.info("StrehlLandscape ready (n_grid={}, dim={})", n_grid, landscape.dim)
    if landscape.dim != DIM:
        raise ValueError(f"landscape dim {landscape.dim} != DIM {DIM}")
    init_strehl = float(landscape.score(INIT_X)[0])
    logger.info("init Strehl (flat DM) = {:.4f}", init_strehl)

    results = run_all_algorithms(landscape)
    results["SPGD"] = run_spgd(landscape)
    plot_strehl_curves(results, OUT_DIR)
    plot_convergence_speed(results, OUT_DIR)
    plot_spot_before_after(results, landscape, OUT_DIR)
    plot_summary_bars(results, OUT_DIR)
    save_summary_csv(results, init_strehl, OUT_DIR)
    write_report(results, init_strehl, OUT_DIR)
    logger.info("All outputs written to {}", OUT_DIR)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--n-grid",
        type=int,
        default=N_GRID,
        help=f"Simulation FFT grid size (default: {N_GRID}; use 128 for smoke runs)",
    )
    args = parser.parse_args()
    main(n_grid=args.n_grid)
