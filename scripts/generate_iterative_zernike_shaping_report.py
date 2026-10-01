from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from loguru import logger

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from ao_shaping.algorithm.signal_processing.iterative_zernike_shaping import (
    IterativeZernikeShapingConfig,
    IterativeZernikeShapingOptimizer,
)
from ao_shaping.drivers.sim.slm_shaping_bench import (
    ShapingBenchConfig,
    composite_score,
    compute_metrics,
    forward_intensity,
    gs_shape,
    make_target,
    spgd_shape,
)
from ao_shaping.optimizer.wfless.iterative_zernike_shaping import (
    IterativeZernikePibConfig,
)
from ao_shaping.utils.wavefront.zernike_utils import generate_zernike_phase


N = 64
PAD = 8
TARGET_SIDE = 43  # far-field px ~ 30 um ~ 2.2 Airy diameters
SEED = 0
# 0.15/0.10/0.10 waves RMS (coefficient is the mode's RMS phase in rad,
# so waves = coefficient / 2*pi). Strehl ~ exp(-sum a^2) ~ 0.19.
GOLDEN = {(2, 0): 0.94, (2, -2): 0.63, (4, 0): 0.63}


def make_out_dir() -> Path:
    out = ROOT / "docs" / "iterative_zernike_shaping"
    out.mkdir(parents=True, exist_ok=True)
    return out


def bench_cfg() -> ShapingBenchConfig:
    return ShapingBenchConfig(
        n_grid=N, target_side_px=TARGET_SIDE, far_field_padding=PAD, seed=SEED
    )


def build_actual_far_field() -> tuple[np.ndarray, np.ndarray]:
    """Build the reference 'actual' far-field (golden Zernike, flat SLM phase).

    Returns:
        ``(far_field, zernike_phase)``. The SLM phase is flat, matching the
        optimizer's reference, so the initial panel shows the same state the A↔B
        loop starts from.
    """
    cfg = bench_cfg()
    zp = np.nan_to_num(
        np.asarray(generate_zernike_phase(GOLDEN, resolution=(N, N), n_max=4), dtype=np.float64),
        nan=0.0,
    )
    ff = forward_intensity(zp, cfg)
    return ff / (ff.sum() + 1e-12), zp


def run_gs_baseline() -> tuple[float, float, float, np.ndarray]:
    """Run single-pass GS shaping as the single-pass baseline.

    Returns:
        (score, pib, cv, gs_phase): composite score, PIB, CV, and the GS phase.
    """
    cfg = bench_cfg()
    target = make_target(cfg)
    gs = gs_shape(cfg, n_iters=200, seed=SEED)
    ff = forward_intensity(gs.phase, cfg)
    ff = ff / (ff.sum() + 1e-12)
    m = compute_metrics(ff, target)
    return composite_score(m), m["PIB"], m["CV"], gs.phase


def run_spgd_baseline() -> tuple[float, float, float, np.ndarray]:
    """Run sensorless SPGD on the same 64×64 grid as a second baseline.

    SPGD is the canonical sensorless beam-shaping method; running it on the
    same grid as the iterative A↔B loop makes the comparison apples-to-apples
    (the 0.89 figure cited elsewhere is a 1920×1200 hardware-grid result and
    is not comparable to this 64×64 simulation).

    Returns:
        (score, pib, cv, spgd_phase): composite score, PIB, CV, and the SPGD
        phase on the full grid.
    """
    cfg = bench_cfg()
    target = make_target(cfg)
    spgd = spgd_shape(cfg, n_iters=600, delta=0.1, lr=0.02, seed=SEED, dim=8)
    ff = forward_intensity(spgd.phase, cfg)
    ff = ff / (ff.sum() + 1e-12)
    m = compute_metrics(ff, target)
    return composite_score(m), m["PIB"], m["CV"], spgd.phase


def run_iterative(n_zernike: int = 0) -> dict:
    """Run the adaptive free-form refinement loop (GS warm start).

    ``n_zernike=0`` (default) skips the Zernike calibration pass, which is a
    documented negative result on this model; set 4 to run the ablation.

    Returns:
        Result dict from ``optimize_iterative_zernike_shaping``.
    """
    from ao_shaping.optimizer.wfless.iterative_zernike_shaping import (
        optimize_iterative_zernike_shaping,
    )

    cfg = IterativeZernikePibConfig(
        n_grid=N,
        n_zernike=n_zernike,
        target_side_px=TARGET_SIDE,
        seed=SEED,
        zernike_lr=0.05,
        slm_lr=0.02,
        calib_iters=50,
        shaping_iters=100,
        max_outer_iters=5,
        far_field_padding=PAD,
    )
    return optimize_iterative_zernike_shaping(cfg)


def plot_initial_vs_final(
    ff_init: np.ndarray, res: dict, target: np.ndarray, out: Path
) -> None:
    """Figure 1: initial far-field vs final far-field vs target."""
    ff_final = res["far_field"]

    # Zoom + log scale: at full frame the aberrated halo is invisible against
    # the peak, and the target box is a few pixels of a 512-wide grid.
    size = ff_init.shape[0]
    half = min(size // 2, 90)
    c = size // 2
    window = (slice(c - half, c + half), slice(c - half, c + half))

    panels = [
        ("初始远场 (golden)", ff_init[window]),
        ("迭代后远场 (GS 预热 + 细化)", ff_final[window]),
        ("目标方形", target[window]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (title, img) in zip(axes, panels):
        im = ax.imshow(
            img, cmap="inferno", norm=LogNorm(vmin=max(img.max() * 1e-4, 1e-12), vmax=img.max())
        )
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out / "initial_vs_final.png", dpi=150)
    plt.close(fig)


def plot_score_history(res: dict, gs_score: float, out: Path) -> None:
    """Figure 2: score convergence history + GS baseline."""
    hist = res["score_history"]
    iters = [h["outer_iter"] for h in hist]
    scores = [h["score"] for h in hist]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iters, scores, "o-", color="#2196F3", linewidth=2, label="GS 预热 + 细化")
    ax.axhline(gs_score, color="#FF5722", linestyle="--", linewidth=1.5,
               label=f"GS 单遍 ({gs_score:.3f})")
    ax.axhline(res["final_score"], color="#4CAF50", linestyle=":", linewidth=1.5,
               label=f"最终 ({res['final_score']:.3f})")
    ax.set_xlabel("外迭代次数")
    ax.set_ylabel("综合评分 (PIB + 均匀性)")
    ax.set_title("GS 预热 + 自由相位细化 — 评分收敛")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "score_history.png", dpi=150)
    plt.close(fig)


def plot_zernike_coeffs(res: dict, out: Path) -> None:
    """Figure 3: calibrated Zernike coefficients vs golden."""
    coeffs = res["zernike_coeffs"]
    modes = sorted(coeffs.keys())
    nms = [f"({n},{m})" for n, m in modes]
    calib = [coeffs[(n, m)] for n, m in modes]
    golden_vals = [GOLDEN.get((n, m), 0.0) for n, m in modes]

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(nms))
    w = 0.35
    ax.bar(x - w / 2, golden_vals, w, color="#FFC107", label="Golden (参考)")
    ax.bar(x + w / 2, calib, w, color="#2196F3", label="校准后")
    ax.set_xticks(x)
    ax.set_xticklabels(nms, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("系数 (rad)")
    ax.set_title("Zernike 系数: golden vs 校准后")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "zernike_coeffs.png", dpi=150)
    plt.close(fig)


def plot_phase_evolution(res: dict, out: Path) -> None:
    """Figure 4: SLM phase (raw radians) before/after mod 2π."""
    phase = res["slm_phase"]
    phase_mod = np.mod(phase, 2 * np.pi)
    vmax = max(abs(phase.min()), abs(phase.max()))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(phase, cmap="twilight", vmin=-vmax, vmax=vmax)
    axes[0].set_title("SLM 相位 (raw rad)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(phase_mod, cmap="twilight", vmin=0, vmax=2 * np.pi)
    axes[1].set_title("SLM 相位 (mod 2π)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out / "phase_evolution.png", dpi=150)
    plt.close(fig)


def plot_comparison_bars(
    init_score: float, gs_score: float, spgd_score: float, res: dict, out: Path
) -> None:
    """Figure 5: bar chart comparing initial / GS / SPGD / iterative scores."""
    labels = ["初始\n(golden)", "GS 单遍\n(200 it)", "SPGD\n(600 it)", "迭代细化\n(GS 预热)"]
    scores = [init_score, gs_score, spgd_score, res["final_score"]]
    colors = ["#9E9E9E", "#FFC107", "#FF9800", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, scores, color=colors, edgecolor="black", linewidth=0.5)
    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{s:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("综合评分")
    ax.set_title("评分对比: 初始 vs GS vs SPGD vs 迭代细化")
    ax.set_ylim(0, max(scores) * 1.2)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "score_comparison.png", dpi=150)
    plt.close(fig)


def main() -> None:
    logger.info("Generating iterative Zernike shaping report")
    out = make_out_dir()
    t0 = datetime.now()

    # Build reference
    actual_ff, golden_phase = build_actual_far_field()
    cfg = bench_cfg()
    target = make_target(cfg)

    # Initial state == the reference far-field (golden Zernike, flat SLM phase)
    ff_init = actual_ff
    m_init = compute_metrics(ff_init, target)
    init_score = composite_score(m_init)
    logger.info("Initial score={:.4f} PIB={:.4f} CV={:.4f}", init_score, m_init["PIB"], m_init["CV"])

    # GS baseline
    gs_score, gs_pib, gs_cv, gs_phase = run_gs_baseline()
    logger.info("GS score={:.4f} PIB={:.4f} CV={:.4f}", gs_score, gs_pib, gs_cv)

    # SPGD baseline (same grid, sensorless black-box)
    spgd_score, spgd_pib, spgd_cv, spgd_phase = run_spgd_baseline()
    logger.info("SPGD score={:.4f} PIB={:.4f} CV={:.4f}", spgd_score, spgd_pib, spgd_cv)

    # Iterative (GS warm start, free-form refinement; Zernike calibration off)
    res = run_iterative(n_zernike=0)
    final_score = res["final_score"]
    m_final = res["metrics"]
    logger.info(
        "Iterative score={:.4f} PIB={:.4f} CV={:.4f} n_outer={} converged={}",
        final_score, m_final["PIB"], m_final["CV"], res["n_outer_iters"], res["converged"],
    )

    # Ablation: the Zernike calibration pass (documented negative result)
    res_z = run_iterative(n_zernike=4)
    z_score = res_z["metrics"]["score"]
    logger.info(
        "Ablation: Zernike calibration ON -> score={:.4f} ({:+.1f}% vs Zernike OFF)",
        z_score, (z_score - final_score) / final_score * 100,
    )

    improvement_vs_gs = (final_score - gs_score) / gs_score * 100 if gs_score > 0 else 0
    improvement_vs_spgd = (final_score - spgd_score) / spgd_score * 100 if spgd_score > 0 else 0
    improvement_vs_init = (final_score - init_score) / init_score * 100 if init_score > 0 else 0
    ranking = sorted(
        [
            ("iterative", final_score),
            ("initial", init_score),
            ("GS", gs_score),
            ("SPGD", spgd_score),
        ],
        key=lambda item: item[1],
        reverse=True,
    )
    logger.info(
        "Measured score ranking (high->low): {}",
        " > ".join(f"{name} {score:.3f}" for name, score in ranking),
    )
    logger.info(
        "Iterative vs baselines: SPGD {:+.1f}%, GS {:+.1f}%, initial {:+.1f}%",
        improvement_vs_spgd,
        improvement_vs_gs,
        improvement_vs_init,
    )

    # Plots
    plot_initial_vs_final(ff_init, res, target, out)
    plot_score_history(res, gs_score, out)
    plot_zernike_coeffs(res, out)
    plot_phase_evolution(res, out)
    plot_comparison_bars(init_score, gs_score, spgd_score, res, out)

    # Save data
    data = {
        "timestamp": t0.isoformat(),
        "n_grid": N,
        "far_field_padding": PAD,
        "far_field_size": cfg.far_field_size,
        "far_field_pixel_size_um": cfg.far_field_pixel_size * 1e6,
        "target_side_px": TARGET_SIDE,
        "target_side_um": TARGET_SIDE * cfg.far_field_pixel_size * 1e6,
        "objective": "0.5*PIB + 0.5*(1/(1+CV)) in the grid-centred target support",
        "seed": SEED,
        "golden_coeffs": {str(k): v for k, v in GOLDEN.items()},
        "initial": {"score": init_score, "PIB": m_init["PIB"], "CV": m_init["CV"]},
        "gs": {"score": gs_score, "PIB": gs_pib, "CV": gs_cv},
        "spgd": {"score": spgd_score, "PIB": spgd_pib, "CV": spgd_cv},
        "iterative": {
            "score": final_score,
            "PIB": m_final["PIB"],
            "CV": m_final["CV"],
            "n_outer_iters": res["n_outer_iters"],
            "converged": res["converged"],
            "score_history": res["score_history"],
            "zernike_coeffs": {str(k): v for k, v in res["zernike_coeffs"].items()},
        },
        "improvement_vs_gs_pct": improvement_vs_gs,
        "improvement_vs_spgd_pct": improvement_vs_spgd,
        "improvement_vs_init_pct": improvement_vs_init,
        "zernike_ablation": {
            "zernike_on_score": z_score,
            "zernike_off_score": final_score,
            "note": "Zernike calibration pass is a negative result: it degrades the score.",
        },
    }
    (out / "data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))

    # Save arrays
    np.save(out / "actual_far_field.npy", actual_ff)
    np.save(out / "target.npy", target)
    np.save(out / "final_far_field.npy", res["far_field"])
    np.save(out / "slm_phase.npy", res["slm_phase"])
    np.save(out / "init_far_field.npy", ff_init)
    np.save(out / "gs_phase.npy", gs_phase)
    np.save(out / "spgd_phase.npy", spgd_phase)

    elapsed = (datetime.now() - t0).total_seconds()
    logger.info("Report generated in {:.1f}s. Output: {}", elapsed, str(out))
    logger.info("Initial:   score={:.4f}", init_score)
    logger.info("GS:        score={:.4f}", gs_score)
    logger.info("SPGD:      score={:.4f}", spgd_score)
    logger.info(
        "Iterative: score={:.4f} ({:+.1f}% vs SPGD, {:+.1f}% vs GS, {:+.1f}% vs initial)",
        final_score, improvement_vs_spgd, improvement_vs_gs, improvement_vs_init,
    )
    logger.info(
        "Figures: initial_vs_final.png, score_history.png, zernike_coeffs.png, "
        "phase_evolution.png, score_comparison.png"
    )


if __name__ == "__main__":
    main()
