"""Run the closed-loop SLM far-field beam-shaping simulation bench and emit a
markdown report with per-method metrics and figures.

This script drives the simulated 2f-Fourier SLM shaping bench to compare
representative beam-shaping
methods from the literature on one identical optical model and one identical
target. It is a *report generator* (lives in ``scripts/`` per repo rule) and
writes its markdown + figures to ``docs/beam_shaping/papers/``.

Run:
    source .venv/bin/activate   # python 3.13 + torch
    python scripts/generate_beam_shaping_papers_report.py

Outputs (under docs/beam_shaping/papers/):
    figures/<method>_<stamp>.png     far-field intensity per method
    figures/target_<stamp>.png       the target pattern
    beam_shaping_papers.md           this report (metric table + links)
    results_<stamp>.json             all metrics as JSON (for downstream reuse)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# sys.path bootstrap (repo rule)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "libs"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # must be set BEFORE pyplot (repo-wide rule)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import rcParams  # noqa: E402

rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

from loguru import logger  # noqa: E402

from ao_shaping.drivers.sim.slm_shaping_bench import (  # noqa: E402
    ShapingBenchConfig,
    compute_metrics,
    forward_intensity,
    make_target,
)
from ao_shaping.optimizer.wfless.slm_shaping_bench import (  # noqa: E402
    analytic_amplitude_target,
    differentiable_shape,
    gs_shape,
    spgd_shape,
)

OUT_DIR = ROOT / "docs" / "beam_shaping" / "papers"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

STAMP = time.strftime("%Y%m%d_%H%M%S")

CONFIG = ShapingBenchConfig(
    n_grid=128,
    aperture_size=12e-3,
    wavelength=532e-9,
    focal_length=0.125,
    target_side_px=32,
    target_kind="square",
    seed=0,
)


def save_figure(intensity: np.ndarray, title: str, name: str) -> str:
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(intensity, cmap="inferno")
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    path = FIG_DIR / f"{name}_{STAMP}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def main() -> int:
    target = make_target(CONFIG)
    save_figure(target, "Target (square, 32 px)", "target")

    rows: list[dict] = []
    results: dict[str, dict] = {}

    def record(method: str, res) -> None:
        inten = res.intensity
        inten = inten / (inten.max() + 1e-12)
        center = np.unravel_index(np.argmax(inten), inten.shape)[::-1]
        m = compute_metrics(inten, target, center=center)
        fig = save_figure(inten, res.method, res.method)
        rows.append(
            {
                "method": res.method,
                "PIB": m["PIB"],
                "efficiency": m["efficiency"],
                "CV": m["CV"],
                "Strehl": m["Strehl"],
                "zero_order": m["zero_order"],
                "n_iters": res.n_iters,
                "figure": fig,
            }
        )
        results[res.method] = {
            "metrics": m,
            "n_iters": res.n_iters,
            "figure": fig,
            "history": res.history[-5:],
        }
        logger.info(
            "  {:<28} PIB={:.3f}  CV={:.3f}  Strehl={:.3f}  zero_order={:.3f}",
            res.method,
            m["PIB"],
            m["CV"],
            m["Strehl"],
            m["zero_order"],
        )

    logger.info("=== SLM far-field beam-shaping bench (2f Fourier) ===")

    # --- baselines ----------------------------------------------------------
    record("analytic_amplitude_target", analytic_amplitude_target(CONFIG))
    logger.info("[baseline] unshaped (zero phase):")
    raw = forward_intensity(np.zeros((CONFIG.n_grid, CONFIG.n_grid)), CONFIG)
    raw = raw / (raw.max() + 1e-12)
    c = np.unravel_index(np.argmax(raw), raw.shape)[::-1]
    m0 = compute_metrics(raw, target, center=c)
    save_figure(raw, "Unshaped (zero phase)", "unshaped")
    rows.append(
        {
            "method": "unshaped_zero_phase",
            "PIB": m0["PIB"],
            "efficiency": m0["efficiency"],
            "CV": m0["CV"],
            "Strehl": m0["Strehl"],
            "zero_order": m0["zero_order"],
            "n_iters": 0,
            "figure": str(FIG_DIR / f"unshaped_{STAMP}.png"),
        }
    )
    logger.info(
        "  {:<28} PIB={:.3f}  CV={:.3f}  Strehl={:.3f}  zero_order={:.3f}",
        "unshaped_zero_phase",
        m0["PIB"],
        m0["CV"],
        m0["Strehl"],
        m0["zero_order"],
    )

    # --- GS -----------------------------------------------------------------
    t0 = time.time()
    gs = gs_shape(CONFIG, n_iters=150, verbose=False)
    record("gs_shape", gs)
    logger.info("gs elapsed: {:.1f}s", time.time() - t0)

    # --- differentiable (torch) -------------------------------------------
    try:
        t0 = time.time()
        df = differentiable_shape(CONFIG, n_iters=200, lr=0.05)
        record("differentiable_shape", df)
        logger.info("diff elapsed: {:.1f}s", time.time() - t0)
    except Exception as e:  # noqa: BLE001
        logger.warning("differentiable_shape failed: {}", e)

    # --- SPGD ---------------------------------------------------------------
    t0 = time.time()
    sp = spgd_shape(CONFIG, n_iters=400, dim=8)
    record("spgd_freeform", sp)
    logger.info("spgd elapsed: {:.1f}s", time.time() - t0)

    # --- persist JSON -------------------------------------------------------
    with open(OUT_DIR / f"results_{STAMP}.json", "w") as fh:
        json.dump({"config": vars(CONFIG), "results": results}, fh, indent=2, default=float)

    # --- markdown report ----------------------------------------------------
    md = [
        "# 闭环 SLM 远场光斑整形 — 仿真基准报告",
        "",
        f"> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        "> 模型: 2f-Fourier 单相位 SLM (高斯入射 × 相位板 → 透镜 + 传播 → 焦平面)  ",
        f"> 配置: n_grid={CONFIG.n_grid}, aperture={CONFIG.aperture_size*1e3:.0f}mm, "
        f"λ={CONFIG.wavelength*1e9:.0f}nm, f={CONFIG.focal_length*1e3:.0f}mm, "
        f"target={CONFIG.target_kind} {CONFIG.target_side_px}px",
        "",
        "## 方法对比",
        "",
        "| 方法 | PIB | efficiency | CV (越小越好) | Strehl | zero_order | 迭代数 |",
        "|------|-----|-----------|--------------|--------|-----------|--------|",
    ]
    for r in rows:
        md.append(
            f"| {r['method']} | {r['PIB']:.3f} | {r['efficiency']:.3f} | "
            f"{r['CV']:.3f} | {r['Strehl']:.3f} | {r['zero_order']:.3f} | {r['n_iters']} |"
        )
    md += ["", "## 图像", ""]
    for r in rows:
        rel = Path(r["figure"]).name
        md.append(f"- **{r['method']}** — ![](figures/{rel})")
    md += ["", "## 说明", ""]
    md += [
        "- **PIB** (power-in-bucket): 目标区域内能量占比，越高越好。",
        "- **CV** (均匀性, 目标区内强度系数变异): 越低越好。",
        "- **zero_order**: 中心零级(未衍射)占比；相位型 SLM 会保留零级，故非零。",
        "- **Strehl**: 归一化重叠（余弦相似度），越接近 1 越匹配目标。",
        "- `analytic_amplitude_target` 是纯振幅基线（理想上界）；`unshaped_zero_phase` 是无整形基线。",
        "",
        "> 本报告由 `scripts/generate_beam_shaping_papers_report.py` 生成，仿真基于 "
        "`ao_shaping.drivers.sim.slm_shaping_bench` 的前向模型和 "
        "`ao_shaping.optimizer.wfless.slm_shaping_bench` 的优化方法。",
    ]
    (OUT_DIR / "beam_shaping_papers.md").write_text("\n".join(md), encoding="utf-8")
    logger.info("Report written: {}", OUT_DIR / "beam_shaping_papers.md")
    logger.info("Figures in: {}", FIG_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
