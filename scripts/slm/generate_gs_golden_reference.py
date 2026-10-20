"""Generate the GS golden-reference comparator report.

**What a "golden reference" is here.** The GS square phase, computed by the
canonical :func:`ao_shaping.algorithm.signal_processing.square_shaping.gs_shape`
in pure simulation, is the **upper bound** a phase-only SLM could reach *if the
model were exact*. Hardware runs can then be read against it: how far a real
bench sits below the ceiling.

**What it is not.** It is **a comparator, not a baseline.** ``gs_golden_phase``
is documented as such, and the repo records the hardware evidence: GS has
repeatedly *lost to a flat phase* on the real bench
(``report/shape_trials/README.md``: "候选未能战胜平相", the SLM was reverted to
flat), and the inverse-shaping package notes GS alone was negative in 90/90 runs.
So a GS number beating flat **in this simulation** is a statement about the
simulation, where the forward model is exact by construction -- it is not a
hardware expectation, and the report says so in its own conclusion rather than
leaving the reader to infer it from a green number.

**Why a ceiling and not another arm in a benchmark.** Because the failure mode it
guards against is *model error*: a hardware candidate that looks good against a
flat reference can still be far from what the optics allow, and the only way to
tell those apart is an absolute reference computed from the model. The ceiling
also makes the GS iteration budget a measured quantity (§2) instead of a guess.

**No GS re-implementation.** The phase comes from
:func:`ml.shaping.gs_reference.gs_golden_phase`, a torch-free leaf that delegates
to the canonical ``gs_shape``; the far field comes from the canonical
``forward_intensity``; every metric comes from
:mod:`ao_shaping.utils.image.beam_metrics`, including the Strehl/M² APIs added
alongside it. Nothing optical is computed twice in this script.

**Fully offline.** Simulated bench only; no camera, no SLM.

Run:
    python scripts/slm/generate_gs_golden_reference.py
    python scripts/slm/generate_gs_golden_reference.py --n-grid 128 --quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts._common import fmt_metric, markdown_table, savefig  # noqa: E402
from scripts._common.provenance import insert_header  # noqa: E402

from ao_shaping.algorithm.signal_processing.square_shaping import (  # noqa: E402
    SquareShapingConfig,
)
from ao_shaping.utils.image.beam_metrics import (  # noqa: E402
    compute_quality_score,
    compute_square_metrics,
    ideal_peak_from_pupil,
    m2_reference_relative,
    strehl_ratio,
)
from ml.shaping.gs_reference import gs_golden_phase  # noqa: E402

REPORT_KEY = "report/shape_trials/gs_golden_reference.md"
OUT_DIR = ROOT / "report" / "shape_trials"
FIG_DIR = OUT_DIR / "figures"


def _forward(phase: np.ndarray, cfg: SquareShapingConfig) -> np.ndarray:
    """Far-field intensity of a pupil phase, via the canonical bench model.

    Imported lazily: ``slm_shaping_bench`` registers camera backends at import
    time, and a report generator has no business doing that at module import.
    """
    from ao_shaping.drivers.sim.slm_shaping_bench import forward_intensity

    return np.asarray(forward_intensity(phase, cfg), dtype=np.float64)


def _score(intensity: np.ndarray, cfg: SquareShapingConfig, center, ideal_peak: float) -> dict:
    """Canonical shape metrics plus the Strehl / relative-M² APIs."""
    metrics = compute_square_metrics(intensity, cfg.target_side_px, center)
    metrics["quality_score"] = compute_quality_score(metrics)
    metrics["strehl"] = strehl_ratio(intensity, ideal_peak)
    return metrics


def _zero_order_center(intensity: np.ndarray) -> tuple[int, int]:
    """Locate the 0-order by ``argmax``, never by geometry.

    The repo rule is explicit: in the 2f Fourier bench the optical axis is the
    frame's global maximum and lands at the geometric centre only by luck.
    """
    row, col = np.unravel_index(int(np.argmax(intensity)), intensity.shape)
    return int(col), int(row)


def _crop(intensity: np.ndarray, center, side: int) -> np.ndarray:
    """Centre crop used for the figure so the spot is legible, not the whole grid."""
    cx, cy = center
    half = max(side // 2, 1)
    return intensity[
        max(cy - half, 0) : cy + half, max(cx - half, 0) : cx + half
    ]


def _figure(
    flat_intensity: np.ndarray,
    gs_intensity: np.ndarray,
    target: np.ndarray,
    sweep: list[dict],
    center,
    crop: int,
    out_path: Path,
) -> Path:
    """Far fields (flat / GS / target), the GS convergence curve, and Strehl."""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(12, 7.5))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.25, 1])

    panels = [
        (flat_intensity, "平相位 (flat)"),
        (gs_intensity, "GS 金标 (模拟上界)"),
        (target, "方形目标"),
    ]
    for index, (image, title) in enumerate(panels):
        ax = fig.add_subplot(grid[0, index])
        view = _crop(image, center, crop)
        ax.imshow(np.log10(np.maximum(view, view.max() * 1e-6)), cmap="inferno")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    ax_score = fig.add_subplot(grid[1, 0])
    ax_score.semilogx(
        [row["n_iters"] for row in sweep],
        [row["quality_score"] for row in sweep],
        "o-",
        label="GS 质量分",
        color="#F58518",
    )
    flat_score = sweep[0]["flat_quality_score"]
    ax_score.axhline(flat_score, linestyle="--", color="#4C78A8", label="平相位")
    ax_score.set_xlabel("GS 迭代次数")
    ax_score.set_ylabel("质量分 (canonical)")
    ax_score.set_title("GS 收敛 vs 平相位", fontsize=10)
    ax_score.grid(alpha=0.3)
    ax_score.legend(fontsize=8)

    ax_cv = fig.add_subplot(grid[1, 1])
    ax_cv.semilogx(
        [row["n_iters"] for row in sweep],
        [row["uniformity_cv"] for row in sweep],
        "o-",
        label="GS 箱内 CV",
        color="#E45756",
    )
    ax_cv.axhline(
        sweep[0]["flat_uniformity_cv"],
        linestyle="--",
        color="#4C78A8",
        label="平相位",
    )
    ax_cv.set_xlabel("GS 迭代次数")
    ax_cv.set_ylabel("箱内 CV (越低越均匀)")
    ax_cv.set_title("均匀度 (未加能量项)", fontsize=10)
    ax_cv.grid(alpha=0.3)
    ax_cv.legend(fontsize=8)

    ax_strehl = fig.add_subplot(grid[1, 2])
    ax_strehl.semilogx(
        [row["n_iters"] for row in sweep],
        [row["strehl"] for row in sweep],
        "o-",
        label="GS Strehl",
        color="#54A24B",
    )
    ax_strehl.axhline(
        sweep[0]["flat_strehl"], linestyle="--", color="#4C78A8", label="平相位"
    )
    ax_strehl.set_xlabel("GS 迭代次数")
    ax_strehl.set_ylabel("Strehl (峰值比)")
    ax_strehl.set_ylim(-0.03, 1.05)
    ax_strehl.set_title("光学 Strehl", fontsize=10)
    ax_strehl.grid(alpha=0.3)
    ax_strehl.legend(fontsize=8)

    fig.suptitle("GS 金标：模拟上界与平相位对照 (无设备)", fontsize=12)
    fig.tight_layout()
    savefig(fig, out_path)
    return out_path


def _body(
    sweep: list[dict],
    flat: dict,
    best: dict,
    cfg: SquareShapingConfig,
    seed: int,
) -> str:
    """Render the report body (the provenance header is stamped at write time)."""
    lines: list[str] = []
    lines.append("# GS 金标参照：模拟上界与平相位对照")
    lines.append("")
    lines.append(
        "**这是什么**：把 GS 方形相位当作**上界**——即「若正向模型完全准确，"
        "纯相位 SLM 最多能达到什么」。硬件运行可以据此读出自己离光学允许的上限还有多远。"
    )
    lines.append("")
    lines.append(
        f"**台架参数**：`n_grid={cfg.n_grid}`、`aperture={cfg.aperture_size * 1e3:.1f} mm`、"
        f"`λ={cfg.wavelength * 1e9:.0f} nm`、`f={cfg.focal_length * 1e3:.0f} mm`、"
        f"`target_side={cfg.target_side_px} px`（远场）、`padding={cfg.far_field_padding}`、"
        f"`seed={seed}`。全部为仿真，无设备。"
    )
    lines.append("")

    lines.append("## 1. 对照结论（数据给出，不是断言）")
    lines.append("")
    beat = best["quality_score"] > flat["quality_score"]
    lines.append(
        markdown_table(
            ["臂", "质量分 (canonical)", "箱内 CV", "箱内能量 EE", "Strehl", "远场 M²(相对)"],
            [
                [
                    "平相位 (flat)",
                    fmt_metric(flat["quality_score"]),
                    fmt_metric(flat["uniformity_cv"]),
                    fmt_metric(flat["encircled_energy"]),
                    fmt_metric(flat["strehl"]),
                    fmt_metric(flat["m2_relative"]),
                ],
                [
                    f"GS 金标 (n_iters={best['n_iters']})",
                    fmt_metric(best["quality_score"]),
                    fmt_metric(best["uniformity_cv"]),
                    fmt_metric(best["encircled_energy"]),
                    fmt_metric(best["strehl"]),
                    fmt_metric(best["m2_relative"]),
                ],
            ],
        )
    )
    lines.append("")
    if beat:
        lines.append(
            f"在**本仿真**中 GS 胜出：质量分 {fmt_metric(best['quality_score'])} 对平相位的 "
            f"{fmt_metric(flat['quality_score'])}，箱内 CV 从 "
            f"{fmt_metric(flat['uniformity_cv'])} 降到 {fmt_metric(best['uniformity_cv'])}。"
        )
        lines.append(
            "> ⚠️ **这是关于仿真的陈述，不是硬件预期。** 仿真里正向模型按构造是精确的，"
            "GS 因此几乎必然占优。硬件上仓库记录的结论相反：GS 反复**输给平相位**"
            "（`report/shape_trials/README.md`：候选未能战胜平相，SLM 被退回平相位），"
            "逆向整形包也记录 GS 单独使用在 90/90 次中为负。"
            "**本表的上界不可当作硬件目标。**"
        )
    else:
        lines.append(
            "在**本仿真**中 GS **没有**胜过平相位。这是与多数 GS 叙事相反的结果，"
            "因此必须照实记录：本台架参数下 GS 不是有效上界，"
            "任何引用「GS 上界」的结论都应先复核 §1 的参数。"
        )
    lines.append("")

    lines.append("## 2. GS 收敛：迭代预算不是猜的")
    lines.append("")
    lines.append(
        markdown_table(
            ["n_iters", "质量分", "箱内 CV", "箱内能量 EE", "Strehl", "相对 M²", "相位范围 (rad, 未包裹)"],
            [
                [
                    str(row["n_iters"]),
                    fmt_metric(row["quality_score"]),
                    fmt_metric(row["uniformity_cv"]),
                    fmt_metric(row["encircled_energy"]),
                    fmt_metric(row["strehl"]),
                    fmt_metric(row["m2_relative"]),
                    f"[{row['phase_min_rad']:.3f}, {row['phase_max_rad']:.3f}]",
                ]
                for row in sweep
            ],
        )
    )
    lines.append("")
    lines.append(
        f"平相位参考：质量分 {fmt_metric(flat['quality_score'])}、"
        f"CV {fmt_metric(flat['uniformity_cv'])}、EE {fmt_metric(flat['encircled_energy'])}、"
        f"Strehl {fmt_metric(flat['strehl'])}。"
        "**把这张表当成 GS 的收敛曲线读**：第一个越过平相位分数的迭代次数，"
        "就是这个台架上 GS 值得花的钱；再多迭代只买仿真里的边际改善。"
    )
    lines.append("")

    lines.append("## 3. 口径与本报告不宣称什么")
    lines.append("")
    lines.append(
        "1. **0 级由 `argmax` 定位，不用几何中心。** 2f 傅里叶台架里光轴落在画面"
        "全局最大值处，几何中心只是碰巧重合；所有箱内指标都按定位到的中心计算。"
    )
    lines.append(
        "2. **箱内 CV 单独看会骗人。** 仓库已记录「只优化 `-CV` 会把目标箱掏空」"
        "（实测 EE 塌到 0.002），所以本报告始终与 EE、Strehl 并列，不给 CV 单独下结论。"
    )
    lines.append(
        "3. **相对 M² 不是绝对 M²。** 单个焦面无法定 M²，故报 BPP 比值"
        f"（`m2_reference_relative`），绝对值需要焦点栈。本表中平相位相对 M² = "
        f"{fmt_metric(flat['m2_relative'])}（它本身就是 BPP 参考）。"
    )
    lines.append(
        f"4. **GS 的均匀度是用峰值换来的，不要误读成像差。** 本次 GS 把箱内 CV 从 "
        f"{fmt_metric(flat['uniformity_cv'])} 压到 {fmt_metric(best['uniformity_cv'])}，"
        f"代价是 Strehl 从 {fmt_metric(flat['strehl'])} 掉到 "
        f"{fmt_metric(best['strehl'])}、相对 M² 升到 {fmt_metric(best['m2_relative'])}。"
        "这三个数**同向**是平坦化变换的正常表现，不是相位误差的表现："
        "要把一个高斯焦斑摊成平顶方块，中心峰值必然下降、二阶矩半径必然变大。"
        "所以**不能**用 Strehl 下降来判定 GS 失败，也不能用它来判定像差——"
        "在这张表里 Strehl 衡量的是「有多少能量离开了 0 级峰值」，"
        "而不是波前质量；要评波前质量必须另看相位本身（未包裹弧度的分布，见 §3.6）。"
    )
    lines.append(
        "5. **Strehl 用同一前向链路的理想峰值做分母**（`ideal_peak_from_pupil`，"
        "同样的零填充），所以它度量的是相位误差而不是数值归一化。"
    )
    lines.append(
        "6. **相位是未包裹弧度。** GS 返回 raw radians；唯一的 `mod 2π` 在 SLM 驱动"
        "（`Santec.create_phase_from_array`）里，本脚本不去碰它。"
    )
    lines.append("")

    lines.append("## 4. 复现")
    lines.append("")
    lines.append("```bash")
    lines.append("python scripts/slm/generate_gs_golden_reference.py")
    lines.append("```")
    lines.append("")
    lines.append(
        "GS 相位由 `ml/shaping/gs_reference.py::gs_golden_phase` 产出（委托给 canonical "
        "`gs_shape`，不复刻）；远场由 canonical `forward_intensity` 计算；"
        "全部指标来自 `ao_shaping.utils.image.beam_metrics`，脚本内不另算光学量。"
    )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GS golden-reference comparator report")
    parser.add_argument("--n-grid", type=int, default=64)
    parser.add_argument("--aperture-size", type=float, default=12e-3, help="m")
    parser.add_argument("--wavelength", type=float, default=1064e-9, help="m")
    parser.add_argument("--focal-length", type=float, default=0.125, help="m")
    parser.add_argument("--target-side", type=int, default=40, help="far-field px")
    parser.add_argument("--padding", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop", type=int, default=96, help="figure crop side, far-field px")
    parser.add_argument(
        "--iters",
        default="1,2,5,10,20,50,100,200",
        help="comma-separated GS iteration budgets",
    )
    parser.add_argument("--quick", action="store_true", help="smoke: 3 budgets, tiny grid")
    args = parser.parse_args()
    if args.quick:
        args.n_grid = min(args.n_grid, 32)
        args.iters = "1,5,20"
        args.crop = min(args.crop, 48)
    budgets = [int(v) for v in args.iters.split(",") if v.strip()]
    if not budgets:
        parser.error("--iters produced no budgets")
    if any(b < 1 for b in budgets):
        parser.error("--iters values must all be >= 1")
    args.budgets = sorted(dict.fromkeys(budgets))
    return args


def main() -> int:
    """Run the comparison and write the report."""
    args = _parse_args()
    cfg = SquareShapingConfig(
        n_grid=args.n_grid,
        aperture_size=args.aperture_size,
        wavelength=args.wavelength,
        focal_length=args.focal_length,
        target_side_px=args.target_side,
        far_field_padding=args.padding,
        seed=args.seed,
    )

    # Diffraction-limited reference: the same Gaussian pupil with no phase error.
    ideal_peak = ideal_peak_from_pupil(
        cfg.make_gaussian_pupil(), far_field_padding=float(cfg.far_field_padding)
    )
    flat_phase = np.zeros((cfg.n_grid, cfg.n_grid))
    flat_intensity = _forward(flat_phase, cfg)
    center = _zero_order_center(flat_intensity)
    flat = _score(flat_intensity, cfg, center, ideal_peak)
    # The flat arm *is* the BPP reference, so its relative M2 is 1.0 by
    # definition. Computing it against itself would be a vacuous call dressed up
    # as a measurement; assigning the identity says what it means.
    flat["m2_relative"] = 1.0
    # BPP reference field: the same flat far field, amplitude-rooted.
    flat_amplitude = np.sqrt(np.maximum(flat_intensity, 0.0))
    logger.info(
        "flat: score {:.4f} cv {:.4f} ee {:.4f} strehl {:.4f} center {}",
        flat["quality_score"],
        flat["uniformity_cv"],
        flat["encircled_energy"],
        flat["strehl"],
        center,
    )

    from ao_shaping.algorithm.signal_processing.square_shaping import make_square_target

    target = make_square_target(cfg)

    sweep: list[dict] = []
    best: dict | None = None
    best_intensity = flat_intensity
    for n_iters in args.budgets:
        phase = gs_golden_phase(cfg, n_iters, args.seed)
        intensity = _forward(phase, cfg)
        row = _score(intensity, cfg, center, ideal_peak)
        row["m2_relative"] = m2_reference_relative(
            np.sqrt(np.maximum(intensity, 0.0)),
            flat_amplitude,
            wavelength_m=cfg.wavelength,
            pixel_pitch_m=cfg.far_field_pixel_size,
        )
        row["n_iters"] = n_iters
        row["phase_min_rad"] = float(np.min(phase))
        row["phase_max_rad"] = float(np.max(phase))
        row["flat_quality_score"] = flat["quality_score"]
        row["flat_uniformity_cv"] = flat["uniformity_cv"]
        row["flat_strehl"] = flat["strehl"]
        sweep.append(row)
        if best is None or row["quality_score"] > best["quality_score"]:
            best = row
            best_intensity = intensity
        logger.info(
            "gs n_iters={}: score {:.4f} cv {:.4f} ee {:.4f} strehl {:.4f} phase=[{:.3f}, {:.3f}] rad",
            n_iters,
            row["quality_score"],
            row["uniformity_cv"],
            row["encircled_energy"],
            row["strehl"],
            row["phase_min_rad"],
            row["phase_max_rad"],
        )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figure_name = "gs_golden_reference.png"
    _figure(
        flat_intensity,
        best_intensity,
        target,
        sweep,
        center,
        args.crop,
        FIG_DIR / figure_name,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    body = _body(sweep, flat, best, cfg, args.seed)
    body = body.replace(
        "## 1. 对照结论（数据给出，不是断言）",
        f"![GS 金标与平相位对照](figures/{figure_name})\n\n"
        "## 1. 对照结论（数据给出，不是断言）",
        1,
    )
    (OUT_DIR / "gs_golden_reference.md").write_text(
        insert_header(body, REPORT_KEY), encoding="utf-8"
    )

    logger.info(
        "wrote {} (+ figures/{}) -- best GS score {:.4f} vs flat {:.4f} at n_iters={}",
        REPORT_KEY,
        figure_name,
        best["quality_score"],
        flat["quality_score"],
        best["n_iters"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())