"""GS 方扫报告的**出图**。

八个 ``fig_*`` 各出一张 PNG，外加一个把它们串起来的 :func:`render_figures`。

⚠️ 仓库级规则：``matplotlib.use("Agg")`` 必须在 ``import matplotlib.pyplot``
之前。本模块自己带这句（而不是指望调用方先 import 过），因为图是在
``--check`` 的只读重渲染里也要算的，那时不能有交互式后端。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
import numpy as np

import matplotlib

matplotlib.use("Agg")  # 必须在 import matplotlib.pyplot 之前

import matplotlib.pyplot as plt

# 与 facade 同在 scripts/slm/ 下，parents[2] 算出的仓库根一致。
# 用**逐个 insert** 而不是 for 循环：scripts/ 那层的守卫
# (tests/ao_shaping/scripts/test_script_file_form_imports.py) 靠解析
# `sys.path.insert` 的字面参数判断「这个脚本能不能以文件方式启动」，
# for 循环里的循环变量它解析不了（会整文件跳过，等于没检查）。
# src 在 scripts 之前：scripts/ml 会遮蔽 src/ml。
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from _common import savefig


from scripts.slm._hw_gs_square_scans_figbase import (
    _annotate_cell,
    _check_frame_params,
    _crop_box,
    batch_tag,
    load_frame_mean,
    locate_frames,
    residual_figure_name,
    setup_matplotlib,
)
from scripts.slm._hw_gs_square_scans_fmt import (
    _row_at,
)
from scripts.slm._hw_gs_square_scans_metrics import (
    flag_rows,
)
from scripts.slm._hw_gs_square_scans_model import (
    BASE_CMAP,
    CROP_FACTOR,
    CellPick,
    FIG_TAG,
    Findings,
    PATHS,
    PEAK_MATCH_ATOL,
    RESIDUAL_CMAP,
    RESIDUAL_VLIM,
    ResidualSet,
    RetestFindings,
    RetestRun,
    Run,
    SATURATION_COUNTS,
)

def fig_sides_objectives(run: Run, f: Findings) -> Path:
    setup_matplotlib()
    x = f.sides_x
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.6))

    ax1.plot(x, f.pib, "o-", color="#1f77b4", label="PIB")
    ax1.set_xlabel("目标方框边长 (camera px)")
    ax1.set_ylabel("PIB (框内功率占比)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(alpha=0.3)
    ax1b = ax1.twinx()
    ax1b.plot(x, f.cv, "s--", color="#d62728", label="CV")
    ax1b.set_ylabel("CV (框内变异系数)", color="#d62728")
    ax1b.tick_params(axis="y", labelcolor="#d62728")
    lines = ax1.get_lines() + ax1b.get_lines()
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="upper left", fontsize=9)
    ax1.set_title("(a) PIB 与 CV 同向上升, 但两者在 score 里正好相反")

    ax2.plot(x, f.score, "o-", color="#2ca02c", label="score")
    peak_x = x[f.score_peak_index]
    ax2.plot([peak_x], [f.score[f.score_peak_index]], "k*", ms=14,
             label=f"峰值 {peak_x:.0f} px")
    ax2.axvline(f.selected_side, color="gray", ls=":", lw=1)
    ax2.annotate(f"选定 {f.selected_side} px\n(近 1000 窗口取到上界)",
                 xy=(f.selected_side, f.score[0]), xytext=(f.selected_side - 520, f.score[0] + 0.03),
                 fontsize=8, color="gray",
                 arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    ax2.set_xlabel("目标方框边长 (camera px)")
    ax2.set_ylabel("score = 0.5·PIB + 0.5/(1+CV)")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9, loc="lower right")
    ax2.set_title("(b) 复合分数在 1500 px 见顶, 之后被 CV 拖下来")
    fig.tight_layout()
    path = PATHS.fig_dir / f"fig1_sides_objectives{FIG_TAG}.png"
    savefig(fig, path)
    return path

def fig_sides_exposure(run: Run, f: Findings) -> Path:
    setup_matplotlib()
    sides = run.stages["sides"]
    x = f.sides_x
    frame_peak = sides.col("frame_peak")
    box_peak = sides.col("box_peak")
    fwhm = sides.col("global_spot_fwhm_px")
    marks = flag_rows(run, sides)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    ax1.plot(x, frame_peak, "o-", color="#1f77b4", label="frame_peak (整帧最大)")
    ax1.plot(x, box_peak, "s--", color="#ff7f0e", label="box_peak (框内最大)")
    ax1.axhline(SATURATION_COUNTS, color="#d62728", ls=":", lw=1.2,
                label=f"8 bit 满量程 {SATURATION_COUNTS:.0f} counts")
    for i, mark in enumerate(marks):
        if "†" in mark or "‡" in mark:
            ax1.plot([x[i]], [frame_peak[i]], "x", color="k", ms=7)
    ax1.set_xlabel("目标方框边长 (camera px)")
    ax1.set_ylabel("counts")
    ax1.set_title("(a) 只有 100 px 那一行顶到满量程; 其余 17 行都有余量")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, loc="center right")

    finite = [(xi, v) for xi, v in zip(x, fwhm) if math.isfinite(v)]
    ax2.plot([p[0] for p in finite], [p[1] for p in finite], "o-", color="#9467bd",
             label="global_spot_fwhm (整帧最亮结构)")
    median_fwhm = float(np.median([q for _, q in finite]))
    for xi, v in zip(x, fwhm):
        if math.isfinite(v) and v > 3.0 * median_fwhm:
            ax2.annotate(f"{v:.1f} px", xy=(xi, v), xytext=(xi + 40, v + 2), fontsize=8)
    ax2.set_xlabel("目标方框边长 (camera px)")
    ax2.set_ylabel("FWHM (camera px)")
    ax2.set_title("(b) 该列量的是 0 级/晕斑, 不是方斑")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    path = PATHS.fig_dir / f"fig2_sides_exposure_geometry{FIG_TAG}.png"
    savefig(fig, path)
    return path

def fig_zernike(run: Run, f: Findings) -> Path:
    setup_matplotlib()
    zern = run.stages["zernike"]
    x = zern.x
    explained_pct = [v * 100.0 for v in f.zern_explained]
    marks = flag_rows(run, zern)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    ax1.plot(x, explained_pct, "o-", color="#1f77b4", label="解释方差占比 (%)")
    ax1.set_xlabel("截断阶数 n_max")
    ax1.set_ylabel("zernike_explained_fraction (%)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(alpha=0.3)
    ax1b = ax1.twinx()
    ax1b.plot(x, f.zern_residual, "s--", color="#d62728", label="残差 RMS (rad)")
    ax1b.set_ylabel("zernike_rms_residual (rad)", color="#d62728")
    ax1b.tick_params(axis="y", labelcolor="#d62728")
    ax1b.set_ylim(min(f.zern_residual) - 0.01, max(f.zern_residual) + 0.01)
    ax1.set_title("(a) 496 个模式只解释 3% 的相位方差; 残差几乎不动")

    ax2.plot(x, f.zern_cv, "o-", color="#2ca02c", label="CV")
    cluster = [c for c in f.zern_clean_cv]
    ax2.set_ylim(min(cluster) - 0.15 * (max(cluster) - min(cluster)),
                 max(cluster) + 0.6 * (max(cluster) - min(cluster)))
    for i, mark in enumerate(marks):
        if "†" in mark:
            ax2.plot([x[i]], [f.zern_cv[i]], "o", mfc="none", mec="k", ms=12)
        if "‡" in mark or "※" in mark:
            ax2.annotate(f"n_max={x[i]:.0f}\nCV={f.zern_cv[i]:.2f} ({mark})\n超出纵轴",
                         xy=(x[i], ax2.get_ylim()[1]), xytext=(x[i] + 1.5, ax2.get_ylim()[1] - 0.55),
                         fontsize=8, color="#d62728", ha="left",
                         arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.9))
    ax2.set_xlabel("截断阶数 n_max")
    ax2.set_ylabel("CV", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax2.grid(alpha=0.3)
    ax2b = ax2.twinx()
    ax2b.plot(x, f.zern_score, "^:", color="#7f7f7f", lw=1, label="score")
    ax2b.set_ylabel("score", color="#7f7f7f")
    ax2b.tick_params(axis="y", labelcolor="#7f7f7f")
    ax2.set_title("(b) 干净行 CV 只从 7.30 摆到 7.70; 空圈 = settle 超时")
    fig.tight_layout()
    path = PATHS.fig_dir / f"fig3_zernike_reconstruction{FIG_TAG}.png"
    savefig(fig, path)
    return path

def fig_blaze(run: Run, f: Findings) -> Path:
    setup_matplotlib()
    blaze = run.stages["blaze"]
    x = blaze.x
    pred = blaze.col("predicted_shift_px")
    meas = blaze.col("measured_shift_y_px")
    marks = flag_rows(run, blaze)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    ax1.plot(x, pred, "o-", color="#1f77b4", label="predicted_shift (模型)")
    dark = [i for i, mark in enumerate(marks) if "‡" in mark]
    keep = [i for i in range(len(x)) if i not in dark]
    ax1.plot([x[i] for i in keep], [meas[i] for i in keep], "s--", color="#ff7f0e",
             label="measured_shift_y (框内质心)")
    for i in dark:
        ax1.plot([x[i]], [meas[i]], "x", color="#d62728", ms=11, mew=2,
                 label="黑帧行的实测值 (不计)")
        ax1.annotate("黑帧伪迹\n不当数据点", xy=(x[i], meas[i]),
                     xytext=(x[i] + 8, meas[i] - 55), fontsize=8, color="#d62728",
                     arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8))
    for i, mark in enumerate(marks):
        if not mark or i in dark:
            continue
        ax1.plot([x[i]], [meas[i]], "o", mfc="none", mec="#d62728", ms=13, mew=1.4)
        ax1.annotate(f"{mark}", xy=(x[i], meas[i]), xytext=(x[i] + 6, meas[i] - 42),
                     fontsize=11, color="#d62728")
    ax1.set_xlabel("闪耀光栅周期 (SLM px)")
    ax1.set_ylabel("位移 (camera px)")
    ax1.set_title("(a) 实测位移系统性小于模型; 周期 16 让框出画 (§)")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, loc="upper right")

    excluded = {i for i, mark in enumerate(marks) if "§" in mark or "‡" in mark}
    keep = [i for i in range(len(x)) if i not in excluded]
    ax2.plot([x[i] for i in keep], [f.blaze_score[i] for i in keep], "o-",
             color="#2ca02c", label="score (干净行)")
    for i in sorted(excluded):
        ax2.plot([x[i]], [f.blaze_score[i]], "x", color="#d62728", ms=9,
                 label="排除行 (§ / ‡)")
        ax2.annotate(f"{marks[i]} score {f.blaze_score[i]:.3f}",
                     xy=(x[i], f.blaze_score[i]), xytext=(x[i] + 4, f.blaze_score[i] - 0.03),
                     fontsize=8, color="#d62728")
    ax2.set_xlabel("闪耀光栅周期 (SLM px)")
    ax2.set_ylabel("score", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax2.grid(alpha=0.3)
    ax2.set_title(f"(b) 去掉 {len(excluded)} 行后 score 随周期单调上升")
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles[:2], labels[:2], fontsize=8, loc="lower right")
    fig.tight_layout()
    path = PATHS.fig_dir / f"fig4_blaze_shift{FIG_TAG}.png"
    savefig(fig, path)
    return path

def fig_retest_neighbors(run: Run, retest: RetestRun, rf: RetestFindings) -> Path:
    """复测邻居比较: 左列 CV、右列 score, 上排光栅、下排 Zernike。"""
    setup_matplotlib()
    base_blaze = run.stages["blaze"]
    base_zern = run.stages["zernike"]
    r_blaze = retest.stages["blaze"]
    r_zern = retest.stages["zernike"]

    b_p = [float(r["blaze_period_px"]) for r in base_blaze.rows]
    b_cv = base_blaze.col("cv")
    b_sc = base_blaze.col("score")
    b_n = [float(r["n_max"]) for r in base_zern.rows]
    bz_cv = base_zern.col("cv")
    bz_sc = base_zern.col("score")
    r_p = [float(r["blaze_period_px"]) for r in r_blaze.rows]
    r_n = [float(r["n_max"]) for r in r_zern.rows]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
    (ax_cv_p, ax_sc_p), (ax_cv_n, ax_sc_n) = axes

    def panel(ax, xs, ys, label, **kw):
        ax.plot(xs, ys, label=label, **kw)
        ax.grid(alpha=0.3)

    # (a) 光栅 CV
    panel(ax_cv_p, b_p, b_cv, "基线 (20 ms)", color="#7f7f7f", ls="--", marker="o", ms=5)
    panel(ax_cv_p, r_p, rf.blaze_cv, "复测 (180 ms)", color="#2ca02c", marker="s", ms=7)
    ax_cv_p.plot([40.0], [float(_row_at(base_blaze.rows, "blaze_period_px", 40.0)["cv"])], "x",
                 color="#d62728",
                 ms=12, mew=2, label="基线 40: 黑帧")
    ax_cv_p.set_yscale("log")
    ax_cv_p.set_xlabel("闪耀光栅周期 (SLM px)")
    ax_cv_p.set_ylabel("CV (对数轴)")
    ax_cv_p.set_title("(a) 基线 40 的 CV 尖峰在复测里消失")
    ax_cv_p.legend(fontsize=8)

    # (b) 光栅 score
    panel(ax_sc_p, b_p, b_sc, "基线 (20 ms)", color="#7f7f7f", ls="--", marker="o", ms=5)
    panel(ax_sc_p, r_p, [float(r["score"]) for r in r_blaze.rows], "复测 (180 ms)",
          color="#2ca02c", marker="s", ms=7)
    ax_sc_p.set_xlabel("闪耀光栅周期 (SLM px)")
    ax_sc_p.set_ylabel("score")
    r_score_p = [float(r["score"]) for r in r_blaze.rows]
    mono_p = all(r_score_p[i] > r_score_p[i - 1] for i in range(1, len(r_score_p)))
    ax_sc_p.set_title(
        "(b) score: 复测 32/40/48 " + ("单调上升" if mono_p else "非单调（无尖峰）"))
    ax_sc_p.legend(fontsize=8, loc="lower right")

    # (c) Zernike CV
    panel(ax_cv_n, b_n, bz_cv, "基线 (20 ms)", color="#7f7f7f", ls="--", marker="o", ms=4)
    panel(ax_cv_n, r_n, rf.zern_cv, "复测 (180 ms)", color="#2ca02c", marker="s", ms=7)
    ax_cv_n.plot([18.0], [float(_row_at(base_zern.rows, "n_max", 18.0)["cv"])], "x",
                 color="#d62728",
                 ms=12, mew=2, label="基线 18: 黑帧")
    ax_cv_n.set_yscale("log")
    ax_cv_n.set_xlabel("截断阶数 n_max")
    ax_cv_n.set_ylabel("CV (对数轴)")
    ax_cv_n.set_title("(c) 基线 18 的 CV 尖峰在复测里消失")
    ax_cv_n.legend(fontsize=8)

    # (d) Zernike score
    panel(ax_sc_n, b_n, bz_sc, "基线 (20 ms)", color="#7f7f7f", ls="--", marker="o", ms=4)
    panel(ax_sc_n, r_n, [float(r["score"]) for r in r_zern.rows], "复测 (180 ms)",
          color="#2ca02c", marker="s", ms=7)
    ax_sc_n.set_xlabel("截断阶数 n_max")
    ax_sc_n.set_ylabel("score")
    r_score_n = [float(r["score"]) for r in r_zern.rows]
    mono_n = all(r_score_n[i] > r_score_n[i - 1] for i in range(1, len(r_score_n)))
    ax_sc_n.set_title(
        "(d) score: 复测 17/18/19 "
        + ("单调上升" if mono_n else f"在 {min(r_score_n):.4f}–{max(r_score_n):.4f} 之间，无尖峰"))
    ax_sc_n.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        f"复测（功率已变, 折合曝光 {rf.equivalent_exposure_ms:.1f} ms）—— 只做批内邻居比较",
        fontsize=11,
    )
    fig.tight_layout()
    path = PATHS.fig_dir / f"fig5_retest_neighbors{FIG_TAG}.png"
    savefig(fig, path)
    return path

def render_figures(run: Run, f: Findings) -> list[Path]:
    PATHS.fig_dir.mkdir(parents=True, exist_ok=True)
    return [
        fig_sides_objectives(run, f),
        fig_sides_exposure(run, f),
        fig_zernike(run, f),
        fig_blaze(run, f),
    ]

def fig_atomic_sides(run: Run, atomic: RetestRun) -> Path:
    """边长 → CV/PIB/score: 本轮实线 + 基线虚线 (定性对照)。"""
    setup_matplotlib()
    a_sides = atomic.stages["sides"]
    b_sides = run.stages["sides"]
    ax_x = a_sides.col("side_px")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.6))

    ax1.plot(ax_x, a_sides.col("cv"), "o-", color="#d62728", label="CV（本轮）")
    ax1.plot(b_sides.x, b_sides.col("cv"), "s--", color="#d62728", alpha=0.45,
             label="CV（基线, 20 ms）")
    ax1.set_xlabel("目标方框边长 (camera px)")
    ax1.set_ylabel("CV", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax1.grid(alpha=0.3)
    ax1b = ax1.twinx()
    ax1b.plot(ax_x, a_sides.col("pib"), "o-", color="#1f77b4", label="PIB（本轮）")
    ax1b.plot(b_sides.x, b_sides.col("pib"), "s--", color="#1f77b4", alpha=0.45,
              label="PIB（基线, 20 ms）")
    ax1b.set_ylabel("PIB", color="#1f77b4")
    ax1b.tick_params(axis="y", labelcolor="#1f77b4")
    handles = ax1.get_lines() + ax1b.get_lines()
    ax1.legend(handles, [ln.get_label() for ln in handles], fontsize=7.5, loc="upper left")
    ax1.set_title("(a) CV / PIB: 形状一致, 高度不可比")

    ax2.plot(ax_x, a_sides.col("score"), "o-", color="#2ca02c", label="score（本轮）")
    ax2.plot(b_sides.x, b_sides.col("score"), "s--", color="#2ca02c", alpha=0.45,
             label="score（基线, 20 ms）")
    scores_a = a_sides.col("score")
    peak_x = max(zip(ax_x, scores_a), key=lambda pair: pair[1])[0]
    plateau_x = [x for x, sc in zip(ax_x, scores_a) if sc >= max(scores_a) - 1e-3]
    ax2.plot([peak_x], [max(scores_a)], "k*", ms=14, label=f"argmax {peak_x:.0f} px")
    if len(plateau_x) > 1:
        ax2.axvspan(min(plateau_x), max(plateau_x), color="k", alpha=0.08,
                    label=f"平台 {min(plateau_x):.0f}–{max(plateau_x):.0f} px")
    ax2.set_xlabel("目标方框边长 (camera px)")
    ax2.set_ylabel("score = 0.5·PIB + 0.5/(1+CV)")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8, loc="lower right")
    ax2.set_title("(b) score 是平台, 不是单点")
    fig.suptitle(f"整轮原子重扫（{atomic.exposure_ms:g} ms, 单一版本）", fontsize=11)
    fig.tight_layout()
    path = PATHS.fig_dir / f"fig10_atomic_sides_{batch_tag(atomic)}.png"
    savefig(fig, path)
    return path

def fig_atomic_blaze(atomic: RetestRun) -> Path:
    """光栅周期 → 预测位移 (1/P) vs 实测位移 (近水平)。"""
    setup_matplotlib()
    blaze = atomic.stages["blaze"]
    x = blaze.col("blaze_period_px")
    pred = blaze.col("predicted_shift_px")
    meas = blaze.col("measured_shift_y_px")
    clipped = [i for i in range(len(x)) if blaze.col("box_clipped_px")[i] > 0]
    keep = [i for i in range(len(x)) if i not in clipped]

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(x, pred, "o-", color="#1f77b4", label="预测位移 (∝ 1/周期)")
    ax.plot([x[i] for i in keep], [meas[i] for i in keep], "s-", color="#ff7f0e",
            label="实测位移 y")
    for i in clipped:
        ax.plot([x[i]], [meas[i]], "x", color="#d62728", ms=11, mew=2,
                label="该行方框出画 (不计入趋势)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.0f}" for v in x])
    ax.minorticks_off()
    ax.set_yticks([50, 100, 200, 400])
    ax.set_yticklabels(["50", "100", "200", "400"])
    ax.set_xlabel("闪耀光栅周期 (SLM px, 对数轴)")
    ax.set_ylabel("位移 (camera px, 对数轴)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax.set_title("预测按 1/周期 下降, 实测几乎水平 —— 比值从 0.22 涨到 1.20")
    fig.tight_layout()
    path = PATHS.fig_dir / f"fig11_atomic_blaze_shift_{batch_tag(atomic)}.png"
    savefig(fig, path)
    return path

def fig_zernike_residual(atomic: RetestRun, residuals: ResidualSet) -> Path:
    """fig12: 左上 = GS 基准相位, 第 1 行 = 残差相位, 第 2 行 = 同配置 CCD 远场。

    **列间共用同一尺度**: 残差行固定 ``±π`` (一个色标), 远场行共用一个
    ``LogNorm(vmin=1, vmax=四列最大峰值)`` —— 同一行内的颜色因此可直接横比。
    行间 / 基准列与残差列之间各加一条细线, 把「两种量」分开。
    """
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D

    setup_matplotlib()
    panels = residuals.panels
    ncols = len(panels) + 1
    fig = plt.figure(figsize=(2.85 * ncols + 1.1, 6.6))
    grid = fig.add_gridspec(2, ncols + 1, hspace=0.24, wspace=0.10,
                            width_ratios=[1.0] * ncols + [0.05])

    ax_base = fig.add_subplot(grid[0, 0])
    im_base = ax_base.imshow(residuals.base, cmap=BASE_CMAP, vmin=-RESIDUAL_VLIM,
                             vmax=RESIDUAL_VLIM, interpolation="nearest")
    ax_base.set_title("GS 基准相位 (raw rad)", fontsize=8.5)
    ax_base.set_xticks([])
    ax_base.set_yticks([])
    # 基准相位的色标占 (1,0) 格: 用子网格把它做窄, 否则会占满整格宽度
    cax_base = fig.add_subplot(grid[1, 0].subgridspec(1, 5, wspace=0.0)[0, 2])
    cb_base = fig.colorbar(im_base, cax=cax_base)
    cb_base.set_label("基准相位 (rad)", fontsize=7.5)
    cb_base.ax.tick_params(labelsize=6.5)

    residual_axes = []
    residual_image = None
    for k, panel in enumerate(panels):
        ax = fig.add_subplot(grid[0, k + 1])
        residual_image = ax.imshow(
            np.ma.masked_where(~panel.mask, panel.residual),
            cmap=RESIDUAL_CMAP, vmin=-RESIDUAL_VLIM, vmax=RESIDUAL_VLIM,
            interpolation="nearest",
        )
        ax.set_title(
            f"n_max={panel.n_max}  残差 RMS={panel.rms_recorded:.3f} rad\n"
            f"解释={panel.explained * 100:.2f}%",
            fontsize=8.0,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        residual_axes.append(ax)
    cax_res = fig.add_subplot(grid[0, ncols])
    cb_res = fig.colorbar(residual_image, cax=cax_res)
    cb_res.set_label("残差 (rad)", fontsize=7.5)
    cb_res.set_ticks([-math.pi, -math.pi / 2.0, 0.0, math.pi / 2.0, math.pi])
    cb_res.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])
    cb_res.ax.tick_params(labelsize=6.5)

    stage_dir = atomic.root / "zernike"
    headers = atomic.stages["zernike"].headers
    frame_vmax = max(float(p.row["frame_peak"]) for p in panels)
    ccd_axes = []
    for k, panel in enumerate(panels):
        ax = fig.add_subplot(grid[1, k + 1])
        ccd_axes.append(ax)
        pick = CellPick(stage="zernike", row=panel.row, role="残差列")
        ref = locate_frames(stage_dir, panel.row, headers)
        if not ref.paths:
            ax.axis("off")
            ax.text(0.5, 0.5, "缺帧", ha="center", va="center", fontsize=9, color="#d62728")
            continue
        _check_frame_params("zernike", pick, ref)
        image = load_frame_mean(ref.paths)
        if abs(float(image.max()) - float(panel.row["frame_peak"])) > PEAK_MATCH_ATOL:
            raise ValueError(
                f"zernike/{panel.config}: 帧最大值 {float(image.max()):.6g} 与 CSV "
                f"frame_peak {panel.row['frame_peak']} 不一致 —— 帧与 CSV 行对不上"
            )
        x0, x1, y0, y1, _ = _crop_box(panel.row, image.shape, CROP_FACTOR)
        shown = image[y0:y1, x0:x1]
        ax.imshow(shown, cmap="inferno",
                  norm=LogNorm(vmin=1.0, vmax=max(frame_vmax, 1.0 + 1e-6)),
                  interpolation="nearest")
        _annotate_cell(ax, pick, crop=(x0, x1, y0, y1))
        peak = float(panel.row["frame_peak"])
        sat = "  饱和" if peak >= SATURATION_COUNTS else ""
        ax.set_title(
            f"CV={float(panel.row['cv']):.3f}  PIB={float(panel.row['pib']):.3f}  "
            f"peak={peak:.0f}{sat}",
            fontsize=8.0,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.98, 0.02, f"×{len(ref.paths)} 帧均值", transform=ax.transAxes,
                fontsize=7.0, color="white", ha="right", va="bottom",
                bbox=dict(fc="black", ec="none", alpha=0.45, pad=1.2))
        del image, shown

    fig.suptitle(
        f"Zernike 截断残差 × 同配置 CCD 远场（{batch_tag(atomic)}, {atomic.exposure_ms:g} ms）",
        fontsize=11,
    )
    # 不用 tight_layout: 色标轴是 subgridspec 造出来的, tight_layout 会报
    # "Axes that are not compatible" 并给出错位的间距 —— 这里显式定边距。
    fig.subplots_adjust(left=0.012, right=0.985, top=0.90, bottom=0.025,
                        hspace=0.30, wspace=0.08)
    # 分隔线必须在 layout 定稿后按**实际**轴位置画 (否则与格子的间距对不上)
    fig.canvas.draw()
    if residual_axes and ccd_axes:
        boxes_top = [ax.get_position() for ax in residual_axes]
        boxes_bottom = [ax.get_position() for ax in ccd_axes]
        y_line = 0.5 * (min(b.y0 for b in boxes_top) + max(b.y1 for b in boxes_bottom))
        fig.add_artist(Line2D([min(b.x0 for b in boxes_top), max(b.x1 for b in boxes_top)],
                              [y_line, y_line], transform=fig.transFigure,
                              color="#999999", lw=0.7, zorder=5))
        x_line = 0.5 * (ax_base.get_position().x1 + boxes_top[0].x0)
        y_lo = min(b.y0 for b in boxes_top + boxes_bottom)
        y_hi = max(b.y1 for b in boxes_top + boxes_bottom)
        fig.add_artist(Line2D([x_line, x_line], [y_lo, y_hi], transform=fig.transFigure,
                              color="#999999", lw=0.7, zorder=5))

    path = PATHS.fig_dir / residual_figure_name(atomic)
    savefig(fig, path)
    return path
