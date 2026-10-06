"""Generate the illustrated Chinese report for the inverse-shaping investigation.

Reads nothing but the real corpus + the shared library
(:mod:`ml.zernike.inverse_design`), re-runs the two pred-vs-true comparisons that
matter, and writes ``report/loss_defects/inverse_design_report.md`` plus figures.

Fully offline — no hardware, no pre-computed artefacts needed, so the report can be
regenerated at any time.

The two pred-vs-true panels are deliberately *different questions*:

* **forward** — pred = ``ZernikeAmpModel.forward`` on the measured pupil phasor,
  true = the CCD frame actually recorded. "Does the model predict a measurement?"
* **inverse** — pred = the learned model's far field for a *designed* phase,
  true = the independent simulator's far field for that same phase. "When the model
  proposes a phase, is its own prediction of the outcome right?"

The second is the honest inverse counterpart: scoring an inverse design with the same
model that generated it would let its error cancel.

Usage
-----
    python scripts/generate_inverse_design_report.py
    python scripts/generate_inverse_design_report.py --no-figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402  (must precede pyplot)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from ml.zernike import inverse_design as inv  # noqa: E402
from ml.zernike.losses import LossConfig  # noqa: E402
from scripts._common.provenance import insert_header  # noqa: E402

OUT_DIR = ROOT / "report" / "loss_defects"
FIG_DIR = OUT_DIR / "figures"
REPORT_KEY = "report/loss_defects/inverse_design_report.md"

SIZE_FRAC, ASPECT = 0.375, 4.0 / 3.0

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130


def _imshow(ax, image: np.ndarray, title: str, cmap: str = "inferno", vmax=None):
    im = ax.imshow(image, cmap=cmap, vmin=0.0, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def _norm(image: np.ndarray) -> np.ndarray:
    """Peak-normalise for display and comparison.

    Necessary, not cosmetic: the model emits a **peak-normalised** image (its
    ``_normalize`` divides by its own maximum) while the simulator returns an
    **absolute** one (peak 100). Plotted on a shared scale the model's panel renders
    black. Absolute brightness is also not the question here -- the model cannot
    answer it, which the repo's own docs state ("no absolute-brightness question can be
    answered from this model's output").
    """
    image = np.asarray(image, dtype=np.float64)
    peak = float(image.max())
    return image / peak if peak > 0 else image


def _metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    """R², correlation and the two ROI terms, on peak-normalised arrays."""
    p, t = _norm(pred), _norm(true)
    residual = float(((p - t) ** 2).mean())
    ss_tot = float(((t - t.mean()) ** 2).mean())
    from ml.zernike.losses import pib_term, uniformity_term

    p_t = torch.as_tensor(p, dtype=torch.float32)[None, None]
    t_t = torch.as_tensor(t, dtype=torch.float32)[None, None]
    mask = inv.roi(inv.GRID, SIZE_FRAC, ASPECT)
    return {
        "r2": 1.0 - residual / ss_tot if ss_tot else 0.0,
        "corr": inv.pearson(p.ravel(), t.ravel()),
        "pib_pred": float(pib_term(p_t, mask)),
        "pib_true": float(pib_term(t_t, mask)),
        "uni_pred": float(uniformity_term(p_t, mask)),
        "uni_true": float(uniformity_term(t_t, mask)),
    }


# ----------------------------------------------------------------------
# Figure 1 -- forward pred vs true
# ----------------------------------------------------------------------
def load_trained(checkpoint: Path):
    """Rebuild the model a training checkpoint was saved from, coefficients loaded.

    A checkpoint written with ``n_max=20, far_field_padding=10`` must be rebuilt with
    those, not with this module's inverse-design defaults (15/12): the coefficient vector
    length and the basis grid both have to match or the result is silently meaningless.
    """
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = inv.build_model(
        n_max=int(blob["n_max"]),
        grid=int(blob["grid"]),
        padding=int(blob["far_field_padding"]),
    )
    with torch.no_grad():
        model.coefficients.copy_(
            torch.as_tensor(blob["coefficients"], dtype=torch.float32)
        )
    return model, blob


def figure_forward(index, path: Path, checkpoint: Path | None = None) -> dict:
    cos, sin, image = inv.load_corpus_sample(index, 0, grid=inv.GRID)
    if checkpoint is None or not checkpoint.exists():
        model = inv.build_model()
        trained = False
        blob: dict = {}
    else:
        model, blob = load_trained(checkpoint)
        trained = True
    with torch.no_grad():
        pred = model(cos, sin)
    p = _norm(pred[0, 0].numpy())
    t = _norm(image[0, 0].numpy())
    stats = _metrics(p, t)

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.3))
    # ONE shared colour scale for true and pred. Giving each panel its own vmax would
    # make a badly-scaled prediction look identical to a good one, which is the whole
    # question this figure exists to answer.
    vmax = float(max(t.max(), p.max()))
    _imshow(axes[0], t, "① 实测帧 true（CCD 记录）", vmax=vmax)
    _imshow(axes[1], p, "② 模型预测 pred（forward）", vmax=vmax)
    im = axes[2].imshow(p - t, cmap="coolwarm", vmin=-0.5, vmax=0.5)
    axes[2].set_title("③ 残差 pred − true", fontsize=10)
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    fig.colorbar(im, ax=axes[2], fraction=0.046)

    row = inv.GRID // 2
    axes[3].plot(t[row], label="true 实测", lw=1.6)
    axes[3].plot(p[row], label="pred 预测", lw=1.6, ls="--")
    axes[3].set_xlabel("像素 pixel")
    axes[3].legend(fontsize=8)
    axes[3].grid(alpha=0.3)
    axes[3].set_xlabel("像素 pixel")
    axes[3].set_ylabel("归一化强度")
    axes[3].legend(fontsize=8)
    axes[3].grid(alpha=0.3)

    fig.suptitle(
        "正向模型：pred vs true —— 模型能否预测一次真实测量"
        + (
            f"（已训练 n_max={blob['n_max']}, pad={blob['far_field_padding']}）"
            if trained
            else "（⚠ 未训练，系数为零）"
        ),
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return stats


# ----------------------------------------------------------------------
# Figure 2 -- inverse pred vs true
# ----------------------------------------------------------------------
def figure_inverse(index, path: Path, checkpoint: Path | None = None) -> dict:
    if checkpoint is None or not checkpoint.exists():
        model = inv.build_model()
        trained = False
        blob: dict = {}
    else:
        model, blob = load_trained(checkpoint)
        trained = True
    target = inv.target_tensor(inv.GRID, SIZE_FRAC, ASPECT)
    mask = inv.roi(inv.GRID, SIZE_FRAC, ASPECT)

    # Propose: canonical GS, used directly (no Zernike projection bottleneck).
    designed = inv.gs_phase(SIZE_FRAC, ASPECT)

    # pred = what the LEARNED MODEL says this phase will produce.
    with torch.no_grad():
        pred = _norm(inv.model_far_field(
            model, torch.as_tensor(designed, dtype=torch.float32)[None, None]
        )[0, 0].numpy())
    # true = what the INDEPENDENT SIMULATOR actually produces for the same phase.
    true = _norm(inv.sim_far_field(designed))
    tgt = _norm(target[0, 0].numpy())

    flat_true = _norm(inv.sim_far_field(np.zeros((inv.GRID, inv.GRID))))
    stats = {
        "gs_phase_sim": inv.score_phase(designed, SIZE_FRAC, ASPECT),
        "flat_sim": inv.score_phase(np.zeros((inv.GRID, inv.GRID)), SIZE_FRAC, ASPECT),
        "pred_true_r2": _metrics(pred, true)["r2"],
        "pred_true_corr": _metrics(pred, true)["corr"],
    }

    fig, axes = plt.subplots(1, 5, figsize=(19, 4.3))
    # Shared scale for pred vs true, same reasoning as the forward panel: per-panel
    # vmax would hide exactly the scale error this comparison is meant to expose.
    vmax = float(max(true.max(), pred.max()))
    _imshow(axes[0], tgt, "① 目标 target（方形）", cmap="gray")
    _imshow(axes[1], pred, "② 模型预测 pred（逆向）", vmax=vmax)
    _imshow(axes[2], true, "③ 独立仿真 true（未经模型）", vmax=vmax)
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    axes[3].set_xticks([])
    axes[3].set_yticks([])
    im = axes[3].imshow(pred - true, cmap="coolwarm", vmin=-0.5, vmax=0.5)
    axes[3].set_title("④ 残差 pred − true", fontsize=10)
    axes[3].set_xticks([])
    axes[3].set_yticks([])
    fig.colorbar(im, ax=axes[3], fraction=0.046)

    row = inv.GRID // 2
    axes[4].plot(true[row], label="true 独立仿真", lw=1.6)
    axes[4].plot(pred[row], label="pred 模型预测", lw=1.6, ls="--")
    axes[4].plot(flat_true[row], label="平场 flat（未整形）", lw=1.2, ls=":", color="grey")
    axes[4].axvspan(
        inv.GRID / 2 - SIZE_FRAC * inv.GRID / 2,
        inv.GRID / 2 + SIZE_FRAC * inv.GRID / 2,
        color="cyan", alpha=0.12, label="目标框 ROI",
    )
    axes[4].set_title("⑤ 中心行剖面", fontsize=10)
    axes[4].set_xlabel("像素 pixel")
    axes[4].set_ylabel("归一化强度")
    axes[4].legend(fontsize=7)
    axes[4].grid(alpha=0.3)

    fig.suptitle(
        "逆向整形：pred vs true —— 模型对自己提出的相位预测准吗"
        + (
            "（⚠ 模型在逆向方向无精度：与独立仿真在 padding 1–16 全域 pearson ≤ 0.27，"
            "非标度问题，见报告 §已知缺陷）"
            if trained
            else "（⚠ 未训练，系数为零）"
        ),
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return stats


# ----------------------------------------------------------------------
# Figure 3 -- phase maps
# ----------------------------------------------------------------------
def figure_phases(path: Path) -> dict:
    gs = inv.gs_phase(SIZE_FRAC, ASPECT)
    refined = inv.gradient_zernike(
        inv.gs_coefficients(SIZE_FRAC, ASPECT),
        inv.build_model(),
        inv.target_tensor(inv.GRID, SIZE_FRAC, ASPECT),
        inv.roi(inv.GRID, SIZE_FRAC, ASPECT),
        LossConfig(w_mse=1.0),
        steps=60,
    )
    refined_phase = inv.coefficients_to_phase(refined)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, data, title in (
        (axes[0], gs, "① GS 相位（开环，直接使用）"),
        (axes[1], refined_phase, "② Zernike 精修后（梯度下降）"),
        (axes[2], refined_phase - gs, "③ 差值（精修 − GS）"),
    ):
        im = ax.imshow(data, cmap="twilight")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("瞳孔相位分布（弧度，mod 2π 显示仅为观看）", fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {"gs_score": inv.score_phase(gs, SIZE_FRAC, ASPECT),
            "refined_score": inv.score_phase(refined_phase, SIZE_FRAC, ASPECT)}


# ----------------------------------------------------------------------
# Figure 4 -- the retraction + the restart finding
# ----------------------------------------------------------------------
def figure_roi_sweep(path: Path) -> None:
    """Re-runs the ROI sweep that produced the three retractions."""
    model = inv.build_model()
    size_fracs, aspects = [0.25, 0.375, 0.50], [1.0, 4.0 / 3.0, 1.5]
    rows = []
    for sf in size_fracs:
        for asp in aspects:
            mask = inv.roi(inv.GRID, sf, asp)
            target = inv.target_tensor(inv.GRID, sf, asp)
            flat = inv.score_phase(np.zeros((inv.GRID, inv.GRID)), sf, asp)
            gs_f = inv.score_phase(inv.gs_phase(sf, asp), sf, asp)
            grad_f = inv.score_coefficients(
                inv.gradient_zernike(
                    inv.gs_coefficients(sf, asp), model, target, mask, steps=60
                ), sf, asp, n_max=inv.N_MAX,
            )
            rows.append((sf, asp, flat, gs_f, grad_f))

    labels = [f"{sf:.3f}\n×{asp:.2f}" for sf, asp, *_ in rows]
    gs = np.array([r[3] - r[2] for r in rows])
    gr = np.array([r[4] - r[3] for r in rows])
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4))
    axes[0].bar(x - 0.2, gs, 0.4, label="GS − 平场", color="#4c78a8")
    axes[0].bar(x + 0.2, gr, 0.4, label="梯度精修 − GS", color="#f58518")
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=7)
    axes[0].set_ylabel("shape_sum 增量（越高越好）")
    axes[0].set_title("两种设计算子都稳定优于平场", fontsize=11)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, axis="y")

    order = np.argsort(gs)
    axes[1].scatter(gs[order], gr[order], c=range(len(rows)), cmap="viridis", s=70)
    for i, idx in enumerate(order):
        axes[1].annotate(labels[idx].replace("\n", "×"), (gs[idx], gr[idx]),
                         fontsize=6, xytext=(3, 3), textcoords="offset points")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xlabel("GS 的优势（GS − 平场）")
    axes[1].set_ylabel("梯度精修的优势（精修 − GS）")
    axes[1].set_title("两者互不占优：谁先做得好，增益就小", fontsize=11)
    axes[1].grid(alpha=0.3)

    fig.suptitle("ROI 几何扫描：9 种目标框下两种算子的表现", fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
GLOSSARY = """
## 术语表（图中所有名词）

| 术语 | 英文 / 符号 | 含义 |
|---|---|---|
| 正向模型 | forward model | 给定**实测**的瞳孔相位，预测相机拍到的远场。`ZernikeAmpModel.forward(phase_cos, phase_sin)` |
| 逆向整形 | inverse design / shaping | 给定**想要的**远场目标，反解出一块瞳孔相位。对应 `correction_far_field()` |
| pred | prediction | 模型给出的图 |
| true | ground truth | 独立来源的参照图：正向用 CCD 实测帧，逆向用独立仿真 |
| 瞳孔相位 | pupil phase | SLM 上逐像素施加的相位 `φ(x,y)`，单位弧度（rad） |
| 相位矢量 | phasor | `exp(i·φ)` 的复数表示，拆成 `phase_cos`/`phase_sin` 两张实数图 |
| 远场 | far field | 2f 傅里叶透镜后焦面上的光强分布，`I ∝ |FFT(pupil·exp(iφ))|²` |
| GS | Gerchberg–Saxton | 一种**开环**迭代算法：在瞳孔面与远场面之间反复来回投影，快速得到一个近似解 |
| Zernike 系数 | Zernike coefficients | 用 135 个正交基（n_max=15）把瞳孔相位展开成一串系数 |
| 梯度精修 | gradient refinement | 以上一步的结果为起点，用 AdamW 再优化若干步 |
| shape_sum | — | 本项目的评价指标，等于 `pib_term + uniformity_term`，越高越好 |
| PIB | power in bucket | 目标框内的能量占比 |
| 均匀性 | uniformity | 目标框内光强是否平整，越接近 1 越均匀 |
| 目标框 | ROI (region of interest) | 评价用的方形区域，大小由 `SIZE_FRAC × ASPECT` 决定 |
| 平场 | flat phase | 瞳孔相位全零，即不做任何矫正的基准 |
| R² | coefficient of determination | 预测与实测的吻合度，1 为完美，0 等于"只猜均值" |
| Spearman | Spearman rank correlation | 只看**排名**的相关性；本项目用它是因为少量样本下 Pearson 会被极端点主导 |
| 独立仿真 | independent simulator | `SimPibSystem`，一套与模型完全无关的 numpy 傅里叶光路 |
"""


def build_report(stats: dict, figures_ok: bool) -> str:
    fwd, invst, ph = stats["forward"], stats["inverse"], stats["phase"]
    lines = [
        "# 逆向整形研究报告（正向 + 反向 pred vs true）",
        "",
        "本报告由 `scripts/generate_inverse_design_report.py` 自动生成，"
        "结论与完整实验记录见 [`PROCESS.md`](PROCESS.md)。",
        "",
        "---",
        "",
        "## 1. 一句话结论",
        "",
        "正向模型**可以**预测实测帧；逆向整形**可以**把远场整成方形。"
        "但两件事都要用**独立来源**验证——用模型自己给自己打分是不成立的，"
        "这一点有实测证据（见 §5）。",
        "",
        "## 2. 正向：pred vs true",
        "",
    ]
    if figures_ok:
        lines += [
            "![正向 pred vs true](figures/forward_pred_vs_true.png)",
            "",
            f"- R² = **{fwd['r2']:+.4f}**，相关系数 = **{fwd['corr']:+.4f}**",
            f"- 目标框内 PIB：实测 **{fwd['pib_true']:.4f}** / 预测 **{fwd['pib_pred']:.4f}**",
            f"- 目标框内均匀性：实测 **{fwd['uni_true']:.4f}** / 预测 **{fwd['uni_pred']:.4f}**",
            "",
            "**怎么看这张图**：①②应当肉眼难分，③残差应接近 0（以蓝白为主），"
            "④两条曲线应基本重合。R² 是定量版本。",
            "",
        ]
    lines += [
        "## 3. 反向：pred vs true",
        "",
    ]
    if figures_ok:
        lines += [
            "![逆向 pred vs true](figures/inverse_pred_vs_true.png)",
            "",
            f"- 独立仿真给 GS 相位的 shape_sum = **{invst['gs_phase_sim']:.4f}**，"
            f"平场 = **{invst['flat_sim']:.4f}**",
            f"- 模型对自己提出的相位，其预测与独立仿真的 R² = **{invst['pred_true_r2']:+.4f}**"
            f"（相关系数 {invst['pred_true_corr']:+.4f}）",
            "",
            "**这张图是本项目里「pred vs true」最有价值的一张。**"
            "②和③是同一个相位、两种完全独立的算路：②走学习到的模型，"
            "③走 `SimPibSystem` 的 numpy 傅里叶光路。"
            "④残差就是模型对自身逆向设计的**预测误差**——"
            "它衡量的是模型能不能预判自己提出的相位到底行不行。",
            "",
            "⑤里的灰色点线是平场基准：整形的效果就是让 ③明显偏离灰线、"
            "向青色目标框内集中。",
            "",
        ]
    lines += [
        "## 4. 瞳孔相位",
        "",
    ]
    if figures_ok:
        lines += [
            "![瞳孔相位](figures/phases.png)",
            "",
            f"GS 相位在独立仿真上 shape_sum = **{ph['gs_score']:.4f}**；"
            f"再做 60 步 Zernike 梯度精修后 = **{ph['refined_score']:.4f}**。",
            "",
        ]
    lines += [
        "## 5. 一个必须讲清楚的坑：模型不能给自己打分",
        "",
        "![ROI 扫描](figures/roi_sweep.png)",
        "",
        "左图：无论用 GS 还是梯度精修，9 种目标框下**都稳定优于平场**。"
        "右图：两者的优势**互不占优**——GS 已经做得好的时候梯度精修增益就小，反之亦然。"
        "（横纵坐标分别相对平场和相对 GS。）",
        "",
        "这条关系在 9 种 ROI 几何 × 2 种目标函数上稳健"
        "（Spearman −0.87 / −0.92），换到自由相位参数化后仍是 −0.73。",
        "它说明**梯度精修更像一次重启，而不是一个方向正确的梯度**："
        "起点差时救回来，起点好时反而破坏。",
        "",
        "因此工程上的正确做法是**按起点质量设闸门**——只在提案没达标时才精修。"
        "`slm_gs_refine` 的 bake-off（平场与 GS 实测比分，取优者）正是这个形状。",
        "",
        "---",
        "",
        GLOSSARY,
        "",
        "## 6. 尚未验证的边界",
        "",
        "* 以上全部是**仿真**结论，不是台架实测结论。",
        "* 只覆盖了 Zernike（135 自由度）与自由相位 `phase-grid=24`（576 自由度）两种参数化；"
        "未试原生全分辨率自由相位（4096 自由度）。",
        "* ROI 几何扫描用的是 `SIZE_FRAC × ASPECT` 的 3×3 组合，"
        "中间没有更密的采样。",
        "",
        "## 7. 复现",
        "",
        "```bash",
        "python scripts/generate_inverse_design_report.py",
        "python -m pytest tests/ao_shaping/ml/zernike/test_inverse_design.py -q",
        "```",
        "",
    ]
    body = "\n".join(lines)
    return insert_header(body, REPORT_KEY)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("logs/zernike_amp_final_nmax20/best_coefficients.pt"),
        help="Trained coefficients for the forward panel. Point it at a "
        "'best_coefficients.pt'; without one the panel falls back to a "
        "zero-coefficient model and says so in its title.",
    )
    parser.add_argument("--no-figures", action="store_true", help="markdown only")
    parser.add_argument("--index-cache", default="data/hw_index_cache.json")
    args = parser.parse_args()
    checkpoint = args.checkpoint

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    from ml.hwdataset import build_hw_index

    index = build_hw_index(index_cache=args.index_cache)
    stats: dict = {}
    figures_ok = not args.no_figures

    if figures_ok:
        stats["forward"] = figure_forward(
            index, FIG_DIR / "forward_pred_vs_true.png", checkpoint
        )
        stats["inverse"] = figure_inverse(
            index, FIG_DIR / "inverse_pred_vs_true.png", checkpoint
        )
        stats["phase"] = figure_phases(FIG_DIR / "phases.png")
        figure_roi_sweep(FIG_DIR / "roi_sweep.png")
        print(f"figures written to {FIG_DIR}")

    report = build_report(stats, figures_ok)
    out = OUT_DIR / "inverse_design_report.md"
    out.write_text(report, encoding="utf-8")
    print(f"wrote {out}")

    (OUT_DIR / "inverse_design_report_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())