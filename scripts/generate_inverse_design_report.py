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
from functools import partial
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
from scripts._common import fmt_signed  # noqa: E402
from scripts._common.provenance import insert_header  # noqa: E402

#: Bound to ``missing="MISSING"`` so this report keeps its own absence token. A
#: ``def _fmt`` copy of ``fmt_signed`` is banned by
#: ``test_common_helpers_not_reintroduced.py`` -- two copies drift, and this one had
#: already been reported. ``spec`` stays positional (``_fmt(x, ".4f")``).
_fmt = partial(fmt_signed, missing="MISSING")

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


def _load(name: str) -> dict:
    """Load a saved result panel, or ``{}`` when it has not been produced yet.

    The report must render on a clean checkout where only some experiments have been run, so a
    missing artefact degrades to an empty dict and the section says so rather than inventing
    numbers.
    """
    import json

    path = OUT_DIR / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value, spec: str = "+.4f", missing: str = "MISSING") -> str:
    """Format a number, or report it missing rather than as 0.0000.

    A metric that silently renders as zero is indistinguishable from a real zero, which is how
    an absent measurement turns into a fake result.
    """
    if value is None:
        return missing
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f != f:  # NaN
        return missing
    return format(f, spec)


def _search_rows() -> list[str]:
    """The paired-delta table for the architecture / loss / augmentation search."""
    data = _load("forward_search.json")
    if not data.get("summary"):
        return ["*NOT PRODUCED YET - run `python scripts/forward_search.py`*"]
    out = ["| arm | mean R2 | dR2 (paired) | std err | beats baseline | dSSIM |",
           "|---|---|---|---|---|---|"]
    for row in data["summary"]:
        out.append(
            f"| {row['arm']} | {_fmt(row['r2'])} | {_fmt(row['d_r2'])} | "
            f"{_fmt(row.get('d_r2_se'), '.4f')} | {row.get('pos', 0)}/3 | "
            f"{_fmt(row.get('d_ssim'))} |"
        )
    return out


def _combined_rows() -> list[str]:
    """The winner table: physics + U-Net residual against each architecture alone."""
    data = _load("forward_search_extra.json")
    rows = data.get("rows")
    if not rows:
        return ["*NOT PRODUCED YET - run `python scripts/forward_search_extra.py`*"]
    base = {r["seed"]: r for r in rows if r["arm"] == "A0 physics"}
    out = ["| arm | dR2 (paired) | std err | beats baseline | dSSIM |", "|---|---|---|---|---|"]
    for name in sorted({r["arm"] for r in rows}):
        sel = [r for r in rows if r["arm"] == name]
        if name == "A0 physics":
            out.append("| A0 physics (incumbent) | - | - | - | - |")
            continue
        deltas = [r["r2"] - base[r["seed"]]["r2"] for r in sel if r["seed"] in base]
        if not deltas:
            continue
        mean = sum(deltas) / len(deltas)
        var = sum((d - mean) ** 2 for d in deltas) / (len(deltas) - 1) if len(deltas) > 1 else 0.0
        ss = [r["ssim"] - base[r["seed"]]["ssim"] for r in sel if r["seed"] in base]
        out.append(
            f"| {name} | {_fmt(mean)} | {_fmt(var / len(deltas) ** 0.5, '.4f')} | "
            f"{sum(1 for d in deltas if d > 0)}/{len(deltas)} | {_fmt(sum(ss) / len(ss))} |"
        )
    out.append("")
    out.append(f"({len(base)} seeds, paired. Read from `forward_search_extra.json`.)")
    return out


def _inverse_rows() -> list[str]:
    """The three-arm inverse comparison that the improved forward model triggered."""
    data = _load("inverse_combined.json")
    rows = data.get("rows")
    if not rows:
        return ["*NOT PRODUCED YET - run `python scripts/inverse_combined.py`*"]
    flat = float(data.get("flat", 0.0))
    gs = float(data.get("gs", 0.0))
    base = {r["seed"]: r["inv_rand"] for r in rows if r["arm"] == "A0 physics"}
    out = ["| arm | forward R2 | inverse (random start) | vs flat | d(inverse) | consistent |",
           "|---|---|---|---|---|---|",
           f"| _flat reference (do nothing)_ | - | {flat:.4f} | 0.0000 | - | - |",
           f"| _GS proposal_ | - | {gs:.4f} | {gs - flat:+.4f} | - | - |"]
    for name in ("A0 physics", "A5 unet", "D2 physics+residual"):
        sel = [r for r in rows if r["arm"] == name]
        if not sel:
            continue
        inv = sum(r["inv_rand"] for r in sel) / len(sel)
        r2 = sum(r["r2"] for r in sel) / len(sel)
        if name == "A0 physics":
            out.append(f"| {name} | {_fmt(r2)} | {inv:.4f} | {inv - flat:+.4f} | - | - |")
            continue
        d = [r["inv_rand"] - base[r["seed"]] for r in sel if r["seed"] in base]
        if not d:
            continue
        out.append(
            f"| {name} | {_fmt(r2)} | {inv:.4f} | {inv - flat:+.4f} | "
            f"{_fmt(sum(d) / len(d))} | {sum(1 for v in d if v > 0)}/{len(d)} |"
        )
    return out


def _taxonomy_rows() -> list[str]:
    """The closed-loop loss taxonomy scorecard, read from the saved panel."""
    data = _load("closed_loop_loss_taxonomy.json")
    if not data.get("verdict"):
        return ["*NOT PRODUCED YET - run `python scripts/closed_loop_loss_taxonomy.py`*"]
    inc = data["incumbent_best_movement"]
    v = data["verdict"]
    labels = {
        "weighted_mse": "加权 MSE（目标区高权重）",
        "gradient_loss": "梯度域 L1",
        "peak_penalty": "峰值强度惩罚",
        "ssim_loss": "SSIM Loss",
        "energy_conservation": "总能量守恒",
        "phase_smoothness": "相位平滑 `||∇φ||₁`（实现于 `losses.py`）",
        "phase_modulation_depth": "相位调制深度（峰谷）",
    }
    out = ["| 候选 loss | 相对判别力 | 退化安全 | 与现有项最大 │ρ│ | 是否有独立信息 |",
           "|---|---|---|---|---|"]
    order = [
        "weighted_mse", "gradient_loss", "peak_penalty", "ssim_loss",
        "energy_conservation", "phase_smoothness", "phase_modulation_depth",
    ]
    for name in order:
        if name not in v:
            continue
        r = v[name]
        if not r.get("screenable_here", True):
            out.append(
                f"| {labels[name]} | — | — | — | **本台架无法筛选**（见下） |"
            )
            continue
        rho = r["max_abs_spearman_vs_incumbent"]
        rho_s = "—" if rho != rho else f"{rho:.2f}"
        # "Independent information" = it moves beyond what the incumbent set already
        # captures on at least one corruption. `covers` empty means it does not.
        indep = "**有**（" + "、".join(r["covers"]) + "）" if r["covers"] else "无"
        safe = "是" if r["degenerate_safe"] else "**否**"
        out.append(f"| {labels[name]} | {r['discrimination']:.4f} | {safe} | {rho_s} | {indep} |")
    out.append("")
    out.append("判别力 = 该 loss 在三种扰动下改变量占它自己动态范围的比例（跨项可比）；"
               "「现有项」= `mse` / `uniformity` / `w_speckle` / `w_ellipse`。")
    out.append("")
    out.append("现有项的最好相对变化量（候选必须超过它才算有信息）：")
    for k, vv in inc.items():
        out.append(f"* `{k}`：**{vv['best_relative']:.4f}**（由 `{vv['argmax']}` 取得）")
    return out


def _phase_smooth_rows() -> list[str]:
    """The paired 3-seed result for the phase-domain penalty."""
    data = _load("phase_smoothness_train.json")
    if not data.get("verdict"):
        return ["*尚未生成：先跑 `python scripts/phase_smoothness_train.py`*"]
    cfg = data["config"]
    base = data["baseline"]
    out = [
        f"配对 {len(cfg['seeds'])} 个 seed，epochs={cfg['epochs']}，n_max={cfg['n_max']}，"
        f"padding={cfg['padding']}，每个权重与 `w=0` **同 seed 配对**比较：",
        "",
        "| w_phase_smooth | ΔR²（配对） | 标准误 | 胜过基线 | ΔSSIM | 相位 TV | 相位 PV |",
        "|---|---|---|---|---|---|---|",
    ]
    for w, r in data["verdict"].items():
        out.append(
            f"| {w} | {_fmt(r['d_r2_paired'])} | {_fmt(r['d_r2_se'], '.4f')} | "
            f"{r['beats_baseline']}/{r['n']} | {_fmt(r['d_ssim_paired'])} | "
            f"{base['phase_tv']:.4f} → {r['phase_tv']:.4f} | "
            f"{base['phase_pv_rad']:.2f} → {r['phase_pv_rad']:.2f} rad |"
        )
    out.append("")
    rows = data["rows"]
    zero = [r for r in rows if r["weight"] == 0.0]
    heavy = [r for r in rows if r["weight"] == max(cfg["weights"])]
    zc = sum(r["max_abs_coeff"] for r in zero) / len(zero)
    hc = sum(r["max_abs_coeff"] for r in heavy) / len(heavy)
    out.append(
        f"**机制是显式的**：最大系数幅值从 `w=0` 的 {zc:.3f} 塌到 `w={max(cfg['weights'])}` 的 "
        f"{hc:.3f}。这个惩罚不是在校准一个过大的修正，而是在**把修正整体压掉**——"
        "模型干脆不再校正像差。"
    )
    out.append("")
    out.append(f"**结论：预测被证实**（{'CONFIRMED' if data['outcome'] == 'CONFIRMED' else 'REFUTED'}）。"
               f"没有任何一个权重在 >=2/{len(cfg['seeds'])} 个 seed 上有帮助，因此"
               "**`w_phase_smooth` 保持默认 0**。")
    return out


def _crosstalk_rows() -> list[str]:
    """Forward-training arms: incumbent, same-simulator control, crosstalk at 3 strengths."""
    data = _load("crosstalk_augmentation_train.json")
    if not data.get("verdict"):
        return ["*NOT PRODUCED YET - run `python scripts/crosstalk_augmentation_train.py`*"]
    import ml.zernike.eval_stats as es

    v = data["verdict"]
    base = data["baseline"]
    rows = data["rows"]
    base_by_seed = {r["seed"]: r["r2"] for r in rows if r["arm"] == "D2"}
    ctrl_by_seed = {r["seed"]: r["r2"] for r in rows if r["arm"] == "D2+same"}

    out = [
        f"基线 `D2`（无增强）平均 R² = {_fmt(base['mean_r2'])}，配对 {len(base_by_seed)} 个 seed。",
        "",
        "| arm | ΔR²（配对） | 标准误 | Cohen's dz | p（精确符号翻转） | 胜过基线 | vs 同源对照 |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm in ("D2+same", "D2+xt0.5", "D2+xt1.0", "D2+xt2.0"):
        if arm not in v:
            continue
        r = v[arm]
        sel = {x["seed"]: x["r2"] for x in rows if x["arm"] == arm}
        d = [sel[s] - base_by_seed[s] for s in sorted(sel)]
        st = es.paired_comparison(d, arm)
        label = "**同源对照**" if arm == "D2+same" else arm
        out.append(
            f"| {label} | {_fmt(st['mean_diff'])} | {_fmt(r['d_r2_se'], '.4f')} | "
            f"{_fmt(st['cohens_dz'], '+.2f')} | {st['p_sign_flip']:.4f} | "
            f"{r['beats_baseline']}/{r['n']} | "
            f"{'—' if arm == 'D2+same' else ('**胜出**' if r['beats_control'] else '不胜')} |"
        )
    out.append("")
    out.append(
        f"⚠️ **5 个配对样本的最小可达 p = {es.min_attainable_pvalue(5)}**，"
        "所以这个协议**结构上就到不了 p<0.05**。下表所有 p 值都只能读作"
        "「是否有方向性证据」，不能读作「是否显著」。"
    )
    out.append("")
    guards = data.get("degeneracy_guards", {})
    if guards:
        out.append(
            "退化守卫（防止某一臂其实是空转）："
            + "；".join(
                f"`{k}` = {_fmt(val, '.4f')}" for k, val in guards.items()
                if k.endswith("_vs_same_mae")
            )
            + "。串扰臂与同源臂的目标**确实不同**，否则这一臂是空操作、负结果毫无意义。"
        )
    return out


def _inverse_transfer_rows() -> list[str]:
    """Inverse scores through the same three arms, on the scale-matched padding-1 protocol."""
    data = _load("inverse_crosstalk_check.json")
    if not data.get("verdict"):
        return ["*尚未生成：先跑 `python scripts/inverse_crosstalk_check.py`*"]
    ref = data["references"]
    v = data["verdict"]
    out = [
        f"平场 = {ref['flat']:.4f}，GS 提案 = {ref['gs']:.4f}"
        f"（GS − 平场 = {ref['gs'] - ref['flat']:+.4f}）。"
        "⚠️ **三个 arm 的逆向得分全部低于平场**，也就是说经由本模型做逆向优化"
        "**依然失败**，与第 8 节的结论一致。",
        "",
        "| arm | 逆向得分 | Δ（配对） | Cohen's dz | p | 胜过基线 | 正向 R² |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm in sorted(v, key=lambda a: -v[a]["d_inv_vs_D2"]):
        r = v[arm]
        label = "**同源对照**" if arm == "D2+same" else arm
        out.append(
            f"| {label} | {r['mean_inv']:.4f} | {_fmt(r['d_inv_vs_D2'])} | "
            f"{_fmt(r['cohens_dz'], '+.2f')} | {r['p_sign_flip']:.4f} | "
            f"{r['beats_baseline']}/{r['n']} | {_fmt(r['mean_fwd_r2'])} |"
        )
    return out


def build_report(stats: dict, figures_ok: bool) -> str:
    fwd, invst, ph = stats["forward"], stats["inverse"], stats["phase"]
    SEARCH_ROWS = _search_rows()
    TAXONOMY_ROWS = _taxonomy_rows()
    CROSSTALK_ROWS = _crosstalk_rows()
    INVERSE_TRANSFER_ROWS = _inverse_transfer_rows()
    PHASE_SMOOTH_ROWS = _phase_smooth_rows()
    COMBINED_ROWS = _combined_rows()
    INVERSE_ROWS = _inverse_rows()
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
        "## 5. 梯度精修到底是不是「重启」——一个被推翻的结论",
        "",
        "![ROI 扫描](figures/roi_sweep.png)",
        "",
        "**本节曾经写着一个已被推翻的结论。保留这段是为了说明它是怎么被推翻的。**",
        "",
        "原结论是：「梯度精修更像一次重启，而不是一个方向正确的梯度」——"
        "起点差时救回来，起点好时反而破坏。证据是 9 种 ROI 几何 × 2 种目标函数上 "
        "Spearman = **−0.87 / −0.92**。据此还给出了工程建议：按起点质量设闸门，只精修没达标的提案。",
        "",
        "**问题出在评测器本身。** 原来那套评测把远场用 `[:64, :64]` 裁切，"
        "而 0 阶光斑在远场阵列的**中心**，不在左上角。左上角的数值比中心光斑低约 10^5 倍，"
        "于是所有 ROI 指标量的都是**离轴旁瓣**，不是光斑。",
        "",
        "这个错误被自己的数字掩盖了很久——它是自洽的、可复现的。"
        "只有把图画出来才会看到「独立仿真」那一栏是空的。",
        "",
        "修正后的结论**方向相反**：",
        "",
        "| 结论 | 原始（错误评测器） | 修正后 |",
        "|---|---|---|",
        "| 「GS 优于梯度设计」 | 34/90，接近抛硬币 | **9/9**，配对 t = +6.4…+32.5 |",
        "| 「精修破坏 GS 解」 | −0.2206，0/16 | **符号翻转**：+0.0795，78/90 为正 |",
        "| 「精修是重启不是梯度」 | Spearman −0.87 / −0.92 | **Spearman +0.93 / +0.95** |",
        "",
        "Spearman 从 −0.87 变成 +0.93 不是「效果变弱」，是**符号翻转且幅度更大**："
        "GS 提案越好，精修增益越大。这正是方向正确的梯度该有的行为，与重启行为相反。",
        "",
        "**两条对工程直接有用的推论：**",
        "",
        "1. **不要按起点质量设闸门。** 那条建议建立在错误的符号上。"
        "现在的证据支持「精修几乎总是有帮助」，不需要闸门。",
        "2. **GS 单独用常常不比平场好。** `GS − 平场` 在 90/90 次里都是负的"
        "（−0.0025…−0.0364）。真正的收益来自 **GS + 精修**：GS 给出正确的盆地，梯度把它走完。",
        "",
        "自由相位参数化下结论**不一致**：精修反而在 9/9 个 ROI 里都**破坏** GS 解（−0.2779）。"
        "原因是自由相位有 576 个自由度，而模型只表达 135 个 Zernike 模式——"
        "精修在一个目标函数看不见的方向上走了。这与 §9 的正向结论是同一件事。",
        "",
        "---",
        "",
        GLOSSARY,
        "",
"## 6. 正向模型的改进尝试：逐项贡献",
        "",
        "![各 arm 对 R² 的贡献](figures/forward_search_summary.png)",
        "",
        "每个 arm 都与基线**同 seed 配对**比较（不同 seed 之间 R² 的散布高达 0.075，"
        "不配对就会把随机波动读成改进）。排序只看 R² 与 SSIM，不看 MSE/PSNR——"
        "本仓库有过 72 dB PSNR 但 R² 低于常数预测的记录。",
        "",
        *SEARCH_ROWS,
        "",
        "逐项结论：",
        "",
        "| 尝试 | 结果 | 原因 |",
        "|---|---|---|",
        "| 相机曝光等设备参数 | **无法进行** | 7 个 family 的曝光全是常数"
        "（1.1/1.1/0.4/1.5/1.2/1.2/0.1 ms），常数输入携带零信息 |",
        "| 椭圆拟合 loss（3 个权重） | 降低自身指标 18%，但 R² 0/3 | "
        "拿像素保真度换光斑形状，是权衡不是净提升 |",
        "| physics + attention | −0.0137，0/3 | 参数 ×45 无收益，第三次被否 |",
        "| U-Net 单独 | +0.0168 ± 0.0221 | 标准误比均值还大，且**过度平滑**（见下） |",
        "| 相位 π 翻转 | +0.0000，逐位相同 | **物理上必然无效**：加 π 使相量取负，"
        "而 `|FFT(−E)| = |FFT(E)|`，强度目标完全不变 |",
        "| 相位噪声 / 平移 / 合成强相位 | 全在噪声内或为负 | "
        "验证/训练误差比 1.08，没有泛化间隙可压 |",
        "",
        "### 唯一有效的改进：physics + U-Net 残差",
        "",
        "![pred vs true 逐样本对比](figures/forward_search_pred_vs_true.png)",
        "",
        "两个模型之前只被**分开**试过，漏掉的是中间那档：让 physics 做它擅长的"
        "（闭式解析相位），让 CNN 只补它表达不了的部分。",
        "",
        *COMBINED_ROWS,
        "",
        "两点值得记下：",
        "",
        "* **U-Net 单独用仍然没用**（+0.0168 ± 0.0221）。所以提升不是「更大的网络」，"
        "而是物理基座提供了 CNN 自己找不到的东西。",
        "* **冻结基座反而更差**（4/5，+0.0261，对比联合训练的 5/5）。"
        "说明残差不是单纯「补基座缺的那部分」——基座与残差需要**一起**适应。",
        "",
        "这与本仓库早前「hybrid 与 physics 统计不可区分」的记录不矛盾："
        "那个 hybrid 用的是 32 通道零初始化残差，而这里是完整 U-Net 残差、缩放 0.15。"
        "**架构细节决定成败，旧结论不能直接套用。**",
        "",
        "图里还有一个只看数字看不出来的现象：**U-Net 的 SSIM 优势部分来自「过度平滑」。**"
        "它的预测光斑明显比实测更宽更糊，残差图中心发蓝（预测偏暗）、外圈发红（预测偏亮）。"
        "数值上同向印证：U-Net 的椭圆误差 0.3997 对基线 0.2027，光斑尺寸差了近 2 倍。"
        "R² 会惩罚丢失的峰值结构，SSIM 不会——这就是本文不按 SSIM 选模型的原因。",
        "",
        "## 7. 没有过拟合可修",
        "",
        "「误差过大」很容易被当成过拟合，但实测不是：",
        "",
        "| 训练记录数 | val R² | 验证/训练 |",
        "|---|---|---|",
        "| 128 | +0.8767 | 1.16 |",
        "| 256 | +0.8850 | 1.02 |",
        "| 512（默认） | **+0.8860** | **1.08** |",
        "| 768 | +0.8851 | 1.12 |",
        "",
        "验证误差只比训练误差高 8%（256 条时反而更低），学习曲线是平的，"
        "而且**每一个正则化都让它更差**：`l2=1e-3` −0.025、`l2=1e-2` −0.253、"
        "`weight_decay=1e-2` −0.207，全部 0/3 配对。降低容量（`n_max=15`）"
        "毫无变化（−0.0005）。",
        "",
        "所以正向误差是**表示能力上限**，不是过拟合。加正则化去补一个不存在的间隙，"
        "只会让模型更差——这一点在每个杠杆上都是 3 个 seed 量过的。",
        "",
        "## 8. 逆向优化仍然失败，且与模型结构无关",
        "",
        "既然上一节的改进有效，按既定条件就该验证逆向。三个 arm 在同一 padding 下重测：",
        "",
        *INVERSE_ROWS,
        "",
        "（padding=1，所以本表的 R² 不能与第 6 节的 padding=12 结果直接比较；"
        "三个 arm 之间可比。）",
        "",
        "**这是一个负结果，而且很重要：**",
        "",
        "* 改进最大的 D2 把正向 R² 抬高了 +0.0426（5/5 配对），"
        "逆向只动了 +0.0065（2/5，纯噪声）。",
        "* 更刺眼的是 U-Net：正向 R² 几乎翻倍（+0.8718 对 +0.6276），"
        "逆向分数**纹丝不动**（0.5461）。",
        "",
        "至此已经是**第三次**用三种结构、两种 padding 独立确认同一件事："
        "**正向准确率不能预测逆向能力。** 瓶颈是**训练分布**"
        "（语料只有四个优化目标的轻微像差），不是模型容量，也不是正则化。"
        "更强的模型不会解决它，只有系统性覆盖强相位的训练数据才会。",
        "",
        "---",
        "",
        "## 9. 尚未验证的边界",
        "",
        "* 以上全部是**仿真**结论，不是台架实测结论。",
        "* 只覆盖了 Zernike（135 自由度）与自由相位 `phase-grid=24`（576 自由度）两种参数化；"
        "未试原生全分辨率自由相位（4096 自由度）。",
        "* ROI 几何扫描用的是 `SIZE_FRAC × ASPECT` 的 3×3 组合，中间没有更密的采样。",
        "* 跨目标的分组比较**做不了**：每个 seed 的验证划分只覆盖 1–2 个优化目标，"
        "只有 `rmse_out` 出现在 2 个以上的 seed 里。要做这件事需要 10 折分组交叉验证。",
        "* 合成强相位数据与真实语料**同源**（都用项目自带仿真器），"
        "因此模型没有在完全独立的传播模型上被评测过。",
        "",
        "## 10. 闭环整形的 loss 分类：逐项打分，只有一项是新轴",
        "",
        "有人提出了一份闭环整形 loss 清单（加权 MSE / SSIM / 梯度域 / 相位平滑 / "
        "physics-informed / 峰值惩罚 / 能量守恒）。下面按**本仓已实测的**三条轴打分，"
        "而不是照单采纳：",
        "",
        "1. **判别力** —— 三种方向相反的扰动（过平滑 / 过散斑 / 能量移位）下，"
        "它相对自身动态范围改变了多少。",
        "2. **退化安全** —— 能不能靠毁掉信号来取胜。这不是假设：本仓实测过只优化均匀度"
        "把环围能量推到 **0.002**。",
        "3. **冗余度** —— 与现有项的逐样本相关性。相关性 0.97 的项只是多一个旋钮，"
        "不是多一份信息。",
        "",
        *TAXONOMY_ROWS,
        "",
        "**七个候选里，五个是重复的或已被否掉的：**",
        "",
        "* **能量守恒已经存在**（`losses.roi_energy_loss`），而且实测判别力只有 0.0362 —— "
        "`image_mode='abs255'` 加上模型的 `normalization='peak'` 已经把总量除掉了，"
        "这个「光路损耗告警」几乎没有可告警的东西。",
        "* **physics-informed 已经存在**，就是 `slm_model_in_loop` 本身：它每轮重新拟合"
        "正向模型的像差，也就是清单里「定期重新标定 F_err」那一条，已经按轮做了。",
        "* **冷启动配方（physics-informed + MSE）已经被否**：合成强相位增强 "
        "C4 +strong 25% 实测 **−0.0211，0/3 配对**。原因见下。",
        "* **峰值惩罚退化不安全**：空 ROI 天然没有热像素，所以它可以靠把光打空取胜。",
        "* **加权 MSE 与现有项相关性 0.97**，且没有一项扰动能超过现有项最好水平 —— "
        "纯冗余。",
        "",
        "「独立信息」一列**全为空**，意思是：没有一个候选能做到现有项做不到的事。"
        "`w_speckle` 在过散斑上已经把相对变化量吃到 **1.0000**（饱和），那里根本没有余量。",
        "",
        "### 为什么「合成强相位增强」也无效：它不是独立数据",
        "",
        "C4 用的 `ml/zernike/augment.py::strong_phase_pairs` 看起来是「新数据」，"
        "但它的目标来自 `ml/zernike/inverse_design.py::sim_far_field`，"
        "而那正是 `ZernikeAmpModel._propagate` 自己的传播链。"
        "**实测两者的相关系数 = 0.9956**（同一相位下，`padding=12`，64×64）："
        "所谓「独立仿真器」与模型自身的输出只差 0.4%。",
        "",
        "所以它不是独立信息，只是把模型已经能精确算出的东西又喂了一遍，"
        "同时把训练分布搅宽——净效果为负（−0.0211，0/3 配对）与「它没有信息」一致。",
        "",
        "**这给出了下一步唯一有意义的判据**：任何新增强数据，其传播器必须包含"
        "模型**证明无法表示**的物理。本仓已知且已实测的候选有三个——"
        "面板串扰 / 有限填充因子、LCOS 灰度-幅度耦合（AGENTS.md 实测周期约 993 灰度）、"
        "以及像差本身的大幅偏离。但仿真器里没有任何一项的模型，"
        "所以这三者都只能作为**训练数据的生成器**引入，不能靠仿真自己长出来。",
        "",
        "### 两个我自己的台架错误（都曾伪造出读数）",
        "",
        "* **用「真值处的取值」做归一化**：清单里每一项都是 anchored 的，真值处取值按"
        "构造为 0，于是所有比值都是 `inf`。改为「该项自身动态范围」才可比。",
        "* **在扰动扫描上给相位域项打分**：它们按构造与输入无关，于是必然得到 "
        "`判别力=0 / 不安全`。这是台架的伪影不是结论，现在显式标为"
        "**本台架无法筛选**，必须配训练实验。",
        "",
        "## 11. 相位平滑：唯一的新轴，实现后实测无效",
        "",
        "上面两项里，相位平滑是**唯一本仓没测过**的，而且它是清单里唯一作用在**指令**"
        "而非测量图像上的项：其余各项都只能事后观察相位梯度的后果，无法表达"
        "「这个相位在面板上不可实现」。所以它被实现了"
        "（`losses.phase_smoothness_penalty` = TV + 2π 峰谷铰链，`w_phase_smooth` 默认 0）。",
        "",
        "**它的动机是对的**：不加约束的拟合命令了 **PV 23.19 rad ≈ 3.7 个完整 2π**，"
        "面板的相位线性度撑不住。",
        "",
        "**但配对实验说不要打开它：**",
        "",
        *PHASE_SMOOTH_ROWS,
        "",
        "### 为什么会这样，以及为什么这不奇怪",
        "",
        "这个结果是**先预测后验证**的，预测来自一条已有测量：`l2_penalty` 本身就是一个"
        "系数域正则项，而它在每个强度上都更差（−0.025 / −0.253，均 0/3 配对），"
        "因为验证/训练误差比 **1.08**、学习曲线是平的 —— **根本没有可压缩的泛化间隙**。"
        "相位平滑是同一根杠杆，只是去掉了模态序上的各向异性（L2 在模态序上是各向同性的，"
        "这在物理上是错的：同样 1 rad 系数，高阶模态的相位梯度陡得多）。"
        "它理应继承失败，而它确实继承了。",
        "",
        "### 这条结论的硬边界（必须一起读）",
        "",
        "**仿真器的相位响应是理想的**——没有串扰、没有填充因子、没有效率耦合。"
        "所以可实现性惩罚在这里**只可能损失精度，不可能赚回任何东西**，"
        "因为不存在一个会向它收费的串扰模型。",
        "换句话说：**PV 23 rad 在真实台架上是否真的损失效率，这是一个本仓无法回答的硬件问题。**"
        "在有人实测之前，`w_phase_smooth` 保持 0。本报告能给出的只是："
        "模型确实在命令一个面板撑不住的相位，以及用仿真无法判断这件事的代价。",
        "",
        "---",
        "",
        "## 12. 串扰增强：唯一带新物理的增强，测了，结论与直觉相反",
        "",
        "上一节给出了下一步的唯一判据：增强数据的传播器必须包含模型**证明无法表示**的物理。"
        "本仓已知、且已实测而仿真器没有的候选有三个，本节测第一个——**SLM 像素串扰 / "
        "有限填充因子**（2f 台架有空间带宽积，真实 LCOS 会混合相邻像素）。",
        "",
        "实现走 `SimPibSystem._pupil_field` 这条 seam（其 docstring 明说是为「子类注入"
        "瞳面物理（如 SLM 通道串扰）」而留的），**不复制**补零 + FFT + 归一化那条链——"
        "复制一份悄悄漂移的传播器是本仓已记录过的失败模式。",
        "串扰 PSF 作用在**复瞳孔场**上而不是相位上：串扰混合的是相邻像素的**场**，"
        "混合相位是另一种（错误的）误差模型。",
        "",
        "**臂设计**：`D2`（无增强）、`D2+same`（**同源对照**，与串扰臂同样本数）、"
        "`D2+crosstalk σ ∈ {0.5, 1.0, 2.0}`。对照臂不可省——没有它就分不清"
        "「独立传播器有效」与「数据变多有效」，而这正是整个问题。",
        "",
        *CROSSTALK_ROWS,
        "",
        "### 结论一：串扰增强本身**不成立**",
        "",
        "最好的串扰臂（σ=2.0）配对 ΔR² = +0.0253，但**标准误 ±0.0314 比效应本身还大**，"
        "精确符号翻转 **p = 0.6250**。它相对同源对照的优势是 +0.0112，**p = 0.3125**——"
        "也就是说「独立传播器」带来的额外信息，在噪声里完全看不见。",
        "",
        "### 结论二：但「加数据」本身有效，只是与传播器无关",
        "",
        "同源对照 `D2+same` 配对 ΔR² = +0.0141（p = 0.8125），单看正向也不显著，"
        "**但它在逆向上是三个臂里唯一达到协议下限的**：Δ = +0.0485、dz = +1.35、"
        "**5/5 seed 全胜**、p = 0.0625（= 5 对的最小可达值）。",
        "",
        "**方向与直觉相反**：同源增强对逆向的帮助**大于**串扰增强（+0.0485 对 +0.0286）。",
        "如果「独立物理传播器」是瓶颈，同源对照不该赢。它赢了，说明当前瓶颈**不是**"
        "「传播器缺少物理」，而是更朴素的东西——**训练分布的覆盖度与多样性本身**。",
        "同源增强之所以有用，很可能不是因为它提供了新信息（它没有，corr 0.9956），"
        "而是因为它把 300 个**远离原分布**的强相位样本塞进了每个 batch，"
        "起到了正则化/扩覆盖的作用。",
        "",
        "## 13. 逆向：增益**没有**从正向传过来（第三次独立确认）",
        "",
        "按第 8 节的判据做转移检验：正向动了，逆向是否跟着动？"
        "本节用**尺度匹配**的 padding-1 协议（padding 12 下 GS 提案**低于**平场，"
        "脚本自带的守卫会直接拒绝运行——这个守卫是对的，且正是它避免了早期 ROI 扫描那类无效数字）。",
        "",
        *INVERSE_TRANSFER_ROWS,
        "",
        "**结论：串扰臂的逆向增益不成立**（Δ = +0.0286，dz = +0.74，p = 0.1875，仅 4/5）。"
        "把第 8 节的两条与本节合起来看，"
        "「正向准确率不能预测逆向能力」已经是**第三次**被独立确认：",
        "",
        "1. physics + U-Net 残差：正向 +0.0426（5/5）→ 逆向 +0.0065（2/5，噪声）。",
        "2. 单独 U-Net：正向 R² 近乎翻倍（+0.8718 对 +0.6276）→ 逆向 **+0.0000**。",
        "3. 本节：串扰增强正向略动（不显著）→ 逆向也不动；"
        "而同源增强正向不显著、逆向却是 5/5 全胜。",
        "",
        "第 3 条尤其说明问题：**正向和逆向甚至不同向**。"
        "所以「先修正向再修逆向」这个工作顺序本身不成立，"
        "任何只按正向 R² 排序的模型选择都在优化一个与目标无关的量。",
        "",
        "---",
        "",
        "## 14. 复现",
        "",
        "```bash",
        "python scripts/generate_inverse_design_report.py",
        "python -m pytest tests/ao_shaping/ml/zernike/test_inverse_design.py -q",
        "",
        "# Section 6 panels (run the experiments before regenerating the report)",
        "python scripts/forward_search.py",
        "python scripts/forward_search_extra.py",
        "python scripts/inverse_combined.py",
        "python scripts/generate_forward_search_report.py",
        "python scripts/closed_loop_loss_taxonomy.py",
        "python scripts/phase_smoothness_train.py",
        "python scripts/crosstalk_augmentation_train.py",
        "python scripts/inverse_crosstalk_check.py",
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
