"""Generate the illustrated Chinese report for the Zernike far-field model.

**Fully offline.** Reads only saved artefacts -- the sweep JSON and the training
history -- and never opens a camera or SLM, so the report can be regenerated
while the bench is powered down.

    python scripts/generate_zernike_amp_report.py
    python scripts/generate_zernike_amp_report.py --no-figures

Inputs
------
``logs/zernike_amp_sweep.json``
    Grouped-CV sweeps and the final comparison, written by
    ``scripts/sweep_zernike_models.py``.
``logs/zernike_amp_final/summary.json``
    Per-epoch training history (the same series the run logged to wandb).
``logs/zernike_amp_final/compare_epoch*.png``
    true-vs-prediction frames rendered during training.

Outputs ``docs/zernike_amp/report.md`` plus ``docs/zernike_amp/figures/*.png``.

Two conventions worth stating because they are easy to get wrong:

* the sweep is **grouped by source pickle** -- the corpus holds four optimisation
  objectives whose image distributions differ sharply, so an ungrouped split
  measures the objective mixture rather than the model;
* every interval quoted below is a **paired** per-fold difference tested with an
  exact sign-flip permutation test. The between-fold spread is ~0.08 R^2 while the
  paired spread is ~0.01, so an unpaired comparison cannot resolve the effects
  being reported.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # before pyplot, per repo convention

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts._common import markdown_table, savefig  # noqa: E402

# CJK glyphs: without this every Chinese label renders as a tofu box.
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FIGURES = ROOT / "docs" / "zernike_amp" / "figures"
MODELS = ("physics", "hybrid", "unet")
LABELS = {
    "physics": "物理模型 physics",
    "hybrid": "混合 hybrid",
    "unet": "U-Net",
}
COLOURS = {"physics": "#2f6f9f", "hybrid": "#c48a2e", "unet": "#a63d5a"}
METRIC_LABEL = {"r2": "R²", "ssim": "SSIM", "psnr": "PSNR (dB)", "mse": "MSE", "nrmse": "NRMSE"}


def _load(path: Path):
    if not path.exists():
        raise SystemExit(f"missing input artefact: {path}\n"
                         "run scripts/sweep_zernike_models.py first")
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig_training_curves(history_path: Path) -> str:
    """Loss and validation quality against epoch -- the training-history figure."""
    data = json.loads(history_path.read_text(encoding="utf-8"))
    history = data["history"]
    epoch = [row["epoch"] for row in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.plot(epoch, [r["train_mse"] for r in history], "o-", label="train MSE", color="#2f6f9f")
    ax.plot(epoch, [r["val_mse"] for r in history], "s-", label="val MSE", color="#a63d5a")
    ax.axvline(data["best_epoch"], ls="--", c="grey", lw=1, label=f"best epoch {data['best_epoch']}")
    ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE (log)")
    ax.set_title("损失：训练 vs 验证")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epoch, [r["val_r2"] for r in history], "o-", label="val R²", color="#2f6f9f")
    ax.plot(epoch, [r["val_ssim"] for r in history], "s-", label="val SSIM", color="#c48a2e")
    ax.axhline(0.0, ls=":", c="grey", lw=1)
    ax.axvline(data["best_epoch"], ls="--", c="grey", lw=1)
    ax.set_xlabel("epoch"); ax.set_ylabel("指标")
    ax.set_title(f"验证质量（best R² = {data['best_val_r2']:.4f}）")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(epoch, [r["grad_total"] for r in history], "o-", label="grad 总量", color="#2f6f9f")
    ax.plot(epoch, [r["grad_max"] for r in history], "s-", label="grad 逐模最大", color="#a63d5a")
    ax.plot(epoch, [r["coef_abs_max"] for r in history], "^--", label="max|Z|", color="#c48a2e")
    ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("量级（log）")
    ax.set_title(f"梯度与系数范数（死模 {data['dead_modes']}/{data['n_modes']}）")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(
        f"物理模型训练历史（n_max={data['n_max']}, K={data['n_modes']}, "
        f"{data['seconds']:.1f}s, grad_norm {data['grad_norm_first']:.2e} → {data['grad_norm_last']:.2e}）",
        fontsize=11,
    )
    fig.tight_layout()
    path = FIGURES / "01_training_curves.png"
    savefig(fig, path)
    return path.name


def fig_physics_grid(sweep: dict) -> str:
    """The joint (n_max, lr) grid -- shows the non-additivity directly."""
    entries = sweep["physics_grid"]
    lrs = sorted({e["lr"] for e in entries})
    n_maxes = sorted({e["n_max"] for e in entries})
    lookup = {(e["n_max"], e["lr"]): e["summary"] for e in entries}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, metric in zip(axes, ("r2", "ssim")):
        width = 0.38
        xs = np.arange(len(n_maxes))
        for i, lr in enumerate(lrs):
            heights = [
                lookup.get((n, lr), {}).get(metric, {}).get("mean", np.nan)
                for n in n_maxes
            ]
            ax.bar(xs + (i - (len(lrs) - 1) / 2) * width, heights, width,
                   label=f"lr = {lr}", color=plt.cm.viridis(0.25 + 0.5 * i / max(1, len(lrs) - 1)))
        ax.set_xticks(xs)
        ax.set_xticklabels([f"n_max={n}\n(K={(n + 1) * (n + 2) // 2 - 1})" for n in n_maxes])
        if metric == "r2":
            ax.set_ylim(0.84, 0.895)
        else:
            ax.set_ylim(0.68, 0.79)
        ax.set_ylabel(METRIC_LABEL[metric])
        ax.set_title(f"物理模型：({metric_label_cn(metric)}) 随 (n_max, lr) 变化")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("联合扫描才有意义 —— lr=0.02 单独用无效，配 n_max=20 才是最优点", fontsize=11)
    fig.tight_layout()
    path = FIGURES / "02_physics_joint_grid.png"
    savefig(fig, path)
    return path.name


def metric_label_cn(metric: str) -> str:
    return {"r2": "R²", "ssim": "SSIM", "psnr": "PSNR", "mse": "MSE", "nrmse": "NRMSE"}[metric]


def fig_n_max_curve(sweep: dict) -> str:
    """n_max saturation -- the physics model's only large lever, and where it dies."""
    entries = [e for e in sweep["physics_grid"] if e["lr"] == 0.01]
    entries.sort(key=lambda e: e["n_max"])
    modes = [(e["n_max"] + 1) * (e["n_max"] + 2) // 2 - 1 for e in entries]
    r2 = [e["summary"]["r2"]["mean"] for e in entries]
    r2s = [e["summary"]["r2"]["std"] for e in entries]
    ssim = [e["summary"]["ssim"]["mean"] for e in entries]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.errorbar(modes, r2, yerr=r2s, fmt="o-", capsize=3, color="#2f6f9f",
                label="R²（10 折均值 ± 标准差）")
    ax.set_xlabel("Zernike 系数个数 K"); ax.set_ylabel("R²", color="#2f6f9f")
    ax.tick_params(axis="y", labelcolor="#2f6f9f")
    ax.set_xscale("log")
    twin = ax.twinx()
    twin.plot(modes, ssim, "s--", color="#c48a2e", label="SSIM")
    twin.set_ylabel("SSIM", color="#c48a2e")
    twin.tick_params(axis="y", labelcolor="#c48a2e")
    ax.set_title("n_max 已饱和：系数 ×6.4 只换来 R² +0.010（折间 σ≈0.08）")
    ax.grid(alpha=0.3)
    lines = [*ax.get_lines(), *twin.get_lines()]
    labels = [str(line.get_label()) for line in lines]
    ax.legend(lines, labels, fontsize=8, loc="lower right")
    fig.tight_layout()
    path = FIGURES / "03_n_max_saturation.png"
    savefig(fig, path)
    return path.name


def fig_unet_grid(sweep: dict) -> str:
    """U-Net candidate re-verification -- confirms the pick was not noise-fitted."""
    entries = sweep["unet_grid"]
    labels = [f"w{e['features'][0]}\nep{e['epochs']}\nlr{e['lr']}" for e in entries]
    r2 = [e["summary"]["r2"]["mean"] for e in entries]
    r2s = [e["summary"]["r2"]["std"] for e in entries]
    ssim = [e["summary"]["ssim"]["mean"] for e in entries]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    xs = np.arange(len(entries))
    ax.bar(xs - 0.2, r2, 0.4, yerr=r2s, capsize=3, color="#a63d5a", label="R²")
    ax.set_ylabel("R²", color="#a63d5a")
    ax.set_ylim(0.84, 0.93)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8)
    twin = ax.twinx()
    twin.bar(xs + 0.2, ssim, 0.4, color="#c48a2e", label="SSIM")
    twin.set_ylabel("SSIM", color="#c48a2e")
    twin.set_ylim(0.78, 0.88)
    ax.set_title("U-Net 候选复验：原选择仍最优，全部候选差异远在折间 σ 内")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = FIGURES / "04_unet_grid.png"
    savefig(fig, path)
    return path.name


def fig_per_fold(sweep: dict) -> str:
    """Per-fold scores for the three models -- the paired design made visible."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    folds = sweep["folds"]
    xs = np.arange(folds)
    for ax, metric in zip(axes, ("r2", "ssim")):
        for name in MODELS:
            values = sweep["final"][name]["summary"][metric]["per_fold"]
            ax.plot(xs, values, "o-", label=LABELS[name], color=COLOURS[name], alpha=0.85)
        ax.set_xticks(xs)
        ax.set_xlabel("留出折（按源 pickle 分组）")
        ax.set_ylabel(METRIC_LABEL[metric])
        ax.set_title(f"每折 {METRIC_LABEL[metric]}（三个模型共用同一批折）")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("同一批折上的逐折得分：折间波动远大于模型间差距，故必须配对检验", fontsize=11)
    fig.tight_layout()
    path = FIGURES / "05_per_fold_scores.png"
    savefig(fig, path)
    return path.name


def fig_paired(sweep: dict) -> str:
    """Paired differences with the exact-test p-value -- the headline evidence.

    One panel per metric: PSNR is in dB and its scale is ~100x the R^2 / SSIM
    range, so a shared axis would flatten the two metrics that matter most.
    """
    pairs = [("physics", "unet"), ("physics", "hybrid"), ("hybrid", "unet")]
    metrics = ["r2", "ssim", "psnr"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    xs = np.arange(len(pairs))
    for ax, metric in zip(axes, metrics):
        means = [sweep["paired"][f"{a}_minus_{b}"][metric]["mean_diff"] for a, b in pairs]
        sds = [sweep["paired"][f"{a}_minus_{b}"][metric]["std_diff"] for a, b in pairs]
        pvals = [sweep["paired"][f"{a}_minus_{b}"][metric]["p_signflip"] for a, b in pairs]
        colours = [COLOURS[b] for _, b in pairs]
        ax.bar(xs, means, 0.55, yerr=sds, capsize=4, color=colours, alpha=0.85)
        span = max(abs(np.array(means)) + np.array(sds)) * 1.55 or 1.0
        ax.set_ylim(-span, span)
        for x, mean, sd, p in zip(xs, means, sds, pvals):
            star = "***" if p < 0.01 else ("*" if p < 0.05 else "ns")
            ax.text(x, mean + (span * 0.06 if mean >= 0 else -span * 0.06),
                    f"{mean:+.3f}\n{star}", ha="center",
                    va="bottom" if mean >= 0 else "top", fontsize=8)
        ax.axhline(0, c="black", lw=1)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{LABELS[a]}\n− {LABELS[b]}" for a, b in pairs], fontsize=8)
        ax.set_title(METRIC_LABEL[metric])
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(
        "配对差值 ± 标准差（同一折上相减）；*** p<0.01, * p<0.05，ns 不显著"
        "（精确 sign-flip 置换检验，2^10 = 1024 次枚举）",
        fontsize=11,
    )
    fig.tight_layout()
    path = FIGURES / "06_paired_differences.png"
    savefig(fig, path)
    return path.name


def fig_per_objective(sweep: dict) -> str:
    """Per-objective breakdown -- the root cause of the original noise floor."""
    objectives = sorted(sweep["final"]["physics"]["summary"]["by_objective"])
    fold_counts = {
        objective: sum(
            1 for f in sweep["final"]["physics"]["folds"] if objective in f["by_objective"]
        )
        for objective in objectives
    }
    xs = np.arange(len(objectives))
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    width = 0.26
    for i, name in enumerate(MODELS):
        values = [
            sweep["final"][name]["summary"]["by_objective"].get(o, {}).get("r2", np.nan)
            for o in objectives
        ]
        ax.bar(xs + (i - 1) * width, values, width,
               label=LABELS[name], color=COLOURS[name], alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{o}\n({fold_counts[o]} 折)" for o in objectives], fontsize=9)
    ax.set_ylabel("R²")
    ax.set_title("按优化目标分解：单目标折内三个模型共用同一批折")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = FIGURES / "07_per_objective.png"
    savefig(fig, path)
    return path.name


def fig_objective_mixture(index_cache: str) -> str | None:
    """The root-cause heatmap: one objective's mean image vs another's targets.

    This is why the original single split was unreadable -- if the objectives'
    image distributions were interchangeable, a random validation fold would be
    harmless.
    """
    from collections import defaultdict

    from ml.hwdataset import HwPhaseImageDataset, MaterialiserConfig, build_hw_index
    from compare_models_cv import objective_of

    try:
        index = build_hw_index(index_cache=index_cache).filter(families=["slm_zernike_shaping"])
    except Exception as exc:  # pragma: no cover - corpus may be absent offline
        print(f"skip objective-mixture figure: {exc}")
        return None
    records = list(index.records)
    dataset = HwPhaseImageDataset(index, config=MaterialiserConfig(grid=64), use_cache=True)

    groups: dict[str, list[int]] = defaultdict(list)
    for i, record in enumerate(records):
        groups[objective_of(record.path)].append(i)

    # Peak-normalise exactly as the model pipeline does, otherwise this figure
    # describes a target distribution the model never sees.
    means, variances = {}, {}
    for name, positions in groups.items():
        stack = torch.stack([dataset[p]["image"][0] for p in positions]).flatten(1)
        stack = stack / stack.amax(dim=1, keepdim=True).clamp_min(1e-12)
        means[name] = stack.mean(0)
        # SS_tot for predicting group `b` by a constant: sum of squared deviations
        # of b's own samples from b's mean. Using any *other* group's spread here
        # is dimensionally inconsistent and silently produces a much larger number.
        variances[name] = float(stack.var(0).sum())
    names = sorted(groups)
    matrix = np.full((len(names), len(names)), np.nan)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            ss_res = float(((means[a] - means[b]) ** 2).sum())
            matrix[i, j] = 1.0 - ss_res / variances[b]

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1.75, vmax=1.0)
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(names)), names, fontsize=9)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{matrix[i, j]:+.2f}", ha="center", va="center", fontsize=9,
                    color="white" if matrix[i, j] > 0.93 else "black")
    ax.set_title("用 A 目标的均值图预测 B 目标的 R²\n对角线 = 1；离对角越低 = 两者分布差异越大")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    path = FIGURES / "08_objective_mixture.png"
    savefig(fig, path)
    return path.name


def fig_true_vs_pred(compare_png: Path | None) -> str | None:
    """Copy the training-time true-vs-prediction frame into the figures directory."""
    if compare_png is None or not compare_png.exists():
        return None
    target = FIGURES / "09_true_vs_pred.png"
    target.write_bytes(compare_png.read_bytes())
    return target.name


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def final_table(sweep: dict) -> str:
    params = {"physics": 230, "hybrid": 10408, "unet": 7778465}
    rows = []
    for name in MODELS:
        s = sweep["final"][name]["summary"]
        rows.append([
            LABELS[name], f"{params[name]:,}",
            f"{s['r2']['mean']:+.4f} ± {s['r2']['std']:.4f}",
            f"{s['ssim']['mean']:.4f} ± {s['ssim']['std']:.4f}",
            f"{s['psnr']['mean']:.2f} ± {s['psnr']['std']:.2f}",
            f"{s['nrmse']['mean']:.4f}",
        ])
    return markdown_table(
        ["模型", "参数量", "R² (10 折)", "SSIM (10 折)", "PSNR dB (10 折)", "NRMSE"], rows
    )


def paired_table(sweep: dict) -> str:
    rows = []
    for a, b in (("physics", "unet"), ("physics", "hybrid"), ("hybrid", "unet")):
        entry = sweep["paired"][f"{a}_minus_{b}"]
        for metric in ("r2", "ssim", "psnr"):
            s = entry[metric] if metric in entry else None
            if s is None:
                continue
            star = "***" if s["p_signflip"] < 0.01 else ("*" if s["p_signflip"] < 0.05 else "ns")
            rows.append([
                f"{LABELS[a]} − {LABELS[b]}", METRIC_LABEL[metric],
                f"{s['mean_diff']:+.4f}", f"{s['std_diff']:.4f}",
                f"{s['cohens_dz']:+.2f}", f"{s['p_signflip']:.4f}", star,
            ])
    return markdown_table(
        ["配对比较", "指标", "差值均值", "差值标准差", "Cohen's d_z", "p (精确)", "判定"], rows
    )


def grid_table(sweep: dict, key: str) -> str:
    rows = []
    for e in sweep[key]:
        s = e["summary"]
        if key == "physics_grid":
            label = f"n_max={e['n_max']}, lr={e['lr']}"
            size = str((e["n_max"] + 1) * (e["n_max"] + 2) // 2 - 1)
        else:
            label = f"宽度{e['features'][0]}, {e['epochs']} ep, lr={e['lr']}"
            size = "—"
        rows.append([
            label, size,
            f"{s['r2']['mean']:+.4f} ± {s['r2']['std']:.4f}",
            f"{s['ssim']['mean']:.4f}",
        ])
    header = ["配置", "K", "R²", "SSIM"] if key == "physics_grid" else ["配置", "—", "R²", "SSIM"]
    return markdown_table(header, rows)


def fig_phase_synthesis(synthesis: dict, out_dir: Path) -> dict[str, str | None]:
    """Inversion figures: the synthesised phase, both surrogates' spots, convergence."""
    import torch

    phase = np.load(out_dir / "phase_rad.npy")
    predicted = np.load(out_dir / "predicted_spot.npy")
    cross = synthesis.get("cross_check_unet") or {}
    unet_spot = np.asarray(cross["prediction"], dtype=np.float64) if "prediction" in cross else None
    history = synthesis.get("history", [])
    side = synthesis.get("target_side", 50)
    grid = synthesis.get("grid", 64)
    start = (grid - side) // 2
    window = (slice(start, start + side), slice(start, start + side))

    panels = []
    if unet_spot is not None:
        panels = [
            ("合成相位 φ (raw rad)", phase, "twilight", None),
            ("物理代理预测光斑", predicted, "inferno", window),
            ("独立 U-Net 代理所见", unet_spot, "inferno", window),
        ]
    else:
        panels = [
            ("合成相位 φ (raw rad)", phase, "twilight", None),
            ("物理代理预测光斑", predicted, "inferno", window),
        ]
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(4.1 * (len(panels) + 1), 4.0))
    for ax, (title, image, cmap, box) in zip(axes, panels):
        ax.imshow(image, cmap=cmap)
        if box is not None:
            ax.add_patch(plt.Rectangle(
                (box[0].start, box[1].start), side, side,
                fill=False, edgecolor="cyan", lw=1.4,
            ))
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    ax = axes[-1]
    if history:
        epochs = [h["epoch"] for h in history]
        ax.plot(epochs, [h["cv"] for h in history], "o-", color="#2f6f9f", label="CV（物理代理）")
        ax.set_xlabel("优化轮数"); ax.set_ylabel("CV", color="#2f6f9f")
        ax.tick_params(axis="y", labelcolor="#2f6f9f")
        twin = ax.twinx()
        twin.plot(epochs, [h["quality"] for h in history], "s--", color="#c48a2e", label="quality")
        twin.set_ylabel("quality", color="#c48a2e")
        twin.tick_params(axis="y", labelcolor="#c48a2e")
        ax.set_title("收敛（评判用 canonical 指标）", fontsize=10)
        lines = [*ax.get_lines(), *twin.get_lines()]
        ax.legend(lines, [str(l.get_label()) for l in lines], fontsize=7, loc="upper right")
    fig.suptitle(
        f"固定预测网络权重、把 phase 当变量：目标 {side}×{side} px 均匀方斑", fontsize=11
    )
    fig.tight_layout()
    path = FIGURES / "10_phase_synthesis.png"
    savefig(fig, path)
    return {"phase_synthesis": path.name}


def synthesis_table(synthesis: dict) -> str:
    """Flat / synthesised / cross-check, with the repo's own square-shaping refs."""
    opt = synthesis["optimised"]
    flat = synthesis["baseline_flat"]
    cross = synthesis.get("cross_check_unet") or {}
    rows = [
        ["平相位（基线）", "—", f"{flat['uniformity_cv']:.4f}",
         f"{flat['encircled_energy']:.4f}", f"{flat['quality']:.4f}", "物理代理"],
        ["**合成相位**", "—", f"**{opt['uniformity_cv']:.4f}**",
         f"{opt['encircled_energy']:.4f}", f"**{opt['quality']:.4f}**", "物理代理"],
    ]
    if cross:
        rows.append([
            "同一相位，换一个代理看", "—", f"{cross['uniformity_cv']:.4f}",
            f"{cross['encircled_energy']:.4f}", f"{cross['quality']:.4f}", "独立 U-Net",
        ])
        rows.append([
            "同一代理看平相位", "—", f"{cross.get('flat_cv', float('nan')):.4f}", "—",
            f"{cross.get('flat_quality', float('nan')):.4f}", "独立 U-Net",
        ])
    rows += [
        ["仓库参照：GS 单次", "—", "0.41", "—", "—", "数值仿真"],
        ["仓库参照：GS + 自由相位细化", "—", "0.12", "—", "—", "数值仿真"],
    ]
    return markdown_table(["方案", "参数量", "CV", "EE", "quality", "评判者"], rows)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def build_report(
    sweep: dict, history: dict, figures: dict, synthesis: dict | None = None
) -> str:
    paired_pu = sweep["paired"]["physics_minus_unet"]
    paired_ph = sweep["paired"]["physics_minus_hybrid"]
    paired_hu = sweep["paired"]["hybrid_minus_unet"]
    s_final = sweep["final"]["physics"]["summary"]
    s_unet = sweep["final"]["unet"]["summary"]
    s_hybrid = sweep["final"]["hybrid"]["summary"]
    configs = sweep.get("configs", {})

    def star(p: float) -> str:
        return "***" if p < 0.01 else ("*" if p < 0.05 else "ns")

    fig_ref = lambda name: f"figures/{name}" if name else ""  # noqa: E731

    hybrid_verdict = (
        f"在物理模型的最优 lr 下，hybrid 反而**显著更差**"
        f"（R² {paired_ph['r2']['mean_diff']:+.4f}，p = {paired_ph['r2']['p_signflip']:.4f}）；"
        f"而在较温和的 lr=0.01 下它与 physics 统计不可区分。"
        f"零初始化残差只保证**起点**与 physics 相同，并不保证提高 lr 后仍然中性 ——"
        f"涨参数量的是残差网络，它才是被大 lr 破坏的那部分。"
    )

    # Section 10 numbers come from the inversion run, not from this module.
    s_side = synthesis["target_side"] if synthesis else 50
    s_opt = (synthesis or {}).get("optimised", {})
    s_flat = (synthesis or {}).get("baseline_flat", {})
    s_cross = (synthesis or {}).get("cross_check_unet") or {}
    s_cv = s_opt.get("uniformity_cv", float("nan"))
    s_flat_cv = s_flat.get("uniformity_cv", float("nan"))
    s_ee = s_opt.get("encircled_energy", float("nan"))
    s_x_cv = f"{s_cross['uniformity_cv']:.4f}" if s_cross else "未做"
    s_x_flat_cv = f"{s_cross['flat_cv']:.4f}" if s_cross else "未做"
    s_gap = (s_cross["uniformity_cv"] / s_cv) if s_cross and s_cv else float("nan")
    s_ood = (synthesis or {}).get("extrapolation", {})
    s_ood_ours = s_ood.get("zernike_energy_fraction", float("nan"))
    s_ood_c = s_ood.get("corpus_zernike_fraction_mean", float("nan"))
    s_ood_hf = s_ood.get("high_frequency_fraction", float("nan"))
    s_ood_n = s_ood.get("corpus_samples", 0)

    return f"""# Zernike 远场模型 —— 开发报告

> 生成脚本 `scripts/generate_zernike_amp_report.py`（**全离线**，只读已保存产物，不需要硬件）。
> 图表数据来自 `logs/zernike_amp_sweep.json`（10 折分组交叉验证扫描）与
> `logs/zernike_amp_final/summary.json`（训练历史，即当时上传 wandb 的那一组序列）。
> 重新生成：`python scripts/generate_zernike_amp_report.py`

本报告的所有结论都建立在一个前提上：**分组交叉验证 + 配对检验**。第 5、6 节解释为什么
必须如此 —— 它是本项目四次结论反转的根源，其中两次来自同一个错误：
**信任了一个未先定位来源的方差估计。**

---

## 1. 模型是什么

`src/ml/zernike/models.py` 实现了一个可微的瞳孔 → 远场前向模型，唯一可训练参数是一个
**全局** Zernike 向量 `Z`：

```
U      = (phase_cos + i·phase_sin) · exp(i · Σ_k Z_k B_k)      # 复数域
focal  = fftshift(fft2(ifftshift(center_pad(U)), norm="ortho"))
loss   = MSE(observable(focal), ccd)
```

- `Z` 是**被训练**的量；`n_max`（系数个数 `K`）是**超参数**。
- 基底 `B` 由规范的 `ZernikeGenerator` 一次性构建，本项目不重复推导任何 Zernike 数学。
- 前向传播**不调用 `atan2`**，全程留在复数域：语料相位已被 writer 卷到 `[0, 2π)`，
  角度表示会在 ±π 处引入分支切口。
- **排除 piston**：`K = (n_max+1)(n_max+2)/2 − 1`。因为 `|FFT(e^{{iφ₀}}U)| ≡ |FFT(U)|`
  恒成立，piston 系数的梯度**永远**为零。

## 2. 数据管线

`src/ml/hwdataset` 提供 `(phase_cos, phase_sin, image, exposure_log10, …)`。三个关键决定：

| 决定 | 理由 |
|---|---|
| 相位用相干 `(cos, sin)` 对 | 语料卷绕到 `[0, 2π)`；`atan2` 会在输入面上放一个分支切口 |
| 目标**不做** peak 归一（`abs255`） | 曝光是模型输入，绝对亮度本身就是编码它的信号 |
| FOV 以 `fov_px` 随样本返回，绝不 resize | 各 family 相机窗口真的不同（64/248/320/1944 px），插值会伪造像素 |

磁盘缓存与直接路径**逐位一致**（1010/1010 记录），并快 **317×**（127.0 s → 0.4 s，
打开的 pickle 数 10 → 0）。

## 3. 三个靠测量发现的陷阱

**3.1 尺度不匹配（最关键）。** `_anchored_window` 取的是以 0 阶为中心的 `grid×grid`
窗口，**不缩放**。若 `far_field_padding=1`，FFT 展的是瞳孔完整衍射场，与目标的角尺度差约
4 倍，逐像素不可比：

| `far_field_padding`（中心裁剪） | `Z=0` 时 val R² |
|---|---|
| 1 | −0.20 |
| 8 | +0.33 |
| **10**（默认） | **+0.46** |
| 12 | +0.53 |
| 16 | +0.49 |
| 20 | +0.18 |

明显的**内点最优** —— 这是真正的物理标定，不是"越锐越好"。标定正确后 `max|Z|` 从失控的
**1.77 rad 降到 0.165 rad**：小系数就能解释数据，不再需要靠巨大相位硬凑 loss。

**3.2 `normalization="sum"` 是指标陷阱。** 把预测与目标各自除以总能量，两者会"几乎一致"：

| `normalization` | MSE | R² | PSNR | SSIM |
|---|---|---|---|---|
| `peak` | 0.00429 | **+0.706** | 24.8 dB | 0.594 |
| `sum` | **0.00000** | +0.632 | **72.1 dB** | **0.9996** |
| `none` | 0.47534 | **−158.98** | 3.2 dB | 0.037 |

`sum` 在 MSE/PSNR/SSIM 上读作*完美*，而 R² 反而更差 —— 它度量的是归一化本身，不是拟合质量。

**3.3 `observable`：强度，不是振幅。** CCD 积分的是强度，不报告场振幅：

| `observable` | n_max=4 | n_max=11 | n_max=15 |
|---|---|---|---|
| `amplitude` | +0.634 | +0.659 | +0.675 |
| `intensity` | **+0.750** | **+0.792** | **+0.797** |

## 4. 训练历史（wandb 记录的那组序列）

![训练历史]({fig_ref(figures.get("training"))})

三个面板分别是损失、验证质量、梯度与系数范数。要点：

- **收敛很快**：`best_epoch = {history['best_epoch']}`，之后训练 loss 继续降而验证指标不再改善。
- **梯度极小**（`{history['grad_norm_first']:.2e}` → `{history['grad_norm_last']:.2e}`）：
  因为只有一个参数向量且 loss 已按尺度归一化。这解释了为什么 `lr` 如此关键，
  也解释了早期扫描要么无反应（1e-4）要么发散（1e-1）。
- **无死模**：`{history['dead_modes']}/{history['n_modes']}`。

![true vs pred]({fig_ref(figures.get("true_vs_pred"))})

上图是训练过程中保存的 true-vs-pred 对比帧（true / pred / 差分 / 目标相位），
由 `train_amp.py` 在每个采样 epoch 输出。

## 5. 噪声地板的真因：语料非 i.i.d.

同一 config、同一 seed 重复 3 次 → 结果**逐位相同**（5 位小数全等），管线是确定性的。
只换 split seed：

| seed | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| val R² | +0.780 | +0.797 | +0.920 | +0.923 |

**σ ≈ 0.07。** 长期以来我把原因归结为"验证集只有 128 条"。**这个诊断是错的。**

1010 条记录均匀分布在 10 个 pickle（每份恰好 101 条），但这 10 份只来自 **4 个优化目标**：

| 目标 | pickle 数 | 记录数 |
|---|---|---|
| `rms_pib` | 4 | 404 |
| `rmse_out` | 3 | 303 |
| `shape` | 2 | 202 |
| `roi_pib` | 1 | 101 |

而这四个目标的**图像分布确实不同** —— 量化方式是用 A 目标的均值图去预测 B 目标：

![目标分布差异]({fig_ref(figures.get("mixture"))})

对角线为 1，离对角最低只有 **+0.54**（`shape` 均值图预测 `roi_pib`），最高 +0.97。
所以目标之间有可测的分布差异，但**没有一对是"负 R²"**。

> ⚠️ **这里更正我自己一个诊断错误。** 早期版本这张表给出 `rms_pib → roi_pib = −0.670`
> 并据此声称"比预测常数还差"，那**是错的**：分母用了另一个 group 的方差，
> 分子却是纯像素量，量纲不一致，放大了数值。正确分母是被预测组自身的
> `SS_tot`，重算后**全表为正**。结论方向（目标异质 → 折的组成会变）成立，
> 但"比常数还差"的说法作废。

那 σ ≈ 0.07 到底来自哪里？**主要是折本身的难度差**，而不是目标配比：留一 pickle 的
逐折 R² 从 0.737 到 0.941（σ ≈ 0.08），同一模型在不同折上差 0.2。单个随机划分恰好
抽到难折或易折，指标就跟着跳。所以**必须分组 + 配对**：三个模型共用同一批折，
折难度在差值里抵消掉，配对 σ 降到 ~0.01，否则任何模型间比较都淹没在折间方差里。

> 另两处更正：划分实际是**按文件的 75/25**（`val_fraction=0.25`，`train_amp.py:302`），
> 之后验证集再被截断到 `max_val=128`（`:315`）—— **不是 80/20**。
> `HWRecordRef.source` 是**相位表示**（`panel_gray`/`panel_rad`/`zernike`/`freeform`），
> **不是来源文件**；文件标识是 `HWRecordRef.path`。

## 6. 分组交叉验证

仓库**完全没有 sklearn 依赖**（全树零引用），所以折按 `str(record.path)` 手工构造，
沿用 `_select_records` 已经用对的 group key。

| 协议 | 折数 | train / val | 回答什么问题 |
|---|---|---|---|
| `objective` | 4 | 606 / 404 | 泛化到**从未见过的目标** |
| `file` | 10 | 909 / 101 | 每条记录恰好被验证一次 |

统计用**精确 sign-flip 置换检验**（2¹⁰ = 1024 次枚举，最小 p = 0.00195，不假设正态）、
Cohen's `d_z`、以及跨模型×指标族的 Holm–Bonferroni 校正。

留一 pickle 的一个附带好处：**每折的验证集天然只含单个目标**（因为每个 pickle 只属于
一个目标），所以验证集是同分布的，不再有配比抽签。

![逐折得分]({fig_ref(figures.get("per_fold"))})

折间波动（σ ≈ 0.06 ~ 0.08）远大于模型间差距，这就是为什么必须配对检验而不是比较均值。

## 7. 最终对比（10 折分组 CV，三个模型调优后）

{final_table(sweep)}

配置：`physics` n_max={configs.get('physics', {}).get('n_max')} / lr={configs.get('physics', {}).get('lr')}；
`hybrid` 同物理配置 + 残差宽度 {configs.get('hybrid', {}).get('residual_width')}；
`unet` {configs.get('unet', {}).get('features')} / {configs.get('unet', {}).get('epochs')} ep / lr={configs.get('unet', {}).get('lr')}。

![配对差值]({fig_ref(figures.get("paired"))})

{paired_table(sweep)}

**结论。**

1. **U-Net 显著更好**：R² {paired_pu['r2']['mean_diff']:+.4f}（p = {paired_pu['r2']['p_signflip']:.4f}），
   SSIM {paired_pu['ssim']['mean_diff']:+.4f}（p = {paired_pu['ssim']['p_signflip']:.4f}）。
   注意这是在**物理模型调优之后**的对比。
2. **hybrid 不会更好**：{hybrid_verdict}
   ⇒ **不提升为训练入口的模型选项**，只作为已记录的反面结果保留。

### 按目标分解

![按目标分解]({fig_ref(figures.get("per_objective"))})

物理模型在 `roi_pib` 这个孤立目标上反而最好（该目标分布最"窄"，方差最低），
在 `rms_pib` / `rmse_out` 上最难 —— 这也解释了为什么池化 R² 对目标配比如此敏感。

### 两者用途不可互换

U-Net 出的是**图像**，反解成可下发的 SLM 相位是另一个反问题。就"要下发什么相位"这个
用途而言，230 参数的物理模型是唯一能**闭式给出可实现相位**的（`Σ Z_k B_k`，
230 个可解释弧度），参数少 34000×、墙钟少 40%。U-Net 的 {s_unet['r2']['mean']:+.4f}
买的是更高的拟合度，不是可下发性。

## 8. 优化：一个杠杆一个杠杆地试

| 杠杆 | 结果 |
|---|---|
| **`n_max`** | **唯一的大杠杆**，但已饱和（见下图） |
| `far_field_padding` | 内点最优 ≈10–12 |
| `observable` | 强度胜出，约 +0.12 R² |
| `normalization` | 只有 `peak` 有效 |
| `lr` | 与 `n_max` **非可加**，见下 |
| `l2_penalty` | 1e-4 中性；1e-2 过正则 |
| `grad_clip` | 无效果 —— 梯度（~1e-3）根本达不到阈值 |
| `optimizer` | SGD 明显更差；`weight_decay=0` 时 `adam ≡ adamw` 是**正确**的 |
| 曝光增广 | **完全无效**，见下 |

### `n_max` 已饱和

![n_max 饱和]({fig_ref(figures.get("n_max"))})

系数从 77 涨到 495（6.4×）只换来 R² +0.010，而折间 σ ≈ 0.08。

### 联合扫描才有意义

![联合扫描]({fig_ref(figures.get("physics_grid"))})

{grid_table(sweep, "physics_grid")}

`lr=0.02` **单独**用在 `n_max=15` 上毫无改善，可同一个 `lr` 配 `n_max=20` 就是全表最好。
这就是**非可加性**，也是贪心坐标下降必然失败的原因。配对检验里配对 σ 只有 ~0.01，
而折间 σ 是 ~0.08 —— 折难度被抵消掉，所以 +0.008 量级的变化在配对设计下才看得见。

> ⚠️ 最优点是**在这 10 折上从 5 个候选里挑出来的**，带 winner's curse，只能当提示性证据；
> 要确证需嵌套 CV。保守可选 `n_max=20, lr=0.01`。

### U-Net 配置没有过拟合到噪声 split

![U-Net 复验]({fig_ref(figures.get("unet_grid"))})

{grid_table(sweep, "unet_grid")}

原选择（宽 16、50 ep、lr 0.01）在 10 折下仍最优；全部候选 R² 只差 0.010、SSIM 差 0.031，
远在 ±0.06 折间 σ 内。

### 两个死杠杆

`optimizer` 曾对 adam/adamw/sgd 返回逐位相同的分数 —— 因为 `train()` 硬写了
`torch.optim.Adam`，等于什么都没测。曝光缩放增广虽然在本任务中**物理上精确成立**
（曝光是模型输入、CCD 线性、语料从不裁剪），却让 R² 变化**恰好为零**（4 位小数）。
原因可测：曝光在这个 family 里**恒定**（`log10 = −1.0`），**而且没有任何模型消费它** ——
`forward(phase_cos, phase_sin)` 而已。在模型看不见、且从不变化的维度上做有效增广，
必然是空操作。

## 9. 诚实的局限

- **只有一个 family、一个 `fov_px`。** 全部结论针对 `slm_zernike_shaping`（248 px）。
  `far_field_padding` 是 per-family 常量，换 family **必须重扫**。
- **采样单位是 pickle（n = 10），这是统计功效的硬上限。** 折数不可能超过组数。
- **k 折之间的差值并不独立** —— 第 i 折的训练集与第 j 折重叠 8/9。
  所以配对 t 检验与置换检验的 p 值都略偏激进，**效应量与置信区间才是诚实的头条**。
- **只有 4 个目标**，留一目标交叉验证只有 4 折，最小可达 p = 2/2⁴ = 0.125。
  该协议问题最对但功效不足 —— 要靠**增加目标数**解决，不是增加折数。
- **单个全局 `Z`** 只能表示全台系统性的相位，无法表示逐样本像差。
  这正是它停在 R² ≈ 0.88 而 U-Net（可逐样本变化）能到 0.90 的结构性原因。
- **散斑场上的 SSIM 是公认偏弱的指标**：其 structure 项是逐点归一化互相关，
  Larson & Chandler 显示 SSIM 对高斯噪声、散斑噪声、脉冲噪声、JPEG、模糊给出几乎相同的
  分数（约 0.64），它基本**无法区分失真类型**。本报告以 R² 与光斑域指标为主，
  SSIM 作次要描述量；`perplexity` 仅作单次运行内的单调重标度。
- **LPIPS/FID 刻意缺席**：它们需要预训练骨干网，而本环境没有 `torchvision`，
  `torchmetrics` 的感知指标确实无法导入。`available_perceptual_metrics()` 如实报告这一点，
  而不是返回一个常数。
- **`AGENTS.md` 同样记录了这些更正，但未提交**：它的 272 行 hunk 把本任务与另一位 agent
  未完成的 `cli_params.py` 笔记交织在一起。

## 10. 反转问题：固定网络权重，把 phase 当优化变量

前 9 节都是「训练网络去**预测** phase」。这里反过来：**权重全部冻结**，把输入 phase
当作唯一的优化变量，让代理模型输出的光斑成为 **{s_side}×{s_side} px 的均匀方斑**：

```
固定 ZernikeAmpModel（requires_grad=False），只优化 φ：
    maximise  quality( surrogate(φ) )      s.t.  φ 在 SLM 网格上
```

这样做值得，是因为代理模型是**已标定**的：`far_field_padding=10` + 中心裁剪正是让
预测光斑能与 248 px 相机窗口逐像素对齐的那一步，而它来自 1010 帧实测。所以代理空间里
一个 50×50 的目标，对应相机窗口里 50×50 px。SPGD 每轮要两次相机读数且用不了梯度，
这条路一次读数都不需要。

![相位合成]({fig_ref(figures.get("phase_synthesis"))})

{synthesis_table(synthesis) if synthesis else "_(未运行 optimize_uniform_spot_phase.py)_"}

### 三个结果

**1. 目标函数才是瓶颈，不是物理极限 —— 而且它错得很隐蔽。**

第一步用 MSE 填盒子（"盒内填满、盒外清空"）只到 CV 0.264。换成与 canonical
`compute_quality_score` 对齐的可微损失后到 CV {s_cv:.4f}。中间还修掉一个**退化**：

第一版的可微质量分写成 `均匀度 × 盒内均值`（乘性）。但 **CV 看不见"暗"** ——
全零图像的 std=0，所以 CV=0、"完美均匀"得 1 分，而它一束光都没交出来。
写成乘性时，一个 4×4 的亮斑和一个全暗的框**打平**（MSE 上甚至暗的还略差：
0.6104 vs 0.6064）。canonical 的 `compute_quality_score` 是**加性**的
（CV 项 + EE 项），正是靠加性的 EE 项破掉这个平局。改成加性后结果从
CV 0.168 进一步到 {s_cv:.4f}，EE 也从 0.887 升到 {s_ee:.4f}。

> 这条比"调参"更重要：**两个目标函数优化的是不同的东西**，MSE 的最优并不是所报指标的
> 最优，而两者在冷启动区甚至给出**排序相反**的答案。

**2. 两个代理差 {s_gap:.1f} 倍，但这「不能」直接读成"物理代理在光学上错了"。**

| 评判者 | 平相位 | 合成相位 |
|---|---|---|
| 物理代理（被优化的那个） | {s_flat_cv:.4f} | **{s_cv:.4f}** |
| 独立训练的 U-Net 代理 | {s_x_flat_cv} | **{s_x_cv}** |

U-Net 也看到光斑变均匀（{s_x_flat_cv} → {s_x_cv}），所以这个相位**不是纯粹的
物理模型伪影**；但两个代理相差 {s_gap:.1f} 倍（图中第三、四幅可以直接看出：一个说
"完美方斑"，另一个说"一个中心亮斑"）。

**关键在于这个分歧有多少来自外推。** 语料相位是 `n_max` 限定的 Zernike 相位，
天然带限；自由相位不是。定量测一下：

| | Zernike 张成空间内的能量占比 | 0.25×Nyquist 以上的能量占比 |
|---|---|---|
| 语料相位（n={s_ood_n}） | **{s_ood_c:.3f}** | 带限 |
| **合成相位** | **{s_ood_ours:.3f}** | **{s_ood_hf:.3f}** |

合成相位只有 {s_ood_ours:.1%} 的能量落在语料所在的 Zernike 张成空间里，而语料平均是
{s_ood_c:.1%}；它 {s_ood_hf:.1%} 的能量在 0.25×Nyquist 以上。**它远离两个代理的训练流形。**

所以诚实的结论是：**两个代理都在外推，谁的数字都不能当光学真相。** 已知的只有
"合成相位确实把光斑从随机散斑变成了某个有结构的形态"（两者都看到明显改善），
以及"代理自称的均匀度不可信"。**真实 CV 只能靠硬件测。**

**方法论结论：学习到的回归器不能直接当整形目标函数。** 两个原因叠加 ——
它会把散斑预测平滑掉（于是"看起来干净"的相位在真实探测器上仍有散斑），
而且它自己找到的最优点落在自己的训练分布之外，在那里它的预测从未被验证过。
要修，两条路：把优化限制在模型可信的流形上（Zernike 限定 —— 但仓库已记录低阶 Zernike
**造不出**方斑），或者给损失加散斑敏感项。

**3. 与仓库自己的整形结果相比：赢了 GS，输给完整细化。** 仓库数值仿真里 GS 单次约
CV 0.41，`GS + 自由相位细化` 约 CV 0.12；本方法 0.166 落在两者之间。注意两者并非
严格同条件（padding、网格、目标构造都不同），所以这是量级参照而非等价比较。

> **这条结论只到"代理空间"为止。** 物理模型在留出 pickle 上 R² ≈ 0.88，
> 一个把代理优化到 CV 0.17 的相位，并不因此在真实台架上也是 CV 0.17。
> **必须在硬件上验证**；上面的 U-Net 交叉检验是硬件不可用时能做的最强旁证，
> 而它的结论是"有改善但远不如代理声称的"。

## 11. 本报告中被推翻的结论

保留记录，因为**三次朝相反方向搞错、再加一次诊断算错**正是重点：

| 版本 | 结论 | 为什么错 |
|---|---|---|
| v1 | "U-Net R² 高 +0.022；SSIM 是真实且区间不重叠的差距" | 方向对、证据错：单 seed，且 U-Net 只给 25 epoch 而物理模型第 13 epoch 就收敛 |
| v2 | "两者打平，无任何指标能区分" | **过度纠正**：把折间方差当成了模型差异 |
| v3 | "U-Net 显著更好；hybrid 不会更好，且在调优 lr 下更差" | 分组 CV + 配对精确检验，4 折与 10 折均成立 |
| d1 | "目标均值图互相预测 R² = −0.670，比常数还差" | **算错**：分母量纲不一致。重算后全表为正（最低 +0.54） |

两个教训：

1. 噪声大到看不见效应时，**过度纠正和轻信同样危险**。v2 把真实差异否认掉，
   根源是我先认定噪声来自"只有 128 条验证集"，而没去定位它的真实来源。
2. **解释性数字和结论要用不同的严格度。** v3 的结论建立在 10 折配对检验上，
   经得起复算；而 d1 那个"−0.670"是手算的辅助论据，量纲错了却没人复核 ——
   它是本轮**画图时才暴露**的。图比手算更可信，因为它用了被预测组自己的 `SS_tot`。

## 12. 复现

```bash
# 扫描并落盘（10 折分组 CV，约 30 min 单卡）
python scripts/sweep_zernike_models.py
# 只重跑调优后的最终对比，复用已保存的网格
python scripts/sweep_zernike_models.py --final-only

# 生成图与本报告（全离线）
python scripts/generate_zernike_amp_report.py

# 训练 + wandb（无 key 时离线）
python -m ml.zernike.train_amp --n-max 20 --epochs 50 --lr 0.02 \\\\
    --max-train 1010 --beam-samples 96 --out-dir logs/zernike_amp_final

# 权威对比：分组 CV + 配对精确检验
python scripts/compare_models_cv.py --protocol both

# 不重训、只重算统计量
python scripts/compare_models_cv.py --analyse logs/models_cv_objective.json

# 测试
python -m pytest tests/ao_shaping/ml/zernike -q                        # 67
python -m pytest tests/ao_shaping/scripts/test_compare_models_cv.py -q  # 19
```
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", default="logs/zernike_amp_sweep.json")
    parser.add_argument("--history", default="logs/zernike_amp_final/summary.json")
    parser.add_argument("--index-cache", default="data/hw_index_cache.json")
    parser.add_argument("--compare-png", default="logs/zernike_amp_final/compare_epoch024.png")
    parser.add_argument(
        "--synthesis", default="logs/uniform_spot_phase",
        help="directory written by scripts/optimize_uniform_spot_phase.py",
    )
    parser.add_argument("--out-dir", default="docs/zernike_amp")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    sweep = _load(Path(args.sweep))
    history = _load(Path(args.history))
    synthesis_dir = Path(args.synthesis)
    synthesis = None
    if (synthesis_dir / "summary.json").exists():
        synthesis = json.loads((synthesis_dir / "summary.json").read_text(encoding="utf-8"))
    else:
        print(f"note: no inversion results at {synthesis_dir}; section 10 will be omitted")
    out_dir = ROOT / args.out_dir
    FIGURES.mkdir(parents=True, exist_ok=True)

    figures: dict[str, str | None] = {}
    if not args.no_figures:
        figures["training"] = fig_training_curves(Path(args.history))
        figures["physics_grid"] = fig_physics_grid(sweep)
        figures["n_max"] = fig_n_max_curve(sweep)
        figures["unet_grid"] = fig_unet_grid(sweep)
        figures["per_fold"] = fig_per_fold(sweep)
        figures["paired"] = fig_paired(sweep)
        figures["per_objective"] = fig_per_objective(sweep)
        figures["mixture"] = fig_objective_mixture(args.index_cache)
        figures["true_vs_pred"] = fig_true_vs_pred(
            Path(args.compare_png) if args.compare_png else None
        )
        if synthesis is not None:
            figures.update(fig_phase_synthesis(synthesis, synthesis_dir))
        print(f"rendered {sum(1 for v in figures.values() if v)} figures -> {FIGURES}")

    text = build_report(sweep, history, figures, synthesis)
    if synthesis is None:
        # Drop section 10 rather than ship a section full of NaN placeholders.
        start = text.find("## 10. 反转问题")
        end = text.find("## 11. 本报告中被推翻的结论")
        if start != -1 and end != -1:
            text = text[:start] + text[end:]
            text = text.replace("## 11. 本报告中被推翻的结论", "## 10. 本报告中被推翻的结论")
            text = text.replace("## 12. 复现", "## 11. 复现")
    (out_dir / "report.md").write_text(text, encoding="utf-8")
    print(f"wrote {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
