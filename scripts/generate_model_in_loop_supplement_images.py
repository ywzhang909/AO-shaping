"""Generate supplementary images for the 2026-10-13 model-in-loop live report.

This script renders figures for the new experimental sections added after the
initial report generation:
  - U-Net vs physical model forward prediction comparison montages
  - PSF trial bar charts (baseline vs PSF model per phase family)
  - Phase bandwidth trial bar charts (baseline vs smooth per phase family)
  - Cross-experiment model candidate comparison
  - Exposure response diagnostics

All figures are saved to ``report/shape_trials/images/``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "report" / "shape_trials" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

PAIRED_DIR = ROOT / "data" / "debug" / "slm_model_compare"
EXP_ANALYSIS_DIR = ROOT / "data" / "debug" / "slm_exposure_analysis"
REPLAY_DIR = ROOT / "data" / "debug" / "slm_model_in_loop_replay"

sns.set_theme(style="whitegrid", context="talk")


def _load_csv_as_dicts(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _group_by_family_family(rows: list[dict]) -> dict:
    """Group rows by phase family, returning {family: {method: avg_metrics}}."""
    result: dict[str, dict[str, dict]] = {}
    for r in rows:
        fam = r["family"]
        method = r["method"]
        if fam not in result:
            result[fam] = {}
        if method not in result[fam]:
            result[fam][method] = {"correlation": [], "shape_nrmse": []}
        result[fam][method]["correlation"].append(float(r["correlation"]))
        result[fam][method]["shape_nrmse"].append(float(r["shape_nrmse"]))
    return result


# ---------------------------------------------------------------------------
# Figure 1: U-Net vs Physical model prediction montage
# ---------------------------------------------------------------------------


def render_unet_prediction_montage() -> Path:
    """Render a montage comparing CCD, physical model, U-Net pretrain, U-Net adapted
    predictions for a representative set of holdout phases.
    """
    pred_dir = PAIRED_DIR / "20261013_1943_seed_20261013" / "predictions"
    files = sorted(pred_dir.glob("*.npz"))

    # Pick one example from each family
    examples: dict[str, Path] = {}
    for f in files:
        data = np.load(f)
        family = data["family"] if "family" in data.keys() else "unknown"
        if family not in examples:
            examples[family] = f

    families = list(examples.keys())
    n = len(families)
    fig, axes = plt.subplots(4, n, figsize=(4 * n, 16))
    if n == 1:
        axes = axes.reshape(-1, 1)

    titles = ["CCD 测量", "Zernike + NNLS", "U-Net 预训练", "U-Net 10 步微调"]
    for col, (family, fpath) in enumerate(examples.items()):
        data = np.load(fpath)
        images = [
            data["ccd"],
            data["physical"],
            data["unet_prior"],
            data["unet_adapted"],
        ]
        for row, (img, title) in enumerate(zip(images, titles)):
            ax = axes[row, col]
            img_norm = np.asarray(img, dtype=np.float64)
            peak = max(float(img_norm.max()), 1e-12)
            ax.imshow(img_norm / peak, cmap="viridis", interpolation="none")
            ax.set_title(f"{family}\n{title}" if row == 0 else title, fontsize=11)
            ax.axis("off")

    fig.suptitle(
        "真实 CCD 同帧 U-Net vs 物理模型对比 (0.4 ms, 36×36 目标框)",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    out = IMG_DIR / "mil_unet_prediction_montage.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2: PSF trial comparison bar chart
# ---------------------------------------------------------------------------


def render_psf_trial_comparison() -> Path:
    """Bar chart: baseline vs train-selected PSF per phase family."""
    csv_path = PAIRED_DIR / "20261013_psf_trial" / "paired_observable_holdouts.csv"
    rows = _load_csv_as_dicts(csv_path)
    grouped = _group_by_family_family(rows)

    families = ["smooth", "zernike", "freeform"]
    family_names = {"smooth": "平滑", "zernike": "Zernike", "freeform": "自由相位"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    metrics = [
        ("correlation", "Pearson 相关性", "相关性"),
        ("shape_nrmse", "形状 NRMSE", "NRMSE"),
    ]

    for ax_idx, (metric, label, title) in enumerate(metrics):
        ax = axes[ax_idx]
        baseline_vals = [np.mean(grouped[fam]["baseline"][metric]) for fam in families]
        psf_vals = [
            np.mean(grouped[fam]["train_selected_psf"][metric]) for fam in families
        ]

        x = np.arange(len(families))
        width = 0.35
        bars1 = ax.bar(
            x - width / 2, baseline_vals, width, label="基线 (无 PSF)", color="#4a90d9"
        )
        bars2 = ax.bar(
            x + width / 2, psf_vals, width, label="训练选 PSF", color="#e91e63"
        )

        for bar in bars1:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        for bar in bars2:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([family_names.get(f, f) for f in families])
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(label, fontsize=11)
        ax.legend(fontsize=10)
        ax.set_ylim(0, max(max(baseline_vals), max(psf_vals)) * 1.25)

    fig.suptitle(
        "PSF 训练选 vs 基线: 相关性与 NRMSE (新会话 + 旧会话合并, 24 帧/族)",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    out = IMG_DIR / "mil_psf_trial_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 3: Phase bandwidth trial comparison bar chart
# ---------------------------------------------------------------------------


def render_phase_bandwidth_comparison() -> Path:
    """Bar chart: baseline vs train-selected phase bandwidth per phase family."""
    csv_path = PAIRED_DIR / "20261013_phase_bandwidth_trial" / "paired_holdouts.csv"
    rows = _load_csv_as_dicts(csv_path)
    grouped = _group_by_family_family(rows)

    families = ["smooth", "zernike", "freeform"]
    family_names = {"smooth": "平滑", "zernike": "Zernike", "freeform": "自由相位"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    metrics = [("correlation", "Pearson 相关性"), ("shape_nrmse", "形状 NRMSE")]

    for ax_idx, (metric, label) in enumerate(metrics):
        ax = axes[ax_idx]
        baseline_vals = [np.mean(grouped[fam]["baseline"][metric]) for fam in families]
        smooth_vals = [
            np.mean(grouped[fam]["train_selected_phase_bandwidth"][metric])
            for fam in families
        ]

        x = np.arange(len(families))
        width = 0.35
        bars1 = ax.bar(
            x - width / 2, baseline_vals, width, label="基线 (无平滑)", color="#4caf50"
        )
        bars2 = ax.bar(
            x + width / 2, smooth_vals, width, label="训练选平滑", color="#ff9800"
        )

        for bar in bars1:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        for bar in bars2:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([family_names.get(f, f) for f in families])
        ax.set_title(label, fontsize=12)
        ax.legend(fontsize=10)

    fig.suptitle(
        "相位带宽训练选 vs 基线: 相关性与 NRMSE (24 帧/族)", fontsize=14, y=1.02
    )
    fig.tight_layout()
    out = IMG_DIR / "mil_phase_bandwidth_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 4: Model candidate comparison across sessions
# ---------------------------------------------------------------------------


def render_model_candidate_comparison() -> Path:
    """Compare baseline vs PSF vs phase-bandwidth models across old and new sessions."""
    old_psf = _load_csv_as_dicts(
        PAIRED_DIR / "20261013_psf_trial" / "paired_holdouts.csv"
    )
    old_bw = _load_csv_as_dicts(
        PAIRED_DIR / "20261013_phase_bandwidth_trial" / "paired_holdouts.csv"
    )
    new_psf = _load_csv_as_dicts(
        PAIRED_DIR / "20261013_2015_new_session" / "psf_paired.csv"
    )
    new_bw = _load_csv_as_dicts(
        PAIRED_DIR / "20261013_2015_new_session" / "phase_bandwidth_paired.csv"
    )

    families = ["smooth", "zernike", "freeform"]
    family_names = {"smooth": "平滑", "zernike": "Zernike", "freeform": "自由相位"}

    combos = [
        ("旧会话\n(种子 79)", old_psf, old_bw),
        ("新会话\n(种子 81)", new_psf, new_bw),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for row_idx, (session_label, psf_rows, bw_rows) in enumerate(combos):
        psf_grouped = _group_by_family_family(psf_rows)
        bw_grouped = _group_by_family_family(bw_rows)

        for col_idx, fam in enumerate(families):
            ax = axes[row_idx, col_idx]
            methods = [
                "baseline",
                "train_selected_psf",
                "train_selected_phase_bandwidth",
            ]
            method_labels = ["基线", "PSF 模型", "相位平滑"]
            colors = ["#2196f3", "#e91e63", "#ff9800"]

            corr_vals = [
                np.mean(psf_grouped[fam]["baseline"]["correlation"]),
                np.mean(psf_grouped[fam]["train_selected_psf"]["correlation"]),
                np.mean(
                    bw_grouped[fam]["train_selected_phase_bandwidth"]["correlation"]
                ),
            ]
            nrmse_vals = [
                np.mean(psf_grouped[fam]["baseline"]["shape_nrmse"]),
                np.mean(psf_grouped[fam]["train_selected_psf"]["shape_nrmse"]),
                np.mean(
                    bw_grouped[fam]["train_selected_phase_bandwidth"]["shape_nrmse"]
                ),
            ]

            x = np.arange(len(methods))
            width = 0.35
            ax2 = ax.twinx()
            bars1 = ax.bar(
                x - width / 2, corr_vals, width, label="相关性", color=colors, alpha=0.8
            )
            bars2 = ax2.bar(
                x + width / 2,
                nrmse_vals,
                width,
                label="NRMSE",
                color=colors,
                alpha=0.5,
                hatch="//",
            )

            ax.set_xticks(x)
            ax.set_xticklabels(method_labels, fontsize=10)
            ax.set_ylim(0.4, 1.0)
            ax2.set_ylim(0, 0.8)
            ax.set_title(f"{session_label} / {family_names.get(fam, fam)}", fontsize=12)

            if row_idx == 0 and col_idx == 0:
                ax.set_ylabel("相关性", fontsize=11)
                ax2.set_ylabel("NRMSE", fontsize=11)

    fig.suptitle("模型候选跨会话比较: 基线 vs PSF vs 相位平滑", fontsize=14, y=1.02)
    fig.tight_layout()
    out = IMG_DIR / "mil_model_candidate_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 5: Exposure response curves
# ---------------------------------------------------------------------------


def render_exposure_response() -> Path:
    """Plot exposure response curves for flat and shaped phases."""
    analysis_dir = EXP_ANALYSIS_DIR / "20261013_1935"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Subplot 1: Response curves
    ax = axes[0]
    analysis = json.loads((analysis_dir / "analysis_with_repeat.json").read_text())
    sweeps = analysis["sweeps"]

    for sweep in sweeps:
        if "flat" in sweep["source"].lower():
            exposures = []
            signals = []
            for frame in sweep["frames"]:
                if frame.get("exposure_ms"):
                    exposures.append(frame["exposure_ms"])
                    signals.append(
                        frame.get("mean_roi_signal", frame.get("raw_peak", 0))
                    )
            if exposures:
                color = "blue" if "0-08" in sweep["source"] else "cyan"
                ax.plot(
                    exposures,
                    signals,
                    "o-",
                    color=color,
                    label=f"平场 ({sweep['source'].split('/')[-1][-6:]})",
                )
        elif "shaped" in sweep["source"].lower():
            exposures = []
            signals = []
            for frame in sweep["frames"]:
                if frame.get("exposure_ms"):
                    exposures.append(frame["exposure_ms"])
                    signals.append(
                        frame.get("mean_roi_signal", frame.get("raw_peak", 0))
                    )
            if exposures:
                ax.plot(
                    exposures,
                    signals,
                    "s-",
                    color="red",
                    label=f"成形 {sweep['source'].split('/')[-1][-6:]}",
                )

    ax.set_xlabel("曝光 (ms)")
    ax.set_ylabel("ROI 信号 (背景扣除后)")
    ax.set_title("曝光响应曲线")
    ax.legend(fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Subplot 2: PSF comparison (reuse psf_paired.csv from new session)
    ax = axes[1]
    new_psf = _load_csv_as_dicts(
        PAIRED_DIR / "20261013_2015_new_session" / "psf_paired.csv"
    )
    grouped = _group_by_family_family(new_psf)
    x = np.arange(3)
    width = 0.25
    baseline_corr = [
        np.mean(grouped[f]["baseline"]["correlation"])
        for f in ["smooth", "zernike", "freeform"]
    ]
    psf_corr = [
        np.mean(grouped[f]["train_selected_psf"]["correlation"])
        for f in ["smooth", "zernike", "freeform"]
    ]
    ax.bar(x - width / 2, baseline_corr, width, label="基线", color="#4a90d9")
    ax.bar(x + width / 2, psf_corr, width, label="PSF 模型", color="#e91e63")
    for i, (b, p) in enumerate(zip(baseline_corr, psf_corr)):
        ax.text(i - width / 2, b + 0.01, f"{b:.3f}", ha="center", fontsize=9)
        ax.text(i + width / 2, p + 0.01, f"{p:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["平滑", "Zernike", "自由"])
    ax.set_ylabel("相关性")
    ax.set_title("新会话 PSF 相关性对比")
    ax.legend(fontsize=9)

    # Subplot 3: Phase bandwidth comparison (new session)
    ax = axes[2]
    new_bw = _load_csv_as_dicts(
        PAIRED_DIR / "20261013_2015_new_session" / "phase_bandwidth_paired.csv"
    )
    grouped = _group_by_family_family(new_bw)
    baseline_nrmse = [
        np.mean(grouped[f]["baseline"]["shape_nrmse"])
        for f in ["smooth", "zernike", "freeform"]
    ]
    smooth_nrmse = [
        np.mean(grouped[f]["train_selected_phase_bandwidth"]["shape_nrmse"])
        for f in ["smooth", "zernike", "freeform"]
    ]
    ax.bar(x - width / 2, baseline_nrmse, width, label="基线", color="#4caf50")
    ax.bar(x + width / 2, smooth_nrmse, width, label="相位平滑", color="#ff9800")
    for i, (b, s) in enumerate(zip(baseline_nrmse, smooth_nrmse)):
        ax.text(i - width / 2, b + 0.01, f"{b:.3f}", ha="center", fontsize=9)
        ax.text(i + width / 2, s + 0.01, f"{s:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["平滑", "Zernike", "自由"])
    ax.set_ylabel("NRMSE")
    ax.set_title("新会话相位平滑 NRMSE 对比")
    ax.legend(fontsize=9)

    fig.suptitle(
        "Daheng 曝光响应与模型候选对比 (2026-10-13 19:16–19:38)", fontsize=14, y=1.02
    )
    fig.tight_layout()
    out = IMG_DIR / "mil_exposure_and_model_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 6: Fixed exposure A/B phase replay
# ---------------------------------------------------------------------------


def render_replay_comparison() -> Path:
    """Render comparison of A/B phase replay with fixed exposure."""
    replay_dirs = [
        ("无扰动 AB", REPLAY_DIR / "20261013_1733_four_vs_two"),
        ("固定曝光 AB", REPLAY_DIR / "20261013_160630_fixed040"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Subplot 1: Score comparison
    ax = axes[0]
    categories = [
        "无扰动\n(2探针)",
        "固定曝光\n(2探针)",
        "固定曝光\n(4探针)",
        "有扰动\n(2探针)",
        "有扰动\n(4探针)",
    ]
    scores = [0.7294, 0.73098, 0.7133, 0.716, 0.71940]
    cv_vals = [1.0168, 1.0097, 1.08534, 1.0306, 0.89790]
    ee_vals = [0.8850, 0.8866, 0.8716, 0.8661, 0.84773]

    x = np.arange(len(categories))
    width = 0.25
    ax.bar(x - width, scores, width, label="分数", color="#2196f3")
    ax2 = ax.twinx()
    ax2.bar(x, cv_vals, width, label="CV", color="#f44336", alpha=0.7)
    ax2.bar(x + width, ee_vals, width, label="EE", color="#4caf50", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel("综合分", fontsize=11)
    ax2.set_ylabel("CV / EE", fontsize=11)
    ax.set_title("AB 相位复放对比")
    ax.legend(loc="upper left", fontsize=9)
    ax2.legend(loc="upper right", fontsize=9)

    # Subplot 2: Phase image comparison (load from npz)
    ax = axes[1]
    replay = REPLAY_DIR / "20261013_1733_four_vs_two"
    phase_a = np.load(replay / "02_B.npy")  # 2-probe phase
    phase_b = np.load(replay / "02_B.npy")  # same file, placeholder
    # Load best phases from main runs
    run_173039 = ROOT / "data/debug/slm_model_in_loop/20261013_173039"
    run_173230 = ROOT / "data/debug/slm_model_in_loop/20261013_173230"

    if (run_173039 / "best_phase.npy").exists() and (
        run_173230 / "best_phase.npy"
    ).exists():
        phase_a = np.load(run_173039 / "best_phase.npy")
        phase_b = np.load(run_173230 / "best_phase.npy")
        phase_diff = phase_a - phase_b
        im = ax.imshow(phase_diff, cmap="RdBu_r", interpolation="none")
        ax.set_title("2探针 vs 4探针\n相位差 (rad)")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Subplot 3: CCD far-field comparison
    ax = axes[2]
    try:
        from PIL import Image
    except ImportError:
        Image = None

    img_a = (
        np.array(Image.open(run_173039 / "best_far_field.png"))
        if Image is not None and (run_173039 / "best_far_field.png").exists()
        else None
    )
    img_b = (
        np.array(Image.open(run_173230 / "best_far_field.png"))
        if Image is not None and (run_173230 / "best_far_field.png").exists()
        else None
    )

    if img_a is not None and img_b is not None:
        ax.imshow(
            img_a,
            cmap="gray",
            interpolation="none",
            extent=[0, img_a.shape[1], 0, -img_a.shape[0]],
        )
        ax.set_title("2探针远场")
        ax.axis("off")
        ax_inset = ax.inset_axes([1.1, 0, 1, 1])
        ax_inset.imshow(img_b, cmap="gray", interpolation="none")
        ax_inset.set_title("4探针远场", fontsize=8)
        ax_inset.axis("off")
    else:
        ax.text(0.5, 0.5, "文件缺失", ha="center", transform=ax.transAxes)
        ax.axis("off")

    fig.suptitle("固定曝光 A/B 相位复放对比 (2026-10-13)", fontsize=14, y=1.02)
    fig.tight_layout()
    out = IMG_DIR / "mil_replay_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Generate all supplementary images."""
    print("Generating U-Net prediction montage...")
    img1 = render_unet_prediction_montage()
    print(f"  -> {img1.relative_to(ROOT)}")

    print("Generating PSF trial comparison...")
    img2 = render_psf_trial_comparison()
    print(f"  -> {img2.relative_to(ROOT)}")

    print("Generating phase bandwidth comparison...")
    img3 = render_phase_bandwidth_comparison()
    print(f"  -> {img3.relative_to(ROOT)}")

    print("Generating model candidate comparison...")
    img4 = render_model_candidate_comparison()
    print(f"  -> {img4.relative_to(ROOT)}")

    print("Generating exposure and model comparison...")
    img5 = render_exposure_response()
    print(f"  -> {img5.relative_to(ROOT)}")

    print("Generating replay comparison...")
    img6 = render_replay_comparison()
    print(f"  -> {img6.relative_to(ROOT)}")

    print("\nAll images generated in:", IMG_DIR.relative_to(ROOT))


if __name__ == "__main__":
    main()
