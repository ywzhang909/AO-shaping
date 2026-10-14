"""Generate supplementary images for the 2026-10-14 model-in-loop live report.

This script renders figures showing phase and CCD images for each experimental condition:
- Phase images from best_phase.npy files
- CCD images from best_far_field.png files

All figures are saved to `report/shape_trials/images/`.
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

ROOT = Path(__file__).resolve().parents[2]
IMG_DIR = ROOT / "report" / "shape_trials" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")


def _load_experimental_data() -> list[dict]:
    """Load experimental data from the report table."""
    # Hardcoded data from the report for 2026-10-14
    return [
        {
            "dir": "20261014_105041",
            "condition": "无扰动，方形",
            "six_holdout_avg": 0.8153,
            "free_phase_avg": 0.7473,
            "pred_corr": 0.8129,
            "cv_ee": "0.9019 / 0.9209",
            "score": 0.7629,
        },
        {
            "dir": "20261014_105205",
            "condition": "扰动，方形",
            "six_holdout_avg": 0.8084,
            "free_phase_avg": 0.7601,
            "pred_corr": 0.8096,
            "cv_ee": "0.8650 / 0.8741",
            "score": 0.7390,
        },
        {
            "dir": "20261014_105339",
            "condition": "扰动，长方形",
            "six_holdout_avg": 0.8112,
            "free_phase_avg": 0.7664,
            "pred_corr": 0.8219,
            "cv_ee": "0.7142 / 0.7622",
            "score": 0.6907,
        },
        {
            "dir": "20261014_105438",
            "condition": "扰动，三角形",
            "six_holdout_avg": 0.8173,
            "free_phase_avg": 0.7846,
            "pred_corr": 0.8226,
            "cv_ee": "0.6624 / 0.7173",
            "score": 0.6710,
        },
        {
            "dir": "20261014_105535",
            "condition": "扰动，五角星",
            "six_holdout_avg": 0.8112,
            "free_phase_avg": 0.7722,
            "pred_corr": 0.8430,
            "cv_ee": "0.6529 / 0.6134",
            "score": 0.6101,
        },
        {
            "dir": "20261014_110203",
            "condition": "扰动，长方形，Step A 零步，种子 86",
            "six_holdout_avg": 0.8316,
            "free_phase_avg": 0.8213,
            "pred_corr": 0.8016,
            "cv_ee": "0.6851 / 0.6944",
            "score": 0.6540,
        },
        {
            "dir": "20261014_110305",
            "condition": "扰动，长方形，Step A 10 步，种子 86",
            "six_holdout_avg": 0.8417,
            "free_phase_avg": 0.7767,
            "pred_corr": 0.8037,
            "cv_ee": "0.8088 / 0.7471",
            "score": 0.6694,
        },
    ]


def render_phase_ccd_comparison() -> Path:
    """Create a side-by-side comparison of phase and CCD images for all experimental conditions."""
    experiments = _load_experimental_data()
    
    # Set up the plot - 2 columns (Phase, CCD), N rows (one per experiment)
    n_experiments = len(experiments)
    fig, axes = plt.subplots(n_experiments, 2, figsize=(12, 4 * n_experiments))
    
    # If only one experiment, axes needs to be reshaped
    if n_experiments == 1:
        axes = axes.reshape(1, -1)
    
    # Column labels
    axes[0, 0].set_title("相位 (Phase)", fontsize=14, pad=20)
    axes[0, 1].set_title("CCD 远场 (Far-field)", fontsize=14, pad=20)
    
    for idx, exp in enumerate(experiments):
        dir_path = ROOT / "data" / "debug" / "slm_model_in_loop" / exp["dir"]
        
        # Load and display phase image
        phase_ax = axes[idx, 0]
        try:
            phase_data = np.load(dir_path / "best_phase.npy")
            # Normalize phase data for visualization (to -pi to pi range for better contrast)
            phase_viz = np.angle(np.exp(1j * phase_data))  # Wrap to [-pi, pi]
            im_phase = phase_ax.imshow(phase_viz, cmap="RdBu_r", vmin=-np.pi, vmax=np.pi)
            phase_ax.set_axis_off()
            plt.colorbar(im_phase, ax=phase_ax, fraction=0.046, pad=0.04)
        except Exception as e:
            phase_ax.text(0.5, 0.5, f"相位数据\n加载失败\n{str(e)[:50]}...", 
                         ha="center", va="center", transform=phase_ax.transAxes,
                         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
            phase_ax.set_axis_off()
        
        # Load and display CCD image
        ccd_ax = axes[idx, 1]
        try:
            ccd_img = np.array(Image.open(dir_path / "best_far_field.png"))
            # Convert RGBA to RGB if needed
            if ccd_img.shape[2] == 4:
                ccd_rgb = ccd_img[:, :, :3]
            else:
                ccd_rgb = ccd_img
            ccd_ax.imshow(ccd_rgb)
            ccd_ax.set_axis_off()
        except Exception as e:
            ccd_ax.text(0.5, 0.5, f"CCD图像\n加载失败\n{str(e)[:50]}...", 
                       ha="center", va="center", transform=ccd_ax.transAxes,
                       bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
            ccd_ax.set_axis_off()
        
        # Add condition label and metrics as text
        label_text = (
            f"{exp['condition']}\n"
            f"目录: {exp['dir']}\n"
            f"6留出均值: {exp['six_holdout_avg']:.4f}\n"
            f"自由相位: {exp['free_phase_avg']:.4f}\n"
            f"预测相关: {exp['pred_corr']:.4f}\n"
            f"CV/EE: {exp['cv_ee']}\n"
            f"分数: {exp['score']:.4f}"
        )
        
        # Add text box in the bottom left of the phase image
        phase_ax.text(
            0.02, 0.98, label_text,
            transform=phase_ax.transAxes,
            fontsize=8,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
        )
    
    plt.suptitle(
        "2026-10-14 Model-in-Loop Experimental Results\n"
        "Left: Phase (rad), Right: CCD Far-field Intensity",
        fontsize=16,
        y=0.98
    )
    plt.tight_layout()
    
    out = IMG_DIR / "mil_phase_ccd_comparison_20261014.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    """Generate all supplementary images for 2026-10-14 report."""
    print("Generating phase and CCD comparison image...")
    img = render_phase_ccd_comparison()
    print(f"  -> {img.relative_to(ROOT)}")
    print("\nAll images generated in:", IMG_DIR.relative_to(ROOT))


if __name__ == "__main__":
    main()
