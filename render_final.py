"""Re-render far field + cross-section plots from raw PKL _img data.

The previous cross-section plots were extracted from rendered PNGs (which
carry a white matplotlib canvas + inferno colormap), causing the center
detection to land on the white margin instead of the beam.  This script
reads the raw ``_img`` frame from each trial's PKL (record 1 = GS stage)
and regenerates both the far-field render and the X/Y intensity profiles.
"""

import numpy as np
import pickle
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = "data/debug/slm_gs_refine"
FIG = "report/shape_trials/figures"

# old filenames to remove
OLD = [
    "gs_issue_live_20261015_20261015_205654_phase.png",
    "gs_issue_live_20261015_20261015_205654_farfield.png",
    "gs_issue_live_20261015_20261015_205848_phase.png",
    "gs_issue_live_20261015_20261015_205848_farfield.png",
    "gs_issue_live_20261015_20261015_210048_phase.png",
    "gs_issue_live_20261015_20261015_210048_farfield.png",
    "gs_issue_live_20261015_20261015_210155_phase.png",
    "gs_issue_live_20261015_20261015_210155_farfield.png",
    "gs_issue_live_20261015_20261015_211345_phase.png",
    "gs_issue_live_20261015_20261015_211345_farfield.png",
    "gs_issue_live_20261015_20261015_205654_cross_section.png",
    "gs_issue_live_20261015_20261015_205848_cross_section.png",
    "gs_issue_live_20261015_20261015_210048_cross_section.png",
    "gs_issue_live_20261015_20261015_210155_cross_section.png",
    "gs_issue_live_20261015_20261015_211345_cross_section.png",
]
for f in OLD:
    p = os.path.join(FIG, f)
    if os.path.exists(p):
        os.remove(p)
        print(f"Removed old: {f}")

trials = [
    ("20261015_205654", "plain", 0.6123),
    ("20261015_205848", "weighted", 0.7793),
    ("20261015_210048", "adaptive-feedback", 0.7064),
    ("20261015_210155", "plain 复测", 0.6152),
    ("20261015_211345", "plain 再复测", 0.6080),
]

for dirname, method, cv in trials:
    trial_dir = os.path.join(BASE, dirname)

    # ---- Load raw intensity from PKL (record 1 = GS stage) ----
    pkl_path = os.path.join(trial_dir, f"slm_gs_refine_{dirname}.pkl")
    with open(pkl_path, "rb") as f:
        pkl = pickle.load(f)
    raw = np.asarray(pkl[1]["_img"], dtype=float)
    raw_norm = raw / (raw.max() + 1e-12)

    # ---- Load phase from best_phase.npy ----
    phase = np.load(os.path.join(trial_dir, "best_phase.npy"))
    phase_wrapped = np.mod(phase, 2 * np.pi)

    # ---- Render phase image (clean, no white margin) ----
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(
        phase_wrapped, cmap="twilight", origin="upper", vmin=0, vmax=2 * np.pi
    )
    ax.set_title(f"{method} (CV={cv}) — SLM 相位", fontsize=11)
    ax.set_xlabel("column (px)")
    ax.set_ylabel("row (px)")
    fig.colorbar(im, ax=ax, label="phase (rad)", shrink=0.8)
    fig.tight_layout()
    phase_out = os.path.join(FIG, f"gs_issue_live_phase_{dirname}.png")
    fig.savefig(phase_out, dpi=120)
    plt.close(fig)

    # ---- Render far field (clean, no white margin) ----
    fig, ax = plt.subplots(figsize=(5, 5))
    im2 = ax.imshow(raw_norm, cmap="inferno", origin="upper")
    ax.set_title(f"{method} (CV={cv})", fontsize=11)
    ax.set_xlabel("column (px)")
    ax.set_ylabel("row (px)")
    fig.colorbar(im2, ax=ax, label="归一强度", shrink=0.8)
    fig.tight_layout()
    far_out = os.path.join(FIG, f"gs_issue_live_farfield_{dirname}.png")
    fig.savefig(far_out, dpi=120)
    plt.close(fig)

    # ---- Cross-section from raw data through the beam center ----
    cy, cx = divmod(np.argmax(raw), raw.shape[1])
    x_cross = raw_norm[cy, :]
    y_cross = raw_norm[:, cx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(np.arange(len(x_cross)), x_cross, color="steelblue", linewidth=1.0)
    ax1.axvline(cx, color="red", linestyle="--", alpha=0.7, label=f"中心 x={cx}")
    ax1.set_title(f"{method} (CV={cv}) — X 截面 (row={cy})", fontsize=11)
    ax1.set_xlabel("column (px)")
    ax1.set_ylabel("归一强度")
    ax1.set_ylim(bottom=0)
    ax1.legend(fontsize=9)

    ax2.plot(np.arange(len(y_cross)), y_cross, color="darkorange", linewidth=1.0)
    ax2.axvline(cy, color="red", linestyle="--", alpha=0.7, label=f"中心 y={cy}")
    ax2.set_title(f"{method} (CV={cv}) — Y 截面 (col={cx})", fontsize=11)
    ax2.set_xlabel("row (px)")
    ax2.set_ylabel("归一强度")
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=9)

    fig.suptitle(f"GS {method} 远场强度截面 ({dirname})", fontsize=13, y=1.02)
    fig.tight_layout()
    cs_out = os.path.join(FIG, f"gs_issue_live_xsec_{dirname}.png")
    fig.savefig(cs_out, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"{dirname}: phase={phase_out}, far={far_out}, xsec={cs_out}")
    print(f"  center=({cy},{cx}), peak={raw_norm[cy, cx]:.3f}")

print("\nDone.")
