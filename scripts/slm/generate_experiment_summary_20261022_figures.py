"""重跑 2026-10-15 GS 五轮对照的相位/远场/截面图 (report/shape_trials/figures/)。

删除条件 (AGENTS.md 「scripts/ 目录」规则 3): 其图片 gs_issue_live_phase_*.png /
gs_issue_live_farfield_*.png 仍被活报告
``report/shape_trials/gs_refine/gs_issue_live_20261015.md`` 引用 (结论见该报告与
``report/shape_trials/experiment_summary_20261016.md``)。当
report/shape_trials/figures/ 下这组图不再需要重跑时可删。

"""
import numpy as np
from loguru import logger
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

BASE = Path("data/debug/slm_gs_refine")
FIG_DIR = Path("report/shape_trials/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

trials = [
    ("20261015_205654", "plain", 0.6123),
    ("20261015_205848", "weighted", 0.7793),
    ("20261015_210048", "adaptive-feedback", 0.7064),
    ("20261015_210155", "plain_复测", 0.6152),
    ("20261015_211345", "plain_再复测", 0.6080),
]

for trial_dir, method, cv_val in trials:
    trial_dir_path = BASE / trial_dir

    phase_file = trial_dir_path / "best_phase.npy"
    far_path = trial_dir_path / "best_far_field.png"

    if not phase_file.exists() or not far_path.exists():
        logger.info(f"跳过 {trial_dir}: 文件不存在")
        continue

    # 加载相位
    phase = np.load(phase_file)
    phase_wrapped = np.mod(phase, 2 * np.pi)

    # 加载远场
    img = np.array(Image.open(far_path)).astype(float)
    if img.ndim == 3 and img.shape[2] >= 3:
        img_gray = img[:, :, :3].mean(axis=2)
    else:
        img_gray = img.astype(float)
    img_gray = np.clip(img_gray, 0, None)

    # 找光斑中心
    cy, cx = np.unravel_index(np.argmax(img_gray), img_gray.shape)

    # X 截面（横向穿过中心行）
    x_cross = img_gray[cy, :]
    # Y 截面（垂直穿过中心列）
    y_cross = img_gray[:, cx]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # X 方向
    ax0 = axes[0]
    ax0.plot(
        np.arange(len(x_cross)),
        x_cross,
        color="steelblue",
        linewidth=1.0,
        label="X 截面",
    )
    ax0.axvline(cx, color="red", linestyle="--", alpha=0.7, label=f"中心 x={cx}")
    ax0.set_title(f"{method} — X 方向光强截面 (row={cy})", fontsize=11)
    ax0.set_xlabel("column (px)", fontsize=10)
    ax0.set_ylabel("归一强度", fontsize=9)
    ax0.set_xlim(0, len(x_cross))
    ax0.legend(fontsize=8)

    # Y 方向
    ax1 = axes[1]
    ax1.plot(
        np.arange(len(y_cross)),
        y_cross,
        color="darkorange",
        linewidth=1.0,
        label="Y 截面",
    )
    ax1.axvline(cy, color="red", linestyle="--", alpha=0.7, label=f"中心 y={cy}")
    ax1.set_title(f"{method} — Y 方向光强截面 (col={cx})", fontsize=11)
    ax1.set_xlabel("row (px)", fontsize=9)
    ax1.set_ylabel("归一强度", fontsize=9)
    ax1.set_xlim(0, len(y_cross))
    ax1.legend(fontsize=8)

    fig.suptitle(f"{method} 相位与远场 (CV={cv_val})", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_phase = FIG_DIR / f"gs_issue_live_phase_{trial_dir}.png"
    out_far = FIG_DIR / f"gs_issue_live_farfield_{trial_dir}.png"
    out_xsec = FIG_DIR / f"gs_issue_live_farfield_{trial_dir}_xy_profile.png"

    fig.savefig(out_phase, dpi=120)
    fig.savefig(out_far, dpi=120)
    fig.savefig(out_xsec, dpi=120)
    plt.close(fig)

    logger.info(f"Saved: {out_phase.name}")
    logger.info(f"Saved: {out_far.name}")
    logger.info(f"Saved: {out_xsec.name}")

logger.info("\nDone! All images rendered.")
