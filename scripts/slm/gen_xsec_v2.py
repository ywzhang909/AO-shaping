import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob

FIG_DIR = "report/shape_trials/figures"


def load_image_data(run_dir):
    """加载远场图像数据，优先使用原始 .npy 数据，否则使用 best_far_field.png"""
    # 尝试加载原始 .npy 数据（排除相位、拟合系数、源振幅、raw 原始帧）
    npy_files = sorted([f for f in os.listdir(run_dir) if f.endswith(".npy")])
    ccd_npy = [
        f for f in npy_files if not f.startswith(("best_", "fitted_", "source_", "raw"))
    ]

    if ccd_npy:
        arr = np.load(os.path.join(run_dir, ccd_npy[0])).astype(float)
        arr = np.clip(arr, 0, None)
        if arr.max() > 0:
            arr = arr / arr.max()
        return arr, f"原始数据: {ccd_npy[0]}"

    # 否则使用 best_far_field.png（inferno 渲染图，仅作参考）
    far_path = os.path.join(run_dir, "best_far_field.png")
    if os.path.exists(far_path):
        img = np.array(Image.open(far_path)).astype(float)
        if img.ndim == 3 and img.shape[2] >= 3:
            gray = img[:, :, :3].mean(axis=2)
        else:
            gray = img
        gray = np.clip(gray, 0, None)
        if gray.max() > 0:
            gray = gray / gray.max()
        return gray, "渲染图: best_far_field.png"

    return None, None


# 需要生成截面图的运行目录列表
run_dirs = [
    ("data/debug/slm_model_in_loop/20261014_105041", "20261014_105041"),
    ("data/debug/slm_model_in_loop/20261014_105205", "20261014_105205"),
    ("data/debug/slm_model_in_loop/20261014_105339", "20261014_105339"),
    ("data/debug/slm_model_in_loop/20261014_105438", "20261014_105438"),
    ("data/debug/slm_model_in_loop/20261014_105535", "20261014_105535"),
    ("data/debug/slm_model_in_loop/20261014_110203", "20261014_110203"),
    ("data/debug/slm_model_in_loop/20261014_110305", "20261014_110305"),
    (
        "data/debug/slm_model_in_loop_replay/20261014_1115_zero_vs_refit",
        "20261014_1115_zero_vs_refit",
    ),
    ("data/debug/slm_model_in_loop/20261014_170820", "20261014_170820"),
    ("data/debug/slm_model_in_loop/20261014_171043", "20261014_171043"),
    ("data/debug/slm_model_in_loop/20261014_171219", "20261014_171219"),
    ("data/debug/slm_model_in_loop/20261014_171301", "20261014_171301"),
    ("data/debug/slm_model_in_loop/20261014_171340", "20261014_171340"),
    (
        "data/debug/slm_model_in_loop/20261016_1041_uniformity_baseline",
        "20261016_1041_uniformity_baseline",
    ),
    (
        "data/debug/slm_model_in_loop/20261016_1042_uniformity_weighted",
        "20261016_1042_uniformity_weighted",
    ),
    (
        "data/debug/slm_model_in_loop/20261016_1048_zero_refit",
        "20261016_1048_zero_refit",
    ),
]

for run_dir, name in run_dirs:
    img, source = load_image_data(run_dir)
    if img is None:
        print(f"跳过 {run_dir}: 无可用图像数据")
        continue

    h, w = img.shape
    cy, cx = np.unravel_index(np.argmax(img), img.shape)

    x_line = img[cy, :]
    y_line = img[:, cx]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(np.arange(w), x_line, color="steelblue", linewidth=1.0)
    axes[0].axvline(cx, color="red", linestyle="--", alpha=0.7, label=f"中心 x={cx}")
    axes[0].set_title(f"{name} X 截面 (row={cy})")
    axes[0].set_xlabel("column (px)")
    axes[0].set_ylabel("归一强度")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(fontsize=8)

    axes[1].plot(np.arange(h), y_line, color="darkorange", linewidth=1.0)
    axes[1].axvline(cy, color="red", linestyle="--", alpha=0.7, label=f"中心 y={cy}")
    axes[1].set_title(f"{name} Y 截面 (col={cy})")
    axes[1].set_xlabel("row (px)")
    axes[1].set_ylabel("归一强度")
    axes[1].set_ylim(bottom=0)
    axes[1].legend(fontsize=8)

    fig.suptitle(f"{name} ({source})", fontsize=11)
    fig.tight_layout()

    out = os.path.join(FIG_DIR, f"{name}_farfield_xsec.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"{run_dir} -> {out} (中心=({cx},{cy}), 数据源={source})")

print("全部完成。")
