"""Cold-start 随机成形的逐迭代过程动画: 下发相位 / 实测远场 / 模型预测三联对比.

读 ``data/debug/slm_transfer/cold_start_20261026_1004/<run>/`` 下的
``iterations.pkl`` (每轮的下发相位与 metrics) 和 ``capture_*.npz`` (每块的 CCD 原始
帧与 ``model_phase``), 把同一 ``model_phase`` 再前向一次得到预测光斑, 输出
``report/shape_trials/cold_start/figures/cold_start_<run>_process.gif``: 上排 = 瞳面
相位 / 实测远场 / 模型预测, 下排 = 相位变化 (vs 初始) / metrics 文本 / loss 轨迹。

会话数据产生于另一台机器的实验系统, 所以这个生成器是**离线**的: 只读已保存的
``data/debug`` 产物, 不打开任何设备。路径一律从 ``__file__`` 推导, 与运行时的
工作目录无关。
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter

from ao_shaping.drivers.sim.slm_shaping_bench import (
    SquareShapingConfig,
    forward_intensity,
)

#: 面板标题里的中文需要一款含 CJK 字形的字体。这是操作系统资源而不是仓库位置,
#: 没有 ``__file__`` 相对写法, 所以先探测再回落到 matplotlib 的 rcParams 别名,
#: 缺字体的机器只是退回默认 sans-serif 而不会失败。
_FONT_PATH = "C:/Windows/Fonts/msyh.ttc"
if os.path.exists(_FONT_PATH):
    fm.fontManager.addfont(_FONT_PATH)
    _font_prop = fm.FontProperties(fname=_FONT_PATH)
    plt.rcParams["font.family"] = _font_prop.get_name()
    plt.rcParams["font.sans-serif"] = [_font_prop.get_name(), "SimHei", "SimSun"]
else:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

BASE = ROOT / "data/debug/slm_transfer/cold_start_20261026_1004"
FIG_DIR = ROOT / "report/shape_trials/cold_start/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

run_name = "random_formation"
label = "random_formation"
run_dir = BASE / run_name

iters = pickle.load(open(run_dir / "iterations.pkl", "rb"))
iters_sorted = sorted(iters, key=lambda r: (r["block"], r["iteration"]))
total_iters = len(iters_sorted)

caps = sorted([f for f in os.listdir(run_dir) if f.startswith("capture_") and f.endswith(".npz")])

frames_data = []
for i, rec in enumerate(iters_sorted):
    cap_idx = min(rec["block"] * (len(caps) // 4) + min(rec["iteration"] // 2, (len(caps) // 4) - 1), len(caps) - 1)
    z = np.load(run_dir / caps[cap_idx], allow_pickle=True)
    ccd_frame = z["raw_frames"].mean(axis=0).astype(float)
    ccd_frame = np.clip(ccd_frame, 0, None)
    if ccd_frame.max() > 0:
        ccd_frame = ccd_frame / ccd_frame.max()

    model_phase = z["model_phase"]
    cfg = SquareShapingConfig(
        n_grid=128,
        aperture_size=128 * 8e-6,
        wavelength=1064e-9,
        focal_length=0.125,
        target_side_px=80,
        far_field_padding=3,
    )
    pred_intensity = forward_intensity(model_phase, cfg)
    pred_intensity = np.clip(pred_intensity, 0, None)
    if pred_intensity.max() > 0:
        pred_intensity = pred_intensity / pred_intensity.max()

    phase_wrapped = np.mod(rec["phase"] + np.pi, 2 * np.pi) - np.pi

    frames_data.append((rec, ccd_frame, pred_intensity, phase_wrapped))

fig, axes = plt.subplots(2, 3, figsize=(14, 8.4))
writer = PillowWriter(fps=4.0)

for index, (rec, ccd_frame, pred_intensity, phase_wrapped) in enumerate(frames_data):
    for row in axes:
        for ax in row:
            ax.clear()

    terms = rec["terms"]

    axes[0][0].imshow(phase_wrapped, cmap="twilight", origin="upper", vmin=-np.pi, vmax=np.pi)
    axes[0][0].set_title("SLM 瞳面相位 (rad)", fontsize=11)
    axes[0][0].set_xticks([])
    axes[0][0].set_yticks([])

    axes[0][1].imshow(ccd_frame, cmap="inferno", origin="upper")
    axes[0][1].set_title("CCD 远场光斑", fontsize=11)
    axes[0][1].set_xticks([])
    axes[0][1].set_yticks([])

    axes[0][2].imshow(pred_intensity, cmap="inferno", origin="upper")
    axes[0][2].set_title("模型预测光斑", fontsize=11)
    axes[0][2].set_xticks([])
    axes[0][2].set_yticks([])

    if index > 0:
        initial_phase = frames_data[0][3]
        diff = phase_wrapped - initial_phase
        axes[1][0].imshow(diff, cmap="coolwarm", origin="upper", vmin=-np.pi, vmax=np.pi)
        axes[1][0].set_title("相位变化 (vs 初始)", fontsize=11)
        axes[1][0].set_xticks([])
        axes[1][0].set_yticks([])
    else:
        axes[1][0].text(0.5, 0.5, "初始相位 (无变化)", ha="center", va="center", fontsize=10)
        axes[1][0].axis("off")

    axes[1][1].axis("off")
    axes[1][1].set_title("metrics (模型 / 目标)", fontsize=11)
    combined_text = (
        "【正向预测模型】\n"
        "  loss: %.4f\n" % terms['loss'] +
        "  grad_norm: %.4f\n" % terms['gradient_norm'] +
        "  phase_rms: %.4f\n" % terms['phase_rms'] +
        "  mode: %s\n" % terms['mode'] +
        "  step: %d\n" % terms['step'] +
        "\n【目标 metrics】\n"
        "  CV: %.4f\n" % terms['l_cv'] +
        "  EE_ROI: %.4f\n" % terms['ee_roi'] +
        "  shape_loss: %.4f\n" % terms['shape_loss'] +
        "  iteration: %d\n" % rec['iteration'] +
        "  block: %d\n" % rec['block'] +
        "  total: %d/%d" % (index+1, total_iters)
    )
    axes[1][1].text(0.05, 0.95, combined_text, transform=axes[1][1].transAxes,
                    fontsize=8.5, va="top", family="sans-serif")

    axes[1][2].clear()
    x_so_far = list(range(index + 1))
    loss_so_far = [r["terms"]["loss"] for r, _, _, _ in frames_data[:index+1]]
    axes[1][2].plot(x_so_far, loss_so_far, color="steelblue", linewidth=1.0)
    axes[1][2].set_title("Loss 轨迹", fontsize=11)
    axes[1][2].set_xlabel("迭代")
    axes[1][2].set_ylabel("loss")
    axes[1][2].grid(alpha=0.3)

    fig.suptitle(f"{label} - Block {rec['block']} Iter {rec['iteration']} (总 {index+1}/{total_iters})", fontsize=12)
    fig.tight_layout()

    if index == 0:
        writer.setup(fig, FIG_DIR / f"cold_start_{run_name}_process.gif", dpi=100)
    writer.grab_frame()

writer.finish()
plt.close(fig)
print(f"GIF saved: {FIG_DIR / f'cold_start_{run_name}_process.gif'}")
