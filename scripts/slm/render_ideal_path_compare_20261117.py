"""离线重渲染理想光路静态测试对比图（英文标题，修复 CJK 字体方框）。

从 ``data/debug/ideal_path_static_test/<ts>/summary.json`` + ``pattern_*.npz``
读取原始数组，重绘 ``compare_<label>.png``（相位 / 下发灰度 / CCD 实测三联）
与 ``overview.png``（CCD 实测远场总览）。纯离线，不访问任何硬件。

用法::

    .venv\\Scripts\\python.exe scripts/slm/render_ideal_path_compare_20261117.py \
        --data-dir data/debug/ideal_path_static_test/20261117_195803
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save_panel(ax: plt.Axes, data: np.ndarray, title: str, vmax: float) -> None:
    """在 ax 上画一张灰度图并设色标。"""
    ax.set_title(title, fontsize=9)
    im = ax.imshow(data, cmap="gray", aspect="equal")
    im.set_clim(0.0, vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def render_compare(
    axs: list[plt.Axes],
    phase: np.ndarray,
    sent: np.ndarray,
    ccd: np.ndarray,
    label: str,
    ccd_peak: float,
) -> None:
    """画三联对比图。"""
    gray_max = float(np.max(sent)) if sent.size else 1023.0
    ccd_vmax = max(50.0, float(ccd_peak) * 2.5)
    cbs = []
    cbs.append(_save_panel(axs[0], phase, f"phase (rad) - {label}", float(np.max(phase) or 6.0)))
    cbs.append(_save_panel(axs[1], sent, f"sent gray - {label}", gray_max))
    cbs.append(_save_panel(axs[2], ccd, f"CCD measured - {label}", ccd_vmax))
    for ax, cb in zip(axs, cbs):
        ax.figure.colorbar(cb, ax=ax, fraction=0.046, pad=0.04)


def render_overview(axs: list[plt.Axes], ccd_imgs: list[tuple[str, np.ndarray]], vmax: float) -> None:
    """画 CCD 实测远场总览。"""
    for ax, (label, img) in zip(axs, ccd_imgs):
        im = ax.imshow(img, cmap="gray", aspect="equal")
        im.set_clim(0.0, vmax)
        ax.set_title(label, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="采集产物目录")
    parser.add_argument("--out-dir", default=None, help="输出目录 (默认同 data-dir)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    ccd_vmax = 250.0

    for pat in summary["patterns"]:
        label = pat["label"]
        npz = data_dir / pat["files"]["npz"]
        z = np.load(npz)
        phase = z["phase"]
        sent = z["sent_gray"]
        ccd = z["ccd"]
        fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
        fig.suptitle(
            f"Ideal-path static test - {label} (model_forward, observed=false)",
            fontsize=11,
        )
        render_compare(axs, phase, sent, ccd, label, pat["ccd_peak_counts"])
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out_dir / f"compare_{label}.png", dpi=140)
        plt.close(fig)
        print(f"[render] compare_{label}.png")

    # overview
    ccd_imgs: list[tuple[str, np.ndarray]] = []
    for pat in summary["patterns"]:
        z = np.load(data_dir / pat["files"]["npz"])
        ccd_imgs.append((pat["label"], z["ccd"]))
    fig, axs = plt.subplots(1, len(ccd_imgs), figsize=(3 * len(ccd_imgs), 3.2))
    if len(ccd_imgs) == 1:
        axs = [axs]
    render_overview(axs, ccd_imgs, ccd_vmax)
    fig.suptitle("CCD measured far-field (ideal / model-forward path)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "overview.png", dpi=140)
    plt.close(fig)
    print("[render] overview.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
