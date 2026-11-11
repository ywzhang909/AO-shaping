#!/usr/bin/env python3
"""生成棋盘格 LUT 探测报告的全部配图 + CCD 扫描 GIF。

数据 (只读, 已落盘):
  data/debug/lut_checker_cells_20261111/summary.json            (5 尺寸 retained 曲线)
  data/debug/lut_checker_20261111/order_reconstruction_v4.json  (65 点双通道重建)
  data/debug/lut_checker_20261111/frames/*.npy                  (CCD 帧)

输出: report/slm/lut_checker_20261111/figures/
  01_cells_sweep.png      尺寸扫描 retained(g)
  02_two_channel.png      0 级 + 对角级 + 全图功率 vs g
  03_delta_eff.png        标量反演 Δ_eff(g) 与 |M|(g)
  04_spot_montage.png     关键灰度 CCD 画面 (flat/g0/g256/g496/g992)
  05_checker_scan.gif     CCD 扫描动画
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# CJK 字体回退 (Windows 中文字体; 避免标题变豆腐块)
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

REPO = Path(__file__).resolve().parents[2]
CELLS_D = REPO / "data/debug/lut_checker_cells_20261111"
LUT_D = REPO / "data/debug/lut_checker_20261111"
OUT = REPO / "report/slm/lut_checker_20261111/figures"

CY, CX = 1102, 1221          # 零级中心 (65 点扫描)
CELLS = 24
OFF = 157.0 * 24.0 / CELLS   # 对角级单轴偏移 (px)
C4 = 4.0 * (2 / np.pi) ** 4  # 0.657 (4 对角级合计系数)
NOMINAL_PI = 496.5           # 名义 π 灰度 (2π=993)


def load_summary_cells() -> dict:
    return json.loads((CELLS_D / "summary.json").read_text(encoding="utf-8"))


def load_recon() -> dict:
    return json.loads((LUT_D / "order_reconstruction_v4.json").read_text(encoding="utf-8"))


def load_frame(path: Path) -> np.ndarray:
    arr = np.load(path).astype(np.float64)
    return np.maximum(arr - float(np.median(arr)), 0.0)


def crop_roi(arr: np.ndarray, half: int) -> np.ndarray:
    return arr[CY - half:CY + half + 1, CX - half:CX + half + 1]


def fig_cells_sweep(summary: dict, out: Path) -> None:
    gray = np.array(summary["gray"])
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    cols = plt.cm.viridis(np.linspace(0, 0.85, len(summary["per_cells"])))
    for point, color in zip(summary["per_cells"], cols):
        curve = np.array(point["retained_curve"])
        ax.plot(gray, curve, color=color, lw=1.6, label=f"cells={point['cells']}")
        ax.plot(point["valley_gray"], point["retained_min"], "o", color=color, ms=5)
    ax.axhline(0.25, color="0.4", ls="--", lw=0.8,
               label="单标量场 q=1 下界 (1−|M|)²/4")
    ax.axvline(NOMINAL_PI, color="0.6", ls=":", lw=0.8)
    ax.text(NOMINAL_PI, 1.02, "名义 π = 496.5", rotation=0, fontsize=7,
            color="0.4", ha="center")
    ax.set_xlabel("灰度 g")
    ax.set_ylabel("retained = P₀(checker)/P₀(flat)")
    ax.set_ylim(0.2, 1.02)
    ax.set_xlim(-20, 1043)
    ax.set_title("棋盘格单元尺寸扫描: 0 级谷值不随尺寸变化 (跨 5 尺寸 <4.2%)",
                 fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_two_channel(recon: dict, out: Path) -> None:
    rows = recon["rows"]
    gray = np.array([x["gray"] for x in rows])
    zero = np.array([x["X"] for x in rows])
    diag = np.array([x["Ysum"] for x in rows])
    total = np.array([x["total"] for x in rows])
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    ax.plot(gray, zero, "C0", lw=1.6, label="0 级 X = P₀/Pflat")
    ax.plot(gray, diag, "C3", lw=1.6, label="4 对角级 Σ = P_diag/Pflat")
    ax.plot(gray, total, "C7", lw=1.2, ls=":", label="全图总功率 / flat (≈1, 恒定)")
    ax.axvline(NOMINAL_PI, color="0.6", ls=":", lw=0.8)
    ax.text(NOMINAL_PI, 1.02, "名义 π", fontsize=7, color="0.4", ha="center")
    ax.set_xlabel("灰度 g")
    ax.set_ylabel("相对功率 (除以 flat 帧)")
    ax.set_xlim(-20, 1043)
    ax.set_ylim(0, 1.25)
    ax.set_title("双通道实测: 0 级谷值≈0.52 时对角级≈0.34, 全图功率恒定",
                 fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_delta_eff(recon: dict, out: Path) -> None:
    rows = recon["rows"]
    gray = np.array([x["gray"] for x in rows])
    delta = np.array([x["Ddeg"] for x in rows])
    abs_m = np.array([x["absM"] for x in rows])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.9), dpi=150)
    ax1.plot(gray, delta, "C0", lw=1.6)
    ax1.axvline(NOMINAL_PI, color="0.6", ls=":", lw=0.8)
    ax1.axhline(180, color="0.4", ls="--", lw=0.8)
    ax1.axhline(90, color="C3", ls="-.", lw=1.0)
    ax1.text(20, 182, "目标 π (180°)", fontsize=7, color="0.4")
    ax1.text(20, 93, "反演天花板 2·asin√q, q≈0.5 → 90°", fontsize=6.5, color="C3")
    ax1.set_xlabel("灰度 g")
    ax1.set_ylabel("等效相位台阶 Δ_eff (deg)")
    ax1.set_xlim(-20, 1043)
    ax1.set_ylim(-10, 200)
    ax1.set_title("标量反演 Δ_eff 峰值 ~90° (不等于 LC 真实相位未到 π)", fontsize=8.5)
    ax1.grid(alpha=0.3)
    ax2.plot(gray, abs_m, "C2", lw=1.6)
    ax2.axhline(1.0, color="0.4", ls="--", lw=0.8)
    ax2.text(20, 1.02, "|M|=1 (纯相位)", fontsize=7, color="0.4")
    ax2.set_xlabel("灰度 g")
    ax2.set_ylabel("|M| = t_g/t_0")
    ax2.set_xlim(-20, 1043)
    ax2.set_ylim(0.9, 1.25)
    ax2.set_title("|M| ≈ 1.08 恒定 (闭合偏差, 非串扰独立证据)", fontsize=8.5)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_spot_montage(out: Path) -> None:
    keys = [("flat", None), ("g0", 0), ("g256", 256), ("g496", 496),
            ("g992", 992), ("g1023", 1023)]
    # 半宽必须 > OFF (=157, cells=24 的对角级单轴偏移), 否则 166 行的 within-ROI
    # 判断恒为假, 白色十字一个都画不出来 (旧值 140 就是这个 bug)。
    half = 190
    fig, axes = plt.subplots(2, 3, figsize=(8.8, 5.0), dpi=130)
    im = None
    for ax, (name, gray_value) in zip(axes.ravel(), keys):
        fname = "flat.npy" if name == "flat" else f"checker_g{gray_value:04d}.npy"
        arr = load_frame(LUT_D / "frames" / fname)
        roi = crop_roi(arr, half)
        vmax = float(np.percentile(roi, 99.5))
        im = ax.imshow(roi, cmap="inferno", vmin=0, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("flat" if name == "flat" else f"g={gray_value}", fontsize=8)
        if name != "flat":
            for scale_y in (-1, 1):
                for scale_x in (-1, 1):
                    off_y = int(round(scale_y * OFF))
                    off_x = int(round(scale_x * OFF))
                    if abs(off_y) <= half and abs(off_x) <= half:
                        ax.plot(half + off_x, half + off_y, "w+", ms=6, mew=0.8)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    fig.suptitle("CCD 探测画面 (零级 ROI, 对角级白色十字为预测位置)", fontsize=9)
    fig.savefig(out)
    plt.close(fig)


def gif_scan(out: Path) -> None:
    half = 120
    # 帧清单直接来自磁盘, 而不是 range(0, 1024, 16): 后者恰好停在 1008, 丢掉
    # g=1023 (扫描的真实终点)。glob 保证每个实测灰度都进 GIF 且不虚构缺失点。
    grays = sorted(
        int(p.stem.removeprefix("checker_g"))
        for p in (LUT_D / "frames").glob("checker_g*.npy")
    )
    rois = [
        crop_roi(load_frame(LUT_D / "frames" / f"checker_g{g:04d}.npy"), half)
        for g in grays
    ]
    # 逐帧自适应亮度会把 ~0.5 的零级谷值拉回同一视亮度, 正好抹掉 GIF 想展示的
    # 现象 (g496 峰值 64 会被归一到和 g0 峰值 124 一样亮)。改用全序列固定上限,
    # 让谷值的真实变暗在动画里可见。
    vmax = float(max(np.percentile(roi, 99.9) for roi in rois))
    frames = []
    for gray_value, roi in zip(grays, rois):
        raw = (np.clip(roi / max(vmax, 1e-9), 0, 1) * 255).astype(np.uint8)
        img = Image.fromarray(raw).convert("RGB")
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, 132, 22), fill=(0, 0, 0))
        draw.text((7, 6), f"g={gray_value:4d}", fill=(255, 255, 255))
        frames.append(img.convert("P", palette=Image.Palette.ADAPTIVE, colors=256))
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=120,
                   loop=0, optimize=True)
    print(f"GIF: {len(frames)} 帧 (g={grays[0]}…{grays[-1]}, 固定亮度上限 "
          f"{vmax:.0f}) -> {out}")


def main() -> int:
    if not (CELLS_D / "summary.json").exists():
        print(f"缺 cells summary: {CELLS_D / 'summary.json'}")
        return 1
    if not (LUT_D / "order_reconstruction_v4.json").exists():
        print(f"缺重建结果: {LUT_D / 'order_reconstruction_v4.json'}")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    summary = load_summary_cells()
    recon = load_recon()
    fig_cells_sweep(summary, OUT / "01_cells_sweep.png")
    fig_two_channel(recon, OUT / "02_two_channel.png")
    fig_delta_eff(recon, OUT / "03_delta_eff.png")
    fig_spot_montage(OUT / "04_spot_montage.png")
    gif_scan(OUT / "05_checker_scan.gif")
    for path in sorted(OUT.glob("*")):
        print(f"  {path.name}  {path.stat().st_size / 1024:.0f} KiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
