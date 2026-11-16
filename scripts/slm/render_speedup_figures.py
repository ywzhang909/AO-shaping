# -*- coding: utf-8 -*-
"""渲染「提速前后对比」报告的物理读回帧对比图与数据图表。

数据来源全部为实测落盘产物（``report/slm/differentiable_opt/artifacts/<stamp>/``）：
CCD 读回帧取 ``ccd_flat.npz`` / ``ccd_hw_iter_*.npz`` / ``ccd_optimized.npz`` 的
``frame`` 键，指标取 ``run_config.json`` 的 ``comparison.best``。生成四类图，输出到
``report/slm/differentiable_opt/figures/``，文件名统一 ``speedup_`` 前缀：

1. ``speedup_ccd_frames_<stamp>.png`` —— 单运行采集序列：平场 / 3 次硬件迭代 /
   优化帧，99.5 分位定标（与 ``hw_differentiable_phase_opt.py::plot_ccd`` 同口径）。
2. ``speedup_ccd_iters_compare.png`` —— 不同迭代次数的优化帧并排对比。
3. ``speedup_ccd_metrics_compare.png`` —— 跨轮 CCD 指标柱状图（FWHM/峰值/空心度）。
4. ``speedup_hw_iter_trends.png`` —— 实测损失/梯度范数/曝光随硬件迭代趋势。

用法::

    python scripts/slm/render_speedup_figures.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# 与 scripts/slm/hw_differentiable_phase_opt.py 同款引导: 以 `python scripts/...`
# 直跑时 cwd 不在 sys.path 上, 只有脚本自己的目录在。
sys.path.insert(0, str(ROOT))

from scripts._common.plot_style import setup_cjk_font

setup_cjk_font()  # 中文标签, 否则 DejaVu Sans 全画成豆腐块

REPO = ROOT
ARTIFACT_DIR = REPO / "report/slm/differentiable_opt/artifacts"
FIG_DIR = REPO / "report/slm/differentiable_opt/figures"

# (stamp, 展示标签, iters, 日期)
RUNS = [
    ("20261116_015802", "phys_20261116_01", 120, "11-16"),
    ("20261116_024048", "phys_20261116_02", 20, "11-16"),
    ("20261116_114427", "20261116_114424", 120, "11-16"),
    ("20261022_091111", "20261022_091111", 150, "10-22"),
    ("20261022_132915", "20261022_132915", 200, "10-22"),
    ("20261023_012505", "20261023_012505", 150, "10-23"),
]

# 有 npz 帧、可出采集序列图的运行
FRAME_RUNS = RUNS[:3]


def _load_frame(stamp: str, name: str) -> np.ndarray | None:
    path = ARTIFACT_DIR / stamp / name
    if not path.exists():
        return None
    with np.load(path) as data:
        return np.asarray(data["frame"], dtype=float)


def _load_metrics(stamp: str) -> dict:
    cfg = json.loads((ARTIFACT_DIR / stamp / "run_config.json").read_text(encoding="utf-8-sig"))
    return (cfg.get("comparison") or {}).get("best", {}) or {}


def _show_frame(ax: plt.Axes, frame: np.ndarray, title: str) -> None:
    """按 99.5 百分位定标渲染 CCD 帧（与 plot_ccd 同口径）。"""
    a = np.asarray(frame, dtype=float)
    hi = float(np.percentile(a, 99.5)) if a.size else 1.0
    im = ax.imshow(a, cmap="inferno", vmin=0.0, vmax=max(hi, 1e-9), origin="upper")
    ax.set_title(f"{title}\n峰值 {float(a.max()):.1f}, 99.5% {hi:.1f}")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _save(fig: plt.Figure, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def render_frames_per_run() -> list[Path]:
    """单运行采集序列图：平场 / 3 次硬件迭代 / 优化帧。"""
    out: list[Path] = []
    for stamp, label, iters, _ in FRAME_RUNS:
        frames: list[tuple[str, np.ndarray | None]] = [
            ("平场", _load_frame(stamp, "ccd_flat.npz")),
            ("硬件迭代 1", _load_frame(stamp, "ccd_hw_iter_000.npz")),
            ("硬件迭代 2", _load_frame(stamp, "ccd_hw_iter_001.npz")),
            ("硬件迭代 3", _load_frame(stamp, "ccd_hw_iter_002.npz")),
            ("优化帧", _load_frame(stamp, "ccd_optimized.npz")),
        ]
        present = [(t, f) for t, f in frames if f is not None]
        fig, axes = plt.subplots(1, len(present), figsize=(3.1 * len(present), 3.4))
        if len(present) == 1:
            axes = [axes]
        for ax, (title, frame) in zip(axes, present):
            _show_frame(ax, frame, title)
        fig.suptitle(f"{label}（iters={iters}）CCD 读回帧采集序列", fontsize=11)
        fig.tight_layout()
        out.append(_save(fig, f"speedup_ccd_frames_{stamp}.png"))
    return out


def render_iters_compare() -> Path:
    """不同迭代次数的优化帧并排对比。"""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
    for ax, (stamp, label, iters, _) in zip(axes, FRAME_RUNS):
        frame = _load_frame(stamp, "ccd_optimized.npz")
        metrics = _load_metrics(stamp)
        _show_frame(ax, frame, f"{label}\niters={iters}")
        fwhm = metrics.get("ccd_fwhm_px")
        peak = metrics.get("ccd_peak")
        hollow = metrics.get("ccd_hollowness")
        caption = (
            f"CCD FWHM {fwhm:.1f} px · 峰值 {peak:.0f}\n空心度 {hollow:.3f}"
            if fwhm is not None and peak is not None and hollow is not None
            else ""
        )
        ax.set_title(ax.get_title() + "\n" + caption, fontsize=8.5)
    fig.suptitle("不同迭代次数的 CCD 读回帧优化光斑对比（物理采集）", fontsize=11)
    fig.tight_layout()
    return _save(fig, "speedup_ccd_iters_compare.png")


def render_metrics_bars() -> Path:
    """跨轮 CCD 指标柱状图：FWHM / 峰值 / 空心度。"""
    labels = [label for _, label, _, _ in RUNS]
    fwhm = [_load_metrics(s).get("ccd_fwhm_px") or 0.0 for s, _, _, _ in RUNS]
    peak = [_load_metrics(s).get("ccd_peak") or 0.0 for s, _, _, _ in RUNS]
    hollow = [_load_metrics(s).get("ccd_hollowness") or 0.0 for s, _, _, _ in RUNS]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    x = np.arange(len(labels))
    for ax, values, title, fmt in (
        (axes[0], fwhm, "CCD FWHM（px）", "{:.1f}"),
        (axes[1], peak, "CCD 峰值（counts）", "{:.0f}"),
        (axes[2], hollow, "空心度", "{:.3f}"),
    ):
        ax.bar(x, values, color="#2f6fb2", width=0.62)
        for xi, v in zip(x, values):
            ax.text(xi, v, fmt.format(v), ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=20, ha="right")
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("CCD 读回帧指标跨轮对比（6 轮物理实测）", fontsize=11)
    fig.tight_layout()
    return _save(fig, "speedup_ccd_metrics_compare.png")


def render_hw_iter_trends() -> Path:
    """实测损失/梯度范数/曝光随硬件迭代趋势（3 轮各 3 次测量锚定）。"""
    series: dict[str, list[list[float]]] = {"loss": [], "grad": [], "expo": []}
    for stamp, label, _, _ in FRAME_RUNS:
        csv_path = ARTIFACT_DIR / stamp / "history_hw.csv"
        if not csv_path.exists():
            continue
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig")))
        series["loss"].append([float(r["loss"]) for r in rows])
        series["grad"].append([float(r["grad_norm"]) for r in rows])
        series["expo"].append([float(r["exposure_ms"]) for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    iters = [1, 2, 3]
    colors = ["#2f6fb2", "#c0504d", "#4f9d69"]
    for ax, key, title, fmt in (
        (axes[0], "loss", "实测损失（硬件读回）", "{:.3e}"),
        (axes[1], "grad", "梯度范数", "{:.3f}"),
        (axes[2], "expo", "单帧曝光（ms）", "{:.3f}"),
    ):
        for (label, _, _, _), vals, color in zip(FRAME_RUNS, series[key], colors):
            ax.plot(iters, vals, marker="o", color=color, label=label)
        ax.set_xticks(iters)
        ax.set_title(title, fontsize=9.5)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5)
    fig.suptitle("硬件测量锚定迭代趋势（3 轮 × 3 次，物理读回帧）", fontsize=11)
    fig.tight_layout()
    return _save(fig, "speedup_hw_iter_trends.png")


def main() -> None:
    outputs: list[Path] = []
    outputs.extend(render_frames_per_run())
    outputs.append(render_iters_compare())
    outputs.append(render_metrics_bars())
    outputs.append(render_hw_iter_trends())
    for path in outputs:
        print(f"written: {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
