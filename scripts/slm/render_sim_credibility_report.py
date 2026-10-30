"""渲染 `report/simulation/sim_credibility/sim_credibility_20261030.md` 的图。

这份报告的**数字**不是本脚本算的 —— 它们由两个生产者落盘到 ``artifacts/``:

* ``artifacts/direction_corpus_20261030.json`` — ``scripts/ml/analyze_sim2f_direction_corpus.py``
  (孪生对「变化方向」的信息量; 它自己画 ``figures/direction_corpus_20261030.png``)
* ``artifacts/differentiable_20261030.json`` — ``scripts/slm/analyze_sim_differentiable.py``
  (损失函数缺陷复现 + 可微整形有效性 + 模型/参数敏感性)

本脚本只做**渲染**: 读上面两份 JSON, 画可微与敏感性三张图。正文由人从 JSON 写成,
所以重跑本脚本不会改动任何一个数字或一句话 —— 这是有意的: 数字的真源是 JSON,
不是这张图, 也不是正文。

用法::

    python scripts/slm/render_sim_credibility_report.py

坐标轴标签用英文: 报告里既有的一张图 (方向余弦直方图) 也是英文标签, 且 matplotlib
的 CJK 字体在各机器上不保证存在, 缺字形会把标签渲染成方框。

两条画图约定 (来自 2026-10-30 的读图复核, 改图时别退回去):

* 柱状图的分组键是 ``(shape, aspect_ratio)`` 而**不是** ``shape`` —— 长方形目标是
  canonical 的 ``square`` 形状加 ``aspect_ratio=2``, 只按 ``shape`` 分组会把长方形
  那一组整个吞掉 (画出来只剩三组, 且无人报错)。
* 图例一律放在坐标轴**外面** (轴下方): 放在轴内会压住柱子和标注; 对数轴上旋转 90°
  的数字标注还要额外 4 倍顶部留白, 否则顶端那根柱子的标注会被上边框切掉。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts._common import savefig

REPORT_DIR = ROOT / "report" / "simulation" / "sim_credibility"
ART_DIR = REPORT_DIR / "artifacts"
FIG_DIR = REPORT_DIR / "figures"
STAMP = "20261030"

#: 分组键 = (canonical 形状名, 长宽比)。长方形不是独立形状, 是 2:1 的正方形目标。
CASE_LABEL = {
    ("square", 1.0): "square",
    ("square", 2.0): "rectangle 2:1",
    ("circle", 1.0): "circle",
    ("gaussian", 1.0): "gaussian",
}
CASE_ORDER = tuple(CASE_LABEL)

#: 对数轴上旋转标注的顶部留白倍数 (实测: 1.05 倍会被上边框切掉)。
LOG_HEADROOM = 4.0

ALGO_COLOR = {
    "flat-baseline": "#8a8a8a",
    "gs": "#3b6ea5",
    "backprop-flat": "#b0483a",
    "backprop-random": "#c98a3f",
    "backprop": "#b0483a",
    "spgd-sim": "#2f7a4f",
}


def _load(name: str) -> dict:
    path = ART_DIR / name
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _case(row: dict) -> tuple[str, float]:
    return (row["shape"], float(row.get("aspect_ratio", 1.0)))


def _rows_by_case(rows: list[dict], algorithm: str) -> dict[tuple[str, float], dict]:
    """按 ``(shape, aspect_ratio)`` 收集某个算法的那一行。"""
    out: dict[tuple[str, float], dict] = {}
    for row in rows:
        if row.get("algorithm") != algorithm:
            continue
        out.setdefault(_case(row), row)
    return out


def _present_cases(rows: list[dict], algorithms) -> list[tuple[str, float]]:
    seen = {_case(r) for r in rows if r.get("algorithm") in algorithms}
    return [case for case in CASE_ORDER if case in seen]


def _grouped_bars(ax, cases, series, *, ylabel, title, logy=False):
    """``series`` = [(label, {case: value})], 逐形状并排柱。"""
    x = np.arange(len(cases))
    width = 0.8 / max(len(series), 1)
    finite = [
        values[c] for _label, values in series for c in cases
        if c in values and np.isfinite(values[c])
    ]
    for i, (label, values) in enumerate(series):
        vals = [values.get(c, np.nan) for c in cases]
        offset = (i - (len(series) - 1) / 2.0) * width
        bars = ax.bar(
            x + offset,
            vals,
            width * 0.92,
            label=label,
            color=ALGO_COLOR.get(label, None),
        )
        for bar, val in zip(bars, vals):
            if not np.isfinite(val):
                continue
            ax.annotate(
                f"{val:.3g}",
                (bar.get_x() + bar.get_width() / 2.0, val),
                textcoords="offset points",
                xytext=(0, 2),
                ha="center",
                va="bottom",
                fontsize=6.5,
                rotation=90 if logy else 0,
            )
    if logy:
        ax.set_yscale("log")
        low = min(finite)
        ax.set_ylim(bottom=low * 0.35, top=max(finite) * LOG_HEADROOM)
    else:
        ax.set_ylim(0.0, max(finite) * 1.22)
    ax.set_xticks(x)
    ax.set_xticklabels([CASE_LABEL[c] for c in cases], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.legend(
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=len(series),
        frameon=False,
    )


def figure_as_shipped(data: dict) -> Path:
    """``run_benchmark`` 按现状报出的 CV / EE (目标为 sum 归一化)。"""
    rows = data["benchmark_as_shipped"]
    algos = ("gs", "backprop", "spgd-sim")
    cases = _present_cases(rows, algos)
    cv = {a: {c: r["uniformity_cv"] for c, r in _rows_by_case(rows, a).items()} for a in algos}
    ee = {a: {c: r["encircled_energy"] for c, r in _rows_by_case(rows, a).items()} for a in algos}
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    _grouped_bars(
        axes[0],
        cases,
        [(a, cv[a]) for a in algos],
        ylabel="uniformity CV (lower better)",
        title="as shipped: CV (log scale)",
        logy=True,
    )
    _grouped_bars(
        axes[1],
        cases,
        [(a, ee[a]) for a in algos],
        ylabel="encircled energy",
        title="as shipped: EE",
    )
    fig.suptitle(
        "run_benchmark as shipped (128x128, 300 iters): backprop and spgd-sim both stall",
        fontsize=9,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24, top=0.86)
    path = FIG_DIR / f"benchmark_as_shipped_{STAMP}.png"
    savefig(fig, path)
    return path


def figure_mask_fixed(data: dict) -> Path:
    """目标改成 0/1 掩膜后的同一组基准 (GS 基线同口径, 平场基线只在 CV 面板)。"""
    rows = data["benchmark_mask_fixed"]["rows"]
    cv_algos = ("flat-baseline", "gs", "backprop-flat", "backprop-random")
    ee_algos = ("gs", "backprop-flat", "backprop-random")
    cases = _present_cases(rows, ee_algos)
    cv = {a: {c: r["uniformity_cv"] for c, r in _rows_by_case(rows, a).items()} for a in cv_algos}
    ee = {a: {c: r["encircled_energy"] for c, r in _rows_by_case(rows, a).items()} for a in ee_algos}
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    _grouped_bars(
        axes[0],
        cases,
        [(a, cv[a]) for a in cv_algos],
        ylabel="uniformity CV (lower better)",
        title="0/1 mask target: CV (log scale)",
        logy=True,
    )
    _grouped_bars(
        axes[1],
        cases,
        [(a, ee[a]) for a in ee_algos],
        ylabel="encircled energy",
        title="0/1 mask target: EE (a flat frame reads 1.00 -- degenerate)",
    )
    fig.suptitle(
        "target switched to a 0/1 mask (128x128, 300 iters): uniformity wins, energy loses",
        fontsize=9,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24, top=0.86)
    path = FIG_DIR / f"benchmark_mask_fixed_{STAMP}.png"
    savefig(fig, path)
    return path


def figure_sensitivity(data: dict) -> Path:
    """参数扰动下同一个最优相位的表现, 以及两个仿真后端分支的交叉评估。"""
    ms = data["model_sensitivity"]
    nominal = ms["nominal"]
    nominal_text = {
        "distance": f"{nominal['distance']:.4g} m",
        "wavelength": f"{nominal['wavelength'] * 1e9:.0f} nm",
        "cell_spacing": f"{nominal['cell_spacing'] * 1e6:.0f} um",
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))

    ax = axes[0]
    style = {
        "distance": ("#3b6ea5", "o", "distance"),
        "wavelength": ("#b0483a", "s", "wavelength"),
        "cell_spacing": ("#2f7a4f", "^", "cell spacing"),
    }
    for param in ("distance", "wavelength", "cell_spacing"):
        pts = [r for r in ms["sensitivity"] if r["parameter"] == param]
        pts.sort(key=lambda r: r["rel_change"])
        xs = [100.0 * r["rel_change"] for r in pts]
        ys = [r["encircled_energy"] for r in pts]
        color, marker, label = style[param]
        ax.plot(
            xs,
            ys,
            marker=marker,
            color=color,
            label=f"{label} (nominal {nominal_text[param]})",
            linewidth=1.4,
            markersize=4,
        )
    # 三条曲线在 0 % 处重合, 所以标一次就够 —— 逐条标注会三个字符串叠在一起。
    peak = next(
        r["encircled_energy"] for r in ms["sensitivity"] if r["parameter"] == "distance" and r["rel_change"] == 0.0
    )
    ax.axvline(0.0, color="#8a8a8a", linewidth=0.8, linestyle=":")
    ax.annotate(
        f"nominal: EE={peak:.3f} (all three)",
        (0.0, peak),
        textcoords="offset points",
        xytext=(8, 6),
        fontsize=7,
    )
    ax.set_xlabel("parameter deviation from nominal (%)")
    ax.set_ylabel("encircled energy (0/1 mask, asm phase)")
    ax.set_title("same optimized phase, nominal parameters perturbed", fontsize=9)
    ax.set_ylim(0.0, 0.62)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)

    ax = axes[1]
    order = [
        ("fft", "fft", "fft -> fft"),
        ("fft", "asm", "fft -> asm"),
        ("asm", "fft", "asm -> fft"),
        ("asm", "asm", "asm -> asm"),
    ]
    lookup = {(r["optimized_on"], r["evaluated_on"]): r for r in ms["cross_model"]}
    labels = [label for _a, _b, label in order]
    values = [lookup[(a, b)]["encircled_energy"] for a, b, _l in order]
    cvs = [lookup[(a, b)]["uniformity_cv"] for a, b, _l in order]
    colors = ["#2f7a4f" if a == b else "#b0483a" for a, b, _l in order]
    bars = ax.bar(labels, values, color=colors)
    for bar, val, cv in zip(bars, values, cvs):
        ax.annotate(
            f"EE={val:.3f}\nCV={cv:.3g}",
            (bar.get_x() + bar.get_width() / 2.0, val),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=7,
        )
    ax.set_ylabel("encircled energy")
    ax.set_ylim(0.0, 1.12)
    ax.set_title("phase optimized on one backend, scored on the other", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.26, top=0.88)
    path = FIG_DIR / f"model_sensitivity_{STAMP}.png"
    savefig(fig, path)
    return path


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    diff = _load(f"differentiable_{STAMP}.json")
    written = [
        figure_as_shipped(diff),
        figure_mask_fixed(diff),
        figure_sensitivity(diff),
    ]
    for path in written:
        print(f"写出 {path.relative_to(ROOT).as_posix()}")
    print(f"方向余弦图由 scripts/ml/analyze_sim2f_direction_corpus.py 生成, 本脚本不重画: "
          f"{(FIG_DIR / f'direction_corpus_{STAMP}.png').relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
