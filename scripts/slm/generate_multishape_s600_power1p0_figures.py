"""Figures for the 1.0-power s600 lanes (05–08) of the 2026-10-29 campaign.

The 200 px lanes ran at laser power 0.7; the power was raised to 1.0 mid-campaign
and lanes 05–08 were re-run/new-run at 1.0.  Those four sessions have no figure
in the report yet (they are quoted only as tables), which makes the shape-to-shape
comparison hard to eyeball: every shape uses its own core mask, so the `cv`
*levels* are not comparable and only the seed→final `dCV` is.

This script draws exactly that: one panel per shape, `dCV` per exposure rung with
the replicate-spread half-width as the error bar, coloured by the audit verdict.

Every number comes from :func:`scripts.slm.audit_replay_blocks.audit_block` /
``summarise_block`` themselves — the same functions that print the table in the
live log — so a figure can never drift from the table it illustrates.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_replay_blocks import audit_block, summarise_block
from generate_multishape_scale_live_figures import (
    _save,
    use_cjk_font,
    SHAPE_LABELS,
)

SHAPES = ("circle", "square", "triangle", "star")
SESSIONS = {shape: f"p20261029_analytic_init_s600/{shape}" for shape in SHAPES}
DEFAULT_CAMPAIGN = "data/debug/slm_transfer/multishape_nogs_20261029"
DEFAULT_OUT = "report/shape_trials/multishape_scale/figures"
FIGURE_NAME = "audit_dcv_s600_power1p0_20261029.png"

#: reason -> (bar colour, hatch, legend label)
VERDICT_STYLE = {
    "cv_improved_beyond_noise": ("#2E7D32", None, "supported（改善越过重复离散）"),
    "wrong_direction": ("#C62828", None, "wrong_direction（反向变差）"),
    "no_difference": ("#8C8C8C", None, "no_difference（差异在离散内）"),
    "insufficient_repeats": ("#B8860B", None, "insufficient（重复不足）"),
    "no_seed": ("#FFFFFF", "///", "无 seed 行（不能成对，不判）"),
}


def _rungs(directory: Path) -> tuple[dict, list[dict]]:
    """Per-exposure rungs of one session's replay block, straight from the audit."""
    summary = summarise_block(audit_block(directory))
    rungs = []
    for key, group in summary["groups"].items():
        pair = group["pairs"][0]["verdict"] if group["pairs"] else None
        rungs.append(
            dict(
                exposure=float(key.split("ms/")[0]),
                n=group["n"],
                cv=group["cv_mean"],
                ee=group["ee_rel_mean"],
                delta=None if pair is None else pair["delta_cv"],
                half=None if pair is None else pair.get("noise_half_width"),
                reason="no_seed" if pair is None else pair["reason"],
            )
        )
    rungs.sort(key=lambda rung: rung["exposure"])
    return summary, rungs


def _panel(ax, shape: str, summary: dict, rungs: list[dict]) -> None:
    xs = np.arange(len(rungs))
    deltas = [0.0 if rung["delta"] is None else rung["delta"] for rung in rungs]
    halves = [0.0 if rung["half"] is None else rung["half"] for rung in rungs]
    colors = [VERDICT_STYLE[rung["reason"]][0] for rung in rungs]
    hatches = [VERDICT_STYLE[rung["reason"]][1] for rung in rungs]

    for x, delta, half, color, hatch in zip(xs, deltas, halves, colors, hatches):
        ax.bar(
            x,
            delta,
            width=0.55,
            color=color,
            edgecolor="#333333",
            linewidth=0.8,
            hatch=hatch,
            zorder=3,
        )
        if half:
            ax.errorbar(
                x,
                delta,
                yerr=half,
                fmt="none",
                ecolor="#222222",
                elinewidth=1.2,
                capsize=5,
                zorder=4,
            )

    ax.axhline(0.0, color="#333333", linewidth=1.0, zorder=2)
    # A band of +/- the median replicate half-width makes "does this bar clear the
    # measurement spread" readable without reading the numbers.
    finite = [half for half in halves if half]
    if finite:
        band = float(np.median(finite))
        ax.axhspan(-band, band, color="#2E7D32", alpha=0.10, zorder=1)
        ax.text(
            0.99,
            band,
            f"±中位重复离散 {band:.3f}",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=8,
            color="#2E7D32",
        )

    for x, rung in zip(xs, rungs):
        if rung["delta"] is None:
            ax.text(x, 0.0, "无 seed", ha="center", va="center", fontsize=8, rotation=90)

    ax.set_xticks(xs)
    ax.set_xticklabels(
        [
            f"{rung['exposure']:.1f} ms\nn={rung['n']} · ee {rung['ee']:.2f}"
            for rung in rungs
        ],
        fontsize=8,
    )
    ax.set_ylabel("dCV（final − seed）", fontsize=9)
    usable = summary["n_usable"]
    total = summary["n_rows"]
    ax.set_title(
        f"{SHAPE_LABELS.get(shape, shape)}   有效 {usable}/{total} 行",
        fontsize=10,
    )
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)


def build_figure(campaign: Path, out: Path, *, dpi: float) -> Path:
    use_cjk_font()
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4), sharey=True)
    used: dict[str, tuple[str, str | None]] = {}
    for ax, shape in zip(axes.ravel(), SHAPES):
        summary, rungs = _rungs(campaign / SESSIONS[shape] / "common_replay")
        if not rungs:
            ax.set_visible(False)
            continue
        _panel(ax, shape, summary, rungs)
        for rung in rungs:
            color, hatch, label = VERDICT_STYLE[rung["reason"]]
            used[rung["reason"]] = (color, hatch, label)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="#333333",
                      hatch=hatch, label=label)
        for color, hatch, label in used.values()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        fontsize=8.5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "600 px 四形状 · 激光功率 1.0 · 冻结光阑重放的逐档配对"
        "（误差棒 = final 重复离散半宽，n=4，95% t）",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    return _save(fig, out, dpi)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign", type=Path, default=Path(DEFAULT_CAMPAIGN))
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUT) / FIGURE_NAME)
    parser.add_argument("--dpi", type=float, default=150.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    build_figure(args.campaign, out, dpi=args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
