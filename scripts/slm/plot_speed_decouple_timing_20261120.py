# -*- coding: utf-8 -*-
"""Offline figures for the 2026-11-20 SLM/CCD speed-decouple report.

Reads the real capture artifacts on disk (no hardware, no network, no new
measurements) and writes two PNG figures. Re-runnable from any working dir.

Usage (from repo root):
    .venv/Scripts/python.exe scripts/slm/plot_speed_decouple_timing_20261120.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# scripts/slm/<this file> -> parents[2] is the repo root
ROOT = Path(__file__).resolve().parents[2]
CAPTURE_DIR = ROOT / "data" / "debug" / "slm_transfer" / "speed_decouple_capture_20261120_v2"
FIG_DIR = ROOT / "report" / "shape_trials" / "figures"

RUN_SUMMARY = CAPTURE_DIR / "run_summary.json"
DEVICE_LOG = CAPTURE_DIR / "device_log.json"
FIG_FPS = FIG_DIR / "speed_decouple_20261120_fps.png"
FIG_TIMING = FIG_DIR / "speed_decouple_20261120_timing.png"

PLANNED_FPS = 10.0  # planned acquisition rate (spec constant, not from data)
MAX_GAP_S = 2.0  # discard cross-transition gaps (settle / phase generation)


def load_json(path: Path) -> list[dict[str, Any]]:
    """Load a JSON array of objects from ``path`` using UTF-8."""
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        msg = f"expected a JSON array in {path}"
        raise TypeError(msg)
    return data


def pick_cjk_font() -> str | None:
    """Return the first usable Chinese font family, or None for English labels."""
    for candidate in (
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "MS Gothic",
    ):
        try:
            font_manager.findfont(candidate, fallback_to_default=False)
        except (ValueError, OSError):
            continue
        return candidate
    return None


def txt(use_cjk: bool, cjk: str, en: str) -> str:
    """Pick the Chinese label when a CJK font is available, else English."""
    return cjk if use_cjk else en


def assign_segments(entries: list[dict[str, Any]]) -> list[int]:
    """Give each log entry a segment id; a new id starts at each transition."""
    segment = -1
    result: list[int] = []
    for entry in entries:
        action = entry.get("action")
        if action == "transition_begin":
            segment += 1
        elif action == "transition_end":
            result.append(segment)
            segment += 1  # settle / next-begin belong to the following gap
            continue
        result.append(segment)
    return result


def decompose_gaps(entries: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    """Return (display->grab, grab->display) gaps in seconds.

    Only adjacent pairs inside the same transition segment are kept, with a
    strictly positive gap below MAX_GAP_S (excludes cross-transition settle
    and phase-generation time).
    """
    segments = assign_segments(entries)
    d2g: list[float] = []
    g2d: list[float] = []
    for i in range(len(entries) - 1):
        if segments[i] != segments[i + 1]:
            continue
        gap = (entries[i + 1]["monotonic_ns"] - entries[i]["monotonic_ns"]) / 1e9
        if not (0.0 < gap < MAX_GAP_S):
            continue
        pair = (entries[i]["action"], entries[i + 1]["action"])
        if pair == ("display", "grab"):
            d2g.append(gap)
        elif pair == ("grab", "display"):
            g2d.append(gap)
    return d2g, g2d


def plot_fps(runs: list[dict[str, Any]], font: str | None) -> None:
    """Bar chart of achieved measurement FPS for all transitions."""
    use_cjk = font is not None
    labels = [str(r["label"]) for r in runs]
    fps = [float(r["achieved_measure_fps"]) for r in runs]

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
    ax.bar(range(len(fps)), fps, color="#1f4e79", width=0.72)
    ax.axhline(PLANNED_FPS, color="#b22222", ls="--", lw=1.8,
               label=txt(use_cjk, f"计划 {PLANNED_FPS:.0f} fps", f"planned {PLANNED_FPS:.0f} fps"))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7.5)
    ax.set_ylim(0, PLANNED_FPS * 1.08)
    ax.set_ylabel(txt(use_cjk, "实测测量帧率 / fps", "achieved measure FPS"), fontsize=11)
    ax.set_xlabel(txt(use_cjk, "过渡（目标形状 / 相位来源）", "transition (target / phase source)"),
                  fontsize=11)
    ax.set_title(
        txt(
            use_cjk,
            f"实验 ~{statistics.median(fps):.1f} fps vs 计划 {PLANNED_FPS:.0f} fps，SLM 写入为瓶颈",
            f"measured ~{statistics.median(fps):.1f} fps vs planned {PLANNED_FPS:.0f} fps; SLM write is the bottleneck",
        ),
        fontsize=12,
    )
    ax.text(
        0.02, 0.94,
        txt(
            use_cjk,
            f"30 次过渡：实测中位数 {statistics.median(fps):.2f} fps "
            f"(范围 {min(fps):.2f}–{max(fps):.2f})；计划 {PLANNED_FPS:.0f} fps",
            f"30 transitions: median {statistics.median(fps):.2f} fps "
            f"(range {min(fps):.2f}-{max(fps):.2f}); planned {PLANNED_FPS:.0f} fps",
        ),
        transform=ax.transAxes, va="top", fontsize=9.5, color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#bbbbbb", alpha=0.9),
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.86, bottom=0.32)
    fig.savefig(FIG_FPS)
    plt.close(fig)


def plot_timing(d2g: list[float], g2d: list[float], font: str | None) -> None:
    """Boxplots of CCD expose+readout vs SLM write+slot rotate gaps."""
    use_cjk = font is not None
    labels = [
        txt(use_cjk, "CCD 曝光+读出", "CCD expose+readout"),
        txt(use_cjk, "SLM 写入+换槽", "SLM write+slot rotate"),
    ]
    medians = [statistics.median(d2g), statistics.median(g2d)]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.boxplot([d2g, g2d], tick_labels=labels, widths=0.5, showfliers=True,
               medianprops=dict(color="#b22222", lw=1.8))
    for x, med in enumerate(medians, start=1):
        ax.text(x, med, f" {med:.3f} s", va="center", fontsize=10, color="#b22222")

    ax.set_ylabel(txt(use_cjk, "相邻事件间隔 / s", "adjacent-event gap / s"), fontsize=11)
    ax.set_title(
        txt(
            use_cjk,
            f"单帧耗时分解：中位数 {medians[0]:.3f} s vs {medians[1]:.3f} s（SLM 更慢）",
            f"per-frame timing: median {medians[0]:.3f} s vs {medians[1]:.3f} s (SLM slower)",
        ),
        fontsize=12,
    )
    ax.text(
        0.5, 0.95,
        txt(
            use_cjk,
            f"相邻 display→grab: n={len(d2g)}；grab→下一 display: n={len(g2d)}",
            f"adjacent display->grab: n={len(d2g)}; grab->next display: n={len(g2d)}",
        ),
        transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color="#333333",
    )
    ax.grid(axis="y", alpha=0.3)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.84, bottom=0.14)
    fig.savefig(FIG_TIMING)
    plt.close(fig)


def main() -> None:
    """Load artifacts, write both figures, and print the key statistics."""
    font = pick_cjk_font()
    if font:
        plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
        print(f"font: {font} (Chinese labels)")
    else:
        print("font: DejaVu Sans (English labels; no CJK font available)")
    plt.rcParams["axes.unicode_minus"] = False

    runs = load_json(RUN_SUMMARY)
    entries = load_json(DEVICE_LOG)
    fps = [float(r["achieved_measure_fps"]) for r in runs]
    d2g, g2d = decompose_gaps(entries)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_fps(runs, font)
    plot_timing(d2g, g2d, font)

    print(f"run_summary: {len(runs)} transitions")
    print(f"device_log: {len(entries)} entries")
    print(f"display->grab pairs: {len(d2g)}")
    print(f"grab->display pairs: {len(g2d)}")
    print(
        f"fps min/med/max: {min(fps):.4f} / {statistics.median(fps):.4f} / {max(fps):.4f}"
    )
    print(f"CCD expose+readout gap median: {statistics.median(d2g):.6f} s")
    print(f"SLM write+slot rotate gap median: {statistics.median(g2d):.6f} s")
    print(f"saved: {FIG_FPS}")
    print(f"saved: {FIG_TIMING}")


if __name__ == "__main__":
    main()
