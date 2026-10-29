"""Figures of the in-experiment multishape × three-scale record (2026-10-28).

Reads the Session A run directory written by ``slm-transfer-shapes`` and renders
the figures referenced by
``report/shape_trials/multishape_scale/multishape_scale_live_20261028.md``,
including the `Recorder`-driven process animation. Offline: the run directories
are the only source -- no SLM, no camera, and nothing is written under
``data/``.

Usage::

    python scripts/slm/generate_multishape_scale_live_figures.py \\
        --out report/shape_trials/multishape_scale/figures \\
        --run-root data/debug/slm_transfer/multishape_scale_20261028/s1200

All 16 stages that the plan defines are complete on disk: four shapes × four
stages (``transport_cv`` / ``mono12_cv`` / ``mono12_continue`` /
``mono12_dense``), 4 + 8 + 12 + 4 = 28 measured blocks per shape, 112 in total.
``common_replay`` / ``common_replay_dense`` hold in-progress writes from the
next step and are deliberately **not opened**; ``records/`` (CSV + HDF5 +
sidecars, written by the runner) is not a figure source either -- the stage
directories are.  Nothing is inferred: every number in a figure comes from a
``manifest.json`` / ``summary.json`` / ``iterations.pkl`` already on disk.

The Recorder table itself (CSV + HDF5 + ``.npy`` sidecars) is deliberately not
written here: one record carries a 640×640 phase and a decimated frame, so the
sidecars of this 112-block session are ~160 MB.  The animation and the plots are
the deliverable; the runner's ``--records-only`` path owns the table.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from ao_shaping.tools.slm.transfer import recording
from ao_shaping.utils.io.file import Recorder
from ao_shaping.utils.io.record_mode import RecordMode
from ao_shaping.utils.io.recorder_keys import RECORD_IMAGE

STAMP = "_20261028"

STAGES = ("transport_cv", "mono12_cv", "mono12_continue", "mono12_dense")

#: Stage plan of Session A.  All 16 stages finished; a missing stage directory is
#: skipped with a warning instead of being drawn from a half-written manifest,
#: and ``_stage_windows`` refuses a shape whose records do not cover the plan.
STAGE_PLAN = {
    "square": STAGES,
    "triangle": STAGES,
    "star": STAGES,
    "circle": STAGES,
}

#: Drawn as one row each, in run order (the order the stages were executed).
SHAPES = tuple(STAGE_PLAN)

SHAPE_LABELS = {
    "square": "正方形 (square)",
    "triangle": "三角形 (triangle)",
    "star": "五角星 (star)",
    "circle": "圆形 (circle)",
}
SHAPE_COLORS = {
    "square": "#4C72B0",
    "triangle": "#DD8452",
    "star": "#55A868",
    "circle": "#8172B2",
}
STAGE_SHORT = {
    "transport_cv": "第 1 段 transport_cv (8 bit)",
    "mono12_cv": "第 2 段 mono12_cv (12 bit)",
    "mono12_continue": "第 3 段 mono12_continue (12 bit)",
    "mono12_dense": "第 4 段 mono12_dense (12 bit, 宽载波)",
}
STAGE_SAT_SHORT = {
    "transport_cv": "transport",
    "mono12_cv": "mono12_cv",
    "mono12_continue": "mono12_cont",
    "mono12_dense": "mono12_dense",
}
CANDIDATES = (
    ("backprojection", "反投影 backprojection", "#55A868"),
    ("transport", "传输 transport", "#C44E52"),
)

#: 12 bit full scale and the core guard the stages accept on (``250/255``).
FULL_SCALE_12BIT = 4095.0
CORE_SATURATION = FULL_SCALE_12BIT * 250.0 / 255.0  # 4014.7
FULL_SCALE_8BIT = 255.0

DEFAULT_OUT = "report/shape_trials/multishape_scale/figures"
DEFAULT_RUN_ROOT = "data/debug/slm_transfer/multishape_scale_20261028/s1200"


# --------------------------------------------------------------------------- #
# session reading (device-free)
# --------------------------------------------------------------------------- #
def _read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_manifests(run_root: Path) -> dict[tuple[str, str], list[dict]]:
    """``{(shape, stage): manifest rows}`` for the stages that are complete."""
    manifests: dict[tuple[str, str], list[dict]] = {}
    for shape, stages in STAGE_PLAN.items():
        for stage in stages:
            path = Path(run_root) / "shape" / shape / stage / "manifest.json"
            if not path.is_file():
                print(f"[skip] 阶段不存在: {path}", file=sys.stderr)
                continue
            rows = _read_json(path)
            if not rows:
                print(f"[skip] manifest 为空: {path}", file=sys.stderr)
                continue
            manifests[(shape, stage)] = rows
    if not manifests:
        raise SystemExit(f"没有可用的已完阶段目录: {run_root}")
    return manifests


def stage_seed_rows(manifests: dict[tuple[str, str], list[dict]]) -> dict[tuple[str, str], dict]:
    """The stage's own opening re-measurement (``{shape}_seed``).

    Every stage starts by re-measuring the phase it inherited, so the seed is the
    same-phase 12 bit reference of the previous stage's ``*_final`` -- the pair
    §4(c) sizes the re-measurement noise on.  It is *not* a block.
    """
    seeds: dict[tuple[str, str], dict] = {}
    for key, rows in manifests.items():
        wanted = f"{key[0]}_seed"
        for row in rows:
            if str(row.get("label")) == wanted:
                seeds[key] = row
                break
        else:
            raise SystemExit(f"{key} 的 manifest 没有 {wanted} 行 (段首 seed 重测)")
    return seeds


def stage_final_rows(manifests: dict[tuple[str, str], list[dict]]) -> dict[tuple[str, str], dict]:
    """The stage's closing measurement (``*_final``), i.e. the §3.2 table value."""
    finals: dict[tuple[str, str], dict] = {}
    for key, rows in manifests.items():
        last = rows[-1]
        if not str(last.get("label", "")).endswith("_final"):
            raise SystemExit(f"{key} 的 manifest 末条不是 *_final: {last.get('label')}")
        finals[key] = last
    return finals


def saturation_by_stage(
    manifests: dict[tuple[str, str], list[dict]],
) -> dict[tuple[str, str], tuple[int, str]]:
    """``{(shape, stage): (max saturated pixels, label holding it)}``.

    Scoped over **all** manifest rows, as the §3.2 table is.  The maximum often
    sits on a seed / ``seed_replay`` / ``final`` row rather than on a block
    (triangle ``mono12_cv``: block 1516 px vs ``seed_replay`` 1574 px), so the
    per-block figure keeps its own scope and both are reported, never merged.
    """
    table: dict[tuple[str, str], tuple[int, str]] = {}
    for key, rows in manifests.items():
        best = max(rows, key=lambda row: int(row.get("saturation_pixels") or 0))
        table[key] = (int(best.get("saturation_pixels") or 0), str(best.get("label")))
    return table


def peaks_by_capture(manifests: dict[tuple[str, str], list[dict]]) -> dict[tuple[str, str], dict]:
    """``{(shape/stage, label): manifest row}`` -- full-frame peak and saturation."""
    table: dict[tuple[str, str], dict] = {}
    for (shape, stage), rows in manifests.items():
        for row in rows:
            table[(f"{shape}/{stage}", row["label"])] = row
    return table


def build_recorder(run_root: Path, manifests: dict[tuple[str, str], list[dict]]) -> Recorder:
    """One Recorder record per CCD block, stages appended in run order.

    Mirrors ``transfer_shapes_runner._rebuild_records``: the block numbering is
    owned by the recorder, so no two stages collide on an epoch / id.
    """
    recorder = Recorder(mark="cv_candidate", mode=RecordMode.MIN)
    for shape, stages in STAGE_PLAN.items():
        for stage in stages:
            if (shape, stage) not in manifests:
                continue
            added = recording.append_run(recorder, Path(run_root) / "shape" / shape / stage,
                                         f"{shape}/{stage}")
            print(f"[record] {shape}/{stage}: {added} 块")
    if not recorder.history:
        raise SystemExit("Recorder 未收到任何块记录")
    return recorder


# --------------------------------------------------------------------------- #
# derived series
# --------------------------------------------------------------------------- #
def _block_cv(record: dict) -> float:
    """The block's measured CV: candidate B, or A1 where the block has no B.

    ``transport_cv`` accepts on the model loss and measures only A1, so that
    stage's trajectory is the anchor series; the other three measure A1 -> B.
    """
    candidate = record.get("cv_candidate")
    if candidate is not None:
        return float(candidate)
    anchor = record.get("cv_anchor")
    if anchor is None:
        raise SystemExit(f"块 {record.get('label')} 既无 cv_candidate 也无 cv_anchor")
    return float(anchor)


def _records_of(records: list[dict], shape: str) -> list[dict]:
    return [record for record in records if record.get("shape") == shape]


def _stage_windows(records: list[dict]) -> dict[str, list[tuple[str, int, int]]]:
    """``{shape: [(stage, start, stop)]}``, block index within each shape.

    The plan is also a guard -- a stage directory whose manifest holds fewer
    blocks than the plan expects cannot be drawn as if it were measured data.
    """
    table: dict[str, list[tuple[str, int, int]]] = {}
    for shape, stages in STAGE_PLAN.items():
        rows = _records_of(records, shape)
        if not rows:
            continue
        windows: list[tuple[str, int, int]] = []
        start = 0
        for stage in stages:
            count = sum(1 for record in rows if record.get("stage") == f"{shape}/{stage}")
            if not count:
                raise SystemExit(f"{shape}/{stage} 没有块记录, 与 STAGE_PLAN 不符")
            windows.append((stage, start, start + count))
            start += count
        if start != len(rows):
            raise SystemExit(f"{shape} 有 {len(rows)} 块记录, 但 {list(stages)} 只覆盖 {start} 块")
        table[shape] = windows
    if not table:
        raise SystemExit("无法从记录中切出阶段窗口")
    return table


def _counts_per_ms(record: dict, image: np.ndarray) -> np.ndarray:
    """Mono8-equivalent counts per millisecond (same unit as the process GIF).

    The stages meter independently and the bit depth changes, so raw counts are
    neither comparable across blocks nor displayable side by side.
    """
    exposure = float(record.get("exposure_ms") or 0.0)
    if not exposure:
        return np.asarray(image, dtype=np.float64)
    bit_depth = int(record.get("bit_depth") or 8)
    return np.asarray(image, dtype=np.float64) / (exposure * float(2 ** (bit_depth - 8)))


# --------------------------------------------------------------------------- #
# plotting helpers
# --------------------------------------------------------------------------- #
def use_cjk_font() -> None:
    """Same font setup as ``recording.render_process_gif`` (DejaVu has no CJK)."""
    import matplotlib.font_manager as fm

    font_path = "C:/Windows/Fonts/msyh.ttc"
    if Path(font_path).exists():
        fm.fontManager.addfont(font_path)
        family = fm.FontProperties(fname=font_path).get_name()
        plt.rcParams["font.family"] = family
        plt.rcParams["font.sans-serif"] = [family, "SimHei", "SimSun"]
    else:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
    plt.rcParams["axes.unicode_minus"] = False


def _draw_stage_windows(ax, windows: list[tuple[str, int, int]], *, labels: bool) -> None:
    for stage, start, stop in windows:
        if start:
            ax.axvline(start - 0.5, color="#8C8C8C", linestyle="--", linewidth=0.9, zorder=0)
        if labels:
            ax.text((start + stop - 1) / 2.0, 1.015, STAGE_SHORT[stage],
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=8.0, color="#333333")


def _block_ticks(windows: list[tuple[str, int, int]]) -> list[int]:
    ticks = [windows[0][1]]
    ticks += [stop - 1 for _, _, stop in windows]
    return ticks


def _save(fig, path: Path, dpi: float) -> Path:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {path} ({path.stat().st_size} bytes)")
    return path


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig_cv_trajectory(records: list[dict], finals: dict[tuple[str, str], dict],
                      seeds: dict[tuple[str, str], dict], path: Path, *, dpi: float) -> Path:
    """Per-block measured CV, one row per shape, 28 blocks each.

    Each row owns its y range: one shared range would be set by triangle's
    1.93 first block and would flatten the 1–2 pp in-stage wobble that §4(c)
    shows is below the re-measurement noise anyway.  Per stage the row marks the
    opening ``seed`` (same-phase re-measurement) and the closing ``*_final``.
    """
    windows = _stage_windows(records)
    drawn = [shape for shape in SHAPES if _records_of(records, shape)]
    fig, axes = plt.subplots(len(drawn), 1, figsize=(11.2, 2.85 * len(drawn) + 1.1),
                             squeeze=False)
    rows_axes = axes[:, 0]
    handles = [
        Line2D([], [], color="#555555", marker="o", markersize=4.0, linewidth=1.4,
               label="已实测块 (CV)"),
        Line2D([], [], color="#555555", marker="^", markersize=7.0, linestyle="none",
               label="段首 seed (同相位重测)"),
        Line2D([], [], color="#555555", linestyle=":", linewidth=1.7,
               label="段末 *_final (纵线, §3.2 表值)"),
    ]
    for ax, shape in zip(rows_axes, drawn):
        rows = _records_of(records, shape)
        color = SHAPE_COLORS[shape]
        xs = np.arange(len(rows))
        ys = np.asarray([_block_cv(record) for record in rows], dtype=float)
        ax.plot(xs, ys, marker="o", markersize=4.0, linewidth=1.4, color=color, zorder=3)
        marks = [float(ys.min()), float(ys.max())]
        for stage, start, stop in windows[shape]:
            seed = float(seeds[(shape, stage)]["metrics"]["cv"])
            final = float(finals[(shape, stage)]["metrics"]["cv"])
            marks += [seed, final]
            ax.hlines(final, start, stop - 1, color=color, linestyle=":", linewidth=1.7,
                      alpha=0.95, zorder=2)
            # The seed is acquired on the boundary *before* the stage's first
            # block, so the marker sits on the boundary line.  Values stay out of
            # this figure on purpose: with 28 blocks per row a number next to
            # every marker lands on a block; §3.2 and 图 3-3 carry the values.
            ax.plot([start - 0.75], [seed], marker="^", markersize=7.0, color=color,
                    markeredgecolor="white", markeredgewidth=0.6, linestyle="none", zorder=4,
                    clip_on=False)
        lo, hi = min(marks), max(marks)
        margin = max(0.012, 0.06 * (hi - lo))
        ax.set_ylim(lo - margin, hi + margin)
        ax.set_xlim(-1.6, len(rows) + 0.6)
        ax.set_xticks(_block_ticks(windows[shape]))
        ax.grid(alpha=0.3)
        _draw_stage_windows(ax, windows[shape], labels=True)
        ax.set_ylabel(f"{SHAPE_LABELS[shape]}\n实测核心 CV", fontsize=8.5)
    rows_axes[0].legend(handles=handles, fontsize=8, loc="upper right", frameon=True,
                        framealpha=0.92)
    rows_axes[-1].set_xlabel("块序号 (同一形状内, 每块 = 一次 A1→B 采集)")
    fig.suptitle(f"逐块实测核心 CV (s1200, 2026-10-28; {len(drawn)} 形状 × 4 段, "
                 f"每形状 28 块; 每行纵轴按本形状数据取, 不共用)", fontsize=11)
    fig.subplots_adjust(hspace=0.55, top=0.94)
    return _save(fig, path, dpi)


def fig_stage_final_cv(finals: dict[tuple[str, str], dict],
                       seeds: dict[tuple[str, str], dict],
                       saturations: dict[tuple[str, str], tuple[int, str]],
                       path: Path, *, dpi: float) -> Path:
    """4 stage groups × 4 shape bars: the closing CV, with the stage seed beside it.

    Bar = ``*_final``; the short black line = the stage's opening ``seed`` (same
    phase, re-measured), joined to its bar by a dashed stem.  Four of the 16
    seeds (the 8 bit ``transport_cv`` openings: 1.0408 / 1.9206 / 0.9464 /
    1.2453) sit far above the 12 bit finals, and letting them set the y range
    crushed every bar into 10 % of the height.  They are therefore marked at the
    top edge with ``▲`` instead of being drawn to scale, and the note says so.

    Stages whose manifest holds any saturated pixel carry the count in red --
    triangle's 8 bit stage (3905 px) and square's (277 px) included, otherwise a
    clipped reading would pass as a plain CV.
    """
    stages = [stage for stage in STAGES if any((shape, stage) in finals for shape in SHAPES)]
    drawn = [shape for shape in SHAPES if any((shape, stage) in finals for stage in stages)]
    values = [float(finals[key]["metrics"]["cv"]) for key in finals]
    low, high = min(values), max(values)
    bottom = low - 0.09          # free strip under every bar, for the note
    ceiling = high + 0.06        # room for the 饱和 text above the tallest bar
    fig, ax = plt.subplots(figsize=(11.8, 5.4))
    width = 0.80 / len(drawn)
    for offset, shape in enumerate(drawn):
        color = SHAPE_COLORS[shape]
        for index, stage in enumerate(stages):
            key = (shape, stage)
            if key not in finals:
                continue
            x = index + (offset - (len(drawn) - 1) / 2.0) * width
            final = float(finals[key]["metrics"]["cv"])
            seed = float(seeds[key]["metrics"]["cv"])
            ax.bar(x, final, width=width * 0.9, color=color,
                   label=SHAPE_LABELS[shape] if index == 0 else None)
            # The value lives *inside* the bar: with 16 bars whose seeds sit a
            # few thousandths above their finals, any label above the bar lands
            # on the next cap or stem.
            ax.annotate(f"{final:.4f}", (x, final), textcoords="offset points",
                        xytext=(0, -3), ha="center", va="top", fontsize=7.2,
                        color="white")
            if seed <= ceiling - 0.02:
                ax.plot([x - width * 0.36, x + width * 0.36], [seed, seed], color="#222222",
                        linewidth=1.9, solid_capstyle="round", zorder=4)
                ax.plot([x, x], [final, seed], color="#222222", linewidth=0.8,
                        linestyle=(0, (1, 1.4)), zorder=3)
                sat_base = max(final, seed)
            else:
                # The 8 bit openings (1.04 … 1.92) are 3–5× the 12 bit finals; to
                # scale they would flatten every bar into 10 % of the height.
                ax.text(x, bottom + 0.008, f"seed\n{seed:.4f}↑", ha="center", va="bottom",
                        fontsize=6.4, color="white", linespacing=1.35)
                sat_base = final
            saturated, _where = saturations[key]
            if saturated > 0:
                ax.annotate(f"饱和 {saturated}", (x, sat_base), textcoords="offset points",
                            xytext=(0, 5), ha="center", va="bottom", fontsize=6.8,
                            color="#C44E52")
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels([STAGE_SHORT[stage].replace(" (", "\n(") for stage in stages], fontsize=8.5)
    ax.set_ylabel("核心 CV")
    ax.set_ylim(bottom, ceiling)
    ax.set_title(f"各阶段末块 CV (柱, *_final) 与段首 seed 重测 · s1200, 2026-10-28 · "
                 f"{len(drawn)} 形状 × {len(stages)} 段", fontsize=10.5, pad=34)
    ax.grid(axis="y", alpha=0.3)
    handles, _labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="#222222", linewidth=1.9, label="段首 seed (同相位重测)"))
    handles.append(Line2D([], [], color="#222222", marker="^", linestyle="none",
                          markersize=6, label="seed 高于纵轴上限 (标在柱底)"))
    ax.legend(handles=handles, fontsize=8.0, loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=len(handles), frameon=False)
    # Bars are clipped at ``bottom``, so the note cannot live under them: it goes
    # in the strip between the axes frame and the title.
    note = ("柱 = 段末 *_final CV (柱内白字); 黑色短横 + 细虚线 = 该段段首 seed (同一相位重测); "
            "柱底「seed N↑」= 该 seed 高于纵轴上限 (按数值读, 不按比例)\n"
            "红字「饱和 N」= 该阶段 manifest 全部行 (含 seed / seed_replay / final) 的最大饱和像素数; "
            f"纵轴起点 {bottom:.2f} (非 0)")
    if len(stages) < len(STAGES):
        note += f"; 仅 {len(stages)}/{len(STAGES)} 段有数据"
    ax.text(0.0, 1.005, note, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=6.6, color="#666666", linespacing=1.6)
    return _save(fig, path, dpi)


def fig_exposure_peak_saturation(records: list[dict],
                                 peaks: dict[tuple[str, str], dict],
                                 saturations: dict[tuple[str, str], tuple[int, str]],
                                 path: Path, *, dpi: float) -> Path:
    """Exposure / full-frame peak / saturated-pixel audit over all measured blocks.

    All four shapes share the same 28-block stage layout, so one block axis holds
    them.  The saturated-pixel panel annotates every stage that lost pixels to the
    ceiling; where the manifest-wide maximum sits on a seed / final row instead of
    a block, that row is named rather than silently promoted to a block value.
    """
    windows = _stage_windows(records)
    shapes = [shape for shape in SHAPES if _records_of(records, shape)]
    if not shapes:
        raise SystemExit("审计图没有可用形状")
    reference = windows[shapes[0]]
    for shape in shapes[1:]:
        if windows[shape] != reference:
            raise SystemExit(f"{shape} 与 {shapes[0]} 的阶段块数不同, 无法共用一条块轴")
    fig, axes = plt.subplots(3, 1, figsize=(12.4, 10.4), sharex=True)
    fig.subplots_adjust(right=0.74)
    ax_exp, ax_peak, ax_sat = axes
    for shape in shapes:
        rows = _records_of(records, shape)
        if not rows:
            continue
        color = SHAPE_COLORS[shape]
        xs = np.arange(len(rows))
        exposure = [float(record.get("exposure_ms") or np.nan) for record in rows]
        peak = [float(peaks[(record["stage"], record["label"])]["peak"]) for record in rows]
        saturation = [float(record.get("saturation_pixels") or 0) for record in rows]
        bits = np.array([int(record.get("bit_depth") or 8) for record in rows])
        is_12 = bits == 12
        for ax, series in ((ax_exp, exposure), (ax_peak, peak), (ax_sat, saturation)):
            ax.plot(xs, series, color=color, linewidth=1.2, alpha=0.75,
                    label=SHAPE_LABELS[shape])
        for ax, series, marker in ((ax_peak, peak, "o"), (ax_sat, saturation, "o")):
            values = np.asarray(series, dtype=float)
            ax.plot(xs[is_12], values[is_12], marker, color=color, markersize=4.0,
                    linestyle="none")
            ax.plot(xs[~is_12], values[~is_12], "s", color=color, markersize=5.0,
                    linestyle="none", markerfacecolor="none")

    ax_exp.set_ylabel("曝光 exposure_ms")
    ax_peak.set_ylabel("全幅峰值 peak (counts)")
    ax_peak.set_ylim(0.0, 4400.0)
    ax_peak.axhline(FULL_SCALE_12BIT, color="#C44E52", linestyle="-", linewidth=1.0)
    ax_peak.axhline(CORE_SATURATION, color="#C44E52", linestyle=":", linewidth=1.2)
    ax_peak.axhline(FULL_SCALE_8BIT, color="#55A868", linestyle="--", linewidth=1.0)
    # Left-edge labels with an opaque box: peak samples sit on the 12 bit guard,
    # so plain text would be unreadable wherever the series passes behind it.
    label_box = {"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none",
                 "alpha": 0.88}
    ax_peak.text(0.005, FULL_SCALE_12BIT - 90, "12 bit 满量程 4095",
                 transform=ax_peak.get_yaxis_transform(), fontsize=7.5, color="#C44E52",
                 bbox=label_box)
    ax_peak.text(0.005, CORE_SATURATION - 300, "核心饱和阈值 4095×250/255 ≈ 4015",
                 transform=ax_peak.get_yaxis_transform(), fontsize=7.5, color="#C44E52",
                 bbox=label_box)
    ax_peak.text(0.005, FULL_SCALE_8BIT + 60, "8 bit 满量程 255",
                 transform=ax_peak.get_yaxis_transform(), fontsize=7.5, color="#55A868",
                 bbox=label_box)
    ax_sat.set_ylabel("饱和像素数 saturation_pixels")
    ax_sat.set_yscale("symlog", linthresh=1.0)
    ax_sat.set_ylim(0.0, None)  # counts, never negative: no meaningless -10^n ticks
    ax_sat.set_xlabel("块序号 (同一形状内)")
    ax_sat.set_xticks(_block_ticks(reference))

    def block_max(shape: str, stage: str) -> tuple[int, int]:
        """(block index, saturation_pixels) of the worst block in one stage."""
        cells = [(index, record) for index, record in enumerate(_records_of(records, shape))
                 if record.get("stage") == f"{shape}/{stage}"]
        index, worst = max(cells, key=lambda item: int(item[1].get("saturation_pixels") or 0))
        return index, int(worst.get("saturation_pixels") or 0)

    # Three short callouts only.  Seven full labels ("triangle mono12_continue
    # 44 px (triangle_seed 57)") are 250-330 px wide, and at linthresh=1 the
    # late-stage counts (15 … 684) share one decade of the symlog axis, so they
    # necessarily land on each other.  The numbers §4 quotes stay on the panel;
    # every stage maximum is listed in the table to the right of the panels.
    for shape, stage, offset, align in (
            ("triangle", "transport_cv", (0, -13), ("center", "top")),
            ("square", "transport_cv", (0, 9), ("center", "bottom")),
            ("triangle", "mono12_cv", (11, 8), ("left", "bottom"))):
        if (shape, stage) not in saturations:
            continue
        index, value = block_max(shape, stage)
        saturated, _where = saturations[(shape, stage)]
        text = f"{saturated} px" if saturated <= value else f"{value}\n段 {saturated}"
        ax_sat.annotate(text, (index, max(value, 1.0)), textcoords="offset points",
                        xytext=offset, ha=align[0], va=align[1], fontsize=7.0,
                        color=SHAPE_COLORS[shape], linespacing=1.4,
                        bbox={"boxstyle": "round,pad=0.15", "facecolor": "white",
                              "edgecolor": "none", "alpha": 0.85})

    rows = []
    for shape in shapes:
        for stage, _start, _stop in reference:
            if (shape, stage) not in saturations:
                continue
            saturated, _where = saturations[(shape, stage)]
            if saturated <= 0:
                continue
            rows.append((saturated, f"{shape} {stage}: 段 {saturated} px / 块 "
                                    f"{block_max(shape, stage)[1]} px"))
    rows.sort(reverse=True)
    empty = sum(1 for shape in shapes for stage, _a, _b in reference
                if saturations.get((shape, stage), (0, ""))[0] <= 0)
    table = ["各阶段最大饱和像素数", "(manifest 全部行, 含 seed / seed_replay / final)", ""]
    table += [line for _value, line in rows]
    if empty:
        table += ["", f"其余 {empty} 个形状×阶段组合: 0 px"]
    fig.text(0.752, ax_sat.get_position().y1 + 0.005, "\n".join(table), ha="left", va="top",
             fontsize=7.0, color="#333333", linespacing=1.6)

    for ax in axes:
        ax.grid(alpha=0.3)
        for _, start, _stop in reference:
            if start:
                ax.axvline(start - 0.5, color="#8C8C8C", linestyle="--", linewidth=0.9, zorder=0)
    _draw_stage_windows(ax_exp, reference, labels=True)
    handles = [Line2D([], [], color=SHAPE_COLORS[shape], linewidth=1.6,
                      label=SHAPE_LABELS[shape]) for shape in shapes]
    handles += [Line2D([], [], color="#555555", marker="o", linestyle="none", label="12 bit"),
                Line2D([], [], color="#555555", marker="s", linestyle="none",
                       markerfacecolor="none", label="8 bit")]
    ax_peak.legend(handles=handles, fontsize=8, loc="lower right", ncol=2)
    ax_exp.set_title(f"曝光 / 全幅峰值 / 饱和像素逐块审计 (s1200, 2026-10-28; "
                     f"{' + '.join(shapes)} 全部 {sum(len(_records_of(records, shape)) for shape in shapes)} 块; "
                     f"symlog 下饱和 0 显示为 0)", fontsize=11, pad=24)
    return _save(fig, path, dpi)


def fig_seed_bakeoff(run_root: Path, path: Path, *, dpi: float) -> Path:
    """square seed bake-off: backprojection vs transport, CV with the per-frame band."""
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    positions = [0, 1]
    plotted: list[float] = []
    for key, label, color in CANDIDATES:
        folder = Path(run_root) / "seed_bakeoff" / f"square_{key}"
        manifest = folder / "manifest.json"
        if not manifest.is_file():
            print(f"[skip] 无 bake-off 目录: {folder}", file=sys.stderr)
            continue
        rows = _read_json(manifest)
        means = [float(row["metrics"]["cv"]) for row in rows]
        lows = [min(float(frame["cv"]) for frame in row.get("per_frame", [])) for row in rows]
        highs = [max(float(frame["cv"]) for frame in row.get("per_frame", [])) for row in rows]
        plotted += means + lows + highs
        ax.fill_between(positions, lows, highs, color=color, alpha=0.18, linewidth=0)
        ax.plot(positions, means, marker="o", markersize=6, linewidth=1.6, color=color, label=label)
        for index, (x_value, mean) in enumerate(zip(positions, means)):
            text = f"末次 {mean:.3f}" if index == len(means) - 1 else f"{mean:.3f}"
            ax.annotate(text, (x_value, mean), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize=9, color=color)
    ax.set_xticks(positions)
    ax.set_xticklabels(["seed (初相位采集)", "final (对比后采集)"])
    ax.set_xlim(-0.35, 1.65)
    if plotted:  # leave room so the point labels are not clipped by the frame
        span = max(plotted) - min(plotted)
        pad = max(span * 0.10, 0.004)
        ax.set_ylim(min(plotted) - pad, max(plotted) + pad * 1.6)
    ax.set_ylabel("实测核心 CV (8 bit)")
    ax.set_title("square 初相位对比择优: 反投影 1.022 < 传输 1.231 (色带 = 同次采集 3 帧的 min–max)",
                 fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc="center left")
    return _save(fig, path, dpi)


def fig_ccd_first_last(records: list[dict], path: Path, *, dpi: float) -> Path:
    """First vs last measured frame per shape (the only two far fields compared)."""
    shapes = [shape for shape in SHAPES if _records_of(records, shape)]
    fig, axes = plt.subplots(len(shapes), 2, figsize=(8.6, 3.3 * len(shapes)), squeeze=False)
    for row_index, shape in enumerate(shapes):
        rows = _records_of(records, shape)
        picked = [rows[0], rows[-1]]
        images = [_counts_per_ms(record, record[RECORD_IMAGE]) if RECORD_IMAGE in record else None
                  for record in picked]
        available = [image for image in images if image is not None]
        vmax = float(np.percentile(np.maximum(available[0], available[1]), 99.5)) if len(available) == 2 else 1.0
        vmax = vmax if vmax > 0 else 1.0
        for column, (record, image) in enumerate(zip(picked, images)):
            ax = axes[row_index][column]
            if image is not None:
                ax.imshow(image, cmap="inferno", vmin=0.0, vmax=vmax)
            stage = str(record.get("stage", "")).split("/")[-1]
            ax.set_title(f"{SHAPE_LABELS[shape]} · {'首块' if column == 0 else '末块'} "
                         f"{stage} 块 {record.get('block')}\n"
                         f"CV {_block_cv(record):.4f} · {float(record.get('exposure_ms') or 0):.1f} ms · "
                         f"{int(record.get('bit_depth') or 8)} bit", fontsize=8.5)
            ax.axis("off")
    fig.suptitle("首块 vs 末块实测远场 (Mono8 等效 counts/ms, 每形状共用色标 99.5 分位)", fontsize=10.5)
    fig.tight_layout()
    return _save(fig, path, dpi)


def _process_title(records: list[dict], stride: int) -> str:
    """Title carrying the true stage coverage *and* the animation's sampling."""
    stages = {shape: STAGE_PLAN[shape] for shape in SHAPES}
    if all(len(count) == len(STAGES) for count in stages.values()):
        coverage = f"{'/'.join(SHAPES)} 各 {len(STAGES)} 段, 全部已实测"
    else:
        coverage = ", ".join(
            f"{shape} {len(count)} 段" if len(count) == len(STAGES) else f"{shape} 仅 {count[0]}"
            for shape, count in stages.items())
    suffix = f" · 每 {stride} 块取 1" if stride > 1 else ""
    return (f"多形状物理预测模型闭环整形 · s1200 2026-10-28 · "
            f"{coverage} ({len(records)} 块){suffix}")


def render_process(recorder: Recorder, path: Path, *, fps: float, max_mb: float,
                   max_frames: int) -> Path:
    """The Recorder process animation, strided down if it would be too heavy."""
    records = list(recorder.history)
    stride = max(1, math.ceil(len(records) / max_frames)) if max_frames else 1
    while True:
        frames = records[::stride]
        recording.render_process_gif(frames, path, fps=fps, title=_process_title(records, stride))
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb <= max_mb or stride >= len(records):
            print(f"[gif] {path} ({path.stat().st_size} bytes, {len(frames)} 帧, {size_mb:.1f} MB)")
            return path
        print(f"[gif] {size_mb:.1f} MB 超过 {max_mb:.0f} MB, 加倍步长重渲染", file=sys.stderr)
        stride *= 2


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT,
                        help="Session A 运行根目录 (只读; 默认 %(default)s)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="图输出目录 (默认 %(default)s)")
    parser.add_argument("--dpi", type=float, default=150.0, help="PNG 分辨率 (默认 %(default)s)")
    parser.add_argument("--fps", type=float, default=2.0, help="GIF 帧率 (默认 %(default)s)")
    parser.add_argument("--gif-max-mb", type=float, default=16.0,
                        help="GIF 体积上限; 超过则按步长抽帧重渲染 (默认 %(default)s)")
    parser.add_argument("--gif-max-frames", type=int, default=56,
                        help="GIF 最多帧数 (默认 %(default)s: 112 块按 2 块取 1 = 56 帧)")
    parser.add_argument("--only", default=None,
                        help="只渲染某一张 (文件名前缀, 如 cv_trajectory)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_root = Path(args.run_root)
    if not (run_root / "shape").is_dir():
        raise SystemExit(f"运行根目录不含 shape/: {run_root}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifests = load_manifests(run_root)
    seeds = stage_seed_rows(manifests)
    finals = stage_final_rows(manifests)
    saturations = saturation_by_stage(manifests)
    peaks = peaks_by_capture(manifests)
    use_cjk_font()

    recorder = build_recorder(run_root, manifests)
    records = list(recorder.history)
    print(f"[record] 共 {len(records)} 块 ({', '.join(SHAPES)}), 阶段 {len(manifests)} 个")

    produced: list[Path] = []
    wanted = args.only

    def want(name: str) -> bool:
        return not wanted or name.startswith(wanted)

    if want("process_s1200"):
        produced.append(render_process(recorder, out / f"process_s1200{STAMP}.gif",
                                       fps=args.fps, max_mb=args.gif_max_mb,
                                       max_frames=args.gif_max_frames))
    if want("cv_trajectory"):
        produced.append(fig_cv_trajectory(records, finals, seeds,
                                          out / f"cv_trajectory_s1200{STAMP}.png", dpi=args.dpi))
    if want("stage_final_cv"):
        produced.append(fig_stage_final_cv(finals, seeds, saturations,
                                           out / f"stage_final_cv_s1200{STAMP}.png", dpi=args.dpi))
    if want("exposure_peak_saturation"):
        produced.append(fig_exposure_peak_saturation(
            records, peaks, saturations,
            out / f"exposure_peak_saturation_s1200{STAMP}.png", dpi=args.dpi))
    if want("seed_bakeoff"):
        produced.append(fig_seed_bakeoff(run_root, out / f"seed_bakeoff_square{STAMP}.png",
                                         dpi=args.dpi))
    if want("ccd_first_last"):
        produced.append(fig_ccd_first_last(records, out / f"ccd_first_last_s1200{STAMP}.png",
                                           dpi=args.dpi))

    print("\n".join(f"{path}\t{path.stat().st_size} B" for path in produced))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())