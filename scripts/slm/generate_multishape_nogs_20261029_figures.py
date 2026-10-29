"""Figures for the 2026-10-29 no-GS multishape session (200 px analytic init).

Reads the campaign directory written by ``slm-transfer-shapes`` and renders the
figures referenced by
``report/shape_trials/multishape_scale/live_20261029_device_session.md``.
Offline: the run directories are the only source -- no SLM, no camera.

Every number comes from a file already on disk:

* ``shape/<shape>/<stage>/capture_records.pkl`` -- one row per CCD capture with
  its measured CV (the trajectory and dumbbell figures, and the stage table in
  the live log, all quote this one source; ``summary.json`` holds the per-block
  *decision* metric, which is a different cut and is not used here);
* ``<shape>/common_replay/`` -- the archived replay frames plus the frozen
  aperture. The audit figures call
  :func:`scripts.slm.audit_replay_blocks.audit_block` /
  ``summarise_block`` **themselves** rather than re-deriving CV, so a figure can
  never disagree with the table in the live log; and the frames are read once by
  that same code path (background subtraction included).

Usage::

    python scripts/slm/generate_multishape_nogs_20261029_figures.py \\
        --out report/shape_trials/multishape_scale/figures

The GIF lands in ``<out>/../images/`` -- animations are illustrations, not
plotted measurements (``report/AGENTS.md`` §4).

Nothing is written under ``data/``. ``common_replay_aborted_50_125_225ms`` (the
aborted first replay attempt) is deliberately not opened: its rows are not a
completed condition, and drawing them next to the released block would invite
exactly the comparison the audit refuses.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "slm"))


# The audit is the single implementation of "what does this replay block say".
# Importing it (instead of re-reading the frames here) is what keeps the figures
# and the printed table identical.
from audit_replay_blocks import audit_block, summarise_block

# Style helpers and vocabulary shared with the 2026-10-28 session figures.
from generate_multishape_scale_live_figures import (
    _save, use_cjk_font, SHAPE_COLORS, SHAPE_LABELS,
)

STAMP = "_s200_20261029"
DEFAULT_CAMPAIGN = "data/debug/slm_transfer/multishape_nogs_20261029"
DEFAULT_SESSION = "p20261029_analytic_init_s200"
DEFAULT_OUT = "report/shape_trials/multishape_scale/figures"

#: Run order of the campaign (the order the audit table in the live log uses).
SHAPES = ("circle", "square", "triangle", "star")

STAGES = ("transport_cv", "mono12_cv", "mono12_continue", "mono12_dense")


VERDICT_STYLE = {
    # reason -> (colour, marker, human label)
    "cv_improved_beyond_noise": ("#2E7D32", "o", "supported（改善超噪声）"),
    "wrong_direction": ("#C62828", "X", "wrong_direction（反向变差）"),
    "no_difference": ("#8C8C8C", "s", "no_difference（差异在噪声内）"),
    "insufficient_repeats": ("#B8860B", "^", "insufficient（重复不足）"),
    "missing_cv": ("#8C8C8C", "v", "missing_cv（无 CV）"),
}


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _audit_payload(session: Path, shapes) -> dict[str, dict]:
    """Per-shape audit result plus the exposure bookkeeping the figures show."""
    payload = {}
    for shape in shapes:
        directory = session / shape / "common_replay"
        if not directory.is_dir():
            continue
        block = audit_block(directory)
        if not block.get("present"):
            continue
        summary = summarise_block(block)
        rows = block["rows"]
        groups = []
        for key, group in summary["groups"].items():
            exposure = float(str(key).split("ms")[0])
            pairs = group.get("pairs") or []
            replicates = group.get("replicates") or {}
            # ``noise`` is the replicate spread of the unchanged (seed) phase:
            # with one seed row there is none, so the comparison is reported as
            # insufficient rather than borrowing the phase spread.
            # ``judge_pair`` returns the half width on some branches only (a
            # ``no_difference`` verdict carries it, a ``wrong_direction`` one does
            # not), so fall back to the block's own replicate spread -- that is the
            # same number the verdict was tested against.
            replicate_half = [float(stat["half_width"]) for stat in replicates.values()
                              if stat.get("half_width") is not None]
            noise = pairs[0]["verdict"].get("noise_half_width") if pairs else None
            if noise is None and replicate_half:
                noise = max(replicate_half)
            groups.append(dict(
                exposure=exposure, key=key, n=group["n"],
                cv_mean=group["cv_mean"], ee_mean=group["ee_rel_mean"],
                zero_excluded=group["zero_excluded_fraction"],
                noise=noise,
                pairs=pairs,
            ))
        groups.sort(key=lambda item: item["exposure"])
        # Exposures that were replayed but produced no comparable group (every row
        # clipped, or no seed row): the figure has to show that they were dropped,
        # not silently omit them.
        used = {round(item["exposure"], 3) for item in groups}
        dropped = {}
        for row in rows:
            exposure = row.get("exposure_ms")
            if exposure is None:
                continue
            exposure = round(float(exposure), 3)
            if exposure in used:
                continue
            stat = dropped.setdefault(exposure, dict(total=0, clipped=0, errors=0))
            if row.get("error"):
                stat["errors"] += 1
            else:
                stat["total"] += 1
                stat["clipped"] += int(bool(row.get("clipped")))
        clipped_by_exposure = {}
        for row in rows:
            if row.get("error") or row.get("exposure_ms") is None:
                continue
            exposure = round(float(row["exposure_ms"]), 3)
            clip = clipped_by_exposure.setdefault(exposure, dict(total=0, clipped=0))
            clip["total"] += 1
            clip["clipped"] += int(bool(row.get("clipped")))
        payload[shape] = dict(
            block=block, summary=summary, groups=groups, dropped=dropped,
            clipped=clipped_by_exposure, aperture=block.get("aperture"),
            rows=rows,
            n_rows=summary["n_rows"], n_usable=summary["n_usable"],
            n_clipped=len(summary["clipped"]),
        )
    return payload


def _captures(session: Path) -> dict[tuple[str, str], list[dict]]:
    """``(shape, stage) -> capture records`` -- every CCD capture, not every block.

    This is the source the §5.2 stage table quotes (first / last capture CV, and
    its 行数 column), so a figure drawn from it cannot disagree with that table.
    A ``summary.json`` row instead reports the per-block *decision* metric, which
    is a different cut of the same run.
    """
    out: dict[tuple[str, str], list[dict]] = {}
    for shape_dir in sorted(p for p in session.iterdir() if p.is_dir()):
        shape = shape_dir.name
        base = shape_dir / "shape" / shape
        if not base.is_dir():
            continue
        for stage_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            path = stage_dir / "capture_records.pkl"
            if not path.is_file():
                continue
            with path.open("rb") as handle:
                records = pickle.load(handle)
            out[(shape, stage_dir.name)] = records
    return out


def _trajectory(captures: dict[tuple[str, str], list[dict]], shape: str) -> dict:
    """Cumulative capture axis plus the observed CV of one shape.

    ``transport_cv`` accepted on the model's own loss, so its CVs are
    observations rather than a decision metric -- drawn dashed and labelled, not
    hidden, because the 8-bit stage is exactly where the trajectory is flat.
    """
    x: list[int] = []
    cv: list[float] = []
    windows: list[tuple[str, int, int, tuple[int, ...]]] = []
    cursor = 0
    for stage in STAGES:
        records = captures.get((shape, stage))
        if not records:
            continue
        start = cursor
        for record in records:
            value = (record.get("metrics") or {}).get("cv")
            if value is None:
                continue
            x.append(cursor)
            cv.append(float(value))
            cursor += 1
        bits = tuple(sorted({int(record.get("bit_depth") or 0) for record in records}))
        windows.append((stage, start, cursor, bits))
    return dict(x=x, cv=cv, windows=windows)


def fig_cv_trajectory(traj: dict[str, dict], path: Path, *, dpi: float) -> Path:
    """Observed core CV per capture -- the line the §5.2 table is a summary of."""
    shapes = [shape for shape in SHAPES if shape in traj]
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 7.8), squeeze=False)
    for ax, shape in zip(axes.ravel(), shapes):
        data = traj[shape]
        x = np.asarray(data["x"])
        cv = np.asarray(data["cv"])
        if not x.size:
            ax.axis("off")
            continue
        colour = SHAPE_COLORS[shape]
        for index, (stage, start, stop, bits) in enumerate(data["windows"]):
            if index % 2:
                ax.axvspan(start - .5, stop - .5, color="#F2F2F2", zorder=0)
            if start:
                ax.axvline(start - .5, color="#8C8C8C", ls="--", lw=.8, zorder=1)
            model_only = stage == "transport_cv"
            ax.text((start + stop - 1) / 2, .97,
                    f"{stage}\n{'8bit' if 8 in bits else '12bit'}",
                    transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=7.2, color="#333333",
                    bbox=dict(boxstyle="round,pad=.15", fc="white", ec="none", alpha=.75))
            mask = (x >= start) & (x < stop)
            ax.plot(x[mask], cv[mask], color=colour if not model_only else "#7A7A7A",
                    ls="--" if model_only else "-", lw=1.2 if model_only else 1.6,
                    marker="o", markersize=2.6, zorder=3)
            ax.text(start + .6, .045, f"{cv[mask][0]:.3f}\n→{cv[mask][-1]:.3f}",
                    transform=ax.get_xaxis_transform(), fontsize=6.6, color="#333333",
                    va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=.15", fc="white", ec="none", alpha=.75))
        ax.set_title(SHAPE_LABELS[shape], fontsize=10, pad=8, loc="left")
        ax.set_ylabel("实测核心 CV（越低越均匀）", fontsize=8.5)
        ax.set_xlabel("采集序号（四段连续计数）", fontsize=8.5)
        ax.grid(alpha=.25, lw=.6)
        ax.tick_params(labelsize=8)
    for ax in axes.ravel()[len(shapes):]:
        ax.axis("off")
    handles = [
        plt.Line2D([], [], color="#444444", lw=1.6, marker="o", markersize=3,
                   label="mono12 段（12bit，CV 即接受判据）"),
        plt.Line2D([], [], color="#7A7A7A", lw=1.2, ls="--", marker="o", markersize=3,
                   label="transport_cv（8bit，接受判据为模型损失，CV 仅观测）"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8.5, frameon=False)
    fig.suptitle("200 px 解析初始化：逐采集核心 CV（与 §5.2 阶段表同源，均取自 capture_records.pkl）",
                 fontsize=11, y=.985)
    fig.tight_layout(rect=(0, .05, 1, .965))
    return _save(fig, path, dpi)


def fig_stage_dumbbell(traj: dict[str, dict], path: Path, *, dpi: float) -> Path:
    """每段首/末采集 CV：即 §5.2 表两列的可视化，一段内部的净变化一眼可比。"""
    shapes = [shape for shape in SHAPES if shape in traj]
    fig, ax = plt.subplots(figsize=(11.4, 4.8))
    width = .19
    for offset, shape in enumerate(shapes):
        data = traj[shape]
        colour = SHAPE_COLORS[shape]
        x = np.asarray(data["x"])
        cv = np.asarray(data["cv"])
        for index, (stage, start, stop, bits) in enumerate(data["windows"]):
            mask = (x >= start) & (x < stop)
            if not mask.any():
                continue
            first, last = float(cv[mask][0]), float(cv[mask][-1])
            xs = offset + (index - 1.5) * width
            model_only = stage == "transport_cv"
            ax.annotate("", xy=(xs, last), xytext=(xs, first),
                        arrowprops=dict(arrowstyle="-|>",
                                        color=colour if not model_only else "#7A7A7A",
                                        lw=1.4, ls="--" if model_only else "-",
                                        shrinkA=1.5, shrinkB=1.5, alpha=.95))
            ax.scatter([xs], [first], s=30, facecolor="white", edgecolor=colour,
                       linewidth=1.2, zorder=3)
            ax.scatter([xs], [last], s=34, color=colour, zorder=3)
            ax.text(xs, max(first, last) + .022, f"{last - first:+.3f}", ha="center",
                    va="bottom", fontsize=6.8,
                    color=colour if not model_only else "#7A7A7A")
    ax.set_xticks(range(len(shapes)))
    ax.set_xticklabels([SHAPE_LABELS[shape] for shape in shapes], fontsize=9)
    ax.set_ylabel("核心 CV（空心 = 段首采集，实心 = 段末采集）", fontsize=8.5)
    ax.set_xlabel("每一形状内从左到右 = ① transport_cv ② mono12_cv ③ mono12_continue ④ mono12_dense",
                  fontsize=8.5)
    ax.grid(axis="y", alpha=.25, lw=.6)
    ax.tick_params(labelsize=8)
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + .07 * (high - low))
    fig.suptitle("200 px 解析初始化：逐段首末核心 CV（与 §5.2 表同源；灰色虚线 = transport_cv，接受判据为模型损失）",
                 fontsize=10.5, y=.98)
    fig.tight_layout(rect=(.02, 0, 1, .93))
    return _save(fig, path, dpi)


def fig_audit_dcv(payload: dict[str, dict], path: Path, *, dpi: float) -> Path:
    """The §5.1 evidence: dCV per (shape, exposure) with the replicate noise band.

    A point outside its own whisker is a difference bigger than the unchanged
    phase's own spread. Two situations are shown rather than dropped: a group
    with no comparable pair is labelled ``无对照`` on its tick, and an exposure
    whose every row was clipped is shaded with a struck-out tick -- plotting
    either one at zero would read as "no change", which is not what happened.
    """
    shapes = [shape for shape in SHAPES if shape in payload]
    fig, axes = plt.subplots(2, len(shapes), figsize=(3.55 * len(shapes), 7.2),
                             squeeze=False)
    for col, shape in enumerate(shapes):
        data = payload[shape]
        exposures = sorted(set(data["clipped"]) | {group["exposure"] for group in data["groups"]})
        position = {exposure: index for index, exposure in enumerate(exposures)}
        status: dict[int, str] = {}
        cv_values: list[float] = [0.0]
        ee_values: list[float] = [0.0]

        ax = axes[0][col]
        for group in data["groups"]:
            index = position[group["exposure"]]
            pairs = group["pairs"]
            if not pairs:
                status[index] = f"n={group['n']} 无对照"
                continue
            verdict = pairs[0]["verdict"]
            reason = verdict.get("reason") or "insufficient_repeats"
            colour, marker, _ = VERDICT_STYLE.get(reason, VERDICT_STYLE["insufficient_repeats"])
            delta = verdict.get("delta_cv")
            if delta is None:
                status[index] = f"n={group['n']} 无 CV"
                continue
            noise = verdict.get("noise_half_width")
            if noise is None:
                noise = group.get("noise")
            status[index] = f"n={group['n']} 可用"
            if noise:
                ax.errorbar(index, delta, yerr=noise, fmt="none", ecolor=colour,
                            elinewidth=1.3, capsize=3, alpha=.85, zorder=2)
                cv_values += [delta + noise, delta - noise]
            else:
                cv_values.append(delta)
            ax.scatter(index, delta, s=46, color=colour, marker=marker, zorder=3)
            ax.annotate(f"{delta:+.3f}", (index, delta), textcoords="offset points",
                        xytext=(9, 0), ha="left", va="center", fontsize=7, color=colour)
        for exposure, stat in sorted(data["dropped"].items()):
            index = position[exposure]
            status[index] = f"整组截断 {stat['clipped']}/{stat['total']}"
            ax.axvspan(index - .42, index + .42, color="#EFEFEF", zorder=0)
            ax.scatter(index, 0, s=34, color="#AAAAAA", marker="x", linewidths=1.5, zorder=3)
        ax.axhline(0, color="#333333", lw=.9, zorder=1)
        _finish_audit_axis(ax, position, status, cv_values)
        ax.set_title(f"{SHAPE_LABELS[shape]}\n有效 {data['n_usable']}/{data['n_rows']} 行",
                     fontsize=9)
        if col == 0:
            ax.set_ylabel("dCV = candidate − baseline（越低越好）", fontsize=8.5)

        ax = axes[1][col]
        for group in data["groups"]:
            index = position[group["exposure"]]
            if not group["pairs"]:
                continue
            verdict = group["pairs"][0]["verdict"]
            reason = verdict.get("reason") or "insufficient_repeats"
            colour, marker, _ = VERDICT_STYLE.get(reason, VERDICT_STYLE["insufficient_repeats"])
            delta = verdict.get("delta_ee_rel")
            if delta is None:
                continue
            ee_values.append(delta)
            ax.scatter(index, delta, s=42, color=colour, marker=marker, zorder=3)
            ax.annotate(f"{delta:+.3f}", (index, delta), textcoords="offset points",
                        xytext=(9, 0), ha="left", va="center", fontsize=7, color=colour)
        for exposure in data["dropped"]:
            index = position[exposure]
            ax.axvspan(index - .42, index + .42, color="#EFEFEF", zorder=0)
            ax.scatter(index, 0, s=30, color="#AAAAAA", marker="x", linewidths=1.4, zorder=3)
        ax.axhline(0, color="#333333", lw=.9, zorder=1)
        _finish_audit_axis(ax, position, status, ee_values, label_status=False)
        if col == 0:
            ax.set_ylabel("d ee_rel = candidate − baseline（能量占比变化）", fontsize=8.5)

    handles = [plt.Line2D([], [], color=colour, marker=marker, ls="none", markersize=7,
                          label=label)
               for reason, (colour, marker, label) in VERDICT_STYLE.items()]
    handles.append(plt.Line2D([], [], color="#AAAAAA", marker="x", ls="none", markersize=7,
                              label="整组截断（无可用行）"))
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.suptitle("冻结光阑重放的逐档配对（误差棒 = 同相位 4 次重复的半宽；点越出误差棒才算差异）",
                 fontsize=10.5, y=.985)
    fig.tight_layout(rect=(0, .045, 1, .95))
    return _save(fig, path, dpi)


def _finish_audit_axis(ax, position: dict[float, int], status: dict[int, str],
                       values: list[float], *, label_status: bool = True) -> None:
    """Ticks carry each exposure's own outcome, so panels can hold different exposures."""
    low, high = min(values), max(values)
    span = max(high - low, 1e-3)
    ax.set_ylim(low - .28 * span, high + .22 * span)
    ax.set_xticks(list(position.values()))
    ticks = []
    for exposure, index in position.items():
        line = f"{exposure:g} ms"
        ticks.append(f"{line}\n{status[index]}" if label_status and index in status else line)
    ax.set_xticklabels(ticks, fontsize=7)
    ax.set_xlim(-.6, len(position) - .4)
    ax.grid(axis="y", alpha=.25, lw=.6)
    ax.tick_params(axis="y", labelsize=7.5)


def _mean_cv(rows: list[dict]) -> float | None:
    """Mean CV of one phase's rows (``None`` when nothing measured)."""
    values = [float(row["cv"]) for row in rows if row.get("cv") is not None]
    return float(np.mean(values)) if values else None


def _fmt_cv(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _cropped(frame: np.ndarray, aperture: dict, *, pad: float = 1.35) -> np.ndarray:
    cx, cy = aperture["centre_xy"]
    half = aperture["half_side"] * pad
    x0, x1 = int(round(cx - half)), int(round(cx + half))
    y0, y1 = int(round(cy - half)), int(round(cy + half))
    x0, y0 = max(x0, 0), max(y0, 0)
    return np.asarray(frame)[y0:y1, x0:x1]


def _mean_frame(directory: Path, rows: list[dict], aperture: dict) -> np.ndarray | None:
    frames = []
    for row in rows:
        path = directory / str(row["file"])
        if not path.is_file():
            continue
        with np.load(path) as payload:
            frames.append(payload["raw_frames"].mean(0))
    if not frames:
        return None
    return _cropped(np.mean(frames, axis=0), aperture)


def _overlay_aperture(ax, aperture: dict, crop: np.ndarray) -> None:
    """Draw the box / zero-order disk in *cropped* pixel coordinates."""
    cx, cy = aperture["centre_xy"]
    half = aperture["half_side"]
    pad = 1.35
    x0 = cx - half * pad
    y0 = cy - half * pad
    height, width = crop.shape[:2]
    box = Rectangle(((cx - half) - x0, (cy - half) - y0), 2 * half, 2 * half,
                    fill=False, edgecolor="#00E5FF", lw=1.0, alpha=.9)
    ax.add_patch(box)
    if aperture.get("exclude_zero") and aperture.get("zero_xy"):
        zx, zy = aperture["zero_xy"]
        radius = float(aperture.get("zero_radius") or 0)
        if radius > 0:
            import matplotlib.patches as mpatches

            circle = mpatches.Circle((zx - x0, zy - y0), radius, fill=False,
                                     edgecolor="#FFD54F", lw=.9, alpha=.9, ls="--")
            ax.add_patch(circle)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)


def fig_frozen_aperture(payload: dict[str, dict], session: Path, path: Path, *,
                        dpi: float) -> Path:
    """首档与末档的判据帧：ABBA 到底比较了哪两张图，肉眼可见。

    The panels show the **judged frame** of each phase (the first row of the
    phase, which is exactly what ``judge_pair`` compares), so the CV printed in
    the title and the ``dCV`` badge are the audit's own numbers -- not a mean
    that would quietly differ from the table.
    """
    shapes = [shape for shape in SHAPES if shape in payload]
    fig, axes = plt.subplots(len(shapes), 4, figsize=(12.6, 2.9 * len(shapes)),
                             squeeze=False)
    for row, shape in enumerate(shapes):
        data = payload[shape]
        directory = session / shape / "common_replay"
        aperture = data["aperture"]
        groups = [group for group in data["groups"] if group["pairs"]]
        if not groups or aperture is None:
            for ax in axes[row]:
                ax.axis("off")
            continue
        picks = [groups[0], groups[-1]] if len(groups) > 1 else [groups[0]]
        for col, group in enumerate(picks):
            members = [item for item in data["rows"]
                       if round(float(item["exposure_ms"]), 3) == round(group["exposure"], 3)
                       and not item.get("error")]
            seeds = [item for item in members if item.get("phase") == "seed"]
            finals = [item for item in members if item.get("phase") == "final"]
            verdict = group["pairs"][0]["verdict"]
            delta = verdict.get("delta_cv")
            for offset, (rows, label) in enumerate((
                    (seeds[:1], f"{group['exposure']:g} ms · seed（A，判据帧）"),
                    (finals[:1], f"{group['exposure']:g} ms · final（B，判据帧）"))):
                ax = axes[row][col * 2 + offset]
                ax.set_xticks([])
                ax.set_yticks([])
                frame = _mean_frame(directory, rows, aperture)
                if frame is None:
                    ax.axis("off")
                    continue
                image = np.clip(frame - np.median(frame), 0, None) ** 0.7
                vmax = np.percentile(image, 99.7) or image.max() or 1.0
                ax.imshow(image, cmap="inferno", vmax=vmax, interpolation="nearest")
                _overlay_aperture(ax, aperture, image)
                cv = rows[0].get("cv") if rows else None
                ax.set_title(f"{label}\nCV={_fmt_cv(cv)}", fontsize=7.5)
            axes[row][col * 2].set_ylabel(SHAPE_LABELS[shape].split(" ")[0], fontsize=9)
            axes[row][col * 2].annotate(
                f"dCV={delta:+.3f}" if delta is not None else "dCV=—",
                xy=(.02, .04), xycoords="axes fraction", fontsize=7.5, color="white",
                bbox=dict(boxstyle="round,pad=.2", fc="#00000088", ec="none"))
        if len(picks) == 1:
            for ax in axes[row][2:]:
                ax.axis("off")
            dropped = ", ".join(f"{exposure:g} ms"
                                for exposure in sorted(data["dropped"]))
            axes[row][2].text(.5, .5, f"其余档位整组截断，无可比较帧\n{dropped}",
                              transform=axes[row][2].transAxes, ha="center", va="center",
                              fontsize=8, color="#666666")
    fig.suptitle("冻结光阑下的判据帧（seed = A，final = B）：青框 = 判定框，黄虚圆 = 排除的零级",
                 fontsize=10.5, y=.995)
    fig.tight_layout(rect=(0.035, 0, 1, .97))
    return _save(fig, path, dpi)


def fig_saturation(payload: dict[str, dict], path: Path, *, dpi: float) -> Path:
    """每档被截断的行数：为什么那几档退出了分组，一眼可见。

    One row per shape, because each shape was replayed at its **own** exposure
    ladder (23.9-71.7 ms for the circle, 61.2-183.6 ms for the star). A shared
    column axis would put two different exposures under one tick.
    """
    shapes = [shape for shape in SHAPES if shape in payload]
    fig, axes = plt.subplots(len(shapes), 1, figsize=(7.8, 1.5 * len(shapes) + 1.2),
                             sharex=True, squeeze=False)
    for row, shape in enumerate(shapes):
        data = payload[shape]
        ax = axes[row][0]
        exposures = sorted(data["clipped"])
        totals = [data["clipped"][exposure]["total"] for exposure in exposures]
        clipped = [data["clipped"][exposure]["clipped"] for exposure in exposures]
        fraction = [hit / total if total else 0.0 for hit, total in zip(clipped, totals)]
        ys = list(range(len(exposures)))
        ax.barh(ys, fraction, height=.62, color=SHAPE_COLORS[shape], alpha=.85)
        for y, frac, hit, total in zip(ys, fraction, clipped, totals):
            ax.text(min(frac, 1.0) + .015, y, f"{hit}/{total}" + ("" if hit else "（无截断）"),
                    va="center", ha="left", fontsize=7.5, color="#333333")
        ax.set_yticks(ys)
        ax.set_yticklabels([f"{exposure:g} ms" for exposure in exposures], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.22)
        ax.set_ylim(len(exposures) - .4, -.6)
        ax.grid(axis="x", alpha=.25, lw=.6)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.set_title(f"{SHAPE_LABELS[shape]} · 有效 {data['n_usable']}/{data['n_rows']} 行",
                     fontsize=8.5, loc="left")
    axes[-1][0].set_xlabel("被冻结光阑判为截断的行占比（截断行的 CV / ee_rel 不是该相位的测量）",
                           fontsize=8.5)
    fig.suptitle("各档截断行占比：整档全截断的档位已从全部分组剔除", fontsize=10.5, y=.985)
    fig.tight_layout(rect=(0, 0, 1, .95))
    return _save(fig, path, dpi)


def fig_common_replay_gif(payload: dict[str, dict], session: Path, path: Path, *,
                          fps: float, max_mb: float) -> Path:
    """4-panel animation of the archived replay frames, in capture order."""
    import matplotlib.animation as animation

    shapes = [shape for shape in SHAPES if shape in payload]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 8.4), squeeze=False)
    panels = {}
    for ax, shape in zip(axes.ravel(), shapes):
        data = payload[shape]
        directory = session / shape / "common_replay"
        aperture = data["aperture"]
        if aperture is None:
            ax.axis("off")
            continue
        frames = []
        for row in data["rows"]:
            if row.get("error"):
                continue
            path_row = directory / str(row["file"])
            if not path_row.is_file():
                continue
            with np.load(path_row) as payload_row:
                frame = _cropped(payload_row["raw_frames"].mean(0), aperture)
            # Same gamma stretch as the static frame figure: the far-field core
            # holds most of the energy, so a linear map leaves the speckle
            # structure invisible at any scale that does not clip the core.
            image = np.clip(frame - np.median(frame), 0, None) ** 0.6
            frames.append((image, row))
        if not frames:
            ax.axis("off")
            continue
        vmax = max(np.percentile(item[0], 99.7) for item in frames) or 1.0
        panels[shape] = dict(frames=frames, vmax=vmax, ax=ax, aperture=aperture)
        ax.imshow(frames[0][0], cmap="inferno", vmax=vmax, interpolation="nearest")
        _overlay_aperture(ax, aperture, frames[0][0])
        ax.set_xticks([])
        ax.set_yticks([])
    steps = max(len(panel["frames"]) for panel in panels.values()) if panels else 1
    axes_list = list(panels)

    def update(index):
        for shape in axes_list:
            panel = panels[shape]
            frames = panel["frames"]
            image, row = frames[min(index, len(frames) - 1)]
            ax = panel["ax"]
            ax.images[0].set_data(image)
            tag = "截断" if row.get("clipped") else ("可用" if row.get("cv") is not None else "—")
            cv = "—" if row.get("cv") is None else f"{row['cv']:.3f}"
            ax.set_title(f"{SHAPE_LABELS[shape]}\n{row['label']} | CV={cv} | {tag}",
                         fontsize=8)
        fig.suptitle(f"冻结光阑重放帧（第 {index + 1}/{steps} 帧，按采集顺序）", fontsize=10)
        return []

    anim = animation.FuncAnimation(fig, update, frames=steps, interval=1000 / fps,
                                   blit=False)
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    size_mb = path.stat().st_size / 1e6
    if size_mb > max_mb:
        print(f"[figure] WARNING {path.name} is {size_mb:.1f} MB (limit {max_mb:.0f} MB); "
              f"lower --gif-fps or reduce the panel count")
    print(f"[figure] {path} ({path.stat().st_size} bytes)")
    return path


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", type=Path, default=Path(DEFAULT_CAMPAIGN))
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUT))
    parser.add_argument("--dpi", type=float, default=150.0)
    parser.add_argument("--gif-fps", type=float, default=2.0)
    parser.add_argument("--gif-max-mb", type=float, default=8.0)
    parser.add_argument("--only", nargs="*", default=None,
                        help="只渲染这些图（cv / dumbbell / audit / aperture / saturation / gif）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    session = Path(args.campaign) / args.session
    if not session.is_dir():
        print(f"no such session: {session}")
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    use_cjk_font()

    captures = _captures(session)
    if not captures:
        print(f"no capture records under {session}")
        return 2
    traj = {shape: _trajectory(captures, shape) for shape in SHAPES
            if any(key[0] == shape for key in captures)}
    payload = _audit_payload(session, SHAPES)

    only = set(args.only or ())
    def wanted(name: str) -> bool:
        return not only or name in only

    if wanted("cv"):
        fig_cv_trajectory(traj, out / f"cv_trajectory{STAMP}.png", dpi=args.dpi)
    if wanted("dumbbell"):
        fig_stage_dumbbell(traj, out / f"stage_dumbbell{STAMP}.png", dpi=args.dpi)
    if wanted("audit"):
        fig_audit_dcv(payload, out / f"audit_dcv{STAMP}.png", dpi=args.dpi)
    if wanted("aperture"):
        fig_frozen_aperture(payload, session, out / f"frozen_aperture{STAMP}.png",
                            dpi=args.dpi)
    if wanted("saturation"):
        fig_saturation(payload, out / f"saturation_groups{STAMP}.png", dpi=args.dpi)
    if wanted("gif"):
        # Animations live in `images/` (report/AGENTS.md 4): a GIF is an
        # illustration of the run, not a plotted measurement.
        gif_dir = out.parent / "images"
        gif_dir.mkdir(parents=True, exist_ok=True)
        fig_common_replay_gif(payload, session, gif_dir / f"common_replay{STAMP}.gif",
                              fps=args.gif_fps, max_mb=args.gif_max_mb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
