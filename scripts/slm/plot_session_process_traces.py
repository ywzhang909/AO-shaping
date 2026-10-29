"""Plot one shaping session's per-stage optimization traces from Recorder CSVs.

Reads ``<session-dir>/shape/<shape>/<stage>/recording/process.csv`` for every
stage that produced a recording, writes a per-stage summary JSON **before**
touching matplotlib (so a plotting failure cannot cost the numbers), then renders
a two-panel figure: (a) ``cv`` and (b) ``exposure_ms`` against the cumulative
recorded frame index, with stage boundaries drawn and each stage's first/last
``cv`` labelled.

Why two panels: ``core_peak`` is not comparable across stages (8-bit vs 12-bit
with ``counts_scale = 16``), but ``cv`` (dimensionless) and ``exposure_ms``
(physical) are. Anything else would repeat the ``core_peak / exposure`` mistake
documented in ``report/shape_trials/multishape_scale/live_20261029_device_session.md``
§9.5.

Usage::

    python scripts/slm/plot_session_process_traces.py \\
        --session-dir data/debug/slm_transfer/multishape_nogs_20261029/p20261029_strict_cold_s200/star \\
        --out report/shape_trials/multishape_scale/figures/star_stage_traces_20261029.png \\
        --json report/shape_trials/multishape_scale/artifacts/star_stage_traces_20261029.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from generate_multishape_scale_live_figures import use_cjk_font

# Canonical stage order of the four-stage shaping recipe (tools/slm/transfer/stages.py).
STAGE_ORDER = ("transport_cv", "mono12_cv", "mono12_continue", "mono12_dense")

USECOLS = (
    "cv",
    "core_peak",
    "model_loss",
    "label",
    "stage",
    "exposure_ms",
    "bit_depth",
    "saturation_pixels",
)


class StageTrace(NamedTuple):
    shape: str
    stage: str
    frame: pd.DataFrame
    csv_path: pathlib.Path


def _stage_key(shape: str, stage: str) -> tuple[str, int, str]:
    try:
        order = STAGE_ORDER.index(stage)
    except ValueError:
        order = len(STAGE_ORDER)
    return (shape, order, stage)


def discover_stages(session_dir: pathlib.Path) -> list[StageTrace]:
    """Return every stage under ``<session-dir>/shape/<shape>/<stage>`` with a recording."""
    traces: list[StageTrace] = []
    for shape_dir in sorted(p for p in (session_dir / "shape").iterdir() if p.is_dir()):
        for stage_dir in sorted(p for p in shape_dir.iterdir() if p.is_dir()):
            csv_path = stage_dir / "recording" / "process.csv"
            if not csv_path.exists():
                continue
            frame = pd.read_csv(csv_path, usecols=lambda c: c in USECOLS)
            if frame.empty:
                continue
            traces.append(StageTrace(shape_dir.name, stage_dir.name, frame, csv_path))
    traces.sort(key=lambda t: _stage_key(t.shape, t.stage))
    return traces


def summarise(traces: list[StageTrace]) -> dict:
    """Per-stage numbers, computed before anything is drawn."""
    stages = []
    for trace in traces:
        frame = trace.frame
        cv = frame["cv"].astype(float)
        exposure = frame["exposure_ms"].astype(float)
        stages.append(
            {
                "shape": trace.shape,
                "stage": trace.stage,
                "csv": str(trace.csv_path),
                "frames": int(len(frame)),
                "cv_first": float(cv.iloc[0]),
                "cv_min": float(cv.min()),
                "cv_last": float(cv.iloc[-1]),
                "cv_delta": float(cv.iloc[-1] - cv.iloc[0]),
                "exposure_first_ms": float(exposure.iloc[0]),
                "exposure_last_ms": float(exposure.iloc[-1]),
                "first_label": str(frame["label"].iloc[0]),
                "last_label": str(frame["label"].iloc[-1]),
                "bit_depth": sorted({int(v) for v in frame["bit_depth"].dropna().unique()}),
                "saturated_frames": int((frame["saturation_pixels"].fillna(0) > 0).sum()),
                "model_loss_first": float(frame["model_loss"].astype(float).iloc[0]),
                "model_loss_last": float(frame["model_loss"].astype(float).iloc[-1]),
            }
        )
    return {
        # process.csv -> recording / <stage> / <shape> / shape / <session>
        "session_dir": str(traces[0].csv_path.parents[4]) if traces else None,
        "shapes": sorted({t.shape for t in traces}),
        "total_frames": sum(s["frames"] for s in stages),
        "stages": stages,
    }


def plot(traces: list[StageTrace], summary: dict, out_path: pathlib.Path) -> None:
    use_cjk_font()
    offsets = []
    cursor = 0
    for trace in traces:
        offsets.append(cursor)
        cursor += len(trace.frame)

    fig, axes = plt.subplots(2, 1, figsize=(11.4, 6.4), sharex=True)

    for (trace, offset, entry) in zip(traces, offsets, summary["stages"]):
        x = range(offset, offset + len(trace.frame))
        axes[0].plot(x, trace.frame["cv"].astype(float), lw=0.9,
                     label=f"{trace.stage}（cv {entry['cv_first']:.3f} → {entry['cv_last']:.3f}）")
        axes[1].plot(x, trace.frame["exposure_ms"].astype(float), lw=0.9, label=trace.stage)

    boundaries = [offsets[0]] + [o for o in offsets[1:]] + [cursor]
    for ax in axes:
        for boundary in boundaries:
            ax.axvline(boundary, color="0.6", lw=0.7, ls=":")
        ax.grid(alpha=0.25, lw=0.5)

    # Stage names ride the top edge of the cv panel only — a second set over the
    # exposure panel collided with its legend on the first render.
    for offset, trace in zip(offsets, traces):
        axes[0].annotate(trace.stage, xy=(offset, 1.0), xycoords=("data", "axes fraction"),
                         xytext=(4, -11), textcoords="offset points", fontsize=8, color="0.3")

    axes[0].set_ylabel("cv（冻结核心）")
    axes[0].set_title(
        f"{summary['shapes'][0]} 盲跑四段逐帧记录：{len(traces)} 段 / "
        f"{summary['total_frames']} 帧（段界为虚线；每段由模型步重新播种，段内不必单调）"
    )
    axes[1].set_ylabel("曝光 (ms)")
    axes[1].set_xlabel("累计记录帧序（四段首尾相接）")
    headroom = 1.18 * max(e["exposure_last_ms"] for e in summary["stages"])
    axes[1].set_ylim(0, headroom)
    axes[1].set_yticks([0, 5, 10, 15, 20, 25, 30])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.09, 1, 1))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", required=True, type=pathlib.Path,
                        help="session directory holding shape/<shape>/<stage>/recording/process.csv")
    parser.add_argument("--out", required=True, type=pathlib.Path, help="figure PNG path")
    parser.add_argument("--json", type=pathlib.Path, default=None,
                        help="per-stage summary JSON (written before plotting)")
    args = parser.parse_args(argv)

    traces = discover_stages(args.session_dir)
    if not traces:
        print(f"no process.csv under {args.session_dir}", flush=True)
        return 2

    summary = summarise(traces)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[json] {args.json}", flush=True)

    for entry in summary["stages"]:
        print(
            f"{entry['stage']:<18} frames={entry['frames']:<4} "
            f"cv {entry['cv_first']:.3f} -> {entry['cv_last']:.3f} "
            f"({entry['cv_delta']:+.3f})  expo {entry['exposure_first_ms']:.1f} -> "
            f"{entry['exposure_last_ms']:.1f} ms  sat_frames={entry['saturated_frames']}",
            flush=True,
        )
    print(f"total frames: {summary['total_frames']}", flush=True)

    plot(traces, summary, args.out)
    print(f"[figure] {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
