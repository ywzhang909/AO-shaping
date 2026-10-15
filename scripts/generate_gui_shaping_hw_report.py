"""Offline figures + summary for the GUI closed-loop shaping report (2026-10-14).

Reads the saved run artefacts under ``data/debug/slm_gs_refine/<ts>/`` (CSV
history, replay ``*.npy`` frames and ``measurements.json``) and renders every
figure and the summary table used by
``report/shape_trials/gui_closed_loop_shaping_20261014.md``.

Pure numpy + matplotlib: no hardware, no SDK, instruments may be powered off.

    python scripts/generate_gui_shaping_hw_report.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data" / "debug" / "slm_gs_refine"
OUT = ROOT / "report" / "shape_trials" / "images" / "gui_shaping_20261014"

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

#: The three GUI-driven runs of 2026-10-14 (see the report §2 for context).
#: Keys are the run directory names under ``data/debug/slm_gs_refine/``.
SESSIONS = {
    "20261014_100441": "A 10:04 首次 GUI 闭环",
    "20261014_104105": "B 10:41 重构后复跑",
    "20261014_143035": "C 14:30 远端合并后复跑",
}
EXPOSURE_LADDER_MS = (0.02, 0.05, 0.10, 0.20, 0.40)
#: Flat-field peak measured by the two read-only preflights (14:29, same day).
PREFLIGHT_PEAKS = {0.02: 59.3, 0.05: 146.7, 0.10: 234.0, 0.20: 254.7, 0.40: 254.7}
PREFLIGHT_PEAKS_LATE = {0.02: 50.0, 0.05: 125.7, 0.10: 211.3, 0.20: 254.7}


@dataclass(frozen=True)
class Run:
    ts: str
    label: str
    history: list[dict]
    replay: list[dict]
    best: np.ndarray | None
    flat: np.ndarray | None


def _load_csv(path: Path) -> list[dict]:
    import csv

    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_run(ts: str, label: str) -> Run | None:
    directory = RUNS / ts
    csvs = sorted(directory.glob("*.csv"))
    if not csvs:
        return None
    history = _load_csv(csvs[0])
    replay_path = directory / "replay" / "measurements.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8")) if replay_path.exists() else []
    best = flat = None
    replay_dir = directory / "replay"
    if (replay_dir / "02_best.npy").exists():
        best = np.load(replay_dir / "02_best.npy")
        flat = np.load(replay_dir / "00_flat.npy")
    return Run(ts=ts, label=label, history=history, replay=replay, best=best, flat=flat)


def _metric(run: Run, stage: str, field: str) -> float | None:
    for row in run.history:
        if row["stage"] == stage and row.get(field):
            return float(row[field])
    return None


def _best_of(run: Run, field: str) -> float | None:
    values = [float(r[field]) for r in run.history if r.get(field)]
    return max(values) if values else None


def _best_row(run: Run) -> dict[str, str]:
    """The record-setting row — metrics must come from *this* row.

    Taking a per-column max over the history instead would mix stages (the flat
    row has the largest CV, the GS/best rows the largest PIB), which silently
    reports a combination no measured frame ever produced.
    """
    return max(run.history, key=lambda r: float(r["score"]))


def _best_metric(run: Run, field: str) -> float | None:
    row = _best_row(run)
    return float(row[field]) if row.get(field) else None


def fig_exposure_ladder() -> Path:
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    xs = np.arange(len(EXPOSURE_LADDER_MS))
    early = [PREFLIGHT_PEAKS[e] for e in EXPOSURE_LADDER_MS]
    late = [PREFLIGHT_PEAKS_LATE.get(e, np.nan) for e in EXPOSURE_LADDER_MS[:4]]
    ax.plot(xs, early, "o-", label="10:03 预检")
    ax.plot(xs[:4], late[:4], "s--", label="14:29 预检")
    ax.axhline(250, color="crimson", ls=":", lw=1.2)
    ax.text(len(xs) - 0.4, 252, "饱和保护阈值 250", color="crimson", fontsize=8, ha="right")
    ax.axhline(125, color="grey", ls=":", lw=1)
    ax.text(0.05, 128, "选定工作点 0.05 ms", color="grey", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{e:g}" for e in EXPOSURE_LADDER_MS])
    ax.set_xlabel("曝光 (ms)")
    ax.set_ylabel("平场全帧峰值 (count/255)")
    ax.set_ylim(0, 285)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = OUT / "fig1_exposure_ladder.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_convergence(runs: list[Run]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for run in runs:
        refine = [r for r in run.history if r["stage"] == "refine"]
        if not refine:
            continue
        epochs = [int(float(r["epoch"])) for r in refine]
        scores = [float(r["score"]) for r in refine]
        axes[0].plot(epochs, scores, marker="o", ms=3, lw=1, label=run.label)
    axes[0].set_xlabel("SPGD 轮次")
    axes[0].set_ylabel("综合分")
    axes[0].set_title("闭环内 SPGD 轨迹（不含 flat/GS 阶段）")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.25)

    labels, flat_v, gs_v, best_v = [], [], [], []
    for run in runs:
        if not run.replay:
            continue
        by_phase: dict[str, list[float]] = {}
        for row in run.replay:
            by_phase.setdefault(str(row["phase"]), []).append(float(row["score"]))
        if not {"flat", "gs", "best"} <= set(by_phase):
            continue
        labels.append(run.label.split()[0])
        flat_v.append(float(np.mean(by_phase["flat"])))
        gs_v.append(float(np.mean(by_phase["gs"])))
        best_v.append(float(np.mean(by_phase["best"])))
    x = np.arange(len(labels))
    axes[1].bar(x - 0.25, flat_v, 0.25, label="flat")
    axes[1].bar(x, gs_v, 0.25, label="GS")
    axes[1].bar(x + 0.25, best_v, 0.25, label="best")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0.45, 0.82)
    axes[1].set_ylabel("综合分（交错重放均值）")
    axes[1].set_title("交错重放：因果口径")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25, axis="y")
    fig.tight_layout()
    path = OUT / "fig2_convergence_and_replay.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_far_field(runs: list[Run]) -> Path:
    rows = [r for r in runs if r.flat is not None and r.best is not None]
    if not rows:
        return OUT / "fig3_far_field.png"
    fig, axes = plt.subplots(len(rows), 3, figsize=(9.0, 3.1 * len(rows)))
    axes = np.atleast_2d(axes)
    for row, run in enumerate(rows):
        for col, (frame, name) in enumerate(
            ((run.flat, "flat"), (None, "GS"), (run.best, "best"))
        ):
            ax = axes[row, col]
            if frame is None:
                ax.axis("off")
                ax.text(0.5, 0.5, "GS 相位由重放脚本按记录回放\n（见 replay/measurements.json）",
                        ha="center", va="center", fontsize=7, transform=ax.transAxes)
                continue
            im = ax.imshow(frame, cmap="inferno")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{run.label.split()[0]} {name}  peak={frame.max():.0f}", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path = OUT / "fig3_far_field.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_box_profile(runs: list[Run]) -> Path:
    rows = [r for r in runs if r.best is not None and r.flat is not None]
    if not rows:
        return OUT / "fig4_box_profile.png"
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    run = rows[-1]
    centre = run.best.shape[0] // 2
    for frame, name, style in ((run.flat, "flat", "--"), (run.best, "best", "-")):
        axes[0].plot(frame[centre], style, lw=1, label=name)
        axes[1].plot(frame[:, centre], style, lw=1, label=name)
    axes[0].set_title("过零级中心行的行剖面")
    axes[1].set_title("过零级中心列的列剖面")
    for ax in axes:
        ax.set_xlabel("像素")
        ax.set_ylabel("强度 (count)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle(f"末次会话（{run.label}）中心剖面：能量由紧凑峰铺向方形footprint", fontsize=9)
    fig.tight_layout()
    path = OUT / "fig4_box_profile.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def summarise(runs: list[Run]) -> dict:
    table = []
    for run in runs:
        flat = _metric(run, "flat", "score")
        gs = _metric(run, "gs", "score")
        best = _best_of(run, "score")
        by_phase: dict[str, list[dict]] = {}
        for row in run.replay:
            by_phase.setdefault(str(row["phase"]), []).append(row)
        table.append(
            {
                "ts": run.ts,
                "label": run.label,
                "in_run": {
                    "flat": flat,
                    "gs": gs,
                    "best": best,
                    "cv_best": _best_metric(run, "cv"),
                    "pib_best": _best_metric(run, "pib"),
                    "best_stage": _best_row(run)["stage"],
                    "best_epoch": _best_row(run).get("epoch", ""),
                    "gs_adopted": bool(flat is not None and gs is not None and gs > flat),
                },
                "replay": {
                    phase: {
                        "score_mean": float(np.mean([float(r["score"]) for r in rows])),
                        "score_spread": float(
                            np.ptp([float(r["score"]) for r in rows])
                        ),
                        "cv_mean": float(np.mean([float(r["cv"]) for r in rows])),
                        "pib_mean": float(np.mean([float(r["pib"]) for r in rows])),
                        "n": len(rows),
                    }
                    for phase, rows in by_phase.items()
                },
            }
        )
    summary = {
        # A list of records, not a float-keyed object: JSON object keys are
        # strings, so ``{"0.05": ...}`` would silently change type on the first
        # round-trip and make the artefact awkward to consume.
        "exposure_ladder": {
            "saturation_guard": 250,
            "working_point_ms": 0.05,
            "rows": [
                {
                    "exposure_ms": exposure,
                    "peak_1003": PREFLIGHT_PEAKS[exposure],
                    "peak_1429": PREFLIGHT_PEAKS_LATE.get(exposure),
                }
                for exposure in EXPOSURE_LADDER_MS
            ],
        },
        "runs": table,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = [r for r in (_load_run(ts, label) for ts, label in SESSIONS.items()) if r]
    if not runs:
        print("no run directories found", file=sys.stderr)
        return 1
    made = [fig_exposure_ladder(), fig_convergence(runs), fig_far_field(runs), fig_box_profile(runs)]
    summary = summarise(runs)
    for path in made:
        if path.exists():
            print(f"wrote {path.relative_to(ROOT)}")
    print(f"wrote {(OUT / 'summary.json').relative_to(ROOT)}")
    for row in summary["runs"]:
        replay = row["replay"]
        head = "  ".join(
            f"{phase}={replay[phase]['score_mean']:.4f}" for phase in ("flat", "gs", "best") if phase in replay
        )
        print(f"  {row['ts']}: {head}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())