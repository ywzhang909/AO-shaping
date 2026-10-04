"""Offline acceptance report for the Daheng ``slm-pib`` online regression runs.

Reads every ``data/debug/slm_pib_online/<stamp>/`` directory written by the
``AO_RUN_HARDWARE=1`` online suite and produces:

* ``report.md`` — the acceptance write-up: noise floor / SNR verdict table,
  gate-observability table (applied / fold / noise), the J-trajectory
  environment-drift diagnosis, and a per-run section;
* ``figures/snr_by_delta.png`` — SNR per perturbation ``delta`` with the
  ``unusable`` / ``usable`` / ``strong`` verdict bands;
* ``figures/gate_timeline_<tag>.png`` — per-epoch ``applied``/``fold``/``noise``
  gate timeline for each smoke run;
* ``figures/j_trajectory_<tag>.png`` — ``J`` and ``max_brt`` vs epoch, with the
  environmental-drift correlation annotated.

**Fully offline**: it never opens hardware. It only reads the saved recorder
pickles + summary JSONs, so a report can be regenerated at any time (per repo
convention, report generation lives in ``scripts/`` — see ``scripts/README.md``).

Recorder row contract (mirrors
``tests/ao_shaping/optimizer/wfless/test_slm_zernike_pib_online.py``):

* row ``0`` is the initialisation baseline, carries **no** ``_gate`` key, and
  is counted in ``rows`` only — the invariant is
  ``applied + stalled + 1 == rows``;
* the optimizer writes the per-epoch verdict under the private ``_gate`` key
  with values ``"applied"`` / ``"fold"`` / ``"noise"``;
* records predating that key (heuristic-branch rows, older pickles) are
  classified with the ``_c``-stagnation fallback, which over-counts
  ``applied`` because a noise-gated row still advances ``_c`` via
  ``optimizer.update(zeros)``.

Usage:
    python scripts/generate_slm_pib_online_report.py
    python scripts/generate_slm_pib_online_report.py --root data/debug/slm_pib_online
    python scripts/generate_slm_pib_online_report.py -o report/slm_pib_online
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: must be set before importing pyplot

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
# `scripts._common` lives in this package, so the REPO ROOT (not just `src`)
# must be importable. A direct `python scripts/<name>.py` does not put it
# there; pytest does, via `pythonpath = ["src", ".", "scripts"]` in pyproject.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._common import fmt_general

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# CJK-capable font fallbacks (per repo script convention).
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DEFAULT_ROOT = ROOT / "data" / "debug" / "slm_pib_online"

# `scripts._common` lives in this package, so the REPO ROOT (not just
# `src`) must be importable. Direct `python scripts/<name>.py` does not put
# it there; pytest does via `pythonpath = ["src", ".", "scripts"]`.
sys.path.insert(0, str(DEFAULT_ROOT))
DEFAULT_OUT = ROOT / "report" / "slm_pib_online"

GATE_ORDER = ("applied", "fold", "noise")
GATE_COLORS = {"applied": "#2ca02c", "fold": "#d62728", "noise": "#ff7f0e"}

#: SNR verdict bands used by the online suite (see ``summary_snr.json``).
SNR_STRONG = 2.5
SNR_USABLE = 2.0


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Run:
    """One ``data/debug/slm_pib_online/<stamp>/`` directory."""

    stamp: str
    path: Path
    kind: str  # "snr" | "smoke"
    tag: str  # "snr" | "smoke_f1" | "smoke_f4"
    summary: dict[str, Any] = field(default_factory=dict)
    rows: list[dict[str, Any]] = field(default_factory=list)
    #: True when every epoch row carries an explicit ``_gate`` verdict.
    has_explicit_gate: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def epochs(self) -> list[dict[str, Any]]:
        """Epoch rows only — the baseline (row 0) has no gate verdict."""
        return self.rows[1:]

    def gate_stats(self) -> dict[str, Any]:
        """Applied / stalled / fold / noise counts.

        Byte-identical in semantics to ``_classify_rows`` in the online test so
        the report can never disagree with the test's own assertion.
        """
        stats: dict[str, Any] = {
            "rows": len(self.rows),
            "applied": 0,
            "stalled": 0,
            "fold": 0,
            "noise": 0,
        }
        prev_c: np.ndarray | None = None
        for row in self.rows:
            c = np.asarray(row["_c"])
            gate = row.get("_gate")
            if gate is not None:
                stats[gate] = stats.get(gate, 0) + 1
                if gate != "applied":
                    stats["stalled"] += 1
            elif prev_c is not None:
                if np.array_equal(c, prev_c):
                    stats["stalled"] += 1
                    if float(row["_diff"]) == 0.0:
                        stats["fold"] += 1
                    else:
                        stats["noise"] += 1
                else:
                    stats["applied"] += 1
            prev_c = c
        return stats

    def series(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        """``(epochs, values)`` for a scalar row key, baseline excluded."""
        xs: list[float] = []
        ys: list[float] = []
        for row in self.epochs:
            value = row.get(key)
            if value is None:
                continue
            xs.append(float(row.get("_epoch", len(xs) + 1)))
            ys.append(float(value))
        return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def _load_run(run_dir: Path) -> Run:
    """Build a :class:`Run` from one artifact directory."""
    summaries = sorted(run_dir.glob("summary_*.json"))
    if not summaries:
        raise FileNotFoundError(f"no summary_*.json in {run_dir}")
    summary_path = summaries[0]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    tag = summary_path.stem.removeprefix("summary_")

    rows: list[dict[str, Any]] = []
    missing_recorder = False
    recorder_files = sorted(run_dir.glob("recorder_*.pkl"))
    if recorder_files:
        obj = pickle.loads(recorder_files[0].read_bytes())
        # The SNR sweep measures J directly instead of driving a Recorder, so
        # its debug writer legitimately dumps ``None``; the data lives in the
        # summary JSON.
        if obj is None or not hasattr(obj, "history"):
            missing_recorder = True
        else:
            rows = list(obj.history)

    run = Run(
        stamp=run_dir.name,
        path=run_dir,
        kind="snr" if tag == "snr" else "smoke",
        tag=tag,
        summary=summary,
        rows=rows,
    )
    run.has_explicit_gate = bool(rows) and all(
        row.get("_gate") is not None for row in rows[1:]
    )
    if missing_recorder:
        run.warnings.append(
            "`recorder_*.pkl` holds `None` (this sweep does not drive a Recorder); "
            "all reported values come from the summary JSON"
        )
    if rows and not run.has_explicit_gate:
        run.warnings.append(
            "records predate the explicit `_gate` verdict; gate counts fall back "
            "to the `_c`-stagnation heuristic and over-count `applied`"
        )
    return run


def load_runs(root: Path) -> list[Run]:
    """Load every artifact directory under ``root``, oldest first."""
    runs: list[Run] = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            runs.append(_load_run(run_dir))
        except Exception as exc:  # noqa: BLE001 - one bad dir must not kill the report
            print(f"  ! skipping {run_dir.name}: {type(exc).__name__}: {exc}")
    return runs


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def drift_correlation(run: Run) -> tuple[float, float] | None:
    """``(pearson(J, max_brt), slope)`` over the epoch rows, or ``None``.

    The ep15 failure mode is environmental: the beam drifts across the sensor,
    so the tracked objective ``J`` and the frame peak ``max_brt`` move together
    (a strong negative correlation here) even though the Zernike coefficients
    are barely moving.
    """
    _, j = run.series("J")
    _, brt = run.series("max_brt")
    n = min(j.size, brt.size)
    if n < 3:
        return None
    j, brt = j[:n], brt[:n]
    if np.allclose(j, j[0]) or np.allclose(brt, brt[0]):
        return None
    corr = float(np.corrcoef(j, brt)[0, 1])
    slope = float(np.polyfit(brt, j, 1)[0])
    return corr, slope


def build_findings(runs: list[Run]) -> list[str]:
    """Headline conclusions derived from the loaded runs."""
    findings: list[str] = []
    snr_runs = [r for r in runs if r.kind == "snr"]
    smoke_runs = [r for r in runs if r.kind == "smoke"]

    for run in snr_runs:
        snr = run.summary.get("snr_by_delta", {})
        verdict = run.summary.get("verdict", {})
        best_delta = max(snr, key=lambda d: snr[d]) if snr else None
        if best_delta is not None:
            findings.append(
                f"**{run.stamp}** — noise floor sigma_J="
                f"{run.summary.get('sigma_j', float('nan')):.3e}; best SNR "
                f"{snr[best_delta]:.3f} at delta={best_delta} "
                f"({verdict.get(best_delta, '?')})."
            )
        if snr and all(verdict.get(d) == "unusable" for d in snr):
            findings.append(
                f"  - {run.stamp} is an **all-unusable** epoch: the J measurement "
                f"noise floor dominated the gradient signal at every delta."
            )
        elif snr:
            usable = [d for d in snr if verdict.get(d) != "unusable"]
            if usable:
                findings.append(
                    f"  - {run.stamp} reaches a usable gradient at "
                    f"delta={', '.join(sorted(usable))}."
                )

    for run in smoke_runs:
        stats = run.gate_stats()
        rows = stats["rows"]
        if rows <= 1:
            continue
        applied_pct = 100.0 * stats["applied"] / (rows - 1)
        findings.append(
            f"**{run.stamp}** ({run.tag}, delta={run.summary.get('delta')}, "
            f"n_eval_frames={run.summary.get('n_eval_frames')}) — "
            f"{stats['applied']}/{rows - 1} epochs applied the gradient "
            f"({applied_pct:.0f}%), {stats['fold']} fold-gated, "
            f"{stats['noise']} noise-gated."
        )
        drift = drift_correlation(run)
        if drift is not None and abs(drift[0]) > 0.9:
            findings.append(
                f"  - J and max_brt correlate at **{drift[0]:+.4f}** — the "
                f"objective is tracking environmental beam drift, not the "
                f"coefficients."
            )
    return findings


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_snr(runs: list[Run], out: Path) -> Path | None:
    """SNR per delta across SNR runs, with verdict bands."""
    snr_runs = [r for r in runs if r.summary.get("snr_by_delta")]
    if not snr_runs:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / len(snr_runs)
    xs = np.arange(3)
    for i, run in enumerate(snr_runs):
        snr = run.summary["snr_by_delta"]
        deltas = sorted(snr, key=float)
        values = [snr[d] for d in deltas]
        colors = [GATE_COLORS["applied"] if v >= SNR_STRONG else "#1f77b4" for v in values]
        ax.bar(xs + i * width - 0.4 + width / 2, values, width * 0.9,
               label=run.stamp, color=colors)
        for x, v in zip(xs + i * width - 0.4 + width / 2, values):
            ax.text(x, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels(deltas, rotation=20)
    ax.axhline(SNR_USABLE, ls="--", c="gray", lw=1, label=f"usable ({SNR_USABLE})")
    ax.axhline(SNR_STRONG, ls=":", c="k", lw=1, label=f"strong ({SNR_STRONG})")
    ax.set_ylabel("SNR (signal / sigma_J)")
    ax.set_xlabel("perturbation delta")
    ax.set_title("slm-pib online: gradient SNR per delta")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "snr_by_delta.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_gate_timeline(run: Run, out: Path) -> Path | None:
    """Per-epoch gate verdict timeline."""
    epochs = run.epochs
    if not epochs:
        return None
    gates = [str(r.get("_gate", "unknown")) for r in epochs]
    xs = np.arange(len(gates))
    fig, ax = plt.subplots(figsize=(9, 2.6))
    for i, g in enumerate(gates):
        ax.bar(i, 1, color=GATE_COLORS.get(g, "gray"), edgecolor="none")
    ax.set_xlim(-0.5, len(gates) - 0.5)
    ax.set_yticks([])
    ax.set_xlabel("epoch")
    ax.set_title(f"{run.stamp} ({run.tag}) gate timeline  "
                 f"[{', '.join(f'{g}={gates.count(g)}' for g in GATE_ORDER if g in gates)}]")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=GATE_COLORS[g]) for g in GATE_ORDER
    ]
    ax.legend(handles, list(GATE_ORDER), loc="upper right", fontsize=8, ncol=3)
    fig.tight_layout()
    path = out / f"gate_timeline_{run.stamp}_{run.tag}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_j_trajectory(run: Run, out: Path) -> Path | None:
    """J and max_brt vs epoch, annotated with the drift correlation."""
    xj, j = run.series("J")
    _, brt = run.series("max_brt")
    if xj.size < 2:
        return None
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(xj, j, "-o", ms=3, label="J (objective)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("J", color="tab:blue")
    ax2 = ax.twinx()
    n = min(j.size, brt.size)
    ax2.plot(xj[:n], brt[:n], "-s", ms=3, c="tab:orange", label="max_brt")
    ax2.set_ylabel("max_brt", color="tab:orange")
    drift = drift_correlation(run)
    title = f"{run.stamp} ({run.tag}) J trajectory"
    if drift is not None:
        title += f"  — corr(J, max_brt)={drift[0]:+.4f}"
    ax.set_title(title)
    fig.tight_layout()
    path = out / f"j_trajectory_{run.stamp}_{run.tag}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report(runs: list[Run], out: Path, figures: list[Path]) -> Path:
    """Render ``report.md``."""
    snr_runs = [r for r in runs if r.kind == "snr"]
    smoke_runs = [r for r in runs if r.kind == "smoke"]
    findings = build_findings(runs)
    lines: list[str] = [
        "# SLM-PIB Online Acceptance Report (Daheng CCD)",
        "",
        "Offline report generated from `data/debug/slm_pib_online/`. No hardware",
        "was opened; every number below is read from the saved recorder pickles",
        "and summary JSONs written by the `AO_RUN_HARDWARE=1` online suite.",
        "",
        f"Runs analysed: **{len(runs)}** "
        f"({len(snr_runs)} SNR sweep, {len(smoke_runs)} smoke).",
        "",
        "## 1. Findings",
        "",
    ]
    lines += [f"- {f}" for f in findings] or ["- (no findings)"]
    lines += [
        "",
        "## 2. Noise floor / SNR verdict",
        "",
        "`sigma_J` is the standard deviation of the tracked objective over 15",
        "repeat frames at a fixed (flat) phase — the pure measurement noise floor.",
        "SNR is the mean |J| response to a `+delta`/`-delta` pair divided by it.",
        "",
        "| run | center | sigma_J | " + " | ".join(
            f"SNR@delta={d}" for d in _union_deltas(snr_runs)
        ) + " |",
        "|---|---|---|" + "---|" * len(_union_deltas(snr_runs)),
    ]
    for run in snr_runs:
        snr = run.summary.get("snr_by_delta", {})
        verdict = run.summary.get("verdict", {})
        cells = []
        for d in _union_deltas(snr_runs):
            if d in snr:
                cells.append(f"{fmt_general(snr[d], '.3f')} ({verdict.get(d, '?')})")
            else:
                cells.append("-")
        center = run.summary.get("center", ["?", "?"])
        lines.append(
            f"| {run.stamp} | {center[0]:.0f},{center[1]:.0f} | "
            f"{fmt_general(run.summary.get('sigma_j'), '.4e')} | " + " | ".join(cells) + " |"
        )
    lines += [
        "",
        f"Verdict bands: `unusable` < {SNR_USABLE} <= `usable` < {SNR_STRONG} "
        f"<= `strong`.",
        "",
    ]
    if figures:
        rel = figures[0]
        lines += [f"![SNR by delta]({rel.parent.name}/{rel.name})", ""]

    lines += [
        "## 3. Gate observability",
        "",
        "The optimizer records a per-epoch verdict under the private `_gate` key:",
        "`applied` (gradient adopted), `fold` (in-ROI energy collapse — evaluation",
        "abandoned), `noise` (gradient below the noise gate — coefficients frozen).",
        "",
        "The row-0 baseline carries no verdict and is counted in `rows` only, so the",
        "invariant is `applied + stalled + 1 == rows`.",
        "",
        "| run | tag | delta | n_eval_frames | rows | applied | fold | noise | gate source |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for run in smoke_runs:
        s = run.gate_stats()
        source = "explicit `_gate`" if run.has_explicit_gate else "legacy heuristic"
        lines.append(
            f"| {run.stamp} | {run.tag} | {fmt_general(run.summary.get('delta'))} | "
            f"{run.summary.get('n_eval_frames', '?')} | {s['rows']} | {s['applied']} | "
            f"{s['fold']} | {s['noise']} | {source} |"
        )
    lines.append("")

    for fig in figures:
        if fig.name.startswith("gate_timeline"):
            lines += [f"![{fig.stem}]({fig.parent.name}/{fig.name})", ""]

    lines += [
        "## 4. Environment-drift diagnosis",
        "",
        "The historical ep15 instability is environmental, not coefficient-driven:",
        "the beam drifts across the sensor, so `J` and the frame peak `max_brt` move",
        "together. A strong |corr| here means the tracked objective is following the",
        "drift rather than the shaping quality.",
        "",
        "| run | tag | corr(J, max_brt) | slope dJ/d(max_brt) | J range |",
        "|---|---|---|---|---|",
    ]
    for run in smoke_runs:
        drift = drift_correlation(run)
        _, j = run.series("J")
        jrange = f"{j.min():.4g} .. {j.max():.4g}" if j.size else "-"
        if drift is None:
            lines.append(f"| {run.stamp} | {run.tag} | n/a | n/a | {jrange} |")
        else:
            lines.append(
                f"| {run.stamp} | {run.tag} | {drift[0]:+.4f} | {drift[1]:+.4g} "
                f"| {jrange} |"
            )
    lines.append("")
    for fig in figures:
        if fig.name.startswith("j_trajectory"):
            lines += [f"![{fig.stem}]({fig.parent.name}/{fig.name})", ""]

    lines += ["## 5. Per-run detail", ""]
    for run in runs:
        lines += [f"### {run.stamp} — {run.tag}", ""]
        for key, value in run.summary.items():
            if key == "final_c":
                continue
            lines.append(f"- `{key}`: `{value}`")
        for warning in run.warnings:
            lines.append(f"- ⚠️ {warning}")
        lines.append("")

    lines += [
        "## 6. Reproduction",
        "",
        "```bash",
        "# regenerate this report (offline)",
        "python scripts/generate_slm_pib_online_report.py",
        "",
        "# re-run the hardware suite (requires AO_RUN_HARDWARE=1 + online devices)",
        "AO_RUN_HARDWARE=1 uv run pytest "
        "tests/ao_shaping/optimizer/wfless/test_slm_zernike_pib_online.py -v",
        "```",
        "",
    ]
    path = out / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _union_deltas(snr_runs: list[Run]) -> list[str]:
    deltas: set[str] = set()
    for run in snr_runs:
        deltas.update(run.summary.get("snr_by_delta", {}).keys())
    return sorted(deltas, key=float)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="directory holding the <stamp>/ artifact folders")
    parser.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT,
                        help="output directory for report.md + figures/")
    args = parser.parse_args()

    if not args.root.is_dir():
        print(f"error: artifact root not found: {args.root}")
        return 1

    print(f"reading {args.root}")
    runs = load_runs(args.root)
    if not runs:
        print(f"error: no usable runs under {args.root}")
        return 1
    print(f"  loaded {len(runs)} run(s)")

    out = args.out
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    # Idempotent: drop figures from a previous (possibly differently-named) run
    # so the report never links a stale PNG.
    for stale in fig_dir.glob("*.png"):
        stale.unlink()

    figures: list[Path] = []
    snr_fig = plot_snr(runs, fig_dir)
    if snr_fig:
        figures.append(snr_fig)
    for run in runs:
        if run.kind != "smoke":
            continue
        for fig in (plot_gate_timeline(run, fig_dir), plot_j_trajectory(run, fig_dir)):
            if fig:
                figures.append(fig)

    report = write_report(runs, out, figures)
    print(f"wrote {report}")
    for fig in figures:
        print(f"wrote {fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
