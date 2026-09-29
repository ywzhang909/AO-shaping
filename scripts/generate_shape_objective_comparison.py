"""Offline comparison of the three shape objectives on real measured frames.

Scores every camera frame recorded by the Daheng ``slm-pib`` online suite under
three independent shape objectives and reports whether they **agree**:

1. ``square_quality_score(cv, ee, ar)`` — the hand-tuned composite used by the
   square SPGD optimizer (uniformity / encircled energy / aspect ratio),
   ``[0, 1]``, higher is better;
2. ``compute_quality_score(compute_square_metrics(...))`` — the weighted
   composite in ``utils.image.beam_metrics`` used by the reporting path,
   ``[0, 1]``, higher is better;
3. ``1 - Pearson(measured, target)`` — the FourierGSNet ``shaping_loss``
   migrated into the hardware path (``utils.image.target.metrics``), unbounded,
   lower is better.

The point of the report is the **rank agreement**: migrating GSNet's loss onto
hardware is only sound if its ordering of candidate beams matches the
hand-tuned scores. Where the two disagree, the frames are printed so the
disagreement can be inspected.

**Fully offline**: reads the saved recorder pickles only; it never opens
hardware. ``1 - Pearson`` is imported from the canonical target-metrics module
— this script deliberately does **not** reimplement it.

Usage:
    python scripts/generate_shape_objective_comparison.py
    python scripts/generate_shape_objective_comparison.py --target-side 50
    python scripts/generate_shape_objective_comparison.py -o docs/slm_pib_online
"""

from __future__ import annotations

import argparse
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: must be set before importing pyplot

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# CJK-capable font fallbacks (per repo script convention).
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DEFAULT_ROOT = ROOT / "data" / "debug" / "slm_pib_online"
DEFAULT_OUT = ROOT / "docs" / "slm_pib_online"
DEFAULT_SIDE = 50


# ---------------------------------------------------------------------------
# Objective adapters
# ---------------------------------------------------------------------------


def _load_pearson():
    """Import the canonical ``1 - Pearson`` metric.

    Imported lazily so this report fails with a clear message rather than an
    opaque ``ImportError`` traceback if the migration has not landed yet.
    """
    try:
        from ao_shaping.utils.image.target.metrics import pearson_shape_metric
    except ImportError as exc:  # pragma: no cover - depends on migration state
        raise SystemExit(
            "error: `pearson_shape_metric` is not available in "
            "`ao_shaping.utils.image.target.metrics`.\n"
            "       The FourierGSNet Pearson migration must land before this "
            "report can be generated."
        ) from exc
    return pearson_shape_metric


def objective_square_quality(img: np.ndarray, center: tuple[float, float],
                             side: int) -> float:
    """``square_quality_score`` — higher is better."""
    from ao_shaping.optimizer.wfless.slm_square_shaping import square_quality_score
    from ao_shaping.utils.image.beam_metrics import compute_square_metrics

    m = compute_square_metrics(img, side, center)
    return float(square_quality_score(
        cv=m["uniformity_cv"],
        encircled_energy=m["encircled_energy"],
        aspect_ratio=m["aspect_ratio"],
    ))


def objective_beam_metrics_quality(img: np.ndarray, center: tuple[float, float],
                                   side: int) -> float:
    """``compute_quality_score`` — higher is better."""
    from ao_shaping.utils.image.beam_metrics import (
        compute_quality_score,
        compute_square_metrics,
    )

    return float(compute_quality_score(compute_square_metrics(img, side, center)))


def objective_pearson_loss(img: np.ndarray, center: tuple[float, float],
                           side: int) -> float:
    """``1 - Pearson`` — lower is better."""
    pearson_shape_metric = _load_pearson()
    loss, _energy = pearson_shape_metric(
        img, center, "square", side, 1.0
    )
    return float(loss)


# ---------------------------------------------------------------------------
# Statistics (no SciPy — no new third-party dependency)
# ---------------------------------------------------------------------------


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks, matching ``scipy.stats.rankdata`` tie handling."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=np.float64)
    ranks[order] = np.arange(1, a.size + 1, dtype=np.float64)
    # Average the ranks inside each tie group.
    sorted_a = a[order]
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation; ``nan`` when either series is constant."""
    if x.size < 3 or y.size < 3:
        return float("nan")
    rx, ry = _rankdata(x), _rankdata(y)
    if np.allclose(rx, rx[0]) or np.allclose(ry, ry[0]):
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Linear correlation; ``nan`` when either series is constant."""
    if x.size < 3 or y.size < 3:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class FrameSet:
    """Frames from one recorder pickle, plus their per-row gate verdicts."""

    stamp: str
    tag: str
    epochs: np.ndarray
    frames: list[np.ndarray]
    gates: list[str]

    @property
    def label(self) -> str:
        return f"{self.stamp}_{self.tag}"


def load_frame_sets(root: Path) -> list[FrameSet]:
    """Load every readable recorder pickle under ``root``."""
    out: list[FrameSet] = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for pkl in sorted(run_dir.glob("recorder_*.pkl")):
            try:
                obj = pickle.loads(pkl.read_bytes())
                if obj is None or not hasattr(obj, "history"):
                    continue  # the SNR sweep dumps ``None``; no frames to compare
            except (OSError, pickle.UnpicklingError, EOFError):
                continue
            rows = list(obj.history)
            if len(rows) < 2:
                continue
            frames = [np.asarray(r["_img"]) for r in rows[1:] if "_img" in r]
            epochs = [float(r.get("_epoch", i + 1)) for i, r in enumerate(rows[1:])]
            gates = [str(r.get("_gate", "unknown")) for r in rows[1:]]
            if not frames:
                continue
            out.append(FrameSet(
                stamp=run_dir.name,
                tag=pkl.stem.removeprefix("recorder_"),
                epochs=np.asarray(epochs[: len(frames)]),
                frames=frames,
                gates=gates[: len(frames)],
            ))
    return out


def spot_center(frame: np.ndarray) -> tuple[float, float]:
    """Zero-order spot as ``(x, y)`` via ``argmax`` — never the frame centre.

    On the 2f bench the optical axis lands wherever the beam happens to be, so
    the frame centre is not a valid anchor (see the repo anti-pattern list).
    """
    y, x = np.unravel_index(int(np.argmax(frame)), frame.shape)
    return float(x), float(y)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_objective_traces(fs: FrameSet, ys: dict[str, np.ndarray],
                          out: Path) -> Path:
    """All three objectives vs epoch (Pearson on a secondary inverted axis)."""
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    epochs = fs.epochs
    titles = [
        ("square_quality_score", "higher better"),
        ("compute_quality_score", "higher better"),
        ("1 - Pearson (GSNet loss)", "lower better"),
    ]
    for ax, (key, note) in zip(axes, titles):
        ax.plot(epochs, ys[key], "-o", ms=3)
        ax.set_ylabel(key.split("_")[0])
        ax.set_title(f"{key}  ({note})", fontsize=9)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("epoch")
    fig.suptitle(f"{fs.label} — three shape objectives on the same frames", fontsize=11)
    fig.tight_layout()
    path = out / f"objective_traces_{fs.label}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_agreement(scatter: list[tuple[str, np.ndarray, np.ndarray]],
                   out: Path) -> Path:
    """Scatter of the composite score against the Pearson loss, per frame set."""
    fig, axes = plt.subplots(1, len(scatter), figsize=(5 * len(scatter), 4.4),
                             squeeze=False)
    for ax, (label, comp, pear) in zip(axes[0], scatter):
        ax.scatter(comp, pear, s=18, alpha=0.75)
        rho = spearman(comp, pear)
        r = pearson_r(comp, pear)
        ax.set_xlabel("square_quality_score (higher better)")
        ax.set_ylabel("1 - Pearson (lower better)")
        ax.set_title(f"{label}\nSpearman={rho:+.3f}  r={r:+.3f}", fontsize=9)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out / "objective_agreement.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(v: float, spec: str = "+.4f") -> str:
    return "n/a" if not np.isfinite(v) else format(v, spec)


def analyse(fs: FrameSet, side: int) -> tuple[dict[str, np.ndarray], tuple[float, float]]:
    """Score every frame under all three objectives."""
    center = spot_center(fs.frames[0])
    ys: dict[str, list[float]] = {k: [] for k in
                                 ("square_quality_score", "compute_quality_score",
                                  "pearson_loss")}
    for frame in fs.frames:
        ys["square_quality_score"].append(
            objective_square_quality(frame, center, side))
        ys["compute_quality_score"].append(
            objective_beam_metrics_quality(frame, center, side))
        ys["pearson_loss"].append(objective_pearson_loss(frame, center, side))
    arrays = {k: np.asarray(v, dtype=np.float64) for k, v in ys.items()}
    return arrays, center


def write_report(results: list[tuple[FrameSet, dict[str, np.ndarray], tuple[float, float]]],
                 side: int, out: Path, figures: list[Path]) -> Path:
    lines: list[str] = [
        "# Shape Objective Comparison Report",
        "",
        "Offline comparison of three shape objectives scored on the **same** real",
        f"Daheng CCD frames recorded by the `slm-pib` online suite (target: square,",
        f"side {side} px, anchored at the baseline `argmax`).",
        "",
        "| objective | source | polarity | bounded |",
        "|---|---|---|---|",
        "| `square_quality_score` | `optimizer/wfless/slm_square_shaping.py` | higher better | `[0, 1]` |",
        "| `compute_quality_score` | `utils/image/beam_metrics.py` | higher better | `[0, 1]` |",
        "| `1 - Pearson` | `utils/image/target/metrics.py` (FourierGSNet `shaping_loss`) | lower better | unbounded |",
        "",
        "## 1. Rank agreement",
        "",
        "Migrating the GSNet loss onto hardware is only sound if it orders candidate",
        "beams like the hand-tuned scores. Spearman is the agreement that matters",
        "(Pearson loss is a *loss*, so a **negative** correlation with a higher-is-better",
        "score is the expected sign of agreement).",
        "",
        "| run | n frames | Spearman(square, pearson) | Spearman(metrics, pearson) | Spearman(square, metrics) |",
        "|---|---|---|---|---|",
    ]
    for fs, ys, _c in results:
        sq, bm, pe = (ys["square_quality_score"], ys["compute_quality_score"],
                      ys["pearson_loss"])
        lines.append(
            f"| {fs.label} | {sq.size} | {_fmt(spearman(sq, pe))} | "
            f"{_fmt(spearman(bm, pe))} | {_fmt(spearman(sq, bm))} |"
        )
    lines += ["", "## 2. Value ranges", "",
              "| run | square_quality_score | compute_quality_score | 1 - Pearson |",
              "|---|---|---|---|"]
    for fs, ys, _c in results:
        def rng(a: np.ndarray) -> str:
            return f"{a.min():.4f} .. {a.max():.4f}"
        lines.append(
            f"| {fs.label} | {rng(ys['square_quality_score'])} | "
            f"{rng(ys['compute_quality_score'])} | {rng(ys['pearson_loss'])} |"
        )

    lines += [
        "",
        "The two composite scores are bounded in `[0, 1]` and therefore cannot",
        "express *how bad* a frame is — they saturate. The Pearson loss is",
        "unbounded, so on a folded or dark frame it keeps growing while the",
        "composite scores flatten out. That unbounded tail is what lets the",
        "hardware gate reject an epoch; the composites cannot.",
    ]

    for fig in figures:
        lines += ["", f"![{fig.stem}]({fig.parent.name}/{fig.name})"]

    lines += ["", "## 3. Per-run objective traces", ""]
    for fs, ys, center in results:
        lines += [
            f"### {fs.label}",
            "",
            f"- frames: **{fs.epochs.size}** (epochs {fs.epochs.min():.0f}"
            f"-{fs.epochs.max():.0f})",
            f"- target anchor `(x, y)` = ({center[0]:.0f}, {center[1]:.0f}), "
            f"from the baseline frame `argmax`",
            f"- gate verdicts: "
            + ", ".join(f"{g}={fs.gates.count(g)}" for g in
                        sorted(set(fs.gates))),
            "",
        ]
        # Show the frames where the objectives disagree most — those are the
        # interesting ones for judging the migration.
        sq, pe = ys["square_quality_score"], ys["pearson_loss"]
        order = np.argsort(-(sq - _normalise(pe)))
        lines += ["| epoch | gate | square_quality_score | 1 - Pearson |",
                  "|---|---|---|---|"]
        for i in order[:5]:
            lines.append(
                f"| {fs.epochs[i]:.0f} | {fs.gates[i]} | {sq[i]:.4f} | {pe[i]:.4f} |"
            )
        lines.append("")

    lines += [
        "## 4. Reproduction",
        "",
        "```bash",
        "python scripts/generate_shape_objective_comparison.py",
        "```",
        "",
    ]
    path = out / "objective_comparison.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _normalise(a: np.ndarray) -> np.ndarray:
    """Min-max scale to `[0, 1]` so two different units can be compared."""
    span = a.max() - a.min()
    return (a - a.min()) / span if span > 0 else np.zeros_like(a)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--target-side", type=int, default=DEFAULT_SIDE)
    parser.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.root.is_dir():
        print(f"error: artifact root not found: {args.root}")
        return 1

    frame_sets = load_frame_sets(args.root)
    if not frame_sets:
        print(f"error: no recorded frames under {args.root}")
        return 1
    print(f"loaded {len(frame_sets)} frame set(s)")

    results: list[tuple[FrameSet, dict[str, np.ndarray], tuple[float, float]]] = []
    for fs in frame_sets:
        ys, center = analyse(fs, args.target_side)
        results.append((fs, ys, center))
        print(f"  {fs.label}: {fs.epochs.size} frames @ center"
              f"({center[0]:.0f},{center[1]:.0f})")

    out = args.out
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    figures = [plot_agreement([(fs.label, ys["square_quality_score"],
                                ys["pearson_loss"]) for fs, ys, _c in results],
                              fig_dir)]
    for fs, ys, _c in results:
        figures.append(plot_objective_traces(fs, ys, fig_dir))

    report = write_report(results, args.target_side, out, figures)
    print(f"wrote {report}")
    for fig in figures:
        print(f"wrote {fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
