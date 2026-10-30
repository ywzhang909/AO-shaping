"""Offline, read-only summary of one ``blackbox_path`` transfer run (no device).

    python scripts/slm/summarize_blackbox_path_20261029.py data/shape_trials/<run>
    python scripts/slm/summarize_blackbox_path_20261029.py data/shape_trials/<run> --metrics

Prints the stage journal, the per-stage artifact accounting (file count / bytes,
so "过程数据是否全部落盘" is answered by numbers rather than by trust), the fitted
parameters and the CCD acceptance rows.

``--metrics`` additionally **recomputes** every anchored capture with the
canonical :func:`~ao_shaping.algorithm.signal_processing.multishape_objective.image_metrics`
and writes ``summary/metrics_recomputed.json``.  That is what makes a metric added
*after* a run (e.g. ``target_pearson``) retroactively available for that run: the
raw CCD frames and the frozen ``target`` / ``core`` grids are on disk, so the
number can be re-derived instead of re-measured.  Nothing else writes into the run
directory, and no driver is imported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _stage_accounting(run: Path) -> list[tuple[str, int, float]]:
    rows = []
    for child in sorted(run.iterdir()):
        if not child.is_dir():
            continue
        files = [p for p in child.rglob("*") if p.is_file()]
        rows.append((child.name, len(files), sum(p.stat().st_size for p in files) / 1e6))
    return rows


def _recompute(run: Path) -> dict:
    from ao_shaping.algorithm.signal_processing.multishape_objective import (
        ShapeObjective, image_metrics)

    anchored = run / "anchored"
    manifest = _json(anchored / "manifest.json") or []
    rows = []
    for entry in manifest:
        path = anchored / entry["file"]
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as data:
            target = data["target"]
            core = data["core"]
            xy = data["camera_xy"]
            frames = data["raw_frames"]
        objective = ShapeObjective(target)
        core_matches = bool(np.array_equal(objective.core, core))
        frame = frames.mean(0)[xy[..., 1].astype(int), xy[..., 0].astype(int)]
        metrics = image_metrics(frame, objective)
        rows.append({"label": entry["label"], "file": entry["file"],
                     "exposure_ms": entry["frames"][0]["exposure_ms"],
                     "core_matches_frozen": core_matches,
                     **{k: (None if v is None else float(v)) for k, v in metrics.items()}})
    return {"run": run.name, "captures": rows}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--metrics", action="store_true",
                        help="recompute image_metrics for every anchored capture")
    args = parser.parse_args(argv)
    run = args.run.resolve()
    print(f"run: {run}")

    session = _json(run / "session.json")
    if session:
        settings = session.get("settings", {})
        print("\n== settings ==")
        for key, value in settings.items():
            print(f"  {key:18s} {value}")
        if session.get("resolved"):
            print(f"  resolved           {session['resolved']}")

    journal = _json(run / "journal.json") or []
    print("\n== stages ==")
    for entry in journal:
        print(f"  {entry.get('stage'):16s} {entry.get('status'):9s} "
              f"{entry.get('started_at', '')} -> {entry.get('finished_at', '')}")
        for key in ("error", "message"):
            if entry.get(key):
                print(f"      {key}: {entry[key]}")

    print("\n== artifacts on disk ==")
    for name, count, megabytes in _stage_accounting(run):
        print(f"  {name:20s} files={count:5d}  {megabytes:9.1f} MB")

    fit = _json(run / "fit" / "result.json")
    if fit:
        print("\n== fit ==")
        print(f"  iteration {fit.get('iteration')}  loss {fit.get('loss'):.4f}")
        params = fit.get("parameters", {})
        for key in ("defocus_diopter", "origin", "amplitude_center", "amplitude_width",
                    "gain", "background", "bias_counts", "sigma_ccd_px"):
            if key in params:
                print(f"  {key:18s} {params[key]}")
        residual = params.get("residual_noll_radians") or {}
        if residual:
            print("  residual Noll (rad) " + " ".join(
                f"{n}:{float(v):+.3f}" for n, v in residual.items()))
        metrics = fit.get("metrics") or []
        if metrics:
            corr = [m["correlation"] for m in metrics]
            l2 = [m["relative_l2"] for m in metrics]
            print(f"  probes: n={len(metrics)} corr {min(corr):.3f}..{max(corr):.3f} "
                  f"relative_l2 {min(l2):.3f}..{max(l2):.3f}")

    summary = _json(run / "summary" / "results.json")
    if summary:
        print("\n== acceptance ==")
        for shape in summary.get("shapes", []):
            print(f"  {shape['shape']}")
            for label in ("seed", "final"):
                row = shape.get(label)
                if not row:
                    continue
                measured = row.get("measured", {})
                print(f"    {label:6s} loss {row.get('loss'):.4f} "
                      f"corr {row.get('correlation'):.3f} "
                      f"exposure {row.get('exposure_ms')} ms")
                print("           " + "  ".join(
                    f"{key}={value:.4f}" for key, value in measured.items()
                    if isinstance(value, (int, float))))
            direction = shape.get("direction") or {}
            if direction:
                print("    direction " + "  ".join(
                    f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in direction.items()))

    if args.metrics:
        payload = _recompute(run)
        out = run / "summary" / "metrics_recomputed.json"
        out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"\n== recomputed metrics -> {out.relative_to(run)} ==")
        print(f"  {'label':20s} {'cv':>7s} {'ee_roi':>8s} {'pearson':>8s} {'mean':>9s}")
        for row in payload["captures"]:
            cv = row["cv"]
            pearson = row["target_pearson"]
            print(f"  {row['label']:20s} {'n/a' if cv is None else f'{cv:7.4f}'} "
                  f"{row['ee_roi']:8.5f} "
                  f"{'n/a' if pearson is None else f'{pearson:8.4f}'} "
                  f"{row['mean']:9.2f}")
        mismatched = [row["label"] for row in payload["captures"]
                      if not row["core_matches_frozen"]]
        if mismatched:
            print(f"  WARNING core mask mismatch: {mismatched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
