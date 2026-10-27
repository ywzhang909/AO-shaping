"""Offline metric traces from incremental experiment records; no devices."""
from __future__ import annotations
from loguru import logger

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def fit_trace(directory):
    directory = Path(directory)
    provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    with (directory / "iterations.pkl").open("rb") as stream:
        history = pickle.load(stream)
    train = np.asarray([row["split"] == "train" for row in provenance["rows"]])
    result = []
    for entry in history:
        row = dict(iteration=entry["iteration"], loss=entry["loss"], timestamp_utc=entry["timestamp_utc"])
        for name, mask in (("train", train), ("validation", ~train)):
            for metric in ("correlation", "relative_l2", "cv_abs_error"):
                row[f"{name}_{metric}"] = float(np.mean([m[metric] for m, keep in zip(entry["metrics"], mask) if keep]))
        result.append(row)
    return result


def measurement_trace(directory):
    directory = Path(directory)
    records = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    decisions = json.loads((directory / "summary.json").read_text(encoding="utf-8")) if (directory / "summary.json").exists() else []
    return dict(measurements=[dict(label=r["label"], loss=r["loss"], exposure_ms=r["exposure_ms"],
                                  timestamp_utc=r["timestamp_utc"], **r["metrics"]) for r in records],
                decisions=decisions)


def select_checkpoint(directories, out, *, cv_weight=.25):
    """Validation selection includes CV error; final CCD acceptance is independent.

    The dimensionless CV weight is an explicit selection heuristic, not a
    physical acceptance threshold. No validation pixels enter parameter fitting.
    """
    from .adaptation import validation_quality
    candidates = []
    for directory in map(Path, directories):
        provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
        with (directory / "iterations.pkl").open("rb") as stream:
            history = pickle.load(stream)
        for entry in history:
            quality = validation_quality(entry, provenance["rows"])
            candidates.append(dict(directory=directory, entry=entry, quality=quality,
                                   score=quality["score"]+cv_weight*quality["cv_abs_error"]))
    if not candidates or not np.isfinite(cv_weight) or cv_weight < 0:
        raise ValueError("finite nonnegative CV weight and recorded candidates required")
    selected = min(candidates, key=lambda c: c["score"])
    directory = selected["directory"]
    checkpoint = json.loads((directory / "frozen_parameters.json").read_text(encoding="utf-8"))
    checkpoint.update(selected["entry"]["parameters"])
    decision = dict(source=str(directory.resolve()), iteration=selected["entry"]["iteration"],
                    cv_weight=cv_weight, score=selected["score"], quality=selected["quality"],
                    selection="validation phases; requires independent paired CCD acceptance",
                    candidates=[dict(source=str(c["directory"]), iteration=c["entry"]["iteration"],
                                     score=c["score"], quality=c["quality"]) for c in candidates])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "frozen_parameters.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    (out / "selection.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--select-fits", type=Path, nargs="+")
    parser.add_argument("--cv-weight", type=float, default=.25)
    args = parser.parse_args()
    if args.select_fits:
        decision = select_checkpoint(args.select_fits, args.out, cv_weight=args.cv_weight)
        logger.info(json.dumps({k: v for k, v in decision.items() if k != "candidates"}, indent=2))
        return
    result = {}
    if args.fit:
        result["fit"] = fit_trace(args.fit)
    if args.run:
        result["physical"] = measurement_trace(args.run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.fit:
        logger.info(json.dumps(dict(initial=result["fit"][0], latest=result["fit"][-1]), indent=2))
    if args.run:
        logger.info(json.dumps(result["physical"], indent=2))


if __name__ == "__main__":
    main()
