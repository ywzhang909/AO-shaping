"""Offline transfer prediction diagnostics and experimental committee ranking.

Disagreement is an uncalibrated ranking score, never an acceptance gate or a
probability of correctness. No operation in this module opens SLM/CCD devices.
Report rendering remains in scripts/slm/transfer_uncertainty_experiment.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from loguru import logger


def normalized(images: np.ndarray) -> np.ndarray:
    """Normalize nonnegative intensity by each image's spatial mean."""
    return images / np.maximum(images.mean(axis=(-2, -1), keepdims=True), 1e-9)


def disagreement(predictions: np.ndarray) -> np.ndarray:
    """Return per-command RMS ensemble standard deviation after normalization.

    Input axes: member, command, camera row, camera column. Overall gain
    differences are removed; common model bias remains invisible.
    """
    if predictions.ndim != 4 or min(predictions.shape) < 1 or predictions.shape[0] < 2:
        raise ValueError("predictions must have shape (members>=2, commands, rows, columns)")
    if not np.isfinite(predictions).all() or np.any(predictions < 0):
        raise ValueError("predictions must be finite nonnegative intensities")
    unit = normalized(predictions)
    return np.sqrt(unit.var(axis=0).mean(axis=(-2, -1)))


def select_diverse(scores: np.ndarray, phases: np.ndarray, count: int) -> list[int]:
    """Rank by score, rejecting wrapped commands with distance <= 0.05.

    Distance is mean(1-cos(delta phase)), invariant to integer 2*pi wrapping.
    This rejects near duplicates; it does not ensure phase-family coverage.
    """
    if phases.ndim != 3 or scores.shape != (len(phases),):
        raise ValueError("scores and phases must have shapes (commands,) and (commands, rows, columns)")
    if not np.isfinite(scores).all() or not np.isfinite(phases).all():
        raise ValueError("scores and phases must be finite")
    if not 1 <= count <= len(phases):
        raise ValueError("count must be between one and the candidate count")
    selected: list[int] = []
    for index in np.argsort(-scores, kind="stable"):
        if all(float(np.mean(1 - np.cos(phases[index] - phases[old]))) > .05
               for old in selected):
            selected.append(int(index))
        if len(selected) == count:
            return selected
    raise ValueError("insufficient diverse candidate phases")


def evaluate(prediction: np.ndarray, measured: np.ndarray) -> dict:
    """Compare paired intensity stacks on the same fixed camera grid and units."""
    from ao_shaping.tools.slm.transfer.fit import metrics

    if prediction.ndim != 3 or prediction.shape != measured.shape or min(prediction.shape) < 1:
        raise ValueError("prediction and measured must have identical nonempty (commands, rows, columns) shapes")
    if not np.isfinite(prediction).all() or not np.isfinite(measured).all():
        raise ValueError("prediction and measured must be finite")
    if np.any(prediction < 0) or np.any(measured < 0):
        raise ValueError("prediction and measured must be nonnegative intensities")
    details = metrics(prediction, measured)
    shape_error = np.sqrt(((normalized(prediction) - normalized(measured)) ** 2)
                          .mean(axis=(-2, -1))
                          / np.maximum((normalized(measured) ** 2).mean(axis=(-2, -1)), 1e-9))
    return dict(mean_ncc=float(np.mean([row["correlation"] for row in details])),
                mean_relative_l2=float(np.mean([row["relative_l2"] for row in details])),
                mean_shape_relative_l2=float(shape_error.mean()),
                shape_relative_l2=shape_error.tolist(), per_phase=details)


def predict(checkpoint: Path, phases: np.ndarray, xy: np.ndarray, exposure: np.ndarray,
            device: str) -> np.ndarray:
    """Predict counts/ms using a frozen checkpoint and recorded exposure values."""
    import torch

    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        FrozenDefocusForward,
        load_frozen_parameters,
    )

    if len(phases) == 0 or exposure.shape != (len(phases),):
        raise ValueError("one exposure value is required per phase")
    model = FrozenDefocusForward(load_frozen_parameters(checkpoint), xy, device=device)
    with torch.no_grad():
        return np.concatenate([model.predict(phases[i:i+4], exposure_ms=exposure[i:i+4])
                               .cpu().numpy() for i in range(0, len(phases), 4)])


def fit_recorded_subset(directory: Path, checkpoint: Path, calibration: list[dict],
                        training: list[dict], validation: list[dict], payloads: dict,
                        *, iterations: int, span: int, device: str) -> Path:
    """Fit recorded phases; duplicate training rows implement a phase bootstrap.

    Use a fresh output directory for every input subset. Test rows are rejected
    so final evaluation observations cannot silently become fitting examples.
    """
    from ao_shaping.tools.slm.transfer.fit import fit

    if any(row["split"] == "test" for row in calibration + training + validation):
        raise ValueError("test rows cannot participate in subset fitting")
    directory.mkdir(parents=True, exist_ok=False)
    dataset = directory / "dataset"
    dataset.mkdir()
    rows = calibration + [dict(row, split="train") for row in training]
    rows += [dict(row, split="validation") for row in validation]
    (dataset / "manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (dataset / "config.json").write_text(json.dumps({"experiment": directory.name}), encoding="utf-8")
    fit(SimpleNamespace(dataset=dataset, checkpoint=checkpoint, out=directory / "fit",
                        iterations=iterations, span=span, device=device,
                        adaptation="full", radiometry="exposure", samples=payloads))
    return directory / "fit/frozen_parameters.json"


def main() -> None:
    """Offer offline rank/evaluate commands over saved numerical arrays."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    rank = commands.add_parser("rank", help="experimental ranking, no device action")
    rank.add_argument("--input", type=Path, required=True,
                      help="NPZ with predictions[M,N,H,W] and phases[N,G,G]")
    rank.add_argument("--count", type=int, required=True)
    rank.add_argument("--out", type=Path, required=True)
    evaluate_parser = commands.add_parser("evaluate", help="paired fixed-window intensity errors")
    evaluate_parser.add_argument("--input", type=Path, required=True,
                                 help="NPZ with prediction[N,H,W] and measured[N,H,W]")
    evaluate_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        if args.command == "rank":
            scores = disagreement(data["predictions"])
            result = dict(selected=select_diverse(scores, data["phases"], args.count),
                          scores=scores.tolist(), calibrated=False,
                          note="Experimental ranking only; validate against residuals and random sampling")
        else:
            result = evaluate(data["prediction"], data["measured"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Saved offline {} diagnostics to {}", args.command, args.out)


if __name__ == "__main__":
    main()
