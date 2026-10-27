"""Calibrate effective blur/radiometry using measured shapes, with shape-family validation.

Square/triangle seed images are training; rectangle/star seeds are validation.
No validation image selects the effective blur. Other physical parameters stay fixed.
"""

from __future__ import annotations

import argparse
from loguru import logger
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from ao_shaping.tools.slm.transfer.capture import clock
from ao_shaping.tools.slm.transfer.validate import ShapeObjective, image_metrics
from ao_shaping.utils.image.target import POLYGON_SHAPE_CHOICES


def available_seed_shapes(rows):
    """Require both training shapes and at least one unseen shape family."""
    labels = {r["label"] for r in rows}
    # Single vocabulary (issue #199 P1): the polygon seed family, as a documented
    # subset view of the canonical TARGET_SHAPE_CHOICES.
    shapes = [s for s in POLYGON_SHAPE_CHOICES if s + "_seed" in labels]
    if not {"square", "triangle"}.issubset(shapes):
        raise ValueError("square and triangle seed captures required for training")
    if not {"rectangle", "star"}.intersection(shapes):
        raise ValueError("rectangle or star seed capture required for validation")
    return shapes


def calibrate(args):
    import torch
    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        FrozenDefocusForward,
        load_frozen_parameters,
    )

    torch.set_num_threads(4)
    params = load_frozen_parameters(args.checkpoint)
    rows = json.loads((args.source / "manifest.json").read_text())
    samples = []
    for shape in available_seed_shapes(rows):
        row = next(r for r in rows if r["label"] == shape + "_seed")
        with np.load(args.source / row["file"]) as data:
            xy = data["camera_xy"]
            measured = (
                data["raw_frames"].mean(0)[
                    xy[..., 1].astype(int), xy[..., 0].astype(int)
                ]
                / row["exposure_ms"]
            )
            samples.append(
                dict(
                    shape=shape,
                    phase=data["model_phase"],
                    image=measured,
                    xy=xy,
                    objective=ShapeObjective(data["target"]),
                    row=row,
                )
            )
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "config.json").write_text(
        json.dumps(
            dict(
                source=str(args.source.resolve()),
                checkpoint=str(args.checkpoint.resolve()),
                train_shapes=["square", "triangle"],
                sigmas=args.sigmas,
                **clock(),
            ),
            indent=2,
        )
    )
    train = np.array([s["shape"] in {"square", "triangle"} for s in samples])
    measured = np.stack([s["image"] for s in samples])
    normalization = np.sqrt((measured[train] ** 2).mean((1, 2))).clip(1e-9)
    weights = np.broadcast_to(
        (1 / normalization)[:, None, None], measured[train].shape
    ).ravel()
    results = []
    for sigma in args.sigmas:
        trial = replace(params, sigma_ccd_px=sigma, gain=1.0, background=0.0, bias_counts=0.)
        predictions = []
        for sample in samples:
            forward = FrozenDefocusForward(trial, sample["xy"], device="cpu")
            with torch.no_grad():
                predictions.append(forward.predict(sample["phase"])[0].numpy())
        optical = np.stack(predictions)
        design = np.stack(
            (optical[train].ravel(), np.ones(optical[train].size)), axis=-1
        )
        bias = np.asarray([params.bias_counts/s["row"]["exposure_ms"] for s in samples])[:, None, None]
        gain, background = nnls(
            design * weights[:, None], (measured-bias)[train].ravel() * weights
        )[0]
        prediction = optical * gain + background + bias
        metrics = []
        scores = []
        for sample, im, p in zip(samples, measured, prediction):
            truth = image_metrics(im, sample["objective"])
            predicted = image_metrics(p, sample["objective"])
            relative_mse = float(((im - p) ** 2).mean() / max((im**2).mean(), 1e-9))
            cv_relative_error = (predicted["cv"] - truth["cv"]) / max(truth["cv"], 0.1)
            scores.append(relative_mse + cv_relative_error**2)
            metrics.append(
                dict(
                    shape=sample["shape"],
                    measured=truth,
                    predicted=predicted,
                    correlation=float(np.corrcoef(im.ravel(), p.ravel())[0, 1]),
                    relative_mse=relative_mse,
                )
            )
        entry = dict(
            sigma=sigma,
            gain=float(gain),
            background=float(background),
            train_score=float(np.mean(np.array(scores)[train])),
            validation_score=float(np.mean(np.array(scores)[~train])),
            metrics=metrics,
            **clock(),
        )
        results.append(entry)
        np.savez_compressed(
            args.out / f"sigma_{sigma:g}.npz", prediction=prediction, measured=measured
        )
        logger.info(json.dumps(entry))
    best = min(results, key=lambda r: r["train_score"])
    selected = replace(
        params,
        sigma_ccd_px=best["sigma"],
        gain=best["gain"],
        background=best["background"],
    )
    (args.out / "frozen_parameters.json").write_text(
        json.dumps(selected.to_dict(), indent=2)
    )
    (args.out / "result.json").write_text(
        json.dumps(
            dict(
                selected=best,
                variants=results,
                rule="minimum training relative image MSE + relative core CV error squared",
            ),
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0, 2.0, 3.515916109085083],
    )
    args = parser.parse_args()
    if any(s < 0 for s in args.sigmas):
        parser.error("sigma must be nonnegative")
    calibrate(args)


if __name__ == "__main__":
    main()
