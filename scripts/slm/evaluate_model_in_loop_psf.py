"""Select a small CCD PSF on train probes and score frozen hardware holdouts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import nnls

from ao_shaping.optimizer.wfless.slm_model_in_loop import (
    SlmModelInLoopConfig, _make_optimizer, _to_model_grid,
)
from ao_shaping.utils.image.beam_metrics import prepare_frame, zero_order_center


SIGMAS = (0.0, 0.5, 1.0, 1.5, 2.0)


def fit_gain_leakage(
    predictions: list[np.ndarray], measured: list[np.ndarray], flat: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Fit nonnegative coefficients and return train relative shape error."""
    if not predictions or len(predictions) != len(measured):
        raise ValueError("matched nonempty train probes required")
    features = np.concatenate([
        np.stack((pred.ravel(), flat.ravel()), axis=1) for pred in predictions
    ])
    targets = np.concatenate([image.ravel() for image in measured])
    weights, _ = nnls(features, targets)
    error = float(np.linalg.norm(features @ weights - targets) / np.linalg.norm(targets))
    return weights, error


def _score(
    prediction: np.ndarray, measured: np.ndarray, observed_mask: np.ndarray,
) -> tuple[float, float]:
    a = np.maximum(np.asarray(prediction, dtype=np.float64), 0)[observed_mask]
    b = np.maximum(np.asarray(measured, dtype=np.float64), 0)[observed_mask]
    correlation = float(np.corrcoef(a, b)[0, 1])
    # A nonnegative scalar removes run-to-run power drift from the shape error.
    scale = max(0.0, float(a @ b / max(a @ a, 1e-30)))
    error = float(np.linalg.norm(scale * a - b) / np.linalg.norm(b))
    return correlation, error


def load_run(run_dir: Path) -> tuple[object, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[dict, np.ndarray, np.ndarray]]]:
    """Load the frozen 0.4 ms model, flat frame, and measured train/holdout probes."""
    geometry = json.loads((run_dir / "bench_geometry.json").read_text(encoding="utf-8"))
    recorded = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    region = int(geometry["region"])
    size = int(geometry["far_field_size"])
    config = SlmModelInLoopConfig(
        region=region, far_field_padding=size // region,
        n_orders=int(recorded["n_orders"]),
    )
    coefficients = np.load(run_dir / "fitted_coefficients.npy", allow_pickle=False)
    amplitude = np.load(run_dir / "source_amplitude.npy", allow_pickle=False)
    optimizer = _make_optimizer(config, coefficients, amplitude)
    root = run_dir / "probes"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    with np.load(root / "round_00_flat_00.npz") as record:
        flat_frame = prepare_frame(record["frame"])
    center = zero_order_center(flat_frame, refine=False)

    def model_grid(frame: np.ndarray) -> np.ndarray:
        return np.asarray(_to_model_grid(
            prepare_frame(frame), center, size,
            float(geometry["camera_px_per_model_px"]), 1,
        ), dtype=np.float64)

    flat = model_grid(flat_frame)
    observed_mask = _to_model_grid(
        np.ones_like(flat_frame, dtype=np.float32), center, size,
        float(geometry["camera_px_per_model_px"]), 1,
    ) > 0.5
    samples = []
    for item in manifest:
        if item["split"] not in {"train", "holdout"}:
            continue
        if abs(float(item["exposure_ms"]) - 0.4) > 1e-6:
            raise ValueError(f"{run_dir}: non-matched probe exposure")
        with np.load(root / item["file"]) as record:
            measured = (
                np.asarray(record["model_grid"], dtype=np.float64)
                if "model_grid" in record else model_grid(record["frame"])
            )
            phase = np.asarray(record["phase"], dtype=np.float64).copy()
        samples.append((item, phase, measured))
    return optimizer, coefficients, amplitude, flat, observed_mask, samples


def evaluate_run(run_dir: Path, sigmas: tuple[float, ...] = SIGMAS) -> list[dict]:
    if not sigmas or 0.0 not in sigmas or any(not np.isfinite(s) or s < 0 for s in sigmas):
        raise ValueError("PSF candidates must contain zero and be finite and nonnegative")
    optimizer, coefficients, amplitude, flat, observed_mask, raw_samples = load_run(run_dir)
    samples = [
        (item, optimizer.forward_intensity(coefficients, phase, amplitude), measured)
        for item, phase, measured in raw_samples
    ]
    train = [sample for sample in samples if sample[0]["split"] == "train"]
    holdout = [sample for sample in samples if sample[0]["split"] == "holdout"]
    if not train or not holdout:
        raise ValueError(f"{run_dir}: missing train or holdout probes")

    candidates = []
    for sigma in sigmas:
        blurred = lambda image, _sigma=sigma: (
            gaussian_filter(image, _sigma) if _sigma else image)
        weights, error = fit_gain_leakage(
            [blurred(pred) for _, pred, _ in train],
            [measured for _, _, measured in train], flat,
        )
        candidates.append((error, sigma, weights))
    _, selected_sigma, weights = min(candidates, key=lambda row: row[0])
    baseline_weights = next(row[2] for row in candidates if row[1] == 0)
    rows = []
    for item, predicted, measured in holdout:
        for label, sigma, fitted in (
            ("baseline", 0.0, baseline_weights),
            ("train_selected_psf", selected_sigma, weights),
        ):
            image = fitted[0] * (
                gaussian_filter(predicted, sigma) if sigma else predicted
            ) + fitted[1] * flat
            correlation, error = _score(image, measured, observed_mask)
            rows.append({
                "run": run_dir.name, "family": item["family"],
                "index": int(item["index"]), "method": label,
                "sigma_model_px": sigma, "correlation": correlation,
                "shape_nrmse": error,
                "train_relative_error": next(row[0] for row in candidates if row[1] == sigma),
                "model_gain": float(fitted[0]), "flat_leakage": float(fitted[1]),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    args = parser.parse_args()
    rows = [row for path in args.run_dirs for row in evaluate_run(path)]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for method in ("baseline", "train_selected_psf"):
        subset = [row for row in rows if row["method"] == method]
        print(method, "corr", np.mean([row["correlation"] for row in subset]),
              "nrmse", np.mean([row["shape_nrmse"] for row in subset]))


if __name__ == "__main__":
    main()
