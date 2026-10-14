"""Test a three-parameter camera-grid correction on archived CCD probes.

The fit uses training phases only. All saved holdout phases remain unseen until
the fitted transform and nonnegative gain/leakage are frozen.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize, nnls

from ao_shaping.optimizer.wfless.slm_model_in_loop import (
    SlmModelInLoopConfig,
    _make_optimizer,
    _prediction_correlation,
    _to_model_grid,
    prepare_frame,
)
from ao_shaping.utils.image.beam_metrics import zero_order_center


def warp_prediction(image: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    """Move a predicted intensity by (dx, dy) and scale about its FFT center."""
    dx, dy, scale = map(float, parameters)
    height, width = image.shape
    yy, xx = np.indices(image.shape, dtype=np.float64)
    xx = (xx - width // 2 - dx) / scale + width // 2
    yy = (yy - height // 2 - dy) / scale + height // 2
    return map_coordinates(image, [yy, xx], order=1, mode="constant", cval=0.0)


def fit_nuisance(
    predictions: list[np.ndarray], measurements: list[np.ndarray],
    flat_grid: np.ndarray,
) -> tuple[float, float]:
    features = [
        np.stack((prediction.ravel(), flat_grid.ravel()), axis=1)
        for prediction in predictions
    ]
    weights, _ = nnls(np.concatenate(features), np.concatenate(
        [measured.ravel() for measured in measurements]
    ))
    return float(weights[0]), float(weights[1])


def evaluate_run(run_dir: Path, coefficient_kind: str) -> dict:
    saved = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    keys = {field.name for field in fields(SlmModelInLoopConfig)} - {"callback"}
    config = SlmModelInLoopConfig(**{key: value for key, value in saved.items() if key in keys})
    manifest = json.loads((run_dir / "probes" / "manifest.json").read_text(encoding="utf-8"))
    geometry = json.loads((run_dir / "bench_geometry.json").read_text(encoding="utf-8"))
    amplitude = np.load(run_dir / "source_amplitude.npy", allow_pickle=False)
    fitted = np.load(run_dir / "fitted_coefficients.npy", allow_pickle=False)
    coefficients = np.zeros_like(fitted) if coefficient_kind == "zero" else fitted
    optimizer = _make_optimizer(config, coefficients, amplitude)
    flat_file = next(row["file"] for row in manifest if row["split"] == "flat")
    with np.load(run_dir / "probes" / flat_file, allow_pickle=False) as data:
        flat_frame = np.asarray(data["frame"], dtype=np.float64)
    center = zero_order_center(prepare_frame(flat_frame), refine=False)
    grid_size = int(geometry["far_field_size"])
    cell_scale = float(geometry["camera_px_per_model_px"])
    flat_grid = _to_model_grid(flat_frame, center, grid_size, cell_scale, 1)

    records = []
    for row in manifest:
        if row["split"] not in {"train", "holdout"}:
            continue
        with np.load(run_dir / "probes" / row["file"], allow_pickle=False) as data:
            phase = np.asarray(data["phase"], dtype=np.float64)
            measured = _to_model_grid(data["frame"], center, grid_size, cell_scale, 1)
        records.append({
            "file": row["file"], "split": row["split"], "family": row["family"],
            "exposure_ms": row["exposure_ms"],
            "prediction": optimizer.forward_intensity(coefficients, phase, amplitude),
            "measured": measured,
        })
    train = [row for row in records if row["split"] == "train"]
    holdout = [row for row in records if row["split"] == "holdout"]
    measured_train = [row["measured"] for row in train]

    def assess(parameters: np.ndarray) -> tuple[float, float, float]:
        warped = [warp_prediction(row["prediction"], parameters) for row in train]
        gain, leakage = fit_nuisance(warped, measured_train, flat_grid)
        correlations = [
            _prediction_correlation(gain * pred + leakage * flat_grid, measured)
            for pred, measured in zip(warped, measured_train)
        ]
        return float(np.mean(correlations)), gain, leakage

    identity = np.array([0.0, 0.0, 1.0])
    baseline_train, baseline_gain, baseline_leakage = assess(identity)
    result = minimize(
        lambda parameters: 1.0 - assess(parameters)[0]
        + 0.0005 * float(np.sum(((parameters - identity) / [3.0, 3.0, 0.15]) ** 2)),
        identity, method="Powell",
        bounds=[(-4.0, 4.0), (-4.0, 4.0), (0.8, 1.2)],
        options={"maxiter": 30, "xtol": 1e-3, "ftol": 1e-5},
    )
    calibrated = np.asarray(result.x, dtype=np.float64)
    fitted_train, gain, leakage = assess(calibrated)

    rows = []
    for row in holdout:
        original = baseline_gain * row["prediction"] + baseline_leakage * flat_grid
        corrected = gain * warp_prediction(row["prediction"], calibrated) + leakage * flat_grid
        rows.append({
            "file": row["file"], "family": row["family"],
            "exposure_ms": row["exposure_ms"],
            "baseline_correlation": _prediction_correlation(original, row["measured"]),
            "corrected_correlation": _prediction_correlation(corrected, row["measured"]),
        })
    return {
        "run_dir": str(run_dir), "coefficient_kind": coefficient_kind,
        "parameters_dx_dy_scale": calibrated.tolist(), "fit_success": bool(result.success),
        "train_correlation_baseline": baseline_train,
        "train_correlation_corrected": fitted_train,
        "baseline_gain_leakage": [baseline_gain, baseline_leakage],
        "corrected_gain_leakage": [gain, leakage], "holdouts": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    results = [
        evaluate_run(path, kind) for path in args.run_dirs
        for kind in ("zero", "short_refit")
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for item in results:
        rows = item["holdouts"]
        for family in ("all", "freeform"):
            subset = rows if family == "all" else [r for r in rows if r["family"] == family]
            before = np.mean([r["baseline_correlation"] for r in subset])
            after = np.mean([r["corrected_correlation"] for r in subset])
            print(item["run_dir"], item["coefficient_kind"], family,
                  f"{before:.4f} -> {after:.4f}", item["parameters_dx_dy_scale"])


if __name__ == "__main__":
    main()
