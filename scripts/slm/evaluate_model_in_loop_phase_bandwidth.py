"""Test circular phase smoothing on frozen Santec/Daheng forward-model probes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from scripts.slm.evaluate_model_in_loop_psf import _score, fit_gain_leakage, load_run


SIGMAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def circular_phase_blur(phase: np.ndarray, sigma: float) -> np.ndarray:
    """Blur a phase command without treating 0 and 2*pi as different values."""
    phase = np.asarray(phase, dtype=np.float64)
    if not np.isfinite(phase).all() or not np.isfinite(sigma) or sigma < 0:
        raise ValueError("phase and sigma must be finite; sigma must be nonnegative")
    if sigma == 0:
        return phase.copy()
    phasor = np.exp(1j * phase)
    return np.angle(
        gaussian_filter(phasor.real, sigma, mode="reflect")
        + 1j * gaussian_filter(phasor.imag, sigma, mode="reflect")
    )


def evaluate_run(run_dir: Path, sigmas: tuple[float, ...] = SIGMAS) -> list[dict]:
    if not sigmas or 0.0 not in sigmas or any(not np.isfinite(s) or s < 0 for s in sigmas):
        raise ValueError("phase candidates must contain zero and be finite and nonnegative")
    optimizer, coefficients, amplitude, flat, observed_mask, samples = load_run(run_dir)
    train = [sample for sample in samples if sample[0]["split"] == "train"]
    holdout = [sample for sample in samples if sample[0]["split"] == "holdout"]
    if not train or not holdout:
        raise ValueError(f"{run_dir}: missing train or holdout probes")

    def forward(phase: np.ndarray, sigma: float) -> np.ndarray:
        return optimizer.forward_intensity(
            coefficients, circular_phase_blur(phase, sigma), amplitude,
        )

    candidates = []
    for sigma in sigmas:
        weights, error = fit_gain_leakage(
            [forward(phase, sigma) for _, phase, _ in train],
            [measured for _, _, measured in train], flat,
        )
        candidates.append((error, sigma, weights))
    _, selected_sigma, selected_weights = min(candidates, key=lambda row: row[0])
    baseline_weights = next(row[2] for row in candidates if row[1] == 0)
    rows = []
    for item, phase, measured in holdout:
        for method, sigma, weights in (
            ("baseline", 0.0, baseline_weights),
            ("train_selected_phase_bandwidth", selected_sigma, selected_weights),
        ):
            prediction = weights[0] * forward(phase, sigma) + weights[1] * flat
            correlation, error = _score(prediction, measured, observed_mask)
            rows.append({
                "run": run_dir.name, "family": item["family"],
                "index": int(item["index"]), "method": method,
                "sigma_model_px": sigma, "correlation": correlation,
                "shape_nrmse": error,
                "train_relative_error": next(row[0] for row in candidates if row[1] == sigma),
                "model_gain": float(weights[0]), "flat_leakage": float(weights[1]),
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
    for method in ("baseline", "train_selected_phase_bandwidth"):
        subset = [row for row in rows if row["method"] == method]
        print(method, "corr", np.mean([row["correlation"] for row in subset]),
              "nrmse", np.mean([row["shape_nrmse"] for row in subset]))


if __name__ == "__main__":
    main()
