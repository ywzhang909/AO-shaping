"""Evaluate a saved hardware forward model on disjoint phase probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from ao_shaping.optimizer.wfless.slm_model_in_loop import (
    SlmModelInLoopConfig,
    _make_optimizer,
    _prediction_correlation,
    _to_model_grid,
)
from ao_shaping.utils.image.beam_metrics import zero_order_center


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir
    geometry = json.loads((run_dir / "bench_geometry.json").read_text(encoding="utf-8"))
    region = int(geometry["region"])
    far_field_size = int(geometry["far_field_size"])
    config = SlmModelInLoopConfig(
        region=region,
        far_field_padding=far_field_size // region,
        n_orders=10,
    )
    coefficients = np.load(run_dir / "fitted_coefficients.npy")
    amplitude = np.load(run_dir / "source_amplitude.npy")
    optimizer = _make_optimizer(config, coefficients, amplitude)
    records_dir = run_dir / "probes"
    flat = np.load(records_dir / "round_00_flat_00.npz")["frame"]
    center = zero_order_center(flat, refine=False)
    flat_grid = _to_model_grid(
        flat, center, far_field_size, geometry["camera_px_per_model_px"], 1
    )
    train_features = []
    train_targets = []
    for path in sorted(records_dir.glob("round_01_train_*.npz")):
        record = np.load(path)
        prediction = optimizer.forward_intensity(
            coefficients, record["phase"], amplitude
        )
        measured = _to_model_grid(
            record["frame"],
            center,
            far_field_size,
            geometry["camera_px_per_model_px"],
            1,
        )
        train_features.append(np.stack((prediction.ravel(), flat_grid.ravel()), axis=1))
        train_targets.append(measured.ravel())
    weights, _ = nnls(np.concatenate(train_features), np.concatenate(train_targets))
    print(f"train fit: predicted={weights[0]:.4f}, flat_leakage={weights[1]:.4f}")
    for path in sorted(records_dir.glob("round_01_*.npz")):
        if "train" in path.name:
            continue
        record = np.load(path)
        prediction = optimizer.forward_intensity(
            coefficients, record["phase"], amplitude
        )
        measured = record["model_grid"].astype(np.float64)
        corrected = weights[0] * prediction + weights[1] * flat_grid
        print(
            f"{path.stem}: direct={_prediction_correlation(prediction, measured):.4f} "
            f"corrected={_prediction_correlation(corrected, measured):.4f}"
        )


if __name__ == "__main__":
    main()
