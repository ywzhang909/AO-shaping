"""Compare zero and short-refit forward models on archived hardware holdouts.

Each candidate fits only gain and flat leakage on the run's training probes.
Holdout CCD grids are read from the recorder and never used for fitting.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import numpy as np

from ao_shaping.optimizer.wfless.slm_model_in_loop import (
    SlmModelInLoopConfig,
    _fit_flat_leakage,
    _make_optimizer,
    _prediction_correlation,
    _to_model_grid,
    prepare_frame,
)
from ao_shaping.utils.image.beam_metrics import zero_order_center


def evaluate_run(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "probes" / "manifest.json").read_text(encoding="utf-8"))
    saved_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    names = {field.name for field in fields(SlmModelInLoopConfig)} - {"callback"}
    config = SlmModelInLoopConfig(**{key: value for key, value in saved_config.items() if key in names})
    geometry = json.loads((run_dir / "bench_geometry.json").read_text(encoding="utf-8"))
    source = np.load(run_dir / "source_amplitude.npy", allow_pickle=False)
    fitted = np.load(run_dir / "fitted_coefficients.npy", allow_pickle=False)
    zero = np.zeros_like(fitted)
    entries = {row["split"]: row for row in manifest if row["split"] == "flat"}
    flat = np.load(run_dir / "probes" / entries["flat"]["file"], allow_pickle=False)
    flat_frame = np.asarray(flat["frame"], dtype=np.float64)
    center = zero_order_center(prepare_frame(flat_frame), refine=False)
    size = int(geometry["far_field_size"])
    scale = float(geometry["camera_px_per_model_px"])
    flat_grid = _to_model_grid(flat_frame, center, size, scale, 1)

    train = []
    for row in manifest:
        if row["split"] != "train":
            continue
        with np.load(run_dir / "probes" / row["file"], allow_pickle=False) as data:
            measured = _to_model_grid(data["frame"], center, size, scale, 1)
            train.append((measured, np.asarray(data["phase"], dtype=np.float64)))

    candidates = {}
    for label, coefficients in (("zero", zero), ("short_refit", fitted)):
        optimizer = _make_optimizer(config, coefficients, source)
        gain, leakage = _fit_flat_leakage(optimizer, coefficients, train, source, flat_grid)
        rows = []
        for row in manifest:
            if row["split"] != "holdout":
                continue
            with np.load(run_dir / "probes" / row["file"], allow_pickle=False) as data:
                predicted = optimizer.forward_intensity(coefficients, data["phase"], source)
                predicted = gain * predicted + leakage * flat_grid
                measured = np.asarray(data["model_grid"], dtype=np.float64)
                rows.append({
                    "file": row["file"], "family": row["family"],
                    "exposure_ms": row["exposure_ms"],
                    "correlation": _prediction_correlation(predicted, measured),
                })
        candidates[label] = {"gain": gain, "flat_leakage": leakage, "holdouts": rows}
    return {"run_dir": str(run_dir), "candidates": candidates}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    results = [evaluate_run(path) for path in args.run_dirs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for result in results:
        print(result["run_dir"])
        for name, candidate in result["candidates"].items():
            rows = candidate["holdouts"]
            mean = np.mean([row["correlation"] for row in rows])
            free = np.mean([row["correlation"] for row in rows if row["family"] == "freeform"])
            print(f"  {name}: all={mean:.4f}, freeform={free:.4f}")


if __name__ == "__main__":
    main()
