"""Cross-fit saved freeform CCD probes to test Step-A probe-family choice."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path

import numpy as np

from ao_shaping.optimizer.wfless.model_in_loop_shaping import _fit_aberration_at_probes
from ao_shaping.optimizer.wfless.slm_model_in_loop import (
    SlmModelInLoopConfig,
    _fit_flat_leakage,
    _make_optimizer,
    _prediction_correlation,
    _to_model_grid,
    prepare_frame,
    trust_region_clamp,
)
from ao_shaping.utils.image.beam_metrics import zero_order_center


def evaluate_run(run_dir: Path) -> dict:
    payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    keys = {field.name for field in fields(SlmModelInLoopConfig)} - {"callback"}
    config = SlmModelInLoopConfig(**{key: value for key, value in payload.items() if key in keys})
    manifest = json.loads((run_dir / "probes" / "manifest.json").read_text(encoding="utf-8"))
    geometry = json.loads((run_dir / "bench_geometry.json").read_text(encoding="utf-8"))
    source = np.load(run_dir / "source_amplitude.npy", allow_pickle=False)
    fitted = np.load(run_dir / "fitted_coefficients.npy", allow_pickle=False)
    flat_file = next(row["file"] for row in manifest if row["split"] == "flat")
    with np.load(run_dir / "probes" / flat_file, allow_pickle=False) as data:
        flat_frame = np.asarray(data["frame"], dtype=np.float64)
    center = zero_order_center(prepare_frame(flat_frame), refine=False)
    size = int(geometry["far_field_size"])
    scale = float(geometry["camera_px_per_model_px"])
    flat_grid = _to_model_grid(flat_frame, center, size, scale, 1)

    records = []
    for row in manifest:
        if row["split"] not in {"train", "holdout"}:
            continue
        with np.load(run_dir / "probes" / row["file"], allow_pickle=False) as data:
            records.append({
                "file": row["file"], "split": row["split"], "family": row["family"],
                "phase": np.asarray(data["phase"], dtype=np.float64),
                "grid": _to_model_grid(data["frame"], center, size, scale, 1),
            })
    train = [row for row in records if row["split"] == "train"]
    free = [row for row in records if row["family"] == "freeform"]
    if len(train) != 2 or len(free) != 2:
        raise ValueError("cross-fit requires two original training and two freeform holdout frames")

    results = []
    for test_index in range(2):
        extra_train = free[1 - test_index]
        test = free[test_index]
        for label, training, steps in (
            ("zero", train, 0),
            ("smooth_10", train, 10),
            ("mixed_10", [*train, extra_train], 10),
            ("mixed_30", [*train, extra_train], 30),
        ):
            optimizer = _make_optimizer(config, np.zeros_like(fitted), source)
            frames = [(row["grid"], row["phase"]) for row in training]
            if steps:
                coefficients, _, _, optimizer = _fit_aberration_at_probes(
                    optimizer, frames, steps,
                    rearm=lambda current: _make_optimizer(config, current, source),
                )
            else:
                coefficients = np.zeros_like(fitted)
            coefficients, _, _ = trust_region_clamp(
                coefficients, np.zeros_like(fitted), config.trust_region_c_l2
            )
            gain, leakage = _fit_flat_leakage(
                optimizer, coefficients, frames, source, flat_grid
            )
            prediction = (
                gain * optimizer.forward_intensity(coefficients, test["phase"], source)
                + leakage * flat_grid
            )
            results.append({
                "test_file": test["file"], "extra_train_file": extra_train["file"],
                "candidate": label, "steps": steps,
                "correlation": _prediction_correlation(prediction, test["grid"]),
                "gain": gain, "leakage": leakage,
                "coefficient_l2": float(np.linalg.norm(coefficients)),
            })
    return {"run_dir": str(run_dir), "folds": results}


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
        for label in ("zero", "smooth_10", "mixed_10", "mixed_30"):
            values = [row["correlation"] for row in result["folds"]
                      if row["candidate"] == label]
            print(f"  {label}: {values[0]:.4f}, {values[1]:.4f}, mean {np.mean(values):.4f}")


if __name__ == "__main__":
    main()
