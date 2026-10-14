"""Compare low-order Step-A refits on archived, independent CCD holdouts."""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
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
from ao_shaping.utils.wavefront.zernike_calc import calc_n_zernike_terms


def evaluate_run(run_dir: Path, orders: tuple[int, ...]) -> dict:
    saved = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    keys = {field.name for field in fields(SlmModelInLoopConfig)} - {"callback"}
    base = SlmModelInLoopConfig(**{key: value for key, value in saved.items() if key in keys})
    manifest = json.loads((run_dir / "probes" / "manifest.json").read_text(encoding="utf-8"))
    geometry = json.loads((run_dir / "bench_geometry.json").read_text(encoding="utf-8"))
    source = np.load(run_dir / "source_amplitude.npy", allow_pickle=False)
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
    holdout = [row for row in records if row["split"] == "holdout"]
    frames = [(row["grid"], row["phase"]) for row in train]

    candidates = []
    for order in orders:
        config = replace(base, n_orders=order)
        initial = np.zeros(calc_n_zernike_terms(order), dtype=np.float64)
        for steps in (0, 10, 30):
            optimizer = _make_optimizer(config, initial, source)
            if steps:
                coefficients, _, _, optimizer = _fit_aberration_at_probes(
                    optimizer, frames, steps,
                    rearm=lambda current: _make_optimizer(config, current, source),
                )
            else:
                coefficients = initial.copy()
            coefficients, _, _ = trust_region_clamp(
                coefficients, initial, config.trust_region_c_l2
            )
            gain, leakage = _fit_flat_leakage(
                optimizer, coefficients, frames, source, flat_grid
            )
            scored = []
            for row in holdout:
                predicted = (
                    gain * optimizer.forward_intensity(coefficients, row["phase"], source)
                    + leakage * flat_grid
                )
                scored.append({
                    "file": row["file"], "family": row["family"],
                    "correlation": _prediction_correlation(predicted, row["grid"]),
                })
            candidates.append({
                "n_orders": order, "active_modes": len(initial) - len(config.frozen_modes),
                "steps": steps, "gain": gain, "leakage": leakage,
                "coefficient_l2": float(np.linalg.norm(coefficients)),
                "holdouts": scored,
            })
    return {"run_dir": str(run_dir), "candidates": candidates}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--orders", type=int, nargs="+", default=(3, 4, 6, 10))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    results = [evaluate_run(path, tuple(args.orders)) for path in args.run_dirs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for result in results:
        print(result["run_dir"])
        for candidate in result["candidates"]:
            rows = candidate["holdouts"]
            all_mean = np.mean([row["correlation"] for row in rows])
            free_mean = np.mean([row["correlation"] for row in rows
                                 if row["family"] == "freeform"])
            print(candidate["n_orders"], candidate["steps"],
                  f"all={all_mean:.4f}", f"freeform={free_mean:.4f}")


if __name__ == "__main__":
    main()
