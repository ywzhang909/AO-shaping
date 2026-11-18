"""Compare propagated subpixel SLM interpolation on saved hardware phases."""

from __future__ import annotations

import argparse
from loguru import logger
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom
from scipy.optimize import nnls

from ao_shaping.optimizer.wfless.slm_model_in_loop import (
    _prediction_correlation,
    _to_model_grid,
    prepare_frame,
)
from ao_shaping.utils.image.beam_metrics import zero_order_center


def propagated_intensity(
    phase: np.ndarray, amplitude: np.ndarray, output_size: int, factor: int,
) -> np.ndarray:
    """Bilinearly interpolate the commanded phase before the centred FFT."""
    if factor == 1:
        input_phase, input_amplitude = phase, amplitude
    else:
        input_phase = zoom(phase, factor, order=1)
        input_amplitude = zoom(amplitude, factor, order=1)
    field = input_amplitude * np.exp(1j * input_phase)
    padded_size = output_size * factor
    half = (padded_size - field.shape[0]) // 2
    field = np.pad(field, ((half, padded_size - field.shape[0] - half),) * 2)
    far_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field), norm="ortho"))
    center = padded_size // 2
    half = output_size // 2
    window = far_field[center - half:center + half, center - half:center + half]
    return np.abs(window) ** 2


def evaluate_run(run_dir: Path, factors: tuple[int, ...]) -> dict:
    manifest = json.loads((run_dir / "probes" / "manifest.json").read_text(encoding="utf-8"))
    geometry = json.loads((run_dir / "bench_geometry.json").read_text(encoding="utf-8"))
    amplitude = np.load(run_dir / "source_amplitude.npy", allow_pickle=False)
    flat_file = next(row["file"] for row in manifest if row["split"] == "flat")
    with np.load(run_dir / "probes" / flat_file, allow_pickle=False) as data:
        flat_frame = np.asarray(data["frame"], dtype=np.float64)
    center = zero_order_center(prepare_frame(flat_frame), refine=False)
    size = int(geometry["far_field_size"])
    scale = float(geometry["camera_px_per_model_px"])
    flat_grid = _to_model_grid(flat_frame, center, size, scale, 1)

    records = []
    for row in manifest:
        if row["split"] not in {"train", "holdout", "shaped"}:
            continue
        with np.load(run_dir / "probes" / row["file"], allow_pickle=False) as data:
            records.append({
                "file": row["file"], "split": row["split"], "family": row["family"],
                "phase": np.asarray(data["phase"], dtype=np.float64),
                "grid": _to_model_grid(data["frame"], center, size, scale, 1),
            })
    train = [row for row in records if row["split"] == "train"]
    holdout = [row for row in records if row["split"] in {"holdout", "shaped"}]
    candidates = []
    for factor in factors:
        for row in records:
            row["predicted"] = propagated_intensity(row["phase"], amplitude, size, factor)
        features = np.concatenate([
            np.stack((row["predicted"].ravel(), flat_grid.ravel()), axis=1)
            for row in train
        ])
        target = np.concatenate([row["grid"].ravel() for row in train])
        weights, _ = nnls(features, target)
        scored = []
        for row in holdout:
            prediction = weights[0] * row["predicted"] + weights[1] * flat_grid
            scored.append({
                "file": row["file"], "family": row["family"], "split": row["split"],
                "correlation": _prediction_correlation(prediction, row["grid"]),
            })
        candidates.append({
            "factor": factor, "gain": float(weights[0]),
            "leakage": float(weights[1]), "results": scored,
        })
    return {"run_dir": str(run_dir), "candidates": candidates}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--factors", nargs="+", type=int, default=(1, 2, 4))
    args = parser.parse_args()
    results = [evaluate_run(path, tuple(args.factors)) for path in args.run_dirs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for result in results:
        logger.info(result["run_dir"])
        for candidate in result["candidates"]:
            rows = candidate["results"]
            for family in ("holdout", "freeform", "shaped"):
                subset = [row for row in rows if row["split"] == family]
                if family == "freeform":
                    subset = [row for row in rows if row["family"] == family]
                mean = np.mean([row["correlation"] for row in subset])
                logger.info(candidate["factor"], family, f"{mean:.4f}")


if __name__ == "__main__":
    main()
