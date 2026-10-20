"""Compare saved GS-refine phases with their paired CCD frames offline.

The FFT assumes the ideal Gaussian illumination and linear SLM phase response
used by the GS warm start. Its output is a diagnostic, not a calibrated camera
prediction. Only load trusted local debug pickles.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from ao_shaping.algorithm.signal_processing.square_shaping import SquareShapingConfig
from ao_shaping.drivers.sim.slm_shaping_bench import forward_intensity
from ao_shaping.utils.slm.gs_debug_optics import grayscale_to_radians
from ao_shaping.utils.io.recorder_keys import (
    RECORD_IMAGE,
    RECORD_PHASE,
)


def sample_camera(
    far: np.ndarray,
    *,
    far_pixel_m: float,
    camera_pixel_m: float,
    camera_shape: tuple[int, int],
    center_xy: tuple[float, float],
) -> np.ndarray:
    """Sample a centered FFT at CCD pitch around a fixed optical-axis pixel."""
    if far.ndim != 2 or far.shape[0] != far.shape[1]:
        raise ValueError("far must be a square 2D intensity image")
    if far_pixel_m <= 0 or camera_pixel_m <= 0:
        raise ValueError("pixel pitches must be positive")
    yy, xx = np.indices(camera_shape, dtype=np.float64)
    cx, cy = center_xy
    scale = camera_pixel_m / far_pixel_m
    origin = far.shape[0] // 2
    return map_coordinates(
        np.asarray(far, dtype=np.float64),
        [origin + (yy - cy) * scale, origin + (xx - cx) * scale],
        order=1,
        mode="constant",
        cval=0.0,
    )


def image_metrics(
    predicted: np.ndarray,
    measured: np.ndarray,
    *,
    center_xy: tuple[float, float],
    side: int,
) -> dict[str, float]:
    """Compare two frames at one fixed camera ROI after unit-sum scaling."""
    pred = np.asarray(predicted, dtype=np.float64)
    obs = np.asarray(measured, dtype=np.float64)
    if pred.shape != obs.shape or pred.ndim != 2:
        raise ValueError("predicted and measured must be same-shape 2D arrays")
    if not np.isfinite(pred).all() or not np.isfinite(obs).all():
        raise ValueError("images must be finite")
    if pred.min() < 0 or obs.min() < 0 or pred.sum() <= 0 or obs.sum() <= 0:
        raise ValueError("images must have positive, nonnegative intensity")
    pred = pred / pred.sum()
    obs = obs / obs.sum()
    cx, cy = center_xy
    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    if side < 1 or x0 < 0 or y0 < 0 or x0 + side > obs.shape[1] or y0 + side > obs.shape[0]:
        raise ValueError("target box leaves camera frame")
    mask = np.s_[y0 : y0 + side, x0 : x0 + side]
    pbox, obox = pred[mask], obs[mask]
    return {
        "pearson": float(np.corrcoef(pred.ravel(), obs.ravel())[0, 1]),
        "intensity_overlap": float(np.minimum(pred, obs).sum()),
        "predicted_pib": float(pbox.sum()),
        "measured_pib": float(obox.sum()),
        "predicted_cv": float(pbox.std() / pbox.mean()),
        "measured_cv": float(obox.std() / obox.mean()),
    }


def compare_run(
    directory: Path,
    *,
    pupil_center_xy: tuple[int, int],
    wavelength_nm: float = 1064.0,
    keys: tuple[int, ...] | None = None,
) -> dict:
    pickles = list(directory.glob("*.pkl"))
    if len(pickles) != 1 or not pickles[0].with_suffix(".json").is_file():
        raise ValueError("run directory must contain one PKL with a matching JSON sidecar")
    config = json.loads(pickles[0].with_suffix(".json").read_text(encoding="utf-8"))
    with pickles[0].open("rb") as stream:
        rows = pickle.load(stream)  # noqa: S301 - trusted local hardware record
    radius = float(config["beam_radius_px"])
    n = int(round(2 * radius))
    if n < 2 or abs(n - 2 * radius) > 1e-9:
        raise ValueError("beam radius must be an integer or half-integer number of pixels")
    actual_wavelength_nm = float(config["slm_wavelength"] or wavelength_nm)
    if actual_wavelength_nm <= 0:
        raise ValueError("wavelength must be positive")
    cfg = SquareShapingConfig(
        n_grid=n,
        aperture_size=n * float(config["panel_pixel_um"]) * 1e-6,
        wavelength=actual_wavelength_nm * 1e-9,
        focal_length=float(config["focal_length_m"]),
        far_field_padding=int(config["far_field_padding"]),
    )
    cx, cy = pupil_center_xy
    x0, y0 = int(round(cx - radius)), int(round(cy - radius))
    selected = sorted(rows) if keys is None else list(keys)
    results = []
    for key in selected:
        row = rows[key]
        measured = np.asarray(row[RECORD_IMAGE])
        gray = np.asarray(row[RECORD_PHASE])
        pupil_gray = gray[y0 : y0 + n, x0 : x0 + n]
        if pupil_gray.shape != (n, n):
            raise ValueError(f"pupil for record {key} leaves SLM panel")
        far = forward_intensity(grayscale_to_radians(pupil_gray), cfg)
        center = (float(rows[min(rows)]["cx"]), float(rows[min(rows)]["cy"]))
        predicted = sample_camera(
            far,
            far_pixel_m=cfg.far_field_pixel_size,
            camera_pixel_m=float(config["camera_pixel_um"]) * 1e-6,
            camera_shape=measured.shape,
            center_xy=center,
        )
        metrics = image_metrics(
            predicted, measured, center_xy=center, side=int(row["target_side"])
        )
        results.append({"key": key, "score": float(row["score"]), **metrics})
    return {
        "run": str(directory),
        "pupil_center_xy": list(pupil_center_xy),
        "wavelength_nm": actual_wavelength_nm,
        "assumptions": "linear gray phase, ideal Gaussian pupil, FFT, fixed CCD center; no fitted calibration",
        "records": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="GS refine debug run directory")
    parser.add_argument("--pupil-center", required=True, help="SLM x,y pixels")
    parser.add_argument("--wavelength-nm", type=float, default=1064.0, help="used when run JSON has slm_wavelength=0")
    parser.add_argument("--keys", type=int, nargs="*", help="record keys (default: all)")
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    args = parser.parse_args()
    try:
        center = tuple(int(part) for part in args.pupil_center.split(","))
        if len(center) != 2:
            raise ValueError
    except ValueError:
        parser.error("--pupil-center must be x,y in SLM pixels")
    report = compare_run(
        args.run,
        pupil_center_xy=center,
        wavelength_nm=args.wavelength_nm,
        keys=tuple(args.keys) if args.keys else None,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
