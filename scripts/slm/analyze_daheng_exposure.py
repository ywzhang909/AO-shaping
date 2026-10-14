"""Analyse fixed-phase Daheng exposure sweeps without assuming linear counts.

The fitted curves describe an ROI integral for one fixed optical phase. They
must not be applied pixelwise or transferred to another phase without a true
dark frame and an independent validation sweep.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from ao_shaping.utils.image.beam_metrics import compute_square_metrics, prepare_frame


def response_fits(exposures: np.ndarray, signals: np.ndarray) -> dict:
    """Fit low-parameter candidates and test held-out exposure levels."""
    t = np.asarray(exposures, dtype=np.float64)
    y = np.asarray(signals, dtype=np.float64)
    if t.shape != y.shape or t.ndim != 1 or len(t) < 4:
        raise ValueError("need at least four matched exposure and signal values")
    if not np.isfinite(t).all() or not np.isfinite(y).all() or np.any(t <= 0):
        raise ValueError("exposure and signal must be finite; exposure must be positive")
    levels = np.unique(t)
    if len(levels) < 3:
        raise ValueError("need at least three distinct exposure levels")
    results = {}
    for degree, name in ((1, "affine"), (2, "quadratic")):
        if len(levels) <= degree + 1:
            continue
        coeff = np.polyfit(t, y, degree)
        prediction = np.polyval(coeff, t)
        cv_errors = []
        for level in levels:
            train = t != level
            if np.unique(t[train]).size <= degree:
                continue
            cv_coeff = np.polyfit(t[train], y[train], degree)
            cv_errors.extend((np.polyval(cv_coeff, t[~train]) - y[~train]).tolist())
        results[name] = {
            "coefficients_high_to_low": coeff.tolist(),
            "relative_rms": float(np.sqrt(np.mean((prediction - y) ** 2)) / y.mean()),
            "leave_one_level_out_relative_rms": (
                float(np.sqrt(np.mean(np.square(cv_errors))) / y.mean())
                if cv_errors else None
            ),
            "monotone_on_measured_range": bool(
                np.all(np.diff(np.polyval(coeff, np.linspace(t.min(), t.max(), 100))) > 0)
            ),
        }
        if degree == 1:
            results[name]["intercept_over_slope_ms"] = float(coeff[1] / coeff[0])
    means = np.array([y[t == level].mean() for level in levels])
    if len(levels) >= 3 and np.all(np.diff(means) > 0):
        lut = PchipInterpolator(levels, means)
        results["monotone_cubic_lut"] = {
            "exposures_ms": levels.tolist(),
            "mean_roi_signal": means.tolist(),
            "relative_rms_on_repeats": float(
                np.sqrt(np.mean((lut(t) - y) ** 2)) / y.mean()
            ),
            "status": "interpolates measured levels; no independent exposure-level validation",
        }
    results["double_exponential"] = {
        "status": "not fitted: independent levels and repeats insufficient for four-plus parameters"
    }
    results["logistic"] = {
        "status": "not fitted: independent levels and repeats insufficient for four-plus parameters"
    }
    return results


def pixelwise_response_diagnostic(
    exposures: np.ndarray, frames: list[np.ndarray], *, reference_ms: float | None = None,
) -> list[dict]:
    """Compare bright-pixel response at each level to one fixed-phase reference.

    This is a shape diagnostic, not a dark subtraction or a response inversion.
    The common mask excludes dim, quantized pixels and near-saturated pixels.
    """
    t = np.asarray(exposures, dtype=np.float64)
    stack = np.asarray(frames, dtype=np.float64)
    if stack.ndim != 3 or stack.shape[0] != len(t) or not np.isfinite(stack).all():
        raise ValueError("expected one finite 2-D frame for each exposure")
    levels = np.unique(t)
    reference_ms = float(levels[-1] if reference_ms is None else reference_ms)
    if reference_ms not in levels:
        raise ValueError("reference exposure must be measured")
    reference = stack[t == reference_ms].mean(axis=0)
    result = []
    for level in levels:
        image = stack[t == level].mean(axis=0)
        mask = (reference >= 12) & (reference < 240) & (image >= 12) & (image < 240)
        count = int(mask.sum())
        if count < 100:
            result.append({"exposure_ms": float(level), "bright_pixels": count,
                           "status": "insufficient common unsaturated bright pixels"})
            continue
        ratios = image[mask] / reference[mask]
        scale = float(np.median(ratios))
        residual = image[mask] / scale - reference[mask]
        result.append({
            "exposure_ms": float(level), "bright_pixels": count,
            "median_count_ratio_to_reference": scale,
            "ratio_p10_p90": np.percentile(ratios, [10, 90]).tolist(),
            "relative_shape_rmse_after_scalar": float(
                np.sqrt(np.mean(residual**2)) / np.sqrt(np.mean(reference[mask]**2))
            ),
            "bright_pixel_correlation": float(np.corrcoef(image[mask], reference[mask])[0, 1]),
        })
    return result


def analyze_sweep(root: Path) -> dict:
    rows = json.loads((root / "measurements.json").read_text(encoding="utf-8"))
    if not rows:
        raise ValueError(f"{root}: empty sweep")
    if any(not 0 < float(r["readback_ms"]) <= 0.4 for r in rows):
        raise ValueError(f"{root}: Daheng exposure outside (0, 0.4] ms")
    center = (int(rows[0]["center_x"]), int(rows[0]["center_y"]))
    frames = []
    raw_frames = []
    for row in rows:
        with np.load(root / f"{int(row['index']):02d}.npz") as record:
            frame = np.asarray(record["frame"], dtype=np.float64)
            raw = np.asarray(record["raw"])
        if raw.max() >= 250:
            raise ValueError(f"{root}: saturated raw frame {row['index']}")
        raw_frames.append(frame)
        cx, cy = center
        cropped = frame[cy - 128 : cy + 128, cx - 128 : cx + 128]
        if cropped.shape != (256, 256):
            raise ValueError(f"{root}: frozen camera window falls off sensor")
        frames.append(prepare_frame(cropped))
    reference = frames[int(np.argmax([r["readback_ms"] for r in rows]))]
    per_frame = []
    for row, image in zip(rows, frames):
        metric = compute_square_metrics(image, 100, (128.0, 128.0))
        per_frame.append({
            "index": int(row["index"]),
            "exposure_ms": float(row["readback_ms"]),
            "timestamp_local": row["time_local"],
            "raw_peak": float(row["raw_peak"]),
            "correlation_to_max_exposure": float(
                np.corrcoef(image.ravel(), reference.ravel())[0, 1]
            ),
            "cv": float(metric["uniformity_cv"]),
            "ee": float(metric["encircled_energy"]),
        })
    t = np.array([float(r["readback_ms"]) for r in rows])
    signal = np.array([float(r["signal_roi_sum"]) for r in rows])
    return {
        "source": str(root),
        "fixed_center_xy": list(center),
        "signal_definition": "recorded 192x192 ROI, background subtracted",
        "response_models": response_fits(t, signal),
        "pixelwise_response_diagnostic": pixelwise_response_diagnostic(t, raw_frames),
        "frames": per_frame,
        "dark_frame_available": False,
        "pixelwise_correction_applied": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("sweeps", type=Path, nargs="+")
    args = parser.parse_args()
    result = {"sweeps": [analyze_sweep(root) for root in args.sweeps]}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for sweep in result["sweeps"]:
        print(Path(sweep["source"]).name, sweep["response_models"]["affine"])


if __name__ == "__main__":
    main()
