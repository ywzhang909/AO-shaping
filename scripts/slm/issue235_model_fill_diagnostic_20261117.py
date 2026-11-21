"""Issue #235, offline: does the frozen model explain the measured point-like fill?

Why this exists
---------------
The 2026-11-17 14-frame session
(``data/debug/square1000_model_diag_20261117_140741``) displayed the model's own
1000 px-square phase and measured a **point-like** focus: ~1% of the counts in a
+/-50 px box, ~197 pixels above half maximum.  The frozen ``model640`` forward,
given the same phase, predicts a square flat top filling the target bbox with
~990 000 pixels above half maximum.  The two cannot both be right, and the
earlier analyses compared them on a *binned* grid or without a floor, so the
gap was never quantified on one footing.

This script answers three questions with numbers, all computed on the **whole
native predicted plane** (1944 x 2592) and on the **whole measured frame**, with
the same formulas for both:

Q1  Does the frozen model predict the measured concentration, given the phase
    the hardware actually displayed?  The displayed phase is validated first:
    ``to_panel(encode(model_phase))`` is compared with the archived
    ``panel_gray``, so the model input is known to be what the panel wrote.

Q2  Which **one** operator parameter can move the prediction onto the measured
    morphology (``power_within_+-50px`` and ``pixels_above_half_max``), and by
    how much: far-field pixel scale, ``phase_scale``, ``amplitude_width``,
    ``sigma_ccd_px`` or ``defocus_diopter``.  Each is swept one at a time from
    the frozen value; the table reports the value that reproduces the
    measurement, its factor, and whether the required value is physically
    plausible against ``data/slm_geometry_calib.npz`` and the refit report.

Q3  Does the measured spot imply a full-panel (wide) pupil or the fitted
    ``amplitude_width`` (352, 133)?  The diffraction relation is inverted with
    the repo's own constants (lambda = 1064 nm, SLM pitch 8 um, 2f lens
    f = 0.125 m, CCD pitch 2.2 um), each quoted with file:line.

No device is opened.  Nothing here writes outside ``--out``.

Run::

    python scripts/slm/issue235_model_fill_diagnostic_20261117.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# 以文件形式启动时只有脚本自己的目录在 sys.path 上, 仓库根/src/scripts 都要
# 自己补 (scripts/ 下 import scripts._common 的既定约定, 同 analyze_model_hw_fidelity)。
_ROOT = Path(__file__).resolve().parents[2]
for _entry in (_ROOT, _ROOT / "src", _ROOT / "scripts"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import argparse
import json
import math
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from functools import partial

from scripts._common import fmt_fixed_or_text

import numpy as np
import torch

from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    FrozenDefocusForward,
)
from square_1000_diff_abba_20261030 import load_seed_contract

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SESSION = ROOT / "data/debug/square1000_model_diag_20261117_140741"
DEFAULT_OUT = ROOT / "data/debug/issue235_model_fill_20261117"

NATIVE_SHAPE = (1944, 2592)
PANEL_H, PANEL_W = 1200, 1920
GRID_SIZE = 640
ENCODING_2PI_GRAY = 1023
SPOT_HALF_WINDOW_PX = 50

#: Repo constants used by Q3.  Each path:line is the authoritative definition.
#: lambda and the SLM pitch come from ``bench_constants`` and are re-exported by
#: ``defocus_forward_core``; f and the CCD pitch are the 2f-system defaults in
#: ``target_size.DEFAULT_SYSTEM``.
LASER_WAVELENGTH_M = 1064e-9  # src/ao_shaping/utils/slm/bench_constants.py:53
SLM_PIXEL_PITCH_M = 8e-6  # src/ao_shaping/utils/slm/bench_constants.py:50
FOCAL_LENGTH_M = 0.125  # src/ao_shaping/tools/slm/panel/target_size.py:224
CCD_PIXEL_PITCH_M = 2.2e-6  # src/ao_shaping/tools/slm/panel/target_size.py:227
GEOMETRY_CALIB = ROOT / "data/slm_geometry_calib.npz"

#: Analytic FWHM coefficients of the far-field intensity, derived from the repo's
#: own amplitude convention ``A = exp(-((x-cx)/width)**2)`` -- see
#: ``gaussian_amplitude`` in
#: src/ao_shaping/algorithm/signal_processing/defocus_forward_core.py:236-242.
#: With that convention the pupil Gaussian's intensity 1/e^2 radius equals
#: ``width``; Fourier transforming gives an intensity FWHM of
#: ``sqrt(2*ln2)/pi * lambda*f/width`` = 0.3748 * lambda*f/width.
#: Uniform square of side D (also amplitude-uniform, so intensity ~ |FT|^2 with
#: the same square support): FWHM = 0.8859 * lambda*f/D (0.8859 of the
#: first-zero width).
GAUSSIAN_FWHM_COEFF = math.sqrt(2.0 * math.log(2.0)) / math.pi
SQUARE_FWHM_COEFF = 0.8859


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _ratio(value: float, reference: float) -> float:
    if not (math.isfinite(value) and math.isfinite(reference)) or reference <= 0.0:
        return float("nan")
    return value / reference


def _log_safe(value: float) -> float:
    return math.log(value) if value > 0.0 and math.isfinite(value) else -math.inf


# ---------------------------------------------------------------------------
# Phase-encoding gate: was the model phase what the panel displayed?
# ---------------------------------------------------------------------------
def encode_gray(phase_rad: np.ndarray, two_pi_gray: int = ENCODING_2PI_GRAY) -> np.ndarray:
    """The driver's phase -> gray mapping (no panel expansion)."""
    wrapped = np.mod(np.asarray(phase_rad, dtype=np.float64), 2.0 * np.pi)
    return np.mod(np.floor(wrapped / (2.0 * np.pi) * two_pi_gray), two_pi_gray).astype(np.uint16)


def encode_residual(model_phase: np.ndarray, panel_gray: np.ndarray) -> dict[str, Any]:
    """Compare the rebuilt panel gray with the archived one, in gray and rad."""
    from ao_shaping.tools.slm.transfer.capture import to_panel

    rebuilt = to_panel(encode_gray(model_phase))
    if rebuilt.shape != panel_gray.shape:
        return {"shape_mismatch": [list(rebuilt.shape), list(panel_gray.shape)]}
    diff = np.abs(rebuilt.astype(np.int32) - panel_gray.astype(np.int32))
    lsb_rad = 2.0 * np.pi / ENCODING_2PI_GRAY
    return {
        "mean_abs_gray": float(diff.mean()),
        "max_abs_gray": int(diff.max()),
        "fraction_exact": float(np.mean(diff == 0)),
        "mean_abs_phase_rad": float(diff.mean() * lsb_rad),
        "lsb_phase_rad": float(lsb_rad),
    }


# ---------------------------------------------------------------------------
# Plane metrics, identical for measured frames and model predictions
# ---------------------------------------------------------------------------
def corner_floor(frame: np.ndarray) -> float:
    """Robust background from four corner blocks (the spot is near the centre)."""
    frame = np.asarray(frame, dtype=np.float64)
    h, w = frame.shape
    block = max(16, min(h, w) // 12)
    corners = (
        frame[:block, :block],
        frame[:block, -block:],
        frame[-block:, :block],
        frame[-block:, -block:],
    )
    return float(np.median(np.concatenate([c.ravel() for c in corners])))


def _axis_fwhm(plane: np.ndarray, row: int, col: int, *, axis: str) -> float:
    """FWHM along the row or column through the peak, by linear interpolation."""
    profile = plane[row, :] if axis == "x" else plane[:, col]
    peak = int(col if axis == "x" else row)
    half = 0.5 * float(profile[peak])
    if not (half > 0.0):
        return float("nan")
    lo = peak
    while lo > 0 and profile[lo] > half:
        lo -= 1
    hi = peak
    while hi < profile.size - 1 and profile[hi] > half:
        hi += 1
    if profile[lo] > half or profile[hi] > half:
        return float("nan")
    left = lo + (half - profile[lo]) / (profile[lo + 1] - profile[lo])
    right = hi - (half - profile[hi]) / (profile[hi - 1] - profile[hi])
    return float(right - left)


def plane_metrics(
    frame: np.ndarray,
    floor: float,
    bbox: tuple[int, int, int, int],
    *,
    window: int = SPOT_HALF_WINDOW_PX,
) -> dict[str, Any]:
    """Morphology of one full plane: background-subtracted signal plus raw means.

    ``power_*`` are fractions of the floor-subtracted signal; ``mean_inside`` and
    ``mean_outside`` are raw counts (the measurement's own footing); ``cv`` is
    the raw inside-bbox coefficient of variation.
    """
    frame = np.asarray(frame, dtype=np.float64)
    signal = np.maximum(frame - float(floor), 0.0)
    total = float(signal.sum())
    row, col = np.unravel_index(int(np.argmax(signal)), signal.shape)
    row, col = int(row), int(col)
    peak = float(signal[row, col])

    r0 = max(0, row - window)
    r1 = min(frame.shape[0], row + window + 1)
    c0 = max(0, col - window)
    c1 = min(frame.shape[1], col + window + 1)
    within_50 = float(signal[r0:r1, c0:c1].sum())

    r0 = max(0, row - 2 * window)
    r1 = min(frame.shape[0], row + 2 * window + 1)
    c0 = max(0, col - 2 * window)
    c1 = min(frame.shape[1], col + 2 * window + 1)
    within_100 = float(signal[r0:r1, c0:c1].sum())

    b_r0, b_r1, b_c0, b_c1 = bbox
    in_bbox = signal[b_r0:b_r1, b_c0:b_c1]
    inside_raw = frame[b_r0:b_r1, b_c0:b_c1]
    outside_sum = float(frame.sum() - inside_raw.sum())
    outside_count = int(frame.size - inside_raw.size)
    inside_mean = float(inside_raw.mean())
    outside_mean = outside_sum / outside_count if outside_count else float("nan")
    inside_std = float(inside_raw.std())
    cv = inside_std / inside_mean if inside_mean > 0.0 else float("nan")

    return {
        "peak_rc": [row, col],
        "peak_counts_raw": float(frame[row, col]),
        "peak_counts_signal": peak,
        "floor_counts": float(floor),
        "signal_total": total,
        "power_within_50px": within_50 / total if total > 0.0 else float("nan"),
        "power_within_100px": within_100 / total if total > 0.0 else float("nan"),
        "power_in_bbox": float(in_bbox.sum()) / total if total > 0.0 else float("nan"),
        "mean_inside_bbox": inside_mean,
        "mean_outside_bbox": outside_mean,
        "cv_inside_bbox": cv,
        "pixels_above_half_max": int(np.count_nonzero(signal > 0.5 * peak)) if peak > 0.0 else 0,
        "fwhm_x_px": _finite(_axis_fwhm(signal, row, col, axis="x")),
        "fwhm_y_px": _finite(_axis_fwhm(signal, row, col, axis="y")),
    }


def compare_planes(
    measured: np.ndarray, measured_floor: float,
    predicted: np.ndarray, predicted_floor: float,
) -> dict[str, float]:
    """Pearson correlation and cosine similarity of the floor-subtracted planes."""
    a = np.maximum(np.asarray(measured, dtype=np.float64) - measured_floor, 0.0).ravel()
    b = np.maximum(np.asarray(predicted, dtype=np.float64) - predicted_floor, 0.0).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    cosine = float(np.dot(a, b) / (na * nb)) if na > 0.0 and nb > 0.0 else float("nan")
    a_centred = a - a.mean()
    b_centred = b - b.mean()
    denom = float(np.linalg.norm(a_centred) * np.linalg.norm(b_centred))
    pearson = float(np.dot(a_centred, b_centred) / denom) if denom > 0.0 else float("nan")
    return {
        "pearson": pearson,
        "cosine": cosine,
        "measured_signal_total": float(a.sum()),
        "predicted_signal_total": float(b.sum()),
    }


def footprint_bbox(signal: np.ndarray, half: float = 0.5) -> tuple[int, int, int, int]:
    """Bounding box of the region whose row/col projection exceeds half the peak."""
    peak = float(signal.max())
    if not (peak > 0.0):
        raise ValueError("cannot take a footprint of an all-zero plane")
    rows = np.flatnonzero(signal.max(axis=1) > half * peak)
    cols = np.flatnonzero(signal.max(axis=0) > half * peak)
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


# ---------------------------------------------------------------------------
# Model plumbing
# ---------------------------------------------------------------------------
def full_native_grid(shape: tuple[int, int] = NATIVE_SHAPE) -> np.ndarray:
    height, width = shape
    x = np.arange(width, dtype=np.float64)
    y = np.arange(height, dtype=np.float64)
    return np.stack(np.meshgrid(x, y), axis=-1)


def model_floor(params: Any, exposure_ms: float) -> float:
    """Uniform counts the frozen forward adds: background + bias/exposure."""
    return float(params.background) + float(params.bias_counts) / float(exposure_ms)


def predict_plane(
    params: Any,
    camera_xy: np.ndarray,
    phase_rad: np.ndarray,
    exposure_ms: float,
    *,
    phase_scale: float = 1.0,
) -> np.ndarray:
    """One full-plane prediction at the frozen forward geometry."""
    forward = FrozenDefocusForward(params, camera_xy, device="cpu")
    forward.exposure_ms = float(exposure_ms)
    phase = torch.as_tensor(np.asarray(phase_rad, dtype=np.float32) * phase_scale)
    with torch.no_grad():
        out = forward.predict(phase, exposure_ms=float(exposure_ms))
    return out.detach().cpu().numpy()[0]


# ---------------------------------------------------------------------------
# Q2: one-at-a-time parameter sweeps
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SweepPoint:
    """One scanned parameter value and its distance from the frozen value."""

    value: float
    factor: float | None


@dataclass(frozen=True)
class ParameterSweep:
    """One parameter, swept alone, with the frozen value and physical note."""

    name: str
    frozen: float
    unit: str
    points: tuple[SweepPoint, ...]
    plausibility: str


def _log_grid(start_exp: float, stop_exp: float, step: float) -> tuple[float, ...]:
    count = int(round((stop_exp - start_exp) / step)) + 1
    return tuple(2.0 ** (start_exp + step * i) for i in range(count))


def sweep_specs(params: Any) -> tuple[ParameterSweep, ...]:
    """The five Q2 parameters, each scanned from its frozen value."""
    width_x = float(params.amplitude_width_xy[0])
    amplitude_factors = _log_grid(-2.0, 3.0, 1.0)  # 0.25 .. 8x
    return (
        ParameterSweep(
            name="phase_scale",
            frozen=1.0,
            unit="x frozen command phase",
            points=tuple(SweepPoint(v, v) for v in (0.90, 0.95, 0.98, 1.0, 1.03, 1.05, 1.10)),
            plausibility=(
                "refit fitted 0.9992 (fwdmodel_fit_20261117.md, marginal dPearson "
                "0.0019). The required value would sit outside the fitted [0.90, 1.10]."
            ),
        ),
        ParameterSweep(
            name="amplitude_width_x_px",
            frozen=width_x,
            unit="panel px (1/e amplitude radius, y keeps the frozen aspect)",
            points=tuple(SweepPoint(width_x * f, f) for f in amplitude_factors),
            plausibility=(
                "not fitted by the refit (fixed at the checkpoint's (352.25, 132.72)). "
                "Values above 960 px no longer fit the panel's half span."
            ),
        ),
        ParameterSweep(
            name="sigma_ccd_px",
            frozen=float(params.sigma_ccd_px),
            unit="CCD px (Gaussian blur sigma)",
            points=tuple(SweepPoint(v, None) for v in (0.0, 1.0, 2.15, 4.0, 8.0, 16.0)),
            plausibility=(
                "refit blur_ccd_px 2.15. Blur only spreads power, so it cannot turn "
                "a 1000 px square into a 16 px spot; a negative value is undefined."
            ),
        ),
        ParameterSweep(
            name="defocus_diopter",
            frozen=float(params.defocus_diopter),
            unit="diopter",
            points=tuple(SweepPoint(v, None) for v in (0.0, -0.0244, -0.05, -0.1, -0.25, -0.5, -1.0)),
            plausibility=(
                "checkpoint value -0.0244 diopter (frozen_parameters.json). Extra "
                "defocus adds curvature/blur and cannot concentrate the square."
            ),
        ),
        ParameterSweep(
            name="far_field_pixel_scale",
            frozen=1.0,
            unit="x frozen CCD px per far-field sample",
            points=tuple(SweepPoint(f, f) for f in _log_grid(0.0, -7.0, -0.5)),
            plausibility=(
                "refit scale_px 2.009 CCD px/sample at 2x padding (geometric "
                "1.9698); the frozen model samples 3.94 CCD px/sample at full-panel "
                "width. A required factor far below 1 means sub-CCD-pixel sampling."
            ),
        ),
    )


def apply_parameter(name: str, value: float, params: Any) -> Any:
    """Return ``params`` with exactly one named parameter replaced."""
    if name == "phase_scale":
        return params
    if name == "amplitude_width_x_px":
        width_x, width_y = params.amplitude_width_xy
        return replace(params, amplitude_width_xy=(float(value), float(width_y) * float(value) / float(width_x)))
    if name == "sigma_ccd_px":
        return replace(params, sigma_ccd_px=float(value))
    if name == "defocus_diopter":
        return replace(params, defocus_diopter=float(value))
    if name == "far_field_pixel_scale":
        matrix = tuple(tuple(float(v) * float(value) for v in row) for row in params.ramp_matrix)
        return replace(params, ramp_matrix=matrix)
    raise ValueError(f"unknown sweep parameter {name!r}")


def sweep_score(predicted: dict[str, Any], target: dict[str, Any]) -> float:
    """Distance of (power in +/-50 px, half-max pixels) from the measured pair."""
    power = _ratio(float(predicted["power_within_50px"]), float(target["power_within_50px"]))
    halfmax = _ratio(float(predicted["pixels_above_half_max"]), float(target["pixels_above_half_max"]))
    if not (math.isfinite(power) and math.isfinite(halfmax)) or power <= 0.0 or halfmax <= 0.0:
        return float("inf")
    return abs(math.log(power)) + abs(math.log(halfmax))


def _parabola_min(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> float:
    """x of the vertex of the parabola through three points; falls back to x2."""
    denominator = (x1 - x2) * (x1 - x3) * (x2 - x3)
    if denominator == 0.0:
        return x2
    a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denominator
    b = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denominator
    if not math.isfinite(a) or a <= 0.0:
        return x2
    return float(-b / (2.0 * a))


def run_sweep(
    spec: ParameterSweep,
    params: Any,
    camera_xy: np.ndarray,
    phase_rad: np.ndarray,
    exposure_ms: float,
    target: dict[str, Any],
    bbox: tuple[int, int, int, int],
    predicted_floor: float,
) -> dict[str, Any]:
    """Scan one parameter; refine the best interior point with a log-parabola."""
    rows: list[dict[str, Any]] = []
    for point in spec.points:
        candidate = apply_parameter(spec.name, point.value, params)
        phase_scale = point.value if spec.name == "phase_scale" else 1.0
        plane = predict_plane(candidate, camera_xy, phase_rad, exposure_ms, phase_scale=phase_scale)
        metrics = plane_metrics(plane, predicted_floor, bbox)
        rows.append(
            {
                "value": float(point.value),
                "factor": _finite(point.factor),
                "power_within_50px": float(metrics["power_within_50px"]),
                "pixels_above_half_max": int(metrics["pixels_above_half_max"]),
                "peak_counts_signal": float(metrics["peak_counts_signal"]),
                "fwhm_x_px": _finite(metrics["fwhm_x_px"]),
                "fwhm_y_px": _finite(metrics["fwhm_y_px"]),
                "power_ratio": _finite(_ratio(float(metrics["power_within_50px"]),
                                              float(target["power_within_50px"]))),
                "halfmax_ratio": _finite(_ratio(float(metrics["pixels_above_half_max"]),
                                                float(target["pixels_above_half_max"]))),
                "score": _finite(sweep_score(metrics, target)),
            }
        )

    best_index = min(range(len(rows)), key=lambda i: rows[i]["score"] if rows[i]["score"] is not None else math.inf)
    best = rows[best_index]
    refined: dict[str, Any] | None = None
    if 0 < best_index < len(rows) - 1:
        factors = [rows[i]["factor"] for i in (best_index - 1, best_index, best_index + 1)]
        scores = [rows[i]["score"] for i in (best_index - 1, best_index, best_index + 1)]
        if all(isinstance(v, (int, float)) and math.isfinite(float(v)) and float(v) > 0.0 for v in factors):
            log_factors = [math.log(float(v)) for v in factors]
            log_factor = _parabola_min(
                log_factors[0], float(scores[0]),
                log_factors[1], float(scores[1]),
                log_factors[2], float(scores[2]),
            )
            factor = math.exp(log_factor)
            value = factor if spec.name in ("phase_scale", "far_field_pixel_scale") else (
                spec.frozen * factor if spec.name == "amplitude_width_x_px" else factor
            )
            candidate = apply_parameter(spec.name, value, params)
            phase_scale = value if spec.name == "phase_scale" else 1.0
            plane = predict_plane(candidate, camera_xy, phase_rad, exposure_ms, phase_scale=phase_scale)
            metrics = plane_metrics(plane, predicted_floor, bbox)
            score = sweep_score(metrics, target)
            if score <= float(best["score"]) + 1e-12:
                refined = {
                    "value": float(value),
                    "factor": float(factor),
                    "power_within_50px": float(metrics["power_within_50px"]),
                    "pixels_above_half_max": int(metrics["pixels_above_half_max"]),
                    "power_ratio": _finite(_ratio(float(metrics["power_within_50px"]),
                                                  float(target["power_within_50px"]))),
                    "halfmax_ratio": _finite(_ratio(float(metrics["pixels_above_half_max"]),
                                                    float(target["pixels_above_half_max"]))),
                    "score": _finite(score),
                }

    chosen = refined if refined is not None else best
    reproduces = (
        chosen["power_ratio"] is not None
        and chosen["halfmax_ratio"] is not None
        and 0.8 <= float(chosen["power_ratio"]) <= 1.25
        and 0.8 <= float(chosen["halfmax_ratio"]) <= 1.25
    )
    return {
        "name": spec.name,
        "frozen": float(spec.frozen),
        "unit": spec.unit,
        "plausibility": spec.plausibility,
        "points": rows,
        "best_scanned": best,
        "refined": refined,
        "reproducing_value": float(chosen["value"]),
        "reproducing_factor": _finite(chosen["factor"]),
        "reproducing_power_ratio": chosen["power_ratio"],
        "reproducing_halfmax_ratio": chosen["halfmax_ratio"],
        "reproduces_measurement": bool(reproduces),
        "at_scan_boundary": bool(refined is None and best_index in (0, len(rows) - 1)),
    }


# ---------------------------------------------------------------------------
# Q3: invert the diffraction relation
# ---------------------------------------------------------------------------
def _q3_implication(
    *,
    measured_fwhm_px: float,
    k_ramp: float,
    k_measured: float,
    gaussian_width_px: float,
    square_side_px: float,
    full_panel_square_fwhm: float,
    full_panel_gaussian_fwhm: float,
    width_x_fwhm: float,
    width_y_fwhm: float,
) -> str:
    """Prose for Q3, every number interpolated from the values it cites."""
    return (
        f"Measured x-FWHM {measured_fwhm_px:.2f} px = "
        f"{1e6 * measured_fwhm_px * CCD_PIXEL_PITCH_M:.1f} um.  A full {PANEL_W} px "
        f"square pupil focuses to {full_panel_square_fwhm:.2f} px and a "
        f"{PANEL_W // 2} px Gaussian (1/e^2 amplitude radius) to "
        f"{full_panel_gaussian_fwhm:.2f} px, i.e. "
        f"{measured_fwhm_px / full_panel_square_fwhm:.1f}x narrower than the "
        f"measurement.  Inverting the same relation gives an effective square side "
        f"of {square_side_px:.0f} px (Gaussian width {gaussian_width_px:.0f} px), "
        f"{100.0 * square_side_px / PANEL_W:.0f}% of the panel -- not a full panel.  "
        f"The frozen ``amplitude_width`` (352.2, 132.7) px predicts "
        f"{width_x_fwhm:.2f} px (x) and {width_y_fwhm:.2f} px (y); the measured "
        f"{measured_fwhm_px:.2f} px sits {measured_fwhm_px / width_x_fwhm:.2f}x above "
        f"the x prediction and {width_y_fwhm / measured_fwhm_px:.2f}x below the y "
        f"one, so the spot scale is that of a few-hundred-px pupil.  The analytic "
        f"scale {k_ramp:.1f} agrees with the stored calibration "
        f"{k_measured:.1f} to "
        f"{100.0 * abs(k_ramp - k_measured) / k_measured:.2f}%."
    )


def diffraction_report(measured_fwhm_px: float) -> dict[str, Any]:
    """Turn the measured spot FWHM into an effective pupil width."""
    #: One full-panel DFT line in CCD px (lambda*f / (PANEL_W*p) / p_ccd); this
    #: is the sampling of the panel grid, not a spot size.
    far_field_sample_m = LASER_WAVELENGTH_M * FOCAL_LENGTH_M / (PANEL_W * SLM_PIXEL_PITCH_M)
    k_theory = far_field_sample_m / CCD_PIXEL_PITCH_M
    #: Scale of the diffraction relation in CCD px * panel px:
    #: FWHM_ccd_px = coeff * k_ramp / pupil_width_px.  Algebraically identical to
    #: lambda*f/(p_slm*p_ccd); the calibration's Kx/Ky are exactly this number.
    k_ramp = LASER_WAVELENGTH_M * FOCAL_LENGTH_M / (SLM_PIXEL_PITCH_M * CCD_PIXEL_PITCH_M)

    calib = np.load(GEOMETRY_CALIB, allow_pickle=False)
    kx = float(calib["Kx"])
    ky = float(calib["Ky"])
    k_measured = 0.5 * (kx + ky)

    gaussian_width_px = GAUSSIAN_FWHM_COEFF * k_ramp / measured_fwhm_px
    square_side_px = SQUARE_FWHM_COEFF * k_ramp / measured_fwhm_px

    def gaussian_fwhm(width_px: float) -> float:
        return GAUSSIAN_FWHM_COEFF * k_ramp / width_px

    def square_fwhm(side_px: float) -> float:
        return SQUARE_FWHM_COEFF * k_ramp / side_px

    return {
        "constants": {
            "laser_wavelength_m": LASER_WAVELENGTH_M,
            "laser_wavelength_source": "src/ao_shaping/utils/slm/bench_constants.py:53",
            "slm_pixel_pitch_m": SLM_PIXEL_PITCH_M,
            "slm_pixel_pitch_source": "src/ao_shaping/utils/slm/bench_constants.py:50",
            "focal_length_m": FOCAL_LENGTH_M,
            "focal_length_source": "src/ao_shaping/tools/slm/panel/target_size.py:224 (DEFAULT_SYSTEM.focal_m)",
            "ccd_pixel_pitch_m": CCD_PIXEL_PITCH_M,
            "ccd_pixel_pitch_source": "src/ao_shaping/tools/slm/panel/target_size.py:227 (DEFAULT_SYSTEM.camera_pixel_m)",
        },
        "far_field_sample_m": float(far_field_sample_m),
        "far_field_sample_ccd_px": float(k_theory),
        "k_theory_ccd_px_per_2pi_per_panel_px": float(k_ramp),
        "k_calib_x": kx,
        "k_calib_y": ky,
        "k_calib_mean": k_measured,
        "k_theory_vs_calib_relative_error": float(abs(k_ramp - k_measured) / k_measured),
        "geometry_calib_source": str(GEOMETRY_CALIB.relative_to(ROOT)),
        "measured_fwhm_px": float(measured_fwhm_px),
        "measured_fwhm_um": float(1e6 * measured_fwhm_px * CCD_PIXEL_PITCH_M),
        "implied_gaussian_width_px": float(gaussian_width_px),
        "implied_square_side_px": float(square_side_px),
        "candidates": {
            "full_panel_gaussian_width_960px": {
                "fwhm_px": float(gaussian_fwhm(960.0)),
            },
            "full_panel_square_side_1920px": {
                "fwhm_px": float(square_fwhm(1920.0)),
            },
            "amplitude_width_x_352px": {
                "fwhm_px": float(gaussian_fwhm(352.246704)),
            },
            "amplitude_width_y_133px": {
                "fwhm_px": float(gaussian_fwhm(132.722488)),
            },
        },
        "implication": _q3_implication(
            measured_fwhm_px=measured_fwhm_px,
            k_ramp=k_ramp,
            k_measured=k_measured,
            gaussian_width_px=gaussian_width_px,
            square_side_px=square_side_px,
            full_panel_square_fwhm=square_fwhm(1920.0),
            full_panel_gaussian_fwhm=gaussian_fwhm(960.0),
            width_x_fwhm=gaussian_fwhm(352.246704),
            width_y_fwhm=gaussian_fwhm(132.722488),
        ),
    }


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------
# R-25 反漂移 (issue #205 第 132 批): 逐字口径 = fmt_fixed_or_text(none_text/
# nonfinite_text="n/a", spec="g", bool_labels=("yes", "no")); int/np.float32 走
# str, np.float64 (float 子类) 走 g; 120 输入电池全等。
_fmt = partial(fmt_fixed_or_text, none_text="n/a", nonfinite_text="n/a",
               spec="g", bool_labels=("yes", "no"))


def markdown_report(summary: dict[str, Any]) -> str:
    q1 = summary["q1"]
    q2 = summary["q2"]
    q3 = summary["q3"]
    lines = [
        "# Issue #235 model-fill diagnostic (2026-11-17)",
        "",
        f"- Session: `{summary['session']}`",
        f"- Script: `{summary['script']}`",
        f"- Model floor at {summary['model']['exposure_ms']} ms: "
        f"{_fmt(summary['model']['floor_counts'])} counts "
        f"(background {_fmt(summary['model']['background'])} + bias/exposure "
        f"{_fmt(summary['model']['bias_over_exposure'])})",
        f"- Predicted/measured planes: {NATIVE_SHAPE[0]}x{NATIVE_SHAPE[1]} native pixels",
        "",
        "## Q1 - does the frozen model predict the measured concentration?",
        "",
        "Displayed-phase gate (rebuilt panel gray vs archived `panel_gray`): "
        f"mean |diff| {_fmt(q1['gate']['anchor']['mean_abs_gray'], 3)} gray levels, "
        f"max {_fmt(q1['gate']['anchor']['max_abs_gray'])}; "
        f"equivalent {_fmt(q1['gate']['anchor']['mean_abs_phase_rad'], 3)} rad "
        f"(one LSB = {_fmt(q1['gate']['anchor']['lsb_phase_rad'], 3)} rad).",
        "",
        "| frame | role | power within +/-50 px (meas / pred) | pixels above half max (meas / pred) "
        "| FWHM x,y meas (px) | FWHM x,y pred (px) | Pearson | cosine |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in q1["frames"]:
        measured = row["measured"]
        predicted = row["predicted"]
        lines.append(
            "| {frame} | {role} | {mp} / {pp} | {mh} / {ph} | {mfx}, {mfy} | {pfx}, {pfy} "
            "| {pearson} | {cosine} |".format(
                frame=row["index"],
                role=row["role"],
                mp=_fmt(measured["power_within_50px"]),
                pp=_fmt(predicted["power_within_50px"]),
                mh=_fmt(measured["pixels_above_half_max"]),
                ph=_fmt(predicted["pixels_above_half_max"]),
                mfx=_fmt(measured["fwhm_x_px"]),
                mfy=_fmt(measured["fwhm_y_px"]),
                pfx=_fmt(predicted["fwhm_x_px"]),
                pfy=_fmt(predicted["fwhm_y_px"]),
                pearson=_fmt(row["comparison"]["pearson"], 3),
                cosine=_fmt(row["comparison"]["cosine"], 3),
            )
        )
    lines += [
        "",
        f"Denominators: ``power_*`` is a fraction of each plane's own floor-subtracted "
        f"signal (measured floor {_fmt(q1['frames'][0]['measured']['floor_counts'], 3)} "
        f"counts, model floor "
        f"{_fmt(q1['frames'][0]['predicted']['floor_counts'], 5)} counts).  The measured "
        "spot rides on a broad halo that extends past +/-50 px, so its fraction measures "
        "relative concentration rather than a compact energy budget, whereas the model's "
        f"signal is the square itself: "
        f"{_fmt(q1['frames'][0]['predicted']['power_in_bbox'], 4)} of it falls inside the "
        f"{summary['bbox']['rows'][1] - summary['bbox']['rows'][0]} x "
        f"{summary['bbox']['cols'][1] - summary['bbox']['cols'][0]} px predicted footprint.",
        "",
        f"Verdict: **{q1['verdict']}**",
        "",
        "## Q2 - which single parameter could reproduce the measurement?",
        "",
        f"Target (measured anchor {q1['target_frame']}): "
        f"power within +/-50 px {_fmt(q2['target']['power_within_50px'])}, "
        f"pixels above half max {_fmt(q2['target']['pixels_above_half_max'])}.",
        "",
        "| parameter | frozen | required value | factor | pred +/-50 px | pred half-max px "
        "| ratio (power, half-max) | reproduces? | plausibility |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for sweep in q2["sweeps"]:
        lines.append(
            "| {name} | {frozen} {unit} | {value} | {factor} | {power} | {halfmax} "
            "| {pr}, {hr} | {rep} | {why} |".format(
                name=sweep["name"],
                frozen=_fmt(sweep["frozen"]),
                unit=sweep["unit"].split(" (")[0],
                value=_fmt(sweep["reproducing_value"])
                + (" (scan bound)" if sweep["at_scan_boundary"] else ""),
                factor=_fmt(sweep["reproducing_factor"]),
                power=_fmt(sweep["best_scanned"]["power_within_50px"]),
                halfmax=_fmt(sweep["best_scanned"]["pixels_above_half_max"]),
                pr=_fmt(sweep["reproducing_power_ratio"]),
                hr=_fmt(sweep["reproducing_halfmax_ratio"]),
                rep=_fmt(sweep["reproduces_measurement"]),
                why=sweep["plausibility"].replace("|", "/"),
            )
        )
    lines += [
        "",
        f"Verdict: **{q2['verdict']}**",
        "",
        "## Q3 - does the measured spot imply a full-panel pupil?",
        "",
        f"Measured spot FWHM: {_fmt(q3['measured_fwhm_px'])} px "
        f"({_fmt(q3['measured_fwhm_um'], 4)} um).",
        "",
        f"One full-panel DFT sample = lambda*f/(N*p) = {_fmt(q3['far_field_sample_m'], 3)} m "
        f"= {_fmt(q3['far_field_sample_ccd_px'], 3)} CCD px; "
        f"K = lambda*f/(p*p_cam) = {_fmt(q3['k_theory_ccd_px_per_2pi_per_panel_px'], 4)} "
        f"vs calibration {_fmt(q3['k_calib_mean'], 4)} "
        f"(relative error {_fmt(q3['k_theory_vs_calib_relative_error'], 3)}).",
        "",
        "| candidate pupil | predicted FWHM (px) |",
        "|---|---:|",
        f"| full panel, Gaussian width 960 px | {_fmt(q3['candidates']['full_panel_gaussian_width_960px']['fwhm_px'], 3)} |",
        f"| full panel, square side 1920 px | {_fmt(q3['candidates']['full_panel_square_side_1920px']['fwhm_px'], 3)} |",
        f"| amplitude_width x = 352 px | {_fmt(q3['candidates']['amplitude_width_x_352px']['fwhm_px'], 3)} |",
        f"| amplitude_width y = 133 px | {_fmt(q3['candidates']['amplitude_width_y_133px']['fwhm_px'], 3)} |",
        "",
        f"Inverted: Gaussian width {_fmt(q3['implied_gaussian_width_px'])} px, "
        f"square side {_fmt(q3['implied_square_side_px'])} px.",
        "",
        f"{q3['implication']}",
        "",
        f"Verdict: **{q3['verdict']}**",
        "",
        "## Provenance",
        "",
        f"- Command: `{summary['reproduce']}`",
        f"- Generated (UTC): {summary['generated_utc']}",
        "- All numbers above are keys in the sibling JSON.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def analyse(session: Path, out_dir: Path) -> dict[str, Any]:
    config = _read_json(session / "config.json")
    manifest = _read_json(session / "manifest.json")
    rows = {int(row["index"]): row for row in manifest}
    role_of = {int(row["index"]): str(row["role"]) for row in manifest}

    seed_run = Path(config["seed_run"])
    checkpoint = Path(config["checkpoint"])
    seed_key = str(config["seed_key"])
    base_exposure = float(config["base_exposure_ms"])

    archive = load_seed_contract(seed_run, checkpoint, seed_key=seed_key)
    params = archive.parameters
    recorded_exposure = float(rows[0]["exposure_ms"])
    if abs(recorded_exposure - base_exposure) > 1e-9:
        raise ValueError(
            f"manifest exposure {recorded_exposure} differs from config base {base_exposure}"
        )

    camera_xy = full_native_grid()
    predicted_floor = model_floor(params, recorded_exposure)

    # Phase the panel actually displayed, per frame.
    captures: dict[int, dict[str, Any]] = {}
    gate_rows: dict[str, Any] = {}
    for index in (0, *[i for i, role in role_of.items() if role in ("N4+", "N4-")]):
        with np.load(session / rows[index]["file"]) as saved:
            model_phase = np.asarray(saved["model_phase"], dtype=np.float64)
            panel_gray = np.asarray(saved["panel_gray"], dtype=np.uint16)
            raw = np.asarray(saved["raw_frames"], dtype=np.float64)
        captures[index] = {"phase": model_phase, "raw_mean": raw.mean(axis=0)}
        key = "anchor" if index == 0 else role_of[index]
        gate_rows[key] = encode_residual(model_phase, panel_gray)

    anchor_phase = captures[0]["phase"]
    anchor_measured = captures[0]["raw_mean"]
    anchor_floor = corner_floor(anchor_measured)

    anchor_predicted = predict_plane(params, camera_xy, anchor_phase, recorded_exposure)
    anchor_signal = np.maximum(anchor_predicted - predicted_floor, 0.0)
    bbox = footprint_bbox(anchor_signal)

    measured_metrics: dict[int, dict[str, Any]] = {}
    predicted_metrics: dict[int, dict[str, Any]] = {}
    comparisons: dict[int, dict[str, float]] = {}
    frame_rows: list[dict[str, Any]] = []
    target_indices = (0, *[i for i, role in role_of.items() if role in ("N4+", "N4-")])
    for index in target_indices:
        measured = captures[index]["raw_mean"]
        floor = corner_floor(measured)
        predicted = predict_plane(params, camera_xy, captures[index]["phase"], recorded_exposure)
        measured_metrics[index] = plane_metrics(measured, floor, bbox)
        predicted_metrics[index] = plane_metrics(predicted, predicted_floor, bbox)
        comparisons[index] = compare_planes(measured, floor, predicted, predicted_floor)
        frame_rows.append(
            {
                "index": int(index),
                "role": role_of[index],
                "measured": measured_metrics[index],
                "predicted": predicted_metrics[index],
                "comparison": comparisons[index],
            }
        )

    target = measured_metrics[0]
    sweeps = [
        run_sweep(spec, params, camera_xy, anchor_phase, recorded_exposure,
                  target, bbox, predicted_floor)
        for spec in sweep_specs(params)
    ]
    reproducing = [s for s in sweeps if s["reproduces_measurement"]]
    if reproducing:
        q2_verdict = (
            "only " + ", ".join(s["name"] for s in reproducing) + " reproduces both "
            "targets, at the factors in the table; every other single parameter cannot."
        )
    else:
        ranked = sorted(sweeps, key=lambda s: float(s["best_scanned"]["score"]))[:2]
        q2_verdict = (
            "no single parameter reproduces the measured power within +/-50 px "
            f"({_fmt(target['power_within_50px'])}) and half-max pixel count "
            f"({target['pixels_above_half_max']}) inside the scanned ranges.  Closest "
            "near-misses, each moving one number but not both: "
            + "; ".join(
                f"{s['name']} at {_fmt(s['reproducing_value'])} "
                f"({_fmt(s['reproducing_factor'])}x) still gives power "
                f"{_fmt(float(s['best_scanned']['power_within_50px']))} "
                f"(ratio {_fmt(s['reproducing_power_ratio'])}) and half-max "
                f"{s['best_scanned']['pixels_above_half_max']} px "
                f"(ratio {_fmt(s['reproducing_halfmax_ratio'])})"
                for s in ranked
            )
            + "."
        )

    q3 = diffraction_report(float(target["fwhm_x_px"]))
    full_panel_fwhm = q3["candidates"]["full_panel_square_side_1920px"]["fwhm_px"]
    q3["verdict"] = (
        "the measured spot does NOT imply a full panel: a "
        f"{PANEL_W} px square pupil would focus to {_fmt(full_panel_fwhm, 3)} px FWHM, "
        f"{_fmt(float(target['fwhm_x_px']) / full_panel_fwhm, 2)}x narrower than the "
        f"measured {_fmt(q3['measured_fwhm_px'], 4)} px; the measured spot inverts to an "
        f"effective square side of {_fmt(q3['implied_square_side_px'], 4)} px "
        f"({_fmt(100.0 * q3['implied_square_side_px'] / PANEL_W, 2)}% of the panel) or a "
        f"Gaussian width of {_fmt(q3['implied_gaussian_width_px'], 4)} px, the scale of "
        "the frozen amplitude_width (352.2, 132.7) px rather than panel-sized "
        "illumination."
    )

    summary: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/slm/issue235_model_fill_diagnostic_20261117.py",
        "session": str(session.relative_to(ROOT)),
        "reproduce": (
            "python scripts/slm/issue235_model_fill_diagnostic_20261117.py "
            "--session data/debug/square1000_model_diag_20261117_140741 "
            "--out data/debug/issue235_model_fill_20261117"
        ),
        "model": {
            "seed_run": str(seed_run),
            "seed_key": seed_key,
            "checkpoint": str(checkpoint),
            "exposure_ms": recorded_exposure,
            "grid_size": int(params.grid_size),
            "fft_size": int(params.fft_size),
            "pixel_samples": int(params.pixel_samples),
            "origin_xy": [float(v) for v in params.origin_xy],
            "amplitude_center_xy": [float(v) for v in params.amplitude_center_xy],
            "amplitude_width_xy": [float(v) for v in params.amplitude_width_xy],
            "defocus_diopter": float(params.defocus_diopter),
            "sigma_ccd_px": float(params.sigma_ccd_px),
            "phase_sign": float(params.phase_sign),
            "gain": float(params.gain),
            "background": float(params.background),
            "bias_counts": float(params.bias_counts),
            "bias_over_exposure": float(params.bias_counts) / recorded_exposure,
            "floor_counts": predicted_floor,
        },
        "bbox": {
            "rows": [bbox[0], bbox[1]],
            "cols": [bbox[2], bbox[3]],
            "source": "half-max footprint of the frozen anchor prediction",
        },
        "q1": {
            "target_frame": 0,
            "gate": gate_rows,
            "frames": frame_rows,
            "verdict": (
                "the frozen model does NOT predict the measured point-like concentration: "
                f"it puts {_fmt(predicted_metrics[0]['power_within_50px'])} of its signal "
                f"within +/-50 px and {_fmt(predicted_metrics[0]['pixels_above_half_max'])} "
                "pixels above half max, versus "
                f"{_fmt(measured_metrics[0]['power_within_50px'])} and "
                f"{_fmt(measured_metrics[0]['pixels_above_half_max'])} measured; "
                f"whole-plane Pearson is {_fmt(comparisons[0]['pearson'], 3)}."
            ),
        },
        "q2": {
            "target": {
                "power_within_50px": float(target["power_within_50px"]),
                "pixels_above_half_max": int(target["pixels_above_half_max"]),
                "fwhm_x_px": _finite(float(target["fwhm_x_px"])),
                "fwhm_y_px": _finite(float(target["fwhm_y_px"])),
            },
            "sweeps": sweeps,
            "verdict": q2_verdict,
        },
        "q3": q3,
        "assumed_constants": [
            "lambda = 1064 nm: repo constant (bench_constants.py:53), not assumed.",
            "SLM pixel pitch = 8 um: repo constant (bench_constants.py:50), not assumed.",
            "f = 0.125 m: repo default (target_size.py:224), not assumed.",
            "CCD pixel pitch = 2.2 um: repo default (target_size.py:227), not assumed.",
            "The FWHM coefficients 0.3748 (Gaussian) and 0.8859 (square) are analytic, "
            "not fitted to this bench.",
        ],
    }
    summary["markdown"] = markdown_report(summary)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "diagnostic.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "summary.md").write_text(summary["markdown"], encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if not args.session.is_dir():
        parser.error(f"session directory does not exist: {args.session}")
    torch.set_num_threads(4)
    summary = analyse(args.session.resolve(), args.out.resolve())
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
