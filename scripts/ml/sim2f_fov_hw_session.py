"""Resumable one-camera full-frame session for the sim2f FOV work.

The script separates the three things a hardware FOV measurement must not mix:

1. **plan** -- deterministic, offline.  It writes the commands (calibration
   screens, the +/- Zernike probe bank, and the shape family) as ``.npy`` phases
   plus a schema-versioned ``plan.json``.  No wall clock enters the plan, so two
   runs with the same arguments produce byte-identical files.
2. **capture** -- one SLM and one camera, opened once.  Every command is read as
   **one native full-sensor frame**; ROI choice happens later, offline.  The
   manifest is rewritten atomically after each item, so Ctrl-C leaves a session
   that resumes by skipping valid frames and re-capturing missing/corrupt ones.
3. **offline stages** -- ``finalize`` crops the native frames to the square
   analysis window and writes the repository's standard ``session.json`` +
   ``process_data/`` recording; ``calibrate`` fits a bench state to that
   recording; ``analyze`` scores a full-frame prediction.  None of these opens a
   device.

``--dry-run`` replaces the devices with :class:`~ml.sim2f.bench.SimInstrument`
and exercises the same capture/resume/finalize code path.  It is a plumbing
check, not a statement about the real bench.

Usage::

    python scripts/ml/sim2f_fov_hw_session.py plan --out runs/sim2f_fov
    python scripts/ml/sim2f_fov_hw_session.py capture --session runs/sim2f_fov --dry-run
    python scripts/ml/sim2f_fov_hw_session.py finalize --session runs/sim2f_fov
    python scripts/ml/sim2f_fov_hw_session.py calibrate --session runs/sim2f_fov --out runs/sim2f_fov/calibration
    python scripts/ml/sim2f_fov_hw_session.py analyze --session runs/sim2f_fov --out runs/sim2f_fov/analysis
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from ml.sim2f.bench import (
    HwInstrument,
    SimInstrument,
    recorded_grayscale,
)
from ml.sim2f.calibrate import (
    CalibratedBench,
    calibrate_bench,
    load_bench_session,
    load_calibration,
    save_calibration,
)
from ml.sim2f.config import (
    AberrationConfig,
    CalibrateConfig,
    Sim2fConfig,
)
from ml.sim2f.extrapolate import frame_np, moment_metrics
from ml.sim2f.fit import probe
from ml.sim2f.losses import prepare_frame
from ml.sim2f.optics import Sim2f
from ml.sim2f.recorder_io import BenchRecorder
from ml.sim2f.window import CameraWindow

PLAN_SCHEMA = "sim2f_fov_hw_session/plan/v1"
MANIFEST_SCHEMA = "sim2f_fov_hw_session/manifest/v1"
SESSION_SCHEMA = "sim2f_fov_hw_session/session/v1"
SUMMARY_SCHEMA = "sim2f_fov_hw_session/analyze/v1"

PLAN_NAME = "plan.json"
MANIFEST_NAME = "capture_manifest.json"
SESSION_NAME = "session.json"
SUMMARY_NAME = "summary.json"
PHASE_DIR = "phases"
FRAME_DIR = "frames"
PANEL_DIR = "panel"
PROCESS_DIR = "process_data"

TARGETS: tuple[str, ...] = ("ring", "cross", "frame", "two_spot", "two_spot_wide")
PROBE_NOLL: tuple[int, ...] = tuple(range(4, 12))
PROBE_AMPLITUDE_RAD = 0.35
CALIB_SCREENS = 24
LOW_ORDER_NOLL: tuple[int, ...] = tuple(range(2, 12))
GS_ITERS = 48
FIT_ITERS = 180
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Small, shared utilities
# ---------------------------------------------------------------------------
def _json_write(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically: a reader sees the old file or the whole new one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def _json_read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _array_sha256(array: np.ndarray) -> str:
    data = np.ascontiguousarray(array, dtype=np.float32)
    return hashlib.sha256(data.tobytes()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp.npy")
    np.save(temp, array)
    os.replace(temp, path)


def _load_npy(path: Path) -> np.ndarray:
    return np.load(path)


def _crop_square(
    frame: np.ndarray, center_xy: tuple[int, int], size: int
) -> np.ndarray:
    """Crop a square window, clamped so it always lies inside the frame."""
    height, width = frame.shape
    if size > min(height, width):
        raise ValueError(
            f"window_px={size} does not fit a {height}x{width} native frame"
        )
    cx, cy = int(center_xy[0]), int(center_xy[1])
    left = min(max(cx - size // 2, 0), width - size)
    top = min(max(cy - size // 2, 0), height - size)
    return np.ascontiguousarray(frame[top : top + size, left : left + size])


def _round_window(value: int) -> int:
    if value <= 0 or value % 4:
        raise ValueError(f"window_px must be a positive multiple of 4, got {value}")
    return int(value)


def _crop_layout(
    native_shape: tuple[int, int],
    zero_order_xy: tuple[int, int],
    full_px: int,
) -> tuple[int, int, tuple[int, int]]:
    """Return the native crop origin and the zero-order's in-crop position.

    The prediction is a ``full_px`` square centred on the zero-order.  The
    measured native frame is cropped to the same square, but the crop is
    clamped to the sensor, so next to an edge the recorded zero-order can sit
    away from the square centre.  The returned ``sq_center`` is that position
    in cropped pixel coordinates, i.e. the target for aligning the prediction.
    """
    height, width = int(native_shape[0]), int(native_shape[1])
    cx, cy = int(zero_order_xy[0]), int(zero_order_xy[1])
    half = full_px // 2
    col0 = max(0, min(cx - half, width - full_px))
    row0 = max(0, min(cy - half, height - full_px))
    return col0, row0, (cx - col0, cy - row0)


def _zero_order_centroid_xy(frame: np.ndarray) -> tuple[float, float] | None:
    """``argmax`` + 3x3 weighted centroid of the zero-order, in native ``(x, y)``.

    This repeats the arithmetic of ``ml.sim2f.bench._zero_order_centroid``
    locally on purpose: that helper is private to the library and the session
    script must not depend on library internals just for a diagnostic.  On a
    flat frame the zero-order is the global maximum; the centroid of the 3x3
    around it recovers the sub-pixel position.  Returns ``None`` when that
    window carries no positive weight (for example an all-zero frame), which is
    a measurement that does not exist rather than a zero offset.
    """
    data = np.asarray(frame, dtype=np.float64)
    if data.ndim != 2 or data.size == 0:
        return None
    row, col = np.unravel_index(int(np.argmax(data)), data.shape)
    r0, r1 = max(0, int(row) - 1), min(data.shape[0], int(row) + 2)
    c0, c1 = max(0, int(col) - 1), min(data.shape[1], int(col) + 2)
    weighted = np.clip(data[r0:r1, c0:c1], 0.0, None)
    total = float(weighted.sum())
    if total <= 0.0:
        return None
    rows = np.arange(r0, r1, dtype=np.float64)[:, None]
    cols = np.arange(c0, c1, dtype=np.float64)[None, :]
    row_f = float((weighted * rows).sum() / total)
    col_f = float((weighted * cols).sum() / total)
    return col_f, row_f


def _roll_to_zero_order(
    predicted: Any, sq_center: tuple[int, int], full_px: int
) -> tuple[np.ndarray, np.ndarray]:
    """Roll a square prediction onto the recorded zero-order.

    The forward model emits a ``full_px`` square centred on the zero-order,
    while the cropped measurement has the zero-order at ``sq_center``.  Rolling
    by the difference makes the two share a pixel grid.  Pixels that would wrap
    across an edge are marked invalid instead of being scored as bright spots.
    """
    pred = frame_np(predicted, normalise=False)
    if pred.shape != (full_px, full_px):
        raise ValueError(
            f"prediction is {pred.shape}, expected a square of {full_px} px"
        )
    half = full_px // 2
    dy = int(sq_center[1]) - half
    dx = int(sq_center[0]) - half
    rolled = np.roll(pred, shift=(dy, dx), axis=(0, 1))
    valid = np.zeros((full_px, full_px), dtype=bool)
    y0, y1 = (dy, full_px) if dy >= 0 else (0, full_px + dy)
    x0, x1 = (dx, full_px) if dx >= 0 else (0, full_px + dx)
    if y1 > y0 and x1 > x0:
        valid[y0:y1, x0:x1] = True
    return rolled, valid


def _l1_over(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    denom = float(truth[mask].sum())
    if denom <= 0:
        return float("nan")
    return float(np.abs(pred[mask] - truth[mask]).sum() / denom)


def _ncc_over(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    a = pred[mask]
    b = truth[mask]
    if a.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def _geometric_half_width_px(cfg: Sim2fConfig) -> float:
    """Half-width, in camera pixels, of the forward model's far-field window.

    The FFT grid spans ``fft_length * far_field_pitch_m`` metres in the focal
    plane, and the camera samples that span at ``ccd_pixel_um``.  The product
    ``fft_length * far_field_pitch_m`` is the physical window, so the result
    does not depend on the prediction's pixel values (unlike ``rolled > 0``).
    """
    return (
        float(cfg.fft_length)
        * float(cfg.far_field_pitch_m)
        / (2.0 * float(cfg.ccd_pixel_um) * 1e-6)
    )


def _geometric_support(
    cfg: Sim2fConfig, sq_center: tuple[int, int], full_px: int
) -> np.ndarray:
    """Model-window mask centred on the recorded zero-order.

    Pixel centres are the integer array indices, and the zero-order has already
    been moved to ``sq_center`` by :func:`_roll_to_zero_order`.  The caller must
    intersect the result with the roll's ``aligned`` mask.  This mask describes
    where the model has data, not where the prediction happened to be positive.

    Args:
        cfg: Full-frame configuration (its ``fft_length`` and pitch decide the
            physical window).
        sq_center: Recorded zero-order ``(x, y)`` in the rolled frame.
        full_px: Side length of the square frame.

    Returns:
        Boolean ``(full_px, full_px)`` mask.
    """
    half_width = _geometric_half_width_px(cfg)
    yy, xx = np.mgrid[0:full_px, 0:full_px]
    return (np.abs(xx - float(sq_center[0])) <= half_width) & (
        np.abs(yy - float(sq_center[1])) <= half_width
    )


def _calibration_metrics(
    predicted: np.ndarray, truth: np.ndarray, support: np.ndarray
) -> dict[str, float]:
    """L1/NCC/gain scored like ``calibrate._scores(fit_gain=True)``.

    Both frames are prepared on the *full* cropped grid -- median pedestal
    removed, negatives clipped (:func:`ml.sim2f.losses.prepare_frame`) -- so the
    numbers are comparable with ``calibrate_summary.json``.  The closed-form
    gain and the residuals are then evaluated only on ``support``, matching the
    geometric region the model actually predicts.  Degenerate cases return
    ``NaN``; ``analyze`` converts those to JSON ``null``.

    Args:
        predicted: Full cropped model frame (already rolled to the zero-order).
        truth: Full cropped measured frame.
        support: Geometric model window (and roll ``aligned``) mask.

    Returns:
        ``{"l1_calib", "ncc_calib", "gain_calib"}``.
    """
    result = {
        "l1_calib": float("nan"),
        "ncc_calib": float("nan"),
        "gain_calib": float("nan"),
    }
    if not bool(support.any()):
        return result

    def prepared(frame: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(np.asarray(frame, dtype=np.float32))[None, None]
        return prepare_frame(tensor)[0, 0].detach().cpu().numpy().astype(np.float64)

    pred = prepared(predicted)[support]
    meas = prepared(truth)[support]
    denom = float(np.dot(pred, pred))
    if denom <= _EPS:
        return result
    gain = float(np.dot(pred, meas) / denom)
    result["gain_calib"] = gain

    base = float(meas.sum())
    if base <= 0:
        return result
    result["l1_calib"] = float(np.abs(gain * pred - meas).sum() / base)

    a = pred - pred.mean()
    b = meas - meas.mean()
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm > 0:
        result["ncc_calib"] = float(np.dot(a, b) / norm)
    return result


METRIC_KEYS: tuple[str, ...] = (
    "l1_all",
    "l1_in",
    "l1_out",
    "ncc_all",
    "ncc_in",
    "ncc_out",
    "l1_calib",
    "ncc_calib",
    "gain_calib",
    "ee_in_true",
    "ee_in_pred",
    "ee_out_err",
    "centroid_err_px",
    "d86_err_px",
    "peak_ratio",
    "energy_total_ratio",
    "truth_energy_in_support",
)


def _support_metrics(
    predicted: np.ndarray,
    truth: np.ndarray,
    window: CameraWindow,
    support: np.ndarray,
) -> dict[str, float]:
    """Score a prediction on the geometric model support.

    ``l1_*``/``ncc_*``, centroid, D86 and the EE fractions see only ``support``
    pixels and are normalised on that support (``p`` and ``t`` sum to one
    there).  A pixel the forward model has no data for, or that wrapped across a
    rolled edge, therefore cannot move them.  ``l1_calib``/``ncc_calib``/
    ``gain_calib`` prepare the full cropped frames first and then fit the gain
    on ``support`` (see :func:`_calibration_metrics`).

    The energy metrics use a different basis on purpose: ``energy_total_ratio``
    and ``truth_energy_in_support`` keep each **full frame's** own unit-sum
    normalisation and report the fraction of that global energy that falls
    inside ``support``.  A support-only normalisation would force
    ``energy_total_ratio`` to 1 by construction and hide light placed outside
    the physical window.  ``truth_energy_in_support`` is the measured fraction
    on the same global basis, i.e. how much of the recorded light the model
    window covers.
    """
    nan = float("nan")
    result = dict.fromkeys(METRIC_KEYS, nan)
    raw_pred = np.asarray(predicted, dtype=np.float64)
    raw_true = np.asarray(truth, dtype=np.float64)
    result.update(_calibration_metrics(raw_pred, raw_true, support))

    global_pred = np.clip(raw_pred, 0.0, None)
    global_true = np.clip(raw_true, 0.0, None)
    pred_global_total = float(global_pred.sum())
    true_global_total = float(global_true.sum())
    if bool(support.any()) and pred_global_total > 0 and true_global_total > 0:
        pred_fraction = float(global_pred[support].sum() / pred_global_total)
        true_fraction = float(global_true[support].sum() / true_global_total)
        result["truth_energy_in_support"] = true_fraction
        if true_fraction > 0:
            result["energy_total_ratio"] = pred_fraction / true_fraction

    pred = np.where(support, global_pred, 0.0)
    true = np.where(support, global_true, 0.0)
    true_total = float(true.sum())
    pred_total = float(pred.sum())
    if not bool(support.any()) or true_total <= 0 or pred_total <= 0:
        return result

    p = pred / pred_total
    t = true / true_total
    frame_size = (support.shape[1], support.shape[0])
    inside = support & (window.mask(frame_size) > 0)
    outside = support & ~inside

    result["l1_all"] = _l1_over(p, t, support)
    result["ncc_all"] = _ncc_over(p, t, support)
    result["l1_in"] = _l1_over(p, t, inside)
    result["ncc_in"] = _ncc_over(p, t, inside)
    result["l1_out"] = _l1_over(p, t, outside)
    result["ncc_out"] = _ncc_over(p, t, outside)
    result["ee_in_true"] = float(t[inside].sum())
    result["ee_in_pred"] = float(p[inside].sum())
    result["ee_out_err"] = abs(result["ee_in_pred"] - result["ee_in_true"])

    pred_moments = moment_metrics(p)
    true_moments = moment_metrics(t)
    pred_centre = (pred_moments["cx"], pred_moments["cy"])
    true_centre = (true_moments["cx"], true_moments["cy"])
    if all(math.isfinite(value) for value in pred_centre + true_centre):
        result["centroid_err_px"] = float(
            math.hypot(pred_centre[0] - true_centre[0], pred_centre[1] - true_centre[1])
        )
    if math.isfinite(pred_moments["d86_px"]) and math.isfinite(true_moments["d86_px"]):
        result["d86_err_px"] = abs(
            float(pred_moments["d86_px"] - true_moments["d86_px"])
        )
    true_peak = float(true.max())
    if true_peak > 0:
        result["peak_ratio"] = float(pred.max() / true_peak)
    return result


def _support_diagnostics(support: np.ndarray) -> dict[str, Any]:
    """JSON-safe description of the mask a row was scored on."""
    count = int(support.sum())
    support_shape = [0, 0]
    if count:
        rows, cols = np.nonzero(support)
        support_shape = [
            int(rows.max() - rows.min()) + 1,
            int(cols.max() - cols.min()) + 1,
        ]
        bbox: list[int] | None = [
            int(cols.min()),
            int(rows.min()),
            int(cols.max()) + 1,
            int(rows.max()) + 1,
        ]
    else:
        bbox = None
    return {
        "score_mask_count": count,
        "valid_support_fraction": float(count / support.size),
        "valid_support_shape": support_shape,
        "valid_support_bbox": bbox,
    }


def _group_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate each metric per plan ``group``.

    A group is the plan's ``group`` label (``calib``, ``probe``, ``shape``).
    Every metric reports ``{"mean", "n"}``: the mean over members where it was
    defined and how many that was.  An undefined member (JSON ``null``) is
    skipped rather than counted as zero, so a group mean never invents a score
    for frames whose support was empty.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["group"]), []).append(row)
    summaries: dict[str, Any] = {}
    for group in sorted(grouped):
        members = grouped[group]
        metrics: dict[str, dict[str, Any]] = {}
        for key in METRIC_KEYS:
            values = [
                float(member[key])
                for member in members
                if isinstance(member.get(key), (int, float))
                and math.isfinite(float(member[key]))
            ]
            metrics[key] = {
                "mean": float(np.mean(values)) if values else None,
                "n": len(values),
            }
        summaries[group] = {"items": len(members), "metrics": metrics}
    return summaries


# ---------------------------------------------------------------------------
# Targets and phase generation
# ---------------------------------------------------------------------------
def _target_mask(kind: str, ccd_px: int) -> np.ndarray:
    """A binary target mask, same families as the Lane B reference script."""
    yy, xx = np.mgrid[0:ccd_px, 0:ccd_px].astype(np.float64)
    cx = cy = (ccd_px - 1) / 2.0
    radius = np.hypot(xx - cx, yy - cy)
    out = np.zeros((ccd_px, ccd_px), dtype=np.float64)
    if kind == "ring":
        out[(radius > 0.30 * ccd_px) & (radius < 0.42 * ccd_px)] = 1.0
    elif kind == "cross":
        out[(np.abs(xx - cx) < 0.06 * ccd_px) & (radius < 0.40 * ccd_px)] = 1.0
        out[(np.abs(yy - cy) < 0.06 * ccd_px) & (radius < 0.40 * ccd_px)] = 1.0
    elif kind == "frame":
        square = np.maximum(np.abs(xx - cx), np.abs(yy - cy))
        out[(square > 0.26 * ccd_px) & (square < 0.36 * ccd_px)] = 1.0
    elif kind in ("two_spot", "two_spot_wide"):
        separation = 0.42 * ccd_px if kind == "two_spot" else 0.46 * ccd_px
        for spot_x in (cx - separation, cx + separation):
            out[np.hypot(xx - spot_x, yy - cy) < 0.07 * ccd_px] = 1.0
    else:
        raise ValueError(f"unknown target {kind!r}; choose from {TARGETS}")
    if out.sum() <= 0.0:
        raise ValueError(f"target {kind!r} is empty at ccd_px={ccd_px}")
    return out.astype(np.float32)


def _pad_center(array: np.ndarray, size: int) -> np.ndarray:
    out = np.zeros((size, size), dtype=array.dtype)
    height, width = array.shape
    top = (size - height) // 2
    left = (size - width) // 2
    out[top : top + height, left : left + width] = array
    return out


def _gs_phase(mask: np.ndarray, bench: Sim2f, iterations: int = GS_ITERS) -> np.ndarray:
    """Deterministic Gerchberg-Saxton on the nominal twin.

    The propagation is the twin's own ideal one: the nominal illumination padded
    into ``fft_n``, a centred FFT, and the target amplitude substituted at the
    focal plane.  Static lens/defocus/aberration phases are deliberately absent:
    this is a *plan* seed, and the captured frame -- not this model -- is the
    measurement.
    """
    grid = int(bench.grid)
    fft_n = int(bench.fft_n)
    amplitude = _pad_center(
        bench.nominal_illumination[0, 0].detach().cpu().numpy(), fft_n
    )
    target = torch.as_tensor(mask, dtype=torch.float32)[None, None]
    # FFT samples and CCD pixels have different physical pitches. Resizing the
    # target over the entire FFT silently enlarges a 256px ring by about 4.2x.
    fft_axis = torch.arange(fft_n, dtype=torch.float32) - fft_n // 2
    camera_axis = fft_axis * float(bench.cfg.sampling_ratio) + int(bench.ccd_px) // 2
    normalised = 2 * (camera_axis + 0.5) / int(bench.ccd_px) - 1
    yy, xx = torch.meshgrid(normalised, normalised, indexing="ij")
    coordinates = torch.stack((xx, yy), dim=-1)[None]
    target = torch.nn.functional.grid_sample(
        target, coordinates, mode="bilinear", padding_mode="zeros", align_corners=False
    )[0, 0]
    target = torch.sqrt(target / target.sum().clamp_min(1e-12))
    pupil = torch.as_tensor(amplitude, dtype=torch.float32)
    phase = torch.zeros((fft_n, fft_n), dtype=torch.float32)
    for _ in range(int(iterations)):
        field = pupil * torch.exp(1j * phase)
        focal = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(field)))
        focal = target * torch.exp(1j * torch.angle(focal))
        back = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(focal)))
        phase = torch.angle(back)
    top = (fft_n - grid) // 2
    wrapped = (
        phase[top : top + grid, top : top + grid]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    # Panel embedding interpolates radians, so its input must be continuous.
    # Unwrap preserves each grid sample's phasor; wrapping belongs to the driver
    # after panel resampling, not to the low-resolution control grid.
    from skimage.restoration import unwrap_phase

    return unwrap_phase(wrapped, rng=0).astype(np.float32)


def _fit_phase(
    mask: np.ndarray, bench: CalibratedBench, *, seed: int, iterations: int
) -> np.ndarray:
    """Deterministic Adam phase for the fitted bench (the fitted variant)."""
    target = torch.as_tensor(mask, dtype=torch.float32)[None, None]
    target = target / target.sum().clamp_min(1e-12)
    torch.manual_seed(int(seed))
    phase = torch.zeros(
        (1, int(bench.grid), int(bench.grid)), dtype=torch.float32, requires_grad=True
    )
    optimizer = torch.optim.Adam([phase], lr=0.05)
    for _ in range(int(iterations)):
        optimizer.zero_grad()
        frame = bench.forward(phase, detector=False)
        frame = frame / frame.sum().clamp_min(1e-12)
        loss = (frame - target).abs().sum()
        loss.backward()
        optimizer.step()
    return phase.detach()[0].cpu().numpy().astype(np.float32)


def _zernike_phase(
    bench: Sim2f, noll: int, amplitude_rad: float, n_modes: int = 10
) -> np.ndarray:
    """One Zernike mode as a ``(grid, grid)`` phase, coefficient ``amplitude_rad``."""
    coeffs = np.zeros((1, int(n_modes)), dtype=np.float32)
    coeffs[0, int(noll) - 2] = float(amplitude_rad)
    return bench.aberration(coeffs)[0].detach().cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------
def _flat_item(index: int, group: str, condition: str, repeat: int) -> dict[str, Any]:
    return {
        "id": f"flat_{index:04d}",
        "kind": "flat",
        "group": group,
        "condition": condition,
        "repeat": int(repeat),
        "phase": f"{PHASE_DIR}/flat_{index:04d}.npy",
    }


def build_plan(
    *,
    out: Path,
    grid: int,
    window_px: int,
    seed: int,
    screens: int,
    repeats: int,
    targets: Iterable[str],
    calibration: Path | None,
    fit_iters: int,
) -> dict[str, Any]:
    """Build and write the deterministic plan."""
    if int(screens) < 1:
        raise ValueError("screens must be >= 1")
    if not 2 <= int(repeats) <= 3:
        raise ValueError("repeats must be 2 or 3")
    chosen_targets = tuple(str(name) for name in targets)
    unknown = [name for name in chosen_targets if name not in TARGETS]
    if unknown:
        raise ValueError(f"unknown targets {unknown}; choose from {TARGETS}")

    cfg = replace(Sim2fConfig.calibrated(), grid=int(grid), ccd_px=int(window_px))
    nominal = Sim2f(cfg, device="cpu")
    fitted: CalibratedBench | None = None
    if calibration is not None:
        fitted = CalibratedBench(
            Sim2f(cfg, device="cpu"), load_calibration(calibration)
        )

    items: list[dict[str, Any]] = []
    flat_count = 0
    net_count = 0

    def add_phase(
        item_id: str,
        kind: str,
        group: str,
        repeat: int,
        phase: np.ndarray,
        **extra: Any,
    ) -> None:
        nonlocal flat_count, net_count
        if kind == "flat":
            condition = "a1" if flat_count % 2 == 0 else "a2"
            flat_count += 1
        else:
            condition = "b1" if net_count % 2 == 0 else "b2"
            net_count += 1
        relative = f"{PHASE_DIR}/{item_id}.npy"
        _save_npy(out / relative, np.asarray(phase, dtype=np.float32))
        items.append(
            {
                "id": item_id,
                "kind": kind,
                "group": group,
                "condition": condition,
                "repeat": int(repeat),
                "phase": relative,
                "phase_sha256": _array_sha256(phase),
                **extra,
            }
        )

    def add_flat(group: str, repeat: int) -> None:
        index = len([item for item in items if item["kind"] == "flat"])
        add_phase(f"flat_{index:04d}", "flat", group, repeat, np.zeros((grid, grid)))

    # -- calibration stage: low-order screens + a flat after each --------------
    rng = np.random.default_rng(int(seed))
    basis = nominal.zernike_basis(len(LOW_ORDER_NOLL) + 1)
    low_rows = basis[1 : 1 + len(LOW_ORDER_NOLL)]
    for screen in range(int(screens)):
        coeffs = rng.normal(0.0, 0.12, size=len(LOW_ORDER_NOLL)).astype(np.float32)
        phase = (coeffs[:, None, None] * low_rows.detach().cpu().numpy()).sum(axis=0)
        add_phase(
            f"calib_screen_{screen:03d}",
            "calib_screen",
            "calib",
            0,
            phase.astype(np.float32),
            screen=int(screen),
        )
        add_flat("calib", 0)

    # -- probe bank: +/- Noll 4..11, one flat after each sign ------------------
    for noll in PROBE_NOLL:
        for sign, label in ((1.0, "plus"), (-1.0, "minus")):
            phase = _zernike_phase(nominal, noll, sign * PROBE_AMPLITUDE_RAD)
            add_phase(
                f"probe_noll{noll:02d}_{label}",
                "probe",
                "probe",
                0,
                phase,
                noll=int(noll),
                sign=label,
            )
            add_flat("probe", 0)

    # -- shapes: nominal always, fitted when a calibration was given ----------
    for kind in chosen_targets:
        mask = _target_mask(kind, int(window_px))
        for repeat in range(int(repeats)):
            add_phase(
                f"shape_{kind}_nominal_r{repeat}",
                "shape",
                "shapes",
                repeat,
                _gs_phase(mask, nominal),
                target=kind,
                variant="nominal",
            )
            add_flat("shapes", repeat)
            if fitted is not None:
                add_phase(
                    f"shape_{kind}_fitted_r{repeat}",
                    "shape",
                    "shapes",
                    repeat,
                    _fit_phase(
                        mask,
                        fitted,
                        seed=int(seed) + repeat,
                        iterations=int(fit_iters),
                    ),
                    target=kind,
                    variant="fitted",
                )
                add_flat("shapes", repeat)

    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "seed": int(seed),
        "grid": int(grid),
        "window_px": int(window_px),
        "calibration": None if calibration is None else str(calibration),
        "probe": {
            "noll": [int(noll) for noll in PROBE_NOLL],
            "amplitude_rad": float(PROBE_AMPLITUDE_RAD),
            "n_modes": 10,
        },
        "targets": list(chosen_targets),
        "repeats": int(repeats),
        "screens": int(screens),
        "items": items,
    }
    _json_write(out / PLAN_NAME, plan)
    return plan


# ---------------------------------------------------------------------------
# manifest / capture
# ---------------------------------------------------------------------------
def _manifest_path(session: Path) -> Path:
    return session / MANIFEST_NAME


def _load_plan(session: Path) -> dict[str, Any]:
    path = session / PLAN_NAME
    if not path.is_file():
        raise FileNotFoundError(f"no {PLAN_NAME} under {session}")
    return _json_read(path)


def _entry_valid(
    session: Path, plan: dict[str, Any], entry: dict[str, Any] | None
) -> bool:
    """A done entry is valid iff both arrays exist and match their hashes."""
    if not entry or entry.get("status") != "done":
        return False
    frame = session / str(entry.get("frame", ""))
    panel = session / str(entry.get("panel", ""))
    if not frame.is_file() or not panel.is_file():
        return False
    phase_path = session / str(entry.get("phase", ""))
    if not phase_path.is_file():
        return False
    try:
        return (
            _array_sha256(_load_npy(frame)) == entry.get("frame_sha256")
            and _array_sha256(_load_npy(panel)) == entry.get("panel_sha256")
            and _array_sha256(_load_npy(phase_path)) == entry.get("phase_sha256")
        )
    except (OSError, ValueError, EOFError):
        # A truncated or otherwise unreadable ``.npy`` is a missing item, not a
        # reason to abort the whole resume: re-capture it.
        return False


def _pending_items(
    session: Path, plan: dict[str, Any], manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    entries = manifest.get("items", {})
    return [
        item
        for item in plan["items"]
        if not _entry_valid(session, plan, entries.get(item["id"]))
    ]


def _open_instrument(args: argparse.Namespace, plan: dict[str, Any]):
    if args.dry_run or args.backend == "sim":
        return SimInstrument(bench=Sim2f(Sim2fConfig.calibrated()))
    return HwInstrument(
        exposure_ms=args.exposure_ms,
        ccd_px=int(plan["window_px"]),
        settle_s=args.settle_s,
        cam_type=args.cam_type,
        cam_id=args.cam_id,
    )


def capture(args: argparse.Namespace) -> dict[str, Any]:
    """Capture every pending item as one native full-frame read."""
    session = Path(args.session)
    plan = _load_plan(session)
    manifest_file = _manifest_path(session)
    manifest = (
        _json_read(manifest_file)
        if manifest_file.is_file()
        else {
            "schema": MANIFEST_SCHEMA,
            "plan_sha256": _file_sha256(session / PLAN_NAME),
            "backend": "sim" if (args.dry_run or args.backend == "sim") else "hw",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "items": {},
        }
    )
    if manifest.get("plan_sha256") != _file_sha256(session / PLAN_NAME):
        raise ValueError("capture manifest belongs to a different plan.json")

    pending = _pending_items(session, plan, manifest)
    if not pending:
        print(f"capture: all {len(plan['items'])} items already valid")
        return manifest

    instrument = _open_instrument(args, plan)
    try:
        native = instrument.enable_full_frame()
        manifest["native_shape"] = [int(v) for v in native]
        manifest["zero_order_xy"] = [int(v) for v in instrument.zero_order_xy]
        manifest["exposure_ms"] = getattr(instrument, "exposure_ms", args.exposure_ms)
        _json_write(manifest_file, manifest)
        print(
            f"capture: {len(pending)}/{len(plan['items'])} pending, "
            f"native={native[1]}x{native[0]} zero_order={instrument.zero_order_xy}"
        )
        for index, item in enumerate(pending, start=1):
            phase_path = session / item["phase"]
            phase = np.asarray(_load_npy(phase_path), dtype=np.float32)
            if _array_sha256(phase) != item["phase_sha256"]:
                raise ValueError(f"phase {phase_path} no longer matches plan.json")
            frame = np.asarray(instrument.capture_full_frame(phase), dtype=np.float32)
            panel = np.asarray(instrument.sent_grayscale(phase), dtype=np.uint16)
            frame_rel = f"{FRAME_DIR}/{item['id']}.npy"
            panel_rel = f"{PANEL_DIR}/{item['id']}.npy"
            _save_npy(session / frame_rel, frame)
            _save_npy(session / panel_rel, panel)
            manifest["items"][item["id"]] = {
                "status": "done",
                "kind": item["kind"],
                "group": item["group"],
                "condition": item["condition"],
                "repeat": int(item["repeat"]),
                "phase": item["phase"],
                "phase_sha256": item["phase_sha256"],
                "frame": frame_rel,
                "frame_sha256": _array_sha256(frame),
                "panel": panel_rel,
                "panel_sha256": _array_sha256(panel),
                "exposure_ms": getattr(instrument, "exposure_ms", args.exposure_ms),
                "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            _json_write(manifest_file, manifest)
            print(f"  [{index}/{len(pending)}] {item['id']} -> {frame.shape}")
    finally:
        instrument.close()
    return manifest


# ---------------------------------------------------------------------------
# finalize
# ---------------------------------------------------------------------------
def _require_done(
    session: Path, plan: dict[str, Any], manifest: dict[str, Any]
) -> None:
    missing = _pending_items(session, plan, manifest)
    if missing:
        names = ", ".join(item["id"] for item in missing[:6])
        raise RuntimeError(
            f"{len(missing)} item(s) not captured or invalid: {names}"
            + (" ..." if len(missing) > 6 else "")
        )


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    """Crop native frames and write the standard bench recording."""
    session = Path(args.session)
    plan = _load_plan(session)
    manifest_file = _manifest_path(session)
    if not manifest_file.is_file():
        raise FileNotFoundError(f"no {MANIFEST_NAME} under {session}")
    manifest = _json_read(manifest_file)
    _require_done(session, plan, manifest)

    window_px = _round_window(int(args.window_px or plan["window_px"]))
    native = tuple(int(v) for v in manifest["native_shape"])
    center = tuple(int(v) for v in manifest["zero_order_xy"])
    cfg = replace(
        Sim2fConfig.calibrated(),
        grid=int(plan["grid"]),
        ccd_px=int(window_px),
    )

    recorder = BenchRecorder(mark="frame_peak")
    for index, item in enumerate(plan["items"]):
        entry = manifest["items"][item["id"]]
        frame = np.asarray(_load_npy(session / entry["frame"]), dtype=np.float32)
        panel_path = session / entry["panel"]
        phase = np.asarray(_load_npy(session / item["phase"]), dtype=np.float32)
        if panel_path.is_file():
            panel = np.asarray(_load_npy(panel_path), dtype=np.uint16)
        else:
            panel = recorded_grayscale(phase)
        cropped = _crop_square(frame, center, window_px)
        recorder.append_sample(
            index=index,
            kind=str(item["kind"]),
            condition=str(item["condition"]),
            frame=cropped,
            command_phase=phase,
            panel_grayscale=panel,
            metrics={
                "frame_peak": float(cropped.max()),
                "frame_sum": float(cropped.sum()),
            },
        )

    process_dir = session / PROCESS_DIR
    if process_dir.is_dir():
        shutil.rmtree(process_dir)
    recorder.save(process_dir)

    payload = {
        "schema": SESSION_SCHEMA,
        "backend": str(manifest.get("backend", "?")),
        "bench": cfg.to_dict(),
        "plan_sha256": _file_sha256(session / PLAN_NAME),
        "manifest_sha256": _file_sha256(manifest_file),
        "window_px": int(window_px),
        "native_shape": [int(v) for v in native],
        "zero_order_xy": [int(v) for v in center],
        "items": len(plan["items"]),
        "notes": (
            "Native full frames captured once per command; finalize crops a "
            f"{window_px}px square around the recorded 0-order. Conditions map "
            "flat->a1/a2 and everything else->b1/b2 so load_bench_session can "
            "read the recording."
        ),
    }
    _json_write(session / SESSION_NAME, payload)
    print(
        f"finalize: {len(plan['items'])} rows, window={window_px}px -> "
        f"{session / PROCESS_DIR}"
    )
    return payload


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------
def calibrate(args: argparse.Namespace) -> dict[str, Any]:
    """Offline bench calibration from the finalized recording."""
    session = Path(args.session)
    payload = _json_read(session / SESSION_NAME)
    manifest_path = _manifest_path(session)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no {MANIFEST_NAME} under {session}")
    if payload.get("manifest_sha256") != _file_sha256(manifest_path):
        raise RuntimeError(
            "session.json was written from a different capture manifest; "
            "re-run finalize (refusing to calibrate a stale session)"
        )
    bench_session = load_bench_session(session)
    sim = Sim2f(bench_session.sim_config)
    result = calibrate_bench(
        sim,
        bench_session,
        CalibrateConfig(
            iters=int(args.iters),
            seed=int(args.seed),
            fit_measurement=bool(args.fit_measurement),
            fit_zero_order=bool(args.fit_zero_order),
        ),
    )
    out = Path(args.out)
    json_path, npz_path = save_calibration(out, result)
    summary = {
        "schema": "sim2f_fov_hw_session/calibrate/v1",
        "ok": bool(result.ok),
        "reason": result.reason,
        "score": {name: asdict(score) for name, score in result.score.items()},
        "score_start": {
            name: asdict(score) for name, score in result.score_start.items()
        },
        "n_train": int(result.n_train),
        "n_holdout": int(result.n_holdout),
        "measurement": dict(result.measurement),
        "zero_order": dict(result.zero_order),
        "json": str(json_path),
        "npz": str(npz_path),
    }
    _json_write(out / "calibrate_summary.json", summary)
    print(f"calibrate: ok={result.ok} reason={result.reason} -> {out}")
    return summary


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------
def _phase_array(value: Any) -> np.ndarray:
    array = np.asarray(
        value.detach().cpu().numpy() if torch.is_tensor(value) else value
    )
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"probe requested a batched phase of shape {array.shape}")
    return np.asarray(array, dtype=np.float32)


def _phase_lookup(
    plan: dict[str, Any],
    manifest: dict[str, Any],
    session: Path,
    window_px: int,
    center: tuple[int, int],
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray | None]:
    """Map every planned command to its cropped measured frame."""
    table: list[tuple[np.ndarray, np.ndarray]] = []
    base: np.ndarray | None = None
    for item in plan["items"]:
        phase = np.asarray(_load_npy(session / item["phase"]), dtype=np.float32)
        entry = manifest["items"][item["id"]]
        frame = np.asarray(_load_npy(session / entry["frame"]), dtype=np.float32)
        cropped = _crop_square(frame, center, window_px)
        table.append((phase, cropped))
        if base is None and item["kind"] == "flat":
            base = cropped
    return table, base


def _match_phase(
    target: np.ndarray, table: list[tuple[np.ndarray, np.ndarray]], tol: float = 5e-3
) -> np.ndarray:
    best: np.ndarray | None = None
    best_error = float("inf")
    for phase, frame in table:
        error = float(np.max(np.abs(phase - target)))
        if error < best_error:
            best, best_error = frame, error
    if best is None or best_error > tol:
        raise KeyError(
            f"no planned probe matches the requested phase ({best_error:.3g})"
        )
    return best


def _predict_full_frame(
    model: Any, phase: np.ndarray, residual_map: np.ndarray | None
) -> Any:
    """Predict one full cropped frame for scoring, without detector photometry.

    The metrics compare the model with a *prepared* measurement, so a
    photometric detector response would change the quantity being compared.
    Detector semantics stay in capture and training, where the camera response
    is part of the model.  ``analyze`` always takes this path, including on a
    real backend.
    """
    return model.forward(
        torch.as_tensor(phase)[None],
        aberration=residual_map,
        detector=False,
    )


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    """Offline full-frame prediction and scoring.

    Three offsets are reported separately because they answer different
    questions.  ``zero_order_offset_xy`` is crop clamping: how far the recorded
    zero-order sits from the square centre when the crop is pushed off the
    sensor edge; the roll corrects it exactly.  ``zero_order_residual_px`` is
    recorded versus measured: the flat frame's measured zero-order against the
    position the manifest recorded.  ``zero_order_align_residual_px`` is model
    versus measurement: the measured spot against the point where the model's
    optical axis lands after the roll (``crop_sq_zero_order_xy``).  The measured
    values are ``None`` and listed in ``undefined_metrics`` when the flat frame
    has no positive peak, instead of being invented as zero.
    """
    session = Path(args.session)
    plan = _load_plan(session)
    manifest = _json_read(_manifest_path(session))
    _require_done(session, plan, manifest)
    payload = _json_read(session / SESSION_NAME)
    window_px = _round_window(int(args.window_px or payload["window_px"]))
    native = tuple(int(v) for v in manifest["native_shape"])
    center = tuple(int(v) for v in manifest["zero_order_xy"])
    full_px = int(min(native))
    # 模型整幅输出是以 0 级为中心的方形 (full_px, full_px)；实测原生帧是
    # (native[0], native[1])（如 1944x2592）。把实测帧按 0 级中心裁到同一方形，
    # 只丢弃关于 0 级对称的左右远翼。贴边时裁剪被传感器边缘夹紧，实测 0 级
    # 因而可能偏离方形中心；后续预测按记录的 0 级平移，而不是按方形中心硬贴。
    col0, row0, sq_center = _crop_layout(native, center, full_px)
    half = full_px // 2

    cfg = replace(
        Sim2fConfig.calibrated(),
        grid=int(plan["grid"]),
        ccd_px=int(window_px),
    )
    full_cfg = replace(cfg, ccd_px=full_px)
    calibration = load_calibration(args.calibration) if args.calibration else None
    window_model: Any = Sim2f(cfg, device="cpu")
    full_model: Any = Sim2f(full_cfg, device="cpu")
    if calibration is not None:
        window_model = CalibratedBench(window_model, calibration)
        full_model = CalibratedBench(full_model, calibration)

    table, base_frame = _phase_lookup(plan, manifest, session, window_px, center)
    if base_frame is None:
        raise RuntimeError("plan has no flat item to probe from")
    aberration_cfg = AberrationConfig(n_modes=10)

    def measure(command: Any) -> torch.Tensor:
        return torch.as_tensor(_match_phase(_phase_array(command), table))

    fit_payload: dict[str, Any] = {}
    residual_map: np.ndarray | None = None
    try:
        result = probe(
            window_model,
            aberration_cfg,
            measure=measure,
            measured=base_frame,
        )
        fit_payload = {
            "ok": bool(result.ok),
            "reason": result.reason,
            "rank": int(result.rank),
            "condition": float(result.condition),
            "residual": float(result.residual),
            "residual_start": float(result.residual_start),
            "coeffs": [float(value) for value in result.coeffs],
            "free_noll": [int(value) for value in result.free_noll],
        }
        residual_map = (
            window_model.aberration(result.coeffs[None])
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    except Exception as error:
        fit_payload = {"ok": False, "reason": f"{type(error).__name__}: {error}"}

    window = CameraWindow(
        center=sq_center, size=(window_px, window_px), frame=(full_px, full_px)
    )
    rows: list[dict[str, Any]] = []
    undefined: set[str] = set()
    flat_item = next((item for item in plan["items"] if item["kind"] == "flat"), None)
    measured_xy: tuple[float, float] | None = None
    if flat_item is not None:
        flat_path = session / manifest["items"][flat_item["id"]]["frame"]
        measured_xy = _zero_order_centroid_xy(_load_npy(flat_path))
    measured_source = None if measured_xy is None else str(flat_item["id"])
    measured_residual_px: float | None = None
    align_residual_px: float | None = None
    if measured_xy is not None:
        measured_residual_px = float(
            math.hypot(measured_xy[0] - center[0], measured_xy[1] - center[1])
        )
        align_residual_px = float(
            math.hypot(
                measured_xy[0] - col0 - sq_center[0],
                measured_xy[1] - row0 - sq_center[1],
            )
        )
    else:
        undefined.update(
            {
                "zero_order_measured_xy",
                "zero_order_measured_source",
                "zero_order_residual_px",
                "zero_order_align_residual_px",
            }
        )
    geometric = _geometric_support(full_cfg, sq_center, full_px)
    for item in plan["items"]:
        entry = manifest["items"][item["id"]]
        truth = np.asarray(_load_npy(session / entry["frame"]), dtype=np.float32)
        if truth.shape != (full_px, full_px):
            truth = truth[row0 : row0 + full_px, col0 : col0 + full_px]
        phase = np.asarray(_load_npy(session / item["phase"]), dtype=np.float32)
        predicted = _predict_full_frame(full_model, phase, residual_map)
        rolled, aligned = _roll_to_zero_order(predicted, sq_center, full_px)
        support = aligned & geometric
        metrics = _support_metrics(rolled, truth, window, support)
        row: dict[str, Any] = {
            "id": item["id"],
            "kind": item["kind"],
            "group": item["group"],
            "target": item.get("target"),
            "variant": item.get("variant"),
        }
        for key, value in metrics.items():
            number = float(value)
            if math.isfinite(number):
                row[key] = number
            else:
                # A metric over pixels this session never saw is undefined, not
                # zero: record ``None`` and say so in ``not_implemented``.
                row[key] = None
                undefined.add(key)
        row.update(_support_diagnostics(support))
        rows.append(row)

    not_implemented = {
        "online_analysis": "not_implemented: capture never analyses frames",
        "full_frame_pinn": "not_implemented: no Lane C/PINN port in this script",
        "fitted_phase_quality": (
            "not_implemented: the plan records GS/Adam commands, not a claimed "
            "shape quality; analyze scores measured frames only"
        ),
    }
    if undefined:
        not_implemented["outside_window_metrics"] = (
            "not_implemented: "
            f"{sorted(undefined)} are undefined because the geometric model "
            "support has too few pixels, the prepared frames are degenerate "
            "(clipped crop or wrapped edge), or the flat frame carries no "
            "positive zero-order peak"
        )

    summary = {
        "schema": SUMMARY_SCHEMA,
        "calibration": None if args.calibration is None else str(args.calibration),
        "window_px": int(window_px),
        "native_shape": [int(v) for v in native],
        "zero_order_xy": [int(v) for v in center],
        "zero_order_offset_xy": [
            int(sq_center[0] - half),
            int(sq_center[1] - half),
        ],
        "crop_origin_xy": [int(col0), int(row0)],
        "crop_sq_zero_order_xy": [int(sq_center[0]), int(sq_center[1])],
        "zero_order_measured_xy": (
            None
            if measured_xy is None
            else [float(measured_xy[0]), float(measured_xy[1])]
        ),
        "zero_order_measured_source": measured_source,
        "zero_order_residual_px": measured_residual_px,
        "zero_order_aligned_xy": [float(sq_center[0]), float(sq_center[1])],
        "zero_order_align_residual_px": align_residual_px,
        "geometric_half_width_px": float(_geometric_half_width_px(full_cfg)),
        "fit": fit_payload,
        "undefined_metrics": sorted(undefined),
        "items": rows,
        "groups": _group_summaries(rows),
        "not_implemented": not_implemented,
    }
    _json_write(Path(args.out) / SUMMARY_NAME, summary)
    print(
        f"analyze: {len(rows)} items, fit ok={fit_payload.get('ok')} -> "
        f"{Path(args.out) / SUMMARY_NAME}"
    )
    return summary


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
def preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Report plan/session readiness without touching a device by default."""
    session = Path(args.session)
    plan = _load_plan(session)
    manifest_file = _manifest_path(session)
    manifest = _json_read(manifest_file) if manifest_file.is_file() else None
    pending = (
        _pending_items(session, plan, manifest)
        if manifest is not None
        else list(plan["items"])
    )
    report: dict[str, Any] = {
        "schema": "sim2f_fov_hw_session/preflight/v1",
        "plan": str(session / PLAN_NAME),
        "items": len(plan["items"]),
        "pending": len(pending),
        "finalized": (session / SESSION_NAME).is_file(),
        # 优先用命令行显式覆盖；否则回退到 plan 自带标定，避免生产计划误报 null
        "calibration": str(args.calibration or plan.get("calibration") or "") or None,
    }
    if report["calibration"]:
        report["calibration_ok"] = bool(load_calibration(report["calibration"]).ok)
    if args.dry_run:
        instrument = SimInstrument(bench=Sim2f(Sim2fConfig.calibrated()))
        try:
            report["sim_native_shape"] = [
                int(v) for v in instrument.enable_full_frame()
            ]
            report["sim_zero_order_xy"] = [int(v) for v in instrument.zero_order_xy]
        finally:
            instrument.close()
    _json_write(session / "preflight.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan_cmd = sub.add_parser("plan", help="write a deterministic plan and its phases")
    plan_cmd.add_argument("--out", required=True)
    plan_cmd.add_argument("--grid", type=int, default=128)
    plan_cmd.add_argument("--window-px", type=int, default=256)
    plan_cmd.add_argument("--seed", type=int, default=0)
    plan_cmd.add_argument("--screens", type=int, default=CALIB_SCREENS)
    plan_cmd.add_argument("--repeats", type=int, default=2)
    plan_cmd.add_argument("--targets", nargs="+", default=list(TARGETS))
    plan_cmd.add_argument("--calibration")
    plan_cmd.add_argument("--fit-iters", type=int, default=FIT_ITERS)
    plan_cmd.set_defaults(
        func=lambda args: build_plan(
            out=Path(args.out),
            grid=args.grid,
            window_px=_round_window(args.window_px),
            seed=args.seed,
            screens=args.screens,
            repeats=args.repeats,
            targets=args.targets,
            calibration=None if args.calibration is None else Path(args.calibration),
            fit_iters=args.fit_iters,
        )
    )

    capture_cmd = sub.add_parser("capture", help="capture pending items, resumable")
    capture_cmd.add_argument("--session", required=True)
    capture_cmd.add_argument("--backend", choices=("sim", "hw"), default="hw")
    capture_cmd.add_argument("--dry-run", action="store_true")
    capture_cmd.add_argument("--exposure-ms", type=float)
    capture_cmd.add_argument("--settle-s", type=float, default=0.05)
    capture_cmd.add_argument("--cam-type", default="daheng")
    capture_cmd.add_argument("--cam-id", type=int, default=0)
    capture_cmd.set_defaults(func=capture)

    finalize_cmd = sub.add_parser(
        "finalize", help="crop frames and write the recording"
    )
    finalize_cmd.add_argument("--session", required=True)
    finalize_cmd.add_argument("--window-px", type=int)
    finalize_cmd.set_defaults(func=finalize)

    calibrate_cmd = sub.add_parser("calibrate", help="fit a bench state offline")
    calibrate_cmd.add_argument("--session", required=True)
    calibrate_cmd.add_argument("--out", required=True)
    calibrate_cmd.add_argument("--iters", type=int, default=300)
    calibrate_cmd.add_argument("--seed", type=int, default=0)
    calibrate_cmd.add_argument(
        "--fit-measurement",
        action="store_true",
        help="after the geometry fit, freeze it and fit detector-MTF sigma only",
    )
    calibrate_cmd.add_argument(
        "--fit-zero-order",
        action="store_true",
        help="last stage: fit the coherent zero-order eta alone, bounded to [0, 0.5]",
    )
    calibrate_cmd.set_defaults(func=calibrate)

    analyze_cmd = sub.add_parser("analyze", help="score full-frame predictions")
    analyze_cmd.add_argument("--session", required=True)
    analyze_cmd.add_argument("--out", required=True)
    analyze_cmd.add_argument("--calibration")
    analyze_cmd.add_argument("--window-px", type=int)
    analyze_cmd.set_defaults(func=analyze)

    preflight_cmd = sub.add_parser("preflight", help="report readiness")
    preflight_cmd.add_argument("--session", required=True)
    preflight_cmd.add_argument("--calibration")
    preflight_cmd.add_argument("--dry-run", action="store_true")
    preflight_cmd.set_defaults(func=preflight)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
