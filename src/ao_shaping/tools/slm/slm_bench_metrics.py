"""Pure measurement kernels for characterising the SLM 2f-Fourier bench.

No device objects, no I/O, no CLI: everything here is numpy on frames, so the
whole module is unit-testable offline and safe to import from CI.

This exists so the characterisation probes (:mod:`slm_drift_probe`,
:mod:`slm_floor_probe`, :mod:`slm_abba_probe`) analyse *numbers* rather than
each re-implementing frame preparation and drift statistics. It deliberately
does **not** own device access -- ``slm_bench_probe`` is the canonical
measurement kernel (devices injected, no CLI); this module is the pure
analysis layer that sits downstream of it.

Two frame preparations, and they are NOT interchangeable
-------------------------------------------------------
``finite_clip`` and ``finite_median_subtract`` look like a duplicate pair. They
are not, and unifying them silently breaks an objective:

* ``finite_clip`` mirrors
  :func:`ao_shaping.optimizer.wfless.slm_square_shaping.square_peak_to_background_ratio`.
  It clips negatives and leaves the pedestal alone, because the background
  occupies most of the frame and the median *is* the pedestal.
* ``finite_median_subtract`` mirrors
  :func:`ao_shaping.optimizer.wfless.slm_gs_refine._prepare_frame`. It removes
  the pedestal first, which is what any whole-frame-denominator ratio needs:
  read noise makes roughly half of a raw frame negative, and dividing by a
  near-zero or negative background drives the ratio above 1 (measured
  ``PIB = 1.0120`` on this bench).

Measured consequence of using the wrong one: on a synthetic frame with a
pedestal of 3 counts, median-subtracting before a peak-to-background ratio
drives the result to ~1e5 because the denominator collapses, while clip-only
gives the correct ~34. See ``tests/.../test_slm_bench_metrics.py::
TestPrepDivergence``, which locks both against the live optimizer functions.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ao_shaping.utils.wavefront.matrix_utils import (
    camera_pixel_um_from_focal_scale as _camera_pixel_um_from_focal_scale,
    focal_length_from_camera_pixel as _focal_length_from_camera_pixel,
)

__all__ = [
    "build_block_pattern",
    "camera_pixel_um_from_focal_scale",
    "focal_length_from_camera_pixel",
    "crop_roi",
    "exposure_monotonicity",
    "finite_clip",
    "finite_median_subtract",
    "flat_to_flat_floor",
    "roi_l2",
    "settle_time_s",
    "snr_vs_averages",
]


# ---------------------------------------------------------------------------
# Bench geometry — derive, do not hardcode
# ---------------------------------------------------------------------------

# Re-exported from the leaf `utils` layer so that both the optimizer layer and
# the tools layer share ONE definition (and therefore cannot drift apart).
camera_pixel_um_from_focal_scale = _camera_pixel_um_from_focal_scale
focal_length_from_camera_pixel = _focal_length_from_camera_pixel


# ---------------------------------------------------------------------------
# Frame preparation — the two variants are deliberately different
# ---------------------------------------------------------------------------


def _as_2d_float(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"frame must be 2D, got shape {arr.shape}")
    return arr


def finite_clip(frame: np.ndarray) -> np.ndarray:
    """Mask non-finite pixels to 0 and clip negatives, keeping the pedestal.

    Mirrors the preparation used by
    :func:`~ao_shaping.optimizer.wfless.slm_square_shaping.square_peak_to_background_ratio`.
    Correct for peak/background-style ratios, where subtracting the median would
    destroy the very pedestal that gives the denominator meaning.

    Args:
        frame: 2D camera frame.

    Returns:
        New float64 array, non-negative, same shape.
    """
    arr = _as_2d_float(frame)
    return np.clip(np.where(np.isfinite(arr), arr, 0.0), 0.0, None)


def finite_median_subtract(frame: np.ndarray) -> np.ndarray:
    """Mask non-finite pixels, subtract the median, then clip at 0.

    Mirrors :func:`~ao_shaping.optimizer.wfless.slm_gs_refine._prepare_frame`.
    Required before any ratio whose denominator is a whole-frame statistic:
    symmetric read noise leaves ~half of a raw frame negative, which pushes such
    a ratio above 1 and makes an optimizer chase noise.

    The order matters. Clipping first would rectify the noise distribution and
    invent a DC pedestal proportional to the pixel count.

    Args:
        frame: 2D camera frame.

    Returns:
        New float64 array, non-negative, same shape.
    """
    arr = _as_2d_float(frame)
    clean = np.where(np.isfinite(arr), arr, 0.0)
    return np.clip(clean - float(np.median(clean)), 0.0, None)


# ---------------------------------------------------------------------------
# ROI helpers
# ---------------------------------------------------------------------------


def crop_roi(
    frame: np.ndarray, center: tuple[int, int], half: int
) -> np.ndarray:
    """Crop a ``2*half`` square about ``center``, zero-padding off-frame.

    The spot is frequently not at the frame centre on this bench (measured
    0-order near (674, 1026) on a 1944x2592 frame), so ROI maths must always be
    driven by a measured centre rather than ``shape // 2``.

    Args:
        frame: 2D camera frame.
        center: ROI centre as ``(x, y)`` in pixels.
        half: Half-width; the result is ``2*half`` wide.

    Returns:
        ``(2*half, 2*half)`` float array, zero where the request fell outside.
    """
    arr = _as_2d_float(frame)
    half = int(half)
    if half <= 0:
        raise ValueError(f"half must be positive, got {half!r}")
    h, w = arr.shape
    cx, cy = int(center[0]), int(center[1])
    out = np.zeros((2 * half, 2 * half), dtype=np.float64)
    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x1, w), min(y1, h)
    if sx1 <= sx0 or sy1 <= sy0:
        return out
    out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = arr[sy0:sy1, sx0:sx1]
    return out


def roi_l2(a: np.ndarray, b: np.ndarray) -> float:
    """L2 distance between two same-shape frames or crops.

    Used as the drift observable: on a stable bench two consecutive flat reads
    differ only by noise, whereas an unsettled panel or a drifting laser gives
    a much larger value. Peak intensity is *not* a reliable observable here
    (it is not reproducible run to run); a region sum or norm is.
    """
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    return float(np.linalg.norm(x - y))


def flat_to_flat_floor(
    frames: Sequence[np.ndarray],
) -> tuple[float, list[float]]:
    """Drift floor from consecutive flat reads.

    Args:
        frames: At least two consecutive flat-field frames.

    Returns:
        ``(median, series)`` where ``series`` holds every consecutive
        ``roi_l2`` difference.

    Raises:
        ValueError: If fewer than two frames are given.
    """
    if len(frames) < 2:
        raise ValueError("need at least two frames to estimate a drift floor")
    series = [roi_l2(frames[i], frames[i + 1]) for i in range(len(frames) - 1)]
    return float(np.median(series)), series


# ---------------------------------------------------------------------------
# Settle / drift characterisation
# ---------------------------------------------------------------------------


def settle_time_s(
    deltas: Sequence[float],
    times_s: Sequence[float],
    *,
    frac: float = 0.10,
    run: int = 3,
) -> float | None:
    """First time the settle curve stays within ``frac`` of its final value.

    Replaces "sleep a fixed duration and hope". A fixed wait is invalid on this
    bench because the driver's flip-time estimate under-reports (it reports
    0.0 ms for two phases with similar grey statistics), so the same ramp read
    43.2 px FWHM immediately and 12.8 px three seconds later.

    Args:
        deltas: Observable (e.g. a norm) at each sample.
        times_s: Sample times, seconds.
        frac: Tolerance as a fraction of the final value.
        run: Number of consecutive samples that must all be within tolerance.

    Returns:
        The settle time, or ``None`` if it never settles.

    Raises:
        ValueError: If the two sequences differ in length.
    """
    d = np.asarray(list(deltas), dtype=np.float64)
    t = np.asarray(list(times_s), dtype=np.float64)
    if d.shape != t.shape:
        raise ValueError("deltas and times_s must have the same length")
    if d.size == 0:
        return None
    tol = abs(float(frac)) * abs(float(d[-1]))
    if tol <= 0.0:
        # A zero-valued curve is trivially settled from the first sample.
        return float(t[0]) if abs(float(d[0])) <= 0.0 else None
    for i in range(d.size):
        window = d[i : i + int(run)]
        if window.size < int(run):
            break
        if bool(np.all(np.abs(window - d[-1]) <= tol)):
            return float(t[i])
    return None


def snr_vs_averages(
    signal_norms: Sequence[float],
    floor: float,
    ks: Sequence[int],
) -> dict[int, float]:
    """SNR at K-frame averages against a measured noise floor.

    Useful as a *diagnostic*: if averaging K frames does not improve the SNR,
    the residual is not independent read noise but drift, and averaging longer
    will not help. That distinction decides whether a bench needs a better
    exposure or a better settle protocol.

    Args:
        signal_norms: Observable for each of K identical repeats.
        floor: Single-frame noise floor (same observable, same units).
        ks: Averaging factors to report.

    Returns:
        ``{K: snr}``.

    Raises:
        ValueError: If ``floor`` is not positive.
    """
    f = float(floor)
    if not np.isfinite(f) or f <= 0.0:
        raise ValueError(f"floor must be finite and positive, got {floor!r}")
    x = np.asarray(list(signal_norms), dtype=np.float64)
    out: dict[int, float] = {}
    for k in ks:
        kk = int(k)
        if kk < 1:
            raise ValueError(f"K must be >= 1, got {k!r}")
        mean = float(x[:kk].mean()) if x.size >= kk else float(x.mean())
        out[kk] = mean / (f / np.sqrt(kk))
    return out


def exposure_monotonicity(
    exposures_ms: Sequence[float],
    peaks: Sequence[float],
    *,
    rel_tol: float = 0.02,
    saturation_level: float | None = None,
) -> dict[str, object]:
    """Check that peak brightness rises monotonically with exposure.

    Monotonicity is the cheap way to prove the camera is not being pushed past
    its linear range and that the laser is not drifting across the bracket.

    Args:
        exposures_ms: Exposure settings, ascending.
        peaks: Measured peak per exposure.
        rel_tol: Allowed relative shortfall on each step.
        saturation_level: Detector full-scale value; flags saturation when a
            peak reaches it.

    Returns:
        Dict with ``verdict``, ``ratios``, ``expected`` and ``saturated``.

    Raises:
        ValueError: If the two sequences differ in length.
    """
    e = np.asarray(list(exposures_ms), dtype=np.float64)
    p = np.asarray(list(peaks), dtype=np.float64)
    if e.shape != p.shape:
        raise ValueError("exposures_ms and peaks must have the same length")
    if e.size < 2:
        return {
            "verdict": "insufficient_data",
            "ratios": [],
            "expected": [],
            "saturated": False,
        }

    with np.errstate(divide="ignore", invalid="ignore"):
        expected = e[1:] / e[:-1]
        ratios = p[1:] / p[:-1]
    ok = ratios >= (expected * (1.0 - float(rel_tol)))
    saturated = False
    if saturation_level is not None:
        saturated = bool(np.any(p >= float(saturation_level)))
    return {
        "verdict": "monotonic" if bool(np.all(ok)) and not saturated else
                   "saturated" if saturated else "non_monotonic",
        "ratios": [float(v) for v in ratios],
        "expected": [float(v) for v in expected],
        "saturated": saturated,
    }


# ---------------------------------------------------------------------------
# Pattern construction
# ---------------------------------------------------------------------------


def build_block_pattern(
    coeffs: np.ndarray,
    grid: int,
    panel_shape: tuple[int, int],
) -> np.ndarray:
    """Tile a ``grid x grid`` coefficient map onto the panel as blocks.

    Matches the freeform SPGD basis used by
    ``slm_square_shaping._freeform_phase_radians``: coefficients are
    block-replicated, not applied per pixel, so one DOF covers an 80x50 px SLM
    region. A 24x24 grid on a 1200x1920 panel therefore yields 5x8 blocks.

    Args:
        coeffs: ``grid*grid`` coefficients, row-major.
        grid: Grid edge length.
        panel_shape: ``(height, width)`` of the SLM panel.

    Returns:
        ``panel_shape`` float64 phase array (raw radians, unwrapped).

    Raises:
        ValueError: If ``grid`` is not positive or ``coeffs`` is the wrong size.
    """
    g = int(grid)
    if g <= 0:
        raise ValueError(f"grid must be positive, got {grid!r}")
    c = np.asarray(coeffs, dtype=np.float64).ravel()
    if c.size != g * g:
        raise ValueError(
            f"coeffs must hold grid*grid = {g * g} values, got {c.size}"
        )
    ph, pw = int(panel_shape[0]), int(panel_shape[1])
    blocks = c.reshape(g, g)
    bh, bw = ph // g, pw // g
    if bh < 1 or bw < 1:
        raise ValueError(
            f"grid {g} is too coarse for panel {panel_shape!r}"
        )
    tiled = np.kron(blocks, np.ones((bh, bw), dtype=np.float64))
    # np.kron stops at the first g*bh rows / g*bw cols; pad if the panel is
    # not an exact multiple of the grid.
    if tiled.shape != (ph, pw):
        padded = np.zeros((ph, pw), dtype=np.float64)
        padded[: tiled.shape[0], : tiled.shape[1]] = tiled
        return padded
    return tiled