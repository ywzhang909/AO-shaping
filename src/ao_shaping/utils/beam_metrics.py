"""Beam-shaping quality metrics and pattern normalization helpers.

Pure NumPy functions (no hardware access / no SLM / no camera). These were
migrated from :mod:`ao_shaping.algorithm.beam_shaping_utils` so that metrics
are computed in the leaf ``utils`` layer with a single source of truth.
New code should import from here; the old module re-exports the names for
backward compatibility.

- Spot / beam measurement on a 2D intensity map.
- Mask-aware shaping metrics (uniformity CV, encircled energy).
- Square-beam metrics + combined quality score.
- Pattern normalization / amplitude conversion.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from loguru import logger

from ao_shaping.utils.spots_calc import centroid, radius

__all__ = [
    "intensity_to_amplitude",
    "normalize_pattern",
    "measure_spot_diameter_cam",
    "compute_metrics",
    "compute_shaping_metrics",
    "compute_square_metrics",
    "compute_quality_score",
    "measure_bright_span",
    "clamp_side",
]


def intensity_to_amplitude(
    intensity: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Convert an intensity pattern to amplitude.

    Args:
        intensity: 2D intensity array.
        normalize: If True, normalize the amplitude to ``[0, 1]`` by the
            maximum value.

    Returns:
        Float32 amplitude array.
    """
    amp = np.sqrt(np.asarray(intensity, dtype=np.float64))
    if normalize:
        amax = float(amp.max())
        if amax > 0:
            amp = amp / amax
    return amp.astype(np.float32)


def normalize_pattern(
    pattern: np.ndarray,
    mode: Literal["peak", "sum"] = "peak",
) -> np.ndarray:
    """Normalize a 2D pattern to ``[0, 1]`` (peak) or unit total energy.

    Args:
        pattern: 2D input array.
        mode: ``"peak"`` scales so the maximum value is 1.
            ``"sum"`` scales so the total sum is 1.

    Returns:
        Float32 normalized array with the same shape as ``pattern``.

    Raises:
        ValueError: If ``mode`` is not recognized.
    """
    pattern = np.asarray(pattern, dtype=np.float64)
    pattern = np.nan_to_num(pattern, nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "peak":
        pmax = float(pattern.max())
        if pmax > 0:
            pattern = pattern / pmax
    elif mode == "sum":
        total = float(pattern.sum())
        if total > 0:
            pattern = pattern / total
    else:
        raise ValueError(f"Unknown normalize mode: {mode}")
    return pattern.astype(np.float32)


def measure_spot_diameter_cam(intensity: np.ndarray, energy: float = 0.90) -> float:
    """Measure far-field beam spot diameter (pixels) from an intensity image.

    Uses the intensity centroid as center and the encircled-energy radius
    (default 90%) to derive a spot diameter.

    Args:
        intensity: 2D far-field intensity image.
        energy: Encircled-energy fraction (0~1) for the radius (default 0.90).

    Returns:
        Spot diameter in camera pixels.
    """
    cx, cy = centroid(intensity, return_float=True)
    r = radius(intensity, center=(cx, cy), energy=energy, use_aotools=False)
    return 2.0 * float(r)


def compute_metrics(
    measured: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    """Compute beam-shaping quality metrics between measured and target.

    Both ``measured`` and ``target`` are expected to be 2D intensity or
    amplitude maps of the same shape. They are normalized to ``[0, 1]``
    before comparison so that absolute scale does not matter (the SLM+CCD
    pipeline has an unknown absolute gain).

    Args:
        measured: 2D measured intensity/amplitude map.
        target: 2D target intensity/amplitude map (same shape).

    Returns:
        Dict with ``"mse"`` (normalized intensity MSE), ``"correlation"``
        (Pearson correlation of the flattened maps), and ``"efficiency"``
        (overlap energy ratio).

    Raises:
        ValueError: If shapes do not match.
    """
    m = np.asarray(measured, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if m.shape != t.shape:
        raise ValueError(f"Shape mismatch: measured {m.shape} vs target {t.shape}")

    # Normalize both to unit sum (energy) for scale-invariant comparison
    m_sum = float(m.sum())
    t_sum = float(t.sum())
    if m_sum > 0:
        m = m / m_sum
    if t_sum > 0:
        t = t / t_sum

    mse = float(np.mean((m - t) ** 2))

    # Pearson correlation (guard against zero variance)
    mf, tf = m.flatten(), t.flatten()
    if mf.std() > 1e-8 and tf.std() > 1e-8:
        corr = float(np.corrcoef(mf, tf)[0, 1])
    else:
        corr = 1.0 if np.allclose(mf, tf) else 0.0

    # Efficiency: symmetric overlap in [0, 1]
    total = float(m.sum() + t.sum())
    efficiency = (
        float(2.0 * float(np.minimum(m, t).sum()) / total) if total > 0 else 0.0
    )

    return {"mse": mse, "correlation": corr, "efficiency": efficiency}


def compute_shaping_metrics(
    intensity: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """Compute mask-aware beam-shaping quality metrics.

    Complements :func:`compute_metrics` with uniformity / encircled-energy
    statistics evaluated only inside a boolean target mask.

    Args:
        intensity: 2D intensity array.
        mask: Boolean 2D array of the same shape as ``intensity`` defining
            the target region. If the mask is larger than ``intensity`` it is
            clipped (centered) to the intensity bounds.

    Returns:
        Dict with ``"uniformity_cv"`` (std/mean of the intensity within the
        mask — 0 for a perfectly flat region), ``"encircled_energy"``
        (``sum(intensity[mask]) / sum(intensity)``), ``"peak"`` (max
        intensity, unnormalized) and ``"in_mask_mean"`` (mean intensity inside
        the mask). All values are ``0.0`` when the mask is empty or the total
        intensity is zero (never NaN/inf).

    Raises:
        ValueError: If ``intensity`` or ``mask`` is not a 2D array.
    """
    intensity = np.asarray(intensity, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if intensity.ndim != 2 or mask.ndim != 2:
        raise ValueError("intensity and mask must both be 2D arrays")

    # Clip the mask region to the intensity bounds (centered overlap).
    if mask.shape != intensity.shape:
        h, w = intensity.shape
        mh, mw = mask.shape
        y0 = max((mh - h) // 2, 0)
        x0 = max((mw - w) // 2, 0)
        y1 = min(y0 + h, mh)
        x1 = min(x0 + w, mw)
        clipped = mask[y0:y1, x0:x1]
        padded = np.zeros((h, w), dtype=bool)
        py0 = (h - clipped.shape[0]) // 2
        px0 = (w - clipped.shape[1]) // 2
        padded[py0 : py0 + clipped.shape[0], px0 : px0 + clipped.shape[1]] = clipped
        mask = padded

    n_mask = int(mask.sum())
    total = float(intensity.sum())
    if n_mask == 0 or total <= 0:
        return {
            "uniformity_cv": 0.0,
            "encircled_energy": 0.0,
            "peak": 0.0,
            "in_mask_mean": 0.0,
        }

    in_mask = intensity[mask]
    in_mask_mean = float(in_mask.mean())
    in_mask_std = float(in_mask.std())
    uniformity_cv = in_mask_std / in_mask_mean if in_mask_mean > 0 else 0.0
    encircled_energy = float(in_mask.sum()) / total

    return {
        "uniformity_cv": float(uniformity_cv),
        "encircled_energy": float(encircled_energy),
        "peak": float(intensity.max()),
        "in_mask_mean": in_mask_mean,
    }


def compute_square_metrics(
    intensity: np.ndarray,
    target_side: int,
    center: tuple[float, float],
    energy: float = 0.90,
) -> dict[str, float]:
    """Compute quality metrics for a square beam.

    Args:
        intensity: 2D far-field intensity image.
        target_side: Target square side length (pixels).
        center: Beam center ``(cx, cy)``.
        energy: Encircled-energy fraction (default 0.90).

    Returns:
        Dict with ``aspect_ratio``, ``squareness``, ``uniformity_cv``,
        ``encircled_energy``, ``flatness_factor``, ``intensity_max``,
        ``intensity_mean``.
    """
    intensity = np.asarray(intensity, dtype=np.float64)
    total = float(np.sum(intensity))
    if total <= 0:
        return {
            "aspect_ratio": 1.0,
            "squareness": 0.0,
            "uniformity_cv": 0.0,
            "encircled_energy": 0.0,
            "flatness_factor": 0.0,
            "intensity_max": 0.0,
            "intensity_mean": 0.0,
        }

    cx, cy = int(round(center[0])), int(round(center[1]))
    h, w = intensity.shape

    peak = float(np.max(intensity))
    threshold = 0.5 * peak
    bright = intensity >= threshold
    if bright.any():
        ys, xs = np.nonzero(bright)
        width_bright = int(xs.max()) - int(xs.min()) + 1
        height_bright = int(ys.max()) - int(ys.min()) + 1
        aspect_ratio = max(width_bright, height_bright) / max(
            min(width_bright, height_bright), 1
        )
    else:
        aspect_ratio = 1.0

    half = max(target_side // 2, 1)
    y0 = max(cy - half, 0)
    y1 = min(cy + half, h)
    x0 = max(cx - half, 0)
    x1 = min(cx + half, w)
    region = intensity[y0:y1, x0:x1]
    region_mean = float(np.mean(region))
    region_std = float(np.std(region))
    uniformity_cv = region_std / max(region_mean, 1e-10) if region_mean > 0 else 0.0

    encircled_energy = float(np.sum(region)) / max(total, 1e-10)
    flatness_factor = region_mean / max(peak, 1e-10) if peak > 0 else 0.0

    return {
        "aspect_ratio": float(aspect_ratio),
        "squareness": float(abs(1.0 - aspect_ratio)),
        "uniformity_cv": float(uniformity_cv),
        "encircled_energy": float(encircled_energy),
        "flatness_factor": float(flatness_factor),
        "intensity_max": float(peak),
        "intensity_mean": float(np.mean(intensity)),
    }


def compute_quality_score(metrics: dict[str, float]) -> float:
    """Compute a combined quality score from square metrics.

    Weighted combination of aspect ratio, uniformity, and encircled energy.

    Args:
        metrics: Dict returned by :func:`compute_square_metrics`.

    Returns:
        Float quality score in ``[0, 1]`` (higher is better).
    """
    f_ar = math.exp(-(((metrics["aspect_ratio"] - 1.0) / 0.3) ** 2))
    f_uni = math.exp(-((metrics["uniformity_cv"] / 0.3) ** 2))
    f_ee = float(np.clip(metrics["encircled_energy"], 0.0, 1.0))
    return float(0.3 * f_ar + 0.4 * f_uni + 0.3 * f_ee)


def measure_bright_span(
    intensity: np.ndarray,
    peak_frac: float = 0.5,
) -> tuple[int, int]:
    """Return the ``(width, height)`` of the bright region above ``peak_frac``.

    Args:
        intensity: 2D intensity image.
        peak_frac: Threshold as a fraction of the peak (default 0.5).

    Returns:
        ``(width, height)`` of the bounding box; ``(0, 0)`` if no bright
        region is found.
    """
    intensity = np.asarray(intensity, dtype=np.float64)
    peak = float(np.max(intensity))
    if peak <= 0:
        return 0, 0
    bright = intensity >= peak_frac * peak
    if not bright.any():
        return 0, 0
    ys, xs = np.nonzero(bright)
    return int(xs.max()) - int(xs.min()) + 1, int(ys.max()) - int(ys.min()) + 1


def clamp_side(side: int, height: int, width: int, margin: int = 8) -> int:
    """Clamp a square side length to fit inside a grid with margin.

    Args:
        side: Requested side length.
        height: Grid height.
        width: Grid width.
        margin: Minimum margin from each edge (default 8).

    Returns:
        Clamped side length.
    """
    max_side = min(height, width) - 2 * margin
    if side > max_side:
        logger.warning(
            "Square side {}px exceeds grid limit, clamped to {}px", side, max_side
        )
        return max(1, max_side)
    return max(1, side)