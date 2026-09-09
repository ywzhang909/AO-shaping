"""Shared beam-shaping utilities for SLM+CCD closed-loop runners.

This module centralizes helpers that are common to multiple beam-shaping
runners (e.g. ``gs_hologram_runner`` and ``diff_beam_runner``):

- Target intensity pattern generation (gaussian / circle / ...).
- Loading a target image from disk and normalizing it.
- Capturing a far-field image from a CCD and converting to amplitude.
- Converting an SLM phase pattern (radians) to a uint16 grayscale pattern.
- Computing beam-shaping quality metrics (loss / correlation / efficiency).
- Shared physical constants (wavelength, SLM pixel size, propagation distance).

Keeping these in one place avoids duplicating the same code across the
runners and makes the metric definitions consistent so that results from the
differentiable (backprop) and GS algorithms are directly comparable.

All functions are pure (no hardware access) except ``capture_amplitude`` and
``compute_metrics`` which take already-fetched arrays; hardware I/O is
handled by the callers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence, Tuple

import numpy as np
from loguru import logger

if TYPE_CHECKING:  # pragma: no cover
    from ao_shaping.drivers.ccd import BaseCamera


# ---------------------------------------------------------------------------
# Shared physical / SLM constants
# ---------------------------------------------------------------------------
DEFAULT_WAVELENGTH: float = 1064e-9      # 1064 nm YAG laser (meters)
DEFAULT_SLM_PIXEL_SIZE: float = 8e-6     # SLM pixel pitch (meters)
DEFAULT_DISTANCE: float = 0.1            # default propagation distance (meters)
DEFAULT_MAX_GRAYSCALE: int = 1023        # grayscale value corresponding to 2*pi


def parse_tuple(value: str) -> tuple[int, int]:
    """Parse a string like ``"h,w"`` into a ``(int, int)`` tuple.

    Args:
        value: Comma separated string, e.g. ``"320,320"``.

    Returns:
        A 2-tuple of integers.

    Raises:
        ValueError: If the string does not contain exactly two integers.
    """
    parts = value.split(",")
    if len(parts) != 2:
        raise ValueError(f"Expected 'h,w' format, got '{value}'")
    return int(parts[0]), int(parts[1])


# ---------------------------------------------------------------------------
# Target pattern generation
# ---------------------------------------------------------------------------
def create_target_shape(
    shape: Literal["gaussian", "circle"],
    size: int,
    radius_ratio: float = 0.3,
) -> np.ndarray:
    """Create a 2D target intensity pattern.

    Args:
        shape: Pattern type, ``"gaussian"`` or ``"circle"``.
        size: Side length of the square output array (height == width).
        radius_ratio: Fraction of half-size defining the pattern radius
            (used for both gaussian sigma scaling and circle radius).

    Returns:
        Float32 array of shape ``(size, size)`` with values in ``[0, 1]``.

    Raises:
        ValueError: If ``shape`` is not recognized.
    """
    y, x = np.mgrid[0:size, 0:size]
    cx, cy = (size - 1) / 2, (size - 1) / 2
    radius = radius_ratio * size / 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    if shape == "gaussian":
        sigma = radius / 2
        pattern = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
    elif shape == "circle":
        pattern = (r <= radius).astype(np.float32)
    else:
        raise ValueError(f"Unknown shape: {shape}. Use 'gaussian' or 'circle'.")

    return (pattern / pattern.max()).astype(np.float32) if pattern.max() > 0 else pattern


def load_target_image(path: str | Path) -> np.ndarray:
    """Load a grayscale image as a normalized float target intensity.

    The file may be a ``.npy`` (array) or an image (``.png``/``.jpg``/...).
    The result is normalized to ``[0, 1]`` and cast to float32.

    Args:
        path: File path to the target image or ``.npy`` array.

    Returns:
        Float32 2D array with values in ``[0, 1]``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Target image not found: {path}")

    if path.suffix.lower() == ".npy":
        target = np.load(path)
    else:
        try:
            from skimage import io  # type: ignore
            target = io.imread(str(path))
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "scikit-image required to load image targets. "
                "Install with: pip install scikit-image"
            ) from exc

    if target.ndim > 2:
        target = target[..., 0] if target.shape[-1] > 0 else target.mean(axis=-1)
    target = target.astype(np.float32)
    tmax = target.max()
    if tmax > 0:
        target = target / tmax
    return target


# ---------------------------------------------------------------------------
# SLM phase -> grayscale conversion
# ---------------------------------------------------------------------------
def phase_to_slm_grayscale(
    phase: np.ndarray,
    max_grayscale: int = DEFAULT_MAX_GRAYSCALE,
) -> np.ndarray:
    """Convert a phase map (radians) to an SLM uint16 grayscale map.

    Wraps phase into ``[0, 2*pi)`` and scales to the ``[0, max_grayscale]``
    range. Values are clamped to the grayscale range and cast to ``uint16``.

    Args:
        phase: 2D phase array in radians.
        max_grayscale: The grayscale value that corresponds to ``2*pi``
            (default 1023 for a 10-bit LCOS device).

    Returns:
        ``uint16`` 2D grayscale array.
    """
    phase = np.asarray(phase, dtype=np.float32)
    phase = np.mod(phase, 2 * np.pi)
    gray = (phase / (2 * np.pi)) * max_grayscale
    return np.clip(gray, 0, max_grayscale).astype(np.uint16)


# ---------------------------------------------------------------------------
# CCD far-field capture
# ---------------------------------------------------------------------------
def capture_amplitude(
    camera: "BaseCamera",
    center: Sequence[int] | None = None,
    size: Sequence[int] | None = None,
    n_sample: int = 1,
) -> np.ndarray:
    """Capture the far-field intensity from a CCD and return the amplitude.

    If ``center``/``size`` are provided, the camera window is reset to that
    region first (matching the target pattern's footprint). The captured
    intensity (uint16) is converted to float amplitude ``sqrt(I)`` and
    normalized to ``[0, 1]``.

    Args:
        camera: An open CCD camera object (must expose ``get_numpy_image``
            and optionally ``reset_window``).
        center: ``(cx, cy)`` window center in pixels (None to leave as-is).
        size: ``(h, w)`` window size in pixels (None to leave as-is).
        n_sample: Number of frames to average.

    Returns:
        Float32 2D amplitude array, normalized to ``[0, 1]``.
    """
    if center is not None and size is not None:
        try:
            camera.reset_window(center, size)
        except Exception:  # pragma: no cover - driver may not support
            logger.debug("Camera does not support reset_window; using full frame")

    img = camera.get_numpy_image(n_sample=n_sample, skip_first=True)
    intensity = np.asarray(img, dtype=np.float32)
    intensity = np.nan_to_num(intensity, nan=0.0, posinf=0.0, neginf=0.0)
    amp = np.sqrt(intensity)
    amax = amp.max()
    if amax > 0:
        amp = amp / amax
    return amp.astype(np.float32)


# ---------------------------------------------------------------------------
# Beam-shaping quality metrics
# ---------------------------------------------------------------------------
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
        raise ValueError(
            f"Shape mismatch: measured {m.shape} vs target {t.shape}"
        )

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
    efficiency = float(2.0 * float(np.minimum(m, t).sum()) / total) if total > 0 else 0.0

    return {"mse": mse, "correlation": corr, "efficiency": efficiency}
