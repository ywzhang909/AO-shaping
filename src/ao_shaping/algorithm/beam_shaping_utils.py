"""Shared beam-shaping utilities for SLM+CCD closed-loop runners.

This module centralizes helpers that are common to multiple beam-shaping
runners (e.g. ``gs_hologram_runner`` and ``diff_beam_runner``):

- Target intensity pattern generation (gaussian / circle / square / annular /
  grid / cross).
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

import json
import math
import os
import random
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence, Tuple

import numpy as np
from loguru import logger

from ao_shaping.utils.spots_calc import centroid, radius

if TYPE_CHECKING:  # pragma: no cover
    from ao_shaping.drivers.ccd import BaseCamera


# ---------------------------------------------------------------------------
# Shared physical / SLM constants
# ---------------------------------------------------------------------------
DEFAULT_WAVELENGTH: float = 1064e-9  # 1064 nm YAG laser (meters)
DEFAULT_SLM_PIXEL_SIZE: float = 8e-6  # SLM pixel pitch (meters)
DEFAULT_DISTANCE: float = 0.1  # default propagation distance (meters)
DEFAULT_MAX_GRAYSCALE: int = 1023  # grayscale value corresponding to 2*pi


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
    shape: Literal[
        "gaussian", "circle", "square", "annular", "grid", "cross",
        "rectangle", "pentagon",
    ],
    size: int | tuple[int, int],
    radius_ratio: float = 0.3,
    side: int | None = None,
    as_amplitude: bool = False,
    inner_radius: float | None = None,
    outer_radius: float | None = None,
    nx: int | None = None,
    ny: int | None = None,
    line_width: float = 0.02,
    thickness: float = 0.05,
    aspect_ratio: float = 2.0,
) -> np.ndarray:
    """Create a 2D target intensity or amplitude pattern.

    Args:
        shape: Pattern type. ``"gaussian"``, ``"circle"``, ``"square"``,
            ``"annular"`` (ring), ``"grid"`` (grid lines), ``"cross"``,
            ``"rectangle"`` (aspect-ratio rectangle, long axis horizontal),
            or ``"pentagon"`` (regular pentagon, point-up).
        size: Side length of the square output array, or ``(height, width)``
            tuple for a rectangular grid (e.g. the SLM panel 1200x1920).
        radius_ratio: Fraction of half-size defining the base radius
            (used for gaussian sigma, circle radius, and annular defaults).
        side: Side length in grid pixels. For ``"square"`` it is the full
            side; for ``"rectangle"`` it is the SHORT side (the long side is
            ``side * aspect_ratio``). Defaults to the radius-derived size.
        as_amplitude: If True, return ``sqrt(intensity)`` instead of
            intensity. Useful for Gerchberg-Saxton style algorithms that
            operate on amplitude masks.
        inner_radius: Inner radius of an annular ring (pixels). Defaults to
            ``0.2 * radius``.
        outer_radius: Outer radius of an annular ring (pixels). Defaults to
            ``0.5 * min(height, width) / 2``.
        nx: Number of vertical grid lines. Only used when ``shape == "grid"``.
        ny: Number of horizontal grid lines. Only used when ``shape == "grid"``.
        line_width: Grid line width as a fraction of half-size. Only used when
            ``shape == "grid"`` or ``shape == "cross"``.
        thickness: Cross arm half-width as a fraction of half-size. Only used
            when ``shape == "cross"``.
        aspect_ratio: Width:height ratio of the rectangle (long axis
            horizontal). Only used when ``shape == "rectangle"``.

    Returns:
        Float32 array of shape ``(height, width)`` with values in ``[0, 1]``
        (intensity by default, or amplitude when ``as_amplitude=True``).

    Raises:
        ValueError: If ``shape`` is not recognized.
    """
    if isinstance(size, tuple):
        height, width = int(size[0]), int(size[1])
    else:
        height = width = int(size)

    y, x = np.mgrid[0:height, 0:width]
    cx, cy = (width - 1) / 2, (height - 1) / 2
    radius = radius_ratio * min(height, width) / 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    if shape == "gaussian":
        sigma = radius / 2
        pattern = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
    elif shape == "circle":
        pattern = (r <= radius).astype(np.float32)
    elif shape == "square":
        half = (side if side is not None else int(radius)) / 2.0
        pattern = ((np.abs(x - cx) <= half) & (np.abs(y - cy) <= half)).astype(
            np.float32
        )
    elif shape == "annular":
        inner_r = float(inner_radius if inner_radius is not None else 0.2 * radius)
        outer_r = float(
            outer_radius if outer_radius is not None else 0.5 * min(height, width) / 2
        )
        pattern = ((r >= inner_r) & (r <= outer_r)).astype(np.float32)
    elif shape == "grid":
        nx = int(nx if nx is not None else 5)
        ny = int(ny if ny is not None else 5)
        lw = float(line_width) * min(height, width) / 2
        pattern = np.zeros((height, width), dtype=np.float32)
        for i in range(nx):
            x_pos = cx + (2 * i / max(nx - 1, 1) - 1) * (width / 2)
            pattern[np.abs(x - x_pos) <= lw] = 1.0
        for j in range(ny):
            y_pos = cy + (2 * j / max(ny - 1, 1) - 1) * (height / 2)
            pattern[np.abs(y - y_pos) <= lw] = 1.0
    elif shape == "cross":
        th = float(thickness) * min(height, width) / 2
        pattern = ((np.abs(x - cx) <= th) | (np.abs(y - cy) <= th)).astype(np.float32)
    elif shape == "rectangle":
        # ``side`` is the SHORT side; the long side (horizontal) is
        # ``side * aspect_ratio``. The boolean mask clips to grid bounds.
        short_side = side if side is not None else int(radius)
        half_h = short_side / 2.0
        half_w = short_side * aspect_ratio / 2.0
        pattern = ((np.abs(x - cx) <= half_w) & (np.abs(y - cy) <= half_h)).astype(
            np.float32
        )
    elif shape == "pentagon":
        # Regular pentagon, point-up (one vertex at top), circumradius
        # ``radius``. Vectorized convex-polygon point-in-polygon test over the
        # 5 edges (half-plane checks) — no matplotlib dependency.
        angles = -np.pi / 2 + 2 * np.pi * np.arange(5) / 5
        vx = cx + radius * np.cos(angles)
        vy = cy + radius * np.sin(angles)
        inside = np.ones((height, width), dtype=bool)
        for i in range(5):
            x1, y1 = vx[i], vy[i]
            x2, y2 = vx[(i + 1) % 5], vy[(i + 1) % 5]
            cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
            inside &= cross >= 0
        pattern = inside.astype(np.float32)
    else:
        raise ValueError(
            f"Unknown shape: {shape}. Use 'gaussian', 'circle', 'square', "
            "'annular', 'grid', 'cross', 'rectangle', or 'pentagon'."
        )

    if as_amplitude:
        return np.sqrt(pattern).astype(np.float32)
    return (
        (pattern / pattern.max()).astype(np.float32) if pattern.max() > 0 else pattern
    )


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


def create_target_mask(
    shape: str,
    grid_size: tuple[int, int],
    size: int,
    *,
    sigma: float | None = None,
) -> np.ndarray:
    """Create a normalised target intensity mask.

    Unified target generation helper that supports all shapes used by both the
    differentiable beam-shaping runners (``diff_beam_runner`` /
    ``diff_shaping_runner``). Shapes ``square``, ``circle``, ``gaussian`` and
    ``spot`` are supported.

    Args:
        shape: One of ``"square"``, ``"circle"``, ``"gaussian"``, ``"spot"``.
        grid_size: ``(H, W)`` of the output array.
        size: Characteristic dimension in pixels.
            * square — side length
            * circle — diameter
            * gaussian — ``sigma`` defaults to ``size / 6`` if not given
            * spot — diameter (focused spot, same as circle)
        sigma: Override for the Gaussian standard deviation (pixels).

    Returns:
        ``(H, W)`` ``float64`` mask with values in ``[0, 1]``, centred.
    """
    valid_shapes = {"square", "circle", "gaussian", "spot"}
    if shape not in valid_shapes:
        raise ValueError(f"Invalid shape {shape!r}. Must be one of {valid_shapes}")
    if len(grid_size) != 2:
        raise ValueError(f"grid_size must be a 2-tuple, got {len(grid_size)}D")

    H, W = grid_size
    cy, cx = H / 2.0, W / 2.0
    yy, xx = np.mgrid[0:H, 0:W]

    if shape == "square":
        half = size / 2.0
        mask = ((np.abs(xx - cx) <= half) & (np.abs(yy - cy) <= half)).astype(
            np.float64
        )

    elif shape in {"circle", "spot"}:
        radius = size / 2.0
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        mask = (r2 <= radius**2).astype(np.float64)

    else:  # gaussian
        sig = sigma if sigma is not None else size / 6.0
        mask = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sig**2))
        peak = mask.max()
        if peak > 0:
            mask = mask / peak

    return mask.astype(np.float64)


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
# Square beam shaping (GS): spot measurement -> square sizing -> target
# ---------------------------------------------------------------------------
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


def compute_square_side(
    spot_diameter_cam_px: float,
    factor: float = 1.5,
    p_cam: float = 8e-6,
    d_slm: float = 8e-6,
) -> int:
    """Auto-compute square side (SLM-grid pixels) from the beam spot size.

    The requested physical beam size is ``factor`` times the measured spot
    diameter. If the camera pixel pitch differs from the SLM pixel pitch the
    size is rescaled accordingly::

        side = factor * spot_diameter_cam * (p_cam / d_slm)

    Args:
        spot_diameter_cam_px: Measured spot diameter in camera pixels.
        factor: Square-to-spot size factor (default 1.5).
        p_cam: Camera pixel pitch in meters (default matches SLM pitch).
        d_slm: SLM pixel pitch in meters (default 8e-6).

    Returns:
        Square side length in SLM-grid pixels.
    """
    return int(round(float(factor) * float(spot_diameter_cam_px) * (p_cam / d_slm)))


def build_square_target_amplitude(
    height: int,
    width: int,
    side: int,
) -> np.ndarray:
    """Build a centered square target amplitude on an (height, width) grid.

    Args:
        height: Grid height in pixels.
        width: Grid width in pixels.
        side: Square side length in pixels.

    Returns:
        Float array (height, width) with 1 inside the square, 0 outside.
    """
    target = np.zeros((height, width), dtype=np.float64)
    half = side // 2
    cy = height // 2
    cx = width // 2
    y0 = max(cy - half, 0)
    y1 = min(cy + side - half, height)
    x0 = max(cx - half, 0)
    x1 = min(cx + side - half, width)
    target[y0:y1, x0:x1] = 1.0
    return target


# ---------------------------------------------------------------------------
# Brightness-driven square target (CCD frame -> SLM-grid target)
# ---------------------------------------------------------------------------
def _resize_bilinear(arr: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    """Separable bilinear resize with ``np.interp``-style pixel-center mapping.

    Pure-NumPy replacement for ``skimage.transform.resize`` so the runner does
    not depend on a (fragile on this machine) scikit-image build. Output pixel
    centers are mapped to source coordinates as ``(i + 0.5) * src / dst - 0.5``
    and clamped, matching the standard coordinate convention.
    """
    src_h, src_w = arr.shape
    dst_h, dst_w = out_shape
    if (src_h, src_w) == (dst_h, dst_w):
        return arr.copy()

    ys = (np.arange(dst_h) + 0.5) * src_h / dst_h - 0.5
    xs = (np.arange(dst_w) + 0.5) * src_w / dst_w - 0.5
    ys = np.clip(ys, 0.0, src_h - 1.0)
    xs = np.clip(xs, 0.0, src_w - 1.0)

    y0 = np.floor(ys).astype(np.intp)
    y1 = np.minimum(y0 + 1, src_h - 1)
    wy = (ys - y0)[:, None]
    interp_rows = arr[y0, :] * (1.0 - wy) + arr[y1, :] * wy

    x0 = np.floor(xs).astype(np.intp)
    x1 = np.minimum(x0 + 1, src_w - 1)
    wx = (xs - x0)[None, :]
    return interp_rows[:, x0] * (1.0 - wx) + interp_rows[:, x1] * wx


def crop_resize_to_grid(
    frame: np.ndarray,
    grid_h: int = 1200,
    grid_w: int = 1920,
) -> np.ndarray:
    """Crop a CCD frame to the SLM grid aspect ratio and resize to the grid.

    The 0-order spot is located as the frame global argmax (never assumed to
    be the geometric frame center). The crop is the largest centered rectangle
    with aspect ratio ``grid_h/grid_w`` that fits in the frame, zero-padded if
    it extends past the frame bounds, then resized to the full SLM grid.
    """
    frame = np.asarray(frame, dtype=np.float32)
    frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
    frame_h, frame_w = frame.shape
    cy, cx = np.unravel_index(np.argmax(frame), frame.shape)

    crop_h = min(frame_h, round(frame_w * grid_h / grid_w))
    crop_w = min(round(crop_h * grid_w / grid_h), frame_w)

    y0 = cy - crop_h // 2
    x0 = cx - crop_w // 2
    y1 = y0 + crop_h
    x1 = x0 + crop_w

    y0c = max(y0, 0)
    y1c = min(y1, frame_h)
    x0c = max(x0, 0)
    x1c = min(x1, frame_w)

    crop = frame[y0c:y1c, x0c:x1c]
    if crop.shape != (crop_h, crop_w):
        pad_top = y0c - y0
        pad_bottom = (crop_h - crop.shape[0]) - pad_top
        pad_left = x0c - x0
        pad_right = (crop_w - crop.shape[1]) - pad_left
        crop = np.pad(
            crop,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
        )

    return _resize_bilinear(crop, (grid_h, grid_w)).astype(np.float32)


def square_target_from_measurement(
    frame: np.ndarray,
    side_px: float,
    grid_h: int = 1200,
    grid_w: int = 1920,
) -> tuple[np.ndarray, dict]:
    """Build a uniform-square grid target with a FIXED side in CCD space.

    Exposure/brightness invariant by construction:

    1. Background = 10th percentile of the frame, subtracted (signal stats).
    2. Beam center located at the **frame global maximum (argmax)** of the
       signal — the 0-order spot. This is the project rule (AGENTS.md: locate
       the 0-order by ``argmax``, never by geometry or centroid), and is the
       only method that survives the stray-light halo on the full 2592x1944
       Daheng frame (measured: intensity centroid dragged 150-450 px off;
       argmax stable within ~4 px across runs). Anchors on the *current*
       0-order spot, never on the frame center.
    3. A uniform CCD-space square of **fixed** side ``side_px`` camera pixels
       is centered on the centroid with value ``1 / n_pixels`` (n_pixels =
       actual filled pixel count, so the square sums to exactly 1 regardless
       of sub-pixel centroid alignment).
    4. The CCD-space square is mapped to the SLM grid via
       ``crop_resize_to_grid`` and normalized to ``[0, 1]``.

    Evaluation is exposure-invariant: compare ``frame / frame.sum()`` against
    ``info["target_ccd"]`` (sum == 1) — both sides are normalized, so changing
    exposure or total brightness never changes the target square.

    Args:
        frame: Raw far-field CCD frame of the current beam (flat phase).
        side_px: Fixed square side length in camera pixels (e.g. 20 for a
            20×20 px square on the CCD).
        grid_h: SLM grid height (pixels).
        grid_w: SLM grid width (pixels).

    Returns:
        ``(target_intensity, info)`` — grid-space float32 target in ``[0, 1]``
        for the optimization algorithms, plus a dict: ``side_cam_px`` (fixed
        side in CCD px), ``target_ccd`` (CCD-space float32 normalized square,
        sum == 1, same shape as ``frame`` — use as the evaluation target),
        ``n_pixels`` (actual filled pixel count), ``max_brightness``,
        ``background``, ``total_intensity``, ``centroid``
        ``[cy, cx]`` and ``side_grid_bins`` ``[rows, cols]``.

    Raises:
        ValueError: If no usable signal remains after background subtraction
            or ``side_px`` is not positive.
    """
    frame = np.asarray(frame, dtype=np.float32)
    frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
    frame_h, frame_w = frame.shape
    if side_px <= 0:
        raise ValueError("side_px 必须 > 0")

    bg = float(np.percentile(frame, 10))
    signal = np.clip(frame - bg, 0.0, None)

    total = float(signal.sum())
    smax = float(signal.max())
    if smax <= 0 or total <= 0:
        raise ValueError("实测帧无有效信号 (去背景后总和/峰值 <= 0) — 请检查曝光或光束")

    # Center: 0-order spot located by frame GLOBAL MAX (argmax), per the
    # project rule (AGENTS.md): "0-order = frame global max, never assume it
    # sits at the frame/ROI center". Hardware-verified 2026-09-10 on the
    # 2f bench: the real spot sits at argmax (row,col) ≈ (705,1781) / (701,1780)
    # (two runs, stable), while the intensity centroid is dragged 150-450 px
    # away by pervasive stray light (6.9-7.3% of pixels nonzero, halo energy
    # >> spot) and the corner-max binary centroid drifts 26 px+ at higher
    # exposure. argmax is the only method that pins the true spot reliably.
    #
    # NOTE: do NOT use the full-frame intensity centroid here. Measured on
    # this bench it lands hundreds of px away from the real spot because the
    # constant-offset background removal (10th percentile) cannot subtract the
    # *spatially structured* stray-light halo, whose total energy dominates
    # the moments. The Runner's per-frame recorder calls this same function
    # indirectly via ``centroid`` only for *logging* — do not confuse the log
    # value with the actual spot location.
    #
    # 2026-09-11 flat-phase frame (1200 us, full 2592x1944, saved as
    # data/diff_beam/flat_tiff_20260911/flat_1200us.tiff/.npy) — quantitative
    # reason the centroid fails and why argmax survives:
    #   * 96.37% of pixels are 0; the remaining 3.51% are a *single gray step*
    #     of value 1 spread over EVERY row and column, carrying 62.46% of the
    #     frame energy. Median == 0, so any median/percentile-based background
    #     subtraction leaves this layer untouched and the intensity centroid is
    #     dragged to (834,1578) vs true spot argmax (689,1776).
    #   * Threshold-sweep centroid: for t>=2 the centroid of ``img > t``
    #     instantly converges to (694-695, 1776-1778) and stays within ~1 px up
    #     to t=200. I.e. masking ONLY the gray==1 layer is enough to make any
    #     intensity centroid accurate — but argmax needs no threshold choice at
    #     all, so it remains the robust default.
    #   * corner-max binary centroid fails here too: the gray==1 layer reaches
    #     the frame corners, so a corner-derived threshold (1.0) cannot filter
    #     it and the centroid still lands (715,1834).
    cy, cx = np.unravel_index(np.argmax(signal), signal.shape)
    cy, cx = int(cy), int(cx)

    # Fixed-side square in *camera* space, centered at the measured centroid.
    # Value = 1 / n_pixels (actual filled pixel count): the square sums to
    # exactly 1 (normalized target) no matter how the sub-pixel centroid maps
    # onto the pixel grid. Exposure/brightness changes only scale the raw
    # frame, never the normalized target.
    side = float(np.clip(side_px, 1.0, min(frame_h, frame_w)))

    half = side / 2.0
    y0 = max(0, int(np.floor(cy - half)))
    y1 = min(frame_h, int(np.ceil(cy + half)))
    x0 = max(0, int(np.floor(cx - half)))
    x1 = min(frame_w, int(np.ceil(cx + half)))
    square_ccd = np.zeros_like(frame)
    n_pixels = (y1 - y0) * (x1 - x0)
    if n_pixels > 0:
        square_ccd[y0:y1, x0:x1] = 1.0 / n_pixels

    target = crop_resize_to_grid(square_ccd, grid_h, grid_w)
    tmax = target.max()
    if tmax > 0:
        target = target / tmax
    target = target.astype(np.float32)

    crop_h = min(frame_h, round(frame_w * grid_h / grid_w))
    crop_w = round(crop_h * grid_w / grid_h)
    info = {
        "side_cam_px": side,
        "target_ccd": square_ccd.astype(np.float32),
        "n_pixels": n_pixels,
        "max_brightness": smax,
        "background": bg,
        "total_intensity": total,
        "centroid": [cy, cx],
        "side_grid_bins": [side * grid_h / crop_h, side * grid_w / crop_w],
    }
    return target, info


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
            # Sequences may arrive as lists; the driver expects 2-tuples.
            camera.reset_window(
                (int(center[0]), int(center[1])),
                (int(size[0]), int(size[1])),
            )
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


# ---------------------------------------------------------------------------
# Square beam quality metrics (used by diff_shaping_runner)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# SLM memory-slot rotation (shared by diff_beam_runner and diff_shaping_runner)
# ---------------------------------------------------------------------------
_SLOT_MIN: int = 2
_SLOT_MAX: int = 125
_last_slm_slot: int | None = None


def pick_slm_slot(slm: Any) -> int:
    """Pick a random SLM memory slot in ``[_SLOT_MIN, _SLOT_MAX]`` that differs
    from the last used slot.

    On first call the currently displayed slot is read from the device so the
    rotation survives process restarts (memory mode only).

    Args:
        slm: Open SLM device object exposing ``get_displayed_memory_number``.

    Returns:
        Slot number in ``[2, 125]``.
    """
    global _last_slm_slot
    if _last_slm_slot is None:
        try:
            _last_slm_slot = slm.get_displayed_memory_number()
            logger.info("SLM当前显示槽: {}", _last_slm_slot)
        except Exception as exc:
            logger.debug("读取 SLM 当前显示槽失败: {}", exc)
            _last_slm_slot = None
    candidates = [s for s in range(_SLOT_MIN, _SLOT_MAX + 1) if s != _last_slm_slot]
    slot = random.choice(candidates)
    _last_slm_slot = slot
    return slot


def display_phase(slm: Any, phase_rad: np.ndarray, settle_time_s: float) -> None:
    """Display a radian phase pattern on the SLM using memory-slot rotation.

    Uses ``create_phase_from_array`` (device-authentic 2π conversion +
    correction/LUT) and memory mode only (never DVI).

    Args:
        slm: Open SLM device.
        phase_rad: 2D phase array in radians.
        settle_time_s: Wait time after ``display_memory``.
    """
    gray = slm.create_phase_from_array(phase_rad)
    slot = pick_slm_slot(slm)
    slm.write_phase(gray, memory_number=slot)
    time.sleep(0.05)
    slm.display_memory(slot)
    time.sleep(settle_time_s)


# ---------------------------------------------------------------------------
# Frame recording (shared by diff_beam_runner and diff_shaping_runner)
# ---------------------------------------------------------------------------
_frames_dir: Path | None = None
_frame_counter: int = 0


def init_frame_recording(out_dir: Path) -> None:
    """Create the ``frames/`` directory under ``out_dir`` and reset the counter.

    Args:
        out_dir: Result directory that will contain the ``frames/`` subdir.
    """
    global _frames_dir, _frame_counter
    _frames_dir = Path(out_dir) / "frames"
    _frames_dir.mkdir(parents=True, exist_ok=True)
    _frame_counter = 0


def save_frame_png(frame: np.ndarray, path: Path, title: str) -> None:
    """Render one CCD frame to a PNG (inferno colormap + colorbar).

    Args:
        frame: 2D array to render.
        path: Output PNG path.
        title: Figure title.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4.5))
        im = ax.imshow(np.asarray(frame), cmap="inferno")
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(path, dpi=100)
        plt.close(fig)
    except Exception:  # pragma: no cover - plotting must never break the loop
        logger.warning("PNG 保存失败: {}", path.name)


def record_frame(
    raw: np.ndarray,
    phase_desc: str,
    exposure_ms: float,
) -> dict:
    """Save one raw CCD frame plus a JSONL meta line; log per-frame stats.

    Also returns the meta dict so callers can reuse the recorded stats.

    Args:
        raw: Raw CCD frame array.
        phase_desc: Description of the phase that produced this frame.
        exposure_ms: Exposure time in milliseconds.

    Returns:
        Metadata dict for this frame.
    """
    global _frame_counter
    _frame_counter += 1
    idx = _frame_counter
    frame = np.asarray(raw, dtype=np.float32)
    peak = float(frame.max()) if frame.size else 0.0
    total = float(frame.sum()) if frame.size else 0.0
    if frame.size:
        cx, cy = centroid(frame, return_float=True)
    else:
        cx = cy = 0.0
    meta = {
        "frame": idx,
        "phase": phase_desc,
        "exposure_ms": exposure_ms,
        "peak": peak,
        "sum": total,
        "centroid": [float(cy), float(cx)],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    frames_dir = _frames_dir
    if frames_dir is not None:
        np.save(frames_dir / f"frame_{idx:05d}.npy", np.asarray(raw))
        save_frame_png(
            np.asarray(raw),
            frames_dir / f"frame_{idx:05d}.png",
            f"frame {idx:04d} - {phase_desc} (peak={peak:.0f})",
        )
        with open(frames_dir / "frame_meta.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    logger.info(
        "帧 {:04d}: phase={} peak={:.1f} sum={:.0f} centroid=({:.1f}, {:.1f})",
        idx,
        phase_desc,
        peak,
        total,
        cy,
        cx,
    )
    return meta


# ---------------------------------------------------------------------------
# Auto-exposure helpers (shared by diff_beam_runner and diff_shaping_runner)
# ---------------------------------------------------------------------------
def auto_exposure_target_ms(
    current_ms: float,
    peak: float,
    target_brightness: float = 180.0,
    tol: float = 0.2,
    *,
    min_ms: float = 0.02,
    max_ms: float = 1000.0,
    sat_floor: float = 245.0,
    max_boost: float = 4.0,
    max_cut: float = 0.25,
) -> float:
    """Compute the next exposure (ms) that drives ``peak`` into ``target ± tol``.

    Pure function — no hardware access. The caller applies the result via
    ``camera.reset_exposure_time()``.

    Args:
        current_ms: Current exposure time in milliseconds.
        peak: Measured peak brightness.
        target_brightness: Target peak brightness (default 180).
        tol: Relative tolerance band (default 0.2).
        min_ms: Minimum exposure clamp.
        max_ms: Maximum exposure clamp.
        sat_floor: Hard saturation guard for 8-bit CCD.
        max_boost: Maximum boost factor.
        max_cut: Maximum cut factor.

    Returns:
        Suggested exposure time in milliseconds.
    """
    low, high = target_brightness * (1.0 - tol), target_brightness * (1.0 + tol)
    if low <= peak <= high:
        return float(current_ms)
    if peak >= sat_floor:
        scale = max(target_brightness / peak, 0.1)
    elif peak < low:
        scale = min(target_brightness / peak, max_boost)
    else:  # peak > high
        scale = max(target_brightness / peak, max_cut)
    return float(np.clip(current_ms * scale, min_ms, max_ms))


def auto_exposure_possible(camera: Any) -> bool:
    """Return True if the camera supports runtime exposure adjustment."""
    return callable(getattr(camera, "reset_exposure_time", None))


def apply_auto_exposure(
    camera: Any,
    peak: float,
    current_ms: float,
    target_brightness: float,
    tol: float,
) -> tuple[float, bool]:
    """Adjust camera exposure toward the target peak band.

    Args:
        camera: Camera object with ``reset_exposure_time``.
        peak: Current measured peak.
        current_ms: Current exposure time.
        target_brightness: Target peak brightness.
        tol: Tolerance band.

    Returns:
        ``(actual_exposure_ms, changed)`` tuple.
    """
    next_ms = auto_exposure_target_ms(current_ms, peak, target_brightness, tol)
    if abs(next_ms - current_ms) < 1e-9:
        return current_ms, False
    actual_ms = float(camera.reset_exposure_time(next_ms))
    logger.info(
        "自动曝光: peak={:.1f} -> 曝光 {:.3f} -> {:.3f} ms",
        peak,
        current_ms,
        actual_ms,
    )
    return actual_ms, True


# ---------------------------------------------------------------------------
# Hardware timeout helper
# ---------------------------------------------------------------------------
def call_with_timeout(fn: Any, timeout_s: float, desc: str) -> Any:
    """Run ``fn`` in a daemon thread with a watchdog timeout.

    Hardware SDK calls can hang forever; this bounds the wait and raises
    ``TimeoutError`` if the call does not return in time.

    Args:
        fn: Callable to run.
        timeout_s: Timeout in seconds.
        desc: Description for error messages.

    Returns:
        Return value of ``fn``.

    Raises:
        TimeoutError: If ``fn`` does not complete within ``timeout_s``.
    """
    result: list[Any] = []
    error: list[BaseException] = []

    def _runner() -> None:
        try:
            result.append(fn())
        except BaseException as exc:  # noqa: BLE001
            error.append(exc)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        raise TimeoutError(f"{desc} 超时 ({timeout_s}s)")
    if error:
        raise error[0]
    return result[0] if result else None
