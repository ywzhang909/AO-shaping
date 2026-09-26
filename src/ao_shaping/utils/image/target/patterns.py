"""Parametric target-pattern generation on the SLM / image grid.

Part of the :mod:`ao_shaping.utils.image.target` package (split by type).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np


def create_target_shape(
    shape: Literal[
        "gaussian",
        "circle",
        "square",
        "annular",
        "grid",
        "cross",
        "rectangle",
        "pentagon",
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
    center: tuple[float, float] | None = None,
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
        center: ``(x, y)`` pixel coordinates for the pattern center. Defaults
            to the array center.

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
    cx, cy = center if center is not None else ((width - 1) / 2, (height - 1) / 2)
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
