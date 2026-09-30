"""Interpolation between centered physical sampling grids."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


def centered_coordinates(shape: tuple[int, int], pitch_um: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, y) meshes of pixel-centre coordinates in micrometres."""
    if len(shape) != 2 or any(not isinstance(v, (int, np.integer)) or v < 1 for v in shape):
        raise ValueError("shape must contain two positive integers")
    if not np.isfinite(pitch_um) or pitch_um <= 0:
        raise ValueError("pitch_um must be finite and positive")
    h, w = shape
    x = (np.arange(w) - (w - 1) / 2) * pitch_um
    y = (np.arange(h) - (h - 1) / 2) * pitch_um
    return np.meshgrid(x, y)


def resample_physical_grid(
    values: np.ndarray,
    source_pitch_um: float,
    target_shape: tuple[int, int],
    target_pitch_um: float,
    *,
    order: int = 1,
    fill_value: float | complex = 0.0,
) -> np.ndarray:
    """Sample a 2-D field at target pixel centres in the same physical space.

    Both grids have their geometric centre at (0, 0). A pixel at column ``j``
    has x coordinate ``(j - (width - 1) / 2) * pitch_um``; rows use the same
    convention for y. Values outside the source grid use ``fill_value``.
    This interpolates point samples; it does not conserve integrated flux.
    """
    data = np.asarray(values)
    if data.ndim != 2 or not np.issubdtype(data.dtype, np.number):
        raise ValueError("values must be a numeric 2-D array")
    if min(data.shape) < 1 or not np.all(np.isfinite(data)):
        raise ValueError("values must be non-empty and finite")
    if len(target_shape) != 2 or any(not isinstance(v, (int, np.integer)) or v < 1 for v in target_shape):
        raise ValueError("target_shape must contain two positive integers")
    if not np.isfinite(source_pitch_um) or source_pitch_um <= 0 or not np.isfinite(target_pitch_um) or target_pitch_um <= 0:
        raise ValueError("pixel pitches must be finite and positive")
    if order not in (0, 1, 3):
        raise ValueError("order must be 0, 1, or 3")
    source_h, source_w = data.shape
    xx_um, yy_um = centered_coordinates(target_shape, target_pitch_um)
    yy = yy_um / source_pitch_um + (source_h - 1) / 2
    xx = xx_um / source_pitch_um + (source_w - 1) / 2
    coordinates = np.array([yy, xx])
    if np.iscomplexobj(data):
        real = map_coordinates(data.real, coordinates, order=order, mode="constant", cval=float(np.real(fill_value)))
        imag = map_coordinates(data.imag, coordinates, order=order, mode="constant", cval=float(np.imag(fill_value)))
        return real + 1j * imag
    return map_coordinates(data.astype(np.float64), coordinates, order=order, mode="constant", cval=float(fill_value))
