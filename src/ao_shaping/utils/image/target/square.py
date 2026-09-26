"""Square beam-shaping target sizing and amplitude masks (GS-style).

Part of the :mod:`ao_shaping.utils.image.target` package (split by type).
"""
from __future__ import annotations

import numpy as np


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
