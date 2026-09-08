"""Pattern generation for SLM Hartmann calibration.

Implements the center cosine grayscale pattern from the paper to reduce
pixel crosstalk during phase calibration.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Literal
from loguru import logger


@dataclass
class CosinePatternConfig:
    """Configuration for center cosine grayscale pattern.

    Attributes:
        center_x: Center X coordinate of the measurement region.
        center_y: Center Y coordinate of the measurement region.
        radius_pixels: Radius of the circular phase region in pixels.
            Paper uses 40 pixels radius (80×80 region).
        max_phase_2pi: Maximum phase in units of 2π (paper uses 1.0, i.e., 0~2π).
        background_grayscale: Background grayscale value outside the region.
        smooth_edge_width: Gaussian smoothing width at edges (pixels).
    """

    center_x: int = 960
    center_y: int = 540
    radius_pixels: float = 40.0
    max_phase_2pi: float = 1.0
    background_grayscale: int = 0
    smooth_edge_width: float = 3.0


def generate_center_cosine_pattern(
    config: CosinePatternConfig | None = None,
    output_resolution: tuple[int, int] | None = None,
) -> np.ndarray:
    """Generate center cosine grayscale pattern.

    The pattern follows: g(x,y) = g0 * [1 + cos(π * r / R)] / 2
    where r is distance from center, R is radius, restricted to a circular region.
    The center has maximum phase, monotonically decreasing to zero at the edge.

    This continuous pattern reduces pixel crosstalk compared to traditional
    single-direction gradient patterns with abrupt boundaries.

    Args:
        config: Pattern configuration. Uses defaults if None.
        output_resolution: (width, height) of output array. Uses SLM panel
            resolution (1920, 1200) if None.

    Returns:
        2D uint16 grayscale array suitable for SLM display.
    """
    if config is None:
        config = CosinePatternConfig()

    if output_resolution is None:
        width, height = 1920, 1200
    else:
        width, height = output_resolution

    cx, cy = config.center_x, config.center_y
    R = max(config.radius_pixels, 1.0)
    max_gs = int(1023 * config.max_phase_2pi)

    # Create coordinate grids
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    dx = x_coords - cx
    dy = y_coords - cy
    r = np.sqrt(dx**2 + dy**2)

    # Cosine pattern: max at center, decreases to 0 at edge
    # g(x) = (1 + cos(pi * r / R)) / 2 * max_gs for r <= R
    mask = r <= R

    # Compute raw cosine values
    cos_val = (1.0 + np.cos(np.pi * r / R)) / 2.0

    # Apply smooth edge transition to reduce ringing at boundary
    if config.smooth_edge_width > 0:
        edge_region = (R < r) & (r <= R + config.smooth_edge_width)
        fade = 1.0 - (r[edge_region] - R) / config.smooth_edge_width
        fade = np.clip(fade, 0.0, 1.0)
        cos_val[edge_region] *= fade

    # Mask to circular region
    phase_region = (r <= R) | (config.smooth_edge_width > 0 and edge_region)

    grayscale = np.full((height, width), config.background_grayscale, dtype=np.float64)
    grayscale[phase_region] = cos_val[phase_region] * max_gs

    grayscale = np.clip(grayscale, 0, 1023)

    logger.debug(
        f"Center cosine pattern: center=({cx},{cy}), R={R:.1f}px, "
        f"max_phase={config.max_phase_2pi}*2π, max_gs={max_gs}"
    )

    return grayscale.astype(np.uint16)


def generate_traditional_gradient_pattern(
    config: CosinePatternConfig | None = None,
    direction: Literal["x", "y", "diagonal"] = "x",
    output_resolution: tuple[int, int] | None = None,
) -> np.ndarray:
    """Generate traditional single-direction gradient pattern.

    This creates a linear gradient with abrupt edge boundaries,
    used as a comparison pattern in the paper.

    Args:
        config: Pattern configuration. Uses defaults if None.
        direction: Gradient direction.
        output_resolution: (width, height) of output array.

    Returns:
        2D uint16 grayscale array.
    """
    if config is None:
        config = CosinePatternConfig()

    if output_resolution is None:
        width, height = 1920, 1200
    else:
        width, height = output_resolution

    cx, cy = config.center_x, config.center_y
    R = max(config.radius_pixels, 1.0)
    max_gs = int(1023 * config.max_phase_2pi)

    y_coords, x_coords = np.mgrid[0:height, 0:width]
    dx = x_coords - cx
    dy = y_coords - cy
    r = np.sqrt(dx**2 + dy**2)

    mask = r <= R

    grayscale = np.full((height, width), config.background_grayscale, dtype=np.float64)

    if direction == "x":
        # Gradient along X: left=0, right=max
        normalized = (x_coords - (cx - R)) / (2.0 * R)
    elif direction == "y":
        normalized = (y_coords - (cy - R)) / (2.0 * R)
    else:
        # Diagonal gradient
        normalized = (dx + dy + 2.0 * R) / (4.0 * R)

    normalized = np.clip(normalized, 0.0, 1.0)
    grayscale[mask] = normalized[mask] * max_gs

    return grayscale.astype(np.uint16)


def estimate_phase_from_pattern(
    grayscale_pattern: np.ndarray,
    max_grayscale: int = 1023,
) -> np.ndarray:
    """Estimate nominal phase (in radians) from a grayscale pattern.

    Assumes linear relationship: phase = 2π * g / g_max

    Args:
        grayscale_pattern: 2D uint16 grayscale array.
        max_grayscale: Maximum grayscale value for full 2π phase.

    Returns:
        2D float64 array of phase values in radians.
    """
    max_gs = max(max_grayscale, 1)
    return grayscale_pattern.astype(np.float64) / max_gs * 2.0 * np.pi


def get_pattern_peak_position(
    pattern: np.ndarray,
) -> tuple[int, int]:
    """Find the peak (maximum value) position in a pattern.

    For center cosine patterns, the peak is at the center of the
    circular region and serves as a spatial reference marker for
    matching measurement and compensation regions.

    Args:
        pattern: 2D grayscale array.

    Returns:
        (y, x) position of the maximum value.
    """
    max_val = np.max(pattern)
    if max_val == 0:
        h, w = pattern.shape
        return h // 2, w // 2
    positions = np.argwhere(pattern == max_val)
    return tuple(positions[0])
