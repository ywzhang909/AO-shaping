"""Fixed, reproducible panel phase used for optical-disturbance comparisons."""

from pathlib import Path

import numpy as np

from ao_shaping.utils.wavefront.zernike_utils import generate_zernike_phase


def zernike_panel_disturbance(
    coefficients: dict[int, float], panel_shape: tuple[int, int],
    center: tuple[int, int], radius: int,
) -> np.ndarray:
    """Render raw-radian Noll coefficients around a measured panel beam center."""
    if radius <= 0 or not coefficients:
        raise ValueError("a positive radius and nonempty coefficients are required")
    height, width = panel_shape
    x, y = center
    side = 2 * radius
    if x - radius < 0 or x + radius > width or y - radius < 0 or y + radius > height:
        raise ValueError("disturbance aperture extends beyond the SLM panel")
    patch = generate_zernike_phase(
        coefficients, (side, side), n_max=6, radius=float(radius)
    )
    panel = np.zeros(panel_shape, dtype=np.float32)
    panel[y - radius:y + radius, x - radius:x + radius] = np.nan_to_num(
        patch, nan=0.0
    )
    return panel


def load_panel_disturbance(path: str | None, shape: tuple[int, int]) -> np.ndarray | None:
    """Validate the actual panel map before any phase is displayed."""
    if path is None:
        return None
    panel = np.load(Path(path), allow_pickle=False)
    if panel.shape != shape or not np.issubdtype(panel.dtype, np.floating):
        raise ValueError(f"disturbance must be a floating panel of shape {shape}")
    if not np.isfinite(panel).all():
        raise ValueError("disturbance contains non-finite phase values")
    return np.asarray(panel, dtype=np.float64)
