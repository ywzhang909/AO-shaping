"""Pure geometry helpers to embed pupil-domain phase into full-panel SLM phase."""
from __future__ import annotations

import numpy as np

from ao_shaping.utils.math.array_shape import fit_exact


def panel_shape_from_slm(slm: object) -> tuple[int, int]:
    """Panel ``(height, width)`` from the driver, falling back to the SLM-200 default."""
    res = getattr(slm, "Panel_Res", None)
    if res is not None:
        return int(res[1]), int(res[0])
    return 1200, 1920


def pupil_side(radius_px: float) -> int:
    """Edge length of the square covering the illuminated pupil."""
    return max(32, int(round(2.0 * radius_px)))


def embed_pupil_square(
    box: np.ndarray, panel_shape: tuple[int, int], radius_px: float,
    center_xy: tuple[int, int] | None = None,
) -> np.ndarray:
    """Embed a pupil-square array into a full-panel array, centred.

    The freeform DOF and the GS phase are both computed on a square covering
    only the illuminated pupil. Everything outside stays flat, so no degree of
    freedom is spent on light that never reaches the far field.
    """
    ph, pw = panel_shape
    out = np.zeros((ph, pw), dtype=np.float64)
    side = pupil_side(radius_px)
    cx, cy = center_xy if center_xy is not None else (pw // 2, ph // 2)
    x0, y0 = int(cx) - side // 2, int(cy) - side // 2
    if x0 < 0 or y0 < 0 or x0 + side > pw or y0 + side > ph:
        raise ValueError("pupil square extends beyond the SLM panel")
    out[y0 : y0 + side, x0 : x0 + side] = fit_exact(box, (side, side))
    return out


def compose_pupil_dof(
    base_panel: np.ndarray,
    dof: np.ndarray | None,
    panel_shape: tuple[int, int],
    radius_px: float,
    center_xy: tuple[int, int] | None = None,
) -> np.ndarray:
    """Full-panel raw-radian phase = base (flat or GS) + embedded freeform DOF.

    ``base_panel`` is never mutated, so the accumulated DOF cannot be double
    counted when the best iterate is promoted.
    """
    if dof is None or not np.any(dof):
        return base_panel
    from ao_shaping.utils.image.resample import upsample_bilinear_to_shape

    box = upsample_bilinear_to_shape(dof, (pupil_side(radius_px),) * 2)
    return base_panel + embed_pupil_square(box, panel_shape, radius_px, center_xy)
