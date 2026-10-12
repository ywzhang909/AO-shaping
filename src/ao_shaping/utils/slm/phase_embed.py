"""Pure geometry helpers to embed pupil-domain phase into full-panel SLM phase."""
from __future__ import annotations

import numpy as np


def panel_shape_from_slm(slm: object) -> tuple[int, int]:
    """Panel ``(height, width)`` from the driver, falling back to the SLM-200 default."""
    res = getattr(slm, "Panel_Res", None)
    if res is not None:
        return int(res[1]), int(res[0])
    return 1200, 1920


def pupil_side(radius_px: float) -> int:
    """Edge length of the square covering the illuminated pupil."""
    return max(32, int(round(2.0 * radius_px)))


def fit_exact(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Force ``arr`` to exactly ``shape`` by edge-padding then cropping.

    Mirrors :func:`ao_shaping.utils.image.resample.fit_exact`; kept local so this
    module stays free of the scipy import that ``resample`` pulls in.
    """
    h, w = shape
    out = np.asarray(arr, dtype=np.float64)
    ph, pw = max(0, h - out.shape[0]), max(0, w - out.shape[1])
    if ph or pw:
        out = np.pad(out, ((0, ph), (0, pw)), mode="edge")
    return out[:h, :w]


def embed_pupil_square(
    box: np.ndarray, panel_shape: tuple[int, int], radius_px: float
) -> np.ndarray:
    """Embed a pupil-square array into a full-panel array, centred.

    The freeform DOF and the GS phase are both computed on a square covering
    only the illuminated pupil. Everything outside stays flat, so no degree of
    freedom is spent on light that never reaches the far field.
    """
    ph, pw = panel_shape
    out = np.zeros((ph, pw), dtype=np.float64)
    side = min(pupil_side(radius_px), ph, pw)
    y0, x0 = (ph - side) // 2, (pw - side) // 2
    out[y0 : y0 + side, x0 : x0 + side] = fit_exact(box, (side, side))
    return out


def compose_pupil_dof(
    base_panel: np.ndarray,
    dof: np.ndarray | None,
    panel_shape: tuple[int, int],
    radius_px: float,
) -> np.ndarray:
    """Full-panel raw-radian phase = base (flat or GS) + embedded freeform DOF.

    ``base_panel`` is never mutated, so the accumulated DOF cannot be double
    counted when the best iterate is promoted.
    """
    if dof is None or not np.any(dof):
        return base_panel
    from ao_shaping.utils.image.resample import upsample_bilinear_to_shape

    box = upsample_bilinear_to_shape(dof, (pupil_side(radius_px),) * 2)
    return base_panel + embed_pupil_square(box, panel_shape, radius_px)
