"""Pure geometry helpers to embed pupil-domain phase into full-panel SLM phase."""
from __future__ import annotations

import numpy as np


def panel_shape_from_slm(slm: object) -> tuple[int, int]:
    res = getattr(slm, "Panel_Res", None)
    if res is not None:
        return int(res[1]), int(res[0])
    return 1200, 1920


def pupil_side(radius_px: float) -> int:
    return max(32, int(round(2.0 * radius_px)))


def fit_exact(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    out = np.asarray(arr, dtype=np.float64)
    ph, pw = max(0, h - out.shape[0]), max(0, w - out.shape[1])
    if ph or pw:
        out = np.pad(out, ((0, ph), (0, pw)), mode="edge")
    return out[:h, :w]


def embed_pupil_square(
    box: np.ndarray, panel_shape: tuple[int, int], radius_px: float
) -> np.ndarray:
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
    if dof is None or not np.any(dof):
        return base_panel
    from ao_shaping.utils.image.resample import upsample_bilinear_to_shape

    box = upsample_bilinear_to_shape(dof, (pupil_side(radius_px),) * 2)
    return base_panel + embed_pupil_square(box, panel_shape, radius_px)
