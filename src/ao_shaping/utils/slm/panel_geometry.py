"""SLM panel-geometry placement: put a model-grid phase onto the panel disc.

R-47 attribution correction: :func:`phase_to_panel` is pure panel geometry, not
a bench measurement kernel, so it lives in the ``utils/slm`` leaf rather than
``tools/kernels/bench_kernels`` (which re-exports it for existing callers). This
lets ``optimizer/`` reach it without importing ``tools/`` (the R-46 edge).

Naming trap: this is a **spatial placement** op (model-grid radians -> panel-sized
radians, zero outside the disc). It is NOT the radians->uint16 grayscale
conversion in :func:`ao_shaping.utils.slm.phase_display.phase_to_slm_grayscale`.
The two are different transforms and must not be merged.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom

from ao_shaping.utils.slm.bench_constants import SLM_PANEL_H, SLM_PANEL_W

__all__ = ["phase_to_panel"]


def phase_to_panel(
    phase_model: np.ndarray, disc_radius: int, pupil_center: tuple[int, int]
) -> np.ndarray:
    """Resize a model-grid phase onto the panel disc, centred on the beam.

    Args:
        phase_model: ``(region, region)`` phase, raw unwrapped radians.
        disc_radius: Half-width of the target window, **panel** pixels.
        pupil_center: Beam centre in **panel** pixels as ``(x, y)``.

    Returns:
        A ``(SLM_PANEL_H, SLM_PANEL_W)`` float64 array holding the phase inside
        the disc and zero elsewhere -- the shape the Santec driver expects.

    Raises:
        ValueError: If ``phase_model`` is not square.

    Note:
        ``pupil_center`` must be measured **on the panel**. The camera's 0-order
        is a different coordinate frame entirely (on this bench the two axes are
        swapped and the scales differ by more than 10x), so deriving one from
        the other silently writes the phase where the beam is not.
    """
    phase_model = np.asarray(phase_model, dtype=np.float64)
    if phase_model.ndim != 2 or phase_model.shape[0] != phase_model.shape[1]:
        raise ValueError(f"phase_model must be square, got {phase_model.shape}")
    r = int(disc_radius)
    region = phase_model.shape[0]
    sub = np.asarray(
        zoom(phase_model, (2 * r / region, 2 * r / region), order=1), dtype=np.float64
    )
    panel = np.zeros((SLM_PANEL_H, SLM_PANEL_W), dtype=np.float64)
    cx, cy = int(pupil_center[0]), int(pupil_center[1])
    # Clip the window so an off-centre or oversized disc stays in bounds.
    x0, x1 = max(cx - r, 0), min(cx + r, SLM_PANEL_W)
    y0, y1 = max(cy - r, 0), min(cy + r, SLM_PANEL_H)
    sub = sub[y0 - (cy - r) : y1 - (cy - r), x0 - (cx - r) : x1 - (cx - r)]
    panel[y0:y1, x0:x1] = sub
    return panel
