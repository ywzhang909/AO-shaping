"""Backward-compatible shim for the split :mod:`ao_shaping.utils.image.target` package.

The implementation moved into :mod:`ao_shaping.utils.image.target`, split by type
(``patterns`` / ``metrics`` / ``square`` / ``ccd`` / ``objective``). This module
re-exports the former flat public surface - including the private weight helpers
that ``slm_zernike_pib`` and the tests import - so existing
``from ao_shaping.utils.image.targets import ...`` imports keep working unchanged.

New code should import from the :mod:`ao_shaping.utils.image.target` package.

See Also:
    :mod:`ao_shaping.utils.image.target` (single source of truth; includes
    ``ObjectiveSpec`` and the objective-rule constants).
"""

from __future__ import annotations

import numpy as np

from ao_shaping.utils.image.target import *  # re-export all names
from ao_shaping.utils.image.target import __all__ as _target_all

__all__ = _target_all


def square_mask(shape: tuple[int, int], cy: int, cx: int, side: int) -> np.ndarray:
    """Boolean square support of edge ``side`` centred on ``(cy, cx)``.

    Clipped to the frame.
    """
    h, w = shape
    half = side / 2.0
    y0, y1 = int(round(cy - half)), int(round(cy + half))
    x0, x1 = int(round(cx - half)), int(round(cx + half))
    y0c, y1c = max(0, y0), min(h, y1)
    x0c, x1c = max(0, x0), min(w, x1)
    mask = np.zeros(shape, dtype=bool)
    if y1c > y0c and x1c > x0c:
        mask[y0c:y1c, x0c:x1c] = True
    return mask


def square_mask(shape: tuple[int, int], cy: int, cx: int, side: int) -> np.ndarray:
    """Boolean square support of edge ``side`` centred on ``(cy, cx)``.

    Clipped to the frame. The bench metric functions only ever use the *support*
    of their ``target`` argument, so a boolean mask is a faithful stand-in for a
    target intensity pattern.
    """
    h, w = shape
    half = side / 2.0
    y0, y1 = int(round(cy - half)), int(round(cy + half))
    x0, x1 = int(round(cx - half)), int(round(cx + half))
    y0c, y1c = max(0, y0), min(h, y1)
    x0c, x1c = max(0, x0), min(w, x1)
    mask = np.zeros(shape, dtype=bool)
    if y1c > y0c and x1c > x0c:
        mask[y0c:y1c, x0c:x1c] = True
    return mask
