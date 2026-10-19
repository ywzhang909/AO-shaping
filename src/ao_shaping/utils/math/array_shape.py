"""Exact-shape array fitting that is neither image, wavefront nor device aware.

Lives here because two leaf modules needed the same kernel and each kept a
private copy to avoid pulling in the other's dependency:
``utils/image/resample.py`` imports ``scipy.ndimage``, and
``utils/slm/phase_embed.py`` deliberately does not. A pure-numpy leaf lets both
share one implementation without either gaining a dependency it was avoiding
(R-65 / #87 -- the copies were byte-identical and free to drift).

Leaf contract: standard library and ``numpy`` only, no ``ao_shaping.*`` import
at module level or deferred. Guarded by
``tests/ao_shaping/utils/math/test_math_leaf.py``.
"""

from __future__ import annotations

import numpy as np


def fit_exact(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Force ``arr`` to exactly ``shape`` by edge-padding then cropping.

    Args:
        arr: Input array (H, W) or compatible.
        shape: Target shape ``(height, width)``.

    Returns:
        Array reshaped/padded/cropped to ``shape``, dtype float64.
    """
    h, w = shape
    out = np.asarray(arr, dtype=np.float64)
    ph, pw = max(0, h - out.shape[0]), max(0, w - out.shape[1])
    if ph or pw:
        out = np.pad(out, ((0, ph), (0, pw)), mode="edge")
    return out[:h, :w]
