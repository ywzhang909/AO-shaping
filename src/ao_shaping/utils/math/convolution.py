"""Small pure-numpy convolution kernels shared across the utils leaves.

Two modules carried byte-identical copies of this Gaussian kernel, each with a
comment saying it was hand-rolled so as not to pull ``scipy.ndimage`` into a
leaf (``utils/image/exposure_fusion._gaussian_kernel_1d`` and
``utils/image/target/metrics._loggrad_kernel``). That constraint argues for a
shared dependency-free home, not for duplication -- which is what R-65 / #87
is about.

Leaf contract: standard library and ``numpy`` only, no ``ao_shaping.*`` import
at module level or deferred. Guarded by
``tests/ao_shaping/utils/math/test_math_leaf.py``.
"""

from __future__ import annotations

import numpy as np


def gaussian_kernel_1d(sigma: float) -> np.ndarray:
    """Normalised 1D Gaussian kernel; ``sigma == 0`` is the identity kernel.

    Support is ``ceil(3 * sigma)`` on each side, so the truncated tails hold
    ~99.7% of the mass. Hand-rolled rather than taken from ``scipy.ndimage``
    because the callers are leaves of :mod:`ao_shaping.utils` and must not gain
    a ``scipy`` import to save a line.
    """
    if sigma == 0.0:
        return np.ones(1, dtype=np.float64)
    radius = int(np.ceil(3.0 * sigma))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(offsets**2) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()
