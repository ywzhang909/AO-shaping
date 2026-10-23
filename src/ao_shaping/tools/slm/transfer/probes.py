"""Deterministic phase families with canonical Zernike evaluation."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom

from ao_shaping.utils.wavefront.zernike_utils import generate_zernike_phase


def probe_grid(index: int, size: int, seed: int) -> tuple[str, np.ndarray]:
    """Use disjoint seeds for three phase families; return raw radians."""
    rng = np.random.default_rng(seed + index)
    family = ("zernike", "smooth", "bandlimited")[index % 3]
    if family == "zernike":
        coefficients = {j: float(rng.normal(0, .6)) for j in range(4, 12)}
        phase = generate_zernike_phase(coefficients, (size, size))
    else:
        coarse = 8 if family == "smooth" else 24
        phase = zoom(rng.normal(0, 1, (coarse, coarse)), size / coarse, order=1)
        phase *= 1.5 / max(float(phase.std()), 1e-9)
    return family, np.nan_to_num(np.asarray(phase, dtype=np.float32), nan=0.0)

