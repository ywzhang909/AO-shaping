"""Detached response telemetry for physical stages and neural activations."""
from __future__ import annotations

import numpy as np


def summarize_response(value):
    """Summarize a real/complex array without retaining an autograd graph.

    Complex stages are summarized in magnitude, with separate real/imaginary
    means. Magnitude statistics are not optical intensity. No normalization or
    clipping is applied, and nonfinite values are reported rather than hidden.
    """
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if not array.size or not np.issubdtype(array.dtype, np.number):
        raise ValueError("nonempty numeric response required")
    finite = np.isfinite(array)
    values = np.abs(array[finite]) if np.iscomplexobj(array) else array[finite]
    result = dict(shape=list(array.shape), dtype=str(array.dtype),
                  finite_fraction=float(finite.mean()), complex=bool(np.iscomplexobj(array)))
    if not values.size:
        return result
    values = values.astype(np.float64)
    result.update(mean=float(values.mean()), std=float(values.std()),
                  min=float(values.min()), max=float(values.max()),
                  percentiles=np.percentile(values, [1, 50, 99]).tolist(),
                  zero_fraction=float(np.mean(values == 0)), rms=float(np.sqrt(np.mean(values**2))))
    if np.iscomplexobj(array):
        result.update(real_mean=float(array[finite].real.mean()), imag_mean=float(array[finite].imag.mean()))
    return result
