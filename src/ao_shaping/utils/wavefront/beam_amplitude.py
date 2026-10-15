"""Source / pupil amplitude fields on the model grid (pure numpy leaf).

These are *generators* of the input field that the forward model propagates to
the far field -- not measurement helpers (those live in ``utils.image``) and not
panel placement (``utils.slm``). The module must stay a **leaf**: it imports only
``numpy``, so a hardware-free script (or a unit test) can build a source field
without dragging in a driver, ``torch``, or ``ctypes``.

``gaussian_grid`` used to live in ``tools/kernels/bench_kernels.py`` next to
camera/SLM duck-typed code. It is pure numpy, so it now has a leaf home here;
``bench_kernels`` re-exports it for backward compatibility (R-46 / #62 -- the
last module-scope ``optimizer -> tools`` edge).

.. note::
   ``optimizer/wfless/model_in_loop_shaping.py`` carries two near-duplicates of
   this generator (``_pupil_amplitude`` and the ``amplitude`` closure) that take
   an explicit ``semi_axis_model_px`` **int-clamped** to ``region // 2``. They are
   NOT byte-identical to ``gaussian_grid`` for an odd ``region`` (the aperture
   radius here is the float ``region / 2.0``, so the mask boundary differs).
   Consolidating all three onto one general form is a behaviour-sensitive change
   tracked under R-51 (#67), not this sink.
"""

from __future__ import annotations

import numpy as np


def gaussian_grid(region: int, waist_grid: float) -> np.ndarray:
    """Gaussian illumination on the model grid, masked to the inscribed circle.

    Args:
        region: Model grid edge length in pixels.
        waist_grid: Gaussian waist in **model pixels**.

    Returns:
        A ``(region, region)`` amplitude array, zero outside the inscribed
        circle of radius ``region / 2``.
    """
    yy, xx = np.mgrid[0:region, 0:region]
    r2 = (xx - region / 2.0) ** 2 + (yy - region / 2.0) ** 2
    amp = np.exp(-r2 / (2.0 * max(float(waist_grid), 1e-6) ** 2))
    amp[r2 > (region / 2.0) ** 2] = 0.0
    return amp


__all__ = ["gaussian_grid"]
