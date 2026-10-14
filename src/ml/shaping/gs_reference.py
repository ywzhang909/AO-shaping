"""Torch-free leaf helper: a golden-reference square phase from canonical GS.

This module is a **leaf**: at import time it touches only ``numpy`` (plus the
stdlib and ``loguru``). The canonical GS implementation and every other
``ao_shaping`` dependency are imported **lazily inside the function body**, so
importing this module never drags torch, ``ao_shaping.drivers``,
``ao_shaping.optimizer`` or ``ao_shaping.runners`` into the process. The
``tests/ao_shaping/ml/shaping/test_gs_reference.py`` guards pin that invariant by
walking this file's **AST** (not grep), so a forbidden import hidden inside a
function body is still caught.

:func:`gs_golden_phase` produces a **simulated upper-bound comparator**: a
theoretical-optimum square-shaping SLM phase computed by Gerchberg-Saxton in
pure simulation, kept as an absolute ceiling on the shaping quality a
phase-only SLM could reach. It **delegates** to the canonical
:func:`ao_shaping.algorithm.signal_processing.square_shaping.gs_shape`; the GS
loop is never re-implemented here.

⚠️  GS is a **comparator, not an assumed-good baseline.** On the real bench GS
has repeatedly *lost to a flat phase*: ``report/shape_trials/README.md`` records
hardware runs where the candidate never beat flat ("候选未能战胜平相") and the
SLM was reverted to flat, and the inverse-shaping package docstring
(``ml/shaping/__init__.py``) notes GS alone was negative in 90/90 runs. The
returned phase must therefore never be treated as automatically better than a
flat SLM phase -- measure it on the bench before relying on it.
"""

from __future__ import annotations

import numpy as np

from loguru import logger

__all__ = ["gs_golden_phase"]


def gs_golden_phase(cfg, n_iters: int, seed: int) -> np.ndarray:
    """Return the GS square-shaping phase (raw, unwrapped radians) for ``cfg``.

    This is a **simulated upper-bound comparator**, not a recommended
    controller: it delegates to the canonical
    :func:`ao_shaping.algorithm.signal_processing.square_shaping.gs_shape` and
    returns that result's phase unchanged. GS has previously lost to a flat
    phase in hardware runs (see the module docstring), so the returned phase
    must **never** be assumed automatically good.

    Args:
        cfg: A ``SquareShapingConfig`` (or compatible object) describing the 2f
            Fourier SLM bench -- grid, aperture, wavelength, focal length,
            target size/kind and far-field padding. Typed loosely so this leaf
            does not import ``ao_shaping`` at module import time.
        n_iters: Number of GS iterations; must be ``>= 1``.
        seed: RNG seed; threaded straight into ``gs_shape`` so the result is
            deterministic for a fixed ``(cfg, n_iters, seed)``.

    Returns:
        ``(n_grid, n_grid)`` ``float64`` array of **raw unwrapped radians** in
        the canonical ``(-pi, pi]`` angle convention. It is deliberately *not*
        ``mod 2*pi``: the repository's single mod-2π wrap lives in the SLM
        driver (``Santec.create_phase_from_array``), so callers convert via
        ``utils/slm/phase_display.phase_to_slm_grayscale`` (hardware) or the
        pure fallback (offline/tests).

    Raises:
        ValueError: If ``n_iters <= 0``. ``gs_shape`` itself silently no-ops on
            a non-positive count (it returns the initial field), so the guard
            lives here to keep that footgun out of a "golden reference".
    """
    if n_iters <= 0:
        raise ValueError(f"n_iters must be >= 1, got {n_iters}")

    # Lazy, function-local import: keeps this module a leaf (no ao_shaping /
    # torch at import time) while reusing the one canonical GS implementation.
    from ao_shaping.algorithm.signal_processing.square_shaping import (
        ShapingResult,
        gs_shape,
    )

    result: ShapingResult = gs_shape(cfg, n_iters=n_iters, seed=seed)
    # Copy out of the padded far-field grid so the returned array does not keep
    # the (much larger) internal complex field alive.
    phase = np.array(result.phase, dtype=np.float64)
    logger.debug(
        "gs_golden_phase: shape={} n_iters={} seed={} range=({:.4f}, {:.4f})",
        phase.shape,
        n_iters,
        seed,
        float(phase.min()) if phase.size else float("nan"),
        float(phase.max()) if phase.size else float("nan"),
    )
    return phase
