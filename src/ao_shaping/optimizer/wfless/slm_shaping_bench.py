"""Reference optimization methods for the simulated 2f SLM shaping bench.

Backward-compat re-export shim. The forward model, the objective metrics and
the reference optimizers all live in the canonical module
:mod:`ao_shaping.drivers.sim.slm_shaping_bench`; this module only forwards the
optimizer-layer entry points (``gs_shape`` / ``differentiable_shape`` /
``spgd_shape`` / ``analytic_amplitude_target``).

Keeping a second, hand-copied implementation here is what silently reintroduced
a stale forward model after the 2026-10-01 merge: the copies below used
``beam_backend.focal_plane`` on an **un-padded** pupil while the canonical bench
zero-pads to ``far_field_size`` before the Fraunhofer FFT, so the same method
returned different metrics depending on which module it was imported from.
"""

from __future__ import annotations

from ao_shaping.drivers.sim.slm_shaping_bench import (
    ShapingBenchConfig,
    ShapingResult,
    analytic_amplitude_target,
    differentiable_shape,
    gs_shape,
    spgd_shape,
)

__all__ = [
    "ShapingBenchConfig",
    "ShapingResult",
    "analytic_amplitude_target",
    "differentiable_shape",
    "gs_shape",
    "spgd_shape",
]
