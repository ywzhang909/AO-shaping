"""Differentiable (backpropagation) beam shaping — optimizer-level wrapper.

Thin re-export of the FFT far-field differentiable beam-shaping algorithm
from :mod:`ao_shaping.algorithm.differentiable_beam`. Keep this file as a
facade: any caller wanting the backprop/Gerchberg-Saxton phase-retrieval
pipeline imports it from the optimizer package::

    from ao_shaping.optimizer import differentiable_beam_optimize

Note:
    Unlike :mod:`ao_shaping.optimizer.wfless.differentiable_shaping`, the
    underlying module imports ``torch`` at module top level (torch is a
    hard dependency of the backprop path).
"""

from __future__ import annotations

from ao_shaping.algorithm.differentiable_beam import (
    BeamOptimizeResult,
    differentiable_beam_optimize,
    differentiable_far_field,
    far_field_intensity,
)

__all__ = [
    "BeamOptimizeResult",
    "differentiable_beam_optimize",
    "differentiable_far_field",
    "far_field_intensity",
]