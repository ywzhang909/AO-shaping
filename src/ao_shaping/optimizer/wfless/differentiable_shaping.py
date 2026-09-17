"""Differentiable beam shaping (gradient-descent SLM phase) — optimizer wrapper.

Thin re-export of the closed-loop differentiable shaping algorithm from
:mod:`ao_shaping.algorithm.differentiable_shaping`. Keep this file as a
facade so callers import through the optimizer package::

    from ao_shaping.optimizer import train_beam_shaping

Note:
    ``torch`` stays an *optional* hard dependency — the underlying module
    only accesses torch symbols through its lazy ``_torch()`` accessor and
    raises a clear ``ImportError`` when torch is missing.
"""

from __future__ import annotations

from ao_shaping.algorithm.differentiable_shaping import (
    DifferentiableShapingResult,
    angular_spectrum_propagate_torch,
    create_target_mask,
    efficiency_loss,
    smoothness_regularization,
    total_loss,
    train_beam_shaping,
    uniformity_loss,
    zero_order_penalty,
)

__all__ = [
    "DifferentiableShapingResult",
    "angular_spectrum_propagate_torch",
    "create_target_mask",
    "efficiency_loss",
    "smoothness_regularization",
    "total_loss",
    "train_beam_shaping",
    "uniformity_loss",
    "zero_order_penalty",
]