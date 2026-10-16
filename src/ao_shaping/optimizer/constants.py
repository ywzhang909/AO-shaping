"""Shared constant definitions for the optimizer package.

Centralizes constants that were previously duplicated across multiple
optimizer modules (``OPTIMIZER_MAP``, ``METROPOLIS_ALPHA``) and the optimizer
factory that was previously duplicated five times.

This module is a leaf of the optimizer package: it must not import from other
optimizer submodules (which would create circular imports, e.g. via
``optimizer/__init__.py`` eagerly importing ``wfless.pib``).
"""

from __future__ import annotations

import inspect
import os
from typing import Any

from loguru import logger

from ao_shaping.algorithm.gradient.adam import (
    Adam,
    AdamW,
    AdaMOD,
    Base,
    Muno,
    MunoW,
    SGD,
)

#: Number of frames averaged by ``cam.get_numpy_image`` during optimization probes.
CAM_SAMPLE_ITER: int = 1

#: Target mean brightness (0-255) for the auto-exposure probe that locates the spot.
TEST_EXPOSURE_TIME_BRIGHTNESS: int = 220

#: Ideal spot radius in pixels; overridable via the ``IDEAL_SPOT_RADIUS`` env var.
IDEAL_SPOT_RADIUS: int = int(os.environ.get("IDEAL_SPOT_RADIUS", "6"))

#: Whether the DM keeps its last voltages instead of zeroing on optimizer exit.
KEEP_VOLTAGE_WHEN_EXIT: bool = True

#: Map of optimizer name (CLI/option string) → optimizer class.
OPTIMIZER_MAP: dict[str, type[Base]] = {
    "adam": Adam,
    "adamw": AdamW,
    "adamod": AdaMOD,
    "sgd": SGD,
    "muno": Muno,
    "munow": MunoW,
}

#: Metropolis acceptance scaling factor for SPGD-style optimizers.
METROPOLIS_ALPHA: float = 0.8


def create_optimizer(optimizer_type: str, dim: int, lr: float, **kwargs: Any) -> Base:
    """Create the configured optimizer, forwarding the kwargs it can accept.

    The optimizer family is heterogeneous: ``SGD``'s signature is only
    ``(self, dim, lr)`` while ``Adam`` also takes ``beta1``/``beta2`` and ``AdaMOD``
    adds ``beta3``. One CLI passes the union, so the extras must be filtered per
    class.

    Two things a naive filter gets wrong, both silently:

    * a key the target does not accept was dropped with no word, so
      ``--optimizer sgd`` with ``momentum`` looked accepted and did nothing;
    * a callee declaring ``**kwargs`` never received anything, because a
      var-keyword's parameters are *named* ``kwargs`` -- so the documented
      ``**config.kwargs`` escape hatch could never reach any optimizer.

    So: forward to ``**kwargs`` when the callee declares one, and report anything
    genuinely dropped instead of swallowing it.

    Args:
        optimizer_type: key into :data:`OPTIMIZER_MAP` (case-insensitive); unknown
            names fall back to ``AdaMOD``.
        dim: number of optimisable parameters.
        lr: learning rate.
        **kwargs: forwarded to the constructor when the target accepts them.

    Returns:
        The constructed optimizer.
    """
    optimizer_cls = OPTIMIZER_MAP.get(optimizer_type.lower(), AdaMOD)
    signature = inspect.signature(optimizer_cls.__init__)
    accepts_var_keyword = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    )
    accepted: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in kwargs.items():
        if key in signature.parameters or accepts_var_keyword:
            accepted[key] = value
        else:
            dropped.append(key)
    if dropped:
        logger.warning(
            "{} does not accept {}; ignored. Accepted: {{}}",
            optimizer_cls.__name__,
            ", ".join(sorted(dropped)),
            ", ".join(sorted(signature.parameters)),
        )
    return optimizer_cls(dim, lr=lr, **accepted)