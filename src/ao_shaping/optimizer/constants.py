"""Shared constant definitions for the optimizer package.

Centralizes constants that were previously duplicated across multiple
optimizer modules (``OPTIMIZER_MAP``, ``METROPOLIS_ALPHA``).

This module is a leaf of the optimizer package: it must not import from other
optimizer submodules (which would create circular imports, e.g. via
``optimizer/__init__.py`` eagerly importing ``wfless.pib``).
"""

from __future__ import annotations

from ao_shaping.algorithm.gradient.adam import (
    Adam,
    AdamW,
    AdaMOD,
    Base,
    Muno,
    MunoW,
    SGD,
)

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