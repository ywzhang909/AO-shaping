"""Deprecated re-export shim for :mod:`ao_shaping.optimizer.wfless.slm_zernike_pib`.

``slm_zernike_shaping`` was a sandbox fork of ``slm_zernike_pib`` (TODO 5.17 /
issue #74). The two copies drifted: ``pib`` gained the run-robustness features
(ABBA sampling, fold / noise / peak epoch gating) while ``shaping`` gained the
``w_outside`` energy-loss term and the debug-artifact bundle. Those divergences
have now been merged into ``slm_zernike_pib``, which is the single
implementation.

This module re-exports ``slm_zernike_pib``'s public surface so existing imports
keep resolving. New code must import from ``slm_zernike_pib`` directly; nothing
in production (runners, scripts, GUI) imports this shim.

Example:
    >>> from ao_shaping.optimizer.wfless.slm_zernike_pib import (
    ...     SlmZernikePibConfig,
    ...     optimize_slm_zernike_pib,
    ... )
    >>> recorder = optimize_slm_zernike_pib(
    ...     SlmZernikePibConfig(center="shape", epochs=2000, delta=0.1)
    ... )
"""

from __future__ import annotations

from ao_shaping.optimizer.wfless.slm_zernike_pib import (
    SLM_HEIGHT,
    SLM_RESOLUTION,
    SLM_WIDTH,
    ObjectiveSpec,
    Santec,
    SlmZernikePibConfig,
    optimize_slm_zernike_pib,
)

__all__ = [
    "SLM_HEIGHT",
    "SLM_RESOLUTION",
    "SLM_WIDTH",
    "ObjectiveSpec",
    "Santec",
    "SlmZernikePibConfig",
    "optimize_slm_zernike_pib",
]
