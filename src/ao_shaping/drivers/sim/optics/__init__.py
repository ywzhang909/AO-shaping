"""Optical simulation devices.

This package provides simulated optical devices including SLM, Lens, Aperture, etc.
"""

from ao_shaping.drivers.sim.optics.simulated_slm import (
    SimulatedAperture,
    SimulatedLens,
    SimulatedSLM,
    SimulatedSLMError,
)

__all__ = [
    "SimulatedSLM",
    "SimulatedSLMError",
    "SimulatedLens",
    "SimulatedAperture",
]
