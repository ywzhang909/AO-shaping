"""Simulated devices package.

This package provides simulated hardware devices that integrate with the
device driver framework. These simulated devices wrap the numerical simulation
code from sim.digitaltwin while presenting a Device-compatible interface.

Structure:
    sim/
    ├── base.py          # Base classes for simulated devices
    ├── ccd/             # Simulated cameras
    ├── laser/           # Simulated lasers
    ├── optics/          # Simulated optical elements (SLM, Lens, Aperture)
    └── atmos/           # Simulated atmospheric effects

Example:
    >>> from ao_shaping.drivers.sim import SimulatedLaser, SimulatedSLM
    >>> 
    >>> # Use simulated laser
    >>> laser = SimulatedLaser(power=100, wavelength=1064)
    >>> with laser:
    ...     wave = laser.generate()
    ...
    >>> # Use simulated SLM
    >>> slm = SimulatedSLM(resolution=(1920, 1080))
    >>> with slm:
    ...     slm.set_phase(phase_pattern)
    ...     output = slm.process(wave)
"""

from ao_shaping.drivers.sim.base import (
    OpticalDevice,
    SimulatedDevice,
    SimulatedDeviceError,
    WavefrontProcessor,
)

# CCD/Camera simulations
from ao_shaping.drivers.sim.ccd import SimulatedCCD

# Laser simulations
from ao_shaping.drivers.sim.laser import SimulatedLaser

# Optics simulations
from ao_shaping.drivers.sim.optics import (
    SimulatedAperture,
    SimulatedLens,
    SimulatedSLM,
)

# Atmospheric simulations
from ao_shaping.drivers.sim.atmos import (
    SimulatedATP,
    SimulatedThermalScreen,
    SimulatedTurbulentScreen,
)

__all__ = [
    # Base classes
    "SimulatedDevice",
    "SimulatedDeviceError",
    "OpticalDevice",
    "WavefrontProcessor",
    # CCD
    "SimulatedCCD",
    # Laser
    "SimulatedLaser",
    # Optics
    "SimulatedSLM",
    "SimulatedLens",
    "SimulatedAperture",
    # Atmosphere
    "SimulatedTurbulentScreen",
    "SimulatedThermalScreen",
    "SimulatedATP",
]

# Version
__version__ = "1.0.0"
