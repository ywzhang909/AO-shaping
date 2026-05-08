"""Simulated devices package.

This package provides simulated hardware devices that integrate with the
device driver framework. These simulated devices wrap the numerical simulation
code from sim.digitaltwin while presenting a Device-compatible interface.

Structure:
    sim/
    ├── base.py          # Base classes for simulated devices
    ├── wave.py          # Wave generation, propagation, and metric utilities
    ├── ccd/             # Simulated cameras
    ├── laser/           # Simulated lasers
    ├── optics/          # Simulated optical elements (SLM, Lens, Aperture)
    └── atmos/           # Simulated atmospheric effects

Physics utilities (wrapping sim.digitaltwin):
    create_wave()        # Create a Wave object
    apply_aperture()     # Apply circular aperture
    apply_focus()        # Apply thin lens focus phase
    propagate()          # Angular spectrum propagation
    power_bucket()       # Compute PIB metric
    radius_metric()      # Compute energy-containing radius

Example:
    >>> from ao_shaping.drivers.sim import (
    ...     SimulatedTurbulentScreen,
    ...     create_wave, apply_aperture, apply_focus,
    ...     propagate, power_bucket, radius_metric,
    ... )
    >>>
    >>> wave = create_wave(256, 0.1e-3, 1550e-9)
    >>> apply_aperture(wave, 0.05)
    >>> apply_focus(wave, 0.5)
    >>> turb = SimulatedTurbulentScreen(Cn2=1e-9)
    >>> turb.process(wave)
    >>> propagate(wave, 0.5)
    >>> pib = power_bucket(wave.intensity, wave.x, wave.y, 'origin', 5e-3)
"""

from ao_shaping.drivers.sim.base import (
    OpticalDevice,
    SimulatedDevice,
    SimulatedDeviceError,
    WavefrontProcessor,
)

from ao_shaping.drivers.sim.ccd import SimulatedCCD

from ao_shaping.drivers.sim.laser import SimulatedLaser

from ao_shaping.drivers.sim.optics import (
    SimulatedAperture,
    SimulatedLens,
    SimulatedSLM,
)

from ao_shaping.drivers.sim.atmos import (
    SimulatedATP,
    SimulatedThermalScreen,
    SimulatedTurbulentScreen,
)

from ao_shaping.drivers.sim.dm import SimMicroDM, SimulateDM

from ao_shaping.drivers.sim.wave import (
    WaveGenerator,
    WavePropagator,
    LensApplier,
    ApertureApplier,
    WaveMetric,
    WaveDeviceError,
    create_wave,
    apply_aperture,
    apply_focus,
    propagate,
    power_bucket,
    radius_metric,
)

# Try to import Environment from sim.digitaltwin, but gracefully handle missing dependency
try:
    from sim.digitaltwin.base import Environment
    _has_environment = True
except ImportError:
    Environment = None
    _has_environment = False

# Build __all__ list dynamically
__all__ = [
    # Base classes
    "SimulatedDevice",
    "SimulatedDeviceError",
    "OpticalDevice",
    "WavefrontProcessor",
    # Wave utilities
    "WaveGenerator",
    "WavePropagator",
    "LensApplier",
    "ApertureApplier",
    "WaveMetric",
    "WaveDeviceError",
    "create_wave",
    "apply_aperture",
    "apply_focus",
    "propagate",
    "power_bucket",
    "radius_metric",
    # DM
    "SimMicroDM",
    "SimulateDM",
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

if _has_environment:
    __all__.append("Environment")

__version__ = "1.0.0"
