"""CCD/Camera simulation devices."""

from ao_shaping.drivers.sim.ccd.simulated_ccd import SimulatedCCD, SimulatedCCDError

__all__ = [
    "SimulatedCCD",
    "SimulatedCCDError",
]
