"""CCD/相机模拟设备。"""
from __future__ import annotations

from ao_shaping.drivers.sim.ccd.simulated_ccd import SimulatedCCD, SimulatedCCDError

__all__ = [
    "SimulatedCCD",
    "SimulatedCCDError",
]
