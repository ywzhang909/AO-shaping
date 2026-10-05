"""光学元件模拟设备。

本子包提供光学元件的仿真实现, 包含 SLM、透镜、光阑等。
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
