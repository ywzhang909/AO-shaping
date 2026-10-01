"""Simulated wavefront sensors.

Wraps the OOPAO Shack-Hartmann model in the shared
:class:`~ao_shaping.drivers.wfs.base.BaseWFS` contract, so it is a drop-in
replacement for :class:`~ao_shaping.drivers.wfs.thorlab_wfs.ThorlabWFS`.
"""

from ao_shaping.drivers.sim.wfs.simulated_wfs import SimulatedWFS

__all__ = ["SimulatedWFS"]
