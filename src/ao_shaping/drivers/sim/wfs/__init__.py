"""模拟波前传感器。

把 OOPAO Shack-Hartmann 模型包装进共享的
:class:`~ao_shaping.drivers.wfs.base.BaseWFS` 契约, 因此可直接替换
:class:`~ao_shaping.drivers.wfs.thorlab_wfs.ThorlabWFS`。
"""

from ao_shaping.drivers.sim.wfs.simulated_wfs import SimulatedWFS

__all__ = ["SimulatedWFS"]
