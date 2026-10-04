"""NLight 设备型号专属硬件常量.

与 :mod:`ao_shaping.drivers.dm.nlight.constants` 的分工: 那边放协议 / 编码 /
端点，这边只放这台机器的具体规格。
"""

from __future__ import annotations

#: 致动器数量 (XuWeiDM64)。
DM_NUM: int = 64

#: 驱动板允许的电压范围 (V)。
V_MIN: float = -300.0
V_MAX: float = 499.0

#: 物理上不接的致动器 (索引)。``default_dm_unit_mask`` 据此屏蔽。
DISABLED_ACTUATORS: list[int] = [0]

__all__ = ["DISABLED_ACTUATORS", "DM_NUM", "V_MAX", "V_MIN"]
