"""AsyncMicroDM 驱动的向后兼容 re-export shim.

实现已迁至 :mod:`ao_shaping.drivers.dm.micro.asyn_driver`（与
:mod:`ao_shaping.drivers.slm.santec` 同一子包标准）。新代码请直接
``from ao_shaping.drivers.dm.micro import AsyncMicroDM``；本模块仅为已有
import 路径保留，且不再新增符号。
"""

from __future__ import annotations

from ao_shaping.drivers.dm.micro.asyn_driver import (
    DEFAULT_PORT,
    AsyncMicroDM,
    AsyncR50Controller,
    MicroDMAsync,
    SendResult,
    VoltageConverter,
)
from ao_shaping.drivers.dm.micro.wiring_map import WiringMap

__all__ = [
    "DEFAULT_PORT",
    "AsyncMicroDM",
    "AsyncR50Controller",
    "MicroDMAsync",
    "SendResult",
    "VoltageConverter",
    "WiringMap",
]
