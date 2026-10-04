"""NLight DM 驱动的向后兼容 re-export shim.

实现已迁至 :mod:`ao_shaping.drivers.dm.nlight`（与
:mod:`ao_shaping.drivers.slm.santec` 同一子包标准）::

    nlight/
    ├── driver.py            # NLight
    ├── constants.py         # 驱动层常量 (协议 / 编码 / 端点)
    ├── nlight_constants.py  # 型号专属硬件规格
    ├── udp.py               # DMUdp —— UDP 批量下发
    └── sdk.py               # DMSdk —— Drv_UDPST.dll ctypes 绑定

新代码请直接 ``from ao_shaping.drivers.dm.nlight import NLight``；
本模块仅为已有 import 路径保留，且不再新增符号。
"""

from __future__ import annotations

from ao_shaping.drivers.dm.nlight import (
    NLIGHT_CONFIG,
    NLight,
    NLightParams,
)

__all__ = ["NLIGHT_CONFIG", "NLight", "NLightParams"]
