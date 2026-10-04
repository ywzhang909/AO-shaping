"""NLight 变形镜驱动子包.

``driver`` 提供驱动类 :class:`NLight` 与配置容器 :class:`NLightParams` /
:data:`NLIGHT_CONFIG`; ``constants`` 放驱动层 (协议 / 编码 / 端点) 常量;
``nlight_constants`` 放该型号专属硬件规格; ``udp`` 与 ``sdk`` 是两条互补的
传输通道 (UDP 批量下发 / Drv_UDPST.dll C SDK)。
"""

from __future__ import annotations

from ao_shaping.drivers.dm.nlight.constants import (
    HV_SETTLE_TIME_S,
    MIN_TIME_DELAY,
    NLIGHT_IP,
    NLIGHT_PORT,
    REACHABLE_TIMEOUT_S,
    UDP_HEAD,
    UDP_HEAD_WITH_ECHO,
    UDP_RAW_MAX,
    UDP_RAW_MIN,
    UDP_RAW_OFFSET,
    UDP_RAW_SCALE,
    UDP_RAW_SPAN,
    UDP_REG_IDS,
)
from ao_shaping.drivers.dm.nlight.driver import (
    NLIGHT_CONFIG,
    NLight,
    NLightParams,
)
from ao_shaping.drivers.dm.nlight.nlight_constants import (
    DISABLED_ACTUATORS,
    DM_NUM,
    V_MAX,
    V_MIN,
)
from ao_shaping.drivers.dm.nlight.sdk import DMSdk
from ao_shaping.drivers.dm.nlight.udp import DMUdp

__all__ = [
    "DISABLED_ACTUATORS",
    "DM_NUM",
    "DMUdp",
    "DMSdk",
    "HV_SETTLE_TIME_S",
    "MIN_TIME_DELAY",
    "NLIGHT_CONFIG",
    "NLIGHT_IP",
    "NLIGHT_PORT",
    "NLight",
    "NLightParams",
    "REACHABLE_TIMEOUT_S",
    "UDP_HEAD",
    "UDP_HEAD_WITH_ECHO",
    "UDP_RAW_MAX",
    "UDP_RAW_MIN",
    "UDP_RAW_OFFSET",
    "UDP_RAW_SCALE",
    "UDP_RAW_SPAN",
    "UDP_REG_IDS",
    "V_MAX",
    "V_MIN",
]
