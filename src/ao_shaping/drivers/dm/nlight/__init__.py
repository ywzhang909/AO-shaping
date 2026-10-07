"""NLight 变形镜驱动子包.

``driver`` 提供驱动类 :class:`NLight` 与配置容器 :class:`NLightParams` /
:data:`NLIGHT_CONFIG`; ``constants`` 放驱动层 (协议 / 编码 / 端点) 常量;
``nlight_constants`` 放该型号专属硬件规格; ``udp`` 与 ``sdk`` 是两条互补的
传输通道 (UDP 批量下发 / Drv_UDPST.dll C SDK)。

本包所有符号全部惰性解析: 注册表在模块作用域调用 ``register_dm_lazy``
挂上 ``nlight`` 类型名, 实际的类对象在首次属性访问时才 import。
"""

from __future__ import annotations

from ao_shaping.drivers._lazy import install_lazy_attrs
from ao_shaping.drivers.dm._registry import register_dm_lazy

# 在模块作用域注册类型名, 不触发任何驱动模块的 import。
register_dm_lazy("nlight", "ao_shaping.drivers.dm.nlight.driver", "NLight")

_LAZY_BACKENDS: dict[str, tuple[str, str]] = {
    # --- constants ---
    "HV_SETTLE_TIME_S": (
        "ao_shaping.drivers.dm.nlight.constants",
        "HV_SETTLE_TIME_S",
    ),
    "MIN_TIME_DELAY": ("ao_shaping.drivers.dm.nlight.constants", "MIN_TIME_DELAY"),
    "NLIGHT_IP": ("ao_shaping.drivers.dm.nlight.constants", "NLIGHT_IP"),
    "NLIGHT_PORT": ("ao_shaping.drivers.dm.nlight.constants", "NLIGHT_PORT"),
    "REACHABLE_TIMEOUT_S": (
        "ao_shaping.drivers.dm.nlight.constants",
        "REACHABLE_TIMEOUT_S",
    ),
    "UDP_HEAD": ("ao_shaping.drivers.dm.nlight.constants", "UDP_HEAD"),
    "UDP_HEAD_WITH_ECHO": (
        "ao_shaping.drivers.dm.nlight.constants",
        "UDP_HEAD_WITH_ECHO",
    ),
    "UDP_RAW_MAX": ("ao_shaping.drivers.dm.nlight.constants", "UDP_RAW_MAX"),
    "UDP_RAW_MIN": ("ao_shaping.drivers.dm.nlight.constants", "UDP_RAW_MIN"),
    "UDP_RAW_OFFSET": ("ao_shaping.drivers.dm.nlight.constants", "UDP_RAW_OFFSET"),
    "UDP_RAW_SCALE": ("ao_shaping.drivers.dm.nlight.constants", "UDP_RAW_SCALE"),
    "UDP_RAW_SPAN": ("ao_shaping.drivers.dm.nlight.constants", "UDP_RAW_SPAN"),
    "UDP_REG_IDS": ("ao_shaping.drivers.dm.nlight.constants", "UDP_REG_IDS"),
    # --- driver ---
    "NLIGHT_CONFIG": (
        "ao_shaping.drivers.dm.nlight.driver",
        "NLIGHT_CONFIG",
    ),
    "NLight": ("ao_shaping.drivers.dm.nlight.driver", "NLight"),
    "NLightParams": ("ao_shaping.drivers.dm.nlight.driver", "NLightParams"),
    # --- nlight_constants ---
    "DISABLED_ACTUATORS": (
        "ao_shaping.drivers.dm.nlight.nlight_constants",
        "DISABLED_ACTUATORS",
    ),
    "DM_NUM": ("ao_shaping.drivers.dm.nlight.nlight_constants", "DM_NUM"),
    "V_MAX": ("ao_shaping.drivers.dm.nlight.nlight_constants", "V_MAX"),
    "V_MIN": ("ao_shaping.drivers.dm.nlight.nlight_constants", "V_MIN"),
    # --- sdk ---
    "DMSdk": ("ao_shaping.drivers.dm.nlight.sdk", "DMSdk"),
    # --- udp ---
    "DMUdp": ("ao_shaping.drivers.dm.nlight.udp", "DMUdp"),
}

__getattr__ = install_lazy_attrs(globals(), _LAZY_BACKENDS, __name__)

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
