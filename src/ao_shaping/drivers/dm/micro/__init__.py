"""Micro DM (R50Power) 驱动子包.

``driver`` 提供同步驱动 :class:`MicroDM`、单控制器客户端
:class:`R50Controller` 与电压→载荷换算 :func:`voltages_to_payload`;
``asyn_driver`` 是同协议的 asyncio 版本 (:class:`AsyncMicroDM`);
``constants`` 放帧协议 / 电压编码; ``micro_constants`` 放该型号硬件规格;
``wiring_map`` 是接线表的类型化 JSON schema。

本包所有符号全部惰性解析: 注册表在模块作用域调用 ``register_dm_lazy``
挂上类型名, 实际的类对象在首次属性访问时才 import。
"""

from __future__ import annotations

from ao_shaping.drivers._lazy import install_lazy_attrs
from ao_shaping.drivers.dm._registry import register_dm_lazy

# 在模块作用域注册类型名, 不触发任何驱动模块的 import。
# 这使得 ``list_dm_types()`` 在零驱动加载的情况下就能返回完整列表。
register_dm_lazy("micro", "ao_shaping.drivers.dm.micro.driver", "MicroDM")
register_dm_lazy(
    "asyn_micro", "ao_shaping.drivers.dm.micro.asyn_driver", "AsyncMicroDM"
)

# 全部公开符号的惰性解析映射: 键是 ``__all__`` 中的名字,
# 值是 ``(模块路径, 属性名)``。
_LAZY_BACKENDS: dict[str, tuple[str, str]] = {
    # --- constants ---
    "CMD_RELAY_OFF": ("ao_shaping.drivers.dm.micro.constants", "CMD_RELAY_OFF"),
    "CMD_RELAY_ON": ("ao_shaping.drivers.dm.micro.constants", "CMD_RELAY_ON"),
    "CMD_SET_ALL_CHANNEL_VOLTAGE": (
        "ao_shaping.drivers.dm.micro.constants",
        "CMD_SET_ALL_CHANNEL_VOLTAGE",
    ),
    "CMD_SET_ALL_VOLTAGE_BY_ARR": (
        "ao_shaping.drivers.dm.micro.constants",
        "CMD_SET_ALL_VOLTAGE_BY_ARR",
    ),
    "CMD_SET_CHANNEL_VOLTAGE": (
        "ao_shaping.drivers.dm.micro.constants",
        "CMD_SET_CHANNEL_VOLTAGE",
    ),
    "FOOTER": ("ao_shaping.drivers.dm.micro.constants", "FOOTER"),
    "HEADER": ("ao_shaping.drivers.dm.micro.constants", "HEADER"),
    "PAYLOAD_DAC_GAIN": ("ao_shaping.drivers.dm.micro.constants", "PAYLOAD_DAC_GAIN"),
    "PAYLOAD_FULL_SCALE": (
        "ao_shaping.drivers.dm.micro.constants",
        "PAYLOAD_FULL_SCALE",
    ),
    "PAYLOAD_OFFSET": ("ao_shaping.drivers.dm.micro.constants", "PAYLOAD_OFFSET"),
    "PAYLOAD_SCALE": ("ao_shaping.drivers.dm.micro.constants", "PAYLOAD_SCALE"),
    "WIRING_MAP_PATH": ("ao_shaping.drivers.dm.micro.constants", "WIRING_MAP_PATH"),
    # --- driver ---
    "MAX_CHANNELS": ("ao_shaping.drivers.dm.micro.driver", "MAX_CHANNELS"),
    "MICRO_DM_CONFIG": ("ao_shaping.drivers.dm.micro.driver", "MICRO_DM_CONFIG"),
    "MicroDM": ("ao_shaping.drivers.dm.micro.driver", "MicroDM"),
    "MicroDMConnectionError": (
        "ao_shaping.drivers.dm.micro.driver",
        "MicroDMConnectionError",
    ),
    "MicroDMError": ("ao_shaping.drivers.dm.micro.driver", "MicroDMError"),
    "MicroDMParams": ("ao_shaping.drivers.dm.micro.driver", "MicroDMParams"),
    "MicroDMVoltageError": (
        "ao_shaping.drivers.dm.micro.driver",
        "MicroDMVoltageError",
    ),
    "R50Controller": ("ao_shaping.drivers.dm.micro.driver", "R50Controller"),
    "RelayState": ("ao_shaping.drivers.dm.micro.driver", "RelayState"),
    "voltages_to_payload": (
        "ao_shaping.drivers.dm.micro.driver",
        "voltages_to_payload",
    ),
    # --- micro_constants ---
    "ARRAY_SIDE": ("ao_shaping.drivers.dm.micro.micro_constants", "ARRAY_SIDE"),
    "CHANNELS_PER_CONTROLLER": (
        "ao_shaping.drivers.dm.micro.micro_constants",
        "CHANNELS_PER_CONTROLLER",
    ),
    "DEFAULT_IPS": ("ao_shaping.drivers.dm.micro.micro_constants", "DEFAULT_IPS"),
    "DEFAULT_TIMEOUT": (
        "ao_shaping.drivers.dm.micro.micro_constants",
        "DEFAULT_TIMEOUT",
    ),
    "DM_NUM": ("ao_shaping.drivers.dm.micro.micro_constants", "DM_NUM"),
    "IP_SUFFIX_MAX": ("ao_shaping.drivers.dm.micro.micro_constants", "IP_SUFFIX_MAX"),
    "IP_SUFFIX_MIN": ("ao_shaping.drivers.dm.micro.micro_constants", "IP_SUFFIX_MIN"),
    "MAX_ACTUATORS": (
        "ao_shaping.drivers.dm.micro.micro_constants",
        "MAX_ACTUATORS",
    ),
    "MAX_CONTROLLERS": (
        "ao_shaping.drivers.dm.micro.micro_constants",
        "MAX_CONTROLLERS",
    ),
    "PORT_BASE": ("ao_shaping.drivers.dm.micro.micro_constants", "PORT_BASE"),
    "VOLTAGE_MAX": ("ao_shaping.drivers.dm.micro.micro_constants", "VOLTAGE_MAX"),
    "VOLTAGE_MIN": ("ao_shaping.drivers.dm.micro.micro_constants", "VOLTAGE_MIN"),
    # --- wiring_map ---
    "ChannelEntry": ("ao_shaping.drivers.dm.micro.wiring_map", "ChannelEntry"),
    "ChannelInfo": ("ao_shaping.drivers.dm.micro.wiring_map", "ChannelInfo"),
    "ChannelSchemaDoc": (
        "ao_shaping.drivers.dm.micro.wiring_map",
        "ChannelSchemaDoc",
    ),
    "Group": ("ao_shaping.drivers.dm.micro.wiring_map", "Group"),
    "Metadata": ("ao_shaping.drivers.dm.micro.wiring_map", "Metadata"),
    "RangeInfo": ("ao_shaping.drivers.dm.micro.wiring_map", "RangeInfo"),
    "SchemaDoc": ("ao_shaping.drivers.dm.micro.wiring_map", "SchemaDoc"),
    "SourceFiles": ("ao_shaping.drivers.dm.micro.wiring_map", "SourceFiles"),
    "Summary": ("ao_shaping.drivers.dm.micro.wiring_map", "Summary"),
    "WiringMap": ("ao_shaping.drivers.dm.micro.wiring_map", "WiringMap"),
    "resolve_ips": ("ao_shaping.drivers.dm.micro.wiring_map", "resolve_ips"),
    # --- asyn_driver ---
    "AsyncMicroDM": (
        "ao_shaping.drivers.dm.micro.asyn_driver",
        "AsyncMicroDM",
    ),
    "AsyncR50Controller": (
        "ao_shaping.drivers.dm.micro.asyn_driver",
        "AsyncR50Controller",
    ),
    "MicroDMAsync": (
        "ao_shaping.drivers.dm.micro.asyn_driver",
        "MicroDMAsync",
    ),
    "SendResult": ("ao_shaping.drivers.dm.micro.asyn_driver", "SendResult"),
    "VoltageConverter": (
        "ao_shaping.drivers.dm.micro.asyn_driver",
        "VoltageConverter",
    ),
}

__getattr__ = install_lazy_attrs(globals(), _LAZY_BACKENDS, __name__)

__all__ = [
    "ARRAY_SIDE",
    "AsyncMicroDM",
    "AsyncR50Controller",
    "CHANNELS_PER_CONTROLLER",
    "CMD_RELAY_OFF",
    "CMD_RELAY_ON",
    "CMD_SET_ALL_CHANNEL_VOLTAGE",
    "CMD_SET_ALL_VOLTAGE_BY_ARR",
    "CMD_SET_CHANNEL_VOLTAGE",
    "ChannelEntry",
    "ChannelInfo",
    "ChannelSchemaDoc",
    "DEFAULT_IPS",
    "DEFAULT_TIMEOUT",
    "DM_NUM",
    "FOOTER",
    "Group",
    "HEADER",
    "IP_SUFFIX_MAX",
    "IP_SUFFIX_MIN",
    "MAX_ACTUATORS",
    "MAX_CHANNELS",
    "MAX_CONTROLLERS",
    "MICRO_DM_CONFIG",
    "Metadata",
    "MicroDM",
    "MicroDMAsync",
    "MicroDMConnectionError",
    "MicroDMError",
    "MicroDMParams",
    "MicroDMVoltageError",
    "PAYLOAD_DAC_GAIN",
    "PAYLOAD_FULL_SCALE",
    "PAYLOAD_OFFSET",
    "PAYLOAD_SCALE",
    "PORT_BASE",
    "R50Controller",
    "RangeInfo",
    "RelayState",
    "SchemaDoc",
    "SendResult",
    "SourceFiles",
    "Summary",
    "VOLTAGE_MAX",
    "VOLTAGE_MIN",
    "VoltageConverter",
    "WIRING_MAP_PATH",
    "WiringMap",
    "resolve_ips",
    "voltages_to_payload",
]
