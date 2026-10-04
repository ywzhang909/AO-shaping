"""Micro DM (R50Power) 驱动的向后兼容 re-export shim.

实现已迁至 :mod:`ao_shaping.drivers.dm.micro`（与
:mod:`ao_shaping.drivers.slm.santec` 同一子包标准）::

    micro/
    ├── driver.py            # MicroDM / R50Controller / 电压换算
    ├── asyn_driver.py       # AsyncMicroDM —— asyncio 版本
    ├── constants.py         # 帧协议 / 电压编码
    ├── micro_constants.py   # 型号专属硬件规格
    └── wiring_map.py        # 接线表类型化 JSON schema

新代码请直接 ``from ao_shaping.drivers.dm.micro import MicroDM``；
本模块仅为已有 import 路径保留，且不再新增符号。
"""

from __future__ import annotations

from ao_shaping.drivers.dm.micro.driver import (
    CMD_RELAY_OFF,
    CMD_RELAY_ON,
    CMD_SET_ALL_CHANNEL_VOLTAGE,
    CMD_SET_ALL_VOLTAGE_BY_ARR,
    CMD_SET_CHANNEL_VOLTAGE,
    DEFAULT_IPS,
    DEFAULT_TIMEOUT,
    FOOTER,
    HEADER,
    MAX_ACTUATORS,
    MAX_CHANNELS,
    MAX_CONTROLLERS,
    MICRO_DM_CONFIG,
    MicroDM,
    MicroDMConnectionError,
    MicroDMError,
    MicroDMParams,
    MicroDMVoltageError,
    R50Controller,
    RelayState,
    VOLTAGE_MAX,
    VOLTAGE_MIN,
    WIRING_MAP_PATH,
    voltages_to_payload,
)
from ao_shaping.drivers.dm.micro.wiring_map import (
    ChannelEntry,
    ChannelInfo,
    ChannelSchemaDoc,
    Group,
    Metadata,
    RangeInfo,
    SchemaDoc,
    SourceFiles,
    Summary,
    WiringMap,
)

__all__ = [
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
    "FOOTER",
    "Group",
    "HEADER",
    "MAX_ACTUATORS",
    "MAX_CHANNELS",
    "MAX_CONTROLLERS",
    "MICRO_DM_CONFIG",
    "Metadata",
    "MicroDM",
    "MicroDMConnectionError",
    "MicroDMError",
    "MicroDMParams",
    "MicroDMVoltageError",
    "R50Controller",
    "RangeInfo",
    "RelayState",
    "SchemaDoc",
    "SourceFiles",
    "Summary",
    "VOLTAGE_MAX",
    "VOLTAGE_MIN",
    "WIRING_MAP_PATH",
    "WiringMap",
    "voltages_to_payload",
]
