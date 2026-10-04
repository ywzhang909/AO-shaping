"""Micro DM (R50Power) 驱动子包.

``driver`` 提供同步驱动 :class:`MicroDM`、单控制器客户端
:class:`R50Controller` 与电压→载荷换算 :func:`voltages_to_payload`;
``asyn_driver`` 是同协议的 asyncio 版本 (:class:`AsyncMicroDM`);
``constants`` 放帧协议 / 电压编码; ``micro_constants`` 放该型号硬件规格;
``wiring_map`` 是接线表的类型化 JSON schema。

Example:
    >>> from ao_shaping.drivers.dm.micro import MicroDM
    >>> with MicroDM(ips=["192.168.0.101"]) as dm:
    ...     dm.set_relay_state(True)
"""

from __future__ import annotations

from ao_shaping.drivers._lazy import install_lazy_attrs
from ao_shaping.drivers.dm.micro.constants import (
    CMD_RELAY_OFF,
    CMD_RELAY_ON,
    CMD_SET_ALL_CHANNEL_VOLTAGE,
    CMD_SET_ALL_VOLTAGE_BY_ARR,
    CMD_SET_CHANNEL_VOLTAGE,
    FOOTER,
    HEADER,
    PAYLOAD_DAC_GAIN,
    PAYLOAD_FULL_SCALE,
    PAYLOAD_OFFSET,
    PAYLOAD_SCALE,
    WIRING_MAP_PATH,
)
from ao_shaping.drivers.dm.micro.driver import (
    MAX_CHANNELS,
    MICRO_DM_CONFIG,
    MicroDM,
    MicroDMConnectionError,
    MicroDMError,
    MicroDMParams,
    MicroDMVoltageError,
    R50Controller,
    RelayState,
    voltages_to_payload,
)
from ao_shaping.drivers.dm.micro.micro_constants import (
    ARRAY_SIDE,
    CHANNELS_PER_CONTROLLER,
    DEFAULT_IPS,
    DEFAULT_TIMEOUT,
    DM_NUM,
    IP_SUFFIX_MAX,
    IP_SUFFIX_MIN,
    MAX_ACTUATORS,
    MAX_CONTROLLERS,
    PORT_BASE,
    VOLTAGE_MAX,
    VOLTAGE_MIN,
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

# The async driver is resolved lazily because ``@register_dm("asyn_micro")``
# runs as an import side effect. ``runners/runner_common.py`` snapshots
# ``list_dm_types()`` and *then* imports this package to widen the ``--dm_type``
# list from 6 to 7 entries; an eager import here would collapse that two-step
# ordering and change the help output.
_LAZY_BACKENDS: dict[str, tuple[str, str]] = {
    "AsyncMicroDM": ("ao_shaping.drivers.dm.micro.asyn_driver", "AsyncMicroDM"),
    "AsyncR50Controller": (
        "ao_shaping.drivers.dm.micro.asyn_driver",
        "AsyncR50Controller",
    ),
    "MicroDMAsync": ("ao_shaping.drivers.dm.micro.asyn_driver", "MicroDMAsync"),
    "SendResult": ("ao_shaping.drivers.dm.micro.asyn_driver", "SendResult"),
    "VoltageConverter": ("ao_shaping.drivers.dm.micro.asyn_driver", "VoltageConverter"),
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
    "voltages_to_payload",
]
