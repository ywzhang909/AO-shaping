"""R50Power 设备型号专属硬件常量.

与 :mod:`ao_shaping.drivers.dm.micro.constants` 的分工: 那边放帧协议 /
电压编码，这边只放这台柜子的具体规格。
"""

from __future__ import annotations

#: 致动器阵列边长 (39×39 压电陶瓷矩阵)。
ARRAY_SIDE: int = 39
#: 逻辑通道总数 = ``ARRAY_SIDE ** 2``。
DM_NUM: int = ARRAY_SIDE * ARRAY_SIDE

#: 单台 R50Power 控制器管理的通道数。
CHANNELS_PER_CONTROLLER: int = 50
#: 柜内控制器台数上限。
MAX_CONTROLLERS: int = 26
#: 逻辑通道上限 (50 × 26)。
MAX_ACTUATORS: int = CHANNELS_PER_CONTROLLER * MAX_CONTROLLERS

#: 驱动板电压范围 (V)，来自 R50Power 数据手册。
VOLTAGE_MIN: float = -20.0
VOLTAGE_MAX: float = 120.0

#: TCP 连接 / 发送默认超时 (s)。
DEFAULT_TIMEOUT: float = 10.0

#: 端口 = 10000 + IP 后缀 (10101 → 192.168.0.101)。
PORT_BASE: int = 10000
#: 可用 IP 后缀区间 [IP_SUFFIX_MIN, IP_SUFFIX_MAX]。
IP_SUFFIX_MIN: int = 101
IP_SUFFIX_MAX: int = 126

#: 默认 IP 列表 (192.168.0.101 .. 192.168.0.126)。
DEFAULT_IPS: list[str] = [
    f"192.168.0.{suffix}" for suffix in range(IP_SUFFIX_MIN, IP_SUFFIX_MAX + 1)
]

__all__ = [
    "ARRAY_SIDE",
    "CHANNELS_PER_CONTROLLER",
    "DEFAULT_IPS",
    "DEFAULT_TIMEOUT",
    "DM_NUM",
    "IP_SUFFIX_MAX",
    "IP_SUFFIX_MIN",
    "MAX_ACTUATORS",
    "MAX_CONTROLLERS",
    "PORT_BASE",
    "VOLTAGE_MAX",
    "VOLTAGE_MIN",
]
