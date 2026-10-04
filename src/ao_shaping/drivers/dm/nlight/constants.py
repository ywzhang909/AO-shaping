"""NLight 驱动层常量 (SDK 协议 / 电压编码 / 网络端点).

与设备型号无关的部分集中在此；型号专属的致动器数量等见
:mod:`ao_shaping.drivers.dm.nlight.nlight_constants`.
"""

from __future__ import annotations

# ---- 网络端点 (UDP + SDK 共享) ----
NLIGHT_IP: str = "192.168.6.10"
NLIGHT_PORT: int = 1001

# ---- UDP 帧协议 ----
# 帧头按厂商协议以空格分隔的十六进制字符串组帧后再 bytes.fromhex。
UDP_HEAD_WITH_ECHO: list[str] = list("10 01 2c".split(" "))
UDP_HEAD: list[str] = list("30 01 2c".split(" "))

#: 4 组寄存器的基址 (0x00, 0x40, 0x80, 0xC0)，按寄存器序号取模轮换。
UDP_REG_IDS: list[int] = [0, 16384, 32768, 49152]

# ---- 电压 → 12bit 原始值编码 (UDP 路径) ----
#: 原始值 = (voltage + 500) / 1000 * 4096，随后钳位到 [UDP_RAW_MIN, UDP_RAW_MAX]。
UDP_RAW_SCALE: float = 4096.0
UDP_RAW_OFFSET: float = 500.0
UDP_RAW_SPAN: float = 1000.0
UDP_RAW_MIN: int = 820
UDP_RAW_MAX: int = 4095

# ---- 时序 ----
#: 斜坡下发时每步之间的最小间隔 (s)。
MIN_TIME_DELAY: float = 0.01
#: ``set_hv`` / ``reset_all`` 后等待驱动板稳定的时间 (s)。
HV_SETTLE_TIME_S: float = 0.5
#: ``is_reachable`` 探测时的 socket 超时 (s)。
REACHABLE_TIMEOUT_S: float = 1.0

__all__ = [
    "HV_SETTLE_TIME_S",
    "MIN_TIME_DELAY",
    "NLIGHT_IP",
    "NLIGHT_PORT",
    "REACHABLE_TIMEOUT_S",
    "UDP_HEAD",
    "UDP_HEAD_WITH_ECHO",
    "UDP_RAW_MAX",
    "UDP_RAW_MIN",
    "UDP_RAW_OFFSET",
    "UDP_RAW_SCALE",
    "UDP_RAW_SPAN",
    "UDP_REG_IDS",
]
