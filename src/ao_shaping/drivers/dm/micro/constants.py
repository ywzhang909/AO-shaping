"""R50Power 驱动层常量 (TCP 帧协议 / 电压编码 / 接线表位置).

与设备型号无关的部分集中在此；接线表 (39×39 阵列几何 / 控制器台数) 等
见 :mod:`ao_shaping.drivers.dm.micro.micro_constants`。
"""

from __future__ import annotations

from pathlib import Path

from ao_shaping.utils.io.file import ROOT_DIR

# ---- 帧协议 ----
HEADER: bytes = bytes([0xAA, 0xBB])
FOOTER: bytes = bytes([0xCC, 0xDD])

CMD_SET_CHANNEL_VOLTAGE: int = 0x04
CMD_SET_ALL_CHANNEL_VOLTAGE: int = 0x08
CMD_SET_ALL_VOLTAGE_BY_ARR: int = 0x09
CMD_RELAY_ON: int = 0x06
CMD_RELAY_OFF: int = 0x07

# ---- 电压 → 16bit 载荷编码 ----
# 参考 libs/micro_drive1300/docs/R50PowerV1.m:
#   value = (voltage + 20) / 20 / 3.4 / 3.3 * 65535
#   high = floor(value / 255); low = floor(mod(value, 256))
# 厂商实现 high 用 255 / low 用 256 这一不一致被原样保留 (硬件兼容性);
# 理论一致的做法是 high = raw // 256, low = raw % 256。
PAYLOAD_FULL_SCALE: float = 65535.0
PAYLOAD_DAC_GAIN: float = 20.0 * 3.4 * 3.3
PAYLOAD_SCALE: float = PAYLOAD_FULL_SCALE / PAYLOAD_DAC_GAIN
PAYLOAD_OFFSET: float = 20.0 * PAYLOAD_SCALE

#: 接线表 JSON 路径 (仓库输入资产，锚定包路径而非 CWD)。
WIRING_MAP_PATH: Path = ROOT_DIR / "libs" / "micro_drive1300" / "wiring_map.json"

__all__ = [
    "CMD_RELAY_OFF",
    "CMD_RELAY_ON",
    "CMD_SET_ALL_CHANNEL_VOLTAGE",
    "CMD_SET_ALL_VOLTAGE_BY_ARR",
    "CMD_SET_CHANNEL_VOLTAGE",
    "FOOTER",
    "HEADER",
    "PAYLOAD_DAC_GAIN",
    "PAYLOAD_FULL_SCALE",
    "PAYLOAD_OFFSET",
    "PAYLOAD_SCALE",
    "WIRING_MAP_PATH",
]
