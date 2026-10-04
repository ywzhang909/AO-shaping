"""NLight UDP 电压下发通道.

把电压数组编码成厂商的空格分隔十六进制帧，经 UDP datagram 发往驱动器。
与 :mod:`ao_shaping.drivers.dm.nlight.sdk` 的 ctypes 通道互补: UDP 是高速
批量路径，C SDK 用于高压开关 / 读回。
"""

from __future__ import annotations

import socket
from typing import Any

import numpy as np
import numpy.typing as npt

from ao_shaping.drivers.dm.nlight.constants import (
    NLIGHT_IP,
    NLIGHT_PORT,
    UDP_HEAD,
    UDP_HEAD_WITH_ECHO,
    UDP_RAW_MAX,
    UDP_RAW_MIN,
    UDP_RAW_OFFSET,
    UDP_RAW_SCALE,
    UDP_RAW_SPAN,
    UDP_REG_IDS,
)
from ao_shaping.drivers.dm.nlight.nlight_constants import DM_NUM

#: 一次 ``reset_all`` 下发的补零通道数 (覆盖整个 64 通道寄存器组)。
RESET_CHANNEL_COUNT: int = 256


class DMUdp:
    """Blocking UDP client for one NLight controller."""

    HEAD_WITH_ECHO = UDP_HEAD_WITH_ECHO
    HEAD = UDP_HEAD
    REG_IDS = UDP_REG_IDS

    def __init__(self) -> None:
        self.ip = NLIGHT_IP
        # test ip reachable
        try:
            socket.inet_aton(self.ip)
        except socket.error:
            raise AssertionError("device connection error.")
        self.port = NLIGHT_PORT
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.dm_num = DM_NUM

    @staticmethod
    def _num_hex(num: int) -> str:
        hex_16 = hex(int(num))[2:].zfill(4)
        return hex_16[:2] + " " + hex_16[2:]

    @staticmethod
    def _voltage_hex(num: float, registry: int) -> str:
        _num = int((num + UDP_RAW_OFFSET) / UDP_RAW_SPAN * UDP_RAW_SCALE)
        _num = min(_num, UDP_RAW_MAX)
        _num = max(_num, UDP_RAW_MIN)
        _num += UDP_REG_IDS[registry % 4]
        hex_16 = DMUdp._num_hex(_num)
        return hex_16

    def _send(self, message: str) -> Any:
        hex_message = bytes.fromhex(message)
        return self.sock.sendto(hex_message, (self.ip, self.port))

    def set_voltages(
        self, vs: npt.NDArray[np.floating] | list[float], with_echo: bool = False
    ) -> Any:
        _head = self.HEAD_WITH_ECHO if with_echo else self.HEAD
        send_data = " ".join(
            _head
            + [self._num_hex(self.dm_num)]
            + [self._voltage_hex(v, i) for i, v in enumerate(vs)]
        )
        return self._send(send_data)

    def reset_all(self) -> Any:
        vs = np.zeros(RESET_CHANNEL_COUNT)
        send_data = " ".join(
            self.HEAD_WITH_ECHO
            + [self._num_hex(RESET_CHANNEL_COUNT)]
            + [self._voltage_hex(v, i) for i, v in enumerate(vs)]
        )
        return self._send(send_data) & self._send("10 00 00 00 01 00 03")

    def set_hv(self, hv: bool) -> None:
        raise NotImplementedError("set_hv func not Implement")


__all__ = ["RESET_CHANNEL_COUNT", "DMUdp"]
