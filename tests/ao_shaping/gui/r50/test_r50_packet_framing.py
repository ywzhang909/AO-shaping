"""Packet-framing correctness tests for the R50 GUI send paths.

中央 r50_command 归并后, 四个下发路径 (单控制器 0x08/0x09、单单元 0x09、
分组 0x09、联合 0x09) 用 ``HEADER + cmd + voltages_to_payload(...) + FOOTER``
组帧。这里锁定:
  - 与 MicroDM 驱动参考组帧逐字节一致 (真机实际发送的正是这套字节);
  - 已知电压的金字节, 防止未来改动静默破坏字节编码。
"""

from __future__ import annotations

import numpy as np

from ao_shaping.drivers.dm.MicroDM import (
    CMD_SET_ALL_CHANNEL_VOLTAGE,
    CMD_SET_ALL_VOLTAGE_BY_ARR,
    FOOTER,
    HEADER,
    voltages_to_payload,
)


def _gui_all_channel(voltage: float) -> bytes:
    """单控制器「全选」路径 (r50_single._send_channels all_mode)。"""
    payload = voltages_to_payload(np.asarray([voltage], dtype=np.float64))
    return HEADER + bytes([CMD_SET_ALL_CHANNEL_VOLTAGE, int(payload[0]), int(payload[1])]) + FOOTER


def _gui_bulk_array(arr: np.ndarray) -> bytes:
    """0x09 批量数组路径 (单控制器 bulk / 单单元 / 分组 / 联合 共用)。"""
    return HEADER + bytes([CMD_SET_ALL_VOLTAGE_BY_ARR]) + voltages_to_payload(arr) + FOOTER


class TestAllChannelPacket:
    def test_gui_all_channel_matches_driver_framing(self):
        v = 37.5
        ref = HEADER + bytes([CMD_SET_ALL_CHANNEL_VOLTAGE, *voltages_to_payload(v)[:2]]) + FOOTER
        assert _gui_all_channel(v) == ref

    def test_golden_bytes(self):
        # 37.5V: raw=round((37.5+20)/20/3.4/3.3*65535) big-endian -> hv=0x41 lv=0x99
        assert _gui_all_channel(37.5).hex() == "aabb084199ccdd"

    def test_frame_delimiters(self):
        pkt = _gui_all_channel(0.0)
        assert pkt[:2] == bytes([0xAA, 0xBB])
        assert pkt[-2:] == bytes([0xCC, 0xDD])
        assert pkt[2] == 0x08


class TestBulkArrayPacket:
    def test_gui_matches_driver_framing(self):
        arr = np.array([0.0, 5.0, -5.0, 120.0, 37.5] * 10, dtype=np.float64)  # 50
        ref = HEADER + bytes([CMD_SET_ALL_VOLTAGE_BY_ARR]) + voltages_to_payload(arr) + FOOTER
        assert _gui_bulk_array(arr) == ref

    def test_payload_length_and_delimiters(self):
        arr = np.zeros(50, dtype=np.float64)
        pkt = _gui_bulk_array(arr)
        assert pkt[2] == 0x09
        assert pkt[:2] == bytes([0xAA, 0xBB])
        assert pkt[-2:] == bytes([0xCC, 0xDD])
        # 50 channels * 2 bytes high/low
        assert len(pkt) == 2 + 1 + 100 + 2
