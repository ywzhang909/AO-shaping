import os
import socket
import time
from ctypes import byref, c_bool, c_int32, cdll
from dataclasses import dataclass
from pathlib import Path

import findlibs
import numpy as np
from loguru import logger

from ao_shaping.drivers.dm._registry import register_dm
from ao_shaping.drivers.dm.base import DM
from ao_shaping.utils.device_config import ConfigHandler, DeviceParam, param
from ao_shaping.utils.file import ROOT_DIR as PROJECT_ROOT


# ── NLight 配置参数 ──────────────────────────────────────


_DM_CONFIG_DIR = Path(
    os.environ.get("DM_CONFIG_DIR", PROJECT_ROOT / "data" / "dm_configs")
)


@dataclass
class NLightParams(DeviceParam):
    """NLight DM 配置参数。

    属性名 (attr) 与实际 NLight 实例属性一致:
      - ``max_iter_diff`` → ``self.max_iter_diff``
      - ``max_neibor_diff`` → ``self._max_neibor_diff`` (property)
      - ``keep_when_exit`` → ``self._NLight__keep_when_exit`` (name-mangled)
      - ``safety_mode`` → ``self._safety_mode``
    """
    max_iter_diff: int = param(default=20, cast=int)
    max_neibor_diff: float = param(default=200.0, cast=float, attr="_max_neibor_diff")
    keep_when_exit: bool = param(default=True, cast=bool, attr="_NLight__keep_when_exit")
    safety_mode: bool = param(default=True, cast=bool, attr="_safety_mode")


# 模块级单例，所有 NLight 实例共用
NLIGHT_CONFIG = ConfigHandler(_DM_CONFIG_DIR, "nlight", NLightParams)


def _load_adj_txt():
    return np.loadtxt("data/dm_adj.txt")


@register_dm("nlight")
class NLight(DM):
    DM_NUM: int = 64
    V_Min: float = -300.0
    V_Max: float = 499.0
    MIN_TIME_DELAY = 0.01

    _IP = "192.168.6.10"
    _PORT = 1001

    disabled_actuators: list[int] = [0]

    Units_Adj_Mat = _load_adj_txt()

    @classmethod
    def is_reachable(cls) -> bool:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            result = sock.connect_ex((cls._IP, cls._PORT))
            sock.close()
            return result == 0
        except OSError:
            return False

    def __init__(
        self,
        max_iter_diff=20,
        max_neibor_diff=200,
        keep_when_exit=True,
        safety_mode=True,
    ):
        self._init_values = {
            "max_iter_diff": max_iter_diff,
            "max_neibor_diff": max_neibor_diff,
            "keep_when_exit": keep_when_exit,
            "safety_mode": safety_mode,
        }
        # 使用 defaults + __init__ 参数解析（NLight 无序列号，使用默认配置）
        params = NLIGHT_CONFIG.resolve_from_config({}, init_values=self._init_values)

        assert params.max_iter_diff <= 200
        assert params.max_neibor_diff <= 300

        super().__init__(safety_mode=params.safety_mode)
        self.max_iter_diff = params.max_iter_diff
        self._max_neibor_diff = params.max_neibor_diff

        self.c_driver = DMSdk()
        self.udp_driver = DMUdp()

        self.__keep_when_exit = params.keep_when_exit

    @property
    def max_neibor_diff(self) -> float:
        return self._max_neibor_diff

    @max_neibor_diff.setter
    def max_neibor_diff(self, value: float) -> None:
        self._max_neibor_diff = value

    @property
    def default_dm_unit_mask(self) -> np.ndarray:
        mask = np.ones(self.DM_NUM, dtype=bool)
        mask[0] = False
        return mask

    def load_config(self) -> dict:
        """加载当前设备的配置文件。

        NLight 无硬件序列号，使用固定标识 ``"default"`` 作为配置 key。
        """
        return NLIGHT_CONFIG._manager.load_config("default")

    def save_config(self) -> None:
        """将当前参数保存到 JSON 配置文件。"""
        config = NLIGHT_CONFIG.collect(self)
        NLIGHT_CONFIG._manager.save_config("default", config)
        config_file = NLIGHT_CONFIG._manager._get_config_file("default")
        logger.info(f"NLight 配置已保存: {config_file}")

    def open(self) -> None:
        """Open connection to DM and initialize"""
        self.initialize()

    def close(self) -> None:
        """Close connection to DM and clean up"""
        if not self.__keep_when_exit:
            self.reset_all()
            self.set_hv(False)
            logger.info("DM Turn off high voltages.")
        self.udp_driver.sock.close()

    def transform(self, cmd: np.ndarray) -> np.ndarray:
        """Transform command to DM actuators"""
        return self.transform_voltage(cmd)

    def get_actuator_positions(self) -> np.ndarray:
        """Get positions of DM actuators"""
        return self._last_voltages.copy()

    def get_hardware_info(self) -> dict:
        return {
            "type": "NLight",
            "DM_NUM": self.DM_NUM,
            "V_Min": self.V_Min,
            "V_Max": self.V_Max,
            "max_neibor_diff": self.max_neibor_diff,
            "max_iter_diff": self.max_iter_diff,
            "safety_mode": self._safety_mode,
        }

    def _apply_voltages(self, vs: np.ndarray) -> np.ndarray:
        """Low-level voltage application via UDP driver."""
        vs = np.clip(vs, self.V_Min, self.V_Max)
        if _enable_check_max_voltage_gap := self.max_iter_diff > 0:
            gap = vs - self._last_voltages
            direction = np.sign(gap)
            abs_gap = np.abs(gap)
            while abs_gap.any():
                abs_gap = np.clip(abs_gap - self.max_iter_diff, 0, self.V_Max)
                self.udp_driver.set_voltages(vs + direction * abs_gap)
                time.sleep(self.MIN_TIME_DELAY)
        self.udp_driver.set_voltages(vs)
        self._last_voltages = vs.copy()
        return self._last_voltages

    def initialize(self) -> None:
        self.set_hv(hv=True)

    def reset_all(self):
        self.send_voltages(np.zeros(self.DM_NUM), 0.01)
        if (ret := self.c_driver.reset_all()) == 0:
            self._last_voltages = np.zeros_like(self._last_voltages)
        time.sleep(0.5)
        return ret

    def set_hv(self, hv: bool = True):
        ret = self.c_driver.set_hv(hv)
        time.sleep(0.5)
        return ret

    def get_neighbors(self, unit_id):
        return np.where(self.Units_Adj_Mat[unit_id, :] == 1)[0]

    def check_dm_unit_grad_safe(self, vs):
        if self.max_iter_diff <= 0:
            return True
        diff_mat = (vs[:, None] - vs[None, :]) * self.Units_Adj_Mat
        return not np.any(diff_mat[diff_mat > self.max_neibor_diff])

    def _reset_nerbors_voltage_in_range(self, unit_id, voltages, checked_mask=None):
        if checked_mask is None:
            checked_mask = np.zeros_like(self.Units_Adj_Mat, dtype=bool)
        min_v, max_v = (
            voltages[unit_id] - self.max_neibor_diff,
            voltages[unit_id] + self.max_neibor_diff,
        )
        for nerbor in self.get_neighbors(unit_id):
            if not checked_mask[unit_id, nerbor]:
                voltages[nerbor] = np.clip(voltages[nerbor], min_v, max_v)
                checked_mask[unit_id, nerbor] = checked_mask[nerbor, unit_id] = True
                self._reset_nerbors_voltage_in_range(nerbor, voltages, checked_mask)
        return voltages


class DMUdp:
    global HEAD_WITH_ECHO, HEAD, REG_IDS
    HEAD_WITH_ECHO = "10 01 2c".split(" ")
    HEAD = "30 01 2c".split(" ")
    REG_IDS = [0, 16384, 32768, 49152]  # 0x00, 0x40, 0x80, 0xc0 to dec

    def __init__(self):
        self.ip = "192.168.6.10"
        # test ip reachable
        try:
            socket.inet_aton(self.ip)
        except socket.error:
            raise AssertionError("device connection error.")
        self.port = 1001
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.dm_num = 64

    @staticmethod
    def _num_hex(num: int):
        hex_16 = hex(int(num))[2:].zfill(4)
        return hex_16[:2] + " " + hex_16[2:]

    @staticmethod
    def _voltage_hex(num: float, registry: int):
        _num = int((num + 500) / 1000 * 4096)
        _num = min(_num, 4095)
        _num = max(_num, 820)
        _num += REG_IDS[registry % 4]
        hex_16 = DMUdp._num_hex(_num)
        return hex_16

    def _send(self, message):
        hex_message = bytes.fromhex(message)
        return self.sock.sendto(hex_message, (self.ip, self.port))

    def set_voltages(self, vs, with_echo=False):
        _head = HEAD_WITH_ECHO if with_echo else HEAD
        send_data = " ".join(
            _head
            + [self._num_hex(self.dm_num)]
            + [self._voltage_hex(v, i) for i, v in enumerate(vs)]
        )
        return self._send(send_data)

    def reset_all(self):
        vs = np.zeros(256)
        send_data = " ".join(
            HEAD_WITH_ECHO
            + [self._num_hex(256)]
            + [self._voltage_hex(v, i) for i, v in enumerate(vs)]
        )
        return self._send(send_data) & self._send("10 00 00 00 01 00 03")

    def set_hv(self, hv: bool):
        raise NotImplementedError("set_hv func not Implement")


class DMSdk:
    def __init__(self):
        self.dm_num = 64
        path = findlibs.find("Drv_UDPST")
        if path is None:
            raise Exception("Drv_UDPST.dll not found.")

        dll = cdll.LoadLibrary(path)

        dll.GetConnection2.restype = c_bool
        dll.GetConnection2.argtypes = []

        dll.SetVoltages.restype = c_bool
        dll.SetVoltages.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double, ndim=1, shape=(self.dm_num)),
            c_int32,
            c_int32,
        ]

        dll.SetVoltagesNoEcho.restype = c_bool
        dll.SetVoltagesNoEcho.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double, ndim=1, shape=(self.dm_num)),
            c_int32,
            c_int32,
        ]

        dll.SetHV.restype = c_bool
        dll.SetHV.argtypes = [c_bool, c_bool]

        dll.GetVoltages.restype = c_bool
        dll.GetVoltages.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double, ndim=1, shape=(self.dm_num)),
            c_int32,
            c_int32,
        ]

        self._dll = dll
        assert self._dll.GetConnection2(), "device connection error."

    def set_voltages(self, vs: np.ndarray, with_echo=False):
        func = self._dll.SetVoltages if with_echo else self._dll.SetVoltagesNoEcho
        return func(vs, c_int32(0), c_int32(self.dm_num))

    def reset_all(self):
        return self._dll.ResetAll()

    def set_hv(self, hv: bool):
        return self._dll.SetHV(c_bool(hv), c_bool(True))

    def get_hv(self):
        hv_status = c_bool(False)
        if self._dll.GetHV(byref(hv_status)):
            return hv_status
        else:
            raise Exception("device connection error.")

    def get_voltages(self):
        voltages = np.zeros(self.dm_num, dtype=np.double)
        if self._dll.GetVoltages(voltages, c_int32(0), c_int32(self.dm_num)):
            return voltages
        else:
            raise Exception("device connection error.")
