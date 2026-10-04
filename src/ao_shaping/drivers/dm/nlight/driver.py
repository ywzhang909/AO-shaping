"""NLight 变形镜驱动 (UDP 批量下发 + Drv_UDPST.dll C SDK).

NLight 通过 UDP datagram 批量下发 64 通道电压 (见
:mod:`ao_shaping.drivers.dm.nlight.udp`)，高压开关与电压读回走 C SDK
(:mod:`ao_shaping.drivers.dm.nlight.sdk`)。

硬件事实 (2026-09 实测确认):
  - 致动器邻接矩阵 (``Units_Adj_Mat``) 由
    :func:`ao_shaping.drivers.dm._adjacency.load_adjacency` 惰性加载,
    描述符保证 **import 期不读盘**。
  - 斜坡下发按 ``max_iter_diff`` (单步最大变化) + ``max_neibor_diff``
    (相邻致动器最大压差) 两级约束, 前者由 UDP 逐帧逼近实现。
"""

from __future__ import annotations

import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import numpy.typing as npt
from loguru import logger

from ao_shaping.drivers.dm._adjacency import lazy_adjacency, load_adjacency
from ao_shaping.drivers.dm._registry import register_dm
from ao_shaping.drivers.dm.base import DM
from ao_shaping.drivers.dm.nlight.constants import (
    HV_SETTLE_TIME_S,
    MIN_TIME_DELAY,
    NLIGHT_IP,
    NLIGHT_PORT,
    REACHABLE_TIMEOUT_S,
)
from ao_shaping.drivers.dm.nlight.nlight_constants import (
    DISABLED_ACTUATORS,
    DM_NUM,
    V_MAX,
    V_MIN,
)
from ao_shaping.drivers.dm.nlight.sdk import DMSdk
from ao_shaping.drivers.dm.nlight.udp import DMUdp
from ao_shaping.utils.io.device_config import ConfigHandler, DeviceParam, param
from ao_shaping.utils.io.file import ROOT_DIR as PROJECT_ROOT

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
    keep_when_exit: bool = param(
        default=True, cast=bool, attr="_NLight__keep_when_exit"
    )
    safety_mode: bool = param(default=True, cast=bool, attr="_safety_mode")


# 模块级单例，所有 NLight 实例共用
NLIGHT_CONFIG = ConfigHandler(_DM_CONFIG_DIR, "nlight", NLightParams)

#: ``__init__`` 参数的硬上限 (超出即物理损坏风险)。
MAX_ITER_DIFF_LIMIT = 200
MAX_NEIBOR_DIFF_LIMIT = 300.0


def _load_adj_txt() -> npt.NDArray[np.floating]:
    return load_adjacency()


@register_dm("nlight")
class NLight(DM):
    """NLight 64 通道变形镜驱动.

    UDP 通道负责批量电压下发, C SDK 负责高压开关与读回。``keep_when_exit``
    为 False 时 ``close()`` 会先归零再断高压。
    """

    DM_NUM: int = DM_NUM
    V_Min: float = V_MIN
    V_Max: float = V_MAX
    MIN_TIME_DELAY = MIN_TIME_DELAY

    _IP = NLIGHT_IP
    _PORT = NLIGHT_PORT

    disabled_actuators: list[int] = DISABLED_ACTUATORS

    Units_Adj_Mat = lazy_adjacency()

    @classmethod
    def from_params(cls, params: Any, **overrides: Any) -> Self:
        """Construct an NLight DM from its driver parameter object."""
        kwargs = {
            "max_iter_diff": getattr(params, "max_iter_diff", 20),
            "max_neibor_diff": getattr(params, "max_neibor_diff", 200),
            "keep_when_exit": getattr(params, "keep_when_exit", True),
            "safety_mode": getattr(params, "safety_mode", True),
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    @classmethod
    def is_reachable(cls) -> bool:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(REACHABLE_TIMEOUT_S)

            result = sock.connect_ex((cls._IP, cls._PORT))
            sock.close()
            return result == 0
        except OSError:
            return False

    def __init__(
        self,
        max_iter_diff: int = 20,
        max_neibor_diff: float = 200,
        keep_when_exit: bool = True,
        safety_mode: bool = True,
    ):
        self._init_values = {
            "max_iter_diff": max_iter_diff,
            "max_neibor_diff": max_neibor_diff,
            "keep_when_exit": keep_when_exit,
            "safety_mode": safety_mode,
        }
        # 使用 defaults + __init__ 参数解析（NLight 无序列号，使用默认配置）
        params = NLIGHT_CONFIG.resolve_from_config({}, init_values=self._init_values)

        assert params.max_iter_diff <= MAX_ITER_DIFF_LIMIT
        assert params.max_neibor_diff <= MAX_NEIBOR_DIFF_LIMIT

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
    def default_dm_unit_mask(self) -> npt.NDArray[np.bool_]:
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

    def transform(self, cmd: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Transform command to DM actuators"""
        return self.transform_voltage(cmd)

    def get_actuator_positions(self) -> npt.NDArray[np.floating]:
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

    def _apply_voltages(self, vs: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
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

    def reset_all(self) -> bool:
        self.send_voltages(np.zeros(self.DM_NUM), MIN_TIME_DELAY)
        if (ret := self.c_driver.reset_all()) == 0:
            self._last_voltages = np.zeros_like(self._last_voltages)
        time.sleep(HV_SETTLE_TIME_S)
        return ret

    def set_hv(self, hv: bool = True) -> bool:
        ret = self.c_driver.set_hv(hv)
        time.sleep(HV_SETTLE_TIME_S)
        return ret

    def get_neighbors(self, unit_id: int) -> npt.NDArray[np.intp]:
        return np.where(self.Units_Adj_Mat[unit_id, :] == 1)[0]

    def check_dm_unit_grad_safe(self, vs: npt.NDArray[np.floating]) -> bool:
        if self.max_iter_diff <= 0:
            return True
        diff_mat = (vs[:, None] - vs[None, :]) * self.Units_Adj_Mat
        return not np.any(diff_mat[diff_mat > self.max_neibor_diff])

    def _reset_nerbors_voltage_in_range(
        self,
        unit_id: int,
        voltages: npt.NDArray[np.floating],
        checked_mask: npt.NDArray[np.bool_] | None = None,
    ) -> npt.NDArray[np.floating]:
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


__all__ = ["NLIGHT_CONFIG", "NLight", "NLightParams"]
