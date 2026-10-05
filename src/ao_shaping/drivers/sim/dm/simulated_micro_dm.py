"""用于无硬件测试与开发的模拟 MicroDM。"""

from typing import Any

import numpy as np

from loguru import logger

from ao_shaping.drivers.device_base import DeviceState, DeviceType
from ao_shaping.drivers.dm.base import DM
from ao_shaping.drivers.dm._registry import register_dm
from ao_shaping.model.quantities import DmCommands
from ao_shaping.drivers.dm.micro import (
    MicroDMVoltageError,
    RelayState,
    VOLTAGE_MIN,
    VOLTAGE_MAX,
)


@register_dm("sim_micro")
class SimMicroDM(DM):
    """无硬件情况下 MicroDM 的仿真模式。

    本类用于在没有真实硬件时做测试与开发。它镜像真实 MicroDM 的接口, 但完全在
    内存中运行。

    Attributes:
        DM_Num: 致动器数 (50)。
        V_Min: 最小电压 (-20.0 V)。
        V_Max: 最大电压 (120.0 V)。
    """

    @classmethod
    def is_reachable(cls) -> bool:
        return True

    DM_Num: int = 50
    V_Min: float = VOLTAGE_MIN
    V_Max: float = VOLTAGE_MAX

    @property
    def DM_NUM(self) -> int:
        """DM_Num 的别名 (基类用的是大写写法)。"""
        return self.DM_Num

    device_type = DeviceType.DM
    manufacturer = "R50Power"
    model = "MicroDM-50-Sim"

    def __init__(self, device_id: str = "", safety_mode: bool = True):
        """初始化模拟 MicroDM。

        Args:
            device_id: 设备唯一标识。
            safety_mode: 为 True 时, send_voltages 会从当前电压斜坡到目标值。
        """
        super().__init__(safety_mode=safety_mode)
        self._device_id = device_id

        self._state = DeviceState.DISCONNECTED
        self._relay_state = RelayState.OFF

        logger.debug("SimMicroDM initialized")

    def open(self) -> None:
        """打开仿真连接。"""
        self._state = DeviceState.READY
        logger.info("SimMicroDM connection opened")

    def close(self) -> None:
        """关闭仿真连接。"""
        self._state = DeviceState.DISCONNECTED
        logger.info("SimMicroDM connection closed")

    def is_connected(self) -> bool:
        """检查仿真连接是否活跃。"""
        return self._state == DeviceState.READY

    def get_hardware_info(self) -> dict[str, Any]:
        """获取模拟硬件信息。"""
        return {
            "manufacturer": self.manufacturer,
            "model": self.model,
            "channel_count": self.DM_Num,
            "voltage_range": [self.V_Min, self.V_Max],
            "relay_state": self._relay_state.name,
            "simulation": True,
        }

    def get_actuator_positions(self) -> np.ndarray:
        """获取模拟致动器位置。"""
        return self._last_voltages.copy()

    def transform(self, cmd: np.ndarray) -> np.ndarray:
        """把归一化指令变换到电压量程。

        把 [-1, 1] 线性映射到 [V_Min, V_Max]。
        """
        cmd = np.clip(cmd, -1.0, 1.0)
        return (cmd + 1.0) * (self.V_Max - self.V_Min) / 2.0 + self.V_Min

    def send(self, cmd: np.ndarray | float) -> np.ndarray:
        """向模拟 DM 发送指令。"""
        if isinstance(cmd, np.ndarray):
            return self.send_voltages(cmd)
        if isinstance(cmd, (int, float)):
            return self.set_all_channel_voltage(float(cmd))
        raise MicroDMVoltageError(f"Unsupported command type: {type(cmd)}")

    def _apply_voltages(self, vs: np.ndarray) -> np.ndarray:
        """把电压施加到模拟 DM。"""
        vs = np.clip(vs, self.V_Min, self.V_Max)
        self._last_voltages = vs.copy()
        return self._last_voltages

    def send_voltages(self, vs: DmCommands | np.ndarray, wait_time_s: float = 0.0) -> DmCommands | np.ndarray:
        """发送模拟电压数组, 可选安全斜坡。"""
        values = vs.voltages if isinstance(vs, DmCommands) else np.asarray(vs, dtype=np.float64)
        if values.shape != (self.DM_Num,):
            raise MicroDMVoltageError(
                f"Expected {self.DM_Num} voltages, got {values.shape}"
            )
        return super().send_voltages(vs, wait_time_s=wait_time_s)

    def set_channel_voltage(self, channel: int, voltage: float) -> None:
        """设置模拟单通道电压。"""
        if not 0 <= channel < self.DM_Num:
            raise MicroDMVoltageError(
                f"Channel must be 0-{self.DM_Num - 1}, got {channel}"
            )
        voltage = float(np.clip(voltage, self.V_Min, self.V_Max))
        self._last_voltages[channel] = voltage

    def set_all_voltage_by_arr(self, voltages: np.ndarray) -> None:
        """以数组形式设置所有模拟通道。"""
        voltages = np.asarray(voltages, dtype=np.float64)
        voltages = np.clip(voltages, self.V_Min, self.V_Max)
        self._last_voltages = voltages.copy()

    def set_all_channel_voltage(self, voltage: float) -> np.ndarray:
        """把所有模拟通道设为同一电压。

        Returns:
            实际施加的电压数组。
        """
        voltage = float(np.clip(voltage, self.V_Min, self.V_Max))
        self._last_voltages = np.full(self.DM_Num, voltage)
        return self._last_voltages.copy()

    def set_relay_state(self, state: bool) -> None:
        """设置模拟继电器状态。"""
        self._relay_state = RelayState.ON if state else RelayState.OFF
        logger.info(f"SimMicroDM relay {'opened' if state else 'closed'}")

    def reset_all(self) -> None:
        """把模拟 DM 重置为零电压。"""
        self._last_voltages = np.zeros(self.DM_Num)

    def __repr__(self) -> str:
        return f"SimMicroDM(channels={self.DM_Num}, state={self._state.name})"
