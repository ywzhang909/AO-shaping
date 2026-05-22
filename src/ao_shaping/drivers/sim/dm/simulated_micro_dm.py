"""Simulated MicroDM for testing and development without hardware."""

from typing import Any

import numpy as np

from loguru import logger

from ao_shaping.drivers.device_base import DeviceState, DeviceType
from ao_shaping.drivers.dm.base import DM
from ao_shaping.drivers.dm._registry import register_dm
from ao_shaping.drivers.dm.MicroDM import (
    MicroDMVoltageError,
    RelayState,
    VOLTAGE_MIN,
    VOLTAGE_MAX,
)


@register_dm("sim_micro")
class SimMicroDM(DM):
    """Simulation mode for MicroDM without hardware.

    Use this class for testing and development without actual hardware.
    Mirrors the real MicroDM interface but operates purely in memory.

    Attributes:
        DM_Num: Number of actuators (50).
        V_Min: Minimum voltage (-20.0 V).
        V_Max: Maximum voltage (120.0 V).
    """

    @classmethod
    def is_reachable(cls) -> bool:
        return True

    DM_Num: int = 50
    V_Min: float = VOLTAGE_MIN
    V_Max: float = VOLTAGE_MAX

    @property
    def DM_NUM(self) -> int:
        """Alias for DM_Num (base class expects uppercase)."""
        return self.DM_Num

    device_type = DeviceType.DM
    manufacturer = "R50Power"
    model = "MicroDM-50-Sim"

    def __init__(self, device_id: str = "", safety_mode: bool = True):
        """Initialize simulated MicroDM.

        Args:
            device_id: Unique device identifier.
            safety_mode: If True, send_voltages ramps from current to target.
        """
        super().__init__(safety_mode=safety_mode)
        self._device_id = device_id

        self._state = DeviceState.DISCONNECTED
        self._relay_state = RelayState.OFF

        logger.debug("SimMicroDM initialized")

    def open(self) -> None:
        """Open simulated connection."""
        self._state = DeviceState.READY
        logger.info("SimMicroDM connection opened")

    def close(self) -> None:
        """Close simulated connection."""
        self._state = DeviceState.DISCONNECTED
        logger.info("SimMicroDM connection closed")

    def is_connected(self) -> bool:
        """Check if simulated connection is active."""
        return self._state == DeviceState.READY

    def get_hardware_info(self) -> dict[str, Any]:
        """Get simulated hardware info."""
        return {
            "manufacturer": self.manufacturer,
            "model": self.model,
            "channel_count": self.DM_Num,
            "voltage_range": [self.V_Min, self.V_Max],
            "relay_state": self._relay_state.name,
            "simulation": True,
        }

    def get_actuator_positions(self) -> np.ndarray:
        """Get simulated actuator positions."""
        return self._last_voltages.copy()

    def transform(self, cmd: np.ndarray) -> np.ndarray:
        """Transform normalized command to voltage range.

        Maps [-1, 1] linearly to [V_Min, V_Max].
        """
        cmd = np.clip(cmd, -1.0, 1.0)
        return (cmd + 1.0) * (self.V_Max - self.V_Min) / 2.0 + self.V_Min

    def send(self, cmd: np.ndarray | float) -> np.ndarray:
        """Send command to simulated DM."""
        if isinstance(cmd, np.ndarray):
            return self.send_voltages(cmd)
        if isinstance(cmd, (int, float)):
            return self.set_all_channel_voltage(float(cmd))
        raise MicroDMVoltageError(f"Unsupported command type: {type(cmd)}")

    def _apply_voltages(self, vs: np.ndarray) -> np.ndarray:
        """Apply voltages to simulated DM."""
        vs = np.clip(vs, self.V_Min, self.V_Max)
        self._last_voltages = vs.copy()
        return self._last_voltages

    def send_voltages(self, vs: np.ndarray, wait_time_s: float = 0.0) -> np.ndarray:
        """Send simulated voltage array with optional safety ramping."""
        vs = np.asarray(vs, dtype=np.float64)
        if vs.shape != (self.DM_Num,):
            raise MicroDMVoltageError(
                f"Expected {self.DM_Num} voltages, got {vs.shape}"
            )
        return super().send_voltages(vs, wait_time_s=wait_time_s)

    def set_channel_voltage(self, channel: int, voltage: float) -> None:
        """Set simulated channel voltage."""
        if not 0 <= channel < self.DM_Num:
            raise MicroDMVoltageError(
                f"Channel must be 0-{self.DM_Num - 1}, got {channel}"
            )
        voltage = float(np.clip(voltage, self.V_Min, self.V_Max))
        self._last_voltages[channel] = voltage

    def set_all_voltage_by_arr(self, voltages: np.ndarray) -> None:
        """Set all simulated channels by array."""
        voltages = np.asarray(voltages, dtype=np.float64)
        voltages = np.clip(voltages, self.V_Min, self.V_Max)
        self._last_voltages = voltages.copy()

    def set_all_channel_voltage(self, voltage: float) -> np.ndarray:
        """Set all simulated channels to same voltage.

        Returns:
            The applied voltage array.
        """
        voltage = float(np.clip(voltage, self.V_Min, self.V_Max))
        self._last_voltages = np.full(self.DM_Num, voltage)
        return self._last_voltages.copy()

    def set_relay_state(self, state: bool) -> None:
        """Set simulated relay state."""
        self._relay_state = RelayState.ON if state else RelayState.OFF
        logger.info(f"SimMicroDM relay {'opened' if state else 'closed'}")

    def reset_all(self) -> None:
        """Reset simulated DM to zero voltages."""
        self._last_voltages = np.zeros(self.DM_Num)

    def __repr__(self) -> str:
        return f"SimMicroDM(channels={self.DM_Num}, state={self._state.name})"
