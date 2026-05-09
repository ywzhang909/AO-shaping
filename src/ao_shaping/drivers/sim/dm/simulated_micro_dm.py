"""Simulated MicroDM for testing and development without hardware."""

from typing import Any

import numpy as np

from loguru import logger

from ao_shaping.drivers.device_base import DeviceState, DeviceType
from ao_shaping.drivers.dm.base import DM
from ao_shaping.drivers.dm.MicroDM import MicroDMVoltageError, RelayState


class SimMicroDM(DM):
    """Simulation mode for MicroDM without hardware.

    Use this class for testing and development without actual hardware.

    Attributes:
        DM_Num: Number of actuators (50).
        V_Min: Minimum voltage (-1.0V).
        V_Max: Maximum voltage (6.5V).
    """

    DM_Num: int = 50
    V_Min: float = -1.0
    V_Max: float = 6.5

    device_type = DeviceType.DM
    manufacturer = "R50Power"
    model = "MicroDM-50-Sim"

    def __init__(self, device_id: str = ""):
        """Initialize simulated MicroDM.

        Args:
            device_id: Unique device identifier.
        """
        self._device_id = device_id

        self._last_voltages: np.ndarray = np.zeros(self.DM_Num)
        self._relay_state = RelayState.OFF

        logger.debug("SimMicroDM initialized")

    def _register_parameters(self) -> None:
        """Register device parameters."""
        self.register_parameter("voltage_min", self.V_Min, description="Minimum voltage (V)")
        self.register_parameter("voltage_max", self.V_Max, description="Maximum voltage (V)")
        self.register_parameter("channel_count", self.DM_Num, description="Number of channels")

    def open(self) -> None:
        """Open simulated connection."""
        self._set_state(DeviceState.READY)
        logger.info("SimMicroDM connection opened")

    def close(self) -> None:
        """Close simulated connection."""
        self._set_state(DeviceState.DISCONNECTED)
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

    def transform(self, cmd: np.ndarray) -> np.ndarray:
        """Transform normalized command to voltage range."""
        cmd = np.clip(cmd, -1, 1)
        return (cmd + 1) * (self.V_Max - self.V_Min) / 2 + self.V_Min

    def send(self, cmd: np.ndarray | float) -> np.ndarray:
        """Send command to simulated DM."""
        if isinstance(cmd, np.ndarray):
            return self.send_voltages(cmd)
        elif isinstance(cmd, (int, float)):
            return self.set_all_channel_voltage(float(cmd))
        raise MicroDMVoltageError(f"Unsupported command type: {type(cmd)}")

    def send_voltages(self, vs: np.ndarray, wait_time_s: float = 0.0) -> np.ndarray:
        """Send simulated voltage array."""
        vs = np.clip(vs, self.V_Min, self.V_Max)
        self._last_voltages = vs
        return self._last_voltages.copy()

    def set_channel_voltage(self, channel: int, voltage: float) -> None:
        """Set simulated channel voltage."""
        voltage = np.clip(voltage, self.V_Min, self.V_Max)
        self._last_voltages[channel] = voltage

    def set_all_voltage_by_arr(self, voltage_arr: np.ndarray) -> None:
        """Set all simulated channels by array."""
        voltage_arr = np.clip(voltage_arr, self.V_Min, self.V_Max)
        self._last_voltages = voltage_arr.copy()

    def set_all_channel_voltage(self, voltage: float) -> np.ndarray:
        """Set all simulated channels to same voltage."""
        voltage = np.clip(voltage, self.V_Min, self.V_Max)
        self._last_voltages = np.full(self.DM_Num, voltage)
        return self._last_voltages.copy()

    def set_relay_state(self, state: bool) -> None:
        """Set simulated relay state."""
        self._relay_state = RelayState.ON if state else RelayState.OFF
        logger.info(f"SimMicroDM relay {'opened' if state else 'closed'}")

    def get_actuator_positions(self) -> np.ndarray:
        """Get simulated actuator positions."""
        return self._last_voltages.copy()

    def reset_all(self) -> None:
        """Reset simulated DM to zero voltages."""
        self._last_voltages = np.zeros(self.DM_Num)

    def __repr__(self) -> str:
        return f"SimMicroDM(channels={self.DM_Num}, state={self._state.name})"
