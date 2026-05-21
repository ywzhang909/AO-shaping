from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class DM(ABC):
    """Abstract base class for deformable mirror drivers.

    All DM implementations must provide:
    - Voltage range (V_Min, V_Max)
    - Channel count (DM_NUM)
    - transform/send/open/close/is_connected/get_actuator_positions
    - Optional: send_voltages, set_channel_voltage, set_all_voltage_by_arr
    - Optional: safety checks (check_dm_unit_grad_safe, default_dm_unit_mask)
    """

    DM_NUM: int

    # Voltage range — subclasses override
    V_Min: float = -300.0
    V_Max: float = 500.0

    # Neighbor voltage safety — subclasses override if applicable
    max_neibor_diff: float = 200.0

    @property
    def default_dm_unit_mask(self) -> np.ndarray:
        """Default mask of active actuators (True = active).

        Subclasses override to provide hardware-specific defaults.
        """
        return np.ones(self.DM_NUM, dtype=bool)

    @abstractmethod
    def transform(self, cmd) -> np.ndarray:
        """Transform a normalized command to device-specific values."""
        ...

    @abstractmethod
    def send(self, cmd) -> np.ndarray:
        """Send a command to the DM and return the applied values."""
        ...

    @abstractmethod
    def open(self) -> None:
        """Open connection to the DM."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close connection to the DM."""
        ...

    @abstractmethod
    def get_actuator_positions(self) -> np.ndarray:
        """Get current actuator values."""
        ...

    def is_connected(self) -> bool:
        """Check if the DM is connected."""
        return False

    def get_hardware_info(self) -> dict:
        """Get hardware-specific information."""
        return {"type": type(self).__name__, "DM_NUM": self.DM_NUM}

    @property
    def DM_Num(self) -> int:
        """Alias for DM_NUM (backward compatibility)."""
        return self.DM_NUM

    def send_voltages(self, vs: np.ndarray, wait_time_s: float = 0.001) -> np.ndarray:
        """Send a voltage array to all channels.

        Default implementation delegates to :meth:`send`.
        Subclasses with hardware-specific ramping should override.

        Args:
            vs: Voltage array for all logical channels.
            wait_time_s: Sleep after sending (hardware settling time).

        Returns:
            The applied voltage array.
        """
        vs = np.clip(np.asarray(vs, dtype=np.float64), self.V_Min, self.V_Max)
        result = self.send(vs)
        import time

        time.sleep(wait_time_s)
        return result

    def set_channel_voltage(self, channel: int, voltage: float) -> None:
        """Set voltage for a single channel."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support set_channel_voltage"
        )

    def set_all_voltage_by_arr(self, voltages: np.ndarray) -> None:
        """Set all channel voltages by array."""
        self.send_voltages(voltages)

    def check_dm_unit_grad_safe(self, vs: np.ndarray) -> bool:
        """Check if neighbor voltage differences are within safe limits.

        Default implementation always returns True (no safety check).
        Subclasses with adjacency constraints should override.

        Args:
            vs: Voltage array to check.

        Returns:
            True if safe, False otherwise.
        """
        return True

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
