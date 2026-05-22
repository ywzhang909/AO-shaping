from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np


class DM(ABC):
    """Abstract base class for deformable mirror drivers.

    Unified interface for all DM types:
    - Voltage-range DMs (NLight, MicroDM): send_voltages with optional ramping
    - Phase DMs (ZernikeDM, HadamardDM): send coefficients → phase pattern

    Safety mode (default ON): when enabled, send_voltages automatically
    ramps from current voltages to target in steps bounded by max_neibor_diff.
    """

    DM_NUM: int

    # Voltage range — subclasses override with hardware-specific limits
    V_Min: float = float("-inf")
    V_Max: float = float("inf")

    # Neighbor voltage safety step — subclasses override if applicable
    max_neibor_diff: float = float("inf")

    def __init__(self, safety_mode: bool = True) -> None:
        """Initialize DM with optional safety mode.

        Args:
            safety_mode: If True (default), send_voltages will automatically
                ramp from current voltages to target in steps bounded by
                max_neibor_diff. Set False for direct voltage application.
        """
        self._safety_mode = safety_mode
        dm_num = getattr(self, "DM_NUM", getattr(self, "DM_Num", 0))
        self._last_voltages: np.ndarray = np.zeros(dm_num)

    @property
    def default_dm_unit_mask(self) -> np.ndarray:
        """Default mask of active actuators (True = active)."""
        return np.ones(self.DM_NUM, dtype=bool)

    # ---- Abstract interface ----

    @classmethod
    def is_reachable(cls) -> bool:
        """Check if this DM type's hardware is reachable on the network.

        Returns True if at least one device of this type can be contacted.
        Subclasses must override with type-specific discovery logic.
        """
        return False

    @abstractmethod
    def transform(self, cmd) -> np.ndarray:
        """Transform a normalized command to device-specific values."""
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

    # ---- Default implementations ----

    def send(self, cmd) -> np.ndarray:
        """Send a command to the DM. Delegates to send_voltages for arrays."""
        if isinstance(cmd, np.ndarray):
            return self.send_voltages(cmd)
        raise ValueError(f"Unsupported command type: {type(cmd)}")

    def is_connected(self) -> bool:
        """Check if the DM is connected."""
        return False

    def get_hardware_info(self) -> dict:
        """Get hardware-specific information."""
        return {
            "type": type(self).__name__,
            "DM_NUM": self.DM_NUM,
            "safety_mode": self._safety_mode,
        }

    @property
    def DM_Num(self) -> int:
        """Alias for DM_NUM (backward compatibility)."""
        return self.DM_NUM

    # ---- Voltage transformation (common) ----

    def transform_voltage(self, cmd: np.ndarray) -> np.ndarray:
        """Transform normalized command [-1, 1] to voltage range [V_Min, V_Max].

        This is the common voltage transform used by voltage-range DMs.
        Subclasses can override for custom transform logic.
        """
        cmd = np.clip(np.asarray(cmd, dtype=np.float64), -1.0, 1.0)
        return (cmd + 1.0) * (self.V_Max - self.V_Min) / 2.0 + self.V_Min

    # ---- Voltage ramping (safety mode) ----

    def _ramp_voltages(
        self, target: np.ndarray, step_size: float | None = None
    ) -> np.ndarray:
        """Ramp from current voltages to target in bounded steps.

        Each step changes no channel by more than step_size (defaults to
        max_neibor_diff).  This prevents sudden voltage jumps that could
        damage the DM.

        Subclasses with hardware-specific ramping (e.g. NLight's iter-diff)
        should override this method.

        Args:
            target: Target voltage array.
            step_size: Max per-step change per channel. Defaults to max_neibor_diff.

        Returns:
            The final applied voltage array.
        """
        if step_size is None:
            step_size = self.max_neibor_diff

        if step_size <= 0 or not self._safety_mode:
            return self._apply_voltages(target)

        current = self._last_voltages.copy()
        gap = target - current
        direction = np.sign(gap)
        abs_gap = np.abs(gap)

        while abs_gap.any():
            abs_gap = np.clip(abs_gap - step_size, 0, self.V_Max - self.V_Min)
            intermediate = current + direction * abs_gap
            self._apply_voltages(intermediate)

        self._last_voltages = target.copy()
        return self._last_voltages

    def _apply_voltages(self, vs: np.ndarray) -> np.ndarray:
        """Apply voltages to hardware. Subclasses MUST override.

        This is the low-level method that actually sends voltages to the
        device. Called by _ramp_voltages for each step.

        Args:
            vs: Clipped voltage array.

        Returns:
            The applied voltage array.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _apply_voltages"
        )

    # ---- Public send interface ----

    def send_voltages(self, vs: np.ndarray, wait_time_s: float = 0.001) -> np.ndarray:
        """Send a voltage array to all channels with optional safety ramping.

        When safety_mode is True (default), voltages are ramped from the
        current state to the target in steps bounded by max_neibor_diff.

        Args:
            vs: Voltage array for all logical channels.
            wait_time_s: Sleep after sending (hardware settling time).

        Returns:
            The applied voltage array.
        """
        vs = np.clip(np.asarray(vs, dtype=np.float64), self.V_Min, self.V_Max)
        result = self._ramp_voltages(vs)
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
        """Check if neighbor voltage differences are within safe limits."""
        return True

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
