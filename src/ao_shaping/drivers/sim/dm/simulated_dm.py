"""Simulated DM for testing and development without hardware.

This module provides a generic simulated DM with configurable channel count,
noise simulation, and deformation modeling.

Attributes:
    v_min: Minimum voltage (-300).
    v_max: Maximum voltage (499).
"""

from typing import Any

import numpy as np

from ao_shaping.drivers.dm.base import DM


class SimulateDM(DM):
    """Generic simulated deformable mirror.

    Simulates a deformable mirror with configurable number of actuators,
    voltage limits, noise, and deformation modeling. Useful for testing
    optimization algorithms without hardware.

    Attributes:
        channel: Number of channels (default: 64).
        n_actuators: Number of actuators (default: 64).
        disabled_actuators: List of disabled actuator indices.
        v_min: Minimum voltage.
        v_max: Maximum voltage.
    """

    channel: int = 64
    n_actuators: int = 64
    disabled_actuators: list[int] = []

    v_min: int = -300
    v_max: int = 499

    def __init__(
        self,
        max_iter_diff: int = 20,
        max_neibor_diff: int = 0,
        keep_when_exit: bool = True,
        noise_level: float = 0.01,
    ):
        """Initialize simulated DM.

        Args:
            max_iter_diff: Maximum voltage change per iteration.
            max_neibor_diff: Maximum voltage difference between neighbors.
            keep_when_exit: Whether to keep voltage on exit.
            noise_level: Noise level for voltage simulation.
        """
        super().__init__()
        self.units_adj_mat = self._load_adj_txt()
        self.__last_v = np.zeros(self.channel)
        self.max_iter_diff = max_iter_diff
        self.max_neibor_diff = max_neibor_diff
        self.__keep_when_exit = keep_when_exit
        self.noise_level = noise_level
        self.hv_state = False
        self.deformation_history: list[np.ndarray] = []
        self.voltage_history: list[np.ndarray] = []
        # Simulated deformation model parameters
        self.deformation_model = np.eye(self.channel) * 0.01

    def open(self) -> None:
        """Open simulated connection and initialize."""
        self.initialize()
        print("Simulated DM initialized successfully")

    def close(self) -> None:
        """Close simulated connection."""
        if not self.__keep_when_exit:
            self.reset_all()
            self.set_hv(False)
            print("Simulated DM turned off")
        print("Simulated DM connection closed")

    def transform(self, cmd: np.ndarray) -> np.ndarray:
        """Transform normalized command to voltage range."""
        cmd = np.clip(cmd, -1, 1)
        return (cmd + 1) * (self.v_max - self.v_min) / 2 + self.v_min

    def send(self, cmd: np.ndarray) -> np.ndarray:
        """Send command to simulated DM."""
        if isinstance(cmd, np.ndarray):
            return self.send_voltages(cmd)
        raise ValueError("Unsupported command type. Expected numpy array of voltages.")

    def get_actuator_positions(self) -> np.ndarray:
        """Get simulated actuator positions in a grid layout."""
        x = np.linspace(0, 10, int(np.sqrt(self.channel)))
        y = np.linspace(0, 10, int(np.sqrt(self.channel)))
        xx, yy = np.meshgrid(x, y)
        return np.column_stack((xx.ravel(), yy.ravel()))

    def initialize(self) -> None:
        """Initialize DM: turn on HV and reset voltages."""
        self.set_hv(hv=True)
        self.reset_all()

    def reset_all(self) -> int:
        """Reset all channels to zero voltage."""
        self.send_voltages(np.zeros(self.channel), 0.01)
        self.__last_v = np.zeros_like(self.__last_v)
        return 0

    def send_voltages(self, vs: np.ndarray, wait_time_s: float = 0.001) -> np.ndarray:
        """Send voltages to simulated DM with noise and rate limiting.

        Args:
            vs: Voltage array to send.
            wait_time_s: Wait time per voltage step.

        Returns:
            Current voltage array after simulation.
        """
        vs = np.clip(vs, self.v_min, self.v_max)
        # Add noise to simulate real hardware
        noisy_vs = vs + np.random.normal(0, self.noise_level, size=vs.shape)
        # Apply voltage rate limiting logic
        __gap = noisy_vs - self.__last_v
        if self.max_iter_diff > 0:
            _direction = np.sign(__gap)
            _abs_gap = np.abs(__gap)
            while _abs_gap.any():
                _step = np.minimum(_abs_gap, self.max_iter_diff)
                self.__last_v += _direction * _step
                _abs_gap -= _step
        else:
            self.__last_v = noisy_vs

        # Record voltage history
        self.voltage_history.append(self.__last_v.copy())
        # Calculate simulated deformation (voltage to deformation)
        deformation = self._voltage_to_deformation(self.__last_v)
        self.deformation_history.append(deformation)
        return self.__last_v

    def set_hv(self, hv: bool = True) -> int:
        """Set high voltage state."""
        self.hv_state = hv
        return 0

    def get_hv_state(self) -> bool:
        """Get current high voltage state."""
        return self.hv_state

    def _voltage_to_deformation(self, voltages: np.ndarray) -> np.ndarray:
        """Convert voltages to deformation using a linear model.

        Args:
            voltages: Input voltage array.

        Returns:
            Deformation array.
        """
        deformation = np.dot(self.deformation_model, voltages)
        # Add deformation noise
        deformation += np.random.normal(0, self.noise_level * 0.1, size=deformation.shape)
        return deformation

    @staticmethod
    def _load_adj_txt() -> np.ndarray:
        """Load adjacency matrix or create a default grid structure.

        Returns:
            2D array of shape (64, 64) with adjacency information.
        """
        try:
            return np.loadtxt('data/dm_adj.txt')
        except (FileNotFoundError, OSError):
            # If no adjacency matrix file, create a simple grid structure
            size = int(np.sqrt(64))
            adj = np.zeros((64, 64), dtype=int)
            for i in range(64):
                row, col = i // size, i % size
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < size and 0 <= nc < size:
                        j = nr * size + nc
                        adj[i, j] = 1
            return adj

    def get_deformation_history(self) -> np.ndarray:
        """Get deformation history as array.

        Returns:
            Array of deformation values over time.
        """
        return np.array(self.deformation_history)

    def get_voltage_history(self) -> np.ndarray:
        """Get voltage history as array.

        Returns:
            Array of voltage values over time.
        """
        return np.array(self.voltage_history)

    def clear_history(self) -> None:
        """Clear deformation and voltage history."""
        self.deformation_history = []
        self.voltage_history = []
