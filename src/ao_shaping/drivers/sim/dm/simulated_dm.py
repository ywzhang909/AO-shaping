"""Simulated DM for testing and development without hardware.

This module provides a generic simulated DM with configurable channel count,
noise simulation, and deformation modeling.

Attributes:
    v_min: Minimum voltage (-300).
    v_max: Maximum voltage (499).
"""

from typing import Any

import numpy as np

from ao_shaping.drivers.dm._adjacency import load_adjacency
from ao_shaping.drivers.dm._registry import register_dm
from ao_shaping.drivers.dm.base import DM
from ao_shaping.model.quantities import DmCommands


@register_dm("sim")
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

    #: Canonical base-class names. The lowercase ``v_min``/``v_max`` below were
    #: the only ones defined, so the base default of ``+/-inf`` leaked through
    #: and any caller clipping a command to the DM's range clipped to nothing.
    V_Min: float = -300.0
    V_Max: float = 499.0

    v_min: int = -300
    v_max: int = 499

    @property
    def DM_NUM(self) -> int:
        """Actuator count, as the ``DM`` base class spells it.

        ``DM.DM_Num`` forwards to ``self.DM_NUM``; without this the class raised
        ``AttributeError`` on every ``DM_Num`` access. That went unnoticed while
        the type was unregistered — ``@register_dm("sim")`` is what made the
        gap reachable.
        """
        return self.n_actuators

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

    def send_voltages(self, vs: DmCommands | np.ndarray, wait_time_s: float = 0.001) -> DmCommands | np.ndarray:
        """Send voltages to simulated DM with noise and rate limiting.

        Args:
            vs: Voltage array to send.
            wait_time_s: Wait time per voltage step.

        Returns:
            Current voltage array after simulation.
        """
        typed = isinstance(vs, DmCommands)
        if typed:
            if vs.n_actuators != self.channel or vs.range_min < self.v_min or vs.range_max > self.v_max:
                raise ValueError("DmCommands actuator count or voltage range does not match this DM")
            values = vs.voltages
        else:
            values = np.asarray(vs, dtype=np.float64)
        if values.shape != (self.channel,):
            raise ValueError(f"Expected {self.channel} voltages, got {values.shape}")
        vs = np.clip(values, self.v_min, self.v_max)
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
        self._publish_to_optics(self.__last_v)
        if typed:
            return DmCommands(self.__last_v, self.v_min, self.v_max, self.channel)
        return self.__last_v

    def _publish_to_optics(self, voltages: np.ndarray) -> None:
        """Push the achieved voltages into the shared optical model's DM phase.

        Without this the DM is invisible to ``SimPibSystem``: its ``far_field``
        summed only the SLM command, so DM-driven loops (``pib``, ``combined``)
        ran to completion while the voltages changed nothing.

        The import is deferred because ``drivers/sim/__init__`` imports this
        module *before* ``slm_pib_sim``, so a module-level import would resolve
        against a partially initialised package.

        The *achieved* voltages are published, not the requested ones, so the
        optics sees the same rate-limited and noise-laden state the driver
        actually applied.
        """
        from loguru import logger

        from ao_shaping.drivers.sim.slm_pib_sim import get_system

        try:
            optics = get_system().dm_optics
        except (ImportError, AttributeError) as exc:
            logger.warning("sim DM not coupled into the optical model: {}", exc)
            return
        if optics.n_actuators != self.channel:
            logger.warning(
                "sim DM has {} actuators but the optical model expects {}; "
                "DM voltages will not reach the pupil",
                self.channel,
                optics.n_actuators,
            )
            return
        optics.set_voltages(voltages)
        self._publish_to_wfs(voltages)

    def _publish_to_wfs(self, voltages: np.ndarray) -> None:
        """Also push the voltages to the simulated sensor, if one is live.

        The far-field model and the WFS model are separate optical states: the
        sensor measures the pupil, not the far field, so publishing only to
        ``SimPibSystem`` left ``wf``/``rms-zernike`` reading a flat pupil and
        reporting zero RMS no matter what the DM did.

        Silently skipped when no simulated sensor exists -- on hardware paths,
        or when only the far-field model is in use.
        """
        from ao_shaping.drivers.sim.wfs.simulated_wfs import get_active_sim_wfs

        sensor = get_active_sim_wfs()
        if sensor is None or sensor.dm_optics.n_actuators != self.channel:
            return
        sensor.dm_optics.set_voltages(voltages)

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
        """Adjacency matrix, shared with the real DM drivers.

        Replaces a byte-for-byte duplicate of this loader that read a
        CWD-relative ``data/dm_adj.txt`` and carried its own fallback grid.
        """
        return load_adjacency()


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
