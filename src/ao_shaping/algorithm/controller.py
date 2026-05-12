"""
AO closed-loop controller module.
Control laws: PID / Leaky Integrator (LI) / Quadratic Gaussian (QG) / LQG / MPC / Adaptive Gain.

This module contains pure control law implementations and a high-level orchestrator.
Hardware-specific measurement and actuation are injected via callbacks.
"""

import time
import warnings
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from loguru import logger


class ControlLaw(Enum):
    """Control law enumeration."""
    PID = "pid"
    LEAKY_INTEGRATOR = "leaky"
    QUADRATIC_GAUSSIAN = "qg"
    LQG = "lqg"
    PREDICTIVE = "mpc"
    ADAPTIVE_GAIN = "adaptive"


@dataclass
class LoopConfig:
    """Closed-loop control configuration.

    Attributes:
        n_modes: Number of Zernike modes to control (typically n_slm_terms).
        dt: Sampling period [s].
        Kp: Proportional gain (PID).
        Ki: Integral gain (PID).
        Kd: Derivative gain (PID).
        leak: Leaky integrator leak factor.
        Q_diag: LQR state cost diagonal.
        R_scalar: LQR control cost scalar.
        horizon: Predictive control horizon.
        delay_steps: System delay in sampling periods.
        gain_schedule: Adaptive gain schedule [(start, end, gain, leak), ...].
        rms_target: Target RMS [lambda].
        max_iter: Maximum iterations.
        stall_window: Stall detection window length.
        stall_tol: Relative change threshold for stall detection.
        cancel_tile: Remove WFS tip/tilt during measurement.
    """
    n_modes: int = 15
    dt: float = 0.067
    Kp: float = 0.5
    Ki: float = 0.3
    Kd: float = 0.05
    leak: float = 0.97
    Q_diag: np.ndarray = field(default_factory=lambda: np.ones(15))
    R_scalar: float = 0.1
    horizon: int = 3
    delay_steps: int = 1
    gain_schedule: list[tuple[int, int, float, float]] = field(
        default_factory=lambda: [
            (0, 10, 0.6, 0.95),
            (10, 30, 0.4, 0.97),
            (30, 999, 0.2, 0.99),
        ]
    )
    rms_target: float = 0.05
    max_iter: int = 100
    stall_window: int = 5
    stall_tol: float = 0.02
    cancel_tile: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.Q_diag, list):
            self.Q_diag = np.array(self.Q_diag, dtype=np.float64)


@dataclass
class HardwareConfig:
    """Hardware configuration snapshot for closed-loop recovery.

    Stores SLM and WFS parameters alongside the response matrix,
    enabling automatic hardware parameter restoration during
    closed-loop optimization.
    """
    # SLM parameters
    slm_number: int = 1
    wavelength: int = 1064
    n_max: int = 10
    shift_x: int = 0
    shift_y: int = 0
    correction_csv_path: str = ""

    # WFS parameters
    mla_index: int = 2  # MlaRes enum value (2=Res768)
    exposure_time: float = 0.0
    high_speed: bool = False
    use_custom_ref: bool = False
    pupil_center: tuple[float, float] = (0.0, 0.0)
    pupil_diameter: float = 2.0

    def to_dict(self) -> dict:
        """Serialize to dictionary (for HDF5 JSON storage).

        Returns:
            Dictionary with all hardware parameters.
        """
        return {
            "slm_number": self.slm_number,
            "wavelength": self.wavelength,
            "n_max": self.n_max,
            "shift_x": self.shift_x,
            "shift_y": self.shift_y,
            "correction_csv_path": self.correction_csv_path,
            "mla_index": self.mla_index,
            "exposure_time": self.exposure_time,
            "high_speed": self.high_speed,
            "use_custom_ref": self.use_custom_ref,
            "pupil_center": [float(v) for v in self.pupil_center],
            "pupil_diameter": self.pupil_diameter,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HardwareConfig":
        """Deserialize from dictionary.

        Args:
            d: Dictionary with hardware parameters.

        Returns:
            New HardwareConfig instance.
        """
        return cls(
            slm_number=int(d.get("slm_number", 1)),
            wavelength=int(d.get("wavelength", 1064)),
            n_max=int(d.get("n_max", 10)),
            shift_x=int(d.get("shift_x", 0)),
            shift_y=int(d.get("shift_y", 0)),
            correction_csv_path=str(d.get("correction_csv_path", "")),
            mla_index=int(d.get("mla_index", 2)),
            exposure_time=float(d.get("exposure_time", 0.0)),
            high_speed=bool(d.get("high_speed", False)),
            use_custom_ref=bool(d.get("use_custom_ref", False)),
            pupil_center=tuple(d.get("pupil_center", (0.0, 0.0))),
            pupil_diameter=float(d.get("pupil_diameter", 2.0)),
        )


# =============================================================================
# Solver utilities (module-level functions, like learning_schedule in adam.py)
# =============================================================================


def solve_lqr(Q_diag: np.ndarray, R_scalar: float, n_modes: int) -> np.ndarray:
    """Solve discrete LQR gain via Riccati iteration.

    Computes the optimal state feedback gain K for the discrete-time
    LQR problem with identity dynamics (A=I, B=I).

    Args:
        Q_diag: Diagonal entries of the state cost matrix Q.
        R_scalar: Scalar control cost (R = R_scalar * I).
        n_modes: State dimension.

    Returns:
        LQR gain matrix K of shape (n_modes, n_modes).
    """
    A = np.eye(n_modes)
    B = np.eye(n_modes)
    Q = np.diag(Q_diag)
    R = np.eye(n_modes) * R_scalar

    P = Q.copy()
    for _ in range(1000):
        P_new = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        if np.max(np.abs(P_new - P)) < 1e-8:
            break
        P = P_new

    K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
    logger.debug(f"LQR gain matrix condition number: {np.linalg.cond(K):.1f}")
    return K


def solve_lqr_static(Q_diag: np.ndarray, R_scalar: float, n_modes: int) -> np.ndarray:
    """Static LQR solver (standalone, used by MPC gain computation).

    Args:
        Q_diag: Diagonal entries of the state cost matrix.
        R_scalar: Scalar control cost.
        n_modes: State dimension.

    Returns:
        LQR gain matrix K of shape (n_modes, n_modes).
    """
    A = np.eye(n_modes)
    B = np.eye(n_modes)
    Q = np.diag(Q_diag)
    R = np.eye(n_modes) * R_scalar

    P = Q.copy()
    for _ in range(1000):
        P_new = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        if np.max(np.abs(P_new - P)) < 1e-8:
            break
        P = P_new

    return np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A


def solve_mpc_gain(horizon: int, delay: int, Q_diag: np.ndarray, R_scalar: float) -> np.ndarray:
    """Compute MPC feedback gain (long-horizon LQR approximation).

    Args:
        horizon: Prediction horizon (unused in simplified implementation).
        delay: System delay in steps.
        Q_diag: Diagonal entries of the state cost matrix.
        R_scalar: Scalar control cost.

    Returns:
        MPC gain matrix K of shape (n_modes, n_modes).
    """
    K_lqr = solve_lqr_static(Q_diag * 0.5, R_scalar, len(Q_diag))
    return K_lqr * (1 - 0.1 * delay)


def get_scheduled_gain(
    k: int,
    gain_schedule: list[tuple[int, int, float, float]],
) -> tuple[float, float]:
    """Look up gain and leak values from a time-based schedule.

    Args:
        k: Current iteration index.
        gain_schedule: List of (start, end, gain, leak) tuples.

    Returns:
        Tuple of (gain, leak) for the current iteration.
    """
    for start, end, g, alpha in gain_schedule:
        if start <= k < end:
            return g, alpha
    return 0.1, 0.99


# =============================================================================
# Abstract base controller (like Base in adam.py)
# =============================================================================


class BaseController(ABC):
    """Abstract base class for all AO control laws.

    Each concrete controller implements the `compute` method that
    produces a control output vector from the current WFS measurement.
    """

    def __init__(self, dim: int, dt: float, D_pinv: np.ndarray, s_ref: np.ndarray) -> None:
        self.dim = dim
        self.dt = dt
        self.D_pinv = D_pinv
        self.s_ref = s_ref

    @abstractmethod
    def compute(self, s_meas: np.ndarray, k: int) -> np.ndarray:
        """Compute control output from current measurement.

        Args:
            s_meas: Current slope measurement vector.
            k: Current iteration index (for gain scheduling).

        Returns:
            Control output vector u of shape (dim,).
        """
        ...

    def record_rms(self, rms: float) -> None:
        """Feed back RMS measurement for adaptive controllers.

        The default implementation is a no-op. Adaptive controllers
        override this to track convergence trends.

        Args:
            rms: Current RMS value.
        """

    def reset(self) -> None:
        """Reset internal controller state."""
        pass


# =============================================================================
# Concrete controller implementations (like Adam, SGD in adam.py)
# =============================================================================


class PIDController(BaseController):
    """Proportional-Integral-Derivative controller.

    Implements standard PID feedback control with separate gain terms
    for proportional, integral, and derivative components.
    """

    def __init__(
        self,
        dim: int,
        dt: float,
        D_pinv: np.ndarray,
        s_ref: np.ndarray,
        Kp: float,
        Ki: float,
        Kd: float,
    ) -> None:
        super().__init__(dim, dt, D_pinv, s_ref)
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = np.zeros(dim)
        self.prev_error = np.zeros(dim)

    def compute(self, s_meas: np.ndarray, k: int) -> np.ndarray:
        e_s = s_meas - self.s_ref
        e_a = self.D_pinv @ e_s

        self.integral += e_a * self.dt
        derivative = (e_a - self.prev_error) / self.dt if self.dt > 0 else np.zeros_like(e_a)

        u = self.Kp * e_a + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = e_a
        return u

    def reset(self) -> None:
        self.integral = np.zeros(self.dim)
        self.prev_error = np.zeros(self.dim)


class LeakyIntegratorController(BaseController):
    """Leaky integrator with scheduled gain.

    Commonly used in AO systems for stable convergence.
    The leak factor prevents integrator windup and improves
    robustness against model mismatch.
    """

    def __init__(
        self,
        dim: int,
        dt: float,
        D_pinv: np.ndarray,
        s_ref: np.ndarray,
        gain_schedule: list[tuple[int, int, float, float]],
    ) -> None:
        super().__init__(dim, dt, D_pinv, s_ref)
        self.gain_schedule = gain_schedule
        self.a = np.zeros(dim)

    def compute(self, s_meas: np.ndarray, k: int) -> np.ndarray:
        e_s = s_meas - self.s_ref
        e_a = self.D_pinv @ e_s

        g, alpha = get_scheduled_gain(k, self.gain_schedule)
        self.a = alpha * self.a + g * e_a
        return self.a

    def reset(self) -> None:
        self.a = np.zeros(self.dim)


class QuadraticGaussianController(BaseController):
    """Deterministic LQR (quadratic Gaussian) controller.

    Uses state feedback with an offline-computed LQR gain matrix.
    Assumes identity dynamics (A=I, B=I) for the Zernike mode space.
    """

    def __init__(
        self,
        dim: int,
        dt: float,
        D_pinv: np.ndarray,
        s_ref: np.ndarray,
        Q_diag: np.ndarray,
        R_scalar: float,
    ) -> None:
        super().__init__(dim, dt, D_pinv, s_ref)
        self.Q_diag = Q_diag
        self.R_scalar = R_scalar
        self.a = np.zeros(dim)
        self._K_lqr: np.ndarray | None = None

    def compute(self, s_meas: np.ndarray, k: int) -> np.ndarray:
        a_meas = self.D_pinv @ (s_meas - self.s_ref)

        if self._K_lqr is None:
            self._K_lqr = solve_lqr(self.Q_diag, self.R_scalar, self.dim)

        u = -self._K_lqr @ self.a
        self.a = self.a + u
        return u

    def reset(self) -> None:
        self.a = np.zeros(self.dim)
        self._K_lqr = None


class LQGController(BaseController):
    """Linear Quadratic Gaussian controller.

    Combines a Kalman filter for state estimation with LQR
    state feedback for optimal control under process and
    measurement noise.
    """

    def __init__(
        self,
        dim: int,
        dt: float,
        D_pinv: np.ndarray,
        s_ref: np.ndarray,
        D: np.ndarray,
        Q_diag: np.ndarray,
        R_scalar: float,
    ) -> None:
        super().__init__(dim, dt, D_pinv, s_ref)
        self.D = D
        self.n_meas = D.shape[0]
        self.Q_diag = Q_diag
        self.R_scalar = R_scalar
        self.x_est = np.zeros(dim)
        self.P_est = np.eye(dim) * 0.1
        self._K_lqr: np.ndarray | None = None

    def compute(self, s_meas: np.ndarray, k: int) -> np.ndarray:
        # Kalman prediction
        x_pred = self.x_est
        P_pred = self.P_est + 0.01 * np.eye(self.dim)

        # Kalman update
        z = s_meas - self.s_ref
        H = self.D
        S = H @ P_pred @ H.T + 0.001 * np.eye(self.n_meas)
        K = P_pred @ H.T @ np.linalg.inv(S + 1e-6 * np.eye(self.n_meas))

        self.x_est = x_pred + K @ (z - H @ x_pred)
        self.P_est = (np.eye(self.dim) - K @ H) @ P_pred

        # LQR control
        if self._K_lqr is None:
            self._K_lqr = solve_lqr(self.Q_diag, self.R_scalar, self.dim)

        return -self._K_lqr @ self.x_est

    def reset(self) -> None:
        self.x_est = np.zeros(self.dim)
        self.P_est = np.eye(self.dim) * 0.1
        self._K_lqr = None


class PredictiveController(BaseController):
    """Model predictive controller with delay compensation.

    Uses a pre-computed feedback gain that accounts for system
    delays and a finite prediction horizon.
    """

    def __init__(
        self,
        dim: int,
        dt: float,
        D_pinv: np.ndarray,
        s_ref: np.ndarray,
        horizon: int,
        delay_steps: int,
        Q_diag: np.ndarray,
        R_scalar: float,
    ) -> None:
        super().__init__(dim, dt, D_pinv, s_ref)
        self.horizon = horizon
        self.delay_steps = delay_steps
        self.Q_diag = Q_diag
        self.R_scalar = R_scalar
        self.a = np.zeros(dim)
        self._K_mpc: np.ndarray | None = None

    def compute(self, s_meas: np.ndarray, k: int) -> np.ndarray:
        a_current = self.D_pinv @ (s_meas - self.s_ref)

        if self._K_mpc is None:
            self._K_mpc = solve_mpc_gain(self.horizon, self.delay_steps, self.Q_diag, self.R_scalar)

        u = -self._K_mpc @ a_current
        self.a = a_current + u
        return u

    def reset(self) -> None:
        self.a = np.zeros(self.dim)
        self._K_mpc = None


class AdaptiveGainController(BaseController):
    """Adaptive gain scheduling controller.

    Adjusts gain and leak factors in real-time based on the RMS
    convergence trend. Falls back to conservative parameters when
    divergence is detected.
    """

    def __init__(
        self,
        dim: int,
        dt: float,
        D_pinv: np.ndarray,
        s_ref: np.ndarray,
        gain_schedule: list[tuple[int, int, float, float]],
        default_gain: float = 0.5,
        default_leak: float = 0.97,
    ) -> None:
        super().__init__(dim, dt, D_pinv, s_ref)
        self.gain_schedule = gain_schedule
        self.a = np.zeros(dim)
        self.rms_buffer: list[float] = []
        self.current_gain = default_gain
        self.current_leak = default_leak

    def record_rms(self, rms: float) -> None:
        """Record RMS measurement for trend analysis.

        Args:
            rms: Current RMS value from the wavefront sensor.
        """
        self.rms_buffer.append(rms)

    def compute(self, s_meas: np.ndarray, k: int) -> np.ndarray:
        e_s = s_meas - self.s_ref
        e_a = self.D_pinv @ e_s

        g_base, alpha = get_scheduled_gain(k, self.gain_schedule)

        # Online adaptation based on RMS convergence trend
        if len(self.rms_buffer) >= 3:
            recent = self.rms_buffer[-3:]
            trend = (recent[-1] - recent[0]) / (len(recent) * self.dt) if self.dt > 0 else 0.0

            if trend < -0.01:
                adapt_factor = 1.0  # Fast convergence, maintain gain
            elif trend < 0.001:
                adapt_factor = 0.7  # Slow convergence, reduce gain
                alpha = min(alpha + 0.01, 0.995)
            else:
                adapt_factor = 0.3  # Divergence, emergency reduction
                alpha = 0.90
                warnings.warn(
                    f"Divergence trend detected (trend={trend:.4f}), emergency gain reduction",
                    stacklevel=2,
                )
        else:
            adapt_factor = 1.0

        g = g_base * adapt_factor
        self.current_gain = g
        self.current_leak = alpha

        self.a = alpha * self.a + g * e_a
        return self.a

    def reset(self) -> None:
        self.a = np.zeros(self.dim)
        self.rms_buffer.clear()


