"""
Closed-loop control orchestrator.

Manages the feedback loop: measure, compensate delay, compute control,
apply to hardware, record state, and check convergence criteria.
Hardware I/O is injected via the ``measure_func`` and ``apply_func`` callbacks.

Supports six control laws via controller instances from
:mod:`ao_shaping.algorithm.controller`.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable

import numpy as np
from loguru import logger

from ao_shaping.algorithm.controller import (
    AdaptiveGainController,
    BaseController,
    ControlLaw,
    LoopConfig,
    PIDController,
    LeakyIntegratorController,
    QuadraticGaussianController,
    LQGController,
    PredictiveController,
)


class AOClosedLoop:
    """Closed-loop control orchestrator.

    Manages the feedback loop: measure, compensate delay, compute control,
    apply to hardware, record state, and check convergence criteria.
    Hardware I/O is injected via the ``measure_func`` and ``apply_func``
    callbacks.

    Supports six control laws via controller instances, each encapsulating
    its own internal state and algorithm-specific logic.
    """

    def __init__(
        self,
        D: np.ndarray,
        D_pinv: np.ndarray,
        s_ref: np.ndarray,
        mask_indices: np.ndarray,
        config: LoopConfig,
        excluded_piston: bool = True,
        excluded_tip_tilt: bool = False,
    ) -> None:
        """Initialize the closed-loop controller with system matrices.

        Args:
            D: Response matrix of shape (n_meas, n_modes).
            D_pinv: Pseudo-inverse of shape (n_modes, n_meas).
            s_ref: Reference slope vector of shape (n_meas,).
            mask_indices: Valid sub-aperture indices for slope filtering.
            config: Loop configuration.
            excluded_piston: Whether piston mode is excluded from control.
            excluded_tip_tilt: Whether tip/tilt modes are excluded.
        """
        self.D = D
        self.D_pinv = D_pinv
        self.s_ref = s_ref
        self.mask_indices = mask_indices
        self.cfg = config
        self.excluded_piston = excluded_piston
        self.excluded_tip_tilt = excluded_tip_tilt

        self.n_meas = D.shape[0]
        self.n_modes = D.shape[1]

        # Shared state tracking (used across all control laws)
        self.a_history: list[np.ndarray] = []
        self.rms_history: list[float] = []
        self.s_history: list[np.ndarray] = []
        self.u_history: list[np.ndarray] = []
        self.delay_buffer: deque = deque(maxlen=config.delay_steps + 2)

        # Controller instances for each control law
        n = self.n_modes
        cfg = config
        self.controllers: dict[ControlLaw, BaseController] = {
            ControlLaw.PID: PIDController(n, cfg.dt, D_pinv, s_ref, cfg.Kp, cfg.Ki, cfg.Kd),
            ControlLaw.LEAKY_INTEGRATOR: LeakyIntegratorController(
                n, cfg.dt, D_pinv, s_ref, cfg.gain_schedule
            ),
            ControlLaw.QUADRATIC_GAUSSIAN: QuadraticGaussianController(
                n, cfg.dt, D_pinv, s_ref, cfg.Q_diag, cfg.R_scalar
            ),
            ControlLaw.LQG: LQGController(n, cfg.dt, D_pinv, s_ref, D, cfg.Q_diag, cfg.R_scalar),
            ControlLaw.PREDICTIVE: PredictiveController(
                n, cfg.dt, D_pinv, s_ref, cfg.horizon, cfg.delay_steps, cfg.Q_diag, cfg.R_scalar
            ),
            ControlLaw.ADAPTIVE_GAIN: AdaptiveGainController(
                n, cfg.dt, D_pinv, s_ref, cfg.gain_schedule, cfg.Kp, cfg.leak
            ),
        }

    def run(
        self,
        measure_func: Callable[[], tuple[np.ndarray, float]],
        apply_func: Callable[[np.ndarray], None],
        control_law: ControlLaw = ControlLaw.LEAKY_INTEGRATOR,
        callback: Callable[[np.ndarray, float, int], None] | None = None,
    ) -> dict:
        """Execute closed-loop control.

        Runs the feedback loop: initial measurement, then iterates
        measurement → delay compensation → control computation →
        hardware application → state recording → convergence check.

        Args:
            measure_func: Callable that returns (delta_slopes, rms).
            apply_func: Callable that applies the control output u to hardware.
            control_law: Control law selection.
            callback: Optional per-step callback(u, rms, k).

        Returns:
            Dictionary with results:
            - converged, diverged (bool)
            - n_iter (int)
            - rms_initial, rms_final (float)
            - improvement_db (float)
            - a_history, rms_history, s_history, u_history (ndarray)
            - control_law (str)
            - final_coefficients (ndarray)
        """
        cfg = self.cfg
        controller = self.controllers[control_law]

        logger.info(f"{'='*60}")
        logger.info(f"Closed-loop control start: {control_law.value}")
        logger.info(f"  Modes: {self.n_modes}, Target RMS: {cfg.rms_target}λ")
        logger.info(f"  Max iter: {cfg.max_iter}, Delay: {cfg.delay_steps} steps")
        logger.info(f"{'='*60}")

        # Initial measurement (open loop)
        logger.info("[Init] Open-loop measurement...")
        s0, rms0 = measure_func()
        logger.info(f"  Initial RMS: {rms0:.4f}λ")

        converged = False
        diverged = False
        u = np.zeros(self.n_modes)
        k = -1

        for k in range(cfg.max_iter):
            t_start = time.perf_counter()

            # 1. Measure current slopes
            s_meas, rms_meas = measure_func()

            # 2. Feed RMS to controller (adaptive controllers use this)
            controller.record_rms(rms_meas)

            # 3. Delay compensation (using buffered history)
            if cfg.delay_steps > 0 and len(self.delay_buffer) >= cfg.delay_steps:
                a_delayed = self.delay_buffer[-cfg.delay_steps]
                s_pred = self.D @ a_delayed
                blend = 0.7
                s_meas = blend * s_meas + (1 - blend) * s_pred

            # 4. Compute control output
            u = controller.compute(s_meas, k)

            # 5. Apply control to hardware
            apply_func(u)

            # 6. Record state
            self.a_history.append(u.copy())
            self.rms_history.append(rms_meas)
            self.s_history.append(s_meas.copy())
            self.u_history.append(u.copy())
            self.delay_buffer.append(u.copy())

            # 7. Per-step callback
            if callback:
                callback(u, rms_meas, k)

            # 8. Convergence check
            if rms_meas < cfg.rms_target:
                logger.info(f"Target reached @ iter {k}, RMS={rms_meas:.4f}λ")
                converged = True
                break

            # 9. Stall detection
            if k >= cfg.stall_window:
                recent_vals = self.rms_history[-cfg.stall_window:]
                rel_change = abs(recent_vals[-1] - recent_vals[0]) / (recent_vals[0] + 1e-10)
                if rel_change < cfg.stall_tol:
                    logger.info(f"Stall detected @ iter {k}, RMS={rms_meas:.4f}λ")
                    converged = True
                    break

            # 10. Divergence protection
            if k > 5 and rms_meas > 1.5 * rms0:
                logger.warning(
                    f"Divergence @ iter {k}, RMS={rms_meas:.4f}λ >> {rms0:.4f}λ"
                )
                diverged = True
                break

            # 11. Timing control
            elapsed = time.perf_counter() - t_start
            sleep_time = max(0, cfg.dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

            if k % 10 == 0:
                g_str = (
                    f" g={controller.current_gain:.2f}"
                    if isinstance(controller, AdaptiveGainController)
                    else ""
                )
                logger.info(f"  iter {k:3d}: RMS={rms_meas:.4f}λ{g_str}")

        # Result summary
        final_rms = self.rms_history[-1] if self.rms_history else rms0
        result_dict = {
            "converged": converged,
            "diverged": diverged,
            "n_iter": k + 1,
            "rms_initial": rms0,
            "rms_final": final_rms,
            "improvement_db": (
                20 * np.log10(rms0 / (final_rms + 1e-10)) if rms0 > 0 else 0.0
            ),
            "a_history": np.array(self.a_history) if self.a_history else np.array([]),
            "rms_history": np.array(self.rms_history) if self.rms_history else np.array([]),
            "s_history": np.array(self.s_history) if self.s_history else np.array([]),
            "u_history": np.array(self.u_history) if self.u_history else np.array([]),
            "control_law": control_law.value,
            "final_coefficients": u.copy(),
        }

        logger.info(f"\n{'='*60}")
        status = "Converged" if converged else "Diverged" if diverged else "Not converged"
        logger.info(f"Result: {status}")
        logger.info(f"  Iterations: {result_dict['n_iter']}, RMS: {rms0:.4f}λ → {final_rms:.4f}λ")
        logger.info(f"  Improvement: {result_dict['improvement_db']:.1f} dB")
        logger.info(f"{'='*60}")

        return result_dict

    def reset(self) -> None:
        """Reset all internal state (shared tracking and all controllers)."""
        for ctrl in self.controllers.values():
            ctrl.reset()
        self.a_history.clear()
        self.rms_history.clear()
        self.s_history.clear()
        self.u_history.clear()
        self.delay_buffer.clear()
