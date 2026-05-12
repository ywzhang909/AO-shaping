"""
Pure AO closed-loop controller.
Control laws: PID / Leaky Integrator (LI) / Quadratic Gaussian (QG) / LQG / MPC / Adaptive Gain.

This module contains only the control logic and state management.
Hardware-specific measurement and actuation are injected via measure_func and apply_func callbacks.

Usage:
    >>> from ao_shaping.algorithm.controller import AOClosedLoop, LoopConfig, ControlLaw, HardwareConfig
    >>> config = LoopConfig(n_modes=15)
    >>> loop = AOClosedLoop(D, D_pinv, s_ref, mask_indices, config)
    >>> loop.run(control_law=ControlLaw.LEAKY_INTEGRATOR,
    ...          measure_func=my_measure,
    ...          apply_func=my_apply)
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field
from collections import deque
from collections.abc import Callable
from enum import Enum

import numpy as np
from loguru import logger


class ControlLaw(Enum):
    """控制律枚举"""
    PID = "pid"
    LEAKY_INTEGRATOR = "leaky"
    QUADRATIC_GAUSSIAN = "qg"
    LQG = "lqg"
    PREDICTIVE = "mpc"
    ADAPTIVE_GAIN = "adaptive"


@dataclass
class LoopConfig:
    """闭环控制配置参数

    Attributes:
        n_modes: 控制的Zernike模式数 (通常等于 n_slm_terms)
        dt: 采样周期 [s]
        Kp: 比例增益 (PID)
        Ki: 积分增益 (PID)
        Kd: 微分增益 (PID)
        leak: 泄漏积分器泄漏因子
        Q_diag: LQR状态代价对角线
        R_scalar: LQR控制代价标量
        horizon: 预测控制时域
        delay_steps: 系统延时 (采样周期倍数)
        gain_schedule: 自适应增益调度表 [(start_iter, end_iter, gain, leak), ...]
        rms_target: 目标RMS [λ]
        max_iter: 最大迭代次数
        stall_window: 停滞检测窗口
        stall_tol: 相对变化阈值
        cancel_tile: 测量时去除WFS tip/tilt
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
    """硬件配置快照: 保存SLM和WFS参数以便闭环恢复

    用于与响应矩阵一起保存, 在闭环优化时可以自动载入硬件参数。
    """
    # SLM 参数
    slm_number: int = 1
    wavelength: int = 1064
    n_max: int = 10
    shift_x: int = 0
    shift_y: int = 0
    correction_csv_path: str = ""

    # WFS 参数
    mla_index: int = 2  # MlaRes enum value (2=Res768)
    exposure_time: float = 0.0
    high_speed: bool = False
    use_custom_ref: bool = False
    pupil_center: tuple[float, float] = (0.0, 0.0)
    pupil_diameter: float = 2.0

    def to_dict(self) -> dict:
        """序列化为字典 (用于HDF5 JSON存储)"""
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
    def from_dict(cls, d: dict) -> HardwareConfig:
        """从字典反序列化"""
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


class AOClosedLoop:
    """自适应光学闭环控制器

    核心特性:
    1. 延时补偿: 通过状态预测补偿 SLM响应 + WFS读取 + 计算 的固定延时
    2. 多种控制律: PID / LI / QG / LQG / MPC / 自适应
    3. 异常保护: RMS突增检测, 自动回退, 增益衰减
    4. 完整状态记录: 用于事后分析收敛轨迹

    注意:
        控制符号约定: 正增益将测量残差作为负反馈驱动至零。
        若系统出现发散, 尝试翻转增益符号 (将Kp取负)。
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
            D: Response matrix [n_meas, n_modes], typically deviation_response_matrix[mask_2n]
            D_pinv: Pseudo-inverse [n_modes, n_meas]
            s_ref: Reference slope vector [n_meas]
            mask_indices: Valid sub-aperture indices (for slope filtering)
            config: Loop configuration
            excluded_piston: Whether piston mode is excluded
            excluded_tip_tilt: Whether tip/tilt modes are excluded
        """
        self.D = D
        self.D_pinv = D_pinv
        self.s_ref = s_ref
        self.mask_indices = mask_indices
        self.cfg = config
        self.excluded_piston = excluded_piston
        self.excluded_tip_tilt = excluded_tip_tilt

        # 系统维度
        self.n_meas = D.shape[0]
        self.n_modes = D.shape[1]

        # 状态变量
        self.a = np.zeros(self.n_modes)
        self.a_history: list[np.ndarray] = []
        self.rms_history: list[float] = []
        self.s_history: list[np.ndarray] = []
        self.u_history: list[np.ndarray] = []

        # 延时补偿: 状态缓冲队列
        self.delay_buffer: deque = deque(maxlen=config.delay_steps + 2)

        # PID状态
        self.integral = np.zeros(self.n_modes)
        self.prev_error = np.zeros(self.n_modes)

        # LQG状态估计器
        self.x_est = np.zeros(self.n_modes)
        self.P_est = np.eye(self.n_modes) * 0.1

        # 自适应参数
        self.current_gain = config.Kp
        self.current_leak = config.leak

        # 控制律分发表
        self.controllers: dict[ControlLaw, Callable] = {
            ControlLaw.PID: self._pid_step,
            ControlLaw.LEAKY_INTEGRATOR: self._leaky_step,
            ControlLaw.QUADRATIC_GAUSSIAN: self._qg_step,
            ControlLaw.LQG: self._lqg_step,
            ControlLaw.PREDICTIVE: self._mpc_step,
            ControlLaw.ADAPTIVE_GAIN: self._adaptive_step,
        }

    # ==================== 控制律实现 ====================

    def _pid_step(self, s_meas: np.ndarray, k: int) -> np.ndarray:
        """标准PID控制"""
        cfg = self.cfg
        e_s = s_meas - self.s_ref
        e_a = self.D_pinv @ e_s

        self.integral += e_a * cfg.dt
        derivative = (e_a - self.prev_error) / cfg.dt if cfg.dt > 0 else np.zeros_like(e_a)

        u = cfg.Kp * e_a + cfg.Ki * self.integral + cfg.Kd * derivative
        self.prev_error = e_a
        return u

    def _leaky_step(self, s_meas: np.ndarray, k: int) -> np.ndarray:
        """泄漏积分器 (常用, 稳定性好)"""
        e_s = s_meas - self.s_ref
        e_a = self.D_pinv @ e_s

        g, alpha = self._get_scheduled_gain(k)

        # a_{k+1} = alpha * a_k + g * D^+ * e
        self.a = alpha * self.a + g * e_a
        return self.a

    def _qg_step(self, s_meas: np.ndarray, k: int) -> np.ndarray:
        """二次高斯控制 (确定性LQR)"""
        cfg = self.cfg
        a_meas = self.D_pinv @ (s_meas - self.s_ref)

        if not hasattr(self, "_K_lqr"):
            self._K_lqr = self._solve_lqr(cfg.Q_diag, cfg.R_scalar)

        u = -self._K_lqr @ self.a
        self.a = self.a + u
        return u

    def _lqg_step(self, s_meas: np.ndarray, k: int) -> np.ndarray:
        """线性二次高斯 (Kalman滤波 + LQR)"""
        cfg = self.cfg

        # Kalman预测
        x_pred = self.x_est
        P_pred = self.P_est + 0.01 * np.eye(self.n_modes)

        # Kalman更新
        z = s_meas - self.s_ref
        H = self.D
        S = H @ P_pred @ H.T + 0.001 * np.eye(self.n_meas)
        K = P_pred @ H.T @ np.linalg.inv(S + 1e-6 * np.eye(self.n_meas))

        self.x_est = x_pred + K @ (z - H @ x_pred)
        self.P_est = (np.eye(self.n_modes) - K @ H) @ P_pred

        # LQR控制
        if not hasattr(self, "_K_lqr"):
            self._K_lqr = self._solve_lqr(cfg.Q_diag, cfg.R_scalar)

        u = -self._K_lqr @ self.x_est
        self.a = self.x_est.copy()
        return u

    def _mpc_step(self, s_meas: np.ndarray, k: int) -> np.ndarray:
        """模型预测控制 (含延时补偿)"""
        cfg = self.cfg
        a_current = self.D_pinv @ (s_meas - self.s_ref)

        if not hasattr(self, "_K_mpc"):
            self._K_mpc = self._solve_mpc_gain(
                cfg.horizon, cfg.delay_steps, cfg.Q_diag, cfg.R_scalar
            )

        u = -self._K_mpc @ a_current
        self.a = a_current + u
        return u

    def _adaptive_step(self, s_meas: np.ndarray, k: int) -> np.ndarray:
        """自适应增益调度 + 在线调整"""
        cfg = self.cfg

        e_s = s_meas - self.s_ref
        e_a = self.D_pinv @ e_s

        g_base, alpha = self._get_scheduled_gain(k)

        # 在线自适应: 根据RMS变化率调整
        if len(self.rms_history) >= 3:
            recent = self.rms_history[-3:]
            trend = (recent[-1] - recent[0]) / (len(recent) * cfg.dt) if cfg.dt > 0 else 0.0

            if trend < -0.01:
                adapt_factor = 1.0
            elif trend < 0.001:
                adapt_factor = 0.7
                alpha = min(alpha + 0.01, 0.995)
            else:
                adapt_factor = 0.3
                alpha = 0.90
                warnings.warn(
                    f"检测到发散趋势 (trend={trend:.4f}), 紧急降增益",
                    stacklevel=2,
                )
        else:
            adapt_factor = 1.0

        g = g_base * adapt_factor
        self.current_gain = g
        self.current_leak = alpha

        self.a = alpha * self.a + g * e_a
        return self.a

    # ==================== 辅助方法 ====================

    def _get_scheduled_gain(self, k: int) -> tuple[float, float]:
        """增益调度查询"""
        for start, end, g, alpha in self.cfg.gain_schedule:
            if start <= k < end:
                return g, alpha
        return 0.1, 0.99

    def _solve_lqr(self, Q_diag: np.ndarray, R_scalar: float) -> np.ndarray:
        """离线求解离散LQR增益 (Riccati迭代)"""
        A = np.eye(self.n_modes)
        B = np.eye(self.n_modes)
        Q = np.diag(Q_diag)
        R = np.eye(self.n_modes) * R_scalar

        P = Q.copy()
        for _ in range(1000):
            P_new = Q + A.T @ P @ A - A.T @ P @ B @ \
                    np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
            if np.max(np.abs(P_new - P)) < 1e-8:
                break
            P = P_new

        K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        logger.debug(f"LQR增益矩阵条件数: {np.linalg.cond(K):.1f}")
        return K

    @staticmethod
    def _solve_mpc_gain(
        horizon: int, delay: int, Q_diag: np.ndarray, R_scalar: float
    ) -> np.ndarray:
        """求解MPC反馈增益 (长时域LQR近似)"""
        # 简化实现: 降低Q权重增加稳定性
        K_lqr = AOClosedLoop._solve_lqr_static(Q_diag * 0.5, R_scalar, len(Q_diag))
        return K_lqr * (1 - 0.1 * delay)

    @staticmethod
    def _solve_lqr_static(Q_diag: np.ndarray, R_scalar: float, n_modes: int) -> np.ndarray:
        """静态LQR求解器 (用于MPC)"""
        A = np.eye(n_modes)
        B = np.eye(n_modes)
        Q = np.diag(Q_diag)
        R = np.eye(n_modes) * R_scalar

        P = Q.copy()
        for _ in range(1000):
            P_new = Q + A.T @ P @ A - A.T @ P @ B @ \
                    np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
            if np.max(np.abs(P_new - P)) < 1e-8:
                break
            P = P_new

        return np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A

    # ==================== 主闭环流程 ====================

    def run(
        self,
        measure_func: Callable[[], tuple[np.ndarray, float]],
        apply_func: Callable[[np.ndarray], None],
        control_law: ControlLaw = ControlLaw.LEAKY_INTEGRATOR,
        callback: Callable[[np.ndarray, float, int], None] | None = None,
    ) -> dict:
        """执行闭环控制

        Args:
            measure_func: 测量回调, 返回 (delta_slopes, rms) 的元组
            apply_func: 执行回调, 接收控制器输出系数 u 并应用到硬件
            control_law: 控制律选择
            callback: 每步回调函数 callback(u, rms, k) -> None

        Returns:
            结果字典: {
                converged, diverged, n_iter,
                rms_initial, rms_final, improvement_db,
                a_history, rms_history, s_history, u_history,
                control_law, final_coefficients
            }
        """
        cfg = self.cfg
        controller = self.controllers[control_law]

        logger.info(f"{'='*60}")
        logger.info(f"闭环控制启动: {control_law.value}")
        logger.info(f"  模式数: {self.n_modes}, 目标RMS: {cfg.rms_target}λ")
        logger.info(f"  最大迭代: {cfg.max_iter}, 延时补偿: {cfg.delay_steps}步")
        logger.info(f"{'='*60}")

        # 初始测量 (开环)
        logger.info("[初始化] 开环测量...")
        s0, rms0 = measure_func()
        logger.info(f"  初始RMS: {rms0:.4f}λ")

        converged = False
        diverged = False
        k = 0

        for k in range(cfg.max_iter):
            t_start = time.perf_counter()

            # 1. 测量当前斜率
            s_meas, rms_meas = measure_func()

            # 2. 延时补偿 (使用缓冲的历史状态)
            if cfg.delay_steps > 0 and len(self.delay_buffer) >= cfg.delay_steps:
                a_delayed = self.delay_buffer[-cfg.delay_steps]
                s_pred = self.D @ a_delayed
                blend = 0.7
                s_meas = blend * s_meas + (1 - blend) * s_pred

            # 3. 控制律计算
            u = controller(s_meas, k)

            # 4. 应用控制输出到硬件 (展开+发送由 apply_func 完成)
            apply_func(u)

            # 5. 记录状态
            self.a_history.append(u.copy())
            self.rms_history.append(rms_meas)
            self.s_history.append(s_meas.copy())
            self.u_history.append(u.copy())
            self.delay_buffer.append(u.copy())

            # 6. 回调
            if callback:
                callback(u, rms_meas, k)

            # 7. 收敛判断
            if rms_meas < cfg.rms_target:
                logger.info(f"✓ 达到目标精度 @ iter {k}, RMS={rms_meas:.4f}λ")
                converged = True
                break

            # 8. 停滞检测
            if k >= cfg.stall_window:
                recent_vals = self.rms_history[-cfg.stall_window:]
                rel_change = abs(recent_vals[-1] - recent_vals[0]) / (recent_vals[0] + 1e-10)
                if rel_change < cfg.stall_tol:
                    logger.info(f"✓ 收敛停滞 @ iter {k}, RMS={rms_meas:.4f}λ")
                    converged = True
                    break

            # 9. 发散保护
            if k > 5 and rms_meas > 1.5 * rms0:
                logger.warning(
                    f"发散检测 @ iter {k}, RMS={rms_meas:.4f}λ >> {rms0:.4f}λ"
                )
                diverged = True
                if k > 0 and len(self.a_history) >= 2:
                    self.a = self.a_history[-2] * 0.5
                break

            # 10. 时序控制
            elapsed = time.perf_counter() - t_start
            sleep_time = max(0, cfg.dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

            if k % 10 == 0:
                g_str = (
                    f" g={self.current_gain:.2f}"
                    if control_law == ControlLaw.ADAPTIVE_GAIN
                    else ""
                )
                logger.info(f"  iter {k:3d}: RMS={rms_meas:.4f}λ{g_str}")

        # 结果汇总
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
            "final_coefficients": self.a.copy(),
        }

        logger.info(f"\n{'='*60}")
        status = (
            "收敛" if converged
            else "发散" if diverged
            else "未收敛"
        )
        logger.info(f"闭环结束: {status}")
        logger.info(f"  迭代: {result_dict['n_iter']}, RMS: {rms0:.4f}λ → {final_rms:.4f}λ")
        logger.info(f"  改善: {result_dict['improvement_db']:.1f} dB")
        logger.info(f"{'='*60}")

        return result_dict

    def reset(self) -> None:
        """重置所有内部状态"""
        self.a = np.zeros(self.n_modes)
        self.integral = np.zeros(self.n_modes)
        self.prev_error = np.zeros(self.n_modes)
        self.x_est = np.zeros(self.n_modes)
        self.P_est = np.eye(self.n_modes) * 0.1
        self.a_history.clear()
        self.rms_history.clear()
        self.s_history.clear()
        self.u_history.clear()
        self.delay_buffer.clear()
        self.cancel_delay_buffer = None
        self.current_gain = self.cfg.Kp
        self.current_leak = self.cfg.leak


# 模块导出
__all__ = [
    "ControlLaw",
    "LoopConfig",
    "HardwareConfig",
    "AOClosedLoop",
]
