"""大气仿真 —— 相位屏与传播。

本模块用相位屏提供湍流与热晕效应的仿真实现, 可作为兼容设备接口的包装器使用。
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from ao_shaping.drivers.device_base import DeviceState, DeviceType
from ao_shaping.drivers.sim.beam_backend import make_beam_config, turbulence_phase
from ao_shaping.drivers.sim.base import SimulatedDevice, WavefrontProcessor


class SimulatedTurbulentScreen(WavefrontProcessor):
    """仿真的湍流相位屏。

    本类提供模拟的湍流相位屏, 对波前施加大气湍流效应。

    Example:
        >>> screen = SimulatedTurbulentScreen(Cn2=1e-15, L0=1.0, l0=0.01)
        >>> with screen:
        ...     output = screen.process(input_wave)
    """

    device_type = DeviceType.OTHER
    manufacturer = "Simulation"
    model = "Turbulent Phase Screen"

    def __init__(
        self,
        device_id: str = "",
        dist: float = 1.0,
        Cn2: float = 1e-15,
        L0: float = 1.0,
        l0: float = 0.01,
        harmonic: int = 1,
    ):
        """初始化湍流相位屏。

        Args:
            device_id: 设备唯一标识。
            dist: 传播距离, 单位米。
            Cn2: 折射率结构常数。
            L0: 外尺度, 单位米。
            l0: 内尺度, 单位米。
            harmonic: 次谐波阶数。
        """
        super().__init__(device_id, wavelength=1064.0, npix=512, dpix=1e-3)

        self.dist = dist
        self.Cn2 = Cn2
        self.L0 = L0
        self.l0 = l0
        self.harmonic = harmonic

        self._screen = None
        self._opd = None

        logger.debug(
            f"TurbulentScreen initialized: Cn2={Cn2}, "
            f"L0={L0}m, l0={l0}m"
        )

    def _register_parameters(self) -> None:
        """注册参数。"""
        self.register_parameter(
            "Cn2",
            default_value=self.Cn2,
            min_value=1e-20,
            max_value=1e-12,
            unit="m^{-2/3}",
            description="Refractive index structure constant",
        )
        self.register_parameter(
            "L0",
            default_value=self.L0,
            min_value=0.1,
            max_value=100.0,
            unit="m",
            description="Outer scale",
        )
        self.register_parameter(
            "l0",
            default_value=self.l0,
            min_value=1e-4,
            max_value=0.1,
            unit="m",
            description="Inner scale",
        )

    # ========== SimulatedDevice 实现 ==========

    def compute(self, *args, **kwargs) -> Any:
        """对波前施加湍流相位屏。"""
        if len(args) < 1:
            raise ValueError("Wave argument required")
        return self.process(args[0])

    # ========== WavefrontProcessor 实现 ==========

    def process(self, wave: Any) -> Any:
        """对波前施加湍流相位屏。

        Args:
            wave: 输入波前。

        Returns:
            已施加湍流的波前。
        """
        if not self.is_connected():
            raise RuntimeError("Turbulent screen not connected")

        self._set_state(DeviceState.BUSY)
        try:
            from sim.digitaltwin import screens as dt_screens
            from sim.digitaltwin import base as dt_base

            env = dt_base.Environment()
            env.Cn2 = self.Cn2
            env.L0 = self.L0
            env.l0 = self.l0

            screen = dt_screens.TurbulentScreen(self.dist, env, self.harmonic)
            screen.out(wave)

            self._opd = screen.opd
            return wave
        except Exception as exc:
            logger.warning(f"digitaltwin turbulence path unavailable ({exc}), using fallback")
            return self._apply_turbulence_fallback(wave)
        finally:
            self._set_state(DeviceState.READY)

    def _apply_turbulence_fallback(self, wave: Any) -> Any:
        """直接施加湍流 (回退路径)。"""
        npix = getattr(wave, "npix", 512)
        dpix = getattr(wave, "dpix", 1e-3)
        wavelength = getattr(wave, "wavelength", getattr(wave, "lamd", 1064e-9))

        phase = self._generate_kolmogorov_phase(
            npix=npix,
            dpix=dpix,
            wavelength=float(wavelength),
            cn2=self.Cn2,
            l0=self.l0,
            l_max=self.L0,
            distance=self.dist,
        )
        self._opd = phase

        if hasattr(wave, 'change_wf'):
            wave.change_wf(phase=phase)

        return wave

    def _generate_kolmogorov_phase(
        self,
        npix: int,
        dpix: float,
        wavelength: float,
        cn2: float,
        l0: float,
        l_max: float,
        distance: float,
    ) -> np.ndarray:
        """生成 Von Kármán/Kolmogorov 湍流相位屏。

        Args:
            npix: 像素数。
            dpix: 像元尺寸。

        Returns:
            相位屏数组。
        """
        if cn2 <= 0 or distance <= 0:
            return np.zeros((npix, npix), dtype=float)
        beam_cfg = make_beam_config(
            n_grid=npix,
            aperture_size=npix * dpix,
            wavelength=wavelength,
            cn2=cn2,
            l_max=l_max,
            l_min=l0,
            propagation_distance=distance,
        )
        return turbulence_phase(
            beam_cfg,
            cn2=cn2,
            l_max=l_max,
            l_min=l0,
            propagation_distance=distance,
            rng=self._rng,
        )

    def get_opd(self) -> np.ndarray | None:
        """获取当前 OPD (光程差)。

        Returns:
            OPD 数组, 或 None。
        """
        return self._opd.copy() if self._opd is not None else None


class SimulatedThermalScreen(WavefrontProcessor):
    """仿真的热晕相位屏。

    Example:
        >>> screen = SimulatedThermalScreen(absorb=1e-5, wind=2.0)
        >>> with screen:
        ...     output = screen.process(input_wave)
    """

    device_type = DeviceType.OTHER
    manufacturer = "Simulation"
    model = "Thermal Phase Screen"

    def __init__(
        self,
        device_id: str = "",
        dist: float = 1.0,
        absorb: float = 1e-5,
        wind_x: float = 2.0,
        wind_y: float = 0.0,
        solve_mode: str = "FFT_non_Isobaric",
    ):
        """初始化热晕相位屏。

        Args:
            device_id: 设备唯一标识。
            dist: 传播距离, 单位米。
            absorb: 吸收系数。
            wind_x: x 方向风速 (m/s)。
            wind_y: y 方向风速 (m/s)。
            solve_mode: 求解模式 ('Green', 'FFT_Isobaric', 'FFT_non_Isobaric')。
        """
        super().__init__(device_id, wavelength=1064.0, npix=512, dpix=1e-3)

        self.dist = dist
        self.absorb = absorb
        self.wind_x = wind_x
        self.wind_y = wind_y
        self.solve_mode = solve_mode

        self._opd = None

        logger.debug(
            f"ThermalScreen initialized: absorb={absorb}, "
            f"wind=({wind_x}, {wind_y}) m/s"
        )

    # ========== SimulatedDevice Implementation ==========

    def compute(self, *args, **kwargs) -> Any:
        """对波前施加热晕相位屏。"""
        if len(args) < 1:
            raise ValueError("Wave argument required")
        return self.process(args[0])

    # ========== WavefrontProcessor Implementation ==========

    def process(self, wave: Any) -> Any:
        """对波前施加热晕相位屏。

        Args:
            wave: 输入波前。

        Returns:
            已施加热效应的波前。
        """
        if not self.is_connected():
            raise RuntimeError("Thermal screen not connected")

        self._set_state(DeviceState.BUSY)

        try:
            # 尝试 digitaltwin
            try:
                from sim.digitaltwin import screens as dt_screens
                from sim.digitaltwin import base as dt_base

                # 构造环境
                env = dt_base.Environment()
                env.absorb = self.absorb
                env.wind_x = self.wind_x
                env.wind_y = self.wind_y
                env.density = 1.177  # 标准空气
                env.Cp = 1005
                env.Cv = 718
                env.temperature = 288
                env.Cs2 = 331.3 ** 2
                env.gravity = 9.81

                # 构造并施加相位屏
                screen = dt_screens.ThermalScreen(self.dist, env, self.solve_mode)
                screen.out(wave)

                self._opd = screen.opd
                return wave
            except ImportError:
                logger.warning("sim.digitaltwin not available, using fallback")
                return wave  # 回退: 空操作
        finally:
            self._set_state(DeviceState.READY)

    def get_opd(self) -> np.ndarray | None:
        """获取当前 OPD。"""
        return self._opd.copy() if self._opd is not None else None


class SimulatedATP(SimulatedDevice):
    """仿真的大气传输 (ATP)。

    本类模拟激光在大气中的传播, 包含湍流与热晕效应。

    Example:
        >>> atp = SimulatedATP(prop_dist=3000, layers=10, Cn2=1e-15)
        >>> with atp:
        ...     output = atp.propagate(input_wave)
    """

    device_type = DeviceType.OTHER
    manufacturer = "Simulation"
    model = "Atmospheric Propagation"

    def __init__(
        self,
        device_id: str = "",
        prop_dist: float = 3000.0,
        layers: int = 10,
        Cn2: float = 1e-15,
        Thermal: bool = False,
        Turbulent: bool = True,
    ):
        """初始化大气传输。

        Args:
            device_id: 设备唯一标识。
            prop_dist: 传播距离, 单位米。
            layers: 相位屏层数。
            Cn2: 折射率结构常数。
            Thermal: 启用热晕。
            Turbulent: 启用湍流。
        """
        super().__init__(device_id)

        self.prop_dist = prop_dist
        self.layers = layers
        self.Cn2 = Cn2
        self.Thermal = Thermal
        self.Turbulent = Turbulent

        self._atp = None

        logger.debug(
            f"ATP initialized: distance={prop_dist}m, layers={layers}, "
            f"Cn2={Cn2}, Turbulent={Turbulent}, Thermal={Thermal}"
        )

    # ========== SimulatedDevice Implementation ==========

    def compute(self, *args, **kwargs) -> Any:
        """让波在大气中传播。"""
        if len(args) < 1:
            raise ValueError("Wave argument required")
        return self.propagate(args[0])

    def propagate(self, wave: Any) -> Any:
        """让波在大气中传播。

        Args:
            wave: 输入波前。

        Returns:
            传播后的波前。
        """
        if not self.is_connected():
            raise RuntimeError("ATP not connected")

        self._set_state(DeviceState.BUSY)

        try:
            # 尝试 digitaltwin
            try:
                from sim.digitaltwin import atp as dt_atp
                from sim.digitaltwin import base as dt_base

                # 构造初始环境
                env = dt_base.Environment()
                env.absorb = 5e-6
                env.scatter = 5e-5
                env.wind_x = 2.0
                env.wind_y = 0.0
                env.density = 1.177
                env.Cp = 1005
                env.Cv = 718
                env.temperature = 288
                env.nT = -8.6e-7
                env.atm = 1.0
                env.Cs2 = 331.3 ** 2
                env.gravity = 9.81
                env.Cn2 = self.Cn2
                env.L0 = 1.0
                env.l0 = 0.01

                # 构造 ATP
                atp = dt_atp.ATP(
                    env_init=env,
                    prop_dist=self.prop_dist,
                    layers=self.layers,
                    Turbulent=self.Turbulent,
                    Thermal=self.Thermal,
                )

                # 传播
                atp.out(wave)

                self._atp = atp
                return wave
            except ImportError:
                logger.warning("sim.digitaltwin not available, ATP not executed")
                return wave
        finally:
            self._set_state(DeviceState.READY)

    def set_env_params(self, **kwargs) -> None:
        """设置环境参数。

        Args:
            **kwargs: 环境参数。
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.debug(f"ATP parameter set: {key}={value}")
