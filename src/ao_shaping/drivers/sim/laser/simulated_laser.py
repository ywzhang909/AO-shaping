"""模拟激光器设备。

本模块提供一个继承自 Device 的模拟激光器, 内部走数值仿真, 同时仍能接入现有的
设备框架。
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from ao_shaping.drivers.device_base import DeviceState, DeviceType
from ao_shaping.drivers.sim.base import OpticalDevice, SimulatedDeviceError


class SimulatedLaserError(SimulatedDeviceError):
    """模拟激光器相关错误的异常。"""
    pass


class SimulatedLaser(OpticalDevice):
    """模拟激光光源。

    本类提供一个按指定参数生成波前的模拟激光器。它继承自 OpticalDevice, 以接入
    光学仿真框架。

    Attributes:
        power: 激光功率, 单位瓦。
        wavelength: 工作波长, 单位纳米。
        aperture: 光束口径直径, 单位米。
        beam_quality: 光束质量因子 M²。

    Example:
        >>> laser = SimulatedLaser(power=100, wavelength=1064)
        >>> with laser:
        ...     wave = laser.generate()
        ...     print(f"Generated wave with power {wave.power}")
    """

    device_type = DeviceType.LASER
    manufacturer = "Simulation"
    model = "Simulated Laser"

    def __init__(
        self,
        device_id: str = "",
        power: float = 100.0,
        wavelength: float = 1064.0,
        aperture: float = 0.2,
        beam_quality: float = 1.0,
        enable_noise: bool = True,
        random_seed: int | None = None,
    ):
        """初始化模拟激光器。

        Args:
            device_id: 设备唯一标识。
            power: 激光功率, 单位瓦。
            wavelength: 波长, 单位纳米。
            aperture: 光束口径, 单位米。
            beam_quality: 光束质量因子 M²。
            enable_noise: 是否加入噪声。
            random_seed: 用于可复现性的随机种子。
        """
        super().__init__(device_id, wavelength, enable_noise, random_seed)

        self.power = power
        self.wavelength = wavelength
        self.aperture = aperture
        self.beam_quality = beam_quality

        self._output_enabled = False
        self._current_wave = None

        self._register_parameters()
        self._register_capabilities()

        logger.debug(
            f"SimulatedLaser initialized: "
            f"power={power}W, wavelength={wavelength}nm, "
            f"aperture={aperture}m"
        )

    def _register_parameters(self) -> None:
        """注册激光器参数。"""
        self.register_parameter(
            "power",
            default_value=self.power,
            min_value=0.0,
            max_value=1000.0,
            unit="W",
            description="Laser output power",
        )
        self.register_parameter(
            "wavelength",
            default_value=self.wavelength,
            min_value=300.0,
            max_value=2000.0,
            unit="nm",
            description="Laser wavelength",
        )
        self.register_parameter(
            "aperture",
            default_value=self.aperture,
            min_value=0.01,
            max_value=1.0,
            unit="m",
            description="Beam aperture diameter",
        )
        self.register_parameter(
            "beam_quality",
            default_value=self.beam_quality,
            min_value=1.0,
            max_value=10.0,
            unit="",
            description="Beam quality factor M²",
        )

    def _register_capabilities(self) -> None:
        """注册激光器能力。"""
        self.register_capability(
            "generate",
            description="Generate laser wavefront",
            return_type=object,
        )
        self.register_capability(
            "set_power",
            description="Set laser power",
            parameters=["power"],
        )
        self.register_capability(
            "set_wavelength",
            description="Set laser wavelength",
            parameters=["wavelength"],
        )

    # ========== SimulatedDevice 实现 ==========

    def compute(self, *args, **kwargs) -> Any:
        """生成激光器波前。"""
        return self.generate()

    # ========== 激光器专用方法 ==========

    def generate(self, npix: int = 512, dpix: float = 1e-3) -> Any:
        """生成激光器波前。

        Args:
            npix: 像素数。
            dpix: 像元尺寸, 单位米。

        Returns:
            波前对象 (兼容 sim.digitaltwin.Wave)。
        """
        if not self.is_connected():
            raise RuntimeError("Laser not connected")

        if not self._output_enabled:
            logger.warning("Laser output is disabled")

        self._set_state(DeviceState.BUSY)

        try:
            # 在 digitaltwin 可用时用它生成波前
            wave = self._create_wavefront(npix, dpix)

            if self.beam_quality > 1.0:
                wave = self._apply_beam_quality(wave)

            self._current_wave = wave
            self._output_enabled = True

            logger.debug(f"Generated wavefront: power={self.power}W")
            return wave
        finally:
            self._set_state(DeviceState.READY)

    def _create_wavefront(self, npix: int, dpix: float) -> Any:
        """构造波前对象。

        Args:
            npix: 像素数。
            dpix: 像元尺寸。

        Returns:
            波前对象。
        """
        # 尝试使用 digitaltwin 的基类
        try:
            from sim.digitaltwin.base import Wave as DTWave

            wave = DTWave()
            wave.change_grid(npix, dpix)
            wave.wavelength = self.wavelength * 1e-9  # 换算成米
            wave.refractive = 1.0  # 空气

            # 生成高斯光束
            r = wave.r
            radius = self.aperture / 2 / np.sqrt(2)
            amplitude = np.exp(-(r / radius) ** 2)

            # 设定波前
            wave.wavefront = amplitude * np.exp(0j)

            # 缩放到目标功率
            from sim.digitaltwin import utilities as utils
            intensity = utils.wf2intensity(wave.wavefront, wave.refractive)
            power = intensity.sum() * dpix ** 2
            wave.scale_power(self.power)

            return wave
        except ImportError:
            # 回退: 构造简单的波前字典
            logger.warning("Using fallback wavefront (sim.digitaltwin not available)")
            return self._create_simple_wavefront(npix, dpix)

    def _create_simple_wavefront(self, npix: int, dpix: float) -> dict:
        """构造简单波前字典 (回退)。

        Args:
            npix: 像素数。
            dpix: 像元尺寸。

        Returns:
            简单波前字典。
        """
        # 构造坐标网格
        x = np.linspace(-npix * dpix / 2, npix * dpix / 2 - dpix, npix)
        y = np.linspace(-npix * dpix / 2, npix * dpix / 2 - dpix, npix)
        xx, yy = np.meshgrid(x, y)
        r = np.sqrt(xx ** 2 + yy ** 2)

        # 高斯光束
        radius = self.aperture / 2 / np.sqrt(2)
        amplitude = np.exp(-(r / radius) ** 2)

        wavefront = amplitude * np.exp(0j)

        return {
            "wavefront": wavefront,
            "npix": npix,
            "dpix": dpix,
            "x": xx,
            "y": yy,
            "r": r,
            "wavelength": self.wavelength * 1e-9,
            "refractive": 1.0,
            "power": self.power,
        }

    def _apply_beam_quality(self, wave: Any) -> Any:
        """按光束质量因子 M² 施加退化。

        Args:
            wave: 输入波前。

        Returns:
            已施加光束质量退化的波前。
        """
        # 简化实现: 按 M² 加入相位畸变
        try:
            from sim.digitaltwin import screens as dt_screens
            from sim.digitaltwin import base as dt_base

            # 给湍流相位屏造一个假环境
            env = dt_base.Environment()
            env.Cn2 = 1e-15  # 弱湍流
            env.L0 = 1.0
            env.l0 = 0.01

            # 施加轻微畸变
            screen = dt_screens.TurbulentScreen(0.1, env, harmonic=0)
            screen.out(wave)

            return wave
        except ImportError:
            return wave

    def set_power(self, power: float) -> None:
        """设置激光功率。

        Args:
            power: 功率, 单位瓦。
        """
        if not (0 <= power <= 1000):
            raise ValueError(f"Power {power} out of range [0, 1000]")

        self.power = power
        self.set_parameter_value("power", power)
        logger.info(f"Laser power set to {power} W")

    def set_wavelength(self, wavelength: float) -> None:
        """设置激光波长。

        Args:
            wavelength: 波长, 单位纳米。
        """
        if not (300 <= wavelength <= 2000):
            raise ValueError(f"Wavelength {wavelength} out of range [300, 2000]")

        self.wavelength = wavelength
        self.set_parameter_value("wavelength", wavelength)
        logger.info(f"Laser wavelength set to {wavelength} nm")

    def enable_output(self, enabled: bool) -> None:
        """启用或禁用激光输出。

        Args:
            enabled: True 启用, False 禁用。
        """
        self._output_enabled = enabled
        logger.info(f"Laser output {'enabled' if enabled else 'disabled'}")

    def is_output_enabled(self) -> bool:
        """检查输出是否已启用。"""
        return self._output_enabled

    # ========== OpticalDevice 实现 ==========

    def process(self, wave: Any) -> Any:
        """处理波前 (对激光器直通透传)。

        Args:
            wave: 输入波前。

        Returns:
            同一个波前。
        """
        return wave
