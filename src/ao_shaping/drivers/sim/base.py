"""模拟设备基类。

本模块为仿真光学设备提供基类, 在 Device 框架之上扩展仿真专用功能。

架构:
    - SimulatedDevice: 所有仿真设备的基类
    - OpticalDevice: 光学仿真设备的专用基类
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np
from loguru import logger

from ao_shaping.drivers.device_base import (
    Device,
    DeviceError,
    DeviceState,
    DeviceType,
)


class SimulatedDeviceError(DeviceError):
    """模拟设备错误的基异常。"""
    pass


class SimulatedDevice(Device):
    """仿真设备的基类。

    本类在 Device 基类之上扩展仿真专用功能, 为 AO-Shaping 框架里所有仿真硬件
    设备提供基础。

    Attributes:
        device_type: 仿真设备为 DeviceType.OTHER
        manufacturer: "Simulation"
        model: 具体型号名

    Example:
        >>> class MySimulatedDevice(SimulatedDevice):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self._register_parameters()
        ...
        ...     def compute(self, *args):
        ...         # 这里的仿真逻辑
        ...         return result
    """

    device_type = DeviceType.OTHER
    manufacturer = "Simulation"
    model = "Generic Simulated Device"

    def __init__(
        self,
        device_id: str = "",
        enable_noise: bool = True,
        random_seed: int | None = None,
    ):
        """初始化仿真设备。

        Args:
            device_id: 设备唯一标识。为空时自动生成。
            enable_noise: 是否在仿真中加入噪声。
            random_seed: 用于得到可复现结果的随机种子。
        """
        super().__init__(device_id)

        self._simulation_enabled = True
        self._enable_noise = enable_noise
        self._random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)

        # 为仿真更新元数据
        self._metadata.manufacturer = self.manufacturer
        self._metadata.model = self.model

        logger.debug(f"SimulatedDevice {self.__class__.__name__} initialized")

    # ========== Device 基类实现 ==========

    def open(self) -> None:
        """打开仿真设备连接。"""
        self._set_state(DeviceState.CONNECTING)
        # 模拟连接延迟
        self._set_state(DeviceState.READY)
        logger.info(f"Simulated device {self.device_id} opened")

    def close(self) -> None:
        """关闭仿真设备连接。"""
        self._set_state(DeviceState.DISCONNECTED)
        logger.info(f"Simulated device {self.device_id} closed")

    def is_connected(self) -> bool:
        """检查设备是否已连接。"""
        return self._state == DeviceState.READY

    def get_hardware_info(self) -> dict[str, Any]:
        """获取仿真的硬件信息。

        Returns:
            包含仿真专用硬件信息的字典。
        """
        return {
            "device_type": "simulation",
            "model": self.model,
            "manufacturer": self.manufacturer,
            "simulation_enabled": self._simulation_enabled,
            "noise_enabled": self._enable_noise,
            "random_seed": self._random_seed,
        }

    # ========== 仿真专用方法 ==========

    @abstractmethod
    def compute(self, *args, **kwargs) -> Any:
        """执行仿真计算。

        本方法必须由子类实现, 以完成实际的仿真计算。

        Returns:
            仿真结果 (类型取决于具体实现)。
        """
        pass

    def reset(self) -> None:
        """把仿真状态重置回初始条件。"""
        logger.debug(f"Simulated device {self.device_id} reset")

    def set_seed(self, seed: int) -> None:
        """设置随机种子, 使仿真可复现。

        Args:
            seed: 随机种子值。
        """
        self._random_seed = seed
        self._rng = np.random.default_rng(seed)
        logger.debug(f"Random seed set to {seed}")

    def set_noise(self, enabled: bool) -> None:
        """启用或禁用仿真中的噪声。

        Args:
            enabled: 是否加入噪声。
        """
        self._enable_noise = enabled
        logger.debug(f"Noise {'enabled' if enabled else 'disabled'}")

    def _generate_noise(self, shape: tuple, scale: float = 1.0) -> np.ndarray:
        """生成噪声数组。

        Args:
            shape: 输出数组形状。
            scale: 噪声幅度的缩放因子。

        Returns:
            噪声数组。
        """
        if self._enable_noise:
            return self._rng.normal(0, scale, shape)
        return np.zeros(shape)


class OpticalDevice(SimulatedDevice):
    """光学仿真设备的基类。

    本类在 SimulatedDevice 之上扩展光学专用功能, 包括波长处理与波前处理。

    Attributes:
        wavelength: 工作波长, 单位纳米。

    Example:
        >>> class MyOpticDevice(OpticalDevice):
        ...     def __init__(self, wavelength=1064):
        ...         super().__init__()
        ...         self.wavelength = wavelength
        ...
        ...     def process(self, wave):
        ...         # 处理波前
        ...         return processed_wave
    """

    def __init__(
        self,
        device_id: str = "",
        wavelength: float = 1064.0,
        enable_noise: bool = True,
        random_seed: int | None = None,
    ):
        """初始化光学仿真设备。

        Args:
            device_id: 设备唯一标识。
            wavelength: 工作波长, 单位 nm。
            enable_noise: 是否加入噪声。
            random_seed: 用于可复现性的随机种子。
        """
        super().__init__(device_id, enable_noise, random_seed)

        self.wavelength = wavelength
        self._input_wave: Any | None = None
        self._output_wave: Any | None = None

    def set_input(self, wave: Any) -> None:
        """设置待处理的输入波前。

        Args:
            wave: 输入波前对象。
        """
        self._input_wave = wave
        logger.debug(f"Input wave set for {self.__class__.__name__}")

    def get_output(self) -> Any:
        """获取处理后的输出波前。

        Returns:
            输出波前对象。
        """
        return self._output_wave

    @abstractmethod
    def process(self, wave: Any) -> Any:
        """处理输入波前。

        本方法必须由子类实现, 以完成实际的光学处理。

        Args:
            wave: 待处理的输入波前。

        Returns:
            处理后的波前。
        """
        pass


class WavefrontProcessor(OpticalDevice):
    """波前处理设备的基类。

    用于操控波前的光学仿真专用设备, 例如 SLM、透镜与光阑。
    """

    def __init__(
        self,
        device_id: str = "",
        wavelength: float = 1064.0,
        npix: int = 512,
        dpix: float = 1e-3,
        enable_noise: bool = True,
        random_seed: int | None = None,
    ):
        """初始化波前处理器。

        Args:
            device_id: 设备唯一标识。
            wavelength: 工作波长, 单位 nm。
            npix: 波前数组的像素数。
            dpix: 像元尺寸, 单位米。
            enable_noise: 是否加入噪声。
            random_seed: 用于可复现性的随机种子。
        """
        super().__init__(device_id, wavelength, enable_noise, random_seed)

        self.npix = npix
        self.dpix = dpix
        self._phase_pattern: np.ndarray | None = None

    def set_phase(self, phase: np.ndarray) -> None:
        """设置相位图样。

        Args:
            phase: 二维相位数组, 单位弧度。
        """
        if phase.shape != (self.npix, self.npix):
            logger.warning(
                f"Phase shape {phase.shape} doesn't match device "
                f"({self.npix}, {self.npix})"
            )
        self._phase_pattern = phase

    def get_phase(self) -> np.ndarray | None:
        """获取当前相位图样。

        Returns:
            当前相位图样, 或 None。
        """
        return self._phase_pattern
