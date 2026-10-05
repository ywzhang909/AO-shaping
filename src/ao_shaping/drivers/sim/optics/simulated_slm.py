"""模拟 SLM 与光学元件。

本模块提供光学器件的仿真实现, 包含 SLM (空间光调制器)、透镜、光阑以及其他能接入
仿真框架的光学元件。
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from ao_shaping.drivers.device_base import DeviceState, DeviceType
from ao_shaping.drivers.sim.base import SimulatedDevice, SimulatedDeviceError, WavefrontProcessor


class SimulatedSLMError(SimulatedDeviceError):
    """模拟 SLM 相关错误的异常。"""
    pass


class SimulatedSLM(WavefrontProcessor):
    """模拟空间光调制器。

    本类提供一个对波前施加相位图样的模拟 SLM。它继承自 WavefrontProcessor。

    Attributes:
        resolution: SLM 分辨率 (宽, 高)。
        phase_range: 最大相位调制范围, 单位弧度。

    Example:
        >>> slm = SimulatedSLM(resolution=(1920, 1080))
        >>> with slm:
        ...     phase = np.random.rand(1080, 1920) * 2 * np.pi
        ...     slm.set_phase(phase)
        ...     output = slm.process(input_wave)
    """

    device_type = DeviceType.SLM
    manufacturer = "Simulation"
    model = "Simulated SLM"

    def __init__(
        self,
        device_id: str = "",
        resolution: tuple = (1920, 1080),
        phase_range: float = 2 * np.pi,
        wavelength: float = 1064.0,
        enable_noise: bool = False,
        random_seed: int | None = None,
    ):
        """初始化模拟 SLM。

        Args:
            device_id: 设备唯一标识。
            resolution: SLM 分辨率 (宽, 高)。
            phase_range: 最大相位范围, 单位弧度。
            wavelength: 工作波长, 单位 nm。
            enable_noise: 是否加入相位噪声。
            random_seed: 用于可复现性的随机种子。
        """
        super().__init__(
            device_id,
            wavelength,
            resolution[1],  # npix = 高
            8e-6,  # 典型 SLM 的 dpix ~ 8μm
            enable_noise,
            random_seed,
        )

        self._resolution = resolution
        self.phase_range = phase_range

        self._current_phase: np.ndarray | None = None
        self._phase_loaded = False

        self._register_parameters()
        self._register_capabilities()

        logger.debug(
            f"SimulatedSLM initialized: resolution={resolution}, "
            f"phase_range={phase_range:.2f}π"
        )

    def _register_parameters(self) -> None:
        """注册 SLM 参数。"""
        self.register_parameter(
            "phase_range",
            default_value=self.phase_range,
            min_value=np.pi,
            max_value=4 * np.pi,
            unit="rad",
            description="Maximum phase modulation range",
        )
        self.register_parameter(
            "gamma",
            default_value=1.0,
            min_value=0.5,
            max_value=3.0,
            unit="",
            description="Gamma correction factor",
        )

    def _register_capabilities(self) -> None:
        """注册 SLM 能力。"""
        self.register_capability(
            "set_phase",
            description="Set phase pattern",
            parameters=["phase"],
        )
        self.register_capability(
            "process",
            description="Apply phase to wavefront",
            parameters=["wave"],
            return_type=object,
        )

    # ========== SimulatedDevice 实现 ==========

    def compute(self, *args, **kwargs) -> Any:
        """对波前施加相位。"""
        if len(args) < 1:
            raise ValueError("Wave argument required")
        return self.process(args[0])

    # ========== SLM 专用方法 ==========

    def set_phase(self, phase: np.ndarray) -> None:
        """在 SLM 上设置相位图样。

        Args:
            phase: 二维相位数组, 单位弧度。
        """
        if phase.shape != self._resolution[::-1]:  # 注意: (高, 宽)
            raise ValueError(
                f"Phase shape {phase.shape} doesn't match "
                f"SLM resolution {self._resolution}"
            )

        # 归一化到相位范围
        gamma = self.get_parameter_value("gamma")
        if gamma != 1.0:
            phase = np.power(phase / self.phase_range, 1.0 / gamma) * self.phase_range

        self._current_phase = np.clip(phase, 0, self.phase_range)
        self._phase_loaded = True

        logger.debug(f"Phase pattern loaded: shape={phase.shape}")

    def get_phase(self) -> np.ndarray | None:
        """获取当前相位图样。

        Returns:
            当前相位图样, 或 None。
        """
        return self._current_phase.copy() if self._current_phase is not None else None

    def clear(self) -> None:
        """清除相位图样 (置零)。"""
        self._current_phase = np.zeros(self._resolution[::-1])
        self._phase_loaded = False
        logger.debug("Phase pattern cleared")

    # ========== WavefrontProcessor 实现 ==========

    def process(self, wave: Any) -> Any:
        """对波前施加 SLM 相位图样。

        Args:
            wave: 输入波前 (兼容 sim.digitaltwin.Wave)。

        Returns:
            已施加相位的波前。
        """
        if not self.is_connected():
            raise RuntimeError("SLM not connected")

        if not self._phase_loaded:
            logger.warning("No phase pattern loaded, returning input wave")
            return wave

        self._set_state(DeviceState.BUSY)

        try:
            # 尝试使用 digitaltwin 的光学元件
            try:
                from sim.digitaltwin.optics import SLM as DT_SLM

                # 构造 SLM 并施加
                slm = DT_SLM(self._current_phase, wave.dpix)
                slm.out(wave)

                return wave
            except ImportError:
                # 回退: 直接施加相位
                return self._apply_phase_direct(wave)
        finally:
            self._set_state(DeviceState.READY)

    def _apply_phase_direct(self, wave: Any) -> Any:
        """直接对波前施加相位 (回退)。

        Args:
            wave: 输入波前。

        Returns:
            修改后的波前。
        """
        # 必要时重采样相位
        phase = self._current_phase
        if hasattr(wave, 'npix') and hasattr(wave, 'dpix'):
            if phase.shape != (wave.npix, wave.npix):
                try:
                    from sim.digitaltwin import utilities as utils
                    phase = utils.matrix_size_trans(
                        phase, self.dpix, wave.npix, wave.dpix
                    )
                except ImportError:
                    # 简单重采样
                    from scipy import ndimage
                    factor = wave.npix / phase.shape[0]
                    phase = ndimage.zoom(phase, factor)

        # 施加相位
        if hasattr(wave, 'wavefront'):
            wave.change_wf(phase=phase)

        return wave


class SimulatedLens(SimulatedDevice):
    """模拟聚焦透镜。

    Example:
        >>> lens = SimulatedLens(focus_length=0.5)
        >>> output = lens.process(input_wave)
    """

    device_type = DeviceType.OTHER
    manufacturer = "Simulation"
    model = "Simulated Lens"

    def __init__(
        self,
        device_id: str = "",
        focus_length: float = 0.5,
        wavelength: float = 1064.0,
    ):
        """初始化模拟透镜。

        Args:
            device_id: 设备唯一标识。
            focus_length: 焦距, 单位米 (正值 = 会聚)。
            wavelength: 波长, 单位 nm。
        """
        super().__init__(device_id)

        self.focus_length = focus_length
        self.wavelength = wavelength

    def compute(self, *args, **kwargs) -> Any:
        """对波前施加透镜。"""
        if len(args) < 1:
            raise ValueError("Wave argument required")
        return self.process(args[0])

    def process(self, wave: Any) -> Any:
        """对波前施加透镜相位。

        Args:
            wave: 输入波前。

        Returns:
            聚焦后的波前。
        """
        if not self.is_connected():
            raise RuntimeError("Lens not connected")

        # 尝试 digitaltwin
        try:
            from sim.digitaltwin.optics import Lens as DT_Lens

            lens = DT_Lens(self.focus_length)
            lens.out(wave)
            return wave
        except ImportError:
            # 回退: 直接施加
            return self._apply_lens_direct(wave)

    def _apply_lens_direct(self, wave: Any) -> Any:
        """直接施加透镜相位 (回退)。"""
        if not hasattr(wave, 'r'):
            return wave

        lamd = self.wavelength * 1e-9
        focus_phase = -np.pi * wave.r ** 2 / lamd / self.focus_length

        if hasattr(wave, 'change_wf'):
            wave.change_wf(phase=focus_phase)

        return wave


class SimulatedAperture(SimulatedDevice):
    """模拟光学光阑。

    Example:
        >>> aperture = SimulatedAperture(radius=0.05)
        >>> output = aperture.process(input_wave)
    """

    device_type = DeviceType.OTHER
    manufacturer = "Simulation"
    model = "Simulated Aperture"

    def __init__(
        self,
        device_id: str = "",
        radius: float = 0.05,
    ):
        """初始化模拟光阑。

        Args:
            device_id: 设备唯一标识。
            radius: 光阑半径, 单位米 (正值 = 光阑, 负值 = 遮挡物)。
        """
        super().__init__(device_id)

        self.radius = radius

    def compute(self, *args, **kwargs) -> Any:
        """对波前施加光阑。"""
        if len(args) < 1:
            raise ValueError("Wave argument required")
        return self.process(args[0])

    def process(self, wave: Any) -> Any:
        """对波前施加光阑。

        Args:
            wave: 输入波前。

        Returns:
            已被掩模的波前。
        """
        if not self.is_connected():
            raise RuntimeError("Aperture not connected")

        # Try digitaltwin
        try:
            from sim.digitaltwin.optics import Aperture as DT_Aperture

            aperture = DT_Aperture(self.radius)
            aperture.out(wave)
            return wave
        except ImportError:
            # 回退
            return self._apply_aperture_direct(wave)

    def _apply_aperture_direct(self, wave: Any) -> Any:
        """直接施加光阑 (回退)。"""
        if not hasattr(wave, 'r'):
            return wave

        if self.radius > 0:
            mask = (np.sign(self.radius - wave.r) + 1) / 2
        elif self.radius < 0:
            mask = (np.sign(wave.r + self.radius) + 1) / 2
        else:
            mask = 1

        if hasattr(wave, 'wavefront'):
            wave.wavefront = np.real(wave.wavefront) * mask + 1j * np.imag(wave.wavefront) * mask

        return wave
