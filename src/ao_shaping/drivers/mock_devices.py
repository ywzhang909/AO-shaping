"""用于测试与开发的 Mock 设备。

本模块提供各种硬件设备的模拟实现, 便于在无需真实硬件的情况下进行
测试、开发与演示。
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from loguru import logger

from ao_shaping.drivers.device_base import Device, DeviceError, DeviceState, DeviceType


class MockCameraError(DeviceError):
    """Mock 相机错误时抛出的异常。"""

    pass


class MockCamera(Device):
    """用于测试的 Mock 相机设备。

    模拟一台分辨率、噪声与采集行为均可配置的相机。

    Attributes:
        device_type: DeviceType.CAMERA
        manufacturer: "Mock"
        model: "Simulated Camera"

    示例:
        >>> cam = MockCamera(device_id="mock_cam_001", resolution=(1024, 1024))
        >>> with cam:
        ...     img = cam.capture()
        ...     print(f"Captured: {img.shape}")
    """

    device_type = DeviceType.CAMERA
    manufacturer = "Mock"
    model = "Simulated Camera"

    def __init__(
        self,
        device_id: str = "",
        resolution: tuple[int, int] = (1024, 1024),
        noise_level: float = 5.0,
        simulate_delay: float = 0.01,
        random_seed: int | None = None,
    ):
        """初始化 Mock 相机。

        Args:
            device_id: 唯一设备标识。
            resolution: 图像分辨率 (宽, 高)。
            noise_level: 高斯噪声的标准差。
            simulate_delay: 模拟采集延迟, 单位秒。
            random_seed: 用于可复现输出的随机种子。为 None 时使用随机初始化。
        """
        super().__init__(device_id)

        self._resolution = resolution
        self._noise_level = noise_level
        self._simulate_delay = simulate_delay
        self._frame_counter = 0
        self._last_image: np.ndarray | None = None
        self._rng = np.random.default_rng(random_seed)

        self._register_parameters()
        self._register_capabilities()

    def _register_parameters(self) -> None:
        """注册相机专属参数。"""
        self.register_parameter(
            "exposure_time_ms",
            default_value=20.0,
            min_value=0.1,
            max_value=10000.0,
            unit="ms",
            description="Exposure time in milliseconds",
        )
        self.register_parameter(
            "gain",
            default_value=1.0,
            min_value=1.0,
            max_value=100.0,
            unit="",
            description="Analog gain",
        )
        self.register_parameter(
            "brightness",
            default_value=128.0,
            min_value=0.0,
            max_value=255.0,
            unit="",
            description="Image brightness offset",
        )
        self.register_parameter(
            "contrast",
            default_value=1.0,
            min_value=0.1,
            max_value=10.0,
            unit="",
            description="Image contrast factor",
        )
        self.register_parameter(
            "auto_exposure",
            default_value=False,
            unit="",
            description="Enable auto exposure",
        )

    def _register_capabilities(self) -> None:
        """注册相机能力。"""
        self.register_capability(
            "capture",
            description="Capture single image",
            return_type=np.ndarray,
        )
        self.register_capability(
            "capture_average",
            description="Capture averaged image",
            parameters=["n_frames"],
            return_type=np.ndarray,
        )
        self.register_capability(
            "get_resolution",
            description="Get current resolution",
            return_type=tuple,
        )

    def open(self) -> None:
        """打开 Mock 相机连接。"""
        self._set_state(DeviceState.CONNECTING)
        time.sleep(0.05)  # 模拟连接延迟
        self._frame_counter = 0
        self._set_state(DeviceState.READY)
        logger.info(f"Mock camera {self.device_id} opened")

    def close(self) -> None:
        """关闭 Mock 相机连接。"""
        self._set_state(DeviceState.DISCONNECTED)
        logger.info(f"Mock camera {self.device_id} closed")

    def is_connected(self) -> bool:
        """检查相机是否已连接。"""
        return self._state == DeviceState.READY

    def get_hardware_info(self) -> dict[str, Any]:
        """获取 Mock 硬件信息。"""
        return {
            "serial_number": f"MOCK_CAM_{self.device_id[:8]}",
            "firmware_version": "1.0.0-mock",
            "resolution": self._resolution,
            "pixel_format": "MONO8",
            "sensor_type": "MockSensor",
        }

    def _generate_image(self) -> np.ndarray:
        """生成一张带图案的合成图像。"""
        width, height = self._resolution
        brightness = self.get_parameter_value("brightness")
        contrast = self.get_parameter_value("contrast")
        gain = self.get_parameter_value("gain")

        # 创建基础图案 (渐变 + 若干特征)
        x = np.linspace(0, 4 * np.pi, width)
        y = np.linspace(0, 4 * np.pi, height)
        xx, yy = np.meshgrid(x, y)

        # 合成图案: 正弦波的组合
        pattern = (
            np.sin(xx) * np.cos(yy) * 50 + np.sin(xx * 0.5) * 30 + np.cos(yy * 0.3) * 20
        )

        # 添加若干 "物体" (高斯亮斑)
        for _ in range(5):
            cx, cy = self._rng.integers(0, width), self._rng.integers(0, height)
            sigma = self._rng.uniform(10, 50)
            blob = np.exp(
                -(
                    (xx - cx * 4 * np.pi / width) ** 2
                    + (yy - cy * 4 * np.pi / height) ** 2
                )
                / (2 * sigma**2)
            )
            pattern += blob * 100

        # 应用亮度、对比度与增益
        image = (pattern + brightness) * contrast * gain

        # 添加噪声
        noise = self._rng.normal(0, self._noise_level, image.shape)
        image = image + noise

        # 截断到有效范围
        image = np.clip(image, 0, 255).astype(np.uint8)

        return image

    def capture(self, n_samples: int = 1) -> np.ndarray:
        """从 Mock 相机采集图像。

        Args:
            n_samples: 用于平均的采样数。

        Returns:
            uint8 形式的已采集图像。
        """
        if not self.is_connected():
            raise RuntimeError("Camera not connected")

        self._set_state(DeviceState.BUSY)
        try:
            time.sleep(self._simulate_delay)

            if n_samples == 1:
                img = self._generate_image()
            else:
                # 对多帧取平均
                frames = [
                    self._generate_image().astype(np.float32) for _ in range(n_samples)
                ]
                img = np.mean(frames, axis=0).astype(np.uint8)

            self._last_image = img
            self._frame_counter += 1

            self._emit_data("image", img)
            return img
        finally:
            self._set_state(DeviceState.READY)

    def get_resolution(self) -> tuple[int, int]:
        """获取当前分辨率。"""
        return self._resolution

    def get_twin_state(self) -> dict[str, Any]:
        """获取用于数字孪生同步的状态。"""
        state = super().get_twin_state()
        state["hardware"] = {
            "resolution": self._resolution,
            "frame_counter": self._frame_counter,
        }
        return state


class MockSLMError(DeviceError):
    """Mock SLM 错误时抛出的异常。"""

    pass


class MockSLM(Device):
    """Mock 空间光调制器 (SLM) 设备。

    模拟一台分辨率与相位调制均可配置的 SLM。

    Attributes:
        device_type: DeviceType.SLM
        manufacturer: "Mock"
        model: "Simulated SLM"

    示例:
        >>> slm = MockSLM(device_id="mock_slm_001", resolution=(1920, 1080))
        >>> with slm:
        ...     phase_pattern = np.random.rand(1920, 1080) * 2 * np.pi
        ...     slm.write_phase(phase_pattern)
    """

    device_type = DeviceType.SLM
    manufacturer = "Mock"
    model = "Simulated SLM"

    def __init__(
        self,
        device_id: str = "",
        resolution: tuple[int, int] = (1920, 1080),
        bit_depth: int = 8,
    ):
        """初始化 Mock SLM。

        Args:
            device_id: 唯一设备标识。
            resolution: SLM 分辨率 (宽, 高)。
            bit_depth: 相位表示的位深 (8 或 16)。
        """
        super().__init__(device_id)

        self._resolution = resolution
        self._bit_depth = bit_depth
        self._current_pattern: np.ndarray | None = None
        self._wavelength_nm = 633.0
        self._frame_count = 0

        self._register_parameters()
        self._register_capabilities()

    def _register_parameters(self) -> None:
        """注册 SLM 专属参数。"""
        self.register_parameter(
            "wavelength",
            default_value=633.0,
            min_value=300.0,
            max_value=1100.0,
            unit="nm",
            description="Operating wavelength",
        )
        self.register_parameter(
            "frame_rate",
            default_value=60.0,
            min_value=1.0,
            max_value=120.0,
            unit="Hz",
            description="Frame refresh rate",
        )
        self.register_parameter(
            "phase_range",
            default_value=2 * np.pi,
            min_value=np.pi,
            max_value=4 * np.pi,
            unit="rad",
            description="Maximum phase modulation range",
        )
        self.register_parameter(
            "gamma_correction",
            default_value=1.0,
            min_value=0.5,
            max_value=3.0,
            unit="",
            description="Gamma correction factor",
        )

    def _register_capabilities(self) -> None:
        """注册 SLM 能力。"""
        self.register_capability(
            "write_phase",
            description="Write phase pattern to SLM",
            parameters=["phase_pattern"],
        )
        self.register_capability(
            "write_grayscale",
            description="Write grayscale image to SLM",
            parameters=["image"],
        )
        self.register_capability(
            "get_resolution",
            description="Get SLM resolution",
            return_type=tuple,
        )

    def open(self) -> None:
        """打开 Mock SLM 连接。"""
        self._set_state(DeviceState.CONNECTING)
        time.sleep(0.1)
        self._current_pattern = np.zeros(self._resolution, dtype=np.float32)
        self._set_state(DeviceState.READY)
        logger.info(f"Mock SLM {self.device_id} opened")

    def close(self) -> None:
        """关闭 Mock SLM 连接。"""
        self._current_pattern = None
        self._set_state(DeviceState.DISCONNECTED)
        logger.info(f"Mock SLM {self.device_id} closed")

    def is_connected(self) -> bool:
        """检查 SLM 是否已连接。"""
        return self._state == DeviceState.READY

    def get_hardware_info(self) -> dict[str, Any]:
        """获取 Mock 硬件信息。"""
        return {
            "serial_number": f"MOCK_SLM_{self.device_id[:8]}",
            "firmware_version": "2.0.0-mock",
            "resolution": self._resolution,
            "bit_depth": self._bit_depth,
            "pixel_pitch_um": 8.0,
            "fill_factor": 0.95,
        }

    def write_phase(self, phase_pattern: np.ndarray) -> None:
        """向 Mock SLM 写入相位图案。

        Args:
            phase_pattern: 以弧度表示的相位值二维数组。
        """
        if not self.is_connected():
            raise RuntimeError("SLM not connected")

        if phase_pattern.shape != self._resolution:
            raise ValueError(
                f"Pattern shape {phase_pattern.shape} doesn't match SLM resolution {self._resolution}"
            )

        self._set_state(DeviceState.BUSY)
        try:
            time.sleep(0.005)  # 模拟写入延迟

            # 归一化并应用 gamma 校正
            gamma = self.get_parameter_value("gamma_correction")
            phase_range = self.get_parameter_value("phase_range")

            normalized = np.clip(phase_pattern / phase_range, 0, 1)
            if gamma != 1.0:
                normalized = np.power(normalized, 1.0 / gamma)

            # 转换为目标位深
            max_value = (1 << self._bit_depth) - 1
            self._current_pattern = (normalized * max_value).astype(
                np.uint16 if self._bit_depth > 8 else np.uint8
            )

            self._frame_count += 1
            logger.debug(f"SLM pattern written (frame {self._frame_count})")
        finally:
            self._set_state(DeviceState.READY)

    def write_grayscale(self, image: np.ndarray) -> None:
        """向 Mock SLM 写入灰度图像。

        Args:
            image: 灰度值二维数组 (0-255)。
        """
        if not self.is_connected():
            raise RuntimeError("SLM not connected")

        if image.shape != self._resolution:
            raise ValueError(
                f"Image shape {image.shape} doesn't match SLM resolution {self._resolution}"
            )

        self._set_state(DeviceState.BUSY)
        try:
            time.sleep(0.005)
            self._current_pattern = image.astype(np.uint8)
            self._frame_count += 1
        finally:
            self._set_state(DeviceState.READY)

    def get_current_pattern(self) -> np.ndarray | None:
        """获取当前显示的图案。"""
        return (
            self._current_pattern.copy() if self._current_pattern is not None else None
        )

    def get_resolution(self) -> tuple[int, int]:
        """获取 SLM 分辨率。"""
        return self._resolution

    def get_twin_state(self) -> dict[str, Any]:
        """获取用于数字孪生同步的状态。"""
        state = super().get_twin_state()
        state["hardware"] = {
            "resolution": self._resolution,
            "frame_count": self._frame_count,
            "bit_depth": self._bit_depth,
        }
        return state


class MockDMError(DeviceError):
    """Mock DM 错误时抛出的异常。"""

    pass


class MockDM(Device):
    """Mock 变形镜 (DM) 设备。

    模拟一面致动器数量与电压-形变模型均可配置的变形镜。

    Attributes:
        device_type: DeviceType.DM
        manufacturer: "Mock"
        model: "Simulated DM"

    示例:
        >>> dm = MockDM(device_id="mock_dm_001", n_actuators=64)
        >>> with dm:
        ...     voltages = np.zeros(64)
        ...     dm.apply_voltages(voltages)
        ...     surface = dm.get_surface()
    """

    device_type = DeviceType.DM
    manufacturer = "Mock"
    model = "Simulated DM"

    def __init__(
        self,
        device_id: str = "",
        n_actuators: int = 64,
        voltage_range: tuple[float, float] = (0.0, 300.0),
    ):
        """初始化 Mock DM。

        Args:
            device_id: 唯一设备标识。
            n_actuators: 致动器数量。
            voltage_range: 最小/最大电压范围 (V)。
        """
        super().__init__(device_id)

        self._n_actuators = n_actuators
        self._voltage_range = voltage_range
        self._current_voltages = np.zeros(n_actuators)
        self._surface_shape = (int(np.sqrt(n_actuators)) * 10,) * 2

        # 影响矩阵 (致动器电压到表面形变的映射)
        self._influence_matrix = self._create_influence_matrix()

        self._register_parameters()
        self._register_capabilities()

    def _register_parameters(self) -> None:
        """注册 DM 专属参数。"""
        self.register_parameter(
            "voltage_limit",
            default_value=self._voltage_range[1],
            min_value=self._voltage_range[0],
            max_value=500.0,
            unit="V",
            description="Maximum actuator voltage",
        )
        self.register_parameter(
            "bias_voltage",
            default_value=150.0,
            min_value=0.0,
            max_value=300.0,
            unit="V",
            description="Actuator bias voltage",
        )
        self.register_parameter(
            "settling_time_ms",
            default_value=1.0,
            min_value=0.1,
            max_value=100.0,
            unit="ms",
            description="DM settling time after voltage change",
        )
        self.register_parameter(
            "hysteresis_factor",
            default_value=0.1,
            min_value=0.0,
            max_value=1.0,
            unit="",
            description="Hysteresis effect magnitude",
        )

    def _register_capabilities(self) -> None:
        """注册 DM 能力。"""
        self.register_capability(
            "apply_voltages",
            description="Apply voltages to actuators",
            parameters=["voltages"],
        )
        self.register_capability(
            "get_surface",
            description="Get current mirror surface shape",
            return_type=np.ndarray,
        )
        self.register_capability(
            "reset",
            description="Reset all actuators to zero",
        )

    def _create_influence_matrix(self) -> np.ndarray:
        """创建致动器到表面映射用的影响矩阵。"""
        # 简化模型: 每个致动器产生一个高斯影响
        n_grid = int(np.sqrt(self._n_actuators))
        surface_size = n_grid * 10

        act_pos = np.array(
            [
                [(i + 0.5) / n_grid * surface_size, (j + 0.5) / n_grid * surface_size]
                for i in range(n_grid)
                for j in range(n_grid)
            ]
        )

        x = np.linspace(0, surface_size, surface_size)
        y = np.linspace(0, surface_size, surface_size)
        xx, yy = np.meshgrid(x, y)

        influence = np.zeros((surface_size, surface_size, self._n_actuators))
        sigma = surface_size / n_grid * 0.8

        for i, (ax, ay) in enumerate(act_pos):
            influence[:, :, i] = np.exp(
                -((xx - ax) ** 2 + (yy - ay) ** 2) / (2 * sigma**2)
            )

        return influence.reshape(-1, self._n_actuators)

    def open(self) -> None:
        """打开 Mock DM 连接。"""
        self._set_state(DeviceState.CONNECTING)
        time.sleep(0.1)
        self._current_voltages = np.zeros(self._n_actuators)
        self._set_state(DeviceState.READY)
        logger.info(f"Mock DM {self.device_id} opened ({self._n_actuators} actuators)")

    def close(self) -> None:
        """关闭 Mock DM 连接。"""
        self._current_voltages = np.zeros(self._n_actuators)
        self._set_state(DeviceState.DISCONNECTED)
        logger.info(f"Mock DM {self.device_id} closed")

    def is_connected(self) -> bool:
        """检查 DM 是否已连接。"""
        return self._state == DeviceState.READY

    def get_hardware_info(self) -> dict[str, Any]:
        """获取 Mock 硬件信息。"""
        return {
            "serial_number": f"MOCK_DM_{self.device_id[:8]}",
            "firmware_version": "1.5.0-mock",
            "n_actuators": self._n_actuators,
            "actuator_pitch_um": 300.0,
            "max_voltage": self._voltage_range[1],
            "coating": "Gold",
        }

    def apply_voltages(self, voltages: np.ndarray) -> None:
        """向 DM 致动器施加电压。

        Args:
            voltages: 各致动器对应的电压数组。
        """
        if not self.is_connected():
            raise RuntimeError("DM not connected")

        if len(voltages) != self._n_actuators:
            raise ValueError(
                f"Voltage array length {len(voltages)} doesn't match actuator count {self._n_actuators}"
            )

        self._set_state(DeviceState.BUSY)
        try:
            # 截断到电压范围
            v_limit = self.get_parameter_value("voltage_limit")
            voltages = np.clip(voltages, self._voltage_range[0], v_limit)

            # 模拟稳定时间
            settling_ms = self.get_parameter_value("settling_time_ms")
            time.sleep(settling_ms / 1000.0)

            # 应用迟滞效应
            hysteresis = self.get_parameter_value("hysteresis_factor")
            if hysteresis > 0:
                direction = np.sign(voltages - self._current_voltages)
                voltages = voltages + direction * hysteresis * np.abs(
                    voltages - self._current_voltages
                )

            self._current_voltages = voltages
            logger.debug(
                f"DM voltages applied: min={voltages.min():.2f}, max={voltages.max():.2f}"
            )
        finally:
            self._set_state(DeviceState.READY)

    def get_surface(self) -> np.ndarray:
        """获取当前的镜面形貌。

        Returns:
            以纳米表示镜面形貌的二维数组。
        """
        if not self.is_connected():
            raise RuntimeError("DM not connected")

        # 把电压换算为表面形变
        bias = self.get_parameter_value("bias_voltage")
        effective_voltages = self._current_voltages + bias

        # 简单模型: surface = influence_matrix @ voltages
        surface_flat = self._influence_matrix @ effective_voltages
        surface = surface_flat.reshape(self._surface_shape)

        # 换算为纳米 (简化缩放)
        surface_nm = surface * 100  # 每单位电压效应对应 100 nm

        return surface_nm

    def reset(self) -> None:
        """把所有致动器复位到零电压。"""
        self.apply_voltages(np.zeros(self._n_actuators))

    def get_current_voltages(self) -> np.ndarray:
        """获取当前的致动器电压。"""
        return self._current_voltages.copy()

    def get_twin_state(self) -> dict[str, Any]:
        """获取用于数字孪生同步的状态。"""
        state = super().get_twin_state()
        state["hardware"] = {
            "n_actuators": self._n_actuators,
            "voltage_range": self._voltage_range,
            "current_voltages": self._current_voltages.tolist(),
        }
        return state


class MockWFSError(DeviceError):
    """Mock WFS 错误时抛出的异常。"""

    pass


class MockWFS(Device):
    """Mock 波前传感器 (WFS) 设备。

    模拟一台用于波前测量的 Shack-Hartmann 波前传感器。

    Attributes:
        device_type: DeviceType.WFS
        manufacturer: "Mock"
        model: "Simulated WFS"

    示例:
        >>> wfs = MockWFS(device_id="mock_wfs_001", n_lenslets=32)
        >>> with wfs:
        ...     wf = wfs.measure_wavefront()
        ...     zernike = wfs.fit_zernike(wf, n_modes=15)
    """

    device_type = DeviceType.WFS
    manufacturer = "Mock"
    model = "Simulated WFS"

    def __init__(
        self,
        device_id: str = "",
        n_lenslets: int = 32,
        pupil_size_mm: float = 5.0,
        random_seed: int | None = None,
    ):
        """初始化 Mock WFS。

        Args:
            device_id: 唯一设备标识。
            n_lenslets: 每边的微透镜数量。
            pupil_size_mm: 瞳孔直径, 单位毫米。
            random_seed: 用于可复现输出的随机种子。为 None 时使用随机初始化。
        """
        super().__init__(device_id)

        self._n_lenslets = n_lenslets
        self._pupil_size_mm = pupil_size_mm
        self._spot_image: np.ndarray | None = None
        self._rng = np.random.default_rng(random_seed)

        self._register_parameters()
        self._register_capabilities()

    def _register_parameters(self) -> None:
        """注册 WFS 专属参数。"""
        self.register_parameter(
            "integration_time_ms",
            default_value=10.0,
            min_value=1.0,
            max_value=1000.0,
            unit="ms",
            description="Sensor integration time",
        )
        self.register_parameter(
            "threshold",
            default_value=50.0,
            min_value=0.0,
            max_value=255.0,
            unit="",
            description="Spot detection threshold",
        )
        self.register_parameter(
            "pupil_offset_x",
            default_value=0.0,
            min_value=-5.0,
            max_value=5.0,
            unit="mm",
            description="Pupil center X offset",
        )
        self.register_parameter(
            "pupil_offset_y",
            default_value=0.0,
            min_value=-5.0,
            max_value=5.0,
            unit="mm",
            description="Pupil center Y offset",
        )

    def _register_capabilities(self) -> None:
        """注册 WFS 能力。"""
        self.register_capability(
            "measure_wavefront",
            description="Measure wavefront",
            return_type=np.ndarray,
        )
        self.register_capability(
            "fit_zernike",
            description="Fit Zernike polynomials to wavefront",
            parameters=["n_modes"],
            return_type=np.ndarray,
        )
        self.register_capability(
            "get_spot_image",
            description="Get spot pattern image",
            return_type=np.ndarray,
        )

    def open(self) -> None:
        """打开 Mock WFS 连接。"""
        self._set_state(DeviceState.CONNECTING)
        time.sleep(0.1)
        self._set_state(DeviceState.READY)
        logger.info(
            f"Mock WFS {self.device_id} opened ({self._n_lenslets}x{self._n_lenslets} lenslets)"
        )

    def close(self) -> None:
        """关闭 Mock WFS 连接。"""
        self._spot_image = None
        self._set_state(DeviceState.DISCONNECTED)
        logger.info(f"Mock WFS {self.device_id} closed")

    def is_connected(self) -> bool:
        """检查 WFS 是否已连接。"""
        return self._state == DeviceState.READY

    def get_hardware_info(self) -> dict[str, Any]:
        """获取 Mock 硬件信息。"""
        return {
            "serial_number": f"MOCK_WFS_{self.device_id[:8]}",
            "firmware_version": "3.0.0-mock",
            "n_lenslets": self._n_lenslets,
            "pupil_size_mm": self._pupil_size_mm,
            "lenslet_pitch_um": 150.0,
            "focal_length_mm": 10.0,
        }

    def measure_wavefront(self) -> np.ndarray:
        """测量波前。

        Returns:
            以弧度表示的波前相位二维数组。
        """
        if not self.is_connected():
            raise RuntimeError("WFS not connected")

        self._set_state(DeviceState.BUSY)
        try:
            time.sleep(0.01)  # 模拟测量耗时

            # 生成带若干像差的合成波前
            size = self._n_lenslets * 10
            x = np.linspace(-1, 1, size)
            y = np.linspace(-1, 1, size)
            xx, yy = np.meshgrid(x, y)

            # 模拟若干 Zernike 像差
            z_defocus = 0.5 * (2 * (xx**2 + yy**2) - 1)
            z_astig = 0.3 * (xx**2 - yy**2)
            z_coma_x = 0.2 * (3 * (xx**2 + yy**2) - 2) * xx

            wavefront = z_defocus + z_astig + z_coma_x

            # 添加噪声
            wavefront += self._rng.normal(0, 0.05, wavefront.shape)

            return wavefront
        finally:
            self._set_state(DeviceState.READY)

    def fit_zernike(self, wavefront: np.ndarray, n_modes: int = 15) -> np.ndarray:
        """对测得的波前拟合 Zernike 多项式。

        Args:
            wavefront: 测得的波前。
            n_modes: 要拟合的 Zernike 模式数量。

        Returns:
            Zernike 系数数组。
        """
        # 简化处理: 为演示返回随机系数
        return self._rng.standard_normal(n_modes) * 0.1

    def get_spot_image(self) -> np.ndarray:
        """获取光斑图案图像。

        Returns:
            微透镜光斑的二维图像。
        """
        if not self.is_connected():
            raise RuntimeError("WFS not connected")

        # 生成合成光斑图案
        img_size = self._n_lenslets * 20
        image = np.zeros((img_size, img_size), dtype=np.uint8)

        # 添加光斑
        for i in range(self._n_lenslets):
            for j in range(self._n_lenslets):
                cx = int((i + 0.5) / self._n_lenslets * img_size)
                cy = int((j + 0.5) / self._n_lenslets * img_size)

                # 添加高斯光斑
                y, x = np.ogrid[-10:11, -10:11]
                spot = np.exp(-(x**2 + y**2) / 10) * 200

                y0, x0 = max(0, cy - 10), max(0, cx - 10)
                y1, x1 = min(img_size, cy + 11), min(img_size, cx + 11)
                sy0, sx0 = max(0, 10 - cy), max(0, 10 - cx)
                sy1, sx1 = sy0 + (y1 - y0), sx0 + (x1 - x0)

                image[y0:y1, x0:x1] += spot[sy0:sy1, sx0:sx1].astype(np.uint8)

        self._spot_image = np.clip(image, 0, 255).astype(np.uint8)
        return self._spot_image

    def get_twin_state(self) -> dict[str, Any]:
        """获取用于数字孪生同步的状态。"""
        state = super().get_twin_state()
        state["hardware"] = {
            "n_lenslets": self._n_lenslets,
            "pupil_size_mm": self._pupil_size_mm,
        }
        return state


class MockStageError(DeviceError):
    """Mock 运动台错误时抛出的异常。"""

    pass


class MockStage(Device):
    """Mock 运动台设备。

    模拟一台带位置反馈的线性平移台。

    Attributes:
        device_type: DeviceType.STAGE
        manufacturer: "Mock"
        model: "Simulated Stage"

    示例:
        >>> stage = MockStage(device_id="mock_stage_001", axis="X")
        >>> with stage:
        ...     stage.move_to(10.0)
        ...     pos = stage.get_position()
    """

    device_type = DeviceType.STAGE
    manufacturer = "Mock"
    model = "Simulated Stage"

    def __init__(
        self,
        device_id: str = "",
        axis: str = "X",
        travel_range: tuple[float, float] = (0.0, 100.0),
    ):
        """初始化 Mock 运动台。

        Args:
            device_id: 唯一设备标识。
            axis: 轴名 (X、Y、Z 等)。
            travel_range: 最小/最大行程范围, 单位 mm。
        """
        super().__init__(device_id)

        self._axis = axis
        self._travel_range = travel_range
        self._current_position = travel_range[0]
        self._target_position = self._current_position
        self._is_moving = False

        self._register_parameters()
        self._register_capabilities()

    def _register_parameters(self) -> None:
        """注册运动台专属参数。"""
        self.register_parameter(
            "velocity",
            default_value=10.0,
            min_value=0.1,
            max_value=100.0,
            unit="mm/s",
            description="Stage movement velocity",
        )
        self.register_parameter(
            "acceleration",
            default_value=100.0,
            min_value=1.0,
            max_value=1000.0,
            unit="mm/s^2",
            description="Stage acceleration",
        )
        self.register_parameter(
            "backlash_compensation",
            default_value=0.01,
            min_value=0.0,
            max_value=1.0,
            unit="mm",
            description="Backlash compensation amount",
        )
        self.register_parameter(
            "home_position",
            default_value=self._travel_range[0],
            min_value=self._travel_range[0],
            max_value=self._travel_range[1],
            unit="mm",
            description="Home position",
        )

    def _register_capabilities(self) -> None:
        """注册运动台能力。"""
        self.register_capability(
            "move_to",
            description="Move to absolute position",
            parameters=["position"],
        )
        self.register_capability(
            "move_relative",
            description="Move relative to current position",
            parameters=["distance"],
        )
        self.register_capability(
            "home",
            description="Move to home position",
        )
        self.register_capability(
            "get_position",
            description="Get current position",
            return_type=float,
        )

    def open(self) -> None:
        """打开 Mock 运动台连接。"""
        self._set_state(DeviceState.CONNECTING)
        time.sleep(0.1)
        self._current_position = self._travel_range[0]
        self._set_state(DeviceState.READY)
        logger.info(f"Mock stage {self.device_id} opened (axis {self._axis})")

    def close(self) -> None:
        """关闭 Mock 运动台连接。"""
        self._is_moving = False
        self._set_state(DeviceState.DISCONNECTED)
        logger.info(f"Mock stage {self.device_id} closed")

    def is_connected(self) -> bool:
        """检查运动台是否已连接。"""
        return self._state == DeviceState.READY

    def get_hardware_info(self) -> dict[str, Any]:
        """获取 Mock 硬件信息。"""
        return {
            "serial_number": f"MOCK_STAGE_{self.device_id[:8]}",
            "firmware_version": "1.2.0-mock",
            "axis": self._axis,
            "travel_range_mm": self._travel_range,
            "resolution_um": 0.1,
            "encoder_type": "Incremental",
        }

    def move_to(self, position: float) -> None:
        """移动到绝对位置。

        Args:
            position: 目标位置, 单位 mm。
        """
        if not self.is_connected():
            raise RuntimeError("Stage not connected")

        if not (self._travel_range[0] <= position <= self._travel_range[1]):
            raise ValueError(
                f"Position {position} mm out of range {self._travel_range}"
            )

        self._set_state(DeviceState.BUSY)
        self._is_moving = True
        try:
            velocity = self.get_parameter_value("velocity")
            distance = abs(position - self._current_position)
            move_time = distance / velocity

            # 模拟运动
            time.sleep(move_time)

            self._current_position = position
            logger.debug(f"Stage moved to {position:.3f} mm")
        finally:
            self._is_moving = False
            self._set_state(DeviceState.READY)

    def move_relative(self, distance: float) -> None:
        """相对当前位置移动。

        Args:
            distance: 移动距离, 单位 mm (可为正或负)。
        """
        self.move_to(self._current_position + distance)

    def home(self) -> None:
        """移动到原点位置。"""
        home_pos = self.get_parameter_value("home_position")
        self.move_to(home_pos)

    def get_position(self) -> float:
        """获取当前位置, 单位 mm。"""
        return self._current_position

    def is_moving(self) -> bool:
        """检查运动台是否正在移动。"""
        return self._is_moving

    def get_twin_state(self) -> dict[str, Any]:
        """获取用于数字孪生同步的状态。"""
        state = super().get_twin_state()
        state["hardware"] = {
            "axis": self._axis,
            "travel_range": self._travel_range,
            "current_position": self._current_position,
            "is_moving": self._is_moving,
        }
        return state


class MockLaserError(DeviceError):
    """Mock 激光器错误时抛出的异常。"""

    pass


class MockLaser(Device):
    """Mock 激光器设备。

    模拟一台功率与波长均可调谐的激光源。

    Attributes:
        device_type: DeviceType.LASER
        manufacturer: "Mock"
        model: "Simulated Laser"

    示例:
        >>> laser = MockLaser(device_id="mock_laser_001")
        >>> with laser:
        ...     laser.set_power(10.0)
        ...     laser.set_wavelength(633.0)
        ...     laser.enable_output(True)
    """

    device_type = DeviceType.LASER
    manufacturer = "Mock"
    model = "Simulated Laser"

    def __init__(
        self,
        device_id: str = "",
        wavelength_range: tuple[float, float] = (400.0, 1100.0),
        power_range: tuple[float, float] = (0.0, 100.0),
    ):
        """初始化 Mock 激光器。

        Args:
            device_id: 唯一设备标识。
            wavelength_range: 最小/最大波长范围, 单位 nm。
            power_range: 最小/最大功率范围, 单位 mW。
        """
        super().__init__(device_id)

        self._wavelength_range = wavelength_range
        self._power_range = power_range
        self._current_wavelength = wavelength_range[0]
        self._current_power = 0.0
        self._output_enabled = False
        self._temperature = 25.0

        self._register_parameters()
        self._register_capabilities()

    def _register_parameters(self) -> None:
        """注册激光器专属参数。"""
        self.register_parameter(
            "wavelength",
            default_value=self._wavelength_range[0],
            min_value=self._wavelength_range[0],
            max_value=self._wavelength_range[1],
            unit="nm",
            description="Laser wavelength",
        )
        self.register_parameter(
            "power",
            default_value=0.0,
            min_value=self._power_range[0],
            max_value=self._power_range[1],
            unit="mW",
            description="Laser output power",
        )
        self.register_parameter(
            "temperature",
            default_value=25.0,
            min_value=10.0,
            max_value=50.0,
            unit="C",
            description="Laser diode temperature",
        )
        self.register_parameter(
            "stability_mode",
            default_value=True,
            unit="",
            description="Enable power stability control",
        )

    def _register_capabilities(self) -> None:
        """注册激光器能力。"""
        self.register_capability(
            "enable_output",
            description="Enable/disable laser output",
            parameters=["enabled"],
        )
        self.register_capability(
            "get_power",
            description="Get current output power",
            return_type=float,
        )
        self.register_capability(
            "get_wavelength",
            description="Get current wavelength",
            return_type=float,
        )

    def open(self) -> None:
        """打开 Mock 激光器连接。"""
        self._set_state(DeviceState.CONNECTING)
        time.sleep(0.2)
        self._output_enabled = False
        self._set_state(DeviceState.READY)
        logger.info(f"Mock laser {self.device_id} opened")

    def close(self) -> None:
        """关闭 Mock 激光器连接。"""
        self._output_enabled = False
        self._set_state(DeviceState.DISCONNECTED)
        logger.info(f"Mock laser {self.device_id} closed")

    def is_connected(self) -> bool:
        """检查激光器是否已连接。"""
        return self._state == DeviceState.READY

    def get_hardware_info(self) -> dict[str, Any]:
        """获取 Mock 硬件信息。"""
        return {
            "serial_number": f"MOCK_LASER_{self.device_id[:8]}",
            "firmware_version": "2.1.0-mock",
            "wavelength_range_nm": self._wavelength_range,
            "power_range_mW": self._power_range,
            "beam_diameter_mm": 1.0,
            "beam_quality_m2": 1.05,
        }

    def enable_output(self, enabled: bool) -> None:
        """启用或关闭激光输出。

        Args:
            enabled: True 表示启用输出, False 表示关闭。
        """
        if not self.is_connected():
            raise RuntimeError("Laser not connected")

        if enabled and self._current_power <= 0:
            logger.warning("Attempting to enable laser with zero power")

        self._output_enabled = enabled
        logger.info(f"Laser output {'enabled' if enabled else 'disabled'}")

    def set_power(self, power_mw: float) -> None:
        """设置激光功率。

        Args:
            power_mw: 功率, 单位毫瓦。
        """
        if not self.is_connected():
            raise RuntimeError("Laser not connected")

        if not (self._power_range[0] <= power_mw <= self._power_range[1]):
            raise ValueError(f"Power {power_mw} mW out of range {self._power_range}")

        self._set_state(DeviceState.BUSY)
        try:
            time.sleep(0.05)  # 模拟稳定过程
            self._current_power = power_mw
            self.set_parameter_value("power", power_mw)
            logger.debug(f"Laser power set to {power_mw:.2f} mW")
        finally:
            self._set_state(DeviceState.READY)

    def set_wavelength(self, wavelength_nm: float) -> None:
        """设置激光波长。

        Args:
            wavelength_nm: 波长, 单位纳米。
        """
        if not self.is_connected():
            raise RuntimeError("Laser not connected")

        if not (
            self._wavelength_range[0] <= wavelength_nm <= self._wavelength_range[1]
        ):
            raise ValueError(
                f"Wavelength {wavelength_nm} nm out of range {self._wavelength_range}"
            )

        self._set_state(DeviceState.BUSY)
        try:
            time.sleep(0.5)  # 模拟波长调谐
            self._current_wavelength = wavelength_nm
            self.set_parameter_value("wavelength", wavelength_nm)
            logger.debug(f"Laser wavelength set to {wavelength_nm:.2f} nm")
        finally:
            self._set_state(DeviceState.READY)

    def get_power(self) -> float:
        """获取当前输出功率, 单位 mW。"""
        if self._output_enabled:
            return self._current_power
        return 0.0

    def get_wavelength(self) -> float:
        """获取当前波长, 单位 nm。"""
        return self._current_wavelength

    def is_output_enabled(self) -> bool:
        """检查激光输出是否已启用。"""
        return self._output_enabled

    def _on_parameter_changed(self, name: str, old_value: Any, new_value: Any) -> None:
        """处理参数变化。"""
        if name == "wavelength":
            self._current_wavelength = new_value
        elif name == "power":
            self._current_power = new_value

    def get_twin_state(self) -> dict[str, Any]:
        """获取用于数字孪生同步的状态。"""
        state = super().get_twin_state()
        state["hardware"] = {
            "wavelength_range": self._wavelength_range,
            "power_range": self._power_range,
            "current_wavelength": self._current_wavelength,
            "current_power": self._current_power,
            "output_enabled": self._output_enabled,
        }
        return state


class MockFilterError(DeviceError):
    """Mock 滤光轮错误时抛出的异常。"""

    pass


class MockFilter(Device):
    """Mock 光学滤光轮设备。

    模拟一个带多个滤光片位置的滤光轮。

    Attributes:
        device_type: DeviceType.FILTER
        manufacturer: "Mock"
        model: "Simulated Filter Wheel"

    示例:
        >>> filters = ["Open", "ND1", "ND2", "Red", "Green", "Blue"]
        >>> fw = MockFilter(device_id="mock_filter_001", filters=filters)
        >>> with fw:
        ...     fw.move_to_position(3)  # Move to "Red" filter
    """

    device_type = DeviceType.FILTER
    manufacturer = "Mock"
    model = "Simulated Filter Wheel"

    def __init__(
        self,
        device_id: str = "",
        filters: list[str] | None = None,
    ):
        """初始化 Mock 滤光轮。

        Args:
            device_id: 唯一设备标识。
            filters: 滤光片名称列表。默认为 6 位置滤光轮。
        """
        super().__init__(device_id)

        self._filters = filters or ["Open", "ND1", "ND2", "ND3", "Red", "Block"]
        self._n_positions = len(self._filters)
        self._current_position = 0
        self._is_moving = False

        self._register_parameters()
        self._register_capabilities()

    def _register_parameters(self) -> None:
        """注册滤光轮专属参数。"""
        self.register_parameter(
            "speed",
            default_value=1.0,
            min_value=0.5,
            max_value=5.0,
            unit="rev/s",
            description="Filter wheel rotation speed",
        )
        self.register_parameter(
            "settle_time_ms",
            default_value=100.0,
            min_value=10.0,
            max_value=1000.0,
            unit="ms",
            description="Settling time after move",
        )
        self.register_parameter(
            "home_on_startup",
            default_value=True,
            unit="",
            description="Auto-home on initialization",
        )

    def _register_capabilities(self) -> None:
        """注册滤光轮能力。"""
        self.register_capability(
            "move_to_position",
            description="Move to filter position",
            parameters=["position"],
        )
        self.register_capability(
            "move_to_filter",
            description="Move to filter by name",
            parameters=["filter_name"],
        )
        self.register_capability(
            "get_current_filter",
            description="Get current filter name",
            return_type=str,
        )

    def open(self) -> None:
        """打开 Mock 滤光轮连接。"""
        self._set_state(DeviceState.CONNECTING)
        time.sleep(0.1)

        if self.get_parameter_value("home_on_startup"):
            self._current_position = 0

        self._set_state(DeviceState.READY)
        logger.info(
            f"Mock filter wheel {self.device_id} opened ({self._n_positions} positions)"
        )

    def close(self) -> None:
        """关闭 Mock 滤光轮连接。"""
        self._is_moving = False
        self._set_state(DeviceState.DISCONNECTED)
        logger.info(f"Mock filter wheel {self.device_id} closed")

    def is_connected(self) -> bool:
        """检查滤光轮是否已连接。"""
        return self._state == DeviceState.READY

    def get_hardware_info(self) -> dict[str, Any]:
        """获取 Mock 硬件信息。"""
        return {
            "serial_number": f"MOCK_FILTER_{self.device_id[:8]}",
            "firmware_version": "1.0.0-mock",
            "n_positions": self._n_positions,
            "filters": self._filters,
            "position": self._current_position,
        }

    def move_to_position(self, position: int) -> None:
        """移动到指定的滤光片位置。

        Args:
            position: 滤光片位置索引 (从 0 开始)。
        """
        if not self.is_connected():
            raise RuntimeError("Filter wheel not connected")

        if not (0 <= position < self._n_positions):
            raise ValueError(
                f"Position {position} out of range [0, {self._n_positions})"
            )

        self._set_state(DeviceState.BUSY)
        self._is_moving = True
        try:
            speed = self.get_parameter_value("speed")
            settle_ms = self.get_parameter_value("settle_time_ms")

            # 计算转动时间 (取最短路径)
            distance = min(
                abs(position - self._current_position),
                self._n_positions - abs(position - self._current_position),
            )

            move_time = distance / (speed * self._n_positions)
            time.sleep(move_time + settle_ms / 1000.0)

            self._current_position = position
            logger.debug(
                f"Filter wheel moved to position {position} ({self._filters[position]})"
            )
        finally:
            self._is_moving = False
            self._set_state(DeviceState.READY)

    def move_to_filter(self, filter_name: str) -> None:
        """按名称移动到对应的滤光片。

        Args:
            filter_name: 要移动到的滤光片名称。
        """
        if filter_name not in self._filters:
            raise ValueError(
                f"Filter '{filter_name}' not found. Available: {self._filters}"
            )

        position = self._filters.index(filter_name)
        self.move_to_position(position)

    def get_current_position(self) -> int:
        """获取当前滤光片位置索引。"""
        return self._current_position

    def get_current_filter(self) -> str:
        """获取当前滤光片名称。"""
        return self._filters[self._current_position]

    def get_filter_list(self) -> list[str]:
        """获取可用滤光片列表。"""
        return self._filters.copy()

    def get_twin_state(self) -> dict[str, Any]:
        """获取用于数字孪生同步的状态。"""
        state = super().get_twin_state()
        state["hardware"] = {
            "n_positions": self._n_positions,
            "filters": self._filters,
            "current_position": self._current_position,
            "current_filter": self.get_current_filter(),
        }
        return state


class MockADCError(DeviceError):
    """Mock ADC 错误时抛出的异常。"""

    pass


class MockADC(Device):
    """用于测试的 Mock NI DAQ ADC 设备。

    模拟模拟电压采集, 可选带噪声, 基线电压可配置。

    Attributes:
        device_type: DeviceType.OTHER
        manufacturer: "Mock"
        model: "Simulated ADC"

    示例:
        >>> adc = MockADC(device_id="mock_adc_001", noise_std=0.01)
        >>> with adc:
        ...     voltages = adc.read(samples=10)
        ...     mean_v = adc.read_mean()
    """

    device_type = DeviceType.OTHER
    manufacturer = "Mock"
    model = "Simulated ADC"

    def __init__(
        self,
        device_id: str = "",
        device_name: str = "Dev1",
        channel: str = "ai0",
        sample_rate: int = 5000,
        samples_per_channel: int = 10,
        base_voltage: float = 0.0,
        noise_std: float = 0.01,
        random_seed: int | None = None,
    ):
        """初始化 Mock ADC。

        Args:
            device_id: 唯一设备标识。
            device_name: 模拟的 NI DAQ 设备名。
            channel: 模拟的模拟输入通道。
            sample_rate: 模拟采样率 (Hz)。
            samples_per_channel: 每次读取的采样数。
            base_voltage: 基线输出电压 (V)。
            noise_std: 加到读数上的高斯噪声标准差。
            random_seed: 用于可复现输出的随机种子。
        """
        super().__init__(device_id)

        self._device_name = device_name
        self._channel = channel
        self._sample_rate = sample_rate
        self._samples_per_channel = samples_per_channel
        self._base_voltage = base_voltage
        self._noise_std = noise_std
        self._rng = np.random.default_rng(random_seed)

        self._register_parameters()

    def _register_parameters(self) -> None:
        """注册 ADC 专属参数。"""
        self.register_parameter(
            "device_name",
            self._device_name,
            description="Simulated NI DAQ device name",
        )
        self.register_parameter(
            "channel",
            self._channel,
            description="Simulated analog input channel",
        )
        self.register_parameter(
            "sample_rate",
            self._sample_rate,
            min_value=1,
            max_value=1_000_000,
            unit="Hz",
            description="Simulated acquisition sample rate",
        )
        self.register_parameter(
            "samples_per_channel",
            self._samples_per_channel,
            min_value=1,
            max_value=1_000_000,
            description="Simulated samples per read",
        )
        self.register_parameter(
            "base_voltage",
            self._base_voltage,
            unit="V",
            description="Baseline voltage output",
        )
        self.register_parameter(
            "noise_std",
            self._noise_std,
            min_value=0.0,
            unit="V",
            description="Noise standard deviation",
        )

    def open(self) -> None:
        """打开 Mock ADC 连接。"""
        self._set_state(DeviceState.CONNECTING)
        time.sleep(0.05)
        self._set_state(DeviceState.READY)
        logger.info(
            f"Mock ADC {self.device_id} opened ({self._device_name}/{self._channel})"
        )

    def close(self) -> None:
        """关闭 Mock ADC 连接。"""
        self._set_state(DeviceState.DISCONNECTED)
        logger.info(f"Mock ADC {self.device_id} closed")

    def is_connected(self) -> bool:
        """检查 ADC 是否已连接。"""
        return self._state == DeviceState.READY

    def get_hardware_info(self) -> dict[str, Any]:
        """获取 Mock 硬件信息。"""
        return {
            "serial_number": f"MOCK_ADC_{self.device_id[:8]}",
            "firmware_version": "1.0.0-mock",
            "device_name": self._device_name,
            "channel": self._channel,
            "sample_rate": self._sample_rate,
            "samples_per_channel": self._samples_per_channel,
        }

    def read(self, samples: int | None = None) -> np.ndarray:
        """读取模拟电压采样。

        Args:
            samples: 返回的采样数。默认为 ``samples_per_channel``。

        Returns:
            以伏特表示的模拟电压读数一维数组。
        """
        if not self.is_connected():
            raise RuntimeError("ADC not connected")

        n = samples or self._samples_per_channel
        base = self.get_parameter_value("base_voltage")
        noise = self.get_parameter_value("noise_std")

        self._set_state(DeviceState.BUSY)
        try:
            time.sleep(0.001)  # 模拟读取延迟
            data = base + self._rng.normal(0, noise, n)
            return data.astype(np.float64)
        finally:
            self._set_state(DeviceState.READY)

    def read_mean(self, samples: int | None = None) -> float:
        """读取模拟电压采样并返回其均值。

        Args:
            samples: 采样数。默认为 ``samples_per_channel``。

        Returns:
            平均电压, 单位伏特。
        """
        return float(np.mean(self.read(samples=samples)))

    def get_twin_state(self) -> dict[str, Any]:
        """获取用于数字孪生同步的状态。"""
        state = super().get_twin_state()
        state["hardware"] = {
            "device_name": self._device_name,
            "channel": self._channel,
            "sample_rate": self._sample_rate,
            "samples_per_channel": self._samples_per_channel,
            "base_voltage": self._base_voltage,
            "noise_std": self._noise_std,
        }
        return state
