"""模拟 CCD 相机。

本模块提供一个继承自 BaseCamera 的模拟 CCD 相机, 内部走数值仿真, 同时仍能
接入现有的相机接口。
"""

from __future__ import annotations

import time

import numpy as np
from loguru import logger

from ao_shaping.drivers.ccd.base import BaseCamera, CameraError
from ao_shaping.drivers.device_base import DeviceState


class SimulatedCCDError(CameraError):
    """模拟 CCD 相关错误的异常。"""

    pass


class SimulatedCCD(BaseCamera):
    """模拟 CCD 相机。

    本类提供一个继承自 BaseCamera 的模拟相机, 用数值仿真实现全部必需的抽象方法。
    实际计算在可用时使用 sim.digitaltwin 的算法。

    Attributes:
        resolution: 相机分辨率 (宽, 高)。
        noise_level: 噪声标准差, 单位 ADU。
        exposure_time_ms: 曝光时间, 单位毫秒。

    Example:
        >>> cam = SimulatedCCD(resolution=(1024, 1024), noise_level=5.0)
        >>> with cam:
        ...     img = cam.get_numpy_image()
        ...     print(f"Captured: {img.shape}")
    """

    def __init__(
        self,
        cam_id: int = 0,
        exposure_time_ms: float = 20.0,
        resolution: tuple[int, int] = (1024, 1024),
        noise_level: float = 5.0,
        random_seed: int | None = None,
    ):
        """初始化模拟 CCD。

        Args:
            cam_id: 相机 ID (为兼容 BaseCamera)。
            exposure_time_ms: 曝光时间, 单位 ms。
            resolution: 图像分辨率 (宽, 高)。
            noise_level: 噪声水平, 单位 ADU。
            random_seed: 用于可复现性的随机种子。
        """
        super().__init__(cam_id, exposure_time_ms)

        self._resolution = resolution
        self._noise_level = noise_level
        self._random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)

        self._frame_counter = 0
        self._last_image: np.ndarray | None = None

        logger.debug(
            f"SimulatedCCD initialized: resolution={resolution}, noise={noise_level}"
        )

    def __exit__(self, exc_type, exc_value, traceback):
        """上下文管理器退出。"""
        self.close()

    def initialize(self) -> None:
        """初始化模拟相机。"""
        self._set_state(DeviceState.READY)
        self._frame_counter = 0
        logger.info("SimulatedCCD initialized")

    def open(self) -> None:
        """打开模拟相机。"""
        self.initialize()

    def close(self) -> None:
        """关闭模拟相机。"""
        self._set_state(DeviceState.DISCONNECTED)
        self.cam = None
        logger.info("SimulatedCCD closed")

    def reset_exposure_time(self, time_ms: float) -> float:
        """设置曝光时间。

        Args:
            time_ms: 曝光时间, 单位毫秒。

        Returns:
            实际设置的曝光时间。
        """
        self.exposure_time_ms = float(time_ms)
        return self.exposure_time_ms

    def reset_window(
        self,
        center: tuple[int, int],
        size: tuple[int, int],
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """设置 ROI 窗口。

        Args:
            center: 窗口中心 (x, y)。
            size: 窗口尺寸 (宽, 高)。

        Returns:
            (实际尺寸, 实际中心) 二元组。
        """
        if size == (0, 0):
            return self._resolution, center

        actual_size = (
            min(size[0], self._resolution[0]),
            min(size[1], self._resolution[1]),
        )
        return actual_size, center

    def get_numpy_image(
        self,
        n_sample: int = 1,
        skip_first: bool = True,
    ) -> np.ndarray:
        """从模拟相机采集图像。

        Args:
            n_sample: 平均的采样帧数。
            skip_first: 是否跳过第一帧。

        Returns:
            uint16 数组形式的采集图像。
        """
        self._set_state(DeviceState.BUSY)

        # 模拟曝光时间
        time.sleep(self.exposure_time_ms / 1000.0)

        if n_sample == 1:
            img = self._generate_image()
        else:
            start_idx = 1 if skip_first else 0
            frames = [self._generate_image() for _ in range(start_idx, n_sample)]
            img = np.mean(frames, axis=0).astype(np.uint16)

        self._last_image = img
        self._frame_counter += 1

        self._set_state(DeviceState.READY)
        return img

    def _generate_image(self) -> np.ndarray:
        """生成合成图像。

        Returns:
            模拟图像数组。
        """
        width, height = self._resolution

        # 构造基础图样
        x = np.linspace(0, 4 * np.pi, width)
        y = np.linspace(0, 4 * np.pi, height)
        xx, yy = np.meshgrid(x, y)

        # 合成图样: 正弦波的组合
        pattern = (
            np.sin(xx) * np.cos(yy) * 1000
            + np.sin(xx * 0.5) * 600
            + np.cos(yy * 0.3) * 400
        )

        # 加上若干"特征" (高斯亮斑)
        for _ in range(3):
            cx = self._rng.integers(0, width)
            cy = self._rng.integers(0, height)
            sigma = self._rng.uniform(20, 80)
            blob = (
                np.exp(
                    -(
                        (xx - cx * 4 * np.pi / width) ** 2
                        + (yy - cy * 4 * np.pi / height) ** 2
                    )
                    / (2 * sigma**2)
                )
                * 2000
            )
            pattern += blob

        # 按曝光时间缩放, 以 20 ms 为参考归一化。
        # 非正值意味着"无固定曝光" (新的 CLI 默认值), 此时回落到参考值而不是把图样
        # 缩放成零 —— 全零的帧会把每个仿真指标静默变成纯噪声。
        reference_ms = 20.0
        exposure_ms = float(self.exposure_time_ms)
        exposure_factor = (exposure_ms / reference_ms) if exposure_ms > 0.0 else 1.0
        pattern *= exposure_factor

        # 加入噪声
        noise = self._rng.normal(0, self._noise_level, pattern.shape)
        pattern += noise

        # 裁剪到有效范围
        img = np.clip(pattern, 0, 65535).astype(np.uint16)

        return img

    def enable_auto_exposure(self, enable: bool = True, mode: int = 1) -> bool:
        """启用或禁用自动曝光。

        Args:
            enable: True 启用, False 禁用。
            mode: 自动曝光模式。

        Returns:
            是否成功。
        """
        logger.info(f"Auto exposure {'enabled' if enable else 'disabled'}")
        return True

    def set_auto_exposure_target(self, target: int) -> int:
        """设置自动曝光目标值。

        Args:
            target: 目标亮度值。

        Returns:
            设置的目标值。
        """
        return target

    def get_auto_exposure_state(self) -> dict:
        """获取自动曝光状态。

        Returns:
            含自动曝光设置的字典。
        """
        return {
            "enabled": False,
            "mode": 1,
            "target": 32768,
        }

    def set_auto_exposure_range(
        self,
        max_time_ms: int = 350,
        min_time_ms: int = 0,
        max_gain: int = 300,
        min_gain: int = 100,
    ) -> bool:
        """设置自动曝光范围。

        Args:
            max_time_ms: 最大曝光时间, 单位 ms。
            min_time_ms: 最小曝光时间, 单位 ms。
            max_gain: 最大增益。
            min_gain: 最小增益。

        Returns:
            是否成功。
        """
        return True

    @staticmethod
    def get_cam_list() -> list:
        """获取可用相机列表。

        Returns:
            含本模拟相机的列表。
        """
        return ["SimulatedCCD_0"]

    def _set_state(self, state: str) -> None:
        """设置相机状态。"""
        self.cam = state

    @property
    def resolution(self) -> tuple[int, int]:
        """获取相机分辨率。"""
        return self._resolution

    @property
    def frame_counter(self) -> int:
        """获取帧计数器。"""
        return self._frame_counter
