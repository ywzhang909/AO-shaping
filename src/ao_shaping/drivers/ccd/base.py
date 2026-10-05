"""相机驱动基类接口。"""

from abc import ABC, abstractmethod

import numpy as np


class CameraError(Exception):
    """相机错误异常的基类。"""

    pass


class BaseCamera(ABC):
    """相机驱动的抽象基类。

    所有相机驱动都应继承本类并实现其中的抽象方法。本类提供上下文管理器支持,
    以便自动清理资源。

    单位与约定
    ---------------------
    * 曝光时间: Python 层统一用**毫秒 (ms)**。
      部分 SDK (大恒 gxipy、MIICAM) 内部用**微秒 (µs)**;
      各实现必须在 SDK 边界换算 (写入时 `ms * 1000 -> µs`,
      读取时 `µs / 1000 -> ms`)。
    * 亮度: 原始像素值; dtype 取决于位深
      (通常是 ``uint8`` 或 ``uint16``)。
    * ROI 窗口: ``(width, height)`` 与 ``(center_x, center_y)``, 单位像素。
    """

    def __init__(
        self,
        cam_id: int = 0,
        exposure_time_ms: float = 20.0,
        skip_sampling: bool = False,
    ):
        """初始化相机配置。

        参数:
            cam_id: 相机设备索引。
            exposure_time_ms: 初始曝光时间, 单位毫秒。
                各驱动共同的合法范围: ``0.011 ms`` 到
                ``10000 ms``。硬件支持时也允许亚毫秒值。
            skip_sampling: 是否启用 binning/skipping 以加快采集。
        """
        self.cam_id = int(cam_id)
        self.exposure_time_ms = float(exposure_time_ms)
        self.skip_sampling = skip_sampling

        self.cam = None
        self._sn: str | None = None
        self.cam_width: int = 0
        self.cam_height: int = 0

    def __enter__(self):
        """上下文管理器入口 —— 初始化相机。"""
        self.initialize()
        return self

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        """上下文管理器出口 —— 清理相机资源。

        各实现应关闭相机并释放资源。
        """
        pass

    @abstractmethod
    def initialize(self) -> None:
        """初始化相机设备。

        本方法应当:
        1. 关闭此前已打开的相机设备。
        2. 按 cam_id 查找并打开指定相机。
        3. 设置初始曝光时间、增益与像素格式。
        4. 若 skip_sampling 为 True, 配置分辨率与 binning。
        5. 启动数据流。

        异常:
            ConnectionAbortedError: 未找到相机设备。
            CameraError: 相机初始化失败。
        """
        pass

    @abstractmethod
    def open(self) -> None:
        """打开相机设备 (initialize 的别名)。

        这是给不使用上下文管理器的调用方准备的显式方法。
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """关闭相机设备并释放资源。"""
        pass

    @abstractmethod
    def reset_exposure_time(self, time_ms: float) -> float:
        """设置相机曝光时间。

        曝光时间在 Python 层用**毫秒 (ms)** 表示。写入硬件时, 各实现可能需要先
        换算成 SDK 的原生单位 (通常是微秒)。

        部分相机 (如 MIICAM) 在拉流期间改曝光需要走 ``Stop -> set -> Start``
        循环; 改动后的第一帧可能是陈旧帧, 因此调用方应丢弃它
        (``skip_first=True``)。

        参数:
            time_ms: 新的曝光时间, 单位毫秒。
                典型合法范围: ``0.011 ms`` 到 ``10000 ms``。
                超出硬件范围的值应被钳制。

        返回:
            float: 实际设定的曝光时间, 单位毫秒。

        异常:
            AssertionError: 相机未初始化。
        """
        pass

    @abstractmethod
    def reset_window(
        self,
        center: tuple[int, int] | tuple[np.intp, ...],
        size: tuple[int, int],
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """重设相机 ROI 窗口的大小与位置。

        参数:
            center: 期望的窗口中心位置 (x, y)。
            size: 期望的窗口大小 (width, height)。
                用 (0, 0) 表示最大分辨率。

        返回:
            实际设定的 ((width, height), (center_x, center_y))。

        异常:
            AssertionError: 相机未初始化, 或 center 非法。
        """
        pass

    @abstractmethod
    def get_numpy_image(
        self,
        n_sample: int = 1,
        skip_first: bool = True,
    ) -> np.ndarray:
        """采集相机图像, 可选做平均。

        参数:
            n_sample: 平均的采样数。必须 > 0。
            skip_first: 是否跳过第一帧。这在改曝光或改 ROI 之后尤其有用,
                因为返回的第一帧可能是陈旧或不稳定的。

        返回:
            np.ndarray: 采集到的图像, 为 ``uint8`` 或 ``uint16`` 数组,
            取决于相机当前的位深 / 像素格式。

        异常:
            AssertionError: n_sample 不是正数。
        """
        pass

    @abstractmethod
    def enable_auto_exposure(self, enable: bool = True, mode: int = 1) -> bool:
        """启用或关闭自动曝光。

        参数:
            enable: True 启用, False 关闭。
            mode: 自动曝光模式 (各实现自定义)。

        返回:
            bool: 成功返回 True, 不支持返回 False。
        """
        pass

    @abstractmethod
    def get_auto_exposure_state(self) -> dict:
        """获取当前自动曝光状态。

        返回:
            含以下内容的字典:
                - enabled: bool —— 自动曝光是否已启用
                - mode: int —— 当前模式
                - target: int —— 当前目标亮度
        """
        pass

    @abstractmethod
    def set_auto_exposure_range(
        self,
        max_time_ms: int = 350,
        min_time_ms: int = 0,
        max_gain: int = 300,
        min_gain: int = 100,
    ) -> bool:
        """设置自动曝光的时间与增益范围。

        参数:
            max_time_ms: 最大曝光时间, 单位 ms。
            min_time_ms: 最小曝光时间, 单位 ms。
            max_gain: 最大增益值。
            min_gain: 最小增益值。

        返回:
            bool: 成功返回 True, 不支持返回 False。
        """
        pass

    @staticmethod
    @abstractmethod
    def get_cam_list():
        """获取可用相机设备列表。

        返回:
            可用相机设备列表 (格式各实现自定义)。
        """
        pass

    def _get_grid(self, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        """为图像生成坐标网格。

        参数:
            width: 图像宽度。
            height: 图像高度。

        返回:
            (xv, yv) meshgrid 数组构成的 tuple。
        """
        x = np.arange(0, width)
        y = np.arange(0, height)
        xv, yv = np.meshgrid(x, y)
        return xv, yv
