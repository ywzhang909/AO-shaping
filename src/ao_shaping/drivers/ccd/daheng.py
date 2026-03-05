import numpy as np

import gxipy as gx

from ao_shaping.utils.file import logger
from ao_shaping.drivers.ccd.base import BaseCamera, CameraError


class DahengError(CameraError):
    """Exception raised for Daheng camera errors."""

    pass


class CameraStreamManager(BaseCamera):
    def __init__(self, cam_id: int = 0, exposure_time_ms: int = 20, skip_sampling: bool = False):
        super().__init__(cam_id, exposure_time_ms, skip_sampling)
        self.device_manager = gx.DeviceManager()

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def open(self) -> None:
        """Open the camera device (alias for initialize)."""
        self.initialize()

    def close(self) -> None:
        """Close the camera device and release resources."""
        if self.cam:
            self.cam_width = 0
            self.cam_height = 0
            self.cam.stream_off()
            self.cam.close_device()
            self.cam = None
            self._sn = None

    def initialize(self):
        """
        初始化相机设备。

        此方法执行以下操作：
        1. 关闭之前打开的相机设备（如果有）。
        2. 更新设备列表并检查是否有足够的设备。
        3. 打开指定的相机设备。
        4. 设置相机的曝光时间、增益、像素格式、采样方式、偏移量、宽度和高度。
        5. 更新相机的属性并开启数据流。

        如果没有找到相机设备，将记录错误并抛出连接中止错误。

        参数:
            无

        返回:
            无
        """
        # 关闭之前打开的相机设备（如果有）
        self.__exit__(None, None, None)

        # 更新设备列表并获取设备信息列表
        _, dev_info_list = self.device_manager.update_device_list()
        # 检查设备列表长度是否小于等于指定的相机ID
        if not dev_info_list or len(dev_info_list) <= self.cam_id:
            error_info = f"Camera ID {self.cam_id} not found. "
            if dev_info_list:
                error_info += f" Available cameras: {[_.get('sn') for _ in dev_info_list]}."
            logger.error(error_info)
            raise ConnectionAbortedError(error_info)

        sn = dev_info_list[self.cam_id].get("sn")
        self.cam = self.device_manager.open_device_by_sn(sn)
        assert self.cam, "camera not found"
        # 设置相机的曝光时间
        self.cam.ExposureTime.set(self.exposure_time_ms)
        # 设置相机的增益
        self.cam.Gain.set(0.0)
        # 设置相机的像素格式为MONO8
        self.cam.PixelFormat.set(gx.GxPixelFormatEntry.MONO8)
        if self.skip_sampling:
            # 设置相机的合并因子为2
            self.cam.BinningHorizontal.set(2)
            self.cam.BinningVertical.set(2)

        # 设置相机的水平偏移量为0
        self.cam.OffsetX.set(0)
        self.cam.OffsetY.set(0)
        # 设置相机的宽度为最大宽度
        self.cam.Width.set(self.cam.WidthMax.get())
        self.cam.Height.set(self.cam.HeightMax.get())

        self._sn = sn
        self.__update_properties()
        self.cam.stream_on()

    def enable_auto_exposure(self, enable: bool = True, mode: int = 1) -> bool:
        """
        启用或禁用自动曝光。

        Args:
            enable (bool): True 启用, False 禁用。
            mode (int): 自动曝光模式 (大恒相机不支持模式选择，仅启用/禁用)。

        Returns:
            bool: 是否成功。
        """
        assert self.cam, "camera not initialized"
        try:
            self.cam.ExposureAuto.set(gx.GxAutoEntry.CONTINUOUS if enable else gx.GxAutoEntry.OFF)
            return True
        except Exception as e:
            logger.warning(f"Auto exposure not supported: {e}")
            return False

    def set_auto_exposure_target(self, target: int) -> int:
        """
        设置自动曝光目标亮度。

        Args:
            target (int): 目标亮度值。

        Returns:
            int: 设置的目标值。

        Raises:
            NotImplementedError: 大恒相机不支持直接设置自动曝光目标值。
        """
        assert self.cam, "camera not initialized"
        # 大恒相机不支持直接设置自动曝光目标值
        raise NotImplementedError(
            "Daheng camera does not support setting auto exposure target directly. "
            "Auto exposure uses internal algorithm."
        )

    def get_auto_exposure_state(self) -> dict:
        """
        获取当前自动曝光状态。

        Returns:
            dict: 包含以下字段的字典:
                - enabled: bool - 自动曝光是否启用
                - mode: int - 当前模式 (0=关闭, 1=连续)
                - target: int - 当前目标亮度值 (大恒相机不支持, 返回默认值120)
        """
        assert self.cam, "camera not initialized"
        state = {
            "enabled": False,
            "mode": 0,
            "target": 120,
        }
        try:
            # 大恒相机的 ExposureAuto 返回 GxAutoEntry 枚举值
            # CONTINUOUS = 2, OFF = 0
            auto_entry = self.cam.ExposureAuto.get()
            state["enabled"] = auto_entry == gx.GxAutoEntry.CONTINUOUS
            state["mode"] = 1 if state["enabled"] else 0
        except Exception as e:
            logger.warning(f"Failed to get auto exposure state: {e}")
        return state

    def set_auto_exposure_range(
        self,
        max_time_ms: int = 350,
        min_time_ms: int = 0,
        max_gain: int = 300,
        min_gain: int = 100,
    ) -> bool:
        """
        设置自动曝光的曝光时间和增益范围。

        Args:
            max_time_ms: 最大曝光时间 (毫秒)
            min_time_ms: 最小曝光时间 (毫秒)
            max_gain: 最大增益
            min_gain: 最小增益

        Returns:
            bool: 如果设置成功返回 True, 否则返回 False
        """
        assert self.cam, "camera not initialized"
        try:
            # 大恒相机 SDK 可能支持 ExposureTimeRange 相关设置
            # 但由于 gxipy 封装限制, 这里尝试设置曝光时间范围
            # 如果 SDK 支持, 可以通过 cam.ExposureTime 的 min/max 范围设置
            logger.info(
                f"Setting auto exposure range: time {min_time_ms}-{max_time_ms} ms, "
                f"gain {min_gain}-{max_gain}"
            )
            # 大恒相机 SDK 未提供直接的自动曝光范围设置接口
            # 返回 False 表示不支持此功能
            return False
        except Exception as e:
            logger.warning(f"Failed to set auto exposure range: {e}")
            return False

    def reset_exposure_time(self, time_ms:int):
        """
        重置相机的曝光时间。

        参数:
        time_ms (int): 新的曝光时间，单位为毫秒。必须大于等于20。

        返回:
        int: 实际设置的曝光时间，单位为毫秒。
        """
        assert self.cam, "camera not initialized"
        if time_ms >= 20:
            self.exposure_time_ms = time_ms
        else:
            self.exposure_time_ms = 20
            logger.warning('exposure time must >= 20. set to 20.')
        self.cam.ExposureTime.set(self.exposure_time_ms)
        return self.exposure_time_ms

    def reset_window(self, center:tuple[int,int]|tuple[np.intp,...]=(0,0), size:tuple[int,int]=(0,0)) -> tuple[tuple[int,int], tuple[int,int]]:
        """
        重置相机的窗口大小和位置，以确保图像的中心位于指定的位置。

        参数:
        size (Tuple[int]): 期望的窗口大小，格式为 (宽度, 高度)。
        center (Tuple[int]): 期望的窗口中心位置，格式为 (x坐标, y坐标)。

        返回:
        Tuple[int]: 新的窗口中心位置，格式为 (x坐标, y坐标)。
        """
        # 中心坐标大于0
        assert self.cam, "camera not initialized"
        center = tuple(int(c) for c in center)
        self.cam.stream_off()
        # 如果未指定窗口大小，则使用相机的最大宽度和高度
        if size == (0, 0):
            width, height = (self.cam.WidthMax.get(), self.cam.HeightMax.get())
            x_offset, y_offset = 0, 0
        else:
            width, height = size
            width_quatic = self.cam.Width.get_range()['inc']
            width_quatic = width_quatic*2 if width_quatic%2==1 else width_quatic
            height_quatic = self.cam.Height.get_range()['inc']
            height_quatic = height_quatic*2 if height_quatic%2==1 else height_quatic
            width, height = int(width//width_quatic*width_quatic), int(height//height_quatic*height_quatic)
            # 计算窗口的偏移量，确保中心位置在指定位置
            x_offset, y_offset = center[0]-(width//2), center[1]-(height//2)
            x_offset, y_offset = int(x_offset//width_quatic*width_quatic), int(y_offset//height_quatic*height_quatic)
        assert x_offset>=0 and y_offset>=0, f"窗口中心位置:{center}必须在图像内部，窗口大小:{size}"
        self.cam.Width.set(width)
        self.cam.Height.set(height)
        self.cam.OffsetX.set(x_offset)
        # 设置相机的垂直偏移量，确保偏移量是4的倍数
        self.cam.OffsetY.set(y_offset)

        self.__update_properties()
        self.cam.stream_on()

        # 返回新的窗口中心位置
        return (width, height), (width//2, height//2)

    def __take_one_shot(self) -> np.ndarray:
        """
        拍摄一张相机图像。

        参数:
        无

        返回:
        np.ndarray: 拍摄到的图像数据，数据类型为uint8。
        """
        while True:
            raw_image = self.cam.data_stream[0].get_image()
            if raw_image and (new_img:=raw_image.get_numpy_array()) is not None:
                return new_img
    
    def get_numpy_image(self, n_sample=1, skip_first=True) -> np.ndarray:
        """
        获取相机的图像数据，进行平均处理。

        参数:
        n_sample (int): 采样次数，用于计算平均图像。必须大于0。
        skip_first (bool): 是否跳过第一次采样，默认值为True。

        返回:
        np.ndarray: 处理后的平均图像，数据类型为uint8。
        """
        assert n_sample>0, "采样次数必须大于0"
        numpy_image = np.zeros_like(self.__take_one_shot()) if skip_first else self.__take_one_shot()
        _n_sample = n_sample if skip_first else n_sample-1
        for _ in range(_n_sample):
            numpy_image = numpy_image + self.__take_one_shot()
        avg_img = numpy_image/_n_sample
        return avg_img.astype(np.uint8)

    def __update_properties(self):
        assert self.cam, "camera not initialized"
        self.cam_width = self.cam.Width.get()
        self.cam_height = self.cam.Height.get()
        logger.info(f"Open cam {self._sn} success. width={self.cam_width}, height={self.cam_height}")
        self.xv, self.yv = self.__get_grid(self.cam_width, self.cam_height)

    @staticmethod
    def __get_grid(width, height):
        x = np.arange(0, width)
        y = np.arange(0, height)
        xv, yv = np.meshgrid(x, y)
        return xv, yv

    @staticmethod
    def get_cam_list():
        device_manager = gx.DeviceManager()
        return device_manager.update_device_list()