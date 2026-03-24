import numpy as np

import gxipy as gx

from ao_shaping.drivers.ccd.common import ExposureTime
from ao_shaping.utils.file import logger


class CameraStreamManager:
    def __init__(self, cam_id: int = 0, exposure_time_ms: int = 0, skip_sampling=False):
        self.device_manager = gx.DeviceManager()
        self.cam_id = int(cam_id)
        self.__exposure_time_ms = ExposureTime(exposure_time_ms)
        self.skip_sampling = skip_sampling

        self.cam, self.__sn = None, None
        self.cam_width, self.cam_height = 0, 0

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.cam:
            self.cam_width, self.cam_height = 0, 0
            self.cam.stream_off()
            self.cam.close_device()
            self.cam, self.__sn = None, None

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
                error_info += (
                    f" Available cameras: {[_.get('sn') for _ in dev_info_list]}."
                )
            logger.error(error_info)
            raise ConnectionAbortedError(error_info)

        sn = dev_info_list[self.cam_id].get("sn")
        self.cam = self.device_manager.open_device_by_sn(sn)
        assert self.cam, "camera not found"
        # 设置相机的曝光时间
        float_range = self.cam.ExposureTime.get_range()
        if float_range:
            self.__exposure_time_ms.min = float_range["min"]
            self.__exposure_time_ms.max = float_range["max"]
        else:
            logger.warning(
                f"Exposure time range not found for camera {sn}. Using default value."
            )
        self.cam.ExposureTime.set(self.__exposure_time_ms.ms)
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

        self.__sn = sn
        self.__update_properties()
        self.cam.stream_on()

    def __reset_exposure_time(self, time_ms: float):
        """
        重置相机的曝光时间。

        参数:
        time_ms (int): 新的曝光时间，单位为毫秒。必须大于等于20。

        返回:
        int: 实际设置的曝光时间，单位为毫秒。
        """
        assert self.cam, "camera not initialized"
        time_ms = int(time_ms)
        if time_ms < self.__exposure_time_ms.min:
            v = self.__exposure_time_ms.min
            logger.warning(
                f"exposure time must >= {self.__exposure_time_ms.min}ms. set to {self.__exposure_time_ms.min}ms."
            )
        elif time_ms > self.__exposure_time_ms.max:
            v = self.__exposure_time_ms.max
            logger.warning(
                f"exposure time must <= {self.__exposure_time_ms.max}ms. set to {self.__exposure_time_ms.max}ms."
            )
        else:
            v = time_ms
        self.cam.ExposureTime.set(v)
        self.__exposure_time_ms.ms = self.exposure_time

        return self.exposure_time

    def reset_window(
        self,
        center: tuple[int, int] | tuple[np.intp, ...] = (0, 0),
        size: tuple[int, int] = (0, 0),
    ) -> tuple[tuple[int, int], tuple[int, int]]:
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
        center = int(center[0]), int(center[1])
        self.cam.stream_off()
        # 如果未指定窗口大小，则使用相机的最大宽度和高度
        if size == (0, 0):
            width, height = (self.cam.WidthMax.get(), self.cam.HeightMax.get())
            assert width and height, "camera width and height must be greater than 0"
            x_offset, y_offset = 0, 0
        else:
            width, height = size
            range_w, range_h = self.cam.Width.get_range(), self.cam.Height.get_range()
            assert range_w and range_h, "camera width and height range not found"
            width_quatic = range_w["inc"]
            width_quatic = width_quatic * 2 if width_quatic % 2 == 1 else width_quatic
            height_quatic = range_h["inc"]
            height_quatic = (
                height_quatic * 2 if height_quatic % 2 == 1 else height_quatic
            )
            width, height = (
                int(width // width_quatic * width_quatic),
                int(height // height_quatic * height_quatic),
            )
            # 计算窗口的偏移量，确保中心位置在指定位置
            x_offset, y_offset = center[0] - (width // 2), center[1] - (height // 2)
            x_offset, y_offset = (
                int(x_offset // width_quatic * width_quatic),
                int(y_offset // height_quatic * height_quatic),
            )
        assert x_offset >= 0 and y_offset >= 0, (
            f"窗口中心位置:{center}必须在图像内部，窗口大小:{size}"
        )
        self.cam.Width.set(width)
        self.cam.Height.set(height)
        self.cam.OffsetX.set(x_offset)
        # 设置相机的垂直偏移量，确保偏移量是4的倍数
        self.cam.OffsetY.set(y_offset)

        self.__update_properties()
        self.cam.stream_on()

        # 返回新的窗口中心位置
        return (width, height), (center[0] - x_offset, center[1] - y_offset)

    def __take_one_shot(self) -> np.ndarray:
        """
        拍摄一张相机图像。

        参数:
        无

        返回:
        np.ndarray: 拍摄到的图像数据，数据类型为uint8。
        """
        assert self.cam, "camera not initialized"
        while True:
            raw_image = self.cam.data_stream[0].get_image()
            if raw_image and raw_image.get_status() == gx.GxFrameStatusList.SUCCESS:
                return raw_image.get_numpy_array()

    def get_numpy_image(self, n_sample=1, skip_first=True, denoise=False) -> np.ndarray:
        """
        获取相机的图像数据，进行平均处理。

        参数:
        n_sample (int): 采样次数，用于计算平均图像。必须大于0。
        skip_first (bool): 是否跳过第一次采样，默认值为True。

        返回:
        np.ndarray: 处理后的平均图像，数据类型为uint8。
        """
        numpy_image = np.zeros((n_sample, self.cam_height, self.cam_width))  # type: ignore # ignore
        if skip_first:
            self.__take_one_shot()
        for i in range(n_sample):
            numpy_image[i] = self.__take_one_shot()
        avg_img = np.mean(numpy_image, axis=0)
        if denoise:
            avg_img = avg_img - np.median(avg_img)
            avg_img = np.where(avg_img < 0, 0, avg_img)
        return avg_img.astype(np.uint16)

    def autoset_exposure_time_ms(
        self, target_max_brightness, threshold=5, twice_valid=True
    ):
        """
        自动设置相机的曝光时间，以确保图像的最大亮度在指定的阈值范围内。

        参数:
        n_sample (int): 采样次数，用于计算平均图像。必须大于0，需要和闭环时的采样次数一致。
        target_max_brightness (float): 目标最大亮度值。
        threshold (float): 允许的最大亮度范围，默认值为0.1。必须在(0,1)之间。

        返回:
        np.ndarray: 自动设置后的图像数据，数据类型为uint8。
        """
        assert 0 < threshold, "threshold must larger than 0"
        assert self.cam, "camera not initialized"
        n_sample = 20
        low, high = target_max_brightness - threshold, target_max_brightness + threshold
        low, high = int(max(low, 10)), int(min(high, 254))

        _twice_valid_flag = False
        _img = self.get_numpy_image(n_sample)
        cur_max_brightness = max(np.max(_img), 1)
        while True:
            if low <= cur_max_brightness <= high:
                if _twice_valid_flag or not twice_valid:
                    break
                _twice_valid_flag = True
            else:
                self.exposure_time = self.exposure_time * min(
                    target_max_brightness / cur_max_brightness, 3
                )

            _img = self.get_numpy_image(n_sample)
            cur_max_brightness = max(np.max(_img), 1)

            if (
                self.exposure_time <= self.__exposure_time_ms.min
                and cur_max_brightness > high
            ):
                logger.warning(
                    f"target brightness {target_max_brightness} is too low {cur_max_brightness:.2f}, exposure time {self.exposure_time:.2f}ms force to min"
                )
                break
            elif (
                self.exposure_time >= self.__exposure_time_ms.max
                and cur_max_brightness < low
            ):
                logger.warning(
                    f"target brightness {target_max_brightness} is too high {cur_max_brightness:.2f}, exposure time {self.exposure_time:.2f}ms force to max"
                )
                break

        logger.info(
            f"autoset exposure time to {self.exposure_time:.2f}ms, max brightness={np.max(_img):.2f}"
        )
        return _img

    def __update_properties(self):
        assert self.cam, "camera not initialized"
        self.cam_width = self.cam.Width.get()
        self.cam_height = self.cam.Height.get()
        logger.info(
            f"Open cam {self.__sn} success. width={self.cam_width}, height={self.cam_height}"
        )
        self.xv, self.yv = self.__get_grid(self.cam_width, self.cam_height)

    @property
    def exposure_time(self) -> int:
        assert self.cam, "camera not initialized"
        _exp_time = self.cam.ExposureTime.get()
        return int(_exp_time) if _exp_time else 0

    @exposure_time.setter
    def exposure_time(self, time_ms: int):
        assert self.cam, "camera not initialized"
        self.__reset_exposure_time(time_ms)

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
