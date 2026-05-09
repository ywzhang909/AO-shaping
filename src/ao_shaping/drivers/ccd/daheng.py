import numpy as np

import gxipy as gx

from ao_shaping.drivers.ccd import BaseCamera
from ao_shaping.drivers.ccd.common import ExposureTime
from ao_shaping.utils.file import logger


class DahengCamManager(BaseCamera):
    def __init__(
        self, cam_id: int = 0, exposure_time_ms: float = 0.0, skip_sampling=False
    ):
        self.device_manager = gx.DeviceManager()
        self.cam_id = int(cam_id)
        self.__exposure_time_ms = ExposureTime(exposure_time_ms)
        self.skip_sampling = skip_sampling

        self.cam = None
        self._sn: str | None = None
        self.cam_width, self.cam_height = 0, 0

    def open(self) -> None:
        """Open the camera device (alias for initialize)."""
        self.initialize()

    def close(self) -> None:
        """Close the camera device and release resources."""
        if self.cam:
            self.cam_width, self.cam_height = 0, 0
            self.cam.stream_off()
            self.cam.close_device()
            self.cam = None
            self._sn = None

    @property
    def sn(self) -> str | None:
        """Get the camera serial number."""
        return self._sn

    def is_connected(self) -> bool:
        """Check if camera is connected and ready."""
        return self.cam is not None and self._sn is not None

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

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
        try:
            self.cam = self.device_manager.open_device_by_sn(sn)
        except gx.gxiapi.InvalidAccess as e:
            if "REPEAT_OPENED" in str(e) or "device has been open" in str(e):
                logger.warning(
                    f"Device {sn} already opened, attempting to reinitialize..."
                )
                self.device_manager = gx.DeviceManager()
                self.cam = self.device_manager.open_device_by_sn(sn)
            else:
                raise
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

        self._sn = sn
        self.__update_properties()
        self.cam.stream_on()

    def reset_exposure_time(self, time_ms: int) -> int:
        """Reset the camera exposure time.

        Args:
            time_ms: The new exposure time in milliseconds. Must be >= 20.

        Returns:
            int: The actual exposure time set in milliseconds.
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
        logger.info(f"ROI Window offset: ({x_offset, y_offset})")
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

    def auto_exposure(
        self,
        target_mean: float = 0.5,
        tolerance: float = 0.05,
        max_iterations: int = 10,
        n_sample: int = 1,
    ) -> tuple[int, float]:
        """
        自动曝光调整 - 根据目标平均亮度迭代调整曝光时间。

        算法:
        1. 拍摄图像并计算平均亮度
        2. 如果平均亮度在目标值的tolerance范围内,停止
        3. 否则，根据比例调整曝光时间: new_exp = current_exp * (target / current)
        4. 裁剪到有效范围 [min, max]
        5. 重复直到收敛或达到最大迭代次数

        Args:
            target_mean: 目标平均亮度 (0-1范围, 默认0.5)
            tolerance: 容差范围 (默认0.05, 即5%)
            max_iterations: 最大迭代次数 (默认10)
            n_sample: 每次迭代的采样次数 (默认1)

        Returns:
            tuple[int, float]: (最终曝光时间ms, 最终平均亮度)
        """
        assert self.cam, "camera not initialized"

        target_val = target_mean * 255
        min_exp = self.__exposure_time_ms.min
        max_exp = self.__exposure_time_ms.max

        logger.info(
            f"Auto exposure start: target={target_mean:.2f} ({target_val:.0f}), "
            f"range=[{min_exp}, {max_exp}]ms, max_iter={max_iterations}"
        )

        current_exp = self.exposure_time
        for i in range(max_iterations):
            img = self.get_numpy_image(n_sample, skip_first=True)
            mean_val = np.mean(img)

            if abs(mean_val - target_val) <= tolerance * 255:
                logger.info(
                    f"Auto exposure converged at iter {i+1}: "
                    f"exp={current_exp}ms, mean={mean_val:.1f}"
                )
                return current_exp, mean_val / 255.0

            ratio = target_val / max(mean_val, 1)
            new_exp = int(current_exp * ratio)
            new_exp = max(min_exp, min(max_exp, new_exp))

            if new_exp == current_exp:
                logger.info(
                    f"Auto exposure stable at iter {i+1}: "
                    f"exp={current_exp}ms, mean={mean_val:.1f}"
                )
                return current_exp, mean_val / 255.0

            current_exp = new_exp
            self.reset_exposure_time(current_exp)

            logger.debug(
                f"Auto exposure iter {i+1}: mean={mean_val:.1f}, "
                f"exp={current_exp}ms (target={target_val:.0f})"
            )

        final_img = self.get_numpy_image(n_sample, skip_first=True)
        final_mean = np.mean(final_img)
        logger.warning(
            f"Auto exposure max iterations reached: "
            f"exp={current_exp}ms, mean={final_mean:.1f}"
        )
        return current_exp, final_mean / 255.0

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
            f"Open cam {self._sn} success. width={self.cam_width}, height={self.cam_height}"
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
        self.reset_exposure_time(time_ms)

    @staticmethod
    def __get_grid(width, height):
        x = np.arange(0, width)
        y = np.arange(0, height)
        xv, yv = np.meshgrid(x, y)
        return xv, yv

    @staticmethod
    def get_cam_list():
        """Get list of available cameras."""
        device_manager = gx.DeviceManager()
        return device_manager.update_device_list()

    def enable_auto_exposure(self, enable: bool = True, mode: int = 1) -> bool:
        """Enable or disable auto exposure (not supported on Daheng)."""
        logger.warning("Auto exposure not supported on Daheng camera")
        return False

    def set_auto_exposure_target(self, target: int) -> int:
        """Set auto exposure target brightness (not supported on Daheng)."""
        logger.warning("Auto exposure target not supported on Daheng camera")
        raise NotImplementedError("Auto exposure not supported on Daheng camera")

    def get_auto_exposure_state(self) -> dict:
        """Get current auto exposure state (not supported on Daheng)."""
        return {
            "enabled": False,
            "mode": 0,
            "target": 120,
        }

    def set_auto_exposure_range(
        self,
        max_time_ms: int = 350,
        min_time_ms: int = 0,
        max_gain: int = 300,
        min_gain: int = 100,
    ) -> bool:
        """Set auto exposure time and gain range (not supported on Daheng)."""
        logger.warning("Auto exposure range not supported on Daheng camera")
        return False
