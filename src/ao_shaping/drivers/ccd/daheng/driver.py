import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ao_shaping.drivers.ccd import BaseCamera
from ao_shaping.drivers.ccd.common import ExposureTime
from ao_shaping.utils.device_config import ConfigHandler, DeviceParam, param
from ao_shaping.utils.file import ROOT_DIR as PROJECT_ROOT
from ao_shaping.utils.file import logger

try:
    import gxipy as gx  # type: ignore[import-untyped]
except (ImportError, OSError) as e:
    logger.error(f"Daheng SDK import failed: {e}")

# ── CCD 配置参数 ─────────────────────────────────────────


_CCD_CONFIG_DIR = Path(
    os.environ.get("CCD_CONFIG_DIR", PROJECT_ROOT / "data" / "ccd_configs")
)


@dataclass
class CCDParams(DeviceParam):
    """Daheng 相机配置参数。"""

    cam_id: int = param(default=0, cast=int)
    exposure_time_ms: float = param(default=0.0, cast=float)
    skip_sampling: bool = param(default=False, cast=bool)


# 模块级单例，所有 DahengCamManager 实例共用
CCD_CONFIG = ConfigHandler(_CCD_CONFIG_DIR, "ccd", CCDParams)


class DahengCamManager(BaseCamera):
    def __init__(
        self, cam_id: int = 0, exposure_time_ms: float = 0.0, skip_sampling=False
    ):
        self._init_values = {
            "cam_id": cam_id,
            "exposure_time_ms": exposure_time_ms,
            "skip_sampling": skip_sampling,
        }
        # 使用 defaults + __init__ 参数解析（尚未连接，无序列号）
        params = CCD_CONFIG.resolve_from_config({}, init_values=self._init_values)

        self.device_manager = gx.DeviceManager()
        self.cam_id = params.cam_id
        self.__exposure_time_ms = ExposureTime(params.exposure_time_ms)
        self.skip_sampling = params.skip_sampling

        self.cam = None
        self._sn: str | None = None
        self.cam_width, self.cam_height = 0, 0

    @property
    def cam_type(self) -> str:
        return "daheng"

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

    def load_config(self) -> dict:
        """加载当前设备的配置文件。

        Returns:
            配置字典；无序列号或文件不存在时返回空字典。
        """
        if not self._sn:
            return {}
        return CCD_CONFIG._manager.load_config(self._sn)

    def save_config(self) -> None:
        """将当前参数保存到 JSON 配置文件。"""
        if not self._sn:
            logger.warning("未获取到序列号，跳过配置保存")
            return
        config = CCD_CONFIG.collect(self)
        CCD_CONFIG._manager.save_config(self._sn, config)
        config_file = CCD_CONFIG._manager._get_config_file(self._sn)
        logger.info(f"相机配置已保存: {config_file}")

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
        except gx.gxiapi.InvalidAccess as e:  # type: ignore[attr-defined]
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
            # gxipy SDK 的 ExposureTime 单位是微秒 (μs)，
            # 转换为毫秒 (ms) 存入内部 ExposureTime，保证全驱动统一用 ms。
            self.__exposure_time_ms.min = float_range["min"] / 1000.0
            self.__exposure_time_ms.max = float_range["max"] / 1000.0
            logger.info(
                "Exposure time range: {:.1f}~{:.1f} µs ({:.3f}~{:.3f} ms)",
                float_range["min"],
                float_range["max"],
                self.__exposure_time_ms.min,
                self.__exposure_time_ms.max,
            )
        else:
            logger.warning(
                f"Exposure time range not found for camera {sn}. Using default value."
            )
        # 写入 SDK 时再转回 µs
        self.cam.ExposureTime.set(int(self.__exposure_time_ms.ms * 1000))
        # 关闭 SDK 原生自动曝光，确保手动曝光控制
        try:
            self.cam.ExposureAuto.set("Off")
        except Exception:
            pass
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

        # 按序列号加载并应用配置文件（不覆盖 __init__ 显式参数）
        config = self.load_config()
        CCD_CONFIG.apply_from_config(self, config, init_values=self._init_values)
        # 重新应用曝光时间（配置可能覆盖了 exposure_time_ms）
        self.cam.ExposureTime.set(int(self.__exposure_time_ms.ms * 1000))

        self.__update_properties()
        self.cam.stream_on()

    def reset_exposure_time(self, time_ms: float) -> float:
        """Reset the camera exposure time.

        Args:
            time_ms: The new exposure time in milliseconds.

        Returns:
            float: The actual exposure time set in milliseconds.
        """
        assert self.cam, "camera not initialized"
        time_ms = float(time_ms)
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
        self.cam.ExposureTime.set(int(v * 1000))
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

    def __take_one_shot(self) -> npt.NDArray[np.uint8]:
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

    def get_numpy_image(self, n_sample=1, skip_first=True, denoise=False) -> npt.NDArray[np.uint8]:
        """
        获取相机的图像数据，进行平均处理。

        参数:
        n_sample (int): 采样次数，用于计算平均图像。必须大于0。
        skip_first (bool): 是否跳过第一次采样，默认值为True。

        返回:
        np.ndarray: 处理后的平均图像，数据类型为uint8。
        """
        numpy_image = np.zeros((n_sample, self.cam_height, self.cam_width), dtype=float)
        if skip_first:
            self.__take_one_shot()
        for i in range(n_sample):
            numpy_image[i] += self.__take_one_shot()
        avg_img = np.mean(numpy_image, axis=0)
        if denoise:
            avg_img = avg_img - np.median(avg_img)
            avg_img = np.where(avg_img < 0, 0, avg_img)
        return avg_img.astype(np.uint8)

    def auto_exposure(
        self,
        target_mean: float = 0.5,
        tolerance: float = 0.05,
        max_iterations: int = 10,
        n_sample: int = 1,
        use_sdk_auto: bool = True,
        sdk_settle_frames: int = 3,
    ) -> tuple[float, float]:
        """自动曝光调整 - 优先使用大恒 SDK 原生自动曝光，再用已有的曝光调整方法修正。

        算法 (两阶段):
        阶段 1 - SDK 原生自动曝光 (use_sdk_auto=True 时):
          1. 启用 ExposureAuto="On"，设置 ExpectedGrayValue 为目标亮度
          2. 采集 sdk_settle_frames 帧让 SDK 内部收敛
          3. 读取 SDK 自动调整后的曝光时间作为初值
          4. 关闭 SDK 自动曝光，切换到手动模式

        阶段 2 - 已有曝光调整方法修正:
          1. 拍摄图像并计算平均亮度
          2. 如果平均亮度在目标值的 tolerance 范围内，停止
          3. 否则，根据比例调整曝光时间: new_exp = current_exp * (target / current)
          4. 裁剪到有效范围 [min, max]
          5. 重复直到收敛或达到最大迭代次数

        Args:
            target_mean: 目标平均亮度 (0-1范围, 默认0.5)
            tolerance: 容差范围 (默认0.05, 即5%)
            max_iterations: 第二阶段最大迭代次数 (默认10)
            n_sample: 每次迭代的采样次数 (默认1)
            use_sdk_auto: 是否优先使用 SDK 原生自动曝光 (默认True)
            sdk_settle_frames: SDK 自动曝光收敛等待帧数 (默认3)

        Returns:
            tuple[float, float]: (最终曝光时间ms, 最终平均亮度)
        """
        assert self.cam, "camera not initialized"

        target_val = float(target_mean * 255)
        min_exp = float(self.__exposure_time_ms.min)
        max_exp = float(self.__exposure_time_ms.max)

        # ── 阶段 1: SDK 原生自动曝光 ──
        if use_sdk_auto:
            try:
                sdk_target = int(max(0, min(255, target_val)))
                self.cam.ExposureAuto.set("On")
                self.cam.ExpectedGrayValue.set(sdk_target)
                logger.info(
                    f"[SDK auto-exposure] enabled, target={sdk_target}, "
                    f"waiting {sdk_settle_frames} frames for convergence..."
                )
                # 采集若干帧让 SDK 内部自动曝光收敛
                for _ in range(sdk_settle_frames):
                    self.__take_one_shot()
                # 读取 SDK 自动调整后的曝光时间
                sdk_exp = float(self.exposure_time)
                logger.info(f"[SDK auto-exposure] converged: exp={sdk_exp:.2f}ms")
                # 切换到手动模式，后续用已有方法修正
                self.cam.ExposureAuto.set("Off")
                current_exp = sdk_exp
            except Exception as e:
                logger.warning(
                    f"[SDK auto-exposure] not available, falling back to manual: {e}"
                )
                current_exp = float(self.exposure_time)
        else:
            current_exp = float(self.exposure_time)

        logger.info(
            f"Auto exposure phase 2 start: target={target_mean:.2f} ({target_val:.0f}), "
            f"range=[{min_exp}, {max_exp}]ms, max_iter={max_iterations}"
        )

        # ── 阶段 2: 已有曝光调整方法修正 ──
        for i in range(max_iterations):
            img = self.get_numpy_image(n_sample, skip_first=True)
            mean_val = float(np.mean(img))

            if abs(mean_val - target_val) <= tolerance * 255:
                logger.info(
                    f"Auto exposure converged at iter {i + 1}: "
                    f"exp={current_exp}ms, mean={mean_val:.1f}"
                )
                return current_exp, mean_val / 255.0

            ratio = target_val / max(mean_val, 1)
            new_exp = current_exp * ratio
            new_exp = max(min_exp, min(max_exp, new_exp))

            if new_exp == current_exp:
                logger.info(
                    f"Auto exposure stable at iter {i + 1}: "
                    f"exp={current_exp}ms, mean={mean_val:.1f}"
                )
                return current_exp, mean_val / 255.0

            current_exp = new_exp
            self.reset_exposure_time(current_exp)

            logger.debug(
                f"Auto exposure iter {i + 1}: mean={mean_val:.1f}, "
                f"exp={current_exp}ms (target={target_val:.0f})"
            )

        final_img = self.get_numpy_image(n_sample, skip_first=True)
        final_mean = float(np.mean(final_img))
        logger.warning(
            f"Auto exposure max iterations reached: "
            f"exp={current_exp}ms, mean={final_mean:.1f}"
        )
        return current_exp, final_mean / 255.0

    def autoset_exposure_time_ms(
        self,
        target_max_brightness,
        threshold=5,
        twice_valid=True,
        use_sdk_auto: bool = True,
        sdk_settle_frames: int = 3,
    ):
        """自动设置相机的曝光时间，以确保图像的最大亮度在指定的阈值范围内。

        两阶段策略:
        阶段 1 (use_sdk_auto=True): 启用大恒 SDK 原生自动曝光
          (ExposureAuto="On" + ExpectedGrayValue)，采集若干帧让 SDK 收敛，
          读取 SDK 自动调整后的曝光时间作为初值，然后关闭 SDK 自动曝光。
        阶段 2: 用已有曝光调整方法 (比例迭代) 修正到目标最大亮度。

        参数:
            target_max_brightness (float): 目标最大亮度值。
            threshold (float): 允许的最大亮度范围，默认值为5。
            twice_valid (bool): 是否需要连续两次验证通过。
            use_sdk_auto (bool): 是否优先使用 SDK 原生自动曝光 (默认True)
            sdk_settle_frames (int): SDK 自动曝光收敛等待帧数 (默认3)

        返回:
            np.ndarray: 自动设置后的图像数据，数据类型为uint8。
        """
        assert 0 < threshold, "threshold must larger than 0"
        assert self.cam, "camera not initialized"
        n_sample = 20
        low, high = target_max_brightness - threshold, target_max_brightness + threshold
        low, high = int(max(low, 10)), int(min(high, 254))

        # ── 阶段 1: SDK 原生自动曝光 ──
        if use_sdk_auto:
            try:
                sdk_target = int(max(0, min(255, target_max_brightness)))
                self.cam.ExposureAuto.set("On")
                self.cam.ExpectedGrayValue.set(sdk_target)
                logger.info(
                    f"[SDK auto-exposure] enabled, target={sdk_target}, "
                    f"waiting {sdk_settle_frames} frames for convergence..."
                )
                for _ in range(sdk_settle_frames):
                    self.__take_one_shot()
                logger.info(
                    f"[SDK auto-exposure] converged: exp={self.exposure_time:.2f}ms"
                )
                self.cam.ExposureAuto.set("Off")
            except Exception as e:
                logger.warning(
                    f"[SDK auto-exposure] not available, falling back to manual: {e}"
                )

        # ── 阶段 2: 已有曝光调整方法修正 ──
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
        width_val = self.cam.Width.get()
        height_val = self.cam.Height.get()
        assert width_val is not None and height_val is not None, (
            "camera size not available"
        )
        self.cam_width = int(width_val)
        self.cam_height = int(height_val)
        logger.info(
            f"Open cam {self._sn} success. width={self.cam_width}, height={self.cam_height}"
        )
        self.xv, self.yv = self.__get_grid(self.cam_width, self.cam_height)

    @property
    def exposure_time(self) -> float:
        assert self.cam, "camera not initialized"
        _exp_time = self.cam.ExposureTime.get()
        return _exp_time / 1000.0 if _exp_time else 0.0

    @exposure_time.setter
    def exposure_time(self, time_ms: float):
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
        _, dev_info_list = device_manager.update_device_list()
        return dev_info_list

    def get_exposure_range(self) -> tuple[float, float]:
        """Get the camera's supported exposure time range.

        Returns:
            (min_exposure_ms, max_exposure_ms)
        """
        assert self.cam, "camera not initialized"
        return float(self.__exposure_time_ms.min), float(self.__exposure_time_ms.max)

    def enable_auto_exposure(self, enable: bool = True, mode: int = 1) -> bool:
        """Enable or disable auto exposure.

        Uses Daheng SDK ``ExposureAuto`` enum (``"On"``/``"Off"``).
        When enabling, the camera automatically adjusts exposure time and
        gain to reach the target brightness set via ``ExpectedGrayValue``.

        Args:
            enable: True to enable, False to disable.
            mode: Auto exposure mode (0=disable, 1=continuous, 2=once).
                  Only meaningful when ``enable=True``; Daheng SDK does not
                  expose a separate mode enum, so continuous is always used.

        Returns:
            bool: True if successful, False if not supported.
        """
        assert self.cam, "camera not initialized"
        try:
            if enable:
                self.cam.ExposureAuto.set("On")
                logger.info("Auto exposure enabled (continuous)")
            else:
                self.cam.ExposureAuto.set("Off")
                logger.info("Auto exposure disabled")
            return True
        except Exception as e:
            logger.warning(f"Auto exposure not supported on Daheng camera: {e}")
            return False

    def set_auto_exposure_target(self, target: int) -> int:
        """Set auto exposure target brightness.

        Maps to Daheng SDK ``ExpectedGrayValue`` (IntFeature, 0-255,
        default 120). Only effective when auto exposure is enabled.

        Args:
            target: Target brightness value. Range: 0-255, default: 120.

        Returns:
            The target value that was set.
        """
        assert self.cam, "camera not initialized"
        target = max(0, min(255, target))
        try:
            self.cam.ExpectedGrayValue.set(target)
            logger.info(f"Auto exposure target brightness set to {target}")
        except Exception as e:
            logger.warning(f"Auto exposure target not supported: {e}")
        return target

    def get_auto_exposure_state(self) -> dict:
        """Get current auto exposure state.

        Reads ``ExposureAuto`` enum and ``ExpectedGrayValue`` from the
        Daheng SDK.

        Returns:
            Dictionary containing:
                - enabled: bool - Whether auto exposure is enabled
                - mode: int - Current mode (1=continuous, 0=disabled)
                - target: int - Current target brightness (0-255)
                - exposure_time_ms: float - Current exposure time in ms
        """
        assert self.cam, "camera not initialized"
        state: dict = {
            "enabled": False,
            "mode": 0,
            "target": 120,
            "exposure_time_ms": 0.0,
        }
        try:
            _val, auto_key = self.cam.ExposureAuto.get()
            state["enabled"] = auto_key == "On"
            state["mode"] = 1 if auto_key == "On" else 0
            try:
                state["target"] = self.cam.ExpectedGrayValue.get()
            except Exception:
                pass
            state["exposure_time_ms"] = self.exposure_time
        except Exception as e:
            logger.warning(f"Auto exposure state not supported: {e}")
        return state

    def set_auto_exposure_range(
        self,
        max_time_ms: int = 350,
        min_time_ms: int = 0,
        max_gain: int = 300,
        min_gain: int = 100,
    ) -> bool:
        """Set auto exposure time and gain range.

        Maps to Daheng SDK ``AutoExposureTimeMin``/``Max`` (FloatFeature,
        µs) and ``AutoGainMin``/``Max`` (FloatFeature). These bounds
        constrain the auto-exposure / auto-gain search space.

        Args:
            max_time_ms: Maximum exposure time in ms.
            min_time_ms: Minimum exposure time in ms.
            max_gain: Maximum gain value.
            min_gain: Minimum gain value.

        Returns:
            bool: True if successful, False if not supported.
        """
        assert self.cam, "camera not initialized"
        try:
            self.cam.AutoExposureTimeMin.set(min_time_ms * 1000)
            self.cam.AutoExposureTimeMax.set(max_time_ms * 1000)
            self.cam.AutoGainMin.set(min_gain)
            self.cam.AutoGainMax.set(max_gain)
            logger.info(
                f"Auto exposure range set: time=[{min_time_ms}, {max_time_ms}]ms, "
                f"gain=[{min_gain}, {max_gain}]"
            )
            return True
        except Exception as e:
            logger.warning(f"Auto exposure range not supported: {e}")
            return False

    @property
    def auto_exposure_value(self) -> float:
        """Current auto exposure numerical value (exposure time in ms).

        When auto exposure is enabled, this returns the exposure time
        the SDK has auto-adjusted to. When disabled, it returns the
        manually-set exposure time.

        Returns:
            float: Current exposure time in milliseconds.
        """
        return self.exposure_time
