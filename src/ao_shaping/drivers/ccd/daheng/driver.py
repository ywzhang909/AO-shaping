import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ao_shaping.drivers.ccd import BaseCamera
from ao_shaping.drivers.ccd.common import ExposureTime
from ao_shaping.drivers.ccd.daheng import constants
from ao_shaping.utils.io.device_config import ConfigHandler, DeviceParam, param
from ao_shaping.utils.io.file import ROOT_DIR as PROJECT_ROOT
from ao_shaping.utils.io.file import logger

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
    bit_depth: int = param(default=8, cast=int)


# 模块级单例，所有 DahengCamera 实例共用
CCD_CONFIG = ConfigHandler(_CCD_CONFIG_DIR, "ccd", CCDParams)


class DahengCamera(BaseCamera):
    def __init__(
        self,
        cam_id: int = 0,
        exposure_time_ms: float = 0.0,
        skip_sampling=False,
        bit_depth: int = 8,
    ):
        self._init_values = {
            "cam_id": cam_id,
            "exposure_time_ms": exposure_time_ms,
            "skip_sampling": skip_sampling,
            "bit_depth": bit_depth,
        }
        # 使用 defaults + __init__ 参数解析（尚未连接，无序列号）
        params = CCD_CONFIG.resolve_from_config({}, init_values=self._init_values)

        self.device_manager = gx.DeviceManager()
        self.cam_id = params.cam_id
        self.__exposure_time_ms = ExposureTime(params.exposure_time_ms)
        self.skip_sampling = params.skip_sampling
        self._bit_depth = params.bit_depth

        self.cam = None
        self._sn: str | None = None
        self.cam_width, self.cam_height = 0, 0

    @property
    def cam_type(self) -> str:
        return "daheng"

    def open(self) -> "DahengCamera":
        """Open the camera device (alias for initialize)."""
        self.initialize()
        return self

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
        # Clamp the requested exposure into the device range before writing.
        # An out-of-range value (notably 0 ms, the conventional "auto" sentinel)
        # makes the SDK `set` fail out-of-bounds and the camera silently keeps its
        # previous/default (often long) exposure → saturated frames.
        _req = self.__exposure_time_ms.ms
        _lo, _hi = self.__exposure_time_ms.min, self.__exposure_time_ms.max
        if not (_lo <= _req <= _hi):
            _clamped = min(max(_req, _lo), _hi)
            logger.warning(
                "Exposure {:.3f}ms out of range [{:.3f}, {:.3f}]ms; clamped to {:.3f}ms",
                _req,
                _lo,
                _hi,
                _clamped,
            )
            self.__exposure_time_ms.ms = _clamped
        # 写入 SDK 时再转回 µs
        self.cam.ExposureTime.set(int(self.__exposure_time_ms.ms * 1000))
        # 关闭 SDK 原生自动曝光，确保手动曝光控制
        self.cam.ExposureAuto.set(constants.GxAutoEntry.OFF.value)

        # 设置相机的增益
        self.cam.Gain.set(0.0)
        # 按位深设置相机的像素格式（MONO8 / MONO16）
        self.cam.PixelFormat.set(
            gx.GxPixelFormatEntry.MONO16
            if self._bit_depth == 16
            else gx.GxPixelFormatEntry.MONO8
        )
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

    def get_numpy_image(
        self, n_sample=1, skip_first=True, denoise=False
    ) -> npt.NDArray[np.uint16] | npt.NDArray[np.uint8]:
        """
        获取相机的图像数据，进行平均处理。

        参数:
        n_sample (int): 采样次数，用于计算平均图像。必须大于0。
        skip_first (bool): 是否跳过第一次采样，默认值为True。

        返回:
        np.ndarray: 处理后的平均图像，数据类型由位深决定 (uint16/uint8)。
        """
        out_dtype = np.uint16 if self._bit_depth == 16 else np.uint8
        numpy_image = np.zeros((n_sample, self.cam_height, self.cam_width), dtype=float)
        if skip_first:
            self.__take_one_shot()
        for i in range(n_sample):
            numpy_image[i] += self.__take_one_shot()
        avg_img = np.mean(numpy_image, axis=0)
        if denoise:
            avg_img = avg_img - np.median(avg_img)
            avg_img = np.where(avg_img < 0, 0, avg_img)
        return avg_img.astype(out_dtype)

    def auto_exposure(
        self,
        target_max: float = 40.0,
        tolerance: float = 5.0,
        twice_valid: bool = True,
        max_iterations: int = 20,
        n_sample: int = 1,
        use_sdk_auto: bool = True,
        sdk_settle_frames: int = 3,
    ) -> npt.NDArray[np.uint8]:
        """自动曝光调整 - 优先使用大恒 SDK 原生自动曝光，再用比例迭代修正。

        ``target_max`` / ``tolerance`` 均采用 **0-255 灰度**单位 (与 SDK
        ``ExpectedGrayValue`` 以及历史 ``autoset_exposure_time_ms`` 一致)。

        算法 (两阶段):
        阶段 1 - SDK 原生自动曝光 (use_sdk_auto=True 时):
          1. 启用 ExposureAuto="Once"，设置 ExpectedGrayValue 为目标亮度
          2. 采集 sdk_settle_frames 帧让 SDK 内部收敛
          3. 读取 SDK 自动调整后的曝光时间作为初值
          4. 关闭 SDK 自动曝光，切换到手动模式

        阶段 2 - 比例迭代修正:
          1. 拍摄图像并计算最大亮度
          2. 若峰值落在 ``target_max ± tolerance`` 内 (twice_valid=True 时需连续两次) 则停止
          3. 否则按 ``new_exp = exp * target / peak`` 调整 (单步放大上限 3×)
          4. 裁剪到有效范围 [min, max]，并处理曝光已到边界仍无法达标的情况
          5. 重复直到收敛或达到 max_iterations

        Args:
            target_max: 目标最大亮度 (0-255, 默认40)
            tolerance: 峰值容差 (0-255, 默认5)
            twice_valid: True 时要求连续两次落入容差范围才收敛 (默认True)
            max_iterations: 第二阶段最大迭代次数 (默认20)
            n_sample: 每次估计峰值时的采样帧数 (默认1)
            use_sdk_auto: 是否优先使用 SDK 原生自动曝光 (默认True)
            sdk_settle_frames: SDK 自动曝光收敛等待帧数 (默认3)

        Returns:
            np.ndarray: 调整后采集到的图像 (uint8)。
        """
        assert self.cam, "camera not initialized"
        assert tolerance > 0, "tolerance must be > 0"

        target_val = float(target_max)
        low = int(max(target_val - tolerance, 10))
        high = int(min(target_val + tolerance, 254))
        min_exp = float(self.__exposure_time_ms.min)
        max_exp = float(self.__exposure_time_ms.max)

        # ── 阶段 1: SDK 原生自动曝光 ──
        if use_sdk_auto:
            try:
                sdk_target = int(max(0, min(255, target_val)))
                self.cam.ExposureAuto.set(constants.GxAutoEntry.ONCE.value)
                self.cam.ExpectedGrayValue.set(sdk_target)
                logger.info(
                    "[SDK auto-exposure] enabled, target={}, waiting {} frames "
                    "for convergence...",
                    sdk_target,
                    sdk_settle_frames,
                )
                # 采集若干帧让 SDK 内部自动曝光收敛
                for _ in range(sdk_settle_frames):
                    self.__take_one_shot()
                logger.info(
                    "[SDK auto-exposure] converged: exp={:.2f}ms",
                    float(self.exposure_time),
                )
                # 切换到手动模式，后续用比例迭代修正
                self.cam.ExposureAuto.set(constants.GxAutoEntry.OFF.value)
            except Exception as e:
                logger.warning(
                    "[SDK auto-exposure] not available, falling back to manual: {}",
                    e,
                )

        logger.info(
            "Auto exposure start: target={:.0f}±{:.0f} (range=[{}, {}]ms, max_iter={})",
            target_val,
            tolerance,
            min_exp,
            max_exp,
            max_iterations,
        )

        # ── 阶段 2: 比例迭代修正 ──
        twice_ok = False
        img = self.get_numpy_image(n_sample, skip_first=True)

        for i in range(max(1, int(max_iterations))):
            peak = float(np.max(img))

            if low <= peak <= high:
                if twice_ok or not twice_valid:
                    logger.info(
                        "Auto exposure converged at iter {}: exp={:.3f}ms, max={:.1f}",
                        i + 1,
                        float(self.exposure_time),
                        peak,
                    )
                    return img
                twice_ok = True
            else:
                twice_ok = False
                current = float(self.exposure_time)
                if current <= 0:
                    break
                ratio = min(target_val / max(peak, 1.0), 3.0)
                new_exp = float(np.clip(current * ratio, min_exp, max_exp))
                if abs(new_exp - current) < 1e-9:
                    break
                self.reset_exposure_time(new_exp)
                if new_exp <= min_exp and peak > high:
                    logger.warning(
                        "target brightness {:.0f} unreachable (peak {:.1f}); "
                        "exposure forced to min",
                        target_val,
                        peak,
                    )
                    break
                if new_exp >= max_exp and peak < low:
                    logger.warning(
                        "target brightness {:.0f} unreachable (peak {:.1f}); "
                        "exposure forced to max",
                        target_val,
                        peak,
                    )
                    break

            img = self.get_numpy_image(n_sample, skip_first=True)

        logger.info(
            "Auto exposure finished: exp={:.3f}ms, max={:.1f}",
            float(self.exposure_time),
            float(np.max(img)),
        )
        return img

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
        return dev_info_list if dev_info_list else list()

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
                self.cam.ExposureAuto.set(constants.GxAutoEntry.CONTINUOUS.value)
                logger.info("Auto exposure enabled (continuous)")
            else:
                self.cam.ExposureAuto.set(constants.GxAutoEntry.OFF.value)
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
