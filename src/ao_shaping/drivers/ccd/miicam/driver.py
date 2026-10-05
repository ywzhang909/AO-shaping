from __future__ import annotations

import ctypes
import threading
import time
from typing import Any, Callable, Self

import numpy as np
from loguru import logger

from ao_shaping.drivers.ccd.base import BaseCamera, CameraError
from ao_shaping.drivers.ccd.miicam._sdk_setup import _setup_miicam_sdk

# 导入前先设置好 MIICAM SDK
_MIICAM_AVAILABLE = _setup_miicam_sdk()

if _MIICAM_AVAILABLE:
    import miicam
else:
    miicam = None


class MIICAMError(CameraError):
    """MIICAM 相机错误异常。"""

    pass


# ===== 内部辅助类 =====


class _CallbackSession:
    """管理单次采集或连续拉流的回调模式生命周期。

    跟踪回调模式此前是否已激活, 以免干扰由 ``start_callback_mode`` 启动的
    既有回调流。
    """

    def __init__(self, owner: "MIICamera") -> None:
        self._owner = owner
        self._was_active: bool = False

    def start(self, callback: Callable | None = None) -> None:
        """进入回调模式, 或复用已激活的会话。"""
        self._was_active = self._owner._callback_mode_active
        if not self._was_active:
            self._owner._stop_streaming()
            self._owner._callback_user = callback
            self._owner._callback_buffer_lock = threading.Lock()
            self._owner._callback_buffer = None
            self._owner._callback_frame_info = None
            self._owner._callback_new_frame = threading.Event()
            self._owner.cam.StartPullModeWithCallback(
                self._owner._frame_callback, self._owner
            )
        self._owner._callback_mode_active = True

    def stop(self) -> None:
        """若回调模式是本会话启动的, 则退出它。"""
        if not self._was_active:
            self._owner._stop_streaming()
            time.sleep(0.3)
            self._owner.cam.StartPullModeWithCallback(None, None)
            self._owner._callback_mode_active = False
        self._owner._callback_user = None


class _FramePuller:
    """带重试地从相机拉取帧。"""

    def __init__(self, owner: "MIICamera") -> None:
        self._owner = owner

    def wait_image(self, timeout_ms: int | None = None) -> np.ndarray:
        """WaitImageV3, 超时后重试。

        超时值是自适应的: 若未给定, 取 ``max(300, exposure_ms * 5)``。
        """
        if timeout_ms is None:
            timeout_ms = max(300, int(self._owner.exposure_time_ms * 5))

        bufsize, bits, dtype = self._owner._get_buffer_params()
        buffer = (ctypes.c_char * bufsize)()
        frame_info = miicam.MiicamFrameInfoV3()

        max_retries = 2
        for attempt in range(max_retries):
            try:
                self._owner.cam.WaitImageV3(timeout_ms, buffer, 0, bits, 0, frame_info)
                break
            except miicam.HRESULTException as e:
                hr = getattr(e, "hr", getattr(e, "winerror", 0))
                if hr in (0x8000000A, 0x8001011F):
                    if attempt < max_retries - 1:
                        time.sleep(0.02)
                        continue
                    raise MIICAMError(
                        f"WaitImageV3 timeout after {max_retries} retries"
                    ) from e
                if attempt < max_retries - 1:
                    time.sleep(0.02)
                    continue
                raise MIICAMError(
                    f"WaitImageV3 failed: hr=0x{hr & 0xFFFFFFFF:08x}"
                ) from e

        img_data = np.frombuffer(buffer, dtype=dtype)
        return self._owner._decode_image(img_data)

    def pull_image_v4(self, buffer, bits, frame_info) -> None:
        """PullImageV4."""
        self._owner.cam.PullImageV4(buffer, 0, bits, 0, frame_info)

    def pull_still_image(self, buffer, bits, frame_info) -> None:
        """PullStillImageV2."""
        self._owner.cam.PullStillImageV2(buffer, bits, frame_info)


# ===== 相机主类 =====


class MIICamera(BaseCamera):
    """MIICAM 相机拉流管理器。

    支持两种采集模式:
    - **wait** (默认): 阻塞式 ``WaitImageV3`` 拉取。
    - **callback**: ``StartPullModeWithCallback`` + 软件 ``Trigger`` +
      ``PullImageV4`` (参考 C++ ``demosofttrigger``)。
    """

    MIN_EXPOSURE_MS = 0.011
    MAX_EXPOSURE_MS = 10000.0

    @staticmethod
    def get_exposure_range() -> tuple[float, float]:
        """获取相机支持的曝光时间范围, 单位 ms。

        返回:
            (min_exposure_ms, max_exposure_ms)
        """
        return MIICamera.MIN_EXPOSURE_MS, MIICamera.MAX_EXPOSURE_MS

    @classmethod
    def from_params(cls, params: Any, **overrides: Any) -> Self:
        """由驱动参数对象构造一个 MiiCam 相机。"""
        kwargs = {
            "cam_id": getattr(params, "cam_id", 0),
            "exposure_time_ms": getattr(params, "exposure_time_ms", 20.0),
            "skip_sampling": getattr(params, "skip_sampling", False),
            "bit_depth": getattr(params, "bit_depth", 8),
            "capture_mode": getattr(params, "capture_mode", "wait"),
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    def __init__(
        self,
        cam_id: int = 0,
        exposure_time_ms: float = 20.0,
        skip_sampling: bool = False,
        bit_depth: int = 8,
        capture_mode: str = "wait",
    ):
        """初始化 MIICAM 相机。

        参数:
            cam_id: 相机索引。
            exposure_time_ms: 曝光时间, 单位毫秒。
            skip_sampling: 启用 2x2 binning。
            bit_depth: 输出位深 (8 或 16)。
            capture_mode: "wait" 用 WaitImageV3 (阻塞拉取),
                "callback" 用基于回调的 PullImageV4 (软件触发模式)。
        """
        if capture_mode not in ("wait", "callback"):
            raise ValueError(
                f"capture_mode must be 'wait' or 'callback', got '{capture_mode}'"
            )
        super().__init__(cam_id, exposure_time_ms, skip_sampling)
        self._bit_depth = bit_depth
        self._pixel_format = "MONO8" if bit_depth == 8 else "MONO16"
        self._max_bit_depth = 8  # 稍后从相机更新
        self._capture_mode = capture_mode
        self._callback_mode_active: bool = False

        # 曝光上下限 (ms) —— SDK 的硬件约束
        self._min_exposure_ms = MIICamera.MIN_EXPOSURE_MS
        self._max_exposure_ms = MIICamera.MAX_EXPOSURE_MS

        # 辅助对象 (在 cam 打开之后初始化)
        self._callback_session: _CallbackSession | None = None
        self._frame_puller: _FramePuller | None = None

    @property
    def min_exposure_ms(self) -> float:
        """最小曝光时间, 单位毫秒。"""
        return MIICamera.MIN_EXPOSURE_MS

    @property
    def max_exposure_ms(self) -> float:
        """最大曝光时间, 单位毫秒。"""
        return MIICamera.MAX_EXPOSURE_MS

    # =========================================================================
    # 上下文管理器 / 生命周期
    # =========================================================================

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def open(self) -> "MIICamera":
        """打开相机设备 (initialize 的别名)。"""
        self.initialize()
        return self

    def close(self) -> None:
        """关闭相机设备并释放资源。"""
        if self.cam:
            # 带重试地停止拉流 —— 回调线程可能需要一点时间退出
            for _ in range(5):
                try:
                    self.cam.Stop()
                    break
                except miicam.HRESULTException:
                    time.sleep(0.1)
                except Exception:
                    break

            # 清空待处理帧
            try:
                self.cam.put_Option(miicam.MIICAM_OPTION_FLUSH, 3)
            except Exception:
                pass

            # 留一点时间让回调线程结束
            time.sleep(0.3)

            try:
                self.cam.Close()
            except Exception:
                pass

            self.cam_width = 0
            self.cam_height = 0
            self.cam = None

            # 重新打开前给相机硬件一点时间稳定
            time.sleep(0.5)

    # =========================================================================
    # 初始化
    # =========================================================================

    def initialize(self) -> None:
        """初始化相机设备。"""
        # 关闭此前已打开的相机设备 (若有)
        self.__exit__(None, None, None)

        # 预防性处理: 尝试停掉上一次会话遗留的拉流
        try:
            dev_list = miicam.Miicam.EnumV2()
            if dev_list and len(dev_list) > self.cam_id:
                temp_cam = miicam.Miicam.Open(dev_list[self.cam_id].id)
                if temp_cam:
                    try:
                        temp_cam.Stop()
                    except Exception:
                        pass
                    try:
                        temp_cam.put_Option(miicam.MIICAM_OPTION_FLUSH, 3)
                    except Exception:
                        pass
                    time.sleep(0.5)
                    try:
                        temp_cam.Close()
                    except Exception:
                        pass
                    time.sleep(1.0)
        except Exception:
            pass

        # 更新设备列表并取得设备信息列表
        dev_list = miicam.Miicam.EnumV2()
        if not dev_list or len(dev_list) <= self.cam_id:
            error_info = f"Camera ID {self.cam_id} not found. "
            if dev_list:
                error_info += f" Available cameras: {[_.id for _ in dev_list]}."
            logger.error(error_info)
            raise ConnectionAbortedError(error_info)

        # 按索引打开相机
        self.cam = miicam.Miicam.Open(dev_list[self.cam_id].id)
        if not self.cam:
            raise MIICAMError("Failed to open camera")

        self._init_hardware()
        self._init_streaming()
        self.__update_properties()

        # 相机就绪后再初始化辅助对象
        self._callback_session = _CallbackSession(self)
        self._frame_puller = _FramePuller(self)

    def _init_hardware(self) -> None:
        """设置曝光、增益、位深、像素格式与分辨率。"""
        self._init_exposure()
        self._init_bit_depth()
        self._init_pixel_format()
        self._init_binning()
        self._init_resolution()
        self._detect_raw_format()
        self._init_serial_number()

    def _init_exposure(self) -> None:
        """关闭自动曝光并设置曝光时间与增益。"""
        try:
            self.cam.put_AutoExpoEnable(0)
        except miicam.HRESULTException:
            logger.warning("Could not disable auto exposure during init")

        try:
            self.cam.put_ExpoTime(int(self.exposure_time_ms * 1000))
        except miicam.HRESULTException:
            logger.warning(f"Could not set exposure time to {self.exposure_time_ms}ms")

        try:
            self.cam.put_ExpoAGain(100)
        except miicam.HRESULTException:
            logger.warning("Could not set exposure gain")

    def _init_bit_depth(self) -> None:
        """查询最大位深并配置输出位深。"""
        self._max_bit_depth = self.cam.MaxBitDepth()

        if self._bit_depth == 8:
            output_bitdepth = 0
        elif self._max_bit_depth > 8:
            output_bitdepth = 1
        else:
            output_bitdepth = 0
            logger.warning(
                f"Camera max bit depth is {self._max_bit_depth}, "
                f"requested {self._bit_depth}-bit. Falling back to 8-bit."
            )
            self._bit_depth = 8

        try:
            self.cam.put_Option(miicam.MIICAM_OPTION_BITDEPTH, output_bitdepth)
        except miicam.HRESULTException:
            logger.warning(f"Could not set bit depth to {self._bit_depth}, using 8-bit")
            self._bit_depth = 8

    def _init_pixel_format(self) -> None:
        """按位深设置像素格式。"""
        if self._bit_depth == 8:
            self._pixel_format = "MONO8"
            self._set_mono8_format()
            return

        self._pixel_format = "MONO16"
        if not self._try_set_option(miicam.MIICAM_OPTION_RAW, 1):
            if not self._try_set_option(miicam.MIICAM_OPTION_RGB, 4):
                logger.warning(
                    "Could not set high bit depth mode (RAW or 16-bit Grey), "
                    "falling back to 8-bit"
                )
                self._pixel_format = "MONO8"
                self._bit_depth = 8
                self._set_mono8_format()

    def _set_mono8_format(self) -> None:
        """强制 MONO8 格式并关闭 RAW 模式。"""
        self._try_set_option(miicam.MIICAM_OPTION_RGB, 3)
        self._try_set_option(miicam.MIICAM_OPTION_RAW, 0)

    def _try_set_option(self, option: int, value: int) -> bool:
        """尝试设置某个 SDK 选项, 成功返回 True。"""
        try:
            self.cam.put_Option(option, value)
            return True
        except miicam.HRESULTException:
            return False

    def _init_binning(self) -> None:
        """skip_sampling 为 True 时启用 2x2 binning。"""
        if not self.skip_sampling:
            return
        if not self._try_set_option(miicam.MIICAM_OPTION_BINNING, 0x80 | 2):
            logger.warning(
                "MIICAM_OPTION_BINNING not supported, continuing without binning"
            )

    def _init_resolution(self) -> None:
        """把相机设为最大分辨率并更新尺寸属性。"""
        max_width, max_height = self.cam.get_Resolution(0)
        self.cam.put_Size(max_width, max_height)
        self.cam_width, self.cam_height = self.cam.get_Size()

    def _detect_raw_format(self) -> None:
        """检测实际 raw 格式, 必要时调整像素格式。"""
        raw_fmt, _ = self.cam.get_RawFormat()
        raw_fmt_str = (
            raw_fmt.decode("ascii", errors="replace")
            if isinstance(raw_fmt, bytes)
            else raw_fmt
        )

        if raw_fmt_str in ("YUYV", "VUYY", "UYVY"):
            self._pixel_format = "YUV422"
            logger.info(f"Camera using YUV422 format (raw: {raw_fmt_str})")
        elif raw_fmt_str in ("YMono",):
            logger.info(
                f"Camera using MONO{self._bit_depth} format (raw: {raw_fmt_str})"
            )
        elif raw_fmt_str in ("BGGR", "RGGB", "GRBG", "GBRG"):
            logger.info(
                f"Camera using Bayer {raw_fmt_str} format, treating as MONO{self._bit_depth}"
            )
        else:
            logger.info(
                f"Camera using format: {raw_fmt_str}, treating as MONO{self._bit_depth}"
            )

    def _init_serial_number(self) -> None:
        """查询相机序列号。"""
        try:
            self._sn = self.cam.SerialNumber()
        except Exception:
            self._sn = f"MIICAM_{self.cam_id}"

    def _init_streaming(self) -> None:
        """启动视频流 (不带回调的拉流模式)。"""
        # 预防性地尝试停掉上一次会话遗留的拉流
        try:
            self.cam.Stop()
        except Exception:
            pass
        time.sleep(0.5)

        max_retries = 10
        for attempt in range(max_retries):
            try:
                self.cam.StartPullModeWithCallback(None, None)
                break
            except miicam.HRESULTException:
                if attempt < max_retries - 1:
                    try:
                        self.cam.Stop()
                    except Exception:
                        pass
                    time.sleep(0.5 + 0.5 * attempt)
                else:
                    raise

    # =========================================================================
    # 曝光控制
    # =========================================================================

    def reset_exposure_time(self, time_ms: float) -> float:
        """Set the camera exposure time.

        实验确认 (2026-09): 在视频流运行期间直接调用 ``put_ExpoTime`` **不会生效**——
        抓到的图像仍是旧曝光的停滞/残帧（表现为 max 恒定、图像与历史帧雷同）。
        修改曝光必须按 SDK 要求 **先 Stop 再设置再重启拉流**，并丢弃前 1-2 帧。
        实现与 ``reset_window`` 一致（Stop → put_* → StartPullModeWithCallback）。

        曝光量程经验（MiiCam + 1064nm 激光）:
          - 曝光 <0.1ms 时信号淹没在传感器噪声中（max≤10，无光斑特征），不可用；
          - 建议从 ≥0.2ms 起调节；亮区宽度/均值正确响应曝光，而峰值 (max) 可能被
            锁在 ~85 —— 做质量指标时应优先 mean / 亮区包围盒而非 max。

        参数:
            time_ms: 新的曝光时间, 单位毫秒。
                合法范围: 0.011ms 到 10000ms。超出该范围的值会被钳制。

        返回:
            实际设定的曝光时间, 单位毫秒。
        """
        assert self.cam, "camera not initialized"
        if time_ms < MIICamera.MIN_EXPOSURE_MS:
            self.exposure_time_ms = MIICamera.MIN_EXPOSURE_MS
            logger.warning(
                "exposure time must >= {:.4f}ms. clamped to {:.4f}ms.",
                MIICamera.MIN_EXPOSURE_MS,
                MIICamera.MIN_EXPOSURE_MS,
            )
        elif time_ms > MIICamera.MAX_EXPOSURE_MS:
            self.exposure_time_ms = MIICamera.MAX_EXPOSURE_MS
            logger.warning(
                "exposure time must <= {:.1f}ms. clamped to {:.1f}ms.",
                MIICamera.MAX_EXPOSURE_MS,
                MIICamera.MAX_EXPOSURE_MS,
            )
        else:
            self.exposure_time_ms = time_ms

        # 关键: 流运行中 put_ExpoTime 不生效, 必须先 Stop
        try:
            self.cam.Stop()
        except Exception:
            pass
        time.sleep(0.1)

        self.cam.put_ExpoTime(int(self.exposure_time_ms * 1000))

        # 保持自动曝光关闭 (防止 AGC 覆盖手动曝光)
        try:
            self.cam.put_AutoExpoEnable(0)
        except miicam.HRESULTException:
            pass

        # 重启拉流 (带短暂重试, 仿 _init_streaming)
        for attempt in range(3):
            try:
                self.cam.StartPullModeWithCallback(None, None)
                break
            except miicam.HRESULTException:
                if attempt < 2:
                    try:
                        self.cam.Stop()
                    except Exception:
                        pass
                    time.sleep(0.3 * (attempt + 1))
                else:
                    raise
        # 丢弃切换后的前几帧 (可能仍为旧曝光残留)
        time.sleep(0.15)
        return self.exposure_time_ms

    def auto_exposure(
        self,
        target_max: float = 40.0,
        tolerance: float = 5.0,
        twice_valid: bool = True,
        max_iterations: int = 20,
        n_sample: int = 1,
    ) -> np.ndarray:
        """自动曝光调整 (比例迭代)。

        契约与 :meth:`DahengCamera.auto_exposure` 对齐: ``target_max`` /
        ``tolerance`` 均为 **0-255 灰度**单位, 返回调整后采集到的图像。

        MiiCam 没有可靠的 SDK 自动曝光闭环 (``put_AutoExpoEnable`` 与手动曝光
        相互干扰), 因此这里只做比例迭代: 每次按 ``exp * target / peak`` 调整
        (单步放大上限 3×), 且改曝光必须走 :meth:`reset_exposure_time`
        (Stop → put_ExpoTime → 重启拉流), 故慢于 Daheng。

        经验约束 (见模块文档): 曝光 <0.1ms 时信号淹没在噪声中; 峰值可能被锁在
        ~85, 因此高目标 (如 220) 常常不可达, 此时会以边界曝光退出。

        参数:
            target_max: 目标最大亮度 (0-255, 默认40)。
            tolerance: 峰值容差 (0-255, 默认5)。
            twice_valid: True 时要求连续两次落入容差范围才收敛 (默认True)。
            max_iterations: 最大迭代次数 (默认20)。
            n_sample: 每次估计峰值时的采样帧数 (默认1)。

        返回:
            np.ndarray: 调整后采集到的图像 (uint8/uint16, 取决于位深)。
        """
        assert self.cam, "camera not initialized"
        assert tolerance > 0, "tolerance must be > 0"

        target_val = float(target_max)
        low = int(max(target_val - tolerance, 10))
        high = int(min(target_val + tolerance, 254))
        min_exp = float(self.min_exposure_ms)
        max_exp = float(self.max_exposure_ms)

        logger.info(
            "Auto exposure start: target={:.0f}±{:.0f} (range=[{}, {}]ms, max_iter={})",
            target_val,
            tolerance,
            min_exp,
            max_exp,
            max_iterations,
        )

        twice_ok = False
        img = self.get_numpy_image(n_sample, skip_first=True)

        for i in range(max(1, int(max_iterations))):
            peak = float(np.max(img))

            if low <= peak <= high:
                if twice_ok or not twice_valid:
                    logger.info(
                        "Auto exposure converged at iter {}: exp={:.4f}ms, max={:.1f}",
                        i + 1,
                        self.exposure_time_ms,
                        peak,
                    )
                    return img
                twice_ok = True
            else:
                twice_ok = False
                current = float(self.exposure_time_ms)
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
            "Auto exposure finished: exp={:.4f}ms, max={:.1f}",
            self.exposure_time_ms,
            float(np.max(img)),
        )
        return img

    def enable_auto_exposure(self, enable: bool = True, mode: int = 1) -> bool:
        """启用或关闭自动曝光。

        参数:
            enable: True 启用, False 关闭。
            mode: 自动曝光模式 (0=禁用, 1=连续, 2=单次)。

        返回:
            成功返回 True。
        """
        assert self.cam, "camera not initialized"
        mode_value = 1 if enable else 0
        if enable and mode > 0:
            mode_value = mode
        self.cam.put_AutoExpoEnable(mode_value)
        return True

    def set_auto_exposure_target(self, target: int) -> int:
        """设置自动曝光的目标亮度。

        参数:
            target: 目标亮度值。范围 16-220, 默认 120。

        返回:
            实际设定的目标值。
        """
        assert self.cam, "camera not initialized"
        target = max(16, min(220, target))
        try:
            self.cam.put_AutoExpoTarget(target)
        except miicam.HRESULTException:
            logger.warning("Auto exposure target not supported")
        return target

    def get_auto_exposure_state(self) -> dict:
        """获取当前自动曝光状态。

        返回:
            含 enabled、mode 与 target 的字典。
        """
        assert self.cam, "camera not initialized"
        state = {"enabled": False, "mode": 0, "target": 120}
        try:
            state["mode"] = self.cam.get_AutoExpoEnable()
            state["enabled"] = state["mode"] > 0
            state["target"] = self.cam.get_AutoExpoTarget()
        except miicam.HRESULTException:
            logger.warning("Auto exposure not supported")
        return state

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
            成功返回 True, 不支持返回 False。
        """
        assert self.cam, "camera not initialized"
        try:
            self.cam.put_AutoExpoRange(
                max_time_ms * 1000,
                min_time_ms * 1000,
                max_gain,
                min_gain,
            )
        except miicam.HRESULTException:
            logger.warning("Auto exposure range not supported")
            return False
        return True

    # =========================================================================
    # 窗口 / ROI
    # =========================================================================

    def reset_window(
        self,
        center: tuple[int, int] | tuple[np.intp, ...] = (0, 0),
        size: tuple[int, int] = (0, 0),
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """重设相机窗口的大小与位置。

        参数:
            center: 期望的窗口中心位置 (x, y)。
            size: 期望的窗口大小 (width, height)。用 (0, 0) 表示最大。

        返回:
            实际设定的 ((width, height), (center_x, center_y))。
        """
        assert self.cam, "camera not initialized"
        center = tuple(int(c) for c in center)

        try:
            self.cam.Stop()
        except Exception:
            pass

        if size == (0, 0):
            width, height = self.cam.get_Resolution(0)
            x_offset = 0
            y_offset = 0
        else:
            width, height = int(size[0]), int(size[1])
            x_offset = max(0, center[0] - (width // 2))
            y_offset = max(0, center[1] - (height // 2))

        assert x_offset >= 0 and y_offset >= 0, (
            f"Window center position: {center} must be within image, window size: {size}"
        )

        try:
            self.cam.put_Size(width, height)
        except miicam.HRESULTException:
            width, height = self.cam.get_Resolution(0)
            self.cam.put_Size(width, height)

        self.__update_properties()
        self.cam.StartPullModeWithCallback(None, None)
        return (width, height), (width // 2, height // 2)

    # =========================================================================
    # 图像采集
    # =========================================================================

    def __take_one_shot(self) -> np.ndarray:
        """按配置的采集模式拍摄单张相机图像。"""
        if self._capture_mode == "callback":
            return self.__take_one_shot_callback()
        return self.__take_one_shot_wait()

    def __take_one_shot_wait(self) -> np.ndarray:
        """基于 WaitImageV3 的采集 (阻塞拉取)。"""
        return self._frame_puller.wait_image()

    def __take_one_shot_callback(self) -> np.ndarray:
        """基于回调的采集, 用 PullImageV4 (软件触发模式)。"""
        with self._callback_session as session:
            self.cam.Trigger(1)
            result = self.get_callback_frame(timeout=5.0)
            if result is None:
                raise MIICAMError("Timeout waiting for callback frame after trigger")
            img, _ = result
            return img

    def get_numpy_image(
        self,
        n_sample: int = 1,
        skip_first: bool = True,
    ) -> np.ndarray:
        """获取相机图像数据并做平均。

        参数:
            n_sample: 平均的采样数。必须 > 0。
            skip_first: 是否跳过第一帧 (常常不稳定)。

        返回:
            处理后的平均图像。
        """
        assert n_sample > 0, "Sample count must be > 0"

        first_img = self.__take_one_shot()

        if n_sample == 1:
            return first_img

        numpy_image = np.zeros_like(first_img) if skip_first else first_img.copy()
        _n_sample = n_sample if skip_first else n_sample - 1

        for _ in range(_n_sample):
            numpy_image = numpy_image + self.__take_one_shot()

        avg_img = numpy_image / _n_sample
        return avg_img.astype(np.uint8 if self._bit_depth == 8 else np.uint16)

    # =========================================================================
    # 回调模式 (连续拉流)
    # =========================================================================

    def start_callback_mode(
        self,
        callback: Callable | None = None,
    ) -> None:
        """启动带帧通知回调函数的拉流模式。

        回调运行在 SDK 内部线程里。务必保持轻量 —— 不要在其中做重处理, 也不要
        回头调用 SDK。

        参数:
            callback: 每来一帧新图像就被调用的函数, 签名为 (image, frame_info)。
                      为 None 时使用内部缓冲回调。
        """
        assert self.cam, "camera not initialized"
        self._callback_session.start(callback)
        logger.info("Callback mode started")

    def stop_callback_mode(self) -> None:
        """停止回调模式并释放回调相关资源。"""
        self._callback_session.stop()
        logger.info("Callback mode stopped")

    def _frame_callback(
        self,
        nEvent: int,
        ctx: object,
        user_callback: Callable[[np.ndarray, miicam.MiicamFrameInfoV3], None]
        | None = None,
    ) -> None:
        """SDK 内部回调: 有新帧可用时被调用。"""
        if nEvent == miicam.MIICAM_EVENT_IMAGE:
            try:
                bufsize, bits, dtype = self._get_buffer_params()
                buffer = (ctypes.c_char * bufsize)()
                frame_info = miicam.MiicamFrameInfoV3()
                self._frame_puller.pull_image_v4(buffer, bits, frame_info)
                img_data = np.frombuffer(buffer, dtype=dtype)
                img = self._decode_image(img_data)

                if user_callback is not None:
                    user_callback(img, frame_info)
                else:
                    with self._callback_buffer_lock:
                        self._callback_buffer = img
                        self._callback_frame_info = frame_info
                        self._callback_new_frame.set()
            except Exception:
                pass
        elif nEvent == miicam.MIICAM_EVENT_STILLIMAGE:
            try:
                bufsize, bits, dtype = self._get_buffer_params()
                buffer = (ctypes.c_char * bufsize)()
                frame_info = miicam.MiicamFrameInfoV3()
                self._frame_puller.pull_still_image(buffer, bits, frame_info)
                img_data = np.frombuffer(buffer, dtype=dtype)
                img = self._decode_image(img_data)

                if user_callback is not None:
                    user_callback(img, frame_info)
                else:
                    with self._callback_buffer_lock:
                        self._callback_buffer = img
                        self._callback_frame_info = frame_info
                        self._callback_new_frame.set()
            except Exception:
                pass

    def get_callback_frame(
        self, timeout: float = 1.0
    ) -> tuple[np.ndarray, miicam.MiicamFrameInfoV3] | None:
        """从回调模式取最新一帧。

        参数:
            timeout: 等待新帧的最长时间, 单位秒。

        返回:
            (image, frame_info) 的 tuple; 超时时返回 None。
        """
        if self._callback_new_frame.wait(timeout=timeout):
            with self._callback_buffer_lock:
                self._callback_new_frame.clear()
                return (self._callback_buffer, self._callback_frame_info)
        return None

    # =========================================================================
    # 触发模式 (软件触发)
    # =========================================================================

    def set_trigger_mode(self, mode: int) -> None:
        """设置相机触发模式。

        参数:
            mode: 0 = 视频模式 (默认)
                  1 = 软件 / 仿真触发
                  2 = 外部触发 (上升沿)
                  3 = 外部 + 软件触发
        """
        assert self.cam, "camera not initialized"
        self._stop_streaming()
        self.cam.put_Option(miicam.MIICAM_OPTION_TRIGGER, mode)
        logger.info("Trigger mode set to {}", mode)

    def get_trigger_mode(self) -> int:
        """获取当前触发模式。

        返回:
            当前触发模式 (0=视频, 1=软件, 2=外部, 3=两者)。
        """
        assert self.cam, "camera not initialized"
        return self.cam.get_Option(miicam.MIICAM_OPTION_TRIGGER)

    def trigger(self, n_images: int = 1) -> None:
        """发送一次软件触发。

        调用本方法前, 相机必须处于触发模式 (set_trigger_mode(1)) 且正在拉流
        (StartPullModeWithCallback)。

        参数:
            n_images: 要采集的图像张数。
                      0 = 取消触发, 0xFFFF = 连续。
        """
        assert self.cam, "camera not initialized"
        self.cam.Trigger(n_images)

    def trigger_sync(
        self,
        n_images: int = 1,
        timeout_ms: int = 0,
    ) -> np.ndarray:
        """软件触发并同步等待图像。

        把 Trigger 与 WaitImageV3 合在一次调用里。相机必须处于触发模式且正在拉流。

        参数:
            n_images: 要采集的图像张数 (单次触发填 1)。
            timeout_ms: 超时, 单位 ms。0 = 自适应 (曝光 × 5, 最小 300ms)。

        返回:
            采集到的图像, 为 numpy 数组。
        """
        assert self.cam, "camera not initialized"

        self.cam.Trigger(n_images)

        if timeout_ms == 0:
            timeout_ms = max(300, int(self.exposure_time_ms * 5))

        bufsize, bits, dtype = self._get_buffer_params()
        buffer = (ctypes.c_char * bufsize)()
        frame_info = miicam.MiicamFrameInfoV3()

        max_retries = 2
        for attempt in range(max_retries):
            try:
                self.cam.WaitImageV3(timeout_ms, buffer, 0, bits, 0, frame_info)
                break
            except miicam.HRESULTException as e:
                hr = getattr(e, "hr", 0)
                if hr in (0x8000000A, 0x8001011F) and attempt < max_retries - 1:
                    time.sleep(0.02)
                    continue
                raise MIICAMError(
                    f"trigger_sync timeout: hr=0x{hr & 0xFFFFFFFF:08x}"
                ) from e

        img_data = np.frombuffer(buffer, dtype=dtype)
        return self._decode_image(img_data)

    # =========================================================================
    # 静态图像 (Snap) —— 高分辨率采集
    # =========================================================================

    def snap(self, resolution_index: int = 0xFFFFFFFF) -> None:
        """触发一次静态图像采集 (Snap)。

        相机临时切到指定分辨率, 采集一帧, 再切回预览分辨率。

        参数:
            resolution_index: Snap 使用的分辨率索引。0xFFFFFFFF = 当前
                            预览分辨率。
        """
        assert self.cam, "camera not initialized"
        self.cam.Snap(resolution_index)

    def pull_still_image(self) -> tuple[np.ndarray, miicam.MiicamFrameInfoV3]:
        """在 Snap 事件之后拉取静态图像。

        返回:
            (image, frame_info) 的 tuple。
        """
        assert self.cam, "camera not initialized"

        bufsize, bits, dtype = self._get_buffer_params()
        buffer = (ctypes.c_char * bufsize)()
        frame_info = miicam.MiicamFrameInfoV3()

        self._frame_puller.pull_still_image(buffer, bits, frame_info)

        img_data = np.frombuffer(buffer, dtype=dtype)
        img = self._decode_image(img_data)
        return img, frame_info

    def get_still_resolution_count(self) -> int:
        """获取可用静态图像分辨率的数量。"""
        assert self.cam, "camera not initialized"
        return self.cam.StillResolutionNumber()

    def get_still_resolution(self, index: int) -> tuple[int, int]:
        """获取某个静态图像分辨率的尺寸。"""
        assert self.cam, "camera not initialized"
        return self.cam.get_StillResolution(index)

    # =========================================================================
    # 拉流控制
    # =========================================================================

    def pause(self, b_pause: bool = True) -> None:
        """暂停或恢复视频流。"""
        assert self.cam, "camera not initialized"
        self.cam.Pause(1 if b_pause else 0)

    def _stop_streaming(self) -> None:
        """停止拉流 (内部辅助)。"""
        if self.cam:
            try:
                self.cam.Stop()
            except Exception:
                pass
            time.sleep(0.3)
        self._callback_mode_active = False

    # =========================================================================
    # 内部辅助
    # =========================================================================

    def _get_buffer_params(self) -> tuple[int, int, type]:
        """按当前格式取缓冲区大小、位数与 numpy dtype。"""
        if getattr(self, "_pixel_format", None) == "YUV422":
            return (self.cam_width * self.cam_height * 2, 8, np.uint8)
        elif self._bit_depth == 8:
            return (self.cam_width * self.cam_height, 8, np.uint8)
        else:
            return (self.cam_width * self.cam_height * 2, 16, np.uint16)

    def _decode_image(self, img_data: np.ndarray) -> np.ndarray:
        """按像素格式解码原始图像数据。"""
        if getattr(self, "_pixel_format", None) == "YUV422":
            img_yuv = img_data.reshape((self.cam_height, self.cam_width * 2))
            return img_yuv[:, ::2]
        return img_data.reshape((self.cam_height, self.cam_width))

    def __update_properties(self):
        assert self.cam, "camera not initialized"
        self.cam_width, self.cam_height = self.cam.get_Size()
        logger.info(
            f"Open cam {self._sn} success. width={self.cam_width}, height={self.cam_height}"
        )
        self.xv, self.yv = self.__get_grid(self.cam_width, self.cam_height)

    @staticmethod
    def __get_grid(width: int, height: int):
        x = np.arange(0, width)
        y = np.arange(0, height)
        xv, yv = np.meshgrid(x, y)
        return xv, yv

    @staticmethod
    def get_cam_list():
        """获取可用相机列表。"""
        return miicam.Miicam.EnumV2()
