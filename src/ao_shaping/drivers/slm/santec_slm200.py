"""Santec SLM-200 空间光调制器驱动模块

提供对Santec SLM-200系列空间光调制器的控制接口，
支持相位图显示、波长设置、内存模式等功能。

Agent Wiki: docs/slm-200/agent_wiki.md
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from retrying import retry
from scipy import ndimage

from ao_shaping.drivers.slm.santec_slm200_constants import (
    FLAGS_RATE120,
    GRAY_SCALE_BITS,
    GRAYSCALE_MAX,
    GRAYSCALE_MIN,
    MAX_MEM_SLOTS,
    MAX_PIXEL_FLIP_TIME_MS,
    MEMORY_MODE_INTERNAL,
    MEMORY_NUMBER_MAX,
    MEMORY_NUMBER_MIN,
    PANEL_RES,
    PANEL_SIZE_MM,
    PITCH_UM,
    PIXEL_SIZE_UM,
    RESPONSE_TIME_MS,
    SLM_OK,
    WAVELENGTH_MAX,
    WAVELENGTH_MIN,
    VideoMode,
    get_max_grayscale,
    get_slm_error_message,
)
from ao_shaping.drivers.slm.wavefront_correction import WavefrontCorrection
from ao_shaping.utils.device_config import ConfigHandler, DeviceParam, param
from ao_shaping.utils.file import ROOT_DIR as PROJECT_ROOT

# LCOS 像素翻转时序常量
# 0→2π（满相位量程，_max_gray 灰度）翻转耗时至多 200ms（厂商规格）。
# display_data 的自动等待按最大灰度变化 / 满量程 正比线性估算。
MAX_PIXEL_FLIP_TIME_S = MAX_PIXEL_FLIP_TIME_MS / 1000.0


def apply_lut_remap(gray: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Apply phase→gray compensation lookup table to a grayscale array.

    Maps each pixel's ideal gray value (0..1023) through the inverse_gray LUT
    to obtain the corrected gray value that produces the intended phase.

    The LUT index uses the same truncation as the final ``astype(np.uint16)``
    cast in ``create_phase_from_array``, so an identity LUT is a true no-op.

    Args:
        gray: Grayscale array (float or int) with values in 0..1023.
        lut: Inverse gray table (uint16, length ≥ 1024).

    Returns:
        Corrected grayscale array with the same shape as *gray*, dtype float64.
    """
    idx = np.clip(gray.astype(np.int64), 0, lut.size - 1)
    return lut[idx].astype(np.float64)

# Config directory: <project_root>/data/slm_configs/ or from SLM_CONFIG_DIR env var
_SLM_CONFIG_DIR = Path(
    os.environ.get("SLM_CONFIG_DIR", PROJECT_ROOT / "data" / "slm_configs")
)


# ── SLM 配置参数 dataclass ──────────────────────────────


def _to_int_or_none(v: Any) -> int | None:
    """Convert value to int, returning None if None or empty."""
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


@dataclass
class SLMParams(DeviceParam):
    """SLM 设备参数（加载/检查/应用均基于此 dataclass）。

    字段定义同时充当:
      - :attr:`config_key` — JSON 配置文件的 key
      - :attr:`attr` — 设备实例的属性名
      - :attr:`cast` — 从 JSON 读入时的类型转换
    """

    wavelength: int | None = param(default=None, cast=_to_int_or_none)
    shift_x: int = param(default=0, attr="_shift_x")
    shift_y: int = param(default=0, attr="_shift_y")
    use_120hz: bool = param(default=False, cast=bool, attr="_use_120hz")


# 模块级单例，所有 SantecSLM200 实例共用
SLM_CONFIG = ConfigHandler(_SLM_CONFIG_DIR, "slm", SLMParams)


class SantecSLM200Error(Exception):
    """Santec SLM-200 驱动错误

    包含错误码和可读错误消息，便于调试。

    Attributes:
        code: SLM SDK返回的错误码
        message: 人类可读的错误消息
    """

    def __init__(self, message: str = "", code: int | None = None):
        self.code = code
        if code is not None:
            error_detail = get_slm_error_message(code)
            full_message = (
                f"{message} (错误码: {code}, {error_detail})"
                if message
                else f"错误码: {code}, {error_detail}"
            )
            super().__init__(full_message)
        else:
            super().__init__(message)


def _is_retryable_slant_error(exception) -> bool:
    """Determine if an exception is retryable for SLM operations.

    Retry on SantecSLM200Error as these often indicate transient USB issues
    that may succeed on retry.
    """
    return isinstance(exception, SantecSLM200Error)


class SantecSLM200:
    """Santec SLM-200 空间光调制器驱动类

    提供对SLM-200系列空间光调制器的完整控制，包括：
    - 设备连接与断开
    - 波长和相位范围设置
    - 相位图显示（内存模式）
    - 灰度级控制

    Attributes:
        slm_number: SLM设备编号（1-8）
        wavelength: 当前工作波长（nm）
        is_open: 设备是否已连接

    Example:
        >>> with SantecSLM200(slm_number=1) as slm:
        ...     slm.set_wavelength(1064)  # 1064nm, 0~2π相位
        ...     phase_data = np.zeros((1080, 1920), dtype=np.uint16)
        ...     slm.write_phase(phase_data, memory_number=1)
        ...     slm.display_memory(1)

    硬件规格与实验经验（2026-09 实测确认）:
      - 面板: 1920×1200, 像素间距 8µm（PITCH_UM；制造商标称 10µm 方格含电极间隙，
        实际可调制有效口径 8µm）。GS/相位网格以 PITCH_UM=8µm 作为 SLM 平面采样间距。
      - 像素翻转时序: LCOS 像素 0→2π 满量程翻转耗时至多 200ms
        （``MAX_PIXEL_FLIP_TIME_MS``）。``display_data()`` 在等待时间未指定或为负时，
        按相邻两帧最大灰度变化相对满量程的正比自动等待（首帧保守取最大值）。
      - 内存模式（Memory）: 对同一 slot 连续 ``display_memory`` 是 no-op —— 当目标 slot
        已在显示时设备不刷新 LCOS 面板，屏幕保留上一相位。连续写入必须轮换不同 slot
        （如 ``itertools.cycle([3,4,5])``），或直接使用 ``display_data()``（内置 127 槽轮换）。
      - 平场相位（及任何直接灰度图案）必须以原值 uint16 灰度写入
        （``np.full((h, w), gray, dtype=np.uint16)``），禁止经 ``create_phase_from_array()``
        —— 该函数按弧度处理输入（mod 2π），会把 uint16 灰度值静默损坏。
      - 振幅耦合: 1064nm 下不同平场灰度值产生不同相机亮度，周期 ≈2π ≈993 灰度
        （见 ``scripts/validate_flat_phase_gray.py``）。
      - 像素标定（GS 方形整形）: SLM→相机物理距离约 1000mm，但实测
        k = SLM 网格边长 / 相机亮区宽度 ≈ 0.414（每 SLM 8µm 像素 ≈ 2.417 相机像素，
        等效成像距离 ≈115mm）—— 光路中存在成像元件或非标称傅里叶配置。光路调整后
        必须重新标定（``gs-square --pixel-scale`` 自动标定模式）。
    """

    # 从常量模块导入硬件参数
    Pixel_Size_um = PIXEL_SIZE_UM
    Pitch_um = PITCH_UM
    Panel_Size_mm = PANEL_SIZE_MM
    Panel_Res = PANEL_RES
    Response_time_ms = RESPONSE_TIME_MS
    Gray_Scale_bits = GRAY_SCALE_BITS
    MAX_GRAYSCALE_VALUE = get_max_grayscale()
    MAX_PIXEL_FLIP_TIME_MS = MAX_PIXEL_FLIP_TIME_MS  # 像素全量翻转(0→2π)最大耗时 (ms)
    MAX_PIXEL_FLIP_TIME_S = MAX_PIXEL_FLIP_TIME_S  # 同上，单位秒

    def __init__(
        self,
        slm_number: int = 1,
        use_120hz: bool = False,
        wavelength: int | None = None,
        video_mode: int | VideoMode = VideoMode.Memory,
        shift_x: int | None = None,
        shift_y: int | None = None,
        correction_csv_path: str | Path | None = None,
    ):
        """初始化SLM驱动

        Args:
            slm_number: SLM设备编号（1-8），默认为1
            use_120hz: 是否使用120Hz刷新率，默认为False
            wavelength: 工作波长（nm），默认为None；
                设为None则在open()时从配置文件或设备读取
            video_mode: 视频模式 (0=内存模式, 1=DVI模式)，默认为0
            shift_x: X方向平移像素数（正=右，负=左），设为None从配置文件加载
            shift_y: Y方向平移像素数（正=下，负=上），设为None从配置文件加载
            correction_csv_path: 误差矫正CSV文件路径，默认为None；
                格式如 libs/SLM_DLL_ver.2.51/Wavefront_correction_Data/
                Wavefront_correction_Data_240236000006(520nm).csv；
                若提供则在create_phase_from_array时自动叠加矫正
        """
        # 参数验证仅在设备控制函数中进行，此处不重复验证
        self.slm_number = slm_number
        self._use_120hz = use_120hz
        self.flags = FLAGS_RATE120 if use_120hz else 0
        self.video_mode = video_mode if isinstance(video_mode, int) else int(video_mode)
        self.is_open = False
        self._max_gray = self.MAX_GRAYSCALE_VALUE
        self._memory_phase_cache: dict[int, np.ndarray] = {}
        self._displayed_memory_number: int | None = None
        self._displayed_phase_cache: np.ndarray | None = None

        # 保存init参数，用于open()中优先级判断
        # 与 ConfigHandler 兼容的字典形式
        self._init_values: dict[str, Any] = {
            "wavelength": wavelength,
            "_shift_x": shift_x,
            "_shift_y": shift_y,
            "_use_120hz": use_120hz,
        }

        # 实际运行时值（立即生效，供属性和测试使用）
        self.wavelength: int | None = wavelength
        self._shift_x: int = shift_x if shift_x is not None else 0
        self._shift_y: int = shift_y if shift_y is not None else 0

        self._current_memory_slot = 1

        # 设备序列号（open后获取）
        self._serial_number: str | None = None

        # 波前误差矫正工具（从CSV加载，在create_phase_from_array中叠加）
        # 文件有效性由 WavefrontCorrection.__init__ 内部判断
        self._correction = WavefrontCorrection(correction_csv_path)

        # 相位→灰度补偿查找表（load_lut 加载，create_phase_from_array 中应用）
        self._lut: np.ndarray | None = None
        self._lut_dir: Path | None = None

        # 延迟导入SLM SDK
        try:
            import ao_shaping.drivers.slm._slm_win as slm_sdk

            self._slm = slm_sdk
        except ImportError as e:
            raise SantecSLM200Error(
                f"无法导入SLM SDK (_slm_win): {e}. 请确保已安装Santec SLM驱动程序。"
            ) from e

    def get_serial_number(self, timeout: float = 0.0) -> str | None:
        """读取SLM设备标识。

        参考官方文档与 C++ 样例，依次尝试：
        1. SLM_Ctrl_ReadSDO 读取 Drive board / Option board ID
        2. SLM_Ctrl_ReadSD 读取 Drive board ID
        3. SLM_Ctrl_ReadSO 读取 Option board ID

        Args:
            timeout: 已弃用（保留参数兼容旧调用）。

        Returns:
            设备标识字符串，失败时返回 None
        """
        self._ensure_open()

        drive_id = ctypes.create_string_buffer(16)
        option_id = ctypes.create_string_buffer(16)

        ret = self._slm.SLM_Ctrl_ReadSDO(self.slm_number, drive_id, option_id)
        if ret == SLM_OK:
            serial = drive_id.value.decode("utf-8").strip()
            if serial:
                logger.debug(f"SLM #{self.slm_number} driveboardID: {serial}")
                return serial
            serial = option_id.value.decode("utf-8").strip()
            if serial:
                logger.debug(f"SLM #{self.slm_number} optionboardID: {serial}")
                return serial

        device_id = ctypes.create_string_buffer(16)
        ret = self._slm.SLM_Ctrl_ReadSD(self.slm_number, device_id)
        if ret == SLM_OK:
            serial = device_id.value.decode("utf-8").strip()
            if serial:
                logger.debug(f"SLM #{self.slm_number} driveboardID(ReadSD): {serial}")
                return serial

        option_only = ctypes.create_string_buffer(16)
        ret = self._slm.SLM_Ctrl_ReadSO(self.slm_number, option_only)
        if ret == SLM_OK:
            serial = option_only.value.decode("utf-8").strip()
            if serial:
                logger.debug(f"SLM #{self.slm_number} optionboardID(ReadSO): {serial}")
                return serial

        logger.warning(f"SLM #{self.slm_number} 无法读取设备标识")
        return None

    def get_product_serial_number(self, board: int = 0) -> str | None:
        """读取产品序列号（标签上的 12 位数字）。

        Args:
            board: 0=Drive board, 1=Option board

        Returns:
            12 位产品序列号字符串，失败时返回 None
        """
        self._ensure_open()
        buf = ctypes.create_string_buffer(16)
        ret = self._slm.SLM_Ctrl_ReadPS(self.slm_number, board, buf)
        if ret == SLM_OK:
            serial = buf.value.decode("utf-8").strip()
            if serial:
                return serial
        return None

    def get_lcos_serial_number(self, board: int = 0) -> str | None:
        """读取 LCOS 产品序列号（最长 20 位数字）。

        Args:
            board: 0=Drive board, 1=Option board

        Returns:
            LCOS 序列号字符串，失败时返回 None
        """
        self._ensure_open()
        buf = ctypes.create_string_buffer(32)
        ret = self._slm.SLM_Ctrl_ReadLS(self.slm_number, board, buf)
        if ret == SLM_OK:
            serial = buf.value.decode("utf-8").strip()
            if serial:
                return serial
        return None

    def get_display_name(self) -> str | None:
        """读取 Display Name（EDID 信息，最长 13 位）。

        Returns:
            Display Name 字符串，失败时返回 None
        """
        self._ensure_open()
        buf = ctypes.create_string_buffer(16)
        ret = self._slm.SLM_Ctrl_ReadPN(self.slm_number, buf)
        if ret == SLM_OK:
            name = buf.value.decode("utf-8").strip()
            if name:
                return name
        return None

    def get_version(self) -> str | None:
        """读取版本信息。

        Returns:
            版本字符串，格式如 "DLL:2.5.0,Drive:0322,Option:0321,FPGA:0110"
        """
        self._ensure_open()
        buf = ctypes.create_string_buffer(64)
        ret = self._slm.SLM_Ctrl_ReadVR(self.slm_number, buf)
        if ret == SLM_OK:
            version = buf.value.decode("utf-8").strip()
            if version:
                return version
        return None

    def get_device_info(self) -> dict[str, Any]:
        """收集所有可用的设备标识信息。

        Returns:
            包含 driveboard_id, optionboard_id, product_serial,
            lcos_serial, display_name, version 的字典
        """
        info: dict[str, Any] = {}
        try:
            info["driveboard_id"] = self.get_serial_number()
        except Exception as e:
            logger.debug(f"读取 driveboard_id 失败: {e}")

        try:
            info["optionboard_id"] = self._read_option_id()
        except Exception as e:
            logger.debug(f"读取 optionboard_id 失败: {e}")

        try:
            info["product_serial"] = self.get_product_serial_number(0)
            if not info["product_serial"]:
                info["product_serial"] = self.get_product_serial_number(1)
        except Exception as e:
            logger.debug(f"读取 product_serial 失败: {e}")

        try:
            info["lcos_serial"] = self.get_lcos_serial_number(0)
            if not info["lcos_serial"]:
                info["lcos_serial"] = self.get_lcos_serial_number(1)
        except Exception as e:
            logger.debug(f"读取 lcos_serial 失败: {e}")

        try:
            info["display_name"] = self.get_display_name()
        except Exception as e:
            logger.debug(f"读取 display_name 失败: {e}")

        try:
            info["version"] = self.get_version()
        except Exception as e:
            logger.debug(f"读取 version 失败: {e}")

        return info

    def _read_option_id(self) -> str | None:
        """读取 Option board ID（内部辅助方法）。"""
        buf = ctypes.create_string_buffer(16)
        ret = self._slm.SLM_Ctrl_ReadSO(self.slm_number, buf)
        if ret == SLM_OK:
            return buf.value.decode("utf-8").strip() or None
        return None

    def reboot(self) -> None:
        """重新启动SLM设备。

        调用SDK的SLM_Ctrl_Reboot函数复位设备USB/控制器状态。
        当设备因快速连续open/close等操作导致SLM_Ctrl_ReadSD挂起时，
        调用此方法可使设备恢复正常通信。

        Raises:
            SantecSLM200Error: 重启失败
        """
        ret = self._slm.SLM_Ctrl_Reboot(self.slm_number)
        if ret != SLM_OK:
            raise SantecSLM200Error(f"SLM #{self.slm_number} 重启失败", code=ret)
        logger.info(f"SLM #{self.slm_number} 已重启")
        # 重启后设备需要重新打开
        self.is_open = False
        # Allow device time to stabilize after reboot
        time.sleep(0.5)

    def load_config(self) -> dict:
        """加载当前设备的配置文件

        Returns:
            配置字典；无序列号或文件不存在时返回空字典
        """
        if not self._serial_number:
            return {}
        return SLM_CONFIG._manager.load_config(self._serial_number)

    def save_config(self) -> None:
        """将当前参数保存到JSON配置文件

        先加载已有配置（保留额外字段如 correction_csv_path），
        再更新 SLMParams 字段 + max_gray + video_mode。
        同时记录矫正文件的启用状态与当前生效路径，使下次启动时
        保持与上次一致的矫正行为。
        """
        serial = self._serial_number
        if not serial:
            logger.warning("未获取到序列号，跳过配置保存")
            return

        # 从现有配置文件加载（保留 correction_csv_path 等额外字段）
        config = SLM_CONFIG._manager.load_config(serial)
        # 覆盖 SLMParams 已注册字段
        config.update(SLM_CONFIG.collect(self))
        # 覆盖额外字段
        config["max_gray"] = self._max_gray
        config["video_mode"] = self.video_mode
        # 记录矫正文件启用状态与当前生效路径，
        # 使下次 open() 时保持与上次一致的矫正行为。
        config["correction_enabled"] = self._correction.is_valid
        config["correction_csv_path"] = (
            str(self._correction.csv_path)
            if self._correction.is_valid and self._correction.csv_path is not None
            else None
        )

        SLM_CONFIG._manager.save_config(serial, config)
        config_file = SLM_CONFIG._manager._get_config_file(serial)
        logger.info(f"SLM配置已保存: {config_file}")

    # ── open() 拆分子方法 ────────────────────────────

    def _apply_config_params(self, config: dict) -> None:
        """应用配置参数（open() 子步骤，委托至 ConfigHandler）

        参数优先级（由 ConfigHandler.resolve 保证）:
          1. __init__ 显式参数 (_init_values)
          2. JSON 配置文件 (序列号匹配)
          3. SLMParams 字段默认值

        Args:
            config: 从 load_config() 获取的配置字典
        """
        params = SLM_CONFIG.apply_from_config(
            self,
            config,
            init_values=self._init_values,
        )
        self.flags = FLAGS_RATE120 if self._use_120hz else 0

        logger.info(
            f"SLM #{self.slm_number} 参数: "
            f"wavelength={params.wavelength}, "
            f"shift_x={params.shift_x}, shift_y={params.shift_y}, "
            f"use_120hz={params.use_120hz}"
        )

    def _load_correction(self, config: dict) -> None:
        """加载波前误差矫正数据（委托至 WavefrontCorrection.resolve）

        优先级（由工具类 resolve() 处理）:
          __init__ 显式路径 → 配置文件路径 → 默认路径

        若配置文件记录了 correction_enabled=False，则跳过加载，
        保持矫正禁用状态（与上次关闭时一致）。

        Args:
            config: 从 load_config() 获取的配置字典
        """
        # 配置文件中记录的矫正启用状态（默认 True，向后兼容）
        correction_enabled = config.get("correction_enabled", True)
        if not correction_enabled:
            self._correction = WavefrontCorrection()  # 空实例（is_valid=False）
            logger.info(f"SLM #{self.slm_number} 矫正已禁用（配置文件记录）")
            return

        default_path = (
            PROJECT_ROOT
            / "data"
            / "calibration"
            / "Wavefront_correction_Data_240236000006(520nm).csv"
        )
        resolved = WavefrontCorrection.resolve(
            explicit_path=self._correction.csv_path
            if self._correction.is_valid
            else None,
            config=config,
            panel_resolution=self.Panel_Res,
            default_path=default_path,
        )
        if resolved is not None:
            self._correction = resolved
            logger.info(
                f"SLM #{self.slm_number} 已加载矫正数据: {resolved.csv_path.name}"
            )
        else:
            self._correction = WavefrontCorrection()  # 空实例（is_valid=False）
            logger.info(f"SLM #{self.slm_number} 未加载矫正数据")

    def _setup_wavelength(self) -> None:
        """设置工作波长（open() 子步骤）"""
        if self.wavelength is None:
            self.wavelength, self._max_gray = self.get_wavelength_info()
        else:
            device_wavelength, device_max_gray = self.get_wavelength_info()
            if device_wavelength == self.wavelength:
                logger.info(
                    f"SLM #{self.slm_number} 波长与设备当前值相同，跳过设置 "
                    f"(wavelength={self.wavelength})"
                )
                self._max_gray = device_max_gray
            else:
                self.set_wavelength(self.wavelength, save_to_device=True)

    # ── open / close ──────────────────────────────────

    def open(self) -> None:
        """打开SLM设备连接

        建立与SLM控制器的通信连接。在调用其他方法前必须先调用此方法。
        打开设备后会自动读取序列号并按优先级加载配置：
          1. __init__() 显式传入的参数（最高优先级）
          2. JSON配置文件（按序列号匹配，如有）
          3. 设备读取的默认值或代码默认值（最低）

        Raises:
            SantecSLM200Error: 设备连接失败
        """
        if self.is_open:
            # SDK may be in inconsistent state - try to verify and recover
            try:
                self._check_status()
                logger.warning(f"SLM #{self.slm_number} 已经处于打开状态")
                return
            except SantecSLM200Error:
                # Device in bad state, force reset
                logger.warning(f"SLM #{self.slm_number} 状态异常，尝试复位")
                self.is_open = False
                # Try to close and recover
                with contextlib.suppress(Exception):
                    self._slm.SLM_Ctrl_Close(self.slm_number)
                time.sleep(0.2)

        # 先尝试关闭（确保干净状态）
        with contextlib.suppress(Exception):
            self._slm.SLM_Ctrl_Close(self.slm_number)
        # Small delay for SDK state to settle
        time.sleep(0.1)

        # 打开设备
        ret = self._slm.SLM_Ctrl_Open(self.slm_number)
        if ret != SLM_OK:
            raise SantecSLM200Error(f"无法打开SLM #{self.slm_number}", code=ret)

        self.is_open = True
        logger.info(f"成功打开SLM #{self.slm_number}")

        # 等待设备就绪（参考官方demo，应在读序列号前进行）
        self._wait_for_ready()

        # 读取设备标识信息（参考官方文档 3.2.41-3.2.48）
        try:
            self._serial_number = self.get_serial_number()
            logger.info(f"SLM #{self.slm_number} 标识: {self._serial_number}")
        except SantecSLM200Error as e:
            logger.warning(f"无法读取SLM标识: {e}")
            self._serial_number = None

        # 如果 board ID 为空，尝试读取产品序列号作为备用标识
        if not self._serial_number:
            try:
                self._serial_number = self.get_product_serial_number(0)
                if self._serial_number:
                    logger.info(
                        f"SLM #{self.slm_number} 产品序列号(drive):"
                        f" {self._serial_number}"
                    )
            except Exception as e:
                logger.debug(f"读取产品序列号(drive)失败: {e}")

        if not self._serial_number:
            try:
                self._serial_number = self.get_product_serial_number(1)
                if self._serial_number:
                    logger.info(
                        f"SLM #{self.slm_number} 产品序列号(option):"
                        f" {self._serial_number}"
                    )
            except Exception as e:
                logger.debug(f"读取产品序列号(option)失败: {e}")

        # 按标识加载配置文件
        config: dict = self.load_config()

        # 应用配置参数（波长、平移、刷新率等）
        self._apply_config_params(config)

        # 加载波前误差矫正数据
        self._load_correction(config)

        # 设置波长（如需要）
        self._setup_wavelength()

        # 设置内存模式
        try:
            self._set_memory_mode(self.video_mode)
        except SantecSLM200Error as e:
            self.is_open = False
            raise SantecSLM200Error(f"设置内存模式失败: {e}") from e

    def close(self) -> None:
        """关闭SLM设备连接

        断开与SLM控制器的通信连接，释放资源。
        """
        if not self.is_open:
            return

        # 保存当前配置
        self.save_config()

        ret = self._slm.SLM_Ctrl_Close(self.slm_number)
        if ret == SLM_OK:
            logger.info(f"成功关闭SLM #{self.slm_number}")
        else:
            logger.warning(f"关闭SLM #{self.slm_number}时返回错误码: {ret}")

        self.is_open = False

    def _check_status(self) -> bool:
        """检查SLM设备状态

        Raises:
            SantecSLM200Error: 设备状态异常
        """
        ret = self._slm.SLM_Ctrl_ReadSU(self.slm_number)
        if ret != SLM_OK:
            raise SantecSLM200Error(f"SLM #{self.slm_number} 状态异常", code=ret)
        logger.debug(f"SLM #{self.slm_number} 状态正常")
        return True

    def _wait_for_ready(self, max_retries: int = 10, retry_delay: float = 0.1) -> bool:
        """等待设备就绪（参考官方demo）

        Args:
            max_retries: 最大重试次数，默认10次 (~1s)
            retry_delay: 重试间隔秒数，默认0.1秒

        Returns:
            True if device becomes ready

        Raises:
            SantecSLM200Error: if device never becomes ready
        """
        for attempt in range(max_retries):
            try:
                self._check_status()
                if attempt > 0:
                    logger.info(f"SLM #{self.slm_number} 在 {attempt + 1} 次尝试后就绪")
                return True
            except SantecSLM200Error:
                if attempt < max_retries - 1:
                    logger.debug(f"等待设备就绪... ({attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    logger.error(
                        f"SLM #{self.slm_number} 在 {max_retries} 次尝试后仍未就绪"
                    )
                    raise
        return False

    def _set_memory_mode(self, mode: int | VideoMode) -> None:
        """设置SLM工作模式

        Args:
            mode: 内存模式 (0=内部内存, 1=DVI)，支持int或VideoMode枚举

        Raises:
            SantecSLM200Error: 模式设置失败
        """
        # 转换为int以兼容SDK
        mode_int = int(mode)

        ret = self._slm.SLM_Ctrl_WriteVI(self.slm_number, mode_int)
        if ret != SLM_OK:
            raise SantecSLM200Error("设置内存模式失败", code=ret)

        # 验证设置
        dat32 = ctypes.c_uint32(0)
        self._slm.SLM_Ctrl_ReadVI(self.slm_number, dat32)
        if dat32.value != mode_int:
            raise SantecSLM200Error("内存模式设置验证失败")

        mode_str = "内部内存" if mode_int == MEMORY_MODE_INTERNAL else "DVI"
        logger.info(f"SLM #{self.slm_number} 已设置为{mode_str}模式")

    def set_wavelength(self, wavelength: int, save_to_device: bool = True) -> None:
        """设置SLM工作波长

        设置SLM的工作波长（固定使用0~2π相位范围）。此操作可能需要约40秒完成。
        通常只需在首次使用或更换波长时调用一次。

        Args:
            wavelength: 工作波长（nm），例如 1064，范围 450-1600
            save_to_device: 是否保存到SLM控制器，默认为True

        Raises:
            SantecSLM200Error: 设置失败或参数无效
            RuntimeError: 设备未打开
        """
        self._ensure_open()
        wavelength = int(wavelength)
        # 在设备控制函数中进行参数验证
        if not WAVELENGTH_MIN <= wavelength <= WAVELENGTH_MAX:
            raise SantecSLM200Error(
                f"波长必须在{WAVELENGTH_MIN}-{WAVELENGTH_MAX}nm之间，当前: {wavelength}"
            )

        logger.info(f"设置波长 {wavelength}nm (0~2π相位范围)...")

        # 固定使用 2π 相位范围 (200 = 2*pi)
        phase_range = 200

        # 设置波长和相位范围
        res = self._slm.SLM_Ctrl_WriteWL(self.slm_number, wavelength, phase_range)
        if res != SLM_OK:
            raise SantecSLM200Error(
                f"设置波长/相位范围失败，输入波长为{wavelength}", code=res
            )

        # 保存到设备
        if save_to_device:
            ret = self._slm.SLM_Ctrl_WriteAW(self.slm_number)
            if ret != SLM_OK:
                raise SantecSLM200Error("保存波长设置失败", code=ret)

        self.wavelength = wavelength
        self.get_wavelength_info()
        logger.info(
            f"波长设置完成: {wavelength}nm, "
            f"相位范围: 0~2π, "
            f"2π对应灰度值: {self._max_gray}"
        )

    def get_wavelength_info(self) -> tuple[int, int]:
        """获取当前波长设置信息

        Returns:
            Tuple of (wavelength_nm, max_grayscale_for_2pi)

        Raises:
            SantecSLM200Error: 读取失败
            RuntimeError: 设备未打开
        """
        self._ensure_open()

        wavelength = ctypes.c_uint32(0)
        phase = ctypes.c_uint32(0)

        res = self._slm.SLM_Ctrl_ReadWL(
            self.slm_number, ctypes.byref(wavelength), ctypes.byref(phase)
        )
        if res != SLM_OK:
            raise SantecSLM200Error("读取波长信息失败", code=res)

        wavelength = int(wavelength.value)
        phase_pi = phase.value / 100.0
        if phase_pi <= 0:
            logger.warning(
                f"SLM #{self.slm_number}: SLM_Ctrl_ReadWL returned phase=0;"
                " device wavelength not set"
            )
            self._max_gray = self.MAX_GRAYSCALE_VALUE
            return (
                self.wavelength if self.wavelength is not None else 1064,
                self._max_gray,
            )
        raw = int(2.0 / phase_pi * self.MAX_GRAYSCALE_VALUE)
        self._max_gray = max(1, min(raw, self.MAX_GRAYSCALE_VALUE))
        if raw != self._max_gray:
            logger.warning(
                f"SLM #{self.slm_number}: 计算2π灰度值 {raw} 超出硬件范围 "
                f"[1, {self.MAX_GRAYSCALE_VALUE}]，已钳位至 {self._max_gray}"
            )
            logger.info(f"当前波长: {wavelength}nm, 2π对应灰度值: {self._max_gray}")
        else:
            logger.debug(f"当前波长: {wavelength}nm, 2π对应灰度值: {self._max_gray}")

        return wavelength, self._max_gray

    def calibrate_wavelength(
        self, wavelength: int, save_to_device: bool = False
    ) -> tuple[int, int]:
        """自校正波长，正确计算 2π 对应的灰度值并更新内部状态。

        将设备设为指定波长 + 2π 相位范围，从设备读取实际相位响应并
        计算校正后的 2π 灰度值（即 _max_gray）。

        与 :meth:`set_wavelength` 的区别:
          - 默认**不写 EEPROM** (``save_to_device=False``)，
            避免不必要的 ~40 秒 EEPROM 写入
          - 返回校正后的 ``(wavelength_nm, max_grayscale_for_2pi)`` 元组

        Args:
            wavelength: 工作波长（nm），取值范围 450-1600。
            save_to_device: 是否将波长设置保存到 SLM 控制器 EEPROM，
                默认 ``False``。

        Returns:
            校正后的 ``(wavelength_nm, max_grayscale_for_2pi)`` 元组。

        Raises:
            SantecSLM200Error: 自校正失败（SDK 通信错误、参数无效等）。
            RuntimeError: 设备未打开。
        """
        self._ensure_open()
        wavelength = int(wavelength)
        if not WAVELENGTH_MIN <= wavelength <= WAVELENGTH_MAX:
            raise SantecSLM200Error(
                f"波长必须在 {WAVELENGTH_MIN}-{WAVELENGTH_MAX}nm 之间，"
                f"当前: {wavelength}"
            )

        logger.info(f"波长自校正: {wavelength}nm (0~2π 相位范围)...")

        # 写入波长 + 2π 相位范围到设备 RAM（200 = 2*pi）
        phase_range = 200
        res = self._slm.SLM_Ctrl_WriteWL(self.slm_number, wavelength, phase_range)
        if res != SLM_OK:
            raise SantecSLM200Error(
                f"自校正失败: SLM_Ctrl_WriteWL 返回错误 (wavelength={wavelength})",
                code=res,
            )

        # 可选：保存到 EEPROM
        if save_to_device:
            ret = self._slm.SLM_Ctrl_WriteAW(self.slm_number)
            if ret != SLM_OK:
                raise SantecSLM200Error(
                    "自校正失败: SLM_Ctrl_WriteAW (EEPROM) 返回错误", code=ret
                )

        # 读取设备实际值，计算并校正 _max_gray
        self.wavelength = wavelength
        wavelength_nm, max_gray = self.get_wavelength_info()

        logger.info(f"波长自校正完成: {wavelength_nm}nm, 2π 对应灰度值: {max_gray}")

        return wavelength_nm, max_gray

    def write_phase(
        self,
        phase: np.ndarray,
        memory_number: int = 1,
        memory_mode: int = MEMORY_MODE_INTERNAL,
    ) -> None:
        """将相位数据写入SLM内存

        将NumPy数组格式的相位数据写入SLM的指定内存位置。
        数据类型必须是uint16，表示10位灰度值（0-1023）。

        Args:
            phase: 相位数据数组，shape为(height, width)，dtype为uint16
            memory_number: 内存位置编号（1-128），默认为1
            memory_mode: 内存模式，默认为内部内存模式

        Raises:
            SantecSLM200Error: 写入失败
            ValueError: 数据格式错误
            RuntimeError: 设备未打开
        """
        self._ensure_open()

        if self.video_mode != VideoMode.Memory:
            raise RuntimeError("必须在内存模式下才能写入相位数据")

        # 在设备控制函数中进行参数验证
        if not MEMORY_NUMBER_MIN <= memory_number <= MEMORY_NUMBER_MAX:
            raise ValueError(
                f"内存编号必须在{MEMORY_NUMBER_MIN}-{MEMORY_NUMBER_MAX}"
                f"之间，当前: {memory_number}"
            )

        # 验证数据类型和形状
        if phase.dtype != np.uint16:
            phase = phase.astype(np.uint16)

        if phase.ndim != 2:
            raise ValueError(f"相位数据必须是2D数组，当前维度: {phase.ndim}")

        # 自动叠加波前误差矫正（模 MAX_GRAYSCALE_VALUE+1 环绕）
        if self._correction.is_valid and self._correction.correction_map is not None:
            phase = self._correction.map_error(
                phase, max_grayscale=self.MAX_GRAYSCALE_VALUE
            ).astype(np.uint16)

        height, width = phase.shape

        # 创建ctypes指针
        dat = (ctypes.c_ushort * (width * height))()
        ctypes.memmove(dat, phase.ctypes.data, phase.nbytes)

        # 写入SLM内存
        ret = self._slm.SLM_Ctrl_WriteMI(
            self.slm_number, memory_number, width, height, memory_mode, dat
        )

        if ret != SLM_OK:
            raise SantecSLM200Error(f"写入相位数据到内存#{memory_number}失败", code=ret)

        self._memory_phase_cache[memory_number] = phase.copy()
        logger.debug(f"相位数据已写入SLM #{self.slm_number} 内存#{memory_number}")

    def display_memory(self, memory_number: int) -> None:
        """显示指定内存中的相位图

        将指定内存位置的相位数据显示到SLM上。

        Args:
            memory_number: 内存位置编号（1-128）

        Raises:
            SantecSLM200Error: 显示失败
            RuntimeError: 设备未打开
            ValueError: 内存编号无效
        """
        self._ensure_open()

        # 在设备控制函数中进行参数验证
        if not MEMORY_NUMBER_MIN <= memory_number <= MEMORY_NUMBER_MAX:
            raise ValueError(
                f"内存编号必须在{MEMORY_NUMBER_MIN}-{MEMORY_NUMBER_MAX}"
                f"之间，当前: {memory_number}"
            )

        ret = self._slm.SLM_Ctrl_WriteDS(self.slm_number, memory_number)
        if ret != SLM_OK:
            raise SantecSLM200Error(f"显示内存#{memory_number}失败", code=ret)

        # 固件陷阱: display_memory 对"正在显示的同一槽位"被视为 no-op,
        # LCOS 面板不会刷新——若刚对该槽重写了新相位, 新图案不会上屏。
        if self._displayed_memory_number == memory_number:
            logger.warning(
                f"SLM #{self.slm_number} 连续两次 display_memory 同一内存槽 "
                f"#{memory_number}: 固件视为 no-op, LCOS 面板不会刷新, "
                "新写入的相位不会生效。请轮换到其他槽位"
                "（display_data() 内部已自动轮换 127 个槽位）"
            )

        self._displayed_memory_number = memory_number
        self._displayed_phase_cache = self._memory_phase_cache.get(memory_number)

    def display_video(self, phase):
        # 验证数据类型和形状
        if phase.dtype != np.uint16:
            raise ValueError(f"相位数据类型必须是uint16，当前: {phase.dtype}")

        if phase.ndim != 2:
            raise ValueError(f"相位数据必须是2D数组，当前维度: {phase.ndim}")

        # 应用平移
        phase = self._apply_shift(phase)

        height, width = phase.shape

        # 创建ctypes指针
        dat = (ctypes.c_ushort * (width * height))()
        ctypes.memmove(dat, phase.ctypes.data, phase.nbytes)

        ret = self._slm.SLM_Disp_Data(self.slm_number, width, height, 0, dat)

        if ret != SLM_OK:
            raise SantecSLM200Error("显示相位数据失败", code=ret)

        self._displayed_memory_number = None
        self._displayed_phase_cache = phase.copy()
        logger.debug("相位数据显示")

    def display_data(
        self, phase: np.ndarray, wait_time_s: float | None = None
    ) -> None:
        """将相位数据写入内存并显示，等待像素翻转完成。

        像素下发后的等待时间语义:
          - ``wait_time_s=None`` 或 ``< 0``: 自动估算。按相邻两帧之间最大灰度变化
            相对 2π 满量程（``_max_gray``）的比例线性推算翻转等待时间
            （上限 ``MAX_PIXEL_FLIP_TIME_S``，见 :meth:`_estimate_pixel_flip_wait`）。
            首帧（无上一帧基准）时保守等待最大翻转时间。
          - ``wait_time_s=0``: 不等待，调用方自行负责时序。
          - ``wait_time_s>0``: 使用给定的固定等待时间（秒）。

        Args:
            phase: 相位灰度数组，shape (h, w)，dtype uint16
            wait_time_s: 显示后的等待时间（秒）；None 或负值表示自动估算

        Raises:
            SantecSLM200Error: 写入或显示失败
            RuntimeError: 设备未打开
        """
        self._ensure_open()
        # 在显示前记录当前已显示相位，用于计算最大灰度变化
        # （display_memory/display_video 会同步更新 _displayed_phase_cache）
        prev_phase = self._displayed_phase_cache

        if self.video_mode == VideoMode.DVI:
            self.display_video(phase)
        elif self.video_mode == VideoMode.Memory:
            self._current_memory_slot = (self._current_memory_slot + 1) % MAX_MEM_SLOTS
            self._write_phase_with_retry(phase, self._current_memory_slot + 1)
            self.display_memory(self._current_memory_slot + 1)

        if wait_time_s is None or wait_time_s < 0:
            wait_time_s = self._estimate_pixel_flip_wait(phase, prev_phase)
            logger.debug(
                f"SLM #{self.slm_number} 自动等待 {wait_time_s * 1000:.1f}ms "
                "(按最大灰度变化估算)"
            )

        if wait_time_s > 0:
            time.sleep(wait_time_s)

    def _estimate_pixel_flip_wait(
        self, new_phase: np.ndarray, prev_phase: np.ndarray | None
    ) -> float:
        """按最大灰度变化估算 LCOS 像素翻转等待时间。

        Santec SLM-200 的 LCOS 像素从 0 翻转到 2π（满量程 ``_max_gray`` 灰度）
        耗时至多 ``MAX_PIXEL_FLIP_TIME_MS`` 毫秒（厂商规格）。等待时间线性正比:

            wait = max_gray_change / _max_gray × MAX_PIXEL_FLIP_TIME_S

        相邻帧灰度不变（最大变化为 0）时返回 0，不引入多余延时；
        无上一帧基准（首帧/状态未知）时保守返回最大翻转时间。

        Args:
            new_phase: 即将显示的灰度相位图 (uint16)
            prev_phase: 当前已显示的灰度相位图；None 表示无基准

        Returns:
            估算等待时间（秒），取值 [0, MAX_PIXEL_FLIP_TIME_S]
        """
        if prev_phase is None:
            return self.MAX_PIXEL_FLIP_TIME_S

        max_change = int(
            np.abs(new_phase.astype(np.int32) - prev_phase.astype(np.int32)).max()
        )
        if max_change <= 0:
            return 0.0

        gray_full = self._max_gray if self._max_gray else GRAYSCALE_MAX
        return min(
            self.MAX_PIXEL_FLIP_TIME_S,
            self.MAX_PIXEL_FLIP_TIME_S * max_change / gray_full,
        )

    @retry(
        stop_max_attempt_number=3,
        wait_fixed=100,
        retry_on_exception=_is_retryable_slant_error,
    )
    def _write_phase_with_retry(self, phase: np.ndarray, memory_number: int) -> None:
        """Write phase data with retry logic for transient errors.

        Args:
            phase: Phase data to write
            memory_number: Memory slot to write to (1-128)
        """
        self.write_phase(phase, memory_number)

    def set_grayscale(self, gs: int) -> None:
        """设置SLM灰度值（均匀显示）

        将SLM设置为均匀的灰度值，用于测试或重置。

        Args:
            gs: 灰度值（0-1023）

        Raises:
            RuntimeError: 设备未打开
            ValueError: 灰度值超出范围
        """
        self._ensure_open()

        # 在设备控制函数中进行参数验证
        if not GRAYSCALE_MIN <= gs <= GRAYSCALE_MAX:
            raise ValueError(
                f"灰度值必须在{GRAYSCALE_MIN}-{GRAYSCALE_MAX}之间，当前: {gs}"
            )

        ret = self._slm.SLM_Ctrl_WriteGS(self.slm_number, gs)
        if ret != SLM_OK:
            raise SantecSLM200Error("设置灰度值失败", code=ret)
        self._displayed_memory_number = None
        self._displayed_phase_cache = np.full(
            (self.Panel_Res[1], self.Panel_Res[0]),
            gs,
            dtype=np.uint16,
        )
        logger.debug(f"SLM #{self.slm_number} 灰度值设置为 {gs}")

    def get_displayed_memory_number(self) -> int | None:
        """读取当前正在显示的内存编号。"""
        self._ensure_open()
        memory_number = ctypes.c_uint32(0)
        ret = self._slm.SLM_Ctrl_ReadDS(self.slm_number, ctypes.byref(memory_number))
        if ret != SLM_OK:
            raise SantecSLM200Error("读取当前显示内存失败", code=ret)
        return memory_number.value or None

    def get_current_grayscale(self) -> int:
        """读取当前均匀灰度值。"""
        self._ensure_open()
        gray = ctypes.c_ushort(0)
        ret = self._slm.SLM_Ctrl_ReadGS(self.slm_number, ctypes.byref(gray))
        if ret != SLM_OK:
            raise SantecSLM200Error("读取当前灰度值失败", code=ret)
        return gray.value

    def get_displayed_phase(self) -> tuple[np.ndarray | None, str]:
        """获取当前显示的相位缓存及其来源说明。

        Returns:
            Tuple of (phase_gray, source). `phase_gray` 为 uint16 灰度相位图。
        """
        self._ensure_open()

        try:
            memory_number = self.get_displayed_memory_number()
        except SantecSLM200Error:
            memory_number = self._displayed_memory_number

        if memory_number is not None:
            self._displayed_memory_number = memory_number
            phase = self._memory_phase_cache.get(memory_number)
            if phase is not None:
                self._displayed_phase_cache = phase
                return phase.copy(), f"内存槽 {memory_number}"
            return None, f"内存槽 {memory_number}（未缓存，无法从设备读回整幅相位）"

        if self._displayed_phase_cache is not None:
            return self._displayed_phase_cache.copy(), "直接显示缓存"

        return None, "当前显示相位未缓存，且 SDK 不支持直接读回显存图像"

    def load_phase_from_csv(
        self, filepath: str | Path, skiprows: int = 1, delimiter: str = ","
    ) -> np.ndarray:
        """从CSV文件加载相位数据

        Args:
            filepath: CSV文件路径
            skiprows: 跳过的行数，默认为1（跳过标题行）
            delimiter: 分隔符，默认为逗号

        Returns:
            相位数据数组，dtype为uint16
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"相位文件不存在: {filepath}")

        # 读取CSV文件，跳过第一列（通常是索引）
        phase = np.loadtxt(filepath, delimiter=delimiter, skiprows=skiprows)[
            :, 1:
        ].astype(np.uint16)

        logger.info(f"已从 {filepath} 加载相位数据，形状: {phase.shape}")
        return phase

    def create_phase_from_array(
        self, phase_rad: np.ndarray, max_grayscale: int | None = None
    ) -> np.ndarray:
        """Convert radian phase values to SLM grayscale values.

        Converts phase values in radians (0-2π) to SLM grayscale (0-1023).

        Args:
            phase_rad: Phase array in radians (0-2π), shape (height, width).
            max_grayscale: Grayscale value for 2π radians. Auto-calculated if None.

        Returns:
            Grayscale phase array, dtype=uint16.
        """
        if max_grayscale is None:
            max_grayscale = self._max_gray

        # 验证并调整矩阵shape为SLM面板分辨率 (height, width)
        target_h, target_w = self.Panel_Res[1], self.Panel_Res[0]  # (1200, 1920)

        h, w = phase_rad.shape

        if (h, w) != (target_h, target_w):
            if h > target_h or w > target_w:
                # 过大：从中心裁切
                logger.warning(
                    f"输入相位图尺寸 ({h}, {w}) 超过SLM面板 ({target_h}, {target_w})，"
                    f"将从中心裁切"
                )
                # 计算裁切起始位置
                start_y = (h - target_h) // 2
                start_x = (w - target_w) // 2
                phase_rad = phase_rad[
                    start_y : start_y + target_h, start_x : start_x + target_w
                ]
            else:
                # 过小：四周补0
                logger.warning(
                    f"输入相位图尺寸 ({h}, {w}) 小于SLM面板 ({target_h}, {target_w})，"
                    f"将在四周补0"
                )
                padded = np.zeros((target_h, target_w), dtype=phase_rad.dtype)
                # 居中放置
                start_y = (target_h - h) // 2
                start_x = (target_w - w) // 2
                padded[start_y : start_y + h, start_x : start_x + w] = phase_rad
                phase_rad = padded

        # 确保 max_grayscale 有有效值
        assert max_grayscale is not None, "max_grayscale should be calculated"
        np.nan_to_num(phase_rad, copy=False, nan=0)
        phase_rad = np.mod(phase_rad, 2 * np.pi)
        # 确保输入是float类型以便计算
        phase_rad = phase_rad.astype(np.float64)

        # 将弧度转换为灰度值
        grayscale = phase_rad / (2 * np.pi) * max_grayscale

        # 叠加波前误差矫正（如有）
        if self._correction is not None and self._correction.correction_map is not None:
            grayscale = self._correction.map_error(grayscale, max_grayscale)

        # 应用相位→灰度补偿查找表（如有）
        if self._lut is not None:
            grayscale = apply_lut_remap(grayscale, self._lut)

        # 应用平移
        grayscale = self._apply_shift(grayscale)

        return grayscale.astype(np.uint16)

    def _resize_to_panel(self, data: np.ndarray) -> np.ndarray:
        """将数组裁切或补零至SLM面板分辨率（委托至 WavefrontCorrection）

        若输入尺寸超过面板分辨率，从中心裁切；
        若不足，则居中补零。

        Args:
            data: 输入数组，shape为(height, width)

        Returns:
            调整后的数组，shape为(Panel_Res[1], Panel_Res[0])，dtype float64
        """
        return WavefrontCorrection.resize_to_panel(data, self.Panel_Res)

    def _apply_shift(self, phase: np.ndarray) -> np.ndarray:
        """应用X/Y平移到相位图，空白区域填0

        Args:
            phase: 输入相位图，shape (height, width)

        Returns:
            平移后的相位图，vacated区域填0
        """
        if self._shift_x == 0 and self._shift_y == 0:
            return phase

        # scipy.ndimage.shift: shift > 0 向右/下移动 (order=0 最近邻填充0)
        # shift convention: (shift_y, shift_x) - 注意顺序
        shifted = ndimage.shift(
            phase,
            shift=(self._shift_y, self._shift_x),
            order=0,
            mode="constant",
            cval=0,
        )
        return shifted.astype(phase.dtype)

    @property
    def correction_enabled(self) -> bool:
        """当前是否已加载有效的波前误差矫正数据"""
        return (
            self._correction is not None
            and self._correction.is_valid
            and self._correction.correction_map is not None
        )

    @property
    def correction_csv_path(self) -> Path | None:
        """当前矫正 CSV 文件路径（None 表示未加载）"""
        return self._correction.csv_path if self._correction is not None else None

    def load_correction_from_csv(
        self,
        csv_path: str | Path | None = None,
    ) -> bool:
        """运行时重新加载波前误差矫正 CSV 文件。

        Args:
            csv_path: 矫正 CSV 文件路径。为 None 时使用当前已加载的路径重新计算，
                或清空矫正（当前路径无效时）。

        Returns:
            True 表示已成功加载并计算矫正映射，False 表示未加载有效矫正。
        """
        if csv_path is not None:
            resolved = WavefrontCorrection.resolve(
                explicit_path=csv_path,
                config=None,
                panel_resolution=self.Panel_Res,
            )
        else:
            current = self.correction_csv_path
            resolved = WavefrontCorrection.resolve(
                explicit_path=current if current is not None else "",
                config=None,
                panel_resolution=self.Panel_Res,
            )

        if resolved is not None:
            self._correction = resolved
            logger.info(
                f"SLM #{self.slm_number} 矫正数据已重新加载: {resolved.csv_path.name}"
            )
            return True

        self._correction = WavefrontCorrection()
        logger.info(f"SLM #{self.slm_number} 已清除矫正数据")
        return False

    # ── LUT (phase→gray compensation) ─────────────────────

    @property
    def lut(self) -> np.ndarray | None:
        """Loaded inverse_gray compensation table (uint16, len ~1024) or None."""
        return self._lut

    @property
    def lut_dir(self) -> Path | None:
        """Directory the LUT was loaded from, or None."""
        return self._lut_dir

    def load_lut(self, lut_dir: str | Path | None) -> None:
        """Load phase→gray compensation table from a LUT directory.

        The directory must contain ``lut.npz`` (preferred) or
        ``lut_inverse.csv`` as produced by ``ao_shaping.utils.slm_lut.save_lut()``.

        Args:
            lut_dir: Path to the LUT directory.  ``None`` clears the loaded LUT.
                A missing or malformed directory logs a warning and leaves the
                previous state unchanged.
        """
        if lut_dir is None:
            self._lut = None
            self._lut_dir = None
            logger.info(f"SLM #{self.slm_number} LUT 已清除")
            return

        lut_path = Path(lut_dir)

        if not lut_path.is_dir():
            logger.warning(
                f"SLM #{self.slm_number} LUT 目录不存在: {lut_path}，保持当前状态"
            )
            return

        # Try npz first
        npz_file = lut_path / "lut.npz"
        if npz_file.is_file():
            try:
                data = np.load(str(npz_file), allow_pickle=False)
                if "inverse_gray" not in data:
                    logger.warning(
                        f"SLM #{self.slm_number} lut.npz 缺少 'inverse_gray' 键，"
                        f"可用键: {list(data.keys())}，保持当前状态"
                    )
                    return
                inverse_gray = np.asarray(data["inverse_gray"])
            except (OSError, ValueError) as e:
                logger.warning(
                    f"SLM #{self.slm_number} 读取 lut.npz 失败: {e}，保持当前状态"
                )
                return

            if inverse_gray.ndim != 1 or inverse_gray.size < 1024:
                logger.warning(
                    f"SLM #{self.slm_number} inverse_gray 形状无效 "
                    f"(ndim={inverse_gray.ndim}, size={inverse_gray.size})，"
                    f"保持当前状态"
                )
                return

            self._lut = inverse_gray.astype(np.uint16)
            self._lut_dir = lut_path
            logger.info(
                f"SLM #{self.slm_number} LUT 已从 {npz_file} 加载 "
                f"(len={self._lut.size})"
            )
            return

        # Fallback: lut_inverse.csv
        csv_file = lut_path / "lut_inverse.csv"
        if csv_file.is_file():
            try:
                raw = np.genfromtxt(
                    str(csv_file), delimiter=",", dtype=np.uint16, skip_header=1
                )
                if raw.ndim == 2 and raw.shape[1] >= 2:
                    inverse_gray = raw[:, 1]  # second column: gray
                elif raw.ndim == 1:
                    inverse_gray = raw
                else:
                    logger.warning(
                        f"SLM #{self.slm_number} lut_inverse.csv 列数无效 "
                        f"(shape={raw.shape})，保持当前状态"
                    )
                    return
            except (OSError, ValueError) as e:
                logger.warning(
                    f"SLM #{self.slm_number} 读取 lut_inverse.csv 失败: {e}，"
                    f"保持当前状态"
                )
                return

            if inverse_gray.ndim != 1 or inverse_gray.size < 1024:
                logger.warning(
                    f"SLM #{self.slm_number} lut_inverse.csv 数据长度不足 "
                    f"(len={inverse_gray.size})，保持当前状态"
                )
                return

            self._lut = inverse_gray.astype(np.uint16)
            self._lut_dir = lut_path
            logger.info(
                f"SLM #{self.slm_number} LUT 已从 {csv_file} 加载 "
                f"(len={self._lut.size})"
            )
            return

        logger.warning(
            f"SLM #{self.slm_number} LUT 目录中未找到 lut.npz 或 "
            f"lut_inverse.csv，保持当前状态"
        )

    def set_shift(self, shift_x: int, shift_y: int) -> None:
        """设置平移参数

        Args:
            shift_x: X方向平移像素数（正=右，负=左）
            shift_y: Y方向平移像素数（正=下，负=上）
        """
        self._shift_x = shift_x
        self._shift_y = shift_y
        logger.info(
            f"SLM #{self.slm_number} 平移参数已更新:"
            f" shift_x={shift_x}, shift_y={shift_y}"
        )

    @property
    def shift_x(self) -> int:
        """X方向平移像素数"""
        return self._shift_x

    @property
    def shift_y(self) -> int:
        """Y方向平移像素数"""
        return self._shift_y

    @property
    def temperature(self):
        """
        Read the drive board and option board temperatures.

        Returns
        -------
        (float, float)
            Temperature in Celsius of the drive and option board
        """
        # Note that the Santec documentation suggests using signed
        # integers, but the header requests unsigned integers.
        drive_temp = ctypes.c_uint32(0)
        option_temp = ctypes.c_uint32(0)

        if (
            res := self._slm.SLM_Ctrl_ReadT(self.slm_number, drive_temp, option_temp)
        ) != SLM_OK:
            raise SantecSLM200Error(code=res)

        return (drive_temp.value / 10.0, option_temp.value / 10.0)

    def _ensure_open(self) -> None:
        """确保设备已打开

        Raises:
            RuntimeError: 设备未打开
        """
        if not self.is_open:
            raise RuntimeError("SLM设备未打开，请先调用open()方法")

    def __enter__(self):
        """上下文管理器入口"""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()

    def __repr__(self) -> str:
        """字符串表示"""
        status = "已连接" if self.is_open else "未连接"
        mode_str = "内存模式" if self.video_mode == 0 else "DVI模式"
        return (
            f"SantecSLM200("
            f"编号={self.slm_number}, "
            f"状态={status}, "
            f"波长={self.wavelength}nm, "
            f"相位范围=0~2π, "
            f"模式={mode_str}"
            f")"
        )


def test():
    with SantecSLM200(slm_number=1) as slm:
        slm.set_wavelength(1064)  # 1064nm, 2*pi相位
        phase_data = np.zeros((1080, 1920), dtype=np.uint16)
        slm.write_phase(phase_data, memory_number=1)
        slm.display_memory(1)


if __name__ == "__main__":
    test()
