"""Santec SLM-200 空间光调制器驱动模块

提供对Santec SLM-200系列空间光调制器的控制接口，
支持相位图显示、波长设置、内存模式等功能。
"""

import ctypes
from pathlib import Path
from typing import Optional, Union
from enum import IntEnum

import numpy as np
from scipy import ndimage

from loguru import logger

# SLM SDK常量
SLM_OK = 0
FLAGS_RATE120 = 1

# 内存模式
MEMORY_MODE_INTERNAL = 0  # 内部内存模式
DVI_MODE = 1  # DVI模式


class VideoMode(IntEnum):
    # 0:Memory mode, 1:DVI mode
    Memory = 0
    DVI = 1


class SLMErrorCode(IntEnum):
    """SLM SDK 错误码定义

    对应Santec SLM-200官方手册中的错误代码。
    """

    # SLM 基础错误码
    SLM_OK = 0
    SLM_NG = 1
    SLM_IS_BUSY = 2
    SLM_PARAMETER_ERROR = 3

    # 显示/显示器相关错误
    SLM_INVALID_MONITOR = -1
    SLM_NOT_OPEN_MONITOR = -2
    SLM_OPEN_WINDOW_ERR = -3
    SLM_DATA_FORMAT_ERR = -4
    SLM_FILE_READ_ERR = -101

    # USB/连接相关错误
    SLM_NOT_OPEN_USB = -200

    # 通用错误
    SLM_OTHER_ERROR = -1000

    # FTDI USB驱动错误
    FT_INVALID_HANDLE = -10001
    FT_DEVICE_NOT_FOUND = -10002
    FT_DEVICE_NOT_OPENED = -10003
    FT_IO_ERROR = -10004
    FT_INSUFFICIENT_RESOURCES = -10005
    FT_INVALID_PARAMETER = -10006
    FT_INVALID_BAUD_RATE = -10007
    FT_DEVICE_NOT_OPENED_FOR_ERASE = -10008
    FT_DEVICE_NOT_OPENED_FOR_WRITE = -10009
    FT_FAILED_TO_WRITE_DEVICE = -10010
    FT_EEPROM_READ_FAILED = -10011
    FT_EEPROM_WRITE_FAILED = -10012
    FT_EEPROM_ERASE_FAILED = -10013
    FT_EEPROM_NOT_PRESENT = -10014
    FT_EEPROM_NOT_PROGRAMMED = -10015
    FT_INVALID_ARGS = -10016
    FT_NOT_SUPPORTED = -10017
    FT_NO_MORE_ITEMS = -10018
    FT_TIMEOUT = -10019
    FT_OPERATION_ABORTED = -10020
    FT_RESERVED_PIPE = -10021
    FT_INVALID_CONTROL_REQUEST_DIRECTION = -10022
    FT_INVALID_CONTROL_REQUEST_TYPE = -10023
    FT_IO_PENDING = -10024
    FT_IO_INCOMPLETE = -10025
    FT_HANDLE_EOF = -10026
    FT_BUSY = -10027
    FT_NO_SYSTEM_RESOURCES = -10028
    FT_DEVICE_LIST_NOT_READY = -10029
    FT_DEVICE_NOT_CONNECTED = -10030
    FT_INCORRECT_DEVICE_PATH = -10031
    FT_OTHER_ERROR = -10032


# 错误码到人类可读消息的映射
_SLM_ERROR_MESSAGES = {
    # SLM 基础错误码
    0: "操作成功",
    1: "操作失败",
    2: "SLM 忙碌中，请稍后重试",
    3: "参数错误，请检查输入参数",

    # 显示/显示器相关错误
    -1: "未找到有效的显示器",
    -2: "显示器未打开",
    -3: "窗口打开错误",
    -4: "数据格式错误",
    -101: "数据值超出0-1023范围",

    # USB/连接相关错误
    -200: "USB未连接或未打开",

    # 通用错误
    -1000: "未知错误",

    # FTDI USB驱动错误
    -10001: "USB驱动句柄无效",
    -10002: "未找到USB设备，请检查设备电源和连接",
    -10003: "USB设备已打开",
    -10004: "USB通信错误",
    -10005: "USB资源不足",
    -10006: "USB参数无效",
    -10007: "USB波特率无效",
    -10008: "USB设备未打开(擦除)",
    -10009: "USB设备未打开(写入)",
    -10010: "USB写入失败",
    -10011: "EEPROM读取失败",
    -10012: "EEPROM写入失败",
    -10013: "EEPROM擦除失败",
    -10014: "EEPROM不存在",
    -10015: "EEPROM未编程",
    -10016: "参数无效",
    -10017: "操作不支持",
    -10018: "没有更多项目",
    -10019: "操作超时",
    -10020: "操作中止",
    -10021: "保留管道错误",
    -10022: "无效的控制请求方向",
    -10023: "无效的控制请求类型",
    -10024: "IO等待中",
    -10025: "IO未完成",
    -10026: "句柄结束",
    -10027: "USB设备忙碌",
    -10028: "系统资源不足",
    -10029: "设备列表未就绪",
    -10030: "USB设备未连接",
    -10031: "设备路径错误",
    -10032: "USB其他错误",
}


def _get_slm_error_message(code: int) -> str:
    """获取SLM错误码对应的可读错误消息

    Args:
        code: SLM SDK返回的错误码

    Returns:
        人类可读的错误消息，如果未知则返回"未知错误码"
    """
    return _SLM_ERROR_MESSAGES.get(code, f"未知错误码 ({code})")


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
            error_detail = _get_slm_error_message(code)
            full_message = f"{message} (错误码: {code}, {error_detail})" if message else f"错误码: {code}, {error_detail}"
            super().__init__(full_message)
        else:
            super().__init__(message)


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
    """

    Pixel_Size_um = 7.8
    Pitch_um = 8
    Panel_Size_mm = (15.36, 9.60)
    Panel_Res = (1920, 1200)
    Response_time_ms = 300
    Gray_Scale_bits = 10
    MAX_GRAYSCALE_VALUE = 2**Gray_Scale_bits - 1  # 10位灰阶最大值: 1023

    def __init__(
        self,
        slm_number: int = 1,
        use_120hz: bool = False,
        wavelength: int = 1064,
        video_mode: int | VideoMode = 0,
        shift_x: int = 0,
        shift_y: int = 0,
    ):
        """初始化SLM驱动

        Args:
            slm_number: SLM设备编号（1-8），默认为1
            use_120hz: 是否使用120Hz刷新率，默认为False
            wavelength: 工作波长（nm），默认为1064
            video_mode: 视频模式 (0=内存模式, 1=DVI模式)，默认为0
            shift_x: X方向平移像素数（正=右，负=左），默认为0
            shift_y: Y方向平移像素数（正=下，负=上），默认为0

        Raises:
            SantecSLM200Error: 设备编号无效
        """
        if not 1 <= slm_number <= 8:
            raise SantecSLM200Error(f"SLM编号必须在1-8之间，当前: {slm_number}")
        assert 450 <= wavelength <= 1600, f"{wavelength=} not in range(450, 1600)"
        self.slm_number = slm_number
        self.flags = FLAGS_RATE120 if use_120hz else 0
        self.wavelength = wavelength
        self.video_mode = video_mode if isinstance(video_mode, int) else int(video_mode)
        self.is_open = False
        self._max_gray = self.MAX_GRAYSCALE_VALUE
        self._memory_phase_cache: dict[int, np.ndarray] = {}
        self._displayed_memory_number: int | None = None
        self._displayed_phase_cache: np.ndarray | None = None

        # 平移参数（X正=右，Y正=下，vacated区域填0）
        self._shift_x = shift_x
        self._shift_y = shift_y

        # 延迟导入SLM SDK
        try:
            import ao_shaping.drivers.slm._slm_win as slm_sdk

            self._slm = slm_sdk
        except ImportError as e:
            raise SantecSLM200Error(
                f"无法导入SLM SDK (_slm_win): {e}. 请确保已安装Santec SLM驱动程序。"
            )

    def open(self) -> None:
        """打开SLM设备连接

        建立与SLM控制器的通信连接。在调用其他方法前必须先调用此方法。

        Raises:
            SantecSLM200Error: 设备连接失败
        """
        if self.is_open:
            logger.warning(f"SLM #{self.slm_number} 已经处于打开状态")
            return

        # 先尝试关闭（确保干净状态）
        self._slm.SLM_Ctrl_Close(self.slm_number)

        # 打开设备
        ret = self._slm.SLM_Ctrl_Open(self.slm_number)
        if ret != SLM_OK:
            raise SantecSLM200Error(f"无法打开SLM #{self.slm_number}", code=ret)

        self.is_open = True
        logger.info(f"成功打开SLM #{self.slm_number}")

        # 读取并验证设备状态
        self._check_status()

        # 设置内存模式
        self._set_memory_mode(self.video_mode)

    def close(self) -> None:
        """关闭SLM设备连接

        断开与SLM控制器的通信连接，释放资源。
        """
        if not self.is_open:
            return

        ret = self._slm.SLM_Ctrl_Close(self.slm_number)
        if ret == SLM_OK:
            logger.info(f"成功关闭SLM #{self.slm_number}")
        else:
            logger.warning(f"关闭SLM #{self.slm_number}时返回错误码: {ret}")

        self.is_open = False

    def _check_status(self) -> None:
        """检查SLM设备状态

        Raises:
            SantecSLM200Error: 设备状态异常
        """
        ret = self._slm.SLM_Ctrl_ReadSU(self.slm_number)
        if ret != SLM_OK:
            raise SantecSLM200Error(f"SLM #{self.slm_number} 状态异常", code=ret)
        logger.debug(f"SLM #{self.slm_number} 状态正常")

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
            wavelength: 工作波长（nm），例如 1064
            save_to_device: 是否保存到SLM控制器，默认为True

        Raises:
            SantecSLM200Error: 设置失败
            RuntimeError: 设备未打开
        """
        self._ensure_open()
        assert 450 <= wavelength <= 1600, f"{wavelength=} not in range(450, 1600)"
        logger.info(f"设置波长 {wavelength}nm (0~2π相位范围)...")

        # 固定使用 2π 相位范围 (200 = 2*pi)
        phase_range = 200

        # 设置波长和相位范围
        res = self._slm.SLM_Ctrl_WriteWL(self.slm_number, wavelength, phase_range)
        if res != SLM_OK:
            raise SantecSLM200Error(
                f"设置波长/相位范围失败", code=res
            )

        # 保存到设备
        if save_to_device:
            ret = self._slm.SLM_Ctrl_WriteAW(self.slm_number)
            if ret != SLM_OK:
                raise SantecSLM200Error(f"保存波长设置失败", code=ret)

        self.wavelength = wavelength

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

        dat32_1 = ctypes.c_uint32(0)
        dat32_2 = ctypes.c_uint32(0)

        res = self._slm.SLM_Ctrl_ReadWL(self.slm_number, dat32_1, dat32_2)
        if res != SLM_OK:
            raise SantecSLM200Error(f"读取波长信息失败", code=res)

        wavelength = dat32_1.value

        return wavelength, self._max_gray

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

        if not 1 <= memory_number <= 128:
            raise ValueError(f"内存编号必须在1-128之间，当前: {memory_number}")

        # 验证数据类型和形状
        if phase.dtype != np.uint16:
            phase = phase.astype(np.uint16)

        if phase.ndim != 2:
            raise ValueError(f"相位数据必须是2D数组，当前维度: {phase.ndim}")

        height, width = phase.shape

        # 创建ctypes指针
        dat = (ctypes.c_ushort * (width * height))()
        ctypes.memmove(dat, phase.ctypes.data, phase.nbytes)

        # 写入SLM内存
        ret = self._slm.SLM_Ctrl_WriteMI(
            self.slm_number, memory_number, width, height, memory_mode, dat
        )

        if ret != SLM_OK:
            raise SantecSLM200Error(
                f"写入相位数据到内存#{memory_number}失败", code=ret
            )

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
        """
        self._ensure_open()

        if not 1 <= memory_number <= 128:
            raise ValueError(f"内存编号必须在1-128之间，当前: {memory_number}")

        ret = self._slm.SLM_Ctrl_WriteDS(self.slm_number, memory_number)
        if ret != SLM_OK:
            raise SantecSLM200Error(f"显示内存#{memory_number}失败", code=ret)

        self._displayed_memory_number = memory_number
        self._displayed_phase_cache = self._memory_phase_cache.get(memory_number)
        logger.info(f"SLM #{self.slm_number} 正在显示内存#{memory_number}的相位图")

    def display_data(self, phase: np.ndarray):
        self._ensure_open()
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

        ret = self._slm.SLM_Disp_Data(
            self.slm_number, width, height, 0, dat
        )

        if ret != SLM_OK:
            raise SantecSLM200Error(
                f"显示相位数据失败", code=ret
            )

        self._displayed_memory_number = None
        self._displayed_phase_cache = phase.copy()
        logger.debug("相位数据显示")

    def set_grayscale(self, gs: int) -> None:
        """设置SLM灰度值（均匀显示）

        将SLM设置为均匀的灰度值，用于测试或重置。

        Args:
            gs: 灰度值（0-1023）

        Raises:
            RuntimeError: 设备未打开
        """
        self._ensure_open()
        ret = self._slm.SLM_Ctrl_WriteGS(self.slm_number, gs)
        if ret != SLM_OK:
            raise SantecSLM200Error(f"设置灰度值失败", code=ret)
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
            raise SantecSLM200Error(f"读取当前显示内存失败", code=ret)
        return memory_number.value or None

    def get_current_grayscale(self) -> int:
        """读取当前均匀灰度值。"""
        self._ensure_open()
        gray = ctypes.c_ushort(0)
        ret = self._slm.SLM_Ctrl_ReadGS(self.slm_number, ctypes.byref(gray))
        if ret != SLM_OK:
            raise SantecSLM200Error(f"读取当前灰度值失败", code=ret)
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
        self, filepath: Union[str, Path], skiprows: int = 1, delimiter: str = ","
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
        self, phase_rad: np.ndarray, max_grayscale: Optional[int] = None
    ) -> np.ndarray:
        """将弧度相位值转换为SLM灰度值

        将弧度为单位的相位值（0-2π）转换为SLM的灰度值（0-1023）。

        Args:
            phase_rad: 相位值数组（弧度，0-2π）
            max_grayscale: 2π对应的灰度值，默认为自动计算

        Returns:
            灰度值数组，dtype为uint16
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
                phase_rad = phase_rad[start_y : start_y + target_h, start_x : start_x + target_w]
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
        phase_rad = np.mod(phase_rad, 2*np.pi)
        # 确保输入是float类型以便计算
        phase_rad = phase_rad.astype(np.float64)

        # 将弧度转换为灰度值
        grayscale = (phase_rad / (2 * np.pi) * max_grayscale)      
        # 应用平移
        grayscale = self._apply_shift(grayscale)

        return grayscale

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

    def set_shift(self, shift_x: int, shift_y: int) -> None:
        """设置平移参数

        Args:
            shift_x: X方向平移像素数（正=右，负=左）
            shift_y: Y方向平移像素数（正=下，负=上）
        """
        self._shift_x = shift_x
        self._shift_y = shift_y
        logger.info(f"SLM #{self.slm_number} 平移参数已更新: shift_x={shift_x}, shift_y={shift_y}")

    @property
    def shift_x(self) -> int:
        """X方向平移像素数"""
        return self._shift_x

    @property
    def shift_y(self) -> int:
        """Y方向平移像素数"""
        return self._shift_y

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
