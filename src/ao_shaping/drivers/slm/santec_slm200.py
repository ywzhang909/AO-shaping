"""Santec SLM-200 空间光调制器驱动模块

提供对Santec SLM-200系列空间光调制器的控制接口，
支持相位图显示、波长设置、内存模式等功能。
"""

import ctypes
import os
from pathlib import Path
from typing import Union

import numpy as np
from loguru import logger
from scipy import ndimage

from ao_shaping.drivers.slm.santec_slm200_constants import (
    FLAGS_RATE120,
    GRAY_SCALE_BITS,
    GRAYSCALE_MAX,
    GRAYSCALE_MIN,
    MAX_MEM_SLOTS,
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
from ao_shaping.utils.file import SLMConfigManager

# Config directory: <project_root>/data/slm_configs/ or from SLM_CONFIG_DIR env var
# Project root = 4 levels up from this file: src/ao_shaping/drivers/slm/
_DEFAULT_SLM_CONFIG_DIR = Path(__file__).resolve().parents[4] / "data" / "slm_configs"
_SLM_CONFIG_DIR = Path(os.environ.get("SLM_CONFIG_DIR", _DEFAULT_SLM_CONFIG_DIR))


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

    # 从常量模块导入硬件参数
    Pixel_Size_um = PIXEL_SIZE_UM
    Pitch_um = PITCH_UM
    Panel_Size_mm = PANEL_SIZE_MM
    Panel_Res = PANEL_RES
    Response_time_ms = RESPONSE_TIME_MS
    Gray_Scale_bits = GRAY_SCALE_BITS
    MAX_GRAYSCALE_VALUE = get_max_grayscale()

    def __init__(
        self,
        slm_number: int = 1,
        use_120hz: bool = False,
        wavelength: int | None = None,
        video_mode: int | VideoMode = VideoMode.Memory,
        shift_x: int | None = None,
        shift_y: int | None = None,
        correction_csv_path: Union[str, Path, None] = None,
    ):
        """初始化SLM驱动

        Args:
            slm_number: SLM设备编号（1-8），默认为1
            use_120hz: 是否使用120Hz刷新率，默认为False
            wavelength: 工作波长（nm），默认为1064；
                设为None则从配置文件或设备读取
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
        self._init_wavelength: int | None = wavelength
        self._init_shift_x: int | None = shift_x
        self._init_shift_y: int | None = shift_y

        # 实际运行时值（立即生效，供属性和测试使用）
        self.wavelength: int | None = wavelength
        self._shift_x: int = 0
        self._shift_y: int = 0

        self._current_memory_slot = 1

        # 设备序列号（open后获取）
        self._serial_number: str | None = None

        # 配置管理器
        self._config_manager: SLMConfigManager | None = None

        # 误差矫正数据（从CSV加载，在create_phase_from_array中叠加）
        self._correction_phase: np.ndarray | None = None
        if correction_csv_path is not None:
            correction_path = Path(correction_csv_path)
            if not correction_path.exists():
                raise FileNotFoundError(f"误差矫正文件不存在: {correction_path}")
            raw_correction = self.load_phase_from_csv(correction_path)
            self._correction_phase = self._resize_to_panel(raw_correction)
            logger.info(
                f"SLM #{self.slm_number} 已加载误差矫正数据: {correction_path.name}, "
            )

        # 延迟导入SLM SDK
        try:
            import ao_shaping.drivers.slm._slm_win as slm_sdk

            self._slm = slm_sdk
        except ImportError as e:
            raise SantecSLM200Error(
                f"无法导入SLM SDK (_slm_win): {e}. 请确保已安装Santec SLM驱动程序。"
            )

    def get_serial_number(self) -> str | None:
        """读取SLM设备的序列号。

        通过SDK函数 SLM_Ctrl_ReadSD 获取设备唯一序列号。
        必须在设备打开后调用。

        Returns:
            设备序列号字符串，失败时返回None

        Raises:
            RuntimeError: 设备未打开
        """
        self._ensure_open()
        device_id = ctypes.create_string_buffer(256)
        ret = self._slm.SLM_Ctrl_ReadSD(self.slm_number, device_id)
        if ret != SLM_OK:
            logger.warning(f"读取SLM序列号失败: {get_slm_error_message(ret)}")
            return None
        serial = device_id.value.decode("utf-8").strip()
        logger.debug(f"SLM #{self.slm_number} 序列号: {serial}")
        return serial

    def _init_config_manager(self) -> None:
        """初始化配置管理器"""
        if self._config_manager is None:
            self._config_manager = SLMConfigManager(_SLM_CONFIG_DIR)

    def load_config(self) -> dict:
        """加载当前设备的配置文件

        Returns:
            配置字典；无序列号或文件不存在时返回空字典
        """
        if not self._serial_number:
            return {}
        self._init_config_manager()
        assert self._config_manager is not None, "Config manager should be initialized"
        return self._config_manager.load_config(self._serial_number)

    def save_config(self) -> None:
        """将当前参数保存到JSON配置文件

        配置项包括: serial_number, wavelength, shift_x, shift_y, use_120hz, video_mode
        """
        if not self._serial_number:
            logger.warning("未获取到序列号，跳过配置保存")
            return

        self._init_config_manager()
        assert self._config_manager is not None, "Config manager should be initialized"
        config = {
            "wavelength": self.wavelength,
            "max_gray": self._max_gray,
            "shift_x": self._shift_x,
            "shift_y": self._shift_y,
            "use_120hz": self._use_120hz,
            "video_mode": self.video_mode,
        }
        self._config_manager.save_config(self._serial_number, config)
        config_file = self._config_manager._get_config_file(self._serial_number)
        logger.info(f"SLM配置已保存: {config_file}")

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

        # 读取设备序列号
        try:
            self._serial_number = self.get_serial_number()
            logger.info(f"SLM 序列号: {self._serial_number}")
        except SantecSLM200Error as e:
            logger.warning(f"无法读取SLM序列号: {e}")
            self._serial_number = None

        # 按序列号加载配置文件（如有）
        config: dict = self.load_config()

        # 应用参数优先级：__init__ 显式参数 > config文件 > 默认值
        # wavelength: None=请用配置/设备；其他值（含默认1064）=显式设置
        if self._init_wavelength is not None:
            self.wavelength = self._init_wavelength
        elif "wavelength" in config:
            cfg_wl = config["wavelength"]
            self.wavelength = int(cfg_wl)
        else:
            self.wavelength = None

        # shift_x/shift_y: 优先使用init参数，否则从配置加载
        if self._init_shift_x is not None:
            self._shift_x = self._init_shift_x
        elif "shift_x" in config:
            self._shift_x = config["shift_x"]

        if self._init_shift_y is not None:
            self._shift_y = self._init_shift_y
        elif "shift_y" in config:
            self._shift_y = config["shift_y"]

        logger.info(
            f"SLM #{self.slm_number} 参数: "
            f"wavelength={self.wavelength}, "
            f"shift_x={self._shift_x}, shift_y={self._shift_y}, "
            f"use_120hz={self._use_120hz}"
        )

        # 读取并验证设备状态，必要时从设备读取波长
        if self._check_status():
            if self.wavelength is None:
                self.wavelength, self._max_gray = self.get_wavelength_info()
            else:
                # 先获取设备当前波长，比较后再决定是否设置
                device_wavelength, device_max_gray = self.get_wavelength_info()
                if device_wavelength == self.wavelength:
                    logger.info(
                        f"SLM #{self.slm_number} 波长与设备当前值相同，跳过设置 "
                        f"(wavelength={self.wavelength})"
                    )
                    self._max_gray = device_max_gray
                else:
                    self.set_wavelength(self.wavelength, save_to_device=True)

        # 设置内存模式
        self._set_memory_mode(self.video_mode)

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
                "设置波长/相位范围失败", code=res
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

        res = self._slm.SLM_Ctrl_ReadWL(self.slm_number, wavelength, phase)
        if res != SLM_OK:
            raise SantecSLM200Error("读取波长信息失败", code=res)

        wavelength = int(wavelength.value)
        phase_pi = phase.value / 100.0
        self._max_gray = int(2.0 / phase_pi * self.MAX_GRAYSCALE_VALUE)

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

        if self.video_mode != VideoMode.Memory:
            raise RuntimeError("必须在内存模式下才能写入相位数据")

        # 在设备控制函数中进行参数验证
        if not MEMORY_NUMBER_MIN <= memory_number <= MEMORY_NUMBER_MAX:
            raise ValueError(
                f"内存编号必须在{MEMORY_NUMBER_MIN}-{MEMORY_NUMBER_MAX}之间，当前: {memory_number}"
            )

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
            ValueError: 内存编号无效
        """
        self._ensure_open()

        # 在设备控制函数中进行参数验证
        if not MEMORY_NUMBER_MIN <= memory_number <= MEMORY_NUMBER_MAX:
            raise ValueError(
                f"内存编号必须在{MEMORY_NUMBER_MIN}-{MEMORY_NUMBER_MAX}之间，当前: {memory_number}"
            )

        ret = self._slm.SLM_Ctrl_WriteDS(self.slm_number, memory_number)
        if ret != SLM_OK:
            raise SantecSLM200Error(f"显示内存#{memory_number}失败", code=ret)

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

        ret = self._slm.SLM_Disp_Data(
            self.slm_number, width, height, 0, dat
        )

        if ret != SLM_OK:
            raise SantecSLM200Error(
                "显示相位数据失败", code=ret
            )

        self._displayed_memory_number = None
        self._displayed_phase_cache = phase.copy()
        logger.debug("相位数据显示")

    def display_data(self, phase: np.ndarray):
        self._ensure_open()
        if self.video_mode == VideoMode.DVI:
            self.display_video(phase)

        elif self.video_mode == VideoMode.Memory:
            self._current_memory_slot = (self._current_memory_slot + 1) % MAX_MEM_SLOTS
            self.write_phase(phase, self._current_memory_slot+1)
            self.display_memory(self._current_memory_slot+1)

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
        self, phase_rad: np.ndarray, max_grayscale: int | None = None
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

        # 叠加误差矫正（如有）
        if self._correction_phase is not None:
            grayscale = np.mod(grayscale + self._correction_phase, max_grayscale + 1)

        # 应用平移
        grayscale = self._apply_shift(grayscale)

        return grayscale

    def _resize_to_panel(self, data: np.ndarray) -> np.ndarray:
        """将数组裁切或补零至SLM面板分辨率

        若输入尺寸超过面板分辨率，从中心裁切；
        若不足，则居中补零。

        Args:
            data: 输入数组，shape为(height, width)

        Returns:
            调整后的数组，shape为(Panel_Res[1], Panel_Res[0])
        """
        target_h, target_w = self.Panel_Res[1], self.Panel_Res[0]  # (1200, 1920)
        h, w = data.shape

        if (h, w) == (target_h, target_w):
            return data.astype(np.float64)

        if h > target_h or w > target_w:
            start_y = (h - target_h) // 2
            start_x = (w - target_w) // 2
            result = data[start_y:start_y + target_h, start_x:start_x + target_w]
        else:
            result = np.zeros((target_h, target_w), dtype=data.dtype)
            start_y = (target_h - h) // 2
            start_x = (target_w - w) // 2
            result[start_y:start_y + h, start_x:start_x + w] = data

        return result.astype(np.float64)

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
