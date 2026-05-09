"""Santec SLM-200 静态常量和错误码定义

此模块包含所有Santec SLM-200相关的静态常量，包括错误码、错误消息、
枚举类型和设备参数常量。
"""

from enum import IntEnum


# SDK基础常量
SLM_OK = 0
FLAGS_RATE120 = 1

# 内存模式
MEMORY_MODE_INTERNAL = 0  # 内部内存模式
DVI_MODE = 1  # DVI模式

# 可用内存槽数量
MAX_MEM_SLOTS = 127  # 可用的内存地址为 1~127


class VideoMode(IntEnum):
    """视频模式枚举"""
    Memory = 0  # 内存模式
    DVI = 1     # DVI模式


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
SLM_ERROR_MESSAGES = {
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


def get_slm_error_message(code: int) -> str:
    """获取SLM错误码对应的可读错误消息

    Args:
        code: SLM SDK返回的错误码

    Returns:
        人类可读的错误消息，如果未知则返回"未知错误码"
    """
    return SLM_ERROR_MESSAGES.get(code, f"未知错误码 ({code})")


# 设备硬件参数常量
PIXEL_SIZE_UM = 7.8  # 像素尺寸 (微米)
PITCH_UM = 8  # 像素间距 (微米)
PANEL_SIZE_MM = (15.36, 9.60)  # 面板尺寸 (mm, 宽x高)
PANEL_RES = (1920, 1200)  # 面板分辨率 (宽x高)
RESPONSE_TIME_MS = 300  # 响应时间 (毫秒)
GRAY_SCALE_BITS = 10  # 灰度位数


def get_max_grayscale() -> int:
    """Get maximum grayscale value (2^bits - 1).

    Returns:
        Maximum grayscale value based on GRAY_SCALE_BITS.
    """
    return 2 ** GRAY_SCALE_BITS - 1


# 默认参数值
DEFAULT_WAVELENGTH = 1064
DEFAULT_SHIFT_X = 0
DEFAULT_SHIFT_Y = 0

# SLM编号范围
SLM_NUMBER_MIN = 1
SLM_NUMBER_MAX = 8

# 波长范围 (nm)
WAVELENGTH_MIN = 450
WAVELENGTH_MAX = 1600

# 灰度值范围
GRAYSCALE_MIN = 0
GRAYSCALE_MAX = 1023

# 内存编号范围
MEMORY_NUMBER_MIN = 1
MEMORY_NUMBER_MAX = 128
