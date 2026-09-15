"""Santec SLM-200 设备硬件参数常量

此模块包含 SLM-200 型号专属的设备硬件物理参数 (像素尺寸/间距、面板
分辨率、响应时间、灰度位数等)。驱动类 ``Santec`` 同时服务于 SLM-200 与
SLM-300 等型号: 后续型号可提供各自的设备常量模块 (如 ``slm300_constants``),
SDK 协议/模式/范围等驱动常量见 ``constants``。
"""

# 设备硬件参数常量
PIXEL_SIZE_UM = 7.8  # 像素尺寸 (微米)
PITCH_UM = 8  # 像素间距 (微米)
PANEL_SIZE_MM = (15.36, 9.60)  # 面板尺寸 (mm, 宽x高)
PANEL_RES = (1920, 1200)  # 面板分辨率 (宽x高)
RESPONSE_TIME_MS = 300  # 响应时间 (毫秒)
MAX_PIXEL_FLIP_TIME_MS = 200  # LCOS 像素翻转最大耗时 (毫秒)；0→2π(满相位) 量程翻转
GRAY_SCALE_BITS = 10  # 灰度位数


def get_max_grayscale() -> int:
    """Get maximum grayscale value (2^bits - 1).

    Returns:
        Maximum grayscale value based on GRAY_SCALE_BITS.
    """
    return 2 ** GRAY_SCALE_BITS - 1