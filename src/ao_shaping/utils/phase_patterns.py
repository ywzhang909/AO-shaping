"""相位图案生成工具模块

提供用于 SLM (空间光调制器) 的各种相位图案生成函数，
包括闪耀光栅、聚焦透镜、棋盘格、二元光栅、涡旋光束和 Zernike 多项式等。

使用示例:
    from ao_shaping.utils.phase_patterns import (
        generate_blazed_grating,
        generate_focus,
        SLM_RESOLUTION,
    )

    # 生成水平闪耀光栅
    phase = generate_blazed_grating(period=20, direction="horizontal")
"""

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import zoom
from scipy.special import factorial

# SLM 硬件参数
SLM_RESOLUTION = (1920, 1200)  # SLM 分辨率 (宽, 高)
SLM_BITS = 10
SLM_MAX_VAL = 2**SLM_BITS - 1  # 1023


def generate_blazed_grating(
    period: int = 20, direction: str = "horizontal"
) -> NDArray[np.uint16]:
    """生成闪耀光栅

    Args:
        period: 光栅周期（像素）
        direction: 方向，"horizontal" 或 "vertical"

    Returns:
        相位图案数组，形状为 (1200, 1920)，dtype 为 uint16
    """
    height, width = SLM_RESOLUTION[1], SLM_RESOLUTION[0]
    max_val = SLM_MAX_VAL  # 固定 2π = 1023

    if direction == "horizontal":
        y = np.arange(height)
        grating = (y % period) / period * max_val
        img = np.tile(grating[:, np.newaxis], (1, width))
    else:
        x = np.arange(width)
        grating = (x % period) / period * max_val
        img = np.tile(grating[np.newaxis, :], (height, 1))

    return img.astype(np.uint16)


def generate_focus(
    focal_length: float = 0.5,
    wavelength: float = 532e-9,
    pixel_size: float = 8e-6,
) -> NDArray[np.uint16]:
    """生成聚焦相位 (抛物面)

    Args:
        focal_length: 焦距（米）
        wavelength: 波长（米）
        pixel_size: 像素尺寸（米）

    Returns:
        相位图案数组，形状为 (1200, 1920)，dtype 为 uint16
    """
    height, width = SLM_RESOLUTION[1], SLM_RESOLUTION[0]
    max_val = SLM_MAX_VAL  # 固定 2π = 1023

    x = np.arange(width) - width // 2
    y = np.arange(height) - height // 2
    X, Y = np.meshgrid(x, y)
    R2 = X**2 + Y**2

    phase = (np.pi / wavelength / focal_length) * (R2 * pixel_size**2)
    phase_wrapped = np.mod(phase, 2 * np.pi)
    img = (phase_wrapped / (2 * np.pi) * max_val).astype(np.uint16)

    return img


def generate_checkerboard(period: int = 100) -> NDArray[np.uint16]:
    """生成棋盘格

    Args:
        period: 周期（像素）

    Returns:
        相位图案数组，形状为 (1200, 1920)，dtype 为 uint16
    """
    height, width = SLM_RESOLUTION[1], SLM_RESOLUTION[0]
    max_val = SLM_MAX_VAL  # 固定 2π = 1023

    y = np.arange(height) // period
    x = np.arange(width) // period
    X, Y = np.meshgrid(x, y)

    checker = (X + Y) % 2
    img = (checker * max_val).astype(np.uint16)

    return img


def generate_binary_grating(
    period: int = 8, direction: str = "horizontal"
) -> NDArray[np.uint16]:
    """生成二元光栅 (01光栅)

    Args:
        period: 光栅周期（像素）
        direction: 方向，"horizontal" 或 "vertical"

    Returns:
        相位图案数组，形状为 (1200, 1920)，dtype 为 uint16
    """
    height, width = SLM_RESOLUTION[1], SLM_RESOLUTION[0]
    max_val = SLM_MAX_VAL // 2  # π = 1023/2

    if direction == "horizontal":
        y = np.arange(height)
        grating = np.where(y % period < period // 2, 0, max_val)
        img = np.tile(grating[:, np.newaxis], (1, width))
    else:
        x = np.arange(width)
        grating = np.where(x % period < period // 2, 0, max_val)
        img = np.tile(grating[np.newaxis, :], (height, 1))

    return img.astype(np.uint16)


def generate_vortex(topological_charge: int = 1) -> NDArray[np.uint16]:
    """生成涡旋光束相位

    Args:
        topological_charge: 拓扑荷

    Returns:
        相位图案数组，形状为 (1200, 1920)，dtype 为 uint16
    """
    height, width = SLM_RESOLUTION[1], SLM_RESOLUTION[0]
    max_val = SLM_MAX_VAL  # 固定 2π = 1023

    x = np.arange(width) - width // 2
    y = np.arange(height) - height // 2
    X, Y = np.meshgrid(x, y)

    theta = np.arctan2(Y, X)
    phase = topological_charge * theta
    phase_wrapped = np.mod(phase, 2 * np.pi)
    img = (phase_wrapped / (2 * np.pi) * max_val).astype(np.uint16)

    return img


def generate_zernike(
    n: int = 4, m: int = 0, amplitude: float = 2.0
) -> NDArray[np.uint16]:
    """生成Zernike多项式相位

    Args:
        n: 径向阶数
        m: 角向阶数
        amplitude: 振幅（波长）

    Returns:
        相位图案数组，形状为 (1200, 1920)，dtype 为 uint16
    """
    height, width = SLM_RESOLUTION[1], SLM_RESOLUTION[0]
    max_val = SLM_MAX_VAL  # 固定 2π = 1023

    radius = min(height, width) // 2

    x = (np.arange(width) - width // 2) / radius
    y = (np.arange(height) - height // 2) / radius
    X, Y = np.meshgrid(x, y)

    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)

    mask = R <= 1.0

    def zernike_radial(n: int, m: int, r: NDArray[np.float64]) -> NDArray[np.float64]:
        R_arr = np.zeros_like(r)
        for k in range((n - abs(m)) // 2 + 1):
            coef = ((-1) ** k * factorial(n - k)) / (
                factorial(k)
                * factorial((n + abs(m)) // 2 - k)
                * factorial((n - abs(m)) // 2 - k)
            )
            R_arr += coef * r ** (n - 2 * k)
        return R_arr

    if m >= 0:
        Z = zernike_radial(n, m, R) * np.cos(m * Theta)
    else:
        Z = zernike_radial(n, -m, R) * np.sin(-m * Theta)

    Z = Z * mask
    phase = Z * amplitude * 2 * np.pi
    phase_wrapped = np.mod(phase, 2 * np.pi)
    img = (phase_wrapped / (2 * np.pi) * max_val).astype(np.uint16)

    return img


def resize_to_slm(img: NDArray[np.uint16]) -> NDArray[np.uint16]:
    """将图像调整到SLM分辨率

    Args:
        img: 输入图像数组

    Returns:
        调整大小后的图像，形状为 (1200, 1920)
    """
    target_height, target_width = SLM_RESOLUTION[1], SLM_RESOLUTION[0]

    if img.shape[0] == target_height and img.shape[1] == target_width:
        return img

    zoom_y = target_height / img.shape[0]
    zoom_x = target_width / img.shape[1]
    img_scaled = zoom(img, (zoom_y, zoom_x), order=1)

    return img_scaled.astype(np.uint16)


def load_phase_csv(file_path: str) -> NDArray[np.uint16]:
    """加载CSV格式的相位图案

    Args:
        file_path: CSV 文件路径

    Returns:
        相位数据数组，dtype 为 uint16
    """
    with open(file_path, "r") as f:
        header = f.readline().strip().split(",")
        cols = len(header) - 1

    data = np.loadtxt(file_path, delimiter=",", skiprows=1, usecols=range(1, cols + 1))
    return data.astype(np.uint16)
