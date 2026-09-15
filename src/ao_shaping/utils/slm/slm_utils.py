"""SLM 相位 → 灰度转换与内存槽轮换辅助函数。

纯辅助函数, 通过鸭子类型操作 SLM 设备对象 (任何暴露
``get_displayed_memory_number``, ``create_phase_from_array``,
``display_phase`` 的对象)。此处不执行任何硬件导入;
调用方传入已打开的 SLM 设备。

从 :mod:`ao_shaping.algorithm.beam_shaping_utils` 迁移而来; 旧模块为向后
兼容重新导出这些名称。

内存槽规则 (AGENTS.md 铁律):
- 连续写入必须始终指向不同槽位; 对已在显示的槽调用 ``display_memory``
  是空操作, LCOS 面板不会刷新。
- 推荐模式: 在 2..125 内随机选槽, 排除当前显示槽 (首次调用时通过
  ``get_displayed_memory_number()`` 读取, 跨进程重启仍有效)。
- 仅内存模式 (``video_mode=0``); 绝不自动尝试 DVI 模式。
"""

from __future__ import annotations

import random
import time
from typing import Any

import numpy as np

from loguru import logger

__all__ = [
    "DEFAULT_WAVELENGTH",
    "DEFAULT_SLM_PIXEL_SIZE",
    "DEFAULT_DISTANCE",
    "DEFAULT_MAX_GRAYSCALE",
    "phase_to_slm_grayscale",
    "pick_slm_slot",
    "display_phase",
]

# Shared physical / SLM constants
DEFAULT_WAVELENGTH: float = 1064e-9  # 1064 nm YAG laser (meters)
DEFAULT_SLM_PIXEL_SIZE: float = 8e-6  # SLM pixel pitch (meters)
DEFAULT_DISTANCE: float = 0.1  # default propagation distance (meters)
DEFAULT_MAX_GRAYSCALE: int = 1023  # grayscale value corresponding to 2*pi


def phase_to_slm_grayscale(
    phase: np.ndarray,
    max_grayscale: int = DEFAULT_MAX_GRAYSCALE,
) -> np.ndarray:
    """将相位图 (弧度) 转换为 SLM uint16 灰度图。

    将相位折叠到 ``[0, 2*pi)`` 并缩放到 ``[0, max_grayscale]`` 范围。数值
    被限制在灰度范围内并转换为 ``uint16``。

    Args:
        phase: 以弧度表示的二维相位数组。
        max_grayscale: 对应 ``2*pi`` 的灰度值 (10 位 LCOS 设备默认 1023)。

    Returns:
        ``uint16`` 二维灰度数组。
    """
    phase = np.asarray(phase, dtype=np.float32)
    phase = np.mod(phase, 2 * np.pi)
    gray = (phase / (2 * np.pi)) * max_grayscale
    return np.clip(gray, 0, max_grayscale).astype(np.uint16)


# ---------------------------------------------------------------------------
# SLM memory-slot rotation (shared by diff_beam_runner and diff_shaping_runner)
# ---------------------------------------------------------------------------
_SLOT_MIN: int = 2
_SLOT_MAX: int = 125
_last_slm_slot: int | None = None


def pick_slm_slot(slm: Any) -> int:
    """在 ``[_SLOT_MIN, _SLOT_MAX]`` 内随机选取一个与上次使用槽位不同的
    SLM 内存槽。

    首次调用时从设备读取当前显示槽, 使轮换在进程重启后仍能延续 (仅内存
    模式)。

    Args:
        slm: 已打开的 SLM 设备对象, 暴露 ``get_displayed_memory_number``。

    Returns:
        ``[2, 125]`` 范围内的槽号。
    """
    global _last_slm_slot
    if _last_slm_slot is None:
        try:
            _last_slm_slot = slm.get_displayed_memory_number()
            logger.info("SLM当前显示槽: {}", _last_slm_slot)
        except Exception as exc:
            logger.debug("读取 SLM 当前显示槽失败: {}", exc)
            _last_slm_slot = None
    candidates = [s for s in range(_SLOT_MIN, _SLOT_MAX + 1) if s != _last_slm_slot]
    slot = random.choice(candidates)
    _last_slm_slot = slot
    return slot


def display_phase(slm: Any, phase_rad: np.ndarray, settle_time_s: float) -> None:
    """使用内存槽轮换在 SLM 上显示弧度相位图案。

    使用 ``create_phase_from_array`` (设备原生的 2π 转换 + 矫正/LUT) 且
    仅用内存模式 (绝不使用 DVI)。

    Args:
        slm: 已打开的 SLM 设备。
        phase_rad: 以弧度表示的二维相位数组。
        settle_time_s: ``display_phase`` 后的等待时间。
    """
    slot = pick_slm_slot(slm)
    slm.display_phase(phase_rad, memory_number=slot)
    time.sleep(settle_time_s)
