"""SLM 安全规则工具(与 AGENTS.md 避坑规则一一对应)。

规则 1(灰度 RAW 路径):
    平坦相位必须直接下发 uint16 灰度值(np.full((h,w), gray, dtype=np.uint16)),
    禁止经 create_phase_from_array() 转换(该函数将输入当作弧度 mod 2π,
    uint16 灰度会被静默破坏)。

规则 2(内存槽轮换):
    连续 write_phase + display_memory 到同一槽位是 no-op(LCOS 面板不刷新)。
    必须轮换内存槽(默认 itertools.cycle([3,4,5])),每个新相位写入不同槽位。
"""
from __future__ import annotations

import itertools
from typing import Iterator

import numpy as np

# SLM-200 面板分辨率(像素)
SLM_PANEL_WIDTH = 1920
SLM_PANEL_HEIGHT = 1200
# 10-bit 灰度范围
GRAYSCALE_MIN = 0
GRAYSCALE_MAX = 1023
# 默认轮换槽位(避开 0/1/2,与硬件惯例一致)
DEFAULT_SLOTS = (3, 4, 5)


class SlmRuleError(ValueError):
    """违反 SLM 安全规则。"""


class MemorySlotRotator:
    """内存槽轮换器:保证连续两次相位写入使用不同槽位。

    用法:
        rotator = MemorySlotRotator()
        slot = rotator.next()   # 3
        slot = rotator.next()   # 4
        slot = rotator.next()   # 5
        slot = rotator.next()   # 3 (循环)

    也支持显式指定槽位集合(从 ioc.yaml 读取)。
    """

    def __init__(self, slots: tuple[int, ...] | list[int] | None = None):
        slots = tuple(slots) if slots is not None else DEFAULT_SLOTS
        if len(slots) < 2:
            raise SlmRuleError(f"内存槽轮换至少需要 2 个槽位,当前: {slots}")
        for s in slots:
            if not (1 <= s <= 127):
                raise SlmRuleError(f"SLM 内存槽位越界(1-127): {s}")
        self._slots: tuple[int, ...] = slots
        self._cycle: Iterator[int] = itertools.cycle(slots)

    def next(self) -> int:
        """返回下一个槽位(与上一次必然不同)。"""
        return next(self._cycle)

    @property
    def slots(self) -> tuple[int, ...]:
        return self._slots


def validate_grayscale(value: int | float) -> int:
    """校验 10-bit 灰度值(0-1023)。"""
    v = int(round(float(value)))
    if not (GRAYSCALE_MIN <= v <= GRAYSCALE_MAX):
        raise SlmRuleError(
            f"灰度值越界(0-1023): {value}"
        )
    return v


def flat_phase_grayscale(
    height: int = SLM_PANEL_HEIGHT,
    width: int = SLM_PANEL_WIDTH,
    gray: int = 0,
    dtype: np.dtype | type = np.uint16,
) -> np.ndarray:
    """生成平坦相位灰度图(规则 1:直接 uint16 灰度,不经弧度转换)。

    Args:
        height: 面板高度(默认 1200)。
        width: 面板宽度(默认 1920)。
        gray: 灰度值(0-1023)。
        dtype: 输出 dtype(默认 uint16)。

    Returns:
        shape (height, width) 的灰度数组。
    """
    g = validate_grayscale(gray)
    return np.full((height, width), g, dtype=dtype)


def validate_phase_array(phase: np.ndarray) -> np.ndarray:
    """校验/规范化相位数组(规则 1:仅接受 uint16 灰度)。

    允许输入 uint16(原样返回)。其他整数类型若取值范围在 0-1023 内也接受,
    浮点输入(弧度)会被拒绝,防止误用弧度转灰度。

    Args:
        phase: 相位灰度数组。

    Returns:
        规范化后的 uint16 数组。

    Raises:
        SlmRuleError: 输入非法(浮点/越界/维度错误)。
    """
    arr = np.asarray(phase)
    if arr.ndim != 2:
        raise SlmRuleError(f"相位数组必须为 2D,当前: {arr.ndim}D")
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        # 浮点即弧度:拒绝,防止误经 create_phase_from_array
        raise SlmRuleError(
            "相位数组为浮点类型(疑似弧度)。必须使用 uint16 灰度值,"
            "平坦相位请用 flat_phase_grayscale() 生成。"
        )
    if arr.dtype != np.uint16:
        # 其他整数类型:检查范围后转 uint16
        if arr.size and (arr.min() < GRAYSCALE_MIN or arr.max() > GRAYSCALE_MAX):
            raise SlmRuleError(
                f"灰度值越界(0-1023): min={arr.min()}, max={arr.max()}"
            )
        arr = arr.astype(np.uint16)
    return arr


def validate_dm_voltages(
    voltages: np.ndarray,
    vmin: float = -300.0,
    vmax: float = 499.0,
    n_actuators: int = 64,
) -> np.ndarray:
    """校验 DM 电压数组(通道数 + 范围)。"""
    arr = np.asarray(voltages, dtype=float)
    if arr.ndim != 1:
        raise SlmRuleError(f"DM 电压必须为 1D,当前: {arr.ndim}D")
    if arr.shape[0] != n_actuators:
        raise SlmRuleError(
            f"DM 通道数不匹配: 期望 {n_actuators},实际 {arr.shape[0]}"
        )
    if arr.size and (arr.min() < vmin or arr.max() > vmax):
        raise SlmRuleError(
            f"DM 电压越界({vmin}~{vmax}): min={arr.min()}, max={arr.max()}"
        )
    return arr
