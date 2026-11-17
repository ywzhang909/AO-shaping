"""零级中心"算法选择"层 —— 四种口径 (argmax / 质心 / 框内质心 / 椭圆中心)。

默认椭圆中心 (半高连通域的二阶矩中心)。基础原语 (``despike_frame`` /
``locate_peak``) 仍在 :mod:`ao_shaping.utils.image.beam_metrics`; 本模块只做
选择与组合, 不重复实现。依赖是单向的 ``zero_order`` → ``beam_metrics``。
"""

from __future__ import annotations

import numpy as np

from ao_shaping.utils.image.beam_metrics import (
    _DESPIKE_MIN_K,
    despike_frame,
    locate_peak,
)
from ao_shaping.utils.image.ranks import NDIM_SINGLE_CHANNEL

__all__ = [
    "DEFAULT_ZERO_ORDER_METHOD",
    "ZERO_ORDER_METHODS",
    "zero_order_center_by_method",
]


#: 零级中心算法: (键, 界面显示名)。默认椭圆中心 (半高连通域的二阶矩中心)。
ZERO_ORDER_METHODS: tuple[tuple[str, str], ...] = (
    ("argmax", "最大值"),
    ("centroid", "质心"),
    ("box_centroid", "框内质心"),
    ("ellipse", "椭圆中心"),
)
DEFAULT_ZERO_ORDER_METHOD = "ellipse"


def _corner_background(img: np.ndarray) -> float:
    """四角小块 (边长 ``max(dim // 50, 2)``) 的最大值, 沿用角点背景口径。"""
    height, width = img.shape
    ch = max(int(height // 50), 2)
    cw = max(int(width // 50), 2)
    corners = (img[:ch, :cw], img[:ch, -cw:], img[-ch:, :cw], img[-ch:, -cw:])
    return float(max(float(patch.max()) for patch in corners))


def _largest_half_max_component(img: np.ndarray, level: float) -> np.ndarray | None:
    """半高阈值掩码的 4 连通最大连通域; 无前景返回 ``None``。"""
    from scipy.ndimage import label

    labeled, num = label(img >= level)
    if num == 0:
        return None
    counts = np.bincount(labeled.ravel())
    counts[0] = 0  # 0 号是背景
    idx = int(np.argmax(counts))
    if idx == 0 or counts[idx] == 0:
        return None
    return labeled == idx


def _intensity_centroid(weights: np.ndarray) -> tuple[float, float] | None:
    """非负权重图的一阶矩 ``(x, y)``; 总权重为零返回 ``None``。"""
    total = float(weights.sum())
    if total <= 0.0:
        return None
    cols = np.arange(weights.shape[1], dtype=np.float64)
    rows = np.arange(weights.shape[0], dtype=np.float64)
    cx = float(np.dot(weights.sum(axis=0, dtype=np.float64), cols))
    cy = float(np.dot(weights.sum(axis=1, dtype=np.float64), rows))
    return (cx / total, cy / total)


def _round_center(
    center: tuple[float, float] | None, fallback: tuple[int, int]
) -> tuple[int, int]:
    """浮点中心四舍五入为整数像素; ``None`` 时返回 ``fallback``。"""
    if center is None:
        return fallback
    return (int(round(center[0])), int(round(center[1])))


def zero_order_center_by_method(
    frame: np.ndarray,
    method: str = DEFAULT_ZERO_ORDER_METHOD,
    *,
    threshold_fraction: float = 0.5,
    despike_k: int = 1,
) -> tuple[int, int]:
    """按 ``method`` 取 0 级光斑中心, 默认椭圆中心。

    ``beam_metrics.zero_order_center`` 的 argmax 锚定在杂散光下最稳, 但它把
    整个 0 级压成一个像素; 真实光束不是点, 且帧里常有散斑与热点。本函数
    提供可选的口径面 (不改动 ``zero_order_center`` 的历史行为):

    * ``argmax`` — 全局峰值像素, 等价 ``zero_order_center(frame, refine=False)``;
    * ``centroid`` — 扣角点背景后全画幅一阶矩;
    * ``box_centroid`` — 半高最大连通域外接框内的强度加权质心;
    * ``ellipse`` — 半高最大连通域的二阶矩中心 (默认)。

    Args:
        frame: 二维强度帧。
        method: 算法键, 见 :data:`ZERO_ORDER_METHODS`。
        threshold_fraction: 半高阈值 ``背景 + f * (峰值 - 背景)`` 的比例。
        despike_k: ``k >= 3`` 时先做 ``k x k`` 中值去热点, 同
            :func:`~ao_shaping.utils.image.beam_metrics.zero_order_center`。

    Returns:
        ``(x, y)`` 整数像素坐标 (项目约定); 全暗帧返回帧中心。

    Raises:
        ValueError: 帧不是非空二维图, 或 ``method`` 未知。
    """
    valid = [key for key, _ in ZERO_ORDER_METHODS]
    if method not in valid:
        raise ValueError(f"未知零级中心算法: {method!r} (可选 {valid})")
    img = np.asarray(frame)
    if img.ndim != NDIM_SINGLE_CHANNEL or img.size == 0:
        raise ValueError(f"frame must be a non-empty 2D array, got shape {img.shape}")
    height, width = (int(img.shape[0]), int(img.shape[1]))
    if float(img.max()) <= float(img.min()):
        return (width // 2, height // 2)

    clean = img.astype(np.float64)
    if despike_k >= _DESPIKE_MIN_K:
        clean = np.asarray(despike_frame(img, despike_k), dtype=np.float64)
    peak_y, peak_x = locate_peak(clean)
    argmax_center = (int(peak_x), int(peak_y))
    if method == "argmax":
        return argmax_center

    background = _corner_background(clean)
    weights = np.clip(clean - background, 0.0, None)
    plain = _intensity_centroid(weights)
    if method == "centroid":
        return _round_center(plain, argmax_center)

    level = background + float(threshold_fraction) * (float(clean.max()) - background)
    component = _largest_half_max_component(clean, level)
    if component is None:
        return _round_center(plain, argmax_center)
    ys, xs = np.nonzero(component)
    if method == "ellipse":
        if xs.size < 9:
            return _round_center(plain, argmax_center)
        return (int(round(float(xs.mean()))), int(round(float(ys.mean()))))

    # box_centroid: 外接框内的强度加权质心 (行/列切片即外接框)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    boxed = _intensity_centroid(weights[y0:y1, x0:x1])
    if boxed is None:
        return _round_center(plain, argmax_center)
    return (int(round(boxed[0] + x0)), int(round(boxed[1] + y0)))
