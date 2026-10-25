"""形状→掩膜的纯几何原语（顶点 / 面积 / 包含判定 / 腐蚀）。

Part of the :mod:`ao_shaping.utils.image.target` package (split by type).

本模块从 ``ao_shaping.algorithm.signal_processing.target_masks`` 上移而来
（issue #199 P2：掩膜几何属于 target 层，不该躺在 algorithm 层），并吸收
``patterns.create_target_shape`` 的三角/星形光栅化算式，成为**唯一**的
形状几何实现（issue #199 R1：曾经有 5 份）。

坐标约定（与相机栅格一致）：
* 顶点列序 ``(x, y)``，相对形状中心；
* 栅格下标 ``(row, col) = (y, x)``；
* 顶点用数学角（``+90°`` 在标准数学平面是"上"），但栅格化的图像数组
  ``row`` 向下增长，所以**尖朝上**的三角形首顶点在 ``-90°``
  （``shape_vertices`` 与 ``patterns.create_target_shape`` 同向）；
* ``step_ccd_px = (step_y, step_x)``：每个栅格格子在相机像元上的尺寸。

依赖约束：本模块只 import numpy（包的叶子层测试见
``tests/ao_shaping/utils/test_target_package.py::TestLeafLayer``），
因此 :mod:`patterns` 可以在模块级引用它而不成环。
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

ShapeName = Literal["square", "rectangle", "triangle", "star"]


def _regular_polygon(n: int, *, start_deg: float = 90.0) -> npt.NDArray[np.floating]:
    """单位外接圆正多边形顶点，列序 ``(x, y)``，``start_deg`` 处为首个顶点。"""
    angles = np.deg2rad(start_deg + 360.0 * np.arange(n) / n)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def _star_polygon(n_points: int = 5) -> npt.NDArray[np.floating]:
    """单位外接圆五角星：外顶点与内顶点交替，内半径 = sin(36°)/sin(54°)。

    外环首顶点在 ``-90°``（图像坐标里"尖朝上"），与
    :func:`~ao_shaping.utils.image.target.patterns.create_target_shape`
    的星形同向。注意内半径比（0.7265，transfer 侧历史口径；原 docstring
    误写为 sin(18°)/sin(54°)）与 patterns 的 ``0.42`` 是**两套历史参数**，
    本模块保持 transfer 侧的口径不变（能力报告的历史星形数据即此口径）。
    """
    inner = np.sin(np.deg2rad(180.0 / n_points)) / np.sin(
        np.deg2rad(3.0 * 180.0 / (2.0 * n_points))
    )
    outer = _regular_polygon(n_points, start_deg=-90.0)
    inner_pts = inner * _regular_polygon(
        n_points, start_deg=-90.0 + 180.0 / n_points
    )
    vertices = np.empty((2 * n_points, 2))
    vertices[0::2] = outer
    vertices[1::2] = inner_pts
    return vertices


def polygon_area(vertices: npt.NDArray[np.floating]) -> float:
    """鞋带公式面积（列序 ``(x, y)``，返回绝对面积）。"""
    x = vertices[:, 0]
    y = vertices[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def shape_vertices(
    shape: ShapeName,
    *,
    area_px2: float = 7200.0,
    size_px: tuple[float, float] | None = None,
    rotation_deg: float = 0.0,
) -> npt.NDArray[np.floating]:
    """返回按 ``area_px2`` 归一、以原点为中心、可旋转的形状顶点（单位：相机像元）。

    Args:
        shape: ``"square"`` / ``"rectangle"`` / ``"triangle"`` / ``"star"``。
        area_px2: 目标面积（相机像元²）；四个形状共用同一面积才可比。
        size_px: 仅长方形用，``(长, 短)``，决定长宽比；面积仍由 ``area_px2`` 定。
        rotation_deg: 面内旋转角（逆时针，度）；旋转检验用。

    Returns:
        ``(N, 2)`` 顶点数组，列序 ``(x, y)``。

    Note:
        三角/星形尖朝上（图像坐标）。上移前 ``target_masks`` 的首顶点在
        ``+90°``，在向下增长的 row 上实际尖朝下，与其 docstring 相反；统一到
        ``patterns`` 的口径（issue #199 R1）。轴对齐正方形/长方形不受影响
        （正方形顶点集在 180° 旋转下不变）。
    """
    if shape == "square":
        unit = _regular_polygon(4, start_deg=45.0)  # 轴对齐正方形
    elif shape == "rectangle":
        long_px, short_px = size_px if size_px is not None else (120.0, 60.0)
        if long_px <= 0 or short_px <= 0:
            raise ValueError(f"长方形尺寸必须为正: {size_px}")
        aspect = float(long_px) / float(short_px)
        unit = np.array(
            [[-0.5 * aspect, -0.5], [0.5 * aspect, -0.5], [0.5 * aspect, 0.5], [-0.5 * aspect, 0.5]]
        )
    elif shape == "triangle":
        unit = _regular_polygon(3, start_deg=-90.0)  # 尖朝上正三角形
    elif shape == "star":
        unit = _star_polygon(5)
    else:  # pragma: no cover - 参数层已限制
        raise ValueError(f"未知形状: {shape}")

    scale = float(np.sqrt(area_px2 / polygon_area(unit)))
    vertices = unit * scale
    if rotation_deg:
        theta = np.deg2rad(rotation_deg)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        vertices = vertices @ rot.T
    return vertices


def points_in_polygon(
    xx: npt.NDArray[np.floating],
    yy: npt.NDArray[np.floating],
    vertices: npt.NDArray[np.floating],
) -> npt.NDArray[np.bool_]:
    """偶奇交叉规则的多边形包含判定（向量化，任意简单多边形）。

    Args:
        xx: 待判定点的 x 坐标（图像 col），任意可与 ``yy`` 广播的形状。
        yy: 待判定点的 y 坐标（图像 row）。
        vertices: ``(N, 2)`` 顶点数组，列序 ``(x, y)``，按边界顺序排列。

    Returns:
        与 ``np.broadcast(xx, yy)`` 同形状的布尔数组。

    Note:
        算式与 ``patterns.create_target_shape`` 的三角/星形分支逐位一致
        （同一交叉代数、同一 ``1e-12`` 防护），因此 patterns 委托本函数后
        输出不变；本函数即 R1 收敛后的唯一光栅化包含判定。
    """
    inside = np.zeros(np.broadcast(xx, yy).shape, dtype=bool)
    n = len(vertices)
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        crosses = (y1 > yy) != (y2 > yy)
        x_cross = (x2 - x1) * (yy - y1) / (y2 - y1 + 1e-12) + x1
        inside ^= crosses & (xx < x_cross)
    return inside


def _erode(mask: npt.NDArray[np.bool_], iterations: int) -> npt.NDArray[np.bool_]:
    """四邻域二值腐蚀（Chebyshev 距离内缩），不依赖 scipy。"""
    out = mask.copy()
    for _ in range(int(iterations)):
        shifted = out.copy()
        shifted[1:, :] &= out[:-1, :]
        shifted[:-1, :] &= out[1:, :]
        shifted[:, 1:] &= out[:, :-1]
        shifted[:, :-1] &= out[:, 1:]
        shifted[0, :] = False
        shifted[-1, :] = False
        shifted[:, 0] = False
        shifted[:, -1] = False
        out = shifted
    return out
