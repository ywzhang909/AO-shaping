"""多形状目标掩膜（正方形 / 长方形 / 三角形 / 五角星），按**面积**归一。

为什么单独一个模块：`differentiable_defocus_uniformity.build_masks` 只支持方框，
而非方形目标若仍用外接方框做 CV，框内会混进大量形状外的背景，读数没有意义。
这里只做**掩膜几何**（顶点 → 栅格化 → 内缩核心），不 import torch / 驱动 / GUI，
供优化器、runner 与离线报告共用。

坐标约定（与相机栅格一致）：
* 顶点列序 ``(x, y)``，相对掩膜中心；
* 栅格下标 ``(row, col) = (y, x)``；
* ``step_ccd_px = (step_y, step_x)``：每个栅格格子在相机像元上的尺寸。

面积归一是**必须**的：不同形状只比较"面积相同"的目标才有意义（同一条零级能量
分到相同面积上）。因此本模块接受 ``area_px2``，把单位形状缩放到该面积。
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

ShapeName = Literal["square", "rectangle", "triangle", "star"]

SHAPES: tuple[str, ...] = ("square", "rectangle", "triangle", "star")


def _regular_polygon(n: int, *, start_deg: float = 90.0) -> npt.NDArray[np.floating]:
    """单位外接圆正多边形顶点，列序 ``(x, y)``，``start_deg`` 处为首个顶点。"""
    angles = np.deg2rad(start_deg + 360.0 * np.arange(n) / n)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def _star_polygon(n_points: int = 5) -> npt.NDArray[np.floating]:
    """单位外接圆五角星：外顶点与内顶点交替，内半径 = sin(18°)/sin(54°)。"""
    inner = np.sin(np.deg2rad(180.0 / n_points)) / np.sin(
        np.deg2rad(3.0 * 180.0 / (2.0 * n_points))
    )
    outer = _regular_polygon(n_points)
    inner_pts = inner * _regular_polygon(n_points, start_deg=90.0 + 180.0 / n_points)
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
        unit = _regular_polygon(3)  # 尖朝上正三角形
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
    """偶奇交叉规则的多边形包含判定（向量化，任意简单多边形）。"""
    inside = np.zeros(np.broadcast(xx, yy).shape, dtype=bool)
    n = len(vertices)
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(n):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % n]
            crosses = (y1 > yy) != (y2 > yy)
            x_cross = x1 + (yy - y1) * (x2 - x1) / (y2 - y1)
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


def build_polygon_masks(
    grid_shape: tuple[int, int],
    centre_ij: tuple[int, int],
    vertices: npt.NDArray[np.floating],
    *,
    inner_shrink_px: float,
    step_ccd_px: tuple[float, float],
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """把形状顶点栅格化成 ``(box, core)`` 两张布尔掩膜。

    Args:
        grid_shape: 采样窗口 ``(rows, cols)``。
        centre_ij: 形状中心在窗口内的栅格下标 ``(col, row)``。
        vertices: ``shape_vertices`` 的输出（相机像元，相对中心）。
        inner_shrink_px: 核心掩膜相对外框的内缩量（相机像元），用于避开边缘软过渡。
        step_ccd_px: ``(step_y, step_x)``，每格对应多少相机像元。

    Returns:
        ``(box, core)``：``box`` 为整个形状，``core`` 为内缩后的形状（口径更严）。
    """
    col, row = centre_ij
    step_y, step_x = float(step_ccd_px[0]), float(step_ccd_px[1])
    yy, xx = np.mgrid[: grid_shape[0], : grid_shape[1]].astype(float)

    # 顶点从相机像元换算到栅格单位，再平移到中心
    verts = np.empty_like(vertices, dtype=float)
    verts[:, 0] = vertices[:, 0] / step_x + col
    verts[:, 1] = vertices[:, 1] / step_y + row

    box = points_in_polygon(xx, yy, verts)
    shrink_cells = int(round(inner_shrink_px / (0.5 * (step_x + step_y))))
    core = _erode(box, max(1, shrink_cells)) if shrink_cells > 0 else box.copy()
    return box, core


def shift_mask(mask: npt.NDArray[np.bool_], dy: int, dx: int) -> npt.NDArray[np.bool_]:
    """整体平移掩膜（用于载波把光斑挪走后在**新位置**取同一形状的读数）。"""
    return np.roll(np.roll(mask, int(dy), axis=0), int(dx), axis=1)
