"""多形状目标掩膜（正方形 / 长方形 / 三角形 / 五角星），按**面积**归一。

为什么单独一个模块：非方形目标若仍用外接方框做 CV，框内会混进大量形状外的背景，
读数没有意义。这里只做**掩膜几何**（顶点 → 栅格化 → 内缩核心），不 import torch /
驱动 / GUI，供优化器、runner 与离线报告共用。

本模块从 ``ao_shaping.algorithm.signal_processing.target_masks`` 上移到
``ao_shaping.utils.image.target``（issue #199 P2：掩膜几何属于 target 层），
并与 :mod:`~ao_shaping.utils.image.target.geometry`（原语）、
:mod:`~ao_shaping.utils.image.target.patterns`（canonical 工厂）共同构成
形状→掩膜几何的**唯一**实现（issue #199 R1：曾经有 5 份）。
``differentiable_defocus_uniformity.build_masks`` 的方框几何由本模块的
:func:`build_box_core_masks` 接管；``hw_square_carrier_probe.square_target``
改为委托 ``patterns.create_target_shape``。

坐标约定（与相机栅格一致）：
* 顶点列序 ``(x, y)``，相对掩膜中心；
* 栅格下标 ``(row, col) = (y, x)``；
* ``step_ccd_px = (step_y, step_x)``：每个栅格格子在相机像元上的尺寸。

面积归一是**必须**的：不同形状只比较"面积相同"的目标才有意义（同一条零级能量
分到相同面积上）。因此本模块接受 ``area_px2``，把单位形状缩放到该面积。

几何口径（R1 收敛后的单一约定，像素级契约测试见
``tests/ao_shaping/utils/image/target/test_mask_factories_agree_pixelwise.py``）：
* 轴对齐正方形/长方形走**闭合盒**快路径（``[xmin, xmax] × [ymin, ymax]``），
  与 ``patterns.create_target_shape`` 的方盒逐位一致；三角/星形走
  :func:`~ao_shaping.utils.image.target.geometry.points_in_polygon`。
* 半边取**精确值**（相机像元 → 栅格单位后不再取整到整格）：请求 23 px、步长
  2 px 时半边是 5.75 格，而不是旧 ``build_masks`` 向上取整的 6 格。各向同性
  步长且小数部分 ≤ 0.5 时与旧实现一致（冻结前向的全部实际调用点均落在此）；
  取整会系统性放大方框（23 px 请求变成 26 px 盒）。
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ao_shaping.utils.image.target.geometry import (
    ShapeName,
    _erode,
    points_in_polygon,
    polygon_area,
    shape_vertices,
)
from ao_shaping.utils.image.target.metrics import POLYGON_SHAPE_CHOICES

__all__ = [
    "SHAPES",
    "ShapeName",
    "build_box_core_masks",
    "build_polygon_masks",
    "points_in_polygon",
    "polygon_area",
    "shape_vertices",
    "shift_mask",
]

#: 多边形族形状 —— canonical :data:`TARGET_SHAPE_CHOICES` 的文档化子集视图
#: （issue #199 P1），不是私有副本。transfer 侧的种子采集 / 模糊标定 /
#: 能力报告都只用这一族。
SHAPES: tuple[str, ...] = POLYGON_SHAPE_CHOICES
assert set(SHAPES) <= set(POLYGON_SHAPE_CHOICES)


def _axis_aligned_box(
    vertices: npt.NDArray[np.floating],
) -> tuple[float, float, float, float] | None:
    """顶点恰为外接矩形四角时返回 ``(xmin, xmax, ymin, ymax)``，否则 ``None``。

    ``shape_vertices`` 的正方形顶点按 45° 起步（菱形序），但顶点**集**就是外接
    矩形的四角，因此走闭合盒快路径；旋转过的矩形外接框严格大于多边形，落到
    PIP。三角（3 顶点）/星形（10 顶点）直接判长度排除。
    """
    if len(vertices) != 4:
        return None
    xs = vertices[:, 0]
    ys = vertices[:, 1]
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    corners = {(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)}
    if {(float(vx), float(vy)) for vx, vy in vertices} != corners:
        return None
    return xmin, xmax, ymin, ymax


def _rasterise(
    grid_shape: tuple[int, int],
    centre_ij: tuple[int, int],
    vertices: npt.NDArray[np.floating],
    *,
    inner_shrink_px: float,
    step_ccd_px: tuple[float, float],
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """顶点 → ``(box, core)`` 的唯一栅格化路径（方框与多边形共用）。"""
    col, row = centre_ij
    step_y, step_x = float(step_ccd_px[0]), float(step_ccd_px[1])
    yy, xx = np.mgrid[: grid_shape[0], : grid_shape[1]].astype(float)

    # 顶点从相机像元换算到栅格单位，再平移到中心
    verts = np.empty_like(vertices, dtype=float)
    verts[:, 0] = vertices[:, 0] / step_x + col
    verts[:, 1] = vertices[:, 1] / step_y + row

    bounds = _axis_aligned_box(verts)
    if bounds is not None:
        xmin, xmax, ymin, ymax = bounds
        box = (xx >= xmin) & (xx <= xmax) & (yy >= ymin) & (yy <= ymax)
    else:
        box = points_in_polygon(xx, yy, verts)
    shrink_cells = int(round(inner_shrink_px / (0.5 * (step_x + step_y))))
    core = _erode(box, max(1, shrink_cells)) if shrink_cells > 0 else box.copy()
    return box, core


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
        vertices: :func:`shape_vertices` 的输出（相机像元，相对中心）。
        inner_shrink_px: 核心掩膜相对外框的内缩量（相机像元），用于避开边缘软过渡。
        step_ccd_px: ``(step_y, step_x)``，每格对应多少相机像元。

    Returns:
        ``(box, core)``：``box`` 为整个形状，``core`` 为内缩后的形状（口径更严）。
    """
    return _rasterise(
        grid_shape, centre_ij, vertices,
        inner_shrink_px=inner_shrink_px, step_ccd_px=step_ccd_px,
    )


def build_box_core_masks(
    grid_shape: tuple[int, int],
    centre_ij: tuple[int, int],
    box_px: float,
    inner_shrink_px: float,
    step_ccd_px: tuple[float, float],
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """按 CCD 像元尺寸构造方框/核心掩膜（冻结前向的评估窗）。

    取代 ``differentiable_defocus_uniformity.build_masks``（issue #199 R1）：
    方框几何与 :func:`build_polygon_masks` 的轴对齐快路径是同一条代码路径
    （本函数只是构造四个角点后委托），半边按精确几何取（见模块 docstring）。

    Args:
        grid_shape: 采样窗口 ``(rows, cols)``。
        centre_ij: 方框中心在窗口内的栅格下标 ``(col, row)``。
        box_px: 方框边长（相机像元）。
        inner_shrink_px: 核心掩膜内缩量（相机像元）。
        step_ccd_px: ``(step_y, step_x)``，每格对应多少相机像元。

    Returns:
        ``(box, core)`` 布尔掩膜；``box`` 为闭合方框，``core`` 为内缩方框。
    """
    half = float(box_px) / 2.0
    vertices = np.array(
        [[-half, -half], [half, -half], [half, half], [-half, half]], dtype=float
    )
    return _rasterise(
        grid_shape, centre_ij, vertices,
        inner_shrink_px=inner_shrink_px, step_ccd_px=step_ccd_px,
    )


def shift_mask(mask: npt.NDArray[np.bool_], dy: int, dx: int) -> npt.NDArray[np.bool_]:
    """整体平移掩膜（用于载波把光斑挪走后在**新位置**取同一形状的读数）。"""
    return np.roll(np.roll(mask, int(dy), axis=0), int(dx), axis=1)
