"""形状→掩膜几何的像素级契约（issue #199 R1 收敛测试）。

五个历史实现（``patterns`` / ``differentiable_shaping`` /
``target_masks`` / ``differentiable_defocus_uniformity.build_masks`` /
``hw_square_carrier_probe.square_target``）收敛为一套几何后，这些断言锁住
**跨工厂的像素一致性**与**新口径**：

* 轴对齐正方形/长方形：``masks`` 快路径、``build_box_core_masks`` 与
  ``patterns.create_target_shape`` 的方盒**逐位一致**（闭合盒）；
* 三角/星形：栅格化面积逼近解析面积，尖朝上（与 ``patterns`` 同向）；
* ``patterns`` 三角/星形委托 ``geometry.points_in_polygon`` 后输出不变
  （以委托前的内联算式做孪生比对）；
* 方框半边取精确几何（不再向上取整到整格）——这是相对旧 ``build_masks``
  唯一的行为差异，在此显式钉住。

纯 numpy，无 torch / 无设备。
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.utils.image.target.masks import (
    build_box_core_masks,
    build_polygon_masks,
    shape_vertices,
)
from ao_shaping.utils.image.target.patterns import create_target_shape

GRID = (201, 201)
CENTRE = (100, 100)  # (col, row)
STEP = (2.0, 2.0)


def _as_bool(pattern: np.ndarray) -> np.ndarray:
    return np.asarray(pattern) > 0.5


# --------------------------------------------------------------------------- #
# 轴对齐盒：三个入口必须逐位一致
# --------------------------------------------------------------------------- #
def test_square_box_three_factories_agree_pixelwise() -> None:
    """80 px 方框：build_box_core_masks ≡ build_polygon_masks ≡ patterns。"""
    box, _ = build_box_core_masks(GRID, CENTRE, box_px=80.0, inner_shrink_px=0.0, step_ccd_px=STEP)
    vertices = np.array([[-40.0, -40.0], [40.0, -40.0], [40.0, 40.0], [-40.0, 40.0]])
    box_poly, _ = build_polygon_masks(GRID, CENTRE, vertices, inner_shrink_px=0.0, step_ccd_px=STEP)
    pattern = create_target_shape("square", GRID, side=40.0, center=(100.0, 100.0))

    assert np.array_equal(box, box_poly)
    assert np.array_equal(box, _as_bool(pattern))
    assert int(box.sum()) == 41 * 41  # 闭合盒：|-20..20| 全含


def test_rectangle_box_matches_patterns() -> None:
    """120×60 长方形（面积 7200）：shape_vertices 快路径 ≡ patterns 方盒。"""
    vertices = shape_vertices("rectangle", area_px2=7200.0, size_px=(120.0, 60.0))
    box, _ = build_polygon_masks(GRID, CENTRE, vertices, inner_shrink_px=0.0, step_ccd_px=STEP)
    # patterns 的 rectangle：side 是短边，长边 = side * aspect_ratio
    pattern = create_target_shape(
        "rectangle", GRID, side=30.0, aspect_ratio=2.0, center=(100.0, 100.0)
    )
    assert np.array_equal(box, _as_bool(pattern))
    assert int(box.sum()) == 61 * 31


def test_area_normalised_square_matches_patterns_too() -> None:
    """真实入口：shape_vertices("square", 6400) 与 patterns side=40 同像素。"""
    vertices = shape_vertices("square", area_px2=6400.0)
    box, _ = build_polygon_masks(GRID, CENTRE, vertices, inner_shrink_px=0.0, step_ccd_px=STEP)
    pattern = create_target_shape("square", GRID, side=40.0, center=(100.0, 100.0))
    assert np.array_equal(box, _as_bool(pattern))


def test_box_core_masks_equals_polygon_path_for_same_square() -> None:
    """R1 的核心断言：方框几何只有一条代码路径（含 core 内缩）。"""
    box_a, core_a = build_box_core_masks(GRID, CENTRE, box_px=80.0, inner_shrink_px=3.0, step_ccd_px=STEP)
    vertices = shape_vertices("square", area_px2=6400.0)
    box_b, core_b = build_polygon_masks(
        GRID, CENTRE, vertices, inner_shrink_px=3.0, step_ccd_px=STEP
    )
    assert np.array_equal(box_a, box_b)
    assert np.array_equal(core_a, core_b)
    # shrink = round(3 px / 2 px) = 2 格 ⇒ 41 - 2*2 = 37
    assert int(core_a.sum()) == 37 * 37


def test_closed_box_includes_both_boundary_columns() -> None:
    """闭合盒口径（patterns 约定）：旧 even-odd PIP 是半开盒，边界列只含一侧。"""
    box, _ = build_box_core_masks(GRID, CENTRE, box_px=80.0, inner_shrink_px=0.0, step_ccd_px=STEP)
    # 中心行/列上，±20 格的边界点都在盒内（半开盒会丢掉一侧）
    assert box[100, 80] and box[100, 120]
    assert box[80, 100] and box[120, 100]
    # 边界列本身也是满的 41 格（不是只有中心那一行）
    assert int(box[:, 80].sum()) == 41 and int(box[:, 120].sum()) == 41
    assert int(box[80, :].sum()) == 41 and int(box[120, :].sum()) == 41


def test_box_half_is_exact_geometry_not_snapped_up() -> None:
    """精确几何口径：23 px @ 2 px/格 ⇒ 半边 5.75 格（11 格宽），旧实现取整到 6（13 格）。"""
    box, _ = build_box_core_masks((201, 201), CENTRE, box_px=23.0, inner_shrink_px=0.0, step_ccd_px=STEP)
    assert int(box.any(axis=0).sum()) == 11
    assert int(box.any(axis=1).sum()) == 11


def test_box_core_masks_anisotropic_steps_are_per_axis_exact() -> None:
    """各向异性步长按轴取精确半边（旧实现用平均步长单一取整，两轴同宽）。"""
    box, _ = build_box_core_masks(
        GRID, CENTRE, box_px=80.0, inner_shrink_px=0.0, step_ccd_px=(2.0, 4.0)
    )
    assert int(box.any(axis=0).sum()) == 21  # x: 40/4 = 10 格 ⇒ 21 列
    assert int(box.any(axis=1).sum()) == 41  # y: 40/2 = 20 格 ⇒ 41 行


# --------------------------------------------------------------------------- #
# 三角/星形：解析面积 + 尖朝上 + patterns 孪生比对
# --------------------------------------------------------------------------- #
def test_triangle_rasterised_area_matches_analytic() -> None:
    """等边三角形面积 (3√3/4)R²，R = sqrt(4A/(3√3))；栅格化后 ±2%。"""
    vertices = shape_vertices("triangle", area_px2=7200.0)
    box, _ = build_polygon_masks(GRID, CENTRE, vertices, inner_shrink_px=0.0, step_ccd_px=STEP)
    radius = float(np.sqrt(4.0 * 7200.0 / (3.0 * np.sqrt(3.0))))
    analytic = (3.0 * np.sqrt(3.0) / 4.0) * radius**2
    rasterised = float(box.sum()) * STEP[0] * STEP[1]
    assert rasterised == pytest.approx(analytic, rel=0.02), (rasterised, analytic)


def test_triangle_and_star_are_point_up() -> None:
    """尖朝上（图像坐标）：最顶行在中轴附近；尖朝下时顶点会贴底行。"""
    for shape in ("triangle", "star"):
        vertices = shape_vertices(shape, area_px2=7200.0)
        assert float(vertices[np.argmin(vertices[:, 1]), 0]) == pytest.approx(0.0, abs=1e-9)
        box, _ = build_polygon_masks(GRID, CENTRE, vertices, inner_shrink_px=0.0, step_ccd_px=STEP)
        rows = np.flatnonzero(box.any(axis=1))
        top_cols = np.flatnonzero(box[rows[0]])
        assert abs(float(top_cols.mean()) - CENTRE[0]) <= 3.0, shape


def test_star_rasterised_area_matches_analytic() -> None:
    """五角星（交替半径十边形）面积 5·R·r·sin(36°)；栅格化后 ±2%。"""
    vertices = shape_vertices("star", area_px2=7200.0)
    box, _ = build_polygon_masks(GRID, CENTRE, vertices, inner_shrink_px=0.0, step_ccd_px=STEP)
    outer = float(np.hypot(*vertices[0]))
    inner = float(np.hypot(*vertices[1]))
    analytic = 5.0 * outer * inner * float(np.sin(np.deg2rad(36.0)))
    rasterised = float(box.sum()) * STEP[0] * STEP[1]
    assert rasterised == pytest.approx(analytic, rel=0.02), (rasterised, analytic)


def test_patterns_triangle_star_unchanged_by_delegation() -> None:
    """孪生比对：patterns 委托前的内联交叉算式 vs 委托后的共享原语，逐位一致。"""
    for shape in ("triangle", "star"):
        height = width = 121
        # radius 的推导必须与 patterns 内部同算式（(ratio * min) / 2），
        # 否则浮点末位差会让边界像素翻转，孪生比对失去意义。
        radius_ratio = 40.0 / 60.5
        radius = radius_ratio * min(height, width) / 2
        y, x = np.mgrid[0:height, 0:width]
        cx = cy = 60.0
        if shape == "triangle":
            angles = np.deg2rad([-90.0, 30.0, 150.0])
            radii = np.full(3, radius)
        else:
            angles = np.deg2rad(-90.0 + np.arange(10) * 36.0)
            radii = np.where(np.arange(10) % 2 == 0, radius, radius * 0.42)
        vertices = np.column_stack((cx + radii * np.cos(angles), cy + radii * np.sin(angles)))
        # --- 委托前的内联实现（verbatim from patterns, pre-R1） ---
        legacy = np.zeros((height, width), dtype=bool)
        for (x0, y0), (x1, y1) in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
            crossing = (y0 > y) != (y1 > y)
            edge_x = (x1 - x0) * (y - y0) / (y1 - y0 + 1e-12) + x0
            legacy ^= crossing & (x < edge_x)
        pattern = create_target_shape(shape, (height, width), radius_ratio=radius_ratio)
        assert np.array_equal(_as_bool(pattern), legacy), shape
