"""``utils.image.target.masks`` 的纯几何约定测试（无 torch / 无设备）。

这些断言锁的是**口径**：四种形状必须同面积、外接框与几何一致、核心掩膜严格
落在外框内、旋转只换外接框方向。口径一变，跨形状的 CV 就不可比了。
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.utils.image.target.masks import (
    SHAPES,
    build_polygon_masks,
    points_in_polygon,
    polygon_area,
    shape_vertices,
    shift_mask,
)

GRID = (201, 201)
CENTRE = (100, 100)  # (col, row)
STEP = (2.0, 2.0)


def _masks(shape: str, **kwargs):
    vertices = shape_vertices(shape, area_px2=7200.0, **kwargs)
    return build_polygon_masks(GRID, CENTRE, vertices, inner_shrink_px=3.0, step_ccd_px=STEP)


@pytest.mark.parametrize("shape", SHAPES)
def test_area_normalised_to_target(shape: str) -> None:
    vertices = shape_vertices(shape, area_px2=7200.0)
    assert polygon_area(vertices) == pytest.approx(7200.0, rel=1e-9)


@pytest.mark.parametrize(
    ("shape", "kwargs", "expect_w", "expect_h"),
    [
        ("square", {}, 86, 86),
        ("rectangle", {"size_px": (120.0, 60.0)}, 120, 60),
        ("triangle", {}, 126, 112),
        ("star", {}, 110, 106),
    ],
)
def test_bounding_box(shape: str, kwargs: dict, expect_w: int, expect_h: int) -> None:
    box, _ = _masks(shape, **kwargs)
    width = int(box.any(axis=0).sum()) * int(STEP[1])
    height = int(box.any(axis=1).sum()) * int(STEP[0])
    assert abs(width - expect_w) <= 3, (shape, width)
    assert abs(height - expect_h) <= 3, (shape, height)


@pytest.mark.parametrize("shape", SHAPES)
def test_core_inside_box_and_smaller(shape: str) -> None:
    box, core = _masks(shape)
    assert core.any(), f"{shape}: core 为空"
    assert not (core & ~box).any(), f"{shape}: core 有像素落在 box 外"
    assert core.sum() < box.sum()


@pytest.mark.parametrize("shape", SHAPES)
def test_rasterised_area_close_to_target(shape: str) -> None:
    """栅格化后的像元数 × 单格面积应逼近 7200 px²（±8%）。"""
    box, _ = _masks(shape)
    area_px2 = float(box.sum()) * float(STEP[0]) * float(STEP[1])
    assert area_px2 == pytest.approx(7200.0, rel=0.08), (shape, area_px2)


def test_rotation_swaps_bounding_box() -> None:
    box0, _ = _masks("rectangle", size_px=(120.0, 60.0))
    box90, _ = _masks("rectangle", size_px=(120.0, 60.0), rotation_deg=90.0)
    assert int(box0.any(axis=0).sum()) == int(box90.any(axis=1).sum())
    assert int(box0.any(axis=1).sum()) == int(box90.any(axis=0).sum())
    assert box0.sum() == pytest.approx(box90.sum(), rel=0.05)


def test_shift_mask_moves_by_exact_offset() -> None:
    box, _ = _masks("square")
    moved = shift_mask(box, 0, 10)
    assert np.array_equal(moved[:, 10:], box[:, :-10])
    back = shift_mask(moved, 0, -10)
    assert np.array_equal(back, box)


def test_points_in_polygon_basic() -> None:
    square = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    inside = points_in_polygon(np.array([0.0, 2.0]), np.array([0.0, 0.0]), square)
    assert inside.tolist() == [True, False]


def test_unknown_shape_and_bad_size_raise() -> None:
    with pytest.raises(ValueError):
        shape_vertices("hexagon")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        shape_vertices("rectangle", size_px=(0.0, 60.0))  # type: ignore[arg-type]
