"""CCD 图像旋转配准与整像素画布旋转的回归测试。

锁四件事：
1. 从面板 x/y 两轴实测位移估计旋转（含精确 ±90° 与非 90° 任意角）；
2. 角度符号约定与 :func:`rotate_image_about_center` 的内容旋转一致；
3. 退化/镜像/非正交轴对被拒绝，而不是给出看似合理的角度；
4. 输出整数画布、支持裁剪/填充并保持 dtype。
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.utils.image.axis_order import Point
from ao_shaping.utils.image.rotation import (
    AxisRotationError,
    estimate_axis_rotation,
    rotate_image_about_center,
    rotated_canvas_shape,
)


def _panel_vectors(
    delta_deg: float, mag_x: float = 20.0, mag_y: float = 20.0
) -> tuple[tuple[float, float], tuple[float, float]]:
    """面板 +x/+y 基矢在 CCD 中旋转 ``delta_deg`` 后的实测位移。"""
    c, s = np.cos(np.deg2rad(delta_deg)), np.sin(np.deg2rad(delta_deg))
    return (mag_x * c, mag_x * s), (-mag_y * s, mag_y * c)


@pytest.mark.parametrize(
    ("delta_deg", "expected_correction"),
    [(90.0, -90.0), (-90.0, 90.0), (180.0, -180.0)],
)
def test_exact_right_angle_estimate(
    delta_deg: float, expected_correction: float
) -> None:
    vx, vy = _panel_vectors(delta_deg, 17.0, 23.0)
    est = estimate_axis_rotation(vx, vy)
    assert est.correction_deg == pytest.approx(expected_correction, abs=1e-9)
    wrapped = (est.panel_to_ccd_deg - delta_deg + 180.0) % 360.0 - 180.0
    assert wrapped == pytest.approx(0.0, abs=1e-9)
    assert est.orthogonality_error_deg == pytest.approx(0.0, abs=1e-9)


def test_non_90_estimate_reports_angles_and_scale() -> None:
    vx, vy = _panel_vectors(37.0, 6.0, 15.0)
    est = estimate_axis_rotation(vx, vy)
    assert est.correction_deg == pytest.approx(-37.0, abs=1e-9)
    assert est.panel_x_angle_deg == pytest.approx(37.0, abs=1e-9)
    assert est.panel_y_angle_deg == pytest.approx(127.0, abs=1e-9)
    assert est.orthogonality_error_deg == pytest.approx(0.0, abs=1e-9)
    assert est.scale_ratio == pytest.approx(2.5)
    assert est.panel_x_magnitude_px == pytest.approx(6.0)
    assert est.panel_y_magnitude_px == pytest.approx(15.0)


def test_correction_sign_moves_measured_panel_x_to_columns() -> None:
    """面板 +x 实测向下 (+y) => 修正角 -90°，内容旋转后光斑到 +x。"""
    est = estimate_axis_rotation((0.0, 15.0), (-15.0, 0.0))
    assert est.correction_deg == pytest.approx(-90.0)

    img = np.zeros((41, 41), dtype=np.uint8)
    img[28, 20] = 255  # 中心 (20, 20) 下方 8 px，即面板 +x 方向
    out = rotate_image_about_center(img, est.correction_deg, order=0)
    assert np.unravel_index(int(np.argmax(out)), out.shape) == (20, 28)


def test_rejects_short_collinear_and_mirror_pairs() -> None:
    with pytest.raises(AxisRotationError, match="too small"):
        estimate_axis_rotation((0.0, 0.0), (10.0, 0.0))
    with pytest.raises(AxisRotationError, match="degenerate"):
        estimate_axis_rotation((10.0, 0.0), (20.0, 0.0))
    with pytest.raises(AxisRotationError, match="mirror"):
        estimate_axis_rotation((10.0, 0.0), (0.0, -10.0))
    # 轴互换也是镜像（det < 0），旋转无法同时对齐。
    with pytest.raises(AxisRotationError, match="mirror"):
        estimate_axis_rotation((0.0, 10.0), (10.0, 0.0))


def test_rejects_non_orthogonal_and_invalid_vectors() -> None:
    with pytest.raises(AxisRotationError, match="not orthogonal"):
        estimate_axis_rotation(
            (10.0, 0.0),
            (10.0 * np.cos(np.deg2rad(30.0)), 10.0 * np.sin(np.deg2rad(30.0))),
        )
    with pytest.raises(ValueError, match="2-element"):
        estimate_axis_rotation((1.0, 2.0, 3.0), (0.0, 1.0))
    with pytest.raises(AxisRotationError, match="finite"):
        estimate_axis_rotation((np.nan, 0.0), (1.0, 0.0))


@pytest.mark.parametrize(("angle_deg", "k"), [(90.0, -1), (-90.0, 1)])
def test_right_angle_rotation_matches_numpy_orientation(
    angle_deg: float, k: int
) -> None:
    img = np.arange(9, dtype=np.uint16).reshape(3, 3)
    out = rotate_image_about_center(img, angle_deg, order=0)
    np.testing.assert_array_equal(out, np.rot90(img, k=k))

    non_square = np.arange(12, dtype=np.uint16).reshape(3, 4)
    out_ns = rotate_image_about_center(non_square, angle_deg, order=0)
    assert out_ns.shape == (4, 3)
    np.testing.assert_array_equal(out_ns, np.rot90(non_square, k=k))


def test_rotated_canvas_shape_is_exact_at_right_angles() -> None:
    assert rotated_canvas_shape((3, 4), 0.0) == (3, 4)
    assert rotated_canvas_shape((3, 4), 90.0) == (4, 3)
    assert rotated_canvas_shape((3, 4), -90.0) == (4, 3)
    assert rotated_canvas_shape((3, 4), 180.0) == (3, 4)
    assert rotated_canvas_shape((5, 5), 45.0) == (7, 7)


def test_rotate_supports_pad_crop_and_preserves_dtype() -> None:
    img = np.full((20, 30), 7, dtype=np.uint16)
    padded = rotate_image_about_center(
        img, 30.0, output_shape=(30, 40), pad_value=1234, order=1
    )
    assert padded.shape == (30, 40)
    assert padded.dtype == np.uint16
    assert padded[0, 0] == 1234
    assert np.all(padded[10:20, 15:25] == 7)

    cropped = rotate_image_about_center(img, 30.0, output_shape=(10, 10), order=1)
    assert cropped.shape == (10, 10)
    assert np.all(cropped == 7)

    floats = np.full((10, 10), 1.5, dtype=np.float32)
    out = rotate_image_about_center(floats, 15.0, pad_value=-2.0, order=1)
    assert out.dtype == np.float32
    assert out.shape == rotated_canvas_shape((10, 10), 15.0)


def test_accepts_axis_order_point_per_issue_223() -> None:
    """#223: 位移矢量是项目点序 ``(x, y)``，``axis_order.Point`` 可直接传。"""
    assert estimate_axis_rotation(Point(0, 15), Point(-15, 0)).correction_deg == (
        pytest.approx(estimate_axis_rotation((0.0, 15.0), (-15.0, 0.0)).correction_deg)
    )
    p = Point(3, -4)
    assert (p.x, p.y) == (3, -4)


def test_non_square_rotation_keeps_point_axis_order() -> None:
    """非方帧上点序与数组形状序必须分开：#223 的转置可被这里抓到。"""
    height, width = 120, 320
    peak_row, peak_col = 60, 240  # Point(x=240, y=60)
    img = np.zeros((height, width), dtype=np.uint8)
    img[peak_row, peak_col] = 255

    out = rotate_image_about_center(img, 90.0, order=0)
    # +90° 内容旋转: (dx, dy) -> (-dy, dx)；中心取像素中心。
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    dx, dy = peak_col - cx, peak_row - cy
    out_cx, out_cy = (out.shape[1] - 1) / 2.0, (out.shape[0] - 1) / 2.0
    expected = (
        int(round(out_cy + dx)),
        int(round(out_cx - dy)),
    )
    assert np.unravel_index(int(np.argmax(out)), out.shape) == expected
    assert expected != expected[::-1]  # 非方帧: 转置后不在同一位置
