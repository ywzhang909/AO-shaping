"""CCD 图像与 SLM 面板轴的旋转配准（纯 NumPy/SciPy）。

**唯一坐标/符号约定**（本模块所有角度共用）:

* 图像按 ``(row, col)`` 存储；取 ``x = col``（向右增），``y = row``（向下增）。
* 向量 ``(x, y)`` 的夹角为 ``atan2(y, x)``；因 y 向下，**正角在屏幕上表现为
  顺时针**（行 0 在顶部）。
* :func:`rotate_image_about_center` 的 ``angle_deg`` 是**内容**旋转：输入中从
  中心指向 ``v`` 的特征，在输出中位于 ``R(angle_deg) @ v``（同一 x/y 约定）。
* :func:`estimate_axis_rotation` 的 ``correction_deg`` 是"把实测的**面板 +x**
  与**面板 +y** 位移方向分别转到 CCD +x（列增）与 +y（行增）"所需的图像旋转
  角，可直接交给 :func:`rotate_image_about_center`。

为什么需要实测：面板轴与 CCD 轴之间隔着安装姿态（本系统 SLM 竖向安装，
面板 x 斜坡可在 CCD 上表现为 y 向位移）。若把 90° 写死，一旦安装角不是
正好 90°（或符号反了），后续所有坐标都带着固定偏差。这里只从两轴实测位移
矢量求旋转，并对退化/镜像的输入**报错**而不是给一个看似合理的角度。

轴序（issue #223 单一事实源）：本模块的位移矢量与点一律是项目点序
``(x, y)``（x = 列在前），可直接传
:class:`ao_shaping.utils.image.axis_order.Point`；数组**形状**仍按 numpy 原生
``(rows, cols)``。两种序只在显式边界经 ``rowcol_to_xy`` / ``xy_to_rowcol``
转换，绝不在同一接口里混用——非方帧上 ``(row, col)`` 与 ``(col, row)`` 都合法，
混用是静默转置。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.ndimage import map_coordinates

__all__ = [
    "ROTATION_CONVENTION",
    "AxisRotationError",
    "AxisRotationEstimate",
    "estimate_axis_rotation",
    "rotate_image_about_center",
    "rotated_canvas_shape",
]

#: 角度约定说明，写入分析元数据，保证结果与工具口径一致。
ROTATION_CONVENTION = (
    "CCD array coordinates: x=column (increasing right), y=row (increasing down); "
    "angles are atan2(y, x) so a positive angle rotates image content clockwise "
    "on screen (row 0 at top). correction_deg rotates measured panel +x to +x "
    "(columns) and panel +y to +y (rows)."
)

_RIGHT_ANGLE_EPS_DEG = 1e-9


class AxisRotationError(ValueError):
    """两轴位移对无法给出唯一旋转（退化、镜像或非正交）。"""


@dataclass(frozen=True)
class AxisRotationEstimate:
    """从面板 x/y 位移矢量估计出的旋转。

    Attributes:
        correction_deg: 应施加到 CCD 图像上的**内容**旋转角（度），约定见
            :data:`ROTATION_CONVENTION`。
        panel_to_ccd_deg: 面板轴到 CCD 轴的旋转角，等于 ``-correction_deg``。
        panel_x_angle_deg: 面板 +x 位移矢量在 CCD 中的绝对角度。
        panel_y_angle_deg: 面板 +y 位移矢量在 CCD 中的绝对角度。
        panel_x_magnitude_px: 面板 +x 位移长度（CCD 像素）。
        panel_y_magnitude_px: 面板 +y 位移长度（CCD 像素）。
        orthogonality_error_deg: 两矢量夹角相对 90° 的偏差。
        scale_ratio: 两轴位移长度的较大/较小比（只记录，不作判据）。
        max_residual_px: 各向同性旋转+缩放模型对两矢量的最大拟合残差。
    """

    correction_deg: float
    panel_to_ccd_deg: float
    panel_x_angle_deg: float
    panel_y_angle_deg: float
    panel_x_magnitude_px: float
    panel_y_magnitude_px: float
    orthogonality_error_deg: float
    scale_ratio: float
    max_residual_px: float


def _wrap_180(angle_deg: float) -> float:
    """把角度折到 ``[-180, 180)``。"""
    return float((float(angle_deg) + 180.0) % 360.0 - 180.0)


def _cos_sin_deg(angle_deg: float) -> tuple[float, float]:
    """角度的 (cos, sin)；直角处吸附为精确值，保证 ±90°/180° 无插值残差。"""
    angle = float(angle_deg)
    if not np.isfinite(angle):
        raise ValueError(f"angle_deg must be finite, got {angle_deg!r}")
    quarter = round(angle / 90.0)
    if abs(angle - quarter * 90.0) <= _RIGHT_ANGLE_EPS_DEG:
        table = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))
        return table[int(quarter) % 4]
    rad = np.deg2rad(angle)
    return float(np.cos(rad)), float(np.sin(rad))


def _as_vector(name: str, value: Sequence[float] | npt.ArrayLike) -> np.ndarray:
    vec = np.asarray(value, dtype=np.float64)
    if vec.shape != (2,):
        raise ValueError(
            f"{name} must be a 2-element (x, y) vector, got shape {vec.shape}"
        )
    if not np.all(np.isfinite(vec)):
        raise AxisRotationError(f"{name} must be finite, got {vec.tolist()}")
    return vec


def _validate_shape(shape: Sequence[int]) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError(f"shape must be (height, width), got {shape!r}")
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"shape must be positive, got {shape!r}")
    return height, width


def _ceil_int(value: float) -> int:
    # 减 1e-9 防止精确整数被浮点误差推到下一个整数。
    return int(np.ceil(float(value) - 1e-9))


def estimate_axis_rotation(
    panel_x_delta_xy: Sequence[float] | npt.ArrayLike,
    panel_y_delta_xy: Sequence[float] | npt.ArrayLike,
    *,
    min_magnitude_px: float = 0.5,
    min_sine: float = 0.2,
    max_orthogonality_error_deg: float = 45.0,
) -> AxisRotationEstimate:
    """从面板 x/y 两轴的实测 CCD 位移矢量估计图像旋转角。

    Args:
        panel_x_delta_xy: 面板 +x 斜坡在 CCD 上产生的目标位移 ``(dx, dy)``，
            ``x`` 在前（项目点序）；可直接传
            :class:`~ao_shaping.utils.image.axis_order.Point`。
        panel_y_delta_xy: 面板 +y 斜坡产生的目标位移 ``(dx, dy)``，同样 ``x`` 在前。
        min_magnitude_px: 单轴位移长度下限；更短视为无法定向。
        min_sine: 两矢量夹角正弦下限；低于此判定共线/退化。
        max_orthogonality_error_deg: 夹角偏离 90° 的容限；超过即拒绝。

    Returns:
        :class:`AxisRotationEstimate`。

    Raises:
        AxisRotationError: 矢量非有限、过短、共线、镜像（左手系）或非正交。
        ValueError: 矢量形状不是 ``(2,)``。
    """
    vx = _as_vector("panel_x_delta_xy", panel_x_delta_xy)
    vy = _as_vector("panel_y_delta_xy", panel_y_delta_xy)
    mag_x = float(np.hypot(vx[0], vx[1]))
    mag_y = float(np.hypot(vy[0], vy[1]))
    if mag_x < min_magnitude_px:
        raise AxisRotationError(
            f"panel x displacement too small: {mag_x:.3f} px < {min_magnitude_px} px"
        )
    if mag_y < min_magnitude_px:
        raise AxisRotationError(
            f"panel y displacement too small: {mag_y:.3f} px < {min_magnitude_px} px"
        )

    dot = float(vx[0] * vy[0] + vx[1] * vy[1])
    cross = float(vx[0] * vy[1] - vx[1] * vy[0])
    sin_ab = cross / (mag_x * mag_y)
    cos_ab = dot / (mag_x * mag_y)
    if abs(sin_ab) < min_sine:
        raise AxisRotationError(
            "degenerate axis pair: panel x/y displacements are collinear "
            f"(|sin|={abs(sin_ab):.3f} < {min_sine})"
        )
    if cross <= 0.0:
        # 纯旋转+缩放 det = s^2 > 0；负叉积说明两轴被镜像/互换，旋转无法修复。
        raise AxisRotationError(
            "mirror axis pair: panel x/y displacements are left-handed "
            f"(cross={cross:.3f} <= 0); no rotation can align both axes"
        )

    angle_ab_deg = float(np.degrees(np.arctan2(abs(sin_ab), cos_ab)))
    orthogonality_error = abs(angle_ab_deg - 90.0)
    if orthogonality_error > max_orthogonality_error_deg:
        raise AxisRotationError(
            "panel x/y displacements are not orthogonal: "
            f"angle={angle_ab_deg:.2f} deg, error={orthogonality_error:.2f} deg "
            f"> {max_orthogonality_error_deg} deg"
        )

    # 2D Kabsch/Procrustes: 面板基 (e_x, e_y) -> 实测 (vx, vy) 的最优旋转。
    delta = float(np.arctan2(vx[1] - vy[0], vx[0] + vy[1]))
    cos_d, sin_d = float(np.cos(delta)), float(np.sin(delta))
    scale = 0.5 * ((vx[0] * cos_d + vx[1] * sin_d) + (-vy[0] * sin_d + vy[1] * cos_d))
    pred_x = scale * np.array([cos_d, sin_d], dtype=np.float64)
    pred_y = scale * np.array([-sin_d, cos_d], dtype=np.float64)
    max_residual = float(max(np.linalg.norm(vx - pred_x), np.linalg.norm(vy - pred_y)))

    return AxisRotationEstimate(
        correction_deg=_wrap_180(-np.degrees(delta)),
        panel_to_ccd_deg=_wrap_180(np.degrees(delta)),
        panel_x_angle_deg=_wrap_180(np.degrees(np.arctan2(vx[1], vx[0]))),
        panel_y_angle_deg=_wrap_180(np.degrees(np.arctan2(vy[1], vy[0]))),
        panel_x_magnitude_px=mag_x,
        panel_y_magnitude_px=mag_y,
        orthogonality_error_deg=orthogonality_error,
        scale_ratio=max(mag_x, mag_y) / min(mag_x, mag_y),
        max_residual_px=max_residual,
    )


def rotated_canvas_shape(shape: Sequence[int], angle_deg: float) -> tuple[int, int]:
    """覆盖整幅旋转内容所需的整数画布 ``(height, width)``。

    以像素中心为旋转中心；整 90° 倍数下结果与原图精确换轴/不变。
    这是 numpy 的**数组形状**序（``(rows, cols)``），不是
    :class:`~ao_shaping.utils.image.axis_order.Point` 的点序。
    """
    height, width = _validate_shape(shape)  # AXIS: numpy array shape (rows, cols)
    cos_a, sin_a = _cos_sin_deg(angle_deg)
    half_h = 0.5 * (height - 1)
    half_w = 0.5 * (width - 1)
    out_h = _ceil_int(2.0 * (abs(sin_a) * half_w + abs(cos_a) * half_h)) + 1
    out_w = _ceil_int(2.0 * (abs(cos_a) * half_w + abs(sin_a) * half_h)) + 1
    return max(1, out_h), max(1, out_w)


def rotate_image_about_center(
    image: npt.ArrayLike,
    angle_deg: float,
    *,
    output_shape: Sequence[int] | None = None,
    pad_value: float = 0.0,
    order: int = 3,
    prefilter: bool = True,
) -> np.ndarray:
    """绕图像中心旋转 2D 图像，返回像素中心对齐的整数画布。

    语义（与模块头一致）：内容旋转 ``angle_deg``，即输入中心指向 ``v`` 的
    特征在输出中位于 ``R(angle_deg) @ v``。像素中心采用 ``(size - 1) / 2``，
    因此整 90° 倍数下无半像素偏移。

    Args:
        image: 2D 输入。
        angle_deg: 内容旋转角（度），约定见 :data:`ROTATION_CONVENTION`。
        output_shape: 请求的 ``(height, width)``（numpy 数组形状序，不是点序）；默认
            :func:`rotated_canvas_shape`。小于默认画布时从中心裁剪，大于时用
            ``pad_value`` 填充。
        pad_value: 采样落在输入外时的填充值（整数 dtype 会四舍五入并裁剪）。
        order: 插值阶数 0..5（0=最近邻，1=双线性，默认 3=三次样条）。
        prefilter: 传给 ``scipy.ndimage.map_coordinates`` 的样条预滤波开关。

    Returns:
        与输入 dtype 相同的新数组，形状为 ``output_shape``（整数像素）。

    Raises:
        ValueError: 非 2D、``output_shape`` 非法或角度非有限。
        TypeError: 非数值/布尔 dtype。
    """
    src = np.asarray(image)
    if src.ndim != 2:
        raise ValueError(f"image must be 2D, got shape {src.shape}")
    order_int = int(order)
    if not 0 <= order_int <= 5:
        raise ValueError(f"order must be in 0..5, got {order!r}")

    in_h, in_w = _validate_shape(src.shape)
    out_h, out_w = (
        rotated_canvas_shape(src.shape, angle_deg)
        if output_shape is None
        else _validate_shape(output_shape)
    )
    cos_a, sin_a = _cos_sin_deg(angle_deg)

    # 输出像素中心相对输出中心 -> 输入坐标（pull-back: R(-angle)）。
    x_out = np.arange(out_w, dtype=np.float64) - 0.5 * (out_w - 1)
    y_out = np.arange(out_h, dtype=np.float64) - 0.5 * (out_h - 1)
    cx = 0.5 * (in_w - 1)
    cy = 0.5 * (in_h - 1)
    # 广播到 (out_h, out_w)，避免再建两份完整 meshgrid。
    x_in = cx + cos_a * x_out[None, :] + sin_a * y_out[:, None]
    y_in = cy - sin_a * x_out[None, :] + cos_a * y_out[:, None]

    is_bool = src.dtype == np.bool_
    if is_bool:
        work = src.astype(np.float64)
    elif np.issubdtype(src.dtype, np.integer) or np.issubdtype(src.dtype, np.floating):
        work = src.astype(np.float64)
    else:
        raise TypeError(f"unsupported image dtype: {src.dtype}")

    sampled = map_coordinates(
        work,
        [y_in, x_in],
        order=order_int,
        mode="constant",
        cval=float(pad_value),
        prefilter=bool(prefilter),
    )

    if is_bool:
        return sampled >= 0.5
    if np.issubdtype(src.dtype, np.integer):
        info = np.iinfo(src.dtype)
        return np.clip(np.rint(sampled), info.min, info.max).astype(src.dtype)
    return sampled.astype(src.dtype)
