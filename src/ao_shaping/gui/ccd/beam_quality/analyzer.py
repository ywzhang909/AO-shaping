"""光束质量逐帧分析算法 (纯 numpy/scipy, 不依赖 Qt)。

本模块只做数值计算, 供 ``beam_quality`` 的 Qt 界面在 UI 线程里逐帧调用,
不涉及采集/线程/显示, 因此可以离线单测。

分析项:

* 亮度统计: 峰值、均值、标准差、传感器饱和像素占比 (达到生效满量程);
* 质心: 扣除中值背景后的一阶矩 (过中心剖面的默认中心口径取椭圆拟合中心,
  见 :data:`CENTER_METHODS`);
* 包围椭圆: 以 ``背景 + 0.5 * (峰值 - 背景)`` 二值化, 取最大连通域,
  用该连通域的二阶矩得到椭圆的长短半轴与倾角;
* X/Y 过中心剖面 (默认取 ±1 像素的均值以抑制噪声);
* 剖面的一维高斯拟合 (失败返回 ``None``, 从不抛异常)。

坐标约定: ``(cx, cy)`` 与 ``angle_deg`` 都在**图像坐标**下 (x 向右, y 向下),
角度从 +x 轴量起, 逆时针为正 (在 y 向下的屏幕上即"顺时针为正", 与
``QPainter.rotate`` 的方向一致, 可直接用于叠加绘制)。

椭圆口径: 阈值取在半高 (相对峰值) 处, 因此二阶矩给出的半轴就是**半高全宽
半轴** —— 对高斯光束 ``a = 1.1774 * sigma``, 即 ``fwhm_major = 2 * a = 2.3548 * sigma``。
均匀填充椭圆满足 ``var = a**2 / 4``, 故 ``a = 2 * sqrt(var)``。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger
from scipy import ndimage
from scipy.optimize import least_squares

# 半高阈值对应的半轴/高斯 sigma 之比 sqrt(2*ln2)
HALF_MAX_TO_SIGMA = float(np.sqrt(2.0 * np.log(2.0)))
# 全宽 (FWHM) / sigma
FWHM_TO_SIGMA = 2.0 * HALF_MAX_TO_SIGMA

#: 逐帧分析降本: 大帧的中值背景用 4 像素步进抽样 (全幅 median 的 partition
#: 是 UI 线程热点); 小帧 (< 20 万像素) 仍走精确全幅中值, 保持既有测试口径。
MEDIAN_STRIDE_MIN_PIXELS = 200_000
MEDIAN_STRIDE = 4

#: 可选的中心计算方法: (键, 界面显示名)。
#: 默认 ``"ellipse"`` (椭圆拟合中心, 与 0 级中心默认口径一致); 其余为操作员
#: 可选的替代口径。四者都来自本模块的纯 numpy/scipy 计算 —— ``utils/image``
#: 提供了质心与峰值 (``locate_peak``) 同口径原语, 但椭圆拟合中心只存在于
#: :func:`_ellipse_from_mask`; 且引入 ``utils/image`` 会连带把
#: ``spots_calc`` 的 cuBLAS 预载链拉进 CCD GUI, 故统一在此维护。
CENTER_METHODS: tuple[tuple[str, str], ...] = (
    ("centroid", "质心"),
    ("box_centroid", "框内质心"),
    ("ellipse", "椭圆拟合中心"),
    ("peak", "最大值"),
)
DEFAULT_CENTER_METHOD = "ellipse"

#: 四法一致性容差 (px)。来源: 2026-11-23 四法对账中两两最大差恰为 3.0 px,
#: 等于既定容差, 且 ``ellipse_centre_usable``。
CENTER_SPREAD_TOLERANCE_PX = 3.0


@dataclass
class EllipseParams:
    """包围椭圆 (半高边界) 参数, 单位为原始图像像素。"""

    center: tuple[float, float]
    semi_major: float
    semi_minor: float
    angle_deg: float

    @property
    def fwhm_major(self) -> float:
        """长轴方向的全宽 (2 * 半轴)。"""
        return 2.0 * self.semi_major

    @property
    def fwhm_minor(self) -> float:
        """短轴方向的全宽 (2 * 半轴)。"""
        return 2.0 * self.semi_minor


@dataclass
class GaussianFit:
    """一维高斯拟合结果 (``a * exp(-(x-c)^2 / (2 s^2)) + b``)。"""

    amplitude: float
    center: float
    sigma: float
    offset: float
    r_squared: float

    @property
    def fwhm(self) -> float:
        """由拟合 sigma 换算的全宽。"""
        return FWHM_TO_SIGMA * self.sigma


@dataclass
class FrameMetrics:
    """一帧图像的完整分析结果。"""

    max_value: float
    mean_value: float
    std_value: float
    sat_ratio: float
    cx: float
    cy: float
    ellipse: EllipseParams | None
    x_profile: np.ndarray = field(repr=False)
    y_profile: np.ndarray = field(repr=False)
    x_fit: GaussianFit | None = None
    y_fit: GaussianFit | None = None
    # 剖面/坐标相对原始像素的放大倍数 (2x2 binning 后分析时为 2.0)
    pixel_scale: float = 1.0
    width: int = 0
    height: int = 0
    sat_threshold: float = 255.0
    centers: dict[str, tuple[float, float]] = field(default_factory=dict)
    center_spread_px: float | None = None

    def as_scalars(self) -> dict[str, float]:
        """趋势缓冲需要的标量字段。"""
        ellipse = self.ellipse
        return {
            "max": float(self.max_value),
            "mean": float(self.mean_value),
            "std": float(self.std_value),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "sat_ratio": float(self.sat_ratio),
            # 无椭圆 (全黑/纯背景) 时半轴缺失, 用 NaN 让趋势图跳过该点
            "semi_major": (
                float(ellipse.semi_major) if ellipse is not None else float("nan")
            ),
            "semi_minor": (
                float(ellipse.semi_minor) if ellipse is not None else float("nan")
            ),
        }

    def rescaled(self, scale: float) -> FrameMetrics:
        """把镜像/降采样分析得到的坐标换算回原始像素。

        剖面数组本身保持降采样后的长度 (其 x 轴按 ``pixel_scale`` 解释),
        质心、椭圆与拟合中心/sigma 乘 ``scale``。
        """

        if scale == 1.0:
            return self
        ellipse = self.ellipse
        scaled_ellipse = (
            None
            if ellipse is None
            else EllipseParams(
                center=(ellipse.center[0] * scale, ellipse.center[1] * scale),
                semi_major=ellipse.semi_major * scale,
                semi_minor=ellipse.semi_minor * scale,
                angle_deg=ellipse.angle_deg,
            )
        )
        return FrameMetrics(
            max_value=self.max_value,
            mean_value=self.mean_value,
            std_value=self.std_value,
            sat_ratio=self.sat_ratio,
            cx=self.cx * scale,
            cy=self.cy * scale,
            ellipse=scaled_ellipse,
            x_profile=self.x_profile,
            y_profile=self.y_profile,
            x_fit=_rescale_fit(self.x_fit, scale),
            y_fit=_rescale_fit(self.y_fit, scale),
            pixel_scale=self.pixel_scale * scale,
            width=int(round(self.width * scale)),
            height=int(round(self.height * scale)),
            sat_threshold=self.sat_threshold,
            centers={
                method: (center[0] * scale, center[1] * scale)
                for method, center in self.centers.items()
            },
            center_spread_px=(
                None
                if self.center_spread_px is None
                else self.center_spread_px * scale
            ),
        )


def _rescale_fit(fit: GaussianFit | None, scale: float) -> GaussianFit | None:
    if fit is None or scale == 1.0:
        return fit
    return GaussianFit(
        amplitude=fit.amplitude,
        center=fit.center * scale,
        sigma=fit.sigma * scale,
        offset=fit.offset,
        r_squared=fit.r_squared,
    )


def gaussian_1d(
    x: np.ndarray, amplitude: float, center: float, sigma: float, offset: float
) -> np.ndarray:
    """一维高斯函数 (拟合模型)。"""
    return amplitude * np.exp(-((x - center) ** 2) / (2.0 * sigma**2)) + offset


def _gaussian_1d_jac(
    x: np.ndarray, amplitude: float, center: float, sigma: float, offset: float
) -> np.ndarray:
    """``gaussian_1d`` 对四个参数的解析雅可比 (避免最小二乘的 4 次数值差分)。"""
    d = x - center
    base = np.exp(-(d**2) / (2.0 * sigma**2))
    return np.column_stack(
        (
            base,
            amplitude * base * d / sigma**2,
            amplitude * base * d**2 / sigma**3,
            np.ones_like(x),
        )
    )


def fit_profile(x: np.ndarray, y: np.ndarray) -> GaussianFit | None:
    """对一条剖面做高斯拟合; 失败返回 ``None``, 不抛异常。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 5 or x.size != y.size:
        return None
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return None
    offset0 = float(np.min(y))
    amplitude0 = float(np.max(y) - offset0)
    if amplitude0 <= 0.0:
        return None
    center0 = float(x[int(np.argmax(y))])
    sigma0 = float((x.max() - x.min()) / 6.0)
    if not np.isfinite(sigma0) or sigma0 <= 0.0:
        sigma0 = 1.0
    x_min, x_max = float(x.min()), float(x.max())
    if x_max <= x_min:
        return None
    try:
        result = least_squares(
            lambda p: gaussian_1d(x, *p) - y,
            [amplitude0, center0, sigma0, offset0],
            bounds=(
                [0.0, x_min, 1e-3, -np.inf],
                [np.inf, x_max, max(1e-3, (x_max - x_min)), np.inf],
            ),
            max_nfev=4000,
            jac=lambda p: _gaussian_1d_jac(x, *p),
        )
        if not result.success:
            raise RuntimeError(result.message)
        popt = result.x
    # policy: 拟合失败 = 本次没有该测量值, 返回 None 让上层隐藏标签, 而不是中断整个分析
    except Exception as exc:
        logger.debug("高斯拟合失败: {}", exc)
        return None

    fitted = gaussian_1d(x, *popt)
    residual = float(np.sum((y - fitted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - residual / total if total > 0.0 else 0.0
    if not np.all(np.isfinite(popt)):
        return None
    return GaussianFit(
        amplitude=float(popt[0]),
        center=float(popt[1]),
        sigma=float(popt[2]),
        offset=float(popt[3]),
        r_squared=float(r_squared),
    )


def _as_2d(img: np.ndarray) -> np.ndarray:
    """取单通道 2D 视图 (多通道输入取第一个通道), 转 float32。"""
    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"期望 2D 灰度图, 实际 shape={arr.shape}")
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return arr


def _background_estimate(arr: np.ndarray) -> float:
    """背景中值: 大帧抽样, 小帧精确。抽样只影响亚 count 级背景估计。"""
    if arr.size >= MEDIAN_STRIDE_MIN_PIXELS:
        return float(np.median(arr[::MEDIAN_STRIDE, ::MEDIAN_STRIDE]))
    return float(np.median(arr))


def _largest_component(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """返回最大连通域的 ``(ys, xs)`` 坐标; 无前景返回 ``None``。

    只在非零像素的包围盒内做 ``ndimage.label``: 盒外全为背景, 连通关系与全幅
    标注完全一致; 坐标顺序与对全幅布尔图调用 ``np.nonzero`` 一致。
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    labeled, num = ndimage.label(mask[y0:y1, x0:x1])
    if num == 0:
        return None
    counts = np.bincount(labeled.ravel())
    # 0 号是背景
    counts[0] = 0
    idx = int(np.argmax(counts))
    if idx == 0 or counts[idx] == 0:
        return None
    component_y, component_x = np.nonzero(labeled == idx)
    return component_y + y0, component_x + x0


def _ellipse_from_mask(
    component: tuple[np.ndarray, np.ndarray] | None,
) -> EllipseParams | None:
    """由最大连通域的 ``(ys, xs)`` 坐标求半高边界椭圆。"""
    if component is None:
        return None
    ys, xs = component
    if xs.size < 9:
        return None
    x = xs.astype(np.float64)
    y = ys.astype(np.float64)
    mx = float(x.mean())
    my = float(y.mean())
    dx = x - mx
    dy = y - my
    vx = float(np.mean(dx * dx))
    vy = float(np.mean(dy * dy))
    vxy = float(np.mean(dx * dy))
    half_diff = 0.5 * (vx - vy)
    root = float(np.hypot(half_diff, vxy))
    mid = 0.5 * (vx + vy)
    lam_major = mid + root
    lam_minor = max(mid - root, 0.0)
    # 均匀填充椭圆: var = a^2 / 4
    semi_major = 2.0 * float(np.sqrt(max(lam_major, 0.0)))
    semi_minor = 2.0 * float(np.sqrt(lam_minor))
    angle = 0.5 * float(np.degrees(np.arctan2(2.0 * vxy, vx - vy)))
    if angle > 90.0:
        angle -= 180.0
    elif angle <= -90.0:
        angle += 180.0
    return EllipseParams(
        center=(mx, my),
        semi_major=semi_major,
        semi_minor=semi_minor,
        angle_deg=angle,
    )


def bin_2x2(img: np.ndarray) -> np.ndarray:
    """2x2 均匀合并 (均值), 奇数尺寸时丢弃最后一行/列。

    用于大靶面相机 (>1080p) 在 UI 线程里做实时分析前的降采样: 2688x1520
    降到 1344x760, 计算量降为 1/4, 坐标需由 ``FrameMetrics.rescaled(2.0)`` 还原。
    """
    arr = _as_2d(img)
    h, w = arr.shape
    h2 = h - (h % 2)
    w2 = w - (w % 2)
    view = arr[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2)
    return view.mean(axis=(1, 3), dtype=np.float32)


def _validate_center_method(method: str) -> None:
    """校验中心计算方法键; 未知键抛 ``ValueError``。"""
    if method not in dict(CENTER_METHODS):
        raise ValueError(
            f"未知中心计算方法: {method!r} (可选 {[k for k, _ in CENTER_METHODS]})"
        )


def _box_centroid(
    weight: np.ndarray | None,
    component: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[float, float] | None:
    """半高最大连通域外接框内的一阶矩; 无框/无权重返回 ``None``。"""
    if weight is None or component is None:
        return None
    ys, xs = component
    if xs.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    patch = weight[y0:y1, x0:x1].astype(np.float64, copy=False)
    total = float(patch.sum())
    if total <= 0.0:
        return None
    cols = np.arange(x0, x1, dtype=np.float64)
    rows = np.arange(y0, y1, dtype=np.float64)
    cx = float(np.dot(patch.sum(axis=0), cols) / total)
    cy = float(np.dot(patch.sum(axis=1), rows) / total)
    return (cx, cy)


def _select_center(
    method: str,
    centers: dict[str, tuple[float, float]],
    fallback: tuple[float, float],
) -> tuple[float, float]:
    """从已计算的口径中选择中心; 缺失口径回退至质心。"""
    return centers.get(method, fallback)


def _center_spread(centers: dict[str, tuple[float, float]]) -> float | None:
    """返回所有可用定心口径的最大两两欧氏距离。"""
    points = list(centers.values())
    if len(points) < 2:
        return None
    return max(
        float(np.hypot(left[0] - right[0], left[1] - right[1]))
        for index, left in enumerate(points)
        for right in points[index + 1 :]
    )


def analyze_frame(
    img: np.ndarray,
    *,
    background: float | None = None,
    threshold_ratio: float = 0.5,
    profile_band: int = 1,
    center_method: str = DEFAULT_CENTER_METHOD,
    full_scale: float | None = None,
) -> FrameMetrics:
    """分析一帧灰度图, 返回 :class:`FrameMetrics`.

    饱和占比按传感器的生效满量程统计 (像素值 ``>= sat_threshold``), 不等同于
    8-bit 显示截断。``full_scale=None`` 时为兼容旧版保留 255.0 阈值。

    参数:
        img: 2D 灰度图 (uint8/uint16/float 均可, 多通道输入取首通道)。
        background: 背景电平; 默认取全图中值。
        threshold_ratio: 二值化阈值相对 (峰值 - 背景) 的比例, 默认 0.5 = 半高。
        profile_band: 剖面取样时在中心行/列两侧各取多少像素做均值 (>=0)。
        center_method: 中心计算方法键 (见 :data:`CENTER_METHODS`);
            决定过中心剖面与十字准线所用的 ``(cx, cy)``。
        full_scale: 生效满量程; ``None`` 使用兼容阈值 255.0, 否则必须为有限正值。
    """
    _validate_center_method(center_method)
    if full_scale is None:
        sat_threshold = 255.0
    else:
        try:
            sat_threshold = float(full_scale)
        except (TypeError, ValueError) as exc:
            raise ValueError("full_scale 必须是有限正值") from exc
        if not np.isfinite(sat_threshold) or sat_threshold <= 0.0:
            raise ValueError("full_scale 必须是有限正值")

    arr = _as_2d(img)
    height, width = (int(arr.shape[0]), int(arr.shape[1]))

    peak = float(arr.max()) if arr.size else 0.0
    mean_value = float(arr.mean()) if arr.size else 0.0
    std_value = float(arr.std()) if arr.size else 0.0
    sat_ratio = (
        float(np.count_nonzero(arr >= sat_threshold) / arr.size) if arr.size else 0.0
    )

    bg = _background_estimate(arr) if background is None else float(background)

    # --- 最大值口径; 保持原有 argmax 展平索引约定 ---
    flat = int(np.argmax(arr)) if arr.size else 0
    peak_xy = (float(flat % width), float(flat // width)) if width else (0.0, 0.0)

    # --- 质心 (扣背景后的一阶矩, 用可分离求和避免构造大网格) ---
    weight = np.subtract(arr, bg, dtype=np.float32)
    np.maximum(weight, 0.0, out=weight)
    total = float(weight.sum())
    if total > 0.0:
        col_sums = weight.sum(axis=0, dtype=np.float64)
        row_sums = weight.sum(axis=1, dtype=np.float64)
        centroid = (
            float(np.dot(col_sums, np.arange(width)) / total),
            float(np.dot(row_sums, np.arange(height)) / total),
        )
    else:
        # 全黑/纯背景帧: 质心退化为峰值位置
        centroid = peak_xy

    # --- 半高最大连通域、框内质心与椭圆均各计算一次 ---
    level = bg + float(threshold_ratio) * (peak - bg)
    component = _largest_component(arr > level)
    ellipse = None if component is None else _ellipse_from_mask(component)
    centers: dict[str, tuple[float, float]] = {"centroid": centroid}
    box_center = _box_centroid(weight, component)
    if box_center is not None:
        centers["box_centroid"] = box_center
    if ellipse is not None:
        centers["ellipse"] = ellipse.center
    centers["peak"] = peak_xy
    cx, cy = _select_center(center_method, centers, centroid)
    center_spread_px = _center_spread(centers)

    # --- 过中心剖面 ---
    band = max(int(profile_band), 0)
    row = int(np.clip(round(cy), 0, max(height - 1, 0)))
    col = int(np.clip(round(cx), 0, max(width - 1, 0)))
    rows = arr[max(0, row - band) : min(height, row + band + 1)]
    cols = arr[:, max(0, col - band) : min(width, col + band + 1)]
    x_profile = rows.mean(axis=0).astype(np.float64) if rows.size else np.zeros(0)
    y_profile = cols.mean(axis=1).astype(np.float64) if cols.size else np.zeros(0)

    x_axis = np.arange(x_profile.size, dtype=np.float64)
    y_axis = np.arange(y_profile.size, dtype=np.float64)
    x_fit = fit_profile(x_axis, x_profile)
    y_fit = fit_profile(y_axis, y_profile)

    return FrameMetrics(
        max_value=peak,
        mean_value=mean_value,
        std_value=std_value,
        sat_ratio=sat_ratio,
        cx=cx,
        cy=cy,
        ellipse=ellipse,
        x_profile=x_profile,
        y_profile=y_profile,
        x_fit=x_fit,
        y_fit=y_fit,
        pixel_scale=1.0,
        width=width,
        height=height,
        sat_threshold=sat_threshold,
        centers=centers,
        center_spread_px=center_spread_px,
    )
