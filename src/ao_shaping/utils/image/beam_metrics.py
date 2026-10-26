"""光束整形质量指标与图案归一化辅助函数。

纯 NumPy 函数 (无硬件访问 / 无 SLM / 无相机)。这些函数从
:mod:`ao_shaping.algorithm.beam_shaping_utils` 迁移而来, 使指标在叶子
``utils`` 层计算, 保持单一事实来源。新代码应从这里导入; 旧模块为向后
兼容重新导出这些名称。

- 二维强度图上的光斑 / 光束测量。
- 掩码指标原语 ``masked_cv`` / ``masked_energy_fraction`` —— issue #199 R3 后
  全仓 "框内 CV + ROI 能量" 口径的单一事实来源 (torch 侧为可微软封装)。
- 掩码感知整形指标 (均匀度变异系数, 环围能量)。
- 方形光束指标 + 组合质量评分。
- 图案归一化 / 振幅转换。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt
from loguru import logger

from ao_shaping.utils.image.spots_calc import centroid, radius

#: Zero-variance guard for Pearson correlation in :func:`compute_beam_metrics`.
_STD_EPS: Final[float] = 1e-8

#: Smallest odd median kernel ``despike_frame`` can apply; below it the frame
#: is returned untouched (a 1x1 or 2x2 "median" is not despiking).
_DESPIKE_MIN_K: Final[int] = 3

#: M^2 needs at least this many distinct focal planes to be absolute.
_M2_MIN_PLANES: Final[int] = 3

__all__ = [
    "intensity_to_amplitude",
    "normalize_pattern",
    "measure_spot_diameter_cam",
    "compute_beam_metrics",
    "compute_shaping_metrics",
    "compute_shape_iou",
    "masked_cv",
    "masked_energy_fraction",
    "zero_order_exclusion_mask",
    "compute_square_metrics",
    "compute_quality_score",
    "measure_bright_span",
    "clamp_side",
    "threshold_spot_center",
    "zero_order_center",
    "smart_zero_order_center",
    "median_zero_order_center",
    "clamp_center_to_frame",
    "prepare_frame",
    "locate_peak",
    "strehl_ratio",
    "ideal_peak_from_pupil",
    "beam_radius_second_moment",
    "m2_from_focus_stack",
    "m2_reference_relative",
    "m2_marechal_crosscheck",
]


def intensity_to_amplitude(
    intensity: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """将强度图案转换为振幅。

    Args:
        intensity: 二维强度数组。
        normalize: 若为 True, 按最大值将振幅归一化到 ``[0, 1]``。

    Returns:
        Float32 振幅数组。
    """
    amp = np.sqrt(np.asarray(intensity, dtype=np.float64))
    if normalize:
        amax = float(amp.max())
        if amax > 0:
            amp = amp / amax
    return amp.astype(np.float32)


def normalize_pattern(
    pattern: np.ndarray,
    mode: Literal["peak", "sum"] = "peak",
) -> np.ndarray:
    """将二维图案归一化到 ``[0, 1]`` (峰值) 或单位总能量。

    Args:
        pattern: 二维输入数组。
        mode: ``"peak"`` 按最大值缩放至 1。
            ``"sum"`` 按总和缩放至 1。

    Returns:
        与 ``pattern`` 同形状的 Float32 归一化数组。

    Raises:
        ValueError: 当 ``mode`` 无法识别时。
    """
    pattern = np.asarray(pattern, dtype=np.float64)
    pattern = np.nan_to_num(pattern, nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "peak":
        pmax = float(pattern.max())
        if pmax > 0:
            pattern = pattern / pmax
    elif mode == "sum":
        total = float(pattern.sum())
        if total > 0:
            pattern = pattern / total
    else:
        raise ValueError(f"Unknown normalize mode: {mode}")
    return pattern.astype(np.float32)


def measure_spot_diameter_cam(intensity: np.ndarray, energy: float = 0.90) -> float:
    """从强度图像测量远场光束光斑直径 (像素)。

    以强度质心为中心, 用环围能量半径 (默认 90%) 推导光斑直径。

    Args:
        intensity: 二维远场强度图像。
        energy: 环围能量比例 (0~1), 用于计算半径 (默认 0.90)。

    Returns:
        以相机像素为单位的光斑直径。
    """
    cx, cy = centroid(intensity, return_float=True)
    r = radius(intensity, center=(cx, cy), energy=energy, use_aotools=False)
    return 2.0 * float(r)


def compute_beam_metrics(
    measured: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    """计算实测与目标之间的光束整形质量指标。

    ``measured`` 与 ``target`` 应为同形状的二维强度或振幅图。比较前两者
    均归一化到 ``[0, 1]``, 因此绝对尺度无关紧要 (SLM+CCD 链路存在未知
    的绝对增益)。

    Args:
        measured: 二维实测强度/振幅图。
        target: 二维目标强度/振幅图 (同形状)。

    Returns:
        包含 ``"mse"`` (归一化强度均方误差), ``"correlation"`` (展平后
        两图的 Pearson 相关系数) 与 ``"efficiency"`` (重叠能量比) 的字典。

    Raises:
        ValueError: 当形状不匹配时。
    """
    m = np.asarray(measured, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if m.shape != t.shape:
        raise ValueError(f"Shape mismatch: measured {m.shape} vs target {t.shape}")

    # 将两者归一化为单位总和 (能量), 实现尺度无关比较
    m_sum = float(m.sum())
    t_sum = float(t.sum())
    if m_sum > 0:
        m = m / m_sum
    if t_sum > 0:
        t = t / t_sum

    mse = float(np.mean((m - t) ** 2))

    # Pearson 相关系数 (防止零方差)
    mf, tf = m.flatten(), t.flatten()
    if mf.std() > _STD_EPS and tf.std() > _STD_EPS:
        corr = float(np.corrcoef(mf, tf)[0, 1])
    else:
        corr = 1.0 if np.allclose(mf, tf) else 0.0

    # 效率: [0, 1] 内的对称重叠
    total = float(m.sum() + t.sum())
    efficiency = (
        float(2.0 * float(np.minimum(m, t).sum()) / total) if total > 0 else 0.0
    )

    return {"mse": mse, "correlation": corr, "efficiency": efficiency}


# ---------------------------------------------------------------------------
# 掩码指标原语: issue #199 R3 的全仓唯一口径
# ---------------------------------------------------------------------------
def _masked_values(
    intensity: np.ndarray,
    mask: npt.NDArray[np.bool_] | tuple[Any, ...] | None,
) -> np.ndarray:
    """按 ``mask`` 选值: ``None`` = 全体; 布尔数组; 基础索引元组 (如切片)。"""
    image = np.asarray(intensity)
    return image if mask is None else image[mask]


def masked_cv(
    intensity: np.ndarray,
    mask: npt.NDArray[np.bool_] | tuple[Any, ...] | None = None,
    *,
    floor: float = 0.0,
) -> float:
    """掩膜内强度的变异系数 ``std / mean`` (issue #199 R3 的唯一口径)。

    R3 之前 "框内 CV" 有四份实现 —— :func:`compute_square_metrics` 的区域版、
    ``multishape_objective.image_metrics``、
    ``slm_model_in_loop._metrics_at`` 的非方路径、
    ``hw_square_carrier_probe.metrics`` —— 分子分母与退化策略各不相同; 现全部
    收敛到本函数。各自的退化策略 (``0.0`` / ``None`` / ``nan``) 保留在调用点:
    那是消费者契约, 不是指标定义。

    Args:
        intensity: 二维强度数组。**不转换 dtype**: float32 输入即按 float32
            累计与相除 (``image_metrics`` 的调用方传 float32 相机帧, 其历史
            行为就是 float32 除法)。
        mask: ``None`` = 对 ``intensity`` 全体统计; 布尔数组按掩码选; 也接受
            基础索引元组 (如 ``(slice(y0, y1), slice(x0, x1))``), 供半开盒
            区域使用 (:func:`compute_square_metrics`)。
        floor: 均值分母的下限防护, 复现调用点的历史口径
            (:func:`compute_square_metrics` 用 ``max(mean, 1e-10)``)。

    Returns:
        变异系数; 掩膜为空或均值非正 (含 NaN) 时返回 ``0.0``, 绝不返回
        NaN/inf。
    """
    values = _masked_values(intensity, mask)
    if values.size == 0:
        return 0.0
    mean = values.mean()
    if not mean > 0:
        return 0.0
    return float(values.std() / max(mean, floor))


def masked_energy_fraction(
    intensity: np.ndarray,
    mask: npt.NDArray[np.bool_] | tuple[Any, ...] | None = None,
    *,
    floor: float = 0.0,
    nonnegative: bool = False,
) -> float:
    """掩膜内能量占全帧能量的比例 (issue #199 R3 的唯一口径)。

    与 :func:`masked_cv` 同源: R3 之前 EE 有 ``max(total, 1e-10)`` /
    ``max(total, 1e-9)`` / ``max(total, 1.0)`` / 非负截断四种口径, 现由
    ``floor`` 与 ``nonnegative`` 两个开关复现, 定义本身只有一条。

    Args:
        intensity: 二维强度数组 (同样不转换 dtype)。
        mask: 同 :func:`masked_cv`。
        floor: 分母下限防护 (各调用点的历史口径: 1e-10 / 1e-9 / 1.0 / 无)。
        nonnegative: 分子分母都先把负强度截断为 0
            (``slm_model_in_loop._metrics_at`` 非方路径的口径: 读噪声的负计数
            不计入能量)。

    Returns:
        能量占比; 全帧总能量非正时返回 ``0.0``。
    """
    image = np.asarray(intensity)
    if nonnegative:
        image = np.maximum(image, 0.0)
    total = float(image.sum())
    if total <= 0.0:
        return 0.0
    values = image if mask is None else image[mask]
    return float(values.sum() / max(total, floor))


def compute_shaping_metrics(
    intensity: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """计算掩码感知的光束整形质量指标。

    在 :func:`compute_beam_metrics` 基础上补充均匀度 / 环围能量统计, 且仅在
    布尔目标掩码内部计算。

    Args:
        intensity: 二维强度数组。
        mask: 与 ``intensity`` 同形状的布尔二维数组, 定义目标区域。若掩码
            大于 ``intensity``, 则 (居中) 裁剪到强度边界内。

    Returns:
        包含 ``"uniformity_cv"`` (掩码内强度的标准差/均值, 完全平坦区域
        为 0), ``"encircled_energy"`` (``sum(intensity[mask]) /
        sum(intensity)``), ``"peak"`` (未归一化的最大强度) 与
        ``"in_mask_mean"`` (掩码内平均强度) 的字典。掩码为空或总强度为
        零时所有值均为 ``0.0`` (绝不产生 NaN/inf)。

    Raises:
        ValueError: 当 ``intensity`` 或 ``mask`` 不是二维数组时。
    """
    intensity = np.asarray(intensity, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if intensity.ndim != 2 or mask.ndim != 2:
        raise ValueError("intensity and mask must both be 2D arrays")

    # 将掩码区域裁剪到强度边界内 (居中重叠)。
    if mask.shape != intensity.shape:
        h, w = intensity.shape
        mh, mw = mask.shape
        y0 = max((mh - h) // 2, 0)
        x0 = max((mw - w) // 2, 0)
        y1 = min(y0 + h, mh)
        x1 = min(x0 + w, mw)
        clipped = mask[y0:y1, x0:x1]
        padded = np.zeros((h, w), dtype=bool)
        py0 = (h - clipped.shape[0]) // 2
        px0 = (w - clipped.shape[1]) // 2
        padded[py0 : py0 + clipped.shape[0], px0 : px0 + clipped.shape[1]] = clipped
        mask = padded

    n_mask = int(mask.sum())
    total = float(intensity.sum())
    if n_mask == 0 or total <= 0:
        return {
            "uniformity_cv": 0.0,
            "encircled_energy": 0.0,
            "peak": 0.0,
            "in_mask_mean": 0.0,
        }

    return {
        "uniformity_cv": masked_cv(intensity, mask),
        "encircled_energy": masked_energy_fraction(intensity, mask),
        "peak": float(intensity.max()),
        "in_mask_mean": float(intensity[mask].mean()),
    }


def compute_shape_iou(
    intensity: np.ndarray, mask: np.ndarray, *, peak_fraction: float = 0.5
) -> float:
    """Intersection over union of the bright CCD region and a target mask.

    The threshold is relative to the measured peak, so exposure scaling does
    not change this diagnostic.  A focused spot inside a large target scores
    poorly even when its encircled energy is high.
    """
    image = np.asarray(intensity, dtype=np.float64)
    target = np.asarray(mask, dtype=bool)
    if image.ndim != 2 or image.shape != target.shape:
        raise ValueError("intensity and target mask must have the same 2D shape")
    if not 0 < peak_fraction < 1:
        raise ValueError("peak_fraction must lie in (0, 1)")
    clean = prepare_frame(np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0))
    peak = float(clean.max())
    if peak <= 0 or not target.any():
        return 0.0
    bright = clean >= peak_fraction * peak
    intersection = np.count_nonzero(bright & target)
    union = np.count_nonzero(bright | target)
    return float(intersection / union) if union else 0.0


def zero_order_exclusion_mask(
    flat_frame: np.ndarray,
    target: np.ndarray,
    *,
    radius_px: float | None = None,
    threshold_fraction: float = 0.5,
    margin_px: int = 2,
) -> np.ndarray:
    """Find the optical axis in a flat reference and mask its central core.

    The shaped frame's brightest speckle is not necessarily the zero order.
    With no explicit radius, use the connected half-peak component containing
    the flat-reference peak, plus a small margin. The exclusion is clipped to
    the target and to one quarter of its equivalent side length.
    """
    from scipy.ndimage import label

    flat = np.asarray(flat_frame, dtype=np.float64)
    support = np.asarray(target, dtype=bool)
    if flat.ndim != 2 or flat.shape != support.shape or not support.any():
        raise ValueError("flat_frame and nonempty target must have the same 2D shape")
    if not 0 < threshold_fraction < 1 or margin_px < 0:
        raise ValueError("invalid zero-order threshold or margin")
    if radius_px is not None and (not np.isfinite(radius_px) or radius_px <= 0):
        raise ValueError("radius_px must be positive and finite")
    clean = prepare_frame(np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0))
    cx, cy = zero_order_center(clean, refine=False, despike_k=3)
    peak = float(clean[cy, cx])
    if peak <= 0:
        raise ValueError("flat reference has no detectable zero order")
    if not support[cy, cx]:
        raise ValueError("flat-reference zero order is outside the target")
    yy, xx = np.ogrid[:flat.shape[0], :flat.shape[1]]
    distance2 = (xx - cx) ** 2 + (yy - cy) ** 2
    if radius_px is None:
        components, _ = label(clean >= threshold_fraction * peak)
        component = components == components[cy, cx]
        radius_px = float(np.sqrt(distance2[component].max())) + margin_px
    max_radius = max(1.0, float(np.sqrt(support.sum())) / 4.0)
    return support & (distance2 <= min(float(radius_px), max_radius) ** 2)


def compute_square_metrics(
    intensity: np.ndarray,
    target_side: int,
    center: tuple[float, float],
    energy: float = 0.90,
) -> dict[str, float]:
    """计算方形光束的质量指标。

    Args:
        intensity: 二维远场强度图像。
        target_side: 目标方形边长 (像素)。
        center: 光束中心 ``(cx, cy)``。
        energy: 环围能量比例 (默认 0.90)。

    Returns:
        包含 ``aspect_ratio``, ``squareness``, ``uniformity_cv``,
        ``encircled_energy``, ``flatness_factor``, ``intensity_max``,
        ``intensity_mean`` 的字典。``uniformity_cv`` / ``encircled_energy``
        走 :func:`masked_cv` / :func:`masked_energy_fraction`
        (issue #199 R3 的唯一口径)。
    """
    intensity = np.asarray(intensity, dtype=np.float64)
    total = float(np.sum(intensity))
    if total <= 0:
        return {
            "aspect_ratio": 1.0,
            "squareness": 0.0,
            "uniformity_cv": 0.0,
            "encircled_energy": 0.0,
            "flatness_factor": 0.0,
            "intensity_max": 0.0,
            "intensity_mean": 0.0,
        }

    cx, cy = int(round(center[0])), int(round(center[1]))
    h, w = intensity.shape

    peak = float(np.max(intensity))
    threshold = 0.5 * peak
    bright = intensity >= threshold
    if bright.any():
        ys, xs = np.nonzero(bright)
        width_bright = int(xs.max()) - int(xs.min()) + 1
        height_bright = int(ys.max()) - int(ys.min()) + 1
        aspect_ratio = max(width_bright, height_bright) / max(
            min(width_bright, height_bright), 1
        )
    else:
        aspect_ratio = 1.0

    half = max(target_side // 2, 1)
    y0 = max(cy - half, 0)
    y1 = min(cy + half, h)
    x0 = max(cx - half, 0)
    x1 = min(cx + half, w)
    # 半开盒切片 (取整到整像元) —— 与闭合盒的目标几何 (R1 收敛后的
    # build_box_core_masks) 是两套口径, 此处保持实测帧上的历史行为。
    region = (slice(y0, y1), slice(x0, x1))
    region_mean = float(np.mean(intensity[region]))
    # CV/EE 走 issue #199 R3 的共享原语; ``max(., 1e-10)`` 下限是本函数的
    # 历史口径, 由 ``floor`` 复现。
    uniformity_cv = masked_cv(intensity, region, floor=1e-10)
    encircled_energy = masked_energy_fraction(intensity, region, floor=1e-10)
    flatness_factor = region_mean / max(peak, 1e-10) if peak > 0 else 0.0

    return {
        "aspect_ratio": float(aspect_ratio),
        "squareness": float(abs(1.0 - aspect_ratio)),
        "uniformity_cv": float(uniformity_cv),
        "encircled_energy": float(encircled_energy),
        "flatness_factor": float(flatness_factor),
        "intensity_max": float(peak),
        "intensity_mean": float(np.mean(intensity)),
    }


def compute_quality_score(metrics: dict[str, float]) -> float:
    """由方形指标计算组合质量评分。

    长宽比、均匀度与环围能量的加权组合, 越高越好, 值域 ``[0, 1]``。

    **两个核函数都是单调不饱和的有界形式** (2026-10-06 修正):

    ============================ ==========================================
    核                           形式
    ============================ ==========================================
    ``f_ar``                     ``1 / (1 + |AR - 1|)``
    ``f_uni``                    ``1 / (1 + max(CV, 0))``
    ============================ ==========================================

    ⚠️ **此前用的是 ``exp(-((x/0.3)**2))``, 它在可达范围内就饱和了, 并因此把
    评分排序整个搞反。** 实测: ``f_uni`` 在 CV=0.9 时约 ``1.2e-4``, 在 CV=3 时约
    ``e^-100``, 在 CV=11 (未整形的紧聚焦) 时**恰好为 0** —— 于是 CV 0.9 (均匀方斑)
    与 CV 11 (未整形) 的均匀度得分都是 0, 合成评分退化为 ``0.3*AR + 0.3*EE``。
    环围能量还会掩盖这一点, 因为未整形的紧聚焦天然 EE≈0.99。后果是实测的:
    ``slm-model-in-loop`` 首次实现用它做对比择优, 把 CV≈0.9 的**均匀方斑**排在
    CV≈11 的**未整形聚焦**之后, 于是丢弃了正确相位保留了平场。

    修正后同样两个输入的排序**恢复正确** (AR=1, 均匀方斑 CV=0.9/EE=0.87 得
    **0.7715**, 未整形 CV=11/EE=0.99 得 **0.6303**), 且在整个 CV 可达范围内每一次
    均匀度改善都得到回报 —— 没有阈值可调, 不需要重新标定。

    物理读法: ``CV = std/mean``, 完全发育的散斑场 ``CV = 1``, 平顶目标 ``CV -> 0``。
    ``1/(1+CV)`` 把这个区间单调压到 ``[0, 1/2]``, 既保留了对均匀度的分辨力, 又不会
    让这一项主导总分 (纯饱和指数会让它恒为 0, 即完全放弃该维度)。

    Args:
        metrics: :func:`compute_square_metrics` 返回的字典。

    Returns:
        ``[0, 1]`` 范围内的浮点质量评分 (越高越好)。
    """
    f_ar = 1.0 / (1.0 + abs(float(metrics["aspect_ratio"]) - 1.0))
    f_uni = 1.0 / (1.0 + max(float(metrics["uniformity_cv"]), 0.0))
    f_ee = float(np.clip(metrics["encircled_energy"], 0.0, 1.0))
    return float(0.3 * f_ar + 0.4 * f_uni + 0.3 * f_ee)


def measure_bright_span(
    intensity: np.ndarray,
    peak_frac: float = 0.5,
) -> tuple[int, int]:
    """返回高于 ``peak_frac`` 的亮区 ``(width, height)``。

    Args:
        intensity: 二维强度图像。
        peak_frac: 以峰值比例表示的阈值 (默认 0.5)。

    Returns:
        包围盒的 ``(width, height)``; 未找到亮区时返回 ``(0, 0)``。
    """
    intensity = np.asarray(intensity, dtype=np.float64)
    peak = float(np.max(intensity))
    if peak <= 0:
        return 0, 0
    bright = intensity >= peak_frac * peak
    if not bright.any():
        return 0, 0
    ys, xs = np.nonzero(bright)
    return int(xs.max()) - int(xs.min()) + 1, int(ys.max()) - int(ys.min()) + 1


def clamp_side(side: int, height: int, width: int, margin: int = 8) -> int:
    """将方形边长限制在带边距的网格范围内。

    Args:
        side: 请求的边长。
        height: 网格高度。
        width: 网格宽度。
        margin: 距每条边的最小边距 (默认 8)。

    Returns:
        限制后的边长。
    """
    max_side = min(height, width) - 2 * margin
    if side > max_side:
        logger.warning(
            "Square side {}px exceeds grid limit, clamped to {}px", side, max_side
        )
        return max(1, max_side)
    return max(1, side)


# ---------------------------------------------------------------------------
# 0-order spot centre helpers (shared by any hardware runner needing the
# 2f-bench 0-order location before optimisation)
# ---------------------------------------------------------------------------
def threshold_spot_center(img: np.ndarray) -> tuple[int, int]:
    """Threshold-based 0-order spot centre, robust to degenerate frames.

    Marks every pixel brighter than a corner background estimate and takes that
    mask's centroid. Three guards make it safe on real camera frames:

    * empty mask — the brightest pixels sit *inside* the corner patch itself
      (hot pixel / stray light), so ``img > corner_max`` selects nothing and
      ``scipy.center_of_mass`` would divide by zero → ``centroid()`` would raise
      ``ValueError: cannot convert float NaN to integer``; fall back to a
      relative-threshold centroid;
    * all-dark frame → return the frame centre instead of NaN;
    * non-2D / empty input → ``ValueError``.

    Returns ``(x, y)`` in pixels (project convention).
    """
    frame = np.asarray(img)
    if frame.ndim != 2 or frame.size == 0:
        raise ValueError(f"img must be a non-empty 2D array, got shape {frame.shape}")
    height, width = frame.shape
    if float(frame.max()) <= 0.0:
        return (width // 2, height // 2)

    corner_h = max(int(height // 50), 2)
    corner_w = max(int(width // 50), 2)
    corner = frame[:corner_h, :corner_w]
    mask = frame > float(np.max(corner))
    if np.any(mask):
        cx, cy = centroid(mask)
        return (int(cx), int(cy))

    # Degenerate mask (the corner patch itself holds the brightest pixels, e.g. a
    # hot pixel): drop that patch and take a plain centroid. A relative threshold
    # would still be scaled by the hot pixel's value and drag the centre towards
    # the corner.
    fallback = frame.copy()
    fallback[:corner_h, :corner_w] = 0
    if float(fallback.max()) > 0.0:
        cx, cy = centroid(fallback)
        return (int(cx), int(cy))
    return (width // 2, height // 2)


def despike_frame(img: np.ndarray, k: int = _DESPIKE_MIN_K) -> np.ndarray:
    """Replace isolated hot/cold pixels with a ``k x k`` median.

    A box blur alone is not enough against a *single* hot pixel: a 5000-count
    defect spread over a 5x5 box still reads 200, which beats a dim spot peaking at
    20. Real sensors have defects, and one is enough to send ``argmax`` -- and
    therefore the centroid and every width derived from it -- to the wrong place.
    A median kills isolated outliers outright while leaving a real spot (which is
    spatially correlated) essentially unchanged.

    Lives here, in the leaf ``utils/image`` layer, because **both**
    :mod:`slm_bench_probe` (which owns the bench
    measurement kernel) and the dataset transforms need it, and ``utils`` must not
    import upward from ``tools``. ``slm_bench_probe`` re-exports this name, so
    existing callers are unaffected. It was previously defined only there.
    """
    frame = np.asarray(img, dtype=np.float64)
    k = int(k)
    if k < _DESPIKE_MIN_K:
        return frame
    pad = k // 2
    p = np.pad(frame, pad, mode="edge")
    stack = np.stack(
        [
            p[dy : dy + frame.shape[0], dx : dx + frame.shape[1]]
            for dy in range(k)
            for dx in range(k)
        ],
        axis=0,
    )
    return np.median(stack, axis=0)


def prepare_frame(frame: np.ndarray) -> np.ndarray:
    """Remove the detector background and return a non-negative frame.

    A CCD frame carries symmetric read noise, so roughly half its pixels are
    negative and an unprocessed frame yields ``PIB > 1`` (measured 1.012). The
    beam occupies a small fraction of the frame, so the median is a robust
    background estimate; clipping **after** subtraction is what makes ``PIB <= 1``
    and ``CV`` finite. Clipping the raw frame instead would rectify that noise
    into a pixel-count-sized DC pedestal.

    Note this is deliberately *not* the same convention as
    ``slm_square_shaping.square_quality_score``, which clips without subtracting:
    there the background fills the frame, so the median *is* the pedestal and
    subtracting it makes the ratio explode. The measurements behind both, and the
    4598x read-noise pedestal figure, are in
    ``report/slm_pib_bench/report.md`` §7.

    The input dtype is preserved (a float32 frame stays float32); only
    non-finite samples are masked to 0.

    Args:
        frame: 2D camera frame.

    Returns:
        Non-negative array, same shape and dtype as ``frame``.
    """
    clean = np.where(np.isfinite(frame), frame, 0.0)
    return np.clip(clean - np.median(clean), 0.0, None)


def zero_order_center(
    frame: np.ndarray,
    refine: bool = True,
    half_win: int | None = None,
    despike_k: int = 1,
) -> tuple[int, int]:
    """0 级光斑中心: 全局 argmax 锚定 (+ 窗口内局部质心细化)。

    2f Fourier 光路上 0 级即帧全局最大 (``AGENTS.md`` 多条记录), 因此
    argmax 比角点阈值掩码 (会跳到热像素) 或全画幅质心 (会被杂散光拉偏)
    可靠得多。细化被限制在锚点周围窗口内, 窗口外的光不会移动中心。

    Args:
        frame: 二维远场强度图像。
        refine: 若为 True (默认), 在锚点邻域窗口内用质心细化; 仅返回
            argmax 锚点则传 False。
        half_win: 细化窗口半宽 (像素); None 时取 ``min(shape)//20``
            (下限 8)。
        despike_k: Locate the anchor on a ``k x k`` median-despiked copy of the
            frame when ``k >= 3``. **Default 1 = the bare argmax, unchanged.**

            Bare argmax is robust to *stray light* but blind to *hot pixels*: one
            60000-count defect beats a 255-count 0-order, and since the returned
            centre is used to place the measurement crop, a single defect then
            silently relocates the whole window. Measured with a synthetic defect
            at ``(4, 4)``: the crop moved off the beam entirely. A median kills
            isolated outliers while leaving a spatially correlated spot unchanged,
            so ``despike_k=3`` is the cheap fix; ``refine`` still runs on the
            *original* pixels afterwards, so the reported sub-pixel centre is
            unaffected apart from a better anchor.

    Returns:
        ``(x, y)`` 像素坐标 (项目约定)。全暗帧返回帧中心。
    """
    frame = np.asarray(frame)
    if frame.ndim != 2 or frame.size == 0:
        raise ValueError(f"frame must be a non-empty 2D array, got shape {frame.shape}")
    height, width = frame.shape
    if float(frame.max()) <= 0.0:
        return (width // 2, height // 2)

    search = despike_frame(frame, despike_k) if despike_k >= _DESPIKE_MIN_K else frame
    anchor_y, anchor_x = np.unravel_index(int(np.argmax(search)), frame.shape)
    if not refine:
        return (int(anchor_x), int(anchor_y))

    win = int(half_win) if half_win else max(int(min(frame.shape) // 20), 8)
    y0, y1 = max(0, anchor_y - win), min(height, anchor_y + win + 1)
    x0, x1 = max(0, anchor_x - win), min(width, anchor_x + win + 1)

    patch = frame[y0:y1, x0:x1].astype(np.float64)
    patch = np.clip(patch - float(np.percentile(patch, 20)), 0.0, None)
    if float(patch.sum()) <= 0.0:
        return (int(anchor_x), int(anchor_y))
    # centroid() returns window-local coordinates -> add the window offset.
    cx, cy = centroid(patch, return_float=True)
    cx, cy = cx + x0, cy + y0
    if not np.isfinite(cx) or not np.isfinite(cy):
        return (int(anchor_x), int(anchor_y))
    return (int(round(cx)), int(round(cy)))


def smart_zero_order_center(
    img: np.ndarray,
    margin: int = 6,
    flat_frac: float = 0.4,
) -> tuple[int, int]:
    """0 级光斑中心: argmax 锚定, 遇平坦核 (非空洞) 时扩大窗口细化。

    2f Fourier 光路上 0 级 = 帧全局最大 (``argmax``)。当 ``argmax`` 邻域内
    核部位空洞 (亮度 >= ``flat_frac * max``) 时, 扩大细化窗口以获得更稳定的
    质心 —— 全图质心会被杂散光拉偏 (硬件实测 1394 vs 958px)。核部为空洞
    (如环形 / TEM01 模式) 时, 保持 :func:`zero_order_center` 的默认小窗口,
    否则宽窗口的质心会漂移到环侧。

    这是 :func:`zero_order_center` 的 flat-core 感知封装, 统一了
    ``slm_zernike_pib.py`` 的 ``_smart_center``、``pib.py`` 的
    ``intellij_center``、``combined_optimizer.py`` 的内联 ``intellij_center``
    及 ``slm_square_shaping.py`` 的 ``intelligen_center``-基检测逻辑
    (2026-09 去重合并)。

    Args:
        img: 二维强度图像。
        margin: 核部检查的半窗口 (像素), 同时作为 flat-core 检测的邻域半径。
        flat_frac: 判断核部是否平坦的阈值比例 (``img[core] >= flat_frac * max``)。

    Returns:
        ``(x, y)`` 像素坐标。全暗帧返回帧中心。
    """
    frame = np.asarray(img)
    if frame.ndim != 2 or frame.size == 0:
        raise ValueError(f"img must be a non-empty 2D array, got shape {frame.shape}")
    height, width = frame.shape
    if float(frame.max()) <= 0.0:
        return (width // 2, height // 2)

    center = zero_order_center(frame)
    (cx, cy) = center
    y0, y1 = max(0, cy - margin), min(height, cy + margin)
    x0, x1 = max(0, cx - margin), min(width, cx + margin)
    if (
        y1 > y0
        and x1 > x0
        and np.all(frame[y0:y1, x0:x1] >= float(np.max(frame)) * flat_frac)
    ):
        # Flat (non-hollow) core: widen the refinement window. A full-image
        # centroid (as intelligen_center does) is dragged tens of pixels by
        # stray light — measured (1394 vs 958 on this bench).
        center = zero_order_center(frame, half_win=max(margin * 6, 32))
    return center


def median_zero_order_center(
    frames: list[np.ndarray] | tuple[np.ndarray, ...],
    refine: bool = True,
    half_win: int | None = None,
) -> tuple[int, int]:
    """多帧 0 级中心的逐轴中位数 (对偶发热像素帧鲁棒)。

    实际抓帧时个别帧可能被热像素 / 杂散光主导 (全局 argmax 跳到角落);
    中位数比均值更能容忍孤立的离群帧。

    Args:
        frames: 二维强度帧序列 (非空)。
        refine: 传给 :func:`zero_order_center`。
        half_win: 传给 :func:`zero_order_center`。

    Returns:
        ``(x, y)`` 像素坐标。

    Raises:
        ValueError: 序列为空或任一帧非法。
    """
    if not frames:
        raise ValueError("frames must be a non-empty sequence")
    xs: list[int] = []
    ys: list[int] = []
    for frame in frames:
        x, y = zero_order_center(frame, refine=refine, half_win=half_win)
        xs.append(x)
        ys.append(y)
    return (int(round(float(np.median(xs)))), int(round(float(np.median(ys)))))


def clamp_center_to_frame(
    center: tuple[int | float, int | float],
    frame_shape: tuple[int | float, int | float],
    window: int,
) -> tuple[int, int]:
    """Clamp an ROI centre so a ``window x window`` ROI always fits the frame.

    ``DahengCamera.reset_window`` asserts ``x_offset >= 0 and y_offset >= 0``, so
    a spot detected near a frame edge aborted the whole run. Clamping keeps the
    ROI inside the frame (the spot is simply off-centre) instead of crashing.

    Args:
        center: ``(x, y)`` ROI centre in pixels.
        frame_shape: ``(height, width)`` of the captured image.
        window: requested (square) ROI size in pixels.

    Returns:
        ``(x, y)`` clamped integer centre.
    """
    height, width = int(frame_shape[0]), int(frame_shape[1])
    half = min(max(0, int(window) // 2), min(height, width) // 2)
    max_x = max(half, width - half - 1)
    max_y = max(half, height - half - 1)
    return (
        int(np.clip(int(center[0]), half, max_x)),
        int(np.clip(int(center[1]), half, max_y)),
    )


def locate_peak(frame: np.ndarray, finite: bool = True) -> tuple[int, int]:
    """Locate the global peak as (row, col)."""
    masked = np.asarray(frame, dtype=np.float64)
    if finite:
        masked = np.where(np.isfinite(masked), masked, 0.0)
    iy, ix = np.unravel_index(int(np.argmax(masked)), masked.shape)
    return int(iy), int(ix)


# ---------------------------------------------------------------------------
# Canonical optical metrics: Strehl ratio and M^2 (ISO-11146 second moment)
# ---------------------------------------------------------------------------


def _clean_frame(frame: np.ndarray, name: str) -> np.ndarray:
    """Return a finite, 2D float64 copy of *frame* (NaN/Inf -> 0)."""
    arr = np.asarray(frame, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}")
    return np.where(np.isfinite(arr), arr, 0.0)


def _second_moment_radii(
    weight: np.ndarray,
    pitch_x: float,
    pitch_y: float,
) -> tuple[float, float]:
    """ISO-11146 second-moment radii (w_x, w_y) of a non-negative weight map.

    Per axis w = sqrt(2) * sigma, where sigma is the weighted standard
    deviation of the coordinate. Negative weights are clipped to zero.
    """
    w = np.clip(weight, 0.0, None)
    total = float(w.sum())
    if total <= 0.0:
        return (0.0, 0.0)
    ny, nx = w.shape
    ys = np.arange(ny, dtype=np.float64)
    xs = np.arange(nx, dtype=np.float64)
    wy = w.sum(axis=1)  # marginal over rows (varies with y)
    wx = w.sum(axis=0)  # marginal over cols (varies with x)
    cy = float((ys * wy).sum() / total)
    cx = float((xs * wx).sum() / total)
    var_y = float((((ys - cy) ** 2) * wy).sum() / total)
    var_x = float((((xs - cx) ** 2) * wx).sum() / total)
    w_y = math.sqrt(2.0 * max(var_y, 0.0)) * pitch_y
    w_x = math.sqrt(2.0 * max(var_x, 0.0)) * pitch_x
    return (w_x, w_y)


def strehl_ratio(
    measured: np.ndarray,
    ideal_peak: float,
    *,
    clip: bool = True,
) -> float:
    """Optical Strehl ratio: measured far-field peak / ideal far-field peak.

    The optical Strehl ratio is the ratio of the peak intensity of the
    measured far field to that of a diffraction-limited reference **under
    identical normalization** (same total energy / exposure / pixel scale).
    This is a different quantity from
    :func:`ao_shaping.algorithm.signal_processing.square_shaping.strehl`,
    which is a mean-centred cosine similarity (shape metric, scale
    invariant) and is NOT an optical Strehl.

    Args:
        measured: 2D far-field intensity image (background already removed,
            e.g. via :func:`prepare_frame`).
        ideal_peak: Diffraction-limited far-field peak under the same
            normalization; must be a positive finite number. Compute it
            with :func:`ideal_peak_from_pupil` through the same FFT
            pipeline that produced *measured*.
        clip: Clamp the ratio to ``[0, 1]`` (default ``True``). Detector
            noise can push a measured peak above the ideal; pass
            ``clip=False`` to keep the raw ratio for diagnostics.

    Returns:
        Strehl ratio in ``[0, 1]`` (raw ratio when ``clip=False``).

    Raises:
        ValueError: if *measured* is not 2D or *ideal_peak* is not positive
            and finite.
    """
    frame = _clean_frame(measured, "measured")
    ideal = float(ideal_peak)
    if not math.isfinite(ideal) or ideal <= 0.0:
        raise ValueError(f"ideal_peak must be positive and finite, got {ideal_peak!r}")
    ratio = float(frame.max()) / ideal
    if clip:
        ratio = min(max(ratio, 0.0), 1.0)
    return ratio


def ideal_peak_from_pupil(
    amplitude: np.ndarray,
    *,
    far_field_padding: float = 1.0,
) -> float:
    """Fraunhofer far-field peak of a pupil amplitude (diffraction-limited reference).

    Computes ``max(|FFT2(amplitude)|^2)`` with ``fftshift`` centering, using
    numpy's unnormalized FFT convention. The value is only meaningful
    together with a measured far field produced by the **same** pipeline
    (same padding, same normalization) — see :func:`strehl_ratio`.

    Args:
        amplitude: 2D pupil field amplitude (real or complex).
        far_field_padding: Zero-pad the pupil to ``round(size * padding)``
            (centered) before the FFT to supersample the far field
            (default ``1.0`` = no padding).

    Returns:
        Far-field peak intensity (unnormalized FFT units).

    Raises:
        ValueError: if *amplitude* is not 2D or *far_field_padding* < 1.
    """
    amp = np.asarray(amplitude, dtype=np.complex128)
    if amp.ndim != 2:
        raise ValueError(f"amplitude must be 2D, got shape {amp.shape}")
    pad = float(far_field_padding)
    if not math.isfinite(pad) or pad < 1.0:
        raise ValueError(f"far_field_padding must be >= 1.0, got {far_field_padding!r}")
    if pad > 1.0:
        h, w = amp.shape
        nh, nw = int(round(h * pad)), int(round(w * pad))
        canvas = np.zeros((nh, nw), dtype=np.complex128)
        y0, x0 = (nh - h) // 2, (nw - w) // 2
        canvas[y0 : y0 + h, x0 : x0 + w] = amp
        amp = canvas
    ff = np.fft.fftshift(np.fft.fft2(amp))
    return float(np.max(np.abs(ff) ** 2))


def beam_radius_second_moment(
    intensity: np.ndarray,
    *,
    pixel_pitch_m: float = 1.0,
) -> float:
    """ISO-11146 second-moment beam radius (geometric mean of the two axes).

    Per axis ``w = sqrt(2) * sigma`` (sigma = intensity-weighted standard
    deviation); the returned radius is ``sqrt(w_x * w_y)``. For a Gaussian
    this equals the 1/e^2 radius exactly.

    Remove the background first (e.g. :func:`prepare_frame`): read noise
    inflates the second moment quadratically with frame size.

    Args:
        intensity: 2D intensity image.
        pixel_pitch_m: Physical pixel pitch in meters (default ``1.0`` =
            result in pixels).

    Returns:
        Beam radius (pitch-scaled). ``0.0`` for an all-zero frame.

    Raises:
        ValueError: if *intensity* is not 2D or *pixel_pitch_m* <= 0.
    """
    frame = _clean_frame(intensity, "intensity")
    pitch = float(pixel_pitch_m)
    if not math.isfinite(pitch) or pitch <= 0.0:
        raise ValueError(f"pixel_pitch_m must be positive, got {pixel_pitch_m!r}")
    w_x, w_y = _second_moment_radii(frame, pitch, pitch)
    return float(math.sqrt(w_x * w_y))


def m2_from_focus_stack(
    fields: Sequence[np.ndarray],
    z_positions_m: Sequence[float],
    *,
    wavelength_m: float,
    pixel_pitch_m: float,
) -> float:
    """ISO-11146 M^2 from a focus stack (absolute method; multi-plane only).

    Fits ``w^2(z) = A z^2 + B z + C`` to second-moment radii measured at
    several known axial positions, then

        w0^2 = C - B^2 / (4A),   M^2 = (pi * w0 * sqrt(A)) / lambda

    which is the ISO-11146 parabola fit of ``w^2(z) = w0^2 +
    (M^2 lambda / (pi w0))^2 (z - z0)^2``.

    **A single focal plane cannot measure M^2 absolutely** — the divergence
    term requires propagation. For the single-plane bench use
    :func:`m2_reference_relative`; this function is the absolute reference
    for simulation / focus-stack data only.

    Args:
        fields: Intensity images at the given z positions (background
            removed).
        z_positions_m: Axial positions in meters (>= 3 distinct values).
        wavelength_m: Wavelength in meters.
        pixel_pitch_m: Physical pixel pitch in meters.

    Returns:
        M^2 (dimensionless). ``float('inf')`` when the fit is
        non-physical (A <= 0 or w0^2 <= 0).

    Raises:
        ValueError: on length mismatch, < 3 planes, or non-positive
            wavelength / pitch.
    """
    if len(fields) != len(z_positions_m):
        raise ValueError(
            f"fields ({len(fields)}) and z_positions_m ({len(z_positions_m)}) "
            "must have the same length"
        )
    if len(fields) < _M2_MIN_PLANES:
        raise ValueError(
            "M^2 cannot be measured absolutely from fewer than 3 focal planes; "
            "use m2_reference_relative for a single plane"
        )
    lam = float(wavelength_m)
    pitch = float(pixel_pitch_m)
    if not math.isfinite(lam) or lam <= 0.0:
        raise ValueError(f"wavelength_m must be positive, got {wavelength_m!r}")
    if not math.isfinite(pitch) or pitch <= 0.0:
        raise ValueError(f"pixel_pitch_m must be positive, got {pixel_pitch_m!r}")
    z = np.asarray(z_positions_m, dtype=np.float64)
    if len(np.unique(z)) < _M2_MIN_PLANES:
        raise ValueError("z_positions_m must contain at least 3 distinct values")
    w = np.array(
        [beam_radius_second_moment(f, pixel_pitch_m=pitch) for f in fields],
        dtype=np.float64,
    )
    if np.any(w <= 0.0):
        return float("inf")
    a, b, c = (float(v) for v in np.polyfit(z, w**2, 2))
    if a <= 0.0:
        return float("inf")
    w0_sq = c - (b * b) / (4.0 * a)
    if w0_sq <= 0.0:
        return float("inf")
    return float(math.pi * math.sqrt(w0_sq) * math.sqrt(a) / lam)


def m2_reference_relative(
    field: np.ndarray,
    ideal_field: np.ndarray,
    *,
    wavelength_m: float,
    pixel_pitch_m: float,
) -> float:
    """Reference-relative M^2 at a single plane (primary API for this bench).

    The second-moment beam parameter product ``BPP = w * theta`` is
    invariant under free propagation, with ``theta = lambda * sigma_k``
    where ``sigma_k`` is the second-moment radius of the angular spectrum
    ``|FFT(field)|^2``. The absolute ``M^2 = pi * w * sigma_k`` is
    wavelength-independent (lambda cancels between theta and sigma_k); the
    practical single-plane quantity is the ratio against a known
    diffraction-limited reference field sampled at the same plane:

        M^2_rel = (w * sigma_k) / (w_ref * sigma_k_ref)

    **This is relative to ideal_field, not an absolute M^2.** An absolute
    value requires a focus stack — see :func:`m2_from_focus_stack`.

    Args:
        field: Complex (or real) field amplitude at the measurement plane.
        ideal_field: Diffraction-limited reference field amplitude at the
            same plane (same pixel pitch and sampling).
        wavelength_m: Wavelength in meters. Cancels in the ratio; accepted
            for API symmetry with the other M^2 helpers.
        pixel_pitch_m: Physical pixel pitch in meters.

    Returns:
        Relative M^2 (>= ~1 for a well-sampled beam; 1.0 when *field*
        matches *ideal_field*). ``float('inf')`` when the reference has
        zero power.

    Raises:
        ValueError: on non-2D inputs, shape mismatch, or non-positive
            wavelength / pitch.
    """
    amp = np.asarray(field, dtype=np.complex128)
    ref = np.asarray(ideal_field, dtype=np.complex128)
    if amp.ndim != 2 or ref.ndim != 2:
        raise ValueError(
            f"field and ideal_field must be 2D, got {amp.shape} and {ref.shape}"
        )
    if amp.shape != ref.shape:
        raise ValueError(
            f"field shape {amp.shape} != ideal_field shape {ref.shape}"
        )
    lam = float(wavelength_m)
    pitch = float(pixel_pitch_m)
    if not math.isfinite(lam) or lam <= 0.0:
        raise ValueError(f"wavelength_m must be positive, got {wavelength_m!r}")
    if not math.isfinite(pitch) or pitch <= 0.0:
        raise ValueError(f"pixel_pitch_m must be positive, got {pixel_pitch_m!r}")

    def bpp(amp2d: np.ndarray) -> float:
        near = np.abs(amp2d) ** 2
        near = np.where(np.isfinite(near), near, 0.0)
        if near.sum() <= 0.0:
            return float("nan")
        w_x, w_y = _second_moment_radii(near, pitch, pitch)
        w = math.sqrt(w_x * w_y)
        power = np.abs(np.fft.fftshift(np.fft.fft2(amp2d))) ** 2
        ny, nx = power.shape
        np.fft.fftshift(np.fft.fftfreq(ny, d=pitch))
        np.fft.fftshift(np.fft.fftfreq(nx, d=pitch))
        sk_x, sk_y = _second_moment_radii(power, 1.0 / (nx * pitch), 1.0 / (ny * pitch))
        sigma_k = math.sqrt(sk_x * sk_y)
        return w * sigma_k

    bpp_meas = bpp(amp)
    bpp_ref = bpp(ref)
    if not math.isfinite(bpp_ref) or bpp_ref <= 0.0:
        return float("inf")
    if not math.isfinite(bpp_meas):
        return float("inf")
    return float(bpp_meas / bpp_ref)


def m2_marechal_crosscheck(
    phase_rad: np.ndarray,
    *,
    wavelength_m: float,
    aperture_weight: np.ndarray | None = None,
) -> float:
    """Maréchal cross-check: Strehl ~ exp(-sigma_phi^2), M^2 ~ 1/Strehl.

    ``phase_rad`` is wavefront phase in **radians**, so the Maréchal
    exponent is simply ``-(sigma_phi_rad)^2``: the ``2*pi/lambda`` factor
    converts OPD-in-meters to radians and cancels here. ``wavelength_m``
    is accepted for API symmetry with the other M^2 helpers and is not
    used when the phase is already in radians.

    This is a **sanity check, not a measurement**: the Maréchal
    approximation is only valid for low aberration
    (``sigma_phi <~ pi/2`` rad). Use :func:`m2_reference_relative` (single
    plane) or :func:`m2_from_focus_stack` (focus stack) for real values.

    Args:
        phase_rad: 2D phase map in radians.
        wavelength_m: Wavelength in meters (unused for radian input; kept
            for API symmetry).
        aperture_weight: Optional non-negative aperture mask / weight; the
            phase variance is computed weighted by it. ``None`` = all ones.

    Returns:
        ``exp(sigma_phi_rad^2)`` — the Maréchal-approximated M^2 proxy
        (1.0 for a perfect wavefront). ``float('inf')`` when the aperture
        weight sums to zero.

    Raises:
        ValueError: if *phase_rad* is not 2D or *wavelength_m* <= 0.
    """
    ph = _clean_frame(phase_rad, "phase_rad")
    lam = float(wavelength_m)
    if not math.isfinite(lam) or lam <= 0.0:
        raise ValueError(f"wavelength_m must be positive, got {wavelength_m!r}")
    if aperture_weight is None:
        wt = np.ones_like(ph)
    else:
        wt = _clean_frame(aperture_weight, "aperture_weight")
        if wt.shape != ph.shape:
            raise ValueError(
                f"aperture_weight shape {wt.shape} != phase_rad shape {ph.shape}"
            )
        wt = np.clip(wt, 0.0, None)
    wsum = float(wt.sum())
    if wsum <= 0.0:
        return float("inf")
    mean = float((wt * ph).sum() / wsum)
    var = float((wt * (ph - mean) ** 2).sum() / wsum)
    return float(math.exp(var))
