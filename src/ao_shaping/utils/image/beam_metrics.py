"""光束整形质量指标与图案归一化辅助函数。

纯 NumPy 函数 (无硬件访问 / 无 SLM / 无相机)。这些函数从
:mod:`ao_shaping.algorithm.beam_shaping_utils` 迁移而来, 使指标在叶子
``utils`` 层计算, 保持单一事实来源。新代码应从这里导入; 旧模块为向后
兼容重新导出这些名称。

- 二维强度图上的光斑 / 光束测量。
- 掩码感知整形指标 (均匀性变异系数, 环围能量)。
- 方形光束指标 + 组合质量评分。
- 图案归一化 / 振幅转换。
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from loguru import logger

from ao_shaping.utils.image.spots_calc import centroid, radius

__all__ = [
    "intensity_to_amplitude",
    "normalize_pattern",
    "measure_spot_diameter_cam",
    "compute_metrics",
    "compute_shaping_metrics",
    "compute_square_metrics",
    "compute_quality_score",
    "measure_bright_span",
    "clamp_side",
    "threshold_spot_center",
    "zero_order_center",
    "smart_zero_order_center",
    "median_zero_order_center",
    "clamp_center_to_frame",
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


def compute_metrics(
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
    if mf.std() > 1e-8 and tf.std() > 1e-8:
        corr = float(np.corrcoef(mf, tf)[0, 1])
    else:
        corr = 1.0 if np.allclose(mf, tf) else 0.0

    # 效率: [0, 1] 内的对称重叠
    total = float(m.sum() + t.sum())
    efficiency = (
        float(2.0 * float(np.minimum(m, t).sum()) / total) if total > 0 else 0.0
    )

    return {"mse": mse, "correlation": corr, "efficiency": efficiency}


def compute_shaping_metrics(
    intensity: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """计算掩码感知的光束整形质量指标。

    在 :func:`compute_metrics` 基础上补充均匀性 / 环围能量统计, 且仅在
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

    in_mask = intensity[mask]
    in_mask_mean = float(in_mask.mean())
    in_mask_std = float(in_mask.std())
    uniformity_cv = in_mask_std / in_mask_mean if in_mask_mean > 0 else 0.0
    encircled_energy = float(in_mask.sum()) / total

    return {
        "uniformity_cv": float(uniformity_cv),
        "encircled_energy": float(encircled_energy),
        "peak": float(intensity.max()),
        "in_mask_mean": in_mask_mean,
    }


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
        ``intensity_mean`` 的字典。
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
    region = intensity[y0:y1, x0:x1]
    region_mean = float(np.mean(region))
    region_std = float(np.std(region))
    uniformity_cv = region_std / max(region_mean, 1e-10) if region_mean > 0 else 0.0

    encircled_energy = float(np.sum(region)) / max(total, 1e-10)
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

    长宽比、均匀性与环围能量的加权组合。

    Args:
        metrics: :func:`compute_square_metrics` 返回的字典。

    Returns:
        ``[0, 1]`` 范围内的浮点质量评分 (越高越好)。
    """
    f_ar = math.exp(-(((metrics["aspect_ratio"] - 1.0) / 0.3) ** 2))
    f_uni = math.exp(-((metrics["uniformity_cv"] / 0.3) ** 2))
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


def zero_order_center(
    frame: np.ndarray,
    refine: bool = True,
    half_win: int | None = None,
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

    Returns:
        ``(x, y)`` 像素坐标 (项目约定)。全暗帧返回帧中心。
    """
    frame = np.asarray(frame)
    if frame.ndim != 2 or frame.size == 0:
        raise ValueError(f"frame must be a non-empty 2D array, got shape {frame.shape}")
    height, width = frame.shape
    if float(frame.max()) <= 0.0:
        return (width // 2, height // 2)

    anchor_y, anchor_x = np.unravel_index(int(np.argmax(frame)), frame.shape)
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
