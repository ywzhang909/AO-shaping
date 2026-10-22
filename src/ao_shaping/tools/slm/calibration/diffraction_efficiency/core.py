"""闪耀光栅衍射效率测量: 纯算法层 (无任何硬件导入)。

为什么单独一层
--------------
本模块只用 numpy / scipy / loguru, **不导入任何设备驱动**。仿真后端
(:mod:`ao_shaping.tools.slm.calibration.diffraction_efficiency.simulate`)
与硬件后端 (:mod:`ao_shaping.tools.slm.calibration.diffraction_efficiency.measure`)
共用这里的同一套 **图案生成 → 阶次识别 → 能量积分 → 效率解算 → 相位响应拟合**
代码。仿真验证的就是硬件要走的那条路, 而不是另一条平行实现。

物理模型
--------
一维锯齿 (闪耀) 相位光栅: 周期 P (面板像元), 相位深度 Δφ (弧度), 一个周期内
相位从 0 线性升到 Δφ 再复位。第 m 阶的傅里叶系数::

    a_m = (exp(i(Δφ - 2πm)) - 1) / (i(Δφ - 2πm))
    η_m = |a_m|^2 = sinc²((Δφ - 2πm) / (2π)),   sinc(x) = sin(πx)/(πx)

所以 η_+1 在 Δφ = 2π 处取 1 (理论上限), η_0 在 Δφ = 2π 处取 0。

    ⚠️ **两侧一级衍射不相等** (锯齿光栅不对称): η₊₁ / η₋₁ = ((Δφ+2π)/(Δφ-2π))²。
    在 Δφ = π 处 η₊₁ = 0.405 而 η₋₁ = 0.045 (差 9 倍); Δφ → 2π 时 η₋₁ → 0
    (能量几乎全在 +1 级)。**把两侧当对称处理, 拟合出的相位深度会系统性偏小**
    (数字孪生上实测偏 11.5%)。所以模型分别算两侧再相加。

**只看一级效率定不出 Δφ** —— 它每 2π 振荡一次, Δφ 与 Δφ+2π 给同一条曲线。
零级的零点是一个尖锐标尺: 联立零级与一级才能唯一确定 Δφ ∈ [0, 2π]。
这也是本工具同时积分零级与两侧一级的唯一理由 (只测 +1 级的做法把这条
信息丢掉了, 于是"欠调制/过调制"的判据变得不可靠)。

拟合用的是**离散器件模型** (:func:`discrete_order_efficiencies`) 而不是上面的
理想连续公式: 理想公式是 P→∞ 的极限, 且完全不含器件灰度→相位的非线性 γ。
拿理想公式去拟合真实器件, Δφ_max 与 γ 会**同时**偏掉 —— γ 越大偏得越多
(非线性把相位剖面从均匀斜坡变成弯曲斜坡, 一级能量被抹到相邻阶次)。

测量量与归一化
--------------
三个 ROI 的积分能量: E0 (零级), E+1, E-1。归一化用**三者和**而不是全帧,
于是整体通量、曝光、慢漂移基本抵消::

    r0 = E0 / (E0 + E+1 + E-1)
    r1 = (E+1 + E-1) / (E0 + E+1 + E-1)

模型侧引入**未调制泄漏** ℓ (偏振不纯 / 填充因子不足的那部分光直漏零级)::

    f0   = (1-ℓ)·η0 + ℓ
    f±1  = (1-ℓ)·η1
    r0   = f0 / (f0 + 2·f1)
    r1   = 2·f1 / (f0 + 2·f1)

拟合参数: Δφ_max (满命令处的真实相位深度), γ (灰度→相位的幂律指数),
ℓ (泄漏比例)。因为 r0 + r1 ≡ 1 (同一归一化), 每个扫描点只提供 **1 个
自由度**, 拟合在 r1 上做, r0 的残差作为一致性检查输出。

术语
----
本模块的中文叙述遵守仓库术语表: 瞳面 (pupil), 实验系统, 加性背景 (杂散光) 项。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from loguru import logger
from scipy.optimize import least_squares

from ao_shaping.utils.image.beam_metrics import prepare_frame

__all__ = [
    "BlazeSpec",
    "OrderDetectionError",
    "OrderMeasurement",
    "PhaseResponseFit",
    "Spot",
    "SweepPoint",
    "blaze_gray",
    "blaze_phase",
    "build_linearization_lut",
    "diagnose",
    "discrete_order_efficiencies",
    "evaluate_fwhm_gate",
    "fit_phase_response",
    "FwhmGate",
    "measure_orders",
    "model_ratios",
    "order_efficiencies",
    "run_sweep",
]

#: 判据阈值: 满命令相位深度低于此值 = 欠调制 (相位包裹不完整)。
UNDER_MODULATION_RAD: float = 1.8 * math.pi
#: 判据阈值: 满命令相位深度高于此值 = 过调制 (相位包裹错误, 出现高阶鬼影)。
OVER_MODULATION_RAD: float = 2.2 * math.pi
#: 零级峰值占满量程的上限 (超过即过曝, 效率会被严重低估)。
SATURATION_FRACTION: float = 0.8
#: "显著光斑"的判据: 峰值 ≥ 帧内峰值的这个比例。只有显著光斑才允许参与
#: "零级锚点改用两侧中点"这类结构性判断 —— 否则低相位端 (只有零级可见)
#: 会被两个弱噪声光斑拖走, 零级能量直接测错。
SIGNIFICANT_FRACTION: float = 0.2
#: 一级衍射总效率峰值的下限 (低于此值 = 偏振不纯 / 填充因子不足)。
#: 理想情况下 Δφ=2π 时能量全在 +1 级, 峰值 = 1 - 泄漏。
PEAK_ETA_WARN: float = 0.75


class OrderDetectionError(RuntimeError):
    """帧内找不到可用的衍射阶次 (曝光过低 / 图案未上屏 / 阶次不可分辨)。"""


# ---------------------------------------------------------------------------
# 图案生成
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlazeSpec:
    """一维闪耀光栅规格。

    Attributes:
        period_px: 光栅周期 (面板像元)。必须 ≥ 2, 否则远场相邻阶次不可分辨。
        axis: 光栅条纹方向。``"x"`` = 相位沿**列**变化 (竖直条纹, 衍射阶次
            沿行方向排开); ``"y"`` = 相位沿**行**变化。
        depth_rad: 锯齿相位深度 (弧度, raw 未包裹)。
        pupil_center: 瞳面 (pupil) 中心 (x, y), 面板像元坐标。
        pupil_radius: 瞳面半径 (面板像元); 0 = 不加瞳面掩膜 (整块面板)。
    """

    period_px: int
    axis: str = "x"
    depth_rad: float = 0.0
    pupil_center: tuple[int, int] = (0, 0)
    pupil_radius: int = 0

    def __post_init__(self) -> None:
        if int(self.period_px) < 2:
            raise ValueError(f"period_px 必须 >= 2, 得到 {self.period_px}")
        if str(self.axis).lower() not in ("x", "y"):
            raise ValueError(f"axis 必须是 'x' 或 'y', 得到 {self.axis!r}")
        if float(self.depth_rad) < 0.0:
            raise ValueError(f"depth_rad 不能为负, 得到 {self.depth_rad}")
        if int(self.pupil_radius) < 0:
            raise ValueError(f"pupil_radius 不能为负, 得到 {self.pupil_radius}")


def _pupil_mask(spec: BlazeSpec, panel_shape: tuple[int, int]) -> np.ndarray:
    """瞳面圆盘掩膜 (True = 在瞳面内)。"""
    height, width = int(panel_shape[0]), int(panel_shape[1])
    cx, cy = (int(spec.pupil_center[0]), int(spec.pupil_center[1]))
    dy = np.arange(height, dtype=np.float64)[:, None] - cy
    dx = np.arange(width, dtype=np.float64)[None, :] - cx
    radius = float(spec.pupil_radius)
    return (dx * dx + dy * dy) <= radius * radius


def blaze_phase(spec: BlazeSpec, panel_shape: tuple[int, int]) -> np.ndarray:
    """生成锯齿相位图, 返回 **raw 未包裹弧度** (float64)。

    契约: 本函数**不做** mod 2π —— 全仓库唯一的包裹点是驱动里的
    ``Santec.create_phase_from_array()``。瞳面 (pupil) 外为 0。

    **采样约定**: 一个周期内第 j 个像元 (j = 1..P) 取相位 ``depth·j/P``,
    即斜坡的**整步长是 depth/P**, 最后一个像元恰好落在 ``depth`` 上。
    这与"第 j 个像元取 depth·j/P (j = 0..P−1)"差一个像元的偏置: 后者
    的最大相位只有 ``depth·(P−1)/P``, 会把"命令深度"系统性地缩小
    (P=16 时 6%), 直接进到拟合里就是一个假的欠调制。

    Args:
        spec: 光栅规格。
        panel_shape: ``(height, width)`` 面板尺寸。

    Returns:
        ``(height, width)`` 的 float64 弧度相位。
    """
    height, width = int(panel_shape[0]), int(panel_shape[1])
    period = float(spec.period_px)
    axis = str(spec.axis).lower()
    span = width if axis == "x" else height
    ramp = ((np.arange(span, dtype=np.float64) % period) + 1.0) / period
    phase = np.empty((height, width), dtype=np.float64)
    if axis == "x":
        phase[:] = ramp[None, :]
    else:
        phase[:] = ramp[:, None]
    phase *= float(spec.depth_rad)
    if int(spec.pupil_radius) > 0:
        phase[~_pupil_mask(spec, panel_shape)] = 0.0
    return phase


def blaze_gray(spec: BlazeSpec, panel_shape: tuple[int, int], max_gray: int) -> np.ndarray:
    """生成灰度斜坡 (uint16), 直接写进 SLM 内存。

    与 :func:`blaze_phase` 的区别: 这里**绕开**驱动的 弧度→灰度 管线
    (波前矫正图 / 相位 LUT / 平移), 因此测到的是**器件本征**的
    灰度→相位关系。灰度值上限 ``max_gray`` 必须 ≤ 设备的 2π 灰度, 否则
    相位在斜坡中途绕回, 有效深度被削减。

    Args:
        spec: 光栅规格 (``depth_rad`` 决定斜坡的最大灰度)。
        panel_shape: ``(height, width)`` 面板尺寸。
        max_gray: 设备的 2π 灰度 (SLM-200 @1064nm 为 993)。

    Returns:
        ``(height, width)`` 的 uint16 灰度图。
    """
    if int(max_gray) < 1:
        raise ValueError(f"max_gray 必须 >= 1, 得到 {max_gray}")
    phase = blaze_phase(spec, panel_shape)
    gray = phase / (2.0 * math.pi) * float(max_gray)
    return np.clip(np.rint(gray), 0, int(max_gray)).astype(np.uint16)


# ---------------------------------------------------------------------------
# 阶次识别
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Spot:
    """一个衍射光斑的测量结果。"""

    cy: float
    cx: float
    peak: float
    energy: float
    fwhm_px: float
    ellipticity: float


@dataclass(frozen=True)
class OrderMeasurement:
    """一帧里零级与 ±1 级的完整测量。"""

    zero: Spot | None
    plus_one: Spot | None
    minus_one: Spot | None
    e_zero: float
    e_plus: float
    e_minus: float
    eta_zero: float | None
    eta_plus: float | None
    eta_minus: float | None
    eta_plus_total: float | None
    separation_px: float | None
    predicted_separation_px: float | None
    residual_px: float | None
    peak_adu: float
    zero_center: tuple[float, float] | None
    notes: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """是否至少拿到可用的 ±1 级能量 (拟合的最小输入)。"""
        return self.eta_plus_total is not None


@dataclass(frozen=True)
class SweepPoint:
    """扫描曲线上的一个点。"""

    command: float
    e_zero: float
    e_plus: float
    e_minus: float
    eta_zero: float | None
    eta_plus_total: float | None
    separation_px: float | None
    residual_px: float | None
    peak_adu: float
    fwhm_px: float | None
    ellipticity: float | None
    zero_center: tuple[float, float] | None
    notes: tuple[str, ...] = ()


def _find_peaks(
    clean: np.ndarray, threshold: float, min_separation_px: float, max_spots: int
) -> list[tuple[float, float, float]]:
    """贪心取局部极大: 返回 ``[(cy, cx, peak), ...]``, 按峰值降序。"""
    work = clean.copy()
    height, width = clean.shape
    radius = max(1.0, float(min_separation_px))
    found: list[tuple[float, float, float]] = []
    for _ in range(max(1, int(max_spots))):
        idx = int(np.argmax(work))
        value = float(work.flat[idx])
        if value <= threshold:
            break
        cy, cx = divmod(idx, width)
        found.append((float(cy), float(cx), value))
        y0 = max(0, int(cy - radius))
        y1 = min(height, int(cy + radius) + 1)
        x0 = max(0, int(cx - radius))
        x1 = min(width, int(cx + radius) + 1)
        work[y0:y1, x0:x1] = 0.0
    return found


def _fwhm_profile(profile: np.ndarray) -> float:
    """一维剖面的半高全宽 (像素)。"""
    if profile.size == 0:
        return 0.0
    peak = float(profile.max())
    if peak <= 0.0:
        return 0.0
    above = np.nonzero(profile >= peak / 2.0)[0]
    if above.size < 2:
        return 0.0
    return float(above[-1] - above[0] + 1)


def _integrate_spot(
    clean: np.ndarray, cy: float, cx: float, radius: float
) -> tuple[float, float, float, float, float]:
    """在圆 ROI 内积分光斑。

    Returns:
        ``(energy, cy_centroid, cx_centroid, fwhm_px, ellipticity)``。
    """
    height, width = clean.shape
    y0 = max(0, int(round(cy - radius)))
    y1 = min(height, int(round(cy + radius)) + 1)
    x0 = max(0, int(round(cx - radius)))
    x1 = min(width, int(round(cx + radius)) + 1)
    if y1 <= y0 or x1 <= x0:
        return 0.0, float(cy), float(cx), 0.0, 1.0
    patch = clean[y0:y1, x0:x1]
    dy = np.arange(y0, y1, dtype=np.float64)[:, None] - cy
    dx = np.arange(x0, x1, dtype=np.float64)[None, :] - cx
    mask = (dx * dx + dy * dy) <= radius * radius
    values = np.where(mask, patch, 0.0)
    total = float(values.sum())
    if total <= 0.0:
        return 0.0, float(cy), float(cx), 0.0, 1.0
    my = float((values * dy).sum() / total + cy)
    mx = float((values * dx).sum() / total + cx)
    row_profile = values.sum(axis=1)
    col_profile = values.sum(axis=0)
    fwhm = 0.5 * (_fwhm_profile(row_profile) + _fwhm_profile(col_profile))
    # 二阶矩 → 椭圆率 (1.0 = 圆)。离焦/像散会把它推离 1。
    ddy = (np.arange(y0, y1, dtype=np.float64)[:, None] - my) ** 2
    ddx = (np.arange(x0, x1, dtype=np.float64)[None, :] - mx) ** 2
    s_yy = float((values * ddy).sum() / total)
    s_xx = float((values * ddx).sum() / total)
    s_xy = float((values * (dy * dx)).sum() / total)
    trace = s_xx + s_yy
    det = s_xx * s_yy - s_xy * s_xy
    disc = max(trace * trace / 4.0 - det, 0.0)
    lam_max = trace / 2.0 + math.sqrt(disc)
    lam_min = trace / 2.0 - math.sqrt(disc)
    ellipticity = math.sqrt(lam_max / lam_min) if lam_min > 1e-12 else float("inf")
    return total, my, mx, fwhm, ellipticity


def _pick_orders(
    peaks: list[tuple[float, float, float]],
    frame_peak: float,
    expected_center: tuple[float, float] | None,
    predicted_separation_px: float | None,
    frame_shape: tuple[int, int],
) -> tuple[
    tuple[float, float],
    tuple[float, float, float] | None,
    tuple[float, float, float] | None,
    list[str],
]:
    """从光斑列表里挑出零级锚点与两侧的一级衍射光斑。

    决策顺序 (为什么这么排):

    1. **有标定中心** → 先看标定中心附近 (容差 0.25×预测间距, 下限 25px)
       有没有光斑:
       * 有 → 取最近的那个做零级锚点。标定中心可能因光路漂移偏了几十
         像素, 所以只作"最近者"参考, 不作硬门限。
       * 没有 → **零级被熄灭** (Δφ 接近 2π 时零级能量趋零, 或者它落在
         检出阈值以下)。此时两个一级衍射斑关于零级中心对称, 于是改用
         "关于标定中心最对称的一对光斑的中点"当零级中心。这一支不能省:
         少了它, 熄灭点会被误当成零级, 效率直接算错。
    2. **没有标定中心** → 取最亮的光斑做零级锚点。
    3. 剩下的光斑沿"锚点 → 最亮旁斑"方向投影, 按投影的正负与大小分给
       两侧 (衍射阶次与零级共线)。

    **+/- 的定义**: 沿"零级 → 更亮的一级"方向为正侧。本光路的 SLM 轴与
    传感器轴交叉 (面板 x 向在 CCD 上表现为行方向), 几何标定里的 K 只给
    间距不给指向, 因此这里不把 +/− 绑定到面板轴 —— 拟合只用到两侧
    能量之和, 不受此约定影响。

    Returns:
        ``(anchor_center, side_a, side_b, notes)``。
    """
    notes: list[str] = []
    if not peaks:
        raise OrderDetectionError("帧内未检出任何衍射光斑")

    height, width = frame_shape
    sep = (
        float(predicted_separation_px)
        if predicted_separation_px
        else 0.05 * math.hypot(height, width)
    )
    #: "显著" = 峰值 ≥ 帧内峰值的这个比例。低于它的光斑不参与"锚点改用
    #: 中点"这类**结构性**判断。
    strong = [p for p in peaks if p[2] >= SIGNIFICANT_FRACTION * float(frame_peak)]

    if expected_center is not None:
        tolerance = max(0.25 * sep, 25.0)
        near = [
            p
            for p in peaks
            if math.hypot(p[0] - expected_center[0], p[1] - expected_center[1])
            <= tolerance
        ]
        if near:
            anchor = min(
                near,
                key=lambda p: math.hypot(
                    p[0] - expected_center[0], p[1] - expected_center[1]
                ),
            )
        else:
            # 零级不在检出范围内 (被熄灭或落到阈值以下)
            pair = None
            if len(strong) >= 2:
                candidate = min(
                    combinations(strong, 2),
                    key=lambda pr: _pair_score(pr, expected_center, sep),
                )
                mid_y = 0.5 * (candidate[0][0] + candidate[1][0])
                mid_x = 0.5 * (candidate[0][1] + candidate[1][1])
                if (
                    math.hypot(mid_y - expected_center[0], mid_x - expected_center[1])
                    <= 0.25 * sep
                ):
                    pair = candidate
            if pair is not None:
                notes.append(
                    "标定中心附近没有光斑 (零级被熄灭), 零级中心改用两侧光斑的中点"
                )
                mid_y = 0.5 * (pair[0][0] + pair[1][0])
                mid_x = 0.5 * (pair[0][1] + pair[1][1])
                return (mid_y, mid_x), pair[0], pair[1], notes
            # 只剩一侧可辨 (锯齿光栅在 Δφ→2π 时 −1 级消失) —— 零级中心
            # 沿用标定中心, 不拿噪声光斑凑对。
            notes.append(
                "标定中心附近没有光斑且只有一侧可辨 (零级被熄灭), 零级中心沿用标定中心"
            )
            return (
                (expected_center[0], expected_center[1]),
                strong[0] if strong else None,
                None,
                notes,
            )
    else:
        anchor = peaks[0]  # peaks 已按峰值降序

    rest = [p for p in peaks if p is not anchor]
    if not rest:
        return (anchor[0], anchor[1]), None, None, notes

    # 方向: 锚点 → 最亮的旁斑
    bright = max(rest, key=lambda p: p[2])
    vx, vy = bright[1] - anchor[1], bright[0] - anchor[0]
    norm = math.hypot(vx, vy)
    if norm <= 0.0:
        return (anchor[0], anchor[1]), None, None, notes
    ux, uy = vx / norm, vy / norm

    projections: list[tuple[float, tuple[float, float, float]]] = []
    for p in rest:
        t = (p[1] - anchor[1]) * ux + (p[0] - anchor[0]) * uy
        projections.append((t, p))

    # 一级衍射**只在几何预测的窗口里**找 (0.5×sep ~ 2×sep)。
    #
    # 为什么必须这么挑: 零级光斑自带一圈 **Airy 旁瓣环** (数字孪生上实测
    # 旁瓣峰值 ≈ 零级峰值的 8%), 它离零级只有 ~9px, 而在低调制深度处
    # 一级衍射本身也只占几个百分点 —— 不设窗口的话, 旁瓣环会被当成一级
    # 衍射, 效率直接算成"零级几乎没有、一级占 2/3"这种胡说。
    #
    # 窗口下界 (0.5×sep) 把旁瓣环挡在外面, 上界 (2×sep) 留出几何标定的
    # 不确定度余量。窗口内一个都没有 → 这一点的 r1 就是 0 (深度太浅, 一级
    # 衍射确实不可见), 这是**正确**的读数, 不是失败。
    if predicted_separation_px:
        lo, hi = 0.5 * sep, 2.0 * sep
    else:
        lo, hi = 0.0, math.inf

    positive = min(
        (item for item in projections if lo <= item[0] <= hi),
        key=lambda item: abs(item[0] - sep),
        default=None,
    )
    negative = min(
        (item for item in projections if -hi <= item[0] <= -lo),
        key=lambda item: abs(-item[0] - sep),
        default=None,
    )
    side_a = positive[1] if positive is not None else None
    side_b = negative[1] if negative is not None else None
    if side_a is None and side_b is None:
        if predicted_separation_px:
            strong_outside = [
                p for p in rest if p[2] >= SIGNIFICANT_FRACTION * float(frame_peak)
            ]
            notes.append(
                "几何预测窗口 ("
                f"{lo:.0f}~{hi:.0f}px) 内没有一级衍射光斑, 本点一级效率记为 0"
                + (
                    " (窗口外却有强光斑: 几何标定可能已过期)"
                    if strong_outside
                    else ""
                )
            )
        else:
            side_a, side_b = (rest + [None, None])[:2]
            notes.append("未找到与锚点共线的旁斑, 退回按能量取前两个光斑")

    if side_a is not None and side_b is not None:
        # 只有两侧都"显著"时才允许改写零级中心 —— 否则低相位端 (只有零级
        # 可见, 旁斑是弱噪声) 会被两个弱光斑拖走, 零级能量直接测错。
        both_strong = (
            side_a[2] >= SIGNIFICANT_FRACTION * float(frame_peak)
            and side_b[2] >= SIGNIFICANT_FRACTION * float(frame_peak)
        )
        mid_y = 0.5 * (side_a[0] + side_b[0])
        mid_x = 0.5 * (side_a[1] + side_b[1])
        offset = math.hypot(mid_y - anchor[0], mid_x - anchor[1])
        if both_strong and offset > 0.3 * sep:
            notes.append(
                f"一级衍射中点偏离锚点 {offset:.1f}px (>0.3×间距 {sep:.1f}px), "
                "零级中心改用中点"
            )
            anchor = (mid_y, mid_x, 0.0)
    return (anchor[0], anchor[1]), side_a, side_b, notes


def _pair_score(
    pair: tuple[tuple[float, float, float], tuple[float, float, float]],
    center: tuple[float, float],
    sep: float,
) -> float:
    """"两个光斑是否关于给定中心对称"的打分 (越小越好)。"""
    mid_y = 0.5 * (pair[0][0] + pair[1][0])
    mid_x = 0.5 * (pair[0][1] + pair[1][1])
    offset = math.hypot(mid_y - center[0], mid_x - center[1]) / max(sep, 1.0)
    spacing = 0.5 * math.hypot(
        pair[0][0] - pair[1][0], pair[0][1] - pair[1][1]
    )
    spacing_err = abs(spacing - sep) / max(sep, 1.0)
    return offset + spacing_err


def _wide_spot_metrics(
    clean: np.ndarray, cy: float, cx: float, radius: float
) -> tuple[float, float]:
    """在**大窗口**里测光斑的 FWHM 与椭圆率。

    为什么必须单独开窗口: 能量积分的 ROI 是按几何间距取的 (0.08×间距),
    所以在 ROI 里测 FWHM 会被 ROI 尺寸**截顶** —— 实测离焦时光斑从 9px
    涨到 115px, 而 ROI 半径只有 19px, 于是 FWHM 永远报 ~33px, 读数"看起来
    正常"。大窗口 (半径 < 间距, 不碰到邻居阶次) 才能如实反映光斑宽度。

    Returns:
        ``(fwhm_px, ellipticity)``。
    """
    height, width = clean.shape
    y0 = max(0, int(round(cy - radius)))
    y1 = min(height, int(round(cy + radius)) + 1)
    x0 = max(0, int(round(cx - radius)))
    x1 = min(width, int(round(cx + radius)) + 1)
    if y1 <= y0 or x1 <= x0:
        return 0.0, 1.0
    patch = clean[y0:y1, x0:x1]
    iy = int(round(cy)) - y0
    ix = int(round(cx)) - x0
    if not (0 <= iy < patch.shape[0] and 0 <= ix < patch.shape[1]):
        return 0.0, 1.0
    row_profile = patch[iy, :]
    col_profile = patch[:, ix]
    fwhm = 0.5 * (_fwhm_profile(row_profile) + _fwhm_profile(col_profile))
    total = float(patch.sum())
    if total <= 0.0:
        return fwhm, 1.0
    dy = (np.arange(y0, y1, dtype=np.float64) - cy)[:, None]
    dx = (np.arange(x0, x1, dtype=np.float64) - cx)[None, :]
    s_yy = float((patch * dy * dy).sum() / total)
    s_xx = float((patch * dx * dx).sum() / total)
    s_xy = float((patch * (dy * dx)).sum() / total)
    trace = s_xx + s_yy
    det = s_xx * s_yy - s_xy * s_xy
    disc = max(trace * trace / 4.0 - det, 0.0)
    lam_max = trace / 2.0 + math.sqrt(disc)
    lam_min = trace / 2.0 - math.sqrt(disc)
    ellipticity = math.sqrt(lam_max / lam_min) if lam_min > 1e-12 else float("inf")
    return fwhm, ellipticity


def measure_orders(
    frame: np.ndarray,
    *,
    expected_center: tuple[float, float] | None = None,
    predicted_separation_px: float | None = None,
    roi_radius_px: float | None = None,
    threshold_frac: float = 0.02,
    min_separation_px: float = 8.0,
    max_spots: int = 8,
) -> OrderMeasurement:
    """从一帧远场画面里解出零级与 ±1 级, 并给出相对衍射效率。

    Args:
        frame: 相机原始帧 (二维)。
        expected_center: 标定给出的零级中心 ``(cy, cx)``; 用于挑选锚点。
        predicted_separation_px: 标定 K 预测的 ±1 级间距 (CCD 像素); 用于
            限定阶次搜索窗口 (0.5×~2×) 与残差核对, 不参与能量计算。
        roi_radius_px: 能量积分圆半径 (像素); None = 由预测间距推 (0.08×间距,
            下限 8px)。
        threshold_frac: 光斑检出阈值 (相对帧内峰值)。默认 2%: 一级衍射的
            **弱侧** (锯齿光栅在低调制深度处两侧相差可达 9 倍) 可能只有
            几个百分点, 阈值太高会把它整条丢掉, 使 r1 系统性偏低。零级
            周围的 Airy 旁瓣环不受影响 —— 它被几何窗口挡在外面。
        min_separation_px: 光斑合并的最小间距 (像素)。
        max_spots: 最多检出的光斑数。

    Returns:
        :class:`OrderMeasurement`。

    Raises:
        OrderDetectionError: 帧内无有效信号或检不出光斑。
    """
    clean = prepare_frame(np.asarray(frame, dtype=np.float64))
    peak = float(clean.max()) if clean.size else 0.0
    if not np.isfinite(peak) or peak <= 0.0:
        raise OrderDetectionError("帧内无有效信号 (检查曝光 / 图案是否上屏)")
    height, width = clean.shape
    peaks = _find_peaks(
        clean,
        peak * float(threshold_frac),
        float(min_separation_px),
        int(max_spots),
    )
    if not peaks:
        raise OrderDetectionError("未检出衍射光斑 (阈值过高或画面全暗)")
    anchor, side_a, side_b, notes = _pick_orders(
        peaks, peak, expected_center, predicted_separation_px, (height, width)
    )
    if roi_radius_px is not None:
        radius = float(roi_radius_px)
    elif predicted_separation_px:
        radius = max(8.0, 0.08 * float(predicted_separation_px))
    else:
        radius = 12.0

    # 光斑宽度在**大窗口**里测 (半径 < 阶次间距, 不碰邻居): 能量 ROI 只有
    # 0.08×间距, 在它里面测 FWHM 会被截顶, 离焦时永远报一个"看起来正常"
    # 的小值, 于是"光斑比阶次间距还宽"这种致命状态完全看不出来。
    window_radius = (
        0.45 * float(predicted_separation_px)
        if predicted_separation_px
        else 0.1 * math.hypot(height, width)
    )
    wide_fwhm, wide_ellipticity = _wide_spot_metrics(
        clean, anchor[0], anchor[1], window_radius
    )
    if predicted_separation_px and wide_fwhm > 0.3 * float(predicted_separation_px):
        notes.append(
            f"零级光斑过宽 (FWHM={wide_fwhm:.0f}px > 0.3×间距 "
            f"{0.3 * float(predicted_separation_px):.0f}px): 阶次不可分辨, "
            "本点读数不可信 (先做离焦补偿)"
        )

    e_zero, zy, zx, _zfwhm, _zell = _integrate_spot(clean, anchor[0], anchor[1], radius)
    zi, zj = int(round(zy)), int(round(zx))
    zero_spot = Spot(
        cy=zy,
        cx=zx,
        peak=float(clean[zi, zj]) if 0 <= zi < height and 0 <= zj < width else 0.0,
        energy=e_zero,
        fwhm_px=wide_fwhm,
        ellipticity=wide_ellipticity,
    )

    e_plus = 0.0
    e_minus = 0.0
    plus_spot: Spot | None = None
    minus_spot: Spot | None = None
    for side, is_plus in ((side_a, True), (side_b, False)):
        if side is None:
            continue
        energy, py, px, _pf, _pe = _integrate_spot(clean, side[0], side[1], radius)
        side_fwhm, side_ell = _wide_spot_metrics(clean, py, px, window_radius)
        spot = Spot(py, px, side[2], energy, side_fwhm, side_ell)
        if is_plus:
            e_plus, plus_spot = energy, spot
        else:
            e_minus, minus_spot = energy, spot

    total = e_zero + e_plus + e_minus
    eta_zero = e_zero / total if total > 0.0 else None
    eta_plus = e_plus / total if total > 0.0 else None
    eta_minus = e_minus / total if total > 0.0 else None
    eta_plus_total = (e_plus + e_minus) / total if total > 0.0 else None

    separation: float | None = None
    if plus_spot is not None and minus_spot is not None:
        separation = math.hypot(
            plus_spot.cy - minus_spot.cy, plus_spot.cx - minus_spot.cx
        ) / 2.0
    elif plus_spot is not None or minus_spot is not None:
        one = plus_spot or minus_spot
        separation = math.hypot(one.cy - zy, one.cx - zx)
    residual: float | None = None
    if separation is not None and predicted_separation_px:
        residual = abs(separation - float(predicted_separation_px))

    return OrderMeasurement(
        zero=zero_spot,
        plus_one=plus_spot,
        minus_one=minus_spot,
        e_zero=e_zero,
        e_plus=e_plus,
        e_minus=e_minus,
        eta_zero=eta_zero,
        eta_plus=eta_plus,
        eta_minus=eta_minus,
        eta_plus_total=eta_plus_total,
        separation_px=separation,
        predicted_separation_px=(
            float(predicted_separation_px) if predicted_separation_px else None
        ),
        residual_px=residual,
        peak_adu=peak,
        zero_center=(zy, zx),
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# 扫描协议 (仿真与硬件共用)
# ---------------------------------------------------------------------------


def run_sweep(
    pattern_for: Callable[[float], np.ndarray],
    display: Callable[[np.ndarray], None],
    capture: Callable[[], np.ndarray],
    commands: Sequence[float],
    *,
    expected_center: tuple[float, float] | None = None,
    predicted_separation_px: float | None = None,
    roi_radius_px: float | None = None,
    on_point: Callable[[int, int, SweepPoint], None] | None = None,
) -> list[SweepPoint]:
    """跑完一条扫描曲线 (仿真与硬件共用的唯一协议)。

    Args:
        pattern_for: 命令值 → 灰度图 (uint16) 的构造函数。
        display: 把灰度图下发到设备 (仿真里可以是空实现)。
        capture: 采一帧并返回 (二维数组)。
        commands: 扫描命令序列 (灰度值或弧度, 取决于后端)。
        expected_center: 标定零级中心 ``(cy, cx)``。
        predicted_separation_px: 标定预测的 ±1 级间距。
        roi_radius_px: 能量积分圆半径。
        on_point: 进度回调 ``(序号, 总数, 该点结果)``。

    Returns:
        扫描点列表。

    Raises:
        OrderDetectionError: 某一帧解不出阶次时中断 (数据不可用, 不应静默跳过)。
    """
    points: list[SweepPoint] = []
    total = len(commands)
    for index, command in enumerate(commands, start=1):
        gray = pattern_for(float(command))
        display(gray)
        frame = capture()
        measurement = measure_orders(
            frame,
            expected_center=expected_center,
            predicted_separation_px=predicted_separation_px,
            roi_radius_px=roi_radius_px,
        )
        point = SweepPoint(
            command=float(command),
            e_zero=measurement.e_zero,
            e_plus=measurement.e_plus,
            e_minus=measurement.e_minus,
            eta_zero=measurement.eta_zero,
            eta_plus_total=measurement.eta_plus_total,
            separation_px=measurement.separation_px,
            residual_px=measurement.residual_px,
            peak_adu=measurement.peak_adu,
            fwhm_px=(measurement.zero.fwhm_px if measurement.zero else None),
            ellipticity=(measurement.zero.ellipticity if measurement.zero else None),
            zero_center=measurement.zero_center,
            notes=measurement.notes,
        )
        points.append(point)
        if on_point is not None:
            on_point(index, total, point)
    return points


# ---------------------------------------------------------------------------
# 光斑宽度门禁 (在焦参照值的倍数)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FwhmGate:
    """零级光斑宽度的**相对**门禁。

    门禁不按绝对像素定, 而按"相对在焦参照值的倍数"定:

    * **参照值** = 扫描里命令最小那一点 (平相位) 的零级 FWHM, 也就是这条
      光路在**本次运行开始时**的光斑宽度; 也可以由调用方传入一条已知在焦
      的光路量出来的绝对参照值 (跨次运行可比)。
    * **门限** = ``gate_ratio × 参照值`` (默认 3 倍)。
    * 超过门限的点 = 光斑在扫描过程中变糊了 (机械漂移 / 热漂移 / 焦面移动),
      它的**相位深度读数被截断失真** ⇒ 拟合时剔除, 判定时不可信。

    **零级被熄灭的点** (高调制端, ``eta_zero`` 很小) 的宽度读数是噪声底上的
    读数, 拿它当判据会把最有信息量的那批点误杀, 所以它们**跳过**门禁评估
    (但仍参与拟合)。⚠️ 只有在门禁**没有**触发时这个"跳过"才成立: 一旦
    门禁触发, 整轮扫描就不可信了, 那些被跳过的点同样不可信 (噪声底正是光路
    变糊的产物), 于是它们**一并从拟合里剔除** —— 否则拟合会靠这几个点凑出
    一个看着正常其实荒唐的深度。

    为什么不用绝对阈值: 绝对阈值 ("FWHM > 0.3×阶次间距 = 71px") 只有在
    阶次间距已知时才有意义, 而且它只能抓住"整条光路都离焦"这种极端情况;
    相对门禁抓的是**扫描过程中光斑宽度的变化**, 这才是拟合被污染的直接
    原因。两者互补: 相对门禁管"这一轮里变糊了", 绝对判据管"一开始就糊了"。
    要抓后者, 传入一条**在焦**光路量出来的绝对参照值 (``reference_fwhm_px``)
    —— 那时越限会直接把可用点压到不足 3 个, 拟合**显式失败**而不是给出
    一个偏了的深度。
    """

    reference_fwhm_px: float
    gate_ratio: float
    limit_px: float
    violations: tuple[int, ...] = ()
    max_ratio: float = 1.0
    evaluated_points: int = 0
    skipped_points: int = 0
    skipped_indices: tuple[int, ...] = ()

    @property
    def ok(self) -> bool:
        """是否所有被评估的点都在门限内。"""
        return not self.violations


def evaluate_fwhm_gate(
    points: Sequence[SweepPoint],
    *,
    reference_fwhm_px: float | None = None,
    gate_ratio: float = 3.0,
    min_eta_zero: float = 0.05,
) -> FwhmGate:
    """按"零级光斑宽度相对在焦参照值的倍数"评估这一轮扫描。

    Args:
        points: 扫描点。
        reference_fwhm_px: 绝对参照值 (px)。给了就用它 (跨次运行可比);
            None = 取扫描里**命令最小**那一点 (平相位) 的零级 FWHM 当参照。
        gate_ratio: 门限倍数 (默认 3)。
        min_eta_zero: 零级能量占比低于此值的点**不参与**门禁评估 ——
            高调制端零级被熄灭时, 那里量到的"宽度"是噪声底上的读数, 拿它
            当门禁判据会把最有信息量的那批点误杀。

    Returns:
        :class:`FwhmGate`。
    """
    ratio = float(gate_ratio)
    if ratio <= 0.0:
        raise ValueError(f"gate_ratio 必须为正, 得到 {gate_ratio}")

    reference = (
        float(reference_fwhm_px)
        if reference_fwhm_px is not None and float(reference_fwhm_px) > 0.0
        else float("nan")
    )
    if not np.isfinite(reference):
        # 参照值 = 命令最小的点 (平相位: 零级最亮、最锐)
        candidates = [p for p in points if p.fwhm_px]
        if not candidates:
            return FwhmGate(
                reference_fwhm_px=float("nan"),
                gate_ratio=ratio,
                limit_px=float("nan"),
            )
        reference = float(min(candidates, key=lambda p: p.command).fwhm_px or 0.0)
    if reference <= 0.0:
        return FwhmGate(
            reference_fwhm_px=float("nan"),
            gate_ratio=ratio,
            limit_px=float("nan"),
        )

    limit = ratio * reference
    violations: list[int] = []
    skipped_indices: list[int] = []
    max_ratio = 1.0
    evaluated = 0
    skipped = 0
    for index, point in enumerate(points):
        if not point.fwhm_px:
            continue
        if point.eta_zero is not None and point.eta_zero < min_eta_zero:
            skipped += 1  # 零级被熄灭 → 宽度读数无意义, 不参与门禁
            skipped_indices.append(index)
            continue
        evaluated += 1
        current = float(point.fwhm_px) / reference
        max_ratio = max(max_ratio, current)
        if current > ratio:
            violations.append(index)
    return FwhmGate(
        reference_fwhm_px=reference,
        gate_ratio=ratio,
        limit_px=limit,
        violations=tuple(violations),
        max_ratio=max_ratio,
        evaluated_points=evaluated,
        skipped_points=skipped,
        skipped_indices=tuple(skipped_indices),
    )


# ---------------------------------------------------------------------------
# 理论模型与拟合
# ---------------------------------------------------------------------------


def order_efficiencies(depth_rad: np.ndarray | float) -> tuple[np.ndarray, float]:
    """锯齿光栅零级与 **+1 级** 的理论衍射效率。

    ``η_m = sinc²((Δφ - 2πm)/(2π))``, ``sinc(x) = sin(πx)/(πx)``。

    ⚠️ **两侧一级衍射不相等** (锯齿光栅不对称): 由
    ``|η_m| ∝ |sin((Δφ-2πm)/2)/(Δφ-2πm)|`` 得

        η₊₁ / η₋₁ = ((Δφ + 2π)/(Δφ - 2π))²

    在 Δφ = π 处 η₊₁ = 0.405 而 η₋₁ = 0.045 (差 9 倍); 在 Δφ = 2π 处
    η₋₁ = 0 (能量全在 +1 级)。**把两侧当成对称会让拟合出的相位深度系统性
    偏小** (实测 ~11%) —— 所以模型必须分别算两侧, 见
    :func:`negative_order_efficiency` 与 :func:`discrete_order_efficiencies`。

    Args:
        depth_rad: 相位深度 (弧度), 可为标量或数组。

    Returns:
        ``(eta_zero, eta_one)``; 标量输入时 ``eta_zero`` 为 0-d 数组。
    """
    depth = np.asarray(depth_rad, dtype=np.float64)
    eta_zero = np.sinc(depth / (2.0 * math.pi)) ** 2
    eta_one = np.sinc((depth - 2.0 * math.pi) / (2.0 * math.pi)) ** 2
    return eta_zero, eta_one


def negative_order_efficiency(depth_rad: np.ndarray | float) -> np.ndarray:
    """锯齿光栅 **−1 级** 的理论衍射效率 (与 +1 级不相等, 见 :func:`order_efficiencies`)。"""
    depth = np.asarray(depth_rad, dtype=np.float64)
    return np.sinc((depth + 2.0 * math.pi) / (2.0 * math.pi)) ** 2


def model_ratios(depth_rad: np.ndarray | float, leakage: float) -> tuple[np.ndarray, np.ndarray]:
    """**理想连续**模型给出的归一化比值 ``(r0, r1)``。

    两侧的一级衍射**分别计算** (锯齿光栅不对称, 见 :func:`order_efficiencies`);
    泄漏 ℓ 表示"未被调制的那部分光" (偏振不纯 / 填充因子不足), 它全部
    落在零级里。

    Args:
        depth_rad: 相位深度 (弧度)。
        leakage: 泄漏比例 ∈ [0, 1]。

    Returns:
        ``(r0, r1)``, 满足 ``r0 + r1 == 1``。
    """
    eta_zero, eta_plus = order_efficiencies(depth_rad)
    eta_minus = negative_order_efficiency(depth_rad)
    ell = float(leakage)
    f_zero = (1.0 - ell) * eta_zero + ell
    f_plus = (1.0 - ell) * eta_plus
    f_minus = (1.0 - ell) * eta_minus
    denom = f_zero + f_plus + f_minus
    denom = np.where(denom <= 0.0, np.nan, denom)
    return f_zero / denom, (f_plus + f_minus) / denom


def discrete_order_efficiencies(
    command_fraction: np.ndarray | float,
    depth_max_rad: float,
    gamma: float,
    period_px: int,
    leakage: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """**离散器件模型**给出的归一化比值 ``(r0, r1)``。

    面板上一个周期内第 j 个像元 (j = 1..P) 的灰度是 ``g_j = g_max·u·j/P``
    (``u`` = 命令/满量程), 器件把灰度映射成相位 ``φ_j = Δφ_max·(u·j/P)^γ``。
    于是第 m 阶的傅里叶系数::

        a_m = (1/P)·Σ_j exp(i(φ_j - 2πm(j-1)/P))

    相比理想连续公式 (``sinc²``) 这里有三项**真实器件才有**的效应:

    1. **像元离散化**: 理想公式是 P→∞ 的极限, P=16 时差 0.3%, P=8 时差 1.4%。
    2. **器件灰度→相位的非线性 γ**: 相位剖面不再是均匀斜坡, 一级能量被
       抹开 (爬到相邻阶次)。**拿理想公式去拟合非线性器件, Δφ_max 与 γ
       会同时偏掉** —— γ 越大偏得越多。
    3. **两侧一级不等**: 锯齿光栅的 ±1 级不对称 (见
       :func:`order_efficiencies`), 必须分别算再相加。

    所以拟合用本函数 (离散模型), 理想公式只作为参照与诊断基准。

    Args:
        command_fraction: 命令/满量程 ``u`` ∈ [0, 1], 标量或数组。
        depth_max_rad: 满命令处的真实相位深度 (弧度)。
        gamma: 灰度→相位的幂律指数。
        period_px: 光栅周期 (面板像元)。
        leakage: 未调制泄漏比例 ∈ [0, 1]。

    Returns:
        ``(r0, r1)``; 标量输入时返回长度为 1 的数组。
    """
    u = np.atleast_1d(np.asarray(command_fraction, dtype=np.float64))
    period = int(period_px)
    if period < 2:
        raise ValueError(f"period_px 必须 >= 2, 得到 {period_px}")
    j = np.arange(1, period + 1, dtype=np.float64) / float(period)
    idx = np.arange(period, dtype=np.float64)
    phase = float(depth_max_rad) * np.power(
        np.clip(u, 0.0, None)[:, None] * j[None, :], float(gamma)
    )
    base = np.exp(1j * phase)
    a_zero = base.sum(axis=1) / float(period)
    a_plus = (base * np.exp(-2j * math.pi * idx / float(period))).sum(axis=1) / float(period)
    a_minus = (base * np.exp(+2j * math.pi * idx / float(period))).sum(axis=1) / float(period)
    ell = float(leakage)
    f_zero = (1.0 - ell) * np.abs(a_zero) ** 2 + ell
    f_plus = (1.0 - ell) * np.abs(a_plus) ** 2
    f_minus = (1.0 - ell) * np.abs(a_minus) ** 2
    denom = f_zero + f_plus + f_minus
    denom = np.where(denom <= 0.0, np.nan, denom)
    return f_zero / denom, (f_plus + f_minus) / denom


@dataclass(frozen=True)
class PhaseResponseFit:
    """相位响应拟合结果。"""

    depth_max_rad: float
    gamma: float
    leakage: float
    residual_rms: float
    n_points: int
    converged: bool
    message: str
    commands: np.ndarray
    model_depth_rad: np.ndarray
    model_eta_plus_total: np.ndarray
    measured_eta_plus_total: np.ndarray
    outliers: tuple[int, ...]
    stderr: dict[str, float]
    gate_excluded: int = 0
    gate_reference_fwhm_px: float = float("nan")
    gate_ratio: float = float("nan")

    @property
    def depth_max_pi(self) -> float:
        """满命令相位深度, 以 π 为单位。"""
        return self.depth_max_rad / math.pi


def _fit_model(
    params: np.ndarray,
    u: np.ndarray,
    y: np.ndarray,
    period_px: int,
) -> np.ndarray:
    """scipy ``least_squares`` 的残差函数 (参数向量在前, 固定参数在 ``args``)。"""
    depth_max, gamma, leakage = (float(v) for v in params)
    _, r1 = discrete_order_efficiencies(u, depth_max, gamma, period_px, leakage)
    return np.nan_to_num(r1, nan=0.0) - y


def fit_phase_response(
    commands: Sequence[float],
    eta_plus_total: Sequence[float],
    *,
    command_max: float,
    period_px: int,
    gate: FwhmGate | None = None,
    depth_max_bounds: tuple[float, float] = (0.0, 8.0 * math.pi),
    gamma_bounds: tuple[float, float] = (0.05, 10.0),
) -> PhaseResponseFit:
    """拟合"命令 → 实际相位深度"的响应。

    模型: 实际相位深度 ``Δφ(u) = Δφ_max · u^γ``, ``u = 命令 / command_max``;
    面板上的**离散**相位剖面按 :func:`discrete_order_efficiencies` 计算。
    灰度扫描模式 (``command_max`` = 2π 灰度) 与相位扫描模式
    (``command_max`` = 2π) 用同一个模型: 前者刻画器件本征的灰度→相位律,
    后者刻画"命令弧度 → 器件实际相位"的链路响应 (驱动的 弧度→灰度 换算
    与器件律在模型里合成为同一个幂律)。

    Args:
        commands: 扫描命令值序列。
        eta_plus_total: 对应的 ±1 级总相对效率 (``(E+1+E-1)/(E0+E+1+E-1)``)。
        command_max: 命令上限 (灰度上限或 2π 弧度)。
        period_px: 光栅周期 (面板像元) —— 离散模型需要它。
        gate: 光斑宽度门禁。给了就把**越限的点从拟合里剔除** —— 那些点的
            光斑已经糊到读数被截断, 留着它们会把相位深度**拉偏** (离焦工况
            实测偏 +11%~+12%)。
        depth_max_bounds: Δφ_max 的搜索边界。
        gamma_bounds: γ 的搜索边界。

    Returns:
        :class:`PhaseResponseFit`。
    """
    cmd = np.asarray(commands, dtype=np.float64)
    y = np.asarray(eta_plus_total, dtype=np.float64)
    keep = np.isfinite(cmd) & np.isfinite(y)
    if gate is not None and gate.violations:
        for index in gate.violations:
            if 0 <= index < keep.size:
                keep[index] = False
        # 门禁一旦触发, 整轮扫描就不可信了 —— 那些"因为零级被熄灭而跳过宽度
        # 评估"的点同样不可信 (它们的宽度读数是噪声底上的读数, 而噪声底正是
        # 光路变糊的产物)。把它们留下, 拟合会靠这几个点凑出一个**看着正常其实
        # 荒唐**的深度 (数字孪生实测: 剔除 17/21 个点后, 拟合靠剩下 4 个被跳过
        # 的点给出 1.9775π) —— 那正是这个门禁要消灭的东西。
        for index in gate.skipped_indices:
            if 0 <= index < keep.size:
                keep[index] = False
    cmd, y = cmd[keep], y[keep]
    if cmd.size < 3:
        excluded = (
            len(gate.violations) + len(gate.skipped_indices) if gate is not None else 0
        )
        detail = f" (其中 {excluded} 个被光斑宽度门禁剔除)" if excluded else ""
        raise ValueError(
            f"有效扫描点不足 3 个 (得到 {cmd.size}){detail}, 无法拟合"
        )
    if float(command_max) <= 0.0:
        raise ValueError(f"command_max 必须为正, 得到 {command_max}")
    u = np.clip(cmd / float(command_max), 0.0, 1.0)
    period = int(period_px)

    best = None
    for depth0 in (2.0 * math.pi, 4.0 * math.pi, 6.0 * math.pi):
        for gamma0 in (0.6, 1.0, 1.6):
            for leak0 in (0.0, 0.15):
                try:
                    result = least_squares(
                        _fit_model,
                        x0=np.array([depth0, gamma0, leak0], dtype=np.float64),
                        args=(u, y, period),
                        bounds=(
                            np.array(
                                [
                                    float(depth_max_bounds[0]),
                                    float(gamma_bounds[0]),
                                    0.0,
                                ]
                            ),
                            np.array(
                                [
                                    float(depth_max_bounds[1]),
                                    float(gamma_bounds[1]),
                                    1.0,
                                ]
                            ),
                        ),
                        loss="soft_l1",
                        f_scale=0.02,
                        max_nfev=4000,
                    )
                except Exception as exc:  # 单个初值失败不致命
                    logger.debug("拟合初值 {} 失败: {}", (depth0, gamma0, leak0), exc)
                    continue
                if best is None or float(result.cost) < float(best.cost):
                    best = result
    if best is None:
        raise RuntimeError("相位响应拟合失败: 所有初值都未收敛")

    depth_max, gamma, leakage = (float(v) for v in best.x)
    model_r1 = np.nan_to_num(
        discrete_order_efficiencies(u, depth_max, gamma, period, leakage)[1], nan=0.0
    )
    residual = model_r1 - y
    rms = float(np.sqrt(np.mean(residual**2)))
    mad = float(np.median(np.abs(residual)))
    outliers = tuple(
        int(i) for i in np.nonzero(np.abs(residual) > max(3.0 * mad, 0.02))[0]
    )

    stderr = {"depth_max_rad": float("nan"), "gamma": float("nan"), "leakage": float("nan")}
    try:
        jac = np.asarray(best.jac, dtype=np.float64)
        dof = max(1, jac.shape[0] - jac.shape[1])
        cov = np.linalg.pinv(jac.T @ jac) * (2.0 * float(best.cost) / dof)
        diag = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        stderr = {
            "depth_max_rad": float(diag[0]),
            "gamma": float(diag[1]),
            "leakage": float(diag[2]),
        }
    except Exception as exc:  # 协方差只是附加信息
        logger.debug("拟合协方差计算失败: {}", exc)

    return PhaseResponseFit(
        depth_max_rad=depth_max,
        gamma=gamma,
        leakage=leakage,
        residual_rms=rms,
        n_points=int(cmd.size),
        converged=bool(best.success),
        message=str(best.message),
        commands=cmd,
        model_depth_rad=depth_max * np.power(u, gamma),
        model_eta_plus_total=model_r1,
        measured_eta_plus_total=y,
        outliers=outliers,
        stderr=stderr,
        gate_excluded=(
            (len(gate.violations) + len(gate.skipped_indices))
            if gate is not None and gate.violations
            else (len(gate.violations) if gate is not None else 0)
        ),
        gate_reference_fwhm_px=(
            float(gate.reference_fwhm_px) if gate is not None else float("nan")
        ),
        gate_ratio=float(gate.gate_ratio) if gate is not None else float("nan"),
    )


def build_linearization_lut(
    fit: PhaseResponseFit, *, size: int = 256, gray_max: int = 1023
) -> np.ndarray:
    """由拟合结果生成 **灰度重映射表** (uint16), 使器件相位线性化。

    器件相位律 ``Δφ(u) = Δφ_max·u^γ`` (``u`` = 灰度/满量程)。要让"指令
    灰度索引 i"对应的实际相位线性于 i, 需令输出灰度 ``u_out = u_target^(1/γ)``::

        lut[i] = round(gray_max · (i/(size-1))^(1/γ))

    该表的语义与驱动里的 ``apply_lut_remap`` 一致: **输入灰度索引 → 输出灰度**。

    Args:
        fit: 拟合结果。
        size: 表长。
        gray_max: 设备的 2π 灰度。

    Returns:
        ``size`` 长的 uint16 灰度重映射表。
    """
    if int(size) < 2:
        raise ValueError(f"size 必须 >= 2, 得到 {size}")
    idx = np.arange(int(size), dtype=np.float64) / (int(size) - 1)
    out = float(gray_max) * np.power(idx, 1.0 / max(fit.gamma, 1e-6))
    return np.clip(np.rint(out), 0, int(gray_max)).astype(np.uint16)


# ---------------------------------------------------------------------------
# 诊断
# ---------------------------------------------------------------------------


def diagnose(
    points: Sequence[SweepPoint],
    fit: PhaseResponseFit,
    *,
    full_scale: float | None = None,
    predicted_separation_px: float | None = None,
    gate: FwhmGate | None = None,
) -> dict:
    """给出光路与器件的体检结论 (纯结构化输出, 不写文件)。

    Args:
        points: 扫描点。
        fit: 拟合结果。
        full_scale: 相机满量程 (ADU); 给出后才能判断过曝。
        predicted_separation_px: 标定预测的 ±1 级间距 (用于核对几何标定)。
        gate: 光斑宽度门禁 (相对在焦参照值的倍数)。给了就把"光斑在扫描中
            变糊"这件事写进结论, 并把欠调制/过调制判定标为不可信。

    Returns:
        可 JSON 序列化的结论字典。
    """
    notes: list[str] = []
    depth = fit.depth_max_rad
    if depth < UNDER_MODULATION_RAD:
        modulation = "under"
        notes.append(
            f"满命令相位深度 {depth / math.pi:.2f}π < {UNDER_MODULATION_RAD / math.pi:.1f}π: "
            "欠调制, 相位包裹不完整, 零级残余强"
        )
    elif depth > OVER_MODULATION_RAD:
        modulation = "over"
        notes.append(
            f"满命令相位深度 {depth / math.pi:.2f}π > {OVER_MODULATION_RAD / math.pi:.1f}π: "
            "过调制, 相位包裹错误, 需用线性化查找表截断"
        )
    else:
        modulation = "nominal"
        notes.append(
            f"满命令相位深度 {depth / math.pi:.2f}π 落在 "
            f"{UNDER_MODULATION_RAD / math.pi:.1f}π~{OVER_MODULATION_RAD / math.pi:.1f}π 区间"
        )
    if modulation != "nominal":
        notes.append(
            "⚠️ 该判定隐含三个前提: 器件的灰度→相位响应是幂律、光路在焦 (零级光斑够小)、"
            "填充因子接近 1。前提不成立时判定会**假阳性** (仿真实测: 填充因子 0.8 会把标称器件"
            "报成过调制, 离焦会报 +11% 的深度偏差) —— 见 "
            "report/slm/diffraction_efficiency/sim_nonideality_20261022.md"
        )

    peaks = np.array(
        [p.eta_plus_total for p in points if p.eta_plus_total is not None],
        dtype=np.float64,
    )
    peak_eta = float(peaks.max()) if peaks.size else float("nan")
    if np.isfinite(peak_eta) and peak_eta < PEAK_ETA_WARN:
        notes.append(
            f"一级衍射总效率峰值仅 {peak_eta:.3f} (理想 ≤ 泄漏后应为 1-ℓ): "
            "填充因子偏低, 或入射光含未调制的正交偏振分量"
        )

    residuals = [p.residual_px for p in points if p.residual_px is not None]
    max_residual = float(max(residuals)) if residuals else float("nan")
    if np.isfinite(max_residual) and predicted_separation_px:
        if max_residual > 0.1 * float(predicted_separation_px):
            notes.append(
                f"±1 级位置与标定预测偏离最大 {max_residual:.1f}px "
                f"(>10% 间距): 几何标定可能已过期 (光路动过)"
            )

    ellipt = [p.ellipticity for p in points if p.ellipticity is not None]
    max_ell = float(max(ellipt)) if ellipt else float("nan")
    if np.isfinite(max_ell) and max_ell > 1.25:
        notes.append(
            f"零级光斑椭圆率最大 {max_ell:.2f} (1.0=圆): 像散/离焦或入射角过大"
        )

    peak_adus = [p.peak_adu for p in points]
    max_adu = float(max(peak_adus)) if peak_adus else float("nan")
    saturation = None
    if full_scale:
        saturation = max_adu / float(full_scale)
        if saturation > SATURATION_FRACTION:
            notes.append(
                f"零级峰值 {max_adu:.0f} ADU 达满量程的 {saturation * 100:.0f}% "
                f"(>{SATURATION_FRACTION * 100:.0f}%): 效率会被低估, 需降低曝光"
            )

    if fit.outliers:
        notes.append(
            "拟合残差离群点: 命令值 "
            + ", ".join(f"{fit.commands[i]:.4g}" for i in fit.outliers)
            + " (单帧异常, 建议复测)"
        )

    gate_report: dict = {}
    verdict_trustworthy = True
    if gate is not None and np.isfinite(gate.reference_fwhm_px):
        gate_report = {
            "reference_fwhm_px": float(gate.reference_fwhm_px),
            "gate_ratio": float(gate.gate_ratio),
            "limit_px": float(gate.limit_px),
            "violations": list(gate.violations),
            "max_ratio": float(gate.max_ratio),
            "evaluated_points": int(gate.evaluated_points),
            "skipped_points": int(gate.skipped_points),
            "ok": bool(gate.ok),
        }
        if not gate.ok:
            verdict_trustworthy = False
            notes.append(
                f"零级光斑宽度门禁: {len(gate.violations)}/{gate.evaluated_points} 个点的 "
                f"FWHM 超过在焦参照值 ({gate.reference_fwhm_px:.1f}px) 的 "
                f"{gate.gate_ratio:.0f} 倍 (限 {gate.limit_px:.1f}px, 实测最大 "
                f"{gate.max_ratio:.1f} 倍) —— 这些点已从拟合中剔除, "
                "相位深度判定不可信 (光斑在扫描过程中变糊: 机械/热漂移或焦面移动); "
                f"另有 {gate.skipped_points} 个零级被熄灭的点 (宽度读数无意义) "
                "也一并剔除, 否则拟合会靠它们凑出一个荒唐的深度"
            )
        elif gate.evaluated_points:
            notes.append(
                f"零级光斑宽度门禁通过: 全部 {gate.evaluated_points} 个评估点的 FWHM "
                f"都在在焦参照值 ({gate.reference_fwhm_px:.1f}px) 的 "
                f"{gate.gate_ratio:.0f} 倍以内 (最大 {gate.max_ratio:.1f} 倍)"
            )

    return {
        "modulation": modulation,
        "modulation_verdict_trustworthy": verdict_trustworthy,
        "depth_max_rad": depth,
        "depth_max_pi": depth / math.pi,
        "gamma": fit.gamma,
        "leakage": fit.leakage,
        "peak_eta_plus_total": peak_eta,
        "max_order_position_residual_px": max_residual,
        "max_zero_order_ellipticity": max_ell,
        "max_peak_adu": max_adu,
        "saturation_fraction": saturation,
        "residual_rms": fit.residual_rms,
        "n_points": fit.n_points,
        "fwhm_gate": gate_report,
        "notes": notes,
    }
