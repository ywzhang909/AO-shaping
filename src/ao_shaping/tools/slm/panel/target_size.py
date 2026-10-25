"""目标尺寸（方形目标边长）的自动标定与推荐.

问题
----

2f 实验系统下，方形目标边长存在**最优区间**：

* 太小 ⇒ 零级泄漏 / 散斑 —— 目标盒装不下光束能量，未整形的零级能量堆在中心，
  盒内边缘又补不满；
* 太大 ⇒ 受衍射极限与高频丢失所限 —— 面板的相位自由度合不出大盒所需的
  高空间频率，能量仍集中在自然光斑上，盒的大部分区域是空的。

最优尺寸由四类因素共同决定：

1. **光学硬参数** (λ, f, SLM 像素数 N, 像素间距 p) —— 解析出物理界限
   (:class:`OpticalLimits`)；
2. **器件非理想**（填充因子、相位量化位深、像素串扰）—— 本模块**不建模**，
   只通过「实测指标随尺寸变化」体现（:class:`SizeMetrics`）；
3. **算法配置**（迭代数、初值、约束）—— 扫描时固定并记录在扫描脚本的输出里；
4. **目标图案复杂度**（低频 vs 高频、动态范围、对称性）—— 作为 ``target_type``
   键进查找表（:class:`TargetSizeCalibration`）。

**换透镜 / 波长 / SLM 后必须重新标定**（GRAPHTCS 自校准思想）：把「目标缩放
因子」当可学习参数，用少量实测反馈搜索最优工作点。本模块用**系统签名**
(:meth:`OpticalLimits.signature`) 保证这一点 —— 签名对 (λ, f, N, p, 相机像元,
相机尺寸) 任一参数的改变都会变，:meth:`TargetSizeCalibration.load` 在签名
不匹配时**显式拒绝**沿用旧表，而不是静默使用。

模块边界
--------

* 纯 numpy + 标准库；**不** import torch / 驱动 / SDK。
* 扫描与优化在 ``scripts/slm/sweep_target_size.py``（模型侧 dry-run + 设备侧
  execute）；本模块只提供解析界限、指标容器、峰值窗口选择、单峰拟合与
  查找表。

协议脚本在哪一步调用它（一句话）
--------------------------------

在**开设备后、正式采集前**，用 ``TargetSizeCalibration.load(path, limits=当前
光路的 OpticalLimits)`` 读 ``target_size_calibration.json``（签名不匹配会被
显式拒绝并要求重标定），``cal.recommend(target_type)`` 取出推荐边长（相机
像元），再把这个值直接作为
``ao_shaping.utils.image.target.masks.build_box_core_masks(grid_shape,
centre_ij, box_px, inner_shrink_px, step_ccd_px)`` 的 ``box_px``
冻结方框/核心掩膜。

术语
----

中文叙述遵守 ``docs/terminology_glossary.md``：实验设备一律写「实验系统」，
光学口径一律写「瞳面」，A/B 对照语境一律写「变体」。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from ao_shaping.utils.slm.bench_constants import LASER_WAVELENGTH_M

__all__ = [
    "OpticalLimits",
    "SizeMetrics",
    "TargetSizeCalibration",
    "TargetSizeCalibrationError",
    "default_fractions",
    "select_best_window",
    "composite_scores",
    "degenerate_metrics",
    "fit_optimum",
    "is_valid",
    "DEFAULT_SYSTEM",
    "SYSTEM_SIGNATURE",
]


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_fractions() -> tuple[float, ...]:
    """默认扫描分数：0.05 到 0.80、步长 0.05（**含两端**）。

    用整数步长生成再取整，避免 ``0.05*n`` 的浮点漂移（0.05*6 = 0.30000000000000004）。
    """
    return tuple(round(0.05 * i, 2) for i in range(1, 17))


# ---------------------------------------------------------------------------
# 光学硬参数 → 物理界限
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OpticalLimits:
    """2f 实验系统的光学硬参数，及其解析出的目标尺寸物理界限。

    Attributes:
        wavelength_m: 波长 λ (m)。
        focal_m: 透镜焦距 f (m)。
        slm_pixels: 面板像素数 ``(行数, 列数)``，即 ``(N_y, N_x)``。
        pixel_pitch_m: 面板像素间距 p (m)。
        camera_pixel_m: 相机像元尺寸 (m)。
        camera_shape: 传感器尺寸 ``(行数, 列数)``，即 ``(H, W)`` 像元。
    """

    wavelength_m: float
    focal_m: float
    slm_pixels: tuple[int, int] = (1200, 1920)
    pixel_pitch_m: float = 8e-6
    camera_pixel_m: float = 2.2e-6
    camera_shape: tuple[int, int] = (1944, 2592)

    def __post_init__(self) -> None:
        for name in ("wavelength_m", "focal_m", "pixel_pitch_m", "camera_pixel_m"):
            value = float(getattr(self, name))  # type: ignore[arg-type]
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"OpticalLimits.{name} 必须是正的有限值: {value!r}")
        for name in ("slm_pixels", "camera_shape"):
            pair = getattr(self, name)  # type: ignore[assignment]
            if len(pair) != 2 or any(int(v) <= 0 for v in pair):
                raise ValueError(f"OpticalLimits.{name} 必须是两个正整数: {pair!r}")
            object.__setattr__(self, name, (int(pair[0]), int(pair[1])))

    # -- 解析界限 -----------------------------------------------------------
    @property
    def far_field_sample_m(self) -> float:
        """焦面单像素尺寸 ``Δx = λf/(N_x·p)`` (m)。

        推导：2f 系统中透镜对瞳面做傅里叶变换。面板 x 方向有 ``N_x`` 个像素、
        间距 p，瞳面被离散化为 N_x 个采样、整个孔径宽 ``W = N_x·p``。孔径的
        频谱（DFT）第 k 条线对应空间频率 ``ν_k = k/W`` (cycle/m)，透镜把空间
        频率线性映射到焦面坐标 ``x = f·λ·ν``，于是相邻谱线在焦面的间隔
        （焦面里一个「像素」覆盖的物理尺寸）为

            Δx = f·λ·(1/W) = λf/(N_x·p)。

        例：λ=1064 nm, f=0.125 m, N_x=1920, p=8 µm ⇒ Δx ≈ 8.66 µm ≈ 3.94 相机像元。
        y 方向同理（用 ``N_y`` 代替）。
        """
        return self.wavelength_m * self.focal_m / (self.slm_pixels[1] * self.pixel_pitch_m)

    @property
    def slm_fov_m(self) -> float:
        """最大无混叠视场 ``λf/p`` (m)。

        推导：间距 p 的采样能无混叠表示的最高空间频率是 Nyquist 极限
        ``ν_max = 1/(2p)`` (cycle/m)，对应焦面最大坐标 ``x_max = f·λ/(2p)``，
        全视场 ``FOV = 2·x_max = λf/p``。
        例：λ=1064 nm, f=0.125 m, p=8 µm ⇒ FOV = 16.625 mm。
        """
        return self.wavelength_m * self.focal_m / self.pixel_pitch_m

    @property
    def practical_max_px(self) -> float:
        """实用上限（相机像元）= min(SLM 极限 FOV, 传感器短边) 换算到相机像元。

        两个限制取小：SLM 侧是衍射/混叠极限（``slm_fov_m``），传感器侧是相机
        短边能框住的范围（``min(camera_shape)`` 个像元）。
        """
        slm_limit_px = self.slm_fov_m / self.camera_pixel_m
        sensor_short_px = float(min(self.camera_shape))
        return min(slm_limit_px, sensor_short_px)

    def candidate_sizes_px(self, fractions: Sequence[float] | None = None) -> tuple[float, ...]:
        """候选目标边长（相机像元）= 分数 × ``practical_max_px``。

        默认分数 0.05–0.80 步长 0.05（含两端，16 个）。输出**去重且单调递增**
        （自定义分数乱序时按值排序后去重）。

        Raises:
            ValueError: 去重后少于 2 个不同尺寸（无法做任何曲线分析）。
        """
        fracs = tuple(default_fractions()) if fractions is None else tuple(float(v) for v in fractions)
        raw = [round(f * self.practical_max_px, 2) for f in fracs]
        out: list[float] = []
        for value in sorted(raw):
            if not math.isfinite(value) or value <= 0.0:
                continue
            if out and value == out[-1]:
                continue
            if out and value < out[-1]:
                raise ValueError(f"候选尺寸必须单调递增，但出现 {out[-1]} → {value}")
            out.append(value)
        if len(out) < 2:
            raise ValueError(f"候选尺寸去重后不足 2 个: {out}")
        return tuple(out)

    def signature(self) -> str:
        """对 (λ, f, N, p, 相机像元, 相机尺寸) 的稳定哈希（16 位 hex）。

        任一参数改变 ⇒ 签名改变 ⇒ 旧标定表失效。
        """
        payload = json.dumps(
            {
                "wavelength_m": repr(float(self.wavelength_m)),
                "focal_m": repr(float(self.focal_m)),
                "slm_pixels": [int(v) for v in self.slm_pixels],
                "pixel_pitch_m": repr(float(self.pixel_pitch_m)),
                "camera_pixel_m": repr(float(self.camera_pixel_m)),
                "camera_shape": [int(v) for v in self.camera_shape],
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


#: 本仓库 2f 实验系统的默认光路参数（λ=1064 nm, f=125 mm, 1200×1920 面板,
#: 8 µm 像素间距, 2.2 µm 相机像元, 1944×2592 传感器）。``SYSTEM_SIGNATURE``
#: 是它对应的签名 —— ``TargetSizeCalibration.load(path)`` 不带 ``limits``
#: 参数时用它校验存档是否过期。
DEFAULT_SYSTEM = OpticalLimits(
    wavelength_m=LASER_WAVELENGTH_M,
    focal_m=0.125,
    slm_pixels=(1200, 1920),
    pixel_pitch_m=8e-6,
    camera_pixel_m=2.2e-6,
    camera_shape=(1944, 2592),
)
SYSTEM_SIGNATURE: str = DEFAULT_SYSTEM.signature()


# ---------------------------------------------------------------------------
# 单点指标
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SizeMetrics:
    """一个候选尺寸处的指标（模型侧或实测侧）。

    **分子/分母口径**（2026-10-23 防退化修订；旧口径 E(盒)/E(全图) 的
    efficiency 随尺寸单调上升是恒等式不是效率，已弃用）：

    * ``efficiency`` = E(目标盒) / E(参考窗)。**参考窗在所有尺寸上是同一个
      固定区域**（零级中心的 241×241 相机像元窗；模型侧 = 整个采样网格），
      因此该比值不再随「分母跟着盒子变大」而自我抵消。盒比参考窗大时可 >1
      （盒装下了比参考窗更多的能量），量纲是能量比不是分数。
    * ``efficiency_baseline`` = 同式但用**未优化基线图**（零相位）；
      ``efficiency_vs_baseline`` = 两者之差 = 优化器相对「几何恒等式」的净增益，
      是参与打分的那个量。
    * ``uniformity`` = CV(core) = sqrt(mean((v/μ−1)²))，只在 ``valid=True`` 时
      有意义；``valid=False`` 的空盒点必须从窗口选择中剔除。
    * ``mean_core`` = 目标区（core 掩膜）内的平均强度（图像单位）—— 有效性
      判据的量；``fill`` = 目标盒内**被点亮**的像素占比（像素值 ≥
      ``k×噪声地板``，k 默认 3，与有效性同一阈值）—— 直接回答「这个尺寸的
      盒填上了多少」。
    * ``valid`` = ``mean_core ≥ k × 噪声地板``（:func:`is_valid`，k 默认 3）。
      判据来源：2026-10-23 实测粗扫 97.2–1555.2 px 中，仅 97.2 px 通过
      （mean 1.557 ≥ 3×0.377），194.4 px 起 mean 掉到 0.394→0.018 全部失效，
      CV 被空盒统计推高到 10+ —— 用 3× 地板做门，空盒的 CV 不再参与排序。
    * ``zero_order_ratio`` = E(固定零级盘) / E(参考窗)。零级盘**固定**（零级
      中心、半径 15 px），不再随目标尺寸变化；``zero_abs`` 是同一盘内能量的
      **绝对值**（图像单位），两者一起避免「比值被随尺寸变化的分母稀释」。
    * ``fidelity`` = 模型↔实测强度相关（[-1,1]）；dry-run 无设备时是
      ``fidelity_proxy``（模型预测 vs 存档实测帧），有实测时为真实测相关。

    Attributes:
        size_px: 目标边长（相机像元）。
        efficiency: E(目标盒)/E(参考窗)，优化图；越大越好，≥0（可 >1）。
        uniformity: CV(core)，越小越好（≥0）；``valid=False`` 时不可信。
        zero_order_ratio: E(固定零级盘)/E(参考窗)，[0,1]。
        fidelity: 模型↔实测强度相关（[-1,1]）；无实测为 ``None``。
        target_type: 目标图案复杂度类型（查找表键，如 ``"square"``）。
        efficiency_baseline: E(目标盒)/E(参考窗)，未优化基线图。
        efficiency_vs_baseline: efficiency − efficiency_baseline，打分用。
        zero_abs: 固定零级盘内能量绝对值（图像单位）。
        mean_core: 目标区（core）平均强度（图像单位）。
        fill: 目标盒内被点亮像素占比（≥ k×噪声地板），[0,1]。
        valid: 有效性（mean_core ≥ k×噪声地板，k 默认 3）。
    """

    size_px: float
    efficiency: float
    uniformity: float
    zero_order_ratio: float
    fidelity: float | None = None
    target_type: str = "square"
    efficiency_baseline: float = 0.0
    efficiency_vs_baseline: float = 0.0
    zero_abs: float = 0.0
    mean_core: float = 0.0
    fill: float = 0.0
    valid: bool = True

    def __post_init__(self) -> None:
        for name in (
            "size_px",
            "efficiency",
            "uniformity",
            "zero_order_ratio",
            "efficiency_baseline",
            "zero_abs",
            "mean_core",
        ):
            value = float(getattr(self, name))  # type: ignore[arg-type]
            if not math.isfinite(value):
                raise ValueError(f"SizeMetrics.{name} 不是有限值: {value!r}")
            if value < 0.0:
                raise ValueError(f"SizeMetrics.{name} 必须 ≥ 0: {value!r}")
        if not (0.0 <= self.zero_order_ratio <= 1.0 + 1e-6):
            raise ValueError(f"SizeMetrics.zero_order_ratio 应在 [0,1]: {self.zero_order_ratio!r}")
        if not (0.0 <= float(self.fill) <= 1.0 + 1e-6):
            raise ValueError(f"SizeMetrics.fill 应在 [0,1]: {self.fill!r}")
        if self.fidelity is not None and not (-1.0 <= float(self.fidelity) <= 1.0):
            raise ValueError(f"SizeMetrics.fidelity 应在 [-1,1]: {self.fidelity!r}")
        if not str(self.target_type).strip():
            raise ValueError("SizeMetrics.target_type 不能为空")

    def to_dict(self) -> dict[str, Any]:
        return {
            "size_px": float(self.size_px),
            "efficiency": float(self.efficiency),
            "uniformity": float(self.uniformity),
            "zero_order_ratio": float(self.zero_order_ratio),
            "fidelity": None if self.fidelity is None else float(self.fidelity),
            "target_type": str(self.target_type),
            "efficiency_baseline": float(self.efficiency_baseline),
            "efficiency_vs_baseline": float(self.efficiency_vs_baseline),
            "zero_abs": float(self.zero_abs),
            "mean_core": float(self.mean_core),
            "fill": float(self.fill),
            "valid": bool(self.valid),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SizeMetrics":
        fidelity = data.get("fidelity")
        valid = data.get("valid")
        return cls(
            size_px=float(data["size_px"]),
            efficiency=float(data["efficiency"]),
            uniformity=float(data["uniformity"]),
            zero_order_ratio=float(data["zero_order_ratio"]),
            fidelity=None if fidelity is None else float(fidelity),
            target_type=str(data.get("target_type") or "square"),
            efficiency_baseline=float(data.get("efficiency_baseline") or 0.0),
            efficiency_vs_baseline=float(data.get("efficiency_vs_baseline") or 0.0),
            zero_abs=float(data.get("zero_abs") or 0.0),
            mean_core=float(data.get("mean_core") or 0.0),
            fill=float(data.get("fill") or 0.0),
            # 旧存档没有 valid 字段时按有效处理（向后兼容）。
            valid=True if valid is None else bool(valid),
        )


def is_valid(mean_core: float, signal_floor: float, *, k: float = 3.0) -> bool:
    """有效性判据：``mean_core ≥ k × 噪声地板``（k 默认 3）。

    空盒（目标区几乎没有光）的 CV 由噪声与零值统计主导，数值再大也没有物理
    意义 —— 必须先过这道门。噪声地板（counts/图像单位）由调用方从当次数据
    估计（模型侧 = 冻结前向的 ``background`` 参数；实测侧 = 去本底帧的背景
    标准差），**不同 run 必须用当次的地板**。
    """
    return float(mean_core) >= k * float(signal_floor)


# ---------------------------------------------------------------------------
# 合成分数（峰值窗口与单峰拟合共用）
# ---------------------------------------------------------------------------
def _minmax(values: np.ndarray) -> np.ndarray:
    """min-max 归一到 [0,1]；常数列（max==min）取中性 0.5，不做除法。"""
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo <= 0.0:
        return np.full_like(values, 0.5, dtype=np.float64)
    return (values - lo) / (hi - lo)


def composite_scores(
    points: Sequence[SizeMetrics],
    weight_uniformity: float,
    weight_efficiency: float,
) -> np.ndarray:
    """把不同量纲的 uniformity / efficiency 各自 min-max 归一后合成分数。

    **分数用的是 ``efficiency_vs_baseline``**（相对未优化基线的增量，抗尺寸
    单调性；不是原始 ``efficiency`` —— 后者分母为固定参考区时仍随盒面积单调
    增长，直接参与排序会把「盒子大」误判成「做得好」）：越大越好 ⇒ 分数 =
    归一化值。uniformity 越小越好 ⇒ 分数 = 1 − 归一化值。权重在内部归一化（只
    相对比例有意义）；某一权重为 0 表示该指标不参与（供退化指标剔除时复用）。
    扫描脚本画「尺寸 → 分数」曲线时直接复用本函数，避免第二份口径。
    """
    if weight_uniformity < 0 or weight_efficiency < 0:
        raise ValueError("权重必须非负")
    if weight_uniformity + weight_efficiency <= 0:
        raise ValueError("两个权重不能同时为 0")
    eff = _minmax(np.array([p.efficiency_vs_baseline for p in points], dtype=np.float64))
    uni = _minmax(np.array([p.uniformity for p in points], dtype=np.float64))
    w_u = float(weight_uniformity) / (weight_uniformity + weight_efficiency)
    w_e = float(weight_efficiency) / (weight_uniformity + weight_efficiency)
    return w_u * (1.0 - uni) + w_e * eff


def _require_points(points: Sequence[SizeMetrics], n_min: int = 3) -> list[SizeMetrics]:
    pts = list(points)
    if len(pts) < n_min:
        raise ValueError(
            f"需要至少 {n_min} 个样本点才能做曲线分析，只给了 {len(pts)} 个 —— 显式报错而不是猜"
        )
    for p in pts:
        if not isinstance(p, SizeMetrics):
            raise TypeError(f"期望 SizeMetrics 序列，得到 {type(p)!r}")
    return pts


def _rank(values: np.ndarray) -> np.ndarray:
    """秩（平局取平均秩），供 Spearman 相关用。"""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and values[order[j + 1]] == values[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman 秩相关（[-1,1]）；退化输入（<3 点 / 常数列）返回 0。"""
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    if xv.size < 3 or xv.size != yv.size:
        return 0.0
    rx = _rank(xv) - np.mean(_rank(xv))
    ry = _rank(yv) - np.mean(_rank(yv))
    denom = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
    if denom <= 1e-18:
        return 0.0
    return float((rx * ry).sum() / denom)


def degenerate_metrics(
    points: Sequence[SizeMetrics],
    *,
    cv_range_eps: float = 0.02,
    spearman_eps: float = 0.98,
    zero_range_eps: float = 1e-3,
) -> list[tuple[str, str]]:
    """指标饱和（无区分度）检查。只检查**有效点**（``valid=True``）。

    判据（2026-10-23 实测修订；粗扫 16 档中 efficiency 0.214→0.700 随尺寸
    单调升、zero_order 全程 0.1215 不动，旧口径下二者都会把窗口选到端点）：

    * **efficiency**：Spearman(size, efficiency) > ``spearman_eps``（默认
      0.98）—— 随尺寸单调饱和，量的是「框覆盖了多少光」不是效率；
    * **uniformity**：极差 < ``cv_range_eps``（默认 0.02）—— 区间内几乎不动；
    * **zero_order_ratio**：极差 < ``zero_range_eps``（默认 1e-3）—— 固定零级
      盘 + 固定分母下若仍无变化，说明整形没有改变零级残留，量不出东西。

    Returns:
        ``[(指标名, 说明), ...]``；空列表 = 全部指标有区分度。退化的指标
        **不得计入复合分数**（:func:`select_best_window` 自动剔除），说明里带
        「该指标无区分度，请改用实测一致性」的处置提示。
    """
    valid = [p for p in points if p.valid]
    findings: list[tuple[str, str]] = []
    if len(valid) < 3:
        return findings
    sizes = [p.size_px for p in valid]
    eff = np.asarray([p.efficiency for p in valid], dtype=np.float64)
    cv = np.asarray([p.uniformity for p in valid], dtype=np.float64)
    zero = np.asarray([p.zero_order_ratio for p in valid], dtype=np.float64)
    rho_eff = _spearman(sizes, eff)
    if rho_eff > spearman_eps:
        findings.append(
            (
                "efficiency",
                f"Spearman(size, efficiency) = {rho_eff:.3f} > {spearman_eps}: 随尺寸单调饱和"
                f"（量的是框覆盖了多少光，不是效率），该指标无区分度，请改用实测一致性",
            )
        )
    cv_spread = float(cv.max() - cv.min())
    if cv_spread < cv_range_eps:
        findings.append(
            (
                "uniformity",
                f"极差 {cv_spread:.4f} < {cv_range_eps}: 区间内几乎无变化，该指标无区分度，"
                f"请改用实测一致性",
            )
        )
    zero_spread = float(zero.max() - zero.min())
    if zero_spread < zero_range_eps:
        findings.append(
            (
                "zero_order_ratio",
                f"极差 {zero_spread:.6f} < {zero_range_eps}: 区间内几乎无变化，该指标无区分度，"
                f"请改用实测一致性",
            )
        )
    return findings


# ---------------------------------------------------------------------------
# 峰值窗口
# ---------------------------------------------------------------------------
def select_best_window(
    points: Sequence[SizeMetrics],
    *,
    weight_uniformity: float = 0.5,
    weight_efficiency: float = 0.5,
) -> tuple[float | None, float | None, float | None, list[tuple[str, str]], str]:
    """在「尺寸 → 合成分数」曲线上找峰值，并给出**平台区间** (lo, hi, best)。

    先把 uniformity 与 efficiency_vs_baseline **各自 min-max 归一**（不同量纲，
    不可直接加），合成加权分数；峰值 = 分数最大点；平台 = 从峰值向两侧扩展、
    分数与峰值差**小于 1%** 的连续邻域（差 ≥1% 视为离开了平台）。

    **有效点优先 + 退化拒斥**（2026-10-23 修订，粗扫实测驱动）：

    * **先按 ``valid`` 过滤**（mean_core ≥ k×噪声地板，k 默认 3），只在有效
      点里选；2026-10-23 粗扫中 194.4 px 起全部失效，旧行为（不过滤）会把
      空盒点 1555.2 px 选成「最优」。
    * 有效点为空 ⇒ 返回 ``(None, None, None)`` + 说明「该区间内没有可填满的
      目标，需增大照明/曝光或改小目标」；
    * :func:`degenerate_metrics` 标记的**退化指标不计入复合分数**（权重清零），
      随结果显式返回 ``degenerate=[(名, 说明), ...]``；两项参与排序的指标
      全部退化 ⇒ 同样返回 ``(None, None, None)`` + 说明（拒绝构造假分数）。

    Args:
        points: ``SizeMetrics`` 序列（按尺寸顺序）。
        weight_uniformity: uniformity 项的权重（分数 = 1 − 归一化 uniformity）。
        weight_efficiency: efficiency 项的权重（分数 = 归一化
            efficiency_vs_baseline）。

    Returns:
        ``(lo, hi, best, degenerate, message)``：平台区间两端、峰值尺寸
        （相机像元，均可为 ``None``）、退化指标列表、说明（正常时为空串）。

    Raises:
        ValueError: 样本 <3（总点数，未过滤前）。
    """
    pts = _require_points(points)
    valid = [p for p in pts if p.valid]
    degenerate = degenerate_metrics(valid)
    deg_names = {name for name, _ in degenerate}
    if not valid:
        return (
            None,
            None,
            None,
            degenerate,
            "该区间内没有可填满的目标（所有点 mean_core < k×噪声地板），"
            "需增大照明/曝光或改小目标",
        )
    w_u = 0.0 if "uniformity" in deg_names else float(weight_uniformity)
    w_e = 0.0 if "efficiency" in deg_names else float(weight_efficiency)
    if w_u <= 0.0 and w_e <= 0.0:
        names = ", ".join(name for name, _ in degenerate)
        return (
            None,
            None,
            None,
            degenerate,
            f"参与排序的指标全部退化 ({names})，无法构造合成分数 —— 拒绝给出窗口；"
            f"请改用实测一致性 (fidelity) 或换光路重标定",
        )
    scores = composite_scores(valid, w_u, w_e)
    peak_idx = int(np.argmax(scores))
    if len(valid) < 3:
        # 1–2 个有效点: 平台退化为单点（最佳有效尺寸），显式说明而不是猜区间
        size = float(valid[peak_idx].size_px)
        return (
            size,
            size,
            size,
            degenerate,
            f"只有 {len(valid)} 个有效点 (<3)，窗口退化为单点 {size:.1f} px",
        )
    peak_score = float(scores[peak_idx])
    plateau_threshold = peak_score * 0.99  # 与峰值差 <1% 视为同一平台
    lo = peak_idx
    while lo > 0 and float(scores[lo - 1]) >= plateau_threshold:
        lo -= 1
    hi = peak_idx
    while hi < len(scores) - 1 and float(scores[hi + 1]) >= plateau_threshold:
        hi += 1
    return (float(valid[lo].size_px), float(valid[hi].size_px), float(valid[peak_idx].size_px), degenerate, "")


# ---------------------------------------------------------------------------
# 单峰拟合
# ---------------------------------------------------------------------------
def fit_optimum(points: Sequence[SizeMetrics]) -> tuple[float | None, float | None, float]:
    """在 **log(size)** 上对合成分数做二次拟合（单峰判据）。

    只用**有效点**（``valid=True``，mean_core ≥ k×噪声地板，k 默认 3），
    合成分数口径与 :func:`select_best_window` 一致（``composite_scores``，即
    efficiency_vs_baseline + uniformity）。分数曲线 ``s(u)``（``u = ln(size)``）
    拟合为 ``a·u² + b·u + c``：

    * 顶点（最优尺寸）``u* = −b/(2a)`` ⇒ ``size* = exp(u*)``；
    * 半宽 ``sigma_px``：分数从顶点下降到**顶点值的一半**处的正向尺寸增量
      ``size*·(exp(Δu) − 1)``，其中 ``Δu = sqrt(s*(−2a))^-½``；
    * ``r_squared``：二次拟合的决定系数。

    Returns:
        ``(size_px, sigma_px, r_squared)``。若**不呈单峰**（开口向上 ``a ≥ 0``、
        顶点值 ≤0、或 ``r_squared < 0.5``），返回 ``(None, None, r_squared)``
        —— 顶点为 ``None`` 是显式结论「不呈单峰」，不是静默缺省。

    Raises:
        ValueError: 样本 <3。
    """
    pts = _require_points(points)
    valid = [p for p in pts if p.valid]
    if len(valid) < 3:
        raise ValueError(
            f"剔除无效点后只剩 {len(valid)} 个有效样本 (<3)，无法做单峰拟合 —— "
            f"显式报错而不是猜"
        )
    scores = composite_scores(valid, 0.5, 0.5)
    sizes = np.array([p.size_px for p in valid], dtype=np.float64)
    u = np.log(sizes)
    coef = np.polyfit(u, scores, 2)
    a, b, c = (float(v) for v in coef)
    predicted = np.polyval(coef, u)
    ss_res = float(np.sum((scores - predicted) ** 2))
    ss_tot = float(np.sum((scores - scores.mean()) ** 2))
    # 常数分数 (ss_tot≈0) 或拟合劣于均值时 r² 可为负/噪声负值 —— 截断到 0，
    # 判据仍只看 r² < 0.5。
    r_squared = 0.0 if ss_tot <= 0.0 else max(0.0, 1.0 - ss_res / ss_tot)
    # 不呈单峰：开口向上 / 顶点不在分数域内 / 拟合太差 —— 都显式返回 None 顶点。
    if a >= 0.0 or r_squared < 0.5:
        return (None, None, r_squared)
    u_star = -b / (2.0 * a)
    score_star = a * u_star * u_star + b * u_star + c
    if score_star <= 0.0:
        return (None, None, r_squared)
    size_star = math.exp(u_star)
    du = math.sqrt(score_star / (-2.0 * a))
    sigma_px = size_star * (math.exp(du) - 1.0)
    return (float(size_star), float(sigma_px), float(r_squared))


# ---------------------------------------------------------------------------
# 查找表（标定档案）
# ---------------------------------------------------------------------------
class TargetSizeCalibrationError(RuntimeError):
    """目标尺寸标定表无效：签名不匹配（系统已变更，必须重标定）或字段损坏。"""


@dataclass
class TargetSizeCalibration:
    """一次目标尺寸标定的完整档案（查找表）。

    换透镜 / 波长 / SLM 后 :meth:`load` 会因 ``signature`` 不匹配而**显式拒绝**
    旧表 —— 这是刻意的：旧表里的最优区间是旧光路的属性，沿用会静默给出错误
    尺寸。

    Attributes:
        signature: 标定时的系统签名（``OpticalLimits.signature()``）。
        limits: 标定时的光学硬参数（完整保存，便于复核签名）。
        points: 全部样本点（模型侧 / 实测侧）。
        window: 主（square）目标的最优窗口 ``(lo, hi)``（相机像元）。
        best_size: 最优尺寸回退值（窗口中心或单峰顶点）。
        method: 标定方法描述（如 "model-only sweep" / "model+hardware sweep"）。
        notes: 备注。
        recorded_utc: 记录时刻（UTC, ISO 8601）。
        windows: 按 ``target_type`` 分键的窗口（square 与 ``window`` 同步）。
    """

    signature: str
    limits: OpticalLimits
    points: list[SizeMetrics] = field(default_factory=list)
    window: tuple[float, float] | None = None
    best_size: float | None = None
    method: str = ""
    notes: str = ""
    recorded_utc: str = ""
    windows: dict[str, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.signature).strip():
            raise ValueError("TargetSizeCalibration.signature 不能为空")
        if not isinstance(self.limits, OpticalLimits):
            raise TypeError("TargetSizeCalibration.limits 必须是 OpticalLimits")
        if self.window is not None:
            lo, hi = self.window
            if float(hi) < float(lo):
                raise ValueError(f"window 区间反向: {self.window!r}")
        if self.best_size is not None and float(self.best_size) <= 0.0:
            raise ValueError(f"best_size 必须为正: {self.best_size!r}")
        if not self.recorded_utc:
            self.recorded_utc = _now_utc()

    # -- 序列化 -------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        limits = self.limits
        return {
            "signature": self.signature,
            "limits": {
                "wavelength_m": float(limits.wavelength_m),
                "focal_m": float(limits.focal_m),
                "slm_pixels": [int(v) for v in limits.slm_pixels],
                "pixel_pitch_m": float(limits.pixel_pitch_m),
                "camera_pixel_m": float(limits.camera_pixel_m),
                "camera_shape": [int(v) for v in limits.camera_shape],
            },
            "points": [p.to_dict() for p in self.points],
            "window": None if self.window is None else [float(self.window[0]), float(self.window[1])],
            "windows": {k: [float(v[0]), float(v[1])] for k, v in self.windows.items()},
            "best_size": None if self.best_size is None else float(self.best_size),
            "method": str(self.method),
            "notes": str(self.notes),
            "recorded_utc": str(self.recorded_utc),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TargetSizeCalibration":
        ld = data["limits"]
        limits = OpticalLimits(
            wavelength_m=float(ld["wavelength_m"]),
            focal_m=float(ld["focal_m"]),
            slm_pixels=(int(ld["slm_pixels"][0]), int(ld["slm_pixels"][1])),
            pixel_pitch_m=float(ld["pixel_pitch_m"]),
            camera_pixel_m=float(ld["camera_pixel_m"]),
            camera_shape=(int(ld["camera_shape"][0]), int(ld["camera_shape"][1])),
        )
        window = data.get("window")
        windows = {
            str(k): (float(v[0]), float(v[1])) for k, v in dict(data.get("windows") or {}).items()
        }
        best = data.get("best_size")
        return cls(
            signature=str(data["signature"]),
            limits=limits,
            points=[SizeMetrics.from_dict(p) for p in data.get("points", [])],
            window=None if window is None else (float(window[0]), float(window[1])),
            best_size=None if best is None else float(best),
            method=str(data.get("method") or ""),
            notes=str(data.get("notes") or ""),
            recorded_utc=str(data.get("recorded_utc") or ""),
            windows=windows,
        )

    # -- 增删与落盘 ----------------------------------------------------------
    def add(self, point: SizeMetrics) -> None:
        """追加一个样本点（校验由 ``SizeMetrics`` 自身完成）。"""
        if not isinstance(point, SizeMetrics):
            raise TypeError(f"add 期望 SizeMetrics，得到 {type(point)!r}")
        self.points.append(point)

    def save(self, path: str | Path = "target_size_calibration.json") -> Path:
        """落盘（JSON，UTF-8）。缺省目录自动创建。"""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path, *, limits: OpticalLimits | None = None) -> "TargetSizeCalibration":
        """从落盘文件读标定表；**系统签名不匹配时显式拒绝**（必须重标定）。

        Args:
            path: ``target_size_calibration.json``（或 ``save`` 写出的文件）。
            limits: 当前光路的 ``OpticalLimits``；缺省用本模块的
                :data:`DEFAULT_SYSTEM`（即 ``SYSTEM_SIGNATURE``）。

        Raises:
            TargetSizeCalibrationError: 文件缺失 / 损坏 / 签名与当前系统不一致
                （提示必须重新标定，而不是沿用旧表）。
        """
        target = Path(path)
        if not target.is_file():
            raise TargetSizeCalibrationError(f"目标尺寸标定表不存在: {target}")
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TargetSizeCalibrationError(f"目标尺寸标定表不可读: {target} ({exc})") from exc
        if not isinstance(payload, dict):
            raise TargetSizeCalibrationError(f"目标尺寸标定表格式不对: {target} 的顶层不是对象")
        stored = cls.from_dict(payload)
        # 完整性：文件内记录签名必须与文件内 limits 重算签名一致（防半截写坏）。
        if stored.limits.signature() != stored.signature:
            raise TargetSizeCalibrationError(
                f"标定表内部签名不一致（{stored.signature!r} vs {stored.limits.signature()!r}）: "
                f"{target} 可能损坏，必须重新标定"
            )
        # 系统变更校验：当前光路签名 ≠ 存档签名 ⇒ 旧表失效。
        current = limits if limits is not None else DEFAULT_SYSTEM
        if stored.signature != current.signature():
            raise TargetSizeCalibrationError(
                f"系统签名不匹配（存档 {stored.signature!r} ≠ 当前 {current.signature()!r}）: "
                f"标定表来自不同的 (λ, f, N, p, 相机) 组合，已拒绝沿用 —— 必须重新标定"
            )
        return stored

    # -- 推荐 ----------------------------------------------------------------
    def recommend(self, target_type: str = "square", *, constraint: str = "largest_valid") -> float:
        """推荐目标边长（相机像元）。

        ``constraint`` 在**有效点**（``valid=True``，mean_core ≥ k×噪声地板）里
        做取舍：

        * ``"largest_valid"``（默认）：有效点里**最大**的尺寸 —— 照明/曝光能填满
          的最大边长。2026-10-23 实测细扫 40–240 px 的有效区间为 40–100 px：
          40 px 均匀度最好但填得少，100 px 填得最大，默认取大者；
        * ``"best_balance"``：有效点里按 CV × 效率折中，
          分数 = (1 − minmax(CV)) × minmax(efficiency)，取 argmax —— 「均匀且
          填得上」的折中点。

        该 ``target_type`` 没有有效点时，回退优先级：窗口中心 → ``best_size``
        → **显式报错**（绝不静默给默认值）。

        Args:
            target_type: 查找表键（如 ``"square"``）。
            constraint: ``"largest_valid"``（默认）或 ``"best_balance"``。

        Raises:
            ValueError: ``constraint`` 不是两个合法值之一。
            TargetSizeCalibrationError: 没有有效点、也没有窗口 / ``best_size``
                回退（该区间内没有可填满的目标）。
        """
        if constraint not in ("largest_valid", "best_balance"):
            raise ValueError(
                f"未知 constraint {constraint!r}: 只支持 'largest_valid' 或 'best_balance'"
            )
        valid = [p for p in self.points if p.valid and p.target_type == target_type]
        if not valid:
            valid = [p for p in self.points if p.valid]
        if valid:
            if constraint == "largest_valid":
                return max(float(p.size_px) for p in valid)
            cv = np.array([p.uniformity for p in valid], dtype=np.float64)
            eff = np.array([p.efficiency for p in valid], dtype=np.float64)
            scores = (1.0 - _minmax(cv)) * _minmax(eff)
            return float(valid[int(np.argmax(scores))].size_px)
        window = self.windows.get(target_type)
        if window is None and target_type == "square":
            window = self.window
        if window is not None:
            return 0.5 * (float(window[0]) + float(window[1]))
        if self.best_size is not None:
            return float(self.best_size)
        raise TargetSizeCalibrationError(
            f"target_type={target_type!r} 没有有效点（该区间内没有可填满的目标，"
            f"需增大照明/曝光或改小目标），也没有窗口 / best_size 回退: "
            f"必须先完成目标尺寸标定（不静默给默认值）"
        )
