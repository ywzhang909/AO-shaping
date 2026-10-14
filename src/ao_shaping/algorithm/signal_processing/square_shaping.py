"""SLM 远场方形整形算法 (NumPy / torch 可选, 无驱动依赖)。

本模块实现三种开环远场整形方法, 在 2f-傅里 Fourier 模型下运行:

* **GS** (:func:`gs_shape`) — 经典 Gerchberg-Saxton, 幅度约束 = 远场 sqrt(target)。
* **UniformSquareShaper** — 带权重 / HIO 变体的 GS (class-based API)。
* **differentiable** (:func:`differentiable_shape`) — torch 可微分梯度下降。
* **SPGD** (:func:`spgd_shape`) — 无感知黑盒 SPGD, 作用于粗网格 freeform 相位。
* **analytic_amplitude_target** (:func:`analytic_amplitude_target`) — 纯幅度
  单板基线 (不可达, 但给出纯相位器件的上界参考)。

正向模型与 :mod:`ao_shaping.drivers.sim.iterative_zernike_shaping` 一致:

    field = gauss_pupil * exp(1j*phase)
    far_field = fftshift(fft2(ifftshift(pad(field))))

所有相位都是**原始弧度, 未包裹**; 唯一的 mod-2π 在 SLM 驱动
``create_phase_from_array``。

torch 仅 ``differentiable_shape`` 需用, 导入是惰性的。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np



__all__ = [
    "SquareShapingConfig",
    "ShapingResult",
    "make_square_target",
    "GS_METHODS",
    "GS_SELECT",
    "gs_shape",
    "UniformSquareShaper",
    "differentiable_shape",
    "spgd_shape",
    "analytic_amplitude_target",
    "power_in_bucket",
    "efficiency",
    "uniformity_cv",
    "strehl",
    "zero_order_fraction",
    "compute_bench_metrics",
    "composite_from_pib_cv",
    "composite_score",
]


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SquareShapingConfig:
    """2f 傅里 Fourier SLM 整形台架的算法参数。

    该配置**不依赖任何 sim / 驱动模块** —— Gaussian 瞳面和正向模型都在算法层内联
    计算。它镜像 :class:`ao_shaping.drivers.sim.slm_shaping_bench.ShapingBenchConfig`
    的算法相关字段: ``cn2``/``l_max``/``l_min`` 等大气参数在本模块的前向模型中是
    **死配置** (永不路由), 因此不携带。

    Attributes:
        n_grid: SLM / 仿真网格边长。
        aperture_size: SLM 口径宽度 (m)。
        wavelength: 波长 (m)。
        focal_length: 透镜焦距 (m)。
        target_side_px: 目标方形边长, 计在远场 (相机) 像素。
        target_kind: "square" | "circle" | "gaussian" | "annulus"。
        zero_order_margin_px: 0 级保护半径 (远场 px)。
        far_field_padding: FFT 前的零填充倍数。
        seed: RNG 种子。
    """

    n_grid: int = 256
    aperture_size: float = 12e-3
    wavelength: float = 532e-9
    focal_length: float = 0.125
    target_side_px: int = 60
    target_kind: str = "square"
    zero_order_margin_px: float = 10.0
    far_field_padding: int = 8
    seed: int = 0
    w_loggrad: float = 0.0
    """Weight of the intensity-space structural term (default OFF).

    ``0.0`` reproduces the pre-existing behaviour **bit for bit**: the term is
    skipped entirely rather than added with a zero coefficient, so turning it on
    is a deliberate act and turning it off restores the exact previous numbers.

    When > 0 the shaping score becomes
    ``composite_score(metrics) - w_loggrad * log_gradient_difference_metric(...)``.
    The term is the repo's already-canonical intensity-space structural metric
    (:func:`ao_shaping.utils.image.target.log_gradient_difference_metric`), chosen
    over inventing a new one: a speckled frame and a smooth one can enclose the
    same power but have very different log gradients, which is exactly the
    discrimination ``PIB`` / ``CV`` cannot make.

    This flag does **not** affect :func:`gs_shape`: GS is an alternating
    projection, not a descent on a scalar loss, so it has nowhere to consume a
    penalty term. It applies to the scalar-scored methods (:func:`spgd_shape`).
    """

    def __post_init__(self) -> None:
        if self.far_field_padding < 1:
            raise ValueError(
                f"far_field_padding must be >= 1, got {self.far_field_padding}"
            )
        if self.target_kind not in ("square", "circle", "gaussian", "annulus"):
            raise ValueError(f"unknown target_kind {self.target_kind!r}")
        if not math.isfinite(self.w_loggrad) or self.w_loggrad < 0.0:
            raise ValueError(
                f"w_loggrad must be >= 0 and finite, got {self.w_loggrad!r}"
            )

    @property
    def far_field_size(self) -> int:
        """零填充远场 (相机) 网格的边长。"""
        return self.n_grid * self.far_field_padding

    @property
    def far_field_pixel_size(self) -> float:
        """远场像元间距 (m)。

        ``lambda * f / (far_field_padding * aperture_size)`` —— 与 n_grid 无关。
        """
        return (
            self.wavelength
            * self.focal_length
            / (self.aperture_size * self.far_field_padding)
        )

    def make_gaussian_pupil(self) -> np.ndarray:
        """SLM 瞳面 Gaussian 入射幅度 (n_grid x n_grid, float64)。

        与 :func:`ao_shaping.drivers.sim.beam_backend.gaussian_pupil` 数学等价
        (w0 = aperture_size / 3.5, 口径截断), 但**不导入** beam_backend ——
        因此算法层对 ``AO_OOPAO_BACKEND`` 环境变量等外部状态一无所知。

        >>> cfg = SquareShapingConfig(n_grid=128)
        >>> cfg.make_gaussian_pupil().shape
        (128, 128)
        """
        n = self.n_grid
        pixel_size = self.aperture_size / n
        w0 = self.aperture_size / 3.5
        coords = np.arange(-n / 2, n / 2) * pixel_size
        y, x = np.meshgrid(coords, coords, indexing="ij")
        r = np.sqrt(x**2 + y**2)
        cutoff = self.aperture_size / 2
        return np.exp(-(r**2) / w0**2) * (r <= cutoff)


# ---------------------------------------------------------------------------
# 结果容器
# ---------------------------------------------------------------------------


@dataclass
class ShapingResult:
    """一次整形运行的产出。"""

    method: str
    phase: np.ndarray
    intensity: np.ndarray
    metrics: dict[str, float]
    n_iters: int
    history: list[dict[str, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 正向模型 (2f 傅里叶)
# ---------------------------------------------------------------------------

# 远场网格上的归一化坐标范围
_L2 = 2.0  # pupil radius in normalised coords


def _pad_centred(arr: np.ndarray, far_field_size: int) -> np.ndarray:
    """把方形瞳孔网格数组居中补零到 ``far_field_size``。

    瞳孔占据中央的 ``n_grid x n_grid`` 块; 补零区域就是 SLM 口径之外的暗区域。
    """
    n = arr.shape[0]
    m = far_field_size
    if m <= n:
        return arr
    out = np.zeros((m, m), dtype=arr.dtype)
    start = (m - n) // 2
    out[start : start + n, start : start + n] = arr
    return out


def _fraunhofer_intensity_from_field(
    field: np.ndarray, far_field_size: int
) -> np.ndarray:
    """补零 FFT 得到的焦平面强度。

    2f 台架把 SLM 瞳孔场映射到傅里叶变换 (透镜相位 + 传播到 z=f)。瞳孔被补零到
    ``far_field_size`` 后做居中 FFT; 不补零会混叠成点阵 (每个束腰半径 ~1.1 px)。

    Args:
        field: 瞳孔场 (复数)。可以是 n_grid 网格 (会补零) 或已补零的 far_field_size。
        far_field_size: 目标远场网格边长。
    """
    padded = _pad_centred(field, far_field_size)
    focal = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))
    return np.abs(focal) ** 2


def _forward_intensity(phase: np.ndarray, cfg: SquareShapingConfig) -> np.ndarray:
    """纯相位 SLM 图样 → 远场强度 (未归一化)。

    Gaussian 入射场乘以 ``exp(1j*phase)`` (纯相位 SLM), 补零 FFT 到焦平面。
    """
    pupil = cfg.make_gaussian_pupil().astype(np.complex128)
    field = pupil * np.exp(1j * np.asarray(phase, dtype=np.complex128))
    return _fraunhofer_intensity_from_field(field, cfg.far_field_size)


def make_square_target(cfg: SquareShapingConfig) -> np.ndarray:
    """在远场网格上构造归一化目标 (sum=1)。

    目标尺寸以**远场 (补零后) 像素**计, 使目标函数在尺度上稳定。
    """
    n = cfg.far_field_size
    y, x = np.ogrid[:n, :n]
    cx, cy = n // 2, n // 2
    side = int(cfg.target_side_px)
    target = np.zeros((n, n))
    if cfg.target_kind == "square":
        target[cy - side // 2 : cy + side // 2, cx - side // 2 : cx + side // 2] = 1.0
    elif cfg.target_kind == "circle":
        target[(x - cx) ** 2 + (y - cy) ** 2 <= (side / 2) ** 2] = 1.0
    elif cfg.target_kind == "gaussian":
        s = side / 4
        target = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s**2))
    elif cfg.target_kind == "annulus":
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        target[(r >= side / 2 - side / 4) & (r <= side / 2 + side / 4)] = 1.0
    else:
        raise ValueError(f"unknown target_kind {cfg.target_kind!r}")
    return target / target.sum()


def _initial_field(
    cfg: SquareShapingConfig,
    *,
    seed: int | None,
    base_phase: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """GS 的初始场 (Gaussian 入射 + 随机扰动), 已补零到远场网格。

    返回 ``(field, amp_slm, amp_target, target, base)``:
    * ``field``: 远场网格上的初始复场 ``amp_slm * exp(1j*(base + noise))``;
    * ``amp_slm``: 补零后的 Gaussian 瞳面幅度;
    * ``amp_target``: 目标幅度 (``sqrt(target)``);
    * ``target``: 目标强度;
    * ``base``: 固定的瞳孔相位 (raw 弧度, 补零后), 未使用时为零。
    """
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    target = make_square_target(cfg)
    amp_target = np.sqrt(target)
    pupil = cfg.make_gaussian_pupil()
    amp_slm = _pad_centred(pupil, cfg.far_field_size).astype(np.float64)
    if base_phase is None:
        base = np.zeros(amp_slm.shape, dtype=np.float64)
    else:
        base = _pad_centred(
            np.asarray(base_phase, dtype=np.float64), cfg.far_field_size
        )
    field = amp_slm * np.exp(1j * (base + rng.normal(0, 0.1, size=amp_slm.shape)))
    return field, amp_slm, amp_target, target, base


# ---------------------------------------------------------------------------
# 目标函数 / 质量指标
# ---------------------------------------------------------------------------


def _rolled_support(target: np.ndarray, center: tuple[int, int] | None) -> np.ndarray:
    """目标的布尔支撑域, 平移使其中心落在 ``center`` 上。

    ``center`` 是 ``(col, row)`` (每个调用方都用的 ``np.unravel_index(...)[::-1]``
    约定)。行按行偏移、列按列偏移 —— 弄反会在离轴光束时把支撑域转置, 静默让优
    化器追一个已不再覆盖光束的框。``center=None`` 时支撑域原样使用 (居中)。
    """
    sup = target > 0
    if center is not None:
        row_shift = int(round(center[1] - target.shape[0] // 2))
        col_shift = int(round(center[0] - target.shape[1] // 2))
        sup = np.roll(sup, (row_shift, col_shift), axis=(0, 1))
    return sup


def power_in_bucket(
    intensity: np.ndarray,
    target: np.ndarray,
    *,
    center: tuple[int, int] | None = None,
) -> float:
    """桶内功率: 落在目标区域内的总功率占比。"""
    total = intensity.sum()
    if total <= 0:
        return 0.0
    return float(intensity[_rolled_support(target, center)].sum() / total)


def efficiency(
    intensity: np.ndarray,
    target: np.ndarray,
    *,
    center: tuple[int, int] | None = None,
) -> float:
    """环围 / 桶内能量 = 目标支撑域内的功率。"""
    return power_in_bucket(intensity, target, center=center)


def uniformity_cv(
    intensity: np.ndarray,
    target: np.ndarray,
    *,
    center: tuple[int, int] | None = None,
) -> float:
    """目标支撑域*内部*强度的变异系数。

    越低越好 (0 = 完全均匀)。若目标支撑域内没有功率则返回 +inf。
    """
    vals = intensity[_rolled_support(target, center)]
    if vals.size == 0 or vals.mean() <= 0:
        return float("inf")
    return float(vals.std() / vals.mean())


def strehl(intensity: np.ndarray, target: np.ndarray) -> float:
    """仿真图样与目标图样的归一化重叠 (类 Strehl)。

    .. deprecated:: naming
        这是**均值中心化的余弦相似度** (形状相似度量), 不是光学 Strehl 比:
        它对整体尺度不变, 也不参照衍射极限理想远场。光学 Strehl 请使用
        :func:`ao_shaping.utils.image.beam_metrics.strehl_ratio`。
        数值行为已冻结 —— ``compute_bench_metrics["Strehl"]`` 与既有测试
        逐字节依赖本实现, 不得改动。
    """
    a = intensity
    b = target
    a = a - a.mean()
    b = b - b.mean()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na <= 0 or nb <= 0:
        return 0.0
    return float(np.dot(a.ravel(), b.ravel()) / (na * nb))


def zero_order_fraction(
    intensity: np.ndarray,
    center: tuple[int, int],
    *,
    zero_order_margin_px: float = 10.0,
) -> float:
    """落在中央 0 级保护带内的总功率占比。

    Args:
        intensity: (补零后) 远场强度。
        center: 0 级位置 ``(col, row)``。
        zero_order_margin_px: 保护半径 (远场 px)。
    """
    n = intensity.shape[0]
    y, x = np.ogrid[:n, :n]
    r = np.sqrt((x - int(center[0])) ** 2 + (y - int(center[1])) ** 2)
    m = r <= zero_order_margin_px
    total = intensity.sum()
    return float(intensity[m].sum() / total) if total > 0 else 0.0


def compute_bench_metrics(
    intensity: np.ndarray,
    target: np.ndarray,
    *,
    center: tuple[int, int] | None = None,
    zero_order_margin_px: float = 10.0,
) -> dict[str, float]:
    """计算综述中所用的整套目标函数。

    ``center`` 是目标框中心: ``None`` 表示使用目标自身居中的支撑域, 这对居中
    的仿真台架是正确的。**不要**把它默认成强度的 ``argmax`` —— 对类散斑场, 全
    局最大值在 ~1e-3 扰动下跳动, 让 ``argmax``-滚动的框让 PIB/CV 不连续。
    """
    peak = np.unravel_index(np.argmax(intensity), intensity.shape)[::-1]
    return {
        "PIB": power_in_bucket(intensity, target, center=center),
        "efficiency": efficiency(intensity, target, center=center),
        "CV": uniformity_cv(intensity, target, center=center),
        "Strehl": strehl(intensity, target),
        "zero_order": zero_order_fraction(
            intensity, peak, zero_order_margin_px=zero_order_margin_px
        ),
    }


def composite_from_pib_cv(
    pib: float,
    cv: float,
    *,
    w_pib: float = 0.5,
    w_unif: float = 0.5,
) -> float:
    """综合评分: 0.5*PIB + 0.5*(1/(1+CV))。

    均匀性按 ``1/(1+cv)`` 计分, 而不是裁剪过的线性映射 —— 裁剪项
    ``1 - min(cv/cv_ref, 1)`` 对任何可达的平顶 CV 都饱和到零, 让目标退化
    为纯 PIB。``1/(1+cv)`` 单调且永不饱和, 奖励每一分均匀性改善。
    """
    return w_pib * pib + w_unif * (1.0 / (1.0 + cv))


def composite_score(
    m: dict[str, float],
    *,
    w_pib: float = 0.5,
    w_unif: float = 0.5,
) -> float:
    """需要*最大化*的综合标量评分 (SPGD / 可微分的奖励)。"""
    pib = m.get("PIB", 0.0)
    cv = m.get("CV", float("inf"))
    return composite_from_pib_cv(pib, cv, w_pib=w_pib, w_unif=w_unif)


def shaping_score(
    intensity: np.ndarray,
    target: np.ndarray,
    *,
    center: tuple[int, int] | None = None,
    w_loggrad: float = 0.0,
) -> float:
    """Scalar shaping score, optionally including the intensity-space term.

    With ``w_loggrad == 0`` this is **exactly** :func:`composite_score` applied to
    :func:`compute_bench_metrics` -- the branch returns early rather than adding a
    zero coefficient, so the default path stays bit-for-bit identical and no new
    computation is performed on it.

    With ``w_loggrad > 0`` the returned value is
    ``composite - w_loggrad * log_gradient_difference_metric(intensity, target)``.
    Both halves are "higher is better", so the penalty is *subtracted*.

    The structural term is the repo's canonical intensity-space metric
    (:func:`~ao_shaping.utils.image.target.log_gradient_difference_metric`), not a
    new one: a speckled frame and a smooth frame can enclose the same power yet
    differ enormously in log gradient, which is the discrimination ``PIB``/``CV``
    cannot make and which matters when a shaping run is scored on structure rather
    than throughput.

    Args:
        intensity: Far-field intensity.
        target: Target intensity on the same grid.
        center: 0-order as ``(cx, cy)``; located by ``argmax`` when ``None``.
        w_loggrad: Weight of the structural penalty; must be >= 0 and finite.

    Returns:
        A scalar to maximise.

    Raises:
        ValueError: if ``w_loggrad`` is negative or non-finite.
    """
    if not math.isfinite(w_loggrad) or w_loggrad < 0.0:
        raise ValueError(f"w_loggrad must be >= 0 and finite, got {w_loggrad!r}")
    if w_loggrad <= 0.0:
        return composite_score(compute_bench_metrics(intensity, target, center=center))

    from ao_shaping.utils.image.target import log_gradient_difference_metric

    base = composite_score(compute_bench_metrics(intensity, target, center=center))
    return base - w_loggrad * log_gradient_difference_metric(intensity, target)


# ---------------------------------------------------------------------------
# 方法 1: Gerchberg-Saxton (幅度 <-> 相位约束)
# ---------------------------------------------------------------------------


def _gs_initial_field(
    cfg: SquareShapingConfig,
    *,
    seed: int | None,
    base_phase: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """GS 的初始场, 已补零到远场網格。

    本函数是 :func:`_initial_field` 的薄封装, 保持旧名字以兼容
    ``test_uniform_square_shaper`` 等测试。
    """
    return _initial_field(cfg, seed=seed, base_phase=base_phase)


def gs_shape(
    cfg: SquareShapingConfig,
    *,
    n_iters: int = 200,
    relax: float = 1.0,
    seed: int | None = None,
    verbose: bool = False,
    return_phase_only: bool = True,
    base_phase: np.ndarray | None = None,
) -> ShapingResult:
    """单块纯相位 SLM 的 Gerchberg-Saxton 整形。

    幅度约束 = 远场中的 sqrt(target), 相位约束 = 任意 (SLM 平面)。最终相位即
    SLM 图样 (raw 弧度, 不 mod 2π)。

    Args:
        cfg: 整形配置 (SquareShapingConfig)。
        n_iters: GS 迭代次数。
        relax: Fienup 松弛 (0, 1]。
        seed: 覆盖 ``cfg.seed``。
        verbose: 每 20 轮记录一次 history。
        return_phase_only: 保留接口兼容 (当前总是返回相位)。
        base_phase: 固定的瞳孔相位 (raw 弧度, n_grid x n_grid), GS 解加在其上。
    """
    field, amp_slm, amp_target, target, _base = _initial_field(
        cfg, seed=seed, base_phase=base_phase
    )

    history: list[dict[str, float]] = []
    for i in range(n_iters):
        ff = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
        ff = ff * (amp_target / (np.abs(ff) + 1e-12)) * relax
        ff = ff * (1 - relax) + amp_target * np.exp(1j * np.angle(ff)) * relax
        field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(ff)))
        field = amp_slm * np.exp(1j * np.angle(field))
        if verbose and i % 20 == 0:
            inten = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))) ** 2
            inten = inten / inten.max()
            center = np.unravel_index(np.argmax(inten), inten.shape)[::-1]
            m = compute_bench_metrics(inten, target, center=center)
            history.append({"iter": i, **m})

    inten = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))) ** 2
    inten = inten / (inten.sum() + 1e-12)
    center = np.unravel_index(np.argmax(inten), inten.shape)[::-1]
    metrics = compute_bench_metrics(inten, target, center=center)
    n = cfg.n_grid
    pad = (cfg.far_field_size - n) // 2
    pupil_phase = np.angle(field)[pad : pad + n, pad : pad + n]
    return ShapingResult(
        method="gerchberg_saxton",
        phase=pupil_phase,
        intensity=inten,
        metrics=metrics,
        history=history,
        n_iters=n_iters,
    )


# ---------------------------------------------------------------------------
# 方法 1b: 均匀方形整形的加权 GS / HIO 变体 (class-based)
# ---------------------------------------------------------------------------

GS_METHODS: tuple[str, ...] = (
    "weighted", "hio", "hio-weighted", "adaptive-feedback"
)
GS_SELECT: tuple[str, ...] = ("best", "last")


class UniformSquareShaper:
    """均匀方形远场整形的 GS 变体优化器 (class-based API)。

    与 ``gs_shape`` 共享同一个初始场与交替投影骨架, 但把远场幅度约束换成三种
    变体之一:

    * ``weighted``: 逐像素权重 ``W`` 的乘法更新 (``W *= (A_target/|F|)^gamma``,
      再裁剪到 ``[1/weight_clip, weight_clip]`` 并归一化到均值 1)。
    * ``hio``: Fienup HIO 更新 (``G' = A*B/|B|`` 在支撑域内, ``G - beta*B`` 在
      支撑域外)。
    * ``hio-weighted``: 前 ``hio_fraction`` 比例的迭代走 HIO (探索), 之后走
      weighted (收敛抛光)。
    * ``adaptive-feedback``: 目标区内按当前/目标振幅比调节反馈。接近目标时
      保留更多当前振幅, 偏差大时趋向目标振幅。

    能量守恒: 构造时把 ``amp_target`` 重标定到与瞳孔总能量一致; 因子 c 在
    ``angle(c*X) = angle(X)`` 下被抵消, 故 ``gamma=0`` 时与 ``gs_shape`` 逐位等价。
    """

    def __init__(
        self,
        cfg: SquareShapingConfig,
        *,
        n_iters: int = 200,
        method: str = "weighted",
        gamma: float = 0.5,
        weight_clip: float = 4.0,
        beta: float = 0.9,
        hio_fraction: float = 0.7,
        relax: float = 1.0,
        seed: int | None = None,
        verbose: bool = False,
        base_phase: np.ndarray | None = None,
        select: str = "best",
    ) -> None:
        if method not in GS_METHODS:
            raise ValueError(f"unknown method {method!r}; expected one of {GS_METHODS}")
        if not (0.0 <= gamma < 1.0):
            raise ValueError(f"gamma must be in [0, 1), got {gamma}")
        if weight_clip <= 1.0:
            raise ValueError(f"weight_clip must be > 1.0, got {weight_clip}")
        if not (0.0 <= beta <= 1.0):
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        if not (0.0 <= hio_fraction <= 1.0):
            raise ValueError(f"hio_fraction must be in [0, 1], got {hio_fraction}")
        if n_iters < 1:
            raise ValueError(f"n_iters must be >= 1, got {n_iters}")
        if select not in GS_SELECT:
            raise ValueError(f"select must be one of {GS_SELECT}, got {select!r}")

        self.cfg = cfg
        self.method = method
        self.n_iters = n_iters
        self.gamma = gamma
        self.weight_clip = weight_clip
        self.beta = beta
        self.hio_fraction = hio_fraction
        self.relax = relax
        self.verbose = verbose
        self.select = select

        field, amp_slm, amp_target, target, base = _initial_field(
            cfg, seed=seed, base_phase=base_phase
        )
        # 能量守恒: 把目标幅度重标到与瞳孔总能量一致。
        n_total = amp_slm.size
        rescale = np.sqrt(
            n_total
            * float(np.sum(amp_slm**2))
            / max(float(np.sum(amp_target**2)), 1e-30)
        )
        amp_target = amp_target * rescale

        self.field = field
        self.amp_slm = amp_slm
        self.amp_target = amp_target
        self.target = target
        self.base = base
        self.support = amp_target > 0
        self.W = np.ones_like(amp_target)
        self.g = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
        self._iter = 0
        self.history: list[dict[str, float]] = []
        self.pad = (cfg.far_field_size - cfg.n_grid) // 2

    def update(self) -> np.ndarray:
        """执行一步迭代, 返回裁剪后的 pupil 相位 (raw 弧度, 不 mod 2π)。"""
        if self.method == "hio" or (
            self.method == "hio-weighted"
            and self._iter < round(self.hio_fraction * self.n_iters)
        ):
            self._hio_step()
        elif self.method == "adaptive-feedback":
            self._adaptive_feedback_step()
        else:
            self._weighted_step()
        self._iter += 1
        n = self.cfg.n_grid
        p = self.pad
        return np.angle(self.field)[p : p + n, p : p + n]

    def _weighted_step(self) -> None:
        ff = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(self.field)))
        ratio = np.ones_like(self.amp_target)
        ratio[self.support] = (
            self.amp_target[self.support] / (np.abs(ff)[self.support] + 1e-12)
        ) ** self.gamma
        self.W = np.clip(self.W * ratio, 1.0 / self.weight_clip, self.weight_clip)
        self.W /= max(float(self.W[self.support].mean()), 1e-12)
        ff = (self.W * self.amp_target) * ff / (np.abs(ff) + 1e-12)
        ff = (
            ff * (1 - self.relax)
            + (self.W * self.amp_target) * np.exp(1j * np.angle(ff)) * self.relax
        )
        self.field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(ff)))
        self.field = self.amp_slm * np.exp(1j * np.angle(self.field))

    def _adaptive_feedback_step(self) -> None:
        ff = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(self.field)))
        current = np.abs(ff)
        target = self.amp_target
        closeness = np.minimum(current, target) / np.maximum(
            np.maximum(current, target), 1e-12
        )
        feedback = self.beta * closeness
        amplitude = np.where(
            self.support,
            feedback * current + (1.0 - feedback) * target,
            0.0,
        )
        ff = amplitude * np.exp(1j * np.angle(ff))
        self.field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(ff)))
        self.field = self.amp_slm * np.exp(1j * np.angle(self.field))

    def _hio_step(self) -> None:
        pupil = self.amp_slm * np.exp(
            1j * np.angle(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(self.g))))
        )
        b = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil)))
        self.g = np.where(
            self.support,
            self.amp_target * b / (np.abs(b) + 1e-12),
            self.g - self.beta * b,
        )
        self.field = self.amp_slm * np.exp(
            1j * np.angle(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(self.g))))
        )

    def _score(self) -> tuple[float, dict[str, float]]:
        """当前迭代的远场评分。"""
        inten = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(self.field)))) ** 2
        inten = inten / (inten.sum() + 1e-12)
        y, x = np.unravel_index(np.argmax(inten), inten.shape)
        m = compute_bench_metrics(inten, self.target, center=(int(x), int(y)))
        return composite_from_pib_cv(m["PIB"], m["CV"]), m

    def run(self) -> ShapingResult:
        """跑完 ``n_iters`` 步, 返回与 ``gs_shape`` 同构的结果。

        ⚠️ **``select="best"`` (默认) 返回历史最优迭代, 而不是最后一次。**
        """
        n = self.cfg.n_grid
        p = self.pad
        track = self.select == "best"
        best_score = -np.inf
        best_field: np.ndarray | None = None
        best_iter = -1
        for _ in range(self.n_iters):
            self.update()
            score, m = self._score()
            if track and score > best_score:
                best_score = score
                best_iter = self._iter - 1
                best_field = self.field[p : p + n, p : p + n].copy()
            if self.verbose and (self._iter - 1) % 20 == 0:
                self.history.append({"iter": self._iter - 1, **m})
        if track and best_field is not None:
            picked = best_field
        else:
            picked = self.field[p : p + n, p : p + n]
            best_iter = self._iter - 1
        field = np.zeros(self.amp_slm.shape, dtype=np.complex128)
        field[p : p + n, p : p + n] = picked
        inten = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))) ** 2
        inten = inten / (inten.sum() + 1e-12)
        y, x = np.unravel_index(np.argmax(inten), inten.shape)
        center = (int(x), int(y))
        metrics = compute_bench_metrics(inten, self.target, center=center)
        metrics = {**metrics, "best_iter": float(best_iter)}
        pupil_phase = np.angle(picked)
        return ShapingResult(
            method=self.method,
            phase=pupil_phase,
            intensity=inten,
            metrics=metrics,
            history=self.history,
            n_iters=self._iter,
        )


# ---------------------------------------------------------------------------
# 方法 2: 可微分梯度下降 (torch, 可选)
# ---------------------------------------------------------------------------


def differentiable_shape(
    cfg: SquareShapingConfig,
    *,
    n_iters: int = 300,
    lr: float = 0.05,
    seed: int | None = None,
) -> ShapingResult:
    """可微分远场整形: 对 SLM 相位做梯度下降。

    Loss = -(overlap + 0.5 * PIB)。需要 torch, 惰性导入。
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("differentiable_shape requires torch") from exc

    target = make_square_target(cfg)
    tgt = torch.as_tensor(target, dtype=torch.float64).double()
    n = cfg.n_grid
    m = cfg.far_field_size
    pad = (m - n) // 2

    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    init_phase = torch.as_tensor(
        rng.normal(0, 0.1, size=(n, n)), dtype=torch.float64
    ).double()
    param = torch.nn.Parameter(init_phase)
    opt = torch.optim.Adam([param], lr=lr)

    in_amp = torch.as_tensor(cfg.make_gaussian_pupil(), dtype=torch.float64).double()

    def intensity(phase: Any) -> Any:
        field = in_amp * torch.exp(1j * phase)
        if m > n:
            padded = torch.zeros((m, m), dtype=torch.complex128)
            padded[pad : pad + n, pad : pad + n] = field
            field = padded
        focal = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(field)))
        return focal.real**2 + focal.imag**2

    history: list[dict[str, float]] = []
    for i in range(n_iters):
        opt.zero_grad()
        inten = intensity(param)
        inten = inten / inten.sum()
        a = inten - inten.mean()
        b = tgt - tgt.mean()
        overlap = (a * b).sum() / (a.norm() * b.norm() + 1e-12)
        sup = (tgt > 0).double()
        pib = (inten * sup).sum() / inten.sum()
        loss = -overlap - 0.5 * pib
        loss.backward()
        opt.step()
        if i % 50 == 0:
            with torch.no_grad():
                inten_np = intensity(param).numpy()
                inten_np = inten_np / inten_np.max()
                center = np.unravel_index(np.argmax(inten_np), inten_np.shape)[::-1]
                metrics_hist = compute_bench_metrics(inten_np, target, center=center)
                history.append({"iter": i, **metrics_hist})

    with torch.no_grad():
        inten_np = intensity(param).numpy()
        inten_np = inten_np / (inten_np.sum() + 1e-12)
    center = np.unravel_index(np.argmax(inten_np), inten_np.shape)[::-1]
    metrics = compute_bench_metrics(inten_np, target, center=center)
    return ShapingResult(
        method="differentiable",
        phase=param.detach().numpy(),
        intensity=inten_np,
        metrics=metrics,
        history=history,
        n_iters=n_iters,
    )


# ---------------------------------------------------------------------------
# 方法 3: SPGD (SLM 相位上的无感知黑盒梯度)
# ---------------------------------------------------------------------------


def spgd_shape(
    cfg: SquareShapingConfig,
    *,
    n_iters: int = 600,
    delta: float = 0.1,
    lr: float = 0.02,
    seed: int | None = None,
    dim: int | None = None,
) -> ShapingResult:
    """低维 (Zernike / 粗) 相位基上的无感知 SPGD。

    SLM 相位被参数化为少量 freeform 系数 (一个粗网格), 使随机梯度搜索在仿真中
    可行。奖励是综合评分 (PIB + 均匀性)。
    """
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    target = make_square_target(cfg)
    d = dim or 16
    n_par = d * d
    phase_flat = rng.normal(0, 0.05, size=n_par)

    def upsample(vec: np.ndarray) -> np.ndarray:
        """把 d×d freeform 系数网格映射为完整 SLM 相位 (kron 上采样)。"""
        ph = np.kron(vec.reshape(d, d), np.ones((cfg.n_grid // d, cfg.n_grid // d)))
        if ph.shape != (cfg.n_grid, cfg.n_grid):
            ph = ph[: cfg.n_grid, : cfg.n_grid]
        return ph

    def eval_score(vec: np.ndarray) -> float:
        inten = _forward_intensity(upsample(vec), cfg)
        inten = inten / inten.max()
        center = np.unravel_index(np.argmax(inten), inten.shape)[::-1]
        # Default cfg.w_loggrad == 0 -> shaping_score == the historical
        # composite_score, bit for bit.
        return shaping_score(
            inten, target, center=center, w_loggrad=float(cfg.w_loggrad)
        )

    score = eval_score(phase_flat)
    g = np.zeros(n_par)
    history: list[dict[str, float]] = [
        {"iter": 0, "score": score, "PIB": 0, "CV": float("inf")}
    ]
    for i in range(1, n_iters):
        sgn = rng.choice([-1, 1], size=n_par)
        cand = phase_flat + delta * sgn
        s2 = eval_score(cand)
        g = 0.95 * g + 0.05 * sgn * (s2 - score)
        phase_flat = phase_flat + lr * g
        score = eval_score(phase_flat)
        if i % 50 == 0:
            inten = _forward_intensity(upsample(phase_flat), cfg)
            inten = inten / inten.max()
            center = np.unravel_index(np.argmax(inten), inten.shape)[::-1]
            m = compute_bench_metrics(inten, target, center=center)
            history.append({"iter": i, "score": score, "PIB": m["PIB"], "CV": m["CV"]})

    ph = upsample(phase_flat)
    inten = _forward_intensity(ph, cfg)
    inten = inten / (inten.sum() + 1e-12)
    center = np.unravel_index(np.argmax(inten), inten.shape)[::-1]
    metrics = compute_bench_metrics(inten, target, center=center)
    metrics["score"] = composite_score(metrics)
    return ShapingResult(
        method="spgd_freeform",
        phase=ph,
        intensity=inten,
        metrics=metrics,
        history=history,
        n_iters=n_iters,
    )


# ---------------------------------------------------------------------------
# 方法 4: 解析的"纯幅度"目标 (单板纯幅度基线)
# ---------------------------------------------------------------------------


def analytic_amplitude_target(cfg: SquareShapingConfig) -> ShapingResult:
    """平凡基线: 目标本身 (幅度整形, 无相位)。

    作为 "一块纯幅度 (非纯相位) SLM 会产出什么" 的参照 —— 给出纯相位器件可达
    质量的上界。
    """
    target = make_square_target(cfg)
    metrics = compute_bench_metrics(target, target)
    return ShapingResult(
        method="analytic_amplitude_baseline",
        phase=np.zeros((cfg.n_grid, cfg.n_grid)),
        intensity=target,
        metrics=metrics,
        n_iters=0,
    )
