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

from ao_shaping.utils.image.exposure_fusion import _gaussian_kernel_1d



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
    "analytic_ot_phase",
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
    """2f 傅里 Fourier SLM 整形实验系统的算法参数。

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
    target_edge_soft_px: float = 0.0
    """Soft-edge width of the GS target amplitude (Gaussian sigma, far-field px).

    ``0.0`` (the default) reproduces the hard-edged square bit for bit. A few
    pixels of super-Gaussian falloff suppress the spectral leakage a hard edge
    inevitably produces (sinc-type ringing and high-frequency speckle), which
    is the cheapest available uniformity gain -- changing the *target* beats
    changing the *algorithm*. Applies to the GS-family amplitude constraint
    only; ``make_square_target`` (and therefore every metric that derives its
    support from it) stays hard-edged, so PIB / CV keep their meaning.
    """
    target_gaussian_sigma_px: float | None = None
    """Override the ``gaussian`` target's sigma (far-field px).

    ``None`` (the default) keeps the historical ``target_side_px / 4`` bit for
    bit. Set it when the caller's gaussian target is defined by a *ratio* of its
    characteristic size rather than by that fixed quarter: the ring/sigma
    geometry must match whatever mask the caller then *measures* PIB/CV on,
    otherwise the GS target and the objective are two different shapes.
    """
    target_inner_radius_px: float | None = None
    target_outer_radius_px: float | None = None
    """Override the ``annulus`` target's inner / outer radius (far-field px).

    ``None`` (the default) keeps the historical ``side/4 .. 3*side/4`` bit for
    bit. The historical ring is a fixed 1:3 radius ratio at a *fixed* outer
    radius, which cannot express "outer diameter = target side, inner = 60% of
    it"; a caller that defines the ring that way must pass both radii or its
    measured mask would sit on a different ring than the GS phase aims at.
    """
    target_amplitude: np.ndarray | None = field(default=None, repr=False, compare=False)
    """Optional **far-field amplitude** target to inject instead of a built-in kind.

    ``None`` (the default) keeps every existing branch bit for bit. When given
    it must be a ``(far_field_size, far_field_size)`` non-negative array; GS then
    constrains the far field to ``sqrt(target)`` with ``target = amplitude**2``
    normalised to unit sum, exactly like the built-in kinds. This is the
    injection point for shapes the built-in kinds cannot express (rectangle,
    grid, cross, pentagon) -- so the caller keeps a single GS loop and a single
    geometry source (``utils.image.target.create_target_shape``) instead of a
    second copy of either.

    Excluded from ``__eq__``/``repr``: an array-valued field would make config
    equality raise ("truth value of an array is ambiguous") and dump megabytes
    into a log line.
    """
    init_noise: float = 0.1
    """Sigma (radians) of the random perturbation on the GS initial field.

    ``0.1`` is the historical value and keeps the default path bit-for-bit.
    Set ``0.0`` when the initial field is an analytic transport map
    (:func:`analytic_ot_phase`): its structure is the point, and noise only
    degrades it.
    """
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
        if self.target_gaussian_sigma_px is not None and not (
            math.isfinite(self.target_gaussian_sigma_px)
            and self.target_gaussian_sigma_px > 0.0
        ):
            raise ValueError(
                "target_gaussian_sigma_px must be > 0 and finite when given, got "
                f"{self.target_gaussian_sigma_px!r}"
            )
        for name in ("target_inner_radius_px", "target_outer_radius_px"):
            value = getattr(self, name)
            if value is not None and not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be > 0 and finite when given, got {value!r}")
        if (
            self.target_inner_radius_px is not None
            and self.target_outer_radius_px is not None
            and self.target_inner_radius_px >= self.target_outer_radius_px
        ):
            raise ValueError(
                "target_inner_radius_px must be smaller than target_outer_radius_px, "
                f"got inner={self.target_inner_radius_px} outer="
                f"{self.target_outer_radius_px}"
            )
        if self.target_amplitude is not None:
            amp = np.asarray(self.target_amplitude)
            if amp.ndim != 2:
                raise ValueError(
                    f"target_amplitude must be 2-D, got shape {amp.shape}"
                )
            if not np.isfinite(amp).all() or float(amp.min()) < 0.0:
                raise ValueError("target_amplitude must be finite and non-negative")
            if float(amp.sum()) <= 0.0:
                raise ValueError("target_amplitude must carry some energy")
        if not math.isfinite(self.w_loggrad) or self.w_loggrad < 0.0:
            raise ValueError(
                f"w_loggrad must be >= 0 and finite, got {self.w_loggrad!r}"
            )
        if not math.isfinite(self.target_edge_soft_px) or self.target_edge_soft_px < 0.0:
            raise ValueError(
                "target_edge_soft_px must be >= 0 and finite, got "
                f"{self.target_edge_soft_px!r}"
            )
        if not math.isfinite(self.init_noise) or self.init_noise < 0.0:
            raise ValueError(
                f"init_noise must be >= 0 and finite, got {self.init_noise!r}"
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
    """把方形瞳面网格数组居中补零到 ``far_field_size``。

    瞳面占据中央的 ``n_grid x n_grid`` 块; 补零区域就是 SLM 口径之外的暗区域。
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

    2f 实验系统把 SLM 瞳面场映射到傅里叶变换 (透镜相位 + 传播到 z=f)。瞳面被补零到
    ``far_field_size`` 后做居中 FFT; 不补零会混叠成点阵 (每个束腰半径 ~1.1 px)。

    Args:
        field: 瞳面场 (复数)。可以是 n_grid 网格 (会补零) 或已补零的 far_field_size。
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

    ``cfg.target_amplitude`` 非 ``None`` 时直接用它 (取其平方并归一化到
    ``sum=1``, 与内建分支同一口径) —— 这是内建 ``target_kind`` 表达不了的形状
    (rectangle / grid / cross / pentagon) 的注入入口。``None`` 时逐位等于历史
    实现。
    """
    n = cfg.far_field_size
    y, x = np.ogrid[:n, :n]
    cx, cy = n // 2, n // 2
    side = int(cfg.target_side_px)
    if cfg.target_amplitude is not None:
        amp = np.asarray(cfg.target_amplitude, dtype=np.float64)
        if amp.shape != (n, n):
            raise ValueError(
                f"target_amplitude shape {amp.shape} != far-field grid ({n}, {n})"
            )
        target = amp**2
        total = float(target.sum())
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("target_amplitude carries no energy on the far-field grid")
        return target / total
    target = np.zeros((n, n))
    if cfg.target_kind == "square":
        target[cy - side // 2 : cy + side // 2, cx - side // 2 : cx + side // 2] = 1.0
    elif cfg.target_kind == "circle":
        target[(x - cx) ** 2 + (y - cy) ** 2 <= (side / 2) ** 2] = 1.0
    elif cfg.target_kind == "gaussian":
        if cfg.target_gaussian_sigma_px is None:
            s = side / 4
        else:
            s = float(cfg.target_gaussian_sigma_px)
        target = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s**2))
    elif cfg.target_kind == "annulus":
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        inner = (
            side / 2 - side / 4
            if cfg.target_inner_radius_px is None
            else float(cfg.target_inner_radius_px)
        )
        outer = (
            side / 2 + side / 4
            if cfg.target_outer_radius_px is None
            else float(cfg.target_outer_radius_px)
        )
        target[(r >= inner) & (r <= outer)] = 1.0
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
    * ``amp_target``: 目标幅度 (``sqrt(target)``; ``cfg.target_edge_soft_px>0``
      时为软边目标幅度, 能量与硬边目标一致);
    * ``target``: 目标强度 (**硬边支撑域** —— 指标与支撑域以它为准; 内建方斑取
      值恰为 0/1, ``cfg.target_amplitude`` 注入时是注入幅度的平方);
    * ``base``: 固定的瞳面相位 (raw 弧度, 补零后), 未使用时为零。

    ``cfg.init_noise`` scales the random part; ``0.0`` makes the initial field
    exactly ``base_phase`` (used with :func:`analytic_ot_phase`).
    """
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    target = make_square_target(cfg)
    pupil = cfg.make_gaussian_pupil()
    amp_slm = _pad_centred(pupil, cfg.far_field_size).astype(np.float64)
    if base_phase is None:
        base = np.zeros(amp_slm.shape, dtype=np.float64)
    else:
        base = _pad_centred(
            np.asarray(base_phase, dtype=np.float64), cfg.far_field_size
        )
    field = amp_slm * np.exp(
        1j * (base + rng.normal(0.0, cfg.init_noise, size=amp_slm.shape))
    )
    if cfg.target_edge_soft_px > 0.0:
        amp_target = _soft_edge_amplitude(target, cfg.target_edge_soft_px)
    else:
        amp_target = np.sqrt(target)
    return field, amp_slm, amp_target, target, base


# ---------------------------------------------------------------------------
# 目标域整形: 软边目标幅度 + 解析最优传输 (OT) 初值
#
# 综述的两条核心建议: (1) 改目标比改算法划算 —— 硬边方斑的 sinc 型频谱泄漏
# 必然产生振铃, 几个像素的超高斯下降沿即可把均匀度从 ~85% 推到 >90%;
# (2) 解析/OT 初值 + 少量迭代精修是目前性价比最高的组合 —— OT 解无相位涡旋、
# 可插值、几乎不需要调参, 作为任意迭代算法的通用初值可把收敛迭代数减半。
#
# ``_gaussian_kernel_1d`` is imported from the single implementation in
# :mod:`ao_shaping.utils.image.exposure_fusion` (issue #199 R4: a byte-identical
# second copy used to live here). For every ``sigma > 0`` the two are
# numerically identical; ``_soft_edge_amplitude`` guards ``sigma <= 0`` before
# calling, so the ``sigma == 0`` identity arm is unused from this module.
# ---------------------------------------------------------------------------


def _soft_edge_amplitude(target: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gaussian-softened target **amplitude** on the far-field grid.

    The hard square intensity is blurred with a Gaussian of ``sigma_px``
    (far-field px) and renormalised to unit energy before the square root, so
    the constraint's total energy matches the hard target's and ``relax``
    keeps its meaning. Blurring the *intensity* (not the amplitude) keeps the
    edge profile monotone, which is what suppresses the ringing.
    """
    if sigma_px <= 0.0:
        return np.sqrt(target)
    kernel = _gaussian_kernel_1d(sigma_px)
    radius = (kernel.size - 1) // 2
    rows = np.zeros_like(target)
    padded = np.pad(target, ((radius, radius), (0, 0)))
    for i, weight in enumerate(kernel):
        rows += weight * padded[i : i + target.shape[0], :]
    out = np.zeros_like(target)
    padded = np.pad(rows, ((0, 0), (radius, radius)))
    for i, weight in enumerate(kernel):
        out += weight * padded[:, i : i + target.shape[1]]
    total = float(out.sum())
    if not math.isfinite(total) or total <= 0.0:
        return np.sqrt(target)
    return np.sqrt(out / total)


def _relax_at(relax: float, ramp_frac: float, n_iters: int, epoch: int) -> float:
    """Effective Fienup relaxation at ``epoch`` under a linear tighten ramp.

    ``ramp_frac <= 0`` returns ``relax`` unchanged -- the historical constant
    schedule. Otherwise the relaxation ramps linearly from ``relax`` up to 1.0
    over the first ``ramp_frac * n_iters`` iterations: a loose amplitude
    constraint early (the target amplitude may float) escapes the ringing
    basin faster, and the tightened tail converges. Requires ``relax < 1`` to
    have any effect.
    """
    if ramp_frac <= 0.0:
        return relax
    ramp_iters = max(1, int(round(ramp_frac * max(1, n_iters))))
    frac = min(1.0, (epoch + 1) / ramp_iters)
    return relax + (1.0 - relax) * frac


def _amplitude_error(ff: np.ndarray, amp_target: np.ndarray) -> float:
    """Relative RMS mismatch between a far field and the target amplitude."""
    denom = float(np.sqrt(np.mean(amp_target**2)))
    if denom <= 0.0:
        return 0.0
    return float(np.sqrt(np.mean((np.abs(ff) - amp_target) ** 2)) / denom)


def analytic_ot_phase(
    cfg: SquareShapingConfig,
    *,
    target_side_px: int | None = None,
    amplitude: np.ndarray | None = None,
) -> np.ndarray:
    """Separable Monge (Fermat-CDF) optimal-transport phase, input -> square.

    Models "transport the input amplitude onto a uniform square" as a Monge
    optimal-transport problem and returns the Kantorovich potential in the
    separable (Fermat-CDF) form: each axis is mapped through the CDF of its
    own marginal, and the phase is the sum of the two 1-D potentials.

    On a 2f-Fourier bench a pupil-plane phase slope ``phi'`` displaces the
    focal-plane ray by ``X = f*lambda*phi'/(2*pi)``, so the potential that
    realises the transport map ``X(x) = s*(2*F(x)-1)`` (``F`` the marginal
    CDF, ``s`` the target half-width) is::

        phi(x) = 2*pi*s/(lambda*f) * integral of (2*F(x')-1) dx'

    Properties that matter for the warm start:

    * **no phase vortices** -- a principal source of GS efficiency loss and
      residual speckle;
    * **interpolable** -- the coarse multi-resolution pass upsamples it
      losslessly enough for the fine pass to continue from;
    * **almost no hyper-parameters** -- the only input is the target side.

    Limitation: it is a geometric-optics solution, so it is exact only when
    the target is many diffraction-limited spots across. Keep 10-50 GS
    refinement iterations after it; use it as an *initial field*, not as the
    final answer.

    Args:
        cfg: Shaping configuration; supplies the grid, the input amplitude
            model (:meth:`SquareShapingConfig.make_gaussian_pupil`), the
            wavelength / focal length and the far-field pixel size.
        target_side_px: Target square side in far-field pixels (default
            ``cfg.target_side_px``).
        amplitude: Optional measured input amplitude on the pupil grid
            (``cfg.n_grid`` square). When given, the marginals are taken from
            it instead of the Gaussian model -- the documented remedy for a
            bench whose real illumination is not the modelled Gaussian.

    Returns:
        The phase in **raw unwrapped radians** on the ``cfg.n_grid`` pupil
        grid (the same convention every other phase in this module uses).
    """
    n = cfg.n_grid
    if amplitude is None:
        amp = cfg.make_gaussian_pupil()
    else:
        amp = np.asarray(amplitude, dtype=np.float64)
        if amp.shape != (n, n):
            raise ValueError(f"amplitude shape {amp.shape} != pupil grid ({n}, {n})")
        if not np.isfinite(amp).all() or amp.min() < 0.0:
            raise ValueError("amplitude must be finite and non-negative")
    intensity = amp**2
    marginal_x = intensity.sum(axis=0)
    marginal_y = intensity.sum(axis=1)
    total_x = float(marginal_x.sum())
    total_y = float(marginal_y.sum())
    if not (total_x > 0.0 and total_y > 0.0):
        raise ValueError("degenerate input amplitude: a marginal integrates to zero")
    cdf_x = np.cumsum(marginal_x / total_x)
    cdf_y = np.cumsum(marginal_y / total_y)
    side = cfg.target_side_px if target_side_px is None else int(target_side_px)
    if side < 1:
        raise ValueError(f"target_side_px must be >= 1, got {side}")
    half_side_m = 0.5 * float(side) * cfg.far_field_pixel_size
    pitch_m = cfg.aperture_size / n
    scale = 2.0 * math.pi * half_side_m * pitch_m / (cfg.wavelength * cfg.focal_length)
    phi_x = scale * np.cumsum(2.0 * cdf_x - 1.0)
    phi_y = scale * np.cumsum(2.0 * cdf_y - 1.0)
    return phi_y[:, None] + phi_x[None, :]


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
    的仿真实验系统是正确的。**不要**把它默认成强度的 ``argmax`` —— 对类散斑场, 全
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

    均匀度按 ``1/(1+cv)`` 计分, 而不是裁剪过的线性映射 —— 裁剪项
    ``1 - min(cv/cv_ref, 1)`` 对任何可达的平顶 CV 都饱和到零, 让目标退化
    为纯 PIB。``1/(1+cv)`` 单调且永不饱和, 奖励每一分均匀度改善。
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
    convergence_tol: float = 0.0,
    relax_ramp_frac: float = 0.0,
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
        base_phase: 固定的瞳面相位 (raw 弧度, n_grid x n_grid), GS 解加在其上。
            配 ``cfg.init_noise = 0`` 即为纯解析初值 (如
            :func:`analytic_ot_phase`)。
        convergence_tol: 收敛判据 —— 远场幅度 RMS 误差的相对变化小于该值即
            收工 (0 = 跑满 ``n_iters``, 历史行为)。GS 在振铃解附近空转是常见
            的时间浪费, 该判据典型可省 20%-40% 迭代。只在 >0 时付出每轮一次
            误差度量。
        relax_ramp_frac: 分阶段放开约束 —— 前 ``frac * n_iters`` 轮内把
            ``relax`` 线性收紧到 1.0 (0 = 恒定 ``relax``, 历史行为)。松弛的
            早期约束允许目标区幅度浮动, 比全程硬约束更快跳出振铃。
    """
    field, amp_slm, amp_target, target, _base = _initial_field(
        cfg, seed=seed, base_phase=base_phase
    )

    history: list[dict[str, float]] = []
    n_run = 0
    prev_err: float | None = None
    for i in range(n_iters):
        relax_i = _relax_at(relax, relax_ramp_frac, n_iters, i)
        ff = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
        if convergence_tol > 0.0:
            # 收敛判据: 远场幅度误差的相对变化低于 tol 即收工, 避免在振铃解
            # 附近空转。误差取自**施加约束前**的远场 (即当前重建的真实误差)。
            err = _amplitude_error(ff, amp_target)
            if (
                prev_err is not None
                and abs(prev_err - err) <= convergence_tol * max(prev_err, 1e-12)
            ):
                break
            prev_err = err
        # Fienup 松弛: 约束后的远场与**约束前**的远场按 relax 混合。
        # (历史实现先把 ff 缩放到目标幅度, 再与**缩放后的自己**混合, 结果只差
        # 一个正实数因子, 随后被瞳面 angle 投影完全抵消 —— relax<1 时是恒等
        # 变换, 参数从未生效。angle 仍取缩放后的场, 故 relax=1 时与历史实现
        # 逐字节相同, 默认路径不变。)
        ff_scaled = ff * (amp_target / (np.abs(ff) + 1e-12))
        constrained = amp_target * np.exp(1j * np.angle(ff_scaled))
        ff = ff * (1 - relax_i) + constrained * relax_i
        field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(ff)))
        field = amp_slm * np.exp(1j * np.angle(field))
        n_run = i + 1
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
    if convergence_tol > 0.0:
        # 让调用方看得出是「跑满」还是「提前收敛」。
        metrics = {**metrics, "iters_run": float(n_run)}
    n = cfg.n_grid
    pad = (cfg.far_field_size - n) // 2
    pupil_phase = np.angle(field)[pad : pad + n, pad : pad + n]
    return ShapingResult(
        method="gerchberg_saxton",
        phase=pupil_phase,
        intensity=inten,
        metrics=metrics,
        history=history,
        n_iters=n_run,
    )


# ---------------------------------------------------------------------------
# 方法 1b: 均匀方形整形的加权 GS / HIO 变体 (class-based)
# ---------------------------------------------------------------------------

GS_METHODS: tuple[str, ...] = (
    "weighted", "hio", "hio-weighted", "adaptive-feedback", "sf", "dm"
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
    * ``sf``: Fienup 2006 Simplified Hybrid Input-Output。与 ``hio`` 同构, 但
      支撑域外的反馈只用**输出** ``b``: ``(1 - beta) * b`` (HIO 用 ``g - beta*b``)。
    * ``dm``: Elser 2003 Difference Map。两个投影 ``P_A`` (远场幅度约束) 与
      ``P_B`` (瞳面幅度约束) 交替, 由 ``dm_gamma_a`` / ``dm_gamma_b`` 控制。

    **DM → HIO 退化 (Elser 2003)**: 取 ``beta=1``, ``gamma_A = -1/beta_HIO``,
    ``gamma_B = +1/beta_HIO`` 时, difference map 退化为 HIO (松弛系数
    ``beta_HIO``)。各变体本身由
    ``tests/ao_shaping/algorithm/signal_processing/test_square_shaping.py`` 按
    ``GS_METHODS`` 参数化覆盖, 但**该退化等式没有数值验证**（无对应测试）。

    能量守恒: 构造时把 ``amp_target`` 重标定到与瞳面总能量一致; 因子 c 在
    ``angle(c*X) = angle(X)`` 下被抵消, 故 ``gamma=0`` 时与 ``gs_shape`` 逐位等价。

    ``convergence_tol`` / ``relax_ramp_frac`` / ``base_phase`` 与
    :func:`gs_shape` 同义: 评分收敛判据 (评分每轮都已计算, 不增加成本)、
    分阶段收紧的幅度约束、以及固定初值 (配 ``cfg.init_noise=0`` 即纯解析初值)。
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
        dm_gamma_a: float = -1.0,
        dm_gamma_b: float = 1.0,
        relax: float = 1.0,
        convergence_tol: float = 0.0,
        relax_ramp_frac: float = 0.0,
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
        if not math.isfinite(dm_gamma_a):
            raise ValueError(f"dm_gamma_a must be finite, got {dm_gamma_a}")
        if not math.isfinite(dm_gamma_b):
            raise ValueError(f"dm_gamma_b must be finite, got {dm_gamma_b}")
        if n_iters < 1:
            raise ValueError(f"n_iters must be >= 1, got {n_iters}")
        if select not in GS_SELECT:
            raise ValueError(f"select must be one of {GS_SELECT}, got {select!r}")
        if not math.isfinite(convergence_tol) or convergence_tol < 0.0:
            raise ValueError(
                f"convergence_tol must be >= 0 and finite, got {convergence_tol!r}"
            )
        if not (0.0 <= relax_ramp_frac <= 1.0):
            raise ValueError(
                f"relax_ramp_frac must be in [0, 1], got {relax_ramp_frac!r}"
            )

        self.cfg = cfg
        self.method = method
        self.n_iters = n_iters
        self.gamma = gamma
        self.weight_clip = weight_clip
        self.beta = beta
        self.hio_fraction = hio_fraction
        self.dm_gamma_a = dm_gamma_a
        self.dm_gamma_b = dm_gamma_b
        self.relax = relax
        self.convergence_tol = convergence_tol
        self.relax_ramp_frac = relax_ramp_frac
        self.verbose = verbose
        self.select = select

        field, amp_slm, amp_target, target, base = _initial_field(
            cfg, seed=seed, base_phase=base_phase
        )
        # 能量守恒: 把目标幅度重标到与瞳面总能量一致。
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
        # 支撑域取**硬边**目标: 软边目标幅度的 Gaussian 尾没有精确零点,
        # ``amp_target > 0`` 会变成整幅画面, 加权/HIO 的信号区就失效了。
        # 软边关闭时两者逐字节相同 (方斑取值恰为 0/1)。
        self.support = target > 0
        self.W = np.ones_like(amp_target)
        self.g = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
        self._iter = 0
        self.history: list[dict[str, float]] = []
        self.pad = (cfg.far_field_size - cfg.n_grid) // 2

    def _relax_now(self) -> float:
        """Current-iteration relaxation under the optional tighten ramp."""
        return _relax_at(self.relax, self.relax_ramp_frac, self.n_iters, self._iter)

    def update(self) -> np.ndarray:
        """执行一步迭代, 返回裁剪后的 pupil 相位 (raw 弧度, 不 mod 2π)。"""
        if self.method == "sf":
            self._sf_step()
        elif self.method == "dm":
            self._dm_step()
        elif self.method == "hio" or (
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
        relax = self._relax_now()
        ff = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(self.field)))
        ratio = np.ones_like(self.amp_target)
        ratio[self.support] = (
            self.amp_target[self.support] / (np.abs(ff)[self.support] + 1e-12)
        ) ** self.gamma
        self.W = np.clip(self.W * ratio, 1.0 / self.weight_clip, self.weight_clip)
        self.W /= max(float(self.W[self.support].mean()), 1e-12)
        # Fienup 松弛: 与**约束前**的远场混合 (同 :func:`gs_shape` 的修正;
        # 旧的「先缩放再与自身混合」只差一个正实数因子, 被 angle 投影抵消,
        # relax<1 时从未生效; angle 取缩放后的场, relax=1 逐字节不变)。
        ff_scaled = (self.W * self.amp_target) * ff / (np.abs(ff) + 1e-12)
        constrained = (self.W * self.amp_target) * np.exp(1j * np.angle(ff_scaled))
        ff = ff * (1 - relax) + constrained * relax
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

    def _sf_step(self) -> None:
        """Fienup 2006 Simplified Hybrid Input-Output (SF) 一步。

        与 :meth:`_hio_step` 同构, 唯一区别是支撑域外的反馈只用**输出** ``b``:
        ``(1 - beta) * b`` (HIO 用 ``g - beta * b``)。复用同一个 ``self.beta``。
        """
        pupil = self.amp_slm * np.exp(
            1j * np.angle(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(self.g))))
        )
        b = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil)))
        self.g = np.where(
            self.support,
            self.amp_target * b / (np.abs(b) + 1e-12),
            (1.0 - self.beta) * b,
        )
        self.field = self.amp_slm * np.exp(
            1j * np.angle(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(self.g))))
        )

    def _dm_step(self) -> None:
        """Elser 2003 Difference Map (DM) 一步。

        两个投影:

        * ``P_A`` — 远场幅度约束 (支撑域内取目标幅度, 域外恒等);
        * ``P_B`` — 瞳面幅度约束 (经瞳面往返)。

        更新 ``g_{k+1} = g_k + beta * [P_A(f_B(g_k)) - P_B(f_A(g_k))]``, 其中
        ``f_A(g) = (1 + gamma_A) P_A(g) - gamma_A g``,
        ``f_B(g) = (1 + gamma_B) P_B(g) - gamma_B g``。

        取 ``beta=1``, ``gamma_A = -1/beta_HIO``, ``gamma_B = +1/beta_HIO`` 时
        退化为 HIO (见类 docstring)。
        """
        g = self.g
        # P_A: 远场幅度约束 (支撑域内取目标幅度, 域外恒等)。
        p_a = np.where(
            self.support,
            self.amp_target * g / (np.abs(g) + 1e-12),
            g,
        )
        # P_B: 瞳面幅度约束 (经瞳面往返)。
        pupil = self.amp_slm * np.exp(
            1j * np.angle(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(g))))
        )
        p_b = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil)))
        f_a = (1.0 + self.dm_gamma_a) * p_a - self.dm_gamma_a * g
        f_b = (1.0 + self.dm_gamma_b) * p_b - self.dm_gamma_b * g
        # P_A(f_B) 与 P_B(f_A)。
        p_a_fb = np.where(
            self.support,
            self.amp_target * f_b / (np.abs(f_b) + 1e-12),
            f_b,
        )
        pupil_fa = self.amp_slm * np.exp(
            1j * np.angle(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(f_a))))
        )
        p_b_fa = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_fa)))
        self.g = g + self.beta * (p_a_fb - p_b_fa)
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

        ``convergence_tol > 0`` 时按综合评分的变化提前收工 (评分已每轮计算,
        判据本身不增加成本); 实际步数记入 ``n_iters``。
        """
        n = self.cfg.n_grid
        p = self.pad
        track = self.select == "best"
        best_score = -np.inf
        best_field: np.ndarray | None = None
        best_iter = -1
        prev_score: float | None = None
        for _ in range(self.n_iters):
            self.update()
            score, m = self._score()
            if track and score > best_score:
                best_score = score
                best_iter = self._iter - 1
                best_field = self.field[p : p + n, p : p + n].copy()
            if self.convergence_tol > 0.0 and prev_score is not None:
                if (
                    abs(prev_score - score)
                    <= self.convergence_tol * max(abs(prev_score), 1e-12)
                ):
                    break
            prev_score = score
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
    可行。奖励是综合评分 (PIB + 均匀度)。
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
