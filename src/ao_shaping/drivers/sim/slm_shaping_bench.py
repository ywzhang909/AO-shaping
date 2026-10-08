"""闭环 SLM 远场整形仿真台架。

本模块提供**单块纯相位 SLM** 远场整形 (2f 傅里叶模型) 的自包含、无硬件仿真:

    gaussian input field  ->  SLM phase exp(1j*phi)  ->  Fraunhofer far field
    ->  intensity on the (zero-padded) camera grid

它只把 ``beam_backend`` 用于入射光束 (``gaussian_pupil`` / ``make_beam_config``);
焦平面则直接由一次补零 FFT 算出。它提供:

* 目标函数 / 质量指标 (PIB、efficiency、uniformity/CV、Strehl、overlap、zero-order
  fraction、structure similarity), 它们作为论文综述中所报告的*目标函数*;
* 参考性的*相位生成器* / 优化环路, 在仿真层面复现文献中的代表性方法
  (Gerchberg-Saxton、IFTA、可微分梯度下降、SPGD 无感知黑盒、Zernike 基 SPGD,
  以及一个解析的"纯幅度"单板目标)。

其目的是让单个可复现的脚本在同一个光学模型上跑遍多种整形方法, 并用完全相同的指标
比较它们。

本模块除 ``differentiable_shape`` 之外只用 NumPy, 而后者需要 torch。torch 的 import 是
惰性的 (在函数内部), 因此没装 torch 时本模块仍能干净地导入, 且 numpy 路径保持可用。

关于物理模型的说明
------------------
2f 傅里叶台架 (SLM 前焦面 → f 透镜 → 相机后焦面) 把 SLM 瞳孔场映射到它的**傅里叶
变换** (透镜相位 + 传播到 ``z=f``, 除一个全局相位外就等于夫琅禾费变换)。因此在 FFT
之前瞳孔要补零到 ``far_field_size``, 使焦平面被正确采样: 同尺寸的 FFT 是以
``w_far/dx_far = aperture/(pi*w0) = 3.5/pi = 1.11`` px 每腰半径对焦平面采样的 ——
这是模型中一个与 ``n_grid`` 无关的常数 —— 它会把经透镜相位调制的瞳孔混叠成一个点阵
(见 ``far_field_padding``)。0 级光斑位于全局强度最大值处 (用 ``argmax`` 定位, 从不
按几何定位)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ao_shaping.drivers.sim.beam_backend import (
    BeamSimConfig,
    gaussian_pupil,
    make_beam_config,
)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ShapingBenchConfig:
    """2f 傅里叶 SLM 整形台架的参数。

    Attributes:
        n_grid: SLM / 仿真网格边长 (正方形)。
        aperture_size: SLM 口径宽度, 单位米。
        wavelength: 激光波长, 单位米。
        focal_length: 透镜焦距, 单位米 (决定 2f 的尺度)。
        cn2: 大气湍流强度 (0 => 无湍流)。
        l_max: 湍流外尺度 (m)。
        l_min: 湍流内尺度 (m)。
        target_side_px: 目标方形/区域边长, 单位**远场 (相机) 像素**, 即补零后的
            远场网格 (``far_field_size``) 上的像素, 而不是瞳孔网格上的。
        target_kind: "square" | "circle" | "gaussian" | "annulus"。
        zero_order_margin_px: 报告 0 级功率占比时所用的保护半径 (远场 px); 纯相位
            SLM 会把未衍射的 0 级留在图样中心。
        far_field_padding: 夫琅禾费 FFT 之前施加到瞳孔上的整数补零因子, 即远场网格
            为 ``n_grid * far_field_padding``。不补零时, 同尺寸 FFT 是以
            ``w_far/dx_far = aperture/(pi*w0) = 3.5/pi = 1.11`` px 每腰半径对焦平面
            采样的, *与 n_grid 无关*, 无法表示一个聚焦光斑 (会混叠成一个点阵)。
            8x 给出 ~8.9 px/腰 (足以体现像差形态); 16x 与仓库内部标准一致
            (``SimPibSystem`` 与 ``generate_zernike_farfield_sim_report`` 都用
            512 -> 8192)。
        seed: 用于可复现性的 RNG 种子。
    """

    n_grid: int = 256
    aperture_size: float = 12e-3  # 12 mm SLM 口径
    wavelength: float = 532e-9  # 532 nm
    focal_length: float = 0.125  # 125 mm 透镜
    cn2: float = 0.0  # 0 => 干净 (无湍流) 基线
    l_max: float = 0.1
    l_min: float = 1e-3
    target_side_px: int = 60
    target_kind: str = "square"
    zero_order_margin_px: float = 10.0
    far_field_padding: int = 8
    seed: int = 0

    def __post_init__(self) -> None:
        if self.far_field_padding < 1:
            raise ValueError(
                f"far_field_padding must be >= 1, got {self.far_field_padding}"
            )

    @property
    def far_field_size(self) -> int:
        """补零后远场 (相机) 网格的边长。"""
        return self.n_grid * self.far_field_padding

    @property
    def far_field_pixel_size(self) -> float:
        """远场 (相机) 像元间距, 单位米。

        对补零到 ``M = n_grid * far_field_padding`` 的夫琅禾费 FFT, 焦平面间距为
        ``lambda*f/(M*dx_pupil) =
        lambda*f/(far_field_padding*aperture_size)`` —— 与 n_grid 无关。
        """
        return (
            self.wavelength
            * self.focal_length
            / (self.aperture_size * self.far_field_padding)
        )

    def make_beam_config(self) -> BeamSimConfig:
        return make_beam_config(
            n_grid=self.n_grid,
            aperture_size=self.aperture_size,
            wavelength=self.wavelength,
            cn2=self.cn2,
            l_max=self.l_max,
            l_min=self.l_min,
            propagation_distance=self.focal_length,
        )


# ---------------------------------------------------------------------------
# 结果容器
# ---------------------------------------------------------------------------
@dataclass
class ShapingResult:
    """台架上一次整形运行的产出。"""

    method: str
    phase: np.ndarray
    intensity: np.ndarray
    metrics: dict[str, float]
    n_iters: int
    history: list[dict[str, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 正向模型 (2f 傅里叶)
# ---------------------------------------------------------------------------
def _pad_centred(arr: np.ndarray, cfg: ShapingBenchConfig) -> np.ndarray:
    """把方形瞳孔网格数组居中补零到远场网格。

    瞳孔占据中央的 ``n_grid x n_grid`` 块; 补零区域就是 SLM 口径之外那片 (暗的)
    面积。
    """
    n = arr.shape[0]
    m = cfg.far_field_size
    if m <= n:
        return arr
    out = np.zeros((m, m), dtype=arr.dtype)
    start = (m - n) // 2
    out[start : start + n, start : start + n] = arr
    return out


def _fraunhofer_intensity(field: np.ndarray, cfg: ShapingBenchConfig) -> np.ndarray:
    """瞳孔场的补零夫琅禾费 (焦平面) 强度。

    2f 傅里叶台架把 SLM 瞳孔场映射到它的傅里叶变换 (透镜相位 + 传播到 ``z=f``, 除一个
    全局相位外恰好就是这个)。瞳孔被居中并在 FFT 之前**补零**到 ``far_field_size``,
    使焦平面被充分过采样; 不补零的同尺寸 FFT 只以 ~1.1 px 每腰半径对焦平面采样
    (一个与 ``n_grid`` 无关的模型常数), 并会混叠成一个点阵。

    变换约定: ``F = fftshift(fft2(ifftshift(f)))``, 使居中的瞳孔映射到居中的远场,
    与 ``SimPibSystem.far_field`` 以及仓库里的 ``beam_simulation._ft`` 辅助函数一致。

    Args:
        field: ``n_grid`` 网格上的瞳孔场 (复数)。
        cfg: 台架配置 (提供补零因子)。

    Returns:
        ``far_field_size`` 网格上的非负强度 (未归一化)。
    """
    padded = _pad_centred(field, cfg)
    focal = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))
    return np.abs(focal) ** 2


def forward_intensity(phase: np.ndarray, cfg: ShapingBenchConfig) -> np.ndarray:
    """把纯相位 SLM 图样传播到远场; 返回强度。

    高斯入射场乘以 ``exp(1j*phase)`` (纯相位 SLM), 再由一次**补零**的夫琅禾费 FFT
    传播到焦平面。返回 ``far_field_size`` 网格上的非负强度数组 (未归一化)。
    """
    beam_cfg = cfg.make_beam_config()
    field = gaussian_pupil(beam_cfg).astype(np.complex128) * np.exp(
        1j * np.asarray(phase)
    )
    return _fraunhofer_intensity(field, cfg)


def make_target(cfg: ShapingBenchConfig) -> np.ndarray:
    """在相机网格上构造归一化的远场目标图样 (sum=1)。

    目标以**远场 (补零后) 网格**为中心; 其尺寸是 ``cfg.target_side_px`` 个远场像素,
    使目标函数在尺度上稳定。
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


# ---------------------------------------------------------------------------
# 目标函数 / 质量指标
# ---------------------------------------------------------------------------
def _rolled_support(target: np.ndarray, center: tuple[int, int] | None) -> np.ndarray:
    """目标的布尔支撑域, 被平移使其中心落在 ``center`` 上。

    ``center`` 是 ``(col, row)`` (每个调用方都用的 ``np.unravel_index(...)[::-1]``
    约定)。行 (轴 0) 按*行*偏移平移、列 (轴 1) 按*列*偏移平移 —— 把这两个弄反会在
    离轴光束时把支撑域转置, 从而静默地让优化器去追一个已不再覆盖光束的支撑域。
    ``center`` 为 None 时支撑域原样使用 (网格居中)。调用方必须对每个指标都用*同一个*
    中心, 使桶始终跟着光束走 (与硬件闭环的 ``argmax`` 规则一致)。
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
    """桶内功率: 落在目标区域内的总功率占比。

    目标区域取自 ``target`` (它的支撑域)。若 ``center`` 以 ``(x, y)`` 给出, 目标会
    以它为中心重新居中 (目标跟着实测的 0 级 / 光束质心走), 这与硬件环路一致。
    """
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

    越低越好 (0 = 完全均匀)。若目标支撑域内没有功率则返回 +inf。支撑域被平移到
    ``center`` 的方式与 :func:`power_in_bucket` 完全一致, 因此 PIB 与 CV 描述的
    始终是同一块区域。
    """
    vals = intensity[_rolled_support(target, center)]
    if vals.size == 0 or vals.mean() <= 0:
        return float("inf")
    return float(vals.std() / vals.mean())


def strehl(intensity: np.ndarray, target: np.ndarray) -> float:
    """仿真图样与目标图样之间的归一化重叠 (类 Strehl)。

    用两张能量分布图样的余弦相似度 (归一化内积); 1.0 = 完全匹配。
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
        intensity: (补零后) 相机网格上的远场强度。
        center: 0 级位置, 以 ``(col, row)`` 给出 —— 即每个调用方都用的
            ``np.unravel_index(...)[::-1]`` 约定。
        zero_order_margin_px: 保护半径, 单位远场 (相机) 像素。默认值约等于一个
            Airy 半径 (8x 补零下为 10 px); 换其他补零倍数时要按
            ``cfg.zero_order_margin_px`` 缩放。
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

    ``center`` 是目标框中心: ``None`` (默认) 表示使用目标自身那个网格居中的支撑域,
    这对本**居中**的仿真台架是正确的。**不要**把它默认成强度的 ``argmax``: 对类
    散斑的场, 在 ~1e-3 的模型扰动下全局最大值会在几乎相等的晶粒之间跳动, 于是按
    argmax 平移的框会让 PIB/CV 不连续, 并让优化器去追一个并不覆盖光束的框。0 级保护
    仍以强度峰值为中心 (它测的就是这个)。
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
    """综合评分公式, 由 numpy 与 torch 两条目标函数路径共用。

    均匀性按 ``1 / (1 + cv)`` 计分, 而不是裁剪过的线性映射。裁剪项
    ``1 - min(cv/cv_ref, 1)`` 对任何可达的平顶 CV 都饱和到零 (目标框内的 CV 并非
    尺度无关, 所以不存在一个通用的 ``cv_ref``), 那会静默地把目标函数退化成纯粹的桶内
    能量, 从而奖励把光集中而不是把光抹平。``1/(1+cv)`` 是单调且永不饱和的, 因此会
    奖励每一分均匀性改善, 且没有阈值需要调。
    """
    return w_pib * pib + w_unif * (1.0 / (1.0 + cv))


def composite_score(
    m: dict[str, float],
    *,
    w_pib: float = 0.5,
    w_unif: float = 0.5,
) -> float:
    """需要*最大化*的综合标量评分 (SPGD / 可微分的奖励)。

    把桶内能量与一项均匀性项结合起来, 使优化器不会只顾把能量灌进框里却任其参差
    (这是仅用 CV 作目标的一个已知失败模式)。
    """
    pib = m.get("PIB", 0.0)
    cv = m.get("CV", float("inf"))
    return composite_from_pib_cv(pib, cv, w_pib=w_pib, w_unif=w_unif)


# ---------------------------------------------------------------------------
# 方法 1: Gerchberg-Saxton (幅度 <-> 相位约束)
# ---------------------------------------------------------------------------
def gs_shape(
    cfg: ShapingBenchConfig,
    *,
    n_iters: int = 200,
    relax: float = 1.0,
    seed: int | None = None,
    verbose: bool = False,
    return_phase_only: bool = True,
    base_phase: np.ndarray | None = None,
) -> ShapingResult:
    """单块纯相位 SLM 的 Gerchberg-Saxton 整形。

    幅度约束 = 远场中的 sqrt(target), 相位约束 = 任意 (SLM 平面)。最终相位就是 SLM
    图样。

    GS 跑在**补零后的远场网格**上 (瞳孔被补零到 ``far_field_size``), 因此正/反向那一对
    FFT 是精确且采样正确的变换, 返回的强度与 ``make_target`` 的网格一致。瞳孔支撑域约束
    只保留中央的 ``n_grid`` 块 (光只存在于 SLM 口径之内)。

    Args:
        base_phase: 固定的瞳孔相位 (raw 弧度, 形状 ``(n_grid, n_grid)``), GS 解会被加到
            它之上, 例如一个实测像差的预矫正 ``-Z_est``。它在*瞳孔约束内部*被施加,
            因此 GS 整形的是已矫正的瞳孔, 而返回的相位是总相位
            ``base_phase + delta``。
    """
    beam_cfg = cfg.make_beam_config()
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    target = make_target(cfg)
    amp_target = np.sqrt(target)
    amp_slm = _pad_centred(np.abs(gaussian_pupil(beam_cfg)), cfg).astype(np.float64)
    if base_phase is None:
        base = np.zeros(amp_slm.shape, dtype=np.float64)
    else:
        base = _pad_centred(np.asarray(base_phase, dtype=np.float64), cfg)
    # 初始场: SLM 平面上的高斯入射
    field = amp_slm * np.exp(1j * (base + rng.normal(0, 0.1, size=amp_slm.shape)))

    history = []
    for i in range(n_iters):
        # SLM → 远场
        ff = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
        # 远场中的幅度约束
        ff = ff * (amp_target / (np.abs(ff) + 1e-12)) * relax
        # 向目标幅度回松
        ff = ff * (1 - relax) + amp_target * np.exp(1j * np.angle(ff)) * relax
        # 远场 → SLM 平面
        field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(ff)))
        # SLM 平面中的纯相位约束 (保留高斯幅度, 设定相位)
        field = amp_slm * np.exp(1j * np.angle(field))
        if verbose and i % 20 == 0:
            inten = _fraunhofer_intensity(field, cfg)
            inten = inten / inten.max()
            center = np.unravel_index(np.argmax(inten), inten.shape)[::-1]
            m = compute_bench_metrics(inten, target, center=center)
            history.append({"iter": i, **m})
    inten = _fraunhofer_intensity(field, cfg)
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
# 方法 2: 可微分梯度下降 (torch, 可选)
# ---------------------------------------------------------------------------
def differentiable_shape(
    cfg: ShapingBenchConfig,
    *,
    n_iters: int = 300,
    lr: float = 0.05,
    seed: int | None = None,
) -> ShapingResult:
    """可微分远场整形: 对 SLM 相位做梯度下降。

    Loss = 1 - overlap(target, sim) + lambda * (1 - PIB)。需要 torch, 它是惰性导入的,
    因此本模块其余部分没有它也能工作。
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("differentiable_shape requires torch") from exc

    beam_cfg = cfg.make_beam_config()
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    target = make_target(cfg)
    tgt = torch.as_tensor(target, dtype=torch.float64).double()
    n = cfg.n_grid
    m = cfg.far_field_size
    pad = (m - n) // 2
    init_phase = torch.as_tensor(
        rng.normal(0, 0.1, size=(n, n)), dtype=torch.float64
    ).double()
    param = torch.nn.Parameter(init_phase)
    opt = torch.optim.Adam([param], lr=lr)

    in_amp = torch.as_tensor(
        np.abs(gaussian_pupil(beam_cfg)), dtype=torch.float64
    ).double()

    def intensity(phase: Any) -> Any:
        field = in_amp * torch.exp(1j * phase)
        if m > n:
            padded = torch.zeros((m, m), dtype=torch.complex128)
            padded[pad : pad + n, pad : pad + n] = field
            field = padded
        focal = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(field)))
        return focal.real**2 + focal.imag**2

    history = []
    for i in range(n_iters):
        opt.zero_grad()
        inten = intensity(param)
        inten = inten / inten.sum()
        # 目标重叠度 (类余弦, 归一化)
        a = inten - inten.mean()
        b = tgt - tgt.mean()
        overlap = (a * b).sum() / (a.norm() * b.norm() + 1e-12)
        # PIB: 目标支撑域内的功率
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
    cfg: ShapingBenchConfig,
    *,
    n_iters: int = 600,
    delta: float = 0.1,
    lr: float = 0.02,
    seed: int | None = None,
    dim: int | None = None,
) -> ShapingResult:
    """低维 (Zernike / 粗) 相位基上的无感知 SPGD。

    SLM 相位被参数化为少量 freeform 系数 (一个粗网格), 使随机梯度搜索在仿真中可行。
    奖励是综合评分 (PIB + 均匀性)。
    """
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    target = make_target(cfg)
    d = dim or 16  # 16x16 freeform 基
    n_par = d * d
    phase_flat = rng.normal(0, 0.05, size=n_par)

    def upsample(vec: np.ndarray) -> np.ndarray:
        """把 d×d 的 freeform 系数网格映射为完整的 SLM 相位 (kron 上采样)。"""
        ph = np.kron(vec.reshape(d, d), np.ones((cfg.n_grid // d, cfg.n_grid // d)))
        if ph.shape != (cfg.n_grid, cfg.n_grid):
            ph = ph[: cfg.n_grid, : cfg.n_grid]
        return ph

    def eval_score(vec: np.ndarray) -> float:
        inten = forward_intensity(upsample(vec), cfg)
        inten = inten / inten.max()
        center = np.unravel_index(np.argmax(inten), inten.shape)[::-1]
        m = compute_bench_metrics(inten, target, center=center)
        return composite_score(m)

    score = eval_score(phase_flat)
    g = np.zeros(n_par)
    history = [{"iter": 0, "score": score, "PIB": 0, "CV": float("inf")}]
    for i in range(1, n_iters):
        sgn = rng.choice([-1, 1], size=n_par)
        cand = phase_flat + delta * sgn
        s2 = eval_score(cand)
        g = 0.95 * g + 0.05 * sgn * (s2 - score)
        phase_flat = phase_flat + lr * g
        score = eval_score(phase_flat)
        if i % 50 == 0:
            inten = forward_intensity(upsample(phase_flat), cfg)
            inten = inten / inten.max()
            center = np.unravel_index(np.argmax(inten), inten.shape)[::-1]
            m = compute_bench_metrics(inten, target, center=center)
            history.append({"iter": i, "score": score, "PIB": m["PIB"], "CV": m["CV"]})
    ph = upsample(phase_flat)
    inten = forward_intensity(ph, cfg)
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
def analytic_amplitude_target(cfg: ShapingBenchConfig) -> ShapingResult:
    """平凡基线: 目标本身 (幅度整形, 无相位)。

    作为"一块纯幅度 (非纯相位) SLM 会产出什么"的参照 —— 它给出了纯相位器件可达质量
    的上界。
    """
    target = make_target(cfg)
    metrics = compute_bench_metrics(target, target)
    return ShapingResult(
        method="analytic_amplitude_baseline",
        phase=np.zeros((cfg.n_grid, cfg.n_grid)),
        intensity=target,
        metrics=metrics,
        n_iters=0,
    )


__all__ = [
    "ShapingBenchConfig",
    "ShapingResult",
    "forward_intensity",
    "make_target",
    "compute_bench_metrics",
    "composite_score",
    "composite_from_pib_cv",
    "gs_shape",
    "differentiable_shape",
    "spgd_shape",
    "analytic_amplitude_target",
]
