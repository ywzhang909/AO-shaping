"""把 Zernike 像差系数拟合到实测远场强度上。

迭代式 model-in-the-loop SLM 光束整形管线的 "步骤 A": 在 SLM 相位图**固定**
的前提下, 拟合 Zernike 像差系数 ``c``, 使可微的 FFT 正向模型能复现实测到的
远场强度。数字孪生随后用更新后的相位重新测量, 循环再一次次地重新拟合。

模型
-----
在 ``region x region`` 的瞳面网格上::

    U        = A * exp(i * (phi_slm + sum_j c_j * Z_j))
    I_model  = |fftshift(fft2(ifftshift(U), norm="ortho"))|**2

FFT 约定与数字孪生 (:meth:`SimFourierGSNetEnv.render_intensity`) 所用的
一致, 所以在使用原生高斯光束 (``w0 = 250 * region / 512``) 时, 本模型的
远场在同一网格上与孪生逐位相同。

``A`` 默认为那份原生高斯, 可按次运行覆盖。``phi_slm`` 以**未包裹的原始
弧度**消费: 本仓库的相位生成器都返回原始弧度, 唯一的 ``mod 2*pi`` 包裹
点在 SLM 驱动里, 所以这里不做任何包裹。

损失与梯度
-----------
目标函数镜像 :meth:`DifferentiableBeamOptimizer._intensity_loss`: 对实测远场
做*峰值归一化*的 MSE, ``i / (i.max() + 1e-8)``。之所以必须用峰值而非能量
归一化, 是因为 FFT 把能量集中到少数像素上, 那会在 float32 里下溢。

这一步在 ``differentiable_beam.py`` 意义下是**测量锚定**的: 损失的*目标*
与*归一化尺度*都取自实测图像, 因此交给模型 Jacobian ``dI_model/dc`` 的上游
强度梯度是锚定在真实数据上, 而不是模型自己那张图上。

.. note::
   在 :class:`DifferentiableBeamOptimizer` 里锚点与目标是*不同*的两张图
   (硬件帧 vs 期望图形), 所以可以在测量处求损失得到 ``dL/dI``, 再用
   ``i_sim.backward(gradient=i_meas.grad)`` 推过模型 Jacobian。而这里测量
   *就是*拟合目标, 于是那种字面写法会在目标自身处求值, 给出恒为零的上游
   梯度。因此锚点改由归一化尺度承担 (``peak_anchor``, 取自测量),
   残差则取在模型图像上 —— 这在代数上是同一个锚定梯度, 且非退化。

Attributes:
    n_coefficients: 拟合的 Zernike 系数个数。
    n_orders: Zernike 基的最大径向阶数。
    region: 方形瞳面 / 远场网格的边长。
    radius: Zernike 瞳面半径 (像素)。
    device: 优化所在的 torch 设备。
    coefficient_tensor: 实时的系数参数张量 (requires_grad)。
    coefficients: 当前系数的 numpy 副本 (已 detach), 单位弧度。
    loss_history: 每一步优化的损失值。
    last_loss: 最后一步的损失, 首次更新之前为 ``None``。
    converged: 是否满足提前停止条件。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import math

import numpy as np
import numpy.typing as npt
from loguru import logger
from scipy import ndimage

from ao_shaping.algorithm.signal_processing.iterative_base import IterativeOptimizer
from ao_shaping.utils.wavefront.calibration_terms import (
    FAMILY_NAMES,
    PupilAffine,
    TravelScale,
    apply_pupil_affine,
    apply_travel_scale,
)
from ao_shaping.utils.wavefront.model_error_terms import ModelErrorTerms
from ao_shaping.utils.wavefront.zernike_calc import (
    ZernikeGenerator,
    calc_n_zernike_terms,
)

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    import torch

#: 数字孪生的参考 region (``SimFourierGSNetEnv.BeamParams``)。
TWIN_REGION = 512
#: 数字孪生的原生高斯束腰, 以孪生 region 的像素计。
TWIN_W0 = 250.0
#: 加到每个峰值归一化除数上的保护量, 与
#: :meth:`DifferentiableBeamOptimizer._intensity_loss` 保持一致。
PEAK_EPS = 1e-8
#: 平台期检测: 当最优损失连续 ``PLATEAU_PATIENCE`` 步的改善量都不超过
#: ``RELATIVE_TOL`` (或 ``ABSOLUTE_FLOOR``) 时就停止。
RELATIVE_TOL = 1e-5
ABSOLUTE_FLOOR = 1e-12
PLATEAU_PATIENCE = 30
#: 无论平台期启发式怎么说, 在这么多步之前都不停止。
MIN_ITERATIONS = 5
#: 傅里叶域损失的权重 α。``0`` = 关闭 (默认), 此时正向模型与损失的输出
#: 与引入本特性之前**逐位相同**。论文用的值是 7e-4 (仿真) / 5e-4 / 1e-3。
DEFAULT_FOURIER_WEIGHT = 0.0
#: 图像域损失的种类: ``"mse"`` (历史行为) 或 ``"l1"`` (论文用的那一种)。
LOSS_KINDS = ("mse", "l1")
#: 自标定参数的 Adam 学习率 (像差系数用 :attr:`lr`, 两者相差 1-2 个量级)。
DEFAULT_CALIBRATION_LR = 1e-3

#: Zernike 基伪逆的进程内缓存。键是 (系数个数, 网格边长, 瞳面半径), 值是
#: ``(K, N)`` 的 float64 伪逆。伪逆只依赖基本身, 而基只依赖这三个数 ——
#: 每一轮 MIL 循环都会重建优化器, 不缓存的话每轮都要重算一次 SVD。
_PROJECTOR_CACHE: dict[tuple[int, int, int], npt.NDArray[np.float64]] = {}


def _projector_for_basis(
    basis: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """返回 Zernike 基的最小二乘伪逆 ``(K, N)``。

    Zernike 模式在**离散**网格上并非严格正交 (实测 Gram 矩阵的非对角元最大
    约为对角元的 5%), 所以投影必须用伪逆而不是简单的 ``basis @ phase``。

    Args:
        basis: Zernike 基, 形状 ``(K, R, R)``。

    Returns:
        ``(K, N)`` 的 float64 伪逆, ``N = R * R``。
    """
    n_coeffs, region, radius = basis.shape
    key = (int(n_coeffs), int(region), int(radius))
    cached = _PROJECTOR_CACHE.get(key)
    if cached is not None:
        return cached
    flat = np.ascontiguousarray(basis.reshape(int(n_coeffs), -1))
    # ``rcond`` 取 1e-10: 基的条件数是两位数, 而 numpy 的默认 (1e-15) 会把
    # 数值噪声当成有效方向, 把伪逆放大成噪声。
    projector = np.linalg.pinv(flat.T, rcond=1e-10)
    _PROJECTOR_CACHE[key] = projector
    return projector



def _torch():
    """返回 ``torch`` 模块; 缺失时抛出 :class:`ImportError`。

    Returns:
        ``torch`` 顶层模块。

    Raises:
        ImportError: 若未安装 PyTorch。
    """
    try:
        import torch as _t
    except ImportError:
        raise ImportError(
            "zernike_coefficient_optimizer requires PyTorch. "
            "Install with: uv sync --group ml"
        ) from None
    return _t


def _numpy_dtype(dtype_name: str) -> Any:
    """把工作精度名称映射到对应的 numpy dtype。"""
    return np.float64 if dtype_name == "float64" else np.float32


# ---------------------------------------------------------------------------
# 误差项 (命令相位 → 实际相位 的 LUT 非线性 + 焦面装调误差)
#
# 这些函数是 MIL 正向模型的**唯一**误差实现: Step A 的预测 (拟合) 与
# Step B 的合成都调用它们, 所以标定与合成同源。
#
# 作用位置 (与孪生 bench_twin 的正向链逐项对应):
#   1. ``apply_lut_nonlinearity``  —— 瞳面相位**之前** (命令 → 实际相位)
#   2. 倾斜                      —— 焦面复场上 (图样平移 f · tilt)
#   3. 离焦                      —— 焦面复场上 (角谱传递函数), 在 |·|² 之前
# ---------------------------------------------------------------------------


def apply_lut_nonlinearity(
    phase: torch.Tensor,
    gamma: float,
    phi_max_rad: float = 2.0 * math.pi,
) -> torch.Tensor:
    """命令相位 → 实际相位 (SLM 的 LUT 非线性)。

    语义与孪生 :meth:`~ao_shaping.drivers.sim.bench_twin.BenchTwin.apply_lut`
    一致: ``actual = phi_max · (wrap(cmd) / 2π) ** gamma``, 包裹用
    ``torch.remainder`` (取值落在 ``[0, 2π)``)。

    ``gamma == 1`` 时**原样返回**输入: 2π 周期器件上包裹不改变强度, 但会
    改变最后几个比特, 而本模型引入误差项之前的输出必须一字节不差, 所以
    恒等映射走的是「不包裹」的捷径。

    **灰度量化在可微路径里省略**: 器件把相位量化到 10 bit 灰度 (2π = 993 级),
    量化误差 ≤ 一个量化级 = ``2π/993 ≈ 6.3e-3 rad`` (硬件) 或
    ``2π/1023 ≈ 6.1e-3 rad`` (孪生的 10 bit 默认)。它是**不可微**的阶梯
    (取整), 放进拟合路径只会给梯度引入量化噪声; 而它对远场强度的相对影响
    在光斑型图样上约为 ``2·σ_φ`` (σ_φ = 一个量化级 / √12), 即 1e-3 量级 ——
    比 γ 本身带来的偏差小两个数量级, 所以把 γ 标准到 0.1 以内时它不再是
    限制项。

    Args:
        phase: 命令相位张量 (raw 未包裹弧度)。
        gamma: LUT 的 Gamma 指数 (``1`` = 理想线性)。
        phi_max_rad: 满灰度对应的相位 (rad)。

    Returns:
        实际相位张量, 与输入同形同 dtype。
    """
    torch = _torch()
    if float(gamma) == 1.0:
        return phase
    two_pi = 2.0 * math.pi
    wrapped = torch.remainder(phase, two_pi)
    return float(phi_max_rad) * (wrapped / two_pi) ** float(gamma)


def apply_focal_defocus(
    field: torch.Tensor,
    defocus_m: float,
    wavelength_m: float,
    far_field_pitch_m: float,
) -> torch.Tensor:
    """相机离焦: 角谱传递函数作用在**焦面复场**上。

    ``H = exp(i·k·Δz·sqrt(1 - (λf_x)² - (λf_y)²))``, 其中 ``f_x = m_x / (N·p)``
    是焦面坐标的**共轭频率** (cycles/m, ``m_x`` 是整数频移索引); 倏逝分量
    (``λf > 1``) 置 0, **不得出 NaN** —— 负的辐角先被夹到 0 再开方, 于是
    ``exp`` 的那一支恒为 1, 由 ``where`` 选成 0。

    该写法与「瞳面乘二次相位」**逐位等价** (焦面复场的频谱就是瞳面, 所以
    离焦在频域就是给瞳面加二次相位)。⚠️ 因此它与 MIL 模型里的 **Noll 4
    (离焦) 系数是同一个自由度**: 不要既加「面板 Zernike 离焦」又加「相机
    离焦」去双重表达 (Step A 拟合出的系数会把它吸收掉)。

    ``Δz == 0`` 是恒等 (逐位相同)。

    Args:
        field: 焦面复场 (以 ``fftshift`` 口径存放, 即光学轴在中心)。
        defocus_m: 离焦量 (m, 正负皆可)。
        wavelength_m: 工作波长 (m)。
        far_field_pitch_m: 焦面采样间距 (m)。

    Returns:
        离焦后的焦面复场。
    """
    torch = _torch()
    if float(defocus_m) == 0.0:
        return field
    size = int(field.shape[-1])
    lam = float(wavelength_m)
    k_wave = 2.0 * math.pi / lam
    # ``dtype`` matters: ``torch.fft.fftfreq`` defaults to the *default* dtype
    # (float32), and the transfer function's argument is ``1 - (λf)²`` — a
    # float32 frequency grid perturbs that by ~1e-7, which is 3% of the whole
    # (λf)² term and shows up as a visible change in the defocused spot.
    freq = torch.fft.fftfreq(size, dtype=field.real.dtype) / float(
        far_field_pitch_m
    )  # cycles/m
    arg = 1.0 - (lam * freq[None, :]) ** 2 - (lam * freq[:, None]) ** 2
    propagating = arg > 0.0
    safe_arg = torch.clamp(arg, min=0.0)
    transfer = torch.where(
        propagating,
        torch.exp(1j * k_wave * float(defocus_m) * torch.sqrt(safe_arg)),
        torch.zeros_like(arg, dtype=torch.complex128),
    )
    un = torch.fft.ifftshift(field)
    propagated = torch.fft.ifft2(
        torch.fft.fft2(un, norm="ortho") * transfer.to(field.dtype), norm="ortho"
    )
    return torch.fft.fftshift(propagated)


def apply_focal_tilt(
    field: torch.Tensor,
    tilt_x_rad: float,
    tilt_y_rad: float,
    focal_length_m: float,
    far_field_pitch_m: float,
) -> torch.Tensor:
    """光路倾斜: 焦面图样平移 ``f · tilt``。

    物理上, 光路倾斜 = 入射波相对透镜倾斜, 其**精确**效果是把远场图样平移
    ``f · tilt``。本实现用傅里叶移位精确完成该平移, 与孪生
    :meth:`~ao_shaping.drivers.sim.bench_twin.BenchTwin.apply_tilt` 逐位一致。

    ⚠️ **实现方向**: 「在焦面上乘一个线性相位」**不是**等价写法 —— 按位移
    定理它等价于**平移瞳面**, 远场强度完全不变 (孪生实测 0 级光斑不动)。
    要让远场图样平移, 可以在**瞳面**加线性相位斜坡 (等价), 也可以像这里
    一样直接在焦面上做精确平移。本实现选后者, 因为它与孪生的数值结果
    逐位对齐 (孪生也用焦面平移)。

    Args:
        field: 焦面复场 (以 ``fftshift`` 口径存放)。
        tilt_x_rad: x 方向的倾斜 (rad)。
        tilt_y_rad: y 方向的倾斜 (rad)。
        focal_length_m: 2f 透镜焦距 (m)。
        far_field_pitch_m: 焦面采样间距 (m)。

    Returns:
        平移后的焦面复场。
    """
    torch = _torch()
    dx = float(focal_length_m) * float(tilt_x_rad)
    dy = float(focal_length_m) * float(tilt_y_rad)
    if dx == 0.0 and dy == 0.0:
        return field
    size = int(field.shape[-1])
    freq = torch.fft.fftfreq(size, dtype=field.real.dtype) * size  # 整数频移
    shift = torch.exp(
        -2j
        * math.pi
        * (freq[:, None] * dy + freq[None, :] * dx)
        / (size * float(far_field_pitch_m))
    )
    un = torch.fft.ifftshift(field)
    shifted = torch.fft.ifft2(
        torch.fft.fft2(un, norm="ortho") * shift.to(field.dtype), norm="ortho"
    )
    return torch.fft.fftshift(shifted)


def apply_model_error_terms(field: torch.Tensor, terms: ModelErrorTerms) -> torch.Tensor:
    """把误差项作用在**焦面复场**上 (离焦与倾斜)。

    Args:
        field: 焦面复场。
        terms: 误差项容器。

    Returns:
        施加误差后的焦面复场; 误差全零时**原样返回** (逐位不变)。
    """
    if terms is None or terms.is_identity:
        return field
    terms.validate_focal_geometry()
    result = apply_focal_defocus(
        field,
        float(terms.camera_defocus_m),
        float(terms.wavelength_m),
        float(terms.far_field_pitch_m),
    )
    return apply_focal_tilt(
        result,
        float(terms.tilt_x_rad),
        float(terms.tilt_y_rad),
        float(terms.focal_length_m),
        float(terms.far_field_pitch_m),
    )


def _to_tensor(
    x: npt.NDArray[np.floating],
    device: torch.device,
    dtype: str = "float32",
) -> torch.Tensor:
    """把实数二维 numpy 数组转成 ``device`` 上的 torch 张量。

    Args:
        x: 源数组。
        device: 目标 torch 设备。
        dtype: 工作精度, ``"float32"`` 或 ``"float64"``。
    """
    torch = _torch()
    arr = np.ascontiguousarray(x, dtype=_numpy_dtype(dtype))
    return torch.from_numpy(arr).to(device)


@dataclass
class ZernikeCoefficientResult:
    """一次 :meth:`ZernikeCoefficientOptimizer.run` 调用的结果。

    Attributes:
        coefficients: 最终拟合出的系数, 单位弧度, 形状
            ``(calc_n_zernike_terms(n_orders),)``, 按 Noll 序。
        history: 逐迭代记录; 始终含 ``"loss"``。
        iterations: 实际执行的优化步数。
        converged: 是否在预算耗尽之前就满足了停止条件。
    """

    coefficients: npt.NDArray[np.floating]
    history: dict[str, list] = field(default_factory=dict)
    iterations: int = 0
    converged: bool = False


class ZernikeCoefficientOptimizer(IterativeOptimizer):
    """把 Zernike 像差系数拟合到实测远场上。

    该优化器是有状态的: 构造一次, 然后反复调用 :meth:`update` 以手动控制,
    或调用 :meth:`run` 跑完整循环。

    数字孪生契约
    -------------
    远场复现 ``SimFourierGSNetEnv.render_intensity`` 的核心约定,
    即 ``|fftshift(fft2(ifftshift(U), norm="ortho"))|**2`` 且
    ``U = A * exp(1j * patch)``, 其中包含两个容易漏掉、而本类刻意匹配的细节:

    * **被掩膜的是 patch, 不是像差。** 孪生算的是
      ``nan_to_num(phi_slm + aberration, nan=0.0)``, 所以在圆形 Zernike
      瞳面之外被压平的是*两者之和*。若把 SLM 相位施加到整个画面,
      模型与孪生的差异就会达到 O(100%)。
    * **振幅仍然作用在瞳面之外**, 因为孪生只掩膜相位, 从不掩膜振幅。

    用 ``dtype="float64"`` 时, 远场与 float64 孪生一致到 ~1 ulp
    (相对 2.7e-16); float32 默认值一致到相对 ~7e-8, 这属于 float32
    舍入而非建模差异。

    损失
    ----
    记录的损失是模型远场与实测远场之间的峰值归一化 MSE。
    ``differentiable_beam.py`` 里那种字面的两阶段 "测量锚定" 技巧
    (把损失反传进实测强度) 在这里是退化的, 因为实测图像对系数而言是常量
    -- 在目标处 ``dL/dI_meas == 0``, 不贡献任何梯度。因此本类改用实测
    *峰值* 作为固定的归一化锚点, 这等价于标准的峰值归一化 MSE, 同时让
    目标与归一化因子在 ``c`` 上都是常量, 梯度于是只经由模型强度流动。

    学习率
    ------
    用高阶基 (``n_orders=10``, 66 个系数) 时, 目标函数存在一个浅的局部盆地,
    它能把损失吸收掉却让系数仍然是错的。在 ``region=64`` 网格上以三模式
    真值实测: 同一问题在 ``lr in {0.02, 0.03, 0.1}`` 下收敛到 loss ~9e-13
    且 ``|c - c_true| < 1e-4``, 而在 ``lr in {0.05, 0.08}`` 下停在
    loss ~6.9e-6 且 ``|c - c_true|`` ~0.49。归一化强度的 Jacobian 在该工作点
    条件良好 (条件数 ~5.6, ``sigma_min`` ~12.5), 所以这个盆地是优化伪影
    而不是可辨识性上限 -- 从同一起点直接做最小二乘求解能恢复 ``c_true``。
    因此使用 10 阶基的调用方应当校验*系数*误差, 而不只是看损失:
    单看低损失并不能证明系数被找到了。
    """

    def __init__(
        self,
        n_orders: int = 10,
        region: int = 64,
        radius: int | None = None,
        initial_coefficients: npt.NDArray[np.floating] | None = None,
        lr: float = 0.1,
        max_iterations: int = 500,
        device: str | None = None,
        seed: int | None = None,
        dtype: str = "float32",
        far_field_size: int | None = None,
        frozen_modes: tuple[int, ...] = (),
        pupil_oversampling: int = 1,
        error_terms: ModelErrorTerms | None = None,
        fourier_weight: float = DEFAULT_FOURIER_WEIGHT,
        loss_kind: str = "mse",
        affine_params: PupilAffine | None = None,
        affine_lr: float = DEFAULT_CALIBRATION_LR,
        travel_gammas: tuple[float, ...] | None = None,
        travel_lr: float | None = None,
    ) -> None:
        """初始化优化器并校验所有输入。

        Args:
            n_orders: Zernike 基的最大径向阶数, 必须 >= 1。
                该基按 Noll 1976 序含 ``calc_n_zernike_terms(n_orders)`` 个模式
                (Noll 4 = 离焦, Noll 5 = 散光)。
            region: 方形网格的边长。必须 >= 16 且为偶数。
            radius: Zernike 瞳面半径 (像素)。默认为 ``region // 2``。
                瞳面之外的模式求值为零。
            initial_coefficients: 可选的起始系数, 单位弧度, 形状
                ``(n_coefficients,)``。默认为全零; 给了 ``seed`` 时则取
                一个带种子的小随机抽样。
            lr: Adam 学习率。必须 > 0。
            max_iterations: :meth:`run` 的迭代预算。必须 >= 1。
            device: torch 设备字符串。若为 ``None``, 自动检测: CUDA 可用
                时使用 ``"cuda"``, 否则 ``"cpu"`` —— 与
                ``differentiable_shaping`` 保持一致, 使 model-in-loop 在有 GPU
                时不必显式传参即可加速。
            seed: 随机初始系数的可选 RNG 种子。
            dtype: 工作精度, ``"float32"`` (默认) 或 ``"float64"``。
                数字孪生以 float64 计算远场, 所以 ``"float64"`` 才是让本模型
                与 ``SimFourierGSNetEnv`` 逐位相同的精度; float32 保持默认的
                快速, 并符合本仓库 differentiable-beam 的约定, 与孪生一致到
                float32 舍入 (相对 ~1e-7)。
            far_field_size: FFT 之前对瞳面光场做零填充的可选尺寸。
                ``None`` (默认) 保持在 ``region`` 上变换的历史行为, 这正是
                数字孪生原生 1:1 模式的做法。传一个更大的 2 的幂以便对*同一*
                视场做更精细的采样 -- 与真实相机对比时必须这样做, 因为相机
                像素远小于未填充时的 ``lambda * f / (region * d_slm)`` 间距。
                给出时必须 >= ``region``。
            frozen_modes: 在整个拟合过程中强制保持为零的 Noll 索引 (1 起)。
                默认为空, 即所有模式都可动。当拟合到实测远场强度时传
                ``(1, 2, 3)``: piston 与 tilt 对 ``|E|**2`` 不可见
                (见 :meth:`_apply_mode_mask`), 把它们放开会让拟合把无法建模
                的残余吸收到退化方向上。
            error_terms: 可选的误差项容器 (命令相位 → 实际相位 的 LUT 非线性
                + 相机离焦 + 光路倾斜)。``None`` (默认) = 理想系统, 此时
                正向模型的输出与引入本参数之前**逐位相同**。几何常数
                (波长 / 焦距 / 焦面节距) 由误差项容器自己携带。
            fourier_weight: 傅里叶域 L1 项的权重 α。``0`` (默认) = 关闭,
                此时损失与正向模型的输出与引入本参数之前**逐位相同**。
                论文用的是 7e-4 (仿真) / 5e-4 / 1e-3, 且论文的敏感性扫描
                显示结果对 α 不敏感 (1e-5…1e-3 基本平坦)。
            loss_kind: 图像域损失的种类。``"mse"`` (默认) = 历史的峰值归一化
                MSE; ``"l1"`` = 论文用的 L1。**不要**为了跟论文一致就顺手改
                默认值: MSE 与 L1 的收敛行为不同, 换过去会改变既有结果。
            affine_params: 可选的瞳面仿射自标定量 τ_θ (6 个标量)。
                ``None`` (默认) = 关闭, 此时正向模型的输出与引入本参数之前
                **逐位相同**。``PupilAffine()`` = 单位变换 (也是恒等)。
            affine_lr: τ_θ 的 Adam 学习率。论文给的是 1e-6…1e-3, 比像差
                系数 (1e-2) 小 1-2 个量级。
            travel_gammas: 可选的逐模式族行程缩放 γ (长度 = 7, 见
                :data:`~ao_shaping.utils.wavefront.calibration_terms.\
FAMILY_NAMES`)。``None`` (默认) = 关闭, 此时正向模型的输出与引入本参数
                之前**逐位相同**。全零 = 无缩放 (也是恒等)。
            travel_lr: γ 的 Adam 学习率。``None`` (默认) = 与
                :attr:`affine_lr` 相同 (论文就是这么设的: 行程 γ 用 θ 的
                学习率)。

        Raises:
            ValueError: 任一参数越界, 或任一数组形状不对时。
        """
        torch = _torch()
        super().__init__(max_iterations=max_iterations)

        if not isinstance(n_orders, int) or isinstance(n_orders, bool) or n_orders < 1:
            raise ValueError(f"n_orders must be an integer >= 1, got {n_orders!r}")
        if not isinstance(region, int) or isinstance(region, bool):
            raise ValueError(f"region must be an integer, got {region!r}")
        if region < 16 or region % 2 != 0:
            raise ValueError(f"region must be >= 16 and even, got {region!r}")
        if radius is None:
            radius = region // 2
        elif not isinstance(radius, int) or isinstance(radius, bool) or radius < 1:
            raise ValueError(f"radius must be None or an integer >= 1, got {radius!r}")
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError(f"lr must be a positive finite float, got {lr!r}")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise ValueError(f"seed must be None or an integer, got {seed!r}")
        if dtype not in ("float32", "float64"):
            raise ValueError(f"dtype must be 'float32' or 'float64', got {dtype!r}")
        if far_field_size is None:
            far_field_size = int(region)
        elif (
            not isinstance(far_field_size, int)
            or isinstance(far_field_size, bool)
            or far_field_size < region
        ):
            raise ValueError(
                f"far_field_size must be None or an integer >= region "
                f"({region}), got {far_field_size!r}"
            )
        self._far_field_size = int(far_field_size)
        if (
            not isinstance(pupil_oversampling, int)
            or isinstance(pupil_oversampling, bool)
            or pupil_oversampling < 1
        ):
            raise ValueError("pupil_oversampling must be a positive integer")
        self._pupil_oversampling = pupil_oversampling
        self._error_terms = (
            error_terms if error_terms is not None else ModelErrorTerms()
        )
        if self._error_terms.needs_focal_geometry:
            # Fail at construction: a defocus or a tilt that cannot be
            # evaluated would otherwise surface as a NaN far field deep inside
            # the fit, long after the hardware minutes were spent.
            self._error_terms.validate_focal_geometry()

        if not np.isfinite(fourier_weight) or fourier_weight < 0.0:
            raise ValueError(
                f"fourier_weight must be finite and >= 0, got {fourier_weight!r}"
            )
        self._fourier_weight = float(fourier_weight)
        if loss_kind not in LOSS_KINDS:
            raise ValueError(
                f"loss_kind must be one of {LOSS_KINDS}, got {loss_kind!r}"
            )
        self._loss_kind = str(loss_kind)
        if not np.isfinite(affine_lr) or affine_lr <= 0:
            raise ValueError(
                f"affine_lr must be a positive finite float, got {affine_lr!r}"
            )
        self._affine_lr = float(affine_lr)
        self._travel_lr = (
            self._affine_lr if travel_lr is None else float(travel_lr)
        )
        if not np.isfinite(self._travel_lr) or self._travel_lr <= 0:
            raise ValueError(
                f"travel_lr must be a positive finite float, got {travel_lr!r}"
            )
        # 自标定参数是**优化器状态**, 不是构造参数: 它们的初值由
        # ``affine_params`` / ``travel_gammas`` 给出, 之后由拟合自己推动。
        self._affine_seed: PupilAffine | None = (
            None if affine_params is None else affine_params.clamped()
        )
        self._travel_seed: TravelScale | None = (
            None if travel_gammas is None else TravelScale(travel_gammas).clamped()
        )

        mask = np.ones(calc_n_zernike_terms(n_orders), dtype=np.float64)
        for noll in frozen_modes:
            index = int(noll)
            if not 1 <= index <= mask.size:
                raise ValueError(
                    f"frozen_modes entries must be Noll indices in 1..{mask.size}, "
                    f"got {noll!r}"
                )
            mask[index - 1] = 0.0
        # 保持为 numpy, 首次使用时才转成张量: 在 __init__ 的这个位置
        # 设备还没解析出来。
        self._frozen_mask: npt.NDArray[np.float64] | None = None if mask.all() else mask
        self._mode_mask: Any = None

        n_coeffs = calc_n_zernike_terms(n_orders)
        if initial_coefficients is None:
            if seed is None:
                initial = np.zeros(n_coeffs, dtype=np.float64)
            else:
                rng = np.random.default_rng(seed)
                initial = rng.normal(0.0, 0.1, n_coeffs)
        else:
            initial = np.asarray(initial_coefficients, dtype=np.float64)
            if initial.ndim != 1 or initial.shape[0] != n_coeffs:
                raise ValueError(
                    f"initial_coefficients must have shape ({n_coeffs},) for "
                    f"n_orders={n_orders}, got {initial.shape}"
                )
            if not np.all(np.isfinite(initial)):
                raise ValueError("initial_coefficients must be finite")
        self._initial_coefficients = initial.copy()

        self._n_orders = n_orders
        self._region = region
        self._radius = radius
        self._n_coeffs = n_coeffs
        self._lr = float(lr)
        self._seed = seed
        self._dev = (
            torch.device(device)
            if device is not None
            else (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        )
        self._dtype_name = dtype
        self._np_dtype = _numpy_dtype(dtype)

        # 一次性预计算 Zernike 基, numpy 与 torch 两条路径共用。
        self._basis_np = self._build_basis()
        self._basis_t = _to_tensor(self._basis_np, self._dev, dtype)
        # 瞳面掩膜 (内为 1, 外为 0), 对应孪生对相位之和所用的
        # ``nan_to_num(patch, nan=0.0)``。
        self._aperture_t = _to_tensor(self._aperture_np, self._dev, dtype)

        self._default_amplitude = self.native_amplitude(region)
        self._source_amplitude = self._default_amplitude
        self._loss_history: list[float] = []
        self._converged = False
        self._no_improve = 0

        # --- 自标定参数 (可选) ---------------------------------------------
        # 两者都是**优化器状态**: 它们跨轮次保留 (每轮重建优化器时由
        # ``set_calibration`` 把上一轮的取值带过来), 因为它们描述的是实验
        # 系统而不是某一次拟合。
        self._affine: Any = None
        if self._affine_seed is not None:
            self._affine = torch.tensor(
                self._affine_seed.to_vector(),
                dtype=self._basis_t.dtype,
                device=self._dev,
                requires_grad=True,
            )
        self._travel_gammas: Any = None
        self._projector_t: Any = None
        self._basis_flat_t: Any = None
        if self._travel_seed is not None:
            self._travel_gammas = torch.tensor(
                self._travel_seed.to_vector(),
                dtype=self._basis_t.dtype,
                device=self._dev,
                requires_grad=True,
            )
            self._basis_flat_t = _to_tensor(
                self._basis_np.reshape(self._n_coeffs, -1), self._dev, dtype
            )
            self._projector_t = _to_tensor(
                _projector_for_basis(self._basis_np), self._dev, dtype
            )

        self._restart(restore_initial=True)

        logger.debug(
            "ZernikeCoefficientOptimizer ready: n_coefficients={} region={} "
            "radius={} lr={} device={}",
            n_coeffs,
            region,
            radius,
            self._lr,
            self._dev,
        )

    # ------------------------------------------------------------------
    # 构造辅助函数
    # ------------------------------------------------------------------
    def _build_basis(self) -> npt.NDArray[np.floating]:
        """构建按 Noll 序排列的 Zernike 基, 瞳面之外为零。

        Returns:
            形状为 ``(n_coeffs, region, region)`` 的二维数组; 第 ``j`` 个模式
            是 Noll 索引 ``j + 1``。瞳面外的 NaN 映射为零。
        """
        generator = ZernikeGenerator(
            (self._region, self._region),
            radius=self._radius,
            n_orders=self._n_orders,
        )
        basis = np.zeros((self._n_coeffs, self._region, self._region))
        for index in range(self._n_coeffs):
            weights = np.zeros(self._n_coeffs)
            weights[index] = 1.0
            mode = np.asarray(generator.generate_noll(weights), dtype=np.float64)
            if index == 0:
                # 生成器在同一个圆形瞳面上求值所有模式, 所以 piston 模式的有限支撑
                # *就是* 那个瞳面。在这里导出掩膜 (而不是重新推导几何),
                # 使它与孪生严格一致 —— 孪生正是用同一套 NaN-出瞳面约定
                # 做掩膜的。
                self._aperture_np = np.isfinite(mode)
            basis[index] = np.nan_to_num(mode, nan=0.0)
        return basis

    def _restart(self, restore_initial: bool = False) -> None:
        """重置系数张量、Adam 状态与循环记账。

        Args:
            restore_initial: 为 ``True`` 时把系数重置为传给 ``__init__`` 的值;
                否则保留当前系数, 只重建优化器 / 循环状态, 从而让重复的
                :meth:`run` 暖启动。
        """
        torch = _torch()
        start = self._initial_coefficients if restore_initial else self.coefficients
        coefficients = torch.from_numpy(
            np.ascontiguousarray(start, dtype=self._np_dtype)
        ).to(self._dev)
        coefficients.requires_grad_(True)
        self._coefficients = coefficients
        self._opt = self._build_optimizer(coefficients)
        self._iteration = 0
        self._convergence_history = []
        self._best_value = float("inf")
        self._loss_history = []
        self._converged = False
        self._no_improve = 0

    def _build_optimizer(self, coefficients: Any) -> Any:
        """按参数分组建立 Adam (像差系数一组, 自标定参数各一组)。

        分组学习率是论文的做法: 像差系数用 1e-2, 仿射 θ 用 1e-6…1e-3,
        行程 γ 用 θ 的学习率。三组的量纲互不相同, 用同一个 lr 会让自标定
        参数要么不动、要么飞掉。

        没有自标定参数时走**单组**的旧写法, 以保证默认路径逐位不变。
        """
        torch = _torch()
        groups: list[dict[str, Any]] = []
        if self._affine is not None:
            groups.append({"params": [self._affine], "lr": float(self._affine_lr)})
        if self._travel_gammas is not None:
            groups.append(
                {"params": [self._travel_gammas], "lr": float(self._travel_lr)}
            )
        if not groups:
            return torch.optim.Adam([coefficients], lr=self._lr)
        groups.insert(0, {"params": [coefficients], "lr": float(self._lr)})
        return torch.optim.Adam(groups)

    # ------------------------------------------------------------------
    # 公开的辅助函数
    # ------------------------------------------------------------------
    @staticmethod
    def native_amplitude(region: int) -> npt.NDArray[np.floating]:
        """返回 ``region`` 网格上数字孪生的原生高斯光束。

        束腰随网格线性缩放, 使 ``region=TWIN_REGION`` 精确复现
        ``BeamParams.w0 = 250.0``, 由此得到的远场与数字孪生一致。

        Args:
            region: 方形网格的边长 (像素)。

        Returns:
            二维 float64 振幅图, 轴上为 1, 轴外为高斯分布。
        """
        w0 = TWIN_W0 * (region / TWIN_REGION)
        yy, xx = np.mgrid[0:region, 0:region]
        r2 = (yy - region / 2) ** 2 + (xx - region / 2) ** 2
        return np.exp(-r2 / (2 * w0**2))

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------
    @property
    def n_coefficients(self) -> int:
        """拟合的 Zernike 系数个数。"""
        return self._n_coeffs

    @property
    def n_orders(self) -> int:
        """Zernike 基的最大径向阶数。"""
        return self._n_orders

    @property
    def region(self) -> int:
        """方形网格的边长 (像素)。"""
        return self._region

    @property
    def radius(self) -> int:
        """Zernike 瞳面半径 (像素)。"""
        return self._radius

    @property
    def device(self) -> str:
        """优化所在的 torch 设备。"""
        return str(self._dev)

    @property
    def error_terms(self) -> ModelErrorTerms:
        """正向模型使用的误差项 (理想系统时为全零的默认容器)。"""
        return self._error_terms

    # ------------------------------------------------------------------
    # 自标定量 (傅里叶域损失 / 瞳面仿射 τ_θ / 行程缩放 γ)
    # ------------------------------------------------------------------
    @property
    def fourier_weight(self) -> float:
        """傅里叶域 L1 项的权重 α (``0`` = 关闭)。"""
        return float(self._fourier_weight)

    @property
    def loss_kind(self) -> str:
        """图像域损失的种类 (``"mse"`` 或 ``"l1"``)。"""
        return str(self._loss_kind)

    @property
    def affine(self) -> PupilAffine | None:
        """当前的瞳面仿射 τ_θ; ``None`` = 该自标定量关闭。"""
        if self._affine is None:
            return None
        return PupilAffine.from_vector(
            self._affine.detach().cpu().numpy().astype(np.float64)
        )

    @property
    def travel(self) -> TravelScale | None:
        """当前的逐模式族行程缩放 γ; ``None`` = 该自标定量关闭。"""
        if self._travel_gammas is None:
            return None
        return TravelScale.from_vector(
            self._travel_gammas.detach().cpu().numpy().astype(np.float64)
        )

    def set_calibration(
        self,
        affine: PupilAffine | None = None,
        travel: TravelScale | None = None,
    ) -> None:
        """设置自标定参数的取值 (跨轮次携带, 或回滚)。

        MIL 循环每一轮都会重建优化器, 所以上一轮推到的自标定取值必须
        **显式**带过来, 否则每一轮都从单位变换重新开始, 参数永远学不到
        东西。同一个入口也用于**回滚**: 逐轮信任域越界时把参数退回本轮开始
        的取值。

        Args:
            affine: 新的 τ_θ; ``None`` = 保持现状。
            travel: 新的 γ; ``None`` = 保持现状。

        Raises:
            ValueError: 取值越界时。
        """
        torch = _torch()
        if affine is not None and self._affine is not None:
            clamped = affine.clamped()
            with torch.no_grad():
                self._affine.copy_(
                    torch.tensor(
                        clamped.to_vector(),
                        dtype=self._affine.dtype,
                        device=self._affine.device,
                    )
                )
        if travel is not None and self._travel_gammas is not None:
            clamped = travel.clamped()
            with torch.no_grad():
                self._travel_gammas.copy_(
                    torch.tensor(
                        clamped.to_vector(),
                        dtype=self._travel_gammas.dtype,
                        device=self._travel_gammas.device,
                    )
                )

    def calibration_state(self) -> dict[str, Any]:
        """返回自标定参数的当前取值 (供逐轮记录)。

        Returns:
            一个字典, 含 ``"affine"`` / ``"travel"`` 两个子字典 (关闭时为空
            字典) 以及 ``"fourier_weight"`` / ``"loss_kind"``。
        """
        affine = self.affine
        travel = self.travel
        return {
            "fourier_weight": float(self._fourier_weight),
            "loss_kind": str(self._loss_kind),
            "affine": {} if affine is None else affine.to_dict(),
            "travel": {} if travel is None else travel.to_dict(),
            "travel_families": list(FAMILY_NAMES),
        }

    def set_error_terms(self, terms: ModelErrorTerms) -> None:
        """换掉正向模型使用的误差项。

        给「扫描一个误差参数」的用法用 (例如 :func:`fit_lut_gamma`): 误差项
        容器是 frozen 的, 所以换掉它必须走这个入口, 而不是就地改字段。

        Args:
            terms: 新的误差项。

        Raises:
            ValueError: 误差项需要的几何常数缺失时。
        """
        if terms is None:
            raise ValueError("set_error_terms needs a ModelErrorTerms instance")
        if terms.needs_focal_geometry:
            terms.validate_focal_geometry()
        self._error_terms = terms

    @property
    def iterations(self) -> int:
        """迄今为止执行的优化步数。"""
        return self._iteration

    @property
    def coefficient_tensor(self) -> torch.Tensor:
        """实时的系数参数张量 (requires_grad)。"""
        return self._coefficients

    @property
    def coefficients(self) -> npt.NDArray[np.floating]:
        """当前系数的 numpy 副本 (已 detach), 单位弧度。"""
        return self._coefficients.detach().cpu().numpy().astype(np.float64)

    @property
    def loss_history(self) -> list[float]:
        """每一步优化的损失值。"""
        return self._loss_history

    @property
    def last_loss(self) -> float | None:
        """最后一步的损失, 首次更新之前为 ``None``。"""
        return self._loss_history[-1] if self._loss_history else None

    @property
    def converged(self) -> bool:
        """是否满足提前停止条件。"""
        return self._converged

    @property
    def is_converged(self) -> bool:
        """迭代预算耗尽或拟合收敛时为 ``True``。

        把基类的预算检查与本类自己的判据结合起来: 一个绝对的损失下限,
        或者最优损失连续 :data:`PLATEAU_PATIENCE` 步没有改善的平台期。

        Returns:
            :meth:`run` 应当停止时为 ``True``。
        """
        if self._iteration >= self.max_iterations:
            return True
        if self._iteration < MIN_ITERATIONS or not self._loss_history:
            return False
        if self._loss_history[-1] <= ABSOLUTE_FLOOR:
            self._converged = True
            return True
        if self._no_improve >= PLATEAU_PATIENCE:
            # 平台期是真正的收敛判据, 不是预算耗尽,
            # 所以结果必须把它报告为已收敛。
            self._converged = True
            return True
        return False

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------
    def generate_basis(self) -> npt.NDArray[np.floating]:
        """返回 Zernike 基缓存的一份副本。

        Returns:
            按 Noll 序、形状为 ``(n_coefficients, region, region)`` 的数组;
            瞳面之外的项恰好为零。返回的是副本, 所以改动它不会影响优化器。
        """
        return self._basis_np.copy()

    def set_source_amplitude(self, source_amplitude: npt.NDArray[np.floating]) -> None:
        """设置后续步骤所用的源面 (瞳面) 振幅。

        Args:
            source_amplitude: 形状为 ``(region, region)`` 的二维振幅图。

        Raises:
            ValueError: 形状不对, 或取值非有限时。
        """
        amplitude = np.asarray(source_amplitude, dtype=np.float64)
        if amplitude.shape != (self._region, self._region):
            raise ValueError(
                f"source_amplitude shape {amplitude.shape} must be "
                f"({self._region}, {self._region})"
            )
        if not np.all(np.isfinite(amplitude)):
            raise ValueError("source_amplitude must be finite")
        self._source_amplitude = amplitude.copy()

    def forward_intensity(
        self,
        coefficients: npt.NDArray[np.floating],
        phase_slm: npt.NDArray[np.floating],
        source_amplitude: npt.NDArray[np.floating] | None = None,
    ) -> npt.NDArray[np.floating]:
        """求一个系数向量的远场强度。

        这是正向模型的唯一事实来源; 它与 :meth:`update` 所微分的
        是同一套计算, 只是不建计算图。可以用它合成一份参考测量。

        Args:
            coefficients: 一维系数向量, 单位弧度, 形状
                ``(n_coefficients,)``。
            phase_slm: 二维固定 SLM 相位, **未包裹的原始弧度**, 形状
                ``(region, region)``。
            source_amplitude: 本次调用可选的二维振幅覆盖值。

        Returns:
            模型网格上的二维 float64 远场强度。

        Raises:
            ValueError: 任一数组形状不对或含非有限值时。
        """
        torch = _torch()
        values = np.asarray(coefficients, dtype=np.float64)
        if values.ndim != 1 or values.shape[0] != self._n_coeffs:
            raise ValueError(
                f"coefficients must have shape ({self._n_coeffs},), got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("coefficients must be finite")
        phase_t = _to_tensor(
            self._validate_phase(phase_slm), self._dev, self._dtype_name
        )
        amplitude = self._resolve_amplitude(source_amplitude)
        with torch.no_grad():
            intensity = self._far_field_intensity(
                torch.from_numpy(np.ascontiguousarray(values, dtype=self._np_dtype)).to(
                    self._dev
                ),
                phase_t,
                _to_tensor(amplitude, self._dev, self._dtype_name),
            )
        return intensity.detach().cpu().numpy().astype(np.float64)

    def far_field_tensor(
        self,
        coefficients: npt.NDArray[np.floating],
        phase_slm: npt.NDArray[np.floating] | Any,
        source_amplitude: npt.NDArray[np.floating] | None = None,
    ) -> Any:
        """求一个系数向量的远场强度, **保留计算图**。

        与 :meth:`forward_intensity` 是同一套前向计算, 唯一区别是不进
        ``torch.no_grad()``: 返回的张量仍挂在图上, 因此调用方可以对**相位**
        求梯度 (系数在这里是常量样本, 不是待优化参数)。不确定性探针选择器
        正是靠这一点把 ``dU/dphase`` 反传到探针网格上。

        Args:
            coefficients: 一维系数向量, 单位弧度, 形状 ``(n_coefficients,)``。
                作为常量喂入, 不建梯度。
            phase_slm: 二维 SLM 相位, **未包裹的原始弧度**, 形状
                ``(region, region)``。可以是 numpy 数组, 也可以是调用方自建
                的 ``requires_grad`` torch 张量 (后者原样上设备, 由调用方控制
                精度与梯度)。
            source_amplitude: 本次调用可选的二维振幅覆盖值。

        Returns:
            挂在图上的二维远场强度张量 (设备/精度同优化器内部)。

        Raises:
            ValueError: 任一数组形状不对或含非有限值时。
        """
        values = np.asarray(coefficients, dtype=np.float64)
        if values.ndim != 1 or values.shape[0] != self._n_coeffs:
            raise ValueError(
                f"coefficients must have shape ({self._n_coeffs},), got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("coefficients must be finite")

        torch = _torch()
        if torch.is_tensor(phase_slm):
            phase_t = phase_slm.to(self._dev)
        else:
            phase_t = _to_tensor(
                self._validate_phase(phase_slm), self._dev, self._dtype_name
            )
        amplitude = self._resolve_amplitude(source_amplitude)
        coeff_t = torch.from_numpy(
            np.ascontiguousarray(values, dtype=self._np_dtype)
        ).to(self._dev)
        return self._far_field_intensity(
            coeff_t, phase_t, _to_tensor(amplitude, self._dev, self._dtype_name)
        )

    def reset(self) -> None:
        """把系数重置回初值并清空历史。"""
        self._restart(restore_initial=True)
        logger.debug("ZernikeCoefficientOptimizer reset to initial coefficients")

    def update(
        self, i_meas: npt.NDArray[np.floating], phase_slm: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """执行一步 Adam 并返回下一个系数向量。

        这一步是测量锚定的: 峰值归一化的 MSE 取在模型远场与实测远场之间,
        目标与归一化尺度都取自测量, 梯度则经 FFT 反传进系数。

        若优化器已经收敛, 这一步为空操作, 直接返回当前系数而不推进。

        Args:
            i_meas: 二维实测远场强度。形状不同时会重采样到模型网格
                (真实 CCD 帧是 250x248)。
            phase_slm: 二维固定 SLM 相位, **未包裹的原始弧度**, 形状
                ``(region, region)``。

        Returns:
            更新后的系数, 以 numpy 数组返回 (已 detach), 单位弧度。

        Raises:
            ValueError: 测量或相位图无效时。
        """
        if self._converged:
            return self.coefficients

        torch = _torch()
        target, peak_anchor = self._prepare_measurement(i_meas)
        phase_t = _to_tensor(
            self._validate_phase(phase_slm), self._dev, self._dtype_name
        )
        amplitude_t = _to_tensor(self._source_amplitude, self._dev, self._dtype_name)

        self._opt.zero_grad(set_to_none=True)
        intensity = self._far_field_intensity(self._coefficients, phase_t, amplitude_t)
        # 测量锚定的梯度: 目标与归一化尺度都来自实测图像;
        # 残差落在模型图像上。
        loss = self._anchored_intensity_loss(intensity, target, peak_anchor)
        if self._fourier_weight > 0.0:
            loss = loss + float(self._fourier_weight) * self._fourier_domain_loss(
                intensity, target, peak_anchor
            )
        loss.backward()
        self._opt.step()
        self._apply_mode_mask()
        self._clamp_calibration()

        with torch.no_grad():
            recorded = float(self._intensity_loss(intensity, target).detach().cpu())
        self._loss_history.append(recorded)
        self._update_stagnation(recorded)
        self._record(recorded)
        return self.coefficients

    def run(
        self,
        i_meas: npt.NDArray[np.floating],
        phase_slm: npt.NDArray[np.floating],
        source_amplitude: npt.NDArray[np.floating] | None = None,
    ) -> ZernikeCoefficientResult:
        """跑拟合循环, 直至收敛或用尽迭代预算。

        Adam 状态与损失历史会先重建, 但当前系数被保留, 所以重复调用可以从
        上一次拟合暖启动。调用 :meth:`reset` 可回到初始系数。

        Args:
            i_meas: 二维实测远场强度, 任意形状。
            phase_slm: 二维固定 SLM 相位, **未包裹的原始弧度**, 形状
                ``(region, region)``。
            source_amplitude: 形状为 ``(region, region)`` 的可选二维振幅覆盖
                值; 只有新建一个优化器才会恢复到原生高斯。

        Returns:
            一个 :class:`ZernikeCoefficientResult`, 持有拟合出的系数、
            逐迭代的 ``"loss"`` 历史、步数以及是否满足停止条件。
        """
        self._restart()
        if source_amplitude is not None:
            self.set_source_amplitude(source_amplitude)
        logger.debug(
            "ZernikeCoefficientOptimizer.run start: budget={} n_coefficients={}",
            self.max_iterations,
            self._n_coeffs,
        )
        for _ in range(self.max_iterations):
            self.update(i_meas, phase_slm)
            if self.is_converged:
                break
        result = ZernikeCoefficientResult(
            coefficients=self.coefficients,
            history={"loss": list(self._loss_history)},
            iterations=self._iteration,
            converged=self._converged,
        )
        logger.info(
            "ZernikeCoefficientOptimizer.run done: iterations={} converged={} "
            "loss {} -> {}",
            result.iterations,
            result.converged,
            result.history["loss"][0] if result.history["loss"] else float("nan"),
            result.history["loss"][-1] if result.history["loss"] else float("nan"),
        )
        return result

    # ------------------------------------------------------------------
    # 模型与损失
    # ------------------------------------------------------------------
    def _apply_mode_mask(self) -> None:
        """在一步优化器之后把冻结的系数就地清零。

        piston 与 tilt (Noll 1-3) 从远场*强度*上是**不可辨识**的:
        piston 是一个全局相位, 使 ``|E|**2`` 完全不变; tilt 只是把光斑
        平移, 而同窗口或按 argmax 对齐的比较几乎察觉不到。把它们放开并非
        无害 —— 没有可辨识信号时, 优化器会把残余停在那几个退化方向上,
        而不是报告 "没有像差"。在实验采集上实测: 拟合出的系数范数有
        56% 落在 Noll 1-3, 而拟合质量毫无改善。冻结它们遵循本仓库既有的
        约定 (``slm_square_shaping --zernike-mask`` 同样把 Noll 1-3 强制为零)。
        """
        if self._frozen_mask is None:
            return
        torch = _torch()
        if self._mode_mask is None:
            self._mode_mask = torch.from_numpy(self._frozen_mask).to(self._dev)
        with torch.no_grad():
            self._coefficients.mul_(self._mode_mask)

    def _clamp_calibration(self) -> None:
        """在一步优化器之后把自标定参数夹回硬边界。

        论文的约束是 ``γ ≥ −1`` (缩放因子非负) 与 ``o ≥ 0`` (物体振幅非负,
        在 :func:`~ao_shaping.utils.wavefront.calibration_terms.\
apply_pupil_affine` 里施加)。τ_θ 的边界是本仓库加的: 6 个分量都是几何量,
        越界意味着模型已经离开它能解释的物理范围。

        夹边界**不是**逐轮信任域 —— 逐轮信任域在
        :mod:`ao_shaping.optimizer.wfless.slm_model_in_loop` 那一层, 它管的是
        「这一轮最多能走多远」, 这里管的是「参数不许越界」。
        """
        torch = _torch()
        if self._affine is not None:
            with torch.no_grad():
                self._affine[0].clamp_(PupilAffine.AMP_MIN, PupilAffine.AMP_MAX)
                self._affine[1].clamp_(PupilAffine.AMP_MIN, PupilAffine.AMP_MAX)
                self._affine[2].clamp_(-PupilAffine.OFF_LIMIT, PupilAffine.OFF_LIMIT)
                self._affine[3].clamp_(-PupilAffine.OFF_LIMIT, PupilAffine.OFF_LIMIT)
                self._affine[4].clamp_(-PupilAffine.ROT_LIMIT, PupilAffine.ROT_LIMIT)
                self._affine[5].clamp_(PupilAffine.GRID_MIN, PupilAffine.GRID_MAX)
        if self._travel_gammas is not None:
            with torch.no_grad():
                self._travel_gammas.clamp_(
                    TravelScale.GAMMA_MIN, TravelScale.GAMMA_MAX
                )

    @property
    def far_field_size(self) -> int:
        """正向模型产出的远场网格边长。

        除非通过 ``far_field_size`` 请求了零填充, 否则等于 ``region``。
        在*瞳面*网格上构建目标的调用方, 必须先按
        ``far_field_size / region`` 把它缩放, 才能与模型远场比较 ——
        因为一旦涉及填充, 远场像素覆盖的物理范围就与瞳面像素不同了。
        """
        return self._far_field_size

    def far_field_intensity_tensor(
        self,
        coefficients: torch.Tensor,
        phase_slm: torch.Tensor,
        amplitude: torch.Tensor,
    ) -> torch.Tensor:
        """可微的正向远场强度 (与 :meth:`forward_intensity` 同一套计算)。

        这是**给调用方的同一个前向**: Step B 的全频随机相位合成直接调用它, 所以
        标定 (Step A 的拟合) 与合成 (Step B) 不可能落在不同的正向链上 ——
        误差项 (LUT 非线性 / 相机离焦 / 光路倾斜) 的作用位置与顺序由这一处
        代码决定。

        张量的设备与精度由**调用方**决定 (基与掩膜随之转换), 所以调用方不必
        与本优化器的设置一致; 在实际使用中两者本就一致。

        Args:
            coefficients: 一维系数张量 (弧度), 形状 ``(n_coefficients,)``。
            phase_slm: 二维命令相位张量 (raw 未包裹弧度)。
            amplitude: 二维瞳面振幅张量。

        Returns:
            挂在图上的二维远场强度张量。
        """
        torch = _torch()
        dtype = coefficients.dtype
        device = coefficients.device
        basis = self._basis_t.to(device=device, dtype=dtype)
        aperture = self._aperture_t.to(device=device, dtype=dtype)
        aberration = torch.einsum("j,jhw->hw", coefficients, basis)
        command = apply_lut_nonlinearity(
            phase_slm,
            float(self._error_terms.lut_gamma),
            float(self._error_terms.phi_max_rad),
        )
        if self._travel_gammas is not None:
            command = apply_travel_scale(
                command,
                self._travel_gammas.to(device=device, dtype=dtype),
                self._projector_t.to(device=device, dtype=dtype),
                self._basis_flat_t.to(device=device, dtype=dtype),
            )
        if self._affine is not None:
            patch, amplitude = apply_pupil_affine(
                command, aberration, amplitude, self._affine.to(device=device, dtype=dtype)
            )
        else:
            patch = command + aberration
        if self._pupil_oversampling == 1:
            patch = patch * aperture
        from ao_shaping.utils.wavefront.fraunhofer import interpolated_pupil_field

        field = interpolated_pupil_field(
            patch, amplitude, self._far_field_size, self._pupil_oversampling
        )
        field = apply_model_error_terms(field, self._error_terms)
        return field.real.pow(2) + field.imag.pow(2)

    def _far_field_intensity(
        self,
        coefficients: torch.Tensor,
        phase_slm: torch.Tensor,
        amplitude: torch.Tensor,
    ) -> torch.Tensor:
        """当前系数的可微远场强度。

        与数字孪生的约定一致:
        ``|fftshift(fft2(ifftshift(U), norm="ortho"))|**2`` 且
        ``U = A * exp(1j * patch)``。

        ``patch`` 严格遵循孪生: 先把 SLM 相位与像差相加, 再对*和*做掩膜,
        因为孪生施加的是 ``nan_to_num(phi_slm + aberration, nan=0.0)``,
        而像差在圆形瞳面之外是 NaN。于是总相位在瞳面之外是平的 (零),
        而高斯振幅在那里依然作用 —— 若只给像差乘掩膜, 那里就会留下
        ``exp(1j * phi_slm)``, 从而与孪生不一致。

        当 ``far_field_size`` 大于 ``region`` 时, 瞳面光场会在 FFT 之前被
        零填充到该尺寸。填充**不会**改变视场 —— 采样范围仍是
        ``lambda * f / d_slm`` —— 它只是以 ``far_field_size / region``
        倍的精细度去采样它。这在真实实验系统上很重要: 远场像素间距是
        ``lambda * f / (P * d_slm)``, 所以未填充的 ``P = region = 256``
        网格间距约 65 um, 会把本实验系统约 27 um 的光斑渲染成一个亚像素的
        delta (sigma ~0.4 px), 既无法与像素约 2.2 um 的相机比较, 更谈不上
        拿它做整形。数字孪生通过其零填充尺寸 ``P`` 暴露同一个开关;
        规范的填充采样另见 ``generate_zernike_farfield_sim_report``
        (FAR_N = 8192)。

        **误差项** (:attr:`error_terms`) 串在同一条链上, 位置与孪生一致:
        LUT 非线性作用在**命令相位**上 (瞳面相位之前), 离焦与倾斜作用在
        **焦面复场**上 (``|·|²`` 之前)。误差项全零时, 本函数与引入误差项
        之前的实现**逐位相同** (γ = 1 的 LUT 是恒等映射, 连 2π 包裹都不
        引入)。
        """
        torch = _torch()
        from ao_shaping.utils.wavefront.fraunhofer import interpolated_pupil_field

        aberration = torch.einsum("j,jhw->hw", coefficients, self._basis_t)
        command = apply_lut_nonlinearity(
            phase_slm,
            float(self._error_terms.lut_gamma),
            float(self._error_terms.phi_max_rad),
        )
        if self._travel_gammas is not None:
            command = apply_travel_scale(
                command,
                self._travel_gammas,
                self._projector_t,
                self._basis_flat_t,
            )
        if self._affine is not None:
            patch, amplitude = apply_pupil_affine(
                command, aberration, amplitude, self._affine
            )
        else:
            patch = command + aberration
        if self._pupil_oversampling == 1:
            patch = patch * self._aperture_t
        field = interpolated_pupil_field(
            patch, amplitude, self._far_field_size, self._pupil_oversampling
        )
        field = apply_model_error_terms(field, self._error_terms)
        return field.real.pow(2) + field.imag.pow(2)

    def _intensity_loss(
        self, intensity: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """强度图与目标之间的峰值归一化 MSE。

        上游 :func:`focal_intensity` 用的是补零夫琅禾费 FFT, **刻意不换角谱
        (ASM) 传播** (``signal_processing/angular_spectrum.py`` 提供规范实现):
        在 2f 实验系统几何实测, ASM 在 ``z = f`` 处返回的是**传播后的光束**
        (``max|out| = 1.0066``, 峰值落在瞳面位置), 而不是透镜的傅里叶变换;
        真正的夫琅禾费图样需``z >> a**2/lambda = 271 m``。边界由
        ``test_angular_spectrum.py::TestNotAFraunhoferSubstitute`` 锁定。

        与 :meth:`DifferentiableBeamOptimizer._intensity_loss` 完全一致:
        峰值归一化到最大 1.0 后取均方差。float32 下的峰值归一化避免了
        能量归一化在 FFT 把能量集中到少数像素时所遭受的下溢;
        ``+ PEAK_EPS`` 项则保护零峰值的情形。

        Args:
            intensity: 用自身峰值归一化的二维强度张量。
            target: 二维峰值归一化的目标张量。

        Returns:
            标量 MSE 损失张量, 对 ``intensity`` 可微。
        """
        torch = _torch()
        normalized = intensity / (intensity.max() + PEAK_EPS)
        if self._loss_kind == "l1":
            return torch.mean(torch.abs(normalized - target))
        return torch.mean((normalized - target) ** 2)

    def _anchored_intensity_loss(
        self,
        intensity: torch.Tensor,
        target: torch.Tensor,
        peak_anchor: torch.Tensor,
    ) -> torch.Tensor:
        """用于梯度那一步的测量锚定损失。

        目标与归一化尺度都来自实测图像; 只有残差在模型上求值。这是
        ``differentiable_beam.py`` 里测量锚定梯度的非退化形式: 在测量处
        以测量自身为目标求峰值归一化损失, 会给出恒为零的上游梯度。

        Args:
            intensity: 二维模型强度张量。
            target: 二维峰值归一化的实测张量。
            peak_anchor: 标量张量, 用作归一化尺度的实测峰值。

        Returns:
            标量损失张量, 对 ``intensity`` 可微。
        """
        torch = _torch()
        scaled = intensity / peak_anchor
        if self._loss_kind == "l1":
            return torch.mean(torch.abs(scaled - target))
        return torch.mean((scaled - target) ** 2)

    def _fourier_domain_loss(
        self,
        intensity: torch.Tensor,
        target: torch.Tensor,
        peak_anchor: torch.Tensor,
    ) -> torch.Tensor:
        """傅里叶域的 L1 残差: ``mean(|Re[fft2(I_model)] − Re[fft2(I_meas)]|)``。

        论文 (GRAPHYCS) 的损失是 ``Σ|I_n − q_n| + α·|F{I_n} − F{q_n}|``,
        傅里叶项取的是 **``Re[fft2]``** 的 L1 —— 不是复场, 也不是幅度谱。
        本实现与论文的差别有两条, 都写在这里以免被当成 bug:

        * **归一化**: 论文用「整组 min(正值) 扣背景 + 组内 max 归一」, 本
          仓库用 :func:`~ao_shaping.utils.image.beam_metrics.prepare_frame`
          (中值扣除 + clip)。**不替换**既有的预处理 —— 只在傅里叶项内部沿用
          既有路径的**峰值归一化** (模型侧除以实测峰值锚点, 实测侧已是峰值
          归一化), 所以两项的量纲与既有损失一致。
        * **求和 vs 均值**: 论文写的是像素**和**, 本实现取**均值**。两项都
          取均值时比值不变, 所以论文的 α 仍然可用; 但 α 的绝对含义依赖
          图像域项的量纲 (默认是 MSE, 论文是 L1), 这是与论文的一处已知
          差异, 实测数值见 ``scripts/slm/model_in_loop_selfcal_bakeoff.py``
          在已录真实探针上的对比择优。

        ⚠️ ``Re[fft2]`` 的 **DC 分量 = 图像的总能量**, 所以这一项同时约束了
        能量比 —— 而峰值归一化的 MSE 完全看不到能量差。

        Args:
            intensity: 二维模型强度张量。
            target: 二维峰值归一化的实测张量。
            peak_anchor: 标量张量, 用作归一化尺度的实测峰值。

        Returns:
            标量损失张量, 对 ``intensity`` 可微。
        """
        torch = _torch()
        model = torch.fft.fft2(intensity / peak_anchor).real
        measured = torch.fft.fft2(target).real
        return torch.mean(torch.abs(model - measured))

    # ------------------------------------------------------------------
    # 输入准备与记账
    # ------------------------------------------------------------------
    def _validate_phase(
        self, phase_slm: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """校验固定的 SLM 相位图并以二维数组返回。

        该相位以**未包裹的原始弧度**消费; 本方法绝不做包裹或缩放。
        """
        phase = np.asarray(phase_slm, dtype=np.float64)
        if phase.ndim != 2:
            raise ValueError(f"phase_slm must be 2D, got {phase.ndim}D")
        if phase.shape != (self._region, self._region):
            raise ValueError(
                f"phase_slm shape {phase.shape} must be "
                f"({self._region}, {self._region})"
            )
        if not np.all(np.isfinite(phase)):
            raise ValueError("phase_slm must be finite")
        return phase

    def _resolve_amplitude(
        self, source_amplitude: npt.NDArray[np.floating] | None
    ) -> npt.NDArray[np.floating]:
        """返回要使用的振幅, 默认取已存的那份。"""
        if source_amplitude is None:
            return self._source_amplitude
        amplitude = np.asarray(source_amplitude, dtype=np.float64)
        if amplitude.shape != (self._region, self._region):
            raise ValueError(
                f"source_amplitude shape {amplitude.shape} must be "
                f"({self._region}, {self._region})"
            )
        if not np.all(np.isfinite(amplitude)):
            raise ValueError("source_amplitude must be finite")
        return amplitude

    def _target_grid(self) -> tuple[int, int]:
        """损失计算要求实测帧被重采样到的网格。

        这是**远场**网格, 而不是瞳面网格: ``_far_field_intensity``
        会在 FFT 之前把瞳面零填充到 ``far_field_size``, 所以它返回的是
        ``far_field_size`` 见方的强度, 而 ``_anchored_intensity_loss``
        里的残差只有当测量也落在同一个网格上时才有意义。未配置填充时
        两者恰好相同。

        正是修好这一点才让 ``far_field_size`` 真正可用。此前测量被重采样
        到 ``region`` 而预测是 ``far_field_size``, 于是每一个满足
        ``far_field_size > region`` 的 ``update()`` 都抛形状不匹配 ——
        而这恰恰是 ``far_field_size`` 的 docstring 对真实实验系统所要求的配置,
        因为未填充的网格会把本实验系统约 27 um 的光斑采样成一个亚像素 delta。
        """
        pad = int(self._far_field_size)
        side = pad if pad > 0 else int(self._region)
        return (side, side)

    def _prepare_measurement(
        self, i_meas: npt.NDArray[np.floating]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """对实测远场做缩放、峰值归一化并转成张量。

        Args:
            i_meas: 二维实测强度, 任意形状 (真实 CCD 帧是 250x248,
                而模型网格是正方形)。

        Returns:
            (峰值归一化的目标张量, 用作梯度锚点的标量实测峰值张量) 元组。

        Raises:
            ValueError: 测量不是二维、含非有限值、没有正峰值, 或无法缩放到
                模型网格时。
        """
        torch = _torch()
        measured = np.asarray(i_meas, dtype=np.float64)
        if measured.ndim != 2:
            raise ValueError(f"i_meas must be 2D, got {measured.ndim}D")
        if not np.all(np.isfinite(measured)):
            raise ValueError("i_meas must be finite")

        grid = self._target_grid()
        if measured.shape != grid:
            factors = (grid[0] / measured.shape[0], grid[1] / measured.shape[1])
            measured = np.asarray(
                ndimage.zoom(measured, factors, order=1), dtype=np.float64
            )
            if measured.shape != grid:
                raise ValueError(
                    f"i_meas shape {i_meas.shape} could not be resized to "
                    f"{grid} (got {measured.shape})"
                )
            logger.debug("Resized measured far field {} -> {}", i_meas.shape, grid)

        peak = float(measured.max())
        if peak <= 0:
            raise ValueError("i_meas must have a positive peak")
        target = _to_tensor(measured / peak, self._dev, self._dtype_name)
        return target, torch.tensor(
            peak + PEAK_EPS, dtype=self._basis_t.dtype, device=self._dev
        )

    def _update_stagnation(self, loss_value: float) -> None:
        """推进 :attr:`is_converged` 所用的平台期计数器。"""
        previous_best = self._best_value
        if loss_value <= ABSOLUTE_FLOOR:
            self._converged = True
        if not np.isfinite(previous_best):
            self._no_improve = 0
            return
        tolerance = max(ABSOLUTE_FLOOR, RELATIVE_TOL * abs(previous_best))
        if previous_best - loss_value > tolerance:
            self._no_improve = 0
        else:
            self._no_improve += 1
