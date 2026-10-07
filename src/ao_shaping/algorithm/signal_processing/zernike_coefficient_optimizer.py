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

import numpy as np
import numpy.typing as npt
from loguru import logger
from scipy import ndimage

from ao_shaping.algorithm.signal_processing.iterative_base import IterativeOptimizer
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
            device: torch 设备字符串。默认为 ``"cpu"``, 使运行可复现且绝不
                隐式碰 GPU; 要用 GPU 需显式传 ``"cuda"``。
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
            raise ValueError(
                f"dtype must be 'float32' or 'float64', got {dtype!r}"
            )
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
        self._dev = torch.device(device if device is not None else "cpu")
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
        start = (
            self._initial_coefficients if restore_initial else self.coefficients
        )
        coefficients = torch.from_numpy(
            np.ascontiguousarray(start, dtype=self._np_dtype)
        ).to(self._dev)
        coefficients.requires_grad_(True)
        self._coefficients = coefficients
        self._opt = torch.optim.Adam([coefficients], lr=self._lr)
        self._iteration = 0
        self._convergence_history = []
        self._best_value = float("inf")
        self._loss_history = []
        self._converged = False
        self._no_improve = 0

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
                torch.from_numpy(
                    np.ascontiguousarray(values, dtype=self._np_dtype)
                ).to(self._dev),
                phase_t,
                _to_tensor(amplitude, self._dev, self._dtype_name),
            )
        return intensity.detach().cpu().numpy().astype(np.float64)

    def reset(self) -> None:
        """把系数重置回初值并清空历史。"""
        self._restart(restore_initial=True)
        logger.debug("ZernikeCoefficientOptimizer reset to initial coefficients")

    def update(self, i_meas: npt.NDArray[np.floating], phase_slm: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
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
        amplitude_t = _to_tensor(
            self._source_amplitude, self._dev, self._dtype_name
        )

        self._opt.zero_grad(set_to_none=True)
        intensity = self._far_field_intensity(
            self._coefficients, phase_t, amplitude_t
        )
        # 测量锚定的梯度: 目标与归一化尺度都来自实测图像;
        # 残差落在模型图像上。
        self._anchored_intensity_loss(intensity, target, peak_anchor).backward()
        self._opt.step()
        self._apply_mode_mask()

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
        而不是报告 "没有像差"。在真实硬件采集上实测: 拟合出的系数范数有
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

    @property
    def far_field_size(self) -> int:
        """正向模型产出的远场网格边长。

        除非通过 ``far_field_size`` 请求了零填充, 否则等于 ``region``。
        在*瞳面*网格上构建目标的调用方, 必须先按
        ``far_field_size / region`` 把它缩放, 才能与模型远场比较 ——
        因为一旦涉及填充, 远场像素覆盖的物理范围就与瞳面像素不同了。
        """
        return self._far_field_size

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
        倍的精细度去采样它。这在真实台架上很重要: 远场像素间距是
        ``lambda * f / (P * d_slm)``, 所以未填充的 ``P = region = 256``
        网格间距约 65 um, 会把本台架约 27 um 的光斑渲染成一个亚像素的
        delta (sigma ~0.4 px), 既无法与像素约 2.2 um 的相机比较, 更谈不上
        拿它做整形。数字孪生通过其零填充尺寸 ``P`` 暴露同一个开关;
        规范的填充采样另见 ``generate_zernike_farfield_sim_report``
        (FAR_N = 8192)。
        """
        torch = _torch()
        from ao_shaping.utils.wavefront.fraunhofer import focal_intensity

        aberration = torch.einsum("j,jhw->hw", coefficients, self._basis_t)
        patch = (phase_slm + aberration) * self._aperture_t
        field = amplitude * torch.exp(1j * patch)
        return focal_intensity(field, self._far_field_size)

    def _intensity_loss(
        self, intensity: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """强度图与目标之间的峰值归一化 MSE。

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
        return torch.mean((intensity / peak_anchor - target) ** 2)

    # ------------------------------------------------------------------
    # 输入准备与记账
    # ------------------------------------------------------------------
    def _validate_phase(self, phase_slm: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
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

    def _resolve_amplitude(self, source_amplitude: npt.NDArray[np.floating] | None) -> npt.NDArray[np.floating]:
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
        而这恰恰是 ``far_field_size`` 的 docstring 对真实台架所要求的配置,
        因为未填充的网格会把本台架约 27 um 的光斑采样成一个亚像素 delta。
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
            logger.debug(
                "Resized measured far field {} -> {}", i_meas.shape, grid
            )

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
