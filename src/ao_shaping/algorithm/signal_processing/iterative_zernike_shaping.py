"""迭代式 Zernike + 纯相位 SLM 光束整形优化器 (torch 可微)。

在 2f-傅里叶台上实现一个两阶段迭代方案:

* **阶段 A —— Zernike 标定**: 从一个初始随机 SLM 相位出发, 学出一小组
  (≤ n_max) Zernike 系数 (原始弧度), 使*仿真*远场与*参考*远场 (即 "实际"
  目标) 相匹配。
* **阶段 B —— 纯相位整形**: 冻结标定好的 Zernike 系数, 优化自由形式的
  SLM 相位, 使远场与方形目标相匹配。
* **A↔B 迭代**: 交替执行 A 与 B (提前停止), 直至方斑收敛。

正向模型是带零填充的夫琅禾费 FFT (float64):

    field = gauss_pupil · exp(1j·(zernike_phase + slm_phase))
    far_field = fftshift(fft2(ifftshift(pad(field))))

Zernike 基图来自 canonical 的 ``ZernikeGenerator`` 并转成常量 torch 张量;
只有它们的系数是可训练的。

所有相位生成器都返回**未包裹的原始弧度**; 唯一的 mod-2π 包裹点是 SLM 驱动
``create_phase_from_array`` (仿真中不走这条路)。

基于类的优化器约定 (AGENTS.md): ``__init__`` 校验 + 设置状态;
``update()`` 执行一步并返回下一个状态; ``run()`` 返回结果 dataclass。

需要 torch; 导入是惰性的 (``_torch()``), 因此没装 torch 时本包仍可导入。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from loguru import logger

from ao_shaping.utils.wavefront.zernike_calc import ZernikeGenerator, zernike_modes

__all__ = [
    "IterativeZernikeShapingConfig",
    "IterativeZernikeShapingResult",
    "IterativeZernikeShapingOptimizer",
]


# ---------------------------------------------------------------------------
# 结果 / 配置容器
# ---------------------------------------------------------------------------
@dataclass
class IterativeZernikeShapingConfig:
    """迭代式 Zernike + 纯相位整形优化器的参数。

    Attributes:
        n_grid: SLM 网格边长 (正方形) —— 即瞳面网格。
        n_zernike: Zernike 最大径向阶数 (n_max)。0 = 跳过标定。
        target_side_px: 远场 (相机) 像素计的目标方斑边长,
            即零填充远场网格 (``far_field_size``) 的像素。
        seed: 用于保证可复现的 RNG 种子。
        zernike_lr: Zernike 标定的学习率 (Adam)。这个值要保持小
            (1e-3..1e-2); 远场 MSE 的地形崎岖, 更大的值会让系数发散成非有限。
        slm_lr: SLM 相位整形的学习率 (每轮按余弦衰减)。
        calib_iters: 每轮 Zernike 标定的 Adam 步数。
        shaping_iters: 每轮 SLM 相位整形的 Adam 步数。
        max_outer_iters: A↔B 外层迭代的最大次数。
        early_stop_patience: 连续这么多轮没有改进后停止外层循环。
        early_stop_min_delta: 算作 "有改进" 所需的最小分数提升。
        far_field_padding: 夫琅禾费 FFT 的零填充倍数。远场网格为
            ``n_grid * far_field_padding``; 同尺寸的 FFT 会在每个束腰半径约
            1.1 px 的尺度上欠采样焦平面 (这是一个模型常量), 并混叠成一个
            点阵。必须与构建参考远场所用的台架配置一致。
    """

    n_grid: int = 64
    n_zernike: int = 4
    target_side_px: int = 16
    seed: int = 0
    zernike_lr: float = 0.005
    slm_lr: float = 0.02
    calib_iters: int = 50
    shaping_iters: int = 100
    max_outer_iters: int = 5
    early_stop_patience: int = 2
    early_stop_min_delta: float = 0.005
    far_field_padding: int = 8

    def __post_init__(self) -> None:
        """尽早校验字段 (与优化器构造函数的检查对应)。"""
        if self.n_grid < 4:
            raise ValueError(f"n_grid must be >= 4, got {self.n_grid}")
        if self.n_zernike < 0:
            raise ValueError(f"n_zernike must be >= 0, got {self.n_zernike}")
        if self.shaping_iters <= 0:
            raise ValueError(f"shaping_iters must be > 0, got {self.shaping_iters}")
        if self.max_outer_iters < 1:
            raise ValueError(
                f"max_outer_iters must be >= 1, got {self.max_outer_iters}"
            )
        if self.far_field_padding < 1:
            raise ValueError(
                f"far_field_padding must be >= 1, got {self.far_field_padding}"
            )

    @property
    def far_field_size(self) -> int:
        """零填充远场 (相机) 网格的边长。"""
        return self.n_grid * self.far_field_padding


@dataclass
class IterativeZernikeShapingResult:
    """迭代优化器完整跑一轮的结果。"""

    zernike_coeffs: dict[tuple[int, int], float]
    slm_phase: npt.NDArray[np.floating]
    far_field: npt.NDArray[np.floating]
    target: npt.NDArray[np.floating]
    score_history: list[dict[str, Any]]
    n_outer_iters: int
    converged: bool
    final_score: float


# ---------------------------------------------------------------------------
# torch 惰性导入
# ---------------------------------------------------------------------------
_torch_module: Any = None


def _torch() -> Any:
    """惰性导入并缓存 torch。

    Returns:
        ``torch`` 模块。

    Raises:
        RuntimeError: 若未安装 torch。
    """
    global _torch_module
    if _torch_module is None:
        try:
            import torch as _t
        except ImportError as exc:
            raise RuntimeError(
                "IterativeZernikeShapingOptimizer requires torch. "
                "Install with: uv sync --group ml"
            ) from exc
        _torch_module = _t
    return _torch_module


# ---------------------------------------------------------------------------
# 优化器
# ---------------------------------------------------------------------------
class IterativeZernikeShapingOptimizer:
    """两阶段迭代式 Zernike + 纯相位 SLM 光束整形优化器。

    正向模型为:

        field = gauss_pupil · exp(1j·(zernike_phase + slm_phase))
        far_field = fftshift(fft2(ifftshift(pad(field))))   (float64)

    Zernike 相位使用 canonical 的 RZern 网格约定:
    - 网格: (np.arange(n) - (n-1)/2) / radius, radius = n/2 (单位圆在网格边缘)
    - R = sqrt(x² + y²), θ = atan2(y, x)
    - Z_n^m(r,θ) = R_n^|m|(r) · (cos(mθ) if m≥0 else sin(|m|θ))
    - 基图来自 canonical 的 ``ZernikeGenerator``。

    本类遵循基于类的优化器约定:
    - ``__init__``: 校验配置, 预计算静态张量 (高斯瞳面、目标、
      Zernike 基模式)。
    - ``calibrate_zernike``: 阶段 A 的一轮 (返回更新后的系数)。
    - ``shape_phase``: 阶段 B 的一轮 (返回更新后的 slm 相位)。
    - ``update``: 一次外层迭代 (先 A 再 B, 若 n_zernike=0 则只 B)。
    - ``run``: 带提前停止的完整循环; 返回结果 dataclass。
    """

    def __init__(self, config: IterativeZernikeShapingConfig) -> None:
        """初始化优化器。

        Args:
            config: 优化器参数。

        Raises:
            ValueError: 若 n_grid < 4 或 n_zernike < 0。
        """
        if config.n_grid < 4:
            raise ValueError(f"n_grid must be >= 4, got {config.n_grid}")
        if config.n_zernike < 0:
            raise ValueError(f"n_zernike must be >= 0, got {config.n_zernike}")
        if config.n_zernike > 0 and config.calib_iters <= 0:
            raise ValueError("calib_iters must be > 0 when n_zernike > 0")
        if config.shaping_iters <= 0:
            raise ValueError(f"shaping_iters must be > 0, got {config.shaping_iters}")
        if config.max_outer_iters < 1:
            raise ValueError(
                f"max_outer_iters must be >= 1, got {config.max_outer_iters}"
            )

        self._config = config
        t = _torch()

        # --- 静态张量 (只预计算一次) ---
        n = config.n_grid
        m = config.far_field_size
        radius = n / 2.0

        # 归一化坐标网格 (canonical 的 RZern 约定)
        coords = (t.arange(n, dtype=t.float64) - (n - 1) / 2.0) / radius
        y, x = t.meshgrid(coords, coords, indexing="ij")  # shape (n, n)
        self._x = x
        self._y = y
        self._r = t.sqrt(x**2 + y**2)
        self._theta = t.atan2(y, x)
        self._far_field_size = m
        self._pad = (m - n) // 2

        # 瞳面上的高斯光斑, 与台架同约定
        # (exp(-r^2/w0^2), w0 = aperture/3.5, 在边缘 r = 1 处截断)。
        w0_norm = 2.0 / 3.5
        aperture_mask = (self._r <= 1.0).to(t.float64)
        self._gauss = t.exp(-(self._r**2) / (w0_norm**2)) * aperture_mask

        # Zernike 基模式 (n=1..n_zernike, 排除 piston)
        self._zernike_modes: list[tuple[int, int]] = []
        if config.n_zernike > 0:
            self._zernike_modes = [
                mode for mode in zernike_modes(config.n_zernike) if mode != (0, 0)
            ]
            generator = ZernikeGenerator(
                (n, n), radius=radius, n_orders=config.n_zernike
            )
            self._basis = [
                (
                    nn,
                    mm,
                    t.as_tensor(
                        np.nan_to_num(
                            generator.generate_polynomial({(nn, mm): 1.0}), nan=0.0
                        ),
                        dtype=t.float64,
                    ),
                )
                for nn, mm in self._zernike_modes
            ]
        else:
            self._basis = []
        self._n_zernike_params = len(self._zernike_modes)

        # 目标 (方形, 归一化到总和=1)
        target = t.zeros((m, m), dtype=t.float64)
        half = m // 2
        s = config.target_side_px
        target[half - s // 2 : half + s // 2, half - s // 2 : half + s // 2] = 1.0
        target = target / target.sum()
        self._target = target
        self._target_support = (target > 0).to(t.float64)

        # --- 可训练参数 (初始化为零; 在 run() 里重置) ---
        self._zernike_coeffs: t.nn.Parameter | None = None
        self._slm_phase: t.nn.Parameter | None = None

        logger.debug(
            "IterativeZernikeShapingOptimizer initialized: n_grid={}, n_zernike={}, "
            "zernike_params={}, target_side={}",
            n,
            config.n_zernike,
            self._n_zernike_params,
            config.target_side_px,
        )

    def _zernike_basis(self) -> list[tuple[int, int, Any]]:
        """返回由 canonical 生成器构建好的基张量缓存。

        这些基图来自 :class:`ZernikeGenerator` (``generate_polynomial``,
        瞳面外的 NaN 替换为 0), 并在 ``__init__`` 里一次性转成常量
        torch 张量; 只有系数是可训练的。
        在这里重新推导径向多项式, 就会变成 canonical Zernike 数学的第二份
        会漂移的副本。

        Returns:
            (n, m, basis_tensor) 元组列表。
        """
        return self._basis

    # ------------------------------------------------------------------
    # 正向模型
    # ------------------------------------------------------------------
    def _far_field(self, zernike_coeffs: Any, slm_phase: Any) -> Any:
        """由 Zernike 相位 + SLM 相位算出远场强度。

        瞳面光场零填充到远场网格后, 用居中形式的夫琅禾费 FFT
        (``fftshift(fft2(ifftshift(...)))``) 变换, 与
        ``slm_shaping_bench.forward_intensity`` 一致。

        Args:
            zernike_coeffs: Zernike 系数的一维张量 (原始弧度),
                长度 = n_zernike_params。
            slm_phase: SLM 相位的二维张量 (原始弧度), 形状 (n, n)。

        Returns:
            归一化的远场强度张量 (总和=1), 形状 (M, M)。
        """
        t = _torch()
        if zernike_coeffs is not None and self._n_zernike_params > 0:
            zernike_phase = t.zeros_like(slm_phase)
            for i, (_, _, basis) in enumerate(self._zernike_basis()):
                zernike_phase = zernike_phase + zernike_coeffs[i] * basis
        else:
            zernike_phase = t.zeros_like(slm_phase)

        total_phase = zernike_phase + slm_phase
        field = self._gauss * t.exp(1j * total_phase)
        n = field.shape[0]
        m = self._far_field_size
        if m > n:
            pad = self._pad
            padded = t.zeros((m, m), dtype=t.complex128, device=field.device)
            padded[pad : pad + n, pad : pad + n] = field
            field = padded
        focal = t.fft.fftshift(t.fft.fft2(t.fft.ifftshift(field)))
        intensity = focal.real**2 + focal.imag**2
        intensity = intensity / (intensity.sum() + 1e-12)
        return intensity

    def _zernike_phase(self, coeffs: dict[tuple[int, int], float]) -> npt.NDArray[np.floating]:
        """由系数字典求 Zernike 相位 (原始弧度), 形状 (n, n)。"""
        t = _torch()
        n = self._config.n_grid
        phase = t.zeros((n, n), dtype=t.float64)
        if coeffs and self._n_zernike_params > 0:
            for nm, _, basis in self._zernike_basis():
                if nm in coeffs:
                    phase = phase + float(coeffs[nm]) * basis
        return phase.numpy()

    def _score(self, intensity: Any) -> Any:
        """计算综合质量分 (PIB + 均匀度)。

        分数 = 0.5 * PIB + 0.5 * (1 - min(CV/0.3, 1))

        Args:
            intensity: 归一化的远场强度张量。

        Returns:
            标量张量 (综合分数, 越大越好)。
        """
        t = _torch()
        # 固定的居中支撑区 (与 center=None 的 compute_beam_metrics 一致)。滚动
        # 跟随 argmax 的方框是不连续的: 对类散斑光场, ~1e-3 的模型变化就足以
        # 让 argmax 在近乎等强的晶粒之间跳, 于是优化器追一个已经盖不住光束
        # 的方框。
        support = self._target_support
        pib = (intensity * support).sum()
        vals = intensity[support > 0]
        if vals.numel() == 0 or vals.mean() <= 0:
            cv = t.tensor(float("inf"), dtype=t.float64, device=intensity.device)
        else:
            # 用总体标准差 (unbiased=False) 以匹配 numpy 的 np.std,
            # 台架的 composite_score 用的正是它 -- 否则两个目标函数会不一致。
            cv = vals.std(unbiased=False) / (vals.mean() + 1e-12)
        # 与 slm_shaping_bench.composite_from_pib_cv 保持同步。
        return 0.5 * pib + 0.5 * (1.0 / (1.0 + cv))

    # ------------------------------------------------------------------
    # 阶段 A: Zernike 标定
    # ------------------------------------------------------------------
    def calibrate_zernike(
        self,
        actual_far_field: Any,
        initial_slm_phase: Any,
    ) -> dict[tuple[int, int], float]:
        """跑一轮 Zernike 标定 (阶段 A)。

        在给定固定初始 SLM 相位的前提下, 学习 Zernike 系数 (原始弧度),
        使仿真远场与参考 "实际" 远场之间的差异最小。

        Args:
            actual_far_field: 参考远场 (numpy 或 torch), 形状 (n, n),
                归一化到总和=1。
            initial_slm_phase: 初始 SLM 相位 (numpy 或 torch), 形状 (n, n),
                原始弧度 (未包裹)。

        Returns:
            把 (n, m) 映射到标定后系数 (原始弧度) 的字典。
        """
        t = _torch()
        if self._n_zernike_params == 0:
            return {}

        if isinstance(actual_far_field, np.ndarray):
            target_ff = t.as_tensor(actual_far_field, dtype=t.float64)
        else:
            target_ff = actual_far_field.to(t.float64)

        if isinstance(initial_slm_phase, np.ndarray):
            slm_init = t.as_tensor(initial_slm_phase, dtype=t.float64)
        else:
            slm_init = initial_slm_phase.to(t.float64)

        # 可训练的 Zernike 系数
        zernike_coeffs = t.nn.Parameter(
            t.zeros(self._n_zernike_params, dtype=t.float64)
        )
        opt = t.optim.Adam([zernike_coeffs], lr=self._config.zernike_lr)

        # 和取归一化的远场 => 逐像素值 ~1/m^2, 原始 MSE ~1e-10,
        # 那里 Adam 默认的 eps=1e-8 会盖过梯度。用目标做缩放让损失保持
        # O(1), 这样学习率才真正起作用。
        scale = t.mean(target_ff**2) + 1e-12
        for _ in range(self._config.calib_iters):
            opt.zero_grad()
            ff = self._far_field(zernike_coeffs, slm_init)
            loss = t.mean((ff - target_ff) ** 2) / scale
            loss.backward()
            opt.step()
            if not bool(t.isfinite(zernike_coeffs).all()):
                logger.warning(
                    "Zernike calibration diverged to non-finite coefficients at "
                    "iter {} (zernike_lr={}); returning the last finite estimate.",
                    _,
                    self._config.zernike_lr,
                )
                break

        # 提取标定后的系数
        with t.no_grad():
            coeffs_dict: dict[tuple[int, int], float] = {}
            for i, (n, m, _) in enumerate(self._zernike_basis()):
                coeffs_dict[(n, m)] = float(zernike_coeffs[i].item())
        return coeffs_dict

    # ------------------------------------------------------------------
    # 阶段 B: SLM 相位整形
    # ------------------------------------------------------------------
    def shape_phase(
        self,
        zernike_coeffs: dict[tuple[int, int], float] | None,
        initial_slm_phase: npt.NDArray[np.floating] | None = None,
    ) -> npt.NDArray[np.floating]:
        """跑一轮 SLM 相位整形 (阶段 B)。

        冻结 Zernike 系数, 对照方形目标优化自由形式的 SLM 相位,
        使综合分数 (PIB + 均匀度) 最大化。

        Args:
            zernike_coeffs: 冻结的 Zernike 系数 (原始弧度), 或 None。
            initial_slm_phase: 初始 SLM 相位 (原始弧度), 形状 (n, n)。
                为 None 时初始化为全零。

        Returns:
            优化后的 SLM 相位 (numpy, 原始弧度), 形状 (n, n)。
        """
        t = _torch()
        n = self._config.n_grid

        if zernike_coeffs is None or self._n_zernike_params == 0:
            zernike_vec = t.zeros(self._n_zernike_params, dtype=t.float64)
        else:
            zernike_vec = t.zeros(self._n_zernike_params, dtype=t.float64)
            for i, (nm, _) in enumerate(self._zernike_modes):
                if nm in zernike_coeffs:
                    zernike_vec[i] = zernike_coeffs[nm]

        if initial_slm_phase is None:
            slm_phase = t.nn.Parameter(t.zeros((n, n), dtype=t.float64))
        elif isinstance(initial_slm_phase, np.ndarray):
            slm_phase = t.nn.Parameter(
                t.as_tensor(initial_slm_phase, dtype=t.float64).clone()
            )
        else:
            slm_phase = t.nn.Parameter(initial_slm_phase.to(t.float64).clone())

        base_lr = self._config.slm_lr
        opt = t.optim.Adam([slm_phase], lr=base_lr)
        iters = self._config.shaping_iters
        best_score = float("-inf")
        best_phase = slm_phase.detach().clone()

        for i in range(iters):
            # 余弦衰减: 平直的 slm_lr 会让 Adam 过冲、分数震荡,
            # 于是这一轮会停在一个任意的 (较差的) 迭代点上。
            for group in opt.param_groups:
                group["lr"] = base_lr * 0.5 * (1.0 + np.cos(np.pi * i / max(iters, 1)))
            opt.zero_grad()
            ff = self._far_field(zernike_vec, slm_phase)
            score = self._score(ff)
            score_value = float(score.item())
            if score_value > best_score:
                best_score = score_value
                best_phase = slm_phase.detach().clone()
            (-score).backward()
            opt.step()

        return best_phase.numpy()

    # ------------------------------------------------------------------
    # 一次外层迭代
    # ------------------------------------------------------------------
    def update(
        self,
        actual_far_field: Any,
        zernike_coeffs: dict[tuple[int, int], float],
        slm_phase: npt.NDArray[np.floating],
    ) -> tuple[dict[tuple[int, int], float], npt.NDArray[np.floating]]:
        """执行一次外层迭代: 阶段 A (若 n_zernike > 0), 然后阶段 B。

        Args:
            actual_far_field: 参考远场 (供阶段 A 使用)。
            zernike_coeffs: 当前 Zernike 系数。
            slm_phase: 当前 SLM 相位 (原始弧度)。

        Returns:
            (更新后的 zernike_coeffs, 更新后的 slm_phase) 元组。
        """
        if self._n_zernike_params > 0:
            zernike_coeffs = self.calibrate_zernike(actual_far_field, slm_phase)
        slm_phase = self.shape_phase(zernike_coeffs, slm_phase)
        return zernike_coeffs, slm_phase

    # ------------------------------------------------------------------
    # 带提前停止的完整运行
    # ------------------------------------------------------------------
    def run(
        self,
        actual_far_field: npt.NDArray[np.floating],
        initial_slm_phase: npt.NDArray[np.floating] | None = None,
    ) -> IterativeZernikeShapingResult:
        """运行完整的迭代优化循环。

        Args:
            actual_far_field: 参考远场 (即 "实际" 目标), 形状 (n, n),
                归一化到总和=1。
            initial_slm_phase: 初始 SLM 相位 (原始弧度), 形状 (n, n)。
                为 None 时初始化为全零 (平坦)。

        Returns:
            含全部中间量的 IterativeZernikeShapingResult。
        """
        t = _torch()
        cfg = self._config
        n = cfg.n_grid

        # 整形起点。
        if initial_slm_phase is None:
            slm_phase = np.zeros((n, n), dtype=np.float64)
        else:
            slm_phase = np.asarray(initial_slm_phase, dtype=np.float64)

        # 阶段 A 必须用构建参考时所用的同一个 SLM 相位 (平坦) 来求值;
        # 若用整形的暖启动, 拟合就会与参考不自洽。
        calibration_phase = np.zeros((n, n), dtype=np.float64)

        zernike_coeffs: dict[tuple[int, int], float] = {}
        score_history: list[dict[str, Any]] = []
        converged = False

        # 计算初始分数
        t_arr = _torch()
        actual_t = t_arr.as_tensor(actual_far_field, dtype=t_arr.float64)
        slm_t = t_arr.as_tensor(slm_phase, dtype=t_arr.float64)
        ff = self._far_field(
            t_arr.zeros(self._n_zernike_params, dtype=t_arr.float64)
            if self._n_zernike_params > 0
            else None,
            slm_t,
        )
        init_score = float(self._score(ff).item())
        score_history.append({"outer_iter": 0, "score": init_score, "stage": "init"})

        best_score = init_score
        best_slm_phase = slm_phase
        best_ff = ff.detach().numpy()
        best_zernike: dict[tuple[int, int], float] = dict(zernike_coeffs)
        patience_counter = 0
        first_shape = True

        for outer in range(1, cfg.max_outer_iters + 1):
            # 阶段 A (若适用)
            if self._n_zernike_params > 0:
                zernike_coeffs = self.calibrate_zernike(
                    actual_far_field, calibration_phase
                )

            # 模型会把冻结的 Zernike 加到 SLM 相位上, 所以第一轮整形要把它从初始
            # 相位里减掉: 否则暖启动会被一份模型随后又加回来的 Zernike 污染。
            if first_shape and self._n_zernike_params > 0:
                slm_phase = slm_phase - self._zernike_phase(zernike_coeffs)
                first_shape = False

            # 阶段 B
            slm_phase = self.shape_phase(zernike_coeffs, slm_phase)

            # 计算分数
            zernike_vec = t_arr.zeros(self._n_zernike_params, dtype=t_arr.float64)
            for i, nm in enumerate(self._zernike_modes):
                if nm in zernike_coeffs:
                    zernike_vec[i] = zernike_coeffs[nm]
            slm_t = t_arr.as_tensor(slm_phase, dtype=t_arr.float64)
            ff = self._far_field(zernike_vec, slm_t)
            score = float(self._score(ff).item())
            score_history.append({"outer_iter": outer, "score": score, "stage": "A+B"})

            # 提前停止
            if score > best_score + cfg.early_stop_min_delta:
                best_score = score
                best_slm_phase = slm_phase.copy()
                best_ff = ff.detach().numpy()
                best_zernike = dict(zernike_coeffs)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.early_stop_patience:
                    converged = True
                    logger.info(
                        "Early stop at outer iter {} (patience={}, score={:.4f})",
                        outer,
                        cfg.early_stop_patience,
                        score,
                    )
                    break

        # 返回分数最优的状态 (而非最后一个), 这样 ``final_score``、
        # ``slm_phase`` 与 ``far_field`` 描述的都是同一个产物。
        return IterativeZernikeShapingResult(
            zernike_coeffs=best_zernike,
            slm_phase=best_slm_phase,
            far_field=best_ff,
            target=self._target.detach().numpy(),
            score_history=score_history,
            n_outer_iters=len(score_history) - 1,
            converged=converged,
            final_score=best_score,
        )
