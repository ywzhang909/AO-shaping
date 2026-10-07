"""用梯度下降优化 SLM 相位实现的可微分光束整形。

本模块用一个用 PyTorch 实现的完全可微正向模型 (相位 → FFT/ASM 传播 →
远场强度), 从而支持对 SLM 相位图的直接基于梯度的优化。目标函数把均匀度、
效率、零级抑制与平滑度损失的加权组合, 对照一个目标强度掩码来最小化。

PyTorch 在导入时是**可选**依赖 —— 所有 torch 符号都通过
:func:`_torch` 这个惰性访问器获取, 因此项目其余部分在没装 torch 的情况下
也能 ``import`` 本模块。调用任何依赖 torch 的函数时, 若依赖缺失会收到
明确的 :class:`ImportError`。

Example:
    >>> from ao_shaping.algorithm.signal_processing.differentiable_shaping import (
    ...     create_target_mask, train_beam_shaping,
    ... )
    >>> mask = create_target_mask("square", (128, 128), 40)
    >>> result = train_beam_shaping(mask, (128, 128), iterations=200, seed=42)
    >>> result.phase.shape
    (128, 128)
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from loguru import logger

# Canonical single home for create_target_mask (see the alias below).
from ao_shaping.utils.image.target.patterns import (
    create_target_mask as _canonical_create_target_mask,
)

if TYPE_CHECKING:  # pragma: no cover – type-only imports
    from typing import Any

    from torch import Tensor


# ---------------------------------------------------------------------------
# 惰性 torch 访问器
# ---------------------------------------------------------------------------


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
            "differentiable_shaping requires PyTorch. Install with: uv sync --extra ml"
        ) from None
    return _t


# ---------------------------------------------------------------------------
# 目标掩码生成
# ---------------------------------------------------------------------------


#: Re-export of the canonical ``create_target_mask``.
#:
#: This used to be a byte-identical private copy of the implementation in
#: ``utils/image/target/patterns.py`` (verified equal with docstrings stripped),
#: which is a duplicate implementation waiting to drift -- the failure mode this
#: repo has been bitten by repeatedly. ``utils/`` is the canonical layer because it
#: sits *below* ``algorithm/``; re-exporting upwards keeps this module's public API
#: unchanged while leaving a single implementation to maintain.
#:
#: See utils/image/target/patterns.py for the docstring and semantics.
create_target_mask = _canonical_create_target_mask


# ---------------------------------------------------------------------------
# 损失函数 (torch 张量进, 标量张量出)
# ---------------------------------------------------------------------------


def uniformity_loss(intensity: "Tensor", target: "Tensor") -> "Tensor":
    """目标区域内强度的变异系数。

    值越低表示照明越均匀。目标区域为空或只含零时返回零 (对 NaN 安全)。
    """
    torch = _torch()
    mask = target > 0
    vals = intensity[mask]
    if vals.numel() == 0:
        return torch.tensor(0.0, device=intensity.device, dtype=intensity.dtype)
    mu = vals.mean()
    if mu.abs() < 1e-12:
        return torch.tensor(0.0, device=intensity.device, dtype=intensity.dtype)
    return vals.std() / (mu.abs() + 1e-12)


def efficiency_loss(intensity: "Tensor", target: "Tensor") -> "Tensor":
    """1 减去落在目标内的总能量占比。

    值越低表示能量越集中在期望位置。
    """
    torch = _torch()
    total = intensity.sum()
    if total < 1e-12:
        return torch.tensor(1.0, device=intensity.device, dtype=intensity.dtype)
    target_energy = (intensity * target).sum()
    encircled = target_energy / (total + 1e-12)
    return 1.0 - encircled.clamp(0.0, 1.0)


def zero_order_penalty(intensity: "Tensor") -> "Tensor":
    """对场中心处强 DC / 零级尖峰的惩罚。

    返回一个小中心窗口 (5×5 像素) 内的平均强度, 并用总强度归一化 ——
    值越高表示零级抑制越差。
    """
    torch = _torch()
    H, W = intensity.shape[-2:]
    cy, cx = H // 2, W // 2
    r = 2  # 半窗宽 → 5×5 区域
    region = intensity[..., cy - r : cy + r + 1, cx - r : cx + r + 1]
    total = intensity.sum()
    if total < 1e-12:
        return torch.tensor(0.0, device=intensity.device, dtype=intensity.dtype)
    return region.mean() / (total / (H * W) + 1e-12)


def smoothness_regularization(phase: "Tensor") -> "Tensor":
    """对相位梯度施加的有限差分惩罚。

    相位图恒定时返回零, 并随高频空间变化增大。
    """
    torch = _torch()
    dy = phase[:, 1:] - phase[:, :-1]
    dx = phase[1:, :] - phase[:-1, :]
    return (dy**2).mean() + (dx**2).mean()


def total_loss(
    intensity: "Tensor",
    phase: "Tensor",
    target: "Tensor",
    *,
    w_uniformity: float = 0.4,
    w_efficiency: float = 0.6,
    w_zero_order: float = 0.0,
    w_smoothness: float = 0.0,
) -> "Tensor":
    """所有损失分量的加权和。

    各个分量均非负, 量级也大致相当。

    .. note::
        经实测验证的权重 (report/slm_differential_shaping/): 对于以光束原点
        为中心的目标, 零级惩罚必须为 0 —— 抑制 DC (中心) 会把能量从居中的
        方斑 / 光斑里赶出去, 使围栏能量崩塌 (EE 0.84 -> 0.07, 当
        ``w_zero_order=0.1``)。平滑度正则同样会与锐利方斑边缘所需的高频
        相位成分对抗; 权重取 0 效果最好。``w_efficiency >= w_uniformity``
        先把能量聚起来, 再由均匀度把它压平 (600+ 次迭代可达 CV<0.1;
        学习曲线见报告)。
    """
    return (
        w_uniformity * uniformity_loss(intensity, target)
        + w_efficiency * efficiency_loss(intensity, target)
        + w_zero_order * zero_order_penalty(intensity)
        + w_smoothness * smoothness_regularization(phase)
    )


# ---------------------------------------------------------------------------
# 可微传播
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=32)
def _asm_propagator_torch(
    grid_shape: tuple[int, int],
    dx: float,
    z: float,
    wavelength: float,
    device_str: str,
    dtype_name: str,
):
    """为给定的 (shape, dx, z, λ, device, dtype) 缓存 ASM 传播子。

    传播子只依赖光学几何, 与光场本身无关, 因此可以安全地在多次调用间共享。

    Returns:
        形状为 ``(H, W)`` 的复张量。
    """
    torch = _torch()
    dtype = getattr(torch, dtype_name)
    H, W = grid_shape
    # 用 float64 计算以与 numpy 参考实现在数值上一致,
    # 再转换到光场的 dtype。
    fx = torch.fft.fftfreq(W, dx, device=device_str, dtype=torch.float64)
    fy = torch.fft.fftfreq(H, dx, device=device_str, dtype=torch.float64)
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")
    k = 2.0 * math.pi / wavelength
    kx = 2.0 * math.pi * FX
    ky = 2.0 * math.pi * FY
    kz_sq = k**2 - kx**2 - ky**2
    evanescent = kz_sq < 0
    kz_sq = torch.where(evanescent, torch.zeros_like(kz_sq), kz_sq)
    H_prop = torch.exp(1j * torch.sqrt(kz_sq) * z)
    H_prop[evanescent] = 0.0
    # ifftshift 让传播子与 fft2 的输出排序对齐, 与
    # ao_shaping.algorithm.gerchberg_saxton 里的 numpy 参考实现一致
    # (已验证: 在随机复光场上 torch 与 numpy 的 ASM 一致到 ~1e-11,
    # 因此可以做直接交叉校验, 以及对称的 fft/asm 比较)。
    return torch.fft.ifftshift(H_prop).to(dtype=dtype)


def angular_spectrum_propagate_torch(
    field: "Tensor",
    dx: float,
    z: float,
    wavelength: float,
) -> "Tensor":
    """用 PyTorch FFT 实现的角谱法传播。

    镜像 ``ao_shaping.algorithm.gerchberg_saxton`` 里的 numpy 版
    ``angular_spectrum_propagate``, 但作用于带缓存传播子的 torch 张量。

    Args:
        field: 复光场张量 ``(H, W)``。
        dx: 像素间距, 单位米。
        z: 传播距离 (正 = 正向)。
        wavelength: 波长, 单位米。

    Returns:
        传播后的复光场, 形状与 *field* 相同。
    """
    torch = _torch()
    if field.ndim != 2:
        raise ValueError(f"Field must be 2D, got {field.ndim}D")
    H, W = field.shape[-2:]
    device_str = str(field.device)
    dtype_name = str(field.dtype).split(".")[-1]  # 例如 "complex64"
    H_prop = _asm_propagator_torch(
        (H, W),
        dx,
        z,
        wavelength,
        device_str,
        dtype_name,
    )
    F = torch.fft.fft2(field)
    return torch.fft.ifft2(F * H_prop)


# ---------------------------------------------------------------------------
# 正向模型 (工厂 —— 避免导入期出现 nn.Module)
# ---------------------------------------------------------------------------


def _build_forward(
    propagation: str,
    cell_spacing: float,
    distance: float,
    wavelength: float,
) -> Callable[["Tensor", "Tensor | None"], "Tensor"]:
    """把可微正向模型构建成闭包。

    这样既避免了导入期依赖 torch 的类定义, 又能让传播参数保持局部。

    Returns:
        可调用对象 ``(phase, source_amplitude=None) -> real_intensity``。
    """
    torch = _torch()

    if propagation not in ("fft", "asm"):
        raise ValueError(f"propagation must be 'fft' or 'asm', got {propagation!r}")

    def forward(
        phase: "Tensor",
        source_amplitude: "Tensor | None" = None,
    ) -> "Tensor":
        H, W = phase.shape[-2:]
        if source_amplitude is None:
            amp = torch.ones((H, W), device=phase.device, dtype=phase.dtype)
        else:
            amp = source_amplitude

        field = amp * torch.exp(1j * phase)

        if propagation == "fft":
            out_field = torch.fft.fftshift(torch.fft.fft2(field))
        else:  # asm
            out_field = angular_spectrum_propagate_torch(
                field,
                cell_spacing,
                distance,
                wavelength,
            )

        intensity = out_field.real**2 + out_field.imag**2
        return intensity

    return forward


# ---------------------------------------------------------------------------
# 结果容器
# ---------------------------------------------------------------------------


@dataclass
class DifferentiableShapingResult:
    """一次可微光束整形运行的输出。

    Attributes:
        phase: 最优 / 最终的 ``(H, W)`` ``float64`` 相位, 单位弧度。
        target_intensity: 用于优化的 ``(H, W)`` 目标掩码。
        simulated_intensity: 由最终相位产生的 ``(H, W)`` 强度。
        loss_history: 每次迭代的总损失值。
        iterations: 执行的迭代次数。
        converged: 是否跑完了全部请求的迭代。
    """

    phase: npt.NDArray[np.floating]
    target_intensity: npt.NDArray[np.floating]
    simulated_intensity: npt.NDArray[np.floating]
    loss_history: list[float]
    iterations: int
    converged: bool


# ---------------------------------------------------------------------------
# 主训练循环
# ---------------------------------------------------------------------------


def train_beam_shaping(
    target: "npt.NDArray[np.floating] | Tensor",
    grid_size: tuple[int, int],
    *,
    source_amplitude: "npt.NDArray[np.floating] | Tensor | None" = None,
    initial_phase: "npt.NDArray[np.floating] | Tensor | None" = None,
    propagation: str = "fft",
    optimizer: str = "adam",
    iterations: int = 300,
    lr: float = 3e-2,
    w_uniformity: float = 0.4,
    w_efficiency: float = 0.6,
    w_zero_order: float = 0.0,
    w_smoothness: float = 0.0,
    cell_spacing: float = 8e-6,
    distance: float = 0.1,
    wavelength: float = 1064e-9,
    device: str | None = None,
    seed: int | None = None,
    progress_callback: Callable[[int, float], None] | None = None,
    phase_callback: Callable[["Tensor"], None] | None = None,
) -> DifferentiableShapingResult:
    """用梯度下降优化 SLM 相位图。

    Args:
        target: ``(H, W)`` 目标强度掩码 (numpy 或 torch)。
        grid_size: ``(H, W)`` 网格尺寸。
        source_amplitude: 照明振幅。为 *None* 时取均匀值。
        initial_phase: 起始相位 (弧度)。为 *None* 时取全零。
        propagation: ``"fft"`` (夫琅禾费) 或 ``"asm"`` (角谱)。
        optimizer: ``"adam"`` 或 ``"lbfgs"``。
        iterations: 优化步数。
        lr: 学习率。
        w_uniformity: 均匀度损失的权重。
        w_efficiency: 效率损失的权重。
        w_zero_order: 零级惩罚的权重。
        w_smoothness: 平滑度正则的权重。
        cell_spacing: 像素间距, 单位米。
        distance: 传播距离, 单位米。
        wavelength: 波长, 单位米。
        device: ``"cuda"``、``"cpu"``, 或 *None* 表示自动检测。
        seed: 用于保证可复现的随机种子。
        progress_callback: 每步调用的 ``fn(iteration, loss)``。
        phase_callback: 每步调用的 ``fn(phase_tensor)``。

    Returns:
        字段全部填好的 :class:`DifferentiableShapingResult`。

    Raises:
        ValueError: 输入无效时。

    .. note::
        经实测调优的默认值 (256×256, report/slm_differential_shaping/):
        - ``lr=3e-2`` + ``w_zero_order=0``: 旧默认值 (``lr=1e-2``、
          ``w_zero_order=0.1``) 是坏的 —— 零级惩罚把能量赶出居中目标
          (围栏能量从可达的 0.84 崩到 0.07), 而低学习率又把预测困在
          平凡的均匀临界点上, 对 ``asm`` 尤其明显。用下面的默认值,
          fft/adam 在 600 次迭代内达到 CV<0.1、EE~0.84 (种子 1-3),
          spot 目标 CV~0/EE~0.87, asm 达到 CV~0/EE~0.90。
          L-BFGS (lr=1.0, 60 次外迭代) 给出近乎平顶的方斑 (CV~0)。
        - 能量集中的响应比均匀度更快: ``w_efficiency >= w_uniformity``
          的配比是在能量聚齐之后才把光束压平, 所以下面的取值偏向效率。
        - ``smoothness_regularization`` 会与锐利方斑边缘所需的高频相位
          成分对抗 —— 对这些目标, 权重 0 最优。
    """
    torch = _torch()

    # -- 校验 -----------------------------------------------------------
    if propagation not in ("fft", "asm"):
        raise ValueError(f"propagation must be 'fft' or 'asm', got {propagation!r}")
    if optimizer not in ("adam", "lbfgs"):
        raise ValueError(f"optimizer must be 'adam' or 'lbfgs', got {optimizer!r}")
    if len(grid_size) != 2:
        raise ValueError(f"grid_size must be a 2-tuple, got {len(grid_size)}D")
    H, W = grid_size
    if H <= 0 or W <= 0:
        raise ValueError(f"grid_size must have positive dimensions, got {(H, W)}")
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")

    # 校验目标的维度, 以及是否与 grid_size 匹配
    if isinstance(target, np.ndarray):
        if target.ndim != 2:
            raise ValueError(f"target must be 2D, got {target.ndim}D")
        target_shape = target.shape
    else:  # torch 张量
        if target.dim() != 2:
            raise ValueError(f"target must be 2D, got {target.dim()}D")
        target_shape = tuple(target.shape)

    if target_shape != (H, W):
        raise ValueError(
            f"target shape {target_shape} does not match grid_size {(H, W)}"
        )

    # -- 可复现性 ----------------------------------------------------
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # -- 设备 ---------------------------------------------------------
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    dtype = torch.float32

    # -- 准备张量 ----------------------------------------------------
    if isinstance(target, np.ndarray):
        t_tensor = torch.from_numpy(target).to(device=dev, dtype=dtype)
    else:
        t_tensor = target.to(device=dev, dtype=dtype)

    if source_amplitude is not None:
        if isinstance(source_amplitude, np.ndarray):
            amp = torch.from_numpy(source_amplitude).to(device=dev, dtype=dtype)
        else:
            amp = source_amplitude.to(device=dev, dtype=dtype)
    else:
        amp = torch.ones((H, W), device=dev, dtype=dtype)

    if initial_phase is not None:
        if isinstance(initial_phase, np.ndarray):
            phase_init = torch.from_numpy(initial_phase).to(
                device=dev,
                dtype=dtype,
            )
        else:
            phase_init = initial_phase.to(device=dev, dtype=dtype)
    else:
        # 加入小随机噪声, 避开那种光场为纯实数 (强度在那里是相位的二次函数)
        # 的零梯度退化起点。实测发现 (report/slm_differential_shaping/):
        # 平坦 / 零相位是一个 ASM 永远逃不出的临界点 (损失反而变大而不是
        # 收敛); 0.1 量级的噪声能把预测踢离它。均匀 / `scale=0.1` 这一点
        # 也很关键 —— 1.0 量级的噪声聚焦很慢。
        phase_init = 0.1 * torch.randn((H, W), device=dev, dtype=dtype)

    phase = torch.nn.Parameter(phase_init.clone())

    # -- 正向模型 ----------------------------------------------------
    forward = _build_forward(propagation, cell_spacing, distance, wavelength)

    # -- 优化器 ------------------------------------------------------
    loss_history: list[float] = []

    def _closure():
        """Adam 与 L-BFGS 共用的单次前向 + 反向传播。"""
        opt.zero_grad()
        intensity = forward(phase, amp)
        loss = total_loss(
            intensity,
            phase,
            t_tensor,
            w_uniformity=w_uniformity,
            w_efficiency=w_efficiency,
            w_zero_order=w_zero_order,
            w_smoothness=w_smoothness,
        )
        loss.backward()
        return loss

    if optimizer == "adam":
        # 用 `Any` 是因为 torch 是可选依赖 —— 具体的优化器类型只能
        # 在运行时由 `optimizer` 字符串确定。
        opt: Any = torch.optim.Adam([phase], lr=lr)
    else:
        opt = torch.optim.LBFGS(
            [phase],
            lr=lr,
            max_iter=20,
            history_size=10,
            line_search_fn="strong_wolfe",
        )

    # -- 训练循环 ----------------------------------------------------
    converged = False
    logger.info(
        "Starting differentiable beam shaping: {} iterations, "
        "propagation={}, optimizer={}, device={}",
        iterations,
        propagation,
        optimizer,
        device,
    )

    for it in range(iterations):
        if optimizer == "adam":
            opt.zero_grad()
            loss = _closure()
            if torch.isnan(loss):
                logger.warning("NaN loss at iteration {}; stopping early", it)
                break
            opt.step()
        else:  # lbfgs —— 闭包由优化器内部调用
            loss = opt.step(_closure)
            if loss is None or torch.isnan(loss):
                logger.warning(
                    "L-BFGS returned NaN/None at iteration {}; stopping early", it
                )
                break

        # 把相位包裹到 [0, 2π)。对 Adam 来说每步都做是安全的; 对
        # L-BFGS 来说, 运行中包裹会破坏它的梯度历史记账, 所以只在
        # 最后才包裹。
        if optimizer == "adam":
            with torch.no_grad():
                phase.data = phase.data % (2.0 * math.pi)

        loss_val = float(loss.item())
        loss_history.append(loss_val)

        if progress_callback is not None:
            progress_callback(it, loss_val)

        if phase_callback is not None:
            phase_callback(phase.detach())

        # 每个迭代输出一次进度日志 (library 层为 debug 级别, 由 DEBUG=1 控制;
        # runner 层通过 progress_callback 在 INFO 级别逐迭代输出)
        logger.debug("Iter {}/{}  loss={:.6f}", it + 1, iterations, loss_val)

    if len(loss_history) >= iterations:
        converged = True

    # -- 提取结果 ----------------------------------------------------
    with torch.no_grad():
        final_intensity = forward(phase, amp)
        final_phase = (
            (phase.detach() % (2.0 * math.pi))
            .cpu()
            .numpy()
            .astype(
                np.float64,
            )
        )
        final_intensity_np = (
            final_intensity.detach()
            .cpu()
            .numpy()
            .astype(
                np.float64,
            )
        )
        if isinstance(target, np.ndarray):
            target_np = target.astype(np.float64)
        else:
            target_np = target.detach().cpu().numpy().astype(np.float64)

    logger.info(
        "Beam shaping complete: final loss={:.6f}, converged={}",
        loss_history[-1] if loss_history else float("nan"),
        converged,
    )

    return DifferentiableShapingResult(
        phase=final_phase,
        target_intensity=target_np,
        simulated_intensity=final_intensity_np,
        loss_history=loss_history,
        iterations=len(loss_history),
        converged=converged,
    )
