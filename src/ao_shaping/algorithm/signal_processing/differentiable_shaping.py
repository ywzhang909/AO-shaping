"""用梯度下降优化 SLM 相位实现的可微分光束整形。

本模块用一个用 PyTorch 实现的完全可微物理预测模型 (相位 → FFT/ASM 传播 →
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

import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
from loguru import logger

from ao_shaping.algorithm._torch_support import require_torch
from ao_shaping.algorithm.signal_processing.differentiable_objective import (
    _ZERO_ENERGY_FLOOR as _ZERO_ENERGY_FLOOR,
    efficiency_loss as efficiency_loss,
    smoothness_regularization as smoothness_regularization,
    total_loss as total_loss,
    total_loss_per_candidate as total_loss_per_candidate,
    uniformity_loss as uniformity_loss,
    zero_order_penalty as zero_order_penalty,
)
from ao_shaping.algorithm.signal_processing.shaping_lr_schedule import (
    LR_SCHEDULES as LR_SCHEDULES,
    _lr_schedule_factor as _lr_schedule_factor,
)
from ao_shaping.utils.image.ranks import NDIM_SINGLE_CHANNEL
from ao_shaping.utils.image.target.metrics import DIFFERENTIABLE_SHAPE_CHOICES
from ao_shaping.utils.image.target.patterns import TargetShape, create_target_shape
from ao_shaping.utils.slm.bench_constants import LASER_WAVELENGTH_M, SLM_PITCH_M

if TYPE_CHECKING:  # pragma: no cover – type-only imports
    from typing import Any

    from torch import Tensor

#: ``grid_size`` is always a ``(rows, columns)`` pair, and both validators reject a longer
#: sequence instead of reading the first two entries. This is a tuple arity, not an array
#: rank -- the note in :mod:`ao_shaping.utils.image.ranks` spells out why the two questions
#: must not share one constant.
_GRID_SIZE_ARITY = 2


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
    return require_torch(
        "differentiable_shaping requires PyTorch. Install with: uv sync --extra ml"
    )


# ---------------------------------------------------------------------------
# 目标掩码生成
# ---------------------------------------------------------------------------


def create_target_mask(
    shape: str,
    grid_size: tuple[int, int],
    size: int,
    *,
    sigma: float | None = None,
) -> npt.NDArray[np.float64]:
    """创建归一化的目标强度掩码。

    薄封装: 委托给唯一实现
    :func:`ao_shaping.utils.image.target.patterns.create_target_shape`,
    仅做参数换算 (``size`` → ``radius_ratio`` / ``side``) 与 dtype 归一
    (``float32`` → ``float64``)。公开签名与返回语义保持不变。

    Args:
        shape: ``"square"``, ``"rectangle"``, ``"triangle"``, ``"star"``,
            ``"circle"``, ``"gaussian"`` or ``"spot"``.
        grid_size: 输出数组的 ``(H, W)``。
        size: 以像素为单位的特征尺寸。
            * square —— side length
            * rectangle —— width (height is half the width)
            * triangle/star —— bounding diameter
            * circle —— 直径
            * gaussian —— 若未给出, ``sigma`` 默认为 ``size / 6``
            * spot —— 直径 (聚焦光斑, 与 circle 相同)
        sigma: 高斯标准差的覆盖值 (像素)。

    Returns:
        取值在 ``[0, 1]`` 内、居中的 ``(H, W)`` ``float64`` 掩码。

    Raises:
        ValueError: *shape* 无效或 *grid_size* 不是二维时。
    """
    # Single vocabulary (issue #199 P1): the subset of the canonical
    # ``TARGET_SHAPE_CHOICES`` whose geometry mapping this factory implements.
    # It used to be a private inline set.
    valid_shapes = DIFFERENTIABLE_SHAPE_CHOICES
    if shape not in valid_shapes:
        raise ValueError(f"Invalid shape {shape!r}. Must be one of {valid_shapes}")
    if len(grid_size) != _GRID_SIZE_ARITY:
        raise ValueError(f"grid_size must be a 2-tuple, got {len(grid_size)}D")
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")

    height, width = int(grid_size[0]), int(grid_size[1])
    cx, cy = width / 2.0, height / 2.0
    m = min(height, width)
    typed = cast(TargetShape, shape)

    # Thin delegate to the canonical factory. ``center`` is passed explicitly
    # as the boundary-centred ``(W/2, H/2)`` (not the factory's default
    # ``((W-1)/2, (H-1)/2)``) so the output is pixel-identical to the previous
    # standalone implementation. ``radius_ratio`` is chosen per shape so the
    # factory's ``radius = radius_ratio * min(H, W) / 2`` reproduces the old
    # ``size``-based geometry.
    if shape == "square":
        mask = create_target_shape(typed, (height, width), side=size, center=(cx, cy))
    elif shape == "rectangle":
        # ``side`` is the SHORT side; long side = side * aspect_ratio. The old
        # rectangle was ``size`` wide (horizontal) by ``size / 2`` tall, i.e.
        # short side ``size / 2`` with aspect ratio 2.
        mask = create_target_shape(
            typed, (height, width), side=size / 2.0, aspect_ratio=2.0, center=(cx, cy)
        )
    elif shape in {"triangle", "star"}:
        mask = create_target_shape(
            typed, (height, width), radius_ratio=size / m, center=(cx, cy)
        )
    elif shape in {"circle", "spot"}:
        mask = create_target_shape(
            typed, (height, width), radius_ratio=size / m, center=(cx, cy)
        )
    else:  # gaussian
        sig = sigma if sigma is not None else size / 6.0
        mask = create_target_shape(
            typed, (height, width), radius_ratio=4.0 * sig / m, center=(cx, cy)
        )

    return mask.astype(np.float64)


# ---------------------------------------------------------------------------
# 损失函数 (torch 张量进, 标量张量出) —— 2026-11-16 已拆至 differentiable_objective
# ---------------------------------------------------------------------------

# 本节原有的 uniformity_loss / efficiency_loss / zero_order_penalty /
# smoothness_regularization / total_loss 连同节标题逐字搬到同目录的
# :mod:`ao_shaping.algorithm.signal_processing.differentiable_objective`
# (800 行棘轮), 五个名字在文件顶部 import 块里显式 re-export, 因此
# ``_run_shaping_loop`` 的闭包照旧裸调 ``total_loss``。

# ---------------------------------------------------------------------------
# 可微传播 (2026-11-17 拆至 differentiable_propagation, 见该模块 docstring)
# ---------------------------------------------------------------------------


from ao_shaping.algorithm.signal_processing.differentiable_propagation import (  # noqa: E402
    _asm_propagator_torch as _asm_propagator_torch,
)
from ao_shaping.algorithm.signal_processing.differentiable_propagation import (  # noqa: E402
    _build_forward as _build_forward,
)
from ao_shaping.algorithm.signal_processing.differentiable_propagation import (  # noqa: E402
    angular_spectrum_propagate_torch as angular_spectrum_propagate_torch,
)


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


def _validate_shaping_inputs(
    target: "npt.NDArray[np.floating] | Tensor",
    grid_size: tuple[int, int],
    propagation: str,
    optimizer: str,
    iterations: int,
) -> tuple[int, int]:
    """Validate the run's inputs; return the grid height and width."""
    if propagation not in ("fft", "asm"):
        raise ValueError(f"propagation must be 'fft' or 'asm', got {propagation!r}")
    if optimizer not in ("adam", "lbfgs"):
        raise ValueError(f"optimizer must be 'adam' or 'lbfgs', got {optimizer!r}")
    if len(grid_size) != _GRID_SIZE_ARITY:
        raise ValueError(f"grid_size must be a 2-tuple, got {len(grid_size)}D")
    H, W = grid_size
    if H <= 0 or W <= 0:
        raise ValueError(f"grid_size must have positive dimensions, got {(H, W)}")
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")

    # 校验目标的维度, 以及是否与 grid_size 匹配
    if isinstance(target, np.ndarray):
        if target.ndim != NDIM_SINGLE_CHANNEL:
            raise ValueError(f"target must be 2D, got {target.ndim}D")
        target_shape = target.shape
    else:  # torch 张量
        if target.dim() != NDIM_SINGLE_CHANNEL:
            raise ValueError(f"target must be 2D, got {target.dim()}D")
        target_shape = tuple(target.shape)

    if target_shape != (H, W):
        raise ValueError(
            f"target shape {target_shape} does not match grid_size {(H, W)}"
        )
    return H, W


def _shaping_tensors(
    torch: Any,
    target: "npt.NDArray[np.floating] | Tensor",
    source_amplitude: "npt.NDArray[np.floating] | Tensor | None",
    initial_phase: "npt.NDArray[np.floating] | Tensor | None",
    height: int,
    width: int,
    dev: Any,
    dtype: Any,
) -> tuple[Any, Any, Any]:
    """Build the target, amplitude and initial-phase tensors."""
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
        amp = torch.ones((height, width), device=dev, dtype=dtype)

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
        phase_init = 0.1 * torch.randn((height, width), device=dev, dtype=dtype)
    return t_tensor, amp, phase_init


def _shaping_optimizer(
    torch: Any, phase: Any, optimizer: str, lr: float, lr_schedule: str | None = None
) -> Any:
    """The Adam / L-BFGS optimizer over the phase parameter.

    ``lr_schedule`` 只做**提前校验**; 调度倍率由 :func:`_run_shaping_loop`
    逐迭代施加, L-BFGS 忽略 (循环里会记警告)。
    """
    if lr_schedule not in (None, *LR_SCHEDULES):
        raise ValueError(f"未知学习率调度 {lr_schedule!r}; 可选: {LR_SCHEDULES} 或 None")
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
    return opt


def _resolve_debug_recorder(debug: Any, iterations: int) -> tuple[Any, bool]:
    """把 ``debug`` 归一成 ``(记录器, 是否由本函数负责关闭)``。

    ``debug`` 三种取值: *None* (关闭, 返回 ``None``); 一个输出目录 (str /
    PathLike), 就地建一个记录器并交给本函数关闭; 或一个已有的记录器实例, 其
    生命周期由调用方负责。记录器的导入是**函数内**的 —— ``optimizer.wfless``
    包的 ``__init__`` 会 import 本模块, 模块级导入会成环。
    """
    if debug is None:
        return None, False
    if isinstance(debug, (str, os.PathLike)):
        from ao_shaping.optimizer.wfless.step_b_debug import StepBDebugRecorder

        return (
            StepBDebugRecorder(
                os.fspath(debug), context={"planned_iterations": int(iterations)}
            ),
            True,
        )
    return debug, False



def _shaping_epoch_loop(_closure, base_lr, batched, debug_recorder, iterations, loss_history, lr_schedule, opt, optimizer, per_holder, phase, phase_callback, progress_callback, torch):
# Per-iteration shaping loop lifted verbatim
    for it in range(iterations):
        t0 = time.perf_counter()
        phase_before = phase.detach().clone() if debug_recorder is not None else None
        if optimizer == "adam":
            if lr_schedule is not None:
                factor = _lr_schedule_factor(lr_schedule, it, iterations)
                for group in opt.param_groups:
                    group["lr"] = base_lr * factor
            opt.zero_grad()
            loss = _closure()
            forward_s = time.perf_counter() - t0
            if torch.isnan(loss):
                logger.warning("NaN loss at iteration {}; stopping early", it)
                break
            opt.step()
        else:  # lbfgs —— 闭包由优化器内部调用
            loss = opt.step(_closure)
            forward_s = time.perf_counter() - t0
            if loss is None or torch.isnan(loss):
                logger.warning(
                    "L-BFGS returned NaN/None at iteration {}; stopping early", it
                )
                break
        step_s = (time.perf_counter() - t0) - forward_s

        per = per_holder["per"]
        if per is not None:
            per_d = per.detach()
            best_idx = int(per_d.argmin().item())
            # 上报/记录最优候选的损失, 不是 K 个损失之和 (后者只是反向用的)。
            loss_val = float(per_d[best_idx].item())
        else:
            best_idx = 0
            loss_val = float(loss.item())
        # 记录在相位包裹之前: 包裹会把 (2π) 跳变写进 ‖phase_after-phase_before‖,
        # 让这一步的更新范数读起来像一次猛烈跳变, 而不是 Adam 的真实步长。
        # 张量在这里就转成 numpy: 记录器是纯 numpy 叶子模块, 且 CUDA 张量不能直接
        # 交给 np.asarray。批量模式下记录器只跟踪当前最优候选 (2D)。
        if debug_recorder is not None:

            def _slice(t, _best: int = best_idx):
                return t[_best] if batched else t

            debug_recorder.record(
                iteration=it,
                loss=loss_val,
                grad=(
                    None
                    if phase.grad is None
                    else _slice(phase.grad.detach().float().cpu().numpy())
                ),
                phase_before=(
                    None
                    if phase_before is None
                    else _slice(phase_before.detach().float().cpu().numpy())
                ),
                phase_after=_slice(phase.detach().float().cpu().numpy()),
                forward_s=float(forward_s),
                step_s=float(step_s),
            )

        # 把相位包裹到 [0, 2π)。对 Adam 来说每步都做是安全的; 对
        # L-BFGS 来说, 运行中包裹会破坏它的梯度历史记账, 所以只在
        # 最后才包裹。
        if optimizer == "adam":
            with torch.no_grad():
                phase.data = phase.data % (2.0 * math.pi)

        loss_history.append(loss_val)

        if progress_callback is not None:
            progress_callback(it, loss_val)

        if phase_callback is not None:
            phase_callback(phase.detach()[best_idx] if batched else phase.detach())

        # 每个迭代输出一次进度日志 (library 层为 debug 级别, 由 DEBUG=1 控制;
        # runner 层通过 progress_callback 在 INFO 级别逐迭代输出)
        logger.debug("Iter {}/{}  loss={:.6f}", it + 1, iterations, loss_val)



def _run_shaping_loop(
    torch: Any,
    phase: Any,
    opt: Any,
    forward: Any,
    amp: Any,
    t_tensor: Any,
    *,
    propagation: str,
    optimizer: str,
    iterations: int,
    device: str,
    w_uniformity: float,
    w_efficiency: float,
    w_zero_order: float,
    w_smoothness: float,
    progress_callback: Callable[[int, float], None] | None,
    phase_callback: Callable[["Tensor"], None] | None,
    debug_recorder: Any = None,
    base_lr: float = 0.0,
    lr_schedule: str | None = None,
) -> tuple[list[float], bool]:
    """Run the optimization loop; return the loss history and the converged flag.

    ``lr_schedule is not None`` 时, 每步按 :func:`_lr_schedule_factor` 把 Adam
    ``param_groups`` 的学习率设为 ``base_lr · factor``; ``None`` 时不触碰参数组。

    ``debug_recorder`` (default ``None``) is a
    :class:`~ao_shaping.optimizer.wfless.step_b_debug.StepBDebugRecorder`; when
    given, every iteration also records the gradient statistics, the parameter
    update norm and the forward/step timing split. It is duck-typed on purpose:
    the recorder lives under ``optimizer.wfless``, which imports this module, so
    naming the class here would cycle.
    """
    loss_history: list[float] = []

    # ``phase`` 是 ``(H, W)`` (单候选) 或 ``(K, H, W)`` (模型内并行的 K 候选)。
    # 批量模式下一次前向/反向同时算 K 个候选, 逐候选损失只用于挑最优候选与
    # 记录; 逐候选损失由 :func:`total_loss_per_candidate` 逐候选调用
    # :func:`total_loss` 得到, 与单候选路径逐位一致。
    batched = phase.ndim == 3
    per_holder: dict[str, Any] = {"per": None}

    def _closure():
        """Adam 与 L-BFGS 共用的单次前向 + 反向传播。"""
        opt.zero_grad()
        intensity = forward(phase, amp)
        if batched:
            per = total_loss_per_candidate(
                intensity,
                phase,
                t_tensor,
                w_uniformity=w_uniformity,
                w_efficiency=w_efficiency,
                w_zero_order=w_zero_order,
                w_smoothness=w_smoothness,
            )
            per_holder["per"] = per
            # 各候选损失相互独立, 求和后的梯度与逐候选求和逐位一致。
            loss = per.sum()
        else:
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

    converged = False
    logger.info(
        "Starting differentiable beam shaping: {} iterations, "
        "propagation={}, optimizer={}, device={}",
        iterations,
        propagation,
        optimizer,
        device,
    )
    if lr_schedule is not None and optimizer != "adam":
        logger.warning("lr_schedule={!r} 只对 Adam 生效, 优化器 {} 忽略该调度",
                       lr_schedule, optimizer)

    _shaping_epoch_loop(_closure, base_lr, batched, debug_recorder, iterations, loss_history, lr_schedule, opt, optimizer, per_holder, phase, phase_callback, progress_callback, torch)

    if len(loss_history) >= iterations:
        converged = True

    return loss_history, converged


def _extract_shaping_result(
    torch: Any,
    phase: Any,
    forward: Any,
    amp: Any,
    target: "npt.NDArray[np.floating] | Tensor",
    loss_history: list[float],
    converged: bool,
    w_uniformity: float = 0.4,
    w_efficiency: float = 0.6,
    w_zero_order: float = 0.0,
    w_smoothness: float = 0.0,
) -> DifferentiableShapingResult:
    """Evaluate the final phase and assemble the result record.

    ``phase`` 为 ``(H, W)`` 时按原样返回; 为 ``(K, H, W)`` (模型内并行) 时
    以与优化时相同的目标函数逐候选评估, 返回损失最小的那个候选。
    """
    with torch.no_grad():
        final_intensity = forward(phase, amp)
        if phase.ndim == 3:
            if isinstance(target, np.ndarray):
                target_t = torch.from_numpy(target).to(
                    device=phase.device, dtype=phase.dtype
                )
            else:
                target_t = target.to(device=phase.device, dtype=phase.dtype)
            per = total_loss_per_candidate(
                final_intensity,
                phase,
                target_t,
                w_uniformity=w_uniformity,
                w_efficiency=w_efficiency,
                w_zero_order=w_zero_order,
                w_smoothness=w_smoothness,
            )
            best_idx = int(per.argmin().item())
            selected = phase.detach()[best_idx]
        else:
            best_idx = 0
            selected = phase.detach()
        final_phase = (
            (selected % (2.0 * math.pi))
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
        if phase.ndim == 3:
            final_intensity_np = final_intensity_np[best_idx]
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



def _tbs_prepare_candidates(device, grid_size, initial_phase, iterations, n_candidates, optimizer, propagation, seed, source_amplitude, target, torch):
# Candidate normalization and phase batch setup lifted verbatim
    H, W = _validate_shaping_inputs(
        target, grid_size, propagation, optimizer, iterations
    )

    if n_candidates < 1:
        raise ValueError(f"n_candidates must be >= 1, got {n_candidates}")
    if n_candidates > 1 and optimizer != "adam":
        # L-BFGS 的线搜索会把 K 个候选的梯度混在一起, 模型内并行对它无意义。
        logger.warning(
            "n_candidates={} 仅对 Adam 优化器生效; 优化器 {!r} 回退到单候选",
            n_candidates,
            optimizer,
        )
        n_candidates = 1

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
    t_tensor, amp, phase_init = _shaping_tensors(
        torch, target, source_amplitude, initial_phase, H, W, dev, dtype
    )

    if n_candidates == 1:
        phase = torch.nn.Parameter(phase_init.clone())
    else:
        if initial_phase is None:
            # 每候选独立小噪声起点: 模型内并行的价值就是多个不同起点的候选,
            # 跑完取最优。
            phase_batch = 0.1 * torch.randn(
                (n_candidates, H, W), device=dev, dtype=dtype
            )
        else:
            # 显式初相位 + 逐候选微扰: 保留用户起点, 同时保留候选多样性。
            phase_batch = phase_init.unsqueeze(0) + 0.1 * torch.randn(
                (n_candidates, H, W), device=dev, dtype=dtype
            )
        phase = torch.nn.Parameter(phase_batch)
    return amp, device, phase, t_tensor



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
    lr_schedule: str | None = None,
    w_uniformity: float = 0.4,
    w_efficiency: float = 0.6,
    w_zero_order: float = 0.0,
    w_smoothness: float = 0.0,
    cell_spacing: float = SLM_PITCH_M,
    distance: float = 0.1,
    wavelength: float = LASER_WAVELENGTH_M,
    device: str | None = None,
    seed: int | None = None,
    progress_callback: Callable[[int, float], None] | None = None,
    phase_callback: Callable[["Tensor"], None] | None = None,
    debug: Any = None,
    n_candidates: int = 8,
) -> DifferentiableShapingResult:
    """用梯度下降优化 SLM 相位图。

    Args:
        target: ``(H, W)`` 目标强度掩码 (numpy 或 torch)。
        grid_size: ``(H, W)`` 网格尺寸。
        source_amplitude: 照明振幅。为 *None* 时取均匀值。
        initial_phase: 起始相位 (弧度)。为 *None* 时取 ``0.1 * randn``(标准差 0.1 的
        高斯噪声), **不是**全零 —— 零相位是 ASM 逃不出的临界点, 见
        :func:`_shaping_tensors` 里的初始化注释。
        propagation: ``"fft"`` (夫琅禾费) 或 ``"asm"`` (角谱)。
        optimizer: ``"adam"`` 或 ``"lbfgs"``。
        iterations: 优化步数。
        lr: 学习率。
        lr_schedule: 动态学习率调度 (``None`` = 恒定, 与历史逐位一致):
            ``"cosine"`` / ``"step"`` / ``"warmup-cosine"``, 仅对 Adam 生效。
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
        debug: 每步诊断记录, ``None`` 关闭 (默认)。传一个
            :class:`~ao_shaping.optimizer.wfless.step_b_debug.StepBDebugRecorder`
            实例时, 生命周期由调用方负责 (自己 ``close()``); 传目录路径时本函数
            自建记录器, 跑完写好 ``*_debug.jsonl`` / ``*_debug.json`` 再关闭。
            每次迭代记下正向传输的损失、相位梯度范数/最大值/RMS、参数更新范数,
            以及 ``forward+backward`` 与优化器步各自的耗时 —— 用来判断该提速的是
            物理预测模型还是优化器。
        n_candidates: 模型内并行的候选数, 默认 8。每次迭代用**一次**批量前向/反向
            同时优化 K 个候选, 每步报告最优候选的损失, 最终返回最优候选。
            候选起点: ``initial_phase`` 为 *None* 时每候选独立 ``0.1·randn``;
            显式给出时每候选在其上叠加 ``0.1·randn`` 微扰以保留候选多样性。
            ``n_candidates=1`` 与历史单候选路径逐位一致; 该能力仅对
            ``optimizer="adam"`` 生效, 其它优化器回退为 1。

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

    amp, device, phase, t_tensor = _tbs_prepare_candidates(device, grid_size, initial_phase, iterations, n_candidates, optimizer, propagation, seed, source_amplitude, target, torch)

    # -- 物理预测模型 ----------------------------------------------------
    forward = _build_forward(propagation, cell_spacing, distance, wavelength)

    # -- 优化器 ------------------------------------------------------
    # 调度名的校验在 _shaping_optimizer 里做 (单点, 对任何优化器都生效)。
    opt = _shaping_optimizer(torch, phase, optimizer, lr, lr_schedule)

    # -- 训练循环 ----------------------------------------------------
    recorder, owns_recorder = _resolve_debug_recorder(debug, iterations)
    try:
        loss_history, converged = _run_shaping_loop(
            torch,
            phase,
            opt,
            forward,
            amp,
            t_tensor,
            propagation=propagation,
            optimizer=optimizer,
            iterations=iterations,
            device=device,
            w_uniformity=w_uniformity,
            w_efficiency=w_efficiency,
            w_zero_order=w_zero_order,
            w_smoothness=w_smoothness,
            progress_callback=progress_callback,
            phase_callback=phase_callback,
            debug_recorder=recorder,
            base_lr=lr,
            lr_schedule=lr_schedule,
        )
    finally:
        # 自己建的记录器自己收尾 (写汇总 JSON); 调用方传进来的不动。
        if owns_recorder:
            recorder.close()

    # -- 提取结果 ----------------------------------------------------
    return _extract_shaping_result(
        torch,
        phase,
        forward,
        amp,
        target,
        loss_history,
        converged,
        w_uniformity=w_uniformity,
        w_efficiency=w_efficiency,
        w_zero_order=w_zero_order,
        w_smoothness=w_smoothness,
    )
