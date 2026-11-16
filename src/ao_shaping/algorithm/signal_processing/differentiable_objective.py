"""可微光束整形的目标函数族 (均匀度 / 效率 / 零级抑制 / 平滑度 的加权总损失)。

2026-11-16 从 :mod:`ao_shaping.algorithm.signal_processing.differentiable_shaping`
逐字拆出: 该模块越过 800 行棘轮上限, 于是把这一族只吃 torch 张量、只吐标量张量的
损失函数连同它们的节标题一起搬到这里 —— 函数体、docstring 与前后顺序均未改动。
老导入路径 (``differentiable_shaping.total_loss`` 等) 由父模块显式 re-export,
对外导入面保持不变。

``torch`` 与父模块一样是**可选**依赖: 本模块只在调用时经 :func:`_torch` 惰性取
torch, 导入期不碰 torch, 因此没装 PyTorch 的环境也能 ``import`` 本模块。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ao_shaping.algorithm._torch_support import require_torch

if TYPE_CHECKING:  # pragma: no cover – type-only imports
    from torch import Tensor

#: Floor below which a summed intensity counts as no energy at all: the ratios in
#: ``efficiency_loss`` and ``zero_order_penalty`` are undefined there rather than merely
#: large, so they return their neutral value instead of dividing. Deliberately distinct from
#: the ``1e-12`` in :func:`uniformity_loss`, which tests a *mean* for degeneracy.
_ZERO_ENERGY_FLOOR = 1e-12


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
# 损失函数 (torch 张量进, 标量张量出)
# ---------------------------------------------------------------------------


def uniformity_loss(intensity: "Tensor", target: "Tensor") -> "Tensor":
    """目标区域内强度的变异系数。

    值越低表示照明越均匀。目标区域为空或只含零时返回零 (对 NaN 安全)。
    """
    torch = _torch()
    # Resized target masks have a soft interpolation edge. Keep that edge in
    # the energy term, but measure CV on the intended half-height support.
    mask = target > 0.5
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
    if total < _ZERO_ENERGY_FLOOR:
        return torch.tensor(1.0, device=intensity.device, dtype=intensity.dtype)
    target_energy = (intensity * target).sum()
    encircled = target_energy / (total + _ZERO_ENERGY_FLOOR)
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
    if total < _ZERO_ENERGY_FLOOR:
        return torch.tensor(0.0, device=intensity.device, dtype=intensity.dtype)
    return region.mean() / (total / (H * W) + 1e-12)


def smoothness_regularization(phase: "Tensor") -> "Tensor":
    """对相位梯度施加的有限差分惩罚。

    相位图恒定时返回零, 并随高频空间变化增大。
    """
    _torch()
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
