"""可微光束整形的物理预测模型 (相位 → 传播 → 远场强度)。

2026-11-17 从 :mod:`ao_shaping.algorithm.signal_processing.differentiable_shaping`
逐字拆出: 该模块已到 800 行棘轮上限 (799 行), 而模型内并行 (一次前向/反向同时
优化 K 个候选) 需要在这里支持批量。传播模型与优化循环是两个独立关注点, 因此把
:func:`_asm_propagator_torch` / :func:`angular_spectrum_propagate_torch` /
:func:`_build_forward` 连同节标题搬到这里。老导入路径由父模块显式 re-export,
对外导入面保持不变。

**批量支持**: ``_build_forward`` 的输入相位可以是 ``(H, W)`` 或 ``(K, H, W)``。
FFT 分支只对最后两维做 ``fft2`` / ``fftshift`` (``dim=(-2, -1)``), 因此批量前导维
不会被平移; ASM 分支的传播子形状为 ``(H, W)``, 会对 ``(K, H, W)`` 自然广播。
对 2D 输入, 两条分支都与拆分前逐位一致。

``torch`` 与父模块一样是**可选**依赖: 本模块只在调用时经 :func:`_torch` 惰性取
torch, 导入期不碰 torch, 因此没装 PyTorch 的环境也能 ``import`` 本模块。
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from typing import TYPE_CHECKING

from ao_shaping.algorithm._torch_support import require_torch

if TYPE_CHECKING:  # pragma: no cover – type-only imports
    from torch import Tensor

__all__ = [
    "_asm_propagator_torch",
    "_build_forward",
    "angular_spectrum_propagate_torch",
]


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
        field: 复光场张量 ``(..., H, W)``; 支持 ``(K, H, W)`` 批量。
        dx: 像素间距, 单位米。
        z: 传播距离 (正 = 正向)。
        wavelength: 波长, 单位米。

    Returns:
        传播后的复光场, 形状与 *field* 相同。
    """
    torch = _torch()
    if field.ndim < 2:
        raise ValueError(f"Field must be at least 2D, got {field.ndim}D")
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
# 物理预测模型 (工厂 —— 避免导入期出现 nn.Module)
# ---------------------------------------------------------------------------


def _build_forward(
    propagation: str,
    cell_spacing: float,
    distance: float,
    wavelength: float,
) -> Callable[["Tensor", "Tensor | None"], "Tensor"]:
    """把可微物理预测模型构建成闭包。

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
            # dim=(-2, -1): 只平移空间维。批量前导维 (K,) 不得被 fftshift
            # 环绕, 否则候选之间会互相错位 (对 2D 输入与原行为逐位一致)。
            out_field = torch.fft.fftshift(torch.fft.fft2(field), dim=(-2, -1))
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
