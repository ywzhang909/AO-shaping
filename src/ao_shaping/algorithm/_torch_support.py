"""算法层共享的 torch 惰性导入与峰值归一化强度损失助手。

本模块把 ``algorithm/`` 下逐字重复的两类工具收敛到一处:

* **torch 惰性导入** —— 若干模块此前各写一份 ``try: import torch /
  except ImportError: raise ImportError(<模块专属文案>)``。导入逻辑搬到这里,
  各模块保留一个只传文案的薄包装 ``_torch()``, 于是错误文案仍带模块名,
  测试里对 ``module._torch`` 的 monkeypatch 也照旧生效。
* **峰值归一化强度损失** —— :class:`ZernikeCoefficientOptimizer` 与
  :class:`DifferentiableBeamOptimizer` 的 ``_intensity_loss`` 在数值上逐位
  相同 (前者 docstring 就是这么写的), 统一为 :func:`peak_normalized_loss`。

设计约束
--------
* **不做进程内缓存**: ``_torch()`` 的契约是"每次调用都确认 torch 可导入"。
  若干测试替换 ``builtins.__import__`` 来模拟缺失 torch; 一旦缓存了首次
  导入结果, 这些测试在 torch 已导入的进程里就再也触发不到 ImportError。
* **不在导入期 import torch**: 本模块只依赖 ``typing``, 因此"某模块不依赖
  torch"的门禁 (例如探针模块) 仍然成立。

``IterativeZernikeShapingOptimizer._torch`` 刻意不在此列: 它抛
:class:`RuntimeError` 而非 :class:`ImportError`, 且缓存导入结果 —— 属于不同的
契约, 由该模块自己的薄包装保留。
"""

from __future__ import annotations

from typing import Any


def require_torch(message: str) -> Any:
    """惰性导入 ``torch``; 缺失时抛出带 ``message`` 的 :class:`ImportError`。

    Args:
        message: 缺失 PyTorch 时抛出的完整错误文案 (含安装提示)。

    Returns:
        ``torch`` 顶层模块。

    Raises:
        ImportError: 若无法导入 PyTorch。
    """
    try:
        import torch as _t
    except ImportError:
        raise ImportError(message) from None
    return _t


def peak_normalized_loss(
    intensity: Any,
    target: Any,
    *,
    kind: str = "mse",
    eps: float = 1e-8,
) -> Any:
    """峰值归一化强度损失。

    把 ``intensity`` 按其自身峰值归一化到 1.0 (与已归一化的 ``target``
    匹配), 再取均方 (``"mse"``) 或平均绝对 (``"l1"``) 残差。float32 下的
    峰值归一化避免了能量归一化 (``sum``) 在 FFT 把能量集中到少数像素时
    所出现的下溢; ``+ eps`` 保护零峰值的情形。

    Args:
        intensity: 二维强度张量 (可微)。
        target: 二维峰值归一化的目标张量。
        kind: ``"mse"`` (默认) 或 ``"l1"``。
        eps: 归一化分母的保护项。

    Returns:
        标量损失张量, 对 ``intensity`` 可微。
    """
    torch = require_torch(
        "ao_shaping beam-shaping loss requires PyTorch. "
        "Install with: uv sync --extra ml"
    )
    normalized = intensity / (intensity.max() + eps)
    if kind == "l1":
        return torch.mean(torch.abs(normalized - target))
    return torch.mean((normalized - target) ** 2)
