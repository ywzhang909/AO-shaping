"""梯度优化器 (Adam 家族、加速器)。

子模块可按全路径直接导入, 例如::

    from ao_shaping.algorithm.gradient.adam import Adam, AdamW, AdaMOD
    from ao_shaping.algorithm.gradient.acceleration import _njit

本 ``__init__`` 刻意*不*重新导入子模块: 只有 docstring 的包 init 可以避免
循环导入级联 (子模块之间互相导入, 也会导入顶层 ``ao_shaping.algorithm`` 门面,
在这里急加载它们会让顶层 ``__init__`` 在初始化中途被执行)。所有公开名字改为
经由顶层 ``ao_shaping.algorithm`` 门面暴露。

工厂函数 :func:`make_optimizer` 使用**函数内延迟导入** (deferred import),
因此在模块加载时不会触发任何子模块, 也不会引发上述循环导入问题。
"""

from __future__ import annotations

from typing import Any

__all__ = ["make_optimizer"]


def make_optimizer(kind: Any, dim: int, lr: float = 1.0):
    """按名称构建一个梯度优化器。

    使用 :class:`~ao_shaping.algorithm.gradient.adam.Base` 的注册表查找;
    未识别的名称回退到 ``Adam`` (与 ``slm_gs_refine`` 原先的 ``_make_optimizer``
    行为一致), 这样调用方可以传递任意字符串都能得到可用的优化器。

    构造时尝试 ``cls(dim, lr=lr)``; 若目标类不接受 ``lr`` (历史别名类),
    回退到 ``cls(dim)``。该防御性回退与 :func:`ao_shaping.optimizer.constants.create_optimizer`
    不同 — 后者会报告被丢弃的参数而非静默回退。

    Args:
        kind: 优化器名 (大小写不敏感), 例如 ``"adam"`` / ``"sgd"`` /
            ``"adamw"`` / ``"adamod"`` / ``"muno`` / ``"munow"``。
        dim: 参数向量维度 (必须为正)。
        lr: 学习率 (默认 1.0)。

    Returns:
        构造出的优化器实例 (subclass of
        :class:`~ao_shaping.algorithm.gradient.adam.Base`)。
    """
    from ao_shaping.algorithm.gradient.adam import Base, Adam

    try:
        cls = Base._lookup(kind)
    except ValueError:
        cls = Adam
    try:
        return cls(dim, lr=lr)
    except TypeError:
        return cls(dim)
