"""可微整形的动态学习率调度（纯数学, 不依赖 torch）。

:func:`_lr_schedule_factor` 把"当前迭代的学习率倍率"算成一个纯函数, 由
``differentiable_shaping._run_shaping_loop`` 逐迭代写回 Adam 的
``param_groups["lr"]``。``lr_schedule=None`` 表示恒定学习率, 与历史行为逐位一致。

调度定义的先例来自 :mod:`ao_shaping.algorithm.signal_processing.iterative_zernike_shaping`
的余弦衰减。
"""

from __future__ import annotations

import math

__all__ = ["LR_SCHEDULES", "_lr_schedule_factor"]

#: 支持的动态学习率调度。``None`` 表示恒定学习率 (与历史行为逐位一致)。
LR_SCHEDULES: tuple[str, ...] = ("cosine", "step", "warmup-cosine")


def _lr_schedule_factor(schedule: str, iteration: int, iterations: int) -> float:
    """返回第 ``iteration`` 步 (0 基) 的学习率倍率。

    - ``"cosine"``: 余弦衰减到 0, 与 ``iterative_zernike_shaping`` 的先例同型:
      ``0.5 * (1 + cos(π * i / max(iters, 1)))``, 第 0 步即 ``base_lr``。
    - ``"step"``: 每 ``⌈iters/3⌉`` 步减半。
    - ``"warmup-cosine"``: 前 10 % 的迭代线性升温到 ``base_lr``, 之后余弦
      衰减到 ``0.1 · base_lr``。

    Args:
        schedule: 调度名, 必须取自 :data:`LR_SCHEDULES`。
        iteration: 当前迭代序号 (0 基)。
        iterations: 总迭代数。

    Returns:
        应用于 ``base_lr`` 的倍率, 恒为正。

    Raises:
        ValueError: ``schedule`` 不在 :data:`LR_SCHEDULES` 中。
    """
    if schedule not in LR_SCHEDULES:
        raise ValueError(
            f"未知学习率调度 {schedule!r}; 可选: {LR_SCHEDULES} 或 None"
        )
    if schedule == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * iteration / max(iterations, 1)))
    if schedule == "step":
        block = max(1, math.ceil(iterations / 3.0))
        return 0.5 ** (iteration // block)
    # warmup-cosine
    warmup = max(1, math.ceil(0.1 * iterations))
    if iteration < warmup:
        return (iteration + 1) / warmup
    remaining = max(iterations - 1 - warmup, 1)
    return 0.1 + 0.9 * 0.5 * (
        1.0 + math.cos(math.pi * (iteration - warmup) / remaining)
    )
