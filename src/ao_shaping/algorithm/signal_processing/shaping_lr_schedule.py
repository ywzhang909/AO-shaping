"""可微整形的外部学习率调度因子。

``a60a87ca`` 给 :mod:`differentiable_shaping` 与 :mod:`differentiable_beam`
接上了逐迭代的 lr 调度, 但调度本体文件没有随那次提交入库; 本文件按已入库的
调用点与测试语义补交。``LR_SCHEDULES`` 的每个名字对应
:func:`_lr_schedule_factor` 的一种衰减形状, 由
``differentiable_shaping._run_shaping_loop`` 逐迭代乘到 ``base_lr`` 上;
:class:`DifferentiableBeamOptimizer.set_lr` 走同一条形状。
"""
from __future__ import annotations

import math

#: 可调度的学习率形状; ``None`` (恒定) 不在表内, 由调用点单独放行。
LR_SCHEDULES: tuple[str, ...] = ("cosine", "step", "warmup-cosine")

#: warmup-cosine 前段的线性预热占比。
_WARMUP_RATIO = 0.1
#: step 每过一个里程碑的倍率与最终地板。
_STEP_DECAY = 0.1
_STEP_MILESTONES = (0.5, 0.75)
_STEP_FLOOR = 0.01


def _lr_schedule_factor(schedule: str, index: int, total: int) -> float:
    """第 ``index`` 次迭代 (共 ``total`` 次) 的学习率倍率, 取值 ``(0, 1]``。

    Raises:
        ValueError: 若 ``schedule`` 不在 :data:`LR_SCHEDULES` 中。
    """
    if schedule not in LR_SCHEDULES:
        raise ValueError(f"未知学习率调度 {schedule!r}; 可选: {LR_SCHEDULES}")
    if total <= 1:
        return 1.0
    t = max(0, min(index, total - 1)) / (total - 1)
    if schedule == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * t))
    if schedule == "step":
        factor = 1.0
        for milestone in _STEP_MILESTONES:
            if t >= milestone:
                factor *= _STEP_DECAY
        return max(factor, _STEP_FLOOR)
    if t < _WARMUP_RATIO:
        return (t + 1.0 / (total - 1)) / _WARMUP_RATIO
    u = (t - _WARMUP_RATIO) / (1.0 - _WARMUP_RATIO)
    return 0.5 * (1.0 + math.cos(math.pi * u))
