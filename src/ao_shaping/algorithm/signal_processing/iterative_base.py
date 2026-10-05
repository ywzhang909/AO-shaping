"""迭代式、基于类的光束整形优化器的抽象基类。

同时拥有收敛循环的子类 (即单个 ``update()`` 步骤, 外加一个可选的、反复调用它的
``run()``) 可以继承 :class:`IterativeOptimizer`。这强制执行项目约定:

- ``__init__`` 校验所有参数并初始化状态 (迭代计数器、历史、最优值跟踪)。
- ``update()`` 执行**一**步。返回类型由子类自行决定 (例如相位图、损失值、
  结果 dataclass)。
- ``run()`` 是可选的, 驱动完整循环直至满足停止条件。

具体子类:

- :class:`PhaseWrapOptimizer` (``phase_wrap.py``)
- :class:`DifferentiableBeamOptimizer` (``differentiable_beam.py``)

本类仅供自行管理迭代状态的基于类的优化器使用。基于函数的优化器
(例如 GS、diff-shaping) 不继承它。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class IterativeOptimizer(ABC):
    """迭代式、基于类的光束整形优化器基类。

    Attributes:
        max_iterations: ``run()`` 最多执行的 ``update()`` 步数。
        convergence_history: 每一步之后记录的目标函数值。
        best_value: 迄今为止见到的最优 (最低) 目标函数值。
    """

    def __init__(self, max_iterations: int = 1000):
        """初始化迭代优化器。

        Args:
            max_iterations: ``run()`` 最多执行的更新步数, 必须是正整数。

        Raises:
            ValueError: 若 ``max_iterations`` 不是正整数。
        """
        if not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError(
                f"max_iterations must be a positive integer, got {max_iterations!r}"
            )
        self.max_iterations: int = max_iterations
        self._iteration: int = 0
        self._convergence_history: list[float] = []
        self._best_value: float = float("inf")

    # ------------------------------------------------------------------
    # 抽象 API
    # ------------------------------------------------------------------

    @abstractmethod
    def update(self) -> Any:
        """执行一步优化。

        子类必须:
        - 更新内部状态 (迭代计数器、最优值、历史)
        - 用本步的目标函数值调用 ``self._record(value)``, 使记账保持一致。

        Returns:
            本步的结果 (例如新的相位图, 或当前的目标函数值)。确切类型由实现决定。
        """
        ...  # pragma: no cover

    # ------------------------------------------------------------------
    # 具体的辅助方法
    # ------------------------------------------------------------------

    def run(self) -> Any:
        """运行完整的优化循环。

        反复调用 :meth:`update`, 直到达到 ``max_iterations`` 或
        :attr:`is_converged` 变为 ``True``。

        Returns:
            最后一次 ``update()`` 调用的结果。
        """
        result: Any = None
        for _ in range(self.max_iterations):
            result = self.update()
            if self.is_converged:
                break
        return result

    @property
    def convergence_history(self) -> list[float]:
        """每一步完成后记录的目标函数值。"""
        return list(self._convergence_history)

    @property
    def best_value(self) -> float:
        """迄今为止见到的最优 (最低) 目标函数值。"""
        return self._best_value

    @property
    def is_converged(self) -> bool:
        """迭代预算耗尽时为 ``True``。

        子类可重写此属性以实现更早的停止条件
        (例如平台期检测、梯度范数阈值)。
        """
        return self._iteration >= self.max_iterations

    # ------------------------------------------------------------------
    # 内部记账 (供子类在 ``update`` 内使用)
    # ------------------------------------------------------------------

    def _record(self, value: float) -> None:
        """记录一步的结果并更新记账状态。

        Args:
            value: 本步的目标函数值。
        """
        self._iteration += 1
        self._convergence_history.append(value)
        if value < self._best_value:
            self._best_value = value
