"""供硬件优化循环使用的共享黑盒启发式搜索驱动。

本包里的启发式优化器都是**最小化**自己的适应度函数, 而硬件目标的方向并不
一致 (PIB 是最大化, RMS 是最小化)。本驱动把那些否则会被各优化器重复实现的东西
集中起来:

* ``算法名 -> OptimizerType`` 的映射 (``"spgd"`` 被排除在外 -- 那是梯度方法,
  由调用方自己的循环处理),
* ``maximize`` 的符号处理 (适应度 = ±value),
* 边界裁剪, 让硬件永远收不到越界向量,
* 每次求值后的回调 (记录 / 实时显示),
* 协作式提前中止 (例如用户关掉了实时的 pygame 窗口)。

这是算法层的叶子模块: 它不导入任何硬件, 也不导入任何优化器。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import numpy.typing as npt

from ao_shaping.algorithm.heuristic.heuristic_base import (
    HeuristicOptimizer,
    OptimizerType,
)

#: 启发式算法名 (不含 ``"spgd"``) -> 优化器类型。
HEURISTIC_ALGORITHM_MAP: dict[str, OptimizerType] = {
    "ga": OptimizerType.GA,
    "pso": OptimizerType.PSO,
    "sa": OptimizerType.SA,
    "hc": OptimizerType.HILL_CLIMBING,
    "rs": OptimizerType.RANDOM_SEARCH,
    "cem": OptimizerType.CROSS_ENTROPY,
    "de": OptimizerType.DIFFERENTIAL_EVOLUTION,
}

#: 接受种群规模的启发式算法 (PSO 把同一个旋钮叫作 ``n_particles``)。
POPULATION_ALGORITHMS: frozenset[OptimizerType] = frozenset(
    {
        OptimizerType.GA,
        OptimizerType.PSO,
        OptimizerType.CROSS_ENTROPY,
        OptimizerType.DIFFERENTIAL_EVOLUTION,
    }
)


class SearchAborted(RuntimeError):
    """由 ``should_stop`` 回调抛出, 用于提前结束启发式搜索。"""


@dataclass
class HeuristicSearchResult:
    """:func:`run_heuristic_search` 的结果。

    Attributes:
        best_x: 找到的最佳参数向量 (已裁剪到搜索边界内)。
        best_value: ``best_x`` 的原始目标值 (按调用方的方向)。
        evaluations: 执行过的目标函数求值次数。
        history: 每次求值的原始目标值, 按顺序排列。
    """

    best_x: npt.NDArray[np.float64]
    best_value: float
    evaluations: int
    history: list[float] = field(default_factory=list)


def heuristic_algorithm_choices(include_spgd: bool = True) -> tuple[str, ...]:
    """返回各优化器接受的算法名 (``spgd`` 在最前)。"""
    names = tuple(HEURISTIC_ALGORITHM_MAP)
    return ("spgd", *names) if include_spgd else names


def create_heuristic(
    optimizer_type: OptimizerType,
    dim: int,
    iterations: int,
    bounds: tuple[float, float],
    seed: int | None = None,
    pop_size: int | None = None,
) -> HeuristicOptimizer:
    """经由共享工厂构造一个启发式优化器。

    Args:
        optimizer_type: 启发式选择器 (GA/PSO/SA/HC/RS/CEM/DE)。
        dim: 问题维度。
        iterations: 代数 / 迭代次数 (``n_iterations``)。
        bounds: 每个参数的 ``(低, 高)`` 搜索边界。
        seed: 可选的随机种子, 用于可复现性。
        pop_size: GA/PSO/CEM/DE 可选的种群规模 (SA/HC/RS 不接受, 会被忽略)。

    Returns:
        一个可直接运行的 :class:`HeuristicOptimizer` (调用 ``.optimize(fn, x0)``)。
    """
    factory_kwargs: dict = {
        "n_iterations": max(1, int(iterations)),
        "bounds": (float(bounds[0]), float(bounds[1])),
        "seed": seed,
    }
    if pop_size is not None:
        if optimizer_type is OptimizerType.PSO:
            factory_kwargs["n_particles"] = int(pop_size)
        elif optimizer_type in POPULATION_ALGORITHMS:
            factory_kwargs["pop_size"] = int(pop_size)
    return HeuristicOptimizer.create(optimizer_type, dim=dim, **factory_kwargs)


def run_heuristic_search(
    algorithm: str,
    evaluate: Callable[[npt.NDArray[np.float64]], float],
    dim: int,
    iterations: int,
    bounds: tuple[float, float],
    x0: npt.NDArray[np.float64] | None = None,
    maximize: bool = False,
    seed: int | None = None,
    pop_size: int | None = None,
    on_evaluate: Callable[[npt.NDArray[np.float64], float, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> HeuristicSearchResult:
    """用指定名字的启发式算法驱动 ``evaluate``。

    ``evaluate`` 必须返回**原始**目标值 (按调用方自己的单位/方向); 当
    ``maximize=True`` 时, 本驱动把 ``-value`` 交给 (做最小化的) 启发式, 从而
    搜索它的最大值。

    Args:
        algorithm: :data:`HEURISTIC_ALGORITHM_MAP` 中的一个 (大小写不敏感)。
        evaluate: 黑盒目标函数; 收到的是已裁剪到边界内的向量。
        dim: 问题维度 (例如 Zernike 项数)。
        iterations: 启发式迭代数/代数 (调用方把它们的 ``epochs`` 映射到这里;
            注意种群类方法每一代会做 ``pop_size`` 次求值)。
        bounds: 每个参数的 ``(低, 高)`` 裁剪/搜索边界。
        x0: 可选的初始向量 (GA 会把它纳入种群, 局部方法则用它作起点)。
        maximize: 为 True 时按 ``evaluate`` 上升搜索 (PIB 类), 为 False 时下降
            搜索 (RMS 类)。
        seed: 可选的随机种子。
        pop_size: 可选的种群规模 (GA/PSO/CEM/DE)。
        on_evaluate: 可选的 ``(x, value, evaluation_index)`` 回调, 在每次求值
            之后触发 -- 用它来追加 Recorder 行 / 刷新实时显示。
            ``evaluation_index`` 从 1 开始。
        should_stop: 可选的判定式; 它返回 True 时搜索中止并返回当前最好的结果
            (不抛异常)。

    Returns:
        HeuristicSearchResult, 含最佳向量、它的原始值以及历史。

    Raises:
        ValueError: 未知的 ``algorithm``。
    """
    key = str(algorithm).lower()
    if key not in HEURISTIC_ALGORITHM_MAP:
        raise ValueError(
            f"Unknown heuristic algorithm {algorithm!r}. "
            f"Available: {sorted(HEURISTIC_ALGORITHM_MAP)}"
        )

    lo, hi = float(bounds[0]), float(bounds[1])
    optimizer = create_heuristic(
        HEURISTIC_ALGORITHM_MAP[key], dim, iterations, (lo, hi), seed, pop_size
    )

    evaluations = {"n": 0}
    history: list[float] = []
    best_value: list[float | None] = [None]
    best_x: list[npt.NDArray[np.float64] | None] = [None]

    def fitness(x) -> float:
        x_clipped = np.clip(np.asarray(x, dtype=np.float64), lo, hi)
        value = float(evaluate(x_clipped))

        evaluations["n"] += 1
        history.append(value)
        if best_value[0] is None or (
            value > best_value[0] if maximize else value < best_value[0]
        ):
            best_value[0] = value
            best_x[0] = x_clipped.copy()

        if on_evaluate is not None:
            on_evaluate(x_clipped, value, evaluations["n"])
        if should_stop is not None and should_stop():
            raise SearchAborted

        return -value if maximize else value

    try:
        optimizer.optimize(fitness, init_x=x0)
    except SearchAborted:
        # 已中止 (例如实时窗口被关掉): 在停止检查之前, 当前最好的结果就已经在
        # ``fitness`` 里记录过了, 所以直接落到下面。
        pass

    if best_x[0] is None:
        raise RuntimeError(f"heuristic {algorithm!r} performed no objective evaluation")

    return HeuristicSearchResult(
        best_x=best_x[0],
        best_value=float(best_value[0]),  # type: ignore[arg-type]
        evaluations=evaluations["n"],
        history=history,
    )
