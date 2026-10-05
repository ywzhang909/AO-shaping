"""
禁忌搜索 (Tabu Search) 算法模块

本模块为优化问题提供基于禁忌的自适应邻域搜索的通用实现。它被设计为领域无关的,
可用于任何需要跳出局部最优的优化问题。

模块包含:
- TabuMemory: 短期记忆, 避免重复访问已探索过的候选
- AdaptiveSearchState: 动态调整邻域探索半径
- 候选生成器: 生成搜索候选的方法
- TabuSearchRunner: 高层编排类

Usage:
    from ao_shaping.algorithm.tabu.tabu_search import TabuSearchRunner, TabuMemory, AdaptiveSearchState

    # 创建各组件
    tabu_memory = TabuMemory(capacity=128, quantization=2.0)
    search_state = AdaptiveSearchState(radius=2.0, min_radius=0.5, max_radius=12.0,
                                       expand_ratio=1.4, shrink_ratio=0.75, improvement_tol=1e-4)

    # 创建 runner
    runner = TabuSearchRunner(
        tabu_memory=tabu_memory,
        search_state=search_state,
        candidate_generator=my_candidate_generator,
        safety_check=my_safety_check,
    )

    # 运行搜索
    result = runner.run_search(anchor_v, anchor_value, evaluate_candidate)

Author: AO-Shaping Development Team
Created: 2026-03-26
"""

from collections import deque
from dataclasses import dataclass, field
from collections.abc import Callable

import numpy as np
import numpy.typing as npt


# =============================================================================
# 核心数据结构
# =============================================================================


@dataclass
class TabuMemory:
    """短期禁忌记忆, 用于避免重新探索次优候选。

    本类实现一种基于队列的禁忌记忆, 键为量化后的电压。它把最近探索过的候选的量化
    表示同时存入队列 (用于 FIFO 逐出) 与集合 (用于 O(1) 查找), 从而阻止算法重访
    这些候选。

    量化使禁忌记忆的粒度可以灵活调整。量化值越大, 越多的候选会被视为"相同"而
    被标记为禁忌。

    Attributes:
        capacity: 禁忌记忆最多保存的候选数量。
        quantization: 候选键的量化步长。值越大离散化越粗。

    Example:
        >>> tabu = TabuMemory(capacity=128, quantization=2.0)
        >>> voltages = np.array([10.5, 20.3, 15.7])
        >>> tabu.add(voltages)
        >>> tabu.contains(voltages)
        True
    """

    capacity: int
    quantization: float
    _queue: deque[tuple[int, ...]] = field(init=False, default_factory=deque)
    _keys: set[tuple[int, ...]] = field(init=False, default_factory=set)

    def make_key(self, voltages: npt.NDArray[np.float64]) -> tuple[int, ...]:
        """把电压数组量化为整数键, 供禁忌存储使用。

        本方法按如下步骤把电压数组转成整数元组:
        1. 转为 float64 以保证精度
        2. 除以量化尺度
        3. 四舍五入到最近整数

        Args:
            voltages: 待量化的电压数组。

        Returns:
            表示量化后电压分布的整数元组。
        """
        scale = max(float(self.quantization), 1e-6)
        return tuple(
            np.round(np.asarray(voltages, dtype=np.float64) / scale).astype(int)
        )

    def contains(self, voltages: npt.NDArray[np.float64]) -> bool:
        """检查某个电压分布是否已在禁忌记忆中。

        Args:
            voltages: 待检查的电压数组。

        Returns:
            量化后的电压在禁忌记忆中则返回 True, 否则返回 False。
            capacity <= 0 (禁忌禁用) 时返回 False。
        """
        if self.capacity <= 0:
            return False
        return self.make_key(voltages) in self._keys

    def add(self, voltages: npt.NDArray[np.float64]) -> None:
        """把一个电压分布加入禁忌记忆。

        若键已存在或 capacity <= 0, 本方法不做任何事。
        超出容量时逐出最早的条目 (FIFO)。

        Args:
            voltages: 要加入禁忌记忆的电压数组。
        """
        if self.capacity <= 0:
            return
        key = self.make_key(voltages)
        if key in self._keys:
            return
        self._queue.append(key)
        self._keys.add(key)
        while len(self._queue) > self.capacity:
            expired = self._queue.popleft()
            self._keys.discard(expired)


@dataclass
class AdaptiveSearchState:
    """自适应邻域搜索半径的状态管理。

    本类根据最近的搜索迭代是否带来改进, 管理搜索半径的动态调整。它实现如下
    自适应策略:
    - 发现改进时收缩半径 (利用)
    - 没有改进时扩张半径 (探索)

    半径被截断在 min_radius 与 max_radius 之间, 以防止退化行为。

    Attributes:
        radius: 当前搜索半径。
        min_radius: 允许的最小搜索半径。
        max_radius: 允许的最大搜索半径。
        expand_ratio: 无改进时扩张半径的倍率。
        shrink_ratio: 有改进时收缩半径的倍率。
        improvement_tol: 判定改进是否显著的容差。

    Example:
        >>> state = AdaptiveSearchState(radius=2.0, min_radius=0.5, max_radius=12.0,
        ...                            expand_ratio=1.4, shrink_ratio=0.75, improvement_tol=1e-4)
        >>> state.update_radius(improved=True)  # 收缩
        1.5
        >>> state.update_radius(improved=False)  # 扩张
        2.1
    """

    radius: float
    min_radius: float
    max_radius: float
    expand_ratio: float
    shrink_ratio: float
    improvement_tol: float

    def update_radius(self, improved: bool) -> float:
        """按改进状态更新搜索半径。

        Args:
            improved: 上一轮搜索迭代是否找到了改进。

        Returns:
            更新后的半径值 (已截断到 [min_radius, max_radius])。
        """
        if improved:
            next_radius = self.radius * self.shrink_ratio
        else:
            next_radius = self.radius * self.expand_ratio
        self.radius = float(np.clip(next_radius, self.min_radius, self.max_radius))
        return self.radius


# =============================================================================
# 候选生成
# =============================================================================


def generate_search_candidates(
    anchor_v: npt.NDArray[np.float64],
    radius_scale: float,
    n_samples: int,
    active_mask: npt.NDArray[np.bool_] | None = None,
    rng: np.random.Generator | None = None,
) -> list[npt.NDArray[np.float64]]:
    """在锚点附近生成稠密/稀疏混合的扰动。

    本函数通过扰动一个锚电压向量来生成候选解, 采用混合策略:
    - 一半候选: 高斯扰动 (稠密探索)
    - 一半候选: 稀疏均匀扰动 (稀疏探索)

    稀疏扰动只激活约 35% 的维度并赋予随机幅度, 提供了与稠密方法不同的探索模式。

    Args:
        anchor_v: 待扰动的锚电压向量。
        radius_scale: 高斯扰动的标准差, 也是均匀扰动的尺度。
        n_samples: 要生成的候选扰动数量。
        active_mask: 指示哪些维度需要扰动的二值掩码。
                    None 表示所有维度都激活。
        rng: 随机数生成器。None 表示使用 default_rng。

    Returns:
        候选电压向量的列表。

    Example:
        >>> anchor = np.zeros(64)
        >>> candidates = generate_search_candidates(anchor, 2.0, 8, rng=np.random.default_rng())
        >>> len(candidates)
        8
    """
    if rng is None:
        rng = np.random.default_rng()

    candidates: list[npt.NDArray[np.float64]] = []

    # 若提供了激活掩码则施加它
    if active_mask is not None:
        mask = np.asarray(active_mask, dtype=np.float64)
    else:
        mask = np.ones_like(anchor_v, dtype=np.float64)

    radius_scale = max(float(radius_scale), 1e-6)

    for sample_id in range(max(int(n_samples), 1)):
        # 稠密 (高斯) 与稀疏 (均匀) 扰动交替进行
        if sample_id % 2 == 0:
            # 稠密扰动: 高斯噪声
            perturbation = rng.normal(0.0, radius_scale, size=anchor_v.shape)
        else:
            # 稀疏扰动: 随机符号 + 随机幅度 + 稀疏激活
            signs = (
                rng.binomial(1, 0.5, size=anchor_v.shape).astype(np.float64) * 2.0 - 1.0
            )
            magnitudes = rng.uniform(
                radius_scale * 0.35, radius_scale, size=anchor_v.shape
            )
            sparse_mask = rng.binomial(1, 0.35, size=anchor_v.shape).astype(np.float64)
            perturbation = signs * magnitudes * sparse_mask

        # 施加掩码后加到锚点上
        candidates.append(anchor_v + perturbation * mask)

    return candidates


def should_trigger_search(
    epoch: int,
    enabled: bool,
    warmup: int,
    interval: int,
    patience: int,
    last_best_epoch: int,
) -> bool:
    """判断当前迭代是否应触发自适应搜索。

    本函数检查多个条件以确定是否触发禁忌搜索:
    1. 搜索必须已启用
    2. 必须已过预热期
    3. 必须处在正确的间隔上 (而非每一轮都触发)
    4. 自上次改进以来必须已超过耐心阈值

    Args:
        epoch: 当前优化迭代轮次。
        enabled: 自适应搜索是否启用。
        warmup: 首次搜索之前的最少迭代轮数。
        interval: 两次搜索触发之间的迭代间隔。
        patience: 触发搜索前允许无改进的迭代轮数。
        last_best_epoch: 上次改进所在的迭代轮次。

    Returns:
        应触发搜索则返回 True, 否则返回 False。

    Example:
        >>> should_trigger_search(epoch=500, enabled=True, warmup=200,
        ...                       interval=120, patience=100, last_best_epoch=350)
        True
    """
    if not enabled:
        return False
    if epoch < max(int(warmup), 1):
        return False
    if interval <= 0 or epoch % int(interval) != 0:
        return False
    return (epoch - last_best_epoch) >= max(int(patience), 0)


# =============================================================================
# 禁忌搜索 Runner
# =============================================================================


class TabuSearchRunner:
    """编排基于禁忌的自适应邻域搜索。

    本类提供运行带自适应邻域探索的禁忌搜索的高层接口, 集成:
    - 用于避免重复探索的禁忌记忆
    - 自适应半径管理
    - 候选生成
    - 安全检查
    - 候选评估

    该 runner 通过接受针对特定问题的操作 (候选生成、安全检查、评估) 的回调
    函数来保持领域无关。

    Attributes:
        tabu_memory: 用于跟踪已探索候选的禁忌记忆实例。
        search_state: 用于半径管理的自适应搜索状态。
        candidate_generator: 生成候选解的函数。
        safety_check: 可选的候选安全性校验函数。
        clip_bounds: 可选的电压截断 (min, max) 元组。

    Example:
        >>> def evaluate(voltages):
        ...     # 在这里写你的目标函数
        ...     return {"value": objective_value, "other": data}
        >>>
        >>> runner = TabuSearchRunner(
        ...     tabu_memory=TabuMemory(128, 2.0),
        ...     search_state=AdaptiveSearchState(2.0, 0.5, 12.0, 1.4, 0.75, 1e-4),
        ...     candidate_generator=generate_search_candidates,
        ...     safety_check=lambda v: np.all(np.abs(v) < 100),
        ...     clip_bounds=(-50, 50),
        ... )
        >>> result = runner.run_search(anchor, anchor_value, evaluate)
    """

    def __init__(
        self,
        tabu_memory: TabuMemory,
        search_state: AdaptiveSearchState,
        candidate_generator: Callable[
            [npt.NDArray[np.float64], float, int, npt.NDArray[np.float64] | None, np.random.Generator],
            list[npt.NDArray[np.float64]],
        ]
        | None = None,
        safety_check: Callable[[npt.NDArray[np.float64]], bool] | None = None,
        clip_bounds: tuple[float, float] | None = None,
    ):
        """初始化 TabuSearchRunner。

        Args:
            tabu_memory: 禁忌记忆实例。
            search_state: 自适应搜索状态实例。
            candidate_generator: 生成候选的函数。
                                None 表示使用默认的 generate_search_candidates。
            safety_check: 可选的候选安全性校验函数。
                         None 表示所有候选都视为安全。
            clip_bounds: 可选的电压截断 (min, max) 元组。
        """
        self.tabu_memory = tabu_memory
        self.search_state = search_state
        self.candidate_generator = candidate_generator or generate_search_candidates
        self.safety_check = safety_check
        self.clip_bounds = clip_bounds

    def run_search(
        self,
        anchor_v: npt.NDArray[np.float64],
        anchor_objective: float,
        evaluate_candidate: Callable[[npt.NDArray[np.float64]], dict],
        objective_key: str = "value",
        improvement_tol: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> dict | None:
        """运行一轮禁忌搜索。

        本方法:
        1. 在锚点附近生成候选解
        2. 过滤掉禁忌与不安全的候选
        3. 评估有效候选
        4. 选出带来改进的最佳候选
        5. 更新禁忌记忆与搜索半径
        6. 返回结果; 若没有进展则返回 None

        Args:
            anchor_v: 当前最优电压向量。
            anchor_objective: 锚点处的目标值。
            evaluate_candidate: 评估候选的函数, 返回至少包含 'objective_key'
                              与 'value' 键的 dict。
            objective_key: 评估 dict 中目标值所用的键名。
            improvement_tol: 所需的最小改进量。None 表示使用
                           search_state.improvement_tol。
            rng: 随机数生成器。None 表示使用 default_rng。

        Returns:
            若没有任何候选被评估 (全被拒绝或为空) 则返回 None。
            否则返回包含以下键的 dict:
                - accepted: bool - 是否接受了某个候选
                - voltages: np.ndarray - 最佳候选的电压
                - value: float - 最佳候选处的目标值
                - tabu_hits: int - 因禁忌而跳过的候选数
                - safe_rejects: int - 因安全性被拒的候选数
                - evaluated: int - 已评估的候选数
                - radius: float - 当前搜索半径
                - anchor: str - 锚点来源 ('best' 或 'current')
        """
        if rng is None:
            rng = np.random.default_rng()

        # 使用传入的容差, 或搜索状态中的默认值
        tol = (
            improvement_tol
            if improvement_tol is not None
            else self.search_state.improvement_tol
        )

        # 生成候选 - 用位置参数以保证兼容性
        if self.candidate_generator == generate_search_candidates:
            candidates = self.candidate_generator(
                anchor_v,
                self.search_state.radius,
                8,
                None,
                rng,
            )
        else:
            # 自定义生成器 - 先尝试关键字参数
            try:
                candidates = self.candidate_generator(
                    anchor_v=anchor_v,
                    radius_scale=self.search_state.radius,
                    n_samples=8,
                    active_mask=None,
                    rng=rng,
                )
            except TypeError:
                # 兜底: 位置参数
                candidates = self.candidate_generator(
                    anchor_v,
                    self.search_state.radius,
                    8,
                    None,
                    rng,
                )

        best_candidate: dict | None = None
        tabu_hits = 0
        safe_rejects = 0
        evaluated = 0

        for candidate in candidates:
            # 若指定了边界则施加截断
            if self.clip_bounds is not None:
                candidate = np.clip(candidate, self.clip_bounds[0], self.clip_bounds[1])

            # 检查禁忌
            if self.tabu_memory.contains(candidate):
                tabu_hits += 1
                continue

            # 检查安全性
            if self.safety_check is not None and not self.safety_check(candidate):
                safe_rejects += 1
                self.tabu_memory.add(candidate)
                continue

            # 评估候选
            candidate_eval = evaluate_candidate(candidate)
            evaluated += 1

            # 判断这是否算改进
            candidate_value = candidate_eval.get(
                objective_key, candidate_eval.get("value", 0)
            )
            improved = candidate_value > anchor_objective + tol

            if improved and (
                best_candidate is None
                or candidate_value
                > best_candidate.get(objective_key, best_candidate.get("value", 0))
            ):
                best_candidate = {
                    "voltages": candidate.copy(),
                    **candidate_eval,
                    objective_key: candidate_value,
                }
            else:
                self.tabu_memory.add(candidate)

        # 处理没有有效候选的情况
        if best_candidate is None:
            self.search_state.update_radius(improved=False)
            return {
                "accepted": False,
                "tabu_hits": tabu_hits,
                "safe_rejects": safe_rejects,
                "evaluated": evaluated,
                "radius": self.search_state.radius,
                "anchor": "best",
            }

        # 接受最佳候选
        self.tabu_memory.add(anchor_v)
        self.search_state.update_radius(improved=True)

        best_candidate.update(
            {
                "accepted": True,
                "tabu_hits": tabu_hits,
                "safe_rejects": safe_rejects,
                "evaluated": evaluated,
                "radius": self.search_state.radius,
                "anchor": "best",
            }
        )

        return best_candidate


# =============================================================================
# 工厂函数
# =============================================================================


def create_tabu_search_runner(
    capacity: int = 128,
    quantization: float = 2.0,
    initial_radius: float = 2.0,
    min_radius: float = 0.5,
    max_radius: float = 12.0,
    expand_ratio: float = 1.4,
    shrink_ratio: float = 0.75,
    improvement_tol: float = 1e-4,
    candidate_generator: Callable | None = None,
    safety_check: Callable[[npt.NDArray[np.float64]], bool] | None = None,
    clip_bounds: tuple[float, float] | None = None,
) -> TabuSearchRunner:
    """以默认参数创建 TabuSearchRunner 的工厂函数。

    这是一个便捷函数, 用合理的默认值创建全部所需组件。

    Args:
        capacity: 禁忌记忆容量。
        quantization: 禁忌记忆的量化步长。
        initial_radius: 初始搜索半径。
        min_radius: 最小搜索半径。
        max_radius: 最大搜索半径。
        expand_ratio: 半径扩张倍率。
        shrink_ratio: 半径收缩倍率。
        improvement_tol: 改进容差。
        candidate_generator: 自定义候选生成器, None 表示使用默认实现。
        safety_check: 自定义安全检查, None 表示不做检查。
        clip_bounds: 电压截断边界, 或 None。

    Returns:
        配置好的 TabuSearchRunner 实例。

    Example:
        >>> runner = create_tabu_search_runner(
        ...     capacity=128,
        ...     initial_radius=2.0,
        ...     safety_check=lambda v: np.all(np.abs(v) < 100),
        ... )
    """
    tabu_memory = TabuMemory(capacity=capacity, quantization=quantization)
    search_state = AdaptiveSearchState(
        radius=initial_radius,
        min_radius=min_radius,
        max_radius=max_radius,
        expand_ratio=expand_ratio,
        shrink_ratio=shrink_ratio,
        improvement_tol=improvement_tol,
    )

    return TabuSearchRunner(
        tabu_memory=tabu_memory,
        search_state=search_state,
        candidate_generator=candidate_generator,
        safety_check=safety_check,
        clip_bounds=clip_bounds,
    )
