"""面向基于种群的启发式优化器的引导变异 (Guided Mutation, GM)。

本模块提供三部分内容:

* :class:`GMOperator` -- 算子接口, 以及两个核实现:
  :class:`RankGuidedMutation` 与 :class:`ValueFrequencyGuidedMutation`。
* :class:`GMOptimizerMixin` -- 启发式算法混入即可获得的通用 GM 接口,
  带来 ``use_gm`` / ``_gm_operator`` / ``apply_gm_if_enabled``。
* :func:`guided_mutation` -- 一个装饰器, 包装一次进化步, 并在其运行后立即把
  GM 子代插入种群。

同一个 :class:`GMOperator` 接口背后有两种彼此独立的 GM 定义, 因为
"引导变异" **没有公认的文献定义**:

``RankGuidedMutation``
    一种 *连续* 方案。父代按线性排名权重抽取, 随后对逐步收缩的坐标子集施加
    退火的高斯扰动, 其幅度也随之衰减。这是文献中最接近的对应物 (基于排名的
    自适应变异, 例如 Basak 2021, arXiv:2104.08842, 它自适应的是逐个体的变异
    *率*), 此处把它扩展到共享步长。
    **下面的闭式形式是本仓库自行选定的, 并非引自某篇文献的公式。**

``ValueFrequencyGuidedMutation``
    沿用刘辉等人的离散算子, *A universal feedback-based
    improvement strategy for wavefront-shaping algorithms*, Acta Photonica
    Sinica 52(6):0629002 (2023), doi:10.3788/gzxb20235206.0629002 -- 也就是
    本包中 ``add GM algorithm`` TODO 所指的论文。它维护一个取值频次表
    ``G``, 选出 ``N_G(k)`` 个引导单元, 并按该频次成比例地为它们抽取新的取值。
    此处把离散域改造到连续域的做法是逐维度重采样已记录的精英取值。

两个核都不保证精英保留: ``parent + noise`` 可能比父代更差。需要单调不退化的
调用方必须自己保留精英。GA 原生如此; DE/CEM/PSO 使用的 ``replace_worst``
合并也只会覆盖种群的尾部。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Literal, TypeVar

import numpy as np
import numpy.typing as npt


def _check_population(
    population: npt.NDArray[np.float64],
    fitness: npt.NDArray[np.float64],
    ranks: npt.NDArray[np.intp],
) -> None:
    """显式校验各核的前置条件。

    这些核是具体实现, 所以它们 (而非 mixin) 才是拒绝无法满足的输入的合适位置。

    Raises:
        ValueError: 若种群为空、不是二维, 或其 fitness/ranks 与之不匹配。
    """
    if population.ndim != 2:
        raise ValueError(f"population must be 2-D, got shape {population.shape}")
    rows = population.shape[0]
    if rows == 0:
        raise ValueError("population must contain at least one individual")
    if fitness.shape != (rows,):
        raise ValueError(
            f"fitness must have shape ({rows},), got {fitness.shape}"
        )
    if ranks.shape != (rows,):
        raise ValueError(f"ranks must have shape ({rows},), got {ranks.shape}")


class GMOperator(ABC):
    """引导变异核的接口。

    实现负责把已打分的种群转换为 ``n_offspring`` 个新候选。
    它们不得改动 ``population`` 或 ``fitness``。
    """

    @abstractmethod
    def __call__(
        self,
        population: npt.NDArray[np.float64],
        fitness: npt.NDArray[np.float64],
        ranks: npt.NDArray[np.intp],
        current_iter: int,
        n_offspring: int,
        bounds: tuple[float, float],
        rng: np.random.Generator,
    ) -> npt.NDArray[np.float64]:
        """生成引导变异的子代。

        Args:
            population: 形状为 ``(n, dim)`` 的种群数组。
            fitness: 逐个体的适应度, 形状 ``(n,)``; **越小越好**,
                与本包的最小化约定一致。
            ranks: 逐个体的排名, ``1`` 表示最优。
            current_iter: 从 0 开始的代数下标, 用于退火。
            n_offspring: 要产生的子代数量。
            bounds: 施加到每个子代上的标量 ``(low, high)`` 截断区间。
            rng: 随机数生成器 (绝不使用 numpy 全局 RNG)。

        Returns:
            形状为 ``(n_offspring, dim)`` 的数组, 已截断到 ``bounds`` 内。
        """
        raise NotImplementedError

    def reset(self) -> None:
        """清除跨代状态。无状态的核无需实现。"""


class RankGuidedMutation(GMOperator):
    """连续的排名加权引导变异 (本仓库自定的定义)。

    父代按线性排名权重 ``w_i = (N + 1 - r_i) / (0.5 * N * (N + 1))`` 抽取,
    因此排名 1 权重最大, 且权重之和为 1。每个子代扰动无放回抽取的 ``N_G(k)``
    个坐标, 其中被扰动的比例与高斯幅度都随代数衰减::

        fG(k)  = (g0 - g_end) * k ** (-1 / lam_g) + g_end
        N_G(k) = clip(round(dim * fG(k)), 1, dim)
        sigma(k) = sigma0 * (sigma_min / sigma0) ** (k / lam_s)

    Attributes:
        g0: 初始被扰动的坐标比例。
        g_end: 渐近的被扰动比例。
        lam_g: 被扰动比例的衰减常数。
        sigma0: 初始扰动幅度, 以边界跨度为比例。
        sigma_min: 扰动幅度的下限。
        lam_s: 幅度的衰减常数。
    """

    def __init__(
        self,
        g0: float = 0.10,
        g_end: float = 0.0025,
        lam_g: float = 250.0,
        sigma0: float = 0.10,
        sigma_min: float = 1e-3,
        lam_s: float = 50.0,
    ) -> None:
        """初始化该核。

        Args:
            g0: 初始被扰动的坐标比例, 取值在 ``(0, 1]``。
            g_end: 渐近的被扰动比例, 取值在 ``[0, g0]``。
            lam_g: 被扰动比例的衰减常数; 越大衰减越慢。
            sigma0: 初始幅度, 以边界跨度为比例。
            sigma_min: 幅度下限, 以边界跨度为比例。
            lam_s: 幅度的衰减常数; 越大衰减越慢。

        Raises:
            ValueError: 若调度参数自相矛盾。
        """
        if not 0.0 < g_end <= g0 <= 1.0:
            raise ValueError(f"require 0 < g_end <= g0 <= 1, got g_end={g_end}, g0={g0}")
        if not 0.0 < sigma_min <= sigma0:
            raise ValueError(
                f"require 0 < sigma_min <= sigma0, got {sigma_min}, {sigma0}"
            )
        if lam_g <= 0.0 or lam_s <= 0.0:
            raise ValueError(f"lam_g and lam_s must be > 0, got {lam_g}, {lam_s}")
        self.g0 = float(g0)
        self.g_end = float(g_end)
        self.lam_g = float(lam_g)
        self.sigma0 = float(sigma0)
        self.sigma_min = float(sigma_min)
        self.lam_s = float(lam_s)

    def _fraction(self, k: int) -> float:
        """返回第 ``k`` 代被扰动的坐标比例。"""
        if k <= 0:
            return self.g0
        return (self.g0 - self.g_end) * k ** (-1.0 / self.lam_g) + self.g_end

    def __call__(
        self,
        population: npt.NDArray[np.float64],
        fitness: npt.NDArray[np.float64],
        ranks: npt.NDArray[np.intp],
        current_iter: int,
        n_offspring: int,
        bounds: tuple[float, float],
        rng: np.random.Generator,
    ) -> npt.NDArray[np.float64]:
        low, high = float(bounds[0]), float(bounds[1])
        n, dim = population.shape
        _check_population(population, fitness, ranks)
        span = high - low

        # 线性排名权重: 排名 1 (最优) 获得最大权重。
        weights = (n + 1.0 - ranks) / (0.5 * n * (n + 1.0))
        total = weights.sum()
        weights = weights / total if total > 0 else np.full(n, 1.0 / n)

        parents = population[rng.choice(n, size=n_offspring, p=weights)].copy()

        k = max(int(current_iter), 0)
        n_guided = int(np.clip(round(dim * self._fraction(k)), 1, dim))
        sigma = self.sigma0 * (self.sigma_min / self.sigma0) ** (k / self.lam_s) * span

        for row in range(n_offspring):
            loci = rng.choice(dim, size=n_guided, replace=False)
            parents[row, loci] += rng.normal(0.0, sigma, size=n_guided)

        return np.clip(parents, low, high)


class ValueFrequencyGuidedMutation(GMOperator):
    """离散取值频次 GM, 已适配到连续域。

    沿用刘辉等人, Acta Photonica Sinica 52(6):0629002 (2023): 该论文维护一个
    取值频次表 ``G``, 并按该频次为选中的引导单元成比例地抽取新取值。

    此处采用的连续域适配: 把每个 *坐标* 视作一个"单元", 频次表逐坐标记录每个
    已记录精英取值的出现次数。引导单元按频次抽取 (因此历史上产生过优质个体的
    坐标被扰动得更频繁), 它们的新取值则从该坐标已记录的精英取值中重采样
    (取值频次采样), 再叠加一个退火到零的高斯抖动。

    由于它要重采样精英取值, 首次调用前必须先执行 :meth:`observe`
    (当尚未记录任何历史时, :meth:`__call__` 会自动执行, 用传入的种群作种子)。

    Attributes:
        elite_fraction: 被视为精英的种群比例。
        n_guided: 每个子代的引导单元数, 或 ``None`` 表示由代数下标经
            ``n_guided0`` 与 ``lam_g`` 推导。
        n_guided0: ``n_guided`` 为 None 时第 0 代的引导单元数。
        lam_g: 引导单元数的衰减常数。
        jitter: 加到重采样精英取值上的高斯抖动, 以边界跨度为比例。
        elite_decay: 遗忘某个已记录精英取值的概率, 避免该表无限增长。
    """

    def __init__(
        self,
        elite_fraction: float = 0.2,
        n_guided: int | None = None,
        n_guided0: int = 3,
        lam_g: float = 25.0,
        jitter: float = 0.01,
        elite_decay: float = 0.05,
    ) -> None:
        """初始化该核。

        Args:
            elite_fraction: 被记为精英的种群比例, 取值在 ``(0, 1]``。
            n_guided: 固定的引导单元数, 或 None 表示对其退火。
            n_guided0: 退火时第 0 代的引导单元数。
            lam_g: 引导单元数的衰减常数。
            jitter: 抖动幅度, 以边界跨度为比例。
            elite_decay: 每次调用遗忘某个已记录取值的概率。

        Raises:
            ValueError: 若参数超出范围。
        """
        if not 0.0 < elite_fraction <= 1.0:
            raise ValueError(
                f"elite_fraction must be in (0, 1], got {elite_fraction}"
            )
        if n_guided is not None and n_guided < 1:
            raise ValueError(f"n_guided must be >= 1 or None, got {n_guided}")
        if n_guided0 < 1:
            raise ValueError(f"n_guided0 must be >= 1, got {n_guided0}")
        if lam_g <= 0.0:
            raise ValueError(f"lam_g must be > 0, got {lam_g}")
        if jitter < 0.0:
            raise ValueError(f"jitter must be >= 0, got {jitter}")
        if not 0.0 <= elite_decay < 1.0:
            raise ValueError(f"elite_decay must be in [0, 1), got {elite_decay}")
        self.elite_fraction = float(elite_fraction)
        self.n_guided = n_guided
        self.n_guided0 = int(n_guided0)
        self.lam_g = float(lam_g)
        self.jitter = float(jitter)
        self.elite_decay = float(elite_decay)
        self._elite_values: dict[int, list[float]] = {}

    def reset(self) -> None:
        """遗忘所有已记录的精英取值。"""
        self._elite_values.clear()

    def observe(self, population: npt.NDArray[np.float64], fitness: npt.NDArray[np.float64]) -> None:
        """把本代的精英个体记入取值表。

        Args:
            population: 形状为 ``(n, dim)`` 的种群数组。
            fitness: 逐个体的适应度, 形状 ``(n,)``; 越小越好。
        """
        if population.size == 0:
            return
        n_elite = max(1, int(population.shape[0] * self.elite_fraction))
        elite_idx = np.argsort(fitness)[:n_elite]
        for row in elite_idx:
            for coord, value in enumerate(population[row]):
                bucket = self._elite_values.setdefault(coord, [])
                bucket.append(float(value))
                # 限制表的大小, 避免长时间运行把它撑到无界。
                cap = max(8, n_elite * 8)
                if len(bucket) > cap:
                    del bucket[: len(bucket) - cap]

    def _guided_count(self, dim: int, k: int) -> int:
        """返回第 ``k`` 代的引导单元数。"""
        if self.n_guided is not None:
            return int(np.clip(self.n_guided, 1, dim))
        k = max(int(k), 1)
        scaled = self.n_guided0 * k ** (-1.0 / self.lam_g)
        return int(np.clip(round(scaled), 1, dim))

    def __call__(
        self,
        population: npt.NDArray[np.float64],
        fitness: npt.NDArray[np.float64],
        ranks: npt.NDArray[np.intp],
        current_iter: int,
        n_offspring: int,
        bounds: tuple[float, float],
        rng: np.random.Generator,
    ) -> npt.NDArray[np.float64]:
        low, high = float(bounds[0]), float(bounds[1])
        span = high - low
        n, dim = population.shape
        _check_population(population, fitness, ranks)
        if n_offspring <= 0:
            return np.empty((0, dim), dtype=np.float64)

        # 排名加权的父代, 与连续核使用同样的权重, 使两个算子利用同样的选择压力。
        weights = (n + 1.0 - ranks) / (0.5 * n * (n + 1.0))
        total = weights.sum()
        weights = weights / total if total > 0 else np.full(n, 1.0 / n)

        if not self._elite_values:
            self.observe(population, fitness)
            self.elite_decay = 0.0  # 仅用于播种; 真正的遗忘从下一次调用开始

        children = population[rng.choice(n, size=n_offspring, p=weights)].copy()

        # 已带有精英取值记录的单元就是"引导"单元。
        recorded = [c for c in range(dim) if self._elite_values.get(c)]
        if not recorded:
            return children
        n_guided = min(self._guided_count(dim, current_iter), len(recorded))

        for row in range(n_offspring):
            coords = rng.choice(recorded, size=n_guided, replace=False)
            for coord in coords:
                pool = self._elite_values[coord]
                children[row, coord] = rng.choice(pool)
                if self.jitter > 0.0:
                    children[row, coord] += rng.normal(
                        0.0, self.jitter * span
                    )

        if self.elite_decay > 0.0:
            for coord, pool in self._elite_values.items():
                keep = [v for v in pool if rng.random() >= self.elite_decay]
                if keep:
                    self._elite_values[coord] = keep

        return np.clip(children, low, high)


class GMOptimizerMixin:
    """面向基于种群的优化器的引导变异契约。

    具体优化器通过混入本类、声明 ``_gm_operator`` 支持并实现下面五个钩子来加入
    GM 家族。除开关之外 mixin 自身不持有任何状态, 因此它可以和
    :class:`HeuristicOptimizer` 自由组合, 而不影响 MRO 或构造函数。

    这五个钩子就是全部契约。它们声明为 ``NotImplementedError`` 而非提供基于
    ``getattr`` 的兜底实现, 因为静默取默认值的钩子正是 GM 算子最终改到错误
    种群上的原因:

    ``_gm_population()``
        当前种群, 形状为 ``(n, dim)`` 的数组。
    ``_gm_fitness()``
        对应的适应度数组, 形状 ``(n,)``, **越小越好**。
    ``_gm_iteration()``
        从 0 开始的代数下标, 供各核退火使用。
    ``_gm_offspring()``
        本代 GM 可以贡献多少个候选。
    ``_gm_bounds()``
        每个子代被截断到的标量 ``(low, high)``。

    合并完成后, :meth:`_gm_commit` 会收到新种群, 必须把它写回优化器自身的状态。

    种群本身已是 ``(n, dim)`` 数组的优化器应改为混入
    :class:`NumpyPopulationGM`, 它用三个属性实现了全部六个钩子。
    """

    #: 总开关。为 False 时 GM 不产生任何开销。这是一个普通类属性
    #: (而非 ClassVar), 这样 :meth:`enable_gm` 才能按实例遮蔽它,
    #: 一个优化器的设置绝不会泄漏到另一个优化器。
    use_gm: bool = False

    #: 当前生效的核。``None`` 表示 GM 不可用, 与 ``use_gm`` 写什么无关 --
    #: 两者总是由 :meth:`enable_gm` 一起设置。
    _gm_operator: GMOperator | None = None

    #: 交给 GM 的种群比例。其余名额仍由算法自身的变异算子产生,
    #: 因此种群规模始终不变。
    gm_offspring_fraction: float = 0.2

    #: 由宿主 ``HeuristicOptimizer`` 提供。
    dim: int
    rng: np.random.Generator

    def enable_gm(self, operator: GMOperator, use_gm: bool = True) -> None:
        """安装 GM 核并开启 (或关闭) GM。

        Args:
            operator: 要使用的核。
            use_gm: 本次调用之后 GM 是否启用。
        """
        self._gm_operator = operator
        self.use_gm = use_gm

    def disable_gm(self) -> None:
        """关闭 GM, 但保留已安装的核, 供之后重新启用。"""
        self.use_gm = False

    # ------------------------------------------------------------------
    # 由具体优化器实现的契约
    # ------------------------------------------------------------------
    def _gm_population(self) -> npt.NDArray[np.float64]:
        """返回当前种群, 形状为 ``(n, dim)`` 的数组。"""
        raise NotImplementedError

    def _gm_fitness(self) -> npt.NDArray[np.float64]:
        """返回当前适应度数组, 形状 ``(n,)``; 越小越好。"""
        raise NotImplementedError

    def _gm_iteration(self) -> int:
        """返回从 0 开始的代数下标。"""
        raise NotImplementedError

    def _gm_offspring(self) -> int:
        """返回本代 GM 可以贡献多少个候选。"""
        raise NotImplementedError

    def _gm_bounds(self) -> tuple[float, float]:
        """返回子代被截断到的标量 ``(low, high)``。"""
        raise NotImplementedError

    def _gm_commit(self, population: npt.NDArray[np.float64]) -> None:
        """把 ``population`` 作为优化器的新种群。"""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 共享行为
    # ------------------------------------------------------------------
    def _ranks(self, fitness: npt.NDArray[np.float64]) -> npt.NDArray[np.intp]:
        """返回无并列处理的排名, 最优 (最小) 适应度记为 ``1``。"""
        return fitness.argsort().argsort() + 1

    def apply_gm_if_enabled(self) -> npt.NDArray[np.float64]:
        """从优化器自身的状态生成一批 GM 子代。

        当 GM 关闭或未安装任何核时, 返回空的 ``(0, dim)`` 数组, 所有合并策略都
        把它当作"没有可加的东西"。正因如此该方法可以无条件调用。

        Returns:
            形状为 ``(n_offspring, dim)`` 的数组; 禁用时为 ``(0, dim)``。
        """
        if not self.use_gm or self._gm_operator is None:
            return np.empty((0, self.dim), dtype=np.float64)
        population = self._gm_population()
        fitness = self._gm_fitness()
        n_offspring = self._gm_offspring()
        if n_offspring <= 0:
            return np.empty((0, self.dim), dtype=np.float64)
        children = self._gm_operator(
            population=population,
            fitness=fitness,
            ranks=self._ranks(fitness),
            current_iter=self._gm_iteration(),
            n_offspring=n_offspring,
            bounds=self._gm_bounds(),
            rng=self.rng,
        )
        return np.asarray(children, dtype=np.float64)


class NumpyPopulationGM(GMOptimizerMixin):
    """面向已持有 ndarray 的优化器的 :class:`GMOptimizerMixin`。

    用具体优化器作为其代际状态维护的三个属性实现全部六个钩子:

    ``_population``
        当前的 ``(n, dim)`` 种群。
    ``_fitness_vals``
        其打分, 形状 ``(n,)``。
    ``_current_iter``
        从 0 开始的代数下标。
    """

    _population: npt.NDArray[np.float64]
    _fitness_vals: npt.NDArray[np.float64]
    _current_iter: int
    config: Any  # 提供 .bounds, 由 HeuristicOptimizer 给出

    def _gm_population(self) -> npt.NDArray[np.float64]:
        return self._population

    def _gm_fitness(self) -> npt.NDArray[np.float64]:
        return self._fitness_vals

    def _gm_iteration(self) -> int:
        return self._current_iter

    def _gm_offspring(self) -> int:
        if not self.use_gm or self._gm_operator is None:
            return 0
        n = self._gm_population().shape[0]
        if n == 0:
            return 0
        return int(max(1, round(n * self.gm_offspring_fraction)))

    def _gm_bounds(self) -> tuple[float, float]:
        return self.config.bounds

    def _gm_commit(self, population: npt.NDArray[np.float64]) -> None:
        self._population = population


_GMStep = TypeVar("_GMStep", bound=Callable[..., npt.NDArray[np.float64]])


def _merge_grow(
    evolved: npt.NDArray[np.float64], children: npt.NDArray[np.float64], worst: npt.NDArray[np.intp]
) -> npt.NDArray[np.float64]:
    """把 GM 子代追加在该步自身产生的子代之后。

    种群规模增长 ``len(children)``; 调用方需自行负责最终规模。只有当算法恰好预留
    了那么多名额时才正确 -- 参见 :meth:`NumpyPopulationGM._gm_offspring`。
    """
    return np.vstack([evolved, children])


def _merge_replace_worst(
    evolved: npt.NDArray[np.float64], children: npt.NDArray[np.float64], worst: npt.NDArray[np.intp]
) -> npt.NDArray[np.float64]:
    """用 GM 子代覆盖 ``worst`` 行, 保持规模不变。

    任何种群规模固定的算法都需要它, 因为一旦扩张就会破坏算子自身的索引方式。
    """
    merged = evolved.copy()
    merged[worst] = children
    return merged


def guided_mutation(
    merge: Literal["grow", "replace_worst"] = "grow",
) -> Callable[[_GMStep], _GMStep]:
    """在一次进化步之后把引导变异的子代插入种群。

    被装饰的步必须以 ``(n, dim)`` 数组返回新种群, 且其宿主必须满足
    :class:`GMOptimizerMixin` 契约。GM 关闭时, 包装器原样返回该步的返回值,
    且不消耗任何随机数 -- 正是这一点使未启用的优化器与不加本装饰器运行时
    逐位一致。

    Args:
        merge: ``"grow"`` 追加子代 (该步必须已为它们预留好名额, 因此规模最终不变)。
            ``"replace_worst"`` 则改为覆盖最差的几行, 适用于能容忍固定规模的算法。

    Returns:
        一个包装单次进化步的装饰器。
    """
    merge_fn = _merge_grow if merge == "grow" else _merge_replace_worst

    def decorator(evolve_step: _GMStep) -> _GMStep:
        def wrapper(self: GMOptimizerMixin, *args: Any, **kwargs: Any) -> npt.NDArray[np.float64]:
            evolved = evolve_step(self, *args, **kwargs)
            if not self.use_gm or self._gm_operator is None:
                return evolved

            fitness = self._gm_fitness()
            children = self.apply_gm_if_enabled()
            if children.size == 0:
                return evolved

            if merge_fn is _merge_replace_worst:
                n_replace = children.shape[0]
                worst = np.argsort(self._ranks(fitness), kind="stable")[-n_replace:]
                merged = merge_fn(evolved, children, worst)
            else:
                merged = merge_fn(evolved, children, worst=np.empty(0, dtype=int))
            self._gm_commit(merged)
            return merged

        wrapper.__name__ = evolve_step.__name__
        wrapper.__doc__ = evolve_step.__doc__
        wrapper.__wrapped__ = evolve_step  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator


