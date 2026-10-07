"""梯度下降优化器 (SGD / Adam 家族 / Muon 家族)。

每个具体优化器都通过声明 ``_registry_key`` 来**自行注册**; 本模块不含任何
分发表, 除这些声明之外也不引用自己的子类, 因此新增一个优化器 (比如一个新的
Muon 变体) 只需在这里加一个类, 别的什么都不必动 -- :meth:`Base.create` 只是一次
dict 查找, 所以开闭原则成立。

``create`` 不做任何容错: 关键字参数直接送到构造函数, 由它对任何不接受的参数
抛出指名道姓的 ``TypeError``。

Example:
    >>> from ao_shaping.algorithm.gradient.adam import Base
    >>> opt = Base.create("adamod", dim=64, lr=1.0)
    >>> step = opt.update(gradient)
    >>> opt.reset()
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar, Literal

import numpy as np
import numpy.typing as npt

from ao_shaping.algorithm.base import RegisteredBase


def learning_schedule(
    lr, epoch, epochs, method: Literal["static", "cosin", "exp", "linear"] = "static"
):
    if method == "static":
        return lr
    # Cosine annealing: starts at lr, ends at the 1e-6 floor.
    #
    # This used to be ``lr * cos(pi * epoch / epochs) + 1e-6``, which is a half-wave,
    # not an anneal: cos changes sign at ``epoch == epochs/2`` so the floor does not
    # keep the LR positive, and the second half of every run fed the optimizer a
    # step size of ``-lr + 1e-6`` -- the full initial magnitude, negated, i.e. a
    # direction reversal rather than a step-size change. Fixed by R-44.
    #
    # The form below is the textbook anneal and is deliberately identical to
    # ``optimizer.wf.rms_by_zernike.cosine_annealing_lr`` with ``lr_min=1e-6``:
    # lr at epoch 0, the floor at ``epoch == epochs``, monotone non-increasing in
    # between, and never negative. Do not "simplify" it back to a bare
    # ``lr * cos(...)`` -- that is the defect.
    #
    # NOTE: the ``exp``/``linear`` branches below still use the older
    # ``lr * f(epoch/epochs) + 1e-6`` floor convention, whereas
    # ``rms_by_zernike`` routes the same names to a richer parameterised family
    # (``exponential_decay_lr`` takes a ``decay_factor``). Left alone here: R-44
    # is about the negative learning rate, and reconciling the two families is a
    # separate decision that must not be smuggled in with this fix.
    elif method == "cosin":
        return 1e-6 + (lr - 1e-6) * (1 + np.cos(np.pi * epoch / epochs)) / 2
    # 指数衰减
    elif method == "exp":
        lr = lr * np.exp(-epoch / epochs) + 1e-6
        return lr
    # 线性衰减
    elif method == "linear":
        lr = lr * (1 - epoch / epochs) + 1e-6
        return lr
    else:
        raise ValueError("method must be static, cosin, exp or linear")


class Base(RegisteredBase):
    """梯度优化器的抽象基类。

    子类通过声明 ``_registry_key`` 接入 :meth:`create` 工厂; 声明后即被自动注册::

        class MyOptimizer(Base):
            _registry_key = "mine"

    键按大小写不敏感匹配; 见 :meth:`_normalize_key`。
    """

    #: 本家族的注册表, 以小写优化器名为键。
    _registry: ClassVar[dict[str, type[Base]]] = {}

    @classmethod
    def _normalize_key(cls, key: Any) -> str:
        """把 ``key`` 转成本家族注册时使用的小写名字。"""
        return str(key).lower()

    @classmethod
    def registered_names(cls) -> tuple[str, ...]:
        """返回全部已注册的优化器名, 已排序。"""
        return tuple(sorted(cls._registry))

    @classmethod
    def _describe_keys(cls) -> list[str]:
        """返回已注册的名字, 供错误消息使用。"""
        return list(cls.registered_names())

    def __init__(self, dim: int, lr: float = 1.0):
        """初始化优化器。

        Args:
            dim: 参数向量的维度。必须为正。
            lr: 学习率。

        Raises:
            ValueError: 若 ``dim`` 不是正数。
        """
        self._validate_dim(dim)
        self.dim = dim
        self.lr = lr
        self.t: int = 0

    @abstractmethod
    def update(self, grad: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        pass

    def scale_momentum(self, scaler):
        pass

    def reset(self) -> None:
        """把优化器恢复到刚构造出来的状态。

        步数计数器与所有动量缓冲清零, 于是同一个实例可以在新一轮运行中复用,
        而不会继承上一次残留的动量。
        """
        self.t = 0
        self._reset()

    def _reset(self) -> None:
        """将子类特有的缓冲清零。默认什么都不做。"""

    @classmethod
    def create(cls, name: str, dim: int, lr: float = 1.0, **kwargs: Any) -> Base:
        """按名字创建一个优化器, 在注册表里查找。

        Args:
            name: 已注册的优化器名 (大小写不敏感)。
            dim: 参数向量的维度。
            lr: 学习率。
            **kwargs: 原样转发给构造函数, 由它对任何不接受的参数抛出指名道姓的
                ``TypeError``。

        Returns:
            优化器实例。

        Raises:
            ValueError: 若 ``name`` 没有已注册的实现。
            ValueError: 若 ``dim`` 不是正数。
        """
        return cls._lookup(name)._construct(dim=dim, lr=lr, **kwargs)

    @classmethod
    def _construct(cls, dim: int, lr: float = 1.0, **kwargs: Any) -> Base:
        """由 ``create`` 的关键字参数构造实例。"""
        return cls(dim=dim, lr=lr, **kwargs)


class SGD(Base):
    _registry_key = "sgd"

    def __init__(self, dim: int, lr=1.0):
        super().__init__(dim, lr)

    def update(self, grad: npt.NDArray[np.floating]):
        self.t += 1
        return self.lr * grad


class Adam(Base):
    """
    使用 EMA 来估计二阶矩。这意味着它会遗忘早期的梯度信息。这使得 Adam 的自适应性更强，可以快速适应梯度的局部变化。
    """

    _registry_key = "adam"

    def __init__(self, dim: int, lr=1.0, beta1=0.9, beta2=0.99):
        super().__init__(dim, lr)
        self.beta1 = beta1
        self.beta2 = beta2

        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)

    def update(self, grad: npt.NDArray[np.floating]):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    def scale_momentum(self, scaler: float):
        self.m *= scaler
        self.v *= scaler**2

    def _reset(self):
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)


class AdamW(Adam):
    _registry_key = "adamw"

    def __init__(self, dim: int, lr=1.0, beta1=0.9, beta2=0.99, weight_decay=1e-2):
        super().__init__(dim, lr, beta1, beta2)
        self.weight_decay = weight_decay

    def update(self, grad: npt.NDArray[np.floating]):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return (
            self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            + self.weight_decay * self.lr * self.m
        )


class AdaMOD(Adam):
    """
    AdaMod 是一个基于 Adam 的新的深度学习优化器，但它提供了自动warmup heuristic和长期学习率缓冲。
    从最初的测试来看，AdaMod 是top 5的优化器，很容易击败或超过普通的 Adam，且对学习率超参数不那么敏感，训练曲线更平滑，不需要warmup模式。

    Pros:
    AdaMod保持了自适应学习率自身的指数长期平均值，并在整个训练过程中用这个值来clip任何过高的适应率。
    结果改善了收敛性，不需要warmup，对实际学习率选择的敏感性较低。 记忆的程度由一个新的参数 Beta3控制。

    Cons:
    虽然AdaMod通常比普通的Adam表现更好，但是在更长的训练条件下，SGDM 仍然可能比AdaMod表现更好。

    """

    _registry_key = "adamod"

    def __init__(self, dim: int, lr=1.0, beta1=0.9, beta2=0.99, beta3=0.9995, **kwargs):
        super().__init__(dim, lr, beta1, beta2)
        self.beta3 = beta3
        self.s = 0.0

    def update(self, grad: npt.NDArray[np.floating]):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        gamma = self.lr / (np.sqrt(v_hat) + 1e-8)
        self.s = self.beta3 * self.s + (1 - self.beta3) * gamma
        learning_rate = np.where(gamma < self.s, gamma, self.s)
        return learning_rate * m_hat

    def _reset(self):
        super()._reset()
        self.s = 0.0


class Muno(Base):
    """
    Muno 优化器是一种结合了动量和自适应学习率的优化算法。
    它通过维护梯度的指数移动平均和梯度平方的指数移动平均来动态调整学习率，
    同时引入了额外的机制来稳定训练过程。
    """

    _registry_key = "muno"

    def __init__(
        self, dim: int, lr=1.0, beta1=0.9, beta2=0.999, eps=1e-8, amsgrad=False
    ):
        """
        初始化 Muno 优化器

        Args:
            dim: 参数维度
            lr: 初始学习率
            beta1: 动量项的指数衰减率
            beta2: 梯度平方项的指数衰减率
            eps: 数值稳定性常数
            amsgrad: 是否使用 AMSGrad 变体
        """
        super().__init__(dim, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.amsgrad = amsgrad

        # 初始化动量和梯度平方的累积变量
        self.m = np.zeros(self.dim, dtype=np.float32)  # 动量
        self.v = np.zeros(self.dim, dtype=np.float32)  # 梯度平方的累积
        self.v_max = np.zeros(self.dim, dtype=np.float32)  # AMSGrad 中的最大梯度平方

    def update(self, grad: npt.NDArray[np.floating]):
        """
        更新参数

        Args:
            grad: 当前梯度

        Returns:
            更新步长
        """
        self.t += 1

        # 更新动量(一阶矩估计)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad

        # 更新梯度平方的累积（二阶矩估计）
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2

        # 偏差修正
        m_hat = self.m / (1 - self.beta1**self.t)

        if self.amsgrad:
            # AMSGrad: 维护历史最大值
            self.v_max = np.maximum(self.v_max, self.v)
            v_hat = self.v_max / (1 - self.beta2**self.t)
        else:
            # 标准 Muno
            v_hat = self.v / (1 - self.beta2**self.t)

        # 计算更新步长
        return self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def _reset(self):
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)
        self.v_max = np.zeros(self.dim, dtype=np.float32)


class MunoW(Muno):
    """
    带权重衰减的 Muno 优化器 (MunoW)
    """

    _registry_key = "munow"

    def __init__(
        self,
        dim: int,
        lr=1.0,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
    ):
        """
        初始化 MunoW 优化器

        Args:
            dim: 参数维度
            lr: 初始学习率
            beta1: 动量项的指数衰减率
            beta2: 梯度平方项的指数衰减率
            eps: 数值稳定性常数
            weight_decay: 权重衰减系数
            amsgrad: 是否使用 AMSGrad 变体
        """
        super().__init__(dim, lr, beta1, beta2, eps, amsgrad)
        self.weight_decay = weight_decay

    def update(self, grad: npt.NDArray[np.floating]):
        """
        更新参数

        Args:
            grad: 当前梯度

        Returns:
            更新步长
        """
        self.t += 1

        # 更新动量（一阶矩估计）
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad

        # 更新梯度平方的累积（二阶矩估计）
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2

        # 偏差修正
        m_hat = self.m / (1 - self.beta1**self.t)

        if self.amsgrad:
            # AMSGrad: 维护历史最大值
            self.v_max = np.maximum(self.v_max, self.v)
            v_hat = self.v_max / (1 - self.beta2**self.t)
        else:
            # 标准 MunoW
            v_hat = self.v / (1 - self.beta2**self.t)

        # 计算更新步长，包含权重衰减
        return (
            self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            + self.weight_decay * self.lr * self.m
        )


def zeropower_via_newtonschulz5(G, steps: int = 5):
    """
    用 Newton-Schulz 迭代计算 G 的零次幂 / 正交化。我们选用一个五次迭代, 其系数
    挑得使零点处的斜率最大。就把步数降到最少这个目的而言, 实测上即便继续抬高
    零点处的斜率、越过迭代在整个区间上都不再收敛到 1 的位置, 也依然是有效的。
    因此该迭代产出的并不是 UV^T, 而更像是 US'V^T, 其中 S' 是对角阵且
    S_{ii}' ~ Uniform(0.5, 1.5); 相比 USV^T = G 那个 SVD, 这样做似乎完全不损害
    模型性能。
    """
    assert G.ndim >= 2, "G must be at least 2-dimensional"

    # 五次迭代的系数
    a, b, c = (3.4445, -4.7750, 2.0315)

    # 在 float32 下用 G 的一份副本
    X = G.astype(np.float32)

    # 需要时转置 (当行数 > 列数)
    transposed = False
    if X.shape[-2] > X.shape[-1]:
        X = np.swapaxes(X, -2, -1)
        transposed = True

    # 确保谱范数不超过 1
    norm = np.linalg.norm(X, axis=(-2, -1), keepdims=True)
    X = X / (norm + 1e-7)

    # 执行 NS 迭代
    for _ in range(steps):
        A = np.matmul(X, np.swapaxes(X, -2, -1))
        B = b * A + c * np.matmul(A, A)  # 五次计算
        X = a * X + np.matmul(B, X)

    # 需要时转回去
    if transposed:
        X = np.swapaxes(X, -2, -1)

    return X


def muon_update(grad, momentum_buffer, beta=0.95, ns_steps=5, nesterov=True):
    """
    Muon 更新函数, 施加动量与 Newton-Schulz 正交化

    Args:
        grad: 当前梯度
        momentum_buffer: 动量缓冲
        beta: 动量系数
        ns_steps: Newton-Schulz 步数
        nesterov: 是否使用 Nesterov 动量
    """
    # 更新动量缓冲
    momentum_buffer = beta * momentum_buffer + (1 - beta) * grad

    # 施加 Nesterov 动量或标准动量
    update = grad * (1 - beta) + momentum_buffer * beta if nesterov else momentum_buffer

    # 对卷积核 (4D), 重塑为 2D
    original_shape = update.shape
    if update.ndim == 4:
        update = update.reshape(len(update), -1)

    # 施加 Newton-Schulz 正交化
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)

    # 按维度比值重新缩放
    scale = max(1, update.shape[-2] / update.shape[-1]) ** 0.5
    update = update * scale

    # 需要时重塑回原形状
    if update.shape != original_shape:
        update = update.reshape(original_shape)

    return update, momentum_buffer


class Muon(Base):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon 内部先跑标准的 SGD-momentum, 然后做一次正交化后处理, 其中每个 2D 参数的
    更新量被替换为最近的正交矩阵。为高效完成正交化, 我们使用 Newton-Schulz 迭代。

    Muon 只应用于隐藏权重层。输入嵌入层、最终输出层, 以及任何内部的增益或偏置,
    都应当用 AdamW 之类的标准方法来优化。
    """

    _registry_key = "muon"

    def __init__(self, dim: int, lr=0.02, weight_decay=0, momentum=0.95, ns_steps=5):
        """
        初始化 Muon 优化器

        Args:
            dim: 参数维度
            lr: 学习率
            weight_decay: 权重衰减系数
            momentum: 动量系数
            ns_steps: Newton-Schulz 步数
        """
        super().__init__(dim, lr)
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.ns_steps = ns_steps

        # 初始化动量缓冲
        self.momentum_buffer = np.zeros(dim, dtype=np.float32)

    def update(self, grad: npt.NDArray[np.floating]):
        """
        用 Muon 优化更新参数

        Args:
            grad: 当前梯度

        Returns:
            更新步长
        """
        self.t += 1

        # 施加权重衰减
        if self.weight_decay > 0:
            grad = grad + self.weight_decay * self.momentum_buffer

        # 施加 Muon 更新
        update, self.momentum_buffer = muon_update(
            grad, self.momentum_buffer, beta=self.momentum, ns_steps=self.ns_steps
        )

        # 按学习率缩放
        return -self.lr * update

    def _reset(self):
        self.momentum_buffer = np.zeros(self.dim, dtype=np.float32)


class AdamNS(Base):
    """
    带 Newton-Schulz 正交化后处理的 Adam 优化器
    """

    _registry_key = "adamns"

    def __init__(self, dim: int, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, ns_steps=5):
        """
        初始化 AdamNS 优化器

        Args:
            dim: 参数维度
            lr: 学习率
            betas: 计算梯度及其平方的滑动平均的系数
            eps: 加到分母上以改善数值稳定性的项
            ns_steps: 用于正交化的 Newton-Schulz 步数
        """
        super().__init__(dim, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.ns_steps = ns_steps

        # 初始化缓冲
        self.buf1 = np.zeros(dim, dtype=np.float32)  # 一阶矩估计
        self.buf2 = np.zeros(dim, dtype=np.float32)  # 二阶矩估计

    def adam_update(self, grad):
        """
        标准 Adam 更新
        """
        # 更新有偏的一阶矩估计
        self.buf1 = self.beta1 * self.buf1 + (1 - self.beta1) * grad

        # 更新有偏的二阶原始矩估计
        self.buf2 = self.beta2 * self.buf2 + (1 - self.beta2) * grad**2

        # 计算偏差修正后的一阶矩估计
        buf1c = self.buf1 / (1 - self.beta1**self.t)

        # 计算偏差修正后的二阶原始矩估计
        buf2c = self.buf2 / (1 - self.beta2**self.t)

        # 计算更新量
        return buf1c / (np.sqrt(buf2c) + self.eps)

    def update(self, grad: npt.NDArray[np.floating]):
        """
        用带 Newton-Schulz 正交化的 Adam 更新参数

        Args:
            grad: 当前梯度

        Returns:
            更新步长
        """
        self.t += 1

        # 标准 Adam 更新
        update = self.adam_update(grad)

        # 对更高维的参数, 施加 Newton-Schulz 正交化
        if update.ndim >= 2:
            original_shape = update.shape
            # 需要时重塑为 2D
            if update.ndim > 2:
                update = update.reshape(-1, update.shape[-1])

            # 施加 Newton-Schulz 正交化
            update = zeropower_via_newtonschulz5(update, steps=self.ns_steps)

            # 重塑回去
            if update.shape != original_shape:
                update = update.reshape(original_shape)

        return self.lr * update

    def _reset(self):
        self.buf1 = np.zeros(self.dim, dtype=np.float32)
        self.buf2 = np.zeros(self.dim, dtype=np.float32)


def search_optimal_delta(
    param_dim: int,
    objective_fn: Callable[[npt.NDArray[np.floating]], float],
    apply_fn: Callable[[npt.NDArray[np.floating]], None],
    min_delta: float = 0.01,
    max_delta: float = 100.0,
    n_magnitude_steps: int = 5,
    n_samples_per_delta: int = 5,
    clip_min: float = -50.0,
    clip_max: float = 50.0,
    perturb_mask: npt.NDArray | None = None,
    verbose: bool = True,
) -> tuple[float, dict]:
    """用幅值扫描搜索最优扰动 delta。

    两阶段方案:
    1. 幅值扫描: 测试各个数量级 (1e-2, 1e-1, 1e0, 1e1, 1e2)
    2. 精细扫描: 在最佳数量级区间内测试

    一个有效的 delta 应当在 +/- 两个方向上都带来一致的 RMS 改善。

    Args:
        param_dim: 待优化参数向量的维度。
        objective_fn: 接受参数并返回标量目标值的函数。
        apply_fn: 把参数施加到系统上的函数。
        min_delta: 要测试的最小 delta 值。
        max_delta: 要测试的最大 delta 值。
        n_magnitude_steps: 要测试的数量级步数 (默认: 5)。
        n_samples_per_delta: 每个 delta 的采样数, 用于压低噪声 (默认: 5)。
        clip_min: 参数裁剪的下界。
        clip_max: 参数裁剪的上界。
        perturb_mask: 可选掩码, 指定扰动哪些参数。
        verbose: 是否打印进度信息。

    Returns:
        (optimal_delta, info_dict) 二元组
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print("Auto Delta Detection (Magnitude Scan)")
        print(
            f"Range: [{min_delta}, {max_delta}], Magnitude steps: {n_magnitude_steps}"
        )
        print(f"Samples per delta: {n_samples_per_delta}")
        print(f"{'=' * 60}\n")

    init_params = np.zeros(param_dim, dtype=np.float64)
    apply_fn(init_params)

    baseline_obj = objective_fn(init_params)
    if verbose:
        print(f"Baseline objective: {baseline_obj:.4f}")

    magnitude_results = []
    magnitude_values = np.linspace(-2, 2, n_magnitude_steps)

    if verbose:
        print("\n--- Phase 1: Magnitude Scan ---")

    for mag in magnitude_values:
        test_delta = 10.0**mag
        if test_delta < min_delta or test_delta > max_delta:
            continue

        if verbose:
            print(f"\nTesting magnitude 10^{mag:.1f} (delta={test_delta:.4f}):")

        pos_improvements = []
        neg_improvements = []

        for sample_idx in range(n_samples_per_delta):
            apply_fn(init_params)

            disturb = np.random.randn(param_dim).astype(np.float64)
            disturb *= test_delta

            if perturb_mask is not None:
                disturb *= perturb_mask

            pos_params = np.clip(init_params + disturb, clip_min, clip_max)
            neg_params = np.clip(init_params - disturb, clip_min, clip_max)

            apply_fn(pos_params)
            pos_obj = objective_fn(pos_params)

            apply_fn(neg_params)
            neg_obj = objective_fn(neg_params)

            pos_imp = baseline_obj - pos_obj
            neg_imp = baseline_obj - neg_obj
            pos_improvements.append(pos_imp)
            neg_improvements.append(neg_imp)

            if verbose:
                print(
                    f"  Sample {sample_idx + 1}: pos_imp={pos_imp:.4f}, neg_imp={neg_imp:.4f}"
                )

        avg_pos_imp = np.mean(pos_improvements)
        avg_neg_imp = np.mean(neg_improvements)
        consistency = min(avg_pos_imp, avg_neg_imp) / (
            max(avg_pos_imp, avg_neg_imp) + 1e-8
        )

        magnitude_results.append(
            {
                "magnitude": mag,
                "delta": test_delta,
                "avg_pos_imp": avg_pos_imp,
                "avg_neg_imp": avg_neg_imp,
                "avg_improvement": (avg_pos_imp + avg_neg_imp) / 2,
                "consistency": consistency,
            }
        )

        if verbose:
            print(
                f"  Summary: avg_pos={avg_pos_imp:.4f}, avg_neg={avg_neg_imp:.4f}, consistency={consistency:.2f}"
            )

    best_mag_result = max(magnitude_results, key=lambda x: x["avg_improvement"])
    best_magnitude = best_mag_result["magnitude"]

    if verbose:
        print(
            f"\nBest magnitude: 10^{best_magnitude:.1f} (delta={best_mag_result['delta']:.4f})"
        )
        print("\n--- Phase 2: Fine Scan ---")

    fine_delta_min = 10.0 ** (best_magnitude - 0.5)
    fine_delta_max = 10.0 ** (best_magnitude + 0.5)
    fine_delta_min = max(fine_delta_min, min_delta)
    fine_delta_max = min(fine_delta_max, max_delta)

    n_fine_steps = 10
    fine_deltas = np.linspace(fine_delta_min, fine_delta_max, n_fine_steps)

    fine_results = []

    for delta in fine_deltas:
        if verbose:
            print(f"\nTesting delta = {delta:.4f}:")

        pos_improvements = []
        neg_improvements = []

        for sample_idx in range(n_samples_per_delta):
            apply_fn(init_params)

            disturb = np.random.randn(param_dim).astype(np.float64)
            disturb *= delta

            if perturb_mask is not None:
                disturb *= perturb_mask

            pos_params = np.clip(init_params + disturb, clip_min, clip_max)
            neg_params = np.clip(init_params - disturb, clip_min, clip_max)

            apply_fn(pos_params)
            pos_obj = objective_fn(pos_params)

            apply_fn(neg_params)
            neg_obj = objective_fn(neg_params)

            pos_imp = baseline_obj - pos_obj
            neg_imp = baseline_obj - neg_obj
            pos_improvements.append(pos_imp)
            neg_improvements.append(neg_imp)

        avg_pos_imp = np.mean(pos_improvements)
        avg_neg_imp = np.mean(neg_improvements)

        both_positive = avg_pos_imp > 0 and avg_neg_imp > 0
        avg_improvement = (avg_pos_imp + avg_neg_imp) / 2

        fine_results.append(
            {
                "delta": delta,
                "avg_pos_imp": avg_pos_imp,
                "avg_neg_imp": avg_neg_imp,
                "avg_improvement": avg_improvement,
                "both_positive": both_positive,
                "std_pos": np.std(pos_improvements),
                "std_neg": np.std(neg_improvements),
            }
        )

        status = "✓" if both_positive else "✗"
        if verbose:
            print(
                f"  pos_imp={avg_pos_imp:.4f}±{np.std(pos_improvements):.4f}, neg_imp={avg_neg_imp:.4f}±{np.std(neg_improvements):.4f} {status}"
            )

    valid_results = [r for r in fine_results if r["both_positive"]]

    if valid_results:
        best_result = max(valid_results, key=lambda x: x["avg_improvement"])
        optimal_delta = best_result["delta"]
    else:
        best_result = max(fine_results, key=lambda x: x["avg_improvement"])
        optimal_delta = best_result["delta"]

    apply_fn(init_params)

    if verbose:
        print("\n{'='*60}")
        print("Auto Delta Detection Complete")
        print(f"Optimal delta: {optimal_delta:.4f}")
        print(
            f"Improvement: pos={best_result['avg_pos_imp']:.4f}, neg={best_result['avg_neg_imp']:.4f}"
        )
        print(f"{'=' * 60}\n")

    return float(optimal_delta), {
        "baseline_obj": float(baseline_obj),
        "optimal_delta": float(optimal_delta),
        "magnitude_results": magnitude_results,
        "fine_results": fine_results,
    }
