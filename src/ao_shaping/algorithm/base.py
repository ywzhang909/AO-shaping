"""带键的、由注册表支撑的算法家族的共享基类。

本包中的两个优化器家族 -- ``gradient.adam`` 里的梯度优化器与
``heuristic.heuristic_base`` 里的黑盒启发式 -- 回答同样的三个问题:

* *有哪些实现?* 一个按家族专用选择器建立的注册表
  (梯度用小写名字, 启发式用 ``OptimizerType``);
* *怎么构造一个?* 单一的 ``create`` 入口;
* *怎么复用一个?* 显式的 ``reset``。

本模块持有这层共享接口, 使两个家族不会各自漂移。它刻意**不做容错**:
注册表从不猜测构造函数签名, 从不过滤关键字参数, 也从不替换成某个默认实现。
一个家族恰好提供一个构造钩子 (:meth:`RegisteredBase._construct`), 任何不匹配
都会在犯错的调用处以显眼的 ``TypeError`` 暴露出来, 而不是在这里被悄悄糊平。

新增一个实现只需在子类上加一行声明 (``_registry_key``); 任何工厂函数都不必
新增分支。
"""

from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar, TypeVar

Self = TypeVar("Self", bound="RegisteredBase")


class RegisteredBase(ABC):
    """带键算法家族的注册表、构造与 reset 契约。

    子类通过声明 ``_registry_key`` 来接入 :meth:`create`; 声明后即被自动注册::

        class MyOptimizer(Family):
            _registry_key = "mine"

    键按原样保存。想要大小写不敏感查找的家族覆写 :meth:`_normalize_key` --
    归一化是声明式契约的一部分, 而绝不是 ``create`` 顺手施加的隐式便利。
    """

    #: 选择器 -> 实现。**每个家族声明自己的 dict**; 这里只做类型标注而不赋值,
    #: 因为一份共享的注册表会让梯度优化器的名字满足启发式的查找。
    #: 通过 :meth:`register` 修改它, 绝不要重新绑定。
    _registry: ClassVar[dict[Any, type[Any]]]

    #: 由子类声明, 以便在定义时自动注册。从 ``cls.__dict__`` 读取, 这样中间子类
    #: 绝不会以它父类的键被注册。
    _registry_key: ClassVar[Any | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """注册每一个声明了 ``_registry_key`` 的子类。"""
        super().__init_subclass__(**kwargs)
        declared = cls.__dict__.get("_registry_key")
        if declared is not None:
            cls.register(declared, cls)

    @classmethod
    def _normalize_key(cls, key: Any) -> Any:
        """返回 ``key`` 的规范形式。默认为恒等映射。"""
        return key

    @classmethod
    def register(cls, key: Any, implementation: type[Any]) -> type[Any]:
        """把 ``implementation`` 注册到 ``key`` 之下。

        通常由 :meth:`__init_subclass__` 代你调用; 只有在动态注册实现时才直接调用。

        Args:
            key: 该实现可被访问到的选择器。
            implementation: 要注册的具体子类。

        Returns:
            原样返回 ``implementation``, 因此可当作装饰器使用。

        Raises:
            TypeError: 若 ``implementation`` 不是本家族的子类。
        """
        if not issubclass(implementation, RegisteredBase):
            raise TypeError(
                f"{implementation.__name__} must be a subclass of RegisteredBase"
            )
        cls._registry[cls._normalize_key(key)] = implementation
        return implementation

    @classmethod
    def registered(cls) -> tuple[Any, ...]:
        """返回全部已注册的选择器, 按 ``repr`` 排序。"""
        return tuple(sorted(cls._registry, key=repr))

    @classmethod
    def _describe_keys(cls) -> list[str]:
        """把已注册的选择器转成字符串列表, 供错误消息使用。"""
        return [repr(key) for key in cls.registered()]

    @classmethod
    def _lookup(cls, key: Any) -> type[Any]:
        """返回注册在 ``key`` 之下的实现。

        Raises:
            ValueError: 若 ``key`` 没有已注册的实现。
        """
        implementation = cls._registry.get(cls._normalize_key(key))
        if implementation is None:
            raise ValueError(
                f"Unknown {cls.__name__} type: {key!r}. "
                f"Available: {cls._describe_keys()}"
            )
        return implementation

    @classmethod
    def create(cls: type[Self], key: Any, **kwargs: Any) -> Self:
        """构造注册在 ``key`` 之下的实现。

        Args:
            key: 一个已注册的选择器。
            **kwargs: 原样转发给该实现的构造函数。

        Returns:
            构造出的实例。

        Raises:
            ValueError: 若 ``key`` 没有已注册的实现。
        """
        return cls._lookup(key)._construct(**kwargs)

    @classmethod
    def _construct(cls: type[Self], **kwargs: Any) -> Self:
        """由 ``kwargs`` 构造实例。每个家族覆写一次。

        这是家族决定 ``create`` 的关键字参数如何映射到其构造函数签名的唯一位置。
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement _construct() to be creatable"
        )

    def reset(self) -> None:
        """把实例恢复到运行前的状态, 使其可被复用。

        当从不同初始点重跑同一个优化器, 或针对不同目标函数重跑时, 优先用它而不是
        重新实例化。

        随机数生成器刻意*不*重新播种: 重跑应当探索新的随机性, 而不是克隆上一次的。
        """
        self._reset()

    def _reset(self) -> None:
        """清除家族与子类各自的状态。默认什么都不做。"""

    @staticmethod
    def _validate_dim(dim: int) -> None:
        """拒绝非正的问题维度。

        Raises:
            ValueError: 若 ``dim`` 不是正数。
        """
        if dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim!r}")
