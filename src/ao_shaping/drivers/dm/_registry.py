from __future__ import annotations

from typing import Any, Callable, Type

from loguru import logger

from ao_shaping.drivers.dm.base import DM

# 按类型过滤 kwargs 的映射: 注册名 → 接受的 kwargs 集合
_KWARG_FILTERS: dict[str, set[str]] = {
    "nlight": {
        "keep_when_exit",
        "max_neibor_diff",
        "dm_neibor_diff",
        "max_iter_diff",
        "safety_mode",
    },
    "micro": {
        "ips",
        "timeout",
        "use_wiring_map",
        "exclude_ips",
        "exclude_ids",
        "device_id",
        "safety_mode",
    },
    "zernike": {"n_max", "resolution", "bits", "radius", "safety_mode"},
    "hadamard": {
        "mode_order",
        "resolution",
        "bits",
        "mask_type",
        "radius",
        "safety_mode",
    },
    "sim_micro": {"device_id", "safety_mode"},
    "sim": {
        "device_id",
        "safety_mode",
        "keep_when_exit",
        "max_iter_diff",
        "max_neibor_diff",
        "dm_neibor_diff",
        "noise_level",
    },
    "asyn_micro": {
        "ips",
        "timeout",
        "safety_mode",
    },
}

# 旧版 kwarg 别名: (old_name) → (new_name)
_KWARG_ALIASES: dict[str, dict[str, str]] = {
    "nlight": {"dm_neibor_diff": "max_neibor_diff"},
    "sim": {"dm_neibor_diff": "max_neibor_diff"},
}

#: 守护绑定仿真 DM 类型的那次一次性 import。
_sim_dms_bound = False


def _ensure_sim_dms_bound() -> None:
    """在首次使用注册表时绑定 ``sim`` / ``sim_micro`` DM 类型。

    这些类型是在 ``drivers.sim.dm`` 内用 ``@register_dm(...)`` 声明的。
    若改为从 ``drivers.dm.__init__`` 导入该包, 则会把分层方向倒过来
    (硬件包导入仿真包), 并造成循环导入, 因为 ``simulated_micro_dm``
    又会回头导入 ``drivers.dm.base``。把这次 import 推迟到首次使用注册表时,
    既让依赖保持单向, 又使这些类型无论进程先导入哪个包都能被发现。
    """
    global _sim_dms_bound
    if _sim_dms_bound:
        return
    _sim_dms_bound = True
    try:
        import ao_shaping.drivers.sim.dm  # noqa: F401
    except ImportError as exc:  # pragma: no cover - sim 包是可选的
        logger.debug("simulated DM types unavailable: {}", exc)


class DMRegistry:
    """支持基于装饰器注册的 DM 实现注册表。"""

    def __init__(self) -> None:
        self._registry: dict[str, Type[DM]] = {}

    def register(self, name: str) -> Callable[[Type[DM]], Type[DM]]:
        def decorator(cls: Type[DM]) -> Type[DM]:
            if not issubclass(cls, DM):
                raise TypeError(f"{cls.__name__} must be a subclass of DM")
            self._registry[name.lower()] = cls
            return cls

        return decorator

    def create(self, name: str, **kwargs: Any) -> DM:
        _ensure_sim_dms_bound()
        cls = self._registry.get(name.lower())
        if cls is None:
            raise ValueError(
                f"Unknown DM type: {name!r}. Available: {sorted(self._registry.keys())}"
            )
        return cls(**kwargs)

    def create_dm(self, name: str, **kwargs: Any) -> DM:
        """创建 DM 实例, 按类型专属的接受参数过滤 kwargs。

        这是给那些不清楚每种 DM 类型确切构造函数签名的调用方用的主工厂方法。

        Args:
            name: 已注册的 DM 类型名 (大小写不敏感)。
            **kwargs: 任意 kwargs; 只有被接受的才会转发。

        Returns:
            实例化后的 DM 子类。
        """
        key = name.lower()
        _ensure_sim_dms_bound()
        if key not in self._registry:
            raise ValueError(
                f"Unknown DM type: {name!r}. Available: {sorted(self._registry.keys())}"
            )

        # 应用旧版别名
        for old, new in _KWARG_ALIASES.get(key, {}).items():
            if old in kwargs and new not in kwargs:
                kwargs[new] = kwargs.pop(old)

        accepted = _KWARG_FILTERS.get(key, set())
        filtered = (
            {k: v for k, v in kwargs.items() if k in accepted} if accepted else kwargs
        )
        return self._registry[key](**filtered)

    def has_type(self, name: str) -> bool:
        return name.lower() in self._registry

    def list_types(self) -> list[str]:
        _ensure_sim_dms_bound()
        return sorted(self._registry.keys())

    def list_reachable_types(self) -> list[str]:
        """返回当前硬件可达的 DM 类型排序列表。"""
        _ensure_sim_dms_bound()
        return sorted(
            name for name, cls in self._registry.items() if cls.is_reachable()
        )

    def get_class(self, name: str) -> Type[DM]:
        _ensure_sim_dms_bound()
        cls = self._registry.get(name.lower())
        if cls is None:
            raise ValueError(
                f"Unknown DM type: {name!r}. Available: {sorted(self._registry.keys())}"
            )
        return cls


_global_registry = DMRegistry()


def get_dm_registry() -> DMRegistry:
    return _global_registry


def register_dm(name: str) -> Callable[[Type[DM]], Type[DM]]:
    return _global_registry.register(name)


def create_dm(name: str, **kwargs: Any) -> DM:
    """便捷函数: 经全局注册表创建 DM 实例, 并做 kwarg 过滤。"""
    return _global_registry.create_dm(name, **kwargs)


def list_dm_types() -> list[str]:
    """便捷函数: 列出已注册的 DM 类型。"""
    return _global_registry.list_types()


def list_reachable_dm_types() -> list[str]:
    """便捷函数: 列出当前硬件可达的 DM 类型。"""
    return _global_registry.list_reachable_types()


def resolve_dm(dm_type: str | None = None, **kwargs) -> DM:
    """解析 DM 类型并创建 DM 实例。

    对应 ``wf`` / ``pipeline`` / ``pib`` / ``combined`` / ``dm-matrix``
    这些 runner 共用的 DM 选择块: 显式的 ``dm_type`` 会被转小写后直接使用;
    否则探测可达的 DM 类型并选取唯一可达的那一个, 对零个/多个候选分别报错。

    Args:
        dm_type: 显式的 DM 类型名, 传 ``None`` 则自动检测。
        **kwargs: 转发给 ``create_dm`` 的额外构造函数 kwargs
            (例如 ``keep_when_exit``、``max_neibor_diff``、
            ``dm_neibor_diff``)。

    Returns:
        创建好的 DM 实例。

    Raises:
        RuntimeError: 没有可达的 DM 时, 或 ``dm_type`` 为 ``None``
            却有多个 DM 可达时。
    """
    from loguru import logger

    _ensure_sim_dms_bound()

    if dm_type is not None:
        dm_type = dm_type.lower()
        logger.info("Using specified DM type: {}", dm_type)
    else:
        reachable = list_reachable_dm_types()
        if len(reachable) == 1:
            dm_type = reachable[0]
            logger.info("Auto-detected reachable DM: {}", dm_type)
        elif len(reachable) == 0:
            raise RuntimeError(
                "No DM reachable. Specify --dm_type explicitly or connect a DM."
            )
        else:
            raise RuntimeError(
                f"Multiple DMs reachable ({', '.join(reachable)}). "
                f"Specify --dm_type explicitly to choose one."
            )

    return create_dm(dm_type, **kwargs)
