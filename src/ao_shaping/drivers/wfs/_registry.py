"""波前传感器实现的注册表。

相机走 ``register_camera``/``create_camera`` 解析, DM 走 ``register_dm``/``resolve_dm``;
而在本模块之前 WFS 完全没有接入点, 因此即便写出了仿真传感器也无法替换进 runner。
"""

from __future__ import annotations

from typing import Any, Callable, Type

from loguru import logger

from ao_shaping.drivers.wfs.base import BaseWFS

_KWARG_FILTERS: dict[str, set[str]] = {
    "thorlab": {
        "mla_index",
        "exposure_time",
        "high_speed",
        "use_custom_ref",
        "pupil_diameter",
        "pupil_center",
        "stable_sample_enable",
        "stable_sample_n",
        "stable_variance_threshold",
        "stable_max_attempts",
        "device_id",
    },
    "sim": {
        "resolution",
        "diameter",
        "n_subap",
        "n_pixel_per_subaperture",
        "wavelength_nm",
        "zernike_order",
        "disturbance_cn2",
        "disturbance_seed",
    },
}

_bound = False


def _ensure_bound() -> None:
    """在注册表首次使用时绑定已知的传感器类型。

    仿真传感器是在 ``drivers.sim.wfs`` 内部用 ``@register_wfs("sim")`` 声明的。
    从 ``drivers.wfs.__init__`` 导入那个包会颠倒分层并有形成循环的风险, 因为
    ``SimulatedWFS`` 回头又会导入 ``drivers.wfs.base`` (以及本模块)。推迟到注册表
    首次使用时再绑定, 既让依赖保持单向, 又使类型无论进程先导入哪个包都可被发现。
    """
    global _bound
    if _bound:
        return
    _bound = True
    for module in (
        "ao_shaping.drivers.wfs.thorlab_wfs",
        "ao_shaping.drivers.sim.wfs",
    ):
        try:
            __import__(module)
        except ImportError as exc:  # pragma: no cover - 仿真包为可选依赖
            logger.debug("WFS type from {} unavailable: {}", module, exc)


class WFSRegistry:
    """基于装饰器注册的 WFS 实现注册表。"""

    def __init__(self) -> None:
        self._registry: dict[str, Type[BaseWFS]] = {}

    def register(self, name: str) -> Callable[[Type[BaseWFS]], Type[BaseWFS]]:
        def decorator(cls: Type[BaseWFS]) -> Type[BaseWFS]:
            if not (isinstance(cls, type) and issubclass(cls, BaseWFS)):
                raise TypeError(f"{cls!r} must be a subclass of BaseWFS")
            self._registry[name.lower()] = cls
            return cls

        return decorator

    def create(self, name: str, **kwargs: Any) -> BaseWFS:
        _ensure_bound()
        cls = self._registry.get(name.lower())
        if cls is None:
            raise ValueError(
                f"Unknown WFS type: {name!r}. Available: {sorted(self._registry)}"
            )
        return cls(**kwargs)

    def create_wfs(self, name: str, **kwargs: Any) -> BaseWFS:
        """创建传感器, 只转发该类型接受的 kwargs。

        Runner 不管最终拿到哪种传感器, 都传同一袋选项, 因此无法识别的键会被丢弃
        而不是抛异常。
        """
        _ensure_bound()
        key = name.lower()
        if key not in self._registry:
            raise ValueError(
                f"Unknown WFS type: {name!r}. Available: {sorted(self._registry)}"
            )
        accepted = _KWARG_FILTERS.get(key, set())
        filtered = (
            {k: v for k, v in kwargs.items() if k in accepted} if accepted else kwargs
        )
        return self._registry[key](**filtered)

    def has_type(self, name: str) -> bool:
        _ensure_bound()
        return name.lower() in self._registry

    def list_types(self) -> list[str]:
        _ensure_bound()
        return sorted(self._registry)

    def get_class(self, name: str) -> Type[BaseWFS]:
        _ensure_bound()
        cls = self._registry.get(name.lower())
        if cls is None:
            raise ValueError(
                f"Unknown WFS type: {name!r}. Available: {sorted(self._registry)}"
            )
        return cls


_global_registry = WFSRegistry()


def get_wfs_registry() -> WFSRegistry:
    return _global_registry


def register_wfs(name: str) -> Callable[[Type[BaseWFS]], Type[BaseWFS]]:
    return _global_registry.register(name)


def create_wfs(name: str, **kwargs: Any) -> BaseWFS:
    """经全局注册表创建 WFS 实例, 并做 kwargs 过滤。"""
    return _global_registry.create_wfs(name, **kwargs)


def list_wfs_types() -> list[str]:
    """列出已注册的 WFS 类型名。"""
    return _global_registry.list_types()


def resolve_wfs(wfs_type: str | None = None, **kwargs: Any) -> BaseWFS:
    """解析传感器类型并创建实例。

    ``wfs_type=None`` 时给出 Thorlab, 与该接入点存在之前 runner 调用点的行为一致,
    因此把它们迁移到 ``resolve_wfs`` 是行为保持的。
    """
    name = (wfs_type or "thorlab").lower()
    if wfs_type is None:
        logger.debug("No wfs_type given; defaulting to the Thorlab sensor")
    return create_wfs(name, **kwargs)


__all__ = [
    "WFSRegistry",
    "create_wfs",
    "get_wfs_registry",
    "list_wfs_types",
    "register_wfs",
    "resolve_wfs",
]
