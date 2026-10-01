"""Registry for wavefront-sensor implementations.

Cameras resolve through ``register_camera``/``create_camera`` and DMs through
``register_dm``/``resolve_dm``; before this module WFS had no seam at all, so a
simulated sensor could not be substituted into a runner even once one existed.
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
    """Bind the known sensor types on first registry use.

    The simulated sensor is declared with ``@register_wfs("sim")`` inside
    ``drivers.sim.wfs``. Importing that package from ``drivers.wfs.__init__``
    would invert the layering and risk a cycle, because ``SimulatedWFS`` imports
    ``drivers.wfs.base`` (and this module) back. Deferring to first registry use
    keeps the dependency one-way while making the type discoverable regardless of
    which package the process imported first.
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
        except ImportError as exc:  # pragma: no cover - sim package is optional
            logger.debug("WFS type from {} unavailable: {}", module, exc)


class WFSRegistry:
    """Registry for WFS implementations with decorator-based registration."""

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
        """Create a sensor, forwarding only the kwargs that type accepts.

        Runners pass a uniform bag of options regardless of which sensor they end
        up with, so unrecognised keys are dropped rather than raising.
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
    """Create a WFS instance via the global registry with kwarg filtering."""
    return _global_registry.create_wfs(name, **kwargs)


def list_wfs_types() -> list[str]:
    """List registered WFS type names."""
    return _global_registry.list_types()


def resolve_wfs(wfs_type: str | None = None, **kwargs: Any) -> BaseWFS:
    """Resolve the sensor type and create it.

    ``wfs_type=None`` yields Thorlab, matching what the runner call sites did
    before this seam existed, so migrating them to ``resolve_wfs`` is behaviour
    preserving.
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
