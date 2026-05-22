from __future__ import annotations

from typing import Any, Callable, Type

from ao_shaping.drivers.dm.base import DM

# Type-specific kwargs filter: maps registered name → set of accepted kwargs
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
}

# Legacy kwarg aliases: (old_name) → (new_name)
_KWARG_ALIASES: dict[str, dict[str, str]] = {
    "nlight": {"dm_neibor_diff": "max_neibor_diff"},
}


class DMRegistry:
    """Registry for DM implementations with decorator-based registration."""

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
        cls = self._registry.get(name.lower())
        if cls is None:
            raise ValueError(
                f"Unknown DM type: {name!r}. Available: {sorted(self._registry.keys())}"
            )
        return cls(**kwargs)

    def create_dm(self, name: str, **kwargs: Any) -> DM:
        """Create a DM instance, filtering kwargs by type-specific accepted params.

        This is the primary factory method for callers that don't know
        the exact constructor signature of each DM type.

        Args:
            name: Registered DM type name (case-insensitive).
            **kwargs: Arbitrary kwargs; only accepted ones are forwarded.

        Returns:
            Instantiated DM subclass.
        """
        key = name.lower()
        if key not in self._registry:
            raise ValueError(
                f"Unknown DM type: {name!r}. Available: {sorted(self._registry.keys())}"
            )

        # Apply legacy aliases
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
        return sorted(self._registry.keys())

    def list_reachable_types(self) -> list[str]:
        """Return sorted list of DM types whose hardware is currently reachable."""
        return sorted(
            name for name, cls in self._registry.items() if cls.is_reachable()
        )

    def get_class(self, name: str) -> Type[DM]:
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
    """Convenience: create a DM instance via the global registry with kwarg filtering."""
    return _global_registry.create_dm(name, **kwargs)


def list_dm_types() -> list[str]:
    """Convenience: list registered DM types."""
    return _global_registry.list_types()


def list_reachable_dm_types() -> list[str]:
    """Convenience: list DM types whose hardware is currently reachable."""
    return _global_registry.list_reachable_types()
