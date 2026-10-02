"""Shared base for keyed, registry-backed algorithm families.

Both optimizer families in this package -- the gradient optimizers in
``gradient.adam`` and the black-box heuristics in ``heuristic.heuristic_base`` --
answer the same three questions:

* *which implementations exist?* a registry keyed by a family-specific selector
  (a lowercase name for gradients, an ``OptimizerType`` for heuristics);
* *how do I build one?* a single ``create`` entry point;
* *how do I reuse one?* an explicit ``reset``.

This module owns that shared surface so the two families cannot drift apart. It
deliberately performs **no fault tolerance**: the registry never guesses a
constructor signature, never filters keyword arguments, and never substitutes a
default implementation. A family supplies exactly one construction hook
(:meth:`RegisteredBase._construct`) and any mismatch surfaces as a loud
``TypeError`` at the call site that made the mistake, rather than being silently
papered over here.

Adding an implementation is a one-line declaration on the subclass
(``_registry_key``); no branch is added to any factory.
"""

from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar, TypeVar

Self = TypeVar("Self", bound="RegisteredBase")


class RegisteredBase(ABC):
    """Registry, construction and reset contract for a keyed algorithm family.

    Subclasses opt into :meth:`create` by declaring ``_registry_key``; they are
    then registered automatically::

        class MyOptimizer(Family):
            _registry_key = "mine"

    The key is stored verbatim. Families that want case-insensitive lookup
    override :meth:`_normalize_key` -- normalization is part of the declared
    contract, never an implicit convenience applied by ``create``.
    """

    #: selector -> implementation. **Each family declares its own dict**; it is
    #: annotated here but deliberately not assigned, because a single shared
    #: registry would let a gradient optimizer's name satisfy a heuristic lookup.
    #: Mutate it through :meth:`register`, never by rebinding it.
    _registry: ClassVar[dict[Any, type[Any]]]

    #: Declared by a subclass to auto-register on definition. Read from
    #: ``cls.__dict__`` so an intermediate subclass is never registered under
    #: its parent's key.
    _registry_key: ClassVar[Any | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register every subclass that declares ``_registry_key``."""
        super().__init_subclass__(**kwargs)
        declared = cls.__dict__.get("_registry_key")
        if declared is not None:
            cls.register(declared, cls)

    @classmethod
    def _normalize_key(cls, key: Any) -> Any:
        """Return the canonical form of ``key``. Identity by default."""
        return key

    @classmethod
    def register(cls, key: Any, implementation: type[Any]) -> type[Any]:
        """Register ``implementation`` under ``key``.

        Normally invoked on your behalf by :meth:`__init_subclass__`; call it
        directly only to register an implementation dynamically.

        Args:
            key: Selector this implementation is reachable through.
            implementation: The concrete subclass to register.

        Returns:
            ``implementation``, unchanged, so this can be used as a decorator.

        Raises:
            TypeError: If ``implementation`` is not a subclass of this family.
        """
        if not issubclass(implementation, RegisteredBase):
            raise TypeError(
                f"{implementation.__name__} must be a subclass of RegisteredBase"
            )
        cls._registry[cls._normalize_key(key)] = implementation
        return implementation

    @classmethod
    def registered(cls) -> tuple[Any, ...]:
        """Return every registered selector, sorted by ``repr``."""
        return tuple(sorted(cls._registry, key=repr))

    @classmethod
    def _describe_keys(cls) -> list[str]:
        """Return the registered selectors as strings for error messages."""
        return [repr(key) for key in cls.registered()]

    @classmethod
    def _lookup(cls, key: Any) -> type[Any]:
        """Return the implementation registered under ``key``.

        Raises:
            ValueError: If ``key`` has no registered implementation.
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
        """Build the implementation registered under ``key``.

        Args:
            key: A registered selector.
            **kwargs: Forwarded verbatim to the implementation's constructor.

        Returns:
            The constructed instance.

        Raises:
            ValueError: If ``key`` has no registered implementation.
        """
        return cls._lookup(key)._construct(**kwargs)

    @classmethod
    def _construct(cls: type[Self], **kwargs: Any) -> Self:
        """Build an instance from ``kwargs``. Overridden once per family.

        This is the single place a family decides how ``create``'s keyword
        arguments map onto its constructors' signatures.
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement _construct() to be creatable"
        )

    def reset(self) -> None:
        """Return the instance to its pre-run state so it can be reused.

        Prefer this over re-instantiating when re-running the same optimizer from
        a different initial point or against a different objective function.

        The random generator is deliberately *not* reseeded: a re-run should
        explore fresh randomness rather than clone the previous one.
        """
        self._reset()

    def _reset(self) -> None:
        """Clear family- and subclass-specific state. No-op by default."""

    @staticmethod
    def _validate_dim(dim: int) -> None:
        """Reject a non-positive problem dimension.

        Raises:
            ValueError: If ``dim`` is not positive.
        """
        if dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim!r}")
