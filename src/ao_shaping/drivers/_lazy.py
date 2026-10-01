"""Shared PEP 562 lazy-attribute installation for driver packages.

Importing a driver package must not load a native SDK. The side effects are
real and non-obvious — ``ccd/miicam/driver.py`` calls ``_setup_miicam_sdk()``
at module scope, which rewrites ``sys.path`` and preloads ``MIIUSB.dll`` via
``ctypes.CDLL``; ``ccd/daheng/driver.py`` imports the ``gxipy`` bindings. Since
Python initialises a parent package before importing its submodules, one eager
import makes *every* consumer of the sibling registries pay for it.

The camera family solved this with a module-level ``__getattr__``. That body was
then copied into a second package, so the resolution rules (first-access
binding, ``globals()`` caching so a failed backend is not retried, graceful
degradation to ``None``) existed in two places and could drift. This module is
the single implementation; driver packages call :func:`install_lazy_attrs`.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable

from loguru import logger


def install_lazy_attrs(
    module_globals: dict[str, Any],
    backends: dict[str, tuple[str, str]],
    module_name: str,
    on_error: Callable[[str, BaseException], None] | None = None,
) -> Callable[[str], Any]:
    """Build a PEP 562 ``__getattr__`` resolving ``backends`` on first access.

    Args:
        module_globals: The calling package's ``globals()``. Successfully
            resolved names are cached there, so a name is imported at most once
            and a later direct import of the submodule cannot shadow the binding.
        backends: Public attribute name -> ``(module path, attribute)``.
        module_name: Dotted package name, used in the error message.
        on_error: Optional hook invoked as ``on_error(name, exception)``.
            Defaults to a debug log. Return ``False`` from it to re-raise
            instead of degrading to ``None``.

    Returns:
        A ``__getattr__`` function suitable for the calling package.
    """

    def __getattr__(name: str) -> Any:
        try:
            module_path, attr = backends[name]
        except KeyError:
            raise AttributeError(
                f"module {module_name!r} has no attribute {name!r}"
            ) from None

        try:
            value = getattr(import_module(module_path), attr)
        except Exception as exc:  # a missing SDK must not break the whole package
            if on_error is not None and on_error(name, exc) is False:
                raise
            logger.debug(f"{name} driver not available: {exc}")
            value = None

        module_globals[name] = value
        return value

    return __getattr__


__all__ = ["install_lazy_attrs"]
