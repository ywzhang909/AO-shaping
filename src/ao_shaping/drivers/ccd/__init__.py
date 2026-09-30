"""CCD camera drivers package.

This package provides camera drivers with a unified interface.

SDK-backed backends are exposed **lazily** (PEP 562): ``MIICamera`` and
``DahengCamera`` resolve on first attribute access instead of at package
import. Importing a backend eagerly is not side-effect free — ``miicam.driver``
calls ``_setup_miicam_sdk()``, which mutates ``sys.path`` and loads the native
``MIIUSB.dll`` via ``ctypes.CDLL``, and ``daheng`` imports the ``gxipy``
bindings. Because Python initializes this package before any
``ao_shaping.drivers.ccd.common`` import, the previous eager imports made
*every* consumer of the camera registry (:func:`create_camera`) pay for both
SDKs. Deferring them matches the laziness already documented for the registry
in ``ccd/AGENTS.md``: a missing or broken backend SDK only fails when that
backend is actually requested.

The public names are unchanged, so ``from ao_shaping.drivers.ccd import
MIICamera`` keeps working; a backend that cannot be imported still degrades to
``None`` exactly as before.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from loguru import logger

from ao_shaping.drivers.ccd.base import BaseCamera, CameraError
from ao_shaping.drivers.ccd.ffmpeg import (
    FFmpegCamera,
    FFmpegCameraError,
    ImageFolderCamera,
)

# Public attribute name -> (module to import, attribute within that module).
_LAZY_BACKENDS: dict[str, tuple[str, str]] = {
    "MIICamera": ("ao_shaping.drivers.ccd.miicam.driver", "MIICamera"),
    "MIICAMError": ("ao_shaping.drivers.ccd.miicam.driver", "MIICAMError"),
    "DahengCamera": ("ao_shaping.drivers.ccd.daheng", "DahengCamera"),
}


if TYPE_CHECKING:
    # Static analyzers need the names to be visible; at runtime they are bound
    # by ``__getattr__`` on first access instead.
    from ao_shaping.drivers.ccd.daheng import DahengCamera
    from ao_shaping.drivers.ccd.miicam.driver import MIICAMError, MIICamera


def __getattr__(name: str) -> Any:
    """Resolve an SDK-backed camera class on first access (PEP 562).

    Returns ``None`` when the backend cannot be imported, preserving the
    graceful degradation of the previous eager ``try/except`` imports. The
    result is cached in ``globals()`` so a failed backend is not retried and
    the import cost is paid at most once.
    """
    try:
        module_path, attr = _LAZY_BACKENDS[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None

    try:
        value = getattr(import_module(module_path), attr)
    except Exception as e:  # mirrors the previous catch-all degradation
        logger.debug(f"{name} driver not available: {e}")
        value = None

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "BaseCamera",
    "CameraError",
    "MIICamera",
    "MIICAMError",
    "DahengCamera",
    "FFmpegCamera",
    "FFmpegCameraError",
    "ImageFolderCamera",
]
