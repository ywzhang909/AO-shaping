"""Hardware drivers package.

This package provides unified interfaces for various hardware devices,
including cameras, DMs, and wavefront sensors.
"""

# Device base classes for digital twin management
from ao_shaping.drivers.device_base import (
    Device,
    DeviceCapability,
    DeviceError,
    DeviceMetadata,
    DeviceNotFoundError,
    DeviceParameter,
    DeviceState,
    DeviceType,
)
from ao_shaping.drivers.device_registry import (
    DeviceRegistry,
    RegisteredDevice,
    get_global_registry,
)

# Hardware-specific imports
from .wfs.thorlab_wfs import WFSManager as Thorlab_WFS
from .wfs.thorlab_wfs import MlaRes
from .dm.NLight import NLight as NlightDM

__all__ = [
    # Base classes
    "Device",
    "DeviceCapability",
    "DeviceError",
    "DeviceMetadata",
    "DeviceNotFoundError",
    "DeviceParameter",
    "DeviceRegistry",
    "DeviceState",
    "DeviceType",
    "RegisteredDevice",
    "get_global_registry",
    # Hardware
    "Thorlab_WFS",
    "MlaRes",
    "NlightDM",
]

import logging


logger = logging.getLogger(__name__)


# Try to import CameraStreamManager, preferring Daheng and falling back to MIICAM.
try:
    from .ccd.daheng import CameraStreamManager

    __all__ += ["CameraStreamManager"]
except Exception as daheng_error:
    try:
        from .ccd.miicam import CameraStreamManager

        __all__ += ["CameraStreamManager"]
        logger.warning(
            "Daheng CameraStreamManager not available; using MIICAM fallback: %s",
            daheng_error,
        )
    except Exception as miicam_error:
        logger.warning(
            "CameraStreamManager not available. Daheng import failed: %s; "
            "MIICAM import failed: %s",
            daheng_error,
            miicam_error,
        )
        CameraStreamManager = None
