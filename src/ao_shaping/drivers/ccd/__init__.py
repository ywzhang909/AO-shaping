"""CCD camera drivers package.

This package provides camera drivers with a unified interface.
"""

import logging

from ao_shaping.drivers.ccd.base import BaseCamera, CameraError

logger = logging.getLogger(__name__)

MIICamera = None
MIICAMError = None
DahengCamera = None
CameraStreamManager = None

try:
    from ao_shaping.drivers.ccd.miicam import (
        CameraStreamManager as MIICamera,
        MIICAMError,
    )
except Exception as e:
    logger.debug(f"MIICAM driver not available: {e}")

try:
    from ao_shaping.drivers.ccd.daheng import CameraStreamManager as DahengCamera
except Exception as e:
    logger.debug(f"Daheng driver not available: {e}")

if DahengCamera is not None:
    CameraStreamManager = DahengCamera
elif MIICamera is not None:
    CameraStreamManager = MIICamera

__all__ = [
    "BaseCamera",
    "CameraError",
    "MIICamera",
    "MIICAMError",
    "DahengCamera",
    "CameraStreamManager",
]
