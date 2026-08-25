"""CCD camera drivers package.

This package provides camera drivers with a unified interface.
"""

from loguru import logger

from ao_shaping.drivers.ccd.base import BaseCamera, CameraError

MIICamera = None
MIICAMError = None
DahengCamera = None
CameraStreamManager = None

try:
    from ao_shaping.drivers.ccd.miicam.driver import CameraStreamManager as MIICamera, MIICAMError
except Exception as e:
    logger.debug(f"MIICAM driver not available: {e}")

try:
    from ao_shaping.drivers.ccd.daheng import DahengCamManager as DahengCamera
except Exception as e:
    logger.debug(f"Daheng driver not available: {e}")

if DahengCamera is not None:
    CameraStreamManager = DahengCamera
elif MIICamera is not None:
    CameraStreamManager = MIICamera

from ao_shaping.drivers.ccd.ffmpeg import (
    FFmpegCamera,
    FFmpegCameraError,
    ImageFolderCamera,
)

__all__ = [
    "BaseCamera",
    "CameraError",
    "MIICamera",
    "MIICAMError",
    "DahengCamera",
    "CameraStreamManager",
    "FFmpegCamera",
    "FFmpegCameraError",
    "ImageFolderCamera",
]
