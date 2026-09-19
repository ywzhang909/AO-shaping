"""CCD camera drivers package.

This package provides camera drivers with a unified interface.
"""

from loguru import logger

from ao_shaping.drivers.ccd.base import BaseCamera, CameraError

MIICamera = None
MIICAMError = None
DahengCamera = None

try:
    from ao_shaping.drivers.ccd.miicam.driver import MIICamera, MIICAMError
except Exception as e:
    logger.debug(f"MIICAM driver not available: {e}")

try:
    from ao_shaping.drivers.ccd.daheng import DahengCamera
except Exception as e:
    logger.debug(f"Daheng driver not available: {e}")

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
    "FFmpegCamera",
    "FFmpegCameraError",
    "ImageFolderCamera",
]

