"""CCD camera drivers package.

This package provides camera drivers with a unified interface.
"""

from ao_shaping.drivers.ccd.base import BaseCamera, CameraError
from ao_shaping.drivers.ccd.miicam import CameraStreamManager as MIICamera, MIICAMError
from ao_shaping.drivers.ccd.daheng import CameraStreamManager as DahengCamera
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
