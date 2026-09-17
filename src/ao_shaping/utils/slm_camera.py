"""Shared camera factories for SLM acquisition tools."""

from __future__ import annotations

from typing import Any

from loguru import logger


def open_daheng_camera(cam_id: int, exposure_ms: float) -> Any:
    """Open a Daheng camera using a lazy hardware import."""
    try:
        from ao_shaping.drivers.ccd.daheng import DahengCamManager

        camera = DahengCamManager(cam_id=cam_id, exposure_time_ms=exposure_ms)
        camera.open()
        return camera
    except ImportError as exc:
        logger.warning("Daheng camera unavailable: {}", exc)
        raise
    except Exception as exc:
        logger.error("Daheng camera initialization failed: {}", exc)
        raise


def open_miicam_camera(
    cam_id: int,
    exposure_ms: float,
    bit_depth: int = 8,
) -> Any:
    """Open a MiiCam camera using a lazy hardware import."""
    try:
        from ao_shaping.drivers.ccd.miicam.driver import CameraStreamManager

        camera = CameraStreamManager(
            cam_id=cam_id,
            exposure_time_ms=exposure_ms,
            bit_depth=bit_depth,
        )
        camera.open()
        return camera
    except ImportError as exc:
        logger.warning("MiiCam camera unavailable: {}", exc)
        raise
    except Exception as exc:
        logger.error("MiiCam camera initialization failed: {}", exc)
        raise


__all__ = ["open_daheng_camera", "open_miicam_camera"]
