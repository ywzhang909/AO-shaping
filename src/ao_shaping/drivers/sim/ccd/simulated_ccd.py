"""Simulated CCD camera.

This module provides a simulated CCD camera that inherits from BaseCamera,
allowing it to integrate with the existing camera interface while using
numerical simulation internally.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np
from loguru import logger

from ao_shaping.drivers.ccd.base import BaseCamera, CameraError
from ao_shaping.drivers.device_base import DeviceState


class SimulatedCCDError(CameraError):
    """Exception for simulated CCD errors."""
    pass


class SimulatedCCD(BaseCamera):
    """Simulated CCD camera.
    
    This class provides a simulated camera that inherits from BaseCamera,
    implementing all required abstract methods with numerical simulation.
    The actual computation uses algorithms from sim.digitaltwin when available.
    
    Attributes:
        resolution: Camera resolution (width, height).
        noise_level: Standard deviation of noise in ADU.
        exposure_time_ms: Exposure time in milliseconds.
    
    Example:
        >>> cam = SimulatedCCD(resolution=(1024, 1024), noise_level=5.0)
        >>> with cam:
        ...     img = cam.get_numpy_image()
        ...     print(f"Captured: {img.shape}")
    """
    
    def __init__(
        self,
        cam_id: int = 0,
        exposure_time_ms: int = 20,
        resolution: Tuple[int, int] = (1024, 1024),
        noise_level: float = 5.0,
        random_seed: Optional[int] = None,
    ):
        """Initialize simulated CCD.
        
        Args:
            cam_id: Camera ID (for BaseCamera compatibility).
            exposure_time_ms: Exposure time in ms.
            resolution: Image resolution (width, height).
            noise_level: Noise level in ADU.
            random_seed: Random seed for reproducibility.
        """
        super().__init__(cam_id, exposure_time_ms)
        
        self._resolution = resolution
        self._noise_level = noise_level
        self._random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)
        
        self._frame_counter = 0
        self._last_image: Optional[np.ndarray] = None
        
        logger.debug(
            f"SimulatedCCD initialized: "
            f"resolution={resolution}, noise={noise_level}"
        )
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit."""
        self.close()
    
    def initialize(self) -> None:
        """Initialize the simulated camera."""
        self._set_state(DeviceState.READY)
        self._frame_counter = 0
        logger.info("SimulatedCCD initialized")
    
    def open(self) -> None:
        """Open the simulated camera."""
        self.initialize()
    
    def close(self) -> None:
        """Close the simulated camera."""
        self._set_state(DeviceState.DISCONNECTED)
        self.cam = None
        logger.info("SimulatedCCD closed")
    
    def reset_exposure_time(self, time_ms: int) -> int:
        """Set exposure time.
        
        Args:
            time_ms: Exposure time in milliseconds.
            
        Returns:
            Actual exposure time set.
        """
        self.exposure_time_ms = time_ms
        return time_ms
    
    def reset_window(
        self,
        center: Tuple[int, int],
        size: Tuple[int, int],
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Set ROI window.
        
        Args:
            center: Window center (x, y).
            size: Window size (width, height).
            
        Returns:
            Tuple of (actual_size, actual_center).
        """
        if size == (0, 0):
            return self._resolution, center
        
        actual_size = (
            min(size[0], self._resolution[0]),
            min(size[1], self._resolution[1])
        )
        return actual_size, center
    
    def get_numpy_image(
        self,
        n_sample: int = 1,
        skip_first: bool = True,
    ) -> np.ndarray:
        """Capture image from simulated camera.
        
        Args:
            n_sample: Number of samples to average.
            skip_first: Whether to skip first frame.
            
        Returns:
            Captured image as uint16 array.
        """
        self._set_state(DeviceState.BUSY)
        
        # Simulate exposure time
        time.sleep(self.exposure_time_ms / 1000.0)
        
        if n_sample == 1:
            img = self._generate_image()
        else:
            start_idx = 1 if skip_first else 0
            frames = [
                self._generate_image()
                for _ in range(start_idx, n_sample)
            ]
            img = np.mean(frames, axis=0).astype(np.uint16)
        
        self._last_image = img
        self._frame_counter += 1
        
        self._set_state(DeviceState.READY)
        return img
    
    def _generate_image(self) -> np.ndarray:
        """Generate synthetic image.
        
        Returns:
            Simulated image array.
        """
        width, height = self._resolution
        
        # Create base pattern
        x = np.linspace(0, 4 * np.pi, width)
        y = np.linspace(0, 4 * np.pi, height)
        xx, yy = np.meshgrid(x, y)
        
        # Synthetic pattern: combination of sine waves
        pattern = (
            np.sin(xx) * np.cos(yy) * 1000
            + np.sin(xx * 0.5) * 600
            + np.cos(yy * 0.3) * 400
        )
        
        # Add some "features" (gaussian blobs)
        for _ in range(3):
            cx = self._rng.integers(0, width)
            cy = self._rng.integers(0, height)
            sigma = self._rng.uniform(20, 80)
            blob = np.exp(
                -((xx - cx * 4 * np.pi / width) ** 2 
                  + (yy - cy * 4 * np.pi / height) ** 2) 
                / (2 * sigma ** 2)
            ) * 2000
            pattern += blob
        
        # Scale by exposure time
        exposure_factor = self.exposure_time_ms / 20.0
        pattern *= exposure_factor
        
        # Add noise
        noise = self._rng.normal(0, self._noise_level, pattern.shape)
        pattern += noise
        
        # Clip to valid range
        img = np.clip(pattern, 0, 65535).astype(np.uint16)
        
        return img
    
    def enable_auto_exposure(self, enable: bool = True, mode: int = 1) -> bool:
        """Enable or disable auto exposure.
        
        Args:
            enable: True to enable, False to disable.
            mode: Auto exposure mode.
            
        Returns:
            Success status.
        """
        logger.info(f"Auto exposure {'enabled' if enable else 'disabled'}")
        return True
    
    def set_auto_exposure_target(self, target: int) -> int:
        """Set auto exposure target.
        
        Args:
            target: Target brightness value.
            
        Returns:
            Target value set.
        """
        return target
    
    def get_auto_exposure_state(self) -> dict:
        """Get auto exposure state.
        
        Returns:
            Dictionary with auto exposure settings.
        """
        return {
            "enabled": False,
            "mode": 1,
            "target": 32768,
        }
    
    def set_auto_exposure_range(
        self,
        max_time_ms: int = 350,
        min_time_ms: int = 0,
        max_gain: int = 300,
        min_gain: int = 100,
    ) -> bool:
        """Set auto exposure range.
        
        Args:
            max_time_ms: Maximum exposure time in ms.
            min_time_ms: Minimum exposure time in ms.
            max_gain: Maximum gain.
            min_gain: Minimum gain.
            
        Returns:
            Success status.
        """
        return True
    
    @staticmethod
    def get_cam_list() -> list:
        """Get list of available cameras.
        
        Returns:
            List containing this simulated camera.
        """
        return ["SimulatedCCD_0"]
    
    def _set_state(self, state: str) -> None:
        """Set camera state."""
        self.cam = state
    
    @property
    def resolution(self) -> Tuple[int, int]:
        """Get camera resolution."""
        return self._resolution
    
    @property
    def frame_counter(self) -> int:
        """Get frame counter."""
        return self._frame_counter
