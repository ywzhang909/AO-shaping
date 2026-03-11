"""Base camera driver interface."""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class CameraError(Exception):
    """Base exception for camera errors."""

    pass


class BaseCamera(ABC):
    """Abstract base class for camera drivers.

    All camera drivers should inherit from this class and implement
    the abstract methods. The class provides context manager support
    for automatic resource cleanup.
    """

    def __init__(
        self,
        cam_id: int = 0,
        exposure_time_ms: int = 20,
        skip_sampling: bool = False,
    ):
        """Initialize camera configuration.

        Args:
            cam_id: Camera device index.
            exposure_time_ms: Initial exposure time in milliseconds.
            skip_sampling: Whether to enable binning/skipping for faster capture.
        """
        self.cam_id = int(cam_id)
        self.exposure_time_ms = exposure_time_ms
        self.skip_sampling = skip_sampling

        self.cam = None
        self._sn: str | None = None
        self.cam_width: int = 0
        self.cam_height: int = 0

    def __enter__(self):
        """Context manager entry - initialize camera."""
        self.initialize()
        return self

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit - cleanup camera resources.

        Implementations should close the camera and release resources.
        """
        pass

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the camera device.

        This method should:
        1. Close any previously opened camera device.
        2. Find and open the specified camera by cam_id.
        3. Set initial exposure time, gain, and pixel format.
        4. Configure resolution and binning if skip_sampling is True.
        5. Start the data stream.

        Raises:
            ConnectionAbortedError: If camera device is not found.
            CameraError: If camera initialization fails.
        """
        pass

    @abstractmethod
    def open(self) -> None:
        """Open the camera device (alias for initialize).

        This is an explicit method for non-context-manager usage.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the camera device and release resources."""
        pass

    @abstractmethod
    def reset_exposure_time(self, time_ms: int) -> int:
        """Set the camera exposure time.

        Args:
            time_ms: New exposure time in milliseconds.

        Returns:
            int: Actual exposure time set in milliseconds.

        Raises:
            AssertionError: If camera is not initialized.
        """
        pass

    @abstractmethod
    def reset_window(
        self,
        center: Tuple[int, int] | Tuple[np.intp, ...],
        size: Tuple[int, int],
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Reset the camera ROI window size and position.

        Args:
            center: Expected window center position (x, y).
            size: Expected window size (width, height).
                Use (0, 0) for maximum resolution.

        Returns:
            Tuple of ((width, height), (center_x, center_y)) actually set.

        Raises:
            AssertionError: If camera is not initialized or center is invalid.
        """
        pass

    @abstractmethod
    def get_numpy_image(
        self,
        n_sample: int = 1,
        skip_first: bool = True,
    ) -> np.ndarray:
        """Capture image(s) from camera with optional averaging.

        Args:
            n_sample: Number of samples to average. Must be > 0.
            skip_first: Whether to skip first frame (often unstable).

        Returns:
            np.ndarray: Captured image as uint8 array.

        Raises:
            AssertionError: If n_sample is not positive.
        """
        pass

    @abstractmethod
    def enable_auto_exposure(self, enable: bool = True, mode: int = 1) -> bool:
        """Enable or disable auto exposure.

        Args:
            enable: True to enable, False to disable.
            mode: Auto exposure mode (implementation specific).

        Returns:
            bool: True if successful, False if not supported.
        """
        pass

    @abstractmethod
    def set_auto_exposure_target(self, target: int) -> int:
        """Set auto exposure target brightness.

        Args:
            target: Target brightness value (implementation specific range).

        Returns:
            int: The target value that was set.

        Raises:
            NotImplementedError: If not supported by the camera.
        """
        pass

    @abstractmethod
    def get_auto_exposure_state(self) -> dict:
        """Get current auto exposure state.

        Returns:
            Dictionary containing:
                - enabled: bool - Whether auto exposure is enabled
                - mode: int - Current mode
                - target: int - Current target brightness
        """
        pass

    @abstractmethod
    def set_auto_exposure_range(
        self,
        max_time_ms: int = 350,
        min_time_ms: int = 0,
        max_gain: int = 300,
        min_gain: int = 100,
    ) -> bool:
        """Set auto exposure time and gain range.

        Args:
            max_time_ms: Maximum exposure time in ms.
            min_time_ms: Minimum exposure time in ms.
            max_gain: Maximum gain value.
            min_gain: Minimum gain value.

        Returns:
            bool: True if successful, False if not supported.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_cam_list():
        """Get list of available camera devices.

        Returns:
            List of available camera devices (implementation specific format).
        """
        pass

    def _get_grid(self, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate coordinate grids for the image.

        Args:
            width: Image width.
            height: Image height.

        Returns:
            Tuple of (xv, yv) meshgrid arrays.
        """
        x = np.arange(0, width)
        y = np.arange(0, height)
        xv, yv = np.meshgrid(x, y)
        return xv, yv
