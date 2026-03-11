"""MIICAM camera driver with Device base class integration.

This module provides a MIICAM camera implementation that inherits from
the Device base class, enabling digital twin management and unified
device interface.
"""

import os
import sys
import ctypes
from datetime import datetime
from typing import Any

import numpy as np

from loguru import logger
from ao_shaping.drivers.device_base import Device, DeviceError, DeviceState, DeviceType

# Vendored MIICAM SDK - bundled in this project
_MII_SDK_PATH = os.path.join(os.path.dirname(__file__), "_miicam_sdk")
if _MII_SDK_PATH not in sys.path:
    sys.path.insert(0, _MII_SDK_PATH)

import miicam


class MIICAMDeviceError(DeviceError):
    """Exception raised for MIICAM camera errors."""

    pass


class MIICAMDevice(Device):
    """MIICAM camera driver with Device base class.

    This class provides a unified interface for MIICAM cameras,
    integrating with the digital twin management system.

    Attributes:
        device_type: DeviceType.CAMERA
        manufacturer: "MIICAM"
        model: "USB3.0 Camera"

    Example:
        >>> from ao_shaping.drivers.device_registry import DeviceRegistry
        >>> registry = DeviceRegistry()
        >>>
        >>> cam = MIICAMDevice(device_id="cam_001", cam_id=0)
        >>> registry.register(cam, alias="main_camera", tags=["imaging"])
        >>>
        >>> with cam:
        ...     img = cam.capture()
        ...     print(f"Captured image: {img.shape}")
    """

    device_type = DeviceType.CAMERA
    manufacturer = "MIICAM"
    model = "USB3.0 Camera"

    def __init__(
        self,
        device_id: str = "",
        cam_id: int = 0,
        exposure_time_ms: float = 20.0,
        skip_sampling: bool = False,
    ):
        """Initialize MIICAM device.

        Args:
            device_id: Unique device identifier.
            cam_id: Camera device index.
            exposure_time_ms: Initial exposure time in milliseconds.
            skip_sampling: Whether to enable 2x2 binning.
        """
        super().__init__(device_id)

        self._cam_id = int(cam_id)
        self._skip_sampling = skip_sampling
        self._pixel_format = "MONO8"

        # Camera handle
        self._cam = None
        self._frame_buffer = None

        # Resolution
        self._width = 0
        self._height = 0

        # Register parameters
        self._register_parameters()

        # Register capabilities
        self._register_capabilities()

        # Set initial parameter value
        self.set_parameter_value("exposure_time_ms", exposure_time_ms)

    def _register_parameters(self) -> None:
        """Register camera-specific parameters."""
        self.register_parameter(
            "exposure_time_ms",
            default_value=20.0,
            min_value=0.1,
            max_value=10000.0,
            unit="ms",
            description="Exposure time in milliseconds",
            writable=True,
        )
        self.register_parameter(
            "gain",
            default_value=100,
            min_value=100,
            max_value=500,
            unit="",
            description="Analog gain (100 = 1x)",
            writable=True,
        )
        self.register_parameter(
            "auto_exposure",
            default_value=False,
            unit="",
            description="Enable auto exposure",
            writable=True,
        )
        self.register_parameter(
            "binning",
            default_value=False,
            unit="",
            description="Enable 2x2 binning",
            writable=False,  # Set at initialization
        )
        self.register_parameter(
            "pixel_format",
            default_value="MONO8",
            unit="",
            description="Pixel format (MONO8 or YUV422)",
            writable=False,
        )
        self.register_parameter(
            "frame_rate",
            default_value=30.0,
            min_value=1.0,
            max_value=120.0,
            unit="fps",
            description="Frame rate",
            writable=False,
        )
        self.register_parameter(
            "capture_delay_ms",
            default_value=50.0,
            min_value=0.0,
            max_value=500.0,
            unit="ms",
            description="Frame capture delay before pulling image",
            writable=True,
        )

    def _register_capabilities(self) -> None:
        """Register camera capabilities."""
        self.register_capability(
            "capture",
            description="Capture single image",
            parameters=["exposure_time_ms"],
            return_type=np.ndarray,
        )
        self.register_capability(
            "capture_average",
            description="Capture averaged image",
            parameters=["exposure_time_ms", "n_samples"],
            return_type=np.ndarray,
        )
        self.register_capability(
            "set_roi",
            description="Set region of interest",
            parameters=["center_x", "center_y", "width", "height"],
        )
        self.register_capability(
            "get_resolution",
            description="Get current resolution",
            return_type=tuple,
        )

    def open(self) -> None:
        """Open and initialize the camera.

        Raises:
            MIICAMDeviceError: If camera initialization fails.
            ConnectionError: If camera is not found.
        """
        if self._cam is not None:
            logger.warning(f"Camera {self.device_id} already open")
            return

        self._set_state(DeviceState.CONNECTING)

        try:
            # Update device list
            dev_list = miicam.Miicam.EnumV2()
            if not dev_list or len(dev_list) <= self._cam_id:
                raise ConnectionError(f"Camera ID {self._cam_id} not found")

            # Open camera
            self._cam = miicam.Miicam.Open(dev_list[self._cam_id].id)
            if not self._cam:
                raise MIICAMDeviceError("Failed to open camera")

            # Configure camera
            self._configure_camera()

            # Start streaming
            self._start_streaming()

            # Update metadata
            self._metadata.serial_number = self._get_serial_number()
            self._metadata.last_seen = datetime.now()

            self._set_state(DeviceState.READY)
            logger.info(f"Camera {self.device_id} opened successfully")

        except Exception as e:
            self._set_state(DeviceState.ERROR, str(e))
            raise

    def close(self) -> None:
        """Close camera and release resources."""
        if self._cam:
            try:
                self._cam.Stop()
                self._cam.Close()
            except Exception as e:
                logger.warning(f"Error closing camera: {e}")
            finally:
                self._cam = None
                self._width = 0
                self._height = 0
                self._frame_buffer = None

        self._set_state(DeviceState.DISCONNECTED)
        logger.info(f"Camera {self.device_id} closed")

    def is_connected(self) -> bool:
        """Check if camera is connected and ready."""
        return self._cam is not None and self._state == DeviceState.READY

    def get_hardware_info(self) -> dict[str, Any]:
        """Get camera hardware information."""
        return {
            "cam_id": self._cam_id,
            "serial_number": self._metadata.serial_number,
            "resolution": (self._width, self._height),
            "pixel_format": self._pixel_format,
            "pixel_size_um": 3.45,  # Typical for USB3.0 cameras
        }

    def _configure_camera(self) -> None:
        """Configure camera settings."""
        # Disable auto exposure
        self._cam.put_AutoExpoEnable(0)

        # Set exposure time (convert ms to microseconds)
        exposure_ms = self.get_parameter_value("exposure_time_ms")
        self._cam.put_ExpoTime(int(exposure_ms * 1000))

        # Set gain
        gain = self.get_parameter_value("gain")
        self._cam.put_ExpoAGain(gain)

        # Set pixel format
        try:
            self._cam.put_Option(miicam.MIICAM_OPTION_RGB, 3)  # MONO8
            self._pixel_format = "MONO8"
        except miicam.HRESULTException:
            self._pixel_format = "YUV422"

        # Set binning if requested
        if self._skip_sampling:
            try:
                self._cam.put_Option(miicam.MIICAM_OPTION_BINNING, 0x80 | 2)
                self._parameters["binning"].value = True
            except miicam.HRESULTException:
                logger.warning("Binning not supported")

        # Set to maximum resolution
        max_width, max_height = self._cam.get_Resolution(0)
        self._cam.put_Size(max_width, max_height)
        self._width, self._height = self._cam.get_Size()

    def _start_streaming(self) -> None:
        """Start camera streaming with callback."""
        self._frame_buffer = None

        def frame_callback(nEvent, ctx):
            if nEvent == miicam.MIICAM_EVENT_IMAGE:
                try:
                    bufsize = ctx._width * ctx._height
                    buffer = bytearray(bufsize)
                    ctx._cam.PullImageV4(buffer, 0, 8, 0, None)
                    ctx._frame_buffer = buffer
                except Exception as e:
                    logger.warning(f"Frame callback error: {e}")

        self._cam.StartPullModeWithCallback(frame_callback, self)

    def _get_serial_number(self) -> str:
        """Get camera serial number."""
        try:
            return self._cam.SerialNumber()
        except Exception:
            return f"MIICAM_{self._cam_id}"

    def _on_parameter_changed(self, name: str, old_value: Any, new_value: Any) -> None:
        """Handle parameter changes."""
        if not self._cam:
            return

        if name == "exposure_time_ms":
            self._cam.put_ExpoTime(int(new_value * 1000))
            logger.debug(f"Exposure time set to {new_value}ms")

        elif name == "gain":
            self._cam.put_ExpoAGain(new_value)
            logger.debug(f"Gain set to {new_value}")

        elif name == "auto_exposure":
            mode = 1 if new_value else 0
            self._cam.put_AutoExpoEnable(mode)
            logger.debug(f"Auto exposure {'enabled' if new_value else 'disabled'}")

    def capture(self, n_samples: int = 1, skip_first: bool = True) -> np.ndarray:
        """Capture image from camera.

        Args:
            n_samples: Number of samples for averaging.
            skip_first: Whether to skip first frame.

        Returns:
            Captured image as uint8 array.

        Raises:
            RuntimeError: If camera not connected.
        """
        if not self.is_connected():
            raise RuntimeError("Camera not connected")

        self._set_state(DeviceState.BUSY)
        try:
            img = self._capture_internal(n_samples, skip_first)
            self._emit_data("image", img)
            return img
        finally:
            self._set_state(DeviceState.READY)

    def _capture_internal(self, n_samples: int, skip_first: bool) -> np.ndarray:
        """Internal capture implementation."""
        import time

        def take_one_shot() -> np.ndarray:
            # Calculate buffer size
            if self._pixel_format == "YUV422":
                bufsize = self._width * self._height * 2
            else:
                bufsize = self._width * self._height

            buffer = (ctypes.c_char * bufsize)()
            delay_ms = self.get_parameter_value("capture_delay_ms")
            time.sleep(delay_ms / 1000.0)

            # Pull image with retry
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    self._cam.PullImageV4(buffer, 0, 8, 0, None)
                    break
                except miicam.HRESULTException as e:
                    logger.debug(f"PullImageV4 attempt {attempt + 1}/{max_retries} failed: {e}")
                    time.sleep(0.05)
            else:
                raise MIICAMDeviceError("Failed to capture image after retries")

            # Convert to numpy
            img_data = np.frombuffer(buffer, dtype=np.uint8)

            if self._pixel_format == "YUV422":
                img_yuv = img_data.reshape((self._height, self._width * 2))
                return img_yuv[:, ::2]  # Extract Y channel
            else:
                return img_data.reshape((self._height, self._width))

        # Capture samples
        first_img = take_one_shot()
        if n_samples == 1:
            return first_img

        avg_img = np.zeros_like(first_img) if skip_first else first_img.copy()
        count = n_samples if skip_first else n_samples - 1

        for _ in range(count):
            avg_img = avg_img + take_one_shot()

        return (avg_img / count).astype(np.uint8)

    def set_roi(
        self,
        center: tuple[int, int] = (0, 0),
        size: tuple[int, int] = (0, 0),
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Set region of interest.

        Args:
            center: ROI center (x, y).
            size: ROI size (width, height). Use (0, 0) for max.

        Returns:
            Tuple of ((width, height), (center_x, center_y)).
        """
        if not self.is_connected():
            raise RuntimeError("Camera not connected")

        self._set_state(DeviceState.BUSY)
        try:
            # Stop streaming
            try:
                self._cam.Stop()
            except Exception:
                pass

            if size == (0, 0):
                width, height = self._cam.get_Resolution(0)
                x_offset, y_offset = 0, 0
            else:
                width, height = size
                x_offset = max(0, center[0] - width // 2)
                y_offset = max(0, center[1] - height // 2)

            # Set size
            self._cam.put_Size(width, height)
            self._width, self._height = self._cam.get_Size()

            # Restart streaming
            self._start_streaming()

            return (self._width, self._height), (self._width // 2, self._height // 2)
        finally:
            self._set_state(DeviceState.READY)

    def get_resolution(self) -> tuple[int, int]:
        """Get current resolution."""
        return (self._width, self._height)

    @staticmethod
    def list_cameras():
        """List available cameras."""
        return miicam.Miicam.EnumV2()

    def get_twin_state(self) -> dict[str, Any]:
        """Get state for digital twin synchronization."""
        state = super().get_twin_state()
        state["hardware"] = {
            "cam_id": self._cam_id,
            "resolution": (self._width, self._height),
            "pixel_format": self._pixel_format,
        }
        return state
