import ctypes
import threading
from typing import Callable

import numpy as np
from loguru import logger

from ao_shaping.drivers.ccd.miicam._sdk_setup import _setup_miicam_sdk

# Set up MIICAM SDK before importing
_MIICAM_AVAILABLE = _setup_miicam_sdk()

if _MIICAM_AVAILABLE:
    import miicam

    from ao_shaping.drivers.ccd.base import BaseCamera, CameraError
else:
    miicam = None

    class CameraError(Exception):
        pass

    class BaseCamera:
        pass


class MIICAMError(CameraError):
    """Exception raised for MIICAM camera errors."""

    pass


class CameraStreamManager(BaseCamera):
    def __init__(
        self,
        cam_id: int = 0,
        exposure_time_ms: float = 20.0,
        skip_sampling: bool = False,
        bit_depth: int = 8,
    ):
        """Initialize MIICAM camera.

        Args:
            cam_id: Camera index
            exposure_time_ms: Exposure time in milliseconds
            skip_sampling: Enable 2x2 binning
            bit_depth: Output bit depth (8 or 16). 8-bit for MONO8, 16-bit for
                full sensor bit depth (e.g., 12-bit sensor data in 16-bit container).
                If the camera doesn't support high bit depth, it falls back to 8-bit.
        """
        super().__init__(cam_id, exposure_time_ms, skip_sampling)
        self._bit_depth = bit_depth
        self._pixel_format = "MONO8" if bit_depth == 8 else "MONO16"
        self._max_bit_depth = 8  # Will be updated from camera

    @property
    def cam_type(self) -> str:
        return "miicam"

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def open(self) -> None:
        """Open the camera device (alias for initialize)."""
        self.initialize()

    def close(self) -> None:
        """Close the camera device and release resources."""
        if self.cam:
            import time

            # Stop streaming with retry - the callback thread may need time to exit
            for _ in range(5):
                try:
                    self.cam.Stop()
                    break
                except miicam.HRESULTException:
                    time.sleep(0.1)
                except Exception:
                    break

            # Flush any pending frames
            try:
                self.cam.put_Option(miicam.MIICAM_OPTION_FLUSH, 3)
            except Exception:
                pass

            # Delay to let callback thread finish
            time.sleep(0.3)

            try:
                self.cam.Close()
            except Exception:
                pass

            self.cam_width = 0
            self.cam_height = 0
            self.cam = None

            # Give the camera hardware a moment to settle before re-opening
            time.sleep(0.5)

    def initialize(self) -> None:
        """
        Initialize the camera device.

        This method performs the following operations:
        1. Close any previously opened camera device (if any).
        2. Update device list and check if sufficient devices exist.
        3. Open the specified camera device.
        4. Set exposure time, gain, pixel format, binning, offset, width, and height.
        5. Update camera properties and start data streaming.

        If no camera device is found, an error is logged and ConnectionAbortedError is raised.
        """
        # Close previously opened camera device (if any)
        self.__exit__(None, None, None)

        # Pre-emptive: try to stop any stale stream from a previous session
        # This handles cases where the camera was left streaming from a prior run
        import time as _init_time

        try:
            # Try to open and immediately close to clear any stale state
            dev_list = miicam.Miicam.EnumV2()
            if dev_list and len(dev_list) > self.cam_id:
                temp_cam = miicam.Miicam.Open(dev_list[self.cam_id].id)
                if temp_cam:
                    try:
                        temp_cam.Stop()
                    except Exception:
                        pass
                    try:
                        temp_cam.put_Option(miicam.MIICAM_OPTION_FLUSH, 3)
                    except Exception:
                        pass
                    _init_time.sleep(0.5)
                    try:
                        temp_cam.Close()
                    except Exception:
                        pass
                    _init_time.sleep(1.0)
        except Exception:
            pass

        # Update device list and get device info list
        dev_list = miicam.Miicam.EnumV2()
        if not dev_list or len(dev_list) <= self.cam_id:
            error_info = f"Camera ID {self.cam_id} not found. "
            if dev_list:
                error_info += f" Available cameras: {[_.id for _ in dev_list]}."
            logger.error(error_info)
            raise ConnectionAbortedError(error_info)

        # Open camera by index
        self.cam = miicam.Miicam.Open(dev_list[self.cam_id].id)
        if not self.cam:
            raise MIICAMError("Failed to open camera")

        # Disable auto exposure (some cameras may not support this during init)
        try:
            self.cam.put_AutoExpoEnable(0)
        except miicam.HRESULTException:
            logger.warning("Could not disable auto exposure during init")

        # Set exposure time (convert ms to microseconds; SDK expects integer)
        try:
            self.cam.put_ExpoTime(int(self.exposure_time_ms * 1000))
        except miicam.HRESULTException:
            logger.warning(f"Could not set exposure time to {self.exposure_time_ms}ms")

        # Set gain (default 100 = 1x)
        try:
            self.cam.put_ExpoAGain(100)
        except miicam.HRESULTException:
            logger.warning("Could not set exposure gain")

        # Get camera's maximum bit depth
        self._max_bit_depth = self.cam.MaxBitDepth()

        # Set output bit depth (8 or 16)
        # MIICAM_OPTION_BITDEPTH: 0 = 8 bits, 1 = 16 bits
        # "16-bit" mode outputs full sensor bit depth in 16-bit container
        # (e.g., 12-bit sensor data in 16-bit container)
        if self._bit_depth == 8:
            output_bitdepth = 0
        elif self._max_bit_depth > 8:
            # Camera supports high bit depth, enable 16-bit output
            output_bitdepth = 1
        else:
            output_bitdepth = 0
            logger.warning(
                f"Camera max bit depth is {self._max_bit_depth}, "
                f"requested {self._bit_depth}-bit. Falling back to 8-bit."
            )
            self._bit_depth = 8

        try:
            self.cam.put_Option(miicam.MIICAM_OPTION_BITDEPTH, output_bitdepth)
        except miicam.HRESULTException:
            logger.warning(f"Could not set bit depth to {self._bit_depth}, using 8-bit")
            self._bit_depth = 8

        # Set pixel format based on bit depth
        # MIICAM_OPTION_RGB: 3 = 8-bit Grey, 4 = 16-bit Grey
        # For Bayer sensors, use RAW mode for high bit depth to get full sensor data
        if self._bit_depth == 8:
            self._pixel_format = "MONO8"
            try:
                self.cam.put_Option(miicam.MIICAM_OPTION_RGB, 3)
            except miicam.HRESULTException:
                logger.warning("Could not set MONO8 format, using default")
            # Ensure RAW mode is off for 8-bit
            try:
                self.cam.put_Option(miicam.MIICAM_OPTION_RAW, 0)
            except miicam.HRESULTException:
                pass
        else:
            # For high bit depth mode, try RAW mode first (works with Bayer sensors)
            self._pixel_format = "MONO16"
            raw_mode_set = False
            try:
                self.cam.put_Option(miicam.MIICAM_OPTION_RAW, 1)
                raw_mode_set = True
                logger.info("RAW mode enabled for high bit depth capture")
            except miicam.HRESULTException:
                # Fall back to 16-bit Grey if RAW mode not supported
                try:
                    self.cam.put_Option(miicam.MIICAM_OPTION_RGB, 4)
                    logger.info("16-bit Grey format enabled")
                except miicam.HRESULTException:
                    logger.warning(
                        "Could not set high bit depth mode (RAW or 16-bit Grey), "
                        "falling back to 8-bit"
                    )
                    self._pixel_format = "MONO8"
                    self._bit_depth = 8
                    try:
                        self.cam.put_Option(miicam.MIICAM_OPTION_RGB, 3)
                    except miicam.HRESULTException:
                        logger.warning("Could not set pixel format, using default")
                    try:
                        self.cam.put_Option(miicam.MIICAM_OPTION_RAW, 0)
                    except miicam.HRESULTException:
                        pass

        # Set binning if skip_sampling is True
        # Note: Not all cameras support binning, wrap in try-except
        if self.skip_sampling:
            try:
                # 0x80 | 2 = 2x2 average binning
                self.cam.put_Option(miicam.MIICAM_OPTION_BINNING, 0x80 | 2)
            except miicam.HRESULTException:
                logger.warning(
                    "MIICAM_OPTION_BINNING not supported, continuing without binning"
                )

        # Get maximum resolution
        max_width, max_height = self.cam.get_Resolution(0)

        # Set to maximum resolution (no offset, full ROI)
        self.cam.put_Size(max_width, max_height)

        # Get actual image size after all settings
        self.cam_width, self.cam_height = self.cam.get_Size()

        # Check actual raw format after all settings
        raw_fmt, _ = self.cam.get_RawFormat()
        # Convert bytes to string for comparison
        if isinstance(raw_fmt, bytes):
            raw_fmt_str = raw_fmt.decode("ascii", errors="replace")
        else:
            raw_fmt_str = raw_fmt

        if raw_fmt_str in ("YUYV", "VUYY", "UYVY"):
            self._pixel_format = "YUV422"
            logger.info(f"Camera using YUV422 format (raw: {raw_fmt_str})")
        elif raw_fmt_str in ("YMono",):
            logger.info(
                f"Camera using MONO{self._bit_depth} format (raw: {raw_fmt_str})"
            )
        elif raw_fmt_str in ("BGGR", "RGGB", "GRBG", "GBRG"):
            # Bayer pattern sensor - treated as mono for our purposes
            logger.info(
                f"Camera using Bayer {raw_fmt_str} format, treating as MONO{self._bit_depth}"
            )
        else:
            logger.info(
                f"Camera using format: {raw_fmt_str}, treating as MONO{self._bit_depth}"
            )

        # Get serial number
        try:
            self._sn = self.cam.SerialNumber()
        except Exception:
            self._sn = f"MIICAM_{self.cam_id}"

        # Start streaming (pull mode without callback - we use WaitImageV3)
        import time as _time

        # Pre-emptively try to stop any stale stream from a previous session
        try:
            self.cam.Stop()
        except Exception:
            pass
        _time.sleep(0.5)

        max_retries = 10
        for attempt in range(max_retries):
            try:
                self.cam.StartPullModeWithCallback(None, None)
                break
            except miicam.HRESULTException:
                if attempt < max_retries - 1:
                    try:
                        self.cam.Stop()
                    except Exception:
                        pass
                    _time.sleep(0.5 + 0.5 * attempt)
                else:
                    raise

        self.__update_properties()

    def reset_exposure_time(self, time_ms: float) -> float:
        """
        Reset the camera exposure time.

        Args:
            time_ms (float): New exposure time in milliseconds.
                Valid range: 0.011ms to 10000ms. Values outside this range are clamped.

        Returns:
            float: Actual exposure time set in milliseconds.
        """
        assert self.cam, "camera not initialized"
        # Clamp to valid range: 0.011ms to 10000ms
        if time_ms < 0.011:
            self.exposure_time_ms = 0.011
            logger.warning("exposure time must >= 0.011ms. clamped to 0.011ms.")
        elif time_ms > 10000:
            self.exposure_time_ms = 10000.0
            logger.warning("exposure time must <= 10000ms. clamped to 10000ms.")
        else:
            self.exposure_time_ms = time_ms
        # Convert ms to microseconds for MIICAM (SDK expects integer microseconds)
        self.cam.put_ExpoTime(int(self.exposure_time_ms * 1000))
        return self.exposure_time_ms

    def enable_auto_exposure(self, enable: bool = True, mode: int = 1) -> bool:
        """
        Enable or disable auto exposure.

        Args:
            enable (bool): True to enable, False to disable.
            mode (int): Auto exposure mode:
                0 = disable
                1 = continuous mode (default)
                2 = once mode

        Returns:
            bool: True if successful.
        """
        assert self.cam, "camera not initialized"
        mode_value = 1 if enable else 0
        if enable and mode > 0:
            mode_value = mode
        self.cam.put_AutoExpoEnable(mode_value)
        return True

    def set_auto_exposure_target(self, target: int) -> int:
        """
        Set auto exposure target brightness.

        Args:
            target (int): Target brightness value. Range: 16-220, default: 120.

        Returns:
            int: The target value that was set.
        """
        assert self.cam, "camera not initialized"
        # Clamp to valid range
        target = max(16, min(220, target))
        try:
            self.cam.put_AutoExpoTarget(target)
        except miicam.HRESULTException:
            logger.warning("Auto exposure target not supported")
        return target

    def get_auto_exposure_state(self) -> dict:
        """
        Get current auto exposure state.

        Returns:
            dict: Dictionary containing:
                - enabled: bool - Whether auto exposure is enabled
                - mode: int - Current mode (0=off, 1=continuous, 2=once)
                - target: int - Current target brightness
        """
        assert self.cam, "camera not initialized"
        state = {
            "enabled": False,
            "mode": 0,
            "target": 120,
        }
        try:
            state["mode"] = self.cam.get_AutoExpoEnable()
            state["enabled"] = state["mode"] > 0
            state["target"] = self.cam.get_AutoExpoTarget()
        except miicam.HRESULTException:
            logger.warning("Auto exposure not supported")
        return state

    def set_auto_exposure_range(
        self,
        max_time_ms: int = 350,
        min_time_ms: int = 0,
        max_gain: int = 300,
        min_gain: int = 100,
    ) -> bool:
        """
        Set auto exposure time and gain range.

        Args:
            max_time_ms (int): Maximum exposure time in ms (default: 350)
            min_time_ms (int): Minimum exposure time in ms (default: 0)
            max_gain (int): Maximum gain (default: 300)
            min_gain (int): Minimum gain (default: 100)

        Returns:
            bool: True if successful.
        """
        assert self.cam, "camera not initialized"
        try:
            # Convert ms to microseconds
            self.cam.put_AutoExpoRange(
                max_time_ms * 1000,
                min_time_ms * 1000,
                max_gain,
                min_gain,
            )
        except miicam.HRESULTException:
            logger.warning("Auto exposure range not supported")
            return False
        return True

    def reset_window(
        self,
        center: tuple[int, int] | tuple[np.intp, ...] = (0, 0),
        size: tuple[int, int] = (0, 0),
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """
        Reset the camera window size and position to ensure the image center is at the specified position.

        Args:
            size (Tuple[int]): Expected window size in format (width, height).
            center (Tuple[int]): Expected window center position in format (x, y).

        Returns:
            Tuple[int]: New window center position in format (x, y).
        """
        assert self.cam, "camera not initialized"
        center = tuple(int(c) for c in center)

        # Stop camera first before changing size
        try:
            self.cam.Stop()
        except Exception:
            pass

        # If window size is not specified, use maximum width and height
        if size == (0, 0):
            width, height = self.cam.get_Resolution(0)
            x_offset, y_offset = 0, 0
        else:
            width, height = size
            # Get resolution step (minimum increment)
            # MIICAM uses put_Size with width/height directly
            # No special binning requirements visible in the SDK
            width = int(width)
            height = int(height)
            # Calculate offset to center the window at specified position
            x_offset = center[0] - (width // 2)
            y_offset = center[1] - (height // 2)
            x_offset = max(0, x_offset)
            y_offset = max(0, y_offset)

        assert x_offset >= 0 and y_offset >= 0, (
            f"Window center position: {center} must be within image, window size: {size}"
        )

        # Set window size
        try:
            self.cam.put_Size(width, height)
        except miicam.HRESULTException as e:
            logger.warning(f"put_Size failed: {e}, trying with max resolution")
            width, height = self.cam.get_Resolution(0)
            self.cam.put_Size(width, height)

        self.__update_properties()

        # Restart streaming (pull mode without callback - we use WaitImageV3)
        self.cam.StartPullModeWithCallback(None, None)

        # Return new window size and center
        return (width, height), (width // 2, height // 2)

    def __take_one_shot(self) -> np.ndarray:
        """
        Capture a single camera image using WaitImageV3.

        Returns:
            np.ndarray: Captured image data as uint8 (8-bit mode) or uint16 (14-bit mode) array.
        """
        import ctypes
        import time

        # Determine buffer size and bit depth
        if getattr(self, "_pixel_format", None) == "YUV422":
            bufsize = self.cam_width * self.cam_height * 2
            bits = 8
            dtype = np.uint8
        elif self._bit_depth == 8:
            bufsize = self.cam_width * self.cam_height
            bits = 8
            dtype = np.uint8
        else:
            bufsize = self.cam_width * self.cam_height * 2
            bits = 16
            dtype = np.uint16

        # Allocate buffer and frame info
        buffer = (ctypes.c_char * bufsize)()
        frame_info = miicam.MiicamFrameInfoV3()

        # Wait for frame with retry logic
        # WaitImageV3: blocks until frame arrives or timeout (1000ms)
        # Returns E_PENDING (0x8000000A) or 0x8001011f on timeout
        max_retries = 5
        for attempt in range(max_retries):
            try:
                self.cam.WaitImageV3(1000, buffer, 0, bits, 0, frame_info)
                # Success
                break
            except miicam.HRESULTException as e:
                # Check if it's a timeout error
                hr = e.hr if hasattr(e, "hr") else e.winerror if hasattr(e, "winerror") else 0
                if hr in (0x8000000A, 0x8001011f):
                    # Timeout - retry
                    if attempt < max_retries - 1:
                        time.sleep(0.05)
                        continue
                    raise MIICAMError(
                        f"WaitImageV3 timeout after {max_retries} retries"
                    ) from e
                # Other error - retry
                if attempt < max_retries - 1:
                    time.sleep(0.05)
                    continue
                raise MIICAMError(
                    f"WaitImageV3 failed: hr=0x{hr & 0xFFFFFFFF:08x}"
                ) from e

        # Convert buffer to numpy array
        img_data = np.frombuffer(buffer, dtype=dtype)

        # Handle YUV422 format - extract Y channel (luminance)
        if getattr(self, "_pixel_format", None) == "YUV422":
            img_yuv = img_data.reshape((self.cam_height, self.cam_width * 2))
            img = img_yuv[:, ::2]  # Take every other column (Y channel only)
        else:
            # MONO8/MONO16 or other formats
            img = img_data.reshape((self.cam_height, self.cam_width))

        return img

    # =========================================================================
    # Callback-based capture (for high-FPS or continuous streaming)
    # =========================================================================

    def start_callback_mode(
        self,
        callback = None,
    ) -> None:
        """Start pull mode with a callback function for frame notification.

        The callback runs in an internal SDK thread. Keep it fast — do NOT
        perform heavy processing or call back into the SDK from within it.

        Args:
            callback: Function called on each new frame with (image, frame_info).
                      If None, uses the internal buffer callback.
        """
        assert self.cam, "camera not initialized"
        self._stop_streaming()

        self._callback_user = callback
        self._callback_buffer_lock = threading.Lock()
        self._callback_buffer = None
        self._callback_frame_info = None
        self._callback_new_frame = threading.Event()

        if callback is None:
            cb = self.__frame_callback
        else:
            cb = lambda nEvent, ctx: ctx.__frame_callback(nEvent, ctx, callback)

        self.cam.StartPullModeWithCallback(cb, self)
        logger.info("Callback mode started")

    def stop_callback_mode(self) -> None:
        """Stop callback mode and release callback resources."""
        self._stop_streaming()
        self._callback_user = None
        self._callback_buffer = None
        self._callback_frame_info = None
        logger.info("Callback mode stopped")

    def __frame_callback(
        self,
        nEvent: int,
        ctx: object,
        user_callback: Callable[[np.ndarray, miicam.MiicamFrameInfoV3], None] | None = None,
    ) -> None:
        """Internal SDK callback: called when a new frame is available."""
        if nEvent == miicam.MIICAM_EVENT_IMAGE:
            try:
                bufsize, bits, dtype = self.__get_buffer_params()
                buffer = (ctypes.c_char * bufsize)()
                frame_info = miicam.MiicamFrameInfoV3()
                self.cam.PullImageV4(buffer, 0, bits, 0, frame_info)
                img_data = np.frombuffer(buffer, dtype=dtype)
                img = self.__decode_image(img_data)

                if user_callback is not None:
                    user_callback(img, frame_info)
                else:
                    with self._callback_buffer_lock:
                        self._callback_buffer = img
                        self._callback_frame_info = frame_info
                        self._callback_new_frame.set()
            except Exception:
                pass
        elif nEvent == miicam.MIICAM_EVENT_STILLIMAGE:
            try:
                bufsize, bits, dtype = self.__get_buffer_params()
                buffer = (ctypes.c_char * bufsize)()
                frame_info = miicam.MiicamFrameInfoV3()
                self.cam.PullStillImageV2(buffer, bits, frame_info)
                img_data = np.frombuffer(buffer, dtype=dtype)
                img = self.__decode_image(img_data)

                if user_callback is not None:
                    user_callback(img, frame_info)
                else:
                    with self._callback_buffer_lock:
                        self._callback_buffer = img
                        self._callback_frame_info = frame_info
                        self._callback_new_frame.set()
            except Exception:
                pass

    def get_callback_frame(
        self, timeout: float = 1.0
    ) -> tuple[np.ndarray, miicam.MiicamFrameInfoV3] | None:
        """Get the latest frame from callback mode.

        Args:
            timeout: Maximum time to wait for a new frame in seconds.

        Returns:
            Tuple of (image, frame_info) or None if timeout.
        """
        if self._callback_new_frame.wait(timeout=timeout):
            with self._callback_buffer_lock:
                self._callback_new_frame.clear()
                return (self._callback_buffer, self._callback_frame_info)
        return None

    # =========================================================================
    # Trigger mode (software trigger)
    # =========================================================================

    def set_trigger_mode(self, mode: int) -> None:
        """Set camera trigger mode.

        Args:
            mode: 0 = video mode (default)
                  1 = software / simulated trigger
                  2 = external trigger (rising edge)
                  3 = external + software trigger
        """
        assert self.cam, "camera not initialized"
        self._stop_streaming()
        self.cam.put_Option(miicam.MIICAM_OPTION_TRIGGER, mode)
        logger.info("Trigger mode set to {}", mode)

    def get_trigger_mode(self) -> int:
        """Get current trigger mode.

        Returns:
            Current trigger mode (0=video, 1=software, 2=external, 3=both).
        """
        assert self.cam, "camera not initialized"
        return self.cam.get_Option(miicam.MIICAM_OPTION_TRIGGER)

    def trigger(self, n_images: int = 1) -> None:
        """Send a software trigger.

        The camera must be in trigger mode (set_trigger_mode(1)) and
        streaming (StartPullModeWithCallback) before calling this.

        Args:
            n_images: Number of images to capture.
                      0 = cancel trigger, 0xFFFF = continuous.
        """
        assert self.cam, "camera not initialized"
        self.cam.Trigger(n_images)

    def trigger_sync(
        self,
        n_images: int = 1,
        timeout_ms: int = 0,
    ) -> np.ndarray:
        """Software trigger and wait for image synchronously.

        Combines Trigger + WaitImageV3 in one call. The camera must be
        in trigger mode and streaming.

        Args:
            n_images: Number of images to capture (1 for single trigger).
            timeout_ms: Timeout in ms. 0 = default (exposure * 102% + 4000ms),
                       0xFFFFFFFF = infinite wait.

        Returns:
            Captured image as numpy array.
        """
        assert self.cam, "camera not initialized"
        import ctypes as _ctypes
        import time as _time

        self.cam.Trigger(n_images)

        bufsize, bits, dtype = self.__get_buffer_params()
        buffer = (_ctypes.c_char * bufsize)()
        frame_info = miicam.MiicamFrameInfoV3()

        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.cam.WaitImageV3(timeout_ms if timeout_ms else 1000, buffer, 0, bits, 0, frame_info)
                break
            except miicam.HRESULTException as e:
                hr = getattr(e, "hr", 0)
                if hr in (0x8000000A, 0x8001011f) and attempt < max_retries - 1:
                    _time.sleep(0.05)
                    continue
                raise MIICAMError(f"trigger_sync timeout: hr=0x{hr & 0xFFFFFFFF:08x}") from e

        img_data = np.frombuffer(buffer, dtype=dtype)
        return self.__decode_image(img_data)

    # =========================================================================
    # Still image (Snap) - high-resolution capture
    # =========================================================================

    def snap(self, resolution_index: int = 0xFFFFFFFF) -> None:
        """Trigger a still image capture (Snap).

        The camera temporarily switches to the specified resolution,
        captures one frame, then switches back to the preview resolution.

        Args:
            resolution_index: Resolution index to snap. 0xFFFFFFFF = current
                            preview resolution. Use get_still_resolution_count()
                            to see available options.
        """
        assert self.cam, "camera not initialized"
        self.cam.Snap(resolution_index)

    def pull_still_image(self) -> tuple[np.ndarray, miicam.MiicamFrameInfoV3]:
        """Pull a still image after Snap event.

        Call this after receiving MIICAM_EVENT_STILLIMAGE (in callback mode)
        or after snap() to retrieve the still image data.

        Returns:
            Tuple of (image, frame_info).
        """
        assert self.cam, "camera not initialized"
        import ctypes as _ctypes

        bufsize, bits, dtype = self.__get_buffer_params()
        buffer = (_ctypes.c_char * bufsize)()
        frame_info = miicam.MiicamFrameInfoV3()

        self.cam.PullStillImageV2(buffer, bits, frame_info)

        img_data = np.frombuffer(buffer, dtype=dtype)
        img = self.__decode_image(img_data)
        return img, frame_info

    def get_still_resolution_count(self) -> int:
        """Get number of available still image resolutions.

        Returns:
            Number of still resolutions (0 = not supported).
        """
        assert self.cam, "camera not initialized"
        return self.cam.StillResolutionNumber()

    def get_still_resolution(self, index: int) -> tuple[int, int]:
        """Get dimensions of a still image resolution.

        Args:
            index: Resolution index.

        Returns:
            (width, height) tuple.
        """
        assert self.cam, "camera not initialized"
        return self.cam.get_StillResolution(index)

    # =========================================================================
    # Stream control
    # =========================================================================

    def pause(self, b_pause: bool = True) -> None:
        """Pause or resume the video stream.

        Args:
            b_pause: True to pause, False to resume.
        """
        assert self.cam, "camera not initialized"
        self.cam.Pause(1 if b_pause else 0)

    def _stop_streaming(self) -> None:
        """Stop streaming (internal helper)."""
        if self.cam:
            try:
                self.cam.Stop()
            except Exception:
                pass
            import time as _time
            _time.sleep(0.3)

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def __get_buffer_params(self) -> tuple[int, int, type]:
        """Get buffer size, bits, and numpy dtype based on current format.

        Returns:
            (bufsize, bits, dtype) tuple.
        """
        if getattr(self, "_pixel_format", None) == "YUV422":
            return (self.cam_width * self.cam_height * 2, 8, np.uint8)
        elif self._bit_depth == 8:
            return (self.cam_width * self.cam_height, 8, np.uint8)
        else:
            return (self.cam_width * self.cam_height * 2, 16, np.uint16)

    def __decode_image(self, img_data: np.ndarray) -> np.ndarray:
        """Decode raw image data based on pixel format.

        Args:
            img_data: Raw image buffer as 1D numpy array.

        Returns:
            Decoded 2D image array.
        """
        if getattr(self, "_pixel_format", None) == "YUV422":
            img_yuv = img_data.reshape((self.cam_height, self.cam_width * 2))
            return img_yuv[:, ::2]
        return img_data.reshape((self.cam_height, self.cam_width))

    def get_numpy_image(
        self,
        n_sample: int = 1,
        skip_first: bool = True,
    ) -> np.ndarray:
        """Get camera image data with averaging.

        Args:
            n_sample (int): Number of samples for averaging. Must be > 0.
            skip_first (bool): Whether to skip first sample, default True.

        Returns:
            np.ndarray: Processed averaged image as uint8 (8-bit mode) or uint16 (14-bit mode) array.
        """
        assert n_sample > 0, "Sample count must be > 0"

        # Get first image to initialize array
        first_img = self.__take_one_shot()

        if n_sample == 1:
            return first_img

        numpy_image = np.zeros_like(first_img) if skip_first else first_img.copy()
        _n_sample = n_sample if skip_first else n_sample - 1

        for _ in range(_n_sample):
            numpy_image = numpy_image + self.__take_one_shot()

        avg_img = numpy_image / _n_sample
        if self._bit_depth == 8:
            return avg_img.astype(np.uint8)
        else:
            return avg_img.astype(np.uint16)

    def __update_properties(self):
        assert self.cam, "camera not initialized"
        self.cam_width, self.cam_height = self.cam.get_Size()
        logger.info(
            f"Open cam {self._sn} success. width={self.cam_width}, height={self.cam_height}"
        )
        self.xv, self.yv = self.__get_grid(self.cam_width, self.cam_height)

    @staticmethod
    def __get_grid(width: int, height: int):
        x = np.arange(0, width)
        y = np.arange(0, height)
        xv, yv = np.meshgrid(x, y)
        return xv, yv

    @staticmethod
    def get_cam_list():
        """Get list of available cameras."""
        return miicam.Miicam.EnumV2()




