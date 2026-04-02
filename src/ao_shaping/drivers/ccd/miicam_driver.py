import os
import sys
import ctypes
import logging

_logger = logging.getLogger(__name__)


def _find_miicam_sdk_path() -> str | None:
    """Find the MIICAM SDK path by checking multiple possible locations.

    Checks in order:
    1. Environment variable MIICAM_SDK_PATH (user-configurable)
    2. Bundled in project (src/ao_shaping/drivers/ccd/_miicam_sdk)
    3. External libs directory (libs/miicamsdk.20240728/python)

    Returns:
        str | None: Path to SDK if found, None otherwise.
    """
    # Option 0: Environment variable (highest priority, user-configurable)
    env_sdk_path = os.environ.get("MIICAM_SDK_PATH")
    if env_sdk_path and os.path.isdir(env_sdk_path):
        return env_sdk_path

    _current_file = os.path.abspath(__file__)
    # _current_file -> src/ao_shaping/drivers/ccd/miicam_driver.py
    # Go up 5 levels: ccd -> drivers -> ao_shaping -> src -> project_root
    _project_root = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(_current_file)))
        )
    )

    _MII_SDK_PATHS = [
        # Option 1: Bundled in project (for development)
        os.path.join(os.path.dirname(__file__), "_miicam_sdk"),
        # Option 2: External libs directory
        os.path.join(_project_root, "libs", "miicamsdk.20240728", "python"),
    ]

    for path in _MII_SDK_PATHS:
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "miicam.py")):
            return path
    return None


def _setup_miicam_sdk() -> bool:
    """Set up the MIICAM SDK by adding its path to sys.path and DLL search path.

    Returns:
        bool: True if SDK was found and set up successfully, False otherwise.
    """
    sdk_path = _find_miicam_sdk_path()
    if sdk_path is None:
        _logger.warning(
            "MIICAM SDK not found. Tried: "
            "1) MIICAM_SDK_PATH env var, "
            "2) bundled '_miicam_sdk', "
            "3) 'libs/miicamsdk.20240728/python'. "
            "Camera functionality will not be available. "
            "Set MIICAM_SDK_PATH to the directory containing miicam.py and miicam.dll."
        )
        return False

    # Add to sys.path so Python can import the miicam module
    if sdk_path not in sys.path:
        sys.path.insert(0, sdk_path)

    # On Windows, add the SDK directory to the DLL search path
    # so miicam.dll and its dependencies can be found
    if sys.platform == "win32":
        try:
            os.add_dll_directory(sdk_path)
        except (OSError, AttributeError):
            # os.add_dll_directory requires Python 3.8+
            # Fall back to adding to PATH
            os.environ["PATH"] = sdk_path + os.pathsep + os.environ.get("PATH", "")

    _logger.debug(f"MIICAM SDK set up successfully from: {sdk_path}")
    return True


# Set up MIICAM SDK before importing
_MIICAM_AVAILABLE = _setup_miicam_sdk()

if _MIICAM_AVAILABLE:
    import numpy as np

    import miicam

    from ao_shaping.utils.file import logger
    from ao_shaping.drivers.ccd.base import BaseCamera, CameraError
else:
    # Provide stubs so the module can still be imported for testing
    np = None
    miicam = None
    logger = logging.getLogger(__name__)

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
    ):
        super().__init__(cam_id, exposure_time_ms, skip_sampling)
        self._pixel_format = "MONO8"  # Default format

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
            for _ in range(3):
                try:
                    self.cam.Stop()
                    break
                except miicam.HRESULTException:
                    time.sleep(0.05)
                except Exception:
                    break

            # Flush any pending frames
            try:
                self.cam.put_Option(miicam.MIICAM_OPTION_FLUSH, 3)
            except Exception:
                pass

            # Small delay to let callback thread finish
            time.sleep(0.1)

            try:
                self.cam.Close()
            except Exception:
                pass

            self.cam_width = 0
            self.cam_height = 0
            self.cam = None
            self._frame_buffer = None

            # Give the camera hardware a moment to settle before re-opening
            time.sleep(0.2)

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

        # Try to set pixel format to 8-bit grey (MONO8 equivalent)
        # MIICAM_OPTION_RGB = 3 means 8-bit grey for monochrome camera
        # Note: Not all cameras support this option, wrap in try-except
        self._pixel_format = "MONO8"  # Default assumption
        try:
            self.cam.put_Option(miicam.MIICAM_OPTION_RGB, 3)
        except miicam.HRESULTException:
            # Option not supported, try YUV format
            try:
                self.cam.put_Option(miicam.MIICAM_OPTION_RGB, 8)  # YUV422
                self._pixel_format = "YUV422"
            except miicam.HRESULTException:
                logger.warning("Could not set pixel format, using default")

        # Get actual raw format from camera
        raw_fmt, _ = self.cam.get_RawFormat()
        print(f"Raw format: {raw_fmt}")
        if raw_fmt == b"YUYV" or raw_fmt == "YUYV":
            self._pixel_format = "YUV422"
            logger.info("Camera using YUV422 format")
        elif raw_fmt == b"YMono" or raw_fmt == "YMono":
            self._pixel_format = "MONO8"
            logger.info("Camera using MONO8 format")

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
        if raw_fmt == b"YUYV" or raw_fmt == "YUYV":
            self._pixel_format = "YUV422"
            logger.info("Camera using YUV422 format")
        else:
            self._pixel_format = "MONO8"
            logger.info(f"Camera using format: {raw_fmt}")

        # Get serial number
        try:
            self._sn = self.cam.SerialNumber()
        except Exception:
            self._sn = f"MIICAM_{self.cam_id}"

        # Start streaming (pull mode with callback) with retry
        # Use a minimal callback that just stores the image
        self._frame_buffer = None
        self._frame_buffer_lock = False  # Lock to prevent reading while updating

        def frame_callback(nEvent, ctx):
            if nEvent == miicam.MIICAM_EVENT_IMAGE:
                try:
                    # Wait for buffer to be free
                    while ctx._frame_buffer_lock:
                        pass
                    ctx._frame_buffer_lock = True
                    try:
                        bufsize = ctx.cam_width * ctx.cam_height
                        buffer = (ctypes.c_char * bufsize)()
                        ctx.cam.PullImageV4(buffer, 0, 8, 0, None)
                        ctx._frame_buffer = buffer
                    finally:
                        ctx._frame_buffer_lock = False
                except Exception:
                    pass

        # Retry StartPullModeWithCallback - the camera may need time to
        # fully stop the previous streaming session (from this or a prior process)
        import time as _time

        # Pre-emptively try to stop any stale stream from a previous session
        try:
            self.cam.Stop()
        except Exception:
            pass
        _time.sleep(0.3)

        max_retries = 7
        for attempt in range(max_retries):
            try:
                self.cam.StartPullModeWithCallback(frame_callback, self)
                break
            except miicam.HRESULTException:
                if attempt < max_retries - 1:
                    # Try to stop again and wait with increasing delay
                    try:
                        self.cam.Stop()
                    except Exception:
                        pass
                    _time.sleep(0.5 + 0.3 * attempt)
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

        # Restart streaming
        self._frame_buffer = None
        self._frame_buffer_lock = False

        def frame_callback(nEvent, ctx):
            if nEvent == miicam.MIICAM_EVENT_IMAGE:
                try:
                    while ctx._frame_buffer_lock:
                        pass
                    ctx._frame_buffer_lock = True
                    try:
                        bufsize = ctx.cam_width * ctx.cam_height
                        buffer = (ctypes.c_char * bufsize)()
                        ctx.cam.PullImageV4(buffer, 0, 8, 0, None)
                        ctx._frame_buffer = buffer
                    finally:
                        ctx._frame_buffer_lock = False
                except Exception:
                    pass

        self.cam.StartPullModeWithCallback(frame_callback, self)

        # Return new window size and center
        return (width, height), (width // 2, height // 2)

    def __take_one_shot(self) -> np.ndarray:
        """
        Capture a single camera image.

        Returns:
            np.ndarray: Captured image data as uint8 array.
        """
        import time

        # Try to get image from callback buffer first
        max_wait = 1.0  # Wait up to 1 second for a new frame
        start_time = time.time()

        while time.time() - start_time < max_wait:
            # Check if we have a valid buffer
            if self._frame_buffer is not None and not self._frame_buffer_lock:
                try:
                    # Convert buffer to numpy array
                    img_data = np.frombuffer(self._frame_buffer, dtype=np.uint8)

                    # Handle YUV422 format - extract Y channel (luminance)
                    if getattr(self, "_pixel_format", None) == "YUV422":
                        # YUV422: 2 bytes per pixel (Y0 U0 Y1 V0 ...)
                        img_yuv = img_data.reshape(
                            (self.cam_height, self.cam_width * 2)
                        )
                        img = img_yuv[
                            :, ::2
                        ]  # Take every other column (Y channel only)
                    else:
                        # MONO8 or other formats
                        img = img_data.reshape((self.cam_height, self.cam_width))

                    return img
                except Exception:
                    pass

            time.sleep(0.01)  # Wait a bit before retrying

        # Fallback: If callback buffer not available, use direct pull
        import ctypes

        # Calculate buffer size based on pixel format
        if getattr(self, "_pixel_format", None) == "YUV422":
            # YUV422: 2 bytes per pixel (Y0 U0 Y1 V0 ...)
            bufsize = self.cam_width * self.cam_height * 2
        else:
            # MONO8 or other 8-bit formats
            bufsize = self.cam_width * self.cam_height

        # Create a ctypes buffer
        buffer = (ctypes.c_char * bufsize)()

        # Give the camera a moment to capture a frame
        time.sleep(0.05)

        # Pull image with retry logic
        # bits=8 for 8-bit grey
        max_retries = 5
        retry_count = 0
        while retry_count < max_retries:
            try:
                self.cam.PullImageV4(buffer, 0, 8, 0, None)
                break  # Success
            except miicam.HRESULTException:
                retry_count += 1
                if retry_count >= max_retries:
                    raise  # Re-raise after max retries
                time.sleep(0.05)  # Wait before retry

        # Convert to numpy array
        img_data = np.frombuffer(buffer, dtype=np.uint8)

        # Handle YUV422 format - extract Y channel (luminance)
        if getattr(self, "_pixel_format", None) == "YUV422":
            # YUV422 stores Y0 U0 Y1 V0, so Y is at even indices (0, 2, 4, ...)
            # Reshape and take every other column
            img_yuv = img_data.reshape((self.cam_height, self.cam_width * 2))
            img = img_yuv[:, ::2]  # Take every other column (Y channel only)
        else:
            # MONO8 or other formats
            img = img_data.reshape((self.cam_height, self.cam_width))

        return img

    def get_numpy_image(self, n_sample: int = 1, skip_first: bool = True) -> np.ndarray:
        """
        Get camera image data with averaging.

        Args:
            n_sample (int): Number of samples for averaging. Must be > 0.
            skip_first (bool): Whether to skip first sample, default True.

        Returns:
            np.ndarray: Processed averaged image as uint8 array.
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
        return avg_img.astype(np.uint8)

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
