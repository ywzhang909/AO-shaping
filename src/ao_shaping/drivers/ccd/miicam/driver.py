import ctypes
import threading
import time
from typing import Callable

import numpy as np
from loguru import logger

from ao_shaping.drivers.ccd.miicam._sdk_setup import _setup_miicam_sdk
from ao_shaping.drivers.ccd.base import BaseCamera, CameraError

# Set up MIICAM SDK before importing
_MIICAM_AVAILABLE = _setup_miicam_sdk()

if _MIICAM_AVAILABLE:
    import miicam
else:
    miicam = None


class MIICAMError(CameraError):
    """Exception raised for MIICAM camera errors."""

    pass


# ===== Private Helper Classes =====


class _CallbackSession:
    """Manages callback mode lifecycle for a single capture or continuous streaming.

    Tracks whether callback mode was already active to avoid interfering
    with existing callback streams started via ``start_callback_mode``.
    """

    def __init__(self, owner: "CameraStreamManager") -> None:
        self._owner = owner
        self._was_active: bool = False

    def start(self, callback: Callable | None = None) -> None:
        """Enter callback mode, or reuse an existing active session."""
        self._was_active = self._owner._callback_mode_active
        if not self._was_active:
            self._owner._stop_streaming()
            self._owner._callback_user = callback
            self._owner._callback_buffer_lock = threading.Lock()
            self._owner._callback_buffer = None
            self._owner._callback_frame_info = None
            self._owner._callback_new_frame = threading.Event()
            self._owner.cam.StartPullModeWithCallback(
                self._owner._frame_callback, self._owner
            )
        self._owner._callback_mode_active = True

    def stop(self) -> None:
        """Leave callback mode if this session started it."""
        if not self._was_active:
            self._owner._stop_streaming()
            time.sleep(0.3)
            self._owner.cam.StartPullModeWithCallback(None, None)
            self._owner._callback_mode_active = False
        self._owner._callback_user = None


class _FramePuller:
    """Pulls frames from the camera with retry logic."""

    def __init__(self, owner: "CameraStreamManager") -> None:
        self._owner = owner

    def wait_image(self, timeout_ms: int | None = None) -> np.ndarray:
        """WaitImageV3 with retry on timeout.

        Timeout is adaptive: if not provided, uses ``max(300, exposure_ms * 5)``.
        """
        if timeout_ms is None:
            timeout_ms = max(300, int(self._owner.exposure_time_ms * 5))

        bufsize, bits, dtype = self._owner._get_buffer_params()
        buffer = (ctypes.c_char * bufsize)()
        frame_info = miicam.MiicamFrameInfoV3()

        max_retries = 2
        for attempt in range(max_retries):
            try:
                self._owner.cam.WaitImageV3(timeout_ms, buffer, 0, bits, 0, frame_info)
                break
            except miicam.HRESULTException as e:
                hr = getattr(e, "hr", getattr(e, "winerror", 0))
                if hr in (0x8000000A, 0x8001011F):
                    if attempt < max_retries - 1:
                        time.sleep(0.02)
                        continue
                    raise MIICAMError(
                        f"WaitImageV3 timeout after {max_retries} retries"
                    ) from e
                if attempt < max_retries - 1:
                    time.sleep(0.02)
                    continue
                raise MIICAMError(
                    f"WaitImageV3 failed: hr=0x{hr & 0xFFFFFFFF:08x}"
                ) from e

        img_data = np.frombuffer(buffer, dtype=dtype)
        return self._owner._decode_image(img_data)

    def pull_image_v4(self, buffer, bits, frame_info) -> None:
        """PullImageV4."""
        self._owner.cam.PullImageV4(buffer, 0, bits, 0, frame_info)

    def pull_still_image(self, buffer, bits, frame_info) -> None:
        """PullStillImageV2."""
        self._owner.cam.PullStillImageV2(buffer, bits, frame_info)


# ===== Main Camera Class =====


class CameraStreamManager(BaseCamera):
    """MIICAM camera stream manager.

    Supports two capture modes:
    - **wait** (default): blocking ``WaitImageV3`` pull.
    - **callback**: ``StartPullModeWithCallback`` + software ``Trigger`` +
      ``PullImageV4`` (参考 C++ ``demosofttrigger``).
    """

    def __init__(
        self,
        cam_id: int = 0,
        exposure_time_ms: float = 20.0,
        skip_sampling: bool = False,
        bit_depth: int = 8,
        capture_mode: str = "wait",
    ):
        """Initialize MIICAM camera.

        Args:
            cam_id: Camera index.
            exposure_time_ms: Exposure time in milliseconds.
            skip_sampling: Enable 2x2 binning.
            bit_depth: Output bit depth (8 or 16).
            capture_mode: "wait" uses WaitImageV3 (blocking pull),
                "callback" uses callback-based PullImageV4 (software trigger mode).
        """
        if capture_mode not in ("wait", "callback"):
            raise ValueError(
                f"capture_mode must be 'wait' or 'callback', got '{capture_mode}'"
            )
        super().__init__(cam_id, exposure_time_ms, skip_sampling)
        self._bit_depth = bit_depth
        self._pixel_format = "MONO8" if bit_depth == 8 else "MONO16"
        self._max_bit_depth = 8  # Will be updated from camera
        self._capture_mode = capture_mode
        self._callback_mode_active: bool = False

        # Helper objects (initialized after cam is opened)
        self._callback_session: _CallbackSession | None = None
        self._frame_puller: _FramePuller | None = None

    # =========================================================================
    # Context manager / lifecycle
    # =========================================================================

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

    # =========================================================================
    # Initialization
    # =========================================================================

    def initialize(self) -> None:
        """Initialize the camera device."""
        # Close previously opened camera device (if any)
        self.__exit__(None, None, None)

        # Pre-emptive: try to stop any stale stream from a previous session
        try:
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
                    time.sleep(0.5)
                    try:
                        temp_cam.Close()
                    except Exception:
                        pass
                    time.sleep(1.0)
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

        self._init_hardware()
        self._init_streaming()
        self.__update_properties()

        # Initialize helper objects after camera is ready
        self._callback_session = _CallbackSession(self)
        self._frame_puller = _FramePuller(self)

    def _init_hardware(self) -> None:
        """Set exposure, gain, bit depth, pixel format, and resolution."""
        self._init_exposure()
        self._init_bit_depth()
        self._init_pixel_format()
        self._init_binning()
        self._init_resolution()
        self._detect_raw_format()
        self._init_serial_number()

    def _init_exposure(self) -> None:
        """Disable auto exposure and set exposure time / gain."""
        try:
            self.cam.put_AutoExpoEnable(0)
        except miicam.HRESULTException:
            logger.warning("Could not disable auto exposure during init")

        try:
            self.cam.put_ExpoTime(int(self.exposure_time_ms * 1000))
        except miicam.HRESULTException:
            logger.warning(f"Could not set exposure time to {self.exposure_time_ms}ms")

        try:
            self.cam.put_ExpoAGain(100)
        except miicam.HRESULTException:
            logger.warning("Could not set exposure gain")

    def _init_bit_depth(self) -> None:
        """Query max bit depth and configure output bit depth."""
        self._max_bit_depth = self.cam.MaxBitDepth()

        if self._bit_depth == 8:
            output_bitdepth = 0
        elif self._max_bit_depth > 8:
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

    def _init_pixel_format(self) -> None:
        """Set pixel format based on bit depth."""
        if self._bit_depth == 8:
            self._pixel_format = "MONO8"
            self._set_mono8_format()
            return

        self._pixel_format = "MONO16"
        if not self._try_set_option(miicam.MIICAM_OPTION_RAW, 1):
            if not self._try_set_option(miicam.MIICAM_OPTION_RGB, 4):
                logger.warning(
                    "Could not set high bit depth mode (RAW or 16-bit Grey), "
                    "falling back to 8-bit"
                )
                self._pixel_format = "MONO8"
                self._bit_depth = 8
                self._set_mono8_format()

    def _set_mono8_format(self) -> None:
        """Force MONO8 format and disable RAW mode."""
        self._try_set_option(miicam.MIICAM_OPTION_RGB, 3)
        self._try_set_option(miicam.MIICAM_OPTION_RAW, 0)

    def _try_set_option(self, option: int, value: int) -> bool:
        """Try to set an SDK option, return True if successful."""
        try:
            self.cam.put_Option(option, value)
            return True
        except miicam.HRESULTException:
            return False

    def _init_binning(self) -> None:
        """Enable 2x2 binning if skip_sampling is True."""
        if not self.skip_sampling:
            return
        if not self._try_set_option(miicam.MIICAM_OPTION_BINNING, 0x80 | 2):
            logger.warning(
                "MIICAM_OPTION_BINNING not supported, continuing without binning"
            )

    def _init_resolution(self) -> None:
        """Set camera to maximum resolution and update size properties."""
        max_width, max_height = self.cam.get_Resolution(0)
        self.cam.put_Size(max_width, max_height)
        self.cam_width, self.cam_height = self.cam.get_Size()

    def _detect_raw_format(self) -> None:
        """Detect actual raw format and adjust pixel format if needed."""
        raw_fmt, _ = self.cam.get_RawFormat()
        raw_fmt_str = (
            raw_fmt.decode("ascii", errors="replace")
            if isinstance(raw_fmt, bytes)
            else raw_fmt
        )

        if raw_fmt_str in ("YUYV", "VUYY", "UYVY"):
            self._pixel_format = "YUV422"
            logger.info(f"Camera using YUV422 format (raw: {raw_fmt_str})")
        elif raw_fmt_str in ("YMono",):
            logger.info(
                f"Camera using MONO{self._bit_depth} format (raw: {raw_fmt_str})"
            )
        elif raw_fmt_str in ("BGGR", "RGGB", "GRBG", "GBRG"):
            logger.info(
                f"Camera using Bayer {raw_fmt_str} format, treating as MONO{self._bit_depth}"
            )
        else:
            logger.info(
                f"Camera using format: {raw_fmt_str}, treating as MONO{self._bit_depth}"
            )

    def _init_serial_number(self) -> None:
        """Query camera serial number."""
        try:
            self._sn = self.cam.SerialNumber()
        except Exception:
            self._sn = f"MIICAM_{self.cam_id}"

    def _init_streaming(self) -> None:
        """Start the video stream (pull mode without callback)."""
        # Pre-emptively try to stop any stale stream from a previous session
        try:
            self.cam.Stop()
        except Exception:
            pass
        time.sleep(0.5)

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
                    time.sleep(0.5 + 0.5 * attempt)
                else:
                    raise

    # =========================================================================
    # Exposure control
    # =========================================================================

    def reset_exposure_time(self, time_ms: float) -> float:
        """Set the camera exposure time.

        Args:
            time_ms: New exposure time in milliseconds.
                Valid range: 0.011ms to 10000ms. Values outside this range are clamped.

        Returns:
            Actual exposure time set in milliseconds.
        """
        assert self.cam, "camera not initialized"
        if time_ms < 0.011:
            self.exposure_time_ms = 0.011
            logger.warning("exposure time must >= 0.011ms. clamped to 0.011ms.")
        elif time_ms > 10000:
            self.exposure_time_ms = 10000.0
            logger.warning("exposure time must <= 10000ms. clamped to 10000ms.")
        else:
            self.exposure_time_ms = time_ms
        self.cam.put_ExpoTime(int(self.exposure_time_ms * 1000))
        return self.exposure_time_ms

    def enable_auto_exposure(self, enable: bool = True, mode: int = 1) -> bool:
        """Enable or disable auto exposure.

        Args:
            enable: True to enable, False to disable.
            mode: Auto exposure mode (0=disable, 1=continuous, 2=once).

        Returns:
            True if successful.
        """
        assert self.cam, "camera not initialized"
        mode_value = 1 if enable else 0
        if enable and mode > 0:
            mode_value = mode
        self.cam.put_AutoExpoEnable(mode_value)
        return True

    def set_auto_exposure_target(self, target: int) -> int:
        """Set auto exposure target brightness.

        Args:
            target: Target brightness value. Range: 16-220, default: 120.

        Returns:
            The target value that was set.
        """
        assert self.cam, "camera not initialized"
        target = max(16, min(220, target))
        try:
            self.cam.put_AutoExpoTarget(target)
        except miicam.HRESULTException:
            logger.warning("Auto exposure target not supported")
        return target

    def get_auto_exposure_state(self) -> dict:
        """Get current auto exposure state.

        Returns:
            Dictionary containing enabled, mode, and target.
        """
        assert self.cam, "camera not initialized"
        state = {"enabled": False, "mode": 0, "target": 120}
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
        """Set auto exposure time and gain range.

        Args:
            max_time_ms: Maximum exposure time in ms.
            min_time_ms: Minimum exposure time in ms.
            max_gain: Maximum gain value.
            min_gain: Minimum gain value.

        Returns:
            True if successful, False if not supported.
        """
        assert self.cam, "camera not initialized"
        try:
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

    # =========================================================================
    # Window / ROI
    # =========================================================================

    def reset_window(
        self,
        center: tuple[int, int] | tuple[np.intp, ...] = (0, 0),
        size: tuple[int, int] = (0, 0),
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Reset the camera window size and position.

        Args:
            center: Expected window center position (x, y).
            size: Expected window size (width, height). Use (0, 0) for maximum.

        Returns:
            ((width, height), (center_x, center_y)) actually set.
        """
        assert self.cam, "camera not initialized"
        center = tuple(int(c) for c in center)

        try:
            self.cam.Stop()
        except Exception:
            pass

        if size == (0, 0):
            width, height = self.cam.get_Resolution(0)
        else:
            width, height = int(size[0]), int(size[1])
            x_offset = max(0, center[0] - (width // 2))
            y_offset = max(0, center[1] - (height // 2))

        assert x_offset >= 0 and y_offset >= 0, (
            f"Window center position: {center} must be within image, window size: {size}"
        )

        try:
            self.cam.put_Size(width, height)
        except miicam.HRESULTException:
            width, height = self.cam.get_Resolution(0)
            self.cam.put_Size(width, height)

        self.__update_properties()
        self.cam.StartPullModeWithCallback(None, None)
        return (width, height), (width // 2, height // 2)

    # =========================================================================
    # Image capture
    # =========================================================================

    def __take_one_shot(self) -> np.ndarray:
        """Capture a single camera image using the configured capture mode."""
        if self._capture_mode == "callback":
            return self.__take_one_shot_callback()
        return self.__take_one_shot_wait()

    def __take_one_shot_wait(self) -> np.ndarray:
        """WaitImageV3-based capture (blocking pull)."""
        return self._frame_puller.wait_image()

    def __take_one_shot_callback(self) -> np.ndarray:
        """Callback-based capture using PullImageV4 (software trigger mode)."""
        with self._callback_session as session:
            self.cam.Trigger(1)
            result = self.get_callback_frame(timeout=5.0)
            if result is None:
                raise MIICAMError("Timeout waiting for callback frame after trigger")
            img, _ = result
            return img

    def get_numpy_image(
        self,
        n_sample: int = 1,
        skip_first: bool = True,
    ) -> np.ndarray:
        """Get camera image data with averaging.

        Args:
            n_sample: Number of samples to average. Must be > 0.
            skip_first: Whether to skip first frame (often unstable).

        Returns:
            Processed averaged image.
        """
        assert n_sample > 0, "Sample count must be > 0"

        first_img = self.__take_one_shot()

        if n_sample == 1:
            return first_img

        numpy_image = np.zeros_like(first_img) if skip_first else first_img.copy()
        _n_sample = n_sample if skip_first else n_sample - 1

        for _ in range(_n_sample):
            numpy_image = numpy_image + self.__take_one_shot()

        avg_img = numpy_image / _n_sample
        return avg_img.astype(np.uint8 if self._bit_depth == 8 else np.uint16)

    # =========================================================================
    # Callback mode (continuous streaming)
    # =========================================================================

    def start_callback_mode(
        self,
        callback: Callable | None = None,
    ) -> None:
        """Start pull mode with a callback function for frame notification.

        The callback runs in an internal SDK thread. Keep it fast — do NOT
        perform heavy processing or call back into the SDK from within it.

        Args:
            callback: Function called on each new frame with (image, frame_info).
                      If None, uses the internal buffer callback.
        """
        assert self.cam, "camera not initialized"
        self._callback_session.start(callback)
        logger.info("Callback mode started")

    def stop_callback_mode(self) -> None:
        """Stop callback mode and release callback resources."""
        self._callback_session.stop()
        logger.info("Callback mode stopped")

    def _frame_callback(
        self,
        nEvent: int,
        ctx: object,
        user_callback: Callable[[np.ndarray, miicam.MiicamFrameInfoV3], None]
        | None = None,
    ) -> None:
        """Internal SDK callback: called when a new frame is available."""
        if nEvent == miicam.MIICAM_EVENT_IMAGE:
            try:
                bufsize, bits, dtype = self._get_buffer_params()
                buffer = (ctypes.c_char * bufsize)()
                frame_info = miicam.MiicamFrameInfoV3()
                self._frame_puller.pull_image_v4(buffer, bits, frame_info)
                img_data = np.frombuffer(buffer, dtype=dtype)
                img = self._decode_image(img_data)

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
                bufsize, bits, dtype = self._get_buffer_params()
                buffer = (ctypes.c_char * bufsize)()
                frame_info = miicam.MiicamFrameInfoV3()
                self._frame_puller.pull_still_image(buffer, bits, frame_info)
                img_data = np.frombuffer(buffer, dtype=dtype)
                img = self._decode_image(img_data)

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
            timeout_ms: Timeout in ms. 0 = adaptive (exposure * 5, min 300ms).

        Returns:
            Captured image as numpy array.
        """
        assert self.cam, "camera not initialized"

        self.cam.Trigger(n_images)

        if timeout_ms == 0:
            timeout_ms = max(300, int(self.exposure_time_ms * 5))

        bufsize, bits, dtype = self._get_buffer_params()
        buffer = (ctypes.c_char * bufsize)()
        frame_info = miicam.MiicamFrameInfoV3()

        max_retries = 2
        for attempt in range(max_retries):
            try:
                self.cam.WaitImageV3(
                    timeout_ms, buffer, 0, bits, 0, frame_info
                )
                break
            except miicam.HRESULTException as e:
                hr = getattr(e, "hr", 0)
                if hr in (0x8000000A, 0x8001011F) and attempt < max_retries - 1:
                    time.sleep(0.02)
                    continue
                raise MIICAMError(
                    f"trigger_sync timeout: hr=0x{hr & 0xFFFFFFFF:08x}"
                ) from e

        img_data = np.frombuffer(buffer, dtype=dtype)
        return self._decode_image(img_data)

    # =========================================================================
    # Still image (Snap) - high-resolution capture
    # =========================================================================

    def snap(self, resolution_index: int = 0xFFFFFFFF) -> None:
        """Trigger a still image capture (Snap).

        The camera temporarily switches to the specified resolution,
        captures one frame, then switches back to the preview resolution.

        Args:
            resolution_index: Resolution index to snap. 0xFFFFFFFF = current
                            preview resolution.
        """
        assert self.cam, "camera not initialized"
        self.cam.Snap(resolution_index)

    def pull_still_image(self) -> tuple[np.ndarray, miicam.MiicamFrameInfoV3]:
        """Pull a still image after Snap event.

        Returns:
            Tuple of (image, frame_info).
        """
        assert self.cam, "camera not initialized"

        bufsize, bits, dtype = self._get_buffer_params()
        buffer = (ctypes.c_char * bufsize)()
        frame_info = miicam.MiicamFrameInfoV3()

        self._frame_puller.pull_still_image(buffer, bits, frame_info)

        img_data = np.frombuffer(buffer, dtype=dtype)
        img = self._decode_image(img_data)
        return img, frame_info

    def get_still_resolution_count(self) -> int:
        """Get number of available still image resolutions."""
        assert self.cam, "camera not initialized"
        return self.cam.StillResolutionNumber()

    def get_still_resolution(self, index: int) -> tuple[int, int]:
        """Get dimensions of a still image resolution."""
        assert self.cam, "camera not initialized"
        return self.cam.get_StillResolution(index)

    # =========================================================================
    # Stream control
    # =========================================================================

    def pause(self, b_pause: bool = True) -> None:
        """Pause or resume the video stream."""
        assert self.cam, "camera not initialized"
        self.cam.Pause(1 if b_pause else 0)

    def _stop_streaming(self) -> None:
        """Stop streaming (internal helper)."""
        if self.cam:
            try:
                self.cam.Stop()
            except Exception:
                pass
            time.sleep(0.3)
        self._callback_mode_active = False

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _get_buffer_params(self) -> tuple[int, int, type]:
        """Get buffer size, bits, and numpy dtype based on current format."""
        if getattr(self, "_pixel_format", None) == "YUV422":
            return (self.cam_width * self.cam_height * 2, 8, np.uint8)
        elif self._bit_depth == 8:
            return (self.cam_width * self.cam_height, 8, np.uint8)
        else:
            return (self.cam_width * self.cam_height * 2, 16, np.uint16)

    def _decode_image(self, img_data: np.ndarray) -> np.ndarray:
        """Decode raw image data based on pixel format."""
        if getattr(self, "_pixel_format", None) == "YUV422":
            img_yuv = img_data.reshape((self.cam_height, self.cam_width * 2))
            return img_yuv[:, ::2]
        return img_data.reshape((self.cam_height, self.cam_width))

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
