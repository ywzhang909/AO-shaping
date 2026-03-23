"""FFmpeg/OpenCV camera driver for video streams and virtual cameras.

This driver provides camera interface compatibility using OpenCV's FFmpeg backend.
It supports:
- Video files (local .mp4, .avi, etc.)
- RTSP/RTMP/HTTP streams
- Virtual camera devices (e.g., OBS virtual camera)
- Webcam devices (via DirectShow on Windows)
- Folder of timestamped image files (virtual CCD)

Note: Exposure time control is simulated - it controls internal frame processing
delay, not actual hardware exposure.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np

from ao_shaping.drivers.ccd.base import BaseCamera, CameraError
from ao_shaping.utils.file import logger
from ao_shaping.utils.timestamp import TimestampParser


class FFmpegCameraError(CameraError):
    """Exception raised for FFmpeg camera errors."""

    pass


class FFmpegCamera(BaseCamera):
    """FFmpeg/OpenCV-based camera driver.

    Provides camera interface compatibility using FFmpeg through OpenCV.
    Supports video files, network streams, and virtual camera devices.

    Args:
        cam_id: Camera identifier - can be:
            - Integer: Device index (0 for default webcam, 1 for second camera, etc.)
            - String: File path or stream URL (e.g., "video.mp4", "rtsp://...")
        exposure_time_ms: Simulated exposure time in milliseconds.
            Note: This is a simulated parameter - actual exposure depends on
            the video source/stream and cannot be controlled.
        skip_sampling: If True, skip frames to reduce capture rate.
            If False, wait for each frame (slower but no frame loss).

    Example:
        # Open a video file
        with FFmpegCamera(cam_id="test_video.mp4") as cam:
            img = cam.get_numpy_image()

        # Open a network stream
        with FFmpegCamera(cam_id="rtsp://192.168.1.100:8554 live stream") as cam:
            img = cam.get_numpy_image()

        # Open virtual camera (Windows)
        with FFmpegCamera(cam_id=1) as cam:  # OBS Virtual Camera usually appears as device 1
            img = cam.get_numpy_image()
    """

    def __init__(
        self,
        cam_id: int | str = 0,
        exposure_time_ms: int = 20,
        skip_sampling: bool = False,
    ):
        """Initialize FFmpeg camera configuration.

        Args:
            cam_id: Camera device index, file path, or stream URL.
            exposure_time_ms: Initial exposure time in milliseconds (simulated).
            skip_sampling: Whether to skip frames for faster capture.
        """
        super().__init__(cam_id, exposure_time_ms, skip_sampling)
        self._cap = None
        self._stream_url = None
        self._is_file = False
        self._fps = 30.0
        self._frame_count = 0
        self._total_frames = 0

    def __enter__(self) -> "FFmpegCamera":
        """Context manager entry - initialize camera."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Context manager exit - cleanup camera resources."""
        self.close()

    def initialize(self) -> None:
        """Initialize the FFmpeg camera/video stream.

        This method:
        1. Closes any previously opened camera.
        2. Opens the specified video source by cam_id.
        3. Retrieves video properties (resolution, fps).
        4. Determines if source is a file, stream, or device.

        Raises:
            ConnectionAbortedError: If camera/video source cannot be opened.
            FFmpegCameraError: If video properties cannot be read.
        """
        # Close previously opened camera
        self.close()

        import cv2

        # Handle different source types
        if isinstance(self.cam_id, int):
            # Device index (webcam, virtual camera)
            self._cap = cv2.VideoCapture(self.cam_id, cv2.CAP_DSHOW)
            self._stream_url = f"device_{self.cam_id}"
            self._is_file = False
        elif isinstance(self.cam_id, str):
            # Check if it's a URL or file path
            if self.cam_id.startswith(("rtsp://", "rtmp://", "http://", "https://")):
                # Network stream
                self._cap = cv2.VideoCapture(self.cam_id)
                self._stream_url = self.cam_id
                self._is_file = False
            else:
                # Local video file
                self._cap = cv2.VideoCapture(self.cam_id)
                self._stream_url = self.cam_id
                self._is_file = True
        else:
            raise FFmpegCameraError(
                f"Invalid cam_id type: {type(self.cam_id)}. "
                "Expected int (device index) or str (file path/URL)."
            )

        # Check if opened successfully
        if not self._cap or not self._cap.isOpened():
            error_info = f"Failed to open video source: {self.cam_id}"
            logger.error(error_info)
            raise ConnectionAbortedError(error_info)

        # Get video properties
        self.cam_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cam_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)

        # Get total frame count (only works for video files, not streams)
        if self._is_file:
            self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            self._total_frames = 0

        # Generate serial number from source
        self._sn = f"FFmpeg_{self.cam_id}"

        # Create coordinate grids
        self.xv, self.yv = self._get_grid(self.cam_width, self.cam_height)

        logger.info(
            f"Open FFmpeg camera {self.cam_id} success. "
            f"width={self.cam_width}, height={self.cam_height}, fps={self._fps}"
        )

    def open(self) -> None:
        """Open the camera device (alias for initialize)."""
        self.initialize()

    def close(self) -> None:
        """Close the camera device and release resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.cam_width = 0
        self.cam_height = 0
        self._frame_count = 0
        logger.info(f"Closed FFmpeg camera: {self.cam_id}")

    def reset_exposure_time(self, time_ms: int) -> int:
        """Set the simulated exposure time.

        Note: This is a SIMULATED parameter - actual exposure depends on
        the video source/stream and cannot be controlled via FFmpeg.
        This setting only adds a delay between frame captures.

        Args:
            time_ms: New exposure time in milliseconds.

        Returns:
            int: The exposure time that was set.

        Raises:
            AssertionError: If camera is not initialized.
        """
        assert self._cap is not None and self._cap.isOpened(), "camera not initialized"
        self.exposure_time_ms = max(1, int(time_ms))
        logger.warning(
            f"[FFmpegCamera] Exposure time is SIMULATED only - adds {self.exposure_time_ms}ms "
            f"delay between captures. Actual exposure depends on video source, not controllable."
        )
        return self.exposure_time_ms

    def reset_window(
        self,
        center: Tuple[int, int] | Tuple[np.intp, ...],
        size: Tuple[int, int],
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Reset the video ROI (not supported for FFmpeg).

        FFmpeg does not support region of interest. Returns current dimensions.

        Args:
            center: Ignored (not supported).
            size: Ignored (not supported).

        Returns:
            Tuple of ((width, height), (center_x, center_y)).

        Raises:
            AssertionError: If camera is not initialized.
        """
        assert self._cap is not None and self._cap.isOpened(), "camera not initialized"
        # FFmpeg doesn't support ROI - return current dimensions
        center_x = self.cam_width // 2
        center_y = self.cam_height // 2
        return (self.cam_width, self.cam_height), (center_x, center_y)

    def get_numpy_image(
        self,
        n_sample: int = 1,
        skip_first: bool = True,
    ) -> np.ndarray:
        """Capture image(s) from FFmpeg video source with optional averaging.

        Args:
            n_sample: Number of samples to average. Must be > 0.
            skip_first: Whether to skip first frame (often unstable for streams).

        Returns:
            np.ndarray: Captured image as uint8 array.

        Raises:
            AssertionError: If n_sample is not positive or camera not initialized.
            FFmpegCameraError: If frame capture fails.
        """
        assert self._cap is not None and self._cap.isOpened(), "camera not initialized"
        assert n_sample > 0, "Sample count must be > 0"

        import cv2

        # Skip first frame if requested
        if skip_first:
            ret, frame = self._cap.read()
            if not ret:
                raise FFmpegCameraError(f"Failed to capture frame from {self.cam_id}")
            self._frame_count += 1

        # Capture frames
        frames = []
        for _ in range(n_sample):
            ret, frame = self._cap.read()
            if not ret:
                # For video files, loop back to beginning
                if self._is_file and self._total_frames > 0:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._frame_count = 0
                    ret, frame = self._cap.read()
                    if not ret:
                        raise FFmpegCameraError(
                            f"Failed to capture frame from {self.cam_id}"
                        )
                else:
                    raise FFmpegCameraError(
                        f"Failed to capture frame from {self.cam_id}. "
                        "Stream may have ended."
                    )

            self._frame_count += 1

            # Convert BGR to grayscale if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frames.append(frame)

            # Simulated exposure time delay
            if self.exposure_time_ms > 0:
                time.sleep(self.exposure_time_ms / 1000.0)

        # Compute average
        avg_img = np.mean(frames, axis=0)

        # Skip sampling: drop frames to reduce capture rate
        if self.skip_sampling and self._is_file:
            # Skip half the frames when skip_sampling is True
            skip_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2
            if skip_count > 0:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, skip_count)

        return avg_img.astype(np.uint8)

    def enable_auto_exposure(self, enable: bool = True, mode: int = 1) -> bool:
        """Enable or disable auto exposure.

        Note: Not supported for FFmpeg sources. Returns False.

        Args:
            enable: Ignored.
            mode: Ignored.

        Returns:
            bool: Always returns False (not supported).
        """
        logger.warning("Auto exposure not supported for FFmpeg sources")
        return False

    def set_auto_exposure_target(self, target: int) -> int:
        """Set auto exposure target brightness.

        Note: Not supported for FFmpeg sources.

        Args:
            target: Ignored.

        Returns:
            int: Returns 0 (not supported).

        Raises:
            NotImplementedError: Always raises (not supported).
        """
        raise NotImplementedError("Auto exposure not supported for FFmpeg sources")

    def get_auto_exposure_state(self) -> dict:
        """Get current auto exposure state.

        Returns:
            dict: Always indicates disabled state.
        """
        return {
            "enabled": False,
            "mode": 0,
            "target": 0,
        }

    def set_auto_exposure_range(
        self,
        max_time_ms: int = 350,
        min_time_ms: int = 0,
        max_gain: int = 300,
        min_gain: int = 100,
    ) -> bool:
        """Set auto exposure time and gain range.

        Note: Not supported for FFmpeg sources.

        Args:
            max_time_ms: Ignored.
            min_time_ms: Ignored.
            max_gain: Ignored.
            min_gain: Ignored.

        Returns:
            bool: Always returns False (not supported).
        """
        logger.warning("Auto exposure range not supported for FFmpeg sources")
        return False

    @staticmethod
    def get_cam_list() -> list:
        """Get list of available video capture devices.

        Note: This is a best-effort list. On Windows, this may only detect
        DirectShow devices. Video files and streams are not enumerated.

        Returns:
            List of available device indices (0-9) that can be tried.
        """
        import cv2

        devices = []

        # Try to detect DirectShow devices on Windows
        if hasattr(cv2, "CAP_DSHOW"):
            for i in range(10):
                try:
                    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                    if cap is not None and cap.isOpened():
                        devices.append(i)
                        cap.release()
                except Exception:
                    pass
        else:
            # Try default backend
            for i in range(5):
                try:
                    cap = cv2.VideoCapture(i)
                    if cap is not None and cap.isOpened():
                        devices.append(i)
                        cap.release()
                except Exception:
                    pass

        return devices


class ImageFolderCamera(BaseCamera):
    """Camera driver that reads images from a folder with timestamp-named files.

    This driver mimics a CCD camera by reading sequentially from a folder containing
    image files named with timestamps (e.g., 1700000000000.png, 1700000000050.png).

    It is useful for:
    - Playing back recorded image sequences
    - Testing optimization pipelines with saved data
    - Virtual CCD simulation from recorded data

    Args:
        cam_id: Path to the folder containing image files.
        exposure_time_ms: Delay between reading frames (simulated exposure).
            Note: This is a simulated parameter - actual capture timing depends
            on the file timestamps, not controllable.
        skip_sampling: If True, skip every other frame when reading.
            If False, read every frame in sequence.

    Example:
        # Open a folder of timestamped images
        with ImageFolderCamera(cam_id="/path/to/images") as cam:
            img = cam.get_numpy_image(n_sample=5)

        # Loop through images (for video-like playback)
        with ImageFolderCamera(cam_id="recordings/20240101") as cam:
            for _ in range(100):
                img = cam.get_numpy_image()
                # Process img...
    """

    # Supported image extensions
    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"}

    def __init__(
        self,
        cam_id: str = "",
        exposure_time_ms: int = 20,
        skip_sampling: bool = False,
    ):
        """Initialize image folder camera.

        Args:
            cam_id: Path to the folder containing image files.
            exposure_time_ms: Delay between readings in milliseconds.
            skip_sampling: Whether to skip every other frame.
        """
        super().__init__(cam_id, exposure_time_ms, skip_sampling)
        self._folder_path: Path | None = None
        self._image_files: list[Path] = []
        self._current_index: int = 0
        self._last_image: np.ndarray | None = None

    def __enter__(self) -> "ImageFolderCamera":
        """Context manager entry - initialize camera."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Context manager exit - cleanup resources."""
        self.close()

    def initialize(self) -> None:
        """Initialize by scanning folder for image files.

        This method:
        1. Closes any previously opened folder.
        2. Validates the folder path exists.
        3. Scans for supported image files.
        4. Sorts files by filename (typically timestamp).
        5. Reads first image to get dimensions.

        Raises:
            ConnectionAbortedError: If folder not found or no images found.
            FFmpegCameraError: If image dimensions cannot be read.
        """
        # Close previously opened folder
        self.close()

        import cv2

        # Convert cam_id to Path
        self._folder_path = Path(self.cam_id)

        # Validate folder exists
        if not self._folder_path.exists():
            raise ConnectionAbortedError(f"Image folder not found: {self.cam_id}")

        if not self._folder_path.is_dir():
            raise ConnectionAbortedError(f"Path is not a directory: {self.cam_id}")

        # Scan for image files
        self._image_files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            self._image_files.extend(self._folder_path.glob(f"*{ext}"))
            self._image_files.extend(self._folder_path.glob(f"*{ext.upper()}"))

        if not self._image_files:
            raise ConnectionAbortedError(
                f"No image files found in folder: {self.cam_id}. "
                f"Supported formats: {self.SUPPORTED_EXTENSIONS}"
            )

        # Sort by timestamp using TimestampParser
        self._image_files = TimestampParser().sort_files(self._image_files)
        self._current_index = 0

        # Read first image to get dimensions
        first_img = self._read_image(self._image_files[0])
        if first_img is None:
            raise FFmpegCameraError(
                f"Failed to read first image: {self._image_files[0]}"
            )

        self.cam_width = first_img.shape[1]
        self.cam_height = first_img.shape[0]
        self._last_image = first_img

        # Generate serial number from folder
        self._sn = f"ImageFolder_{self._folder_path.name}"

        # Create coordinate grids
        self.xv, self.yv = self._get_grid(self.cam_width, self.cam_height)

        logger.info(
            f"Open ImageFolder camera {self.cam_id} success. "
            f"Found {len(self._image_files)} images, "
            f"width={self.cam_width}, height={self.cam_height}"
        )

    def _read_image(self, image_path: Path) -> np.ndarray | None:
        """Read a single image file.

        Args:
            image_path: Path to the image file.

        Returns:
            np.ndarray: Grayscale image, or None if read fails.
        """
        import cv2

        try:
            # Handle .npy files specially
            if image_path.suffix.lower() == ".npy":
                img = np.load(image_path)
            else:
                # Read with OpenCV
                img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

            if img is None:
                logger.warning(f"Failed to read image: {image_path}")
                return None

            return img
        except Exception as e:
            logger.warning(f"Error reading {image_path}: {e}")
            return None

    def open(self) -> None:
        """Open the image folder (alias for initialize)."""
        self.initialize()

    def close(self) -> None:
        """Close the camera and release resources."""
        self._image_files = []
        self._current_index = 0
        self._last_image = None
        self.cam_width = 0
        self.cam_height = 0
        logger.info(f"Closed ImageFolder camera: {self.cam_id}")

    def reset_exposure_time(self, time_ms: int) -> int:
        """Set the simulated exposure time (delay between reads).

        Note: This is a SIMULATED parameter - adds delay between frame reads.
        Actual timing depends on file timestamps, not controllable.

        Args:
            time_ms: New delay time in milliseconds.

        Returns:
            int: The time that was set.

        Raises:
            AssertionError: If camera is not initialized.
        """
        assert self._folder_path is not None, "camera not initialized"
        self.exposure_time_ms = max(0, int(time_ms))
        logger.warning(
            f"[ImageFolderCamera] Delay is SIMULATED only - adds {self.exposure_time_ms}ms "
            f"delay between reads. Actual timing depends on file timestamps."
        )
        return self.exposure_time_ms

    def reset_window(
        self,
        center: Tuple[int, int] | Tuple[np.intp, ...],
        size: Tuple[int, int],
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Reset the image ROI (not supported).

        Image folder camera does not support ROI. Returns current dimensions.

        Args:
            center: Ignored (not supported).
            size: Ignored (not supported).

        Returns:
            Tuple of ((width, height), (center_x, center_y)).

        Raises:
            AssertionError: If camera is not initialized.
        """
        assert self._folder_path is not None, "camera not initialized"
        center_x = self.cam_width // 2
        center_y = self.cam_height // 2
        return (self.cam_width, self.cam_height), (center_x, center_y)

    def get_numpy_image(
        self,
        n_sample: int = 1,
        skip_first: bool = True,
    ) -> np.ndarray:
        """Read image(s) from folder with optional averaging.

        Args:
            n_sample: Number of samples to average. Must be > 0.
            skip_first: Whether to skip first image (often unstable).

        Returns:
            np.ndarray: Captured image as uint8 array.

        Raises:
            AssertionError: If n_sample is not positive or camera not initialized.
            FFmpegCameraError: If image read fails.
        """
        assert self._folder_path is not None, "camera not initialized"
        assert n_sample > 0, "Sample count must be > 0"

        import cv2

        # Skip first image if requested
        if skip_first:
            self._advance_index()
            if self._current_index >= len(self._image_files):
                # Loop back to beginning
                self._current_index = 0

        # Read requested number of images
        frames = []
        for _ in range(n_sample):
            if self._current_index >= len(self._image_files):
                # Loop back to beginning for continuous playback
                self._current_index = 0

            img = self._read_image(self._image_files[self._current_index])
            if img is None:
                raise FFmpegCameraError(
                    f"Failed to read image: {self._image_files[self._current_index]}"
                )

            frames.append(img)
            self._last_image = img
            self._advance_index()

            # Simulated exposure time delay
            if self.exposure_time_ms > 0:
                time.sleep(self.exposure_time_ms / 1000.0)

        # Compute average
        avg_img = np.mean(frames, axis=0)
        return avg_img.astype(np.uint8)

    def _advance_index(self) -> None:
        """Advance the current index, handling skip_sampling."""
        if self.skip_sampling:
            # Skip every other image
            self._current_index += 2
        else:
            self._current_index += 1

    def enable_auto_exposure(self, enable: bool = True, mode: int = 1) -> bool:
        """Enable or disable auto exposure.

        Note: Not supported for image folder. Returns False.

        Args:
            enable: Ignored.
            mode: Ignored.

        Returns:
            bool: Always returns False (not supported).
        """
        logger.warning("Auto exposure not supported for ImageFolder sources")
        return False

    def set_auto_exposure_target(self, target: int) -> int:
        """Set auto exposure target brightness.

        Note: Not supported for image folder.

        Args:
            target: Ignored.

        Returns:
            int: Returns 0 (not supported).

        Raises:
            NotImplementedError: Always raises.
        """
        raise NotImplementedError("Auto exposure not supported for ImageFolder sources")

    def get_auto_exposure_state(self) -> dict:
        """Get current auto exposure state.

        Returns:
            dict: Always indicates disabled state.
        """
        return {
            "enabled": False,
            "mode": 0,
            "target": 0,
        }

    def set_auto_exposure_range(
        self,
        max_time_ms: int = 350,
        min_time_ms: int = 0,
        max_gain: int = 300,
        min_gain: int = 100,
    ) -> bool:
        """Set auto exposure time and gain range.

        Note: Not supported for image folder.

        Args:
            max_time_ms: Ignored.
            min_time_ms: Ignored.
            max_gain: Ignored.
            min_gain: Ignored.

        Returns:
            bool: Always returns False (not supported).
        """
        logger.warning("Auto exposure range not supported for ImageFolder sources")
        return False

    @staticmethod
    def get_cam_list() -> list:
        """Get list of available image folders (not applicable).

        Returns:
            Empty list - image folders are specified by path, not enumerated.
        """
        logger.warning(
            "ImageFolderCamera.get_cam_list() not applicable - specify folder path directly"
        )
        return []
