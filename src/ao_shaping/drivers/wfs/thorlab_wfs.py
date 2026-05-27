# refreces docs : https://github.com/nvladimus/WFS/blob/master/python/Thorlabs-WFS-read-average-wavefront.ipynb

from __future__ import annotations

import ctypes
import os
import shutil
from copy import deepcopy
from ctypes import (
    byref,
    c_bool,
    c_double,
    c_float,
    c_int32,
    c_uint8,
    c_ulong,
    create_string_buffer,
)
from datetime import datetime
from enum import IntEnum
from functools import wraps
from pathlib import Path

import numpy as np
from loguru import logger

from ao_shaping.drivers.device_base import Device, DeviceState, DeviceType
from ao_shaping.drivers.wfs._thorlab_wfs import (
    MAX_SPOTS,
    VI_NULL,
    ViInt32,
    ViStatus,
    load_dll,
    np2c,
)
from ao_shaping.utils.file import DeviceConfigManager

WFS_DEBUG_MODE = os.environ.get("WFS_DEBUG", "0") == "1"

EXP_TIME_LOW = 0.002
EXP_TIME_HIGH = 86
MAX_AUTOEXPOSE_ATTEMPTS = 10
MAX_MLA_INDICES = 16


def require_take_image(func):
    """Decorator ensuring take_image() is called before the wrapped method.

    If no image has been captured yet, automatically calls take_image() first.

    **File transformations**: None. Ensures the DLL has a current spotfield
    image and centroid data before functions that depend on it.

    Note:
        After auto-calling take_image(), sets ``_image_captured = False`` so
        the function itself can set it to ``True`` after its own image capture.
    """
    @wraps(func)
    def wrapper(self: ThorlabWFS, *args, **kwargs):
        if not self._image_captured:
            logger.debug(
                f"{func.__name__} requires take_image() to be called first. "
                f"Automatically calling take_image() now."
            )
            self.take_image()
            self._image_captured = False
        return func(self, *args, **kwargs)
    return wrapper


# define  CAM_RES_1280                  (0) // 1280x1024
# define  CAM_RES_1024                  (1) // 1024x1024
# define  CAM_RES_768                   (2) // 768x768
# define  CAM_RES_512                   (3) // 512x512
# define  CAM_RES_320                   (4) // 320x320 smallest!
class MlaRes(IntEnum):
    Res1280 = 0
    Res1024 = 1
    Res768 = 2
    Res512 = 3
    Res320 = 4

    @classmethod
    def from_str(cls, value: str | int | MlaRes) -> MlaRes:
        """Convert a string or integer to MlaRes enum member.

        Args:
            value: Resolution string ('512', '768', '1024', '1280', '320'),
                   integer (512, 768, 1024, 1280, 320), or MlaRes instance

        Returns:
            MlaRes enum member

        Raises:
            ValueError: If value is not a supported resolution
        """
        # If already a MlaRes instance, return as-is
        if isinstance(value, cls):
            return value

        if isinstance(value, int):
            mapping = {
                1280: cls.Res1280,
                1024: cls.Res1024,
                768: cls.Res768,
                512: cls.Res512,
                320: cls.Res320,
            }
            if value in mapping:
                return mapping[value]
            raise ValueError(f"Invalid resolution {value}. Must be one of: 320, 512, 768, 1024, 1280")

        if not isinstance(value, str):
            raise ValueError(f"Value must be str, int, or MlaRes, got {type(value).__name__}")

        mapping = {
            "1280": cls.Res1280,
            "1024": cls.Res1024,
            "768": cls.Res768,
            "512": cls.Res512,
            "320": cls.Res320,
        }
        if value in mapping:
            return mapping[value]
        raise ValueError(f"Invalid resolution '{value}'. Must be one of: '320', '512', '768', '1024', '1280'")


Mla_pix = {
    MlaRes.Res1280: (1280, 1024),
    MlaRes.Res1024: (1024, 1024),
    MlaRes.Res768: (768, 768),
    MlaRes.Res512: (512, 512),
    MlaRes.Res320: (320, 320),
}


class ThorlabWFS(Device):
    """Thorlabs Wavefront Sensor (WFS) device driver.

    Provides comprehensive control over Thorlabs WFS hardware including
    spotfield image capture, wavefront reconstruction, Zernike fitting,
    and reference management.

    Attributes:
        device_type: DeviceType.WFS
        manufacturer: "Thorlabs"
        model: "WFS"
        version: "1.0.0"
    """

    device_type = DeviceType.WFS
    manufacturer = "Thorlabs"
    model = "WFS"
    version = "1.0.0"

    def __init__(
        self,
        mla_index: MlaRes | str | None = None,
        exposure_time: float | None = None,
        high_speed: bool | None = None,
        use_custom_ref: bool | None = None,
        pupil_diameter: float | None = None,
        pupil_center: tuple | None = None,
        stable_sample_enable: bool = False,
        stable_sample_n: int = 5,
        stable_variance_threshold: float = 0.1,
        stable_max_attempts: int = 50,
        device_id: str = "",
    ):
        """Initialize Thorlabs WFS driver.

        Args:
            mla_index: MLA resolution (if None, loaded from config).
                Positional arg for backward compatibility with WFSManager.
            exposure_time: Exposure time in ms; 0 means auto (if None, loaded from config).
            high_speed: Enable high speed mode (if None, loaded from config).
            use_custom_ref: Use custom reference file (if None, loaded from config).
            pupil_diameter: Pupil diameter in mm (if None, loaded from config).
            pupil_center: (cx, cy) center position (if None, loaded from config).
            stable_sample_enable: Enable automatic stable sample filtering.
            stable_sample_n: Number of stable samples to collect.
            stable_variance_threshold: Variance threshold for stability.
            stable_max_attempts: Max attempts before giving up.
            device_id: Unique device identifier (passed to Device base class).
        """
        super().__init__(device_id)

        # Store initialization parameters (deferred to open())
        if isinstance(mla_index, str):
            mla_index = MlaRes.from_str(mla_index)
        self._init_mla_index: MlaRes | None = mla_index
        self._init_exp_time: float | None = exposure_time
        self._init_high_speed: bool | None = high_speed
        self._init_use_custom_ref: bool | None = use_custom_ref
        self._init_pupil_diameter: float | None = pupil_diameter
        self._init_pupil_center: tuple | None = pupil_center
        self._init_stable_sample_enable: bool = stable_sample_enable
        self._init_stable_sample_n: int = stable_sample_n
        self._init_stable_variance_threshold: float = stable_variance_threshold
        self._init_stable_max_attempts: int = stable_max_attempts

        # Load DLL and initialize instrument handle
        self._lib = load_dll()
        self._wfs_instrument_index = c_int32()
        self.device_name = ''
        self.serial_num = ''
        self._instrument_handle = c_ulong(0)

        # Instance attributes (will be updated in open())
        self.use_custom_ref: bool = False
        self.mla_index: MlaRes = MlaRes.Res768
        self.image_pix = Mla_pix[self.mla_index]
        self.num_spots_x, self.num_spots_y = 0, 0

        self.c_x: float = 0.0
        self.c_y: float = 0.0
        self.d_x: float = 2.0
        self.d_y: float = 2.0

        self._explosure_time: float = 0.0
        self._gain = 1.0
        self.enable_high_speed: bool = False
        self._image_captured = False

        # Stable sampling parameters
        self.stable_sample_enable: bool = False
        self.stable_sample_n: int = 5
        self.stable_variance_threshold: float = 0.1
        self.stable_max_attempts: int = 50

        # Configuration manager
        self._config_manager: DeviceConfigManager | None = None

        # Register parameters and capabilities
        self._register_parameters()
        self._register_capabilities()

    def _register_parameters(self) -> None:
        """Register WFS-specific parameters."""
        self.register_parameter(
            "exposure_time_ms",
            default_value=0.0,
            min_value=EXP_TIME_LOW,
            max_value=EXP_TIME_HIGH,
            unit="ms",
            description="Camera exposure time in milliseconds",
        )
        self.register_parameter(
            "master_gain",
            default_value=1.0,
            min_value=1.0,
            max_value=24.0,
            unit="",
            description="Master gain for the WFS camera",
        )
        self.register_parameter(
            "high_speed_mode",
            default_value=False,
            unit="",
            description="Enable high speed camera mode",
        )
        self.register_parameter(
            "use_custom_ref",
            default_value=False,
            unit="",
            description="Use custom user reference file",
        )
        self.register_parameter(
            "mla_resolution",
            default_value=MlaRes.Res768,
            unit="",
            description="Microlens array resolution setting",
        )
        self.register_parameter(
            "black_level",
            default_value=c_int32(100),
            min_value=0,
            max_value=255,
            unit="",
            description="Black level offset for camera",
        )
        self.register_parameter(
            "trigger_mode",
            default_value=c_int32(0),
            min_value=0,
            max_value=3,
            unit="",
            description="Camera trigger mode (0=internal, 1-3=external)",
        )

    def _register_capabilities(self) -> None:
        """Register WFS capabilities."""
        self.register_capability(
            "measure_wavefront",
            description="Measure wavefront from current spotfield image",
            return_type=np.ndarray,
        )
        self.register_capability(
            "fit_zernike",
            description="Fit Zernike polynomials to measured wavefront",
            return_type=np.ndarray,
        )
        self.register_capability(
            "get_spot_image",
            description="Get current spotfield image",
            return_type=np.ndarray,
        )
        self.register_capability(
            "get_spot_deviations",
            description="Get spot deviations from reference positions",
            return_type=tuple,
        )
        self.register_capability(
            "save_reference",
            description="Save user reference file",
            return_type=Path,
        )
        self.register_capability(
            "load_reference",
            description="Load user reference file",
            return_type=bool,
        )

    # ==================== Device Base Class Overrides ====================

    def open(self) -> None:
        """Open connection to the WFS device and initialize.

        Raises:
            ConnectionError: If WFS is already in use or initialization fails.
        """
        self._set_state(DeviceState.CONNECTING)

        device_in_use = ViInt32()
        device_name = create_string_buffer(256)
        serial_number = create_string_buffer(256)
        resource_name = create_string_buffer(256)
        self._lib.WFS_GetInstrumentListInfo(
            VI_NULL(),
            ViInt32(0),
            byref(self._wfs_instrument_index),
            byref(device_in_use),
            device_name,
            serial_number,
            resource_name,
        )

        # check if WFS is in use, if not, connect to device
        assert not device_in_use, (
            "Wavefront sensor currently in use.... closing program"
        )

        self._lib.WFS_init(
            resource_name, c_bool(False), c_bool(True), byref(self._instrument_handle)
        )
        self.device_name = str(device_name.value, encoding='utf8')
        self.serial_num = str(serial_number.value, encoding='utf8')
        logger.info(
            f"Connected to {self.device_name} with Serial Number {self.serial_num}"
        )

        # Load configuration (init params take priority, then config values)
        config = self.load_config()

        # MLA index
        if self._init_mla_index is not None:
            self.mla_index = self._init_mla_index
        elif "mla_index" in config:
            self.mla_index = MlaRes(config["mla_index"])
        else:
            self.mla_index = MlaRes.Res768

        # exposure time
        if self._init_exp_time is not None:
            self._explosure_time = self._init_exp_time
        elif "exposure_time" in config:
            self._explosure_time = float(config["exposure_time"])
        else:
            self._explosure_time = 0.0

        # Validate exposure time
        if not (self._explosure_time == 0.0 or EXP_TIME_LOW <= self._explosure_time <= EXP_TIME_HIGH):
            logger.warning(
                f"exp_time {self._explosure_time} out of range, resetting to auto"
            )
            self._explosure_time = 0.0

        # high speed
        if self._init_high_speed is not None:
            self.enable_high_speed = self._init_high_speed
        elif "high_speed" in config:
            self.enable_high_speed = bool(config["high_speed"])
        else:
            self.enable_high_speed = False

        # use custom ref
        if self._init_use_custom_ref is not None:
            self.use_custom_ref = self._init_use_custom_ref
        elif "use_custom_ref" in config:
            self.use_custom_ref = bool(config["use_custom_ref"])
        else:
            self.use_custom_ref = False

        # pupil center
        if self._init_pupil_center is not None:
            self.c_x, self.c_y = self._init_pupil_center
        elif "pupil_center" in config:
            self.c_x, self.c_y = config["pupil_center"]
        else:
            self.c_x, self.c_y = 0.0, 0.0

        # pupil diameter
        if self._init_pupil_diameter is not None:
            self.d_x, self.d_y = self._init_pupil_diameter, self._init_pupil_diameter
        elif "pupil_diameter" in config:
            self.d_x, self.d_y = config["pupil_diameter"]
        else:
            self.d_x, self.d_y = 2.0, 2.0

        # stable sample parameters
        self.stable_sample_enable = self._init_stable_sample_enable
        self.stable_sample_n = self._init_stable_sample_n
        self.stable_variance_threshold = self._init_stable_variance_threshold
        self.stable_max_attempts = self._init_stable_max_attempts
        if self.stable_sample_enable:
            logger.info(
                f"Stable sample enabled: n={self.stable_sample_n}, "
                f"threshold={self.stable_variance_threshold}, "
                f"max_attempts={self.stable_max_attempts}"
            )

        if self.enable_high_speed:
            logger.info("high speed mode can only use auto exposure time!")

        self.select_mla(self.mla_index)
        self.set_ref_plane(self.use_custom_ref)
        if self._explosure_time <= 0:
            self._explosure_time, _ = self.optimize_exposure_time_and_gain()
        self.exposure_time = self._explosure_time
        self.high_speed = self.enable_high_speed
        self.pupil = (
            (self.c_x, self.c_y, self.d_x, self.d_y)
            if (self.d_x > 0 and self.d_y > 0)
            else self.optimize_pupil()
        )

        self._set_state(DeviceState.READY)
        logger.info(f"WFS device {self.device_id} ready")

    def close(self) -> None:
        """Close the WFS device connection and release resources."""
        if self._instrument_handle.value > 0:
            # Save configuration
            self.save_config()
            self.enable_high_speed = False

            self._lib.WFS_close(self._instrument_handle)
            self._instrument_handle = c_ulong(0)

        self._set_state(DeviceState.DISCONNECTED)
        logger.info(f"WFS device {self.device_id} closed")

    def is_connected(self) -> bool:
        """Check if the WFS device is connected and ready.

        Returns:
            True if instrument handle is valid and state is READY.
        """
        return self._instrument_handle.value > 0 and self._state == DeviceState.READY

    def get_hardware_info(self) -> dict:
        """Get hardware-specific information.

        Returns:
            Dictionary with serial_number, device_name, manufacturer, model,
            and firmware version.
        """
        return {
            "serial_number": self.serial_num,
            "device_name": self.device_name,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "firmware_version": "N/A",
        }

    # ==================== Config Management ====================

    def _init_config_manager(self) -> None:
        """Initialize WFS configuration manager."""
        if self._config_manager is None:
            # Default config directory: data/wfs_configs/
            project_root = Path(__file__).resolve().parents[4]
            config_dir = project_root / "data" / "wfs_configs"
            self._config_manager = DeviceConfigManager(config_dir, device_type="wfs")

    def load_config(self) -> dict:
        """Load JSON configuration file based on serial number.

        Returns:
            Configuration dict; empty dict if no serial number or file missing.
        """
        if not self.serial_num:
            logger.warning("WFS serial number not available, skipping config load")
            return {}
        self._init_config_manager()
        assert self._config_manager is not None
        return self._config_manager.load_config(self.serial_num)

    def save_config(self) -> None:
        """Save current parameters to JSON configuration file.

        Config items include: serial_number, mla_index, exposure_time, high_speed,
        pupil_center, pupil_diameter, use_custom_ref.
        """
        if not self.serial_num:
            logger.warning("WFS serial number not available, skipping config save")
            return

        self._init_config_manager()
        assert self._config_manager is not None

        config = {
            "mla_index": int(self.mla_index),
            "exposure_time": self._explosure_time,
            "high_speed": self.enable_high_speed,
            "pupil_center": (self.c_x, self.c_y),
            "pupil_diameter": (self.d_x, self.d_y),
            "use_custom_ref": self.use_custom_ref,
        }
        self._config_manager.save_config(self.serial_num, config)
        config_file = self._config_manager._get_config_file(self.serial_num)
        logger.info(f"WFS configuration saved: {config_file}")

    # ==================== Error Handling ====================

    def handle_error(self, err: ViStatus, no_raise: bool = False) -> None:
        """Handle WFS error by retrieving and logging error message.

        Args:
            err: Error code returned from WFS library.
            no_raise: If True, only log error without raising exception.
        """
        info = create_string_buffer(256)
        self._lib.WFS_error_message(self._instrument_handle, err, byref(info))
        logger.error(f"error: {info.value.decode('utf-8')}")
        if not no_raise:
            raise Exception(info.value)

    # ==================== MLA Configuration ====================

    def select_mla(self, mla_index: MlaRes) -> None:
        """Select and configure MLA (Micro Lens Array) holographic element.

        Args:
            mla_index: MLA resolution enum value.

        Note:
            This resets the camera configuration and updates num_spots_x/y.
        """
        self._lib.WFS_SelectMla(self._instrument_handle, 0)
        num_spots_x = c_int32()
        num_spots_y = c_int32()
        self._lib.WFS_ConfigureCam(
            self._instrument_handle,
            c_int32(0),
            c_int32(mla_index.value),
            byref(num_spots_x),
            byref(num_spots_y),
        )
        self.mla_index = mla_index
        self.image_pix = Mla_pix[mla_index]
        self.num_spots_x, self.num_spots_y = num_spots_x.value, num_spots_y.value
        logger.info(
            f"Number of detectable spots in X: {num_spots_x.value} \n"
            + f"Number of detectable spots in Y: {num_spots_y.value}"
        )

    def set_ref_plane(self, custom: bool) -> None:
        """Set reference plane for wavefront measurement.

        **DLL calls** (in order when ``custom=True``):
        1. ``WFS_SetReferencePlane(handle, 1)`` → switch to custom ref plane
        2. ``WFS_LoadUserRefFile(handle)`` → load ``.ref`` file from DLL-managed dir

        **DLL calls** (when ``custom=False``):
        1. ``WFS_SetReferencePlane(handle, 0)`` → switch to factory default

        **File transformations**:
        - When ``custom=True``:
          DLL reads ``<Ref_Dir>/<filename>.ref`` (created by ``save_user_ref()``).
          No files are written or modified.
        - When ``custom=False``:
          No file operations. Only DLL internal state change.

        **Fallback behavior**:
        If ``WFS_LoadUserRefFile`` fails (no ``.ref`` file exists yet),
        falls back to default reference with a warning instead of raising.

        Args:
            custom: If True, load custom user reference file; otherwise use default reference.
        """
        _select = 1 if custom else 0
        if err := self._lib.WFS_SetReferencePlane(
            self._instrument_handle, c_int32(_select)
        ):
            self.handle_error(err, no_raise=True)
            logger.warning(
                "WFS_SetReferencePlane failed, falling back to default reference"
            )
            self.use_custom_ref = False
            return

        if custom:
            if err := self._lib.WFS_LoadUserRefFile(self._instrument_handle):
                self.handle_error(err, no_raise=True)
                logger.warning(
                    "No user reference file available (use save_user_ref() first). "
                    "Falling back to default reference."
                )
                # Fall back to default reference
                self._lib.WFS_SetReferencePlane(self._instrument_handle, c_int32(0))
                self.use_custom_ref = False
            else:
                self.use_custom_ref = True
        else:
            self.use_custom_ref = False

    @require_take_image
    def create_default_user_ref(self) -> bool:
        """Create a default user reference from the current spotfield image.

        **DLL calls**:
        1. ``WFS_CreateDefaultUserReference(handle)`` — computes reference spot
           positions from the most recently captured spotfield image.

        **File transformations**: None.
        The reference is only set in DLL internal memory. To persist to disk,
        call ``save_user_ref()`` afterwards (which writes ``.ref`` file).

        **Typical workflow**:
        ``take_image()`` → ``create_default_user_ref()`` → ``save_user_ref()``

        Returns:
            True on success, False on failure.
        """
        if err := self._lib.WFS_CreateDefaultUserReference(self._instrument_handle):
            self.handle_error(err, no_raise=True)
            logger.error("Failed to create default user reference")
            return False
        logger.info("Default user reference created from current spotfield image")
        return True

    def get_mla_name(self) -> str:
        """Get the currently selected MLA name (e.g., 'MLA150M-5C').

        Calls ``WFS_GetMlaData(handle, 0, ...)`` to retrieve the name of
        the Microlens Array that was selected by ``WFS_SelectMla(handle, 0)``
        during initialization.

        .. important::

           The ``MLAIndex`` parameter of ``WFS_GetMlaData`` is **not** the
           camera resolution code (0=1280, 1=1024, 2=768, 3=512, 4=320).
           It is the ``MLA`` **selection index** (0 to ``MLACount-1``),
           which corresponds to the same value passed to ``WFS_SelectMla``.
           Since the driver always selects MLA 0 via ``WFS_SelectMla(handle, 0)``,
           we pass index 0 here.

        Returns:
            MLA name string (e.g. ``"MLA150M-5C"``), or empty string if the
            DLL call fails.
        """
        mla_name_buffer = create_string_buffer(256)
        err = self._lib.WFS_GetMlaData(
            self._instrument_handle,
            c_int32(0),  # MLA index = 0 (the selected MLA)
            mla_name_buffer,
            byref(c_double()),
            byref(c_double()),
            byref(c_double()),
            byref(c_double()),
            byref(c_double()),
            byref(c_double()),
            byref(c_double()),
        )
        if err != 0:
            self.handle_error(err, no_raise=True)
            name = self._try_get_mla_name_fallback()
            if name:
                logger.warning(
                    f"WFS_GetMlaData failed for index 0, "
                    f"falling back to alternative index: '{name}'"
                )
            return name
        name = mla_name_buffer.value
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        return name.strip("\x00").strip()

    def _try_get_mla_name_fallback(self) -> str:
        """Try alternative MLA indices (1, 2, ... up to 15) for a name.

        If the primary MLA index (0) is not available, this searches higher
        indices as a safety net. The WFS typically has 1-2 calibrated MLAs.

        Returns:
            MLA name string, or empty string if all attempts fail.
        """
        for alt_idx in range(1, MAX_MLA_INDICES):  # try indices 1..MAX_MLA_INDICES-1
            buf = create_string_buffer(256)
            err = self._lib.WFS_GetMlaData(
                self._instrument_handle,
                c_int32(alt_idx),
                buf,
                byref(c_double()),
                byref(c_double()),
                byref(c_double()),
                byref(c_double()),
                byref(c_double()),
                byref(c_double()),
                byref(c_double()),
            )
            if err == 0:
                name = buf.value
                if isinstance(name, bytes):
                    name = name.decode("utf-8")
                name = name.strip("\x00").strip()
                if name:
                    logger.warning(
                        f"_try_get_mla_name_fallback found name '{name}' at index {alt_idx}"
                    )
                    return name
        return ""

    @staticmethod
    def _get_ref_default_dir() -> Path:
        """Get the default WFS reference file directory.

        The Thorlabs WFS DLL saves/loads reference files to/from:
            C:\\Users\\<user>\\Documents\\Thorlabs\\Wavefront Sensor\\Reference

        On Linux, falls back to ~/.local/share/Thorlabs/WFS/Ref or project data directory.

        Returns:
            Path object for the reference directory.
        """
        import platform
        system = platform.system()

        if system == "Windows":
            # Use Path.home() which reliably resolves C:\Users\<username>
            # Fall back to USERPROFILE if home() fails
            home = Path.home()
            if not home or not home.exists():
                user_profile = os.environ.get("USERPROFILE")
                if user_profile:
                    home = Path(user_profile)
                else:
                    logger.warning("Cannot determine user home directory")
                    home = Path("C:/Users/Public")
            return home / "Documents" / "Thorlabs" / "Wavefront Sensor" / "Reference"
        else:
            # Linux/macOS fallback: use XDG_DATA_HOME or project data directory
            xdg_data = os.environ.get("XDG_DATA_HOME", "")
            if xdg_data:
                base = Path(xdg_data)
            else:
                base = Path.home() / ".local" / "share"

            ref_dir = base / "Thorlabs" / "WFS" / "Ref"
            logger.debug(f"[WFS _get_ref_default_dir] Linux fallback: {ref_dir}")

            # Fallback to project data/calibration if not writable
            project_fallback = Path("data") / "calibration" / "wfs_ref"
            if not ref_dir.parent.exists():
                logger.debug(f"[WFS _get_ref_default_dir] XDG path not accessible, using project fallback: {project_fallback}")
                return project_fallback

            return ref_dir

    def _get_ref_filename(self, fallback_if_empty: bool = True) -> str:
        """Construct the ``.ref`` filename that the WFS DLL expects.

        **Filename pattern** (from Thorlabs SDK):
            ``WFS_<serial_number>_<mla_name>_<cam_resol_idx>.ref``

        This filename is used by both ``save_user_ref()`` (to know where the
        DLL wrote the file) and ``load_user_ref()`` (to locate the file for
        copying or reading). It matches what the DLL expects internally.

        **No file transformations** — pure string construction.

        Args:
            fallback_if_empty: If True and ``mla_name`` is empty, use
                ``"unknown"`` as a placeholder so the filename is still valid.

        Returns:
            Filename string like ``WFS_M00224955_MLA150M-5C_0.ref``,
            or ``WFS_M00224955_unknown_3.ref`` if MLA name unavailable.
        """
        mla_name = self.get_mla_name()
        if not mla_name and fallback_if_empty:
            mla_name = "unknown"
            logger.warning(
                f"Cannot determine MLA name (WFS_GetMlaData failed), "
                f"using fallback filename with 'unknown': "
                f"WFS_{self.serial_num}_unknown_{int(self.mla_index)}.ref"
            )
        # cam_resol_idx is the MLA resolution index (self.mla_index value)
        cam_resol_idx = int(self.mla_index)
        filename = f"WFS_{self.serial_num}_{mla_name}_{cam_resol_idx}.ref"
        return filename

    @require_take_image
    def save_user_ref(self, backup_dir: str | Path | None = None) -> Path | None:
        """Save user reference file and create a timestamped backup.

        **DLL calls** (in order):
        1. ``WFS_SetSpotsToUserReference(handle)`` — promotes current spot
           centroids as the reference (internal DLL memory only).
        2. ``WFS_SaveUserRefFile(handle)`` — writes ``.ref`` file to the
           DLL-managed location on disk.

        **File transformations** (step by step):
        ::

            ┌─ DLL writes to:  <Ref_Dir>/<filename>.ref        (CREATED)
            └─ shutil.copy2 →  <backup_dir>/<filename>_<ts>.ref (CREATED)

        - **Ref_Dir** = ``%USERPROFILE%/Documents/Thorlabs/Wavefront Sensor/Reference``
        - **filename** = ``WFS_{serial}_{mla_name}_{res_index}``
        - If the MLA name cannot be determined (``WFS_GetMlaData`` fails), uses
          ``"unknown"`` as fallback in the filename.
        - If the expected filename is not found after DLL save, searches the
          ref directory for any ``WFS_{serial}_*.ref`` file as fallback.
        - The DLL-managed copy may be silently overwritten on next save;
          the timestamped backup preserves history.

        Args:
            backup_dir: Directory to store backup. Defaults to ``data/calibration/``.

        Returns:
            Path to the timestamped backup file, or None on failure.
        """
        if backup_dir is None:
            backup_dir = Path("data/calibration")
        else:
            backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Explicitly set current spot positions as the reference before saving.
        # Without this call, WFS_SaveUserRefFile may have nothing to persist,
        # resulting in a "No User Reference available!" error on subsequent load.
        if err := self._lib.WFS_SetSpotsToUserReference(self._instrument_handle):
            self.handle_error(err, no_raise=True)
            return None

        # Call DLL to save reference file to its default location
        if err := self._lib.WFS_SaveUserRefFile(self._instrument_handle):
            self.handle_error(err, no_raise=True)
            return None

        # Get the source file path (where DLL saved it)
        ref_dir = self._get_ref_default_dir()
        ref_filename = self._get_ref_filename()
        src_path = ref_dir / ref_filename

        if not src_path.exists():
            # Fallback: search ref_dir for any .ref file matching the serial number.
            # The DLL may choose a different filename when MLA name is unavailable.
            logger.warning(
                f"Reference file not found at expected path: {src_path}\n"
                f"Searching {ref_dir} for files matching serial {self.serial_num}..."
            )
            candidates = []
            if ref_dir.exists():
                pattern = f"WFS_{self.serial_num}_*.ref"
                candidates = sorted(ref_dir.glob(pattern))
            if candidates:
                src_path = candidates[0]
                logger.info(f"Found alternative reference file: {src_path}")
            else:
                logger.error(
                    f"No .ref file found for serial {self.serial_num} in {ref_dir}. "
                    f"The DLL (WFS_SaveUserRefFile) may have saved to a different location."
                )
                return None

        # Create timestamped backup filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{src_path.stem}_{timestamp}{src_path.suffix}"
        backup_path = backup_dir / backup_filename

        # Copy to backup location
        try:
            shutil.copy2(src_path, backup_path)
            logger.info(f"User reference saved to DLL location: {src_path}")
            logger.info(f"Backup created at: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return None

    def load_user_ref(self, backup_path: str | Path | None = None) -> bool:
        """Load user reference file from a backup or default location.

        **DLL calls** (in order):
        1. ``WFS_LoadUserRefFile(handle)`` — loads ``.ref`` file from DLL dir
        2. ``WFS_SetReferencePlane(handle, 1)`` — switches to custom ref plane

        **File transformations** (step by step):

        **With ``backup_path`` provided**::
        ::

            shutil.copy2(backup_path → <Ref_Dir>/<filename>.ref)  (COPIED)
            DLL reads ←  <Ref_Dir>/<filename>.ref                  (READ)

        **Without ``backup_path``**::
        ::

            DLL reads ←  <Ref_Dir>/<filename>.ref                  (READ)

        - **Ref_Dir** = ``%USERPROFILE%/Documents/Thorlabs/Wavefront Sensor/Reference``
        - If ``backup_path`` is given, the file is first copied to the DLL's
          expected location so ``WFS_LoadUserRefFile`` can find it.
        - If the file does not exist at the expected path, the DLL returns an
          error (previously "No User Reference available!") — see ``save_user_ref()``
          to create it.

        Args:
            backup_path: Optional path to a backup ``.ref`` file to load.
                If provided, the file is copied to the DLL-managed directory
                before calling ``WFS_LoadUserRefFile``.

        Returns:
            True on success, False on failure.
        """
        # Debug: print current system information
        import platform
        logger.debug(f"[WFS load_user_ref] Platform: {platform.system()}")
        logger.debug(f"[WFS load_user_ref] USERPROFILE: {os.environ.get('USERPROFILE', 'NOT_SET')}")

        if backup_path is not None:
            backup_path = Path(backup_path)
            if not backup_path.exists():
                logger.error(f"Backup file not found: {backup_path}")
                return False

            # Copy backup to DLL's expected location
            ref_dir = self._get_ref_default_dir()
            logger.debug(f"[WFS load_user_ref] ref_dir: {ref_dir}")
            logger.debug(f"[WFS load_user_ref] ref_dir exists: {ref_dir.exists()}")
            logger.debug(f"[WFS load_user_ref] ref_dir parent exists: {ref_dir.parent.exists() if ref_dir.parent != ref_dir else 'N/A'}")

            ref_dir.mkdir(parents=True, exist_ok=True)
            ref_filename = self._get_ref_filename()
            dst_path = ref_dir / ref_filename
            logger.debug(f"[WFS load_user_ref] dst_path: {dst_path}")

            try:
                shutil.copy2(backup_path, dst_path)
                logger.info(f"Copied backup {backup_path} to {dst_path}")
            except Exception as e:
                logger.error(f"Failed to copy backup file: {e}")
                return False

        # Call DLL to load reference file from its default location
        logger.debug("[WFS load_user_ref] Calling WFS_LoadUserRefFile...")
        if err := self._lib.WFS_LoadUserRefFile(self._instrument_handle):
            self.handle_error(err, no_raise=True)
            logger.error("Failed to load user reference file via WFS_LoadUserRefFile")
            # Additional debug: list the DLL's expected reference file path
            ref_dir = self._get_ref_default_dir()
            ref_filename = self._get_ref_filename()
            expected_path = ref_dir / ref_filename
            logger.debug(f"[WFS load_user_ref] Expected path: {expected_path}")
            logger.debug(f"[WFS load_user_ref] Expected path exists: {expected_path.exists()}")
            if not expected_path.exists():
                # List directory contents to help debugging
                try:
                    files = list(ref_dir.glob("*")) if ref_dir.exists() else []
                    logger.debug(f"[WFS load_user_ref] Files in ref_dir: {files}")
                except Exception as list_err:
                    logger.debug(f"[WFS load_user_ref] Cannot list ref_dir: {list_err}")
            return False

        # Also update the reference plane setting
        logger.debug("[WFS load_user_ref] Setting reference plane to custom...")
        if err := self._lib.WFS_SetReferencePlane(self._instrument_handle, c_int32(1)):
            self.handle_error(err, no_raise=True)
            logger.warning("WFS_SetReferencePlane returned error but continuing")

        self.use_custom_ref = True
        logger.info("User reference file loaded successfully")
        return True

    # ==================== Pupil / Image Capture ====================

    def optimize_pupil(self) -> tuple[float, float, float, float]:
        """Optimize pupil detection.

        This function calculates the beam centroid and diameter from the
        current spotfield data.

        Returns:
            tuple[float, float, float, float]: beam centroid x, beam centroid y,
                beam diameter x, beam diameter y
        """
        assert not self.enable_high_speed, "turn off high speed mode first"
        self._lib.WFS_CalcSpotsCentrDiaIntens(
            self._instrument_handle, c_int32(1), c_int32(1)
        )
        beam_centroid_x = c_double()
        beam_centroid_y = c_double()
        beam_diameter_x = c_double()
        beam_diameter_y = c_double()
        self._lib.WFS_CalcBeamCentroidDia(
            self._instrument_handle,
            byref(beam_centroid_x),
            byref(beam_centroid_y),
            byref(beam_diameter_x),
            byref(beam_diameter_y),
        )
        return (
            beam_centroid_x.value,
            beam_centroid_y.value,
            beam_diameter_x.value,
            beam_diameter_y.value,
        )

    def take_image(self, n_sample: int = 10, dynamicNoiseCut: bool = True) -> None:
        """Capture spotfield image and calculate spot centroids/diameters/intensities.

        Args:
            n_sample: Number of auto-exposure samples to take (used when exposure_time <= 0).
            dynamicNoiseCut: Enable dynamic noise floor cutoff for spot calculation.

        Note:
            Sets self._image_captured flag to True upon successful capture.
            For fixed exposure, takes single image; for auto exposure, iterates n_sample times.
        """
        if self._explosure_time > 0:
            if err := self._lib.WFS_TakeSpotfieldImage(self._instrument_handle):
                self.handle_error(err)
            else:
                self._image_captured = True
        else:
            # No fixed exposure time, use auto-exposure with multiple samples
            actual_exposure = c_double()
            actual_gain = c_double()
            for _ in range(n_sample):
                self._lib.WFS_TakeSpotfieldImageAutoExpos(
                    self._instrument_handle, byref(actual_exposure), byref(actual_gain)
                )
            self._image_captured = True

        # Calculate spot centroids, diameters, and intensities
        if res := self._lib.WFS_CalcSpotsCentrDiaIntens(
            self._instrument_handle, c_int32(1 if dynamicNoiseCut else 0), c_int32(0)
        ):
            self.handle_error(res)

    @require_take_image
    def get_spotfiled_image(self) -> np.ndarray:
        """Retrieve the captured spotfield image from the WFS device.

        Returns:
            np.ndarray: 2D uint8 image array of shape (512, 512)

        Raises:
            RuntimeError: If WFS_GetSpotfieldImage fails.
        """
        spots_filed_img = np.empty(MAX_SPOTS, np.uint8)
        if err := self._lib.WFS_GetSpotfieldImage(
            self._instrument_handle,
            spots_filed_img.ctypes.data_as(ctypes.POINTER(c_uint8)),
            byref(c_int32(MAX_SPOTS[0])),
            byref(c_int32(MAX_SPOTS[1])),
        ):
            raise RuntimeError(self.handle_error(err))
        return spots_filed_img

    @require_take_image
    def get_spots_statics(self) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Get spot intensities and centroid positions.

        Returns:
            tuple containing:
                - np.ndarray: spot intensities array of shape (num_spots_x, num_spots_y)
                - tuple[np.ndarray, np.ndarray]: (centroid_x, centroid_y) arrays
                  each of shape (num_spots_x, num_spots_y)

        Note:
            Requires high speed mode to be disabled.
            Spots outside the active pupil region will have NaN values.
        """
        assert not self.enable_high_speed, "turn off high speed mode first"
        spots_intensities = np.empty(MAX_SPOTS, dtype=np.float32)
        spots_center_x = np.empty(MAX_SPOTS, dtype=np.float32)
        spots_center_y = np.empty(MAX_SPOTS, dtype=np.float32)
        if err := self._lib.WFS_CalcSpotsCentrDiaIntens(
            self._instrument_handle, ViInt32(0), ViInt32(1)
        ):
            self.handle_error(err)
        else:
            # spots_diameter_x, spots_diameter_y = spots_intensities.copy(), spots_intensities.copy()
            self._lib.WFS_GetSpotIntensities(self._instrument_handle, spots_intensities)
            self._lib.WFS_GetSpotCentroids(
                self._instrument_handle, spots_center_x, spots_center_y
            )
            # self._lib.WFS_GetSpotDiameters(self._instrument_handle,
            #     np2c(spots_diameter_x), np2c(spots_diameter_y))
        return spots_intensities[: self.num_spots_x, : self.num_spots_y], (
            spots_center_x[: self.num_spots_x, : self.num_spots_y],
            spots_center_y[: self.num_spots_x, : self.num_spots_y],
        )

    def build_subaperture_mask(
        self,
        n_avg: int = 30,
        threshold_ratio: float = 0.3,
        edge_clip: int = 1,
        plot: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build valid subaperture mask for WFS40-5C.

        Args:
            n_avg: Number of frames to average for noise suppression.
            threshold_ratio: Intensity threshold as ratio of max intensity (0.2-0.4 typical).
            edge_clip: Number of lenslet rows/cols to clip from edges (1-2 recommended).
            plot: If True, display visualization.

        Returns:
            tuple: (mask_bool, valid_indices_flat)
                - mask_bool: 2D boolean array of shape (num_spots_x, num_spots_y)
                - valid_indices_flat: 1D array of flattened indices for valid subapertures
        """
        from scipy import ndimage

        intensities = []
        for _ in range(n_avg):
            self.take_image(n_sample=1, dynamicNoiseCut=True)
            spots_intensities = np.empty(MAX_SPOTS, dtype=np.float32)
            self._lib.WFS_GetSpotIntensities(self._instrument_handle, spots_intensities)
            int_mat = spots_intensities[: self.num_spots_x, : self.num_spots_y]
            intensities.append(int_mat)

        int_mean = np.mean(intensities, axis=0)
        int_max = np.max(int_mean)
        threshold = int_max * threshold_ratio

        mask = int_mean > threshold

        if edge_clip > 0:
            mask[:edge_clip, :] = False
            mask[-edge_clip:, :] = False
            mask[:, :edge_clip] = False
            mask[:, -edge_clip:] = False

        mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
        mask = ndimage.binary_closing(mask, structure=np.ones((3, 3)))

        valid_flat = np.where(mask.flatten())[0]

        logger.info(
            f"Valid subapertures: {np.sum(mask)}/{mask.size} "
            f"({np.sum(mask) / mask.size * 100:.1f}%), "
            f"threshold={threshold:.1f} (max={int_max:.1f})"
        )

        if plot:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(int_mean, cmap="hot")
            axes[0].set_title("Mean Intensity")
            axes[1].imshow(mask, cmap="gray")
            axes[1].set_title("Valid Mask")
            overlay = np.dstack([int_mean / int_max] * 3)
            overlay[np.logical_not(mask)] = [0, 0, 1]
            axes[2].imshow(overlay)
            axes[2].set_title("Overlay (invalid=blue)")
            plt.tight_layout()
            plt.show()

        return mask, valid_flat

    def _get_stable_samples(
        self,
        n_samples: int,
        variance_threshold: float,
        max_attempts: int,
    ) -> list[np.ndarray]:
        """Collect stable samples based on variance threshold.

        Args:
            n_samples: Number of stable samples to collect.
            variance_threshold: Maximum allowed variance for a sample to be considered stable.
            max_attempts: Maximum number of attempts before giving up.

        Returns:
            List of stable sample arrays.
        """
        stable_samples = []
        attempts = 0

        while len(stable_samples) < n_samples and attempts < max_attempts:
            self.take_image(n_sample=1, dynamicNoiseCut=True)
            self._lib.WFS_CalcSpotToReferenceDeviations(
                self._instrument_handle, c_int32(0)
            )

            wavefront = np.empty(MAX_SPOTS, dtype=c_float)
            self._lib.WFS_CalcWavefront(
                self._instrument_handle,
                ViInt32(0),
                ViInt32(0),
                wavefront,
            )
            wf_slice = wavefront[: self.num_spots_x, : self.num_spots_y]

            variance = float(np.var(wf_slice))
            if variance < variance_threshold:
                stable_samples.append(wf_slice)
                logger.debug(
                    f"Stable sample {len(stable_samples)}/{n_samples}: "
                    f"variance={variance:.6f} < {variance_threshold}"
                )
            else:
                logger.debug(
                    f"Unstable sample rejected: variance={variance:.6f} >= {variance_threshold}"
                )

            attempts += 1

        if len(stable_samples) < n_samples:
            logger.warning(
                f"Only collected {len(stable_samples)}/{n_samples} stable samples "
                f"after {attempts} attempts"
            )

        return stable_samples

    # ==================== Wavefront Measurement ====================

    @require_take_image
    def get_wavefront(self, cancel_tile: bool = False) -> tuple[np.ndarray, dict]:
        """Calculate wavefront from spot deviations.

        Args:
            cancel_tile: If True, remove tip/tilt from wavefront measurement.

        Returns:
            tuple[np.ndarray, dict]: (wavefront array, statistics dict)
                - wavefront: 2D array of wavefront values in waves
                - statistics: dict with keys 'min', 'max', 'diff', 'mean', 'rms', 'wighted_rms'

        Note:
            Uses measured wavefront (type=0) with adaptive pupil compensation if pupil is defined.
            When stable_sample_enable is True, collects multiple stable samples and returns mean.
        """
        # Use stable sampling if enabled
        if self.stable_sample_enable:
            stable_wfs = self._get_stable_samples(
                self.stable_sample_n,
                self.stable_variance_threshold,
                self.stable_max_attempts,
            )
            if stable_wfs:
                wavefront = np.mean(stable_wfs, axis=0)
            else:
                return np.zeros((self.num_spots_x, self.num_spots_y), dtype=np.float32), {
                    "min": np.nan,
                    "max": np.nan,
                    "diff": np.nan,
                    "mean": np.nan,
                    "rms": np.nan,
                    "wighted_rms": np.nan,
                }
        else:
            if res := self._lib.WFS_CalcSpotToReferenceDeviations(
                self._instrument_handle, c_int32(1 if cancel_tile else 0)
            ):
                self.handle_error(res)
            adaptive_pupil = 0 if (self.d_x and self.d_y) else 1
            wavefront = np.empty(MAX_SPOTS, dtype=c_float)
            wavefront_type = 0
            if err := self._lib.WFS_CalcWavefront(
                self._instrument_handle,
                ViInt32(wavefront_type),
                ViInt32(adaptive_pupil),
                wavefront,
            ):
                self.handle_error(err)
                wavefront = np.zeros((self.num_spots_x, self.num_spots_y), dtype=np.float32)
            else:
                wavefront = deepcopy(wavefront)[: self.num_spots_x, : self.num_spots_y]

        # Calculate statistics from the wavefront (regardless of how it was obtained)
        min_val, max_val, diff_val, mean_val = c_double(), c_double(), c_double(), c_double()
        rms_val, wighted_rms_val = c_double(), c_double()
        self._lib.WFS_CalcWavefrontStatistics(
            self._instrument_handle,
            byref(min_val),
            byref(max_val),
            byref(diff_val),
            byref(mean_val),
            byref(rms_val),
            byref(wighted_rms_val),
        )

        if np.all(wavefront == 0):
            logger.warning("WFS_CalcWavefront returned zero-filled buffer — DLL may not have written data")

        if WFS_DEBUG_MODE:
            wf_variance = np.var(wavefront)
            wf_std = np.std(wavefront)
            logger.debug(f"WFS wavefront stats: var={wf_variance:.6f}, std={wf_std:.6f}, shape={wavefront.shape}")

        return wavefront, {
            "min": min_val.value if not self.stable_sample_enable else float(np.min(wavefront)),
            "max": max_val.value if not self.stable_sample_enable else float(np.max(wavefront)),
            "diff": diff_val.value if not self.stable_sample_enable else float(np.max(wavefront) - np.min(wavefront)),
            "mean": mean_val.value if not self.stable_sample_enable else float(np.mean(wavefront)),
            "rms": rms_val.value if not self.stable_sample_enable else float(np.std(wavefront)),
            "wighted_rms": wighted_rms_val.value if not self.stable_sample_enable else float(np.std(wavefront)),
        }

    def _remove_tilt(self, wavefront: np.ndarray) -> np.ndarray:
        """Remove tilt (tip/tilt) from wavefront by fitting a plane and subtracting it.

        Args:
            wavefront: 2D wavefront array.

        Returns:
            Wavefront with tilt removed.
        """
        # Create coordinate grids (normalized to [-1, 1] for numerical stability)
        ny, nx = wavefront.shape
        y, x = np.meshgrid(np.linspace(-1, 1, ny), np.linspace(-1, 1, nx), indexing="ij")

        # Flatten for fitting
        z = wavefront.flatten()

        # Build design matrix for plane: z = a*x + b*y + c
        A = np.column_stack([x.flatten(), y.flatten(), np.ones_like(x.flatten())])

        # Solve for coefficients using least squares
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
            a, b, c = coeffs

            # Create the tilt plane
            tilt_plane = a * x + b * y + c

            # Subtract the tilt
            wavefront_no_tilt = wavefront - tilt_plane
        except Exception:
            # If fitting fails, just return original
            logger.warning("Failed to fit tilt plane, returning original wavefront")
            return wavefront

        return wavefront_no_tilt

    @staticmethod
    def calc_n_zernike_terms(n: int) -> int:
        """Calculate the number of Zernike terms for a given order.

        Args:
            n: Zernike order.

        Returns:
            Number of Zernike terms.
        """
        return (n + 1) * (n + 2) // 2 + 1

    @require_take_image
    def get_zernike(self, zernike_order: int = 10) -> np.ndarray:
        """Calculate Zernike polynomial coefficients from spot deviations.

        Args:
            zernike_order: Zernike order (max 10, indexed from 0).

        Returns:
            np.ndarray: Zernike coefficients array.

        Raises:
            AssertionError: If zernike_order exceeds 10.
        """
        assert zernike_order <= 10, (
            f"zernike order must be less than or equal to 10, got {zernike_order}"
        )
        roc_mm = c_double()
        coeff_num = self.calc_n_zernike_terms(zernike_order)
        zernike_order_c = c_int32(zernike_order)
        zernike_um = np.empty((coeff_num,), c_float)
        zernike_orders_rms_um = np.empty((zernike_order + 1,), c_float)

        if res := self._lib.WFS_CalcSpotToReferenceDeviations(
            self._instrument_handle, c_int32(0)
        ):
            self.handle_error(res)

        if err := self._lib.WFS_ZernikeLsf(
            self._instrument_handle,
            byref(zernike_order_c),
            zernike_um.ctypes.data_as(ctypes.POINTER(c_float)),
            zernike_orders_rms_um.ctypes.data_as(ctypes.POINTER(c_float)),
            byref(roc_mm),
        ):
            self.handle_error(err)
            return np.empty_like(zernike_um)
        return zernike_um

    # ==================== Spot Deviation ====================

    @require_take_image
    def get_spot_deviation(self, cancel_tile: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Get spot deviation from reference positions.

        Args:
            cancel_tile: If True, remove tip/tilt from deviations.

        Returns:
            tuple[np.ndarray, np.ndarray]: (deviation_x, deviation_y) arrays
               each of shape (num_spots_x, num_spots_y).

        Note:
            When stable_sample_enable is True, collects multiple stable samples and returns mean.
        """
        # Use stable sampling if enabled
        if self.stable_sample_enable:
            stable_x_list = []
            stable_y_list = []
            attempts = 0

            while len(stable_x_list) < self.stable_sample_n and attempts < self.stable_max_attempts:
                self.take_image(n_sample=1, dynamicNoiseCut=True)

                _spots_dev_x = np.empty(MAX_SPOTS, dtype=np.float32)
                _spots_dev_y = np.empty(MAX_SPOTS, dtype=np.float32)

                if (
                    res := self._lib.WFS_CalcSpotToReferenceDeviations(
                        self._instrument_handle, c_int32(1 if cancel_tile else 0)
                    )
                ) == 0:
                    if err := self._lib.WFS_GetSpotDeviations(
                        self._instrument_handle, _spots_dev_x, _spots_dev_y
                    ):
                        self.handle_error(err)
                else:
                    self.handle_error(res)
                    continue

                dev_x = _spots_dev_x[: self.num_spots_x, : self.num_spots_y]
                dev_y = _spots_dev_y[: self.num_spots_x, : self.num_spots_y]

                # Calculate variance as stability metric
                variance = float(np.var(dev_x) + np.var(dev_y))
                if variance < self.stable_variance_threshold:
                    stable_x_list.append(dev_x)
                    stable_y_list.append(dev_y)
                    logger.debug(
                        f"Stable deviation sample {len(stable_x_list)}/{self.stable_sample_n}: "
                        f"variance={variance:.6f} < {self.stable_variance_threshold}"
                    )
                else:
                    logger.debug(
                        f"Unstable deviation sample rejected: variance={variance:.6f} >= {self.stable_variance_threshold}"
                    )

                attempts += 1

            if len(stable_x_list) < self.stable_sample_n:
                logger.warning(
                    f"Only collected {len(stable_x_list)}/{self.stable_sample_n} stable deviation samples "
                    f"after {attempts} attempts"
                )

            if stable_x_list:
                x = np.mean(stable_x_list, axis=0)
                y = np.mean(stable_y_list, axis=0)
            else:
                x = np.zeros((self.num_spots_x, self.num_spots_y), dtype=np.float32)
                y = np.zeros((self.num_spots_x, self.num_spots_y), dtype=np.float32)
        else:
            _spots_deviation_x = np.empty(MAX_SPOTS, dtype=np.float32)
            _spots_deviation_y = np.empty(MAX_SPOTS, dtype=np.float32)

            if (
                res := self._lib.WFS_CalcSpotToReferenceDeviations(
                    self._instrument_handle, c_int32(1 if cancel_tile else 0)
                )
            ) == 0:
                if err := self._lib.WFS_GetSpotDeviations(
                    self._instrument_handle, _spots_deviation_x, _spots_deviation_y
                ):
                    self.handle_error(err)
            else:
                self.handle_error(res)
            x = _spots_deviation_x[: self.num_spots_x, : self.num_spots_y]
            y = _spots_deviation_y[: self.num_spots_x, : self.num_spots_y]

        if np.all(x == 0) and np.all(y == 0):
            logger.warning("WFS_GetSpotDeviations returned zero-filled buffers — DLL may not have written data")

        return x, y

    @require_take_image
    def get_stable_spot_deviation(
        self, intensity_threshold: float = 0.0, cancel_tile: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get spot deviations with low-intensity subapertures zeroed out.

        This function:
        1. Calculates spot intensities and centroids via WFS_CalcSpotsCentrDiaIntens
        2. Calculates spot deviations via WFS_CalcSpotToReferenceDeviations
        3. Zeros out deviation for subapertures where intensity < intensity_threshold

        Args:
            intensity_threshold: Minimum intensity threshold. Subapertures with
                intensity below this value will have their deviations set to 0.
                Default 0.0 means no filtering (all subapertures included).
            cancel_tile: If True, remove tip/tilt from deviations.

        Returns:
            tuple[np.ndarray, np.ndarray]: (deviation_x, deviation_y)
                arrays of shape (num_spots_x, num_spots_y).
                Subapertures with intensity < threshold have deviation = 0.
        """
        # Step 1: Calculate intensities and centroids
        spots_intensities = np.empty(MAX_SPOTS, dtype=np.float32)
        if res := self._lib.WFS_CalcSpotsCentrDiaIntens(
            self._instrument_handle, c_int32(1), c_int32(0)
        ):
            self.handle_error(res)
            return (
                np.zeros((self.num_spots_x, self.num_spots_y), dtype=np.float32),
                np.zeros((self.num_spots_x, self.num_spots_y), dtype=np.float32),
            )

        self._lib.WFS_GetSpotIntensities(self._instrument_handle, spots_intensities)
        intensities = spots_intensities[: self.num_spots_x, : self.num_spots_y]

        # Step 2: Calculate deviations
        _spots_deviation_x = np.empty(MAX_SPOTS, dtype=np.float32)
        _spots_deviation_y = np.empty(MAX_SPOTS, dtype=np.float32)

        if (
            res := self._lib.WFS_CalcSpotToReferenceDeviations(
                self._instrument_handle, c_int32(1 if cancel_tile else 0)
            )
        ) != 0:
            self.handle_error(res)
            return (
                np.zeros((self.num_spots_x, self.num_spots_y), dtype=np.float32),
                np.zeros((self.num_spots_x, self.num_spots_y), dtype=np.float32),
            )

        if err := self._lib.WFS_GetSpotDeviations(
            self._instrument_handle, _spots_deviation_x, _spots_deviation_y
        ):
            self.handle_error(err)
            return (
                np.zeros((self.num_spots_x, self.num_spots_y), dtype=np.float32),
                np.zeros((self.num_spots_x, self.num_spots_y), dtype=np.float32),
            )

        x = _spots_deviation_x[: self.num_spots_x, : self.num_spots_y]
        y = _spots_deviation_y[: self.num_spots_x, : self.num_spots_y]

        # Step 3: Zero out deviations for low-intensity subapertures
        if intensity_threshold > 0.0:
            low_intensity_mask = intensities < intensity_threshold
            x[low_intensity_mask] = 0.0
            y[low_intensity_mask] = 0.0
            zeroed_count = np.sum(low_intensity_mask)
            if zeroed_count > 0:
                logger.info(
                    f"Zeroed deviations for {zeroed_count} subapertures "
                    f"with intensity < {intensity_threshold}"
                )

        return x, y

    # ==================== Exposure / Gain Management ====================

    def optimize_exposure_time_and_gain(self) -> tuple[float, float]:
        """Automatically optimize exposure time and gain for clear spotfield images.

        Takes up to 10 test images, checking device status after each to find usable exposure
        that is neither saturated (power too high) nor too dim (power too low).

        Returns:
            tuple[float, float]: (optimal_exposure_time_ms, optimal_gain)

        Note:
            Does NOT change device settings; caller should apply returned values.
        """
        lib, instrument_handle = self._lib, self._instrument_handle
        # Take a series of images until one is usable. Check the device status after each image to determine usability
        actual_exposure = c_double()
        actual_gain = c_double()
        device_status = c_int32()
        for i in range(MAX_AUTOEXPOSE_ATTEMPTS):
            lib.WFS_TakeSpotfieldImageAutoExpos(
                instrument_handle, byref(actual_exposure), byref(actual_gain)
            )
            lib.WFS_GetStatus(instrument_handle, byref(device_status))
            if device_status.value & 0x00000002:
                logger.warning("Power too high")
            elif device_status.value & 0x00000004:
                logger.warning("Power too low")
            elif device_status.value & 0x00000008:
                logger.warning("High ambient light")
            else:
                logger.info(
                    f"Image is usable at {actual_exposure.value} ms.... breaking loop"
                )
                break
        return actual_exposure.value, actual_gain.value

    def get_exposure_time_range(self) -> tuple[float, float, float]:
        """Get the hardware-supported exposure time range.

        Returns:
            tuple[float, float, float]: (min_exposure_ms, max_exposure_ms, increment_ms)
                All values are in seconds.

        Note:
            Returns cached values if already retrieved; queries hardware on first call.
        """
        if hasattr(self, "_exposure_time_range"):
            return self._exposure_time_range

        min_exp = c_double()
        max_exp = c_double()
        increment = c_double()
        err = self._lib.WFS_GetExposureTimeRange(
            self._instrument_handle,
            byref(min_exp),
            byref(max_exp),
            byref(increment),
        )
        if err != 0:
            self.handle_error(err, no_raise=True)
            # Fall back to constants if hardware query fails
            logger.warning("Failed to get exposure range from hardware, using defaults")
            return EXP_TIME_LOW, EXP_TIME_HIGH, 0.001

        # Convert from microseconds to milliseconds
        self._exposure_time_range = (
            min_exp.value / 1000.0,
            max_exp.value / 1000.0,
            increment.value / 1000.0 if increment.value > 0 else 0.001,
        )
        logger.info(
            f"Exposure time range: {self._exposure_time_range[0]:.3f} ~ "
            f"{self._exposure_time_range[1]:.3f} ms (step: {self._exposure_time_range[2]:.3f} ms)"
        )
        return self._exposure_time_range

    # ==================== Properties ====================

    @property
    def exposure_time(self) -> c_double:
        """Get current exposure time in milliseconds."""
        actual_exposure = c_double()
        self._lib.WFS_GetExposureTime(self._instrument_handle, actual_exposure)
        return actual_exposure

    @exposure_time.setter
    def exposure_time(self, value: float) -> None:
        """Set exposure time.

        Args:
            value: Exposure time in milliseconds.

        Raises:
            AssertionError: If value is outside valid range [0.002, 86] ms.
        """
        assert EXP_TIME_LOW <= value <= EXP_TIME_HIGH, (
            f"exposure time must be in range [{EXP_TIME_LOW}, {EXP_TIME_HIGH}] ms"
        )
        actual_exposure = c_double()
        self._lib.WFS_SetExposureTime(
            self._instrument_handle, c_double(value), byref(actual_exposure)
        )
        logger.info(f"actual exposure time is {actual_exposure.value} ms.")

    @property
    def pupil(self) -> tuple[float, float, float, float]:
        """Get current pupil configuration.

        Returns:
            tuple[float, float, float, float]: (centroid_x, centroid_y, diameter_x, diameter_y) in mm.
        """
        beam_centroid_x = c_double()
        beam_centroid_y = c_double()
        beam_diameter_x = c_double()
        beam_diameter_y = c_double()
        self._lib.WFS_GetPupil(
            self._instrument_handle,
            byref(beam_centroid_x),
            byref(beam_centroid_y),
            byref(beam_diameter_x),
            byref(beam_diameter_y),
        )
        self.c_x, self.c_y = beam_centroid_x.value, beam_centroid_y.value
        self.d_x, self.d_y = beam_diameter_x.value, beam_diameter_y.value
        return self.c_x, self.c_y, self.d_x, self.d_y

    @pupil.setter
    def pupil(self, center_and_diameter: tuple[float, float, float, float]) -> None:
        """Set pupil configuration.

        Args:
            center_and_diameter: (centroid_x, centroid_y, diameter_x, diameter_y) in mm.

        Note:
            If diameter_x or diameter_y is <= 0, optimize_pupil() is called instead.
        """
        c_x, c_y, d_x, d_y = center_and_diameter
        self._lib.WFS_SetPupil(
            self._instrument_handle,
            c_double(c_x),
            c_double(c_y),
            c_double(d_x),
            c_double(d_y),
        )
        logger.info(f"pupil is {c_x=}, {c_y=}, {d_x=}, {d_y=}")
        self.c_x, self.c_y, self.d_x, self.d_y = c_x, c_y, d_x, d_y

    @property
    def high_speed(self) -> bool:
        """Check if high speed mode is enabled.

        Returns:
            bool: True if high speed mode is active.
        """
        enable_high_speed = self._lib.WFS_CheckHighspeedCentroids(
            self._instrument_handle
        ).value
        return enable_high_speed.value

    @high_speed.setter
    def high_speed(self, enable: bool) -> None:
        """Enable or disable high speed mode.

        Args:
            enable: True to enable high speed mode, False to disable.

        Note:
            High speed mode only supports 512x512 resolution and requires auto exposure.
            Automatically re-optimizes pupil after enabling.
        """
        if self.device_name.upper() == 'WFS40-5C':
            logger.warning(f'{self.device_name} not support high speed mode!')
            return

        def __set_high_speed():
            return self._lib.WFS_SetHighspeedMode(
                self._instrument_handle,
                c_int32(1 if enable else 0),
                c_int32(1),
                c_int32(1),
                c_int32(1),
            )

        self.enable_high_speed = False
        if enable:
            self.optimize_exposure_time_and_gain()
            self._lib.WFS_CalcSpotsCentrDiaIntens(
                self._instrument_handle, c_int32(1), c_int32(1)
            )

        if res := __set_high_speed():  # res == 0 means success
            self.handle_error(res, True)
            self.pupil = self.optimize_pupil()
            if res := __set_high_speed():  # try again with auto pupil settings
                self.handle_error(res, True)
        else:
            self.enable_high_speed = enable
            if self.enable_high_speed:
                windowCountX = ViInt32()
                windowCountY = ViInt32()
                windowSizeX = ViInt32()
                windowSizeY = ViInt32()
                windowStartposX = np.zeros(
                    self.num_spots_x, dtype=np.int32
                )  # This parameter returns a one-dimensional array containing the start positions in pixels for spot windows in X direction.
                windowStartposY = np.zeros(
                    self.num_spots_y, dtype=np.int32
                )  # This parameter returns a one-dimensional array containing the start positions in pixels for spot windows in Y direction.

                self._lib.WFS_GetHighspeedWindows(
                    self._instrument_handle,
                    byref(windowCountX),
                    byref(windowCountY),
                    byref(windowSizeX),
                    byref(windowSizeY),
                    np2c(windowStartposX),
                    np2c(windowStartposY),
                )

                self.hs_window_count_x = windowCountX.value
                self.hs_window_count_y = windowCountY.value
                self.hs_window_size_x = windowSizeX.value
                self.hs_window_size_y = windowSizeY.value
                self.hs_window_startpos_x = windowStartposX
                self.hs_window_startpos_y = windowStartposY

        logger.info("high speed mode is " + "on" if self.enable_high_speed else "off")


# Backward compatibility aliases
WFSManager = ThorlabWFS
Thorlab_WFS = ThorlabWFS
