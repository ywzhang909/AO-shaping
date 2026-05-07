# refreces docs : https://github.com/nvladimus/WFS/blob/master/python/Thorlabs-WFS-read-average-wavefront.ipynb

from __future__ import annotations
from enum import IntEnum

import shutil
from copy import deepcopy
from functools import wraps
from loguru import logger
from pathlib import Path
from datetime import datetime

import numpy as np
import ctypes
from ctypes import (
    c_double, c_uint8, c_int32, c_bool, c_float, c_ulong, create_string_buffer, byref
)

from ._thorlab_wfs import (
    load_dll, np2c, VI_NULL, ViInt32, ViStatus)
from ._thorlab_wfs import MAX_SPOTS

EXP_TIME_LOW = 0.002
EXP_TIME_HIGH = 86

def require_take_image(func):
    """
    装饰器：确保在调用需要图像的方法前，已执行过 take_image。
    如果未拍摄，会自动调用 take_image()。
    """
    @wraps(func)
    def wrapper(self:"WFSManager", *args, **kwargs):
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
    def from_str(cls, value: str | int | "MlaRes") -> "MlaRes":
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

class WFSManager:
    """Wavefront Sensor Manager"""

    def __init__(
        self,
        mla_index: MlaRes = MlaRes.Res768,
        exp_time: float = 0.0,
        high_speed: bool = False,
        use_custom_ref: bool = False,
        pupil_diameter: float = 2.0,
        pupil_center: tuple = (0.0, 0.0),
    ):
        """
        mla_index: MlaRes
        exp_time: exposure time in ms, 0 means auto
        high_speed: enable high speed mode, only 512x512 resolution supported
        use_custom_ref: use custom reference file, if not, use default reference file
        pupil_diameter: pupil diameter in mm, default 2.0
        """
        assert mla_index in MlaRes, "mla_index must be one of MlaRes"
        assert exp_time == 0.0 or EXP_TIME_LOW <= exp_time <= EXP_TIME_HIGH, (
            f"exp_time must be in [{EXP_TIME_LOW},{EXP_TIME_HIGH}], now is {exp_time}"
        )

        self._lib = load_dll()
        self.device_id = c_int32()
        self.device_name = ''
        self.serial_num = ''
        self._instrument_handle = c_ulong(0)

        self.use_custom_ref = use_custom_ref
        self.mla_index = mla_index
        self.image_pix = Mla_pix[mla_index]
        self.num_spots_x, self.num_spots_y = 0, 0

        self.c_x, self.c_y = pupil_center
        self.d_x, self.d_y = pupil_diameter, pupil_diameter

        self._explosure_time = exp_time
        self._gain = 1.0
        self.enable_high_speed = high_speed
        if self.enable_high_speed:
            logger.info("high speed mode can only use auto exposure time!")
        self._image_captured = False

    def __enter__(self) -> "WFSManager":
        """Enter context manager, initialize the device connection.

        Returns:
            WFSManager: self instance for use in with statement
        """
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit context manager, close the device connection.

        Args:
            exc_type: Exception type if an exception occurred
            exc_value: Exception value if an exception occurred
            traceback: Traceback if an exception occurred
        """
        self.close()

    def initialize(self):
        device_in_use = ViInt32()
        device_name = create_string_buffer(256)
        serial_number = create_string_buffer(256)
        resource_name = create_string_buffer(256)
        self._lib.WFS_GetInstrumentListInfo(
            VI_NULL(),
            ViInt32(0),
            byref(self.device_id),
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

    def close(self) -> None:
        """Close the WFS device connection and release resources."""
        if self._instrument_handle.value > 0:
            self.enable_high_speed = False

            self._lib.WFS_close(self._instrument_handle)
            self._instrument_handle = c_ulong(0)

    def handle_error(self, err: ViStatus, no_raise: bool = False) -> None:
        """Handle WFS error by retrieving and logging error message.

        Args:
            err: Error code returned from WFS library
            no_raise: If True, only log error without raising exception
        """
        info = create_string_buffer(256)
        error_code = ViStatus(err)
        self._lib.WFS_error_message(self._instrument_handle, error_code, byref(info))
        logger.error(f"error: {info.value.decode('utf-8')}")
        if not no_raise:
            raise Exception(info.value)

    def select_mla(self, mla_index: int) -> None:
        """Select and configure MLA (Micro Lens Array) holographic element.

        Args:
            mla_index: MLA resolution enum value

        Note:
            This resets the camera configuration and updates num_spots_x/y
        """
        self._lib.WFS_SelectMla(self._instrument_handle, 0)
        num_spots_x = c_int32()
        num_spots_y = c_int32()
        self._lib.WFS_ConfigureCam(
            self._instrument_handle,
            c_int32(0),
            c_int32(mla_index),
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

        Args:
            custom: If True, load custom user reference file; otherwise use default reference

        Raises:
            Exception: If loading custom reference file fails
        """
        _select = 1 if custom else 0
        if err := self._lib.WFS_SetReferencePlane(
            self._instrument_handle, c_int32(_select)
        ):
            self.handle_error(err)

        elif custom:
            if err := self._lib.WFS_LoadUserRefFile(self._instrument_handle):
                self.handle_error(err)
            else:
                self.use_custom_ref = True
                return

        self.use_custom_ref = False

    def get_mla_name(self, mla_index: int | None = None) -> str:
        """Get MLA name string (e.g., 'MLA150M-5C') for the specified MLA index.

        Args:
            mla_index: MLA index. If None, uses self.mla_index.

        Returns:
            MLA name string, or empty string on failure.
        """
        if mla_index is None:
            mla_index = self.mla_index
        mla_name_buffer = create_string_buffer(256)
        err = self._lib.WFS_GetMlaData(
            self._instrument_handle,
            c_int32(mla_index),
            mla_name_buffer,
            byref(c_double()),
            byref(c_double()),
            byref(c_double()),
            byref(c_double()),
            byref(c_double()),
            byref(c_double()),
        )
        if err != 0:
            self.handle_error(err, no_raise=True)
            return ""
        name = mla_name_buffer.value
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        return name.strip("\x00").strip()

    @staticmethod
    def _get_ref_default_dir() -> Path:
        """Get the default WFS reference file directory.

        The Thorlabs WFS DLL saves/loads reference files to/from:
            C:\\Users\\<user>\\Documents\\Thorlabs\\Wavefront Sensor\\Reference

        Returns:
            Path object for the reference directory.
        """
        import os
        user_docs = Path(os.environ.get("USERPROFILE", "")) / "Documents"
        return user_docs / "Thorlabs" / "Wavefront Sensor" / "Reference"

    def _get_ref_filename(self) -> str:
        """Construct the .ref filename that the WFS DLL expects.

        Filename pattern from Thorlabs docs:
            WFS_<serial_number>_<mla_name>_<cam_resol_idx>.ref

        Returns:
            Filename string like 'WFS_M00224955_MLA150M-5C_0.ref'.
        """
        mla_name = self.get_mla_name()
        # cam_resol_idx is the MLA resolution index (self.mla_index value)
        cam_resol_idx = int(self.mla_index)
        filename = f"WFS_{self.serial_num}_{mla_name}_{cam_resol_idx}.ref"
        return filename

    @require_take_image
    def save_user_ref(self, backup_dir: str | Path | None = None) -> Path | None:
        """Save user reference file and create a timestamped backup.

        This calls WFS_SaveUserRefFile to save to the DLL-managed location,
        then copies the saved file to the project backup directory with a timestamp.

        Args:
            backup_dir: Directory to store backup. Defaults to 'data/calibration/'.

        Returns:
            Path to the backup file, or None on failure.
        """
        if backup_dir is None:
            backup_dir = Path("data/calibration")
        else:
            backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Call DLL to save reference file to its default location
        if err := self._lib.WFS_SaveUserRefFile(self._instrument_handle):
            self.handle_error(err, no_raise=True)
            return None

        # Get the source file path (where DLL saved it)
        ref_dir = self._get_ref_default_dir()
        ref_filename = self._get_ref_filename()
        src_path = ref_dir / ref_filename

        if not src_path.exists():
            logger.error(f"Reference file not found at expected path: {src_path}")
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

        If backup_path is provided:
        1. Copy the backup file to the DLL's expected location
        2. Then call WFS_LoadUserRefFile to load it

        If backup_path is None:
        - Directly call WFS_LoadUserRefFile to load from DLL's default location

        Args:
            backup_path: Optional path to a backup .ref file to load.

        Returns:
            True on success, False on failure.
        """
        if backup_path is not None:
            backup_path = Path(backup_path)
            if not backup_path.exists():
                logger.error(f"Backup file not found: {backup_path}")
                return False

            # Copy backup to DLL's expected location
            ref_dir = self._get_ref_default_dir()
            ref_dir.mkdir(parents=True, exist_ok=True)
            ref_filename = self._get_ref_filename()
            dst_path = ref_dir / ref_filename

            try:
                shutil.copy2(backup_path, dst_path)
                logger.info(f"Copied backup {backup_path} to {dst_path}")
            except Exception as e:
                logger.error(f"Failed to copy backup file: {e}")
                return False

        # Call DLL to load reference file from its default location
        if err := self._lib.WFS_LoadUserRefFile(self._instrument_handle):
            self.handle_error(err, no_raise=True)
            logger.error("Failed to load user reference file via WFS_LoadUserRefFile")
            return False

        # Also update the reference plane setting
        if err := self._lib.WFS_SetReferencePlane(self._instrument_handle, c_int32(1)):
            self.handle_error(err, no_raise=True)

        self.use_custom_ref = True
        logger.info("User reference file loaded successfully")
        return True

    def optimize_pupil(self):
        """
        This function help to optimize pupil.
        Returns:
            tuple[float, float, float, float]: beam centroid x, beam centroid y, beam diameter x, beam diameter y
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
            n_sample: Number of auto-exposure samples to take (used when exposure_time <= 0)
            dynamicNoiseCut: Enable dynamic noise floor cutoff for spot calculation

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
    def get_spotfiled_image(self, image_loop_counter: int = -1) -> np.ndarray:
        """Retrieve the captured spotfield image from the WFS device.

        Args:
            image_loop_counter: Image loop counter (-1 for latest image)

        Returns:
            np.ndarray: 2D uint8 image array of shape (512, 512)

        Raises:
            RuntimeError: If WFS_GetSpotfieldImageCopy fails
        """
        spots_filed_img = np.empty(MAX_SPOTS, np.uint8)
        if err := self._lib.WFS_GetSpotfieldImageCopy(
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

    @require_take_image
    def get_wavefront(self, cancel_tile: bool = False) -> tuple[np.ndarray, dict]:
        """Calculate wavefront from spot deviations.

        Args:
            cancel_tile: If True, remove tip/tilt from wavefront measurement

        Returns:
            tuple[np.ndarray, dict]: (wavefront array, statistics dict)
                - wavefront: 2D array of wavefront values in waves
                - statistics: dict with keys 'min', 'max', 'diff', 'mean', 'rms', 'wighted_rms'

        Note:
            Uses measured wavefront (type=0) with adaptive pupil compensation if pupil is defined.
        """
        if res := self._lib.WFS_CalcSpotToReferenceDeviations(
            self._instrument_handle, c_int32(1 if cancel_tile else 0)):
            self.handle_error(res)
        adaptive_pupil = 0 if (self.d_x and self.d_y) else 1
        wavefront = np.empty(MAX_SPOTS, dtype=c_float)
        # wavefrontType: 0=Measured Wavefront (direct from spot deviations),
        #                1=Reconstructed Wavefront (requires WFS_CalcReconstrDeviations),
        #                2=Difference (requires WFS_CalcReconstrDeviations)
        # Use 0 (Measured) since it doesn't require reconstruction step
        wavefront_type = 0
        if err := self._lib.WFS_CalcWavefront(
            self._instrument_handle,
            ViInt32(wavefront_type),
            ViInt32(adaptive_pupil),
            wavefront,
        ):
            self.handle_error(err)
        else:
            min, max, diff, mean = c_double(), c_double(), c_double(), c_double()
            rms, wighted_rms = c_double(), c_double()
            self._lib.WFS_CalcWavefrontStatistics(
                self._instrument_handle,
                byref(min),
                byref(max),
                byref(diff),
                byref(mean),
                byref(rms),
                byref(wighted_rms),
            )
            # FIXME: 这个函数的返回值不变
            wavefront = deepcopy(wavefront)[: self.num_spots_x, : self.num_spots_y]

            # wavefront = np.where(wavefront==np.nan, 0, wavefront)
            return wavefront, {
                "min": min.value,
                "max": max.value,
                "diff": diff.value,
                "mean": mean.value,
                "rms": rms.value,
                "wighted_rms": wighted_rms.value if wighted_rms.value > 0 else rms.value
            }
        return wavefront, {
            "min": np.nan,
            "max": np.nan,
            "diff": np.nan,
            "mean": np.nan,
            "rms": np.nan,
            "wighted_rms": np.nan,
        }

    def _remove_tilt(self, wavefront: np.ndarray) -> np.ndarray:
        """Remove tilt (tip/tilt) from wavefront by fitting a plane and subtracting it.

        Args:
            wavefront: 2D wavefront array

        Returns:
            Wavefront with tilt removed
        """
        # Create coordinate grids (normalized to [-1, 1] for numerical stability)
        ny, nx = wavefront.shape
        y, x = np.meshgrid(np.linspace(-1, 1, ny), np.linspace(-1, 1, nx), indexing="ij")

        # Flatten for fitting
        z = wavefront.flatten()

        # Build design matrix for plane: z = a*x + b*y + c
        # Use simple least squares
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
    def calc_n_zernike_terms(n):
        return (n + 1) * (n + 2) // 2 + 1

    @require_take_image
    def get_zernike(self, zernike_order: int = 10) -> np.ndarray:
        """Calculate Zernike polynomial coefficients from spot deviations.

        Args:
            zernike_order: Zernike order (max 10, indexed from 0)

        Returns:
            np.ndarray: Zernike coefficients array

        Raises:
            AssertionError: If zernike_order exceeds 10
        """
        assert zernike_order <= 10, "zernike order must be less than or equal to 10"
        roc_mm = c_double()
        coeff_num = self.calc_n_zernike_terms(zernike_order)
        zernike_order_c = c_int32(zernike_order)
        zernike_um = np.empty((coeff_num,), c_float)
        zernike_orders_rms_um = np.empty((11,), c_float)

        if res := self._lib.WFS_CalcSpotToReferenceDeviations(
            self._instrument_handle, c_int32(0)):
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

    @require_take_image
    def get_spot_deviation(self, cancel_tile: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Get spot deviation from reference positions.

        Args:
            cancel_tile: If True, remove tip/tilt from deviations

        Returns:
            tuple[np.ndarray, np.ndarray]: (deviation_x, deviation_y) arrays
               each of shape (num_spots_x, num_spots_y)
        """
        # FIXME: 这个函数的返回值不变

        _spots_deviation_x = np.empty(MAX_SPOTS, dtype=np.float32)
        _spots_deviation_y = np.empty(MAX_SPOTS, dtype=np.float32)
        # if err:= self._lib.WFS_CalcSpotsCentrDiaIntens(self._instrument_handle, c_int32(1), c_int32(1)):
        #     self.handle_error(err)

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
        for i in range(10):
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

    @property
    def exposure_time(self) -> c_double:
        """Get current exposure time.

        Returns:
            c_double: Exposure time in milliseconds
        """
        actual_exposure = c_double()
        self._lib.WFS_GetExposureTime(self._instrument_handle, actual_exposure)
        return actual_exposure

    @exposure_time.setter
    def exposure_time(self, value: float) -> None:
        """Set exposure time.

        Args:
            value: Exposure time in milliseconds

        Raises:
            AssertionError: If value is outside valid range [0.002, 86] ms
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
            tuple[float, float, float, float]: (centroid_x, centroid_y, diameter_x, diameter_y) in mm
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
            center_and_diameter: (centroid_x, centroid_y, diameter_x, diameter_y) in mm

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
            bool: True if high speed mode is active
        """
        enable_high_speed = self._lib.WFS_CheckHighspeedCentroids(
            self._instrument_handle
        ).value
        return enable_high_speed.value

    @high_speed.setter
    def high_speed(self, enable: bool):
        """Enable or disable high speed mode.

        Args:
            enable: True to enable high speed mode, False to disable

        Note:
            High speed mode only supports 512x512 resolution and requires auto exposure.
            Automatically re-optimizes pupil after enabling.
        """
        """
        instrumentHandle	ViSession	This parameter accepts the Instrument Handle returned by the Init function to select the desired instrument driver session.
        highspeedMode	ViInt32	This parameter determines if the camera's Highspeed Mode is switched on or off.
        adaptCentroids	ViInt32	When Highspeed Mode is selected, this parameter determines if the centroid positions measured in Normal Mode should be used to adapt the spot search windows for Highspeed Mode.
        Otherwise, a rigid grid based on reference spot positions is used in Highspeed Mode.
        substractOffset	ViInt32	This parameter defines an offset level for Highspeed Mode only. All camera pixels will be subtracted by this level before the centroids are being calculated, which increases accuracy.
        Valid range: 0 ... 255
        Note: The offset is only valid in Highspeed Mode and must not set too high to clear the spots within the camera image!
        allowAutoExposure	ViInt32	When Highspeed Mode is selected, this parameter determines if the camera should also calculate the image saturation in order enable the auto exposure feature using function WFS_TakeSpotfieldImageAutoExpos() instead of WFS_TakeSpotfieldImage().
        This option leads to a somewhat reduced measurement speed when enabled.
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

