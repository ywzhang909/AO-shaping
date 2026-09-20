"""Utility functions for AO-Shaping system.

This package is organized into domain subpackages:
- ``io``: File operations, timestamps, config, networking (file, timestamp,
  cli_helpers, device_config, network, handler)
- ``image``: Spot analysis and visualization (spots_calc, beam_metrics,
  targets, resample, display, gs_visualization, hardware_utils)
- ``wavefront``: Wavefront and mode-mixing math (zernike_calc, zernike_utils,
  wavefront_calc, wfs_utils, phase_unwrap, hadamard_calc, matrix_utils,
  pattern_helper, vi)
- ``slm``: SLM state, LUT handling, and phase/display (slm_lut, phase_display)

Legacy top-level paths (``ao_shaping.utils.spots_calc`` etc.) remain importable
via backward-compat shims.
"""

from ao_shaping.utils.image import display, spots_calc
from ao_shaping.utils.io import file, timestamp
from ao_shaping.utils.io.handler import Register
from ao_shaping.utils.slm import slm_lut, phase_display
from ao_shaping.utils.wavefront import (
    hadamard_calc,
    matrix_utils,
    pattern_helper,
    wavefront_calc,
    zernike_calc,
)


# Explicit exports from spots_calc
from ao_shaping.utils.image.spots_calc import (
    calculate_sharpness,
    calculate_sharpness_cupy,
    calculate_sharpness_numba,
    crop,
    crop_cupy,
    crop_numba,
    center_of_mass_numpy,
    center_of_mass_cupy,
    center_of_mass_numba,
    center_of_brightness,
    center_of_brightness_cupy,
    center_of_brightness_numba,
    diffraction_limit,
    jitter_diameter,
    centroid,
    peak_position,
    make_coord,
    gaussian_waist_radius_four_angles,
    radius,
    effective_radius,
    power_bucket,
    power_in_bucket_mask,
    pib_ratio_mask,
    disp,
)

# Explicit exports from file
from ao_shaping.utils.io.file import (
    gen_file_path_inc,
    gen_file_path_uuid,
    gen_date_str,
    gen_date_dir,
    get_init_V_by_rms,
    get_init_V_by_energy,
    save_history,
    Recorder,
)

# Explicit exports from display
from ao_shaping.utils.image.display import (
    ImageVoltagesDisplay,
    ZernikeCalibrationDisplay,
    plot_funcs,
    # Constants
    VOLT_HEIGHT,
    LOG_J_HEIGHT,
    BACKGROUND_COLOR,
    LINE_COLOR,
    ZERN_STABLE_COLOR,
    ZERN_MODERATE_COLOR,
    ZERN_UNSTABLE_COLOR,
    ZERN_BAR_DEFAULT_COLOR,
    ZERN_TEXT_COLOR,
    ZERN_BG_COLOR,
    ZERN_PROGRESS_BG,
    ZERN_PROGRESS_FILL,
)

# Explicit exports from timestamp
from ao_shaping.utils.io.timestamp import (
    TimestampParser,
    parse_timestamp,
    sort_by_timestamp,
)

# Explicit exports from matrix_utils
from ao_shaping.utils.wavefront.matrix_utils import (
    compute_pinv,
    compute_lstsq,
    calc_n_zernike_terms,
    noll_to_index,
    index_to_noll,
)

# Explicit exports from pattern_helper
from ao_shaping.utils.wavefront.pattern_helper import PatternHelper

# Explicit exports from wavefront_calc
from ao_shaping.utils.wavefront.wavefront_calc import (
    normalize_01,
    centroid_calculation,
    calculate_derotation,
    get_zernike_base_matrixs,
    to_color,
    ZernikeCentroidCalculator,
)

# Explicit exports from hadamard_calc
from ao_shaping.utils.wavefront.hadamard_calc import (
    HadamardGenerator,
    calc_n_hadamard_modes,
    hadamard_mode_2d,
    is_hadamard_order,
)

# Explicit exports from zernike_calc
from ao_shaping.utils.wavefront.zernike_calc import (
    ZernikeGenerator,
    fit_zernike,
    zernike_radial,
    calc_n_zernike_terms as calc_n_zernike_terms_zern,
    generate_noll_polynomial,
)

# Explicit exports from cli_helpers
from ao_shaping.utils.io.cli_helpers import (
    parse_tuple,
    setup_coredumpy,
    get_date_dir_name,
    get_timestamp_str,
    create_save_dir,
)

# Define public API
__all__ = [
    # hadamard_calc
    "HadamardGenerator",
    "calc_n_hadamard_modes",
    "hadamard_mode_2d",
    "is_hadamard_order",
    "BACKGROUND_COLOR",
    "LINE_COLOR",
    "LOG_J_HEIGHT",
    "VOLT_HEIGHT",
    "ZERN_BAR_DEFAULT_COLOR",
    "ZERN_BG_COLOR",
    "ZERN_MODERATE_COLOR",
    "ZERN_PROGRESS_BG",
    "ZERN_PROGRESS_FILL",
    "ZERN_STABLE_COLOR",
    "ZERN_TEXT_COLOR",
    "ZERN_UNSTABLE_COLOR",
    # display
    "ImageVoltagesDisplay",
    # pattern_helper
    "PatternHelper",
    "Recorder",
    "Register",
    # timestamp
    "TimestampParser",
    "ZernikeCalibrationDisplay",
    "ZernikeCentroidCalculator",
    # zernike_calc
    "ZernikeGenerator",
    "calc_n_zernike_terms",
    "calc_n_zernike_terms_zern",
    "calculate_derotation",
    # spots_calc
    "calculate_sharpness",
    "calculate_sharpness_cupy",
    "calculate_sharpness_numba",
    "center_of_brightness",
    "center_of_brightness_cupy",
    "center_of_brightness_numba",
    "center_of_mass_cupy",
    "center_of_mass_numba",
    "center_of_mass_numpy",
    "centroid",
    "centroid_calculation",
    "compute_lstsq",
    # matrix_utils
    "compute_pinv",
    # Error handler
    "configure_error_logging",
    "crop",
    "crop_cupy",
    "crop_numba",
    "diffraction_limit",
    "disp",
    "display",
    "effective_radius",
    "file",
    "fit_zernike",
    "gen_date_dir",
    "gen_date_str",
    # file
    "gen_file_path_inc",
    "gen_file_path_uuid",
    "generate_noll_polynomial",
    "get_init_V_by_energy",
    "get_init_V_by_rms",
    "get_zernike_base_matrixs",
    "index_to_noll",
    "jitter_diameter",
    "make_coord",
    "matrix_utils",
    "noll_to_index",
    # wavefront_calc
    "normalize_01",
    "parse_timestamp",
    "pattern_helper",
    "peak_position",
    "plot_funcs",
    "power_bucket",
    "power_in_bucket_mask",
    "pib_ratio_mask",
    "radius",
    "save_history",
    "sort_by_timestamp",
    # Modules (aliases)
    "spots_calc",
    "timestamp",
    "to_color",
    "wavefront_calc",
    "zernike_calc",
    "zernike_radial",
    # cli_helpers
    "parse_tuple",
    "setup_coredumpy",
    "get_date_dir_name",
    "get_timestamp_str",
    "create_save_dir",
]

from loguru import logger


def configure_error_logging(
    log_file: str = "logs/error.log",
    rotation: str = "500 MB",
    level: str = "ERROR",
) -> int:
    """Configure error logging to file.

    This function should be called explicitly if error logging is needed.
    It is NOT called automatically on import to avoid side effects.

    Args:
        log_file: Path to log file.
        rotation: Log rotation policy.
        level: Log level.

    Returns:
        Handler ID for removal.
    """
    return logger.add(
        log_file,
        rotation=rotation,
        encoding="utf-8",
        level=level,
        backtrace=True,
        diagnose=True,
    )
