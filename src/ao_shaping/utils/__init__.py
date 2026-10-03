"""Utility functions for AO-Shaping system.

This package is organized into domain subpackages:
- ``io``: File operations, timestamps, config, networking (file, timestamp,
  cli_helpers, device_config, network, handler)
- ``image``: Spot analysis and image helpers (spots_calc, beam_metrics,
  targets, resample, display, hardware_utils). **Visualization/rendering
  lives in ``ao_shaping.display``**, not here (repo anti-pattern).
- ``wavefront``: Wavefront and mode-mixing math (zernike_calc, zernike_utils,
  wavefront_calc, wfs_utils, phase_unwrap, hadamard_calc, matrix_utils,
  pattern_helper, vi)
- ``slm``: SLM state, LUT handling, and phase/display (slm_lut, phase_display)

Legacy top-level paths (``ao_shaping.utils.spots_calc`` etc.) remain importable
via backward-compat shims.

**Import contract (R-20)**: every re-export below resolves lazily through PEP
562. ``import ao_shaping.utils`` must not drag in numpy / matplotlib / aotools (nor
any optional SDK), because this package sits on the import path of *everything* in
the repo -- an import-time failure here is a failure of `import ao_shaping` itself.
Optional third-party dependencies (notably ``aotools``, used only by
``pattern_helper``) are touched when the symbol that needs them is read, not before.
An unknown name still raises ``AttributeError``.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:  # pragma: no cover - for static analysers only
    from ao_shaping.utils.image import display as display
    from ao_shaping.utils.image import spots_calc as spots_calc
    from ao_shaping.utils.io import file as file
    from ao_shaping.utils.io import timestamp as timestamp
    from ao_shaping.utils.io.handler import Register as Register
    from ao_shaping.utils.slm import slm_lut as slm_lut
    from ao_shaping.utils.slm import phase_display as phase_display
    from ao_shaping.utils.wavefront import hadamard_calc as hadamard_calc
    from ao_shaping.utils.wavefront import matrix_utils as matrix_utils
    from ao_shaping.utils.wavefront import pattern_helper as pattern_helper
    from ao_shaping.utils.wavefront import wavefront_calc as wavefront_calc
    from ao_shaping.utils.wavefront import zernike_calc as zernike_calc
    from ao_shaping.utils.image.spots_calc import calculate_sharpness as calculate_sharpness
    from ao_shaping.utils.image.spots_calc import calculate_sharpness_cupy as calculate_sharpness_cupy
    from ao_shaping.utils.image.spots_calc import calculate_sharpness_numba as calculate_sharpness_numba
    from ao_shaping.utils.image.spots_calc import crop as crop
    from ao_shaping.utils.image.spots_calc import crop_cupy as crop_cupy
    from ao_shaping.utils.image.spots_calc import crop_numba as crop_numba
    from ao_shaping.utils.image.spots_calc import center_of_mass_numpy as center_of_mass_numpy
    from ao_shaping.utils.image.spots_calc import center_of_mass_cupy as center_of_mass_cupy
    from ao_shaping.utils.image.spots_calc import center_of_mass_numba as center_of_mass_numba
    from ao_shaping.utils.image.spots_calc import center_of_brightness as center_of_brightness
    from ao_shaping.utils.image.spots_calc import center_of_brightness_cupy as center_of_brightness_cupy
    from ao_shaping.utils.image.spots_calc import center_of_brightness_numba as center_of_brightness_numba
    from ao_shaping.utils.image.spots_calc import diffraction_limit as diffraction_limit
    from ao_shaping.utils.image.spots_calc import jitter_diameter as jitter_diameter
    from ao_shaping.utils.image.spots_calc import centroid as centroid
    from ao_shaping.utils.image.spots_calc import peak_position as peak_position
    from ao_shaping.utils.image.spots_calc import make_coord as make_coord
    from ao_shaping.utils.image.spots_calc import gaussian_waist_radius_four_angles as gaussian_waist_radius_four_angles
    from ao_shaping.utils.image.spots_calc import radius as radius
    from ao_shaping.utils.image.spots_calc import effective_radius as effective_radius
    from ao_shaping.utils.image.spots_calc import power_bucket as power_bucket
    from ao_shaping.utils.image.spots_calc import power_in_bucket_mask as power_in_bucket_mask
    from ao_shaping.utils.image.spots_calc import pib_ratio_mask as pib_ratio_mask
    from ao_shaping.utils.image.spots_calc import disp as disp
    from ao_shaping.utils.io.file import gen_file_path_inc as gen_file_path_inc
    from ao_shaping.utils.io.file import gen_file_path_uuid as gen_file_path_uuid
    from ao_shaping.utils.io.file import gen_date_str as gen_date_str
    from ao_shaping.utils.io.file import gen_date_dir as gen_date_dir
    from ao_shaping.utils.io.file import get_init_V_by_rms as get_init_V_by_rms
    from ao_shaping.utils.io.file import get_init_V_by_energy as get_init_V_by_energy
    from ao_shaping.utils.io.file import save_history as save_history
    from ao_shaping.utils.io.file import Recorder as Recorder
    from ao_shaping.utils.image.display import ImageVoltagesDisplay as ImageVoltagesDisplay
    from ao_shaping.utils.image.display import ZernikeCalibrationDisplay as ZernikeCalibrationDisplay
    from ao_shaping.utils.image.display import plot_funcs as plot_funcs
    from ao_shaping.utils.image.display import VOLT_HEIGHT as VOLT_HEIGHT
    from ao_shaping.utils.image.display import LOG_J_HEIGHT as LOG_J_HEIGHT
    from ao_shaping.utils.image.display import BACKGROUND_COLOR as BACKGROUND_COLOR
    from ao_shaping.utils.image.display import LINE_COLOR as LINE_COLOR
    from ao_shaping.utils.image.display import ZERN_STABLE_COLOR as ZERN_STABLE_COLOR
    from ao_shaping.utils.image.display import ZERN_MODERATE_COLOR as ZERN_MODERATE_COLOR
    from ao_shaping.utils.image.display import ZERN_UNSTABLE_COLOR as ZERN_UNSTABLE_COLOR
    from ao_shaping.utils.image.display import ZERN_BAR_DEFAULT_COLOR as ZERN_BAR_DEFAULT_COLOR
    from ao_shaping.utils.image.display import ZERN_TEXT_COLOR as ZERN_TEXT_COLOR
    from ao_shaping.utils.image.display import ZERN_BG_COLOR as ZERN_BG_COLOR
    from ao_shaping.utils.image.display import ZERN_PROGRESS_BG as ZERN_PROGRESS_BG
    from ao_shaping.utils.image.display import ZERN_PROGRESS_FILL as ZERN_PROGRESS_FILL
    from ao_shaping.utils.io.timestamp import TimestampParser as TimestampParser
    from ao_shaping.utils.io.timestamp import parse_timestamp as parse_timestamp
    from ao_shaping.utils.io.timestamp import sort_by_timestamp as sort_by_timestamp
    from ao_shaping.utils.wavefront.matrix_utils import compute_pinv as compute_pinv
    from ao_shaping.utils.wavefront.matrix_utils import compute_lstsq as compute_lstsq
    from ao_shaping.utils.wavefront.matrix_utils import calc_n_zernike_terms as calc_n_zernike_terms
    from ao_shaping.utils.wavefront.matrix_utils import noll_to_index as noll_to_index
    from ao_shaping.utils.wavefront.matrix_utils import index_to_noll as index_to_noll
    from ao_shaping.utils.wavefront.pattern_helper import PatternHelper as PatternHelper
    from ao_shaping.utils.wavefront.wavefront_calc import normalize_01 as normalize_01
    from ao_shaping.utils.wavefront.wavefront_calc import centroid_calculation as centroid_calculation
    from ao_shaping.utils.wavefront.wavefront_calc import calculate_derotation as calculate_derotation
    from ao_shaping.utils.wavefront.wavefront_calc import get_zernike_base_matrixs as get_zernike_base_matrixs
    from ao_shaping.utils.wavefront.wavefront_calc import to_color as to_color
    from ao_shaping.utils.wavefront.wavefront_calc import ZernikeCentroidCalculator as ZernikeCentroidCalculator
    from ao_shaping.utils.wavefront.hadamard_calc import HadamardGenerator as HadamardGenerator
    from ao_shaping.utils.wavefront.hadamard_calc import calc_n_hadamard_modes as calc_n_hadamard_modes
    from ao_shaping.utils.wavefront.hadamard_calc import hadamard_mode_2d as hadamard_mode_2d
    from ao_shaping.utils.wavefront.hadamard_calc import is_hadamard_order as is_hadamard_order
    from ao_shaping.utils.wavefront.zernike_calc import ZernikeGenerator as ZernikeGenerator
    from ao_shaping.utils.wavefront.zernike_calc import fit_zernike as fit_zernike
    from ao_shaping.utils.wavefront.zernike_calc import zernike_radial as zernike_radial
    from ao_shaping.utils.wavefront.zernike_calc import calc_n_zernike_terms as calc_n_zernike_terms_zern
    from ao_shaping.utils.wavefront.zernike_calc import generate_noll_polynomial as generate_noll_polynomial
    from ao_shaping.utils.io.cli_helpers import parse_tuple as parse_tuple
    from ao_shaping.utils.io.cli_helpers import setup_coredumpy as setup_coredumpy
    from ao_shaping.utils.io.cli_helpers import get_date_dir_name as get_date_dir_name
    from ao_shaping.utils.io.cli_helpers import get_timestamp_str as get_timestamp_str
    from ao_shaping.utils.io.cli_helpers import create_save_dir as create_save_dir


#: Public name -> ``"module:attr"``. Keys are unique; where two
#: source modules exported the same name (``calc_n_zernike_terms`` from both
#: ``matrix_utils`` and ``zernike_calc``) the LAST one wins, matching what the
#: previous sequential ``from ... import`` block resolved to.
_LAZY_EXPORTS: dict[str, str] = {
    'display': 'ao_shaping.utils.image:display',
    'spots_calc': 'ao_shaping.utils.image:spots_calc',
    'file': 'ao_shaping.utils.io:file',
    'timestamp': 'ao_shaping.utils.io:timestamp',
    'Register': 'ao_shaping.utils.io.handler:Register',
    'slm_lut': 'ao_shaping.utils.slm:slm_lut',
    'phase_display': 'ao_shaping.utils.slm:phase_display',
    'hadamard_calc': 'ao_shaping.utils.wavefront:hadamard_calc',
    'matrix_utils': 'ao_shaping.utils.wavefront:matrix_utils',
    'pattern_helper': 'ao_shaping.utils.wavefront:pattern_helper',
    'wavefront_calc': 'ao_shaping.utils.wavefront:wavefront_calc',
    'zernike_calc': 'ao_shaping.utils.wavefront:zernike_calc',
    'calculate_sharpness': 'ao_shaping.utils.image.spots_calc:calculate_sharpness',
    'calculate_sharpness_cupy': 'ao_shaping.utils.image.spots_calc:calculate_sharpness_cupy',
    'calculate_sharpness_numba': 'ao_shaping.utils.image.spots_calc:calculate_sharpness_numba',
    'crop': 'ao_shaping.utils.image.spots_calc:crop',
    'crop_cupy': 'ao_shaping.utils.image.spots_calc:crop_cupy',
    'crop_numba': 'ao_shaping.utils.image.spots_calc:crop_numba',
    'center_of_mass_numpy': 'ao_shaping.utils.image.spots_calc:center_of_mass_numpy',
    'center_of_mass_cupy': 'ao_shaping.utils.image.spots_calc:center_of_mass_cupy',
    'center_of_mass_numba': 'ao_shaping.utils.image.spots_calc:center_of_mass_numba',
    'center_of_brightness': 'ao_shaping.utils.image.spots_calc:center_of_brightness',
    'center_of_brightness_cupy': 'ao_shaping.utils.image.spots_calc:center_of_brightness_cupy',
    'center_of_brightness_numba': 'ao_shaping.utils.image.spots_calc:center_of_brightness_numba',
    'diffraction_limit': 'ao_shaping.utils.image.spots_calc:diffraction_limit',
    'jitter_diameter': 'ao_shaping.utils.image.spots_calc:jitter_diameter',
    'centroid': 'ao_shaping.utils.image.spots_calc:centroid',
    'peak_position': 'ao_shaping.utils.image.spots_calc:peak_position',
    'make_coord': 'ao_shaping.utils.image.spots_calc:make_coord',
    'gaussian_waist_radius_four_angles': 'ao_shaping.utils.image.spots_calc:gaussian_waist_radius_four_angles',
    'radius': 'ao_shaping.utils.image.spots_calc:radius',
    'effective_radius': 'ao_shaping.utils.image.spots_calc:effective_radius',
    'power_bucket': 'ao_shaping.utils.image.spots_calc:power_bucket',
    'power_in_bucket_mask': 'ao_shaping.utils.image.spots_calc:power_in_bucket_mask',
    'pib_ratio_mask': 'ao_shaping.utils.image.spots_calc:pib_ratio_mask',
    'disp': 'ao_shaping.utils.image.spots_calc:disp',
    'gen_file_path_inc': 'ao_shaping.utils.io.file:gen_file_path_inc',
    'gen_file_path_uuid': 'ao_shaping.utils.io.file:gen_file_path_uuid',
    'gen_date_str': 'ao_shaping.utils.io.file:gen_date_str',
    'gen_date_dir': 'ao_shaping.utils.io.file:gen_date_dir',
    'get_init_V_by_rms': 'ao_shaping.utils.io.file:get_init_V_by_rms',
    'get_init_V_by_energy': 'ao_shaping.utils.io.file:get_init_V_by_energy',
    'save_history': 'ao_shaping.utils.io.file:save_history',
    'Recorder': 'ao_shaping.utils.io.file:Recorder',
    'ImageVoltagesDisplay': 'ao_shaping.utils.image.display:ImageVoltagesDisplay',
    'ZernikeCalibrationDisplay': 'ao_shaping.utils.image.display:ZernikeCalibrationDisplay',
    'plot_funcs': 'ao_shaping.utils.image.display:plot_funcs',
    'VOLT_HEIGHT': 'ao_shaping.utils.image.display:VOLT_HEIGHT',
    'LOG_J_HEIGHT': 'ao_shaping.utils.image.display:LOG_J_HEIGHT',
    'BACKGROUND_COLOR': 'ao_shaping.utils.image.display:BACKGROUND_COLOR',
    'LINE_COLOR': 'ao_shaping.utils.image.display:LINE_COLOR',
    'ZERN_STABLE_COLOR': 'ao_shaping.utils.image.display:ZERN_STABLE_COLOR',
    'ZERN_MODERATE_COLOR': 'ao_shaping.utils.image.display:ZERN_MODERATE_COLOR',
    'ZERN_UNSTABLE_COLOR': 'ao_shaping.utils.image.display:ZERN_UNSTABLE_COLOR',
    'ZERN_BAR_DEFAULT_COLOR': 'ao_shaping.utils.image.display:ZERN_BAR_DEFAULT_COLOR',
    'ZERN_TEXT_COLOR': 'ao_shaping.utils.image.display:ZERN_TEXT_COLOR',
    'ZERN_BG_COLOR': 'ao_shaping.utils.image.display:ZERN_BG_COLOR',
    'ZERN_PROGRESS_BG': 'ao_shaping.utils.image.display:ZERN_PROGRESS_BG',
    'ZERN_PROGRESS_FILL': 'ao_shaping.utils.image.display:ZERN_PROGRESS_FILL',
    'TimestampParser': 'ao_shaping.utils.io.timestamp:TimestampParser',
    'parse_timestamp': 'ao_shaping.utils.io.timestamp:parse_timestamp',
    'sort_by_timestamp': 'ao_shaping.utils.io.timestamp:sort_by_timestamp',
    'compute_pinv': 'ao_shaping.utils.wavefront.matrix_utils:compute_pinv',
    'compute_lstsq': 'ao_shaping.utils.wavefront.matrix_utils:compute_lstsq',
    'calc_n_zernike_terms': 'ao_shaping.utils.wavefront.matrix_utils:calc_n_zernike_terms',
    'noll_to_index': 'ao_shaping.utils.wavefront.matrix_utils:noll_to_index',
    'index_to_noll': 'ao_shaping.utils.wavefront.matrix_utils:index_to_noll',
    'PatternHelper': 'ao_shaping.utils.wavefront.pattern_helper:PatternHelper',
    'normalize_01': 'ao_shaping.utils.wavefront.wavefront_calc:normalize_01',
    'centroid_calculation': 'ao_shaping.utils.wavefront.wavefront_calc:centroid_calculation',
    'calculate_derotation': 'ao_shaping.utils.wavefront.wavefront_calc:calculate_derotation',
    'get_zernike_base_matrixs': 'ao_shaping.utils.wavefront.wavefront_calc:get_zernike_base_matrixs',
    'to_color': 'ao_shaping.utils.wavefront.wavefront_calc:to_color',
    'ZernikeCentroidCalculator': 'ao_shaping.utils.wavefront.wavefront_calc:ZernikeCentroidCalculator',
    'HadamardGenerator': 'ao_shaping.utils.wavefront.hadamard_calc:HadamardGenerator',
    'calc_n_hadamard_modes': 'ao_shaping.utils.wavefront.hadamard_calc:calc_n_hadamard_modes',
    'hadamard_mode_2d': 'ao_shaping.utils.wavefront.hadamard_calc:hadamard_mode_2d',
    'is_hadamard_order': 'ao_shaping.utils.wavefront.hadamard_calc:is_hadamard_order',
    'ZernikeGenerator': 'ao_shaping.utils.wavefront.zernike_calc:ZernikeGenerator',
    'fit_zernike': 'ao_shaping.utils.wavefront.zernike_calc:fit_zernike',
    'zernike_radial': 'ao_shaping.utils.wavefront.zernike_calc:zernike_radial',
    'calc_n_zernike_terms_zern': 'ao_shaping.utils.wavefront.zernike_calc:calc_n_zernike_terms',
    'generate_noll_polynomial': 'ao_shaping.utils.wavefront.zernike_calc:generate_noll_polynomial',
    'parse_tuple': 'ao_shaping.utils.io.cli_helpers:parse_tuple',
    'setup_coredumpy': 'ao_shaping.utils.io.cli_helpers:setup_coredumpy',
    'get_date_dir_name': 'ao_shaping.utils.io.cli_helpers:get_date_dir_name',
    'get_timestamp_str': 'ao_shaping.utils.io.cli_helpers:get_timestamp_str',
    'create_save_dir': 'ao_shaping.utils.io.cli_helpers:create_save_dir',
}


def __getattr__(name: str) -> Any:
    """Resolve a public name on first access (PEP 562)."""
    try:
        target = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
    module_name, _, attr = target.partition(":")
    value = getattr(import_module(module_name), attr)
    # Cache into module globals so later lookups skip __getattr__ entirely.
    # Required: without the cache a subsequent `from ao_shaping.utils import x`
    # would re-enter this path and re-run the import machinery.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """The full public surface, including names not yet resolved."""
    return sorted(set(__all__) | set(_LAZY_EXPORTS) | set(globals()))


__all__ = [
    'HadamardGenerator',
    'calc_n_hadamard_modes',
    'hadamard_mode_2d',
    'is_hadamard_order',
    'BACKGROUND_COLOR',
    'LINE_COLOR',
    'LOG_J_HEIGHT',
    'VOLT_HEIGHT',
    'ZERN_BAR_DEFAULT_COLOR',
    'ZERN_BG_COLOR',
    'ZERN_MODERATE_COLOR',
    'ZERN_PROGRESS_BG',
    'ZERN_PROGRESS_FILL',
    'ZERN_STABLE_COLOR',
    'ZERN_TEXT_COLOR',
    'ZERN_UNSTABLE_COLOR',
    'ImageVoltagesDisplay',
    'PatternHelper',
    'Recorder',
    'Register',
    'TimestampParser',
    'ZernikeCalibrationDisplay',
    'ZernikeCentroidCalculator',
    'ZernikeGenerator',
    'calc_n_zernike_terms',
    'calc_n_zernike_terms_zern',
    'calculate_derotation',
    'calculate_sharpness',
    'calculate_sharpness_cupy',
    'calculate_sharpness_numba',
    'center_of_brightness',
    'center_of_brightness_cupy',
    'center_of_brightness_numba',
    'center_of_mass_cupy',
    'center_of_mass_numba',
    'center_of_mass_numpy',
    'centroid',
    'centroid_calculation',
    'compute_lstsq',
    'compute_pinv',
    'configure_error_logging',
    'crop',
    'crop_cupy',
    'crop_numba',
    'diffraction_limit',
    'disp',
    'display',
    'effective_radius',
    'file',
    'fit_zernike',
    'gen_date_dir',
    'gen_date_str',
    'gen_file_path_inc',
    'gen_file_path_uuid',
    'generate_noll_polynomial',
    'get_init_V_by_energy',
    'get_init_V_by_rms',
    'get_zernike_base_matrixs',
    'index_to_noll',
    'jitter_diameter',
    'make_coord',
    'matrix_utils',
    'noll_to_index',
    'normalize_01',
    'parse_timestamp',
    'pattern_helper',
    'peak_position',
    'plot_funcs',
    'power_bucket',
    'power_in_bucket_mask',
    'pib_ratio_mask',
    'radius',
    'save_history',
    'sort_by_timestamp',
    'spots_calc',
    'timestamp',
    'to_color',
    'wavefront_calc',
    'zernike_calc',
    'zernike_radial',
    'parse_tuple',
    'setup_coredumpy',
    'get_date_dir_name',
    'get_timestamp_str',
    'create_save_dir',
]


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
