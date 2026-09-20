"""Wavefront computation utilities: Zernike modes, wavefront reconstruction,
phase unwrapping, Hadamard modes, matrix ops and mode-mixing helpers.

Moved from the flat ``ao_shaping.utils`` package (refactor D1-W2).
"""

from __future__ import annotations

from ao_shaping.utils.wavefront.pattern_helper import (
    PatternHelper,
    PhaseUnwrapperHelper,
    PhaseWrapOptimizerHelper,
    calc_blazed_grating_period,
)
from ao_shaping.utils.wavefront.hadamard_calc import (
    HadamardGenerator,
    calc_n_hadamard_modes,
    hadamard_mode_2d,
    is_hadamard_order,
)
from ao_shaping.utils.wavefront.matrix_utils import (
    compute_pinv,
    compute_lstsq,
    calc_n_zernike_terms,
    noll_to_index,
    index_to_noll,
)
from ao_shaping.utils.wavefront.phase_unwrap import (
    PhaseUnwrapper,
    UnwrapStrategy,
    unwrap_phase,
)
from ao_shaping.utils.wavefront.wavefront_calc import (
    normalize_01,
    centroid_calculation,
    calculate_derotation,
    get_zernike_base_matrixs,
    to_color,
    ZernikeCentroidCalculator,
)
from ao_shaping.utils.wavefront.zernike_calc import (
    ZernikeGenerator,
    fit_zernike,
    zernike_radial,
    calc_n_zernike_terms as calc_n_zernike_terms_zern,
    generate_noll_polynomial,
)
from ao_shaping.utils.wavefront.zernike_calc import (
    get_zernike_name,
    zernike_modes,
)


__all__ = [
    # pattern_helper
    "PatternHelper",
    "PhaseUnwrapperHelper",
    "PhaseWrapOptimizerHelper",
    "calc_blazed_grating_period",
    # hadamard_calc
    "HadamardGenerator",
    "calc_n_hadamard_modes",
    "hadamard_mode_2d",
    "is_hadamard_order",
    # matrix_utils
    "compute_pinv",
    "compute_lstsq",
    "calc_n_zernike_terms",
    "noll_to_index",
    "index_to_noll",
    # phase_unwrap
    "PhaseUnwrapper",
    "UnwrapStrategy",
    "unwrap_phase",
    # wavefront_calc
    "normalize_01",
    "centroid_calculation",
    "calculate_derotation",
    "get_zernike_base_matrixs",
    "to_color",
    "ZernikeCentroidCalculator",
    # zernike_calc
    "ZernikeGenerator",
    "fit_zernike",
    "zernike_radial",
    "calc_n_zernike_terms_zern",
    "generate_noll_polynomial",
    # wfs_utils
    "get_zernike_name",
    "zernike_modes",
]
