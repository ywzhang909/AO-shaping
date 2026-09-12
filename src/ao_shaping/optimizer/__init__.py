"""
AO-Shaping Optimizer Package

Exports all public optimizer classes and functions for easy access:
    from ao_shaping.optimizer import X
"""

# Wavefront-based optimizers (RMS optimization via WFS)
from ao_shaping.optimizer.wf.rms import optimizer_rms, schedule_lr_delta
from ao_shaping.optimizer.wf.interaction_matrix import (
    calculate_interaction_matrix,
    load_interaction_matrix,
    apply_interaction_matrix,
)
from ao_shaping.optimizer.wf.zernike_response_matrix import (
    ZernikeResponseMatrixResult,
    calibrate_zernike_response_matrix,
    measure_zernike_mode_response,
    save_zernike_response_matrix,
    load_zernike_response_matrix,
    plot_response_matrix,
)

# Wavefront-sensorless optimizers (PIB optimization via CCD)
from ao_shaping.optimizer.wfless.pib import (
    optimize_pib,
    learning_schedule,
    TabuMemory,
    AdaptiveSearchState,
)
from ao_shaping.optimizer.wfless.sim_spgd import (
    optimize_spgd,
    optimize_spgd_zernike,
    optimize_pso,
    optimize_ga,
    optimize_sa,
)

__all__ = [
    # Wavefront-based (RMS)
    "optimizer_rms",
    "schedule_lr_delta",
    "calculate_interaction_matrix",
    "load_interaction_matrix",
    "apply_interaction_matrix",
    "ZernikeResponseMatrixResult",
    "calibrate_zernike_response_matrix",
    "measure_zernike_mode_response",
    "save_zernike_response_matrix",
    "load_zernike_response_matrix",
    "plot_response_matrix",
    # Wavefront-sensorless (PIB)
    "optimize_pib",
    "learning_schedule",
    "TabuMemory",
    "AdaptiveSearchState",
    "optimize_spgd",
    "optimize_spgd_zernike",
    "optimize_pso",
    "optimize_ga",
    "optimize_sa",
    # Differentiable beam shaping (torch-dependent, lazy-exported)
    "BeamOptimizeResult",
    "differentiable_beam_optimize",
    "far_field_intensity",
    "DifferentiableShapingResult",
    "create_target_mask",
    "train_beam_shaping",
]


def __getattr__(name: str):
    """Lazily export the torch-dependent differentiable-shaping symbols.

    ``differentiable_beam`` imports ``torch`` at module top level, so ``import
    ao_shaping.optimizer`` in a torch-less environment must not touch it until
    the symbol is actually accessed (PEP 562). Resolved once, then cached in
    the module namespace.
    """
    if name in {"BeamOptimizeResult", "differentiable_beam_optimize", "far_field_intensity"}:
        from ao_shaping.optimizer.wfless.differentiable_beam import __dict__ as _beam_dict

        value = _beam_dict[name]
    elif name in {
        "DifferentiableShapingResult",
        "create_target_mask",
        "train_beam_shaping",
    }:
        from ao_shaping.optimizer.wfless.differentiable_shaping import (
            __dict__ as _shaping_dict,
        )

        value = _shaping_dict[name]
    else:
        raise AttributeError(f"module 'ao_shaping.optimizer' has no attribute {name!r}")
    globals()[name] = value
    return value
