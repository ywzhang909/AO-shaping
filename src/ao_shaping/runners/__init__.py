"""Runners package for AO-Shaping CLI commands.

Lazy imports via module-level ``__getattr__`` so that runner modules are not
pulled into ``sys.modules`` at package import time. This lets every runner be
executed directly with ``python -m ao_shaping.runners.<name>`` without the
``RuntimeWarning: '<name>' found in sys.modules after import of package
'ao_shaping.runners'`` that eager imports trigger.
"""

from __future__ import annotations

from typing import Callable

__all__ = [
    "wf_run",
    "pib_run",
    "pipeline_run",
    "gs_run",
    "zernike_matrix_run",
    "zernike_closed_loop_run",
    "rms_zernike_run",
    "ga_zernike_run",
    "greedy_zernike_run",
    "dm_matrix_run",
    "alt_voltage_run",
    "full_voltage_run",
    "hadamard_matrix_run",
    "combined_run",
    "slm_square_run",
    "slm_pib_run",
]

_LAZY_RUNNERS: dict[str, str] = {
    "wf_run": "ao_shaping.runners.nlight_dm.wf_runner",
    "pib_run": "ao_shaping.runners.nlight_dm.axis_beam_runner",
    "pipeline_run": "ao_shaping.runners.nlight_dm.pipeline_runner",
    "zernike_matrix_run": "ao_shaping.runners.slm.zernike_matrix_runner",
    "zernike_closed_loop_run": "ao_shaping.runners.slm.zernike_matrix_runner",
    "rms_zernike_run": "ao_shaping.runners.slm.rms_zernike_runner",
    "ga_zernike_run": "ao_shaping.runners.ga_zernike_runner",
    "greedy_zernike_run": "ao_shaping.runners.greedy_zernike_runner",
    "dm_matrix_run": "ao_shaping.runners.dm_matrix_runner",
    "alt_voltage_run": "ao_shaping.runners.micro_drive.alt_voltage_runner",
    "full_voltage_run": "ao_shaping.runners.micro_drive.full_voltage_runner",
    "hadamard_matrix_run": "ao_shaping.runners.hadamard_matrix_runner",
    "combined_run": "ao_shaping.runners.nlight_dm.combined_runner",
    "slm_square_run": "ao_shaping.runners.slm_square_runner",
    "slm_pib_run": "ao_shaping.runners.slm_pib_runner",
}

_IMPORT_ATTRS: dict[str, str] = {
    "zernike_matrix_run": "run",
    "zernike_closed_loop_run": "closed_loop_run",
    "slm_square_run": "run",
    "slm_pib_run": "run",
}


def __getattr__(name: str) -> Callable:
    import importlib

    if name not in _LAZY_RUNNERS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path = _LAZY_RUNNERS[name]
    attr = _IMPORT_ATTRS.get(name, "run")
    module = importlib.import_module(module_path)
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
