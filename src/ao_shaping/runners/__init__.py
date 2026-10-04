"""Runners package for AO-Shaping CLI commands.

Lazy imports via module-level ``__getattr__`` so that runner modules are not
pull into ``sys.modules`` at package import time. This lets every runner be
executed directly with ``python -m ao_shaping.runners.<name>`` without the
``RuntimeWarning: '<name>' found in sys.modules after import of package
'ao_shaping.runners'`` that eager imports trigger.

Until R-27 this module carried fifteen eager ``from ... import run as ...``
statements above the ``__getattr__``. That made every name resolvable before
``__getattr__`` could ever be consulted, so the lazy path was unreachable dead
code and the warning above was live on every ``python -m`` invocation. The
eager block is gone; ``_LAZY_RUNNERS`` and ``_IMPORT_ATTRS`` are the single
source of truth, and
``tests/ao_shaping/runners/test_runners_package_lazy.py`` asserts that
``__all__`` and the map cannot drift apart -- which is precisely what let the two
lists diverge unnoticed before.

Two modules export more than one command, so ``_IMPORT_ATTRS`` is load-bearing
rather than decorative:

* ``slm/zernike_matrix_runner.py`` -> ``zernike_matrix_run`` / ``zernike_closed_loop_run``
* ``micro_drive/voltage_runner.py`` -> ``alt_voltage_run`` / ``full_voltage_run``

Forgetting the second name in each pair hands back the *wrong command* instead
of raising, because ``run`` resolves happily in both modules. When adding a
second command to a module, add the sibling to ``_IMPORT_ATTRS`` in the same
commit.
"""

from __future__ import annotations

from typing import Callable

__all__ = [
    "wf_run",
    "pib_run",
    "pipeline_run",
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
    "slm_gs_refine_run",
    "slm_gsnet_run",
    "slm_model_in_loop_run",
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
    # Both commands live in one module, so they need distinct attributes --
    # see the _IMPORT_ATTRS note below.
    "alt_voltage_run": "ao_shaping.runners.micro_drive.voltage_runner",
    "full_voltage_run": "ao_shaping.runners.micro_drive.voltage_runner",
    "hadamard_matrix_run": "ao_shaping.runners.hadamard_matrix_runner",
    "combined_run": "ao_shaping.runners.nlight_dm.combined_runner",
    "slm_square_run": "ao_shaping.runners.slm_square_runner",
    "slm_pib_run": "ao_shaping.runners.slm_pib_runner",
    "slm_gs_refine_run": "ao_shaping.runners.slm_gs_refine_runner",
    # Was in neither __all__ nor _LAZY_RUNNERS. It resolved only because of the
    # eager imports this module used to carry, so main.py's `slm-gsnet`
    # registration depended on an accident rather than on a declaration.
    "slm_gsnet_run": "ao_shaping.runners.slm_gsnet_runner",
    "slm_model_in_loop_run": "ao_shaping.runners.slm_model_in_loop_runner",
}

#: Override the attribute name only when a module exposes more than one command.
#: ``run`` is the default everywhere else.
_IMPORT_ATTRS: dict[str, str] = {
    "zernike_matrix_run": "run",
    "zernike_closed_loop_run": "closed_loop_run",
    "slm_square_run": "run",
    "slm_pib_run": "run",
    "slm_gs_refine_run": "run",
    # Names that are shared with a sibling export must be listed explicitly.
    # Falling through to the "run" default would hand back the wrong command.
    "wf_run": "run",
    "pib_run": "run",
    "pipeline_run": "run",
    "rms_zernike_run": "run",
    "ga_zernike_run": "run",
    "greedy_zernike_run": "run",
    "dm_matrix_run": "run",
    "alt_voltage_run": "run",
    "full_voltage_run": "full_voltage_run",
    "hadamard_matrix_run": "run",
    "combined_run": "run",
    "slm_gsnet_run": "run",
    "slm_model_in_loop_run": "run",
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
    return sorted(set(globals()) | set(__all__))