"""Route an SLM optimizer module's ``Santec`` binding to the simulated SLM.

Every SLM-Zernike shaping optimizer builds its device through
``Santec.from_params(config.slm)`` resolved at call time from a module-global
``Santec`` name. That indirection is what lets the whole loop run against the
2f-Fourier digital twin instead of hardware, and three separate places were
exploiting it by hand:

* ``runners/slm/gsnet_runner.py::_maybe_sim_patch`` -> ``slm_square_shaping``
* ``scripts/slm_pib_sim_run.py::_patch_santec`` -> ``slm_zernike_pib``
* ``tests/.../test_slm_zernike_objectives_sim.py::_patch_slm`` -> both

The duplication was not benign, because **each site patched a different module
set**. The end-to-end offline test therefore exercised
``slm_zernike_shaping`` while the engine the production callers actually use,
``slm_zernike_pib``, had no offline coverage at all. One function taking the
modules to patch makes the coverage an explicit, checkable argument.

Registering the ``"sim"`` camera type and resetting the shared optical state are
deliberately **not** done here: the callers differ in intent (the
``slm_pib_sim_run`` harness needs a disturbance-aware ``reset_system`` wrapper),
so those stay with the caller.

Layering: this module lives in ``drivers/sim`` and imports nothing from
``optimizer`` — the target modules arrive as arguments. Production optimizer
code never imports it, so the hardware path keeps its lazy, simulation-free
import graph (see ``sim/AGENTS.md`` and the repo rule that hardware packages
must not import simulation packages).
"""

from __future__ import annotations

from types import ModuleType

from ao_shaping.drivers.sim.slm_pib_sim import SimSLMPib


def install_sim_slm(*optimizer_modules: ModuleType) -> None:
    """Point each module's ``Santec`` name at :class:`SimSLMPib`.

    Args:
        *optimizer_modules: Modules whose ``Santec`` global should be
            replaced, e.g. ``slm_zernike_pib``. Every one of them must already
            import ``Santec``; a module that does not is a caller bug.

    Raises:
        AttributeError: A given module has no ``Santec`` global. Without this
            check ``setattr`` would happily *create* the attribute, the patch
            would appear to work, and the module under test would keep opening
            real hardware -- the exact failure this indirection is used to
            avoid. The message names the offending module.
    """
    if not optimizer_modules:
        raise ValueError(
            "install_sim_slm() needs at least one optimizer module to patch; "
            "an empty call would silently leave the real Santec in place"
        )
    for module in optimizer_modules:
        if not hasattr(module, "Santec"):
            raise AttributeError(
                f"{module.__name__!r} has no 'Santec' global to patch. The "
                "optimizer resolves its SLM from that module-level import; "
                "pass the module that performs that import (e.g. "
                "ao_shaping.optimizer.wfless.slm_zernike_pib), not its "
                "transitive dependency."
            )
    for module in optimizer_modules:
        module.Santec = SimSLMPib  # type: ignore[attr-defined]


__all__ = ["SimSLMPib", "install_sim_slm"]