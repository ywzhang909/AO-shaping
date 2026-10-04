"""Tests for :mod:`ao_shaping.drivers.sim.sim_bench_patch`.

The helper replaces an SLM optimizer module's module-global ``Santec`` with the
simulated SLM. That indirection is the only reason the whole shaping loop can
run without hardware, which makes two failure modes worth pinning:

* **a silently unpatched module** — ``setattr`` on a module always succeeds, so
  a wrong module name would look like it worked while the code under test kept
  opening real hardware. Hence the explicit ``hasattr`` pre-check;
* **an empty call** — ``install_sim_slm()`` with no arguments must not be a
  no-op, for the same reason.
"""

from __future__ import annotations

import types

import pytest

from ao_shaping.drivers.sim.sim_bench_patch import SimSLMPib, install_sim_slm


def _module_with_santec(name: str = "fake_optimizer") -> types.ModuleType:
    module = types.ModuleType(name)
    module.Santec = object()  # stand-in for the real driver class
    return module


def test_replaces_the_santec_binding():
    module = _module_with_santec()
    assert module.Santec is not SimSLMPib
    install_sim_slm(module)
    assert module.Santec is SimSLMPib


def test_patches_every_module_given():
    a, b, c = (_module_with_santec(n) for n in ("a", "b", "c"))
    install_sim_slm(a, b, c)
    assert a.Santec is SimSLMPib
    assert b.Santec is SimSLMPib
    assert c.Santec is SimSLMPib


def test_is_idempotent():
    module = _module_with_santec()
    install_sim_slm(module)
    install_sim_slm(module)
    assert module.Santec is SimSLMPib


def test_rejects_a_module_without_a_santec_global():
    """The hazard: setattr would invent the attribute and the patch would lie."""
    module = types.ModuleType("no_santec_here")
    with pytest.raises(AttributeError) as exc:
        install_sim_slm(module)
    assert "no_santec_here" in str(exc.value)
    assert not hasattr(module, "Santec"), "the failed patch must not set anything"


def test_rejects_an_empty_call():
    with pytest.raises(ValueError, match="at least one"):
        install_sim_slm()


def test_the_real_optimizer_modules_expose_santec():
    """Every engine the callers pass must actually have the global.

    This is the check that would have caught the original asymmetry, where the
    end-to-end offline test covered ``slm_zernike_shaping`` while the engine
    production actually calls, ``slm_zernike_pib``, was never driven offline.
    """
    import importlib

    for name in (
        "ao_shaping.optimizer.wfless.slm_zernike_pib",
        "ao_shaping.optimizer.wfless.slm_zernike_shaping",
        "ao_shaping.optimizer.wfless.slm_square_shaping",
    ):
        module = importlib.import_module(name)
        assert hasattr(module, "Santec"), f"{name} does not expose Santec"


def test_helper_does_not_import_the_optimizer():
    """Layering: hardware must never pull in simulation, so this module must
    not reach back into ``optimizer``. It only touches modules handed to it."""
    from ao_shaping.drivers.sim import sim_bench_patch

    source = sim_bench_patch.__doc__ or ""
    assert "Layering" in source
    # No module-level optimizer import and no deferred one inside the function.
    assert "import ao_shaping.optimizer" not in _source_of(sim_bench_patch)


def _source_of(module: types.ModuleType) -> str:
    import inspect

    return inspect.getsource(module)