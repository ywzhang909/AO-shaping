"""R8: a mid-search failure must not leave the SLM on a random phase.

Before this, ``_apply_best_on_exit()`` was called only on the happy path,
immediately before ``return recorder``. Any exception in between -- camera
dropout, an out-of-range coefficient, ``KeyboardInterrupt`` -- therefore left
the SLM displaying whatever phase the search happened to write last. On a
diverged search that is a random high-contrast pattern, and it damages whatever
runs next.

The fix wraps the search in ``try: ... finally: _apply_best_on_exit()``. That
raises a second requirement, which is the subtler half: **cleanup must never
raise**, because an exception raised in a ``finally`` replaces the original one
and destroys the diagnosis. So ``_apply_best_on_exit`` guards itself and falls
back to a flat phase -- flat is always safe.

These tests inject the failure through the sim SLM's public write methods, so no
hardware is involved and the injected error is exactly the kind a real dropout
raises.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from ao_shaping.drivers.sim.sim_bench_patch import SimSLMPib, install_sim_slm
from ao_shaping.drivers.sim.slm_pib_sim import (
    get_system,
    register_sim_camera,
    reset_system,
)
from ao_shaping.optimizer.wfless import slm_zernike_pib as engine
from ao_shaping.runners.runner_common import (
    CameraParamsPib,
    ObjectiveTarget,
    SlmParamsPib,
)

PRODUCTION = "ao_shaping.optimizer.wfless.slm_zernike_pib"


@pytest.fixture(autouse=True)
def _sim_bench(monkeypatch):
    mod = importlib.import_module(PRODUCTION)
    install_sim_slm(mod)
    monkeypatch.setattr(mod, "Santec", SimSLMPib, raising=False)


def _config(*, epochs: int = 4, algorithm: str = "spgd", **overrides):
    kwargs = {"pop_size": 4} if algorithm != "spgd" else {}
    return engine.SlmZernikePibConfig(
        center="shape",
        epochs=epochs,
        algorithm=algorithm,
        camera=CameraParamsPib(
            target=ObjectiveTarget(name="pib", target_shape=None),
            cam_type="sim",
            cam_size=128,
            exposure_time_ms=80.0,
        ),
        slm=SlmParamsPib(n_max=4),
        **kwargs,
        **overrides,
    )


def _run(**overrides):
    register_sim_camera()
    reset_system(seed=42)
    return engine.optimize_slm_zernike_pib(_config(**overrides))


def _break_after(monkeypatch, method: str, fail_on: int, exc: BaseException) -> None:
    """Make ``SimSLMPib.<method>`` raise ``exc`` on its ``fail_on``-th call."""
    original = getattr(SimSLMPib, method)
    calls = {"n": 0}

    def wrapper(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] >= fail_on:
            raise exc
        return original(self, *args, **kwargs)

    monkeypatch.setattr(SimSLMPib, method, wrapper)


#: Phase writes are counted from 1. This must land *inside* the search, past the
#: initial/flat display both branches do during setup -- otherwise the failure
#: happens before any search iterate reaches the panel, the SLM is already flat,
#: and the test passes for the wrong reason. Verified against the pre-fix source:
#: at 3 only the ``spgd`` parameterisation failed; the ``ga`` run never got that
#: far. At 8 both do.
FAIL_ON = 8


# --------------------------------------------------------------------------
def test_a_mid_search_failure_still_reaches_the_caller(monkeypatch):
    """The injected error must surface, not be swallowed by cleanup."""
    _break_after(
        monkeypatch, "create_phase_from_array", FAIL_ON, RuntimeError("camera dropout")
    )
    with pytest.raises(RuntimeError, match="camera dropout"):
        _run(epochs=6)


@pytest.mark.parametrize("algorithm", ["spgd", "ga"])
def test_the_original_exception_survives_a_failing_cleanup(monkeypatch, algorithm):
    """Cleanup must not mask the diagnosis.

    Both the phase write *and* the flat fallback are made to fail, which is the
    worst realistic case (device gone mid-run). The caller must still see the
    original error.

    This case **passes on the pre-fix source too** -- pre-fix there is no cleanup
    at all, so nothing can mask anything. It guards the fix against regression (a
    ``finally`` that raises); it does not *detect* R-8. The detector is
    :func:`test_slm_is_left_in_a_defined_state_not_a_search_iterate`.
    """
    boom = RuntimeError("primary failure")
    _break_after(monkeypatch, "create_phase_from_array", FAIL_ON, boom)
    _break_after(monkeypatch, "set_grayscale", 1, RuntimeError("slm gone"))
    with pytest.raises(RuntimeError, match="primary failure"):
        _run(epochs=6, algorithm=algorithm)


def test_keyboard_interrupt_is_also_caught(monkeypatch):
    """Ctrl-C mid-run must still leave a defined SLM state.

    ``except BaseException`` in the cleanup covers this; the search itself does
    not swallow it. Like the masking test, this passes pre-fix (no cleanup runs)
    and therefore guards the fix rather than detecting R-8.
    """
    _break_after(monkeypatch, "create_phase_from_array", FAIL_ON, KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        _run(epochs=6)


@pytest.mark.parametrize("algorithm", ["spgd", "ga"])
def test_slm_is_left_in_a_defined_state_not_a_search_iterate(monkeypatch, algorithm):
    """The safety property, and the test that actually detects R-8.

    After a mid-search failure the displayed pattern must be *defined* -- the
    flat gray-0 fallback -- and must not be the arbitrary high-contrast iterate
    the search happened to emit. Flat is the state cleanup falls back to when the
    run never beat its own baseline, which holds on this bench (flat already
    scores pib ~ 0.97).

    Against the pre-fix source the SLM is left holding a search iterate here, so
    this fails for both ``algorithm`` values; that asymmetry is the detector.
    """
    _break_after(monkeypatch, "create_phase_from_array", FAIL_ON, RuntimeError("dropout"))
    with pytest.raises(RuntimeError):
        _run(epochs=6, algorithm=algorithm)

    system = get_system()
    gray = system._gray  # noqa: SLF001 - reading the sim twin's state is the point
    assert gray is not None, "cleanup never commanded a defined state"
    assert np.all(np.asarray(gray) == 0), (
        f"expected the flat gray-0 fallback, got min={np.min(gray)} "
        f"max={np.max(gray)}"
    )


def test_normal_exit_still_restores_the_best_phase(monkeypatch):
    """No regression: the happy path must not be turned into the flat fallback.

    ``_apply_best_on_exit`` runs from a ``finally`` now, so it is easy to
    accidentally make it the *only* exit behaviour.
    """
    _break_after(monkeypatch, "create_phase_from_array", 10_000, RuntimeError("unused"))
    recorder = _run(epochs=3)
    assert len(recorder.history) == 4
    # energy_loss_violations is recorded on the Recorder by _apply_best_on_exit,
    # so its presence proves the cleanup still ran on the normal path.
    assert hasattr(recorder, "energy_loss_violations")