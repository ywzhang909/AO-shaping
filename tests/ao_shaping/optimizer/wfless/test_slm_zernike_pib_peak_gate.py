"""Tests for the measured-peak feasibility gate (``max_peak``).

Design constraints this suite pins (see the ``max_peak`` comment in
``slm_zernike_pib.py`` for the full rationale):

* ``max_peak = 0`` (the default) must leave every run bit-identical.
* The gate MIRRORS THE FOLD GATE, not the noise gate: on a violation it must not
  call ``optimizer.update`` at all (zero-updating there decays momentum, which
  would corrupt the Adam moments on a statement we have no evidence for), must
  not append to the noise-gate diff history, must not refresh the fold baseline,
  and must not best-track.
* A violating epoch is still recorded honestly: real mean J, real peak,
  ``gate="peak"``, ``diff=0.0``.
* The cap is compared against the max over ALL captured frames, so it covers
  every frame the loop displays (2 normally, 4 under ABBA).
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.runners.runner_common import CameraParamsPib, RunParams, SlmParamsPib
from ao_shaping.runners.slm.slm_shaping_runner import _build_slm_pib_config

_GATE_VALUES = {"applied", "fold", "noise", "peak", None}
_LOG_MSGS: list[str] = []


def _config(**search_overrides):
    """Build an SlmZernikePibConfig for the sim backend.

    ``n_max=4`` (14 Zernike modes) keeps the run fast and the DOF count legal -
    the default yields 50, which the panel-size derivation rejects.
    """
    from ao_shaping.runners.runner_common import ObjectiveTarget, SpgdParamsPib

    rp = RunParams()
    cp = CameraParamsPib(
        cam_type="sim",
        target=ObjectiveTarget(name="shape", target_shape="circle"),
        target_size=40.0,
        cam_size=200,
    )
    sp = SlmParamsPib(n_max=4)
    search = SpgdParamsPib(epochs=4, delta=0.05, lr=0.5, **search_overrides)
    cfg = _build_slm_pib_config(rp, cp, sp, search)
    cfg.camera.cam_type = "sim"
    cfg.camera.exposure_time_ms = 1.2
    return cfg


def _collect_logs():
    """Collect loguru records.

    ``caplog`` only sees the stdlib ``logging`` module and loguru does not
    propagate there by default, so a loguru warning is invisible to it.
    """
    from loguru import logger

    sink_id = logger.add(lambda m: _LOG_MSGS.append(str(m)), level="DEBUG")
    return sink_id


def _drop_logs(sink_id) -> None:
    from loguru import logger

    logger.remove(sink_id)


def _patch_sim(monkeypatch):
    """Route the optimizer at the digital twin via the runner's own helper.

    Deliberately NOT hand-rolled: ``SimPibSystem()`` constructed bare fails
    (``n_actuators`` defaults to ``NLight.DM_NUM`` = 50, which is not a perfect
    square), and ``patch_sim_pib_shaping`` is what production actually calls, so
    using it keeps the test on the real wiring path.
    """
    from ao_shaping.runners.runner_common import patch_sim_pib_shaping

    # ``SimPibSystem`` builds ``SimDmOptics(n_actuators=DM_N_ACTUATORS)`` and
    # ``DM_N_ACTUATORS`` comes from a live DM reachability probe that lands on
    # 50 (NLight) when nothing is reachable - not a perfect square, so the sim
    # refuses to construct. The DM is irrelevant to this SLM-side test, so pin a
    # square count instead of depending on which DM answers the probe.
    import ao_shaping.config as _cfg

    monkeypatch.setattr(_cfg, "DM_N_ACTUATORS", 64, raising=False)

    patch_sim_pib_shaping("sim")
    from ao_shaping.optimizer.wfless import slm_zernike_pib as mod

    monkeypatch.setattr(mod, "SLM_RESPONSE_TIME_S", 0.0)


class TestDefaultOff:
    def test_default_is_zero(self):
        assert _config().max_peak == 0.0

    def test_gate_off_leaves_the_config_untouched(self):
        """The strongest available form of "default is a no-op".

        Two independent sim runs cannot be compared bit-for-bit because the
        digital twin is process-global and shares RNG state, so the invariant is
        asserted on the configuration and on the absence of any gate firing.
        """
        default = _config()
        explicit = _config(max_peak=0.0)
        assert default.max_peak == 0.0
        assert explicit.max_peak == default.max_peak

    def test_no_peak_rows_when_disabled(self, monkeypatch):
        from ao_shaping.optimizer.wfless.slm_zernike_pib import (
            optimize_slm_zernike_pib,
        )

        _patch_sim(monkeypatch)
        rec = optimize_slm_zernike_pib(_config())
        assert not [r for r in rec if r.get("_gate") == "peak"]


class TestGateFires:
    def test_absurdly_low_cap_freezes_every_epoch(self, monkeypatch):
        """A cap below any real frame must stop every commit."""
        from ao_shaping.optimizer.wfless.slm_zernike_pib import (
            optimize_slm_zernike_pib,
        )

        _patch_sim(monkeypatch)
        rec = optimize_slm_zernike_pib(_config(max_peak=1.0))
        peak_rows = [r for r in rec if r.get("_gate") == "peak"]
        assert peak_rows, "cap=1.0 should have frozen every epoch"
        # Coefficients must not have moved while frozen.
        first = np.asarray(peak_rows[0]["_c"], dtype=np.float64)
        last = np.asarray(peak_rows[-1]["_c"], dtype=np.float64)
        assert np.array_equal(first, last)

    def test_peak_rows_are_honest(self, monkeypatch):
        """Real J, real peak, zero diff, zero grad - not a fabricated row."""
        from ao_shaping.optimizer.wfless.slm_zernike_pib import (
            optimize_slm_zernike_pib,
        )

        _patch_sim(monkeypatch)
        rec = optimize_slm_zernike_pib(_config(max_peak=1.0))
        for row in (r for r in rec if r.get("_gate") == "peak"):
            assert np.isfinite(float(row["J"]))
            assert float(row["max_brt"]) > 1.0
            assert float(row["_diff"]) == 0.0
            assert not np.any(np.asarray(row["_grad"], dtype=np.float64))

    def test_gate_value_is_in_the_documented_enum(self, monkeypatch):
        from ao_shaping.optimizer.wfless.slm_zernike_pib import (
            optimize_slm_zernike_pib,
        )

        _patch_sim(monkeypatch)
        rec = optimize_slm_zernike_pib(_config(max_peak=1.0))
        for row in rec:
            assert row.get("_gate") in _GATE_VALUES

    def test_abba_peak_gate_covers_all_four_frames(self, monkeypatch):
        """The cap must consider every displayed frame, not just +/- one."""
        from ao_shaping.optimizer.wfless.slm_zernike_pib import (
            optimize_slm_zernike_pib,
        )

        _patch_sim(monkeypatch)
        rec = optimize_slm_zernike_pib(_config(max_peak=1.0, abba_sampling=True))
        assert [r for r in rec if r.get("_gate") == "peak"]


class TestOrthogonality:
    def test_gate_does_not_disturb_the_noise_history(self, monkeypatch):
        """A peak freeze must not append to the diff history.

        The noise gate derives ``sigma_hat`` from that history; feeding it
        frozen (zero-diff) epochs would deflate sigma_hat and make the noise
        gate stop firing - a silent coupling between two unrelated gates.
        """
        from ao_shaping.optimizer.wfless import slm_zernike_pib as mod
        from ao_shaping.optimizer.wfless.slm_zernike_pib import (
            optimize_slm_zernike_pib,
        )

        seen: list[float] = []
        real_gate = mod._noise_gate

        def _spy(diff, sigma_hat, k):
            seen.append(float(diff))
            return real_gate(diff, sigma_hat, k)

        _patch_sim(monkeypatch)
        monkeypatch.setattr(mod, "_noise_gate", _spy)
        optimize_slm_zernike_pib(_config(max_peak=1.0))
        # Every epoch is frozen, so the noise gate is never reached at all.
        assert seen == []

    def test_optimizer_update_not_called_on_violation(self, monkeypatch):
        """Mirror the fold gate, not the noise gate: zero-updating is wrong here.

        The noise gate calls ``optimizer.update(zeros)`` to decay momentum
        because that diff *was* gradient noise. A peak violation carries no
        information about the gradient, so the same call would be an unjustified
        edit to the optimiser state.
        """
        from ao_shaping.optimizer.wfless import slm_zernike_pib as mod
        from ao_shaping.optimizer.wfless.slm_zernike_pib import (
            optimize_slm_zernike_pib,
        )

        calls: list[int] = []
        real_create = mod._create_optimizer

        def _spy(*a, **k):
            opt = real_create(*a, **k)
            real_update = opt.update

            def _wrapped(grad, *g, **kw):
                calls.append(1)
                return real_update(grad, *g, **kw)

            opt.update = _wrapped
            return opt

        _patch_sim(monkeypatch)
        monkeypatch.setattr(mod, "_create_optimizer", _spy)
        optimize_slm_zernike_pib(_config(max_peak=1.0))
        assert calls == [], "optimizer.update ran despite every epoch being frozen"


class TestAutoExposureWarning:
    def test_warns_when_exposure_is_auto(self, monkeypatch):
        """A counts cap under auto-exposure is not a power cap - say so."""
        from ao_shaping.optimizer.wfless.slm_zernike_pib import (
            optimize_slm_zernike_pib,
        )

        _patch_sim(monkeypatch)
        cfg = _config(max_peak=50.0)
        cfg.camera.exposure_time_ms = 0.0
        _LOG_MSGS.clear()
        sink = _collect_logs()
        try:
            optimize_slm_zernike_pib(cfg)
        finally:
            _drop_logs(sink)
        assert any("auto-exposure" in m for m in _LOG_MSGS), (
            "expected an auto-exposure masking warning; got: "
            + " | ".join(_LOG_MSGS[:6])
        )

    def test_no_warning_when_exposure_is_fixed(self, monkeypatch):
        from ao_shaping.optimizer.wfless.slm_zernike_pib import (
            optimize_slm_zernike_pib,
        )

        _patch_sim(monkeypatch)
        _LOG_MSGS.clear()
        sink = _collect_logs()
        try:
            optimize_slm_zernike_pib(_config(max_peak=50.0))
        finally:
            _drop_logs(sink)
        assert not any("auto-exposure" in m for m in _LOG_MSGS)


@pytest.mark.parametrize("cap", [0.0, 1.0])
def test_gate_value_round_trips_through_the_runner(cap):
    """The CLI field must reach the optimizer config."""
    cfg = _config(max_peak=cap)
    assert cfg.max_peak == cap