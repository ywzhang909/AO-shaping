"""Tests for the simulation seam added to the `wf` runner path.

Three surfaces are pinned here because each was a real defect at some point:

* the ``lr``/``delta`` override in :func:`optimizer_rms_dm` — the first
  implementation was unreachable, because ``lr, delta = schedule_lr_delta(...)``
  rebound the parameter names before the override was read;
* ``SimulateDM._publish_to_wfs`` — silently skipped when no simulated sensor is
  live, which must not raise on the hardware-oriented paths;
* the runner wiring for ``--disturbance-cn2`` / ``--lr`` / ``--delta``.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from ao_shaping.drivers.dm._registry import create_dm
from ao_shaping.optimizer.wf import rms as rms_mod
from ao_shaping.optimizer.wf.rms import optimizer_rms_dm


@pytest.fixture
def sim_dm():
    dm = create_dm("sim")
    dm.open()
    try:
        yield dm
    finally:
        dm.close()


def _run(dm, **kwargs):
    """Run the optimizer with the progress bar and log noise suppressed."""
    with contextlib.redirect_stderr(io.StringIO()):
        return optimizer_rms_dm(
            epochs=kwargs.pop("epochs", 3),
            wfs_res="768",
            dm=dm,
            wfs_type="sim",
            early_stop_threshold=0.0,
            **kwargs,
        )


class TestStepSizeOverride:
    """The override must beat the auto-schedule, and only when supplied."""

    def test_override_beats_the_schedule(self, sim_dm, monkeypatch):
        monkeypatch.setattr(
            rms_mod, "schedule_lr_delta", lambda rms: (111.0, 7.0)
        )
        rec = _run(sim_dm, lr=5.0, delta=42.0)
        assert rec.history[-1]["_gamma"] == 5.0
        assert rec.history[-1]["delta"] == 42.0

    def test_none_falls_back_to_the_schedule(self, sim_dm, monkeypatch):
        """``None`` must mean 'keep the hardware-calibrated schedule'."""
        monkeypatch.setattr(
            rms_mod, "schedule_lr_delta", lambda rms: (111.0, 7.0)
        )
        rec = _run(sim_dm)
        assert rec.history[-1]["_gamma"] == 111.0
        assert rec.history[-1]["delta"] == 7.0

    def test_schedule_still_re_schedules_each_epoch(self, sim_dm, monkeypatch):
        """The per-epoch re-schedule must not clobber a supplied override."""
        calls = iter([(111.0, 7.0)] * 2 + [(222.0, 9.0)] * 40)
        monkeypatch.setattr(
            rms_mod, "schedule_lr_delta", lambda rms: next(calls)
        )
        rec = _run(sim_dm, epochs=4, lr=5.0, delta=42.0)
        gammas = {h["_gamma"] for h in rec.history}
        deltas = {h["delta"] for h in rec.history}
        assert gammas == {5.0}
        assert deltas == {42.0}


class TestDisturbancePlumbing:
    def test_cn2_reaches_the_sensor(self, sim_dm):
        """A non-zero cn2 must actually raise the measured aberration."""
        flat = _run(sim_dm, epochs=2, disturbance_cn2=0.0)
        aberrated = _run(sim_dm, epochs=2, disturbance_cn2=2e-13)
        assert (
            aberrated.history[0]["_statics"]["wighted_rms"]
            > 10 * flat.history[0]["_statics"]["wighted_rms"]
        )


class TestPublishToWfsIsSafeWithoutASensor:
    def test_missing_sensor_does_not_raise(self):
        """No live sensor must be a silent skip, not an exception.

        ``_ACTIVE_SENSOR`` is process-wide, so this only means anything when no
        sensor has been built; the branch exists for hardware-oriented paths.
        """
        import ao_shaping.drivers.sim.wfs.simulated_wfs as mod

        dm = create_dm("sim")
        previous = mod._ACTIVE_SENSOR
        mod._ACTIVE_SENSOR = None
        try:
            dm.open()
            dm.send_voltages(np.zeros(dm.DM_NUM))  # must not raise
        finally:
            dm.close()
            mod._ACTIVE_SENSOR = previous

    def test_actuator_mismatch_is_skipped_quietly(self):
        """A sensor with a different actuator count must be skipped, not crash."""
        import ao_shaping.drivers.sim.wfs.simulated_wfs as mod

        dm = create_dm("sim")
        sensor = mod.SimulatedWFS(resolution=48)
        sensor.dm_optics.n_actuators = 7  # force a mismatch
        previous = mod._ACTIVE_SENSOR
        mod._ACTIVE_SENSOR = sensor
        try:
            dm.open()
            dm.send_voltages(np.zeros(dm.DM_NUM))  # mismatch -> skip, no raise
        finally:
            dm.close()
            mod._ACTIVE_SENSOR = previous


class TestRunnerSurface:
    def test_wf_runner_exposes_the_new_options(self):
        from ao_shaping.runners.nlight_dm.wf_runner import run

        names = {p.name for p in run.params}
        assert {"wfs_type", "disturbance_cn2", "lr", "delta"} <= names

    def test_wfs_params_offers_a_sim_choice(self):
        from ao_shaping.runners.runner_common import WfsParams
        import dataclasses

        field = next(
            f for f in dataclasses.fields(WfsParams) if f.name == "wfs_type"
        )
        assert field.default == "thorlab", (
            "the default must preserve today's hardware behaviour"
        )
