"""Offline contract for the ``--cam_type sim`` wiring of both shaping halves.

Both commands advertise ``--cam_type sim`` as "no hardware required", so the
invariant is narrow and absolute: **on that path neither command may touch a
real device**. It was not, on either half —

* ``spgd-square`` never registered the sim camera or rebound ``Santec``, so the
  "offline" run reached ``Santec`` and failed on a missing USB device;
* ``slm-pib`` registered only the camera (at import of ``slm_pib_sim``), so its
  optimizer still built a real ``Santec`` and died with ``-10002`` *after*
  burning the auto-exposure probe.

No hardware: everything asserted here is either a module-global rebinding or a
real (fast, 2-epoch) run against the 2f-Fourier digital twin.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from ao_shaping.drivers.sim.slm_pib_sim import SimSLMPib
from ao_shaping.runners.runner_common import (
    patch_sim_pib_shaping,
    patch_sim_square_shaping,
)


@pytest.fixture(autouse=True)
def _restore_santec_globals():
    """Undo the module-global rebinding so tests cannot leak into each other."""
    import ao_shaping.optimizer.wfless.slm_square_shaping as square_opt
    import ao_shaping.optimizer.wfless.slm_zernike_pib as pib_opt

    saved = (square_opt.Santec, pib_opt.Santec)
    try:
        yield
    finally:
        square_opt.Santec, pib_opt.Santec = saved


class TestPatchIsANoOpForHardware:
    """A hardware camera type must never install the twin."""

    def test_square_helper_leaves_the_real_driver(self):
        import ao_shaping.optimizer.wfless.slm_square_shaping as opt

        patch_sim_square_shaping("daheng")
        assert opt.Santec is not SimSLMPib

    def test_pib_helper_leaves_the_real_driver(self):
        import ao_shaping.optimizer.wfless.slm_zernike_pib as opt

        patch_sim_pib_shaping("daheng")
        assert opt.Santec is not SimSLMPib


class TestPatchInstallsTheTwin:
    def test_square_helper_rebinds_the_square_optimizer(self):
        import ao_shaping.optimizer.wfless.slm_square_shaping as opt

        patch_sim_square_shaping("sim")
        assert opt.Santec is SimSLMPib

    def test_pib_helper_rebinds_the_pib_optimizer(self):
        import ao_shaping.optimizer.wfless.slm_zernike_pib as opt

        patch_sim_pib_shaping("sim")
        assert opt.Santec is SimSLMPib

    def test_pib_helper_does_not_reset_the_simulated_system(self):
        """The PIB sim harness owns the system; a reset here would drop it.

        ``scripts/slm_pib_sim_run.py`` installs a seeded system *with* a
        disturbance and wraps ``reset_system`` so every call re-attaches it. If
        the runner reset the system itself, a run whose manifest claims a
        disturbance would run without one.
        """
        import ao_shaping.drivers.sim.slm_pib_sim as sim_module

        calls: list[object] = []
        saved = sim_module.reset_system
        sim_module.reset_system = lambda *a, **kw: calls.append((a, kw))
        try:
            patch_sim_pib_shaping("sim")
        finally:
            sim_module.reset_system = saved
        assert calls == [], f"patch_sim_pib_shaping reset the system: {calls}"

    def test_square_helper_still_pins_seed_42(self):
        """Reproducibility is the square half's contract; do not lose it."""
        import ao_shaping.drivers.sim.slm_pib_sim as sim_module

        calls: list[object] = []
        saved = sim_module.reset_system
        sim_module.reset_system = lambda *a, **kw: calls.append((a, kw))
        try:
            patch_sim_square_shaping("sim")
        finally:
            sim_module.reset_system = saved
        assert calls == [((), {"seed": 42})], calls


class TestOfflineRunsReachTheTwin:
    """The end-to-end claim, executed: 2 epochs, no hardware, no probe."""

    def test_spgd_square_runs_offline(self, tmp_path):
        result = CliRunner().invoke(
            _spgd_square(),
            [
                "-e",
                "2",
                "--cam_type",
                "sim",
                "--slm_type",
                "sim",
                "--basis",
                "freeform",
                "--phase-grid",
                "4",
                "--seed",
                "0",
            ],
            obj={"dir": str(tmp_path)},
        )
        assert result.exit_code == 0, result.output
        assert "Best quality" in result.output

    def test_slm_pib_spgd_runs_offline(self, tmp_path):
        from ao_shaping.runners.slm.shaping_runner import spgd

        result = CliRunner().invoke(
            spgd,
            ["-e", "2", "--cam_type", "sim", "-d", str(tmp_path)],
        )
        assert result.exit_code == 0, result.output


def _spgd_square():
    from ao_shaping.runners.slm.shaping_runner import square

    return square
