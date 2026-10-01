"""Contract tests for DM voltage -> pupil phase coupling.

Before this coupling existed, ``SimPibSystem.far_field()`` summed only the SLM
command phase, so ``pib`` and ``combined`` -- which drive DM *voltages* -- ran
to completion while the DM had literally no effect on the optics. The loops
exited 0 and produced CSV files, but the numbers were pure noise: not
"SPGD failed to converge", but "the DM was not in the model".

These tests pin the coupling itself: voltage must produce a localised phase
bump at the right actuator, with the documented voltage->stroke->radian scale,
and it must reach the far field.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.drivers.sim.dm_optics import SimDmOptics


@pytest.fixture
def optics():
    return SimDmOptics(n_actuators=64, slm_shape=(120, 192))


def test_flat_voltages_give_no_phase(optics):
    assert np.count_nonzero(optics.phase()) == 0


def test_single_actuator_gives_a_localised_bump(optics):
    """One actuator of drive must perturb phase near that actuator only."""
    volts = np.zeros(optics.n_actuators)
    volts[27] = optics.v_max
    optics.set_voltages(volts)
    phase = optics.phase()
    assert np.abs(phase).max() > 0.0
    peak = np.unravel_index(np.argmax(np.abs(phase)), phase.shape)
    expected = optics.actuator_grid[27]
    assert abs(peak[0] - expected[0]) <= optics.influence_radius_px + 1
    assert abs(peak[1] - expected[1]) <= optics.influence_radius_px + 1


def test_phase_polarity_follows_voltage(optics):
    volts = np.zeros(optics.n_actuators)
    volts[10] = 50.0
    optics.set_voltages(volts)
    peak = np.unravel_index(np.argmax(np.abs(optics.phase())), optics.phase().shape)
    assert optics.phase()[peak] > 0
    optics.set_voltages(-np.abs(volts))
    assert optics.phase()[peak] < 0


def test_documented_voltage_to_radian_scale(optics):
    """At full drive a single actuator's peak must equal its stroke in radians.

    Pins the scale so a report cannot silently drift: phase = opd * 2pi/lambda.
    """
    assert optics.wavelength_nm == 532, (
        "the sim DM must default to 532 nm to match um_to_waves()"
    )
    volts = np.zeros(optics.n_actuators)
    volts[0] = optics.v_max
    optics.set_voltages(volts)
    peak = float(np.abs(optics.phase()).max())
    expected_rad = optics.stroke_um * 2.0 * np.pi / (optics.wavelength_nm * 1e-3)
    assert peak == pytest.approx(expected_rad, rel=0.02)


def test_voltages_are_clipped_to_the_dm_range(optics):
    """Over-driving must clip to the same result as driving each rail exactly.

    The two rails are asymmetric (``v_min=-300`` vs ``v_max=499``), so the
    negative case clips to ``v_min`` and is deliberately *not* the mirror image
    of the positive one.
    """
    for rail in (optics.v_max, optics.v_min):
        volts = np.zeros(optics.n_actuators)
        volts[0] = rail
        optics.set_voltages(volts)
        at_rail = optics.phase().copy()

        optics.set_voltages(volts * 10)
        assert np.allclose(optics.phase(), at_rail)


def test_response_is_linear_in_voltage(optics):
    """The coupling is a fixed influence matrix, hence linear."""
    volts = np.zeros(optics.n_actuators)
    volts[5] = 30.0
    optics.set_voltages(volts)
    single = optics.phase().copy()
    optics.set_voltages(2 * volts)
    assert np.allclose(optics.phase(), 2.0 * single)


def test_zero_voltage_restores_the_flat_pupil(optics):
    volts = np.zeros(optics.n_actuators)
    volts[3] = 120.0
    optics.set_voltages(volts)
    assert np.abs(optics.phase()).max() > 0
    optics.set_voltages(np.zeros(optics.n_actuators))
    assert np.count_nonzero(optics.phase()) == 0


class TestReachesTheFarField:
    """The point of the whole exercise: DM drive must move the far field."""

    def test_far_field_responds_to_dm_voltage(self, optics):
        from ao_shaping.drivers.sim.slm_pib_sim import SimPibSystem

        system = SimPibSystem(slm_shape=(120, 192), ccd_res=(120, 192))
        system.set_phase_rad(np.zeros((120, 192)))
        flat = system.far_field()

        volts = np.zeros(optics.n_actuators)
        volts[optics.n_actuators // 2] = optics.v_max
        system.dm_optics.set_voltages(volts)
        driven = system.far_field()

        assert not np.array_equal(flat, driven), (
            "DM voltage did not change the far field -- the DM is not coupled "
            "into the optical model"
        )

    def test_dm_phase_does_not_bake_into_the_slm_command(self, optics):
        """Following the SimDisturbance precedent: contributors sum at evaluation.

        Baking DM phase into ``_phase`` would double-count it and would make the
        stored command disagree with what the optimizer believes it sent.
        """
        from ao_shaping.drivers.sim.slm_pib_sim import SimPibSystem

        system = SimPibSystem(slm_shape=(120, 192), ccd_res=(120, 192))
        system.set_phase_rad(np.zeros((120, 192)))
        volts = np.zeros(optics.n_actuators)
        volts[1] = 80.0
        system.dm_optics.set_voltages(volts)
        system.far_field()
        assert np.count_nonzero(system._phase) == 0
