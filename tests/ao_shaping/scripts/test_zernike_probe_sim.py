from __future__ import annotations

import numpy as np
import pytest

from scripts.slm import zernike_probe_sim as sim


def test_zernike_probes_symmetric_randomised_and_same_budget() -> None:
    rng = np.random.default_rng(4)
    phases = sim.zernike_phases(2, 24, rng)
    assert phases.shape == (32, 24, 24)
    assert sim.random_phases(32, 24, rng).shape == phases.shape
    stacked = np.stack(list(sim.basis(24).values()))
    for index, mode in enumerate(sim.MODES):
        # Each random-order repetition is a 16-frame block with one exact ± pair per mode.
        for repeat in range(2):
            block = phases[16*repeat:16*(repeat+1)]
            amp = sim.AMPLITUDES[mode]
            matching = np.isclose(block, amp*stacked[index]).all(axis=(1, 2))
            opposite = np.isclose(block, -amp*stacked[index]).all(axis=(1, 2))
            assert matching.sum() == opposite.sum() == 1
            assert abs(np.flatnonzero(matching)[0] - np.flatnonzero(opposite)[0]) == 1
    with pytest.raises(ValueError):
        sim.zernike_phases(0, 24, rng)


def test_trust_region_bounds_and_l2() -> None:
    import torch

    fit = sim.make_fit(np.zeros((1, 24, 24), np.float32), np.zeros((1, 16), np.float32),
                       size=24, camera_size=16)
    with torch.no_grad():
        fit.d.fill_(3.)
        fit.residual.fill_(1.)
    sim.project_trust_region(fit)
    coefficient = np.r_[float(fit.d.detach())*fit.diopter_scale, fit.residual.detach().numpy()]
    assert abs(coefficient[0]) <= .3 + 1e-7
    assert np.max(np.abs(coefficient[1:])) <= .15 + 1e-7
    assert np.linalg.norm(coefficient) <= .5 + 1e-7


def test_small_two_arm_run_has_holdout_early_stop_and_common_evaluation() -> None:
    holdout = sim.make_holdout(24, 16, 29)
    rng = np.random.default_rng(8)
    phases = sim.zernike_phases(1, 24, rng)
    twin = sim.make_fit(phases, np.zeros((len(phases), 16), np.float32), size=24,
                        camera_size=16, scale=.985, true=True)
    measured = sim.twin_render(twin, rng)
    result = sim.fit_arm(phases, measured, size=24, camera_size=16, holdout=holdout,
                         iterations=12, patience=5)
    assert result["iterations"] <= 12
    assert result["iterations"] >= 7  # Middle block is always exposed before early stopping.
    assert len(result["coefficients"]) == 8
    assert result["nrmse"] == pytest.approx(min(h["nrmse"] for h in result["history"]))
    assert holdout[0].shape[0] == holdout[1].shape[0] == 3


def test_dominance_criterion() -> None:
    matrix = np.diag([2., 3., 4.])
    assert np.all(sim.dominance_ratios(matrix) > 1e10)
    matrix[0, 1] = 2.5
    assert sim.dominance_ratios(matrix)[1] == pytest.approx(3/2.5)


def test_twin_camera_nonidealities_and_nominal_zero_coefficients() -> None:
    phases = np.zeros((1, 24, 24), np.float32)
    twin = sim.make_fit(phases, np.zeros((1, 16), np.float32), size=24,
                        camera_size=16, scale=.985, true=True)
    nominal = sim.make_fit(phases, np.zeros((1, 16), np.float32), size=24, camera_size=16)
    assert float(twin.d)*twin.diopter_scale == pytest.approx(sim.TRUE_C[0])
    assert float(nominal.d) == 0.
    assert np.count_nonzero(nominal.residual.detach().numpy()) == 0
    measured = sim.twin_render(twin, np.random.default_rng(3))
    assert measured.min() > 0.
    assert measured.max() <= 255.25  # saturation precedes additive read noise
