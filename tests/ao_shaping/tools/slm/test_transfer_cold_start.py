"""Cold seeds have no static compensation and need no retrieved phase."""
from types import SimpleNamespace

import numpy as np
import pytest

from ao_shaping.tools.slm.transfer.cold_start import cold_phase


def test_plane_is_only_measured_carrier():
    params = SimpleNamespace(grid_size=16, origin_xy=(20., 30.), ramp_matrix=((1000., 0.), (0., -1000.)))
    f = SimpleNamespace(params=params, pitch=120.)
    phase, meta = cold_phase(f, (30., 30.))
    expected = np.broadcast_to(2*np.pi*.01*np.arange(16)*120., (16, 16))
    np.testing.assert_allclose(phase, expected, rtol=1e-6)
    assert meta["gs_calls"] == 0 and not meta["historical_phase"]
    assert meta["shaping_rms_rad"] == 0
    random, _ = cold_phase(f, (30., 30.), kind="random", seed=42)
    repeat, _ = cold_phase(f, (30., 30.), kind="random", seed=42)
    np.testing.assert_array_equal(random, repeat)
    assert .99 < np.std(random-phase) < 1.01


def test_cold_workflow_has_no_gs_or_unseeded_model_stage(tmp_path):
    from ao_shaping.tools.slm.transfer.workflow import TransferSession
    plan = TransferSession(tmp_path, shapes=("square",), initial_phase="plane", calibration=False).plan()
    assert "CCD_model_blocks" not in plan["stages"]
    assert "nominal_initialization" in plan["stages"]
    assert "CCD_anchored_blocks" in plan["stages"]
    assert plan["settings"]["abba"] is False
    assert "direction_check" not in plan["stages"]


def test_cli_rejects_historical_phase_combined_with_cold_seed(tmp_path, monkeypatch):
    import sys
    import pytest
    from ao_shaping.tools.slm.transfer import validate
    monkeypatch.setattr(sys, "argv", ["validate", "--checkpoint", str(tmp_path / "model.json"),
        "--out", str(tmp_path / "out"), "--zero-xy", "1", "2", "--seed-run", str(tmp_path),
        "--initial-phase", "plane"])
    with pytest.raises(SystemExit):
        validate.main()


@pytest.mark.parametrize("formation", [False, True])
def test_cold_random_phase_can_reduce_physical_model_shape_loss(formation):
    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        DifferentiableUniformityOptimizer, FrozenDefocusForward, FrozenDefocusParameters,
        recentre_window,
    )
    from ao_shaping.algorithm.signal_processing.multishape_objective import ShapeObjective, target_at
    p = FrozenDefocusParameters(0., (100., 100.), (960., 600.), (300., 260.),
        {}, 1., 0., 0., ((7500., 0.), (0., 7500.)), grid_size=32, fft_size=128, pixel_samples=1)
    xy = recentre_window((100., 100.), half_span_px=100., step_px=4.)
    f = FrozenDefocusForward(p, xy, device="cpu")
    phase, _ = cold_phase(f, (100., 100.), kind="random", seed=42)
    objective = ShapeObjective(target_at(xy, "square", (100., 100.), 48.), formation=formation)
    opt = DifferentiableUniformityOptimizer(f, objective, phase, lr=.02, step_rms_cap=.5, device="cpu")
    for _ in range(12):
        opt.update()
    assert opt.loss_history[-1]["loss"] < opt.loss_history[0]["loss"]


def test_formation_prefers_target_energy_over_dim_uniform_background():
    import torch
    from ao_shaping.algorithm.signal_processing.multishape_objective import ShapeObjective
    target = np.zeros((32, 32), dtype=np.float32)
    target[8:24, 8:24] = 1
    dim = torch.ones((32, 32))
    beam = dim.clone()
    beam[8:24, 8:24] = 12
    beam[8:16, 8:24] = 24
    phase = torch.zeros((16, 16))
    objective = ShapeObjective(target, formation=True)
    assert objective.evaluate(beam, phase)[0] < objective.evaluate(dim, phase)[0]
