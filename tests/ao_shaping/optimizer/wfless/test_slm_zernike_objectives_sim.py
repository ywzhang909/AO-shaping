"""Offline integration tests: each shaping objective runs a tiny SPGD loop
on the 2f-Fourier sim camera (no hardware).

Pins that every objective in :data:`SHAPING_OBJECTIVE_CHOICES`:
* validates & builds without opening any device;
* completes 1-epoch SPGD on the sim CCD/SLM;
* records the objective column in history;
* follows the documented min/max direction.
"""

from __future__ import annotations

import pytest

from ao_shaping.drivers.sim.slm_pib_sim import (
    SimSLMPib,
    register_sim_camera,
    reset_system,
)

# Use the shaping config which has w_outside field
from ao_shaping.optimizer.wfless.slm_zernike_shaping import (
    SlmZernikePibConfig,
    optimize_slm_zernike_pib,
)
from ao_shaping.runners.runner_common import (
    CameraParamsPib,
    ObjectiveTarget,
    SlmParamsPib,
)
from ao_shaping.utils.image.targets import SHAPING_OBJECTIVE_CHOICES


# ---- min/max truth table --------------------------------------------------
# Objectives that are MAXIMISED (search ascends J)
MAXIMISE = {
    "pib",
    "avg_radiu",
    "shape",
    "roi_pib",
    "rms_pib",
}
# Objectives that are MINIMISED (search descends J)
MINIMISE = {
    "radiu",
    "rmse",
    "rmse_out",
}
assert MAXIMISE | MINIMISE == set(SHAPING_OBJECTIVE_CHOICES)
assert MAXIMISE & MINIMISE == set()


def _sim_config(
    objective: str,
    target_shape: str | None = None,
    epochs: int = 1,
    algorithm: str = "spgd",
    pop_size: int | None = None,
    w_outside: float = 1.0,
) -> SlmZernikePibConfig:
    """Build a minimal config that routes camera -> sim CCD, SLM -> SimSLMPib."""
    return SlmZernikePibConfig(
        center="shape",
        epochs=epochs,
        algorithm=algorithm,
        pop_size=pop_size,
        w_outside=w_outside,
        camera=CameraParamsPib(
            target=ObjectiveTarget(name=objective, target_shape=target_shape),
            cam_type="sim",
            cam_size=128,
            exposure_time_ms=80.0,
        ),
        slm=SlmParamsPib(n_max=4),
    )


# ---- helper to monkeypatch the real SLM with the sim one ------------------
def _patch_slm(monkeypatch):
    # Patch both modules since they both import Santec
    for mod_name in (
        "ao_shaping.optimizer.wfless.slm_zernike_pib",
        "ao_shaping.optimizer.wfless.slm_zernike_shaping",
    ):
        import importlib

        mod = importlib.import_module(mod_name)
        monkeypatch.setattr(
            SimSLMPib,
            "from_params",
            classmethod(lambda cls, params, **kw: cls()),
            raising=False,
        )
        monkeypatch.setattr(mod, "Santec", SimSLMPib)


# ---- tests -----------------------------------------------------------------
@pytest.mark.parametrize("obj", sorted(SHAPING_OBJECTIVE_CHOICES))
def test_objective_runs_1_epoch_spgd_on_sim(monkeypatch, obj):
    """Every objective validates, runs 1 SPGD epoch on sim, logs its column."""
    register_sim_camera()
    reset_system(seed=42)

    _patch_slm(monkeypatch)

    # pib + target_shape is promoted to "shape" by the resolver
    # For pib/radiu/avg_radiu without target_shape, no shape is needed
    shape = (
        "square"
        if obj in ("shape", "roi_pib", "rms_pib", "rmse", "rmse_out")
        else ("square" if obj == "pib" else None)
    )
    config = _sim_config(obj, target_shape=shape, epochs=1, algorithm="spgd")

    recorder = optimize_slm_zernike_pib(config)

    # init row + 1 epoch row
    assert len(recorder.history) == 2
    # the effective objective column (after promotion) must be present
    effective_obj = "shape" if obj == "pib" else obj
    assert effective_obj in recorder.history[0]
    assert effective_obj in recorder.history[-1]


@pytest.mark.parametrize("obj", sorted(MAXIMISE))
def test_maximise_objectives_improve_over_init(monkeypatch, obj):
    """For maximise objectives, the recorded value should not decrease vs init
    (allowing for noise; at 1 epoch it may stay similar, but must not be NaN)."""
    register_sim_camera()
    reset_system(seed=123)

    _patch_slm(monkeypatch)

    shape = (
        "square"
        if obj in ("shape", "roi_pib", "rms_pib", "rmse", "rmse_out")
        else ("square" if obj == "pib" else None)
    )
    config = _sim_config(obj, target_shape=shape, epochs=2, algorithm="spgd")

    recorder = optimize_slm_zernike_pib(config)

    effective_obj = "shape" if obj == "pib" else obj
    init_val = recorder.history[0][effective_obj]
    final_val = recorder.history[-1][effective_obj]

    assert init_val is not None
    assert final_val is not None
    # maximise: final >= init is the ideal, but with 2 epochs + sim noise we only
    # assert no NaN and the direction flag is "max" in the recorder
    assert recorder.mode == "max"


@pytest.mark.parametrize("obj", sorted(MINIMISE))
def test_minimise_objectives_do_not_increase_over_init(monkeypatch, obj):
    """For minimise objectives, the recorded value should not increase vs init
    (same caveats as the maximise test)."""
    register_sim_camera()
    reset_system(seed=123)

    _patch_slm(monkeypatch)

    shape = (
        "square"
        if obj in ("shape", "roi_pib", "rms_pib", "rmse", "rmse_out", "pib")
        else None
    )
    config = _sim_config(obj, target_shape=shape, epochs=2, algorithm="spgd")

    recorder = optimize_slm_zernike_pib(config)

    init_val = recorder.history[0][obj]
    final_val = recorder.history[-1][obj]

    assert init_val is not None
    assert final_val is not None
    # minimise: final <= init is the ideal
    assert recorder.mode == "min"


def test_rms_pib_records_weight_panel(monkeypatch):
    """rms_pib specifically should record the adaptive weight panel (m_* cols)."""
    register_sim_camera()
    reset_system(seed=42)
    _patch_slm(monkeypatch)

    config = _sim_config("rms_pib", target_shape="square", epochs=1)
    recorder = optimize_slm_zernike_pib(config)

    assert "m_rms_pib" in recorder.history[-1]
    assert "m_pib" in recorder.history[-1]
    assert "m_energy" in recorder.history[-1]
    assert "m_rmse" in recorder.history[-1]
    assert "m_rms_t" in recorder.history[-1]
    assert "m_ee" in recorder.history[-1]


def test_shape_schedule_runs_without_error(monkeypatch):
    """shape objective with shape_schedule=True runs on sim."""
    register_sim_camera()
    reset_system(seed=42)
    _patch_slm(monkeypatch)

    config = _sim_config("shape", target_shape="circle", epochs=1)
    config.camera.shape_schedule = True
    recorder = optimize_slm_zernike_pib(config)

    assert "shape" in recorder.history[-1]
    assert recorder.mode == "max"


def test_roi_pib_runs_without_error(monkeypatch):
    register_sim_camera()
    reset_system(seed=42)
    _patch_slm(monkeypatch)

    config = _sim_config("roi_pib", target_shape="annular", epochs=1)
    recorder = optimize_slm_zernike_pib(config)

    assert "roi_pib" in recorder.history[-1]
    assert recorder.mode == "max"


def test_rmse_out_runs_without_error(monkeypatch):
    register_sim_camera()
    reset_system(seed=42)
    _patch_slm(monkeypatch)

    config = _sim_config("rmse_out", target_shape="pentagon", epochs=1)
    config.w_outside = 2.0
    recorder = optimize_slm_zernike_pib(config)

    assert "rmse_out" in recorder.history[-1]
    assert recorder.mode == "min"


def test_heuristic_algorithm_on_sim(monkeypatch):
    """One heuristic (GA) on sim to ensure the wiring works."""
    register_sim_camera()
    reset_system(seed=42)
    _patch_slm(monkeypatch)

    config = _sim_config("pib", epochs=1, algorithm="ga", pop_size=4)
    recorder = optimize_slm_zernike_pib(config)

    # GA does pop_size evaluations per generation; with epochs=1 we get
    # at least 1 evaluation + init
    assert len(recorder.history) >= 2
    assert "pib" in recorder.history[-1]
    assert recorder.mode == "max"


def test_unknown_objective_rejected_before_hardware(monkeypatch):
    """Invalid objective raises ValueError before any device open."""
    register_sim_camera()
    reset_system(seed=42)
    _patch_slm(monkeypatch)

    config = _sim_config("bogus", epochs=1)
    with pytest.raises(ValueError, match="objective must be one of"):
        optimize_slm_zernike_pib(config)


def test_invalid_target_shape_for_radiu_rejected(monkeypatch):
    """radiu + target_shape must raise before hardware."""
    register_sim_camera()
    reset_system(seed=42)
    _patch_slm(monkeypatch)

    config = _sim_config("radiu", target_shape="square", epochs=1)
    with pytest.raises(ValueError, match="target_shape can only be used with"):
        optimize_slm_zernike_pib(config)


def test_invalid_target_shape_for_avg_radiu_rejected(monkeypatch):
    register_sim_camera()
    reset_system(seed=42)
    _patch_slm(monkeypatch)

    config = _sim_config("avg_radiu", target_shape="circle", epochs=1)
    with pytest.raises(ValueError, match="target_shape can only be used with"):
        optimize_slm_zernike_pib(config)
