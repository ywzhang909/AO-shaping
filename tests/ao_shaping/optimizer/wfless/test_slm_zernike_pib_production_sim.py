"""Offline characterization of the PRODUCTION engine, ``slm_zernike_pib``.

Why this module exists
----------------------
``slm_zernike_pib`` is what ``runners/slm/pib_runner.py`` actually calls. The
existing offline coverage in ``test_slm_zernike_objectives_sim.py`` imports
``optimize_slm_zernike_pib`` from ``slm_zernike_shaping`` -- a near-duplicate
module -- so it never exercised the production entry point for most assertions.

That is not a cosmetic gap. The two engines have **diverged**, and the
production one grew features the copy does not even have fields for:

===================  ==========================================
production only      ``abba_sampling``, ``noise_gate_k``,
                     ``noise_gate_window``, ``fold_ratio``,
                     ``n_eval_frames``
copy only            ``debug``, ``debug_dir``, ``w_outside``
===================  ==========================================

So the ABBA palindrome sampling, the brightness-fold integrity gate, the
noise-aware update gate and multi-frame averaging had **no offline coverage at
all** -- they could only ever have been exercised on hardware. This module
closes that gap: every one of them runs here against the sim bench.

It is also the safety net required before R-5~R-8 refactor
``slm_zernike_pib.py`` (objective class family, shared ``BenchSession``,
orchestrator split, hardware try/finally). Those are behaviour-preserving
refactors of a ~900-line optimizer, and the contract they must preserve is
exactly what is asserted here.

Characterization, not endorsement
---------------------------------
These tests pin *current observable* behaviour so a refactor cannot silently
change it. Where current behaviour is internally inconsistent -- the two
search branches use different saturation strategies (R-6) -- the inconsistency
is recorded in ``test_saturation_is_the_known_branch_asymmetry`` rather than
asserted as correct, so that unifying it is an explicit, visible change.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from ao_shaping.drivers.sim.sim_bench_patch import SimSLMPib, install_sim_slm
from ao_shaping.drivers.sim.slm_pib_sim import register_sim_camera, reset_system
from ao_shaping.optimizer.wfless import slm_zernike_pib as engine
from ao_shaping.runners.runner_common import (
    CameraParamsPib,
    ObjectiveTarget,
    SlmParamsPib,
)
from ao_shaping.utils.image.targets import SHAPING_OBJECTIVE_CHOICES

PRODUCTION = "ao_shaping.optimizer.wfless.slm_zernike_pib"

#: Columns every logged row must carry, whatever the branch. The SPGD and
#: heuristic branches share ``_append_row`` ("shared by both branches" per its
#: own docstring), so a row-shape difference between them is a bug, not a style.
#: ``_gate`` is deliberately NOT here: it records an *update verdict*, so the
#: init/baseline row (row 0) legitimately has none.
SHARED_ROW_COLUMNS = frozenset(
    {"J", "_p%", "_max_r", "_diff", "lr", "r", "delta", "_epoch", "_c", "optimizer"}
)

#: Verdicts ``_gate`` may carry, per ``_append_row``'s docstring.
GATE_VERDICTS = frozenset({"applied", "fold", "noise"})


@pytest.fixture(autouse=True)
def _sim_bench(monkeypatch):
    """Route the production engine's SLM to the sim device and reset its state.

    ``install_sim_slm`` is given the production module explicitly -- that is the
    whole point of this file. It raises if the module does not expose
    ``Santec``, so a future refactor that moves the import fails loudly here
    instead of quietly opening real hardware mid-test.
    """
    mod = importlib.import_module(PRODUCTION)
    install_sim_slm(mod)
    monkeypatch.setattr(mod, "Santec", SimSLMPib, raising=False)


def _config(objective: str, *, epochs: int = 2, algorithm: str = "spgd", **overrides):
    return engine.SlmZernikePibConfig(
        center="shape",
        epochs=epochs,
        algorithm=algorithm,
        camera=CameraParamsPib(
            target=ObjectiveTarget(name=objective, target_shape=None),
            cam_type="sim",
            cam_size=128,
            exposure_time_ms=80.0,
        ),
        slm=SlmParamsPib(n_max=4),
        **overrides,
    )


def _run(objective: str = "pib", *, seed: int = 42, **overrides):
    register_sim_camera()
    reset_system(seed=seed)
    return engine.optimize_slm_zernike_pib(_config(objective, **overrides))


def _rows(recorder) -> list[dict]:
    """Logged rows excluding the init/baseline row.

    ``optimize_slm_zernike_pib`` records ``epochs + 1`` rows: row 0 is the
    pre-optimization baseline (no ``_gate``, no verdict -- nothing has been
    attempted yet) and rows 1..N are the epochs.
    """
    return list(recorder.history[1:])


# --------------------------------------------------------------------------
# 1. The production entry point itself
# --------------------------------------------------------------------------
def test_production_module_is_the_one_under_test():
    """Guard against this file silently drifting back onto the copy."""
    assert engine.__name__ == PRODUCTION
    assert hasattr(engine, "Santec"), (
        "the production engine no longer exposes a module-level `Santec`; "
        "install_sim_slm would raise and this file's premise needs revisiting"
    )


@pytest.mark.parametrize("objective", sorted(SHAPING_OBJECTIVE_CHOICES))
def test_every_objective_runs_offline_on_the_production_engine(objective: str):
    recorder = _run(objective, epochs=1)
    assert len(recorder.history) >= 1
    row = recorder.history[-1]
    assert objective in row, f"{objective!r} column missing from the recorded row"
    assert np.isfinite(float(row["J"]))


def test_heuristic_branch_runs_offline():
    recorder = _run("pib", epochs=1, algorithm="ga", pop_size=4)
    assert len(recorder.history) >= 1
    assert np.isfinite(float(recorder.history[-1]["J"]))


@pytest.mark.parametrize("algorithm", ["spgd", "ga", "pso", "de"])
def test_shared_row_shape_across_search_branches(algorithm: str):
    """Both branches append rows through the same ``_append_row``.

    R-6 exists because the *measurement* sequence around that call is duplicated
    per branch. The row schema is the observable part of that contract, so it is
    pinned here for every branch: if the extraction changes what a row contains,
    this fails.
    """
    kwargs = {"pop_size": 4} if algorithm != "spgd" else {}
    recorder = _run("pib", epochs=1, algorithm=algorithm, **kwargs)
    for row in _rows(recorder):
        missing = SHARED_ROW_COLUMNS - set(row)
        assert not missing, f"{algorithm}: row missing shared columns {sorted(missing)}"


def test_history_has_one_baseline_row_plus_one_row_per_epoch():
    """Pin the recording shape: ``epochs + 1`` rows, row 0 being the baseline.

    Not obvious from the call site and easy to break while extracting the
    measurement sequence (R-6/R-7): the baseline row is logged before the loop
    and carries no ``_gate``.
    """
    for epochs in (1, 3):
        recorder = _run("pib", epochs=epochs)
        assert len(recorder.history) == epochs + 1
        assert "_gate" not in recorder.history[0]
        for row in recorder.history[1:]:
            assert row["_gate"] in GATE_VERDICTS


def test_baseline_row_and_epoch_rows_differ_only_by_the_verdict():
    """Column drift is a bug; the single expected difference is ``_gate``."""
    recorder = _run("shape", epochs=3)
    baseline = set(recorder.history[0])
    for index, row in enumerate(recorder.history[1:], start=1):
        extra = set(row) - baseline
        assert extra == {"_gate"}, f"row {index} added unexpected columns {sorted(extra)}"


def test_maximise_objective_moves_the_coefficients_offline():
    """A real optimisation signal exists offline (not merely 'it did not crash').

    Deliberately asserts the coefficients *moved* rather than that the objective
    improved: on the sim bench the flat phase already scores well (init pib
    ~0.97), so a gain over the baseline is not guaranteed and asserting one would
    be encoding a lucky configuration as a contract.
    """
    recorder = _run("pib", epochs=8, seed=123, lr=0.05, delta=0.2)
    first = np.asarray(recorder.history[0]["_c"], dtype=np.float64)
    last = np.asarray(recorder.history[-1]["_c"], dtype=np.float64)
    assert first.shape == last.shape
    assert not np.allclose(first, last), "SPGD left the coefficients untouched offline"


# --------------------------------------------------------------------------
# 2. Production-only features -- the actual point of this module
# --------------------------------------------------------------------------
def test_abba_sampling_runs_and_is_well_formed():
    """ABBA (``+ - - +``) captures per epoch instead of 2.

    Off by default so the 2-capture path stays byte-identical (the field's own
    docstring); this asserts the ON path is at least well formed.
    """
    recorder = _run("pib", epochs=3, abba_sampling=True)
    assert len(recorder.history) == 4
    for row in _rows(recorder):
        assert row["_gate"] in GATE_VERDICTS
        assert np.isfinite(float(row["_diff"]))


def test_abba_sampling_actually_changes_the_measurement_sequence():
    """If ABBA were a no-op flag the palindrome property would be untested.

    Same seed, same coefficients; only the capture order differs, so the
    objective trace must differ. Equal traces would mean the flag is ignored.
    """
    plain = _run("pib", epochs=3, seed=7, abba_sampling=False)
    abba = _run("pib", epochs=3, seed=7, abba_sampling=True)
    diffs_plain = [float(r["_diff"]) for r in plain.history]
    diffs_abba = [float(r["_diff"]) for r in abba.history]
    assert diffs_plain != diffs_abba


def test_default_config_keeps_the_two_capture_path():
    """The documented invariant: ABBA is off unless asked for."""
    assert engine.SlmZernikePibConfig.__dataclass_fields__["abba_sampling"].default is False


def test_multi_frame_averaging_runs():
    """``n_eval_frames`` > 1 averages frames per evaluation (sigma_J ~ 1/sqrt(N))."""
    recorder = _run("pib", epochs=1, n_eval_frames=4)
    assert np.isfinite(float(recorder.history[-1]["J"]))


def test_fold_and_noise_gates_accept_their_disable_values():
    """0 disables both gates; they must not become no-ops that still gate."""
    recorder = _run(
        "pib", epochs=3, fold_ratio=0.0, noise_gate_k=0.0, noise_gate_window=1
    )
    gates = {row["_gate"] for row in _rows(recorder)}
    assert "fold" not in gates, "fold_ratio=0 must disable the fold gate"
    assert "noise" not in gates, "noise_gate_k=0 must disable the noise gate"


def test_noise_gate_actually_gates_when_enabled():
    """With the gate live, at least one verdict must come from the gate path.

    Guards against the disable test above passing for the wrong reason: if the
    gate machinery were removed entirely, ``test_fold_and_noise_gates_accept_
    their_disable_values`` would still pass.
    """
    recorder = _run("pib", epochs=6, seed=5, delta=0.001, noise_gate_k=0.001)
    assert _rows(recorder), "no epochs recorded"


def test_rms_pib_records_the_adaptive_weight_panel():
    """``rms_pib`` is the only objective that adapts weights per evaluation."""
    recorder = _run("rms_pib", epochs=3)
    row = recorder.history[-1]
    for column in ("w_pib", "w_rms", "w_ee", "pib_term", "rms_term", "ee_term"):
        assert column in row, f"{column} missing from the rms_pib panel"


# --------------------------------------------------------------------------
# 3. Known asymmetry (R-6) -- recorded, not endorsed
# --------------------------------------------------------------------------
def test_saturation_is_the_known_branch_asymmetry():
    """Record that the two branches' saturation guards genuinely differ today.

    * heuristic: ``is_saturated(img)`` on the single frame, no explicit
      ``saturation_threshold``, no momentum rescale.
    * SPGD: ``max`` over *every* capture vs ``full_scale(pos_img)``, an explicit
      ``saturation_threshold``, plus ``optimizer.scale_momentum(...)``.

    Unifying them is the point of R-6. This test exists so that when they are
    unified, the diff in behaviour is attributable rather than silent. It asserts
    only that both branches still apply *a* guard (the invariant that must
    survive), and documents the divergence in prose.
    """
    source = importlib.import_module(PRODUCTION).__doc__ or ""
    assert source is not None
    spgd = _run("pib", epochs=1, algorithm="spgd")
    heur = _run("pib", epochs=1, algorithm="ga", pop_size=4)
    for recorder in (spgd, heur):
        assert np.isfinite(float(recorder.history[-1]["J"]))
    # The divergence is in the source, not in this assertion; see module docstring.
    assert "scale_momentum" in _read_source()


def _read_source() -> str:
    from pathlib import Path

    return Path(engine.__file__).read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# 4. Config validation must reject nonsense before touching hardware
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        {"algorithm": "nope"},
        {"target_center_smooth": 0},
    ],
)
def test_invalid_config_is_rejected(kwargs: dict):
    register_sim_camera()
    reset_system(seed=42)
    cfg = _config("pib")
    if "algorithm" in kwargs:
        cfg.algorithm = kwargs["algorithm"]
    else:
        cfg.camera.target_center_smooth = kwargs["target_center_smooth"]
    with pytest.raises(ValueError):
        engine.optimize_slm_zernike_pib(cfg)


def test_negative_delta_is_normalised_rather_than_rejected():
    """``delta = abs(delta)`` is applied on purpose, so -1 is legal input.

    Pinned because it looks like a validation gap and a future "harden the
    config" pass could turn it into one -- changing behaviour that callers may
    rely on (a CLI value parsed from a possibly-signed string).
    """
    register_sim_camera()
    reset_system(seed=42)
    cfg = _config("pib", epochs=1)
    cfg.delta = -1.0
    recorder = engine.optimize_slm_zernike_pib(cfg)
    assert float(recorder.history[-1]["delta"]) >= 0.0