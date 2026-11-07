"""Issue #229 implementation tests: structured probes, zernike_block tier, online refeed.

All tests are device-free (CPU torch / numpy only). The hardware experiment
plan in the companion report is explicitly NOT executed here.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from ao_shaping.tools.slm.transfer import structured_probes
from ao_shaping.tools.slm.transfer import zernike_block
from ao_shaping.tools.slm.transfer.online_augment import (
    BlockRecord, ModelSwitchGate, SessionDataset, should_refit,
)
from ao_shaping.tools.slm.transfer.structured_probes import (
    AMPLITUDE_LADDER, DEFAULT_MODES, cross_response_matrix, frame_budget,
    flat_field_plan, random_probe_budget_match, structured_phase,
)
from ao_shaping.tools.slm.transfer.structured_probes import structured_probe_plan


# ---------------------------------------------------------------------------
# structured_probe_plan / frame budget
# ---------------------------------------------------------------------------

def test_plan_frame_budget_full_and_compressed():
    plan = structured_probe_plan(repeats=3)
    assert len(plan) == 7 * 3 * 2 * 3
    assert frame_budget(repeats=3) == 126 * 4 + 8 == 512
    plan_r1 = structured_probe_plan(repeats=1)
    assert len(plan_r1) == 7 * 3 * 2
    assert frame_budget(repeats=1) == 42 * 4 + 8 == 176


def test_plan_split_assignment():
    plan = structured_probe_plan(repeats=2)
    by_split: dict[str, int] = {}
    for row in plan:
        by_split[row["split"]] = by_split.get(row["split"], 0) + 1
    # Noll 9: whole mode holdout (3 amps x 2 signs x 2 repeats = 12)
    assert by_split["holdout"] == 12
    # 6 remaining modes: highest level (0.10) is validation (2 signs x 2 reps)
    assert by_split["validation"] == 6 * 2 * 2
    assert by_split["train"] == 6 * 2 * 2 * 2
    # validation level is exactly the third amplitude, holdout is Noll 9 only
    for row in plan:
        if row["split"] == "validation":
            assert row["amplitude"] == AMPLITUDE_LADDER[-1]
            assert row["mode"] != 9
        if row["mode"] == 9:
            assert row["split"] == "holdout"
    # sign pairing is complete
    signs = {(row["mode"], row["amplitude"], row["repeat"]) for row in plan}
    for (mode, amp, rep) in signs:
        positives = [r for r in plan if (r["mode"], r["amplitude"], r["repeat"]) == (mode, amp, rep)
                     and r["sign"] == 1]
        negatives = [r for r in plan if (r["mode"], r["amplitude"], r["repeat"]) == (mode, amp, rep)
                     and r["sign"] == -1]
        assert len(positives) == 1 and len(negatives) == 1


def test_flat_field_plan_is_abba_pairs():
    plan = flat_field_plan(2)
    assert len(plan) == 8  # two ABBA = 8 frames
    assert all(row["split"] == "train" for row in plan)


def test_plan_validation_inputs():
    with pytest.raises(ValueError):
        structured_probe_plan(modes=())
    with pytest.raises(ValueError):
        structured_probe_plan(repeats=0)
    with pytest.raises(ValueError):
        structured_probe_plan(holdout_mode=99)


# ---------------------------------------------------------------------------
# structured_phase / cross_response_matrix
# ---------------------------------------------------------------------------

def test_structured_phase_single_mode_bounded(size=32):
    row = dict(mode=7, amplitude=0.05, sign=1)
    phase = structured_phase(row, size)
    assert phase.shape == (size, size)
    assert phase.dtype == np.float32
    # The Zernike basis is R-normalized: the pattern peak is amplitude × the
    # unit-mode peak (a fixed number, independent of amplitude).
    unit = structured_phase(dict(mode=7, amplitude=1.0, sign=1), size)
    np.testing.assert_allclose(phase, 0.05 * unit, rtol=1e-5)
    # sign flip negates the pattern (Zernike basis is linear)
    row_neg = dict(mode=7, amplitude=0.05, sign=-1)
    np.testing.assert_allclose(structured_phase(row_neg, size), -phase, rtol=1e-6)


def test_cross_response_matrix_orthogonal_modes():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, (16, 16)) ** 2
    modal, v = {}, []
    for i in range(4):
        vec = np.zeros((16, 16))
        vec[i * 4:(i + 1) * 4, i * 4:(i + 1) * 4] = 1.0
        v.append(vec)
        modal[5 + i] = base + vec * 2.0
    matrix, modes = cross_response_matrix(base, modal)
    assert modes == [5, 6, 7, 8]
    assert matrix.shape == (4, 4)
    np.testing.assert_allclose(np.diag(matrix), 1.0, atol=1e-12)
    off = matrix - np.eye(4)
    assert np.abs(off).max() < 1e-12


def test_cross_response_matrix_leaky_modes_show_cross_leakage():
    base = np.zeros((8, 8))
    a = np.ones((8, 8))
    # mode 6 response is a rotated copy of mode 5's -> high off-diagonal
    modal = {5: base + a, 6: base + np.rot90(a)}
    matrix, _ = cross_response_matrix(base, modal)
    assert abs(matrix[0, 1]) > 0.9


# ---------------------------------------------------------------------------
# zernike_block tier
# ---------------------------------------------------------------------------

def _fake_model():
    pytest.importorskip("torch")
    import torch

    params = {name: torch.nn.Parameter(torch.zeros(2))
              for name in ("d", "log_width", "center", "origin")}
    residual = torch.nn.Parameter(torch.zeros(7))
    return type("M", (), {**params, "residual": residual,
                           "residual_modes": [5, 6, 7, 8, 9, 10, 11],
                           "optimizer": None})()


def test_configure_zernike_block_releases_only_selected():
    pytest.importorskip("torch")
    model = _fake_model()
    # illumination released (plan default) — center/origin/width move
    zernike_block.configure_zernike_block(model, [7, 9], release_amplitude=True)
    assert model.center.requires_grad and model.origin.requires_grad
    assert model.log_width.requires_grad is True
    assert model.d.requires_grad is False
    assert model.residual.requires_grad is True
    mask = zernike_block.residual_grad_mask(model, [7, 9])
    assert mask.tolist() == [False, False, True, False, True, False, False]
    # optional illumination: log_width stays frozen
    model2 = _fake_model()
    zernike_block.configure_zernike_block(model2, [7, 9], release_amplitude=False)
    assert model2.log_width.requires_grad is False
    with pytest.raises(ValueError):
        zernike_block.configure_zernike_block(model, [])
    with pytest.raises(ValueError):
        zernike_block.configure_zernike_block(model, [50])


def test_mask_residual_gradient_zeroes_unselected():
    pytest.importorskip("torch")
    import torch

    model = _fake_model()
    model.residual = torch.nn.Parameter(torch.zeros(7), requires_grad=True)
    model.residual.grad = torch.ones(7)  # a nontrivial incoming gradient
    mask = np.array([False, False, True, False, False, False, False])
    zernike_block.mask_residual_gradient(model, mask)
    grad = model.residual.grad.numpy()
    assert grad[2] == 1.0  # selected mode keeps its gradient
    np.testing.assert_array_equal(grad[[0, 1, 3, 4, 5, 6]], 0.0)  # masked modes zeroed


def test_clamp_residual_block_projects_to_cap():
    pytest.importorskip("torch")
    import torch

    model = _fake_model()
    with torch.no_grad():
        model.residual.normal_()
    assert model.residual.norm() > 1.0
    pre = zernike_block.clamp_residual_block(model, cap_rad=0.5)
    assert pre > 0.5
    assert float(model.residual.detach().norm()) == pytest.approx(0.5, abs=1e-6)
    # under-cap vectors are untouched
    with torch.no_grad():
        model.residual.mul_(0.1)
    before = model.residual.detach().clone()
    zernike_block.clamp_residual_block(model)
    np.testing.assert_allclose(model.residual.detach().numpy(), before.numpy(), atol=1e-7)


def test_identifiability_gate_selects_and_rejects():
    # identity cross matrix: all modes orthogonal
    matrix = np.eye(4)
    corr = {5: 0.9, 6: 0.6, 7: 0.2, 8: 0.99}
    verdict = zernike_block.identifiability_gate(matrix, corr)
    assert verdict["selected"] == [5, 6, 8]
    assert verdict["rejected"] == {7: 0.2}
    assert verdict["matrix_ok"] is True
    assert verdict["diag_off_ratio"] == pytest.approx(float("inf"), abs=1.0) or verdict["diag_off_ratio"] > 3
    # an unidentifiable mode drops out without breaking the matrix gate
    assert 7 not in verdict["selected"]


def test_identifiability_gate_rejects_coupled_submatrix():
    # 45-degree-coupled 2-mode block: off-diagonal as large as diagonal
    matrix = np.array([[1.0, 0.99, 0.0, 0.0],
                       [0.99, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0, 1.0]])
    corr = {5: 0.9, 6: 0.9, 7: 0.9, 8: 0.9}
    verdict = zernike_block.identifiability_gate(matrix, corr)
    assert verdict["selected"] == [5, 6, 7, 8]
    assert verdict["matrix_ok"] is False  # diagonal/off-diagonal ratio < 3
    # drop the coupled pair -> the gate passes on the decoupled sub-matrix
    sub_matrix = matrix[np.ix_([0, 2, 3], [0, 2, 3])]
    verdict3 = zernike_block.identifiability_gate(sub_matrix, {5: 0.9, 7: 0.9, 8: 0.9})
    assert verdict3["selected"] == [5, 7, 8]
    assert verdict3["matrix_ok"] is True
    # dimension mismatch is rejected loudly
    with pytest.raises(ValueError):
        zernike_block.identifiability_gate(matrix, {5: 0.9, 7: 0.9, 8: 0.9})


# ---------------------------------------------------------------------------
# online augment
# ---------------------------------------------------------------------------

def _block(i, accepted=True):
    return BlockRecord(block=i, phase_anchor=np.zeros((4, 4)), phase_candidate=np.ones((4, 4)),
                       frames={k: np.full((4, 4), float(i)) for k in ("A1", "B1", "B2", "A2")},
                       accepted=accepted, exposure_ms=1.0)


def test_session_dataset_refeed_and_isolation():
    ds = SessionDataset()
    ds.seed([dict(split="train"), dict(split="validation"), dict(split="holdout")])
    assert ds.counts()["augment"] == 0
    ds.add_block(_block(1))
    ds.add_block(_block(2, accepted=False))  # rejected blocks refed too
    assert ds.counts()["augment"] == 8
    # refeed pool is train/validation/augment; holdout is never eligible
    pool = ds.frames()
    assert len(pool) == 10  # 1 train + 1 validation + 8 augment
    assert all(f["split"] != "holdout" for f in pool)
    augment = ds.frames(splits=("augment",))
    assert len(augment) == 8
    rejected = [f for f in augment if f["block"] == 2]
    assert len(rejected) == 4
    assert all(f["accepted"] is False for f in rejected)


def test_session_dataset_rejects_incomplete_block():
    ds = SessionDataset()
    bad = BlockRecord(block=1, phase_anchor=np.zeros((4, 4)), phase_candidate=np.ones((4, 4)),
                      frames={"A1": np.zeros((4, 4))}, accepted=True, exposure_ms=1.0)
    with pytest.raises(ValueError):
        ds.add_block(bad)


def test_should_refit_cadence():
    assert not should_refit(1) and not should_refit(2)
    assert should_refit(3) and not should_refit(4) and should_refit(6)
    with pytest.raises(ValueError):
        should_refit(0)
    with pytest.raises(ValueError):
        should_refit(3, every=0)


def test_model_switch_gate_pass_fail_rebuild():
    gate = ModelSwitchGate()
    gate.incumbent_validation_nrmse = 0.5
    ok = gate.check(0.45)  # improvement
    assert ok["passed"] and ok["consecutive_failures"] == 0
    assert gate.incumbent_validation_nrmse == pytest.approx(0.45)
    bad = gate.check(0.62)  # 38% degradation > 10%
    assert not bad["passed"] and gate.consecutive_failures == 1
    bad2 = gate.check(0.60)  # second failure -> rebuild
    assert not bad2["passed"] and gate.rebuild_required is True
    # a passing check after a rebuild resets the counter
    gate.incumbent_validation_nrmse = 0.5
    gate.check(0.505)  # 1% degradation, still < 10% -> pass
    assert gate.consecutive_failures == 0


def test_model_switch_gate_guard_regression_blocks():
    gate = ModelSwitchGate()
    gate.incumbent_validation_nrmse = 0.5
    gate.incumbent_guard_correlation = 0.8
    verdict = gate.check(0.51, next_guard_correlation=0.3)
    assert verdict["passed"] is False
    assert verdict["nrmse_ok"] is True and verdict["guard_ok"] is False


# ---------------------------------------------------------------------------
# end-to-end: zernike_block fit on the synthetic dataset
# ---------------------------------------------------------------------------

def test_zernike_block_fit_masks_unselected_modes(tmp_path):
    pytest.importorskip("torch")
    from types import SimpleNamespace

    from ao_shaping.tools.slm.transfer import fit as fit_module
    from ao_shaping.tools.slm.transfer.simulation import generate_dataset

    dataset = tmp_path / "synthetic"
    checkpoint = generate_dataset(dataset, power=1., count=6)
    out = tmp_path / "block"
    fit_module.fit(SimpleNamespace(
        dataset=dataset, checkpoint=checkpoint, out=out, iterations=8, span=120,
        device="cpu", adaptation="zernike_block", selected_modes=[7, 8],
        residual_block_cap=0.5, radiometry="exposure", samples=None,
        patience=8, min_delta=1e-4))
    result = json.loads((out / "result.json").read_text())
    residual = result["parameters"]["residual_noll_radians"]
    # selected modes may move; unselected modes are masked to stay at 0
    for mode in ("5", "6", "9", "10", "11"):
        assert abs(float(residual[mode])) < 1e-5, f"mode {mode} moved: {residual[mode]}"
    # block L2 stays under the cap (or at the cap)
    vec = [float(v) for v in residual.values()]
    assert float(np.linalg.norm(vec)) <= 0.5 + 1e-6
    provenance = json.loads((out / "provenance.json").read_text())
    assert provenance["adaptation"] == "zernike_block"
    assert provenance["selected_modes"] == [7, 8]
    assert provenance["residual_block_cap"] == 0.5


def test_zernike_block_requires_modes(tmp_path):
    pytest.importorskip("torch")
    from types import SimpleNamespace

    from ao_shaping.tools.slm.transfer import fit as fit_module
    from ao_shaping.tools.slm.transfer.simulation import generate_dataset

    dataset = tmp_path / "synthetic"
    checkpoint = generate_dataset(dataset, power=1., count=6)
    with pytest.raises(ValueError, match="selected-modes"):
        fit_module.fit(SimpleNamespace(
            dataset=dataset, checkpoint=checkpoint, out=tmp_path / "x", iterations=2,
            span=120, device="cpu", adaptation="zernike_block", selected_modes=None,
            residual_block_cap=0.5, radiometry="exposure", samples=None,
            patience=8, min_delta=1e-4))


def test_random_probe_budget_match_count_and_shape():
    phases = random_probe_budget_match(n_modes=7, n_frames=28, size=32, seed=7)
    assert len(phases) == 28
    assert all(p.shape == (32, 32) for p in phases)
    with pytest.raises(ValueError):
        random_probe_budget_match(n_modes=7, n_frames=30, size=32, seed=7)
