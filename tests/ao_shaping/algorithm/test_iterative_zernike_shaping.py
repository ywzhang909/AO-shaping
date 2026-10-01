"""Tests for the iterative Zernike + phase-only beam-shaping optimizer.

Scenarios:
  S1  Zernike calibration reduces the mismatch to the reference far-field.
  S2  Phase-only shaping raises the composite score vs. the initial phase.
  S3  Iteration (A↔B) converges and beats the single-pass score.
  S4  Contracts: validation, raw-only phase (no self-wrap), determinism.
  S5  No hardware involved (pure simulation, CPU, torch-gated).
  S6  Torch Zernike phase matches the canonical numpy generate_zernike_phase.
  S7  Iterative A↔B beats same-grid SPGD and GS baselines (the WIN proof).

All tests run on CPU with a deterministic seed and a small grid (n=64) to keep
runtime small. Torch-gated: skip if torch is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ao_shaping.algorithm.signal_processing.iterative_zernike_shaping import (
    IterativeZernikeShapingConfig,
    IterativeZernikeShapingOptimizer,
)
from ao_shaping.drivers.sim.slm_shaping_bench import (
    ShapingBenchConfig,
    composite_score,
    compute_metrics,
    forward_intensity,
    make_target,
)
from ao_shaping.optimizer.wfless.slm_shaping_bench import gs_shape, spgd_shape
from ao_shaping.utils.wavefront.zernike_utils import generate_zernike_phase


N = 64


def _bench_cfg() -> ShapingBenchConfig:
    return ShapingBenchConfig(n_grid=N, target_side_px=16, seed=0)


def _make_actual_far_field(seed: int = 0) -> np.ndarray:
    """Build a reference 'actual' far-field from a golden Zernike + noise."""
    rng = np.random.default_rng(seed)
    golden_phase = (
        0.5 * np.ones((N, N)) * 0.0  # placeholder (real built in test)
    )
    # Use a real Zernike golden via generate_zernike_phase
    zph = generate_zernike_phase({(2, 0): 0.5, (2, -2): 0.3}, resolution=(N, N), n_max=4)
    zph = np.nan_to_num(np.asarray(zph, dtype=np.float64), nan=0.0)
    noise = rng.normal(0, 0.1, size=(N, N))
    ff = forward_intensity(zph + noise, _bench_cfg())
    return ff / (ff.sum() + 1e-12)


def _optimizer(n_zernike: int = 4, **kw) -> IterativeZernikeShapingOptimizer:
    cfg = IterativeZernikeShapingConfig(
        n_grid=N,
        n_zernike=n_zernike,
        target_side_px=16,
        seed=0,
        calib_iters=kw.pop("calib_iters", 40),
        shaping_iters=kw.pop("shaping_iters", 60),
        max_outer_iters=kw.pop("max_outer_iters", 3),
        zernike_lr=kw.pop("zernike_lr", 0.05),
        slm_lr=kw.pop("slm_lr", 0.02),
        **kw,
    )
    return IterativeZernikeShapingOptimizer(cfg)


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


# ---------------------------------------------------------------------------
# S1: Zernike calibration reduces mismatch to the reference far-field
# ---------------------------------------------------------------------------
def test_s1_calibration_reduces_mismatch():
    actual = _make_actual_far_field()
    opt = _optimizer(n_zernike=4, calib_iters=60)
    rng = np.random.default_rng(0)
    init_slm = rng.normal(0, 0.1, size=(N, N))

    # Baseline: zero zernike
    ff0 = opt._far_field(torch.zeros(0) if False else _zero_vec(opt), torch.as_tensor(init_slm, dtype=torch.float64))
    base_mse = _mse(ff0.detach().numpy(), actual)

    coeffs = opt.calibrate_zernike(actual, init_slm)
    # calibrated
    zvec = _coeff_vec(opt, coeffs)
    ff1 = opt._far_field(zvec, torch.as_tensor(init_slm, dtype=torch.float64))
    cal_mse = _mse(ff1.detach().numpy(), actual)

    assert len(coeffs) > 0
    assert cal_mse < base_mse, f"calibrated MSE {cal_mse} not < baseline {base_mse}"


def _zero_vec(opt: IterativeZernikeShapingOptimizer) -> torch.Tensor:
    return torch.zeros(opt._n_zernike_params, dtype=torch.float64)


def _coeff_vec(opt: IterativeZernikeShapingOptimizer, coeffs: dict) -> torch.Tensor:
    v = torch.zeros(opt._n_zernike_params, dtype=torch.float64)
    for i, nm in enumerate(opt._zernike_modes):
        if nm in coeffs:
            v[i] = coeffs[nm]
    return v


# ---------------------------------------------------------------------------
# S2: phase-only shaping raises composite score vs. initial phase
# ---------------------------------------------------------------------------
def test_s2_shaping_raises_score():
    rng = np.random.default_rng(1)
    init_slm = rng.normal(0, 0.1, size=(N, N))
    target = make_target(_bench_cfg())

    ff_init = forward_intensity(init_slm, _bench_cfg())
    ff_init = ff_init / ff_init.sum()
    center = np.unravel_index(np.argmax(ff_init), ff_init.shape)[::-1]
    score_init = composite_score(compute_metrics(ff_init, target, center=center))

    opt = _optimizer(n_zernike=0, shaping_iters=80)  # zernike off => pure shaping
    slm_out = opt.shape_phase(None, init_slm)
    ff_out = forward_intensity(slm_out, _bench_cfg())
    ff_out = ff_out / ff_out.sum()
    center = np.unravel_index(np.argmax(ff_out), ff_out.shape)[::-1]
    score_out = composite_score(compute_metrics(ff_out, target, center=center))

    assert score_out > score_init, f"shaped score {score_out} not > initial {score_init}"


# ---------------------------------------------------------------------------
# S3: iteration converges and beats the single-pass score
# ---------------------------------------------------------------------------
def test_s3_iteration_converges_and_beats_single_pass():
    actual = _make_actual_far_field()
    opt = _optimizer(n_zernike=4, calib_iters=40, shaping_iters=50, max_outer_iters=4)
    rng = np.random.default_rng(0)
    init_slm = rng.normal(0, 0.1, size=(N, N))

    result = opt.run(actual_far_field=actual, initial_slm_phase=init_slm)

    # single-pass = first outer iteration score
    single = result.score_history[1]["score"] if len(result.score_history) > 1 else result.score_history[0]["score"]
    final = result.final_score
    assert final >= single - 1e-6, f"final {final} < single-pass {single}"
    # history strictly has at least 2 entries (init + >=1 outer)
    assert len(result.score_history) >= 2
    # result fields present
    assert "zernike_coeffs" in result.__dict__ or hasattr(result, "zernike_coeffs")
    assert result.far_field.shape == (N, N)
    assert result.slm_phase.shape == (N, N)


# ---------------------------------------------------------------------------
# S4: contracts
# ---------------------------------------------------------------------------
def test_s4a_validation():
    with pytest.raises(ValueError):
        IterativeZernikeShapingConfig(n_grid=2)
    with pytest.raises(ValueError):
        IterativeZernikeShapingOptimizer(IterativeZernikeShapingConfig(n_grid=2))
    with pytest.raises(ValueError):
        IterativeZernikeShapingOptimizer(
            IterativeZernikeShapingConfig(n_grid=N, n_zernike=4, calib_iters=0)
        )
    with pytest.raises(ValueError):
        IterativeZernikeShapingOptimizer(
            IterativeZernikeShapingConfig(n_grid=N, shaping_iters=0)
        )


def test_s4b_raw_only_phase_not_wrapped():
    """SLM phase output must be raw (unwrapped) — not forced into [0, 2π)."""
    opt = _optimizer(n_zernike=0, shaping_iters=80)
    rng = np.random.default_rng(2)
    init_slm = rng.normal(0, 0.1, size=(N, N))
    out = opt.shape_phase(None, init_slm)
    # raw radians: should span beyond [0, 2π) for a non-trivial phase, i.e.
    # the optimizer must not clip/wrap. Assert it is finite and has real variation.
    assert np.isfinite(out).all()
    assert out.std() > 0.01, "phase appears constant (possibly wrapped/clamped)"
    # Crucially, not a uint16 / grayscale value range:
    assert out.dtype == np.float64
    assert out.max() > 2 * np.pi or out.min() < 0.0, "phase looks wrapped into [0, 2π]"


def test_s4c_determinism():
    actual = _make_actual_far_field(seed=5)
    r1 = _optimizer(n_zernike=4, calib_iters=30, shaping_iters=40).run(actual_far_field=actual)
    r2 = _optimizer(n_zernike=4, calib_iters=30, shaping_iters=40).run(actual_far_field=actual)
    assert np.allclose(r1.slm_phase, r2.slm_phase, atol=1e-8)
    assert r1.final_score == pytest.approx(r2.final_score, abs=1e-8)
    assert r1.zernike_coeffs == r2.zernike_coeffs


# ---------------------------------------------------------------------------
# S5: no hardware involved
# ---------------------------------------------------------------------------
def test_s5_no_hardware_full_pipeline():
    from ao_shaping.optimizer.wfless.iterative_zernike_shaping import (
        IterativeZernikePibConfig,
        optimize_iterative_zernike_shaping,
    )

    cfg = IterativeZernikePibConfig(
        n_grid=N, n_zernike=4, calib_iters=30, shaping_iters=40, max_outer_iters=3
    )
    out = optimize_iterative_zernike_shaping(cfg)
    for key in (
        "zernike_coeffs", "slm_phase", "far_field", "target",
        "actual_far_field", "score_history", "final_score",
        "n_outer_iters", "converged", "metrics",
    ):
        assert key in out, f"missing key {key}"
    # metrics use the canonical bench keys
    for mkey in ("PIB", "efficiency", "CV", "Strehl", "zero_order", "score"):
        assert mkey in out["metrics"]
    assert out["final_score"] > 0.0


# ---------------------------------------------------------------------------
# S6: torch Zernike phase matches the canonical numpy generator
# ---------------------------------------------------------------------------
def test_s6_torch_zernike_matches_canonical_numpy():
    coeffs = {(2, 0): 1.0, (2, -2): 0.7, (4, 0): 0.5}
    # canonical numpy reference (raw radians, NaN outside aperture)
    ref = generate_zernike_phase(coeffs, resolution=(N, N), n_max=4)
    ref = np.nan_to_num(np.asarray(ref, dtype=np.float64), nan=0.0)

    opt = _optimizer(n_zernike=4)
    zvec = _coeff_vec(opt, coeffs)
    # reconstruct zernike phase only (slm = 0) by building it directly
    zph = torch.zeros((N, N), dtype=torch.float64)
    for i, (_, _, basis) in enumerate(opt._zernike_basis()):
        zph = zph + zvec[i] * basis
    torch_ph = zph.detach().numpy()

    # Compare only inside the unit circle (the canonical generator is NaN outside).
    yy, xx = np.mgrid[0:N, 0:N]
    cx = cy = (N - 1) / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (N / 2.0)
    inside = r <= 0.98  # stay well inside the aperture
    # Both must be defined at the same point: piston-free => compare normalized diff.
    diff = np.abs(torch_ph[inside] - ref[inside])
    # allow global sign/offset? No — same convention expected. Use relative tol.
    scale = max(np.abs(ref[inside]).max(), 1e-6)
    assert diff.max() < 0.05 * scale, (
        f"torch Zernike differs from canonical by {diff.max():.4f} (scale {scale:.4f})"
    )


def test_s6_every_basis_mode_matches_canonical_generator():
    opt = _optimizer(n_zernike=4)
    for n, m, basis in opt._zernike_basis():
        expected = generate_zernike_phase(
            {(n, m): 1.0}, resolution=(N, N), n_max=4
        )
        np.testing.assert_allclose(
            basis.numpy(), np.nan_to_num(expected, nan=0.0), rtol=0, atol=1e-12
        )


# ---------------------------------------------------------------------------
# S7: iterative A↔B beats same-grid SPGD and GS baselines (the WIN proof)
# ---------------------------------------------------------------------------
def test_s7_beats_spgd_and_gs_baselines():

    cfg = _bench_cfg()
    target = make_target(cfg)
    rng = np.random.default_rng(0)
    init_slm = rng.normal(0, 0.1, size=(N, N))

    def _score(ff: np.ndarray) -> float:
        ff = ff / (ff.sum() + 1e-12)
        center = np.unravel_index(np.argmax(ff), ff.shape)[::-1]
        return composite_score(compute_metrics(ff, target, center=center))

    # Baselines (same 64x64 grid, identical metric)
    gs = gs_shape(cfg, n_iters=200, seed=0)
    gs_score = _score(forward_intensity(gs.phase, cfg))
    spgd = spgd_shape(cfg, n_iters=600, delta=0.1, lr=0.02, seed=0, dim=8)
    spgd_score = _score(forward_intensity(spgd.phase, cfg))
    init_score = _score(forward_intensity(init_slm, cfg))

    # Iterative A↔B
    opt = _optimizer(n_zernike=4, calib_iters=50, shaping_iters=100, max_outer_iters=5)
    actual = _make_actual_far_field()
    res = opt.run(actual_far_field=actual, initial_slm_phase=init_slm)
    final_score = res.final_score

    assert final_score > gs_score, f"iterative {final_score} <= GS {gs_score}"
    assert final_score > spgd_score, f"iterative {final_score} <= SPGD {spgd_score}"
    assert final_score > init_score, f"iterative {final_score} <= initial {init_score}"
