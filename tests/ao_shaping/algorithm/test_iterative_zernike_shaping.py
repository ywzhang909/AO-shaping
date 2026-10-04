"""Tests for the iterative Zernike + phase-only beam-shaping optimizer.

Scenarios:
  S1  Zernike calibration reduces the mismatch to the reference far-field.
  S2  Phase-only shaping raises the composite score vs. the initial phase.
  S3  Iteration (A↔B) converges and matches/exceeds its own single-pass score.
  S4  Contracts: validation, raw-only phase (no self-wrap), determinism.
  S5  No hardware involved (pure simulation, CPU, torch-gated).
  S6  Torch Zernike phase matches the canonical numpy generate_zernike_phase.
  S7  The objective is a single source of truth (algorithm _score == bench
      composite_score) and flat-top shaping (GS) beats spot concentration
      (SPGD) on uniformity -- the physically meaningful discrimination.
  S8  Forward-model regression anchor: the initial far field is a single
      focused spot (NOT the aliased lattice that an un-padded same-size FFT
      produced, which had hundreds of local maxima).

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
from ao_shaping.drivers.sim.slm_shaping_bench import gs_shape, spgd_shape
from ao_shaping.utils.wavefront.zernike_utils import generate_zernike_phase

N = 64
PAD = 8
TARGET_SIDE = 43
GOLDEN = {(2, 0): 0.94, (2, -2): 0.63, (4, 0): 0.63}


def _bench_cfg() -> ShapingBenchConfig:
    return ShapingBenchConfig(
        n_grid=N, target_side_px=TARGET_SIDE, far_field_padding=PAD, seed=0
    )


def _make_actual_far_field() -> np.ndarray:
    """Build the reference 'actual' far-field (golden Zernike, flat SLM phase)."""
    zph = generate_zernike_phase(GOLDEN, resolution=(N, N), n_max=4)
    zph = np.nan_to_num(np.asarray(zph, dtype=np.float64), nan=0.0)
    ff = forward_intensity(zph, _bench_cfg())
    return ff / (ff.sum() + 1e-12)


def _optimizer(n_zernike: int = 4, **kw) -> IterativeZernikeShapingOptimizer:
    cfg = IterativeZernikeShapingConfig(
        n_grid=N,
        n_zernike=n_zernike,
        target_side_px=TARGET_SIDE,
        far_field_padding=PAD,
        seed=0,
        calib_iters=kw.pop("calib_iters", 40),
        shaping_iters=kw.pop("shaping_iters", 60),
        max_outer_iters=kw.pop("max_outer_iters", 3),
        zernike_lr=kw.pop("zernike_lr", 0.005),
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
    init_slm = np.zeros((N, N))

    # Baseline: zero zernike
    ff0 = opt._far_field(_zero_vec(opt), torch.as_tensor(init_slm, dtype=torch.float64))
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
    score_init = composite_score(compute_metrics(ff_init, target))

    opt = _optimizer(n_zernike=0, shaping_iters=80)  # zernike off => pure shaping
    slm_out = opt.shape_phase(None, init_slm)
    ff_out = forward_intensity(slm_out, _bench_cfg())
    ff_out = ff_out / ff_out.sum()
    score_out = composite_score(compute_metrics(ff_out, target))

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
    m = _bench_cfg().far_field_size
    assert result.far_field.shape == (m, m)
    assert result.target.shape == (m, m)
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
    actual = _make_actual_far_field()
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
# S7: objective single source of truth + flat-top vs concentration
# ---------------------------------------------------------------------------
def test_s7_objective_consistency_and_shaping_discrimination():
    """The algorithm ``_score`` must equal the bench ``composite_score``, and
    flat-top shaping (GS) must beat spot concentration (SPGD) on uniformity.

    NOTE: the previous "iterative A<->B beats GS and SPGD" claim does NOT hold
    on the corrected (zero-padded, un-aliased) model once the uniformity term
    is non-degenerate; it was an artefact of the aliased forward field plus a
    uniformly-saturated CV term. See ``docs/iterative_zernike_shaping/data.json``.
    """
    from ao_shaping.drivers.sim.slm_shaping_bench import spgd_shape

    cfg = _bench_cfg()
    target = make_target(cfg)

    def _score(ff: np.ndarray) -> float:
        ff = ff / (ff.sum() + 1e-12)
        return composite_score(compute_metrics(ff, target))

    def _metrics(intensity: np.ndarray):
        intensity = intensity / intensity.sum()
        return compute_metrics(intensity, target)

    gs = gs_shape(cfg, n_iters=200, seed=0)
    gs_m = _metrics(gs.intensity)
    spgd = spgd_shape(cfg, n_iters=600, delta=0.1, lr=0.02, seed=0, dim=8)
    spgd_m = _metrics(spgd.intensity)

    assert gs_m["CV"] < spgd_m["CV"], f"GS CV {gs_m['CV']} >= SPGD CV {spgd_m['CV']}"
    assert _score(gs.intensity) > _score(spgd.intensity)

    opt = _optimizer(n_zernike=0)
    ff = gs.intensity / gs.intensity.sum()
    alg_score = float(opt._score(torch.as_tensor(ff, dtype=torch.float64)).item())
    assert alg_score == pytest.approx(_score(gs.intensity), abs=1e-6)


# ---------------------------------------------------------------------------
# S8: forward-model regression anchor -- the initial spot is not a lattice
# ---------------------------------------------------------------------------
def test_s8_initial_spot_is_single_not_lattice():
    """The initial far field must be a single focused spot, not a dot lattice.

    An un-padded same-size FFT sampled the focal plane at ~1.1 px per waist
    radius (a model constant) and aliased the lens-phase-modulated pupil into
    hundreds of local maxima. The zero-padded Fraunhofer FFT must give one peak.
    """
    from scipy import ndimage

    actual = _make_actual_far_field()
    ff = actual / actual.max()
    maxima = ndimage.maximum_filter(ff, size=3)
    n_peaks = int(((ff == maxima) & (ff > 0.01)).sum())
    assert n_peaks <= 3, f"initial spot has {n_peaks} local maxima (aliased lattice?)"
    peak = np.unravel_index(np.argmax(ff), ff.shape)[::-1]
    centre = ff.shape[0] // 2
    assert abs(peak[0] - centre) <= 1 and abs(peak[1] - centre) <= 1, peak


# ---------------------------------------------------------------------------
# S9: GS warm start + free-form refinement beats GS (the saved result)
# ---------------------------------------------------------------------------
def test_s9_warm_started_refinement_beats_gs():
    """The refinement loop, warm-started from GS, must beat plain GS.

    This locks the earlier failure: with a zero start, or with a warm start
    corrupted by a frozen Zernike, Stage B only tied or worsened GS.
    """
    from ao_shaping.optimizer.wfless.iterative_zernike_shaping import (
        IterativeZernikePibConfig,
        optimize_iterative_zernike_shaping,
    )

    cfg = _bench_cfg()
    target = make_target(cfg)
    gs = gs_shape(cfg, n_iters=100, seed=0)
    gs_ff = forward_intensity(gs.phase, cfg)
    gs_ff = gs_ff / gs_ff.sum()
    gs_score = composite_score(compute_metrics(gs_ff, target))

    out = optimize_iterative_zernike_shaping(
        IterativeZernikePibConfig(
            n_grid=N,
            n_zernike=0,
            target_side_px=TARGET_SIDE,
            far_field_padding=PAD,
            seed=0,
            gs_warm_start=True,
            gs_iters=100,
            calib_iters=20,
            shaping_iters=40,
            max_outer_iters=3,
        )
    )
    score = out["metrics"]["score"]
    assert score > gs_score, f"warm-started refinement {score} <= GS {gs_score}"


# ---------------------------------------------------------------------------
# S10: Zernike calibration stays finite and beats the zero-coefficient baseline
# ---------------------------------------------------------------------------
def test_s10_calibration_is_finite_and_reduces_mismatch():
    """Calibration must not diverge to NaN and must lower the far-field MSE.

    A large ``zernike_lr`` used to blow the coefficients up to non-finite values
    on the rugged far-field MSE landscape; the default is now small and the loop
    bails out on the first non-finite iterate.
    """
    actual = _make_actual_far_field()
    init = np.zeros((N, N))
    opt = _optimizer(n_zernike=4, calib_iters=200)
    coeffs = opt.calibrate_zernike(actual, init)
    assert coeffs, "calibration returned no coefficients"
    for mode, value in coeffs.items():
        assert np.isfinite(value), f"non-finite coefficient for {mode}: {value}"

    zero = _zero_vec(opt)
    m_base = _mse(opt._far_field(zero, torch.as_tensor(init, dtype=torch.float64)).numpy(), actual)
    m_cal = _mse(
        opt._far_field(_coeff_vec(opt, coeffs), torch.as_tensor(init, dtype=torch.float64)).numpy(),
        actual,
    )
    assert m_cal < m_base, f"calibrated MSE {m_cal} not < baseline {m_base}"
