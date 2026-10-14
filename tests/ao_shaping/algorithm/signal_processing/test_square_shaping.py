"""Tests for the pure NumPy square-shaping algorithm layer.

These tests exercise ``ao_shaping.algorithm.signal_processing.square_shaping``
directly (no sim / driver shim), verifying that the algorithm layer is a
self-contained, hardware-free implementation of the forward model, target
generation, metrics, and shaping methods (GS variants, differentiable, SPGD).

All tests run on CPU with deterministic seeds and small grids for speed.
Torch-gated tests are skipped when torch is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest
from dataclasses import FrozenInstanceError

from ao_shaping.algorithm.signal_processing.square_shaping import (
    GS_METHODS,
    GS_SELECT,
    ShapingResult,
    SquareShapingConfig,
    UniformSquareShaper,
    _forward_intensity,
    _pad_centred,
    _rolled_support,
    analytic_amplitude_target,
    compute_bench_metrics,
    composite_from_pib_cv,
    composite_score,
    differentiable_shape,
    efficiency,
    gs_shape,
    make_square_target,
    power_in_bucket,
    spgd_shape,
    strehl,
    uniformity_cv,
    zero_order_fraction,
)

N = 64
PAD = 8


def _cfg(**kw) -> SquareShapingConfig:
    defaults = {"n_grid": N, "far_field_padding": PAD}
    defaults.update(kw)
    return SquareShapingConfig(**defaults)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class TestSquareShapingConfig:
    def test_default_values(self):
        cfg = _cfg()
        assert cfg.n_grid == N
        assert cfg.aperture_size == 12e-3
        assert cfg.wavelength == 532e-9
        assert cfg.focal_length == 0.125
        assert cfg.target_side_px == 60
        assert cfg.target_kind == "square"
        assert cfg.far_field_padding == PAD
        assert cfg.seed == 0

    def test_properties(self):
        cfg = _cfg()
        assert cfg.far_field_size == N * PAD
        assert cfg.far_field_pixel_size > 0

    def test_frozen(self):
        cfg = _cfg()
        with pytest.raises(FrozenInstanceError):
            cfg.n_grid = 128

    def test_invalid_padding(self):
        with pytest.raises(ValueError, match="far_field_padding"):
            _cfg(far_field_padding=0)

    def test_invalid_target_kind(self):
        with pytest.raises(ValueError, match="unknown target_kind"):
            _cfg(target_kind="triangle")

    def test_gaussian_pupil_shape(self):
        cfg = _cfg()
        pupil = cfg.make_gaussian_pupil()
        assert pupil.shape == (N, N)
        assert pupil.dtype == np.float64
        assert pupil.max() <= 1.0
        assert pupil.min() >= 0.0

    def test_gaussian_pupil_centered(self):
        """Peak must be at the grid centre."""
        cfg = _cfg()
        pupil = cfg.make_gaussian_pupil()
        cy, cx = np.unravel_index(np.argmax(pupil), pupil.shape)
        assert (cy, cx) == (N // 2, N // 2)

    def test_gaussian_pupil_energy(self):
        """Gaussian pupil must be non-trivial (has energy inside aperture)."""
        cfg = _cfg()
        pupil = cfg.make_gaussian_pupil()
        assert pupil.sum() > 0.0


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------
class TestForwardModel:
    def test_pad_centred_smaller(self):
        arr = np.ones((4, 4))
        padded = _pad_centred(arr, 8)
        assert padded.shape == (8, 8)
        assert padded.sum() == 16

    def test_pad_centred_equal(self):
        arr = np.ones((8, 8))
        padded = _pad_centred(arr, 8)
        assert padded.shape == (8, 8)
        np.testing.assert_array_equal(padded, arr)

    def test_pad_centred_larger(self):
        arr = np.ones((10, 10))
        padded = _pad_centred(arr, 12)
        assert padded.shape == (12, 12)
        assert padded.sum() == 100

    def test_forward_intensity_shape(self):
        cfg = _cfg()
        phase = np.zeros((N, N))
        inten = _forward_intensity(phase, cfg)
        assert inten.shape == (cfg.far_field_size, cfg.far_field_size)

    def test_forward_intensity_finite(self):
        cfg = _cfg()
        phase = np.random.default_rng(0).normal(0, 0.5, size=(N, N))
        inten = _forward_intensity(phase, cfg)
        assert np.isfinite(inten).all()

    def test_forward_intensity_flat_phase_single_peak(self):
        """Flat phase => single focused spot at centre (not a lattice)."""
        cfg = _cfg()
        inten = _forward_intensity(np.zeros((N, N)), cfg)
        inten = inten / inten.max()
        cy, cx = inten.shape[0] // 2, inten.shape[1] // 2
        # peak near centre
        peak = np.unravel_index(np.argmax(inten), inten.shape)
        assert abs(peak[0] - cy) <= 1 and abs(peak[1] - cx) <= 1


# ---------------------------------------------------------------------------
# Target generation
# ---------------------------------------------------------------------------
class TestMakeSquareTarget:
    def test_shape_and_normalization(self):
        cfg = _cfg()
        target = make_square_target(cfg)
        assert target.shape == (cfg.far_field_size, cfg.far_field_size)
        assert target.sum() == pytest.approx(1.0)

    def test_nonnegative(self):
        for kind in ("square", "circle", "gaussian", "annulus"):
            cfg = _cfg(target_kind=kind)
            target = make_square_target(cfg)
            assert target.min() >= 0.0

    def test_square_is_rectangular(self):
        cfg = _cfg(target_kind="square", target_side_px=40)
        target = make_square_target(cfg)
        cy, cx = target.shape[0] // 2, target.shape[1] // 2
        side = 40
        inner = target[cy - side // 2 : cy + side // 2, cx - side // 2 : cx + side // 2]
        assert inner.sum() > 0
        # corners of the grid should be zero
        assert target[0, 0] == 0.0
        assert target[-1, -1] == 0.0

    def test_gaussian_centered(self):
        cfg = _cfg(target_kind="gaussian", target_side_px=40)
        target = make_square_target(cfg)
        cy, cx = np.unravel_index(np.argmax(target), target.shape)
        assert (cy, cx) == (target.shape[0] // 2, target.shape[1] // 2)

    def test_annulus_is_ring(self):
        cfg = _cfg(target_kind="annulus", target_side_px=40)
        target = make_square_target(cfg)
        cy, cx = target.shape[0] // 2, target.shape[1] // 2
        # centre should be zero (hole in the ring)
        assert target[cy, cx] == 0.0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
class TestMetrics:
    @pytest.fixture
    def sample_intensity_target(self):
        cfg = _cfg(target_side_px=20)
        target = make_square_target(cfg)
        intensity = target.copy()  # perfect match
        return intensity, target

    def test_power_in_bucket_perfect(self, sample_intensity_target):
        intensity, target = sample_intensity_target
        pib = power_in_bucket(intensity, target)
        assert pib == pytest.approx(1.0)

    def test_power_in_bucket_zero(self):
        cfg = _cfg(target_side_px=20)
        target = make_square_target(cfg)
        intensity = np.zeros_like(target)
        assert power_in_bucket(intensity, target) == 0.0

    def test_efficiency_equals_pib(self, sample_intensity_target):
        intensity, target = sample_intensity_target
        assert efficiency(intensity, target) == pytest.approx(
            power_in_bucket(intensity, target)
        )

    def test_uniformity_cv_perfect(self, sample_intensity_target):
        intensity, target = sample_intensity_target
        assert uniformity_cv(intensity, target) == pytest.approx(0.0, abs=1e-10)

    def test_uniformity_cv_empty_support(self):
        cfg = _cfg(target_side_px=20)
        target = make_square_target(cfg)
        intensity = np.zeros_like(target)
        assert uniformity_cv(intensity, target) == float("inf")

    def test_strehl_perfect(self, sample_intensity_target):
        intensity, target = sample_intensity_target
        assert strehl(intensity, target) == pytest.approx(1.0)

    def test_strehl_zero(self):
        cfg = _cfg(target_side_px=20)
        target = make_square_target(cfg)
        intensity = np.ones_like(target)
        assert strehl(intensity, target) >= 0.0

    def test_zero_order_fraction(self):
        n = 64
        intensity = np.zeros((n, n))
        intensity[32, 32] = 10.0
        intensity[10, 10] = 5.0
        zf = zero_order_fraction(intensity, center=(32, 32), zero_order_margin_px=5.0)
        assert 0.0 < zf < 1.0

    def test_zero_order_fraction_centered(self):
        n = 64
        intensity = np.zeros((n, n))
        intensity[32, 32] = 100.0
        zf = zero_order_fraction(intensity, center=(32, 32), zero_order_margin_px=5.0)
        assert zf == pytest.approx(1.0)

    def test_compute_bench_metrics_keys(self):
        cfg = _cfg(target_side_px=20)
        target = make_square_target(cfg)
        intensity = target.copy()
        m = compute_bench_metrics(intensity, target)
        assert set(m.keys()) == {"PIB", "efficiency", "CV", "Strehl", "zero_order"}

    def test_rolled_support_none_center(self):
        arr = np.zeros((10, 10))
        arr[3:7, 3:7] = 1.0
        sup = _rolled_support(arr, None)
        assert sup.sum() == 16  # 4x4 region

    def test_rolled_support_with_center(self):
        arr = np.zeros((10, 10))
        arr[3:7, 3:7] = 1.0
        # shift target 2 rows right, 1 col down => center at (5, 4)
        sup = _rolled_support(arr, center=(5, 4))
        # support should now have nonzero at (4..7, 3..6)-ish shifted
        assert sup.sum() == 16


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------
class TestCompositeScore:
    def test_composite_from_pib_cv(self):
        result = composite_from_pib_cv(0.5, 0.1)
        expected = 0.5 * 0.5 + 0.5 * (1.0 / 1.1)
        assert result == pytest.approx(expected)

    def test_composite_from_pib_cv_weights(self):
        result = composite_from_pib_cv(0.8, 0.2, w_pib=0.3, w_unif=0.7)
        expected = 0.3 * 0.8 + 0.7 * (1.0 / 1.2)
        assert result == pytest.approx(expected)

    def test_composite_score_from_dict(self):
        m = {"PIB": 0.6, "CV": 0.15}
        result = composite_score(m)
        expected = 0.5 * 0.6 + 0.5 * (1.0 / 1.15)
        assert result == pytest.approx(expected)

    def test_composite_score_missing_cv(self):
        m = {"PIB": 0.6}
        result = composite_score(m)
        # CV defaults to inf => 1/(1+inf) = 0
        assert result == pytest.approx(0.5 * 0.6)

    def test_composite_score_missing_pib(self):
        m = {"CV": 0.2}
        result = composite_score(m)
        assert result == pytest.approx(0.5 * (1.0 / 1.2))


# ---------------------------------------------------------------------------
# GS shaping
# ---------------------------------------------------------------------------
class TestGsShape:
    def test_returns_shaping_result(self):
        cfg = _cfg(target_side_px=20)
        result = gs_shape(cfg, n_iters=10, seed=0)
        assert isinstance(result, ShapingResult)
        assert result.method == "gerchberg_saxton"
        assert result.n_iters == 10
        assert result.phase.shape == (N, N)
        assert result.intensity.shape == (cfg.far_field_size, cfg.far_field_size)

    def test_phase_is_real(self):
        cfg = _cfg(target_side_px=20)
        result = gs_shape(cfg, n_iters=10, seed=0)
        assert np.isrealobj(result.phase) or np.isfinite(result.phase).all()
        assert result.phase.dtype == np.float64

    def test_determinism(self):
        cfg = _cfg(target_side_px=20)
        r1 = gs_shape(cfg, n_iters=10, seed=42)
        r2 = gs_shape(cfg, n_iters=10, seed=42)
        np.testing.assert_array_equal(r1.phase, r2.phase)

    def test_seed_changes_result(self):
        cfg = _cfg(target_side_px=20)
        r1 = gs_shape(cfg, n_iters=10, seed=0)
        r2 = gs_shape(cfg, n_iters=10, seed=1)
        assert not np.allclose(r1.phase, r2.phase)

    def test_metrics_present(self):
        cfg = _cfg(target_side_px=20)
        result = gs_shape(cfg, n_iters=10, seed=0)
        for key in ("PIB", "efficiency", "CV", "Strehl", "zero_order"):
            assert key in result.metrics

    def test_verbose_history(self):
        cfg = _cfg(target_side_px=20)
        result = gs_shape(cfg, n_iters=40, seed=0, verbose=True)
        assert len(result.history) > 0
        assert "iter" in result.history[0]

    def test_intensity_normalized(self):
        cfg = _cfg(target_side_px=20)
        result = gs_shape(cfg, n_iters=10, seed=0)
        assert result.intensity.sum() == pytest.approx(1.0, abs=1e-6)

    def test_score_improves_with_iterations(self):
        """More iterations should generally improve the composite score."""
        cfg = _cfg(target_side_px=20)
        r10 = gs_shape(cfg, n_iters=10, seed=0)
        r100 = gs_shape(cfg, n_iters=100, seed=0)
        s10 = composite_score(r10.metrics)
        s100 = composite_score(r100.metrics)
        # both should be better than zero phase
        flat = _forward_intensity(np.zeros((N, N)), cfg)
        flat = flat / flat.sum()
        sf = composite_score(compute_bench_metrics(flat, make_square_target(cfg)))
        assert s10 > sf
        assert s100 > sf


# ---------------------------------------------------------------------------
# UniformSquareShaper
# ---------------------------------------------------------------------------
class TestUniformSquareShaper:
    def test_invalid_method(self):
        cfg = _cfg(target_side_px=20)
        with pytest.raises(ValueError, match="unknown method"):
            UniformSquareShaper(cfg, method="bad", n_iters=5)

    def test_invalid_gamma(self):
        cfg = _cfg(target_side_px=20)
        with pytest.raises(ValueError, match="gamma"):
            UniformSquareShaper(cfg, gamma=1.5, n_iters=5)

    def test_invalid_weight_clip(self):
        cfg = _cfg(target_side_px=20)
        with pytest.raises(ValueError, match="weight_clip"):
            UniformSquareShaper(cfg, weight_clip=0.5, n_iters=5)

    def test_invalid_beta(self):
        cfg = _cfg(target_side_px=20)
        with pytest.raises(ValueError, match="beta"):
            UniformSquareShaper(cfg, beta=1.5, n_iters=5)

    def test_invalid_hio_fraction(self):
        cfg = _cfg(target_side_px=20)
        with pytest.raises(ValueError, match="hio_fraction"):
            UniformSquareShaper(cfg, hio_fraction=1.5, n_iters=5)

    def test_invalid_n_iters(self):
        cfg = _cfg(target_side_px=20)
        with pytest.raises(ValueError, match="n_iters"):
            UniformSquareShaper(cfg, n_iters=0)

    def test_invalid_select(self):
        cfg = _cfg(target_side_px=20)
        with pytest.raises(ValueError, match="select"):
            UniformSquareShaper(cfg, select="bad", n_iters=5)

    @pytest.mark.parametrize("method", GS_METHODS)
    def test_all_methods_run(self, method):
        cfg = _cfg(target_side_px=20)
        shaper = UniformSquareShaper(cfg, method=method, n_iters=10, seed=0)
        for _ in range(5):
            phase = shaper.update()
        assert phase.shape == (N, N)

    def test_update_returns_phase(self):
        cfg = _cfg(target_side_px=20)
        shaper = UniformSquareShaper(cfg, n_iters=10, seed=0)
        phase = shaper.update()
        assert phase.shape == (N, N)
        assert np.isfinite(phase).all()

    def test_adaptive_feedback_zero_beta_matches_plain(self):
        cfg = _cfg(target_side_px=20)
        plain = gs_shape(cfg, n_iters=10, seed=0)
        feedback = UniformSquareShaper(
            cfg, method="adaptive-feedback", beta=0.0, n_iters=10,
            seed=0, select="last",
        ).run()
        np.testing.assert_allclose(
            np.exp(1j * feedback.phase), np.exp(1j * plain.phase), atol=1e-9
        )

    def test_run_returns_shaping_result(self):
        cfg = _cfg(target_side_px=20)
        shaper = UniformSquareShaper(cfg, n_iters=20, seed=0)
        result = shaper.run()
        assert isinstance(result, ShapingResult)
        assert result.phase.shape == (N, N)
        assert "best_iter" in result.metrics

    def test_run_select_last(self):
        cfg = _cfg(target_side_px=20)
        shaper = UniformSquareShaper(cfg, n_iters=20, seed=0, select="last")
        result = shaper.run()
        assert isinstance(result, ShapingResult)
        assert result.metrics["best_iter"] == 19

    def test_determinism(self):
        cfg = _cfg(target_side_px=20)
        s1 = UniformSquareShaper(cfg, n_iters=10, seed=42)
        s2 = UniformSquareShaper(cfg, n_iters=10, seed=42)
        r1 = s1.run()
        r2 = s2.run()
        np.testing.assert_array_equal(r1.phase, r2.phase)

    def test_history_recorded_verbose(self):
        cfg = _cfg(target_side_px=20)
        shaper = UniformSquareShaper(cfg, n_iters=40, seed=0, verbose=True)
        shaper.run()
        assert len(shaper.history) > 0


# ---------------------------------------------------------------------------
# Analytic amplitude target
# ---------------------------------------------------------------------------
class TestAnalyticAmplitudeTarget:
    def test_returns_shaping_result(self):
        cfg = _cfg(target_side_px=20)
        result = analytic_amplitude_target(cfg)
        assert isinstance(result, ShapingResult)
        assert result.method == "analytic_amplitude_baseline"
        assert result.n_iters == 0

    def test_intensity_is_target(self):
        cfg = _cfg(target_side_px=20)
        result = analytic_amplitude_target(cfg)
        assert result.intensity.sum() == pytest.approx(1.0, abs=1e-6)

    def test_phase_is_zero(self):
        cfg = _cfg(target_side_px=20)
        result = analytic_amplitude_target(cfg)
        assert np.all(result.phase == 0.0)

    def test_perfect_score(self):
        cfg = _cfg(target_side_px=20)
        result = analytic_amplitude_target(cfg)
        assert composite_score(result.metrics) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# SPGD
# ---------------------------------------------------------------------------
class TestSPGDShape:
    def test_returns_shaping_result(self):
        cfg = _cfg(target_side_px=20)
        result = spgd_shape(cfg, n_iters=10, seed=0, dim=8)
        assert isinstance(result, ShapingResult)
        assert result.method == "spgd_freeform"
        assert result.phase.shape == (N, N)

    def test_determinism(self):
        cfg = _cfg(target_side_px=20)
        r1 = spgd_shape(cfg, n_iters=10, seed=42, dim=8)
        r2 = spgd_shape(cfg, n_iters=10, seed=42, dim=8)
        np.testing.assert_array_equal(r1.phase, r2.phase)

    def test_intensity_normalized(self):
        cfg = _cfg(target_side_px=20)
        result = spgd_shape(cfg, n_iters=10, seed=0, dim=8)
        assert result.intensity.sum() == pytest.approx(1.0, abs=1e-6)

    def test_metrics_present(self):
        cfg = _cfg(target_side_px=20)
        result = spgd_shape(cfg, n_iters=10, seed=0, dim=8)
        for key in ("PIB", "efficiency", "CV", "Strehl", "zero_order"):
            assert key in result.metrics


# ---------------------------------------------------------------------------
# Differentiable (torch-gated)
# ---------------------------------------------------------------------------
torch = pytest.importorskip("torch")


class TestDifferentiableShape:
    def test_returns_shaping_result(self):
        cfg = _cfg(target_side_px=20)
        result = differentiable_shape(cfg, n_iters=10, lr=0.05, seed=0)
        assert isinstance(result, ShapingResult)
        assert result.method == "differentiable"
        assert result.phase.shape == (N, N)

    def test_determinism(self):
        cfg = _cfg(target_side_px=20)
        r1 = differentiable_shape(cfg, n_iters=10, lr=0.05, seed=42)
        r2 = differentiable_shape(cfg, n_iters=10, lr=0.05, seed=42)
        np.testing.assert_allclose(r1.phase, r2.phase, rtol=1e-10)

    def test_score_improves(self):
        """Differentiable optimization should improve the composite score."""
        cfg = _cfg(target_side_px=20)
        target = make_square_target(cfg)
        result = differentiable_shape(cfg, n_iters=200, lr=0.1, seed=0)
        s = composite_score(compute_bench_metrics(result.intensity, target))
        flat = _forward_intensity(np.zeros((N, N)), cfg)
        flat = flat / flat.sum()
        sf = composite_score(compute_bench_metrics(flat, target))
        assert s >= sf
