"""Tests for the GS-refine Zernike truncation fitter.

The GS-refine experiment fits the best SLM phase it found into a Noll-order
Zernike coefficient vector at several truncation orders, then re-optimises
those coefficients with SPGD. This module is the pure-math bridge between the
two: it crops the illuminated pupil square out of a full SLM panel, least-squares
fits it into Zernike coefficients, and can rebuild a full-panel phase from a
coefficient vector.

Everything here is hardware-free and delegates 100% of the Zernike math to the
canonical engine (:class:`~ao_shaping.utils.wavefront.zernike_calc.ZernikeGenerator`)
and the panel-domain wrapper
(:class:`~ao_shaping.utils.wavefront.pattern_helper.PatternHelper`). The tests
pin the two properties the sweep driver relies on: an exact round-trip through
the generator's own cached grid, and panel-domain consistency with
``embed_pupil_square``.
"""
from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.optimizer.wfless import gs_refine_zernike_fit as fitmod
from ao_shaping.utils.slm.phase_embed import embed_pupil_square, pupil_side
from ao_shaping.utils.wavefront.zernike_calc import (
    ZernikeGenerator,
    calc_n_zernike_terms,
    noll_to_nm,
)

RADIUS = 40.0
SIDE = pupil_side(RADIUS)
PANEL_H = 300
PANEL_W = 400


def _gen(n_max: int, radius: float = RADIUS) -> ZernikeGenerator:
    """Generator on the pupil square for ``radius`` (same grid the module uses)."""
    side = pupil_side(radius)
    return ZernikeGenerator(resolution=(side, side), radius=radius, n_orders=n_max)


def _coeffs(gen: ZernikeGenerator, values: dict[int, float]) -> np.ndarray:
    """Noll-order coefficient vector with ``values`` placed at their indices."""
    c = np.zeros(gen.n_modes, dtype=np.float64)
    for idx, val in values.items():
        c[idx] = val
    return c


class TestNollOrdering:
    """Pin the Noll convention the whole experiment is written against."""

    def test_index_zero_is_piston(self):
        assert noll_to_nm(1) == (0, 0)

    def test_index_three_is_defocus(self):
        assert noll_to_nm(4) == (2, 0)

    def test_index_ten_is_spherical(self):
        assert noll_to_nm(11) == (4, 0)

    def test_generator_agrees_with_noll_to_nm(self):
        gen = _gen(6)
        assert gen.nm_to_noll(0, 0) == 1
        assert gen.nm_to_noll(2, 0) == 4
        assert gen.nm_to_noll(4, 0) == 11


class TestExactRoundTrip:
    def test_recovers_known_coefficients(self):
        gen = _gen(6)
        truth = _coeffs(gen, {0: 0.3, 3: 0.7, 10: -0.2})
        phase = gen.generate_noll(truth)

        result = fitmod.fit_phase_to_zernike(phase, RADIUS, 6)

        assert np.max(np.abs(result.coefficients - truth)) < 1e-12

    def test_result_metadata(self):
        gen = _gen(6)
        truth = _coeffs(gen, {0: 0.3, 3: 0.7, 10: -0.2})
        phase = gen.generate_noll(truth)

        result = fitmod.fit_phase_to_zernike(phase, RADIUS, 6)

        assert result.n_max == 6
        assert result.radius_px == RADIUS
        assert result.side == SIDE
        assert result.n_modes == calc_n_zernike_terms(6)
        assert len(result.coefficients) == result.n_modes
        assert result.coefficients.dtype == np.float64

    def test_reconstructed_is_nan_outside_aperture(self):
        gen = _gen(6)
        truth = _coeffs(gen, {0: 0.3, 3: 0.7, 10: -0.2})
        phase = gen.generate_noll(truth)

        result = fitmod.fit_phase_to_zernike(phase, RADIUS, 6)

        assert result.reconstructed.shape == (SIDE, SIDE)
        assert np.isnan(result.reconstructed).any()
        assert np.isfinite(result.reconstructed).any()
        # The aperture support is exactly the generator's mask.
        assert np.array_equal(
            np.isfinite(result.reconstructed), gen.mask.astype(bool)
        )

    def test_residual_is_zero_for_an_exact_fit(self):
        gen = _gen(6)
        truth = _coeffs(gen, {0: 0.3, 3: 0.7, 10: -0.2})
        phase = gen.generate_noll(truth)

        result = fitmod.fit_phase_to_zernike(phase, RADIUS, 6)

        assert result.rms_residual_rad < 1e-12

    def test_max_abs_coefficient(self):
        gen = _gen(6)
        truth = _coeffs(gen, {0: 0.3, 3: 0.7, 10: -0.2})
        phase = gen.generate_noll(truth)

        result = fitmod.fit_phase_to_zernike(phase, RADIUS, 6)

        assert result.max_abs_coefficient == pytest.approx(0.7, abs=1e-12)


class TestCount:
    @pytest.mark.parametrize("n_max", [1, 4, 6, 8])
    def test_n_modes_matches_triangular_count(self, n_max):
        gen = _gen(n_max)
        phase = gen.generate_noll(np.zeros(gen.n_modes))

        result = fitmod.fit_phase_to_zernike(phase, RADIUS, n_max)

        assert result.n_modes == calc_n_zernike_terms(n_max)
        assert len(result.coefficients) == result.n_modes


class TestMonotoneResidual:
    def test_residual_non_increasing_with_order(self):
        gen12 = _gen(12)
        rng = np.random.default_rng(0)
        phase = gen12.generate_noll(rng.normal(0.0, 0.3, gen12.n_modes))

        residuals = [
            fitmod.fit_phase_to_zernike(phase, RADIUS, n_max).rms_residual_rad
            for n_max in (2, 4, 6, 8, 10, 12)
        ]

        assert all(b <= a + 1e-12 for a, b in zip(residuals, residuals[1:]))
        # The full order must reproduce the phase it was generated from.
        assert residuals[-1] < 1e-12


class TestCrop:
    def test_returns_exact_embedded_square(self):
        gen = _gen(6)
        gs = gen.generate_noll(_coeffs(gen, {0: 0.3, 3: 0.7, 10: -0.2}))
        panel = embed_pupil_square(np.nan_to_num(gs), (PANEL_H, PANEL_W), RADIUS)

        box = fitmod.pupil_square_from_panel(panel, RADIUS)

        assert box.shape == (SIDE, SIDE)
        assert np.array_equal(box, np.nan_to_num(gs))

    def test_default_center_is_panel_centre(self):
        panel = np.arange(PANEL_H * PANEL_W, dtype=np.float64).reshape(
            PANEL_H, PANEL_W
        )

        box = fitmod.pupil_square_from_panel(panel, RADIUS)

        cx, cy = PANEL_W // 2, PANEL_H // 2
        x0, y0 = cx - SIDE // 2, cy - SIDE // 2
        assert np.array_equal(box, panel[y0 : y0 + SIDE, x0 : x0 + SIDE])

    def test_explicit_center_is_honoured(self):
        panel = np.arange(PANEL_H * PANEL_W, dtype=np.float64).reshape(
            PANEL_H, PANEL_W
        )

        box = fitmod.pupil_square_from_panel(panel, RADIUS, center_xy=(100, 100))

        x0, y0 = 100 - SIDE // 2, 100 - SIDE // 2
        assert np.array_equal(box, panel[y0 : y0 + SIDE, x0 : x0 + SIDE])

    def test_returns_float64(self):
        panel = np.zeros((PANEL_H, PANEL_W), dtype=np.float32)
        box = fitmod.pupil_square_from_panel(panel, RADIUS)
        assert box.dtype == np.float64


class TestPanelRoundTrip:
    def test_panel_reconstruction_matches_original(self):
        gen = _gen(6)
        truth = _coeffs(gen, {0: 0.3, 3: 0.7, 10: -0.2})
        panel = fitmod.panel_phase_from_coefficients(truth, RADIUS, (PANEL_H, PANEL_W))

        result = fitmod.fit_best_phase(panel, 6, RADIUS)
        rebuilt = fitmod.panel_phase_from_coefficients(
            result.coefficients, RADIUS, (PANEL_H, PANEL_W)
        )

        assert np.allclose(rebuilt, panel, atol=1e-9)

    def test_recovers_gs_coefficients_from_embedded_panel(self):
        gen = _gen(6)
        truth = _coeffs(gen, {0: 0.3, 3: 0.7, 10: -0.2})
        gs = gen.generate_noll(truth)
        panel = embed_pupil_square(np.nan_to_num(gs), (PANEL_H, PANEL_W), RADIUS)

        result = fitmod.fit_best_phase(panel, 6, RADIUS)

        assert np.max(np.abs(result.coefficients - truth)) < 1e-12

    def test_fit_best_phase_equals_crop_then_fit(self):
        gen = _gen(6)
        truth = _coeffs(gen, {0: 0.3, 3: 0.7, 10: -0.2})
        panel = fitmod.panel_phase_from_coefficients(truth, RADIUS, (PANEL_H, PANEL_W))

        direct = fitmod.fit_best_phase(panel, 6, RADIUS)
        manual = fitmod.fit_phase_to_zernike(
            fitmod.pupil_square_from_panel(panel, RADIUS), RADIUS, 6
        )

        assert np.array_equal(direct.coefficients, manual.coefficients)


class TestPanelPhaseFromCoefficients:
    def test_matches_embed_pupil_square(self):
        gen = _gen(6)
        truth = _coeffs(gen, {0: 0.3, 3: 0.7, 10: -0.2})

        panel = fitmod.panel_phase_from_coefficients(truth, RADIUS, (PANEL_H, PANEL_W))
        reference = embed_pupil_square(
            np.nan_to_num(gen.generate_noll(truth)), (PANEL_H, PANEL_W), RADIUS
        )

        assert panel.shape == (PANEL_H, PANEL_W)
        assert np.allclose(panel, reference, atol=1e-12)

    def test_derives_n_max_and_truncates_long_vector(self):
        # len 30 -> largest n with (n+1)(n+2)//2 <= 30 is n=6 (28 terms).
        long = np.zeros(30, dtype=np.float64)
        long[0] = 0.5
        long[3] = 0.2

        panel = fitmod.panel_phase_from_coefficients(long, RADIUS, (PANEL_H, PANEL_W))

        gen = _gen(6)
        reference = embed_pupil_square(
            np.nan_to_num(gen.generate_noll(long[:28])), (PANEL_H, PANEL_W), RADIUS
        )
        assert np.allclose(panel, reference, atol=1e-12)

    def test_short_vector_is_zero_padded(self):
        # len 1 -> n_max 0 -> piston only; the extra modes are zero.
        panel = fitmod.panel_phase_from_coefficients(
            np.array([0.5]), RADIUS, (PANEL_H, PANEL_W)
        )

        assert panel.shape == (PANEL_H, PANEL_W)
        inside = panel[panel != 0.0]
        assert inside.size > 0
        assert np.allclose(inside, 0.5, atol=1e-12)

    def test_returns_float64(self):
        panel = fitmod.panel_phase_from_coefficients(
            np.zeros(6), RADIUS, (PANEL_H, PANEL_W)
        )
        assert panel.dtype == np.float64


class TestZeroCoefficients:
    def test_all_zero_pupil_gives_zero_coefficients_and_residual(self):
        phase = np.zeros((SIDE, SIDE), dtype=np.float64)

        result = fitmod.fit_phase_to_zernike(phase, RADIUS, 4)

        assert np.all(result.coefficients == 0.0)
        assert result.rms_residual_rad == 0.0
        assert result.max_abs_coefficient == 0.0


class TestValidation:
    def test_n_max_below_one(self):
        with pytest.raises(ValueError):
            fitmod.fit_phase_to_zernike(np.zeros((SIDE, SIDE)), RADIUS, 0)

    def test_radius_non_positive(self):
        with pytest.raises(ValueError):
            fitmod.fit_phase_to_zernike(np.zeros((SIDE, SIDE)), 0.0, 4)

    def test_radius_non_finite(self):
        with pytest.raises(ValueError):
            fitmod.fit_phase_to_zernike(np.zeros((SIDE, SIDE)), np.inf, 4)

    def test_pupil_phase_not_square(self):
        with pytest.raises(ValueError):
            fitmod.fit_phase_to_zernike(np.zeros((SIDE, SIDE + 1)), RADIUS, 4)

    def test_pupil_phase_not_2d(self):
        with pytest.raises(ValueError):
            fitmod.fit_phase_to_zernike(np.zeros(SIDE), RADIUS, 4)

    def test_pupil_phase_wrong_size(self):
        with pytest.raises(ValueError):
            fitmod.fit_phase_to_zernike(np.zeros((SIDE + 2, SIDE + 2)), RADIUS, 4)

    def test_all_nan_input_is_an_error(self):
        with pytest.raises(ValueError):
            fitmod.fit_phase_to_zernike(np.full((SIDE, SIDE), np.nan), RADIUS, 4)

    def test_finite_only_outside_aperture_is_an_error(self):
        # Finite values exist but none inside the aperture, so the fit would be
        # meaningless; it must raise rather than return garbage.
        phase = np.full((SIDE, SIDE), np.nan)
        phase[0, 0] = 1.0
        with pytest.raises(ValueError):
            fitmod.fit_phase_to_zernike(phase, RADIUS, 4)

    def test_panel_shape_non_finite(self):
        with pytest.raises(ValueError):
            fitmod.panel_phase_from_coefficients(
                np.zeros(3), RADIUS, (np.nan, PANEL_W)
            )

    def test_panel_radius_non_positive(self):
        with pytest.raises(ValueError):
            fitmod.panel_phase_from_coefficients(
                np.zeros(3), -1.0, (PANEL_H, PANEL_W)
            )

    def test_panel_radius_non_finite(self):
        with pytest.raises(ValueError):
            fitmod.panel_phase_from_coefficients(
                np.zeros(3), np.nan, (PANEL_H, PANEL_W)
            )

    def test_crop_radius_non_positive(self):
        with pytest.raises(ValueError):
            fitmod.pupil_square_from_panel(np.zeros((PANEL_H, PANEL_W)), 0.0)

    def test_crop_square_outside_panel(self):
        with pytest.raises(ValueError):
            fitmod.pupil_square_from_panel(
                np.zeros((PANEL_H, PANEL_W)), RADIUS, center_xy=(5, 5)
            )


class TestPublicSurface:
    def test_all_exports(self):
        assert set(fitmod.__all__) == {
            "ZernikeFitResult",
            "pupil_square_from_panel",
            "fit_phase_to_zernike",
            "fit_best_phase",
            "panel_phase_from_coefficients",
        }

    def test_result_is_frozen(self):
        gen = _gen(4)
        result = fitmod.fit_phase_to_zernike(
            gen.generate_noll(np.zeros(gen.n_modes)), RADIUS, 4
        )
        with pytest.raises(AttributeError):
            result.n_max = 99
