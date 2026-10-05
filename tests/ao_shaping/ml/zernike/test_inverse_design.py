"""Tests for :mod:`ml.zernike.inverse_design`.

The library replaced copy-pasted helpers in a dozen exploratory scripts, so these
pin the contracts those scripts relied on -- especially the two calibration traps that
silently produced all-zero metrics (NaN outside the aperture disc, and a 64x64
``SimPibSystem`` being out of regime) and the structural invariance from attempt 12.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.zernike import inverse_design as inv
from ml.zernike.losses import LossConfig


class TestTargetsAndMasks:
    def test_square_amplitude_geometry(self) -> None:
        amp = inv.square_amplitude(64, 0.375, 4.0 / 3.0)
        assert amp.shape == (64, 64)
        assert set(np.unique(amp)) <= {0.0, 1.0}
        rows = np.where(amp.any(axis=1))[0]
        cols = np.where(amp.any(axis=0))[0]
        # Every row/column that is lit is lit across the full extent (a filled rect).
        assert amp[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1].sum() == amp.sum()
        # Centred on the grid.
        assert abs(rows.mean() - 31.5) < 1.0
        assert abs(cols.mean() - 31.5) < 1.0

    def test_square_amplitude_is_aspect_dependent(self) -> None:
        square = inv.square_amplitude(64, 0.375, 1.0)
        wide = inv.square_amplitude(64, 0.375, 2.0)
        assert wide.sum() > square.sum(), "wider aspect must cover more pixels"
        assert np.array_equal(square, np.rot90(square)), "a square target is symmetric"

    def test_target_tensor_shape_and_dtype(self) -> None:
        t = inv.target_tensor(32, 0.375, 1.0)
        assert t.shape == (1, 1, 32, 32)
        assert t.dtype == torch.float32

    def test_inscribed_mask_is_a_disc(self) -> None:
        m = inv.inscribed_mask(64)
        assert m.shape == (1, 1, 64, 64)
        plane = m[0, 0].numpy()
        assert set(np.unique(plane)) <= {0.0, 1.0}
        # The inscribed circle covers pi/4 of the square.
        assert plane.mean() == pytest.approx(np.pi / 4, abs=0.01)
        assert plane[32, 32] == 1.0, "centre is inside"
        assert plane[0, 0] == 0.0, "corner is outside"

    def test_roi_mask_matches_the_target_box(self) -> None:
        mask = inv.roi(64, 0.375, 4.0 / 3.0).numpy()
        target = inv.square_amplitude(64, 0.375, 4.0 / 3.0)
        # The loss box must cover the target, not sit inside it.
        assert mask.sum() >= target.sum()


class TestEmbedPhase:
    def test_zeroes_everything_outside_the_disc(self) -> None:
        phase = np.full((inv.GRID, inv.GRID), 0.7)
        panel = inv.embed_phase(phase)
        assert panel.shape == (inv.GRID, inv.GRID)
        # The disc mask is what carries "no light outside the aperture".
        assert panel[0, 0] == 0.0, "corner is outside the disc and carries no light"
        assert panel[inv.GRID // 2, inv.GRID // 2] == pytest.approx(0.7)

    def test_disc_occupies_about_a_quarter_of_the_grid(self) -> None:
        masked = inv.embed_phase(np.ones((inv.GRID, inv.GRID)))
        assert masked.mean() == pytest.approx(np.pi / 4, abs=0.01)

    def test_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="phase must be"):
            inv.embed_phase(np.zeros((32, 32)))


class TestIndependentEvaluator:
    def test_far_field_is_finite_and_correctly_shaped(self) -> None:
        far = inv.sim_far_field(np.zeros((inv.GRID, inv.GRID)))
        assert far.shape == (inv.GRID, inv.GRID)
        assert np.all(np.isfinite(far)), (
            "a non-finite far field means the simulator is out of regime -- "
            "PROCESS.md attempt 1 got all-zero metrics this way instead of an error"
        )

    def test_flat_phase_is_not_all_zero(self) -> None:
        """Guards the exact failure of attempt 1: a metric that is identically zero."""
        assert inv.score_phase(np.zeros((inv.GRID, inv.GRID))) > 0.1

    def test_score_uses_the_canonical_roi_terms(self) -> None:
        from ao_shaping.utils.image.target.metrics import rms_pib_terms

        phase = inv.gs_phase(0.375, 4.0 / 3.0)
        pib, uni = rms_pib_terms(
            inv.sim_far_field(phase),
            (inv.GRID / 2, inv.GRID / 2),
            "rectangle",
            0.375 * inv.GRID,
            4.0 / 3.0,
        )
        assert inv.score_phase(phase, 0.375, 4.0 / 3.0) == pytest.approx(pib + uni)

    def test_gs_beats_flat(self) -> None:
        flat = inv.score_phase(np.zeros((inv.GRID, inv.GRID)))
        assert inv.score_phase(inv.gs_phase()) > flat

    def test_score_is_deterministic(self) -> None:
        phase = inv.gs_phase()
        assert inv.score_phase(phase) == inv.score_phase(phase)


class TestZernikeBridge:
    def test_phase_is_finite_despite_generator_nan_outside_disc(self) -> None:
        """``ZernikeGenerator`` returns NaN outside the aperture; it must be cleared."""
        coeffs = np.zeros(inv.N_MAX * (inv.N_MAX + 3) // 2)
        phase = inv.coefficients_to_phase(coeffs)
        assert np.all(np.isfinite(phase))

    def test_zero_coefficients_give_zero_phase(self) -> None:
        coeffs = np.zeros(inv.N_MAX * (inv.N_MAX + 3) // 2)
        assert np.allclose(inv.coefficients_to_phase(coeffs), 0.0)

    def test_gs_coefficients_drop_piston(self) -> None:
        coeffs = inv.gs_coefficients()
        assert len(coeffs) == inv.N_MAX * (inv.N_MAX + 3) // 2
        assert np.all(np.isfinite(coeffs))

    def test_score_coefficients_agrees_with_score_phase(self) -> None:
        coeffs = inv.gs_coefficients()
        assert inv.score_coefficients(coeffs) == pytest.approx(inv.score_phase(
            inv.coefficients_to_phase(coeffs)
        ))

    def test_zero_coefficients_score_like_flat(self) -> None:
        coeffs = np.zeros(inv.N_MAX * (inv.N_MAX + 3) // 2)
        assert inv.score_coefficients(coeffs) == pytest.approx(
            inv.score_phase(np.zeros((inv.GRID, inv.GRID)))
        )


class TestModelAndOperators:
    def test_build_model_is_deterministic(self) -> None:
        a, b = inv.build_model(), inv.build_model()
        assert torch.equal(a.coefficients, b.coefficients)
        # Coefficients initialise to zeros DETERMINISTICALLY -- this is why every
        # study draws restarts explicitly instead of seeding.
        assert torch.count_nonzero(a.coefficients) == 0

    def test_model_far_field_shape_and_gradients(self) -> None:
        model = inv.build_model()
        phase = torch.zeros(1, 1, inv.GRID, inv.GRID, requires_grad=True)
        out = inv.model_far_field(model, phase)
        assert out.shape == (1, 1, inv.GRID, inv.GRID)
        out.sum().backward()
        assert phase.grad is not None and torch.isfinite(phase.grad).all()
        assert phase.grad.abs().sum() > 0, "no gradient reached the freeform phase"

    def test_aperture_mask_matters(self) -> None:
        """Unmasked vs masked pupil are different functions -- the mask is load-bearing."""
        model = inv.build_model()
        phase = torch.zeros(1, 1, inv.GRID, inv.GRID)
        masked = inv.model_far_field(model, phase)
        phasor = torch.polar(torch.ones_like(phase), phase)
        unmasked = model(phasor.real, phasor.imag)
        assert not torch.allclose(masked, unmasked)

    def test_zero_phase_gives_a_centred_core(self) -> None:
        out = inv.model_far_field(inv.build_model(), torch.zeros(1, 1, inv.GRID, inv.GRID))
        plane = out[0, 0].detach().numpy()
        peak = np.unravel_index(int(plane.argmax()), plane.shape)
        assert abs(peak[0] - inv.GRID // 2) <= inv.GRID // 4
        assert abs(peak[1] - inv.GRID // 2) <= inv.GRID // 4

    def test_gradient_freeform_returns_a_finite_phase(self) -> None:
        model = inv.build_model()
        out = inv.gradient_freeform(
            np.zeros((inv.COARSE_GRID, inv.COARSE_GRID)),
            model,
            inv.target_tensor(),
            inv.roi(),
            steps=5,
        )
        assert out.shape == (inv.GRID, inv.GRID)
        assert np.all(np.isfinite(out))

    def test_gradient_zernike_returns_finite_coefficients(self) -> None:
        model = inv.build_model()
        coeffs = inv.gradient_zernike(
            np.zeros(inv.N_MAX * (inv.N_MAX + 3) // 2),
            model,
            inv.target_tensor(),
            inv.roi(),
            steps=5,
        )
        assert len(coeffs) == inv.N_MAX * (inv.N_MAX + 3) // 2
        assert np.all(np.isfinite(coeffs))


class TestAttempt12Invariance:
    """The structural fact behind attempt 12, as a regression test.

    ``correction_far_field`` reads only ``model.coefficients``, so the Zernike gradient
    path cannot depend on what the model was fitted to. If this ever changes, the
    study's conclusion "forward-model accuracy is not an input" is void.
    """

    @pytest.mark.parametrize("steps,l2,seed", [(0, 0.0, 0), (200, 0.0, 0), (200, 1e-2, 7)])
    def test_refinement_is_invariant_to_the_fitted_state(self, steps: int, l2: float, seed: int) -> None:
        fitted = inv.build_model(seed=seed)
        if steps:
            optimiser = torch.optim.AdamW(
                fitted.parameters(), lr=0.02, weight_decay=l2
            )
            target = inv.target_tensor()
            reference = fitted._normalize(target.clone())  # noqa: SLF001
            for _ in range(steps):
                optimiser.zero_grad()
                ((fitted(reference.new_zeros(1, 1, inv.GRID, inv.GRID) + 0.0,
                         reference.new_zeros(1, 1, inv.GRID, inv.GRID))
                  - reference) ** 2).mean().backward()
                optimiser.step()

        start = inv.gs_coefficients()
        refined = inv.gradient_zernike(
            start, fitted, inv.target_tensor(), inv.roi(), steps=5
        )
        # Re-running with a DIFFERENT fitted model must give the same answer.
        other = inv.build_model(seed=seed + 1)
        refined_again = inv.gradient_zernike(
            start, other, inv.target_tensor(), inv.roi(), steps=5
        )
        assert np.allclose(refined, refined_again), (
            "Zernike refinement started depending on the fitted forward state -- "
            "attempt 12's structural conclusion no longer holds"
        )


class TestStatistics:
    def test_pearson_extremes(self) -> None:
        assert inv.pearson([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
        assert inv.pearson([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)

    def test_spearman_extremes(self) -> None:
        assert inv.spearman([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
        assert inv.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)

    def test_spearman_uses_ranks_not_magnitudes(self) -> None:
        """The reason the study uses ranks: only the ordering may matter.

        Any strictly increasing transform of either argument must leave Spearman
        unchanged -- that is exactly what "uses ranks" means, and it is why the
        restart sweep reports it instead of Pearson.
        """
        x = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, -50.0])
        assert inv.spearman(x, y) == pytest.approx(inv.spearman(x**3, np.exp(y)))
        # ...while Pearson is not invariant to such a transform.
        assert inv.pearson(x, y) != pytest.approx(inv.pearson(x**3, np.exp(y)))

    def test_constant_input_is_zero_not_nan(self) -> None:
        assert inv.pearson([1, 1, 1], [1, 2, 3]) == 0.0
        assert inv.spearman([1, 1, 1], [1, 2, 3]) == 0.0

    def test_weights_default_is_mse(self) -> None:
        assert LossConfig().w_mse == 1.0