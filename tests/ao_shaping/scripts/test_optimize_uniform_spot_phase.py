"""Tests for the phase-inversion script (frozen surrogate, phase as the variable).

The optimisation itself is checked against the properties that make it trustworthy
rather than against a golden number: the weights must stay frozen, the gradient
must actually reach the phase, the target box must be the same pixels the
canonical metric scores, and the reported verdict must be derived from the data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from optimize_uniform_spot_phase import (  # noqa: E402
    PhaseVariable,
    box_slice,
    box_target,
    diagnose_extrapolation,
    score_spot,
    surrogate_loss,
)


class TestBoxGeometry:
    def test_box_is_centred_on_the_optical_axis(self) -> None:
        """grid=64, side=50 -> indices 7..56, centre 31.5 = the fftshift axis."""
        window = box_slice(64, 50)
        assert (window.start, window.stop) == (7, 57)
        indices = np.arange(64)[window]
        assert (indices[0] + indices[-1]) / 2 == pytest.approx(31.5)

    def test_odd_side_cannot_exactly_hit_the_half_pixel_axis(self) -> None:
        """An odd pixel count centres on an integer, so it cannot sit on 31.5.

        Worth pinning because it means a 50 px box (even) and a 51 px box (odd)
        are *not* placed by the same rule, and the scorer must be told which.
        """
        window = box_slice(64, 51)
        indices = np.arange(64)[window]
        assert len(indices) == 51
        # As centred as an even grid allows: within half a pixel of the axis.
        assert abs((indices[0] + indices[-1]) / 2 - 31.5) == pytest.approx(0.5)

    def test_side_larger_than_grid_is_rejected(self) -> None:
        with pytest.raises(SystemExit):
            box_slice(64, 65)

    def test_target_is_a_binary_block_of_the_requested_size(self) -> None:
        target = box_target(64, 50, torch.device("cpu"))
        assert target.shape == (1, 1, 64, 64)
        assert set(torch.unique(target).tolist()) == {0.0, 1.0}
        assert float(target.sum()) == pytest.approx(50 * 50)


class TestTargetMatchesScoredRegion:
    def test_optimised_region_is_the_scored_region(self) -> None:
        """The pixels we fill must be the pixels the canonical metric measures.

        These live in different modules (`compute_square_metrics` re-derives the
        box from `centre` and `side`), so a one-pixel disagreement would mean the
        optimiser is graded on the wrong region.
        """
        from ao_shaping.utils.image.beam_metrics import compute_square_metrics

        grid, side = 64, 50
        start = (grid - side) // 2
        mask = torch.zeros(grid, grid, dtype=torch.bool)
        mask[box_slice(grid, side), box_slice(grid, side)] = True

        image = torch.zeros(grid, grid, dtype=torch.float64).numpy()
        image[mask.numpy()] = 1.0  # exactly the target: uniform, nothing outside
        metrics = compute_square_metrics(
            image, target_side=side, center=(float(start + side // 2),) * 2
        )
        # A perfect target must score CV ~ 0 and all the energy inside.
        assert metrics["uniformity_cv"] == pytest.approx(0.0, abs=1e-9)
        assert metrics["encircled_energy"] == pytest.approx(1.0, abs=1e-9)


class TestLoss:
    def test_mse_is_minimised_by_the_target(self) -> None:
        target = box_target(64, 50, torch.device("cpu"))
        mask = torch.zeros(64, 64, dtype=torch.bool)
        window = box_slice(64, 50)
        mask[window, window] = True
        exact = surrogate_loss(target, target, mask, "mse")
        wrong = surrogate_loss(torch.zeros_like(target), target, mask, "mse")
        assert float(exact) == pytest.approx(0.0)
        assert float(wrong) > 0

    def test_quality_loss_ranks_fill_above_spot_above_dark(self) -> None:
        """The degeneracy this guards.

        A multiplicative `uniformity * fill` ties an all-dark field with a 4x4
        hot blob, because CV cannot see darkness. The additive EE term must
        break that tie in the right order.
        """
        target = box_target(64, 50, torch.device("cpu"))
        mask = torch.zeros(64, 64, dtype=torch.bool)
        window = box_slice(64, 50)
        mask[window, window] = True

        filled = torch.zeros(1, 1, 64, 64)
        filled[0, 0, window, window] = 1.0
        dark = torch.zeros(1, 1, 64, 64)
        hot_spot = torch.zeros(1, 1, 64, 64)
        hot_spot[0, 0, 30:34, 30:34] = 1.0

        # Lower loss is better.
        q_filled = float(surrogate_loss(filled, target, mask, "quality"))
        q_spot = float(surrogate_loss(hot_spot, target, mask, "quality"))
        q_dark = float(surrogate_loss(dark, target, mask, "quality"))
        assert q_filled < q_spot, "a uniform fill must beat a 4x4 blob"
        assert q_spot < q_dark, "a dark field must not outrank a concentrated one"

    def test_mse_marginal_prefers_a_hot_spot_over_leaving_the_box_dark(self) -> None:
        """Why the objective had to change.

        MSE rewards filling the target, but it has no term that prefers *light*
        over *dark*: leaving the box empty and cramming the energy into 4x4 score
        within 0.004 of each other, with the concentrated blob marginally
        **ahead**. A uniform fill should beat both by a wide margin.
        """
        target = box_target(64, 50, torch.device("cpu"))
        mask = torch.zeros(64, 64, dtype=torch.bool)
        window = box_slice(64, 50)
        mask[window, window] = True

        filled = torch.zeros(1, 1, 64, 64)
        filled[0, 0, window, window] = 1.0
        dark = torch.zeros(1, 1, 64, 64)
        hot_spot = torch.zeros(1, 1, 64, 64)
        hot_spot[0, 0, 30:34, 30:34] = 1.0

        mse_filled = float(surrogate_loss(filled, target, mask, "mse"))
        mse_dark = float(surrogate_loss(dark, target, mask, "mse"))
        mse_spot = float(surrogate_loss(hot_spot, target, mask, "mse"))
        assert mse_filled < mse_spot, "MSE must still prefer the uniform fill"
        assert mse_spot < mse_dark, (
            "MSE has no light-vs-dark term, so a concentrated blob edges out an "
            "empty box -- which is exactly the degeneracy the additive EE term "
            "in the quality objective removes"
        )

    def test_quality_loss_is_finite_on_a_dead_start(self) -> None:
        """std/mean is undefined where the box mean is 0, which is the cold start."""
        target = box_target(64, 50, torch.device("cpu"))
        mask = torch.zeros(64, 64, dtype=torch.bool)
        window = box_slice(64, 50)
        mask[window, window] = True
        value = float(surrogate_loss(torch.zeros(1, 1, 64, 64), target, mask, "quality"))
        assert np.isfinite(value)


class TestScoring:
    def test_perfect_box_scores_perfectly(self) -> None:
        image = np.zeros((64, 64), dtype=np.float64)
        window = box_slice(64, 50)
        image[window, window] = 1.0
        metrics = score_spot(image, 50)
        assert metrics["uniformity_cv"] == pytest.approx(0.0, abs=1e-9)
        assert metrics["encircled_energy"] == pytest.approx(1.0, abs=1e-9)
        assert metrics["aspect_ratio"] == pytest.approx(1.0, abs=1e-9)

    def test_quality_stays_in_the_unit_interval(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(5):
            metrics = score_spot(rng.random((64, 64)), 50)
            assert 0.0 <= metrics["quality"] <= 1.0


class TestFrozenWeights:
    def test_gradients_reach_the_phase_and_not_the_coefficients(self) -> None:
        """The whole premise: weights frozen, phase trainable."""
        from ml.zernike.models import ZernikeAmpConfig, ZernikeAmpModel

        model = ZernikeAmpModel(ZernikeAmpConfig(n_max=4, grid=64))
        model.requires_grad_(False)
        assert all(not p.requires_grad for p in model.parameters())

        phase = torch.zeros(64, 64, requires_grad=True)
        cos, sin = torch.cos(phase)[None, None], torch.sin(phase)[None, None]
        model(cos, sin).sum().backward()
        assert phase.grad is not None
        assert torch.isfinite(phase.grad).all()
        assert float(phase.grad.abs().sum()) > 0

    def test_checkpoint_geometry_is_used_not_current_defaults(self, tmp_path) -> None:
        """A checkpoint trained at padding=10 must not be rebuilt at the default."""
        from optimize_uniform_spot_phase import load_frozen_model

        checkpoint = tmp_path / "ckpt.pt"
        torch.save(
            {
                "coefficients": torch.zeros(135),
                "n_max": 15, "grid": 64, "observable": "intensity",
                "normalization": "peak", "far_field_padding": 10,
            },
            checkpoint,
        )
        model = load_frozen_model(checkpoint, torch.device("cpu"))
        assert model.far_field_padding == 10
        assert model.K == 135
        assert all(not p.requires_grad for p in model.parameters())

    def test_coefficient_count_mismatch_is_refused(self, tmp_path) -> None:
        from optimize_uniform_spot_phase import load_frozen_model

        checkpoint = tmp_path / "bad.pt"
        torch.save(
            {
                "coefficients": torch.zeros(7),  # wrong for n_max=15
                "n_max": 15, "grid": 64, "observable": "intensity",
                "normalization": "peak", "far_field_padding": 10,
            },
            checkpoint,
        )
        with pytest.raises(SystemExit, match="coefficients"):
            load_frozen_model(checkpoint, torch.device("cpu"))


class TestExtrapolationDiagnostic:
    def test_zernike_limited_phase_scores_high_and_freeform_low(self) -> None:
        """The check that decides how much the cross-check disagreement means."""
        from ml.zernike.models import ZernikeBasis

        grid, n_max = 64, 15
        basis = ZernikeBasis(grid, n_max).as_tensor().numpy()
        coefficients = np.linspace(-1.0, 1.0, basis.shape[0])
        in_band = torch.from_numpy((basis * coefficients[:, None, None]).sum(0)).float()

        rng = np.random.default_rng(0)
        freeform = torch.from_numpy(rng.normal(0, np.pi, (grid, grid))).float()

        def zfraction(phase: torch.Tensor) -> float:
            return diagnose_extrapolation(phase, index_cache="none", family="none")[
                "zernike_energy_fraction"
            ]

        assert zfraction(in_band) > 0.99  # by construction it IS in the span
        assert zfraction(freeform) < 0.5

class TestPhaseVariable:
    """Band-limited parametrisation: the DOF count and the reconstruction must agree."""

    def test_freeform_dof_is_the_pixel_count(self):
        variable = PhaseVariable(16, 15, torch.device("cpu"), mode="freeform")
        assert variable.tensor.numel() == 16 * 16
        assert variable.to_phase().shape == (16, 16)

    @pytest.mark.parametrize("n_max,dof", [(4, 14), (8, 44), (15, 135), (20, 230)])
    def test_zernike_dof_excludes_piston(self, n_max, dof):
        # (n+1)(n+2)/2 - 1: the -1 is piston, which cannot change intensity.
        variable = PhaseVariable(16, n_max, torch.device("cpu"), mode="zernike")
        assert variable.tensor.numel() == dof
        assert variable.to_phase().shape == (16, 16)

    def test_zernike_reconstruction_is_exact_at_random_coefficients(self):
        # to_phase() must be the plain basis @ coefficients, not an approximation.
        n_max = 8
        variable = PhaseVariable(24, n_max, torch.device("cpu"), mode="zernike")
        with torch.no_grad():
            variable.tensor.copy_(torch.randn(variable.tensor.numel()))
        rebuilt = PhaseVariable(
            24, n_max, torch.device("cpu"), mode="zernike", init=variable.to_phase()[None]
        )
        assert torch.allclose(rebuilt.tensor, variable.tensor, atol=1e-4)

    def test_zernike_phase_lands_in_the_band_limited_span(self):
        # The whole point: a synthesised phase must be reconstructible from its own
        # Zernike coefficients, which is what diagnose_extrapolation measures.
        # Relative tolerance: 135 modes summed reach tens of radians, so an absolute
        # 1e-5 on the phase would be tighter than float32 round-off (~2e-6 relative).
        grid = 64
        variable = PhaseVariable(grid, 15, torch.device("cpu"), mode="zernike")
        with torch.no_grad():
            variable.tensor.copy_(torch.randn(variable.tensor.numel()) * 0.3)
        phase = variable.to_phase()
        again = PhaseVariable(grid, 15, torch.device("cpu"), mode="zernike", init=phase[None])
        scale = float(phase.abs().max())
        error = float((again.to_phase() - phase).abs().max())
        assert error / scale < 1e-5, f"relative reconstruction error {error / scale:.2e}"

    def test_gradients_reach_the_coefficients(self):
        variable = PhaseVariable(16, 4, torch.device("cpu"), mode="zernike")
        variable.to_phase().sum().backward()
        assert variable.tensor.grad is not None
        assert torch.isfinite(variable.tensor.grad).all()
        assert variable.tensor.grad.abs().sum() > 0
