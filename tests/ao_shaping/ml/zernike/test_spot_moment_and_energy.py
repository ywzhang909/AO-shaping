"""Tests for the spot-size (second-moment) loss term and energy-conserving output.

Both were added in response to two specific gaps:

* a pixel-wise MSE is dominated by the bright core, so it barely notices a prediction
  that matches the peak while getting the *spread* wrong;
* peak normalisation destroys the absolute scale, so a loss cannot anchor against
  energy even though the propagator itself is exactly conservative.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.zernike.losses import (
    LossConfig,
    composite_loss,
    roi_mask,
    second_moments,
    spot_moment_gap_term,
)
from ml.zernike.models import ZernikeAmpConfig, ZernikeAmpModel

GRID_N, GRID = 16, 16


def _delta(rows: int, cols: int, row0: int, col0: int) -> torch.Tensor:
    img = torch.zeros(1, 1, GRID_N, GRID_N)
    img[0, 0, row0 : row0 + rows, col0 : col0 + cols] = 1.0
    return img


class TestSecondMoments:
    def test_single_pixel_has_zero_extent(self) -> None:
        var_x, var_y, var_r = second_moments(_delta(1, 1, 8, 8))
        assert float(var_x[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(var_y[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(var_r[0]) == pytest.approx(0.0, abs=1e-6)

    def test_uniform_block_matches_the_discrete_variance(self) -> None:
        """A 3x3 block spanning indices 7..9 has variance (1+0+1)/3 = 2/3 per axis."""
        var_x, var_y, var_r = second_moments(_delta(3, 3, 7, 7))
        assert float(var_x[0]) == pytest.approx(2.0 / 3.0, abs=1e-6)
        assert float(var_y[0]) == pytest.approx(2.0 / 3.0, abs=1e-6)
        assert float(var_r[0]) == pytest.approx(4.0 / 3.0, abs=1e-6)

    def test_radial_moment_is_the_sum_of_the_axes(self) -> None:
        var_x, var_y, var_r = second_moments(_delta(5, 3, 6, 7))
        assert float(var_r[0]) == pytest.approx(float(var_x[0]) + float(var_y[0]))

    def test_elongated_block_separates_the_axes(self) -> None:
        """A bar is wide in one axis and narrow in the other; that must show up."""
        var_x, var_y, _ = second_moments(_delta(1, 9, 8, 4))
        assert float(var_x[0]) > float(var_y[0])

    def test_centred_and_off_centred_give_the_same_variance(self) -> None:
        """Variance is about the centroid, so translating a symmetric blob is a no-op."""
        centred = second_moments(_delta(3, 3, 6, 6))[2]
        shifted = second_moments(_delta(3, 3, 10, 10))[2]
        assert float(centred[0]) == pytest.approx(float(shifted[0]), abs=1e-6)

    def test_shape_is_per_sample_batch(self) -> None:
        batch = torch.cat([_delta(3, 3, 7, 7), _delta(1, 1, 8, 8)], dim=0)
        var_x, _, _ = second_moments(batch)
        assert var_x.shape == (2,)
        assert float(var_x[1]) < float(var_x[0])

    def test_nan_pixels_are_ignored(self) -> None:
        img = _delta(3, 3, 7, 7)
        img[0, 0, 0, 0] = float("nan")
        assert torch.isfinite(second_moments(img)[2]).all()

    def test_mask_restricts_the_measurement(self) -> None:
        img = _delta(1, 1, 2, 2) + _delta(1, 1, 13, 13)
        mask = roi_mask((GRID_N, GRID_N), (3.0, 13.0), "rectangle", 3, 1.0)
        full, _ = second_moments(img)[2], None
        restricted = second_moments(img, mask)[2]
        assert float(restricted) < float(full[0])

    def test_rejects_wrong_rank(self) -> None:
        with pytest.raises(ValueError, match=r"must be \(B,1,H,W\)"):
            second_moments(torch.zeros(1, GRID_N, GRID_N))

    def test_rejects_mask_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="does not match mask"):
            second_moments(_delta(3, 3, 7, 7), roi_mask((8, 8), (4.0, 4.0), "rectangle", 4, 1.0))


class TestSpotMomentGapTerm:
    def test_identical_inputs_give_zero(self) -> None:
        img = _delta(3, 3, 7, 7)
        assert float(spot_moment_gap_term(img, img)[0]) == pytest.approx(0.0, abs=1e-6)

    def test_too_tight_scores_one(self) -> None:
        """A delta spot against a wide target: var_r -> 0, so the relative error -> 1."""
        tight, wide = _delta(1, 1, 8, 8), _delta(9, 9, 4, 4)
        assert float(spot_moment_gap_term(tight, wide)[0]) == pytest.approx(1.0, abs=1e-4)

    def test_is_asymmetric_by_design(self) -> None:
        """``|a - b| / b`` is anchored, so swapping the arguments changes the value.

        That is the point: the reference is always the measurement. A symmetric
        distance would let a prediction be graded against its own scale.
        """
        tight, wide = _delta(1, 1, 8, 8), _delta(7, 7, 5, 5)
        forward = float(spot_moment_gap_term(tight, wide)[0])
        reverse = float(spot_moment_gap_term(wide, tight)[0])
        assert forward != pytest.approx(reverse, abs=1e-6)

    def test_is_anchored_not_absolute(self) -> None:
        """The reference must be the target, so scaling both by k changes nothing."""
        a, b = _delta(3, 3, 7, 7), _delta(5, 5, 6, 6)
        once = float(spot_moment_gap_term(a, b)[0])
        scaled = float(spot_moment_gap_term(a * 100.0, b * 100.0)[0])
        assert once == pytest.approx(scaled, rel=1e-4)

    def test_gradient_reaches_the_prediction(self) -> None:
        pred = _delta(3, 3, 7, 7).clone().requires_grad_(True)
        target = _delta(5, 5, 6, 6)
        spot_moment_gap_term(pred, target).sum().backward()
        assert pred.grad is not None
        assert float(pred.grad.abs().sum()) > 0


class TestCompositeIntegration:
    def test_term_appears_in_the_output(self) -> None:
        mask = roi_mask((GRID_N, GRID_N), (8.0, 8.0), "rectangle", 8, 1.0)
        out = composite_loss(
            _delta(3, 3, 7, 7), _delta(5, 5, 6, 6), mask,
            LossConfig(w_mse=1.0, w_spot_moment=0.5),
        )
        assert "spot_moment" in out
        assert float(out["spot_moment"][0]) > 0

    def test_zero_weight_omits_the_term(self) -> None:
        mask = roi_mask((GRID_N, GRID_N), (8.0, 8.0), "rectangle", 8, 1.0)
        out = composite_loss(
            _delta(3, 3, 7, 7), _delta(5, 5, 6, 6), mask, LossConfig(w_mse=1.0)
        )
        assert "spot_moment" not in out

    def test_weight_scales_the_contribution(self) -> None:
        mask = roi_mask((GRID_N, GRID_N), (8.0, 8.0), "rectangle", 8, 1.0)
        args = (_delta(3, 3, 7, 7), _delta(5, 5, 6, 6), mask)
        one = composite_loss(*args, LossConfig(w_mse=0.0, w_spot_moment=1.0))
        two = composite_loss(*args, LossConfig(w_mse=0.0, w_spot_moment=2.0))
        assert float(two["_mean_total"]) == pytest.approx(
            2.0 * float(one["_mean_total"]), rel=1e-5
        )


class TestConserveEnergy:
    """The propagator is orthonormal, so Parseval already holds; peak normalisation is
    what destroys it. These pin the post-processing that puts it back."""

    @staticmethod
    def _model(conserve: bool, normalization: str = "peak") -> ZernikeAmpModel:
        torch.manual_seed(0)
        return ZernikeAmpModel(
            ZernikeAmpConfig(
                n_max=4, grid=GRID_N, far_field_padding=2,
                normalization=normalization, conserve_energy=conserve,
            )
        )

    @staticmethod
    def _input_total(model: ZernikeAmpModel, cos: torch.Tensor, sin: torch.Tensor) -> float:
        measured = torch.complex(cos, sin)
        unit = torch.polar(
            torch.ones_like(model.correction_phase()), model.correction_phase()
        )[None, None]
        field = measured * unit
        return float((field.real**2 + field.imag**2).sum())

    def test_disabled_preserves_the_old_scale(self) -> None:
        model = self._model(conserve=False)
        ones = torch.ones(1, 1, GRID_N, GRID_N)
        out = model(ones, ones)
        assert float(out.sum()) != pytest.approx(
            self._input_total(model, ones, ones), rel=1e-3
        ), "without the flag the peak normalisation should still cost the total"

    def test_enabled_matches_the_input_total_exactly(self) -> None:
        model = self._model(conserve=True)
        ones = torch.ones(1, 1, GRID_N, GRID_N)
        out = model(ones, ones)
        assert float(out.sum()) == pytest.approx(
            self._input_total(model, ones, ones), rel=1e-5
        )

    def test_anchor_tracks_partial_coherence(self) -> None:
        """The corpus stores coherent block averages, so |phasor| < 1 in general.

        The conserved total must follow the input, not a fixed constant.
        """
        model = self._model(conserve=True)
        half = torch.full((1, 1, GRID_N, GRID_N), 0.5)
        out = model(half, half)
        assert float(out.sum()) == pytest.approx(
            self._input_total(model, half, half), rel=1e-5
        )
        assert float(out.sum()) < float(
            self._model(conserve=True)(torch.ones(1, 1, GRID_N, GRID_N),
                                       torch.ones(1, 1, GRID_N, GRID_N)).sum()
        ), "a dimmer, less coherent input must conserve less energy"

    def test_works_with_sum_normalisation_too(self) -> None:
        model = self._model(conserve=True, normalization="sum")
        ones = torch.ones(1, 1, GRID_N, GRID_N)
        out = model(ones, ones)
        assert float(out.sum()) == pytest.approx(
            self._input_total(model, ones, ones), rel=1e-5
        )

    def test_applies_to_correction_far_field(self) -> None:
        model = self._model(conserve=True)
        out = model.correction_far_field()
        # pupil is a unit-modulus phasor over the whole grid
        assert float(out.sum()) == pytest.approx(float(GRID_N**2), rel=1e-5)

    def test_unnormalized_output_is_left_alone(self) -> None:
        model = self._model(conserve=True)
        raw = model.correction_far_field(normalize=False)
        assert torch.isfinite(raw).all()

    def test_is_differentiable(self) -> None:
        model = self._model(conserve=True)
        phase = torch.zeros(1, 1, GRID_N, GRID_N, requires_grad=True)
        phasor = torch.polar(torch.ones_like(phase), phase)
        model(phasor.real, phasor.imag).sum().backward()
        assert phase.grad is not None and torch.isfinite(phase.grad).all()
