"""Tests for the ellipse-fit gap term and the augmentation helpers.

The ellipse term's whole justification is that it sees things the radial second moment
cannot -- spot position and tilt. So the tests below assert exactly that difference rather
than only checking shapes: a metric that is uniformly better is not the claim being made,
and a test that only checked the diagonal would pass even if the term were blind.
"""

from __future__ import annotations

import math

import pytest
import torch

from ml.zernike.augment import phase_flip, phase_shift, phasor_noise, strong_phase_pairs
from ml.zernike.losses import ellipse_gap_term, ellipse_parameters, roi_mask, spot_moment_gap_term

GRID = 64


def _spot(cx=32.0, cy=32.0, sx=3.0, sy=5.0, theta=0.0) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.arange(GRID, dtype=torch.float32) + 0.5,
        torch.arange(GRID, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    x, y = xs - cx, ys - cy
    c, s = math.cos(theta), math.sin(theta)
    xr, yr = x * c + y * s, -x * s + y * c
    return torch.exp(-(xr**2 / (2 * sx**2) + yr**2 / (2 * sy**2)))[None, None]


@pytest.fixture
def mask() -> torch.Tensor:
    return roi_mask((GRID, GRID), (GRID / 2, GRID / 2), "rectangle", 0.375 * GRID, 4 / 3)


class TestEllipseParameters:
    def test_recovers_a_known_spot(self, mask: torch.Tensor) -> None:
        cx, cy, var_x, var_y, _ = ellipse_parameters(_spot(), mask)
        assert float(cx[0]) == pytest.approx(32.0, abs=1e-3)
        assert float(cy[0]) == pytest.approx(32.0, abs=1e-3)
        assert float(var_x[0]) == pytest.approx(9.0, rel=1e-3)

    def test_returns_batch_shaped_tensors(self, mask: torch.Tensor) -> None:
        """(B,1), not (B,1,1) -- composite_loss stacks these and mixed ranks fail there."""
        batch = torch.cat([_spot(), _spot(cx=40.0)], dim=0)
        for part in ellipse_parameters(batch, mask):
            assert part.shape == (2,)

    def test_rejects_wrong_rank(self) -> None:
        with pytest.raises(ValueError, match=r"\(B,1,H,W\)"):
            ellipse_parameters(torch.ones(4, 4))

    def test_rotation_shows_up_in_covariance(self, mask: torch.Tensor) -> None:
        """The point of the term: tilt is invisible to var_x + var_y but not to cov."""
        _, _, var_x, var_y, cov = ellipse_parameters(_spot(theta=math.pi / 4), mask)
        assert float(cov[0]) != pytest.approx(0.0, abs=1e-3)


class TestEllipseGap:
    def test_zero_on_an_exact_match(self, mask: torch.Tensor) -> None:
        ref = _spot()
        assert float(ellipse_gap_term(ref, ref, mask)["_mean_ellipse"]) == pytest.approx(0.0)

    def test_sees_a_shift_that_the_radial_term_misses(self, mask: torch.Tensor) -> None:
        """The claim the term is bought on: 2 px displacement is free for var_x + var_y.

        The radial term is not *exactly* zero here (~8e-6) because the ROI clips the shifted
        spot slightly; what matters is the four-orders-of-magnitude gap, so that is what is
        asserted rather than an exact zero the geometry does not actually deliver.
        """
        ref = _spot()
        shifted = _spot(cx=34.0)
        ellipse = float(ellipse_gap_term(shifted, ref, mask)["_mean_ellipse"])
        radial = float(spot_moment_gap_term(shifted, ref, mask).mean())
        assert ellipse > 0.05
        assert radial < 1e-4
        assert ellipse / radial > 100.0

    def test_sees_a_rotation_the_radial_term_misses(self, mask: torch.Tensor) -> None:
        ref = _spot()
        rotated = _spot(theta=math.pi / 4)
        ellipse = float(ellipse_gap_term(rotated, ref, mask)["_mean_ellipse"])
        radial = float(spot_moment_gap_term(rotated, ref, mask).mean())
        assert ellipse > 1.0
        assert radial < 0.1

    def test_blank_prediction_is_maximally_wrong(self, mask: torch.Tensor) -> None:
        """Guards against the term being minimised by emitting nothing."""
        gap = float(ellipse_gap_term(torch.zeros(1, 1, GRID, GRID), _spot(), mask)["_mean_ellipse"])
        assert gap > 1.0

    def test_is_anchored(self, mask: torch.Tensor) -> None:
        """Anchored means the reference sets the scale: doubling the reference's spread
        must halve the gap for the same absolute error."""
        ref = _spot(sx=3.0)
        pred = _spot(sx=4.0)
        tight = float(ellipse_gap_term(pred, ref, mask)["_mean_ellipse"])
        loose = float(ellipse_gap_term(pred, _spot(sx=6.0), mask)["_mean_ellipse"])
        assert loose < tight


class TestAugment:
    def test_noise_zero_is_a_no_op(self) -> None:
        c = torch.ones(2, 1, 8, 8)
        s = torch.zeros(2, 1, 8, 8)
        nc, ns = phasor_noise(c, s, 0.0)
        assert torch.equal(nc, c) and torch.equal(ns, s)

    def test_noise_std_is_honoured(self) -> None:
        g = torch.Generator().manual_seed(0)
        c = torch.ones(4096, 1, 1, 1)
        nc, _ = phasor_noise(c, torch.zeros_like(c), 0.05, g)
        assert float((nc - c).std()) == pytest.approx(0.05, rel=0.2)

    def test_flip_negates_both_components(self) -> None:
        g = torch.Generator().manual_seed(0)
        c = torch.full((1, 1, 1, 1), 0.7648)
        s = torch.full((1, 1, 1, 1), 0.6442)
        fc, fs = phase_flip(c, s, g)
        sign_c = float(fc[0, 0, 0, 0]) / 0.7648
        sign_s = float(fs[0, 0, 0, 0]) / 0.6442
        assert sign_c == pytest.approx(sign_s, abs=1e-6)
        assert abs(sign_c) == pytest.approx(1.0)

    def test_shift_actually_moves_the_phase(self) -> None:
        g = torch.Generator().manual_seed(3)
        base = torch.zeros(1, 1, 16, 16)
        base[0, 0, 8, 8] = 1.0
        moved = sum(
            not torch.equal(phase_shift(base.clone(), torch.zeros_like(base), 2, g)[0], base)
            for _ in range(8)
        )
        assert moved >= 6

    def test_strong_pairs_are_physical(self) -> None:
        """Targets must be finite and peak-normalised, or the loss sees garbage."""
        phasor, target = strong_phase_pairs(
            6, grid=GRID, beam_w0=30.0, far_field_padding=1,
            aperture_radius=GRID / 2, seed=1,
        )
        assert phasor.shape == (6, 2, GRID, GRID)
        assert target.shape == (6, 1, GRID, GRID)
        assert bool(torch.isfinite(target).all())
        for i in range(6):
            assert float(target[i].max()) == pytest.approx(1.0, abs=1e-5)
            # unit modulus inside the aperture
            modulus = phasor[i, 0] ** 2 + phasor[i, 1] ** 2
            assert float(modulus[24:40, 24:40].mean()) == pytest.approx(1.0, abs=1e-5)