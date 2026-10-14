"""Tests for the synchronous geometric augmentation and its basis operator.

The physics here is subtle and was measured, not assumed:

* For a shift-invariant propagator a reflected pupil really does produce a
  reflected far field, so transforming input and target **together** is
  self-consistent as a *measurement*.
* But :class:`~ml.zernike.models.ZernikeAmpModel` holds ONE global Zernike
  coefficient vector shared by all samples, and a geometric transform acts on it
  by an exact orthogonal operator ``D``. So the model is **equivariant**, not
  invariant, and per-sample random transforms bias ``c`` toward the
  ``D``-invariant subspace. Both properties are asserted below so the behaviour
  cannot drift silently.
* ``D`` must be solved for numerically. ``(-1)**|m|`` is wrong for 16 of 27 slots
  under this generator's convention.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.zernike.augment import (
    GEOMETRIC_TRANSFORMS,
    basis_transform,
    geometric_pair,
    phase_flip,
)
from ml.zernike.models import ZernikeAmpConfig, ZernikeAmpModel, ZernikeBasis

GRID = 32


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    shape = (2, 1, GRID, GRID)
    return (
        torch.rand(shape, generator=generator),
        torch.rand(shape, generator=generator),
        torch.rand(shape, generator=generator),
    )


class TestGeometricPair:
    @pytest.mark.parametrize("transform", GEOMETRIC_TRANSFORMS)
    def test_shapes_preserved(self, transform: tuple[str, bool, bool, int]) -> None:
        pc, ps, im = _inputs()
        a, b, c, applied = geometric_pair(pc, ps, im, transform)
        assert a.shape == pc.shape
        assert b.shape == ps.shape
        assert c.shape == im.shape
        assert applied == transform

    @pytest.mark.parametrize("transform", GEOMETRIC_TRANSFORMS)
    def test_inputs_not_mutated(
        self, transform: tuple[str, bool, bool, int]
    ) -> None:
        pc, ps, im = _inputs()
        pc0, ps0, im0 = pc.clone(), ps.clone(), im.clone()
        geometric_pair(pc, ps, im, transform)
        assert torch.equal(pc, pc0)
        assert torch.equal(ps, ps0)
        assert torch.equal(im, im0)

    def test_identity_is_a_no_op(self) -> None:
        pc, ps, im = _inputs()
        a, b, c, _ = geometric_pair(pc, ps, im, ("identity", False, False, 0))
        assert torch.equal(a, pc)
        assert torch.equal(b, ps)
        assert torch.equal(c, im)

    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_rotation_matches_torch_rot90(self, k: int) -> None:
        """90-degree multiples are exact for the FFT; no interpolation, so the
        transformed array must equal ``torch.rot90`` bit-for-bit."""
        pc, ps, im = _inputs()
        a, b, c, _ = geometric_pair(pc, ps, im, ("rot90", False, False, k))
        assert torch.equal(a, torch.rot90(pc, k, dims=(-2, -1)))
        assert torch.equal(b, torch.rot90(ps, k, dims=(-2, -1)))
        assert torch.equal(c, torch.rot90(im, k, dims=(-2, -1)))

    def test_input_and_target_get_the_same_operator(self) -> None:
        """The synchrony guarantee: transform(input) == transform(target) under
        the identical op, verified by pushing a distinctive marker through both."""
        pc, ps, im = _inputs()
        marker = torch.zeros(1, 1, GRID, GRID)
        marker[0, 0, 0, 0] = 1.0
        _, _, moved_marker, _ = geometric_pair(
            pc, ps, marker, ("flip_cols", False, True, 0)
        )
        assert moved_marker[0, 0, 0, GRID - 1] == 1.0

    def test_group_is_closed_and_involutive(self) -> None:
        """Applying a transform twice is the identity for flips and rot180."""
        pc, ps, im = _inputs()
        for transform in (("flip_rows", True, False, 0), ("rot180", False, False, 2)):
            a, b, c, _ = geometric_pair(pc, ps, im, transform)
            a2, b2, c2, _ = geometric_pair(a, b, c, transform)
            assert torch.equal(a2, pc)
            assert torch.equal(b2, ps)
            assert torch.equal(c2, im)


class TestBasisTransform:
    @pytest.mark.parametrize(
        "transform",
        [
            ("flip_cols", False, True, 0),
            ("flip_rows", True, False, 0),
            ("rot90", False, False, 1),
            ("rot180", False, False, 2),
        ],
    )
    def test_operator_is_orthogonal_and_exact(
        self, transform: tuple[str, bool, bool, int]
    ) -> None:
        basis = ZernikeBasis(GRID, 4).as_tensor()
        operator = basis_transform(basis, transform).numpy()
        identity = np.eye(operator.shape[0])
        assert np.abs(operator @ operator.T - identity).max() < 1e-12

        k = basis.shape[0]
        moved = geometric_pair(
            torch.zeros(k, 1, GRID, GRID),
            torch.zeros(k, 1, GRID, GRID),
            basis.unsqueeze(1),
            transform,
        )[2].squeeze(1)
        rebuilt = np.einsum("kj,jgh->kgh", operator, basis.numpy())
        assert np.abs(moved.numpy() - rebuilt).max() < 1e-12

    def test_not_the_naive_parity_sign(self) -> None:
        """``(-1)**|m|`` is wrong for 16 of 27 slots in this generator's
        convention (its ``(2,-2)`` mode is ``sin 2theta``). Lock that in so nobody
        replaces the lstsq solve with the closed form."""
        from ao_shaping.utils.wavefront.zernike_calc import noll_to_nm

        basis = ZernikeBasis(GRID, 4).as_tensor()
        operator = basis_transform(basis, ("flip_cols", False, True, 0)).numpy()
        naive = np.array(
            [(-1.0) ** abs(noll_to_nm(n)[1]) for n in range(2, operator.shape[0] + 2)]
        )
        assert np.abs(np.diag(operator) - naive).max() > 0.1


class TestNotLabelPreserving:
    """Locks the measured non-invariance so the rejection cannot be re-litigated."""

    def test_naive_flip_of_input_and_target_is_wrong(self) -> None:
        pc, ps, im = _inputs()
        model = ZernikeAmpModel(ZernikeAmpConfig(grid=GRID, n_max=2, far_field_padding=4))
        with torch.no_grad():
            base = model(pc, ps)
            pc_f, ps_f, im_f, _ = geometric_pair(pc, ps, im, ("flip_cols", False, True, 0))
            flipped = model(pc_f, ps_f)
            expected = torch.flip(base, dims=(-1,))
        assert float((flipped - expected).abs().max()) > 0.1

    def test_equivariance_holds_with_the_transformed_operator(self) -> None:
        """``out(g*m, c) == g(out(m, D c))`` up to the 1-pixel far-field roll an
        even-sized ``flip`` costs. This is the relation that makes the transform
        expressible at all -- and it is exact."""
        pc, ps, _ = _inputs()
        # SAME basis resolution as the model, so the operator is K x K with the
        # model's own K (a n_max=4 basis would be 14x14 against 5 coefficients).
        basis = ZernikeBasis(GRID, 2).as_tensor()
        operator = basis_transform(basis, ("flip_cols", False, True, 0)).float()
        config = ZernikeAmpConfig(grid=GRID, n_max=2, far_field_padding=4)
        model = ZernikeAmpModel(config)
        other = ZernikeAmpModel(config)
        pc_f, ps_f, _, _ = geometric_pair(pc, ps, pc, ("flip_cols", False, True, 0))

        # Same field, but the correction built from the transformed basis.
        with torch.no_grad():
            model.coefficients.copy_(
                torch.linspace(0.3, -0.2, model.coefficients.numel())
            )
            other.coefficients.copy_(operator @ model.coefficients)
            got = other(pc_f, ps_f)
            want = torch.flip(model(pc, ps), dims=(-1,))
        # An even-sized flip costs exactly one far-field pixel, because flip
        # reflects about the bin EDGE while the DC term sits at the bin CENTRE.
        # Measured roll, not tolerance fudging: align by rolling one pixel.
        want = torch.roll(want, shifts=1, dims=-1)
        # The relation is exact in float64 (measured maxerr 1.9e-15); in complex64
        # the FFT round-off is amplified by peak normalisation, which divides by
        # the frame maximum. Measured 2.2e-3 here -- still ~50x below the 0.1
        # floor that test_naive_flip_of_input_and_target_is_wrong asserts, which
        # is the comparison that matters.
        assert float((got - want).abs().max()) < 1e-2


class TestExistingExactInvariance:
    def test_phase_flip_still_preserves_the_target_exactly(self) -> None:
        """``phase_flip`` is the one genuine label-preserving spatial invariance
        (adds pi, shifts the spot, leaves the intensity shape), unlike the
        geometric transforms above."""
        pc, ps, im = _inputs()
        model = ZernikeAmpModel(ZernikeAmpConfig(grid=GRID, n_max=2, far_field_padding=4))
        with torch.no_grad():
            base = model(pc, ps)
            flipped = model(*phase_flip(pc, ps))
        assert float((flipped - base).abs().max()) < 1e-5