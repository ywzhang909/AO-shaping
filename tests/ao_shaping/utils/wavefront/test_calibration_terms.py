"""``calibration_terms``: the pupil affine τ_θ and the travel scale γ.

Covers the parameterisation, the clamps, and the two torch-side applications
(``apply_pupil_affine`` / ``apply_travel_scale``). The *default-off* contract
of the optimizer that consumes them is covered in
``tests/ao_shaping/algorithm/signal_processing/test_zernike_coefficient_optimizer_selfcal.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ao_shaping.utils.wavefront.calibration_terms import (
    AFFINE_PARAM_NAMES,
    FAMILY_NAMES,
    PupilAffine,
    TravelScale,
    apply_pupil_affine,
    apply_travel_scale,
    family_index_vector,
    family_of_noll,
    travel_gain_tensor,
    travel_gain_vector,
)


class TestPupilAffine:
    """The six-scalar in-pupil affine (GRAPHYCS's τ_θ)."""

    def test_identity_is_the_default(self) -> None:
        """A fresh affine must be the identity transform."""
        affine = PupilAffine()
        assert affine.is_identity
        assert affine.to_vector().tolist() == [1.0, 1.0, 0.0, 0.0, 0.0, 1.0]

    def test_parameter_order_matches_the_paper(self) -> None:
        """The vector order is the paper's [amp_x, amp_y, off_x, off_y, rot, grid]."""
        assert AFFINE_PARAM_NAMES == (
            "amp_x",
            "amp_y",
            "off_x",
            "off_y",
            "rot",
            "grid",
        )
        affine = PupilAffine(1.1, 0.9, 0.2, -0.3, 0.05, 1.4)
        assert affine.to_vector().tolist() == [1.1, 0.9, 0.2, -0.3, 0.05, 1.4]

    def test_vector_round_trip(self) -> None:
        """``from_vector(to_vector())`` is the identity."""
        affine = PupilAffine(1.2, 0.8, 0.1, 0.2, 0.3, 0.9)
        assert PupilAffine.from_vector(affine.to_vector()) == affine

    def test_clamped_keeps_every_component_in_range(self) -> None:
        """The clamps bound the scale factors, the offsets and the angle."""
        wild = PupilAffine(amp_x=99.0, amp_y=-5.0, off_x=17.0, off_y=-3.0,
                           rot=12.0, grid=0.01)
        clamped = wild.clamped()
        assert clamped.amp_x == PupilAffine.AMP_MAX
        assert clamped.amp_y == PupilAffine.AMP_MIN
        assert clamped.off_x == PupilAffine.OFF_LIMIT
        assert clamped.off_y == -PupilAffine.OFF_LIMIT
        assert clamped.rot == PupilAffine.ROT_LIMIT
        assert clamped.grid == PupilAffine.GRID_MIN

    def test_step_metrics(self) -> None:
        """The step helpers report the per-parameter change."""
        before = PupilAffine()
        after = PupilAffine(off_x=0.5)
        assert after.max_abs_step(before) == pytest.approx(0.5)
        assert after.step_l2(before) == pytest.approx(0.5)


class TestTravelScale:
    """The per-mode-family travel scale (GRAPHYCS's γ)."""

    def test_default_is_no_scaling(self) -> None:
        """All-zero γ means "no travel error"."""
        scale = TravelScale()
        assert scale.is_identity
        assert scale.scales == tuple([1.0] * len(FAMILY_NAMES))

    def test_clamp_keeps_the_scale_factor_nonnegative(self) -> None:
        """The paper clamps γ >= -1 so the scale factor 1 + γ stays >= 0."""
        wild = TravelScale((-5.0,) * len(FAMILY_NAMES))
        clamped = wild.clamped()
        assert all(value == TravelScale.GAMMA_MIN for value in clamped.gammas)
        assert all(value >= 0.0 for value in clamped.scales)

    def test_upper_bound_is_ours_not_the_papers(self) -> None:
        """The upper bound is this repository's rail, not the paper's."""
        wild = TravelScale((99.0,) * len(FAMILY_NAMES))
        assert all(
            value == TravelScale.GAMMA_MAX for value in wild.clamped().gammas
        )

    def test_wrong_family_count_is_rejected(self) -> None:
        """A γ vector of the wrong length must not be silently accepted."""
        with pytest.raises(ValueError):
            TravelScale((0.1, 0.2))


class TestFamilyMapping:
    """The mode-family grouping of the travel scale."""

    def test_families_cover_every_noll_index(self) -> None:
        """Each Noll index lands in exactly one of the seven families."""
        assert family_of_noll(1) == 0
        assert family_of_noll(2) == 1
        assert family_of_noll(3) == 1
        assert family_of_noll(4) == 2
        assert family_of_noll(5) == 3
        assert family_of_noll(6) == 3
        assert family_of_noll(7) == 4
        assert family_of_noll(8) == 4
        assert family_of_noll(9) == 5
        assert family_of_noll(10) == 5
        assert family_of_noll(11) == 6
        assert family_of_noll(66) == 6

    def test_invalid_noll_is_rejected(self) -> None:
        """Noll indices start at 1."""
        with pytest.raises(ValueError):
            family_of_noll(0)

    def test_gain_vector_expands_per_family(self) -> None:
        """Each mode gets the scale of its family."""
        gammas = np.arange(len(FAMILY_NAMES), dtype=np.float64)
        gains = travel_gain_vector(gammas, 11)
        assert gains[0] == pytest.approx(1.0 + gammas[0])  # Noll 1 -> piston
        assert gains[3] == pytest.approx(1.0 + gammas[2])  # Noll 4 -> defocus
        assert gains[10] == pytest.approx(1.0 + gammas[6])  # Noll 11 -> higher

    def test_family_index_vector_matches_the_scalar_mapping(self) -> None:
        """The vectorised mapping agrees with the scalar one."""
        indices = family_index_vector(20)
        for noll in range(1, 21):
            assert int(indices[noll - 1]) == family_of_noll(noll)


class TestTorchApplications:
    """The two torch-side applications."""

    def test_affine_at_identity_is_a_no_op(self) -> None:
        """The identity affine leaves the total phase and the envelope alone."""
        torch = pytest.importorskip("torch")
        rng = np.random.default_rng(0)
        side = 32
        command = torch.tensor(rng.normal(0, 1, (side, side)), dtype=torch.float64)
        aberration = torch.tensor(
            rng.normal(0, 0.3, (side, side)), dtype=torch.float64
        )
        amplitude = torch.tensor(
            np.exp(-((np.mgrid[0:side, 0:side][0] - side / 2) ** 2) / 50.0),
            dtype=torch.float64,
        )
        params = torch.tensor(PupilAffine().to_vector(), dtype=torch.float64)
        patch, envelope = apply_pupil_affine(
            command, aberration, amplitude, params
        )
        expected = command + aberration
        assert torch.allclose(patch, expected, atol=1e-12)
        assert torch.allclose(envelope, amplitude, atol=1e-12)

    def test_affine_moves_the_envelope(self) -> None:
        """A non-identity affine must actually change the forward model."""
        torch = pytest.importorskip("torch")
        side = 32
        command = torch.zeros((side, side), dtype=torch.float64)
        aberration = torch.zeros((side, side), dtype=torch.float64)
        amplitude = torch.tensor(
            np.exp(-((np.mgrid[0:side, 0:side][0] - side / 2) ** 2) / 50.0),
            dtype=torch.float64,
        )
        params = torch.tensor(
            PupilAffine(off_x=0.4, amp_y=1.3).to_vector(), dtype=torch.float64
        )
        _, envelope = apply_pupil_affine(
            command, aberration, amplitude, params
        )
        assert not torch.allclose(envelope, amplitude, atol=1e-6)

    def test_travel_scale_at_zero_gamma_is_bit_exact(self) -> None:
        """γ = 0 must leave the command phase bit-for-bit unchanged."""
        torch = pytest.importorskip("torch")
        from ao_shaping.algorithm.signal_processing.zernike_coefficient_optimizer import (
            _projector_for_basis,
        )
        from ao_shaping.utils.wavefront.zernike_calc import (
            ZernikeGenerator,
            calc_n_zernike_terms,
        )

        side = 32
        n_orders = 4
        n_coeffs = calc_n_zernike_terms(n_orders)
        generator = ZernikeGenerator((side, side), radius=side // 2, n_orders=n_orders)
        basis = np.zeros((n_coeffs, side, side))
        for index in range(n_coeffs):
            weights = np.zeros(n_coeffs)
            weights[index] = 1.0
            mode = np.asarray(generator.generate_noll(weights), dtype=np.float64)
            basis[index] = np.nan_to_num(mode, nan=0.0)
        projector = torch.tensor(
            _projector_for_basis(basis), dtype=torch.float64
        )
        basis_flat = torch.tensor(
            basis.reshape(n_coeffs, -1), dtype=torch.float64
        )
        rng = np.random.default_rng(3)
        command = torch.tensor(rng.normal(0, 2, (side, side)), dtype=torch.float64)
        gammas = torch.zeros(len(FAMILY_NAMES), dtype=torch.float64)
        scaled = apply_travel_scale(command, gammas, projector, basis_flat)
        assert torch.equal(scaled, command)

    def test_travel_scale_changes_the_command(self) -> None:
        """A non-zero γ must change the command phase."""
        torch = pytest.importorskip("torch")
        from ao_shaping.algorithm.signal_processing.zernike_coefficient_optimizer import (
            _projector_for_basis,
        )
        from ao_shaping.utils.wavefront.zernike_calc import (
            ZernikeGenerator,
            calc_n_zernike_terms,
        )

        side = 32
        n_orders = 4
        n_coeffs = calc_n_zernike_terms(n_orders)
        generator = ZernikeGenerator((side, side), radius=side // 2, n_orders=n_orders)
        basis = np.zeros((n_coeffs, side, side))
        for index in range(n_coeffs):
            weights = np.zeros(n_coeffs)
            weights[index] = 1.0
            mode = np.asarray(generator.generate_noll(weights), dtype=np.float64)
            basis[index] = np.nan_to_num(mode, nan=0.0)
        projector = torch.tensor(
            _projector_for_basis(basis), dtype=torch.float64
        )
        basis_flat = torch.tensor(
            basis.reshape(n_coeffs, -1), dtype=torch.float64
        )
        rng = np.random.default_rng(3)
        command = torch.tensor(rng.normal(0, 2, (side, side)), dtype=torch.float64)
        gammas = torch.zeros(len(FAMILY_NAMES), dtype=torch.float64)
        gammas[2] = 0.3  # the defocus family
        scaled = apply_travel_scale(command, gammas, projector, basis_flat)
        assert not torch.allclose(scaled, command, atol=1e-9)

    def test_travel_gain_tensor_keeps_the_gradient(self) -> None:
        """The differentiable path must not break the γ gradient."""
        torch = pytest.importorskip("torch")
        gammas = torch.zeros(len(FAMILY_NAMES), dtype=torch.float64,
                             requires_grad=True)
        gains = travel_gain_tensor(gammas, 11)
        gains.sum().backward()
        assert gammas.grad is not None
        assert math.isfinite(float(gammas.grad.sum()))
