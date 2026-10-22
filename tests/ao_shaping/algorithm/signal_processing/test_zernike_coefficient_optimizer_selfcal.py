"""Step A's self-calibration: the Fourier-domain loss, τ_θ and γ.

The contract these tests pin down:

* **default off, bit-for-bit.** With ``fourier_weight=0``, no affine and no
  travel scale, the optimizer must not even *call* the new code paths, so the
  forward model and the loss stay byte-identical to the pre-change ones.
* the Fourier term adds a **L1** residual on ``Re[fft2]``;
* the affine and the travel scale are **optimiser state** that survives a
  per-round rebuild (``set_calibration``) and that stays inside its clamps.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.algorithm.signal_processing import (
    zernike_coefficient_optimizer as module_under_test,
)
from ao_shaping.algorithm.signal_processing.zernike_coefficient_optimizer import (
    ZernikeCoefficientOptimizer,
)
from ao_shaping.utils.wavefront.calibration_terms import (
    FAMILY_NAMES,
    PupilAffine,
    TravelScale,
)

torch = pytest.importorskip("torch")


def _optimizer(**kwargs) -> ZernikeCoefficientOptimizer:
    """建一个小的确定性优化器 (CPU, float64)。"""
    options = {
        "n_orders": 4,
        "region": 32,
        "dtype": "float64",
        "device": "cpu",
        "lr": 0.05,
        "max_iterations": 50,
    }
    options.update(kwargs)
    return ZernikeCoefficientOptimizer(**options)


def _sample(seed: int = 0, side: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """返回 (真系数, 探针相位)。"""
    rng = np.random.default_rng(seed)
    coefficients = np.zeros(15, dtype=np.float64)
    coefficients[3] = 0.35
    phase = rng.normal(0.0, 1.5, (side, side))
    return coefficients, phase


class TestDefaultOff:
    """The default path must not touch the new code at all."""

    def test_the_new_entry_points_are_not_called(self, monkeypatch) -> None:
        """With everything off, the affine/travel helpers are never invoked."""
        def _boom(*args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("the self-calibration path was entered")

        monkeypatch.setattr(module_under_test, "apply_pupil_affine", _boom)
        monkeypatch.setattr(module_under_test, "apply_travel_scale", _boom)
        optimizer = _optimizer()
        coefficients, phase = _sample()
        image = optimizer.forward_intensity(coefficients, phase)
        assert np.all(np.isfinite(image))
        assert optimizer.affine is None
        assert optimizer.travel is None

    def test_calibration_state_is_empty_by_default(self) -> None:
        """The recorded state must say "no self-calibration"."""
        state = _optimizer().calibration_state()
        assert state["affine"] == {}
        assert state["travel"] == {}
        assert state["fourier_weight"] == 0.0
        assert state["loss_kind"] == "mse"

    def test_a_zero_weight_fourier_term_leaves_the_loss_alone(self) -> None:
        """``fourier_weight=0`` is the same computation as the default."""
        coefficients, phase = _sample()
        reference = _optimizer()
        same = _optimizer(fourier_weight=0.0)
        image = reference.forward_intensity(coefficients, phase)
        target = image / image.max()
        for _ in range(5):
            reference.update(target, phase)
            same.update(target, phase)
        assert np.array_equal(reference.coefficients, same.coefficients)
        assert reference.loss_history == same.loss_history


class TestFourierTerm:
    """The Fourier-domain L1 residual."""

    def test_the_weight_changes_the_trajectory(self) -> None:
        """A non-zero α must actually enter the optimisation."""
        coefficients, phase = _sample()
        image = _optimizer().forward_intensity(coefficients, phase)
        target = image / image.max()
        plain = _optimizer()
        fourier = _optimizer(fourier_weight=7e-4)
        for _ in range(10):
            plain.update(target, phase)
            fourier.update(target, phase)
        assert not np.array_equal(plain.coefficients, fourier.coefficients)

    def test_the_term_is_the_l1_of_the_real_fft(self) -> None:
        """The residual is ``mean(|Re[fft2(model)] - Re[fft2(target)]|)``."""
        optimizer = _optimizer(fourier_weight=7e-4)
        coefficients, phase = _sample()
        image = optimizer.forward_intensity(coefficients, phase)
        target = torch.tensor(image / image.max(), dtype=torch.float64)
        anchor = torch.tensor(
            float(image.max()) + 1e-8, dtype=torch.float64
        )
        intensity = torch.tensor(image, dtype=torch.float64)
        value = optimizer._fourier_domain_loss(intensity, target, anchor)
        expected = torch.mean(
            torch.abs(
                torch.fft.fft2(intensity / anchor).real
                - torch.fft.fft2(target).real
            )
        )
        assert float(value) == pytest.approx(float(expected), rel=1e-12)

    def test_the_term_is_differentiable(self) -> None:
        """The Fourier residual must reach the coefficients."""
        optimizer = _optimizer(fourier_weight=7e-4)
        coefficients, phase = _sample()
        image = optimizer.forward_intensity(coefficients, phase)
        target = torch.tensor(image / image.max(), dtype=torch.float64)
        anchor = torch.tensor(float(image.max()) + 1e-8, dtype=torch.float64)
        intensity = torch.tensor(image, dtype=torch.float64, requires_grad=True)
        optimizer._fourier_domain_loss(intensity, target, anchor).backward()
        assert intensity.grad is not None
        assert float(intensity.grad.abs().sum()) > 0.0


class TestLossKind:
    """The image-domain loss switch."""

    def test_l1_differs_from_mse(self) -> None:
        """The L1 image-domain loss is a different objective."""
        coefficients, phase = _sample()
        image = _optimizer().forward_intensity(coefficients, phase)
        target = image / image.max()
        mse = _optimizer()
        l1 = _optimizer(loss_kind="l1")
        for _ in range(10):
            mse.update(target, phase)
            l1.update(target, phase)
        assert not np.array_equal(mse.coefficients, l1.coefficients)

    def test_an_unknown_kind_is_rejected(self) -> None:
        """Only the two documented kinds are accepted."""
        with pytest.raises(ValueError):
            _optimizer(loss_kind="huber")


class TestAffine:
    """The pupil-plane affine τ_θ."""

    def test_it_is_fitted_and_reported(self) -> None:
        """With the affine on, the parameters move and are readable."""
        optimizer = _optimizer(affine_params=PupilAffine())
        coefficients, phase = _sample()
        image = optimizer.forward_intensity(coefficients, phase)
        target = image / image.max()
        for _ in range(10):
            optimizer.update(target, phase)
        affine = optimizer.affine
        assert affine is not None
        assert not affine.is_identity
        assert set(affine.to_dict()) == {
            "amp_x", "amp_y", "off_x", "off_y", "rot", "grid"
        }

    def test_it_survives_a_rebuild_through_set_calibration(self) -> None:
        """The affine is bench state, so it must be carried explicitly."""
        optimizer = _optimizer(affine_params=PupilAffine(off_x=0.2))
        coefficients, phase = _sample()
        image = optimizer.forward_intensity(coefficients, phase)
        target = image / image.max()
        for _ in range(5):
            optimizer.update(target, phase)
        moved = optimizer.affine
        assert moved is not None
        fresh = _optimizer(affine_params=PupilAffine())
        fresh.set_calibration(affine=moved)
        assert fresh.affine == moved

    def test_the_parameters_stay_inside_the_clamps(self) -> None:
        """A huge learning rate must not push the affine out of range."""
        optimizer = _optimizer(
            affine_params=PupilAffine(), affine_lr=50.0, lr=0.05
        )
        coefficients, phase = _sample()
        image = optimizer.forward_intensity(coefficients, phase)
        target = image / image.max()
        for _ in range(30):
            optimizer.update(target, phase)
        affine = optimizer.affine
        assert affine is not None
        assert PupilAffine.AMP_MIN <= affine.amp_x <= PupilAffine.AMP_MAX
        assert PupilAffine.AMP_MIN <= affine.amp_y <= PupilAffine.AMP_MAX
        assert -PupilAffine.OFF_LIMIT <= affine.off_x <= PupilAffine.OFF_LIMIT
        assert -PupilAffine.OFF_LIMIT <= affine.off_y <= PupilAffine.OFF_LIMIT
        assert -PupilAffine.ROT_LIMIT <= affine.rot <= PupilAffine.ROT_LIMIT
        assert PupilAffine.GRID_MIN <= affine.grid <= PupilAffine.GRID_MAX


class TestTravel:
    """The per-mode-family travel scale γ."""

    def test_it_is_fitted_and_reported(self) -> None:
        """With the travel scale on, the seven γ values move."""
        optimizer = _optimizer(travel_gammas=(0.0,) * len(FAMILY_NAMES))
        coefficients, phase = _sample()
        image = optimizer.forward_intensity(coefficients, phase)
        target = image / image.max()
        for _ in range(10):
            optimizer.update(target, phase)
        travel = optimizer.travel
        assert travel is not None
        assert not travel.is_identity

    def test_it_survives_a_rebuild_through_set_calibration(self) -> None:
        """Same contract as the affine: bench state, carried explicitly."""
        optimizer = _optimizer(travel_gammas=(0.0,) * len(FAMILY_NAMES))
        coefficients, phase = _sample()
        image = optimizer.forward_intensity(coefficients, phase)
        target = image / image.max()
        for _ in range(5):
            optimizer.update(target, phase)
        moved = optimizer.travel
        assert moved is not None
        fresh = _optimizer(travel_gammas=(0.0,) * len(FAMILY_NAMES))
        fresh.set_calibration(travel=moved)
        assert fresh.travel == moved

    def test_the_scale_factor_never_goes_negative(self) -> None:
        """The paper's clamp: γ >= -1, so the scale factor stays >= 0."""
        optimizer = _optimizer(
            travel_gammas=(0.0,) * len(FAMILY_NAMES), travel_lr=50.0
        )
        coefficients, phase = _sample()
        image = optimizer.forward_intensity(coefficients, phase)
        target = image / image.max()
        for _ in range(30):
            optimizer.update(target, phase)
        travel = optimizer.travel
        assert travel is not None
        assert all(value >= 0.0 for value in travel.scales)

    def test_a_wrong_family_count_is_rejected(self) -> None:
        """γ must have one value per family."""
        with pytest.raises(ValueError):
            _optimizer(travel_gammas=(0.0, 0.0))


class TestValidation:
    """The constructor rejects nonsense."""

    def test_a_negative_fourier_weight_is_rejected(self) -> None:
        """α must be non-negative."""
        with pytest.raises(ValueError):
            _optimizer(fourier_weight=-1e-3)

    def test_a_non_positive_learning_rate_is_rejected(self) -> None:
        """The self-calibration learning rate must be positive."""
        with pytest.raises(ValueError):
            _optimizer(affine_lr=0.0)
        with pytest.raises(ValueError):
            _optimizer(travel_lr=-1.0)
