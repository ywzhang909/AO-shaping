"""Acceptance tests for the zero-initialised residual corrector (Module 7).

Everything here is synthetic and hardware-free: the inner optimizer is the
pure-simulation :class:`ZernikeCoefficientOptimizer`, and every test runs on
CPU float64 forwards.

THE acceptance regression: with a freshly constructed ``ResidualCorrector``,
``CorrectedStepAOptimizer(inner).forward_intensity(...)`` must be BIT-IDENTICAL
(``np.array_equal``, not ``allclose``) to ``inner.forward_intensity(...)``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ao_shaping.algorithm.signal_processing.zernike_coefficient_optimizer import (
    ZernikeCoefficientOptimizer,
)
from ao_shaping.optimizer.wfless.residual_corrector import (
    CorrectedStepAOptimizer,
    ResidualCorrector,
)

REGION = 64
N_ORDERS = 10
SEED = 20261020


def _make_inner(far_field_size: int = REGION) -> ZernikeCoefficientOptimizer:
    """Build the float64 inner optimizer with a Gaussian source amplitude set.

    Args:
        far_field_size: Zero-pad side handed to the inner optimizer.

    Returns:
        A ready-to-use :class:`ZernikeCoefficientOptimizer` (tilt/piston
        frozen, float64 working precision).
    """
    inner = ZernikeCoefficientOptimizer(
        n_orders=N_ORDERS,
        region=REGION,
        far_field_size=far_field_size,
        frozen_modes=(1, 2, 3),
        dtype="float64",
    )
    coords = np.arange(REGION, dtype=np.float64) - (REGION - 1) / 2.0
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    radius = np.sqrt(xx**2 + yy**2)
    aperture = np.exp(-0.5 * (radius / (REGION / 4.0)) ** 2)
    inner.set_source_amplitude(aperture)
    return inner


def _random_phase(rng: np.random.Generator) -> np.ndarray:
    """Return a random raw (unwrapped) probe phase in radians."""
    return rng.uniform(-4.0 * np.pi, 4.0 * np.pi, size=(REGION, REGION))


def _random_coefficients(rng: np.random.Generator, n_coeffs: int) -> np.ndarray:
    """Return a random Zernike coefficient vector in radians."""
    return rng.normal(0.0, 0.3, size=n_coeffs)


def _manual_residual(
    corrector: ResidualCorrector, phase: np.ndarray, i_phys: np.ndarray
) -> np.ndarray:
    """Recompute the wrapper's residual path by hand for an equality check.

    Args:
        corrector: The corrector module to evaluate.
        phase: 2D phase array, radians.
        i_phys: 2D uncorrected far-field intensity.

    Returns:
        The float32 ``(H, W)`` residual the wrapper would add.
    """
    phase_t = torch.from_numpy(
        np.ascontiguousarray(phase, dtype=np.float32)
    ).view(1, 1, *phase.shape)
    i_phys_t = torch.from_numpy(
        np.ascontiguousarray(i_phys, dtype=np.float32)
    ).view(1, 1, *i_phys.shape)
    with torch.no_grad():
        return corrector(phase_t, i_phys_t)[0, 0].cpu().numpy()


def test_zero_init_output_is_exactly_zero() -> None:
    """A fresh corrector returns exactly ``torch.zeros_like(i_phys)``."""
    corrector = ResidualCorrector()
    rng = np.random.default_rng(SEED)
    phase_t = torch.from_numpy(
        np.ascontiguousarray(_random_phase(rng), dtype=np.float32)
    ).view(1, 1, REGION, REGION)
    i_phys_t = torch.rand(1, 1, REGION, REGION, dtype=torch.float32)

    out = corrector(phase_t, i_phys_t)

    assert out.dtype == torch.float32
    assert out.shape == i_phys_t.shape
    assert torch.equal(out, torch.zeros_like(i_phys_t))


def test_parameter_count_below_5000() -> None:
    """The default corrector stays low-capacity (< 5000 parameters)."""
    corrector = ResidualCorrector()
    count = sum(p.numel() for p in corrector.parameters())
    assert count < 5000
    assert corrector.count_parameters() == count


def test_fresh_corrector_forward_is_bit_identical() -> None:
    """ACCEPTANCE REGRESSION: corrected forward == uncorrected, bit-for-bit."""
    inner = _make_inner()
    rng = np.random.default_rng(SEED)
    n_coeffs = int(inner.coefficients.size)
    phases = [_random_phase(rng), _random_phase(rng)]
    coefficient_vectors = [
        _random_coefficients(rng, n_coeffs),
        _random_coefficients(rng, n_coeffs),
    ]

    for phase in phases:
        for coefficients in coefficient_vectors:
            corrected = CorrectedStepAOptimizer(inner).forward_intensity(
                coefficients, phase
            )
            uncorrected = inner.forward_intensity(coefficients, phase)
            assert np.array_equal(corrected, uncorrected)


def test_protocol_attributes_and_frozen_corrector() -> None:
    """The wrapper satisfies the _StepAOptimizer protocol; update freezes R."""
    inner = _make_inner()
    corrector = ResidualCorrector()
    wrapper = CorrectedStepAOptimizer(inner, corrector)
    rng = np.random.default_rng(SEED + 1)

    assert isinstance(wrapper.coefficients, np.ndarray)
    assert np.array_equal(wrapper.coefficients, inner.coefficients)
    assert wrapper.last_loss is None
    assert wrapper.converged == inner.converged

    parameters_before = [p.detach().clone() for p in corrector.parameters()]

    phase = _random_phase(rng)
    uncorrected = inner.forward_intensity(inner.coefficients, phase)
    result = wrapper.update(uncorrected, phase)

    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, inner.coefficients)
    assert wrapper.last_loss == inner.last_loss
    assert wrapper.last_loss is not None and np.isfinite(wrapper.last_loss)
    for before, after in zip(parameters_before, corrector.parameters()):
        assert torch.equal(before, after.detach())


def test_shape_dtype_finiteness_and_residual_equality() -> None:
    """Corrected output: same shape, finite, exactly ``inner + residual``."""
    inner = _make_inner()
    corrector = ResidualCorrector()
    with torch.no_grad():
        corrector.head.weight.fill_(0.25)
        corrector.head.bias.fill_(-0.1)
    wrapper = CorrectedStepAOptimizer(inner, corrector)
    rng = np.random.default_rng(SEED + 2)
    phase = _random_phase(rng)
    coefficients = _random_coefficients(rng, int(inner.coefficients.size))

    uncorrected = inner.forward_intensity(coefficients, phase)
    corrected = wrapper.forward_intensity(coefficients, phase)

    assert corrected.shape == uncorrected.shape
    assert corrected.dtype == uncorrected.dtype
    assert np.all(np.isfinite(corrected))
    residual = _manual_residual(corrector, phase, uncorrected)
    assert np.array_equal(corrected, uncorrected + residual.astype(np.float64))


def test_perturbed_corrector_wiring_is_live() -> None:
    """After perturbing the last conv, corrected differs from uncorrected."""
    inner = _make_inner()
    corrector = ResidualCorrector()
    rng_t = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        corrector.head.weight += torch.randn(
            corrector.head.weight.shape, generator=rng_t
        )
    wrapper = CorrectedStepAOptimizer(inner, corrector)
    rng = np.random.default_rng(SEED + 3)
    phase = _random_phase(rng)
    coefficients = _random_coefficients(rng, int(inner.coefficients.size))

    uncorrected = inner.forward_intensity(coefficients, phase)
    corrected = wrapper.forward_intensity(coefficients, phase)

    assert not np.array_equal(corrected, uncorrected)


def test_bit_identity_on_zero_padded_far_field_grid() -> None:
    """Bit-identity also holds when the far-field grid exceeds the phase grid."""
    inner = _make_inner(far_field_size=2 * REGION)
    rng = np.random.default_rng(SEED + 4)
    phase = _random_phase(rng)
    coefficients = _random_coefficients(rng, int(inner.coefficients.size))

    corrected = CorrectedStepAOptimizer(inner).forward_intensity(coefficients, phase)
    uncorrected = inner.forward_intensity(coefficients, phase)

    assert corrected.shape == (2 * REGION, 2 * REGION)
    assert np.array_equal(corrected, uncorrected)


def test_fit_corrector_moves_parameters_and_changes_forward() -> None:
    """The optional once-per-session fit perturbs the (previously frozen) R."""
    inner = _make_inner()
    corrector = ResidualCorrector()
    wrapper = CorrectedStepAOptimizer(inner, corrector)
    rng = np.random.default_rng(SEED + 5)
    phase = _random_phase(rng)
    parameters_before = [p.detach().clone() for p in corrector.parameters()]

    measured = inner.forward_intensity(inner.coefficients, phase) * 1.3 + 0.01
    wrapper.fit_corrector([(measured, phase)], epochs=2, lr=1e-2)

    assert not all(
        torch.equal(before, after.detach())
        for before, after in zip(parameters_before, corrector.parameters())
    )
    uncorrected = inner.forward_intensity(inner.coefficients, phase)
    corrected = wrapper.forward_intensity(inner.coefficients, phase)
    assert not np.array_equal(corrected, uncorrected)


def test_corrector_rejects_bad_inputs() -> None:
    """Shape/dtype guards raise ValueError on malformed batch tensors."""
    corrector = ResidualCorrector()
    bad_2d = torch.zeros(4, 4, dtype=torch.float32)
    with pytest.raises(ValueError):
        corrector(bad_2d, bad_2d)
    mismatched = torch.zeros(1, 1, 8, 8, dtype=torch.float32)
    with pytest.raises(ValueError):
        corrector(torch.zeros(1, 1, 4, 4, dtype=torch.float32), mismatched)
    with pytest.raises(ValueError):
        ResidualCorrector(hidden=0)
