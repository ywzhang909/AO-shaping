"""Simulation-first tests for :mod:`zernike_coefficient_optimizer`.

All tests are pure torch/numpy on synthetic far fields: no hardware, no SDK
imports, CPU only, deterministic seeds.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from ao_shaping.algorithm.signal_processing.zernike_coefficient_optimizer import (
    PEAK_EPS,
    ZernikeCoefficientOptimizer,
    ZernikeCoefficientResult,
)
from ao_shaping.utils.wavefront.zernike_calc import (
    ZernikeGenerator,
    calc_n_zernike_terms,
)

# Noll 1976 indices used across the tests.
NOLL_DEFO = 4
NOLL_ASTIG = 5
NOLL_SPHERICAL = 11

#: Fast-but-well-conditioned fit used by most tests.  ``region=32`` with
#: ``n_orders=10`` is rank-deficient on a 32 px aperture, so the synthetic
#: recovery uses ``n_orders=6``.
FIT_REGION = 32
FIT_ORDERS = 6
FIT_LR = 0.05

#: Configuration for the 10-order recovery test.  ``region=64`` gives the
#: 66-coefficient basis enough pixels per mode to be well conditioned (the
#: un-normalised Jacobian has cond ~5.6 and sigma_min ~12.5), and ``lr=0.1``
#: lands in the global optimum.  Lower rates are *not* safe here: 0.05 and 0.08
#: both stall in a flat local basin at loss ~6.9e-6 with |dc| ~0.49, while
#: 0.02/0.03/0.1 reach loss ~9e-13 with |dc| < 1e-4.
RECOVERY_REGION = 64
RECOVERY_ORDERS = 10
RECOVERY_LR = 0.1


def gaussian_amplitude(region: int) -> np.ndarray:
    """The native twin Gaussian beam for ``region``."""
    return ZernikeCoefficientOptimizer.native_amplitude(region)


def uniform_phase(region: int, seed: int) -> np.ndarray:
    """Random SLM phase in ``[0, 2*pi)`` as raw unwrapped radians."""
    return np.random.default_rng(seed).uniform(0.0, 2 * np.pi, (region, region))


def coefficients_from_noll(n_coeffs: int, values: dict[int, float]) -> np.ndarray:
    """Build a Noll-ordered coefficient vector from ``{noll: radians}``."""
    coefficients = np.zeros(n_coeffs)
    for noll, radians in values.items():
        coefficients[noll - 1] = radians
    return coefficients


def peak_normalized_mse(model: np.ndarray, measured: np.ndarray) -> float:
    """Peak-normalized MSE between two intensity maps."""
    model_norm = model / (model.max() + PEAK_EPS)
    measured_norm = measured / measured.max()
    return float(np.mean((model_norm - measured_norm) ** 2))


def generator_mode(region: int, n_orders: int, noll_index: int) -> np.ndarray:
    """Raw single-Noll-mode map from the canonical generator (NaN outside).

    The aperture is whatever ``ZernikeGenerator`` decides (it centres the grid
    on ``(region - 1) / 2``), so the mask is taken from the generator rather
    than re-derived here.
    """
    generator = ZernikeGenerator(
        (region, region), radius=region // 2, n_orders=n_orders
    )
    weights = np.zeros(calc_n_zernike_terms(n_orders))
    weights[noll_index - 1] = 1.0
    return np.asarray(generator.generate_noll(weights), dtype=np.float64)


def synthetic_case(
    values: dict[int, float],
    seed: int = 0,
    region: int = FIT_REGION,
    n_orders: int = FIT_ORDERS,
    fixed_phase: bool = False,
    amplitude: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(i_meas, phase_slm, c_true)`` with the optimizer's own model."""
    optimizer = ZernikeCoefficientOptimizer(
        n_orders=n_orders, region=region, max_iterations=1
    )
    phase = (
        np.zeros((region, region)) if fixed_phase else uniform_phase(region, seed)
    )
    if amplitude is None:
        amplitude = gaussian_amplitude(region)
    c_true = coefficients_from_noll(optimizer.n_coefficients, values)
    i_meas = optimizer.forward_intensity(c_true, phase, amplitude)
    return i_meas, phase, c_true


class TestBasis:
    """The cached Zernike basis must match the canonical Noll generator."""

    def test_basis_shape_and_aperture(self) -> None:
        region = 16
        optimizer = ZernikeCoefficientOptimizer(n_orders=10, region=region)
        basis = optimizer.generate_basis()

        assert basis.shape == (calc_n_zernike_terms(10), region, region)
        assert basis.shape == (66, region, region)
        assert np.all(np.isfinite(basis))
        # Outside-aperture NaNs from the generator map to exactly zero.
        outside = ~np.isfinite(generator_mode(region, 10, NOLL_DEFO))
        assert outside.any(), "generator produced no out-of-aperture pixels"
        assert np.all(basis[:, outside] == 0.0)
        # And the aperture interior is not trivially zero.
        assert np.count_nonzero(basis) > 0

    def test_mode_four_matches_generator(self) -> None:
        region = 16
        optimizer = ZernikeCoefficientOptimizer(n_orders=10, region=region)
        basis = optimizer.generate_basis()

        expected = np.nan_to_num(generator_mode(region, 10, NOLL_DEFO), nan=0.0)
        np.testing.assert_allclose(basis[NOLL_DEFO - 1], expected, atol=1e-12)

    def test_generate_basis_returns_a_copy(self) -> None:
        optimizer = ZernikeCoefficientOptimizer(n_orders=3, region=16)
        original = optimizer.generate_basis()
        basis = optimizer.generate_basis()
        basis[:] = 12345.0
        np.testing.assert_array_equal(optimizer.generate_basis(), original)


class TestForwardModel:
    """The forward model must reproduce the digital twin's FFT convention."""

    def test_forward_intensity_matches_numpy_fft(self) -> None:
        region = 16
        optimizer = ZernikeCoefficientOptimizer(n_orders=3, region=region)
        amplitude = gaussian_amplitude(region)
        phase = uniform_phase(region, 3)
        coefficients = np.linspace(-0.2, 0.2, optimizer.n_coefficients)

        model = optimizer.forward_intensity(coefficients, phase, amplitude)

        # Independent numpy evaluation of the same convention. The summed
        # patch is masked, matching the twin's ``nan_to_num(..., nan=0.0)``.
        aberration = np.tensordot(coefficients, optimizer.generate_basis(), axes=(0, 0))
        aperture = np.isfinite(generator_mode(region, 3, 1))
        field = amplitude * np.exp(1j * ((phase + aberration) * aperture))
        expected = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field), norm="ortho"))) ** 2
        np.testing.assert_allclose(model, expected, rtol=1e-5, atol=1e-8)

    def test_zero_coefficients_reduce_to_plain_fft(self) -> None:
        region = 16
        optimizer = ZernikeCoefficientOptimizer(n_orders=4, region=region)
        amplitude = gaussian_amplitude(region)
        phase = np.zeros((region, region))
        model = optimizer.forward_intensity(
            np.zeros(optimizer.n_coefficients), phase, amplitude
        )
        expected = np.abs(
            np.fft.fftshift(
                np.fft.fft2(np.fft.ifftshift(amplitude * np.exp(1j * phase)), norm="ortho")
            )
        ) ** 2
        np.testing.assert_allclose(model, expected, rtol=1e-5, atol=1e-8)

    def test_native_amplitude_matches_twin_waist(self) -> None:
        # region=TWIN_REGION must reproduce the twin's w0=250 Gaussian.
        amplitude = gaussian_amplitude(512)
        assert amplitude[256, 256] == pytest.approx(1.0)
        # exp(-250**2 / (2 * 250**2)) = exp(-0.5)
        assert amplitude[256, 256 + 250] == pytest.approx(np.exp(-0.5), rel=1e-9)

    def test_phase_input_is_not_mutated(self) -> None:
        """The raw-radians contract: the caller's phase array is left alone.

        ``forward_intensity`` is 2*pi-periodic, so wrapping the phase would be
        unobservable in the output. What *is* observable is in-place mutation
        of the caller's array, which the raw-radians contract forbids.
        """
        region = 16
        optimizer = ZernikeCoefficientOptimizer(n_orders=3, region=region)
        phase = uniform_phase(region, 5)
        snapshot = phase.copy()
        i_meas = gaussian_amplitude(region)
        for _ in range(2):
            optimizer.update(i_meas, phase)
        np.testing.assert_array_equal(phase, snapshot)


class TestRecovery:
    """The optimizer must recover a known aberration from a synthetic frame."""

    def test_recovers_synthetic_aberration(self) -> None:
        values = {NOLL_DEFO: 0.6, NOLL_ASTIG: -0.4, NOLL_SPHERICAL: 0.8}
        i_meas, phase, c_true = synthetic_case(
            values, seed=1, region=RECOVERY_REGION, n_orders=RECOVERY_ORDERS
        )
        optimizer = ZernikeCoefficientOptimizer(
            n_orders=RECOVERY_ORDERS,
            region=RECOVERY_REGION,
            lr=RECOVERY_LR,
            max_iterations=500,
        )
        result = optimizer.run(i_meas, phase)

        assert isinstance(result, ZernikeCoefficientResult)
        assert result.coefficients.shape == c_true.shape
        assert result.iterations <= 500
        assert len(result.history["loss"]) == result.iterations

        error = float(np.linalg.norm(result.coefficients - c_true))
        final_mse = peak_normalized_mse(
            optimizer.forward_intensity(result.coefficients, phase), i_meas
        )
        assert error < 0.05, f"coefficient error {error} >= 0.05"
        assert final_mse < 1e-3, f"final MSE {final_mse} >= 1e-3"

    def test_ten_order_recovery_is_not_lr_trapped(self) -> None:
        """The 10-order fit must reach the global optimum, not a flat basin.

        Regression guard for a real failure mode: with ``region=64`` and
        ``n_orders=10`` the peak-normalized objective has a shallow local basin.
        ``lr=0.05`` and ``lr=0.08`` both stall there at loss ~6.9e-6 with a
        coefficient error ~0.49 -- an image that *looks* converged while the
        coefficients are wrong.  The Jacobian is well conditioned there
        (cond ~5.6), so the basin is an optimization artifact, not an
        identifiability limit.  This test pins the good basin.
        """
        values = {NOLL_DEFO: 0.6, NOLL_ASTIG: -0.4, NOLL_SPHERICAL: 0.8}
        i_meas, phase, c_true = synthetic_case(
            values, seed=1, region=RECOVERY_REGION, n_orders=RECOVERY_ORDERS
        )
        optimizer = ZernikeCoefficientOptimizer(
            n_orders=RECOVERY_ORDERS,
            region=RECOVERY_REGION,
            lr=RECOVERY_LR,
            max_iterations=500,
        )
        result = optimizer.run(i_meas, phase)

        error = float(np.linalg.norm(result.coefficients - c_true))
        # The flat basin sits at loss ~6.9e-6; the global optimum is ~1e-12.
        # Require a margin well clear of the trap rather than the 1e-3 bound
        # used by the recovery test.
        assert result.history["loss"][-1] < 1e-9, (
            f"stuck in the flat local basin at loss {result.history['loss'][-1]:.3e}"
        )
        assert error < 0.01, f"coefficient error {error} >= 0.01"

    def test_loss_decreases(self) -> None:
        values = {NOLL_DEFO: 0.6, NOLL_ASTIG: -0.4, NOLL_SPHERICAL: 0.8}
        i_meas, phase, _ = synthetic_case(values, seed=2)
        optimizer = ZernikeCoefficientOptimizer(
            n_orders=FIT_ORDERS, region=FIT_REGION, lr=FIT_LR, max_iterations=500
        )
        history = optimizer.run(i_meas, phase).history["loss"]

        assert len(history) >= 2
        assert history[-1] < history[0]
        assert history[-1] < 0.25 * history[0], (
            f"final loss {history[-1]} not below 25% of initial {history[0]}"
        )

    def test_fixed_phase_defocus_only(self) -> None:
        """A constant (zero) SLM phase must still allow a defocus fit.

        With ``phase_slm = 0`` the far-field intensity is exactly symmetric in
        the defocus sign, so a zero start sits on a symmetric stationary point
        and cannot break the tie. A small seeded defocus start does, which is
        exactly what ``initial_coefficients`` is for.
        """
        region = FIT_REGION
        n_orders = 2
        n_coeffs = calc_n_zernike_terms(n_orders)
        phase = np.zeros((region, region))
        c_true = coefficients_from_noll(n_coeffs, {NOLL_DEFO: 0.7})

        probe = ZernikeCoefficientOptimizer(n_orders=n_orders, region=region)
        i_meas = probe.forward_intensity(c_true, phase)

        start = coefficients_from_noll(n_coeffs, {NOLL_DEFO: 0.1})
        optimizer = ZernikeCoefficientOptimizer(
            n_orders=n_orders,
            region=region,
            initial_coefficients=start,
            lr=0.1,
            max_iterations=500,
        )
        result = optimizer.run(i_meas, phase)

        assert abs(result.coefficients[NOLL_DEFO - 1] - 0.7) < 0.05
        assert result.history["loss"][-1] < 1e-6
        assert result.converged


class TestMeasurementAnchored:
    """The measurement-anchored gradient must be non-degenerate and robust."""

    def test_gradient_flows_into_coefficients(self) -> None:
        values = {NOLL_DEFO: 0.6, NOLL_ASTIG: -0.4, NOLL_SPHERICAL: 0.8}
        i_meas, phase, _ = synthetic_case(values, seed=3)
        optimizer = ZernikeCoefficientOptimizer(
            n_orders=FIT_ORDERS, region=FIT_REGION, lr=FIT_LR, max_iterations=500
        )
        optimizer.update(i_meas, phase)

        grad = optimizer.coefficient_tensor.grad
        assert grad is not None, "no gradient on the coefficient tensor"
        grad_norm = float(np.linalg.norm(grad.detach().cpu().numpy()))
        assert grad_norm > 0.0, "gradient is exactly zero (degenerate anchor)"
        assert np.all(np.isfinite(optimizer.coefficients))

    def test_converges_with_mild_additive_noise(self) -> None:
        values = {NOLL_DEFO: 0.6, NOLL_ASTIG: -0.4, NOLL_SPHERICAL: 0.8}
        i_meas, phase, c_true = synthetic_case(values, seed=4)
        rng = np.random.default_rng(11)
        noisy = i_meas + rng.normal(0.0, 0.01 * i_meas.max(), i_meas.shape)

        optimizer = ZernikeCoefficientOptimizer(
            n_orders=FIT_ORDERS, region=FIT_REGION, lr=FIT_LR, max_iterations=500
        )
        result = optimizer.run(noisy, phase)

        history = result.history["loss"]
        assert history[-1] < 0.25 * history[0]
        # The fit should stay close to the truth despite the noise.
        assert float(np.linalg.norm(result.coefficients - c_true)) < 0.1

    def test_resizes_measurement_to_model_grid(self) -> None:
        """A real 250x248 CCD frame must be resized, not rejected."""
        values = {NOLL_DEFO: 0.6, NOLL_ASTIG: -0.4, NOLL_SPHERICAL: 0.8}
        i_meas, phase, _ = synthetic_case(values, seed=6)
        # Upsample the model frame to the real CCD shape, then hand it back.
        ccd = np.asarray(
            ndimage.zoom(i_meas, (250 / i_meas.shape[0], 248 / i_meas.shape[1]), order=1)
        )
        assert ccd.shape == (250, 248)

        optimizer = ZernikeCoefficientOptimizer(
            n_orders=FIT_ORDERS, region=FIT_REGION, lr=FIT_LR, max_iterations=500
        )
        result = optimizer.run(ccd, phase)
        history = result.history["loss"]
        assert history[-1] < 0.25 * history[0]


class TestSourceAmplitude:
    """The pupil amplitude override must be honoured."""

    def test_uniform_amplitude_override(self) -> None:
        region = FIT_REGION
        amplitude = np.ones((region, region))
        values = {NOLL_DEFO: 0.6, NOLL_ASTIG: -0.4, NOLL_SPHERICAL: 0.8}
        i_meas, phase, c_true = synthetic_case(
            values, seed=7, amplitude=amplitude
        )
        optimizer = ZernikeCoefficientOptimizer(
            n_orders=FIT_ORDERS, region=region, lr=FIT_LR, max_iterations=500
        )
        # The flat-pupil override must be used, so the same aberration that was
        # baked into a flat-pupil measurement is recovered from it.
        result = optimizer.run(i_meas, phase, source_amplitude=amplitude)
        assert np.all(np.isfinite(result.coefficients))
        assert float(np.linalg.norm(result.coefficients - c_true)) < 0.05

    def test_run_amplitude_override_changes_the_fit(self) -> None:
        """A mismatched amplitude must actually be used, not ignored.

        A fresh optimizer is used for each run: ``run`` warm-starts from the
        current coefficients, so reusing one instance would make the second
        fit start from the first one's converged answer.
        """
        region = FIT_REGION
        i_meas, phase, c_true = synthetic_case(
            {NOLL_DEFO: 0.6, NOLL_ASTIG: -0.4, NOLL_SPHERICAL: 0.8}, seed=8
        )
        matched = ZernikeCoefficientOptimizer(
            n_orders=FIT_ORDERS, region=region, lr=FIT_LR, max_iterations=500
        ).run(i_meas, phase)
        mismatched = ZernikeCoefficientOptimizer(
            n_orders=FIT_ORDERS, region=region, lr=FIT_LR, max_iterations=500
        ).run(i_meas, phase, source_amplitude=np.ones((region, region)))

        assert np.all(np.isfinite(mismatched.coefficients))
        # The flat-pupil model cannot reproduce a Gaussian-pupil measurement,
        # so it lands on a visibly different (worse) coefficient set.
        assert float(np.linalg.norm(matched.coefficients - c_true)) < 0.05
        assert float(np.linalg.norm(mismatched.coefficients - c_true)) > 0.1
        # It still reduces its own loss, then plateaus out and says so.
        assert mismatched.history["loss"][-1] < mismatched.history["loss"][0]
        assert mismatched.converged

    def test_set_source_amplitude_rejects_bad_shape(self) -> None:
        optimizer = ZernikeCoefficientOptimizer(n_orders=3, region=16)
        with pytest.raises(ValueError):
            optimizer.set_source_amplitude(np.ones((8, 8)))


class TestValidation:
    """``__init__`` must reject bad input up front."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_orders": 0},
            {"n_orders": -1},
            {"region": 8},
            {"region": 17},
            {"radius": 0},
            {"lr": 0.0},
            {"lr": -0.1},
            {"max_iterations": 0},
            {"initial_coefficients": np.zeros(3)},
            {"initial_coefficients": np.zeros((4, 4))},
            {"initial_coefficients": np.full(6, np.nan)},
            {"seed": "abc"},
        ],
    )
    def test_invalid_construction(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            ZernikeCoefficientOptimizer(**kwargs)

    def test_update_rejects_bad_phase(self) -> None:
        optimizer = ZernikeCoefficientOptimizer(n_orders=3, region=16)
        i_meas = np.ones((16, 16))
        with pytest.raises(ValueError):
            optimizer.update(i_meas, np.ones((8, 8)))
        with pytest.raises(ValueError):
            optimizer.update(i_meas, np.full((16, 16), np.nan))
        with pytest.raises(ValueError):
            optimizer.update(np.ones(16), np.zeros((16, 16)))

    def test_update_rejects_zero_peak_measurement(self) -> None:
        optimizer = ZernikeCoefficientOptimizer(n_orders=3, region=16)
        with pytest.raises(ValueError):
            optimizer.update(np.zeros((16, 16)), np.zeros((16, 16)))

    def test_reset_restores_initial_coefficients(self) -> None:
        start = np.zeros(calc_n_zernike_terms(3))
        start[0] = 0.25
        optimizer = ZernikeCoefficientOptimizer(
            n_orders=3, region=16, initial_coefficients=start
        )
        optimizer._coefficients.data.add_(1.0)
        assert not np.allclose(optimizer.coefficients, start)
        optimizer.reset()
        np.testing.assert_allclose(optimizer.coefficients, start)
        assert optimizer.loss_history == []

    def test_seeded_initial_coefficients_are_reproducible(self) -> None:
        a = ZernikeCoefficientOptimizer(n_orders=4, region=16, seed=5)
        b = ZernikeCoefficientOptimizer(n_orders=4, region=16, seed=5)
        np.testing.assert_allclose(a.coefficients, b.coefficients)

    def test_default_device_is_cpu(self) -> None:
        optimizer = ZernikeCoefficientOptimizer(n_orders=2, region=16)
        assert optimizer.device == "cpu"
        assert str(optimizer.coefficient_tensor.device) == "cpu"


class TestDigitalTwinEquivalence:
    """The forward model must reproduce ``SimFourierGSNetEnv``'s far field.

    These are regression tests for two details that silently break the
    model-in-the-loop contract: the phase patch must be masked outside the
    Zernike aperture (the twin masks the *sum* of SLM phase and aberration),
    and the working precision must be able to reach the twin's float64 FFT.
    """

    REGION = 64
    ORDERS = 10
    NOLLS = {4: 0.6, 5: -0.4, 11: 0.8}

    @staticmethod
    def _twin_far_field(phase: np.ndarray) -> np.ndarray:
        """The twin's pre-envelope, pre-zoom far field.

        The twin's full ``render_intensity`` additionally applies an optical
        sinc envelope and a zoom to the CCD grid, so the comparable quantity is
        the FFT core it builds from the beam, SLM phase and aberrations.
        """
        from ao_shaping.drivers.sim.fouriergsnet_env import (
            BeamParams,
            SimFourierGSNetEnv,
        )

        region = TestDigitalTwinEquivalence.REGION
        env = SimFourierGSNetEnv(
            beam=BeamParams(region=region, w0=250.0 * (region / 512), native=True),
            K_px=region,
            noise_enabled=False,
        )
        env.aberrations = dict(TestDigitalTwinEquivalence.NOLLS)
        env.slm.display_phase(phase)
        patch = np.nan_to_num(
            env._extract_region(env.slm._phase) + env._aberration_phase(), nan=0.0
        )
        field = env._beam_amp * np.exp(1j * patch)
        return np.abs(
            np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field), norm="ortho"))
        ) ** 2

    def test_phase_outside_aperture_does_not_affect_far_field(self) -> None:
        """SLM phase outside the circular aperture must be discarded.

        The twin masks the summed patch, so perturbing the phase strictly
        outside the aperture cannot change its far field. A model that applied
        ``exp(1j * phi_slm)`` everywhere would change here.
        """
        region = self.REGION
        optimizer = ZernikeCoefficientOptimizer(
            n_orders=self.ORDERS, region=region, dtype="float64"
        )
        coefficients = coefficients_from_noll(
            optimizer.n_coefficients, self.NOLLS
        )
        inside = uniform_phase(region, seed=11)
        outside = uniform_phase(region, seed=12)

        aperture = np.isfinite(generator_mode(region, self.ORDERS, 1))
        assert not aperture.all(), "aperture must not cover the whole grid"
        perturbed = np.where(aperture, inside, outside)

        baseline = optimizer.forward_intensity(coefficients, inside)
        np.testing.assert_allclose(
            optimizer.forward_intensity(coefficients, perturbed),
            baseline,
            rtol=0,
            atol=0,
        )

    def test_matches_twin_far_field_in_float64(self) -> None:
        """float64 reaches the twin's precision; float32 is round-off close."""
        region = self.REGION
        phase = uniform_phase(region, seed=13)
        coefficients = coefficients_from_noll(
            calc_n_zernike_terms(self.ORDERS), self.NOLLS
        )
        twin = self._twin_far_field(phase)
        peak = float(twin.max())

        exact = ZernikeCoefficientOptimizer(
            n_orders=self.ORDERS, region=region, dtype="float64"
        )
        rel64 = float(
            np.abs(twin - exact.forward_intensity(coefficients, phase)).max() / peak
        )
        assert rel64 < 1e-12, f"float64 relative error {rel64:.3e} exceeds 1e-12"

        fast = ZernikeCoefficientOptimizer(
            n_orders=self.ORDERS, region=region, dtype="float32"
        )
        rel32 = float(
            np.abs(twin - fast.forward_intensity(coefficients, phase)).max() / peak
        )
        assert rel32 < 1e-5, f"float32 relative error {rel32:.3e} exceeds 1e-5"

    def test_rejects_unknown_dtype(self) -> None:
        with pytest.raises(ValueError, match="dtype"):
            ZernikeCoefficientOptimizer(n_orders=2, region=16, dtype="float16")
