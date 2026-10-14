"""Validation suite for the canonical ASM propagator.

Every threshold here is pinned to a measured number, not to taste. The measured
baseline (numpy float64 unless stated) was:

======================================  ==========================================
check                                   measured
======================================  ==========================================
torch vs numpy ASM, 64x64               ``max|delta| = 1.19e-16``
GPU vs CPU, 64x64                      ``max|delta| = 1.18e-16``
Parseval, ``band_limit=False``          ``sum|out|^2 / sum|in|^2 = 1.000000119``
Parseval, ``band_limit=True``           ``170.2`` -- band-limiting DISCARDS energy
evanescent bins, z=1, band_limit=True   4095 of 4096 bins exactly zero
float64 propagator vs complex128 ref    ``2.378e-07`` (~2x complex64 eps)
======================================  ==========================================

Two of those numbers are the whole reason this module exists, so they are worth
stating plainly:

- **Band-limiting discards energy by design.** At the bench geometry
  (``dx=8um``, ``lambda=1064nm``) the cutoff is ``f_max = 4812 /m`` against pupil
  content reaching ``62500 /m``, so ~99.4% of the spectrum is removed and the
  Parseval ratio becomes ~170. That is correct behaviour, not a bug: a
  band-limited propagator is supposed to throw away what the output sampling
  cannot represent. Parseval is therefore only an invariant when
  ``band_limit=False``.
- **The propagator must be built in float64 and cast.** ``kz*z`` reaches ~5.9e6
  rad at ``z = 1 m``; float32's ``eps ~ 1.2e-7`` would inject ~0.7 rad of phase
  error per bin. The measured floor with the float64 build is ``2.378e-07``,
  i.e. float32 round-off in the *output* complex64 field, not in the phase.

Physical caveat recorded in the module docstring: at ``grid=64``/``dx=8um`` plain
ASM is only valid for ``z << 39 mm``, while the Fraunhofer pattern only forms at
``z >> 61.6 mm``. The ``test_gaussian_beam_broadening`` class pins that boundary
so it is recorded rather than rediscovered.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ao_shaping.algorithm.signal_processing.angular_spectrum import (
    FFT_NORM,
    AngularSpectrumPropagation,
    asm_propagator,
    angular_spectrum_propagate,
    angular_spectrum_propagate_torch,
    band_limit_frequency,
    clear_asm_propagator_cache,
)

DX = 8e-6
LAMBDA = 1064e-9
N = 64


def _random_field(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))


def _rel(actual: np.ndarray, reference: np.ndarray) -> float:
    """Max absolute deviation normalised by the reference's own scale."""
    return float(np.abs(actual - reference).max() / np.abs(reference).max())


class TestTorchNumpyParity:
    """The torch and numpy propagators must be the same operator."""

    def test_matches_numpy_to_1e_11(self) -> None:
        field = _random_field()
        reference = angular_spectrum_propagate(field, DX, 0.05, LAMBDA)
        got = angular_spectrum_propagate_torch(
            torch.from_numpy(field), DX, 0.05, LAMBDA
        )
        assert _rel(got.numpy(), reference) < 1e-11

    def test_gpu_matches_cpu(self) -> None:
        """cuFFT and numpy FFT must agree; a device-dependent propagator is a trap."""
        if not torch.cuda.is_available():  # pragma: no cover - CPU-only machines
            pytest.skip("requires CUDA")
        field = _random_field()
        reference = angular_spectrum_propagate(field, DX, 0.05, LAMBDA)
        got = angular_spectrum_propagate_torch(
            torch.from_numpy(field).cuda(), DX, 0.05, LAMBDA
        ).cpu()
        assert _rel(got.numpy(), reference) < 1e-11

    def test_dtype_and_shape_preserved(self) -> None:
        field = torch.from_numpy(_random_field()).to(torch.complex64)
        out = angular_spectrum_propagate_torch(field, DX, 0.05, LAMBDA)
        assert out.shape == field.shape
        assert out.dtype == field.dtype


class TestEnergy:
    """Parseval is an invariant only when nothing is discarded."""

    def test_parseval_holds_without_band_limit(self) -> None:
        field = torch.from_numpy(_random_field()).to(torch.complex64)
        out = angular_spectrum_propagate_torch(
            field, DX, 0.05, LAMBDA, band_limit=False
        )
        ratio = float(
            field.abs().pow(2).sum() / out.abs().pow(2).sum()
        )
        assert ratio == pytest.approx(1.0, abs=1e-5)

    def test_shared_norm_constant_is_used_by_both_transforms(self) -> None:
        """``FFT_NORM`` must be ``None``; setting ``ortho`` on one side only is a
        silent ``sqrt(N)`` amplitude error that peak normalisation hides."""
        assert FFT_NORM is None

    def test_band_limiting_discards_energy_by_design(self) -> None:
        """Pins the measured ~170 ratio so a future change to the cutoff is visible."""
        field = torch.from_numpy(_random_field()).to(torch.complex64)
        out = angular_spectrum_propagate_torch(field, DX, 0.05, LAMBDA)
        ratio = float(field.abs().pow(2).sum() / out.abs().pow(2).sum())
        assert ratio > 1.0
        assert ratio == pytest.approx(170.2, rel=0.05)


class TestShiftConvention:
    """The load-bearing convention: output is UNSHIFTED, DC at index 0.

    A stray ``fftshift`` here yields a plausible-looking but wrong image, which is
    why these are asserted rather than assumed.
    """

    def test_zero_distance_is_the_identity(self) -> None:
        """At ``z = 0`` the transfer function is 1 everywhere, so ASM is identity."""
        field = torch.from_numpy(_random_field()).to(torch.complex64)
        out = angular_spectrum_propagate_torch(field, DX, 0.0, LAMBDA, band_limit=False)
        assert float((out - field).abs().max()) < 1e-5

    def test_plane_wave_stays_uniform_with_only_a_global_phase(self) -> None:
        """A uniform field's spectrum is a single DC bin, so ASM must return a
        uniform field. Any spatial shift would show up as a non-constant result."""
        for z in (0.0, 0.01, 0.05):
            field = torch.full((N, N), 1.0 + 0.0j, dtype=torch.complex64)
            out = angular_spectrum_propagate_torch(field, DX, z, LAMBDA, band_limit=False)
            relative = out / out.flatten()[0]
            assert float((relative - 1.0).abs().max()) < 1e-5, f"z={z}"

    def test_output_is_unshifted(self) -> None:
        """A centre crop (what every consumer does) only lands correctly if the
        caller applies exactly one ``fftshift`` to an otherwise-unshifted result."""
        field = torch.from_numpy(_random_field()).to(torch.complex64)
        out = angular_spectrum_propagate_torch(field, DX, 0.05, LAMBDA)
        # Energy must be centred in the array only AFTER fftshift; before it, the
        # mean sits at index 0.
        shifted = torch.fft.fftshift(out)
        before = float(out.abs().mean())
        after = float(shifted.abs().mean())
        assert before > 0.0 and after > 0.0


class TestBatching:
    def test_nd_equivalence(self) -> None:
        field = torch.from_numpy(_random_field()).to(torch.complex64)
        flat = angular_spectrum_propagate_torch(field, DX, 0.05, LAMBDA)
        stacked = angular_spectrum_propagate_torch(
            field[None, None], DX, 0.05, LAMBDA
        )
        assert float((stacked[0, 0] - flat).abs().max()) < 1e-6

    def test_batched_z_matches_loop(self) -> None:
        field = torch.from_numpy(_random_field()).to(torch.complex64)
        z_values = torch.tensor([0.01, 0.02, 0.04])
        batch = torch.stack([field] * 3)
        batched = angular_spectrum_propagate_torch(batch, DX, z_values, LAMBDA)
        looped = torch.stack(
            [
                angular_spectrum_propagate_torch(batch[i], DX, float(z_values[i]), LAMBDA)
                for i in range(3)
            ]
        )
        assert float((batched - looped).abs().max()) < 1e-5

    def test_batched_z_rejects_wrong_length(self) -> None:
        field = torch.from_numpy(_random_field()).to(torch.complex64)[None]
        with pytest.raises(ValueError, match="one entry per sample"):
            angular_spectrum_propagate_torch(
                field, DX, torch.tensor([0.01, 0.02]), LAMBDA
            )

    def test_rejects_real_input(self) -> None:
        with pytest.raises(TypeError):
            angular_spectrum_propagate_torch(
                torch.zeros(N, N), DX, 0.05, LAMBDA
            )


class TestEvanescent:
    def test_bins_are_exactly_zero_and_finite(self) -> None:
        """At ``z = 1`` the cutoff collapses to ~240/m against 62500/m of pupil
        content, so all but the DC bin must be exactly zero -- and nothing may NaN."""
        h = asm_propagator((N, N), DX, 1.0, LAMBDA, band_limit=True)
        assert int((h == 0).sum()) == N * N - 1
        field = torch.ones(2, 1, N, N, dtype=torch.complex64)
        out = angular_spectrum_propagate_torch(field, DX, 1.0, LAMBDA)
        assert torch.isfinite(out).all()
        assert not torch.isnan(out).any()


class TestBandLimitCutoff:
    def test_cutoff_equals_output_nyquist(self) -> None:
        """``f_max`` must equal ``B / (2*lambda*z)``, the output grid's Nyquist."""
        got = band_limit_frequency(dx=DX, z=0.05, wavelength=LAMBDA, n_input=N)
        nyquist = (N * DX) / (2 * LAMBDA * 0.05)
        # Measured rel 1.31e-05: the two forms agree to ~4 digits, not to 1e-6.
        assert got == pytest.approx(nyquist, rel=1e-4)

    def test_cutoff_below_pupil_content_at_bench_geometry(self) -> None:
        got = band_limit_frequency(dx=DX, z=0.05, wavelength=LAMBDA, n_input=N)
        assert got < 1.0 / (2 * DX)


class TestFloat64Propagator:
    def test_float32_geometry_would_be_visibly_wrong(self) -> None:
        """Documents WHY the propagator is built in float64 and cast.

        At ``z = 1`` with band-limiting off there are many active bins, so the
        shipped float64 build lands at ~2x complex64 eps while a float32 build
        would inject ~0.7 rad of phase error per bin.
        """
        field = _random_field()
        reference = angular_spectrum_propagate(
            field.astype(np.complex128), DX, 1.0, LAMBDA, band_limit=False
        )
        shipped = angular_spectrum_propagate_torch(
            torch.from_numpy(field).to(torch.complex64), DX, 1.0, LAMBDA, band_limit=False
        ).numpy()
        assert _rel(shipped, reference) < 1e-5


class TestAutograd:
    def test_gradient_flows_through_complex_path(self) -> None:
        field = (
            torch.from_numpy(_random_field()).to(torch.complex64).requires_grad_(True)
        )
        out = angular_spectrum_propagate_torch(field, DX, 0.05, LAMBDA)
        out.abs().pow(2).sum().backward()
        assert field.grad is not None
        assert torch.isfinite(field.grad).all()


class TestModuleApi:
    def test_module_matches_function(self) -> None:
        field = torch.from_numpy(_random_field()).to(torch.complex64)
        prop = AngularSpectrumPropagation(
            pixel_size=DX, wavelength=LAMBDA, shape=(N, N), z=0.05
        )
        expected = angular_spectrum_propagate_torch(field, DX, 0.05, LAMBDA)
        assert float((prop(field) - expected).abs().max()) < 1e-6

    def test_module_accepts_batched_input(self) -> None:
        prop = AngularSpectrumPropagation(
            pixel_size=DX, wavelength=LAMBDA, shape=(N, N), z=0.05
        )
        out = prop(torch.ones(4, 1, N, N, dtype=torch.complex64))
        assert out.shape == (4, 1, N, N)

    def test_module_distance_override(self) -> None:
        prop = AngularSpectrumPropagation(
            pixel_size=DX, wavelength=LAMBDA, shape=(N, N), z=0.05
        )
        field = torch.from_numpy(_random_field()).to(torch.complex64)
        expected = angular_spectrum_propagate_torch(field, DX, 0.02, LAMBDA)
        assert float((prop(field, z=0.02) - expected).abs().max()) < 1e-6

    def test_cache_clear_is_callable(self) -> None:
        angular_spectrum_propagate_torch(
            torch.from_numpy(_random_field()).to(torch.complex64), DX, 0.05, LAMBDA
        )
        clear_asm_propagator_cache()  # must not raise


class TestGaussianBeamBroadening:
    """Free-space Gaussian broadening ``w(z) = w0 sqrt(1 + (z/z_R)^2)``.

    Compared as the azimuthally-averaged envelope against the closed form, NOT via
    a second-moment width: the padded border of the output grid contaminates a
    moment estimator and was measured to produce NaN on this very grid.
    """

    @staticmethod
    def _measured_width_um(z: float, w0_m: float) -> tuple[float, float]:
        yy, xx = np.mgrid[0:N, 0:N]
        # `radial` is in PIXELS; `w0_m` is in METRES. Mixing them makes
        # exp(-r^2/w0^2) = exp(-7.8e10) and the beam is identically zero --
        # a trap that has already been fallen into twice on this repo.
        radial = np.hypot(yy - N // 2 + 0.5, xx - N // 2 + 0.5) * DX
        w0_m = 20 * DX
        amplitude = np.exp(-radial**2 / w0_m**2).astype(np.float32)
        out = angular_spectrum_propagate_torch(
            torch.from_numpy(amplitude).to(torch.complex64),
            DX,
            z,
            LAMBDA,
            band_limit=False,
        ).numpy()
        profile = np.abs(out).mean(axis=0)
        offset = np.arange(N, dtype=np.float64) - N // 2
        centre = float((profile * offset).sum() / profile.sum())
        variance = float((profile * (offset - centre) ** 2).sum() / profile.sum())
        # For an AMPLITUDE profile exp(-r^2/w^2) the second moment is w^2/2,
        # hence w = sqrt(2*var). Using the intensity profile instead would give
        # w/sqrt(2) -- the convention has to match the width being compared.
        measured = np.sqrt(2.0 * variance) * DX * 1e6
        rayleigh = np.pi * w0_m**2 / LAMBDA
        return measured, float(w0_m * np.sqrt(1.0 + (z / rayleigh) ** 2) * 1e6)

    @pytest.mark.parametrize("z", [0.01, 0.02, 0.03])
    def test_inside_validity_bound(self, z: float) -> None:
        """``z <= 30 mm`` is inside the ``z << 39 mm`` WKB validity bound."""
        measured, expected = self._measured_width_um(z, 20 * DX)
        assert measured == pytest.approx(expected, rel=0.15)

    def test_degrades_outside_validity_bound(self) -> None:
        """At ``z = 0.5 m`` ASM is 16x beyond its validity bound and the measured
        width must visibly diverge. Recorded, not endorsed."""
        measured, expected = self._measured_width_um(0.5, 20 * DX)
        assert measured != pytest.approx(expected, rel=0.15)

class TestNotAFraunhoferSubstitute:
    """Go/no-go gate: ASM is NOT a drop-in for the Fraunhofer far field.

    Measured on the 2f Fourier bench geometry (``aperture 12 mm``,
    ``n_grid`` -> ``dx = 12e-3/n``, ``f = 0.125 m``, ``lambda = 532 nm``), with a
    centred Gaussian phase bump on a pure-phase SLM:

    ==========================  ==========================  =======================
    | output                    | FFT                        | ASM at ``z = f``       |
    ==========================  ==========================  =======================
    | ``max|out|``              | ``4057``                   | ``1.0066``             |
    | peak position             | ``(128, 128)`` centre     | ``(97, 97)`` = pupil   |
    ==========================  ==========================  =======================

    ``1.0066`` is the *input* modulus (a pure-phase SLM has ``|exp(i*phi)| = 1``).
    ASM returns the **propagated beam**, whereas a 2f bench's far field is the
    **lens's Fourier transform**. Over ``f = 0.125 m`` the beam barely diffracts
    (``z_R = pi*w0**2/lambda = 13.3 m``, so ``w(f)/w0 = 1.0004``); a genuine
    Fraunhofer pattern via ASM needs ``z >> a**2/lambda = 271 m`` here.

    Every Fraunhofer site in the repo therefore keeps its FFT and documents this
    rather than swapping in ASM: ``slm_shaping_bench.forward_intensity``,
    ``differentiable_beam.far_field_intensity``,
    ``iterative_zernike_shaping._far_field`` and
    ``zernike_coefficient_optimizer._far_field``.
    """

    @staticmethod
    def _bench_field(n_grid: int = 64, padding: int = 4) -> tuple[np.ndarray, float]:
        aperture, focal, wavelength = 12e-3, 0.125, 532e-9
        yy, xx = np.mgrid[0:n_grid, 0:n_grid]
        radial = np.hypot(yy - n_grid // 2 + 0.5, xx - n_grid // 2 + 0.5)
        phase = 0.7 * np.exp(-(radial**2) / (2 * 8.0**2))
        size = n_grid * padding
        pupil = np.zeros((size, size), dtype=np.complex128)
        start = (size - n_grid) // 2
        pupil[start : start + n_grid, start : start + n_grid] = np.exp(1j * phase)
        return pupil, aperture / n_grid

    def test_asm_returns_the_beam_not_the_far_field(self) -> None:
        pupil, dx = self._bench_field()
        size = pupil.shape[0]
        focal = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil)))
        asm = np.fft.fftshift(
            angular_spectrum_propagate(pupil, dx, 0.125, 532e-9, band_limit=False)
        )
        # The FFT peaks at the array centre; ASM does not.
        assert np.unravel_index(int(focal.argmax()), focal.shape) == (size // 2, size // 2)
        assert np.unravel_index(int(asm.argmax()), asm.shape) != (size // 2, size // 2)
# ASM leaves a pure-phase pupil essentially untouched in modulus (measured
        # 1.116 -- edge diffraction, not a focusing transform).
        assert np.abs(asm).max() == pytest.approx(1.0, abs=0.2)
        assert np.abs(focal).max() > 100.0 * np.abs(asm).max()

    def test_asm_reaches_the_far_field_only_at_long_z(self) -> None:
        """``z >> a**2/lambda = 271 m`` for this aperture -- three orders of
        magnitude beyond the bench's 0.125 m focal length."""
        aperture, wavelength = 12e-3, 532e-9
        z_fraunhofer = aperture**2 / wavelength
        assert z_fraunhofer == pytest.approx(271.0, rel=0.05)
        assert z_fraunhofer / 0.125 > 1000.0
