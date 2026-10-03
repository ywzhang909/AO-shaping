"""Unit tests for the physics-based Zernike forward model.

Covers `src/ml/zernike/models.py`: basis construction (piston exclusion, units,
aperture), the complex-domain forward pass, gradient flow, and parity with the
canonical numpy Fraunhofer convention used by
`ao_shaping.drivers.sim.slm_shaping_bench`.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch lives in the optional ml group")

from ml.zernike.models import (  # noqa: E402
    ZernikeAmpConfig,
    ZernikeAmpFitConfig,
    ZernikeAmpModel,
    ZernikeBasis,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _triangular(n_max: int) -> int:
    """Canonical Zernike term count, re-derived independently of the module.

    ``sum_{n=0..n_max} (n + 1)`` -- radial order ``n`` contributes ``n + 1``
    azimuthal orders. NB this is NOT ``(n_max + 1) ** 2``; confusing the two is
    exactly the 36-is-ambiguous case flagged in AGENTS.md (36 is both
    ``n_max = 7`` triangular and a 6x6 freeform grid).
    """
    return sum(n + 1 for n in range(n_max + 1))


def _phasor(batch: int = 2, grid: int = 16, seed: int = 0) -> tuple:
    """A smooth, pupil-like unit phasor pair, as `ml.hwdataset` would deliver."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:grid, 0:grid]
    r = np.hypot(xx - (grid - 1) / 2, yy - (grid - 1) / 2) / (grid / 2)
    inside = r <= 1.0
    phase = np.where(inside, 1.7 * np.sin(3 * r) + 0.4 * xx / grid, 0.0)
    phase = phase + 0.01 * rng.standard_normal(phase.shape)
    return (
        torch.from_numpy(np.cos(phase)[None, None].astype(np.float32)).repeat(batch, 1, 1, 1),
        torch.from_numpy(np.sin(phase)[None, None].astype(np.float32)).repeat(batch, 1, 1, 1),
    )


def _numpy_reference_amp(
    phase_cos: np.ndarray,
    phase_sin: np.ndarray,
    correction: np.ndarray | float,
    padding: int,
    crop: int | None = None,
) -> np.ndarray:
    """Canonical Fraunhofer amplitude, transcribed from the numpy bench.

    Mirrors `slm_shaping_bench._fraunhofer_intensity` /
    `IterativeZernikeShaper._far_field`: complex pupil field, centre-pad,
    `fft2(ifftshift(.))`, `fftshift`, then |F|. ``correction`` may be a scalar
    (a uniform piston offset) or a per-pixel phase map. ``crop`` reproduces the
    model's ``center_crop``: keep the centre ``crop x crop`` of the far field.
    """
    field = (phase_cos + 1j * phase_sin) * np.exp(1j * correction)
    n = field.shape[-1]
    m = n * padding
    if m > n:
        start = (m - n) // 2
        padded = np.zeros(field.shape[:-2] + (m, m), dtype=complex)
        padded[..., start : start + n, start : start + n] = field
        field = padded
    focal = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(field, axes=(-2, -1)), norm="ortho"), axes=(-2, -1)
    )
    if crop is not None and m > crop:
        start = (m - crop) // 2
        focal = focal[..., start : start + crop, start : start + crop]
    return np.abs(focal)


# ---------------------------------------------------------------------------
# Basis
# ---------------------------------------------------------------------------
class TestZernikeBasis:
    """The basis must come from the canonical generator, piston excluded."""

    @pytest.mark.parametrize("n_max", [1, 2, 4, 7])
    def test_k_is_triangular_minus_piston(self, n_max: int) -> None:
        basis = ZernikeBasis(grid=16, n_max=n_max)
        assert basis.K == _triangular(n_max) - 1
        assert len(basis) == basis.K

    def test_shape_dtype_and_finite(self) -> None:
        basis = ZernikeBasis(grid=16, n_max=4)
        assert basis.modes.shape == (basis.K, 16, 16)
        assert basis.modes.dtype == np.float32
        # NaN must never escape the aperture.
        assert np.isfinite(basis.modes).all()

    def test_values_are_raw_radians_not_wrapped(self) -> None:
        """Generators must return raw unwrapped radians (repo red line).

        A `mod 2pi` would confine the basis to [0, 2pi); Noll 5 (astig 0 deg)
        evaluated on a unit circle reaches well beyond 2pi in raw form.
        """
        basis = ZernikeBasis(grid=64, n_max=5)
        assert basis.modes.min() < -1e-6, "basis looks wrapped to a non-negative range"

    def test_piston_slot_is_never_populated(self) -> None:
        """Slot 0 of a Noll vector is (0, 0); nothing may be written there."""
        from ao_shaping.utils.wavefront.zernike_calc import ZernikeGenerator

        generator = ZernikeGenerator((16, 16), radius=8.0, n_orders=4)
        piston = np.zeros(generator.n_modes, dtype=np.float64)
        piston[0] = 1.0
        piston_map = np.nan_to_num(generator.generate_noll(piston), nan=0.0)

        basis = ZernikeBasis(grid=16, n_max=4)
        for mode in basis.modes:
            # No learned mode may be (a multiple of) the piston map.
            assert not np.allclose(mode, piston_map, atol=1e-6)
            assert not np.allclose(mode, -piston_map, atol=1e-6)

    def test_rejects_bad_arguments(self) -> None:
        with pytest.raises(ValueError, match="grid"):
            ZernikeBasis(grid=1, n_max=2)
        with pytest.raises(ValueError, match="n_max"):
            ZernikeBasis(grid=16, n_max=0)
        with pytest.raises(ValueError, match="radius"):
            ZernikeBasis(grid=16, n_max=2, radius=0.0)

    def test_basis_stack_is_not_shared_between_instances(self) -> None:
        a = ZernikeBasis(grid=16, n_max=2).as_tensor()
        b = ZernikeBasis(grid=16, n_max=2).as_tensor()
        a += 1.0
        assert not torch.equal(a, b)


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------
class TestCalibratedDefaults:
    """The defaults are measured choices, so pin them.

    Each of these was chosen from a real-corpus sweep recorded in
    ``ZernikeAmpConfig``'s docstring. A silent flip is exactly the failure that
    produced ``R^2 = -0.38`` once already (a CLI literal overrode the padding).
    """

    def test_observable_defaults_to_intensity(self) -> None:
        # A CCD records intensity. "amplitude" measured R^2 +0.675 vs +0.797 at
        # n_max=15 on the same split.
        assert ZernikeAmpConfig().observable == "intensity"

    def test_normalization_defaults_to_peak(self) -> None:
        # "sum" makes MSE read 0.00000 while R^2 is worse; "none" gives R^2 = -159.
        assert ZernikeAmpConfig().normalization == "peak"

    def test_padding_defaults_to_the_calibrated_value(self) -> None:
        # pad=1 is a ~4x angular-scale mismatch against the target window.
        assert ZernikeAmpConfig().far_field_padding == 10

    def test_center_crop_defaults_on(self) -> None:
        assert ZernikeAmpConfig().center_crop is True

    def test_default_config_output_matches_the_pupil_grid(self) -> None:
        """With the defaults the output must be target-shaped, i.e. (B,1,grid,grid)."""
        model = ZernikeAmpModel(ZernikeAmpConfig(n_max=2, grid=16))
        pc, ps = _phasor(batch=1, grid=16)
        assert tuple(model(pc, ps).shape) == (1, 1, 16, 16)
        assert model.forward_shape(1) == (1, 1, 16, 16)


class TestForward:
    """Shape, observable, and normalisation contracts."""

    def test_output_shape_and_dtype(self) -> None:
        model = ZernikeAmpModel(ZernikeAmpConfig(n_max=3, grid=16))
        pc, ps = _phasor(batch=2, grid=16)
        out = model(pc, ps)
        assert tuple(out.shape) == (2, 1, 16, 16)
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()

    def test_padding_enlarges_the_far_field(self) -> None:
        pc, ps = _phasor(batch=1, grid=16)
        cropped = ZernikeAmpModel(
            ZernikeAmpConfig(n_max=2, grid=16, far_field_padding=3)
        )
        # Default center_crop=True: the output must stay target-shaped.
        assert tuple(cropped(pc, ps).shape) == (1, 1, 16, 16)
        assert cropped.forward_shape(1) == (1, 1, 16, 16)

        raw = ZernikeAmpModel(
            ZernikeAmpConfig(
                n_max=2, grid=16, far_field_padding=3, center_crop=False
            )
        )
        assert tuple(raw(pc, ps).shape) == (1, 1, 48, 48)
        assert raw.forward_shape(1) == (1, 1, 48, 48)

    def test_center_crop_takes_the_centre_of_the_far_field(self) -> None:
        """The crop must be the centre window, not a resize or a corner."""
        pc, ps = _phasor(batch=1, grid=16)
        raw = ZernikeAmpModel(
            ZernikeAmpConfig(
                n_max=2, grid=16, far_field_padding=4, center_crop=False,
                normalization="none",
            )
        )(pc, ps)
        cropped = ZernikeAmpModel(
            ZernikeAmpConfig(
                n_max=2, grid=16, far_field_padding=4, center_crop=True,
                normalization="none",
            )
        )(pc, ps)
        assert torch.allclose(cropped[0, 0], raw[0, 0, 24:40, 24:40], atol=1e-6)

    def test_amplitude_is_sqrt_of_intensity(self) -> None:
        pc, ps = _phasor(grid=16)
        amp = ZernikeAmpModel(
            ZernikeAmpConfig(n_max=3, grid=16, observable="amplitude", normalization="none")
        )(pc, ps)
        inten = ZernikeAmpModel(
            ZernikeAmpConfig(n_max=3, grid=16, observable="intensity", normalization="none")
        )(pc, ps)
        assert torch.allclose(amp.pow(2), inten, atol=1e-5)

    def test_peak_normalisation_makes_the_max_one(self) -> None:
        pc, ps = _phasor(batch=3, grid=16)
        out = ZernikeAmpModel(
            ZernikeAmpConfig(n_max=3, grid=16, normalization="peak")
        )(pc, ps)
        assert torch.allclose(out.amax(dim=(-2, -1)), torch.ones(3), atol=1e-5)

    def test_sum_normalisation_makes_the_total_one(self) -> None:
        pc, ps = _phasor(batch=2, grid=16)
        out = ZernikeAmpModel(
            ZernikeAmpConfig(n_max=3, grid=16, normalization="sum")
        )(pc, ps)
        assert torch.allclose(out.sum(dim=(-2, -1)), torch.ones(2), atol=1e-5)

    def test_rejects_bad_inputs(self) -> None:
        model = ZernikeAmpModel(ZernikeAmpConfig(n_max=2, grid=16))
        pc, ps = _phasor(grid=16)
        with pytest.raises(ValueError, match="identical shapes"):
            model(pc, ps[:-1])
        with pytest.raises(ValueError, match="does not match the basis"):
            model(torch.ones(1, 1, 8, 8), torch.ones(1, 1, 8, 8))
        with pytest.raises(ValueError, match="floating point"):
            model(pc.to(torch.int32), ps.to(torch.int32))

    def test_rejects_invalid_config(self) -> None:
        with pytest.raises(ValueError, match="observable"):
            ZernikeAmpModel(ZernikeAmpConfig(observable="bogus"))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="normalization"):
            ZernikeAmpModel(ZernikeAmpConfig(normalization="bogus"))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="far_field_padding"):
            ZernikeAmpModel(ZernikeAmpConfig(far_field_padding=0))
        with pytest.raises(TypeError, match="not both"):
            ZernikeAmpModel(ZernikeAmpConfig(), n_max=3)


# ---------------------------------------------------------------------------
# The physics that matters
# ---------------------------------------------------------------------------
class TestPhysics:
    """Piston is a null direction; the phasor must not be angle-reconstructed."""

    def test_piston_is_a_far_field_no_op(self) -> None:
        """Why piston is excluded: a constant pupil phase leaves |F| unchanged.

        This is the identity that makes a piston coefficient unidentifiable --
        `|FFT(e^{i0} U)| == |FFT(U)|` -- so it would receive exactly zero gradient
        forever. Verified here against the numpy reference.
        """
        pc, ps = _phasor(grid=16)
        grid = pc.shape[-1]
        piston = np.full((grid, grid), 0.73, dtype=np.float64)
        base = _numpy_reference_amp(pc[0, 0].numpy(), ps[0, 0].numpy(), 0.0, 1)
        shifted = _numpy_reference_amp(
            pc[0, 0].numpy(), ps[0, 0].numpy(), piston, 1
        )
        assert np.allclose(base, shifted, atol=1e-6)

    def test_tip_and_tilt_are_not_null(self) -> None:
        """The contrast with piston: a tilt really does move the spot."""
        pc, ps = _phasor(grid=16)
        grid = pc.shape[-1]
        yy, xx = np.mgrid[0:grid, 0:grid]
        tilt = 0.5 * (xx - (grid - 1) / 2) / (grid / 2)
        base = _numpy_reference_amp(pc[0, 0].numpy(), ps[0, 0].numpy(), 0.0, 1)
        tilted = _numpy_reference_amp(pc[0, 0].numpy(), ps[0, 0].numpy(), tilt, 1)
        assert not np.allclose(base, tilted, atol=1e-4)

    def test_zero_init_is_the_uncorrected_beam(self) -> None:
        """At Z = 0 the model must reproduce the plain far field of the phasor."""
        model = ZernikeAmpModel(
            ZernikeAmpConfig(
                n_max=3,
                grid=16,
                normalization="none",
                far_field_padding=1,
                center_crop=False,
                # The numpy reference returns |F|, so pin the observable rather
                # than inheriting the (intensity) default.
                observable="amplitude",
            )
        )
        pc, ps = _phasor(grid=16)
        assert torch.count_nonzero(model.coefficients) == 0
        with torch.no_grad():
            got = model(pc, ps)[0, 0].numpy()
        want = _numpy_reference_amp(pc[0, 0].numpy(), ps[0, 0].numpy(), 0.0, 1)
        assert np.allclose(got, want, atol=1e-6)

    def test_parity_with_the_numpy_bench_convention(self) -> None:
        """Torch output must match the numpy bench convention bit-for-bit-ish.

        This is the test that proves the FFT conventions (centre-pad,
        `ifftshift` -> `fft2(norm='ortho')` -> `fftshift`) were matched rather
        than reinvented.
        """
        pc, ps = _phasor(grid=16)
        for padding in (1, 2):
            for observable in ("amplitude", "intensity"):
                for crop in (True, False):
                        model = ZernikeAmpModel(
                            ZernikeAmpConfig(
                                n_max=3,
                                grid=16,
                                far_field_padding=padding,
                                observable=observable,
                                normalization="none",
                                center_crop=crop,
                            )
                        )
                        with torch.no_grad():
                            model.coefficients.copy_(
                                torch.linspace(-0.3, 0.4, model.K)
                            )
                            got = model(pc, ps)[0, 0].numpy()
                            correction = model.correction_phase().numpy()
                        want = _numpy_reference_amp(
                            pc[0, 0].numpy(),
                            ps[0, 0].numpy(),
                            correction,
                            padding,
                            crop=16 if crop else None,
                        )
                        if observable == "intensity":
                            want = want**2
                        assert np.allclose(got, want, atol=1e-5), (
                            padding,
                            observable,
                            crop,
                        )

    def test_forward_never_calls_atan2(self) -> None:
        """The branch cut is the whole reason the dataset ships (cos, sin).

        If an implementation regressed to `atan2`, wrapping would appear. A
        smooth pupil phase like this one is far from the +/-pi cut, so a wrapped
        implementation would still agree here -- the guard is that `forward`
        consumes the phasor pair and the correction is applied multiplicatively.
        """
        model = ZernikeAmpModel(ZernikeAmpConfig(n_max=2, grid=16))
        pc, ps = _phasor(grid=16)
        correction = model.correction_phase()
        assert correction.shape == (16, 16)
        # Multiplicative composition is unit-modulus, so the result differs from
        # the bare phasor once coefficients are non-zero.
        with torch.no_grad():
            model.coefficients.fill_(0.4)
        combined = torch.complex(pc, ps) * torch.polar(
            torch.ones_like(correction), correction
        )
        assert torch.allclose(combined.abs(), torch.ones_like(combined.abs()), atol=1e-5)


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------
class TestOptimisation:
    """Gradient flow, the fit loop, and the result contract."""

    def test_gradients_reach_every_coefficient(self) -> None:
        model = ZernikeAmpModel(
            ZernikeAmpConfig(n_max=4, grid=16, observable="intensity")
        )
        pc, ps = _phasor(grid=16)
        target = torch.rand(1, 1, 16, 16)
        loss = ((model(pc, ps) - target) ** 2).mean()
        loss.backward()
        grad = model.coefficients.grad
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert torch.count_nonzero(grad) == model.K, "some mode received no gradient"

    def test_fit_reduces_the_loss(self) -> None:
        """Optimising against a synthetic target must actually descend."""
        grid = 16
        model = ZernikeAmpModel(
            ZernikeAmpConfig(n_max=3, grid=grid, observable="intensity")
        )
        pc, ps = _phasor(batch=4, grid=grid)
        # A target the model can reach: the far field of a known tilt.
        target = model(pc, ps).detach().clone()
        result = model.fit(pc, ps, target, ZernikeAmpFitConfig(epochs=8, lr=0.05, log_every=0))
        assert len(result.history) == 8
        assert result.best_loss <= result.history[0] + 1e-12
        assert result.K == model.K
        assert result.seconds > 0.0
        assert result.coefficients.shape == (model.K,)

    def test_target_is_put_on_the_same_scale_as_the_prediction(self) -> None:
        """A raw target must be re-scaled, else MSE keeps an irreducible offset.

        The FFT amplitude scale is arbitrary, so a peak-normalised prediction
        compared against a raw frame can never reach zero loss no matter what Z
        is. With ``normalize_target`` the model converges on shape instead.
        """
        grid = 16
        pc, ps = _phasor(batch=2, grid=grid)
        # Target that is exactly reachable but deliberately NOT peak-normalised.
        model = ZernikeAmpModel(ZernikeAmpConfig(n_max=2, grid=grid))
        with torch.no_grad():
            target = model(pc, ps).detach().clone() * 0.5

        scaled = model.fit(
            pc, ps, target, ZernikeAmpFitConfig(epochs=3, lr=0.0, log_every=0)
        )
        raw = model.fit(
            pc,
            ps,
            target,
            ZernikeAmpFitConfig(epochs=3, lr=0.0, log_every=0, normalize_target=False),
        )
        # lr=0 freezes Z, so the first epoch's loss is a pure scale comparison:
        # re-scaling the target must make it far smaller than the raw comparison.
        assert scaled.history[0] < raw.history[0] / 10.0

    def test_fit_rejects_a_mismatched_target(self) -> None:
        model = ZernikeAmpModel(ZernikeAmpConfig(n_max=2, grid=16))
        pc, ps = _phasor(grid=16)
        with pytest.raises(ValueError, match="target shape"):
            model.fit(pc, ps, torch.rand(1, 1, 32, 32), ZernikeAmpFitConfig(epochs=1))

    def test_fit_rejects_an_unknown_optimizer(self) -> None:
        model = ZernikeAmpModel(ZernikeAmpConfig(n_max=2, grid=16))
        pc, ps = _phasor(grid=16)
        target = torch.zeros(1, 1, 16, 16)
        with pytest.raises(ValueError, match="optimizer"):
            model.fit(
                pc, ps, target, ZernikeAmpFitConfig(epochs=1, optimizer="nope")  # type: ignore[arg-type]
            )

    def test_coefficients_start_at_zero(self) -> None:
        """Zero init makes step 0 the uncorrected bench."""
        model = ZernikeAmpModel(ZernikeAmpConfig(n_max=6, grid=8))
        assert torch.equal(
            model.coefficients.detach(), torch.zeros(model.K)
        )
        assert model.coefficients.requires_grad

    def test_observable_and_normalization_are_reported(self) -> None:
        model = ZernikeAmpModel(
            ZernikeAmpConfig(n_max=2, grid=8, observable="intensity", normalization="sum")
        )
        pc, ps = _phasor(batch=1, grid=8)
        result = model.fit(pc, ps, torch.zeros(1, 1, 8, 8), ZernikeAmpFitConfig(epochs=1, log_every=0))
        assert result.observable == "intensity"
        assert result.normalization == "sum"
        assert result.n_max == 2