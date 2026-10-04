"""Regression tests for :meth:`ZernikeAmpModel.correction_far_field`.

The inverse-design seam. These guard the three properties the inverse loop depends
on and that nothing else in the suite would catch:

1. ``forward`` really is unusable for inverse design with a zero phasor (the reason
   this method exists at all) -- if that ever stops being true the motivation rots,
   so the test asserts the *defect*, not just the feature.
2. The new method is differentiable and its gradient is genuinely non-zero. A
   zero-gradient shape that still "runs" is the failure mode worth pinning.
3. Its shape/normalisation contract matches ``forward`` so the two are comparable.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.zernike.models import ZernikeAmpConfig, ZernikeAmpModel

GRID, N_MAX = 16, 6


def _model(**overrides) -> ZernikeAmpModel:
    config = ZernikeAmpConfig(n_max=N_MAX, grid=GRID, far_field_padding=2, **overrides)
    torch.manual_seed(0)
    return ZernikeAmpModel(config)


def _zero_phasor() -> tuple[torch.Tensor, torch.Tensor]:
    zeros = torch.zeros(1, 1, GRID, GRID)
    return zeros, zeros.clone()


class TestForwardCannotDriveInverseDesign:
    """The defect that motivated ``correction_far_field``."""

    def test_zero_phasor_forward_is_identically_zero(self) -> None:
        model = _model()
        model.coefficients.data.normal_(0, 0.5)
        cos, sin = _zero_phasor()
        with torch.no_grad():
            out = model(cos, sin)
        assert torch.count_nonzero(out) == 0, (
            "forward() with a zero phasor is expected to be identically zero; "
            "if this changed, re-check whether correction_far_field() is still needed"
        )

    def test_zero_phasor_forward_has_exactly_zero_gradient(self) -> None:
        """Exactly 0.0, not merely small -- the property that makes it useless."""
        model = _model()
        model.coefficients.data.normal_(0, 0.5)
        cos, sin = _zero_phasor()
        model(cos, sin).sum().backward()
        grad = model.coefficients.grad
        assert grad is not None
        assert torch.all(grad == 0.0), f"expected exactly-zero grad, got max {grad.abs().max()}"


class TestCorrectionFarField:
    def test_shape_matches_forward(self) -> None:
        model = _model()
        out = model.correction_far_field()
        assert out.shape == (1, 1, GRID, GRID), (
            "must match forward()'s (B, 1, g, g) contract, otherwise every "
            "comparison against forward is silently broadcasting"
        )

    def test_shape_without_center_crop(self) -> None:
        model = _model(center_crop=False)
        out = model.correction_far_field()
        assert out.shape == (1, 1, GRID * 2, GRID * 2)

    def test_gradient_is_nonzero(self) -> None:
        model = _model()
        model.coefficients.data.normal_(0, 0.5)
        model.correction_far_field().sum().backward()
        grad = model.coefficients.grad
        assert grad is not None
        assert torch.isfinite(grad).all(), "gradient contains NaN/Inf"
        assert grad.abs().max() > 0.0, "gradient is exactly zero -- inverse design cannot move"

    def test_normalize_false_is_peak_scaled(self) -> None:
        model = _model(normalization="peak")
        raw = model.correction_far_field(normalize=False)
        normalised = model.correction_far_field(normalize=True)
        assert torch.allclose(normalised, raw / raw.amax(dim=(-2, -1), keepdim=True), atol=1e-6)

    def test_normalized_output_peaks_at_one(self) -> None:
        model = _model(normalization="peak")
        out = model.correction_far_field()
        assert pytest.approx(1.0, abs=1e-5) == float(out.max())

    def test_amplitude_observable_also_differentiable(self) -> None:
        """The sqrt branch has a singular derivative at 0; it must stay finite."""
        model = _model(observable="amplitude")
        model.coefficients.data.normal_(0, 0.5)
        out = model.correction_far_field()
        out.sum().backward()
        grad = model.coefficients.grad
        assert grad is not None and torch.isfinite(grad).all()
        assert grad.abs().max() > 0.0

    def test_matches_forward_when_correction_is_zero(self) -> None:
        """With zero coefficients both paths reduce to the same bare pupil field.

        This is the real parity check: ``forward`` multiplies the measured phasor
        by ``exp(i*0) = 1``, so a unit phasor must reproduce ``correction_far_field``
        bit-for-bit through the shared propagation.
        """
        model = _model()
        with torch.no_grad():
            model.coefficients.zero_()
            unit = torch.ones(1, 1, GRID, GRID)
            via_forward = model(unit, unit)
            via_correction = model.correction_far_field()
        assert torch.allclose(via_forward, via_correction, atol=1e-6), (
            f"max abs diff {float((via_forward - via_correction).abs().max()):.3e}"
        )

    def test_zero_coefficients_give_a_centred_airy_core(self) -> None:
        """A flat circular pupil transforms to a concentrated Airy core.

        Deliberately *not* asserting exact centrosymmetry: that depends on the
        half-pixel grid-centre convention and fails on an even-sized grid, which
        makes it a fragile invariant rather than a meaningful one. Peak location and
        concentration are the physics.
        """
        model = _model()
        with torch.no_grad():
            model.coefficients.zero_()
            out = model.correction_far_field()
        peak = np.unravel_index(int(out.argmax()), tuple(out.shape))
        assert abs(peak[-2] - GRID // 2) <= GRID // 4, f"peak at {peak}, not centred"
        assert abs(peak[-1] - GRID // 2) <= GRID // 4, f"peak at {peak}, not centred"
        central = out[..., GRID // 4 : 3 * GRID // 4, GRID // 4 : 3 * GRID // 4].sum()
        assert float(central / out.sum()) > 0.5, "flat pupil should concentrate energy centrally"

    def test_attention_is_identity_at_init(self) -> None:
        """Attention must be a no-op at init, so enabling it changes nothing.

        ``out_proj`` is zero-initialised, so this holds by construction rather than
        by poking a parameter -- which is what makes it safe to leave attention
        wired in the shared path.
        """
        plain = _model()
        attended = _model(attention=True, attention_heads=2)
        with torch.no_grad():
            a = plain.correction_far_field()
            b = attended.correction_far_field()
        assert torch.equal(a, b), (
            f"attention at init must be an exact no-op, max dev "
            f"{float((a - b).abs().max()):.3e}"
        )

    def test_attention_is_applied_on_the_shared_path(self) -> None:
        """A non-uniform gate must actually change the shared observable path.

        Note the *weight* is perturbed, not the bias: a uniform bias only rescales
        the intensity, and peak normalisation divides that straight back out, so a
        bias-only change is invisible in the output and would make this test pass
        for the wrong reason.
        """
        plain = _model()
        attended = _model(attention=True, attention_heads=2)
        with torch.no_grad():
            attended.attention.out_proj.weight.normal_(0, 0.5)
            a = plain.correction_far_field()
            b = attended.correction_far_field()
        assert not torch.allclose(a, b), "attention gate had no effect on the output"

    def test_uniform_attention_bias_is_absorbed_by_peak_normalisation(self) -> None:
        """Pins why the test above perturbs weights: uniform gain cancels exactly."""
        plain = _model()
        attended = _model(attention=True, attention_heads=2)
        with torch.no_grad():
            attended.attention.out_proj.bias.fill_(0.5)
            a = plain.correction_far_field()
            b = attended.correction_far_field()
        assert torch.allclose(a, b, atol=1e-6)

    def test_is_deterministic(self) -> None:
        model = _model()
        assert torch.equal(model.correction_far_field(), model.correction_far_field())


class TestCoefficientSpace:
    def test_output_is_finite_for_large_coefficients(self) -> None:
        model = _model()
        model.coefficients.data.fill_(np.pi)
        assert torch.isfinite(model.correction_far_field()).all()

    def test_output_is_finite_for_tiny_coefficients(self) -> None:
        model = _model()
        model.coefficients.data.fill_(1e-8)
        assert torch.isfinite(model.correction_far_field()).all()