"""Tests for the measured-camera transfer stage.

The transfer sits between the optics and the numbers, so two properties matter
more than any fitted value:

* **identity is exact.**  A calibration that never fitted the transfer, and every
  artifact written before the stage existed, must reproduce the old numbers bit
  for bit -- not "to within a rounding error".
* **it is differentiable and physically parameterised.**  The blur must conserve
  flux and the background must be a fraction of a unit-sum frame, so autograd can
  fit both through the same loss as the optical knobs.  A finite-difference check
  pins the sigma gradient, because a kernel whose derivative silently vanishes is
  a knob the optimiser cannot move.

The background's identifiability is checked separately: the prepared-frame score
subtracts the median, so :func:`ml.sim2f.measurement.pedestal_mismatch` is the
term that makes it visible, and that term is tested on a frame whose pedestal is
known by construction.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch", reason="torch lives in the optional ml group")

from ml.sim2f.measurement import (
    MAX_BACKGROUND_FRACTION,
    MAX_PSF_SIGMA_PX,
    MeasurementTransfer,
    apply_transfer,
    gaussian_blur,
    pedestal_mismatch,
)

#: Small enough to run fast; wide enough that a 4 px PSF never sees the border.
SHAPE = (2, 1, 32, 32)


def _frame(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """A smooth, non-negative frame on :data:`SHAPE` (no border discontinuity)."""
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, SHAPE[-2], dtype=dtype),
        torch.linspace(-1.0, 1.0, SHAPE[-1], dtype=dtype),
        indexing="ij",
    )
    signal = torch.exp(-4.0 * (x * x + y * y))
    return signal.view(1, 1, *SHAPE[-2:]).repeat(SHAPE[0], 1, 1, 1)


def test_the_default_transfer_is_the_identity() -> None:
    """No fitted transfer must leave a frame bit-for-bit unchanged."""
    frame = _frame()
    assert torch.equal(apply_transfer(frame, 0.0, 0.0), frame)
    assert torch.equal(MeasurementTransfer().apply(frame), frame)
    assert MeasurementTransfer().is_identity()


def test_a_missing_measurement_block_loads_as_the_identity() -> None:
    """Old ``calibration.json`` files have no ``measurement`` key at all."""
    for payload in (None, {}):
        transfer = MeasurementTransfer.from_dict(payload)
        assert transfer.is_identity()
        assert transfer == MeasurementTransfer()
    assert MeasurementTransfer().to_dict() == {
        "psf_sigma_px": 0.0,
        "background_fraction": 0.0,
        "zero_order_fraction": 0.0,
    }


def test_background_is_uniform_and_a_fraction_of_the_frame_total() -> None:
    """``f`` means the same physical thing at any frame size or exposure scale."""
    frame = _frame()
    frame = frame / frame.sum(dim=(-2, -1), keepdim=True)
    fraction = 0.2
    out = apply_transfer(frame, 0.0, fraction)
    per_pixel = fraction / float(frame.shape[-2] * frame.shape[-1])
    assert torch.allclose(out - frame, torch.full_like(frame, per_pixel), rtol=0, atol=1e-7)
    # Total added intensity is exactly the fraction *per frame*, so the sum stays
    # in the model's own unit-sum units.  The frame carries a batch axis, so the
    # sum is taken per frame, not over the whole tensor.
    added = (out - frame).sum(dim=(-2, -1))
    assert torch.allclose(added, torch.full_like(added, fraction), rtol=0, atol=1e-5)


def test_background_scales_with_the_frame_total() -> None:
    """The pedestal is relative: twice the exposure, twice the added intensity."""
    frame = _frame()
    doubled = 2.0 * frame
    once = apply_transfer(frame, 0.0, 0.1) - frame
    twice = apply_transfer(doubled, 0.0, 0.1) - doubled
    assert torch.allclose(twice, 2.0 * once, rtol=1e-6, atol=1e-9)


def test_background_gradient_is_the_frame_area_in_per_pixel_units() -> None:
    """d(sum over one frame)/d(fraction) = frame total: the parameter is a fraction."""
    frame = _frame(dtype=torch.float64)
    frame = frame / frame.sum(dim=(-2, -1), keepdim=True)
    fraction = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    # Mean over the batch so the check is on one frame's area, not the batch's.
    apply_transfer(frame, 0.0, fraction).flatten(1).sum(dim=1).mean().backward()
    assert fraction.grad is not None
    assert float(fraction.grad) == pytest.approx(1.0, rel=1e-12)


def test_blur_conserves_flux_and_widens_a_point() -> None:
    """A PSF redistributes light; it does not create or destroy it."""
    frame = torch.zeros((1, 1, 32, 32))
    frame[0, 0, 16, 16] = 1.0
    blurred = gaussian_blur(frame, 1.5)
    assert float(blurred.sum()) == pytest.approx(1.0, rel=1e-6)
    assert float(blurred[0, 0, 16, 16]) < 1.0
    assert float(blurred.min()) >= 0.0
    # Isotropic and separable: the 2-D profile is the outer product of one 1-D
    # profile, hence symmetric under a transpose about the point.
    patch = blurred[0, 0, 10:23, 10:23]
    assert torch.allclose(patch, patch.T, rtol=0, atol=1e-7)


def test_blur_is_identity_at_zero_and_smooth_at_small_sigma() -> None:
    """Zero is exact; just above it the kernel must stay finite and normalised."""
    frame = _frame()
    assert torch.equal(gaussian_blur(frame, 0.0), frame)
    tiny = gaussian_blur(frame, 1e-3)
    assert torch.isfinite(tiny).all()
    assert float(tiny.sum()) == pytest.approx(float(frame.sum()), rel=1e-6)


def test_sigma_gradient_matches_a_finite_difference() -> None:
    """The fit can only move sigma if autograd sees the kernel's derivative."""
    frame = _frame(dtype=torch.float64)
    weight = torch.linspace(0.1, 1.0, 32 * 32, dtype=torch.float64).view(1, 1, 32, 32)
    sigma = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    (gaussian_blur(frame, sigma) * weight).sum().backward()
    assert sigma.grad is not None
    analytic = float(sigma.grad)

    step = 1e-4
    plus = (gaussian_blur(frame, torch.tensor(0.8 + step, dtype=torch.float64)) * weight).sum()
    minus = (gaussian_blur(frame, torch.tensor(0.8 - step, dtype=torch.float64)) * weight).sum()
    numeric = float((plus - minus) / (2.0 * step))
    assert analytic == pytest.approx(numeric, rel=1e-5, abs=1e-8)
    assert math.isfinite(analytic)


def test_pedestal_term_identifies_the_background_fraction() -> None:
    """A uniform offset is invisible to the prepared score, not to this term."""
    frame = _frame(dtype=torch.float64)
    fraction = 0.1
    gain = torch.full((SHAPE[0],), 3.0, dtype=torch.float64)
    measured = gain.view(-1, 1, 1, 1) * (frame + fraction / (32.0 * 32.0))

    def mismatch(value: float) -> float:
        prediction = frame + value / (32.0 * 32.0)
        return float(pedestal_mismatch(prediction, measured, gain).mean())

    assert mismatch(fraction) == pytest.approx(0.0, abs=1e-12)
    assert mismatch(0.0) > 0.0
    assert mismatch(fraction) < mismatch(0.0)

    value = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    prediction = frame + value / (32.0 * 32.0)
    pedestal_mismatch(prediction, measured, gain).mean().backward()
    assert value.grad is not None
    # Raising the background moves the prediction towards the measurement.
    assert float(value.grad) < 0.0


def test_zero_order_term_needs_a_template_and_is_off_by_default() -> None:
    """The structural term exists but cannot fire without a measured template."""
    frame = _frame()
    template = torch.full_like(frame, 1.0 / (32.0 * 32.0))
    transfer = MeasurementTransfer(psf_sigma_px=0.7, zero_order_fraction=0.25)
    assert torch.equal(transfer.apply(frame), apply_transfer(frame, 0.7, 0.0))
    with_template = transfer.apply(frame, template=template)
    expected = apply_transfer(frame, 0.7, 0.0) + 0.25 * template
    assert torch.allclose(with_template, expected, rtol=0, atol=1e-7)


def test_negative_or_non_finite_values_are_rejected() -> None:
    """A transfer cannot have a negative width, background or template weight."""
    for kwargs in (
        {"psf_sigma_px": -0.1},
        {"background_fraction": -0.1},
        {"zero_order_fraction": -0.1},
        {"psf_sigma_px": float("nan")},
    ):
        with pytest.raises(ValueError):
            MeasurementTransfer(**kwargs)
    with pytest.raises(ValueError):
        gaussian_blur(_frame(), -1.0)


def test_the_fit_bounds_are_the_documented_ones() -> None:
    """The bounds are a contract with the optimiser, so pin their values."""
    assert MAX_PSF_SIGMA_PX == 4.0
    assert MAX_BACKGROUND_FRACTION == 0.5
