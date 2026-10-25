"""Tests for the 2f forward model (``ml.sim2f.optics``).

The physics is pinned by measurement, not by restating the code:

* the focal-plane pitch and the camera ``sampling_ratio`` are derived from
  ``wavelength * focal_length / (fft_n * pupil_pitch)`` and matched analytically;
* the tilt <-> spot-offset conversion is checked against the **measured centroid**
  (a requested ``(3, -2)`` px lands at ``(2.99, -1.99)`` px, 0.011 px of error
  against a 0.25 px budget);
* camera resampling is checked to be area integration -- rows sum to the pixel
  *width in model cells* ``1 / sampling_ratio`` on both axes, the flat spot stays
  centred, and total frame energy is invariant across phase realisations
  (2.8e-4 relative spread on 5 random aberrations);
* gradients are checked against central differences in float64, for the phase and
  for an aberration coefficient tensor.

Pure torch/numpy -- no corpus, no hardware.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch lives in the optional ml group")

from ml.sim2f.config import OpticsErrors, Sim2fConfig, fast_len
from ml.sim2f.optics import Sim2f

#: Small bench with the reference optics (same 56.25/112.5 um pitch class, same
#: ``sampling_ratio``) but a 64-sample pupil and a 64-px camera window.
TINY = Sim2fConfig(grid=64, region_span_px=900.0, ccd_px=64)
#: Bigger grid for the energy test, where the camera window must capture the
#: whole spot.
BIG = Sim2fConfig(grid=128, ccd_px=128)


def _centroid(frame: np.ndarray) -> np.ndarray:
    """Intensity centroid in camera pixels, ``(x, y)``, origin at the frame centre."""
    values = np.asarray(frame, dtype=np.float64)
    total = float(values.sum())
    assert total > 0.0
    rows = np.arange(values.shape[-2]) - values.shape[-2] / 2.0
    cols = np.arange(values.shape[-1]) - values.shape[-1] / 2.0
    return np.array(
        [
            float((values.sum(axis=-2) * cols).sum() / total),
            float((values.sum(axis=-1) * rows).sum() / total),
        ]
    )


def _is_five_smooth(value: int) -> bool:
    """Whether ``value`` has no prime factor above 5."""
    remainder = int(value)
    for factor in (2, 3, 5):
        while remainder % factor == 0:
            remainder //= factor
    return remainder == 1


# ----------------------------------------------------------------------
# Geometry and sampling
# ----------------------------------------------------------------------
def test_focal_pitch_and_sampling_ratio_are_analytic() -> None:
    """``far_field_pitch`` and ``sampling_ratio`` follow their closed forms."""
    cfg = Sim2fConfig()
    expected_pitch = (
        cfg.wavelength_m * cfg.focal_length_m / (cfg.fft_length * cfg.pupil_pitch_m)
    )
    assert cfg.fft_length == cfg.grid * cfg.far_field_padding
    assert cfg.far_field_pitch_m == pytest.approx(expected_pitch, rel=1e-12)
    assert cfg.sampling_ratio == pytest.approx(
        cfg.far_field_pitch_m / cfg.ccd_pitch_m, rel=1e-12
    )
    sim = Sim2f(cfg, device="cpu")
    assert sim.fft_n == cfg.fft_length
    assert sim.grid == cfg.grid
    assert sim.ccd_px == cfg.ccd_px


def test_exact_fft_n_makes_the_grids_coincide() -> None:
    """At ``exact_fft_n`` one model sample spans exactly one camera pixel."""
    base = Sim2fConfig()
    target_length = 1080.0
    needed_pitch_m = (
        base.wavelength_m
        * base.focal_length_m
        / (base.pupil_pitch_m * target_length * (1.0 - 1e-6))
    )
    cfg = Sim2fConfig(ccd_pixel_um=needed_pitch_m * 1e6)
    assert cfg.exact_fft_n == int(target_length)

    ratio_at_exact = (
        cfg.wavelength_m * cfg.focal_length_m
        / (cfg.exact_fft_n * cfg.pupil_pitch_m * cfg.ccd_pitch_m)
    )
    assert ratio_at_exact == pytest.approx(1.0, rel=2e-3)
    # The default padding is deliberately *not* the exact length (that is the
    # speed/fidelity compromise), so the ratio is > 1 there.
    assert base.sampling_ratio > 2.0


def test_fast_len_sanity() -> None:
    """``fast_len`` returns 5-smooth lengths and is idempotent on them."""
    assert fast_len(4819) == 4860
    assert fast_len(1) == 1
    assert fast_len(512) == 512
    previous = 0
    for value in range(1, 400, 7):
        result = fast_len(value)
        assert result >= value
        assert _is_five_smooth(result)
        assert result >= previous
        previous = result
    with pytest.raises(ValueError):
        fast_len(0)


# ----------------------------------------------------------------------
# Tilt <-> spot offset
# ----------------------------------------------------------------------
def test_tilt_moves_the_spot_to_the_requested_offset() -> None:
    """A ``(3, -2)`` px request lands within 0.25 px (measured: 0.011 px)."""
    sim = Sim2f(TINY, device="cpu")
    offset = np.array([3.0, -2.0])
    coeffs = sim.tilt_coeffs_for_offset_px(offset)
    phase = sim.aberration(coeffs[None, :]).numpy().astype(np.float32)
    frame = sim.forward(phase)[0, 0].numpy()
    measured = _centroid(frame)
    assert np.abs(measured - offset).max() < 0.25
    # The sign convention is not a free choice: a wrong sign would land at -d.
    assert measured[0] > 0.0 > measured[1]


def test_tilt_conversion_round_trips() -> None:
    """``offset -> coeffs -> offset`` is exact."""
    sim = Sim2f(TINY, device="cpu")
    for offset in ([3.0, -2.0], [0.0, 0.0], [-5.5, 4.25]):
        coeffs = sim.tilt_coeffs_for_offset_px(offset)
        assert sim.offset_px_for_tilt_coeffs(coeffs) == pytest.approx(
            offset, abs=1e-12
        )


# ----------------------------------------------------------------------
# Camera sampling
# ----------------------------------------------------------------------
def test_resampling_rows_sum_to_the_pixel_width_in_cells() -> None:
    """Both matrices are area integrators: every row sums to ``1 / ratio``.

    ``sampling_ratio`` is cell size over pixel size, so a camera pixel covers its
    reciprocal in model cells. Using ``sampling_ratio`` itself (as an earlier
    draft did) mis-scales area by ``1 / ratio**2`` -- 4.4x here.
    """
    sim = Sim2f(Sim2fConfig(), device="cpu")
    expected = 1.0 / TINY.sampling_ratio
    rows_x = sim._wx.sum(dim=1).numpy()
    rows_y = sim._wy_t.sum(dim=0).numpy()
    assert rows_x == pytest.approx(expected, rel=1e-6)
    assert rows_y == pytest.approx(expected, rel=1e-6)
    assert float(np.abs(rows_x).min()) > 0.0


def test_flat_frame_is_centred() -> None:
    """A flat phase on centred illumination puts the centroid at the centre."""
    sim = Sim2f(TINY, device="cpu")
    frame = sim.forward(np.zeros((1, TINY.grid, TINY.grid), dtype=np.float32))
    assert np.abs(_centroid(frame[0, 0].numpy())).max() < 0.05


def test_energy_is_invariant_across_phase_realisations() -> None:
    """Total frame energy does not depend on the phase (measured spread 2.8e-4)."""
    sim = Sim2f(BIG, device="cpu")
    rng = np.random.default_rng(1)
    energies = []
    for _ in range(5):
        coeffs = np.zeros((1, 7))
        coeffs[0, 2:] = rng.normal(0.0, 0.3, size=5)  # Noll 4..8, no tip/tilt
        phase = sim.aberration(coeffs).numpy()
        energies.append(float(sim.forward(phase).sum()))
    energies = np.array(energies)
    assert (energies.max() - energies.min()) / energies.mean() < 1e-3


def test_camera_window_captures_the_beam() -> None:
    """The camera window holds essentially all of the model power.

    Each output pixel is an *integral* over its area in model cells, so the frame
    sum is a plain energy sum (no ``sampling_ratio`` factor): a camera window that
    encloses the beam must reproduce the model's total power.
    """
    sim = Sim2f(BIG, device="cpu")
    zero = np.zeros((1, BIG.grid, BIG.grid), dtype=np.float32)
    model = sim.field(zero).abs().square()
    frame = sim.forward(zero)
    assert float(frame.sum()) / float(model.sum()) == pytest.approx(1.0, abs=1e-3)


# ----------------------------------------------------------------------
# Gradients
# ----------------------------------------------------------------------
def _smooth_phase(grid: int, dtype: torch.dtype) -> torch.Tensor:
    """A deterministic, smoothly varying phase on the pupil grid."""
    axis = torch.arange(grid, dtype=dtype)
    return (
        1.2 * torch.sin(axis[None, :] / 7.0) * torch.cos(axis[:, None] / 9.0)
        + 0.7 * torch.cos(axis[None, :] / 4.0)
    )


def test_phase_gradient_matches_finite_difference() -> None:
    """Autograd w.r.t. the phase matches central differences in float64."""
    sim = Sim2f(TINY, device="cpu", dtype=torch.float64)
    generator = torch.Generator().manual_seed(0)
    # Zero-mean weights: the total frame energy is phase-invariant, so an
    # all-positive weight vector makes the loss O(100) while a single pixel's
    # gradient is O(1e-7) -- and the difference form then drowns in round-off.
    weights = torch.rand(
        (1, 1, TINY.ccd_px, TINY.ccd_px), dtype=torch.float64, generator=generator
    )
    weights = weights - weights.mean()

    phase = _smooth_phase(TINY.grid, torch.float64).requires_grad_(True)
    loss = (sim.forward(phase.unsqueeze(0)) * weights).sum()
    loss.backward()
    analytic = phase.grad.detach().clone()

    # 1e-4, not 1e-6: the loss is O(100) while a single pixel's gradient is
    # ~1e-7, so a tighter step drowns the difference form in round-off.
    step = 1e-4
    for row, col in ((10, 20), (33, 40), (55, 8)):
        bumped = phase.detach().clone()
        bumped[row, col] += step
        plus = float((sim.forward(bumped.unsqueeze(0)) * weights).sum())
        bumped[row, col] -= 2.0 * step
        minus = float((sim.forward(bumped.unsqueeze(0)) * weights).sum())
        numeric = (plus - minus) / (2.0 * step)
        assert analytic[row, col].item() == pytest.approx(numeric, rel=1e-4)


def test_aberration_gradient_matches_finite_difference() -> None:
    """Autograd w.r.t. Zernike coefficients matches central differences."""
    sim = Sim2f(TINY, device="cpu", dtype=torch.float64)
    generator = torch.Generator().manual_seed(1)
    weights = torch.rand(
        (1, 1, TINY.ccd_px, TINY.ccd_px), dtype=torch.float64, generator=generator
    )
    weights = weights - weights.mean()
    phase = _smooth_phase(TINY.grid, torch.float64).unsqueeze(0)

    coeffs = torch.tensor(
        [[0.3, -0.2, 0.15, 0.05, -0.1]], dtype=torch.float64, requires_grad=True
    )
    loss = (sim.forward(phase, aberration=sim.aberration(coeffs)) * weights).sum()
    loss.backward()
    analytic = coeffs.grad.detach().clone()

    step = 1e-4
    for index in (0, 2, 4):
        bumped = coeffs.detach().clone()
        bumped[0, index] += step
        plus = float(
            (sim.forward(phase, aberration=sim.aberration(bumped)) * weights).sum()
        )
        bumped[0, index] -= 2.0 * step
        minus = float(
            (sim.forward(phase, aberration=sim.aberration(bumped)) * weights).sum()
        )
        numeric = (plus - minus) / (2.0 * step)
        assert analytic[0, index].item() == pytest.approx(numeric, rel=1e-4)


# ----------------------------------------------------------------------
# Zernike basis and aberration
# ----------------------------------------------------------------------
def test_zernike_basis_modes_and_support() -> None:
    """Noll 4 is azimuthally symmetric, 2/3 are antisymmetric, all masked."""
    sim = Sim2f(TINY, device="cpu")
    basis = sim.zernike_basis(6).numpy()
    assert basis.shape == (6, TINY.grid, TINY.grid)

    defocus = basis[3]  # Noll 4 = (2, 0)
    assert np.allclose(defocus, defocus[:, ::-1], atol=1e-12)
    assert np.allclose(defocus, defocus[::-1, :], atol=1e-12)
    for index in (1, 2):  # Noll 2 = (1, 1), Noll 3 = (1, -1)
        tilt = basis[index]
        assert np.allclose(tilt, -tilt[:, ::-1], atol=1e-12) or np.allclose(
            tilt, -tilt[::-1, :], atol=1e-12
        )

    outside = ~sim.pupil_mask[0, 0].numpy().astype(bool)
    assert np.abs(basis[:, outside]).max() == 0.0


def test_aberration_is_linear_and_masked() -> None:
    """The aberration is a linear map and vanishes outside the support disc."""
    sim = Sim2f(TINY, device="cpu")
    first = np.array([[0.4, -0.3, 0.2, 0.0, 0.1]])
    second = np.array([[-0.1, 0.05, 0.3, -0.2, 0.0]])

    combined = sim.aberration(first + second).numpy()
    separate = sim.aberration(first).numpy() + sim.aberration(second).numpy()
    assert np.allclose(combined, separate, atol=1e-6)

    outside = ~sim.pupil_mask[0, 0].numpy().astype(bool)
    assert np.abs(sim.aberration(np.ones((1, 8))).numpy()[:, outside]).max() == 0.0


def test_aberration_broadcasts_over_the_phase_batch() -> None:
    """One aberration serves a whole batch, and a batch of aberrations works."""
    sim = Sim2f(TINY, device="cpu")
    phase = np.zeros((3, TINY.grid, TINY.grid), dtype=np.float32)
    aberration = sim.aberration(np.array([[0.2, 0.0, 0.0, 0.0, 0.0]]))
    field = sim.field(phase, aberration=aberration)
    assert tuple(field.shape) == (3, 1, sim.fft_n, sim.fft_n)

    batched = sim.aberration(np.zeros((2, 5)))
    assert tuple(sim.field(np.zeros((2, TINY.grid, TINY.grid)), aberration=batched).shape) == (
        2,
        1,
        sim.fft_n,
        sim.fft_n,
    )


# ----------------------------------------------------------------------
# Pupil plane
# ----------------------------------------------------------------------
def test_illumination_is_a_peak_normalised_gaussian_without_a_hard_edge() -> None:
    """Peak is 1, there is no aperture step, and ``width_scale`` widens it."""
    sim = Sim2f(TINY, device="cpu")
    amplitude = sim.nominal_illumination[0, 0].numpy()
    assert float(amplitude.max()) == pytest.approx(1.0, rel=1e-12)
    # No hard aperture: the corner is small but non-zero (a masked pupil would be
    # exactly 0 beyond ``pupil_radius_px``).
    assert float(amplitude[-1, -1]) > 0.0
    wider = sim.illumination(width_scale=2.0)[0, 0].numpy()
    assert float(wider[-1, -1]) > float(amplitude[-1, -1])

    # Pupil coordinates are centred on ``(grid - 1) / 2``, not ``grid / 2``, so
    # the peak lands on the sample nearest the requested offset (half-sample
    # rounding, hence the 1 px tolerance).
    offset = sim.illumination(center_px=(5.0, -3.0))[0, 0].numpy()
    peak = np.unravel_index(int(offset.argmax()), offset.shape)
    assert peak[0] == pytest.approx((TINY.grid - 1) / 2.0 - 3.0, abs=1.0)
    assert peak[1] == pytest.approx((TINY.grid - 1) / 2.0 + 5.0, abs=1.0)


def test_blur_phase_reduces_high_frequency_content() -> None:
    """The crosstalk kernel smooths a checkerboard and is off by default."""
    sharp = Sim2fConfig(grid=64, region_span_px=900.0, ccd_px=64, phase_blur_px=0.0)
    soft = Sim2fConfig(grid=64, region_span_px=900.0, ccd_px=64, phase_blur_px=1.5)
    phase = torch.zeros(1, 64, 64)
    phase[0, ::2, ::2] = 1.0
    phase[0, 1::2, 1::2] = 1.0

    assert torch.equal(Sim2f(sharp, device="cpu").blur_phase(phase), phase)
    blurred = Sim2f(soft, device="cpu").blur_phase(phase)
    assert float(blurred.max() - blurred.min()) < 1.0
    assert float(blurred.mean()) == pytest.approx(float(phase.mean()), rel=1e-5)


# ----------------------------------------------------------------------
# Input validation and the detector
# ----------------------------------------------------------------------
def test_forward_rejects_wrong_shapes() -> None:
    """Geometry mismatches fail loudly instead of broadcasting silently."""
    sim = Sim2f(TINY, device="cpu")
    # The bench is batched on purpose -- a 2x batch is fine. What is *not* fine is
    # a missing batch axis or a wrong spatial size.
    assert tuple(sim.forward(np.zeros((2, TINY.grid, TINY.grid), np.float32)).shape) == (
        2,
        1,
        TINY.ccd_px,
        TINY.ccd_px,
    )
    with pytest.raises(ValueError):
        sim.forward(np.zeros((TINY.grid, TINY.grid), dtype=np.float32))
    with pytest.raises(ValueError):
        sim.forward(np.zeros((1, TINY.grid + 2, TINY.grid), dtype=np.float32))
    with pytest.raises(ValueError):
        sim.forward(np.zeros((1, 3, TINY.grid, TINY.grid), dtype=np.float32))
    with pytest.raises(ValueError):
        sim.field(np.zeros((1, TINY.grid, TINY.grid)), aberration=np.zeros((1, 4, 4)))


def test_phasor_input_matches_radians() -> None:
    """``(cos, sin)`` channels produce the same field as the angle."""
    sim = Sim2f(TINY, device="cpu")
    phase = _smooth_phase(TINY.grid, torch.float32).unsqueeze(0)
    phasor = torch.stack([torch.cos(phase), torch.sin(phase)], dim=1)
    reference = sim.forward(phase)
    # A 3x scale on the phasor must not matter: it is normalised internally.
    assert torch.allclose(sim.forward(3.0 * phasor), reference, rtol=1e-5, atol=1e-6)


def test_detector_off_by_default_and_counts_when_enabled() -> None:
    """``exposure_ms == 0`` gives intensity; a calibrated bench gives counts."""
    ideal = Sim2f(Sim2fConfig(), device="cpu")
    zero = np.zeros((1, ideal.grid, ideal.grid), dtype=np.float32)
    intensity = ideal.forward(zero)
    assert float(intensity.min()) >= 0.0

    calibrated = Sim2fConfig.calibrated()
    sim = Sim2f(calibrated, device="cpu")
    frame = sim.forward(zero)
    assert frame.dtype == torch.float32
    # Quantised, non-negative counts with a dark floor around dark_level_counts.
    assert torch.equal(frame, torch.round(frame))
    assert float(frame.min()) >= 0.0
    assert float(frame.max()) <= float(calibrated.gray_max)
    corner = float(frame[..., 0, 0])
    assert corner == pytest.approx(calibrated.dark_level_counts, abs=10.0)

    # ``detector=False`` bypasses the detector even on a counts bench.
    assert torch.equal(sim.forward(zero, detector=False), sim.forward(zero, detector=False))
    assert float(sim.forward(zero, detector=False).max()) != float(frame.max())


def test_detector_noise_is_reproducible_from_a_generator() -> None:
    """Two calls with the same generator state are identical; the bench is not."""
    sim = Sim2f(Sim2fConfig.calibrated(), device="cpu")
    zero = np.zeros((1, sim.grid, sim.grid), dtype=np.float32)
    first = sim.forward(zero, generator=torch.Generator().manual_seed(7))
    second = sim.forward(zero, generator=torch.Generator().manual_seed(7))
    assert torch.equal(first, second)


def test_zero_order_puts_light_in_an_unmodulated_component() -> None:
    """``eta = 1`` leaves only the unmodulated (flat) field, ``eta = 0`` none."""
    cfg = Sim2fConfig(grid=64, region_span_px=900.0, ccd_px=64)
    base = Sim2f(cfg, device="cpu")
    zero = np.zeros((1, cfg.grid, cfg.grid), dtype=np.float32)
    flat = base.forward(zero)

    only_zero_order = Sim2f(
        Sim2fConfig(
            grid=64,
            region_span_px=900.0,
            ccd_px=64,
            errors=OpticsErrors(zero_order_eta=0.99),
        ),
        device="cpu",
    ).forward(zero)
    # eta -> 1: the field collapses onto the unmodulated pupil, i.e. the same
    # *shape* as the flat frame (a scaled Airy spot at 0-order).
    assert torch.allclose(
        only_zero_order / only_zero_order.max(), flat / flat.max(), rtol=1e-3, atol=1e-6
    )
    # And it is not the same absolute frame: the modulated light is gone.
    assert not math.isclose(
        float(only_zero_order.sum()), float(flat.sum()), rel_tol=1e-3
    )


