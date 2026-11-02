"""Reference checks for the opt-in SLM -> lens -> camera forward model."""
import numpy as np
import pytest
import torch

from ao_shaping.algorithm.signal_processing.defocus_forward_core import (
    collins_field,
    lens_abcd,
)
from ao_shaping.algorithm.signal_processing.full_chain_forward import (
    free_space_field,
    full_chain_intensity,
    pixel_integrated_intensity,
    sample_field_intensity,
)

LAM = 1064e-9
F = .3


def geometry(n=64, pitch=50e-6):
    x = (torch.arange(n, dtype=torch.float64) - n // 2) * pitch
    out = torch.fft.fftshift(torch.fft.fftfreq(n, d=pitch, dtype=torch.float64)) * LAM * F
    yy, xx = torch.meshgrid(x, x, indexing="ij")
    field = torch.exp(-(xx ** 2 + yy ** 2) / (150e-6) ** 2)
    field = field * torch.exp(1j * xx / 100e-6)
    return x, out, field.to(torch.complex128)


def test_full_chain_matches_single_collins_focal_limit():
    x, out, field = geometry()
    a, b, _, d = lens_abcd(F, 0., 0.)
    expected = collins_field(torch, field, x, x, out, out, a=a, b=b, d=d).abs() ** 2
    actual = full_chain_intensity(
        torch, torch.angle(field), x, x, focal_length_m=F,
        amplitude=field.abs(), camera_x_m=out, camera_y_m=out,
    )[0]
    assert abs(actual.sum() - expected.sum()) / expected.sum() < .02
    assert (actual - expected).abs().max() / expected.max() < .05


def test_finite_lens_aperture_makes_slm_offset_observable():
    x, out, _ = geometry()
    yy, xx = torch.meshgrid(x, x, indexing="ij")
    aperture = (xx ** 2 + yy ** 2 < (1e-3) ** 2).to(torch.float64)
    phase = (xx ** 2 + yy ** 2) / (1e-3) ** 2
    common = dict(focal_length_m=F, camera_x_m=out, camera_y_m=out)
    open0 = full_chain_intensity(torch, phase, x, x, **common)
    open1 = full_chain_intensity(torch, phase, x, x, slm_offset_m=.1, **common)
    stop0 = full_chain_intensity(torch, phase, x, x, lens_aperture=aperture, **common)
    stop1 = full_chain_intensity(torch, phase, x, x, slm_offset_m=.1,
                                lens_aperture=aperture, **common)
    open_change = (open1 - open0).abs().max() / open0.max()
    stop_change = (stop1 - stop0).abs().max() / stop0.max()
    assert stop_change > 2 * open_change
    assert stop_change > .02


def test_phase_and_distances_are_differentiable():
    n = 8
    x = (torch.arange(n, dtype=torch.float64) - n // 2) * 100e-6
    out = torch.fft.fftshift(torch.fft.fftfreq(n, d=100e-6, dtype=torch.float64)) * LAM * F
    phase = torch.zeros(n, n, dtype=torch.float64, requires_grad=True)
    slm = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    camera = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    weights = torch.arange(n * n, dtype=torch.float64).reshape(n, n)

    def loss(phi, z1, z2):
        intensity = full_chain_intensity(
            torch, phi, x, x, focal_length_m=F, slm_offset_m=z1,
            camera_offset_m=z2, camera_x_m=out, camera_y_m=out,
        )[0]
        return (intensity * weights).sum()

    assert torch.autograd.gradcheck(loss, (phase, slm, camera), eps=1e-6, atol=1e-6)


def test_camera_sampling_and_pixel_integration_are_consistent():
    n = 16
    x = (torch.arange(n, dtype=torch.float64) - n // 2) * 10e-6
    intensity = torch.ones(1, n, n, dtype=torch.float64)
    points = torch.tensor([[[0., 0.], [10e-6, 10e-6]]], dtype=torch.float64)
    assert torch.allclose(sample_field_intensity(torch, intensity, x, x, points), torch.ones_like(points[..., 0]))
    integrated = pixel_integrated_intensity(
        torch, intensity, x, x, points,
        pixel_pitch_xy_m=(2e-6, 2e-6), samples_per_axis=3,
    )
    torch.testing.assert_close(integrated, torch.ones_like(integrated), atol=1e-12, rtol=0.)


def test_free_space_preserves_contained_beam_energy():
    n, pitch = 64, 50e-6
    x = (torch.arange(n, dtype=torch.float64) - n // 2) * pitch
    yy, xx = torch.meshgrid(x, x, indexing="ij")
    field = torch.exp(-(xx ** 2 + yy ** 2) / (150e-6) ** 2).to(torch.complex128)
    propagated = free_space_field(torch, field, x, x, x, x, F)
    before = (field.abs() ** 2).sum()
    after = (propagated.abs() ** 2).sum()
    assert float(after / before) == pytest.approx(1., rel=1e-3)


def test_zero_distance_requires_matching_axes():
    n = 8
    x = (torch.arange(n, dtype=torch.float64) - n // 2) * 10e-6
    field = torch.ones(1, n, n, dtype=torch.complex128)
    assert torch.equal(free_space_field(torch, field, x, x, x, x, 0.), field)
    shifted = x + 1e-6
    with pytest.raises(ValueError, match="identical"):
        free_space_field(torch, field, x, x, shifted, x, 0.)
