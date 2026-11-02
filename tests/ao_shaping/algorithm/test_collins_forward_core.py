"""Independent focal limit and local phase/distance gradient checks."""
import numpy as np
import pytest
import torch

from ao_shaping.algorithm.signal_processing.defocus_forward_core import (
    collins_field, lens_abcd,
)

LAM = 1064e-9
F = .3


def geometry(n=24):
    x = (torch.arange(n, dtype=torch.float64)-n//2)*30e-6
    out = torch.fft.fftshift(torch.fft.fftfreq(n, d=30e-6, dtype=torch.float64))*LAM*F
    yy, xx = torch.meshgrid(x, x, indexing="ij")
    field = torch.exp(-(xx**2+yy**2)/(150e-6)**2).to(torch.complex128)
    field *= torch.exp(1j*(xx/100e-6 + yy**2/100e-12))
    return x, out, field


def propagate(field, x, out, slm=0., camera=0.):
    a, b, _, d = lens_abcd(F, slm, camera)
    return collins_field(torch, field, x, x, out, out, a=a, b=b, d=d)


def test_abcd_matches_matrix_product():
    z1, z2 = .32, .297
    p1 = np.array([[1., z1], [0., 1.]])
    p2 = np.array([[1., z2], [0., 1.]])
    lens = np.array([[1., 0.], [-1/F, 1.]])
    actual = np.array(lens_abcd(F, z1-F, z2-F)).reshape(2, 2)
    np.testing.assert_allclose(actual, p2 @ lens @ p1, atol=1e-15)
    assert np.linalg.det(actual) == pytest.approx(1.)


def test_focal_limit_matches_fft_complex_field():
    x, out, field = geometry()
    expected = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(field)))
    expected *= (x[1]-x[0])**2/(1j*LAM*F)
    torch.testing.assert_close(propagate(field, x, out), expected, atol=1e-12, rtol=1e-11)


def test_slm_offset_changes_only_phase_when_ccd_is_focal():
    x, out, field = geometry()
    nominal = propagate(field, x, out)
    shifted = propagate(field, x, out, slm=.08)
    torch.testing.assert_close(shifted.abs()**2, nominal.abs()**2, atol=1e-14, rtol=1e-12)
    assert not torch.allclose(shifted, nominal)


def test_phase_and_offsets_gradcheck():
    x, out, field = geometry(10)
    phase = torch.zeros_like(field.real, requires_grad=True)
    slm = torch.tensor(.04, dtype=torch.float64, requires_grad=True)
    camera = torch.tensor(.003, dtype=torch.float64, requires_grad=True)

    def intensity(phi, z1, z2):
        return propagate(field*torch.exp(1j*phi), x, out, z1, z2).abs()**2

    assert torch.autograd.gradcheck(intensity, (phase, slm, camera), eps=1e-6, atol=1e-7)
    assert not torch.allclose(intensity(phase, slm, camera), intensity(phase, slm, camera*0))


def test_slm_distance_is_unidentifiable_at_focal_camera():
    x, out, field = geometry(10)
    slm = torch.tensor(.04, dtype=torch.float64, requires_grad=True)
    weights = torch.arange(100, dtype=torch.float64).reshape(10, 10)
    loss = (propagate(field, x, out, slm=slm).abs()**2*weights).sum()
    loss.backward()
    assert abs(slm.grad) < 1e-10


def test_anchored_image_keeps_the_physical_phase_jacobian():
    x, out, field = geometry(10)
    phase = torch.zeros_like(field.real, requires_grad=True)
    predicted = propagate(field*torch.exp(1j*phase), x, out, .04, .003).abs()**2
    # An anchor changes the operating image, not the derivative of the model.
    measured = predicted.detach()*1.2 + .001
    anchored = measured + predicted-predicted.detach()
    torch.testing.assert_close(anchored, measured)
    weights = torch.arange(100, dtype=torch.float64).reshape(10, 10)
    grad = torch.autograd.grad((predicted*weights).sum(), phase, retain_graph=True)[0]
    anchored_grad = torch.autograd.grad((anchored*weights).sum(), phase)[0]
    torch.testing.assert_close(anchored_grad, grad)


def test_singular_and_nonuniform_grid_rejected():
    x, out, field = geometry()
    with pytest.raises(ValueError, match="imaging"):
        collins_field(torch, field, x, x, out, out, a=1., b=0., d=1.)
    x[3] += 1e-6
    with pytest.raises(ValueError, match="uniform"):
        propagate(field, x, out)
