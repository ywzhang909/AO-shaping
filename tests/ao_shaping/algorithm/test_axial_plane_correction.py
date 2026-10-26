"""ABCD input-plane correction, backward compatibility and phase gradients."""
from dataclasses import replace

import numpy as np
import pytest

from ao_shaping.algorithm.signal_processing.defocus_forward_core import axial_plane_correction


def test_abcd_composition_matches_matrix_multiplication():
    for a, b, offset in ((.2, .12, .08), (-.1, .15, -.03), (0., .12, .4)):
        nominal = np.array([[a, b], [-1/b, 0.]])
        result = nominal @ np.array([[1., offset], [0., 1.]])
        d, scale = axial_plane_correction(a/b, offset)
        assert d == pytest.approx(result[0, 0]/result[0, 1])
        assert scale == pytest.approx(result[0, 1]/b)
    assert axial_plane_correction(0., .5) == (0., 1.)
    with pytest.raises(ValueError, match="singular"):
        axial_plane_correction(2., -.5)


def test_axial_forward_equals_effective_calibration_and_keeps_phase_gradient():
    torch = pytest.importorskip("torch")
    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import FrozenDefocusForward, FrozenDefocusParameters
    p = FrozenDefocusParameters(.2, (20., 20.), (960., 600.), (300., 300.), {}, 1., 0., 0.,
                                ((2000., 0.), (0., 2000.)), fft_size=64, grid_size=16, pixel_samples=1)
    y, x = np.mgrid[10:31:2, 10:31:2]
    xy = np.stack((x, y), -1)
    offset = .1
    d, scale = axial_plane_correction(p.defocus_diopter, offset)
    adjusted = replace(p, slm_axial_offset_m=offset)
    equivalent = replace(p, defocus_diopter=d, gain=p.gain/scale**2,
                         ramp_matrix=tuple(tuple(v*scale for v in row) for row in p.ramp_matrix))
    phase = torch.tensor(np.random.default_rng(42).normal(size=(16, 16)), dtype=torch.float32, requires_grad=True)
    model = FrozenDefocusForward(adjusted, xy, device="cpu")
    prediction = model.predict(phase)
    expected = FrozenDefocusForward(equivalent, xy, device="cpu").predict(phase)
    torch.testing.assert_close(prediction, expected, rtol=3e-5, atol=1e-7)
    prediction.square().sum().backward()
    index = np.unravel_index(phase.grad.abs().argmax().item(), phase.shape)
    epsilon = .001
    plus, minus = phase.detach().clone(), phase.detach().clone()
    plus[index] += epsilon
    minus[index] -= epsilon
    numerical = (model.predict(plus).square().sum()-model.predict(minus).square().sum())/(2*epsilon)
    torch.testing.assert_close(phase.grad[index], numerical, rtol=.03, atol=1e-6)
    assert FrozenDefocusParameters.from_dicts(adjusted.to_dict(), p.ramp_matrix).slm_axial_offset_m == offset
    # A quadratic incident wavefront is distinct from propagation A/B. At the
    # Fourier plane the latter is zero, even with nonzero input aberration.
    focal = replace(p, axial_reference_curvature_diopter=0., slm_axial_offset_m=.3)
    assert focal.effective_parameters().defocus_diopter == p.defocus_diopter
    assert focal.effective_parameters().ramp_matrix == p.ramp_matrix
    assert focal.effective_parameters().gain == p.gain
