"""SLM nonlinearity must not act on the frozen optical aberration."""
import numpy as np
import torch

from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    FrozenDefocusForward, FrozenDefocusParameters,
)
from ao_shaping.algorithm.signal_processing.physical_forward_fit import PhysicalForwardFit


def test_nonlinear_fit_and_frozen_prediction_match():
    size = 24
    phases = np.random.default_rng(30).normal(size=(2, size, size)).astype(np.float32)
    xx, yy = np.meshgrid(np.linspace(1330, 1390, 14), np.linspace(990, 1030, 14))
    xy = np.stack([xx, yy], -1)
    matrix = ((465.146, 7174.691), (-7533.176, -30.278))
    fit = PhysicalForwardFit(
        phases, np.zeros((2, 14, 14), np.float32), xy, matrix, .35,
        np.ones(2, dtype=bool), (960., 600.), initial_defocus=.35,
        fft_size=96, pixel_samples=2, device="cpu", slm_phase_a=.15, slm_phase_b=-.03,
    )
    width = tuple(float(v) for v in fit.log_width.exp().detach())
    params = FrozenDefocusParameters(
        defocus_diopter=.35, origin_xy=(1360., 1020.),
        amplitude_center_xy=(960., 600.), amplitude_width_xy=width,
        residual_noll_radians={}, gain=1., background=0., sigma_ccd_px=0.,
        ramp_matrix=matrix, fft_size=96, pixel_samples=2, grid_size=size,
        slm_phase_a=.15, slm_phase_b=-.03,
    )
    frozen = FrozenDefocusForward(params, xy, device="cpu")
    fit.z4 = frozen.z4.clone()
    with torch.no_grad():
        torch.testing.assert_close(fit.prediction(), frozen.predict(phases), rtol=0, atol=0)

    # Both the nonlinear response and static defocus must be active in this test.
    with torch.no_grad():
        fit.d.zero_()
        assert not torch.allclose(fit.prediction(), frozen.predict(phases))
