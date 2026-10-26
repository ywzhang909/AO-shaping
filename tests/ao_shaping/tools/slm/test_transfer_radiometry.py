"""Exposure-independent counts must become bias/exposure in normalized images."""
from dataclasses import replace

import numpy as np
import pytest

from ao_shaping.tools.slm.transfer.radiometry import fit_radiometry


def test_recovers_shared_gain_background_and_counts_bias_without_validation_leakage():
    optical = np.random.default_rng(4).uniform(size=(6, 10, 10))
    exposure = np.array([.1, .5, 2., .2, .8, 3.])
    train = np.array([True, True, True, False, False, False])
    measured = 3*optical+.4+.8/exposure[:, None, None]
    prediction, params = fit_radiometry(optical, measured, exposure, train)
    np.testing.assert_allclose(prediction, measured, atol=1e-10)
    np.testing.assert_allclose(list(params.values()), [3, .4, .8], atol=1e-10)
    measured[~train] = 1e6
    _, changed = fit_radiometry(optical, measured, exposure, train)
    assert changed == params


def test_frozen_forward_requires_exposure_and_preserves_phase_gradient():
    import torch
    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        FrozenDefocusForward, FrozenDefocusParameters,
    )
    p = FrozenDefocusParameters(0., (10., 10.), (960., 600.), (300., 260.), {},
        1., .4, 0., ((7500., 0.), (0., 7500.)), grid_size=16, fft_size=64, pixel_samples=1)
    yy, xx = np.mgrid[:21, :21]
    xy = np.stack((xx, yy), axis=-1)
    old = FrozenDefocusForward(p, xy, device="cpu")
    new = FrozenDefocusForward(replace(p, bias_counts=.8), xy, device="cpu")
    phase = torch.tensor(np.random.default_rng(2).normal(size=(16, 16)), dtype=torch.float32, requires_grad=True)
    baseline = old.predict(phase)
    with pytest.raises(ValueError, match="requires exposure"):
        new.predict(phase)
    corrected = new.predict(phase, exposure_ms=.2)
    torch.testing.assert_close(corrected, baseline+4.)
    a = torch.autograd.grad(baseline.square().sum(), phase, retain_graph=True)[0]
    b = torch.autograd.grad((corrected-4.).square().sum(), phase)[0]
    torch.testing.assert_close(a, b)
    batch = new.predict(np.zeros((2, 16, 16)), exposure_ms=[.2, .4])
    torch.testing.assert_close(batch[0]-batch[1], torch.full_like(batch[0], 2.))
    with pytest.raises(ValueError, match="positive"):
        new.predict(phase, exposure_ms=0)
    assert FrozenDefocusParameters.from_dicts(new.params.to_dict(), p.ramp_matrix).bias_counts == .8
