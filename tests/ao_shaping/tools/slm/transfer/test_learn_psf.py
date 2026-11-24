"""Optional bounded, trainable static CCD Gaussian PSF for the transfer fit."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    FrozenDefocusForward,
)
from ao_shaping.algorithm.signal_processing.physical_forward_fit import PhysicalForwardFit
from ao_shaping.tools.slm.transfer import fit as fit_module
from ao_shaping.tools.slm.transfer import radiometry as radiometry_module
from ao_shaping.tools.slm.transfer.adaptation import configure_adaptation


def _model(*, learn_psf: bool = False, initial_sigma: float = 1.0,
           sigma_max: float = 8.0, learn_geometry: bool = False) -> PhysicalForwardFit:
    yy, xx = np.mgrid[-2:3, -2:3]
    camera_xy = np.stack((xx + 1360, yy + 1020), axis=-1)
    return PhysicalForwardFit(
        np.zeros((2, 8, 8), dtype=np.float32), np.ones((2, 5, 5), dtype=np.float32),
        camera_xy, np.diag([10000.0, 12000.0]), 0.0, [True, False], [960.0, 600.0],
        fft_size=32, learn_geometry=learn_geometry,
        learn_psf=learn_psf, initial_psf_sigma_ccd_px=initial_sigma,
        psf_sigma_max_ccd_px=sigma_max,
    )


def _comps(*, sigma: float, kernel) -> SimpleNamespace:
    pattern = np.arange(1, 2 * 5 * 5 + 1, dtype=np.float32).reshape(2, 5, 5)
    return SimpleNamespace(
        images=pattern,
        exposure_values=np.ones(2, dtype=np.float64),
        train=np.array([True, False]),
        kernel=kernel,
        matrix=np.diag([10000.0, 12000.0]),
        old=SimpleNamespace(sigma_ccd_px=sigma),
    )


def test_psf_is_disabled_without_a_parameter_or_kernel() -> None:
    model = _model()
    assert model.psf_raw is None
    assert model.psf_sigma() is None
    assert model.psf_sigma_value() is None
    assert model.psf_kernel() is None
    assert "psf_raw" not in dict(model.named_parameters())


def test_disabled_fit_loss_sigma_follows_the_checkpoint() -> None:
    import torch

    model = _model()
    comps = _comps(sigma=2.5, kernel=FrozenDefocusForward._build_blaze_kernel(
        SimpleNamespace(_device=torch.device("cpu")), 2.5))
    _, entry, _ = fit_module.fit_loss(model, comps)
    assert entry["parameters"]["sigma_ccd_px"] == 2.5


def test_learned_psf_is_bounded_normalised_and_differentiable() -> None:
    import torch

    model = _model(learn_psf=True, initial_sigma=3.0)
    assert model.psf_raw is not None and model.psf_raw.requires_grad
    kernel = model.psf_kernel()
    assert tuple(kernel.shape) == (1, 1, 25, 25)
    torch.testing.assert_close(kernel.sum(), torch.tensor(1.0), atol=1e-5, rtol=0)

    # Strictly inside (0, max) even when the raw parameter is driven far out.
    model.psf_raw.data.fill_(1000.0)
    assert 0.0 < model.psf_sigma_value() < 8.0
    model.psf_raw.data.fill_(-1000.0)
    assert 0.0 < model.psf_sigma_value() < 8.0

    # The loss must reach the sigma through the convolution.
    model.psf_raw.data.fill_(0.5)
    from torch.nn.functional import conv2d

    prediction = model.prediction()
    loss = (conv2d(prediction[:, None], model.psf_kernel(), padding=12) ** 2).mean()
    loss.backward()
    gradient = model.psf_raw.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_learned_kernel_matches_the_frozen_export_convention() -> None:
    import torch

    model = _model(learn_psf=True, initial_sigma=1.0)
    sigma_max = model.psf_sigma_max_ccd_px
    for target in (0.75, 2.0, 6.5):
        ratio = target / sigma_max
        model.psf_raw.data.fill_(float(np.log(ratio / (1.0 - ratio))))
        frozen_like = FrozenDefocusForward._build_blaze_kernel(
            SimpleNamespace(_device=torch.device("cpu")), target)
        torch.testing.assert_close(model.psf_kernel(), frozen_like, atol=1e-6, rtol=0)


def test_initial_sigma_is_clamped_away_from_the_bounds() -> None:
    for initial in (1e-6, 8.0):
        model = _model(learn_psf=True, initial_sigma=initial, sigma_max=8.0)
        sigma = model.psf_sigma_value()
        assert 0.0 < sigma < 8.0


def test_configure_adaptation_freezes_or_releases_the_psf_only() -> None:
    model = _model(learn_psf=True, learn_geometry=True)
    configure_adaptation(model, "full", learn_psf=False)
    assert model.psf_raw.requires_grad is False
    assert model.geometry_raw.requires_grad is False

    configure_adaptation(model, "full", learn_psf=True)
    assert model.psf_raw.requires_grad is True
    # Releasing the PSF never releases the geometry correction.
    assert model.geometry_raw.requires_grad is False

    # The historical two-argument call keeps its behaviour: psf stays frozen.
    configure_adaptation(model, "full")
    assert model.psf_raw.requires_grad is False


def test_fit_loss_uses_the_dynamic_kernel_exactly_once(monkeypatch) -> None:
    import torch
    from torch.nn.functional import conv2d

    def identity_radiometry(optical, measured, exposure_ms, train, *, mode="exposure"):
        return None, dict(gain=1.0, background=0.0, bias_counts=0.0)

    monkeypatch.setattr(radiometry_module, "fit_radiometry", identity_radiometry)
    model = _model(learn_psf=True, initial_sigma=2.0)
    # A deliberately different frozen kernel (same 25x25 padding contract):
    # using it, or convolving twice, would change the prediction.
    box = torch.ones((1, 1, 25, 25), dtype=torch.float32)
    comps = _comps(sigma=0.0, kernel=box / box.sum())

    base = model.prediction()
    expected = conv2d(base[:, None], model.psf_kernel(), padding=12)[:, 0]
    _, entry, predicted = fit_module.fit_loss(model, comps)

    np.testing.assert_allclose(predicted, expected.detach().cpu().numpy(), rtol=1e-5, atol=1e-6)
    double = conv2d(expected[:, None], comps.kernel, padding=12)[:, 0]
    assert not np.allclose(predicted, double.detach().cpu().numpy())
    np.testing.assert_allclose(entry["parameters"]["sigma_ccd_px"],
                               model.psf_sigma_value(), rtol=1e-6)
