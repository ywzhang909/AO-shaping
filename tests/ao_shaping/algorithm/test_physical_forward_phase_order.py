"""SLM nonlinearity must not act on the frozen optical aberration."""
import numpy as np
import pytest
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


def _phase_sign_case(defocus: float):
    size, half = 32, 12
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    phase = (0.25 * np.sin(2 * np.pi * (xx + 2 * yy) / size)
             + 0.15 * np.cos(2 * np.pi * yy / 9.)).astype(np.float32)[None]
    coords = np.arange(-half, half + 1, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(coords, coords)
    xy = np.stack([grid_x, grid_y], axis=-1)
    matrix = ((109.11873563, 7451.64925596), (-7205.68698343, 78.24473736))
    base = dict(
        defocus_diopter=defocus, origin_xy=(0., 0.),
        amplitude_center_xy=(960., 600.), amplitude_width_xy=(440., 440.),
        residual_noll_radians={}, gain=1., background=0., sigma_ccd_px=0.,
        ramp_matrix=matrix, fft_size=128, pixel_samples=1, grid_size=size,
    )

    def predict(sign: float, command=None):
        params = FrozenDefocusParameters(**{**base, "phase_sign": sign})
        forward = FrozenDefocusForward(params, xy, device="cpu")
        with torch.no_grad():
            return forward.predict(phase if command is None else command)

    return phase, predict


def test_phase_sign_negates_only_the_commanded_phase():
    """``phase_sign=-1`` is exactly "run the plus model on the negated command".

    On hardware the panel phase enters the propagation with a *negative* sign:
    with ``+1`` every single-mode Zernike screen Z5-Z11 (0.10/0.05/0.02 rad)
    measures a negative signed direction agreement (42/42, median -0.47), and
    negating the commanded phase flips all 42 positive.  The display chain is
    exonerated (``panel_gray`` reverse-solves to ``+panel_phase`` at 0.0035 rad
    RMS vs 0.179 rad for the negated hypothesis), so the defect is this model
    convention.

    Only the command flips: the fitted defocus and residual aberration are the
    model's own terms.  Negating those as well is a *global* conjugation and
    leaves every screen correlation at ~0, which is why the two cases below are
    asserted separately.
    """
    phase, predict = _phase_sign_case(defocus=.1139)
    plus = predict(1.)
    minus = predict(-1.)
    # The default must stay bit-for-bit the historical model.
    torch.testing.assert_close(predict(1.), plus, rtol=0, atol=0)
    torch.testing.assert_close(minus, predict(1., -phase), rtol=0, atol=0)
    assert not torch.allclose(plus, minus)
    # A global conjugation would be a point reflection; with defocus active the
    # minus model must NOT be one.
    with pytest.raises(AssertionError):
        torch.testing.assert_close(
            minus, plus.flip(-1).flip(-2), rtol=0, atol=1e-4 * float(plus.max()))


def test_phase_sign_without_defocus_point_reflects_the_far_field():
    """With no fitted defocus left to keep its sign, the flip is a conjugation.

    That point reflection about the 0-order is precisely why a defocus-dominated
    fit cannot see the polarity error: a radially symmetric far field is
    invariant under conjugation.
    """
    phase, predict = _phase_sign_case(defocus=0.)
    plus = predict(1.)
    minus = predict(-1.)
    torch.testing.assert_close(
        minus, plus.flip(-1).flip(-2), rtol=0, atol=1e-4 * float(plus.max()))
    assert not torch.allclose(plus, minus)


def test_phase_sign_rejects_values_other_than_unit_polarity():
    params = FrozenDefocusParameters(
        defocus_diopter=.1, origin_xy=(0., 0.), amplitude_center_xy=(960., 600.),
        amplitude_width_xy=(440., 440.), residual_noll_radians={}, gain=1.,
        background=0., sigma_ccd_px=0., ramp_matrix=((1., 0.), (0., 1.)),
        fft_size=32, pixel_samples=1, grid_size=8, phase_sign=.5,
    )
    xy = np.zeros((3, 3, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="phase_sign"):
        FrozenDefocusForward(params, xy, device="cpu")
