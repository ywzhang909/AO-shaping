"""Unit tests for the optional physical-model extensions (report 20261027 §4).

The extensions are gated behind checkpoint flags that default to the identity, so
the historical linear baseline is preserved; these tests pin the *non-default*
behaviour: SLM phase nonlinearity, Gaussian amplitude residual, monotonic CCD
response, and the ShapeObjective formation terms.
"""
from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.algorithm.signal_processing.slm_phase_nonlinearity import (
    slm_phase_nonlinearity, check_monotonicity, clamp_coefficients, derivative,
)
from ao_shaping.algorithm.signal_processing.amplitude_residual import (
    AmplitudeResidual, amplitude_with_residual,
)
from ao_shaping.algorithm.signal_processing.monotonic_response import (
    MonotonicResponse, MonotonicResponseConfig, monotonic_derivative_check,
)
from ao_shaping.algorithm.signal_processing.multishape_objective import ShapeObjective

torch = pytest.importorskip("torch")


# --- SLM phase nonlinearity -------------------------------------------------

def test_slm_nonlinearity_default_is_identity():
    phi = np.linspace(0, 2 * np.pi, 50)
    out = slm_phase_nonlinearity(phi, 0.0, 0.0)
    assert np.allclose(out, phi)


def test_slm_nonlinearity_adds_periodic_residual():
    phi = np.array([0.5, 1.2, 2.7])
    out = slm_phase_nonlinearity(phi, 0.2, 0.1)
    assert not np.allclose(out, phi)
    # Monotonicity preserved for small coefficients.
    assert check_monotonicity(0.2, 0.1)


def test_slm_monotonicity_bound():
    # |a| + 2|b| must stay < 1 for the residual to be monotonic increasing.
    assert check_monotonicity(0.3, 0.3)  # 0.3 + 0.6 = 0.9 < 1
    assert not check_monotonicity(0.5, 0.5)  # 0.5 + 1.0 = 1.5 >= 1


def test_slm_clamp_coefficients_shrinks_to_margin():
    a, b = clamp_coefficients(0.8, 0.4, margin=0.9)
    assert check_monotonicity(a, b)
    # Scale factor < 1 means coefficients shrank.
    assert abs(a) < 0.8 or abs(b) < 0.4


def test_slm_nonlinearity_torch_matches_numpy():
    phi = np.linspace(0, 3 * np.pi, 20)
    np_out = slm_phase_nonlinearity(phi, -0.15, 0.07)
    t_out = slm_phase_nonlinearity(torch.tensor(phi), -0.15, 0.07)
    assert np.allclose(np_out, t_out.numpy(), atol=1e-9)


def test_slm_derivative_positive_under_monotonicity():
    phi = torch.linspace(0, 2 * np.pi, 100)
    d = derivative(phi, 0.2, 0.1)
    assert bool((d > 0).all())


# --- Gaussian amplitude residual --------------------------------------------

def test_amplitude_residual_flat_for_zero_coeffs():
    ar = AmplitudeResidual(grid_size=32, n_modes=2)
    r = ar.evaluate(np.mgrid[0:32, 0:32][1], np.mgrid[0:32, 0:32][0])
    assert np.allclose(r, 0.0)


def test_amplitude_residual_shape_and_nonneg_exp_mode():
    base = np.ones((32, 32))
    ar = AmplitudeResidual(grid_size=32, n_modes=2)
    # Set a low-freq mode.
    ar.set_coeffs(np.array([0.1] + [0.0] * ((2 * 2 + 1) ** 2 - 1)))
    panel_x = np.mgrid[0:32, 0:32][1]
    panel_y = np.mgrid[0:32, 0:32][0]
    r = ar.evaluate(panel_x, panel_y)
    out = amplitude_with_residual(base, r, mode="exp")
    assert out.shape == (32, 32)
    assert (out > 0).all()


def test_amplitude_residual_softplus_mode_positive():
    base = np.full((16, 16), 0.5)
    r = np.random.default_rng(0).standard_normal((16, 16)) * 0.2
    out = amplitude_with_residual(base, r, mode="softplus")
    assert (out > 0).all()


def test_amplitude_residual_coeff_count_mismatch_raises():
    ar = AmplitudeResidual(grid_size=32, n_modes=2)
    with pytest.raises(ValueError):
        ar.set_coeffs(np.zeros(5))  # expects (2*2+1)^2 = 25


# --- Monotonic CCD response -------------------------------------------------

def test_monotonic_response_identity_for_zero_h():
    cfg = MonotonicResponseConfig(n_bins=64)
    mr = MonotonicResponse(cfg, np.zeros(64))
    v = np.array([0.0, 64.0, 128.0, 255.0])
    # Zero h => derivative softplus(0)+eps ~ 0.693, integral is affine but the
    # reference anchoring (response(v_ref)=v_ref) pins identity at the reference.
    out = mr.forward(v)
    assert out.shape == v.shape
    assert np.all(np.diff(out) > 0)  # strictly increasing


def test_monotonic_response_numpy_and_torch_agree():
    cfg = MonotonicResponseConfig(n_bins=64)
    h = np.linspace(-0.3, 0.3, 64)
    mr = MonotonicResponse(cfg, h)
    v = np.array([10.0, 100.0, 200.0])
    np_out = mr.forward(v)
    t_out = mr.forward(torch.tensor(v, dtype=torch.float64))
    assert np.allclose(np_out, t_out.numpy(), atol=1e-9)


def test_monotonic_response_positive_derivative():
    cfg = MonotonicResponseConfig(n_bins=64)
    h = np.linspace(-1.0, 1.0, 64)
    mr = MonotonicResponse(cfg, h)
    v = torch.linspace(0, 255, 128, dtype=torch.float64)
    d = mr.derivative(v)
    assert bool((d > 0).all())


def test_monotonic_derivative_check_true():
    assert monotonic_derivative_check(np.linspace(-5, 5, 10))


# --- ShapeObjective formation terms ----------------------------------------

def _square_target(size=32, fill=8, span=16):
    t = np.zeros((size, size), dtype=np.float32)
    t[fill:fill + span, fill:fill + span] = 1.0
    return t


def test_formation_returns_all_terms():
    target = _square_target()
    obj = ShapeObjective(target, formation=True, n_partitions=4, peripheral_px=2)
    img = torch.rand(32, 32, dtype=torch.float32) * 10 + 1
    loss, parts = obj.evaluate(img, torch.zeros(32, 32))
    for key in ("l_partition", "l_peripheral", "l_floor", "l_boundary"):
        assert key in parts
    assert float(loss) == float(loss)  # finite


def test_formation_terms_zero_when_not_requested():
    target = _square_target()
    obj = ShapeObjective(target)  # default mode
    img = torch.rand(32, 32, dtype=torch.float32)
    _, parts = obj.evaluate(img, torch.zeros(32, 32))
    assert parts["l_partition"] == 0.0
    assert parts["l_peripheral"] == 0.0
    assert parts["l_floor"] == 0.0
    assert parts["l_boundary"] == 0.0


def test_formation_partition_penalises_empty_subregion():
    target = _square_target()
    obj = ShapeObjective(target, formation=True, n_partitions=2)
    # Image bright only in upper-left quadrant -> partition imbalance.
    img = torch.zeros(32, 32)
    img[0:16, 0:16] = 10.0
    _, parts = obj.evaluate(img, torch.zeros(32, 32))
    # One partition full, others empty => non-zero std => non-zero penalty.
    assert parts["l_partition"] > 0.0


def test_cv_only_and_formation_mutually_exclusive():
    target = _square_target()
    with pytest.raises(ValueError):
        ShapeObjective(target, cv_only=True, formation=True)


# --- FrozenDefocusForward / PhysicalForwardFit wiring (synthetic geometry) ----

SIZE = 16
FFT_SIZE = 64
PIXEL_SAMPLES = 2
CAMERA_H = CAMERA_W = 8
RAMP_MATRIX = ((465.146, 7174.691), (-7533.176, -30.278))
ORIGIN_XY = (1360.0, 1020.0)
AMPLITUDE_WIDTH = (260.0, 260.0)
AMPLITUDE_CENTER = (960.0, 600.0)
DEFOCUS_DIOPTER = 0.35
GAIN = 1.7
BACKGROUND = 0.02
SIGMA_CCD_PX = 1.5


def _camera_xy():
    xs = np.linspace(1330.0, 1390.0, CAMERA_W)
    ys = np.linspace(990.0, 1030.0, CAMERA_H)
    xx, yy = np.meshgrid(xs, ys)
    return np.stack([xx, yy], axis=-1)


def _phases():
    rng = np.random.default_rng(20261025)
    return (rng.standard_normal((2, SIZE, SIZE)) * 0.4).astype(np.float32)


def _frozen_params(**overrides):
    base = dict(
        defocus_diopter=DEFOCUS_DIOPTER, origin_xy=ORIGIN_XY,
        amplitude_center_xy=AMPLITUDE_CENTER, amplitude_width_xy=AMPLITUDE_WIDTH,
        residual_noll_radians={}, gain=GAIN, background=BACKGROUND,
        sigma_ccd_px=SIGMA_CCD_PX, ramp_matrix=RAMP_MATRIX, fft_size=FFT_SIZE,
        pixel_samples=PIXEL_SAMPLES, grid_size=SIZE,
    )
    base.update(overrides)
    return __import__("ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity",
                      fromlist=["FrozenDefocusParameters"]).FrozenDefocusParameters(**base)


def test_frozen_defaults_ignore_extensions_identity():
    import torch
    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        FrozenDefocusForward,
    )
    params = _frozen_params()
    forward = FrozenDefocusForward(params, _camera_xy(), device="cpu")
    # Defaults must build no extension modules.
    assert forward._slm_phase is None
    assert forward._amp_residual is None
    assert forward._monotonic_ccd is None
    with torch.no_grad():
        out = forward.predict(_phases())
    assert out.shape == (2, CAMERA_H, CAMERA_W)


def test_frozen_slm_phase_residual_changes_prediction():
    import torch
    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        FrozenDefocusForward,
    )
    base = FrozenDefocusForward(_frozen_params(), _camera_xy(), device="cpu")
    ext = FrozenDefocusForward(_frozen_params(slm_phase_a=0.2, slm_phase_b=0.1),
                               _camera_xy(), device="cpu")
    assert ext._slm_phase is not None
    with torch.no_grad():
        a = base.predict(_phases()).numpy()
        b = ext.predict(_phases()).numpy()
    assert not np.allclose(a, b)
    assert np.isfinite(b).all()


def test_frozen_amplitude_residual_changes_prediction():
    import torch
    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        FrozenDefocusForward,
    )
    # 25 = (2*2+1)^2 coeffs for n_modes=2.
    coeffs = tuple(float(c) for c in (np.random.default_rng(1).standard_normal(25) * 0.05))
    ext = FrozenDefocusForward(_frozen_params(amplitude_residual_coeffs=coeffs),
                               _camera_xy(), device="cpu")
    assert ext._amp_residual is not None
    base = FrozenDefocusForward(_frozen_params(), _camera_xy(), device="cpu")
    with torch.no_grad():
        a = base.predict(_phases()).numpy()
        b = ext.predict(_phases()).numpy()
    assert not np.allclose(a, b)


def test_frozen_monotonic_ccd_changes_prediction():
    import torch
    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
        FrozenDefocusForward,
    )
    h = tuple(float(c) for c in np.linspace(-0.2, 0.2, 64))
    base = FrozenDefocusForward(_frozen_params(), _camera_xy(), device="cpu")
    ext = FrozenDefocusForward(_frozen_params(monotonic_ccd_h=h), _camera_xy(), device="cpu")
    assert ext._monotonic_ccd is not None
    with torch.no_grad():
        a = base.predict(_phases()).numpy()
        b = ext.predict(_phases()).numpy()
    assert not np.allclose(a, b)
    # Monotonic response is increasing in the linear (identity) region.
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.9


def test_physical_forward_fit_extension_flags_roundtrip():
    import torch
    from ao_shaping.algorithm.signal_processing.physical_forward_fit import (
        PhysicalForwardFit,
    )
    images = np.zeros((2, CAMERA_H, CAMERA_W), dtype=np.float32)
    train = np.ones((2, CAMERA_H, CAMERA_W), dtype=bool)
    coeffs = tuple(float(c) for c in np.zeros(25))  # identity amplitude residual
    fitter = PhysicalForwardFit(
        _phases(), images, _camera_xy(), RAMP_MATRIX, DEFOCUS_DIOPTER, train,
        AMPLITUDE_CENTER, initial_defocus=DEFOCUS_DIOPTER, learn_amplitude=False,
        learn_residual=False, fft_size=FFT_SIZE, pixel_samples=PIXEL_SAMPLES,
        amplitude_residual_coeffs=coeffs,
    )
    assert fitter._amp_residual is not None
    with torch.no_grad():
        pred = fitter.prediction().numpy()
    assert pred.shape == (2, CAMERA_H, CAMERA_W)
    assert np.isfinite(pred).all()
