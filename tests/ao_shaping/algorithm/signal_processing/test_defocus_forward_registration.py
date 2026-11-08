"""Issue #237: the optional global subpixel 2D registration shift.

The registration is one global ``(dy, dx)`` in **CCD pixels** applied to the
camera sampling correspondence inside ``FrozenDefocusForward`` and
``PhysicalForwardFit``.  These tests pin the four properties the change claims:

* **default off and byte-preserving** -- ``None`` must hand the historical
  sampling grid through untouched (the helper returns the *same object*), and
  ``0.0`` must reproduce the disabled prediction with ``np.array_equal``.  The
  historical numbers themselves stay pinned by the pre-existing
  ``defocus_forward_core_golden.json`` twin of the same geometry;
* **semantics** -- an enabled ``(dy, dx)`` must be equivalent to displacing the
  camera coordinates by the same amount, i.e. it moves the correspondence and
  nothing else: no panel grid, resolution or camera grid shape change;
* **differentiability** -- the fitted shift must carry a finite gradient through
  the torch sampling step, and only when the caller opts in;
* **serialisation** -- the value must round-trip through ``to_dict`` /
  ``from_dicts`` / ``load_frozen_parameters``, and old checkpoints without the
  key must load as disabled.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ao_shaping.algorithm.signal_processing.defocus_forward_core import (
    registration_sample_grid,
    registration_shift_vector,
)
from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    FrozenDefocusForward,
    FrozenDefocusParameters,
    load_frozen_parameters,
)

# --- shared synthetic geometry, same block as test_defocus_forward_core.py ---
SIZE = 16
FFT_SIZE = 64
PIXEL_SAMPLES = 2
CAMERA_H = CAMERA_W = 12
RAMP_MATRIX = ((465.146, 7174.691), (-7533.176, -30.278))
ORIGIN_XY = (1360.0, 1020.0)
AMPLITUDE_WIDTH = (260.00006103515625, 260.00006103515625)
AMPLITUDE_CENTER = (960.0, 600.0)
DEFOCUS_DIOPTER = 0.35
GAIN = 1.7
BACKGROUND = 0.02
SIGMA_CCD_PX = 1.5
#: Camera sample spacing in CCD px for ``_camera_xy`` below.
CAMERA_STEP_PX = 60.0 / (CAMERA_W - 1)


def _camera_xy() -> np.ndarray:
    xs = np.linspace(1330.0, 1390.0, CAMERA_W)
    ys = np.linspace(990.0, 1030.0, CAMERA_H)
    xx, yy = np.meshgrid(xs, ys)
    return np.stack((xx, yy), axis=-1)


def _phases() -> np.ndarray:
    rng = np.random.default_rng(20261025)
    return (rng.standard_normal((2, SIZE, SIZE)) * 0.4).astype(np.float32)


def _params(registration: Any = None) -> FrozenDefocusParameters:
    return FrozenDefocusParameters(
        defocus_diopter=DEFOCUS_DIOPTER,
        origin_xy=ORIGIN_XY,
        amplitude_center_xy=AMPLITUDE_CENTER,
        amplitude_width_xy=AMPLITUDE_WIDTH,
        residual_noll_radians={},
        gain=GAIN,
        background=BACKGROUND,
        sigma_ccd_px=SIGMA_CCD_PX,
        ramp_matrix=RAMP_MATRIX,
        fft_size=FFT_SIZE,
        pixel_samples=PIXEL_SAMPLES,
        grid_size=SIZE,
        registration_shift_ccd_px=registration,
    )


def _forward(registration: Any = None, camera_xy: np.ndarray | None = None,
             **overrides: Any) -> FrozenDefocusForward:
    fields = {**_params(registration).__dict__, **overrides}
    return FrozenDefocusForward(FrozenDefocusParameters(**fields),
                                _camera_xy() if camera_xy is None else camera_xy,
                                device="cpu")


def _predict(registration: Any = None, camera_xy: np.ndarray | None = None) -> np.ndarray:
    import torch

    forward = _forward(registration, camera_xy)
    with torch.no_grad():
        return forward.predict(_phases()).numpy()


# ---------------------------------------------------------------- default off


def test_registration_defaults_to_disabled() -> None:
    assert _params().registration_shift_ccd_px is None
    assert _forward().registration_shift is None
    assert _params().to_dict()["registration_shift_ccd_px"] is None


def test_disabled_helper_returns_the_untouched_grid_object() -> None:
    """Disabled means *no arithmetic at all*: the historical grid flows through."""
    import torch

    xy = torch.as_tensor(_camera_xy(), dtype=torch.float32)
    assert registration_sample_grid(torch, xy, None) is xy
    with pytest.raises(ValueError, match="disabled"):
        registration_shift_vector(torch, None)


@pytest.mark.parametrize("shift", [(0.0, 0.0), (0, 0), [0.0, 0.0], np.zeros(2)])
def test_zero_shift_is_bit_identical_to_disabled(shift: Any) -> None:
    """A zero shift goes through the sampling path and must move no byte."""
    assert np.array_equal(_predict(None), _predict(shift))


# ------------------------------------------------------------------ semantics


def test_nonzero_shift_changes_the_prediction() -> None:
    shifted = _predict((2.0, 3.0))
    assert shifted.shape == _predict(None).shape
    assert not np.array_equal(shifted, _predict(None))


@pytest.mark.parametrize("shift", [(0.0, CAMERA_STEP_PX), (CAMERA_STEP_PX, 0.0),
                                   (0.527, 1.762), (-1.0, 2.5)])
def test_shift_is_equivalent_to_displacing_the_camera_coordinates(shift: tuple) -> None:
    """``(dy, dx)`` moves the camera sampling points, in CCD px, and nothing else.

    Independent path: build the same forward from camera coordinates already
    displaced by ``-(dy, dx)`` with registration off.  Same correspondence, so
    the predictions must agree to sampling round-off.
    """
    dy, dx = shift
    xy = _camera_xy()
    displaced = xy - np.array([dx, dy], dtype=xy.dtype)[None, None, :]
    assert np.allclose(_predict(shift), _predict(None, displaced), rtol=0, atol=1e-5)


def test_registration_leaves_panel_and_camera_grid_alone() -> None:
    """No geometry collateral damage: panel resolution, camera shape, Zernike basis."""
    plain, shifted = _forward(), _forward((0.527, 1.762))
    assert shifted.camera_xy.shape == plain.camera_xy.shape == (CAMERA_H, CAMERA_W, 2)
    assert shifted.params.grid_size == plain.params.grid_size == SIZE
    assert shifted.params.fft_size == plain.params.fft_size == FFT_SIZE
    assert int(shifted.z4.numel()) == int(plain.z4.numel())
    assert np.array_equal(shifted.z4.detach().numpy(), plain.z4.detach().numpy())
    # Output stays on the same camera grid: shape and sampling coordinates.
    assert _predict((0.527, 1.762)).shape == _predict(None).shape


def test_invalid_registration_values_are_rejected() -> None:
    import torch

    from ao_shaping.algorithm.signal_processing.physical_forward_fit import PhysicalForwardFit

    for bad in [(np.nan, 0.0), (np.inf, 0.0), (1.0,), (1.0, 2.0, 3.0), "ab"]:
        with pytest.raises((ValueError, TypeError)):
            _params(bad)
        with pytest.raises((ValueError, TypeError)):
            PhysicalForwardFit(_phases(), np.zeros((2, CAMERA_H, CAMERA_W), dtype=np.float32),
                               _camera_xy(), RAMP_MATRIX, DEFOCUS_DIOPTER,
                               np.ones((2, CAMERA_H, CAMERA_W), dtype=bool), AMPLITUDE_CENTER,
                               initial_defocus=DEFOCUS_DIOPTER, learn_amplitude=False,
                               learn_residual=False, fft_size=FFT_SIZE,
                               pixel_samples=PIXEL_SAMPLES, basis_arrays=None,
                               registration_shift_ccd_px=bad)
    with pytest.raises(ValueError, match="shape"):
        registration_shift_vector(torch, torch.zeros(3))


# ------------------------------------------------------------ differentiability


def _fitter(registration: Any = None, **kwargs):
    from ao_shaping.algorithm.signal_processing.physical_forward_fit import PhysicalForwardFit

    return PhysicalForwardFit(
        _phases(), np.zeros((2, CAMERA_H, CAMERA_W), dtype=np.float32), _camera_xy(),
        RAMP_MATRIX, DEFOCUS_DIOPTER, np.ones((2, CAMERA_H, CAMERA_W), dtype=bool),
        AMPLITUDE_CENTER, initial_defocus=DEFOCUS_DIOPTER, learn_amplitude=False,
        learn_residual=False, fft_size=FFT_SIZE, pixel_samples=PIXEL_SAMPLES,
        basis_arrays=None, registration_shift_ccd_px=registration, **kwargs)


def test_fitted_registration_is_differentiable() -> None:
    import torch

    model = _fitter((0.4, -0.3), learn_registration=True)
    assert isinstance(model.registration_shift, torch.nn.Parameter)
    assert model.registration_shift.requires_grad
    prediction = model.prediction()
    prediction.square().mean().backward()
    grad = model.registration_shift.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert torch.any(grad != 0)
    # one parameter group, two scalars, ordered (dy, dx)
    groups = [g["params"] for g in model.optimizer.param_groups
              if any(p is model.registration_shift for p in g["params"])]
    assert len(groups) == 1


def test_registration_is_only_trainable_when_opted_in() -> None:
    """Default off stays off: a fixed shift is not a trainable parameter."""
    fixed = _fitter((0.4, -0.3))
    assert fixed.registration_shift.requires_grad is False
    # The tensor is the source of truth, so the exported value is float32-exact.
    assert fixed.registration_value == pytest.approx((0.4, -0.3), abs=1e-7)
    assert not any(p is fixed.registration_shift
                   for group in fixed.optimizer.param_groups for p in group["params"])
    assert _fitter().registration_shift is None
    assert _fitter().registration_value is None
    with pytest.raises(ValueError, match="needs registration_shift_ccd_px"):
        _fitter(None, learn_registration=True)


def test_enabled_fitter_matches_disabled_fitter_at_zero_shift() -> None:
    assert np.array_equal(_fitter().prediction().detach().numpy(),
                          _fitter((0.0, 0.0)).prediction().detach().numpy())


@pytest.mark.parametrize("shift", [None, (0.0, 0.0), (0.527, 1.762), (2.0, -3.0)])
def test_twin_guard_holds_with_registration(shift: Any) -> None:
    """The two forwards must still agree byte-for-byte once registration is on.

    Same coupling as ``test_two_forwards_agree_bit_for_bit``: the fitter applies
    no refit post-processing, so the frozen twin is built with identity gain,
    zero background and no CCD blur, and is handed the fitter's own recovered
    width and ``z4``.
    """
    import torch

    model = _fitter(shift)
    recovered = model.log_width.exp().detach()
    frozen = _forward(shift,
                      amplitude_width_xy=(float(recovered[0]), float(recovered[1])),
                      gain=1.0, background=0.0, sigma_ccd_px=0.0)
    model.z4 = torch.tensor(frozen.z4.detach().cpu().numpy(), dtype=torch.float32)
    with torch.no_grad():
        expected = frozen.predict(_phases()).numpy()
        actual = model.prediction().numpy()
    assert expected.shape == actual.shape == (2, CAMERA_H, CAMERA_W)
    assert np.array_equal(expected, actual), (
        f"max abs diff {np.abs(expected - actual).max():.3e}"
    )


# --------------------------------------------------------------- serialisation


def test_registration_round_trips_through_dict_and_json() -> None:
    shift = (0.5267430172174841, 1.7623859421066448)
    payload = json.loads(json.dumps(_params(shift).to_dict()))
    back = FrozenDefocusParameters.from_dicts(payload, RAMP_MATRIX, fft_size=FFT_SIZE,
                                             pixel_samples=PIXEL_SAMPLES, grid_size=SIZE)
    assert back.registration_shift_ccd_px == shift
    assert isinstance(back.registration_shift_ccd_px, tuple)
    # A list from JSON must not leak out as a list.
    assert _params([0.25, -0.75]).registration_shift_ccd_px == (0.25, -0.75)


def test_load_frozen_parameters_round_trip(tmp_path: Path) -> None:
    shift = (0.527, 1.762)
    checkpoint = {**_params(shift).to_dict(), "ramp_matrix": [list(r) for r in RAMP_MATRIX],
                  "grid_size": SIZE, "fft_size": FFT_SIZE, "pixel_samples": PIXEL_SAMPLES}
    path = tmp_path / "frozen_parameters.json"
    path.write_text(json.dumps(checkpoint), encoding="utf-8")
    loaded = load_frozen_parameters(tmp_path)
    assert loaded.registration_shift_ccd_px == shift
    assert np.array_equal(_predict(shift), _predict(loaded.registration_shift_ccd_px))
    # An older checkpoint without the key loads as disabled.
    legacy = {key: value for key, value in checkpoint.items()
              if key != "registration_shift_ccd_px"}
    (tmp_path / "frozen_parameters.json").write_text(json.dumps(legacy), encoding="utf-8")
    assert load_frozen_parameters(tmp_path).registration_shift_ccd_px is None
