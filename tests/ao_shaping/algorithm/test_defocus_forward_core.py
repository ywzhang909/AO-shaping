"""R2/P5: the two physical forwards must stay numerically identical.

``PhysicalForwardFit`` (fitting) and ``FrozenDefocusForward`` (frozen
inference) are two implementations of the same physics: panel geometry,
Gaussian illumination, camera sampling grids and the FFT irradiance
scaling all appear twice. Issue #199 R2 extracts that shared core into
``defocus_forward_core``; this file pins that the extraction moves no
number:

* the twin test drives both forwards over identical geometry (the fitted
  ``z4`` is handed to the fitter through ``basis_arrays``) and requires
  **bit-identical** predictions -- the only comparison that is stable
  across platforms, because both paths run in the same process against
  the same library build;
* the golden pins the frozen forward's own post-processing (blaze
  broadening, gain, background) with the repo's shape + tolerance scheme
  (bit digests do not survive library-version changes, see
  ``test_differentiable_shaping_golden.py``);
* the reproducibility test catches nondeterminism.

The golden is regenerated with::

    AO_DEFOCUS_FORWARD_GOLDEN_UPDATE=1 pytest tests/ao_shaping/algorithm/test_defocus_forward_core.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    FrozenDefocusForward,
    FrozenDefocusParameters,
)
from ao_shaping.algorithm.signal_processing.defocus_forward_core import (
    PANEL_ILLUMINATION_CEILING_PX,
    PANEL_PITCH_PX,
    WAVELENGTH_M,
    diopter_scale,
    irradiance_scale,
)

GOLDEN = Path(__file__).with_name("defocus_forward_core_golden.json")
UPDATE = os.environ.get("AO_DEFOCUS_FORWARD_GOLDEN_UPDATE") == "1"

# --- shared synthetic geometry (no checkpoint needed: CI must run this) ---
SIZE = 32
FFT_SIZE = 128
PIXEL_SAMPLES = 2
CAMERA_H = CAMERA_W = 12
#: A real fitted panel -> camera map (see test_differentiable_defocus_uniformity).
RAMP_MATRIX = ((465.146, 7174.691), (-7533.176, -30.278))
ORIGIN_XY = (1360.0, 1020.0)
#: The fitter's illumination width is hardcoded as ``log(260)`` in its
#: ``__init__`` and recovered with ``exp()``, which is 1 ULP off 260 in
#: float32. The frozen forward is handed that exact recovered value (see
#: ``_frozen``): the twin compares the shared math, and the width is an
#: input to it, not part of it.
AMPLITUDE_WIDTH = (260.00006103515625, 260.00006103515625)
AMPLITUDE_CENTER = (960.0, 600.0)
DEFOCUS_DIOPTER = 0.35
GAIN = 1.7
BACKGROUND = 0.02
SIGMA_CCD_PX = 1.5


def _camera_xy() -> np.ndarray:
    xs = np.linspace(1330.0, 1390.0, CAMERA_W)
    ys = np.linspace(990.0, 1030.0, CAMERA_H)
    xx, yy = np.meshgrid(xs, ys)
    return np.stack([xx, yy], axis=-1)


def _phases() -> np.ndarray:
    rng = np.random.default_rng(20261025)
    return (rng.standard_normal((2, SIZE, SIZE)) * 0.4).astype(np.float32)


def _frozen(
    *,
    sigma: float = SIGMA_CCD_PX,
    gain: float = GAIN,
    background: float = BACKGROUND,
    amplitude_width: tuple[float, float] = AMPLITUDE_WIDTH,
) -> FrozenDefocusForward:
    params = FrozenDefocusParameters(
        defocus_diopter=DEFOCUS_DIOPTER,
        origin_xy=ORIGIN_XY,
        amplitude_center_xy=AMPLITUDE_CENTER,
        amplitude_width_xy=amplitude_width,
        residual_noll_radians={},
        gain=gain,
        background=background,
        sigma_ccd_px=sigma,
        ramp_matrix=RAMP_MATRIX,
        fft_size=FFT_SIZE,
        pixel_samples=PIXEL_SAMPLES,
        grid_size=SIZE,
    )
    return FrozenDefocusForward(params, _camera_xy(), device="cpu")


def _fitter() -> Any:
    """The fitting forward over the same geometry (its width is hardcoded)."""
    from ao_shaping.algorithm.signal_processing.physical_forward_fit import (
        PhysicalForwardFit,
    )

    images = np.zeros((2, CAMERA_H, CAMERA_W), dtype=np.float32)
    train = np.ones((2, CAMERA_H, CAMERA_W), dtype=bool)
    return PhysicalForwardFit(
        _phases(),
        images,
        _camera_xy(),
        RAMP_MATRIX,
        DEFOCUS_DIOPTER,
        train,
        AMPLITUDE_CENTER,
        initial_defocus=DEFOCUS_DIOPTER,
        learn_amplitude=False,
        learn_residual=False,
        fft_size=FFT_SIZE,
        pixel_samples=PIXEL_SAMPLES,
        basis_arrays=None,
    )


def test_two_forwards_agree_bit_for_bit() -> None:
    """The twin guard: identical geometry in, identical bytes out."""
    import torch

    fitter = _fitter()
    # Self-calibrate the frozen width to the fitter's exp(log(260)) recovery.
    recovered = fitter.log_width.exp().detach()
    width = (float(recovered[0]), float(recovered[1]))
    assert width[0] != 260.0, "exp(log(260)) round-trip changed; re-check the coupling"
    frozen = _frozen(sigma=0.0, gain=1.0, background=0.0, amplitude_width=width)
    # The fitter must use the very same z4 the frozen forward sampled.
    fitter.z4 = torch.tensor(frozen.z4.detach().cpu().numpy(), dtype=torch.float32)
    with torch.no_grad():
        expected = frozen.predict(_phases()).numpy()
        actual = fitter.prediction().numpy()
    assert expected.shape == actual.shape == (2, CAMERA_H, CAMERA_W)
    assert np.array_equal(expected, actual), (
        f"max abs diff {np.abs(expected - actual).max():.3e}"
    )


def _shape(array: np.ndarray) -> dict[str, Any]:
    finite = array[np.isfinite(array)]
    return {
        "shape": list(array.shape),
        "min": float(finite.min()) if finite.size else 0.0,
        "max": float(finite.max()) if finite.size else 0.0,
        "mean": float(finite.mean()) if finite.size else 0.0,
        "sum": float(finite.sum()) if finite.size else 0.0,
    }


def _golden_trace() -> dict[str, Any]:
    import torch

    forward = _frozen()
    with torch.no_grad():
        predicted = forward.predict(_phases()).numpy()
    return {
        "predicted": _shape(predicted),
        "blaze_kernel": _shape(forward._blaze_kernel.numpy()),
        "illumination": _shape(forward.illumination_weights()),
    }


def test_frozen_post_processing_matches_golden() -> None:
    trace = _golden_trace()
    if UPDATE:
        GOLDEN.write_text(
            json.dumps(trace, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        expected = trace
    else:
        assert GOLDEN.exists(), (
            f"missing golden {GOLDEN.name}; regenerate with "
            "AO_DEFOCUS_FORWARD_GOLDEN_UPDATE=1"
        )
        expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    for key in expected:
        got, want = trace[key], expected[key]
        assert got["shape"] == want["shape"], f"{key} shape moved: {got} != {want}"
        for field in ("min", "max", "mean", "sum"):
            assert abs(got[field] - want[field]) <= 1e-6 * max(1.0, abs(want[field])), (
                f"{key}.{field} moved: {got[field]} != {want[field]}"
            )


def test_frozen_predict_is_reproducible() -> None:
    import torch

    forward = _frozen()
    with torch.no_grad():
        first = forward.predict(_phases()).numpy()
        second = forward.predict(_phases()).numpy()
    assert np.array_equal(first, second)


def test_core_constants_and_scales() -> None:
    """The extracted constants must keep the values the two classes hardcoded."""
    assert PANEL_PITCH_PX == 1920.0
    assert PANEL_ILLUMINATION_CEILING_PX == 1200.0
    assert WAVELENGTH_M == 1064e-9
    # Z4 = sqrt(3)*(2r^2 - 1) over a 960 px half-span pupil at 8 um pitch.
    assert diopter_scale() == pytest.approx(
        float(np.pi * (960 * 8e-6) ** 2 / (2 * np.sqrt(3) * 1064e-9)), rel=0, abs=0.0
    )
    # Irradiance units are preserved when pitch or FFT size changes.
    assert irradiance_scale(15.0, 512) == 1.0
    # pitch x2 and fft x2 -> inner ratio (2**2 * 2) = 8, squared = 64.
    assert irradiance_scale(30.0, 1024) == pytest.approx(64.0)
