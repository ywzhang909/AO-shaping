"""Scoring contracts for the holdout-ceiling analysis.

The run's holdout gate scores the forward model over the whole model grid, but the
camera only ever observes a crop of it. Everything here is offline: no run
directory, no device, no writes -- synthetic frames pushed through the very same
mapping the script uses, plus the pure scoring helpers.

The recorded bench geometry is reproduced exactly below, because the numbers in the
report (288 observed cells out of 1024, i.e. 7.91%) are the invariant that matters:
they are what makes a recorded 0.11 score a statement about observability rather
than about model quality.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from ao_shaping.optimizer.wfless.slm_model_in_loop import (
    _prediction_correlation,
    _to_model_grid,
)
from ao_shaping.utils.wavefront.zernike_calc import ZernikeGenerator
from scripts.slm.analyze_holdout_ceiling import (
    _bandwidth,
    _family_of,
    _family_rows,
    _remove_flat,
    _support_mask,
)

# The recorded run: a 300 px camera window resampled onto a 1024-cell far-field
# grid with camera_px_per_model_px = 1.0467 and one quarter turn for the camera
# mount. _to_model_grid crops span = round(1024 * 1.0467) = 1072 camera pixels
# around the frozen optical axis before resizing, so the window lands on 288 of the
# 1024 cells -- 7.91% of the grid.
CAMERA_PX = 300
CELL_SCALE = 1.0467
FAR_FIELD = 1024
ROTATION = 1


def _frame_at_peak(frame: np.ndarray, value: float) -> np.ndarray:
    """Copy ``frame`` with its brightest pixel pinned to exactly ``value``."""
    pinned = np.array(frame, dtype=np.float64, copy=True)
    row, col = np.unravel_index(int(np.argmax(pinned)), pinned.shape)
    pinned[row, col] = value
    return pinned


def _phase_families(n: int = 128) -> dict[str, np.ndarray]:
    """The three probe families the recorded run contains, from one fixed seed.

    Full-band noise, the very same draw smoothed past sigma = 2, and a low-order
    Zernike sum in radians over the aperture.
    """
    rng = np.random.default_rng(20261019)
    draw = rng.standard_normal((n, n))
    generator = ZernikeGenerator(resolution=(n, n), radius=n / 2, n_orders=6)
    coefficients = rng.standard_normal(generator._cart.nk) * 0.5
    zernike = np.nan_to_num(generator.generate_noll(coefficients), nan=0.0)
    return {
        "freeform": draw,
        "smooth": gaussian_filter(draw, 2.5),
        "zernike": zernike,
    }


def _non_json_scalars(value: Any, path: str = "payload") -> list[str]:
    """Paths of every leaf ``json`` would not round-trip with its type intact."""
    if isinstance(value, dict):
        return [found for key, item in value.items() for found in _non_json_scalars(item, f"{path}.{key}")]
    if isinstance(value, (list, tuple)):
        return [found for index, item in enumerate(value) for found in _non_json_scalars(item, f"{path}[{index}]")]
    if isinstance(value, np.generic):
        return [f"{path} (numpy {type(value).__name__})"]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return []
    return [f"{path} ({type(value).__name__})"]


def test_camera_footprint_is_a_ones_frame_not_the_nonzero_pixels() -> None:
    """The footprint has to come from the mapping, never from the scene.

    "Which pixels of the mapped grid are nonzero?" is not a camera-footprint
    question. The forward model's far field carries signal over the whole grid, so
    that rule answers ~100% there; the same rule collapses to nothing as soon as a
    frame has a dark corner. The footprint instead pushes an all-ones frame of the
    camera's own shape through the identical mapping, so it inherits the crop, the
    interpolation and the rotation and depends on the scene not at all: 288 cells
    of 1024, 7.91%.
    """
    center = (CAMERA_PX // 2, CAMERA_PX // 2)
    shape = (FAR_FIELD, FAR_FIELD)
    frame = np.ones((CAMERA_PX, CAMERA_PX))

    support = _support_mask(frame, center, FAR_FIELD, CELL_SCALE, ROTATION, shape)
    assert np.sqrt(support.sum()) == pytest.approx(288.0, abs=0.5)
    assert support.mean() == pytest.approx(0.0791, abs=0.002)
    assert support.any()
    assert not support.all()

    # The false alarm: on a grid the model works on, a speckle prediction sits on a
    # nonzero pedestal everywhere, so "nonzero" claims the whole grid.
    rng = np.random.default_rng(4)
    prediction = gaussian_filter(rng.random(shape), 3.0) + 1e-3
    naive = prediction != 0
    assert naive.mean() > 0.99
    assert naive.sum() > 10 * support.sum()

    # Mapped from a real camera window the naive rule can only ever under-report:
    # the resampling is a non-negative linear operator, so a nonzero output has a
    # nonzero input inside its own support. It is right by luck when the scene
    # fills the window, and it follows the scene when it does not.
    mapped = _to_model_grid(frame, center, FAR_FIELD, CELL_SCALE, ROTATION)
    assert not np.any((mapped != 0) & ~support)

    black = np.zeros((CAMERA_PX, CAMERA_PX))
    assert not (_to_model_grid(black, center, FAR_FIELD, CELL_SCALE, ROTATION) != 0).any()
    assert _support_mask(black, center, FAR_FIELD, CELL_SCALE, ROTATION, shape).sum() == support.sum()


@pytest.mark.parametrize(
    ("camera_px", "cell_scale", "far_field"),
    [
        (300, 1.0467, 1024),  # the recorded bench
        (248, 1.11, 1024),
        (320, 1.85, 4096),
        (256, 1.0, 1024),
        (300, 2.0, 1024),  # integer scale: block average instead of interpolation
        (64, 0.5, 512),  # camera undersampling the model grid
    ],
)
def test_support_area_follows_the_camera_window(camera_px: int, cell_scale: float, far_field: int) -> None:
    """A window of ``n`` camera px over ``span`` camera px of grid covers ``(n/span)**2``.

    The bound is deliberately loose: the resize adds a one-cell halo around the
    block, and the halo is what makes the naive nonzero mask agree with the
    footprint instead of being the other way round.
    """
    span = round(far_field * cell_scale)
    support = _support_mask(
        np.ones((camera_px, camera_px)),
        (camera_px // 2, camera_px // 2),
        far_field,
        cell_scale,
        ROTATION,
        (far_field, far_field),
    )
    assert support.any()
    assert not support.all()
    assert support.mean() == pytest.approx((camera_px / span) ** 2, rel=0.10)


def test_support_mask_rejects_a_grid_shape_the_mapping_never_returned() -> None:
    """It is handed the shape a real mapping returned, so a mismatch is a bug."""
    with pytest.raises(ValueError, match="does not match the mapped grid"):
        _support_mask(np.ones((CAMERA_PX, CAMERA_PX)), (150, 150), FAR_FIELD, CELL_SCALE, ROTATION, (1024, 512))


def test_bandwidth_classifies_the_three_probe_families() -> None:
    """The manifest does not name a family, so it is recovered from the spectrum.

    The thresholds (0.3 on the high-frequency fraction, 0.4 on the gradient rms) sit
    in empty gaps, not on the measured values: the recorded run gives 0.004-0.030
    for the smooth and Zernike probes against 0.805 for the full-band ones, and
    0.10-0.20 against 0.72-1.09 for the Zernike pair.
    """
    families = _phase_families()

    measured = {name: _bandwidth(phase) for name, phase in families.items()}
    assert all(isinstance(value, float) for pair in measured.values() for value in pair)
    assert {name: _family_of(phase) for name, phase in families.items()} == {
        "freeform": "freeform",
        "smooth": "smooth",
        "zernike": "zernike",
    }

    high = {name: pair[0] for name, pair in measured.items()}
    gradient = {name: pair[1] for name, pair in measured.items()}

    # The 0.3 threshold sits in a gap an order of magnitude wide on both sides.
    assert max(high["smooth"], high["zernike"]) < 0.3 / 10
    assert high["freeform"] > 0.3 * 2

    # The 0.4 gradient threshold likewise separates the two low-bandwidth families.
    assert gradient["smooth"] < 0.4 / 2
    assert gradient["zernike"] > 0.4


def test_remove_flat_strips_the_pedestal_and_leaves_the_speckle() -> None:
    """A probe whose light scatters out of the zeroth order leaves the flat field behind."""
    rng = np.random.default_rng(3)
    speckle = rng.random((96, 96)) * 100.0
    flat = rng.random((96, 96)) * 50.0 + 3.0
    measured = speckle + 0.7 * flat

    stripped = _remove_flat(measured, flat)
    np.testing.assert_allclose(stripped, speckle, atol=1.0)
    assert abs(_prediction_correlation(stripped, flat)) < 1e-9
    # The pedestal really was there: the input correlates with the flat reference.
    assert _prediction_correlation(measured, flat) > 0.1


def test_remove_flat_leaves_a_pedestal_free_measurement_alone() -> None:
    """The removal is an unconditional least-squares projection, not a conditional one.

    With no pedestal in the measurement there is nothing to strip, but the code
    still subtracts the best multiple of the reference, which removes that single
    direction from the speckle. The contract is therefore "the speckle survives",
    not a bit-exact identity: the perturbation stays under 1% of the frame's range
    and leaves the correlation with the speckle above 0.999.
    """
    rng = np.random.default_rng(8)
    speckle = rng.random((96, 96)) * 100.0
    flat = rng.random((96, 96)) * 50.0 + 3.0

    stripped = _remove_flat(speckle, flat)
    deviation = float(np.abs(stripped - speckle).max())
    span = float(speckle.max() - speckle.min())
    assert deviation < 0.01 * span
    assert _prediction_correlation(stripped, speckle) > 0.999


def test_remove_flat_survives_a_constant_reference() -> None:
    """Zero variance in the reference must return the input, not divide by zero."""
    rng = np.random.default_rng(12)
    measured = rng.random((32, 32)) * 100.0

    stripped = _remove_flat(measured, np.full((32, 32), 12.5))
    np.testing.assert_array_equal(stripped, measured)


def test_ceiling_is_one_only_when_the_camera_saw_the_whole_grid() -> None:
    """``corr(prediction, prediction * support)`` is the perfect-model ceiling.

    The crop is applied to the measurement side, so a perfect model has to carry the
    same crop. That is 1.0 only when the support is the whole grid, and strictly
    below it as soon as modelled energy falls outside what the camera observed.
    """
    rng = np.random.default_rng(21)
    prediction = gaussian_filter(rng.random((64, 64)), 2.0) + 1e-3
    everything = np.ones((64, 64), dtype=bool)

    assert _prediction_correlation(prediction, prediction * everything) == pytest.approx(1.0)

    partial = rng.random((64, 64)) < 0.1
    ceiling = _prediction_correlation(prediction, prediction * partial)
    assert 0.0 < ceiling < 1.0

    # The stricter the crop, the lower the ceiling.
    tighter = rng.random((64, 64)) < 0.01
    assert _prediction_correlation(prediction, prediction * tighter) < ceiling


def test_emitted_records_survive_a_json_round_trip() -> None:
    """The payload is written with ``json.dumps``, so no numpy scalars may leak.

    This bug class has already bitten this repository once: a numpy scalar survives
    ``json.dumps`` and comes back as a plain int/float, which silently changes the
    type a downstream consumer sees.
    """
    probes = [
        {
            "file": "probe_000.npz",
            "split": "holdout",
            "family": "smooth",
            "high_frequency_fraction": 0.0082,
            "gradient_rms": 0.0207,
            "exposure_ms": 20.0,
            "recorded_correlation": 0.788,
            "corr_full": 0.781,
            "corr_support": 0.767,
            "ceiling": 1.0,
            "energy_fraction_in_support": 0.9952,
            "flux_ratio_to_flat": 0.87,
            "flat_correlation": 0.12,
            "corr_support_flat_removed": 0.771,
        },
        {
            "file": "probe_001.npz",
            "split": "holdout",
            "family": "freeform",
            "high_frequency_fraction": 0.8055,
            "gradient_rms": 0.4842,
            "exposure_ms": 20.0,
            "recorded_correlation": 0.113,
            "corr_full": 0.154,
            "corr_support": 0.154,
            "ceiling": 0.477,
            "energy_fraction_in_support": 0.1068,
            "flux_ratio_to_flat": 0.072,
            "flat_correlation": 0.03,
            "corr_support_flat_removed": 0.161,
        },
    ]
    result = {
        "run_dir": "synthetic",
        "unknown_config_fields": ["renamed_field"],
        "roi_center": [150.0, 150.0],
        "cell_scale": CELL_SCALE,
        "camera_rotation_quarters": ROTATION,
        "grid": {
            "pupil_region": 128,
            "far_field_size": FAR_FIELD,
            "mapped_shape": [FAR_FIELD, FAR_FIELD],
            "cam_size": CAMERA_PX,
            "camera_px_per_model_px": CELL_SCALE,
            "support_model_px": 288.0,
            "support_area_pct": 7.91,
        },
        "gate": 0.5,
        "probes": probes,
        "holdout_null": {"smooth": {"pairs": 3, "mean": 0.11, "max": 0.19}},
    }

    payload = {"families": _family_rows(result), **result}
    assert _non_json_scalars(payload) == []
    assert json.loads(json.dumps(payload, indent=2, ensure_ascii=False)) == payload