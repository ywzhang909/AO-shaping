"""Offline tests for the ``roi_pib`` objective (maximise brightness in the target ROI).

``roi_pib`` is the bucket-ratio ("PIB") evaluated over the **target-shaped** ROI
instead of a radius bucket: ``score = sum(I[roi]) / sum(I)``. These tests pin the
two properties that make it usable on hardware:

* it rewards light *inside* the target and reports the exact energy fraction;
* it is exposure / laser-drift invariant (a pure ratio), so a run is comparable
  across brightness changes;
* an empty target box scores 0 - it cannot be "won" by emptying the box the way a
  ``-CV``-only objective can.
"""

from __future__ import annotations

import numpy as np

from ao_shaping.optimizer.wfless.slm_zernike_pib import (
    roi_energy_loss,
    roi_pib_metric,
    target_shape_roi,
)


def test_roi_pib_is_the_in_roi_energy_fraction() -> None:
    image = np.zeros((64, 80), dtype=np.float64)
    image[30:34, 38:42] = 1.0
    center = (40.0, 32.0)
    roi = target_shape_roi((64, 80), center, "rectangle", 10, 4 / 3)

    score, energy = roi_pib_metric(image, center, "rectangle", 10, 4 / 3)

    assert np.isclose(energy, float(image[roi].sum()) / float(image.sum()))
    assert score == energy
    assert 0.0 < score <= 1.0


def test_roi_pib_rewards_light_inside_the_target() -> None:
    center = (40.0, 32.0)
    inside = np.zeros((64, 80), dtype=np.float64)
    inside[30:34, 38:42] = 1.0
    outside = np.zeros((64, 80), dtype=np.float64)
    outside[5:9, 5:9] = 1.0

    s_in, _ = roi_pib_metric(inside, center, "rectangle", 10, 4 / 3)
    s_out, _ = roi_pib_metric(outside, center, "rectangle", 10, 4 / 3)

    assert s_in > s_out


def test_roi_pib_is_invariant_to_global_brightness() -> None:
    center = (40.0, 32.0)
    image = np.zeros((64, 80), dtype=np.float64)
    image[30:34, 38:42] = 1.0

    base, _ = roi_pib_metric(image, center, "rectangle", 10, 4 / 3)
    scaled, _ = roi_pib_metric(image * 4.0, center, "rectangle", 10, 4 / 3)

    assert np.isclose(base, scaled)


def test_roi_pib_empty_target_box_scores_zero() -> None:
    image = np.zeros((64, 80), dtype=np.float64)
    image[5:9, 5:9] = 1.0

    score, energy = roi_pib_metric(image, (60.0, 50.0), "square", 4)

    assert score == 0.0
    assert energy == 0.0


def test_roi_pib_all_light_inside_scores_one() -> None:
    image = np.zeros((64, 80), dtype=np.float64)
    image[30:34, 38:42] = 1.0

    score, energy = roi_pib_metric(image, (40.0, 32.0), "square", 8)

    assert np.isclose(score, 1.0)
    assert np.isclose(energy, 1.0)


def test_roi_pib_handles_zero_total_intensity() -> None:
    score, energy = roi_pib_metric(np.zeros((32, 32)), (16.0, 16.0))

    assert score == 0.0
    assert energy == 0.0


def test_roi_pib_rejects_non_2d_input() -> None:
    try:
        roi_pib_metric(np.zeros((4, 4, 3)), (2.0, 2.0))
    except ValueError:
        return
    raise AssertionError("3D input must raise ValueError")


# --------------------------------------------------------------------------- #
# Safety guard: in-ROI energy loss (default limit 0.6 -> abandon)
# --------------------------------------------------------------------------- #


def test_roi_energy_loss_is_fractional() -> None:
    assert roi_energy_loss(0.5, 0.5) == 0.0
    assert np.isclose(roi_energy_loss(0.5, 0.2), 0.6)
    assert np.isclose(roi_energy_loss(0.5, 0.4), 0.2)
    assert roi_energy_loss(0.5, 0.7) < 0.0


def test_roi_energy_loss_without_protection() -> None:
    assert roi_energy_loss(0.0, 0.0) == 0.0
    assert roi_energy_loss(-1.0, 0.1) == 0.0
    assert roi_energy_loss(float("nan"), 0.1) == 0.0


def test_roi_energy_loss_default_limit_semantics() -> None:
    """Default 0.6 limit: a >60% in-ROI energy drop is abandoned, 50% is allowed."""
    limit = 0.6
    assert roi_energy_loss(0.50, 0.19) > limit
    assert roi_energy_loss(0.50, 0.25) <= limit
