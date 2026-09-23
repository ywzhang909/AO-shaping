"""Offline tests for the ``rmse`` objective (minimise the normalised-image RMSE).

``rmse`` is the non-PIB beam-shaping objective: the captured frame and the
target shape are each normalised to unit sum (``sum(img) = sum(target) = 1.0``),
making the comparison exposure / laser-drift invariant, and the pixelwise RMSE
between the two normalised maps is **minimised**. Driving it to zero pushes the
beam to a flat, uniform intensity inside the target shape and zero elsewhere.

These tests pin the properties that make it usable on hardware:

* a perfect uniform fill inside the target scores exactly 0;
* it is invariant to global brightness / exposure changes;
* a concentrated spot scores worse than the same energy spread uniformly
  (the minimiser is steered toward flat fill, not toward dumping energy);
* light outside the target increases the error;
* a dark / un-normalisable frame returns a strong penalty, never NaN.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from click.testing import CliRunner

from ao_shaping.optimizer.wfless.slm_zernike_pib import (
    optimize_slm_zernike_pib,
    rmse_shape_metric,
    target_shape_roi,
)
from ao_shaping.runners.slm_pib_runner import _effective_objective_key, run


def test_rmse_is_named_tuple_of_score_and_energy() -> None:
    image = np.zeros((64, 80), dtype=np.float64)
    image[30:34, 38:42] = 1.0

    rmse, energy = rmse_shape_metric(image, (40.0, 32.0), "rectangle", 10, 4 / 3)

    assert rmse >= 0.0
    assert 0.0 <= energy <= 1.0
    # Light fully inside the ROI -> in-ROI energy fraction is 1.
    assert np.isclose(energy, float(image[target_shape_roi((64, 80), (40.0, 32.0), "rectangle", 10, 4 / 3)].sum()) / image.sum())


def test_rmse_is_zero_for_perfect_uniform_fill() -> None:
    center = (40.0, 32.0)
    roi = target_shape_roi((64, 80), center, "rectangle", 10, 4 / 3)
    image = np.zeros((64, 80), dtype=np.float64)
    # Any uniform intensity inside the box, none outside: after normalisation
    # the frame equals the normalised target exactly.
    image[roi] = 3.0

    rmse, energy = rmse_shape_metric(image, center, "rectangle", 10, 4 / 3)

    assert rmse == 0.0
    assert np.isclose(energy, 1.0)


def test_rmse_is_invariant_to_global_brightness() -> None:
    center = (40.0, 32.0)
    image = np.zeros((64, 80), dtype=np.float64)
    image[30:34, 38:42] = 1.0

    base, _ = rmse_shape_metric(image, center, "rectangle", 10, 4 / 3)
    scaled, _ = rmse_shape_metric(image * 4.0, center, "rectangle", 10, 4 / 3)

    assert np.isclose(base, scaled)


def test_rmse_orders_fills_by_uniformity() -> None:
    # Minimising the RMSE must rank: uniform fill < slightly-apertured fill <
    # concentrated spot — i.e. it steers toward flat fill, never toward dumping
    # energy, even though all three layouts keep the same in-ROI energy.
    center = (40.0, 32.0)
    shape = ("rectangle", 10, 4 / 3)
    roi = target_shape_roi((64, 80), center, *shape)
    yy, xx = np.nonzero(roi)

    uniform = np.zeros((64, 80), dtype=np.float64)
    uniform[roi] = 1.0
    perturbed = uniform.copy()
    perturbed[yy[0], xx[0]] = 0.0  # one hole in an otherwise perfect fill
    spot = np.zeros_like(uniform)
    spot[yy[len(yy) // 2], xx[len(xx) // 2]] = float(uniform.sum())

    rmse_u, e_u = rmse_shape_metric(uniform, center, *shape)
    rmse_p, e_p = rmse_shape_metric(perturbed, center, *shape)
    rmse_s, e_s = rmse_shape_metric(spot, center, *shape)

    assert rmse_u == 0.0
    assert e_u == 1.0 and e_p == 1.0 and e_s == 1.0  # all light inside the ROI
    assert rmse_u < rmse_p < rmse_s


def test_rmse_penalizes_light_outside_the_target() -> None:
    center = (40.0, 32.0)
    shape = ("rectangle", 10, 4 / 3)
    roi = target_shape_roi((64, 80), center, *shape)

    # Build the patterns FROM the ROI mask (pixel-exact) rather than guessing
    # placement: the same total energy either fills the target box or sits
    # shifted off it.
    inside = roi.astype(np.float64)
    outside = np.roll(roi, 20, axis=(0, 1)).astype(np.float64)  # no wrap (interior)

    rmse_in, e_in = rmse_shape_metric(inside, center, *shape)
    rmse_out, e_out = rmse_shape_metric(outside, center, *shape)

    assert np.isclose(e_in, 1.0)
    assert rmse_in == 0.0
    assert e_out < e_in
    assert rmse_out > rmse_in


def test_rmse_handles_zero_total_intensity() -> None:
    rmse, energy = rmse_shape_metric(np.zeros((32, 32)), (16.0, 16.0))

    assert rmse == 1e3
    assert energy == 0.0


def test_rmse_handles_nan_frames() -> None:
    image = np.ones((32, 32), dtype=np.float64)
    image[0, 0] = np.nan

    rmse, energy = rmse_shape_metric(image, (16.0, 16.0))

    assert rmse == 1e3
    assert energy == 0.0


def test_rmse_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError, match="img must be 2D"):
        rmse_shape_metric(np.zeros((4, 4, 3)), (2.0, 2.0))


def test_rmse_accepts_target_shape_in_validation() -> None:
    # ``rmse`` is a target-shape objective: supplying target_shape must NOT raise
    # the "target_shape can only be used" ValueError that guards the other
    # objectives (i.e. it is not remapped away to ``shape``). The validation
    # runs before any hardware is opened, so reaching hardware/open errors
    # (anything except that specific ValueError) is the pass condition.
    try:
        optimize_slm_zernike_pib(
            center="shape",
            epochs=1,
            objective="rmse",
            target_shape="circle",
        )
    except ValueError as exc:
        assert "target_shape can only be used" not in str(exc)
    except Exception:
        pass  # hardware / camera errors - validation already accepted the args


def test_effective_objective_key_keeps_rmse() -> None:
    # The runner-side remap must keep ``rmse`` (like roi_pib/rms_pib) instead of
    # folding it into ``shape``, so debug artifacts record the right key.
    assert _effective_objective_key("rmse", "circle") == "rmse"
    assert _effective_objective_key("rmse", None) == "rmse"
    assert _effective_objective_key("roi_pib", "circle") == "roi_pib"
    assert _effective_objective_key("shape", "circle") == "shape"
    assert _effective_objective_key("pib", "circle") == "shape"  # legacy remap kept


def test_slm_pib_cli_exposes_rmse_objective() -> None:
    result = CliRunner().invoke(run, ["spgd", "--help"])
    assert result.exit_code == 0, result.output

    match = re.search(r"--objective\s+\[([^\]]+)\]", result.output)
    assert match, result.output
    cli_choices = {c.strip() for c in match.group(1).split("|")}
    assert "rmse" in cli_choices