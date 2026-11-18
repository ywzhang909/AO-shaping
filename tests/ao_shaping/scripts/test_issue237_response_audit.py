"""Narrow tests for the pure helpers in the issue #237 response audit script.

No capture data and no devices: these cover the arithmetic that the audit's
conclusions rest on (weighted fits, held-out errors, design collinearity,
endpoint shape, fixed ROI, global clipping masks).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "issue237_response_audit",
    _ROOT / "scripts/slm/issue237_response_audit_20261118.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

Condition = _MODULE.Condition


def test_weighted_polyfit_recovers_exact_affine() -> None:
    t = np.array([5.0, 8.0, 12.0, 16.0, 20.0])
    y = 0.5 + 0.0833 * t
    fit = _MODULE.weighted_polyfit(t, y, np.full_like(t, 0.01), 1)
    assert fit["coef"][0] == pytest.approx(0.5)
    assert fit["coef"][1] == pytest.approx(0.0833)
    assert max(abs(v) for v in fit["resid"]) < 1e-12
    assert fit["dof"] == 3


def test_weighted_polyfit_needs_more_points_than_parameters() -> None:
    with pytest.raises(ValueError):
        _MODULE.weighted_polyfit(np.array([5.0]), np.array([1.0]), np.ones(1), 1)


def test_leave_one_out_has_zero_error_on_exact_line() -> None:
    t = np.array([5.0, 8.0, 12.0, 16.0, 20.0])
    y = 1.0 - 0.02 * t
    loeo = _MODULE.leave_one_out(t, y, np.full_like(t, 0.5), 1)
    assert loeo["rmse"] == pytest.approx(0.0, abs=1e-9)
    assert loeo["max_abs_z"] == pytest.approx(0.0, abs=1e-9)
    assert [row["heldout_exposure_ms"] for row in loeo["per_point"]] == t.tolist()


def test_affine_loglog_slope_is_below_one_with_a_pedestal() -> None:
    slope = _MODULE.affine_loglog_slope(0.5, 0.0833, 5.0, 20.0)
    expected = np.log((0.5 + 0.0833 * 20.0) / (0.5 + 0.0833 * 5.0)) / np.log(4.0)
    assert slope == pytest.approx(expected)
    assert 0.5 < slope < 1.0


def test_design_diagnostics_quotes_the_5_to_20ms_design() -> None:
    diag = _MODULE.design_diagnostics(np.array([5.0, 8.0, 12.0, 16.0, 20.0]))
    assert diag["n_points"] == 5
    assert diag["t_halfspan_ms"] == pytest.approx(7.5)
    assert diag["intercept_extrapolation_halfspans"] == pytest.approx(12.2 / 7.5)
    assert diag["intercept_variance_inflation"] > 6.0
    assert diag["corr_offset_gain"] < -0.8


def test_endpoint_residuals_flag_a_quadratic_bulge() -> None:
    t = np.array([5.0, 8.0, 12.0, 16.0, 20.0])
    assert _MODULE.endpoint_residuals(t, 2.0 + 3.0 * t) == pytest.approx(
        [0.0] * 5, abs=1e-12
    )
    curved = (t - 12.0) ** 2
    resid = _MODULE.endpoint_residuals(t, curved)
    assert resid[0] == pytest.approx(0.0)
    assert resid[-1] == pytest.approx(0.0)
    assert resid[2] != pytest.approx(0.0)


def test_roi_rect_pads_and_clamps_to_the_image() -> None:
    image = np.zeros((40, 50), dtype=np.float32)
    image[10:20, 15:30] = 200.0
    assert _MODULE.roi_rect(image, 100.0, 3) == (7, 23, 12, 33)
    # A tiny pad on an isophote touching the corners must clamp, not wrap.
    assert _MODULE.roi_rect(image, 50.0, 0) == (10, 20, 15, 30)


def test_dynamic_range_census_counts_levels_strictly_inside_bounds() -> None:
    levels = np.array(
        [
            [21.0, 20.0, 249.0, 251.0],
            [21.0, 249.0, 250.0, 255.0],
        ]
    )
    census = _MODULE.dynamic_range_census(levels, 20.0, 250.0)
    assert census["usable_level_count_hist"] == {0: 1, 1: 2, 2: 1}
    assert census["fraction_with_at_least"]["1"] == pytest.approx(0.75)
    assert census["fraction_with_at_least"]["2"] == pytest.approx(0.25)
    assert census["fraction_with_at_least"]["3"] == pytest.approx(0.0)


def test_saturation_masks_use_every_raw_frame_globally() -> None:
    frames_a = np.zeros((2, 4, 5), dtype=np.uint8)
    frames_a[:, 2, 2] = 255  # clipped in every frame
    frames_a[0, 1, 1] = 255  # clipped once, never again
    frames_b = np.zeros((2, 4, 5), dtype=np.uint8)
    frames_b[:, 2, 2] = 255  # clipped in every frame of both conditions
    frames_b[1, 3, 4] = 255
    conditions = [
        Condition("blank", 5.0, "", "", frames_a),
        Condition("blank", 8.0, "", "", frames_b),
    ]
    ever, always = _MODULE.saturation_masks(conditions)
    assert ever.sum() == 3
    assert always.sum() == 1
    assert bool(always[2, 2])
    assert bool(ever[1, 1]) and bool(ever[3, 4])
