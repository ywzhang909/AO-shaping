"""Fixed-phase exposure fits should reject underdetermined camera curves."""

import numpy as np
import pytest

from scripts.analyze_daheng_exposure import pixelwise_response_diagnostic, response_fits


def test_affine_response_recovers_offset_without_claiming_dark_frame() -> None:
    exposure = np.array([0.02, 0.04, 0.08, 0.12, 0.08, 0.04, 0.02])
    signal = 10000.0 + 300000.0 * exposure
    fit = response_fits(exposure, signal)
    assert fit["affine"]["relative_rms"] < 1e-12
    assert fit["affine"]["intercept_over_slope_ms"] == pytest.approx(1 / 30)
    assert fit["double_exponential"]["status"].startswith("not fitted")
    assert fit["logistic"]["status"].startswith("not fitted")


def test_exposure_fit_needs_distinct_levels() -> None:
    with pytest.raises(ValueError, match="three distinct"):
        response_fits(np.array([0.02, 0.02, 0.04, 0.04]),
                      np.array([2.0, 2.0, 4.0, 4.0]))


def test_pixelwise_diagnostic_distinguishes_scale_from_shape_change() -> None:
    base = np.linspace(30, 150, 400).reshape(20, 20)
    scaled = base * 0.5
    changed = scaled.copy()
    changed[:, :10] *= 1.2
    rows = pixelwise_response_diagnostic(
        np.array([0.1, 0.2, 0.4]), [scaled, changed, base]
    )
    assert rows[0]["relative_shape_rmse_after_scalar"] < 1e-12
    assert rows[1]["relative_shape_rmse_after_scalar"] > 0.04
    assert rows[2]["median_count_ratio_to_reference"] == pytest.approx(1)
