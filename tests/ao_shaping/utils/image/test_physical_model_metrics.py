import numpy as np
import pytest

from ao_shaping.utils.image.physical_model_metrics import (
    compute_directional_metrics,
    compute_forward_metrics,
)


def test_forward_metrics_separates_gain_offset_from_shape_error():
    predicted = np.arange(16, dtype=float).reshape(4, 4) + 1
    measured = 3 * predicted + 7
    core = np.ones_like(predicted, dtype=bool)
    metrics = compute_forward_metrics(predicted, measured, core, exposure_ms=2, full_scale=100)
    assert metrics["pearson_ncc"] > 0.999999
    assert metrics["relative_l2_affine_diagnostic"] < 1e-12
    assert metrics["cv_abs_error"] > 0.1  # Offset is invisible to NCC, but changes CV.
    assert metrics["radiometry_gain"] == pytest.approx(1.5)
    assert metrics["radiometry_offset"] == pytest.approx(3.5)


def test_zero_order_exclusion_is_explicit_and_can_change_cv():
    predicted = np.ones((5, 5), dtype=float)
    measured = predicted.copy()
    measured[2, 2] = 100
    core = np.ones_like(predicted, dtype=bool)
    excluded = np.zeros_like(core)
    excluded[2, 2] = True
    all_pixels = compute_forward_metrics(predicted, measured, core)
    without_zero = compute_forward_metrics(predicted, measured, core, zero_order_exclusion=excluded)
    assert all_pixels["cv_measured"] > 0
    assert without_zero["cv_measured"] == pytest.approx(0.0)
    assert core.all()  # Metrics must never shrink a shared mask in place.


def test_directional_metrics_detect_wrong_gradient_direction():
    core = np.ones((2, 2), dtype=bool)
    plus = np.array([[2., 3.], [4., 5.]])
    minus = np.ones((2, 2))
    measured_plus = minus - (plus - minus)
    measured_minus = minus
    metrics = compute_directional_metrics(plus, minus, measured_plus, measured_minus, core)
    assert metrics["direction_correlation"] < -0.999999
    assert metrics["direction_sign_agreement"] == 0.0


def test_invalid_exposure_and_empty_core_fail_fast():
    image = np.ones((2, 2))
    with pytest.raises(ValueError, match="exposure_ms"):
        compute_forward_metrics(image, image, np.ones_like(image, bool), exposure_ms=0)
    with pytest.raises(ValueError, match="non-empty"):
        compute_forward_metrics(image, image, np.zeros_like(image, bool))


def test_negative_gain_refits_intercept_and_no_response_is_undefined():
    image = np.arange(4.).reshape(2, 2) + 1
    core = np.ones_like(image, bool)
    metrics = compute_forward_metrics(image, 5 - image, core)
    assert metrics["radiometry_gain"] == 0
    assert metrics["radiometry_offset"] == pytest.approx(2.5)
    zero = np.zeros_like(image)
    metrics = compute_forward_metrics(zero, zero, core)
    assert metrics["cv_measured"] is None
    assert metrics["pearson_ncc"] is None
    direction = compute_directional_metrics(image, image, image, image, core)
    assert direction["direction_cosine"] is None
    assert direction["direction_sign_agreement"] is None


def test_sparse_grid_centroid_uses_native_ccd_pixels():
    predicted = np.array([[1., 0.], [0., 0.]])
    measured = np.array([[0., 1.], [0., 0.]])
    yy, xx = np.indices(predicted.shape)
    xy = np.stack((100 + 4 * xx, 200 + 4 * yy), axis=-1)
    result = compute_forward_metrics(predicted, measured, np.ones_like(predicted, bool), camera_xy=xy)
    assert result["centroid_error_px"] == pytest.approx(4)
    assert result["centroid_coordinate_unit"] == "native_ccd_px"


def test_nonfinite_images_and_integer_masks_are_rejected():
    image = np.ones((2, 2))
    with pytest.raises(ValueError, match="finite"):
        compute_forward_metrics(image * np.nan, image, np.ones_like(image, bool))
    with pytest.raises(ValueError, match="non-empty"):
        compute_directional_metrics(image, image, image, image, np.ones_like(image, int))
