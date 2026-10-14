from __future__ import annotations

import pickle

import numpy as np
import pytest

from scripts.slm.compare_gs_refine_forward import compare_run, image_metrics, sample_camera


def test_camera_sampling_uses_fixed_optical_axis_and_pixel_scale() -> None:
    far = np.arange(81, dtype=float).reshape(9, 9)
    sampled = sample_camera(
        far,
        far_pixel_m=2.0,
        camera_pixel_m=1.0,
        camera_shape=(3, 3),
        center_xy=(1, 1),
    )
    assert sampled[1, 1] == far[4, 4]
    assert sampled[1, 2] == pytest.approx((far[4, 4] + far[4, 5]) / 2)


def test_identical_images_have_unit_overlap_and_matching_roi_metrics() -> None:
    image = np.ones((8, 8), dtype=float)
    image[3:5, 3:5] = 4
    result = image_metrics(image, image, center_xy=(4, 4), side=4)
    assert result["pearson"] == pytest.approx(1)
    assert result["intensity_overlap"] == pytest.approx(1)
    assert result["predicted_cv"] == pytest.approx(result["measured_cv"])
    with pytest.raises(ValueError, match="leaves camera frame"):
        image_metrics(image, image, center_xy=(0, 0), side=4)


def test_compare_run_ignores_its_own_output_sidecar(tmp_path) -> None:
    (tmp_path / "run.pkl").write_bytes(pickle.dumps({}))
    (tmp_path / "run.json").write_text("{}", encoding="utf-8")
    (tmp_path / "forward_diagnostic.json").write_text("{}", encoding="utf-8")
    with pytest.raises(KeyError, match="beam_radius_px"):
        compare_run(tmp_path, pupil_center_xy=(2, 2), keys=())
