"""Cold initialization uses measured cross-axis geometry, not historical axes."""
import json

import numpy as np
import pytest

from scripts.slm.bootstrap_transfer_checkpoint import bootstrap


def dataset(tmp_path):
    matrix = np.array([[94., 7468.], [-7537., 122.]])
    rows = [dict(label="reference_start", spot_xy=[1200., 900.], peak=100, exposure_ms=1)]
    for index, axis in enumerate(("x", "y")):
        for sign in (-1, 1):
            rows.append(dict(label=f"ramp_{axis}_64_{sign}", peak=100, exposure_ms=1,
                             spot_xy=(np.array([1200., 900.])+sign*matrix[:, index]/64).tolist()))
    (tmp_path / "manifest.json").write_text(json.dumps(rows))
    (tmp_path / "config.json").write_text(json.dumps(dict(grid=128)))
    return matrix


def test_bootstrap_preserves_measured_geometry_and_marks_nominal(tmp_path):
    expected = dataset(tmp_path)
    params, provenance = bootstrap(tmp_path, center=(700., 800.), defocus=-.06)
    np.testing.assert_allclose(params["ramp_matrix"], expected)
    assert params["origin"] == [1200., 900.]
    assert params["amplitude_center"] == [700., 800.]
    assert params["defocus_diopter"] == -.06
    assert params["residual_noll_radians"] == {str(j): 0 for j in range(5, 12)}
    assert "not fitted" in provenance["status"]


def test_bootstrap_rejects_nonphysical_width(tmp_path):
    dataset(tmp_path)
    with pytest.raises(ValueError, match="positive"):
        bootstrap(tmp_path, width=(0., 100.))


def test_blur_calibration_accepts_three_shapes_and_requires_validation():
    from scripts.slm.calibrate_transfer_blur import available_seed_shapes

    rows = [dict(label=s + "_seed") for s in ("square", "rectangle", "triangle")]
    assert available_seed_shapes(rows) == ["square", "rectangle", "triangle"]
    with pytest.raises(ValueError, match="validation"):
        available_seed_shapes([rows[0], rows[2]])
