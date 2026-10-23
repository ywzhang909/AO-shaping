"""Geometry and split validation for recorded physical-model transfer."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/slm/fit_transfer_multishape.py"
spec = importlib.util.spec_from_file_location("fit_transfer_multishape", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_geometry_recovers_cross_axis_mapping_and_sign():
    expected = np.array([[20., 7400.], [-7600., -50.]])
    rows = []
    for column, axis in enumerate(("x", "y")):
        for period in (64, 128):
            for sign in (-1, 1):
                point = np.array([1220., 945.]) + sign * expected[:, column] / period
                rows.append(dict(label=f"ramp_{axis}_{period}_{sign}", spot_xy=point.tolist()))
    matrix, _ = module.geometry(rows)
    np.testing.assert_allclose(matrix, expected)


def test_geometry_refuses_missing_negative_pair():
    with pytest.raises(ValueError, match="negative ramp"):
        module.geometry([dict(label="ramp_x_64_1", spot_xy=[10., 20.])])


@pytest.mark.parametrize("change", [dict(peak=255), dict(exposure_ms=0), dict(exposure_status="dim"),
                                    dict(settled=dict(settled=False))])
def test_invalid_measurements_do_not_enter_fit(change):
    assert not module.valid(dict(peak=100., exposure_ms=1.) | change)


def test_self_contained_checkpoint_needs_no_upstream_chain(tmp_path):
    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import load_frozen_parameters

    data = dict(defocus_diopter=.12, origin=[1220., 960.], amplitude_center=[900., 700.],
                amplitude_width=[200., 300.], residual_noll_radians={"5": .2}, gain=2.,
                background=.01, sigma_ccd_px=2., ramp_matrix=[[10., 7500.], [-7600., 20.]],
                grid_size=64, fft_size=512, pixel_samples=1)
    path = tmp_path / "frozen_parameters.json"
    path.write_text(json.dumps(data))
    for source in (path, tmp_path):
        loaded = load_frozen_parameters(source)
        assert loaded.grid_size == 64
        assert loaded.fft_size == 512
        assert loaded.ramp_matrix[1][0] == -7600.
