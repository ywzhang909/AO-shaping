"""Verify that the slm_square_runner default parameters produce a Zernike
basis matching the GUI (multi_slm_controller.py Zernike branch):

    radius = 600  (min(SLM_HEIGHT, SLM_WIDTH) / 2)
    init  = Defocus(2,0) + Spherical(4,0)

No hardware is opened — ``optimize_slm_square`` is monkey-patched to capture
its arguments.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from click.testing import CliRunner

from ao_shaping.runners.slm_square_runner import run


def _invoke(args: list[str]):
    runner = CliRunner()
    captured: dict = {}

    def _fake_optimize(**kwargs):
        captured.update(kwargs)
        from ao_shaping.utils.file import Recorder

        rec = Recorder(mark="quality", mode="max")
        rec.append({"quality": 1.0})
        return rec

    # optimize_slm_square is imported lazily inside run(), so patch at the
    # source module where the symbol actually lives.
    with patch(
        "ao_shaping.optimizer.wfless.slm_square_shaping.optimize_slm_square",
        side_effect=_fake_optimize,
    ):
        result = runner.invoke(run, args, obj={"dir": "data"})
    return result, captured


class TestSlmSquareRunnerDefaultsMatchGui:
    def test_default_basis_is_zernike(self):
        result, captured = _invoke(["-e", "1"])
        assert result.exit_code == 0, result.output
        assert captured["basis"] == "zernike"

    def test_default_zernike_radius_is_600(self):
        result, captured = _invoke(["-e", "1"])
        assert result.exit_code == 0, result.output
        assert captured["zernike_radius"] == 600

    def test_default_init_is_defocus_plus_spherical(self):
        result, captured = _invoke(["-e", "1"])
        assert result.exit_code == 0, result.output
        init_c = captured["init_c"]
        assert init_c is not None
        assert len(init_c) == 15  # n_max=4 -> 15 terms
        # Noll 4 = (2,0) defocus -> index 3
        # Noll 11 = (4,0) spherical -> index 10
        assert init_c[3] == pytest.approx(1.0)
        assert init_c[10] == pytest.approx(0.5)
        # All other modes start at 0
        expected = np.zeros(15, dtype=np.float64)
        expected[3] = 1.0
        expected[10] = 0.5
        assert np.allclose(init_c, expected)

    def test_init_coeffs_override(self):
        result, captured = _invoke(["-e", "1", "--init-coeffs", '{"5":2.0,"13":0.7}'])
        assert result.exit_code == 0, result.output
        init_c = captured["init_c"]
        # Noll 5 = (2,-2) astig -> index 4
        # Noll 13 = (4,-2) secondary astig -> index 12
        assert init_c[4] == pytest.approx(2.0)
        assert init_c[12] == pytest.approx(0.7)

    def test_freeform_basis_default_init_is_none(self):
        result, captured = _invoke(["-e", "1", "--basis", "freeform", "--seed", "42"])
        assert result.exit_code == 0, result.output
        assert captured["basis"] == "freeform"
        # freeform init is generated randomly inside optimize_slm_square;
        # the runner passes init_c=None and lets the optimizer handle it.
        assert captured["init_c"] is None
        assert captured["phase_grid"] == 24

    def test_zernike_radius_override(self):
        result, captured = _invoke(["-e", "1", "--zernike-radius", "500"])
        assert result.exit_code == 0, result.output
        assert captured["zernike_radius"] == 500

    def test_init_defocus_spherical_override(self):
        result, captured = _invoke(
            ["-e", "1", "--init-defocus", "2.0", "--init-spherical", "0.7"]
        )
        assert result.exit_code == 0, result.output
        init_c = captured["init_c"]
        assert init_c[3] == pytest.approx(2.0)
        assert init_c[10] == pytest.approx(0.7)

    def test_center_default_is_shape(self):
        result, captured = _invoke(["-e", "1"])
        assert result.exit_code == 0, result.output
        assert captured["center"] == "shape"

    def test_center_centroid_thresh_passthrough(self):
        result, captured = _invoke(["-e", "1", "--center", "centroid_thresh"])
        assert result.exit_code == 0, result.output
        assert captured["center"] == "centroid_thresh"

    def test_center_coordinate_tuple_passthrough(self):
        result, captured = _invoke(["-e", "1", "--center", "320,240"])
        assert result.exit_code == 0, result.output
        assert captured["center"] == (320, 240)
