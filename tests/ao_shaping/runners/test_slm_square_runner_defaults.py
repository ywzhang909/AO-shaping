"""Verify that ``spgd-square``'s default parameters produce a Zernike
basis matching the GUI (``multi_slm_controller.py`` Zernike branch):

    radius = 600  (min(SLM_HEIGHT, SLM_WIDTH) / 2)
    init  = Defocus(2,0) + Spherical(4,0)

No hardware is opened — ``optimize_slm_square`` is monkey-patched to capture its
arguments. The command now lives as the ``square`` subcommand of
``runners/slm/slm_shaping_runner.py`` (merged with the ``slm-pib`` group), and it
hands the optimizer a :class:`SlmSquareConfig` container rather than ~30 flat
kwargs, so the assertions read the captured config.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from click.testing import CliRunner

from ao_shaping.runners.slm.slm_shaping_runner import square


def _invoke(args: list[str], obj: dict | None = None):
    runner = CliRunner()
    captured: dict = {}

    def _fake_optimize(**kwargs):
        captured.update(kwargs)
        from ao_shaping.utils.io.file import Recorder

        rec = Recorder(mark="quality", mode="max")
        rec.append({"quality": 1.0, "_epoch": 0})
        return rec

    # The runner binds ``optimize_slm_square`` at module scope, so that name is
    # what the call resolves through -- patch it there, not in the optimizer
    # module (which the runner never looks up again).
    with patch(
        "ao_shaping.runners.slm.slm_shaping_runner.optimize_slm_square",
        side_effect=_fake_optimize,
    ):
        result = runner.invoke(
            square, args, obj={"dir": "data"} if obj is None else obj
        )
    return result, captured


class TestSlmSquareRunnerDefaultsMatchGui:
    def test_default_basis_is_zernike(self):
        result, captured = _invoke(["-e", "1"])
        assert result.exit_code == 0, result.output
        assert captured["config"].basis == "zernike"

    def test_default_zernike_radius_is_600(self):
        result, captured = _invoke(["-e", "1"])
        assert result.exit_code == 0, result.output
        assert captured["config"].zernike_radius == 600

    def test_default_init_is_defocus_plus_spherical(self):
        result, captured = _invoke(["-e", "1"])
        assert result.exit_code == 0, result.output
        init_c = captured["config"].init_c
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
        init_c = captured["config"].init_c
        # Noll 5 = (2,-2) astig -> index 4
        # Noll 13 = (4,-2) secondary astig -> index 12
        assert init_c[4] == pytest.approx(2.0)
        assert init_c[12] == pytest.approx(0.7)

    def test_freeform_basis_default_init_is_none(self):
        result, captured = _invoke(["-e", "1", "--basis", "freeform", "--seed", "42"])
        assert result.exit_code == 0, result.output
        config = captured["config"]
        assert config.basis == "freeform"
        # freeform init is generated randomly inside optimize_slm_square;
        # the runner passes init_c=None and lets the optimizer handle it.
        assert config.init_c is None
        assert config.phase_grid == 24
        assert config.random_seed == 42

    def test_zernike_radius_override(self):
        result, captured = _invoke(["-e", "1", "--zernike-radius", "500"])
        assert result.exit_code == 0, result.output
        assert captured["config"].zernike_radius == 500

    def test_init_defocus_spherical_override(self):
        result, captured = _invoke(
            ["-e", "1", "--init-defocus", "2.0", "--init-spherical", "0.7"]
        )
        assert result.exit_code == 0, result.output
        init_c = captured["config"].init_c
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

    def test_epochs_stay_positional(self):
        result, captured = _invoke(["-e", "7"])
        assert result.exit_code == 0, result.output
        assert captured["epochs"] == 7


class TestSlmSquareDeltaPinning:
    """``--delta`` must survive the ``lr == 0`` adaptive schedule.

    The schedule *reassigns* ``delta`` every epoch, so a runner that forwards
    the value without marking it pinned turns the flag into a no-op — which is
    exactly the bug ``resolve_spgd_delta`` exists to prevent.
    """

    def test_omitting_delta_leaves_it_adaptive(self):
        result, captured = _invoke(["-e", "1"])
        assert result.exit_code == 0, result.output
        config = captured["config"]
        assert config.delta == pytest.approx(0.1)
        assert config.delta_pinned is False

    def test_explicit_delta_is_pinned(self):
        result, captured = _invoke(["-e", "1", "--delta", "0.35"])
        assert result.exit_code == 0, result.output
        config = captured["config"]
        assert config.delta == pytest.approx(0.35)
        assert config.delta_pinned is True


class TestSlmSquareCliGuards:
    def test_target_side_and_mean_brightness_are_mutually_exclusive(self):
        result, _ = _invoke(
            ["-e", "1", "--target-side", "20", "--target-mean-brightness", "80"]
        )
        assert result.exit_code != 0
        assert "互斥" in result.output

    def test_sim_slm_without_sim_camera_is_rejected(self):
        """Silently opening a real SLM would hang the controller, not fail."""
        result, _ = _invoke(["-e", "1", "--slm_type", "sim"])
        assert result.exit_code != 0
        assert "--cam_type sim" in result.output
