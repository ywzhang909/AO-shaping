"""Tests for greedy_zernike_runner module (B2 dataclass-click conversion, 2026-09).

Locks the converted CLI surface (21 options = GreedyZernikeParams 11 +
WfsParams 5 + ZernikeSlmParams 5) and the run() parameter delivery to
``optimizer_greedy`` — including the fact that ``wfs_exposure_time`` is
NOT delivered (the greedy optimizer never receives it).
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from click.testing import CliRunner

from ao_shaping.drivers import MlaRes

EXPECTED_OPTIONS = [
    # GreedyZernikeParams (11)
    "dir",
    "epochs",
    "n_max",
    "early_stop_threshold",
    "show",
    "n_init",
    "n_directions",
    "perturbation_scale",
    "algorithm",
    "pop_size",
    "debug",
    # WfsParams (5)
    "wfs_res",
    "pupil_diameter",
    "pupil_center",
    "exposure_time_ms",
    "remove_tilt",
    # ZernikeSlmParams (5)
    "wavelength",
    "shift_x",
    "shift_y",
    "slm_number",
    "wait_time",
]


class _FakeRecords:
    """Minimal Recorder stand-in for run()'s post-optimization calls."""

    def get_best_iter(self):
        return (
            {
                "_c": np.zeros(15),
                "_wavefront": [np.zeros((32, 32)), np.zeros((32, 32))],
            },
            (3, 0.12),
        )

    def save_best(self, **kwargs):
        self.saved_kwargs = kwargs


class TestCliOptions:
    """Test CLI command options."""

    def test_command_exists(self):
        """Test that the run command is a Click command."""
        from ao_shaping.runners.greedy_zernike_runner import run

        assert hasattr(run, "callback")
        assert callable(run)

    def test_all_expected_options_exist(self):
        """Test that all 21 expected CLI options are present."""
        from ao_shaping.runners.greedy_zernike_runner import run

        param_names = [p.name for p in run.params]
        assert len(param_names) == 21, f"got {len(param_names)} options"
        for name in EXPECTED_OPTIONS:
            assert name in param_names, f"Missing CLI option: {name}"

    def test_cli_help_shows_perturbation_flag(self):
        """Help text exposes the kebab-case --perturbation-scale flag."""
        from ao_shaping.runners.greedy_zernike_runner import run

        result = CliRunner().invoke(run, ["--help"])
        assert result.exit_code == 0, result.output
        assert "--perturbation-scale" in result.output


class TestRunFunction:
    """Test the main run function with a mocked optimizer (no hardware)."""

    def _invoke(self, args, monkeypatch=None, capture=None):
        from ao_shaping.runners.greedy_zernike_runner import run

        def fake_optimizer_greedy(**kwargs):
            if capture is not None:
                capture.update(kwargs)
            return _FakeRecords()

        with patch(
            "ao_shaping.runners.greedy_zernike_runner.optimizer_greedy",
            side_effect=fake_optimizer_greedy,
        ):
            return CliRunner().invoke(run, args)

    def test_run_completes_with_mocked_optimizer(self):
        result = self._invoke(["--epochs", "2", "--n-max", "4", "--wfs_res", "1024"])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert "完成" in result.output and "RMS" in result.output

    def test_run_delivers_all_params_to_optimizer(self):
        captured: dict = {}

        result = self._invoke(
            [
                "--epochs",
                "5",
                "--n-max",
                "3",
                "--early_stop_threshold",
                "0.05",
                "--n-init",
                "7",
                "--n-directions",
                "4",
                "--perturbation-scale",
                "2.5",
                "--algorithm",
                "ga",
                "--pop_size",
                "12",
                "--wfs_res",
                "512",
                "--pupil_diameter",
                "3.0",
                "--pupil_center",
                "1.5,-2.5",
                "--remove-tilt",
                "--exposure-time-ms",
                "40.0",
                "--wavelength",
                "633",
                "--shift-x",
                "2",
                "--shift-y",
                "-3",
                "--slm-number",
                "2",
                "--wait-time",
                "0.5",
            ],
            capture=captured,
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert captured["epochs"] == 5
        assert captured["n_max"] == 3
        assert captured["n_init"] == 7
        assert captured["n_directions"] == 4
        assert captured["perturbation_scale"] == 2.5
        assert captured["early_stop_threshold"] == 0.05
        assert captured["algorithm"] == "ga"
        assert captured["pop_size"] == 12
        # init_z is always None (first random-position sampling inside optimizer)
        assert captured["init_z"] is None
        # WFS + SLM hardware params
        assert captured["pupil_center"] == (1.5, -2.5)
        assert captured["pupil_diameter"] == 3.0
        assert isinstance(captured["wfs_res"], MlaRes)
        assert captured["remove_tilt"] is True
        assert captured["wavelength"] == 633
        assert captured["shift_x"] == 2
        assert captured["shift_y"] == -3
        assert captured["slm_number"] == 2
        # regression lock: the greedy optimizer never receives exposure time
        assert "wfs_exposure_time" not in captured
        # nor the SLM wait time (stored on ZernikeSlmParams, not delivered)
        assert "wait_time" not in captured


class TestImports:
    """Test that the module imports cleanly."""

    def test_module_imports_clean(self):
        from ao_shaping.runners import greedy_zernike_runner

        assert hasattr(greedy_zernike_runner, "run")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])