"""Tests for the dual-algorithm diff_beam_runner module.

The runner is loaded by *file path* (``importlib``) because importing it
through the ``ao_shaping.runners`` package triggers ``runners/__init__.py``,
whose import chain performs a network ping in ``config.py`` resolving DM
actuators (pre-existing infra issue that hangs in offline environments).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from click.testing import CliRunner

from ao_shaping.algorithm.beam_shaping_utils import compute_metrics

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RUNNER_PATH = _REPO_ROOT / "src" / "ao_shaping" / "runners" / "diff_beam_runner.py"


@pytest.fixture(scope="module")
def runner_module():
    """Load diff_beam_runner.py directly by file path (bypasses package __init__)."""
    spec = importlib.util.spec_from_file_location("diff_beam_runner", _RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestCliOptions:
    """Test the Click command surface."""

    def test_command_exists(self, runner_module):
        assert hasattr(runner_module.run, "callback")
        assert callable(runner_module.run)

    def test_algorithm_choice(self, runner_module):
        """--algorithm must accept only backprop/gs."""
        choice = [p for p in runner_module.run.params if p.name == "algorithm"][0]
        assert sorted(choice.type.choices) == ["backprop", "gs"]

    def test_expected_options_exist(self, runner_module):
        param_names = [p.name for p in runner_module.run.params]
        expected = [
            "algorithm",
            "target_image",
            "target_shape",
            "epochs",
            "lr",
            "iterations",
            "distance",
            "wavelength",
            "slm_wavelength",
            "slm_number",
            "cam_id",
            "cam_center",
            "cam_size",
            "cam_exposure",
            "adaptive",
            "adaptive_iterations",
            "device",
            "seed",
            "save_dir",
            "use_hardware",
            "show",
        ]
        for name in expected:
            assert name in param_names, f"Missing CLI option: {name}"


class TestRunBackprop:
    """Test `run --algorithm backprop` end-to-end in mock mode."""

    def test_backprop_e2e(self, runner_module, tmp_path):
        runner = CliRunner()
        result = runner.invoke(
            runner_module.run,
            [
                "--algorithm",
                "backprop",
                "--target-shape",
                "gaussian",
                "--epochs",
                "2",
                "--lr",
                "0.01",
                "--device",
                "cpu",
                "--seed",
                "0",
                "--save-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # Results must be persisted.
        dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
        assert len(dirs) == 1
        out_dir = dirs[0]
        for fname in [
            "config.json",
            "phase_pattern.npy",
            "target_intensity.npy",
            "measured_intensity.npy",
            "loss_history.npy",
        ]:
            assert (out_dir / fname).exists(), f"Missing output: {fname}"

        # config.json must record the algorithm and metrics.
        import json

        cfg = json.loads((out_dir / "config.json").read_text())
        assert cfg["algorithm"] == "backprop"
        assert set(cfg["metrics"]) == {"mse", "correlation", "efficiency"}

        # Shared metric definition must be usable on the saved arrays.
        import numpy as np

        measured = np.load(out_dir / "measured_intensity.npy")
        target = np.load(out_dir / "target_intensity.npy")
        m = compute_metrics(measured, target)
        assert set(m) == {"mse", "correlation", "efficiency"}


class TestRunGs:
    """Test `run --algorithm gs` end-to-end in mock mode (plain and adaptive)."""

    def test_gs_e2e(self, runner_module, tmp_path):
        runner = CliRunner()
        result = runner.invoke(
            runner_module.run,
            [
                "--algorithm",
                "gs",
                "--target-shape",
                "gaussian",
                "--iterations",
                "5",
                "--save-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0, f"CLI failed: {result.output}"

        dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
        assert len(dirs) == 1
        out_dir = dirs[0]
        for fname in [
            "config.json",
            "phase_pattern.npy",
            "target_intensity.npy",
            "measured_intensity.npy",
            "loss_history.npy",
        ]:
            assert (out_dir / fname).exists(), f"Missing output: {fname}"

        import json

        cfg = json.loads((out_dir / "config.json").read_text())
        assert cfg["algorithm"] == "gs"

    def test_gs_adaptive_without_hardware_falls_back(self, runner_module, tmp_path):
        """--adaptive without --use-hardware must warn and fall back to plain GS."""
        runner = CliRunner()
        result = runner.invoke(
            runner_module.run,
            [
                "--algorithm",
                "gs",
                "--target-shape",
                "gaussian",
                "--iterations",
                "5",
                "--adaptive",
                "--save-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # Fallback warning is emitted by loguru (stderr); verify we still ran.
        assert "Differentiable beam shaping complete!" in result.output


class TestRunValidation:
    """Test CLI input validation."""

    def test_invalid_algorithm_rejected(self, runner_module, tmp_path):
        result = CliRunner().invoke(
            runner_module.run,
            ["--algorithm", "bogus", "--save-dir", str(tmp_path)],
        )
        assert result.exit_code == 2  # Click Choice validation failure

    def test_missing_target_image_rejected(self, runner_module, tmp_path):
        result = CliRunner().invoke(
            runner_module.run,
            [
                "--algorithm",
                "backprop",
                "--target-image",
                str(tmp_path / "does_not_exist.png"),
                "--epochs",
                "1",
                "--save-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 2  # click.Path(exists=True) validation failure


if __name__ == "__main__":
    pytest.main([__file__, "-v"])