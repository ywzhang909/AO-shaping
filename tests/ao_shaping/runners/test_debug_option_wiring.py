"""Tests for the `--debug` option wiring (group / command / DEBUG env).

No hardware: only the resolution helper and the click surfaces are exercised.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from ao_shaping.utils.io.cli_helpers import resolve_debug


def _ctx(parent_obj=None):
    """Fake click context: ``parent_obj=None`` mimics a standalone invocation."""
    parent = None if parent_obj is None else SimpleNamespace(obj=parent_obj)
    return SimpleNamespace(parent=parent)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("DEBUG", raising=False)


def test_group_flag_enables_debug():
    assert resolve_debug(_ctx({"debug": True})) is True


def test_command_flag_enables_debug():
    assert resolve_debug(_ctx({"debug": False}), True) is True


def test_env_var_enables_debug(monkeypatch):
    monkeypatch.setenv("DEBUG", "1")
    assert resolve_debug(_ctx({"debug": False}), None) is True


def test_all_off_is_false():
    assert resolve_debug(_ctx({"debug": False}), None) is False


def test_standalone_context_still_honours_flag():
    assert resolve_debug(None, True) is True
    assert resolve_debug(None, None) is False


def test_standalone_context_still_honours_env(monkeypatch):
    monkeypatch.setenv("DEBUG", "yes")
    assert resolve_debug(None, None) is True


class TestCommandDebugSurface:
    @pytest.mark.parametrize(
        "runner_name",
        ["axis_beam_runner", "combined_runner"],
    )
    def test_nlight_dm_runners_expose_debug(self, runner_name):
        import importlib

        module = importlib.import_module(f"ao_shaping.runners.nlight_dm.{runner_name}")
        result = CliRunner().invoke(module.run, ["--help"])

        assert result.exit_code == 0, result.output
        assert "--debug" in result.output

    def test_slm_pib_exposes_debug(self):
        from ao_shaping.runners.slm.slm_shaping_runner import run

        # run is a click group; --debug lives on the spgd subcommand.
        result = CliRunner().invoke(run, ["spgd", "--help"])

        assert result.exit_code == 0, result.output
        assert "--debug" in result.output


def test_main_group_exposes_debug_flag():
    from ao_shaping.main import cli

    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0, result.output
    assert "--debug" in result.output


def test_gs_refine_honours_main_group_debug(tmp_path, monkeypatch):
    import numpy as np

    from ao_shaping.main import cli
    from ao_shaping.optimizer.wfless import slm_gs_refine
    from ao_shaping.utils.io.file import Recorder

    seen = []

    def fake_optimize(config):
        seen.append(config.record_debug)
        recorder = Recorder(mark="score", mode="max")
        recorder.append(
            {
                "_epoch": 0,
                "stage": "flat",
                "score": 0.2,
                "pib": 0.1,
                "cv": 1.0,
                "_img": np.ones((4, 4)),
                "_phase": np.zeros((4, 4), dtype=np.uint16),
            }
        )
        recorder.append({"_epoch": 1, "stage": "gs", "score": 0.3, "pib": 0.2, "cv": 0.8})
        recorder.append(
            {"_epoch": 2, "stage": "refine", "score": 0.4, "pib": 0.3, "cv": 0.7}
        )
        return recorder

    monkeypatch.setattr(slm_gs_refine, "optimize_slm_gs_refine", fake_optimize)
    result = CliRunner().invoke(
        cli,
        ["--dir", str(tmp_path), "--debug", "slm-gs-refine", "--cam-type", "sim"],
    )

    assert result.exit_code == 0, result.output
    assert seen == [True]
    assert "Debug data saved to:" in result.output
    assert "[adopted]" in result.output
    assert list((tmp_path / "debug").rglob("*.pkl"))
    csv_files = list((tmp_path / "slm_gs_refine").rglob("*.csv"))
    assert len(csv_files) == 1
    csv_text = csv_files[0].read_text(encoding="utf-8")
    assert "score" in csv_text
    assert "_img" not in csv_text
    assert "_phase" not in csv_text
