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
        from ao_shaping.runners.slm_pib_runner import run

        # run is a click group; --debug lives on the spgd subcommand.
        result = CliRunner().invoke(run, ["spgd", "--help"])

        assert result.exit_code == 0, result.output
        assert "--debug" in result.output


def test_main_group_exposes_debug_flag():
    from ao_shaping.main import cli

    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0, result.output
    assert "--debug" in result.output
