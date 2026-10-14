"""Tests for the shared selector dispatcher.

The three ``python -m`` multi-command runners (matrix, zernike search, voltage)
used to each carry their own ``main()``. Those copies all shared one latent bug:
the branch matched any known selector but then dispatched to the *pair* command
unconditionally, so explicitly naming the default selector ran the wrong one --
``python -m ao_shaping.runners.matrix_runner dm-matrix`` executed
``hadamard-matrix``. The dict-based dispatcher fixes that, so it is pinned here.
"""
from __future__ import annotations

import click
import pytest

from ao_shaping.utils.io.cli_helpers import run_selector_dispatch

RECORDED: list[tuple[str, list[str]]] = []


@pytest.fixture(autouse=True)
def _clear_recorded() -> None:
    RECORDED.clear()


def _cmd(name: str) -> click.Command:
    @click.command(name=name)
    def _inner() -> None:
        RECORDED.append((name, []))

    return _inner


def _cmd_recording_argv(name: str) -> click.Command:
    @click.command(
        name=name,
        context_settings={"ignore_unknown_options": True},
        add_help_option=False,
    )
    @click.argument("rest", nargs=-1, type=click.UNPROCESSED)
    def _inner(rest: tuple[str, ...]) -> None:
        RECORDED.append((name, list(rest)))

    return _inner


COMMANDS = {"alpha": _cmd("alpha"), "beta": _cmd("beta")}


class TestSelectorRouting:
    def test_no_selector_runs_the_default(self) -> None:
        with pytest.raises(SystemExit) as exc:
            run_selector_dispatch(COMMANDS, "alpha", [])
        assert exc.value.code == 0
        assert RECORDED == [("alpha", [])]

    def test_explicit_default_selector_runs_the_default(self) -> None:
        # The bug the extraction fixed: this used to dispatch to "beta".
        with pytest.raises(SystemExit) as exc:
            run_selector_dispatch(COMMANDS, "alpha", ["alpha"])
        assert exc.value.code == 0
        assert RECORDED == [("alpha", [])]

    def test_pair_selector_runs_the_pair_command(self) -> None:
        with pytest.raises(SystemExit):
            run_selector_dispatch(COMMANDS, "alpha", ["beta"])
        assert RECORDED == [("beta", [])]

    def test_selector_is_stripped_from_the_child_argv(self) -> None:
        commands = {"alpha": _cmd_recording_argv("alpha"), "beta": _cmd_recording_argv("beta")}
        with pytest.raises(SystemExit):
            run_selector_dispatch(commands, "alpha", ["beta", "--flag", "1"])
        assert RECORDED == [("beta", ["--flag", "1"])]

    def test_argv_defaults_to_sys_argv(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.argv", ["prog", "beta"])
        with pytest.raises(SystemExit):
            run_selector_dispatch(COMMANDS, "alpha")
        assert RECORDED == [("beta", [])]


class TestExitCodes:
    def test_nonzero_exit_from_the_command_is_propagated(self) -> None:
        @click.command()
        def _boom() -> None:
            raise SystemExit(7)

        with pytest.raises(SystemExit) as exc:
            run_selector_dispatch({"alpha": _boom}, "alpha", [])
        assert exc.value.code == 7

    def test_ordinary_exception_becomes_exit_1(self) -> None:
        @click.command()
        def _bad() -> None:
            raise RuntimeError("kaboom")

        with pytest.raises(SystemExit) as exc:
            run_selector_dispatch({"alpha": _bad}, "alpha", [])
        assert exc.value.code == 1


class TestContract:
    def test_default_label_must_be_a_known_command(self) -> None:
        with pytest.raises(KeyError):
            run_selector_dispatch({"alpha": _cmd("alpha")}, "missing", [])