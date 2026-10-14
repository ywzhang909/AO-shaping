"""Guard: **no** CLI long flag may contain an underscore.

Why this exists
---------------
The repo's flag naming was inconsistent: ``--cam-id`` (dash) sat next to
``--cam_type`` (underscore) *in the same command's* ``--help``. That is not a
cosmetic issue — a user who cannot predict which spelling a flag takes will
guess wrong, and the guess fails at parse time, on a bench, mid-run.

This module locks the convention in two independent layers, because either one
alone has a hole:

**Layer 1 — static (declaration sites).** Walks every ``.py`` under ``src/`` and
finds *string literals* that look like a long flag containing an underscore.
Declarations are the single point where a flag name is chosen, so this catches a
missed rename even in a command no test imports. Docstrings are excluded: they
*describe* flags in prose and are handled by the doc sweep, not by this rule.

**Layer 2 — dynamic (resolved commands).** Imports the real Click group and
walks every command recursively (including subcommands such as
``slm-pib spgd``), asserting no resolved :class:`click.Option` exposes an
underscore flag. This is what a user actually types, so it is the layer that
would catch a flag introduced through a path the static scan cannot see.

Both layers are needed. The static scan cannot see a flag built by string
concatenation or synthesised at import time; the dynamic walk only covers the
commands the test imports.

Not covered, on purpose: single-dash short flags (``-c``) and ``/``-joined
boolean pairs (``--warm-start/--no-warm-start``) — those are validated by
Click itself and are not where the drift happened.
"""

from __future__ import annotations

import ast
import pathlib
import re

import click
import pytest

#: tests/ao_shaping/runners/<this file> -> repository root
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"

#: A long flag that contains at least one underscore, e.g. ``--cam_type``.
_UNDERSCORE_LONG_FLAG = re.compile(r"^--[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+$")

#: Anything that merely *starts* with a double dash and looks flag-shaped.
_FLAG_SHAPED = re.compile(r"^--[A-Za-z0-9][A-Za-z0-9_-]*$")


def _read(source_path: pathlib.Path) -> str:
    return source_path.read_text(encoding="utf-8-sig")


def _python_sources() -> list[pathlib.Path]:
    return sorted(SRC_ROOT.rglob("*.py"))


def _flag_like_literals(source_path: pathlib.Path) -> list[tuple[int, str]]:
    """Return ``(lineno, literal)`` for every flag-shaped string literal.

    Docstrings (and other bare string expressions) are skipped: a module
    docstring that *mentions* ``--cam_type`` while documenting the old
    spelling is prose, not a declaration.
    """
    tree = ast.parse(_read(source_path), filename=str(source_path))
    # Docstring node ids of every module/class/function, so their text is skipped.
    docstring_nodes: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstring_nodes.add(id(body[0].value))

    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if id(node) in docstring_nodes:
            continue
        text = node.value
        # A flag literal is the whole string, not a sentence containing one.
        if _FLAG_SHAPED.match(text):
            found.append((node.lineno, text))
    return found


def _iter_commands() -> list[tuple[str, click.Command]]:
    """Every command reachable from the top-level group, recursively."""
    from ao_shaping.main import cli

    out: list[tuple[str, click.Command]] = []

    def walk(command: click.Command, path: str) -> None:
        out.append((path, command))
        if isinstance(command, click.Group):
            for name in sorted(command.commands):
                walk(command.commands[name], f"{path} {name}")

    walk(cli, "main")
    return out


# ---------------------------------------------------------------------------
# Layer 1: static declaration sites
# ---------------------------------------------------------------------------


def test_no_source_declares_an_underscore_long_flag() -> None:
    """No ``.py`` under ``src/`` may declare a ``--foo_bar`` flag literal."""
    offenders: list[str] = []
    for source_path in _python_sources():
        for lineno, literal in _flag_like_literals(source_path):
            if _UNDERSCORE_LONG_FLAG.match(literal):
                rel = source_path.relative_to(REPO_ROOT)
                offenders.append(f"{rel}:{lineno}: {literal!r}")

    assert offenders == [], (
        "CLI long flags must use dashes, not underscores. Found "
        f"{len(offenders)} offending declaration(s):\n" + "\n".join(sorted(offenders))
    )


# ---------------------------------------------------------------------------
# Layer 2: resolved commands
# ---------------------------------------------------------------------------


def test_main_group_exposes_no_underscore_long_flag() -> None:
    """No option on any registered command may expose an underscore flag."""
    offenders: list[str] = []
    for path, command in _iter_commands():
        for param in command.params:
            for opt in (*param.opts, *param.secondary_opts):
                if _UNDERSCORE_LONG_FLAG.match(opt):
                    offenders.append(f"{path}: {opt}")

    assert offenders == [], (
        "Resolved Click options must use dashes, not underscores. Found "
        f"{len(offenders)} offending option(s):\n" + "\n".join(sorted(offenders))
    )


def test_dynamic_scan_is_not_vacuous() -> None:
    """Falsification half: the dynamic layer must actually reach commands.

    A typo in the import path would make ``test_main_group_exposes_no_...``
    pass for the wrong reason (an empty scan). Pin the floor by asserting the
    walk reaches the real command set, subcommands included.

    Note ``--help`` is deliberately NOT the probe: Click injects it lazily via
    ``get_help_option()`` and it never appears in ``Command.params``.
    """
    from ao_shaping.main import cli

    commands = _iter_commands()
    paths = {path for path, _ in commands}

    assert "main" in paths
    assert {f"main {name}" for name in cli.commands} <= paths
    # These two exist to prove the recursion is real: a flat walk of
    # ``cli.commands`` would satisfy every assertion above and miss every
    # subcommand, which is where half the flags live.
    assert {"main slm-pib spgd", "main slm-pib heuristic"} <= paths
    assert len(commands) > len(cli.commands), (
        f"walk found {len(commands)} commands for {len(cli.commands)} registrations"
    )


@pytest.mark.parametrize(
    "bad",
    ["--cam_type", "--a_b", "--x_1"],
)
def test_regex_rejects_underscore_flags(bad: str) -> None:
    """Falsification half: the detector must not be a no-op."""
    assert _UNDERSCORE_LONG_FLAG.match(bad)


@pytest.mark.parametrize("good", ["--cam-type", "--n-max", "--lr", "-c", "--help"])
def test_regex_accepts_dash_flags(good: str) -> None:
    """Falsification half: and it must not over-reject valid spellings."""
    assert not _UNDERSCORE_LONG_FLAG.match(good)