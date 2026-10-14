"""Guard: a command's stacked ``with_params`` classes must not collide.

Why this exists
---------------
``with_params`` attaches each dataclass's options independently, so two classes
stacked on one command can both claim ``--debug``. Click keeps one and silently
drops the other, and ``--help`` looks perfectly normal. That is not theoretical:
this repo shipped ``CombinedAlgorithmParams.debug_flag`` alongside
``CombinedRunParams.debug_flag``, and the algorithm copy was dead because the
runner only ever read the run one.

``_collect_click_annotations`` cannot catch this. It raises on a duplicate within
one class tree, but stacked decorators are separate collections, so the check
never sees the collision. This module walks the resolved commands instead, which
is where the collision actually exists.

The second test is the falsification half: it pins one command's full option set
so that deleting the assertion above cannot silently pass.
"""

from __future__ import annotations

import click
import pytest

from ao_shaping.main import cli

#: Commands whose option surface is pinned. Any change here must be a deliberate
#: review of the diff, not a side effect of adding a dataclass field elsewhere.
_PINNED: dict[str, frozenset[str]] = {
    "combined": frozenset({"--debug", "--cam-type", "--cam-id", "--dm-type", "--root-dir"}),
    "pipeline": frozenset({"--exposure-time-ms", "--wfs-exposure-ms", "--dm-type"}),
}


def _commands() -> list[tuple[str, click.Command]]:
    out: list[tuple[str, click.Command]] = []

    def walk(command: click.Command, path: str) -> None:
        out.append((path, command))
        if isinstance(command, click.Group):
            for name in sorted(command.commands):
                walk(command.commands[name], f"{path} {name}")

    walk(cli, "main")
    return out


def test_no_command_declares_a_duplicate_option_flag() -> None:
    """No option may appear on more than one parameter of the same command."""
    offenders: list[str] = []
    for path, command in _commands():
        seen: dict[str, int] = {}
        for param in command.params:
            for opt in param.opts:
                seen[opt] = seen.get(opt, 0) + 1
        duplicates = {opt: n for opt, n in seen.items() if n > 1}
        if duplicates:
            offenders.append(f"{path}: {duplicates}")

    assert offenders == [], (
        "A duplicated option means two stacked @with_params classes claim the "
        "same flag and Click silently kept one:\n" + "\n".join(offenders)
    )


@pytest.mark.parametrize("name", sorted(_PINNED))
def test_pinned_option_surface_is_intact(name: str) -> None:
    """The pinned commands expose exactly the options this module relies on."""
    command = cli.commands.get(name)
    assert command is not None, f"{name} is not registered"
    exposed = {opt for param in command.params for opt in param.opts}
    missing = _PINNED[name] - exposed
    assert missing == set(), f"{name} no longer exposes {sorted(missing)}"