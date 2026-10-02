"""Golden snapshot of the CLI ``--help`` surface (safety net for T1.1 / T1.3).

Why this exists
---------------
Zero existing tests import ``with_params`` or assert any CLI option surface, and
the 13 test files under ``tests/ao_shaping/tools/slm/`` only import *pure*
helpers. So an upcoming migration of the ~15 hand-written click modules in
``src/ao_shaping/tools/slm/`` to the ``with_params`` dataclass mechanism could
silently rename a flag, drop a flag, change a default or reorder ``--help`` and
the whole suite would stay green. This file is that missing net.

What is frozen
--------------
**Group A** - the commands registered on ``ao_shaping.main:cli``, keyed
``main:<name>``, iterating ``sorted(cli.commands)``.

**Group B** - every ``click.Command`` object found by scanning ``vars(mod)`` of
the 17 hand-written click modules under ``ao_shaping.tools.slm``, keyed
``tools.slm.<module>:<attr>``. ``vars(mod)`` is scanned rather than assuming the
attribute name matches the function name. ``calibration`` contributes two
commands (``main`` and ``main_shift_calib``).

``slm_zernike_sweep_probe`` is now included: it was migrated from argparse to
click (TODO.md R5), and its 21-option ``--help`` is part of the frozen surface.

Normalisation
-------------
Only the whole ``Usage: <something>`` line is rewritten to the literal
``Usage: <prog>`` so the snapshot is prog-name independent. Nothing else in the
help text is altered - option order, help strings, defaults, metavars and the
subcommand listing are all part of the contract. Each command also stores an
integer exit code under a ``:exit`` key, so every command contributes two
parametrized cases.

Regeneration (opt-in only)
-------------------------
The default is **compare**. To re-capture after an intentional CLI change::

    AO_UPDATE_HELP_GOLDEN=1 ./.venv/bin/python -m pytest tests/ao_shaping/runners/test_cli_help_golden.py -q

The rewrite happens in a module-scoped fixture, so it runs once per session
before the first comparison. Review the resulting ``git diff`` on
``_cli_help_golden.json`` (written with ``indent=1, sort_keys=True``,
``ensure_ascii=False`` so Chinese help text stays readable and the diff is
reviewable line-by-line).
"""

from __future__ import annotations

import difflib
import importlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import click
import pytest
from click.testing import CliRunner
from loguru import logger

GOLDEN_PATH = Path(__file__).parent / "_cli_help_golden.json"

EXIT_SUFFIX = ":exit"
PROG = "<prog>"
REGENERATE_ENV_VAR = "AO_UPDATE_HELP_GOLDEN"
_USAGE_RE = re.compile(r"^Usage: .*$", re.MULTILINE)
# `--dm_type` renders click.Choice(DM_TYPES) -- the *live* DM registry -- so its contents depend on
# which modules a process imported first: DM types self-register as an import side effect, and
# runner_common deliberately snapshots the registry before importing asyn_micro_dm. That makes the
# choice list import-order dependent, not a behavioural contract. Scoped to --dm_type on purpose:
# a broader "any [a|b|c]" rule would also hide changes to STATIC choice lists (--camera-type,
# --wfs_type, --objective), which this golden exists to catch.
_DM_TYPE_CHOICE_RE = re.compile(r"^(\s+--dm_type\s+)\[[a-z_0-9|]+\]", re.MULTILINE)

TOOLS_SLM_MODULES: list[str] = [
    "calibration",
    "gray_response",
    "phase_capture",
    "slm_beam_extent",
    "slm_diagnose",
    "slm_exposure_check",
    "slm_lut_runner",
    "slm_panel_locate",
    "slm_phase_resolution",
    "slm_phase_response",
    "slm_tilt_probe",
    "slm_wfs_probe",
    "slm_wfs_reference",
    "slm_zernike_correction",
    "slm_zernike_response",
    "slm_zernike_sweep_probe",
]

#: Currently-colliding resolved click names in Group B, measured on the
#: unmodified tree. ``main`` is the click-default name derived from a bare
#: ``@click.command``; ``run`` is an explicit ``@click.command("run")``.
EXPECTED_COLLIDING_NAMES: dict[str, int] = {"main": 13, "run": 3}


def normalize(text: str) -> str:
    """Normalise the two parts of ``--help`` that are not behavioural contract.

    The ``Usage:`` line carries the prog name, and ``--dm_type``'s choice list is
    the live DM registry (see ``_DM_TYPE_CHOICE_RE``). Everything else - flag
    names, their order, types, defaults and help prose - is compared verbatim.
    """
    text = _USAGE_RE.sub(f"Usage: {PROG}", text)
    return _DM_TYPE_CHOICE_RE.sub(r"\g<1>[<dm-types>]", text)


def invoke_help(cmd: click.Command) -> tuple[str, int]:
    """Invoke ``--help`` under a fixed prog name; return (normalised, exit code)."""
    result = CliRunner().invoke(cmd, ["--help"], prog_name=PROG)
    return normalize(result.output), int(result.exit_code)


def main_group_commands() -> dict[str, click.Command]:
    """Group A: the commands registered on the top-level click group.

    Keyed ``main:<registered-name>`` so the two groups cannot collide.
    """
    from ao_shaping.main import cli

    return {f"main:{name}": cli.commands[name] for name in sorted(cli.commands)}


def tools_slm_commands() -> dict[str, click.Command]:
    """Group B: every ``click.Command`` attribute of the 17 tools/slm modules."""
    found: dict[str, click.Command] = {}
    for module in TOOLS_SLM_MODULES:
        mod = importlib.import_module(f"ao_shaping.tools.slm.{module}")
        for attr, obj in sorted(vars(mod).items()):
            if isinstance(obj, click.Command):
                found[f"tools.slm.{module}:{attr}"] = obj
    return dict(sorted(found.items()))


def collect() -> dict[str, Any]:
    """Capture every command's normalised ``--help`` plus its exit code."""
    golden: dict[str, Any] = {}
    for key, cmd in {**main_group_commands(), **tools_slm_commands()}.items():
        out, code = invoke_help(cmd)
        golden[key] = out
        golden[f"{key}{EXIT_SUFFIX}"] = code
    return golden


def load_golden() -> dict[str, Any]:
    """Load the committed snapshot (fails the test if it is missing)."""
    if not GOLDEN_PATH.exists():
        pytest.fail(
            f"Golden file missing: {GOLDEN_PATH}. Regenerate it with "
            f"{REGENERATE_ENV_VAR}=1 ./.venv/bin/python -m pytest {__file__} -q"
        )
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def all_golden_keys() -> list[str]:
    """Every key in the snapshot: one help key and one ``:exit`` key per command."""
    return sorted(load_golden().keys())


def unified_diff(expected: Any, actual: Any, key: str) -> str:
    """Unified diff so a reviewer can see exactly which flag moved."""
    return "\n".join(
        difflib.unified_diff(
            str(expected).splitlines(),
            str(actual).splitlines(),
            fromfile=f"golden[{key}]",
            tofile=f"current[{key}]",
            lineterm="",
        )
    )


@pytest.fixture(scope="module")
def current() -> dict[str, Any]:
    """Live capture, taken once per session (plus opt-in regeneration)."""
    live = collect()
    if os.environ.get(REGENERATE_ENV_VAR) == "1":
        GOLDEN_PATH.write_text(
            json.dumps(live, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.warning(
            "{} was set: rewrote {} with {} keys",
            REGENERATE_ENV_VAR,
            GOLDEN_PATH,
            len(live),
        )
    return live


@pytest.mark.parametrize("key", all_golden_keys())
def test_cli_help_matches_golden(key: str, current: dict[str, Any]) -> None:
    """Each command contributes two cases: its help text and its exit code."""
    golden = load_golden()
    expected = golden.get(key)
    actual = current.get(key)

    if expected == actual:
        return

    if key.endswith(EXIT_SUFFIX):
        pytest.fail(
            f"exit-code drift for {key}: golden={expected!r} current={actual!r}"
        )

    if actual is None:
        pytest.fail(f"command {key} no longer exists (no click.Command found)")

    pytest.fail(f"--help drift for {key}:\n{unified_diff(expected, actual, key)}")


def test_golden_covers_exactly_the_current_command_set(current: dict[str, Any]) -> None:
    """Adding or removing a command must not slip through unnoticed."""
    golden = load_golden()
    golden_keys = set(golden)
    current_keys = set(current)

    assert current_keys - golden_keys == set(), (
        "new command(s) appeared that the golden does not cover: "
        f"{sorted(current_keys - golden_keys)}"
    )
    assert golden_keys - current_keys == set(), (
        "golden covers command(s) that no longer exist: "
        f"{sorted(golden_keys - current_keys)}"
    )


def test_all_help_exit_codes_are_zero() -> None:
    """Every captured ``--help`` must succeed; a non-zero exit would be a finding."""
    golden = load_golden()
    bad = {
        key: value
        for key, value in golden.items()
        if key.endswith(EXIT_SUFFIX) and value != 0
    }
    assert bad == {}, f"non-zero --help exit codes frozen in golden: {bad}"


def test_command_name_collisions_are_frozen() -> None:
    """Freeze the currently-colliding resolved command name multiset (Group B).

    Measured on the unmodified tree: twelve modules resolve to the bare name
    ``main`` (a bare ``@click.command`` derives the name from the function), and
    three resolve to ``run`` (``gray_response``, ``phase_capture``,
    ``slm_lut_runner``). ``calibration.main_shift_calib`` is the only Group B
    command with an explicit name; it resolves to ``main-shift-calib``.

    Grouping these tools into a single ``slm`` click group is explicitly
    forbidden by ``src/ao_shaping/tools/slm/TODO.md`` section 6 ("--help
    不可用, 且 main.py 会从 2 个变成 22 个工具导入"); doing it correctly would require
    adding an explicit ``name=`` to every single command. This test is the
    tripwire that makes such a rename impossible to do quietly.
    """
    resolved = {key: cmd.name for key, cmd in tools_slm_commands().items()}
    counter = Counter(resolved.values())
    colliding = {name: n for name, n in counter.items() if n > 1}

    assert colliding == EXPECTED_COLLIDING_NAMES, (
        "Group B resolved command-name collision multiset changed: "
        f"golden={EXPECTED_COLLIDING_NAMES} current={colliding}. "
        f"Resolved names: {dict(sorted(resolved.items()))}"
    )
