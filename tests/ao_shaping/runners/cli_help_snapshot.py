"""Shared ``--help`` capture and golden-file IO for the CLI contract tests.

Why one file
------------
Before 2026-10-25 the repo carried **two** golden snapshots of overlapping
command sets, each with its own capture scheme:

* ``test_cli_contract_freeze.py`` (R-35) froze the 19 ``main.py`` commands at a
  forced ``terminal_width=88`` through ``click.Command.get_help``, into
  ``cli_help_golden.json``.
* ``test_cli_help_golden.py`` (T1.1 / T1.3) froze the same 19 commands *plus*
  the 17 ``tools`` probe commands *plus* their ``--help`` exit codes, through
  ``CliRunner`` at the ambient width, into ``_cli_help_golden.json``.

Neither file was the whole surface, the stored text of the shared commands
differed (Usage line, wrapping, ``--dm-type`` placeholder, path separators), so
one intentional CLI change had to be regenerated twice and a reviewer diffing
one file could not see the other. They are merged here: **one file**
(``cli_help_golden.json``), **one capture scheme**, two consumers:

* ``test_cli_help_golden.py`` - primary consumer, parametrised over every key.
* ``test_cli_contract_freeze.py`` - pins its frozen 19-command inventory and
  each command's text against the same snapshot.

Capture scheme (the ``CliRunner`` one, kept because it is the portable half)
---------------------------------------------------------------------------
* ``Usage: <prog>`` - the prog name is not part of the contract.
* ``--dm-type [<dm-types>]`` - the choice list is the live DM registry, whose
  contents depend on which modules a process imported first. Its
  *declaration* is pinned separately (per command, in the contract-freeze
  test).
* ``[default: ...]`` path separators - click renders a ``click.Path`` default
  through ``os.path.sep``, so the raw text would be platform-dependent.

Everything else - flag names, their order, types, defaults, help prose - is
compared verbatim. Each command also stores an integer exit code under a
``:exit`` key, so every command contributes two parametrized cases.

Regeneration (opt-in only)
--------------------------
The default is **compare**. To re-capture after an intentional CLI change::

    AO_UPDATE_HELP_GOLDEN=1 ./.venv/bin/python -m pytest tests/ao_shaping/runners/test_cli_help_golden.py -q

The rewrite happens in a module-scoped fixture, so it runs once per session
before the first comparison. Review the resulting ``git diff`` on
``cli_help_golden.json`` (written with ``indent=1, sort_keys=True``,
``ensure_ascii=False`` so Chinese help text stays readable and the diff is
reviewable line-by-line).
"""

from __future__ import annotations

import difflib
import importlib
import json
import os
import re
from pathlib import Path
from typing import Any

import click
from click.testing import CliRunner
from loguru import logger

#: The one golden snapshot, shared by both CLI contract test modules.
GOLDEN_PATH = Path(__file__).parent / "cli_help_golden.json"

EXIT_SUFFIX = ":exit"
PROG = "<prog>"
REGENERATE_ENV_VAR = "AO_UPDATE_HELP_GOLDEN"
_USAGE_RE = re.compile(r"^Usage: .*$", re.MULTILINE)
# `--dm-type` renders click.Choice(DM_TYPES) -- the *live* DM registry -- so its contents depend on
# which modules a process imported first: DM types self-register as an import side effect, and
# runner_common deliberately snapshots the registry before importing asyn_micro_dm. That makes the
# choice list import-order dependent, not a behavioural contract. Scoped to --dm-type on purpose:
# a broader "any [a|b|c]" rule would also hide changes to STATIC choice lists (--camera-type,
# --wfs-type, --objective), which this golden exists to catch.
_DM_TYPE_CHOICE_RE = re.compile(r"^(\s+--dm-type\s+)\[[a-z_0-9|]+\]", re.MULTILINE)
# Click renders a `click.Path` default through `os.path.sep`, so one and the same command
# prints `data\x.csv` on Windows and `data/x.csv` everywhere else. Comparing that verbatim
# would make this snapshot machine-dependent: the committed one was produced off-Windows, so
# regenerating on Windows flips every path default and reports drift that is not a
# behavioural change. Scoped to the `[default: ...]` suffix on purpose -- help prose elsewhere
# legitimately contains both slashes (e.g. "input/output", "miicam/daheng"), so a repo-wide
# backslash rule would silently hide real edits.
_DEFAULT_PATH_RE = re.compile(r"(\[default:[^\]\n]*(?:\n(?![ \t]*-)[^\]\n]*)*)\]")

TOOLS_SLM_MODULES: list[str] = [
    "tools.slm.calibration.calibration",
    "tools.slm.calibration.gray_response",
    "tools.slm.calibration.phase_capture",
    "tools.slm.calibration.lut_runner",
    "tools.slm.panel.beam_extent",
    "tools.slm.panel.diagnose",
    "tools.slm.panel.panel_locate",
    "tools.slm.panel.phase_resolution",
    "tools.slm.panel.phase_response",
    "tools.slm.panel.tilt_probe",
    "tools.ccd.exposure_check",
    "tools.wfs.wfs_probe",
    "tools.wfs.wfs_reference",
    "tools.wfs.zernike_correction",
    "tools.wfs.zernike_response",
    "tools.zernike.sweep_probe",
]


def normalize(text: str) -> str:
    """Normalise the three parts of ``--help`` that are not behavioural contract.

    The ``Usage:`` line carries the prog name, ``--dm-type``'s choice list is the live DM
    registry (see ``_DM_TYPE_CHOICE_RE``), and a ``click.Path`` default carries the platform
    separator (see ``_DEFAULT_PATH_RE``). Everything else - flag names, their order, types,
    defaults and help prose - is compared verbatim.
    """
    text = _USAGE_RE.sub(f"Usage: {PROG}", text)
    text = _DM_TYPE_CHOICE_RE.sub(r"\g<1>[<dm-types>]", text)
    return _DEFAULT_PATH_RE.sub(lambda m: m.group(1).replace("\\", "/") + "]", text)


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
    """Group B: every ``click.Command`` attribute of the tools probe modules."""
    found: dict[str, click.Command] = {}
    for module in TOOLS_SLM_MODULES:
        mod = importlib.import_module("ao_shaping." + module)
        for attr, obj in sorted(vars(mod).items()):
            if isinstance(obj, click.Command):
                found[f"{module}:{attr}"] = obj
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
        raise AssertionError(
            f"Golden file missing: {GOLDEN_PATH}. Regenerate it with "
            f"{REGENERATE_ENV_VAR}=1 ./.venv/bin/python -m pytest "
            "tests/ao_shaping/runners/test_cli_help_golden.py -q"
        )
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


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


def rewrite_golden(live: dict[str, Any]) -> None:
    """Opt-in regeneration: write ``live`` as the new snapshot (once per session)."""
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


def regeneration_requested() -> bool:
    """True when the operator opted into rewriting the golden for this run."""
    return os.environ.get(REGENERATE_ENV_VAR) == "1"
