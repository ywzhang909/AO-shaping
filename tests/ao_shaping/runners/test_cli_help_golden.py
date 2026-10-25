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

The capture scheme, the normalisation rules and the golden file itself are
shared with ``test_cli_contract_freeze.py`` (R-35) through
``cli_help_snapshot``: since 2026-10-25 there is a single ``cli_help_golden.json``
instead of the former width-88 / CliRunner pair. See that module for the merge
rationale.

Regeneration (opt-in only)
-------------------------
The default is **compare**. To re-capture after an intentional CLI change::

    AO_UPDATE_HELP_GOLDEN=1 ./.venv/bin/python -m pytest tests/ao_shaping/runners/test_cli_help_golden.py -q

The rewrite happens in a module-scoped fixture, so it runs once per session
before the first comparison. Review the resulting ``git diff`` on
``cli_help_golden.json`` (written with ``indent=1, sort_keys=True``,
``ensure_ascii=False`` so Chinese help text stays readable and the diff is
reviewable line-by-line).
"""

from __future__ import annotations

import os
from collections import Counter

import pytest

from tests.ao_shaping.runners.cli_help_snapshot import (
    EXIT_SUFFIX,
    GOLDEN_PATH,
    REGENERATE_ENV_VAR,
    collect,
    load_golden,
    regeneration_requested,
    rewrite_golden,
    tools_slm_commands,
    unified_diff,
)

#: Group B's resolved click names, keyed ``tools.slm.<module>:<attr>``, measured
#: on the unmodified tree. The merged golden normalises the ``Usage:`` line to
#: ``Usage: <prog>`` (prog-name independence), so this mapping is what still
#: pins *which* name each probe command resolves to - a bare
#: ``@click.command`` derives it from the function name, and an explicit
#: ``@click.command("run")`` overrides it.
EXPECTED_RESOLVED_NAMES: dict[str, str] = {
    "tools.ccd.exposure_check:main": "main",
    "tools.slm.calibration.calibration:main": "main",
    "tools.slm.calibration.calibration:main_shift_calib": "main-shift-calib",
    "tools.slm.calibration.gray_response:run": "run",
    "tools.slm.calibration.lut_runner:run": "run",
    "tools.slm.calibration.phase_capture:run": "run",
    "tools.slm.panel.beam_extent:main": "main",
    "tools.slm.panel.diagnose:main": "main",
    "tools.slm.panel.panel_locate:main": "main",
    "tools.slm.panel.phase_resolution:main": "main",
    "tools.slm.panel.phase_response:main": "main",
    "tools.slm.panel.tilt_probe:main": "main",
    "tools.wfs.wfs_probe:main": "main",
    "tools.wfs.wfs_reference:main": "main",
    "tools.wfs.zernike_correction:main": "main",
    "tools.wfs.zernike_response:main": "main",
    "tools.zernike.sweep_probe:main": "main",
}


@pytest.fixture(scope="module")
def current() -> dict[str, str | int]:
    """Live capture, taken once per session (plus opt-in regeneration)."""
    live = collect()
    if regeneration_requested():
        rewrite_golden(live)
    return live


@pytest.mark.parametrize("key", sorted(load_golden().keys()))
def test_cli_help_matches_golden(key: str, current: dict[str, str | int]) -> None:
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


def test_golden_covers_exactly_the_current_command_set(
    current: dict[str, str | int],
) -> None:
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


def test_group_b_resolved_command_names_are_frozen() -> None:
    """Freeze the resolved click name of every Group B command.

    Measured on the unmodified tree: thirteen modules resolve to the bare name
    ``main`` (a bare ``@click.command`` derives the name from the function), and
    three resolve to ``run`` (``gray_response``, ``phase_capture``,
    ``slm_lut_runner``). ``calibration.main_shift_calib`` is the only Group B
    command with an explicit name; it resolves to ``main-shift-calib``.

    Grouping these tools into a single ``slm`` click group is explicitly
    forbidden by ``src/ao_shaping/tools/slm/TODO.md`` section 6 ("--help
    不可用, 且 main.py 会从 2 个变成 22 个工具导入"); doing it correctly would require
    adding an explicit ``name=`` to every single command. This test is the
    tripwire that makes such a rename impossible to do quietly - and since the
    merged golden normalises the ``Usage:`` line, it is the *only* place the
    full mapping (not just the colliding multiset) is pinned.
    """
    resolved = {key: cmd.name for key, cmd in tools_slm_commands().items()}
    assert resolved == EXPECTED_RESOLVED_NAMES, (
        "Group B resolved command names changed: "
        f"golden={EXPECTED_RESOLVED_NAMES} current={resolved}"
    )
    colliding = {name: n for name, n in Counter(resolved.values()).items() if n > 1}
    assert colliding == {"main": 13, "run": 3}, (
        f"Group B collision multiset changed: current={colliding}"
    )
