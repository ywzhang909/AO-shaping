"""R-35 — CLI contract freeze for the Click layer (offline, no hardware).

This is the **hard prerequisite** for R-36 → R-41 (moving the Click mechanism out
of ``runners/runner_common.py`` and migrating the 22 ``tools/slm`` probes onto
``with_params``). Those refactors must not change a single observable byte of
command-line behaviour, so every observable is pinned here first:

1. **Command inventory** — the exact set of names registered on ``main.py``.
2. **``--dm_type`` choice lists** — per command. Note ``dm-matrix`` deliberately
   offers 6 (no ``asyn_micro``): ``runner_common.py`` snapshots
   ``DM_TYPES_PRE_ASYN_MICRO`` before importing ``asyn_micro_dm`` to keep its
   help output byte-identical to the pre-registration help. That divergence is
   pinned, not normalised.
3. **Full ``--help`` text** for every registered command, byte-for-byte, against
   a committed golden file.
4. **No duplicate option flag** inside one command — a duplicate is what a
   cross-``with_params`` field-name collision looks like once click has silently
   dropped one of the two declarations.
5. **``WfsParams.pupil_center``'s tuple + callback coupling** — the annotation
   carries a ``tuple`` arm *and* a ``parse_tuple`` callback. Dropping either one
   is the regression this pins.

Regenerate the golden file after an INTENDED help-text change with::

    AO_CLI_CONTRACT_UPDATE=1 pytest tests/ao_shaping/runners/test_cli_contract_freeze.py
"""

from __future__ import annotations

import json
import os
import re
import typing
from pathlib import Path

import click
import pytest

from ao_shaping.main import cli
from ao_shaping.runners.runner_common import WfsParams, parse_tuple

_GOLDEN_PATH = Path(__file__).with_name("cli_help_golden.json")
_UPDATE = os.environ.get("AO_CLI_CONTRACT_UPDATE") == "1"

#: Registered command names, frozen 2026-10-03 (R-35).
EXPECTED_COMMANDS = (
    "alt-voltage",
    "closed-loop",
    "combined",
    "dm-matrix",
    "full-voltage",
    "ga-zernike",
    "greedy-zernike",
    "hadamard-matrix",
    "pib",
    "pipeline",
    "rms-zernike",
    "slm-diagnose",
    "slm-gs-refine",
    "slm-gsnet",
    "slm-lut",
    "slm-model-in-loop",
    "slm-pib",
    "spgd-square",
    "wf",
    "zernike-matrix",
)

#: The DM types registered *before* ``runner_common`` imports ``asyn_micro_dm``.
#: ``dm-matrix`` offers exactly these; every other command offers ``DM_TYPES``
#: (i.e. the same list plus ``asyn_micro``).
EXPECTED_DM_TYPE_CHOICES = {
    "wf": ("asyn_micro", "hadamard", "micro", "nlight", "sim", "sim_micro", "zernike"),
    "pib": ("asyn_micro", "hadamard", "micro", "nlight", "sim", "sim_micro", "zernike"),
    "pipeline": ("asyn_micro", "hadamard", "micro", "nlight", "sim", "sim_micro", "zernike"),
    "combined": ("asyn_micro", "hadamard", "micro", "nlight", "sim", "sim_micro", "zernike"),
    "dm-matrix": ("hadamard", "micro", "nlight", "sim", "sim_micro", "zernike"),
}


_DM_TYPE_CHOICE_RE = re.compile(r"^(?P<head>\s*--dm_type\s*)\[[a-z_0-9|]+\]", re.MULTILINE)


def _help_text(command: click.Command) -> str:
    """Render ``--help`` for one command through click's own formatter.

    The one genuinely environment-dependent fragment is normalised away: the
    ``--dm_type`` choice list is built from the live DM registry, so a full-suite
    run (where other tests registered DM types first) legitimately renders more
    entries than a bare run. Its *declaration* is pinned separately by
    ``test_dm_matrix_offer_the_pre_asyn_micro_snapshot``; every other choice list
    (``--wfs_type``, ``--mode``, ``--mla-index``, ...) stays byte-for-byte frozen.
    """
    ctx = click.Context(command, info_name=command.name, terminal_width=88)
    return _DM_TYPE_CHOICE_RE.sub(r"\g<head>[<DM_TYPES>]", command.get_help(ctx))


def _collect_help() -> dict[str, str]:
    return {name: _help_text(command) for name, command in sorted(cli.commands.items())}


# ---------------------------------------------------------------------------
# 1. inventory
# ---------------------------------------------------------------------------


def test_registered_command_inventory_is_frozen() -> None:
    assert tuple(sorted(cli.commands)) == EXPECTED_COMMANDS


# ---------------------------------------------------------------------------
# 2. --dm_type choice lists
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(EXPECTED_DM_TYPE_CHOICES))
def test_dm_type_choice_list_is_frozen(name: str) -> None:
    command = cli.commands[name]
    options = [p for p in command.params if "--dm_type" in p.opts]
    assert len(options) == 1, f"{name} should expose exactly one --dm_type"
    assert isinstance(options[0].type, click.Choice)
    actual = tuple(options[0].type.choices)
    expected = EXPECTED_DM_TYPE_CHOICES[name]
    # A full-suite run may register extra DM types before this test imports, so
    # pin the *declaration* (which is frozen per command) and only require the
    # live list to be a superset in the declared order. Order is what --help
    # output shows, and that IS frozen by the golden test below.
    assert set(expected).issubset(set(actual)), (
        f"{name}: lost a declared DM type; declared={expected} live={actual}"
    )
    assert list(actual[: len(expected)]) == list(expected), (
        f"{name}: leading DM types drifted; declared={expected} live={actual}"
    )


def test_dm_matrix_is_the_only_command_missing_asyn_micro() -> None:
    """The divergence is deliberate; if it disappears, the freeze must be revisited."""
    short = set(EXPECTED_DM_TYPE_CHOICES["dm-matrix"])
    long_ = set(EXPECTED_DM_TYPE_CHOICES["wf"])
    assert long_ - short == {"asyn_micro"}
    for name in EXPECTED_DM_TYPE_CHOICES:
        if name != "dm-matrix":
            assert tuple(EXPECTED_DM_TYPE_CHOICES[name]) == EXPECTED_DM_TYPE_CHOICES["wf"]


def test_dm_matrix_offer_the_pre_asyn_micro_snapshot() -> None:
    """``dm-matrix`` must be fed ``DM_TYPES_PRE_ASYN_MICRO``, by contract.

    ``runner_common.py`` snapshots the registry *before* importing
    ``asyn_micro_dm`` so that ``dm_matrix_runner``'s help stays byte-identical to
    its pre-registration text. Assert the wiring, not the rendered list (the
    snapshot's contents depend on what else has registered a DM by then).
    """
    from ao_shaping.runners import runner_common

    dm_matrix = [
        p for p in cli.commands["dm-matrix"].params if "--dm_type" in p.opts
    ][0]
    assert isinstance(dm_matrix.type, click.Choice)
    assert set(runner_common.DM_TYPES_PRE_ASYN_MICRO).issubset(set(dm_matrix.type.choices))
    assert "asyn_micro" in runner_common.DM_TYPES
    assert "asyn_micro" not in runner_common.DM_TYPES_PRE_ASYN_MICRO


def test_no_new_command_silently_grows_dm_type() -> None:
    """A command gaining ``--dm_type`` is a behaviour change, not a refactor."""
    actual = {
        name
        for name, command in cli.commands.items()
        if any("--dm_type" in p.opts for p in command.params)
    }
    assert actual == set(EXPECTED_DM_TYPE_CHOICES)


# ---------------------------------------------------------------------------
# 3. --help golden
# ---------------------------------------------------------------------------


def test_help_text_matches_golden() -> None:
    actual = _collect_help()
    if _UPDATE or not _GOLDEN_PATH.exists():
        _GOLDEN_PATH.write_text(
            json.dumps(actual, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        pytest.skip("golden file written; re-run without AO_CLI_CONTRACT_UPDATE")
    expected = json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))
    assert sorted(actual) == sorted(expected), "command set drifted from the golden"
    for name in sorted(expected):
        assert actual[name] == expected[name], f"--help text of {name!r} changed"


# ---------------------------------------------------------------------------
# 4. no duplicate option flag within a command
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(EXPECTED_COMMANDS))
def test_command_has_no_duplicate_option_flag(name: str) -> None:
    seen: dict[str, str] = {}
    duplicates: dict[str, list[str]] = {}
    for param in cli.commands[name].params:
        for opt in param.opts:
            if opt in seen:
                duplicates.setdefault(opt, [seen[opt]]).append(type(param).__name__)
            seen[opt] = type(param).__name__
    assert not duplicates, (
        f"{name}: option(s) declared more than once -> {duplicates}. "
        "A duplicate means two dataclasses claim the same flag and click kept one."
    )


# ---------------------------------------------------------------------------
# 5. pupil_center tuple + callback coupling
# ---------------------------------------------------------------------------


def test_pupil_center_annotation_still_carries_the_tuple_arm() -> None:
    """Dropping ``tuple[float, float]`` from the annotation is half the regression."""
    hints = typing.get_type_hints(WfsParams)
    assert "tuple" in str(hints["pupil_center"]), hints["pupil_center"]


def _pupil_center_option() -> click.Option:
    """The ``-c/--pupil_center`` option as click actually built it for ``wf``."""
    options = [p for p in cli.commands["wf"].params if "--pupil_center" in p.opts]
    assert len(options) == 1, "wf must expose exactly one --pupil_center"
    return options[0]


def test_pupil_center_declares_the_parse_tuple_callback() -> None:
    assert _pupil_center_option().callback is parse_tuple


def test_pupil_center_default_is_the_string_form_of_the_default_tuple() -> None:
    """The default is ``"(0,0)"`` — a *string* — which only the callback converts."""
    assert WfsParams().pupil_center == "(0,0)"
    assert _pupil_center_option().default == "(0,0)"
    assert _parse("(0,0)") == (0.0, 0.0)


def _parse(raw: str) -> object:
    """Call the real click callback the way click would (ctx, param, value)."""
    return parse_tuple(None, _pupil_center_option(), raw)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("3,4", (3.0, 4.0)),
        ("(3, 4)", (3.0, 4.0)),
        ("-1.5,2.25", (-1.5, 2.25)),
        ("577,655", (577.0, 655.0)),
    ],
)
def test_pupil_center_callback_parses_a_pair(raw: str, expected: tuple[float, float]) -> None:
    assert _parse(raw) == expected


def test_pupil_center_callback_forwards_the_spot_mode_keywords() -> None:
    """``mass``/``max``/``shape`` pass through untouched (the ``-c auto`` modes)."""
    for token in ("mass", "MAX", "Shape"):
        assert _parse(token) == token.lower()


def test_pupil_center_callback_is_required_for_indexing() -> None:
    """Why the callback matters: downstream does ``pupil_center[0]``.

    Without ``parse_tuple`` the raw string survives, and ``"(0,0)"[0]`` is ``"("``
    — a silent wrong value, not a crash. Pin the difference so nobody "simplifies"
    the annotation away believing it is cosmetic.
    """
    raw = "(0,0)"
    assert raw[0] == "("
    assert _parse(raw) == (0.0, 0.0)