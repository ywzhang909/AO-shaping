"""R-37 step 0: pin every ``tools/slm`` probe's flag surface before migrating.

R-37 moves ~200 flag declarations from hand-written ``@click.option`` stacks onto
the shared ``with_params`` convention (TODO R-36). That is 19 probe files and 285
declared flags, and -- unlike the 19 ``main.py`` commands -- **the probes are not
covered by R-35's golden**, because they are ``python -m`` entry points rather than
registered Click commands. So before touching any of them, the surface they
expose has to be pinned.

Why this has to see BOTH declaration forms
------------------------------------------
The migration moves each flag between two syntactic homes::

    @click.option("--exposure-ms", type=float, default=3.0, help="...")   # before
    exposure_ms: Annotated[float, option("--exposure-ms", help="...")] = 3.0   # after

So a scanner that only understands the decorator form reports **zero flags** the
moment a file is migrated -- it would look like the probe had lost every option.
:func:`_declarations` therefore matches on the *called name* rather than on the
syntactic context: anything that calls a function named ``option`` (bare, as
imported from :mod:`ao_shaping.utils.cli.params`, or as ``click.option``) or
``add_argument`` contributes its flags. Both forms land in the same set, so the
golden compares like with like and the files can be migrated one at a time.

Why the flag set and not the rendered help
------------------------------------------
The rendered ``--help`` is kept on disk (``probe_help_golden.json``) as the
reference for regeneration, and ``slm_exposure_check`` was migrated with its help
**byte-identical**. But it cannot be the always-on test: collecting 19 subprocesses
costs ~9 minutes, because each one pays the ~28 s ``import ao_shaping`` baseline.
At that price the guard would simply get switched off.

So the always-on guard derives the flags from the **source AST**. It is fast, needs
no subprocess and no device, and it catches the failure mode that actually matters
in a 287-flag migration: a flag silently dropped, renamed, duplicated, or spelled
two ways.

The rendered help stays reproducible with::

    AO_PROBE_HELP_UPDATE=1 pytest tests/ao_shaping/tools/slm/test_probe_flags.py

which rewrites ``probe_help_golden.json`` and then asserts that nothing changed
beyond ordering. Order of the option block is deliberately *not* asserted: click
accumulates ``__click_params__`` in reverse, so a decorator stack and
``with_params`` can render the same flags in a different order. R-35 documented the
same effect for the registered commands.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
SLM_PKG_DIR = REPO / "src" / "ao_shaping" / "tools" / "slm"
GOLDEN = Path(__file__).with_name("probe_help_golden.json")
UPDATE = os.environ.get("AO_PROBE_HELP_UPDATE") == "1"

#: Every probe that is a ``python -m`` entry point. The six library modules
#: (bench_metrics, bench_probe, scan_analysis, zernike_common, snr_probe,
#: delta_explorer) are deliberately absent: they have no CLI.
PROBES = [
    "calibration",
    "gray_response",
    "phase_capture",
    "slm_abba_probe",
    "slm_beam_extent",
    "slm_diagnose",
    "slm_drift_probe",
    "slm_exposure_check",
    "slm_floor_probe",
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

_FLAG = re.compile(r"^--[a-z0-9][a-z0-9-]*$")
_HELP_FLAG = re.compile(r"--[a-z0-9][a-z0-9-]*")

#: Calls that declare a CLI flag. Matched by *name* so that the decorator form
#: (``@click.option(...)``) and the dataclass form (``option(...)`` inside an
#: ``Annotated``) are recognised identically -- see the module docstring.
_OPTION_CALLS = frozenset({"option", "add_argument"})

#: ``calibration`` overrides click's built-in help flag via
#: ``@click.command(context_settings=dict(help_option_names=["-h", "--help"]))``,
#: so that spelling is a real part of its flag surface. Scoped to that keyword
#: only -- the rest of ``context_settings`` is not a flag declaration.
_HELP_OPTION_CALLS = frozenset({"command", "group"})


def _called_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _split_flags(text: str) -> list[str]:
    """``--display/--no-display`` declares TWO flags; track each half.

    Click's paired-boolean syntax is one string but two flags, and a migration
    that rewrote it as ``"--display", "--no-display"`` (or the reverse) would
    otherwise look like a surface change -- or like nothing at all.
    """
    return [part for part in text.split("/") if _FLAG.match(part)]


def _flag_strings(node: ast.AST) -> list[str]:
    """Flag-looking string constants anywhere inside ``node``.

    A flag may sit in a positional or in a keyword (``name="--x"``); scanning the
    whole call subtree covers both without having to enumerate every spelling.
    """
    out: list[str] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            out += _split_flags(sub.value)
    return out


def _declarations(tree: ast.AST) -> list[list[str]]:
    """One entry per option declaration, so duplicates stay visible."""
    found: list[list[str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _called_name(node)
        if name in _OPTION_CALLS:
            flags = _flag_strings(node)
            if flags:
                found.append(flags)
            continue
        # ``calibration`` renames click's built-in help flag via
        # ``@click.command(context_settings=dict(help_option_names=["-h", "--help"]))``.
        # The keyword belongs to the inner ``dict(...)``, not to ``click.command``,
        # so match it on any call rather than on a particular callee.
        for keyword in node.keywords:
            if keyword.arg == "help_option_names":
                flags = _flag_strings(keyword.value)
                if flags:
                    found.append(flags)
    return found


def _parse(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))


def _declared_flags(path: Path) -> list[str]:
    return sorted({flag for group in _declarations(_parse(path)) for flag in group})


def _golden() -> dict[str, dict[str, object]]:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def test_the_flag_scanner_sees_both_declaration_forms() -> None:
    """The guard is only a migration gate if it understands before *and* after.

    A scanner that understood just the decorator form would report zero flags for
    an already-migrated file, i.e. it would cry wolf on every migrated probe.
    """
    decorator_form = ast.parse(
        "@click.command()\n"
        '@click.option("--exposure-ms", type=float, default=3.0, help="h")\n'
        '@click.option("--display/--no-display", default=False, help="h")\n'
        "def main(exposure_ms, display):\n    pass\n"
    )
    dataclass_form = ast.parse(
        "from dataclasses import dataclass\n"
        "from typing import Annotated\n\n\n"
        "@dataclass\n"
        "class P:\n"
        '    exposure_ms: Annotated[float, option("--exposure-ms", help="h")] = 3.0\n'
        '    display: Annotated[bool, option("--display/--no-display", help="h")] = False\n'
    )
    for form in (decorator_form, dataclass_form):
        flags = sorted({f for group in _declarations(form) for f in group})
        assert flags == ["--display", "--exposure-ms", "--no-display"], flags

    # argparse is covered too, and keeps its own distinct spelling.
    argparse_form = ast.parse(
        'p.add_argument("--cam-id", type=int, default=0, help="h")\n'
        'p.add_argument("--no-hw", action="store_true")\n'
    )
    flags = sorted({f for group in _declarations(argparse_form) for f in group})
    assert flags == ["--cam-id", "--no-hw"], flags

    # calibration overrides click's built-in help flag; that counts as surface.
    help_names_form = ast.parse(
        '@click.command(context_settings=dict(help_option_names=["-h", "--help"]))\n'
        '@click.option("--exposure-ms", default=1.2, help="h")\n'
        "def main(exposure_ms):\n    pass\n"
    )
    flags = sorted({f for group in _declarations(help_names_form) for f in group})
    assert flags == ["--exposure-ms", "--help"], flags


def test_the_probe_list_matches_the_package() -> None:
    """A new probe must be added here or it gets no protection at all."""
    on_disk = {
        p.stem
        for p in SLM_PKG_DIR.glob("*.py")
        if p.stem != "__init__"
        and "__main__" in p.read_text(encoding="utf-8-sig", errors="replace")
    }
    assert set(PROBES) == on_disk, (
        f"only in PROBES: {sorted(set(PROBES) - on_disk)}; "
        f"only on disk: {sorted(on_disk - set(PROBES))}"
    )


@pytest.mark.parametrize("probe", PROBES)
def test_probe_declared_flags_are_unchanged(probe: str) -> None:
    """The always-on guard: what the source DECLARES, not what help renders."""
    expected = _golden()[probe]["declared"]
    actual = _declared_flags(SLM_PKG_DIR / f"{probe}.py")
    assert actual == expected, (
        f"{probe}: flag surface changed.\n"
        f"  dropped:  {sorted(set(expected) - set(actual))}\n"
        f"  added:    {sorted(set(actual) - set(expected))}"
    )


@pytest.mark.parametrize("probe", PROBES)
def test_probe_has_no_duplicate_flag(probe: str) -> None:
    """Click silently keeps the last of two identical flags."""
    groups = _declarations(_parse(SLM_PKG_DIR / f"{probe}.py"))
    counts: dict[str, int] = {}
    for group in groups:
        for flag in group:
            counts[flag] = counts.get(flag, 0) + 1
    dupes = sorted(flag for flag, n in counts.items() if n > 1)
    assert not dupes, f"{probe}: flag(s) declared twice: {dupes}"


@pytest.mark.parametrize("probe", PROBES)
def test_probe_body_literals_are_unchanged(probe: str) -> None:
    """The flag guard is blind to a body rename; this is the other half.

    Turning ``def main(a, b)`` into ``def main(params)`` means rewriting 16
    references across 228 lines of body. Doing that with a text rewrite corrupts
    whatever shares a name with a parameter -- and the two casualties are
    invisible to a flag-name comparison:

    * a keyword-argument NAME: ``Santec(slm_number=slm_number)`` becomes
      ``Santec(params.slm_number=params.slm_number)``, a SyntaxError;
    * a dict key / Recorder kwarg / log format: ``{"csv_path": csv_path}``
      becomes ``{"params.csv_path": params.csv_path}`` -- still valid Python,
      still a passing test, wrong at runtime.

    So pin every string literal inside every function body. Decorator strings are
    excluded on purpose: relocating the option declarations into dataclass fields
    is the one change that is *supposed* to move those strings.
    """
    expected = _golden()[probe]
    lits = _body_literals(SLM_PKG_DIR / f"{probe}.py")
    digest = hashlib.sha256(json.dumps(lits, ensure_ascii=False).encode()).hexdigest()[:16]
    assert len(lits) == expected["body_literals"], (
        f"{probe}: in-body string literal count {len(lits)} != {expected['body_literals']}"
    )
    assert digest == expected["body_literals_sha256"], (
        f"{probe}: in-body string literals changed.\n"
        f"  a body rename likely rewrote a dict key, kwarg name or log format"
    )


def _body_literals(path: Path) -> list[str]:
    out: list[str] = []
    for fn in ast.walk(_parse(path)):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # body only -- see the test docstring for why decorators are excluded
        for stmt in fn.body:
            for node in ast.walk(stmt):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    out.append(node.value)
    return sorted(out)


@pytest.mark.parametrize("probe", PROBES)
def test_no_dataclass_option_carries_a_default(probe: str) -> None:
    """``Annotated[..., option(default=...)]`` raises at import; the flag test cannot see it.

    ``cli_params._patch_defaults`` makes the dataclass field the single source of
    truth and raises ``TypeError`` on a second one. So ``default=`` is legal in
    ``@click.option(...)`` (click handles it) and **fatal** inside
    ``Annotated[..., option(...)]``.

    That distinction is why this check exists rather than a blanket "no default in
    option()": two migrated probes carried ``, default=None`` inside ``option()``
    and every flag-name assertion still passed, because the declarations were all
    present and correct. Only ``--help`` failing to load revealed it -- and a guard
    that needs a 28 s subprocess per module to notice is a guard that gets skipped.
    """
    path = SLM_PKG_DIR / f"{probe}.py"
    tree = _parse(path)
    decorator_calls = {
        id(sub)
        for fn in ast.walk(tree)
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
        for d in fn.decorator_list
        for sub in ast.walk(d)
    }
    offenders = [
        (n.lineno, ast.unparse(kw.value))
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and _called_name(n) == "option"
        and id(n) not in decorator_calls
        for kw in n.keywords
        if kw.arg == "default"
    ]
    assert not offenders, (
        f"{probe}: option() carries default= at line(s) "
        + ", ".join(f"{ln} (={v})" for ln, v in offenders)
        + " -- move it onto the dataclass field, or the module raises TypeError on import"
    )


def test_both_camera_flag_spellings_survive() -> None:
    """R-41: the camera-flag spelling is split across probes, and both are habit.

    Nine probes take ``--cam-type``; ``slm_diagnose`` and ``slm_lut_runner`` take
    ``--camera-type``. Neither is a typo to be swept up -- they are what operators
    already type -- so the contract is that **both stay**, and that no single probe
    grows both spellings (which would be a silent CLI fork, not an alias).

    The golden already pins every flag name, so this test does not add coverage; it
    makes the *intent* legible, so a future "harmonise the spelling" change has to
    come here on purpose rather than ride along in a refactor.
    """
    g = _golden()
    short = {p for p in PROBES if "--cam-type" in g[p]["declared"]}
    long_ = {p for p in PROBES if "--camera-type" in g[p]["declared"]}

    assert len(short) == 8, f"--cam-type lost a probe: {sorted(short)}"
    assert len(long_) == 2, f"--camera-type lost a probe: {sorted(long_)}"
    assert not short & long_, (
        f"{sorted(short & long_)} expose both spellings; pick one or add a real "
        f"alias, do not fork the CLI"
    )
    assert long_ == {"slm_diagnose", "slm_lut_runner"}


@pytest.mark.parametrize(
    ("a", "b", "count_a", "count_b"),
    [
        ("--output", "--out", 8, 4),
        ("--slm-wavelength", "--wavelength", 14, 3),
    ],
)
def test_other_split_flag_spellings_survive(
    a: str, b: str, count_a: int, count_b: int
) -> None:
    """The other two split spellings R-41 swept up, same contract as above."""
    g = _golden()
    users_a = {p for p in PROBES if a in g[p]["declared"]}
    users_b = {p for p in PROBES if b in g[p]["declared"]}

    assert len(users_a) == count_a, f"{a} moved: {len(users_a)} != {count_a}"
    assert len(users_b) == count_b, f"{b} moved: {len(users_b)} != {count_b}"
    assert not users_a & users_b, f"{sorted(users_a & users_b)} expose both {a} and {b}"


def test_the_flag_surface_is_the_size_we_think_it_is() -> None:
    """277 declared flags across 19 probes. A drop here means the scan went blind.

    Declared and help-visible counts differ legitimately: click adds a built-in
    ``--help`` to all 19, and some probes declare ``hidden=True`` options that
    never render. Both numbers are pinned so a change in either direction is seen.

    287 -> 285 on 2026-10-04: ``slm_zernike_sweep_probe`` declared
    ``--save-frames/--no-save-frames``, which argparse cannot express (it has no
    ``/`` syntax, so it registered one literal long option taking a value). Both
    spellings exited rc=2 and ``args.save_frames`` was never read -- frame saving
    is unconditional. The dead, broken declaration was deleted rather than
    "repaired" into a flag that would still control nothing.
    """
    g = _golden()
    declared = sum(len(g[p]["declared"]) for p in PROBES)
    visible = sum(len(g[p]["help_flags"]) for p in PROBES)
    assert declared == 277, f"declared flag inventory is {declared}, expected 277"
    assert visible == 289, f"help-visible flag inventory is {visible}, expected 289"
    assert len(PROBES) == 19


# ---------------------------------------------------------------------------
# the deep check: rendered --help, opt-in because it costs ~9 minutes
# ---------------------------------------------------------------------------


def _rendered_help(probe: str) -> str:
    result = subprocess.run(
        [sys.executable, "-m", f"ao_shaping.tools.slm.{probe}", "--help"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=300, cwd=str(REPO), check=False,
    )
    assert result.returncode == 0, f"{probe}: rc={result.returncode}\n{result.stderr[-400:]}"
    return result.stdout.replace("\r\n", "\n").rstrip("\n")


def _golden_entry(probe: str, help_text: str) -> dict:
    """Re-measure every field the golden stores for ``probe``.

    All five must be recomputed together. Writing only three of them (the
    original bug here) drops ``body_literals``/``body_literals_sha256`` from
    every entry; the run that does the damage still passes, because
    ``expected = _golden()`` is read *before* the write, but the next run dies
    with ``KeyError: 'body_literals'`` in
    :func:`test_probe_body_literals_are_unchanged`. Recomputing is also the
    right semantics: this path is the explicit re-baseline, exactly like
    ``help``/``help_flags``/``declared`` beside it.
    """
    path = SLM_PKG_DIR / f"{probe}.py"
    lits = _body_literals(path)
    return {
        "help": help_text,
        "help_flags": sorted(set(_HELP_FLAG.findall(help_text))),
        "declared": _declared_flags(path),
        "body_literals": len(lits),
        "body_literals_sha256": hashlib.sha256(
            json.dumps(lits, ensure_ascii=False).encode()
        ).hexdigest()[:16],
    }


@pytest.mark.parametrize("probe", PROBES)
def test_golden_entry_rewrites_every_field_it_stores(probe: str) -> None:
    """The regeneration path must not be lossy.

    ``AO_PROBE_HELP_UPDATE=1`` used to write only ``help``/``help_flags``/
    ``declared`` while the golden stores five keys. The run that dropped the
    other two still passed -- ``expected = _golden()`` is read before the write
    -- and the *next* run raised ``KeyError: 'body_literals'``. Nothing caught
    it because the check lived behind the ~9-minute opt-in.

    This re-measures every probe against the committed golden with no
    subprocess at all, so a partial writer fails in milliseconds.
    """
    stored = _golden()[probe]
    rewritten = _golden_entry(probe, stored["help"])

    assert set(rewritten) == set(stored), (
        f"{probe}: the regeneration path writes {sorted(rewritten)} but the golden "
        f"stores {sorted(stored)} -- a field would be silently dropped"
    )
    assert rewritten["body_literals"] == stored["body_literals"], (
        f"{probe}: _golden_entry measures {rewritten['body_literals']} in-body "
        f"literals but the golden stores {stored['body_literals']}"
    )
    assert (
        rewritten["body_literals_sha256"] == stored["body_literals_sha256"]
    ), f"{probe}: _golden_entry digest disagrees with the golden"
    assert rewritten["declared"] == stored["declared"]
    assert rewritten["help_flags"] == stored["help_flags"]


def test_rendered_help_matches_golden_or_differs_only_in_order() -> None:
    if not UPDATE:
        pytest.skip("opt-in: set AO_PROBE_HELP_UPDATE=1 to compare (costs ~9 min)")
    actual = {p: _rendered_help(p) for p in PROBES}
    expected = _golden()
    drift = {}
    for probe in PROBES:
        if actual[probe] != expected[probe]["help"]:
            drift[probe] = {
                "flags_differ": sorted(
                    set(_HELP_FLAG.findall(actual[probe])) ^ set(expected[probe]["help_flags"])
                ),
                "same_lines_reordered": sorted(actual[probe].splitlines())
                == sorted(expected[probe]["help"].splitlines()),
            }
    GOLDEN.write_text(
        json.dumps(
            {p: _golden_entry(p, actual[p]) for p in PROBES},
            indent=2, ensure_ascii=False, sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    hard = {p: d for p, d in drift.items() if d["flags_differ"] or not d["same_lines_reordered"]}
    assert not hard, (
        f"rendered help changed beyond ordering: {hard}. If the change is intended, "
        "review it and commit the regenerated golden alongside."
    )
