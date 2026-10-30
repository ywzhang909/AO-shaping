"""R-32: convention guards (orphan tests, layering, `python -m` consistency).

The TODO asked for these warn-only with a baseline to start from. Two of the three
needed no baseline at all, because the tree is already clean; the third needed a
two-entry baseline after the real breakage was fixed. So all three now hard-fail
on anything new, which is the useful end of "warn-only".

What each guard is for, and why a plain grep is not enough:

* **Orphan tests.** ``testpaths = ["tests"]``, so a ``test_*.py`` under ``src/``
  is never collected -- it runs for nobody while looking like coverage. R-31 found
  one (a near-empty duplicate of a collected test) and deleted it. Grep cannot see
  this because the file exists and looks fine; you have to compare what pytest
  would collect against what is on disk.

* **Layering.** ``utils/`` is the leaf layer and must not depend on
  ``algorithm/``, ``drivers/``, ``optimizer/``, ``runners/``, ``gui/`` or
  ``tools/``. AGENTS.md sanctions two escapes -- ``TYPE_CHECKING`` and a
  function-local import -- so the guard has to distinguish those from a real
  module-scope import, which means walking the AST by nesting depth rather than
  matching lines. All 14 current uses are inside one of the two escapes.

* **`python -m` consistency.** Reorganisation moved runners into
  ``micro_drive/`` and ``slm/`` subpackages and left the documented module paths
  behind, so 11 references pointed at modules that no longer exist -- including
  one emitted into a generated report, which is how a stale path reaches a user.
  All fixed; the baseline below is what legitimately does not resolve.

* **Thread budget.** Every native runtime sizes its thread pool once, at load
  time, and each one reads a *different* env var -- so a plausible-looking
  subset is silently useless and the regression is invisible in the diff. A
  single pytest process asked for 24 (OpenBLAS) + 24 (torch intra-op) + 24
  (torch inter-op) = 72 threads on 32 CPUs, multiplied again by each worker.
  The policy lives in ``tests/conftest.py``; the guard below pins both the var
  set and its position *above* the first native import, which is the part that
  silently rots when someone tidies the imports.

* **Ruff gate breadth.** A ``# noqa`` naming a rule the config never enables
  suppresses nothing, yet still reads to a reviewer as a live exemption and
  masks the next real violation of that rule (issue #169 cleared 864 of them).
  The ruff job in CI is only as wide as ``select``, so the guard below pins
  RUF100 staying in it.

* **CCD device guards survive ``python -O``.** ``assert self.cam`` guarded an
  *external device state* and is stripped wholesale under ``-O``; the tree
  collapsed those 36 sites into ``BaseCamera._ensure_open`` (the SLM
  ``_ensure_open`` paradigm). The AST guard below lives in *this* file because
  ``pyproject`` ``addopts --ignore`` prunes ``tests/ao_shaping/drivers/**`` from
  default discovery, so a guard placed there would run for nobody in CI.
"""

from __future__ import annotations

import ast
import importlib.util
import io
import os
import re
import shutil
import subprocess
import sys
import tokenize
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

#: ``tests/ao_shaping/test_conventions.py`` -> parents[2] is the repo root.
REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"

#: Layers ``utils/`` must not depend on. See AGENTS.md, "模块与包结构".
_HIGHER_LAYERS = ("algorithm", "drivers", "optimizer", "runners", "gui", "tools")

#: Documents that are records rather than instructions. A dated daily log or a
#: closed report is *supposed* to name the paths that were current when it was
#: written; rewriting those would falsify history. TODO.md is a task ledger whose
#: entries quote broken paths on purpose.
#:
#: ``report/`` is the whole experiment-report tree (moved out of ``docs/`` on
#: 2026-10-05 so reports are not mixed with device documentation). It is listed
#: wholesale rather than per-topic: it is entirely records, it is not scanned by
#: ``_LIVE_DOCS`` below, and enumerating its ~30 topics here would rot on the
#: first report added after this edit.
#:
#: ``docs/dev/`` qualifies for the same reason: it holds the completed half of the
#: ledger (``todo_archive.md``), which documents *why* the tree looks the way it
#: does and cites the broken paths it was written against. It is split out of
#: ``TODO.md`` (2026-10-06) so the ledger's actionable half stays small; the split
#: does not change what kind of document either half is.
_HISTORICAL_DOCS = (
    "TODO.md",
    "report/",
    "docs/dev/",
    "docs/slm/",
    "docs/wfs/",
    "docs/micro",
    "docs/miicam/",
    "docs/slm-200/",
    "docs/thorlab-wfs/",
    "docs/daheng/",
    "docs/micro-dm/",
    "docs/TM/",
)

_PY_MODULE_DOCS = sorted(SRC.rglob("*.py")) + sorted((REPO / "scripts").glob("*.py"))
#: Documents that carry *current* instructions, so every ``python -m`` target in
#: them must resolve. ``report/`` is deliberately absent -- see _HISTORICAL_DOCS.
_LIVE_DOCS = (
    [REPO / "README.md", REPO / "AGENTS.md"]
    + sorted(SRC.rglob("*.md"))
    + sorted((REPO / "scripts").glob("*.md"))
    + sorted((REPO / "docs").glob("*.md"))
)


def _is_historical(path: Path) -> bool:
    rel = path.relative_to(REPO).as_posix()
    return any(marker in rel for marker in _HISTORICAL_DOCS)


def test_no_tracked_source_py_starts_with_a_utf8_bom() -> None:
    """No repo-owned ``.py`` may begin with a UTF-8 BOM (``ef bb bf``).

    Regression guard for the BOM trap: an earlier version read source with
    ``encoding="utf-8"`` and ran the text straight into ``ast.parse``, so a
    BOM-prefixed (but perfectly importable) script raised
    ``SyntaxError: invalid non-printable character U+FEFF``. The guards now read
    ``utf-8-sig`` so a BOM can no longer *crash* them -- but a BOM is still
    noise that editors and diffs treat inconsistently, and the issue noted one
    was silently re-introduced after a previous one-off strip. This guard makes
    that a loud failure instead of a silent recurrence.

    Scoped to repo-owned source (``src/``, ``scripts/``, ``tests/``); vendored
    ``libs/`` is excluded so a dependency's own encoding never fails our guard.
    """
    offenders: list[str] = []
    for top in ("src", "scripts", "tests"):
        root = REPO / top
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            if p.read_bytes()[:3] == b"\xef\xbb\xbf":
                offenders.append(p.relative_to(REPO).as_posix())
    assert not offenders, (
        "These .py files start with a UTF-8 BOM -- strip the leading "
        "``ef bb bf`` (the three invisible bytes); it is not required by "
        f"CPython and it confuses editors/diffs and AST-parsing guards: {offenders}"
    )


def _dotted(name: str) -> str:
    return name[: -len(".__init__")] if name.endswith(".__init__") else name


def _module_inventory() -> dict[str, tuple[bool, bool]]:
    """``{module: (has __main__ guard, declares a click command)}``."""
    out: dict[str, tuple[bool, bool]] = {}
    # Two package roots, not one: ``src/ml`` is a standalone top-level package
    # (``python -m ml.hwdataset.inspect``), so an inventory scoped to ``ao_shaping``
    # reads every documented ``ml`` entry point as a non-existent module.
    for path in sorted(p for root in (SRC / "ao_shaping", SRC / "ml") for p in root.rglob("*.py")):
        try:
            text = path.read_text(encoding="utf-8-sig")
            tree = ast.parse(text)
        except (OSError, SyntaxError):
            continue
        rel = _dotted(
            path.relative_to(SRC).with_suffix("").as_posix().replace("/", ".")
        )
        has_click = any(
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", "") in ("command", "group")
            and getattr(getattr(node.func, "value", None), "id", "") == "click"
            for node in ast.walk(tree)
        )
        out[rel] = ("__main__" in text, has_click)
        if path.name == "__main__.py":
            # `python -m <package>` resolves through the package's `__main__.py`;
            # the package's own `__init__.py` holds no entry point and says nothing
            # about one. Without this, a package that is genuinely runnable reads as
            # dead -- and the substring test above only passes by accident when
            # `__init__.py` happens to mention `__main__` in prose.
            pkg = rel[: -len(".__main__")]
            had_main, had_click = out.get(pkg, (False, False))
            out[pkg] = (True, had_click or has_click)
    return out


INVENTORY = _module_inventory()


# ---------------------------------------------------------------------------
# 1. no orphan tests under src/
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pattern", ["test_*.py", "*_test.py", "conftest.py"], ids=str)
def test_src_contains_no_test_files_pytest_would_skip(pattern: str) -> None:
    """``testpaths = ["tests"]`` means anything here is dead weight."""
    found = sorted(p.relative_to(REPO).as_posix() for p in SRC.rglob(pattern))
    assert not found, (
        f"{found} live under src/ but testpaths=['tests'], so pytest never "
        "collects them. Move to tests/ or delete; a duplicate looks like coverage "
        "while running for nobody."
    )


def test_the_testpaths_setting_is_what_makes_the_guard_mean_something() -> None:
    """If ``testpaths`` ever widens to include src/, this guard needs rethinking."""
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8-sig")
    assert re.search(r'testpaths\s*=\s*\["tests"\]', text), (
        "testpaths no longer restricts collection to tests/; re-check whether "
        "orphan tests under src/ are still a problem."
    )


# ---------------------------------------------------------------------------
# 2. utils/ layering
# ---------------------------------------------------------------------------


def _higher_layer_imports(path: Path) -> list[tuple[int, int, str]]:
    """``(lineno, nesting_depth, module)`` for imports of a higher layer.

    Depth 0 is module scope -- the only thing AGENTS.md forbids. Depth >= 1 means
    the import sits inside ``if TYPE_CHECKING:``, a function, or a guard, all of
    which the same document explicitly allows.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    except (OSError, SyntaxError):
        return []

    found: list[tuple[int, int, str]] = []

    def visit(node: ast.AST, depth: int) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                visit(child, depth + 1)
            elif isinstance(child, (ast.If, ast.Try, ast.With)):
                visit(child, depth + 1)
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                mod = (
                    (child.module or "")
                    if isinstance(child, ast.ImportFrom)
                    else child.names[0].name
                )
                parts = mod.split(".")
                if (
                    len(parts) > 1
                    and parts[0] == "ao_shaping"
                    and parts[1] in _HIGHER_LAYERS
                ):
                    found.append((child.lineno, depth, mod))

    visit(tree, 0)
    return found


def test_utils_has_no_module_scope_import_of_a_higher_layer() -> None:
    offenders: list[str] = []
    for path in sorted((SRC / "ao_shaping" / "utils").rglob("*.py")):
        for lineno, depth, mod in _higher_layer_imports(path):
            if depth == 0:
                offenders.append(f"{path.relative_to(REPO).as_posix()}:{lineno} {mod}")
    assert not offenders, (
        "module-scope imports of a higher layer inside utils/ (deferred and "
        f"TYPE_CHECKING forms are fine): {offenders}"
    )


def test_the_deferred_forms_still_exist_so_the_guard_is_not_vacuous() -> None:
    """There are 14 sanctioned escapes today; if that drops to 0 the AST walk
    above is probably not descending and has stopped checking anything."""
    deferred = sum(
        1
        for path in (SRC / "ao_shaping" / "utils").rglob("*.py")
        for _lineno, depth, _mod in _higher_layer_imports(path)
        if depth >= 1
    )
    assert deferred > 0, (
        "no TYPE_CHECKING or function-local higher-layer imports remain under "
        "utils/ -- confirm the guard still detects a module-scope one before "
        "trusting it"
    )


# ---------------------------------------------------------------------------
# 3. `python -m` documentation matches reality
# ---------------------------------------------------------------------------

_PY_M = re.compile(r"python\s+-m\s+((?:ao_shaping|ml)[\w.]*)")
_MD_M = re.compile(r"python\s+-m\s+((?:ao_shaping|ml)[\w.]*)")

#: References that legitimately do not resolve. Each needs a reason, because an
#: unexplained baseline entry is just a suppressed failure.
#:
#: ``ao_shaping.runners.gsnet_train`` was baselined while the gsnet trainer was being
#: moved out of ``ao_shaping.runners``; it left the list once ``scripts/README.md``
#: stopped claiming a ``python -m`` invocation for a module that no longer lived
#: there. Do not re-add it without a live claim to excuse.
_ALLOWED_UNRESOLVED = {
    "ao_shaping.runners": (
        "prose in utils/io/cli_helpers.py uses `python -m ao_shaping.runners...` "
        "as a glob for 'some runner', not a specific module"
    ),
}


def _unresolved() -> dict[str, list[str]]:
    """``{module: [locations]}`` for every documented path that does not resolve."""
    problems: dict[str, list[str]] = {}

    def record(mod: str, where: str) -> None:
        # `python -m pkg` runs pkg/__main__.py, which the inventory keys
        # separately as `pkg.__main__`; a package whose __init__.py carries no
        # __main__ guard or click still resolves when its __main__.py does.
        # A single-file module has no `mod.__main__` key, so the lookup falls
        # through to its own entry.
        entries = [INVENTORY.get(mod), INVENTORY.get(mod + ".__main__")]
        present = [e for e in entries if e is not None]
        if not present:
            problems.setdefault(mod, []).append(f"{where} (no such module)")
        elif not any(any(e) for e in present):
            problems.setdefault(mod, []).append(f"{where} (no __main__, no click)")

    for path in _PY_MODULE_DOCS:
        if _is_historical(path):
            continue
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8-sig").splitlines(), 1
        ):
            for match in _PY_M.finditer(line):
                mod = match.group(1).rstrip(".")
                tail = line[match.end() :][:14]
                if "<" in tail:
                    continue  # `python -m ao_shaping.tools.slm.<name>` template
                record(mod, f"{path.relative_to(REPO).as_posix()}:{lineno}")

    for path in _LIVE_DOCS:
        if _is_historical(path):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in _MD_M.finditer(text):
            mod = match.group(1).rstrip(".")
            if "<" in text[match.end() : match.end() + 14]:
                continue  # `python -m ao_shaping.tools.slm.<name>` template
            record(mod, path.relative_to(REPO).as_posix())

    return problems


def test_documented_python_dash_m_entry_points_resolve() -> None:
    problems = _unresolved()
    unexpected = {
        mod: locs for mod, locs in problems.items() if mod not in _ALLOWED_UNRESOLVED
    }
    assert not unexpected, (
        f"documented `python -m` targets that do not run: {unexpected}. Either the "
        "module moved (update the path) or it never had a __main__ (drop the claim)."
    )


def test_the_baseline_has_not_grown_and_is_still_justified() -> None:
    """A stale baseline entry hides a real break, so both directions are checked."""
    problems = _unresolved()
    assert set(problems) == set(_ALLOWED_UNRESOLVED), (
        "the allow-list and the actual unresolved set have diverged:\n"
        f"  newly broken: {sorted(set(problems) - set(_ALLOWED_UNRESOLVED))}\n"
        f"  now fixed:   {sorted(set(_ALLOWED_UNRESOLVED) - set(problems))}"
    )
    for mod, reason in _ALLOWED_UNRESOLVED.items():
        assert reason and not reason.startswith("TBD"), f"{mod} has no justification"


def test_a_package_may_resolve_only_through_its_main_module() -> None:
    """A package is runnable through ``__main__.py``, not through a mention.

    ``diffraction_efficiency`` is the case that broke the old rule: its
    ``__init__.py`` neither says ``__main__`` nor declares a click command, so the
    inventory read a package that ``python -m`` genuinely runs as dead, and every
    document naming it failed. The reverse hazard is just as real -- a package
    whose ``__init__.py`` merely *mentions* ``__main__`` in prose resolves for the
    wrong reason. Pinning both directions keeps the ``__main__.py`` branch in
    ``_module_inventory`` from being tidied away into dead code.
    """
    silent_packages = [
        mod
        for mod in INVENTORY
        if (pkg := SRC / mod.replace(".", "/")).is_dir()
        and (pkg / "__main__.py").is_file()
        and (pkg / "__init__.py").is_file()
        and "__main__"
        not in (pkg / "__init__.py").read_text(encoding="utf-8-sig", errors="replace")
    ]

    assert silent_packages, (
        "no package resolves through `__main__.py` alone; if that is genuinely "
        "true the branch in _module_inventory is dead code and both it and this "
        "test should go together"
    )
    unrunnable = [mod for mod in silent_packages if not any(INVENTORY[mod])]
    assert not unrunnable, (
        f"these packages have __main__.py but read as unrunnable: {unrunnable}"
    )


def test_the_reorganisation_stale_paths_are_gone_from_the_files_that_had_them() -> None:
    """Pins the actual R-32 fix, file by file, instead of restating a count.

    Reorganising runners into ``micro_drive/`` and ``slm/`` left 11 references
    across these four files pointing at modules that had moved -- including one
    emitted into a generated report, which is how a stale path reaches a user.
    Naming the files and the old paths makes the failure legible; a bare count
    would not say which file regressed.
    """
    # A list, not a dict: alt-voltage and full-voltage now share ONE module, and
    # two identical dict keys would silently drop the first assertion.
    stale = [
        (
            "src/ao_shaping/runners/micro_drive/voltage_runner.py",
            "python -m ao_shaping.runners.alt_voltage_runner",
        ),
        (
            "src/ao_shaping/runners/micro_drive/voltage_runner.py",
            "python -m ao_shaping.runners.full_voltage_runner",
        ),
        (
            "scripts/zernike_bench/generate_zernike_response_matrix_report.py",
            "python -m ao_shaping.runners.zernike_matrix_runner",
        ),
        (
            "src/ao_shaping/tools/slm/cartographer/__init__.py",
            "python -m ao_shaping.tools.slm.cartographer\n",
        ),
    ]
    for rel, old in stale:
        text = (REPO / rel).read_text(encoding="utf-8-sig")
        assert old not in text, f"{rel} still documents the pre-move path {old!r}"

    # And the replacements must resolve, or the fix traded one broken path for another.
    for rel, old in stale:
        text = (REPO / rel).read_text(encoding="utf-8-sig")
        for match in _PY_M.finditer(text):
            mod = match.group(1).rstrip(".")
            if "<" in text[match.end() : match.end() + 14]:
                continue
            info = INVENTORY.get(mod)
            assert info is not None and any(info), (
                f"{rel} now points at {mod}, which also does not run"
            )


# ---------------------------------------------------------------------------
# 4. tests must not write into the tracked docs/ tree
# ---------------------------------------------------------------------------

#: ``TestReport`` targets ``docs/<device>/<device>_report.md`` and mkdirs it in
#: ``__init__`` -- i.e. before any skip can fire. Four of those targets are
#: tracked: docs/slm (82 files), docs/slm-200 (9), docs/wfs (2), docs/miicam (2).
#: The `hardware` marker is declared but pyproject's addopts does NOT filter on it,
#: so a plain `pytest` collects the very tests that rewrite them.
#:
#: Changing that default is deliberately NOT done here: the marker covers 158 test
#: functions across 8 files and some may genuinely pass without hardware, so
#: deselecting by default could silently cut coverage. What is enforced instead is
#: that any test reaching for the harness carries the marker, so a maintainer can
#: filter it with `-m "not hardware"`.
_TEST_REPORT_HARNESS = re.compile(r"\bTest(?:With)?Report\s*\(")
_HARDWARE_MARK = re.compile(r"pytestmark\s*=\s*pytest\.mark\.hardware|mark\.hardware")


def _tracked_docs_paths() -> set[str]:
    import subprocess

    out = subprocess.run(
        ["git", "ls-files", "docs"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(REPO),
        check=False,
    ).stdout.split()
    return {p for p in out if p.endswith((".md", ".png", ".json", ".csv"))}


#: The harness itself defines the class; it is not a consumer.
_HARNESS_MODULE = "tests/ao_shaping/utils/test_report.py"

#: Consumers that write only into an UNTRACKED docs dir. They still leave the tree
#: dirty, but they cannot corrupt a commit, so they are allowed -- and each one
#: needs a reason. Empty since 2026-10-22: the only entry was the simulated
#: miicam report, replaced by the hardware auto-exposure probe
#: (``test_miicam_auto_exposure_report.py``), which does not use this harness.
_ALLOWED_UNTRACKED_WRITERS: dict[str, str] = {}


def _targets_a_tracked_docs_dir(path: Path) -> str | None:
    """The ``device_dir=``/name that makes this test write into tracked docs/, if any."""
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files", "docs"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(REPO),
        check=False,
    ).stdout.split()
    tracked_dirs = {str(Path(f).parent).replace("\\", "/") for f in tracked}

    text = path.read_text(encoding="utf-8-sig", errors="replace")
    # An explicit device_dir OVERRIDES the device name, so it is the answer whether
    # or not it is tracked. Falling through to the name here would flag a
    # TestReport("miicam", device_dir="report/<series>") as writing to
    # docs/miicam, which it does not touch.
    override = re.search(r'device_dir\s*=\s*"([^"]+)"', text)
    if override:
        target = override.group(1).rstrip("/")
        return target if target in tracked_dirs else None
    for match in re.finditer(r'TestReport\(\s*"([\w-]+)"', text):
        candidate = f"docs/{match.group(1)}"
        if candidate in tracked_dirs:
            return candidate
    return None


def test_only_the_harness_or_hardware_tests_write_tracked_docs() -> None:
    """A test that rewrites a TRACKED docs/ file must carry the hardware marker.

    Narrower than "uses the harness": the hazard is corrupting a commit, not
    merely leaving untracked files behind, so a hardware-free writer that
    targets an untracked dir can be whitelisted in
    ``_ALLOWED_UNTRACKED_WRITERS`` with a reason.
    """
    offenders: dict[str, str] = {}
    for path in sorted((REPO / "tests").rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        if rel == _HARNESS_MODULE or rel in _ALLOWED_UNTRACKED_WRITERS:
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        if not _TEST_REPORT_HARNESS.search(text):
            continue
        tracked_target = _targets_a_tracked_docs_dir(path)
        if tracked_target and not _HARDWARE_MARK.search(text):
            offenders[rel] = tracked_target
    assert not offenders, (
        f"these tests write into tracked docs dirs without a `hardware` marker: "
        f"{offenders}. Add `pytestmark = pytest.mark.hardware` so "
        '`-m "not hardware"` can exclude them.'
    )


def test_the_allow_list_has_not_grown_without_a_reason() -> None:
    for rel, reason in _ALLOWED_UNTRACKED_WRITERS.items():
        assert reason and not reason.startswith("TBD"), f"{rel} has no justification"
    # And each allowed writer really does still target an untracked dir, otherwise
    # it graduated into the offender set above and belongs in neither list.
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files", "docs"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(REPO),
        check=False,
    ).stdout.split()
    assert tracked
    {str(Path(f).parent).replace("\\", "/") for f in tracked}
    for rel in _ALLOWED_UNTRACKED_WRITERS:
        target = _targets_a_tracked_docs_dir(REPO / rel)
        assert target is None, (
            f"{rel} now writes into the tracked dir {target}; remove it from the "
            "allow list and add the hardware marker instead"
        )


def test_the_docs_targets_the_harness_writes_to_are_known() -> None:
    """Records the blast radius, so the day it changes someone sees the number.

    Not an assertion about correctness -- the hazard is that a default `pytest`
    rewrites tracked files, which is a decision to make rather than a bug to
    catch. This test exists so the count is not folklore.
    """
    harness = REPO / "tests" / "ao_shaping" / "utils" / "test_report.py"
    assert harness.exists(), "the docs-writing harness moved; re-check this guard"
    tracked = _tracked_docs_paths()
    assert tracked, "git ls-files docs returned nothing; is this a git repo?"
    # Nothing to assert about the exact set -- it changes as reports are added.
    # The guard that matters is the marker check above.
    assert len(tracked) > 0


# ---------------------------------------------------------------------------
# 5. algorithm/ must not write reports
# ---------------------------------------------------------------------------

#: AGENTS.md: "All markdown/illustrated-report **generation** MUST live in
#: `scripts/` ... NEVER under `src/ao_shaping/tools/`, which is reserved for
#: hardware-interaction tools", and the same rule for `algorithm/`: report
#: generation must not sit in the algorithm layer.
#:
#: F-14 was one violation: beam_shaping_benchmark.py wrote CSV+MD+GIF from three
#: functions. It now computes only -- `run_benchmark` / `run_benchmark_suite`
#: no longer take an `output_dir` at all -- while the three helpers that are not
#: I/O (`build_gif_frames`, `to_dataframe`, `HPRINT_KEYS`) stay with the producer.
#: Serialisation moved to scripts/slm/generate_beam_shaping_benchmark_report.py.
_WRITE_CALLS = re.compile(
    r"\b(?:to_csv|write_text|write_bytes|savefig|imsave|mkdir)\s*\(|"
    r"\.save\s*\(|open\s*\([^)]*[\"'][wax]"
)


def test_algorithm_layer_writes_no_reports() -> None:
    offenders: list[str] = []
    for path in sorted((SRC / "ao_shaping" / "algorithm").rglob("*.py")):
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8-sig").splitlines(), 1
        ):
            code = line.split("#", 1)[0]
            if _WRITE_CALLS.search(code):
                offenders.append(
                    f"{path.relative_to(REPO).as_posix()}:{lineno} {code.strip()[:70]}"
                )
    assert not offenders, (
        "the algorithm layer must compute, not serialise -- report generation "
        f"belongs in scripts/ (AGENTS.md anti-pattern). Offenders: {offenders}"
    )


def test_the_benchmark_writer_is_reachable_from_scripts() -> None:
    """Pins where the serialisation went, so it cannot quietly move back."""
    writer = REPO / "scripts" / "slm" / "generate_beam_shaping_benchmark_report.py"
    assert writer.is_file(), "the report writer moved; update this guard"
    text = writer.read_text(encoding="utf-8-sig")
    assert "def write_table(" in text and "def write_artifacts(" in text
    # And the algorithm module must no longer expose them.
    algo = (
        SRC
        / "ao_shaping"
        / "algorithm"
        / "signal_processing"
        / "beam_shaping_benchmark.py"
    ).read_text(encoding="utf-8-sig")
    assert "def _write_table" not in algo and "def _write_artifacts" not in algo
    assert "output_dir" not in algo, (
        "beam_shaping_benchmark still takes an output_dir, so it is still a writer"
    )


# ---------------------------------------------------------------------------
# 6. the thread budget is pinned before any native pool loads
# ---------------------------------------------------------------------------

#: ``tests/conftest.py`` owns the suite's thread budget (16 cores, divided by
#: the worker count). Grep cannot protect this: the pinned vars are load-bearing
#: one runtime at a time -- a plausible-looking subset is silently useless --
#: and their *position*, above the first pool-forming import, is invisible in a
#: diff and rots silently the next time someone tidies the imports.
_ROOT_CONFTEST = REPO / "tests" / "conftest.py"

#: Must match ``CORE_BUDGET`` in the conftest. The first assertion below is what
#: makes that duplication safe: it fails loudly on drift instead of trusting it.
_EXPECTED_CORE_BUDGET = 16

_EXPECTED_THREAD_ENV_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OPENBLAS_DEFAULT_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "NUMBA_NUM_THREADS",
)

#: Runtimes that freeze a pool size at import. ``cupy`` has no CPU pool -- its
#: "threads" are CUDA grid/block -- but it is listed so that a future CPU pool
#: cannot land above the pinning block unnoticed.
_POOL_FORMING_IMPORTS = frozenset(
    {"numpy", "scipy", "torch", "numba", "cupy", "sklearn"}
)


def _conftest_source() -> str:
    return _ROOT_CONFTEST.read_text(encoding="utf-8-sig")


def _module_scope_literal(tree: ast.Module, name: str) -> object:
    """Value of a module-scope assignment to ``name``, or None."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            try:
                return ast.literal_eval(node.value)
            except ValueError:
                return None
    return None


def test_the_conftest_pins_exactly_the_vars_that_are_load_bearing() -> None:
    tree = ast.parse(_conftest_source())
    assert _module_scope_literal(tree, "CORE_BUDGET") == _EXPECTED_CORE_BUDGET, (
        "CORE_BUDGET moved; update _EXPECTED_CORE_BUDGET here in the same commit"
    )
    assert (
        _module_scope_literal(tree, "_THREAD_ENV_VARS") == _EXPECTED_THREAD_ENV_VARS
    ), "the pinned var set drifted -- each var covers a different runtime, "\
       "so dropping one silently restores the oversubscription for that library"
    # MKL_DYNAMIC defaults to TRUE, which makes MKL_NUM_THREADS advisory.
    assert 'os.environ.setdefault("MKL_DYNAMIC", "FALSE")' in _conftest_source(), (
        "without MKL_DYNAMIC=FALSE, MKL is free to pick its own (higher) thread "
        "count and silently ignore MKL_NUM_THREADS"
    )


def test_the_pinning_precedes_every_pool_forming_import() -> None:
    """Position is the half of the policy a diff cannot show.

    A pool sizes itself once, when its library is imported; a later
    ``os.environ`` write does not shrink it, so a reordering here is a silent
    reintroduction of the 72-thread-per-process state.
    """
    pin_lines: list[int] = []
    load_lines: list[str] = []
    for node in ast.parse(_conftest_source()).body:
        # Function bodies run later; only module-scope execution order matters,
        # and the helpers legitimately re-read the environment at call time.
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if "os.environ" in ast.unparse(node):
            pin_lines.append(node.lineno)
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        for name in names:
            if name.split(".")[0] in _POOL_FORMING_IMPORTS:
                load_lines.append(f"{node.lineno}:{name}")

    assert load_lines, (
        "no module-scope pool-forming import is left in tests/conftest.py, so "
        "this guard now proves nothing -- restore the import above the pinning "
        "block, or drop the guard deliberately"
    )
    first_load = min(int(entry.split(":", 1)[0]) for entry in load_lines)
    assert max(pin_lines) < first_load, (
        f"thread pinning runs after a pool-forming import ({load_lines}); the "
        "pool is already sized by then and no env var can shrink it"
    )


def test_the_thread_budget_is_in_force_in_this_process() -> None:
    """The behavioural half: not just that the conftest says so, but that it is."""
    workers = max(1, int(os.environ.get("PYTEST_XDIST_WORKER_COUNT") or "1"))
    override = os.environ.get("AO_TEST_THREADS")
    expected = max(1, (int(override) if override else _EXPECTED_CORE_BUDGET) // workers)

    for var in _EXPECTED_THREAD_ENV_VARS:
        assert os.environ.get(var) == str(expected), (
            f"{var}={os.environ.get(var)!r}, expected {expected} -- the conftest "
            "pinning did not run, so every pool is free to grab all cores"
        )
    assert os.environ.get("MKL_DYNAMIC") == "FALSE"

    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is an optional extra; only the env vars apply")

    import torch

    # The intra-op pool's *default* honours OMP_NUM_THREADS/MKL_NUM_THREADS,
    # but only if nothing sized the pool first, hence the API call as well.
    assert torch.get_num_threads() == expected
    # Inter-op is reachable through no env var at all and is idle outside JIT,
    # so it is held at 1 rather than taking a share of the budget it never uses.
    assert torch.get_num_interop_threads() == 1


# --------------------------------------------------------------------------- #
# Root conftest: the resource guards (CPU/VRAM caps, the OS-thread watch).
#
# A different file from the ``tests/conftest.py`` above, which owns the thread
# *budget*. This one owns the thread *watch*: it samples the process thread count
# so a pool explosion is attributed rather than guessed at.
# --------------------------------------------------------------------------- #

#: The repo-root ``conftest.py``. Deliberately not ``tests/conftest.py``.
_RESOURCE_CONFTEST = REPO / "conftest.py"


def _load_resource_conftest() -> ModuleType:
    """Import the root conftest as a module.

    Safe to load again: module scope only reads the environment and builds
    ``WATCH``. The monitor thread starts in ``pytest_configure``, which this
    does not call.
    """
    spec = importlib.util.spec_from_file_location(
        "_ao_resource_conftest_probe", _RESOURCE_CONFTEST
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_thread_watch_reports_a_count_on_this_platform() -> None:
    """``_os_thread_count`` must return a number here, not ``None``.

    ``/proc/self/status`` is Linux-only, and returning ``None`` off Linux is
    what left the watch inert rather than merely silent: ``_monitor`` skips
    ``None`` counts, so ``WATCH["peak"]`` stayed at ``0``, ``THREAD_ALARM``
    could never fire, and the teardown line always printed ``peak = 0``.
    ``psutil`` supplies the same quantity elsewhere, so ``None`` now means only
    "no source available".

    On Linux this passes off ``/proc`` alone; the ``psutil`` fallback this
    pins is exercised on Windows, where the old code returned ``None``.
    """
    count = _load_resource_conftest()._os_thread_count()

    assert isinstance(count, int), (
        f"expected a real thread count on {sys.platform}, got {count!r} -- the "
        "psutil fallback is what keeps the thread watch alive off Linux"
    )
    assert count >= 1, f"a live process has at least one thread, got {count!r}"


# ---------------------------------------------------------------------------
# 7. the ruff gate stays wide enough to see dead noqa directives
# ---------------------------------------------------------------------------


def _ruff_lint_section() -> dict:
    return tomllib.loads((REPO / "ruff.toml").read_text(encoding="utf-8"))["lint"]


def test_the_ruff_gate_still_selects_ruf100() -> None:
    """A ``# noqa`` naming a rule that is never enabled is a decoy, not a nit.

    Issue #169 cleared 864 of them. They named rules ``select`` never enables
    (``ANN*``, ``BLE001``, ``SLF001``) or rules the config ignores outright
    (``E402``), so each suppressed nothing while still reading to a reviewer as
    a live exemption -- and it would also have hidden the next real violation of
    that same rule. Dropping RUF100 from ``select`` breaks nothing visible; it
    just lets the count grow back.
    """
    lint = _ruff_lint_section()

    assert "RUF100" in lint.get("select", []), (
        "ruff.toml no longer selects RUF100, so unused `# noqa` directives "
        f"regrow unnoticed (select = {lint.get('select', [])})"
    )
    assert "RUF100" not in lint.get("ignore", []), (
        "RUF100 appears in both select and ignore, which cancels the gate"
    )


def test_the_ruf100_gate_is_measured_with_the_project_select_set() -> None:
    """Zero unused directives today, and the measurement must not be done alone.

    RUF100 judges a directive "unused" against the *enabled* rule set, so
    ``--select RUF100`` on its own reclassifies the live ones -- the ``F401``
    exemptions that sit under ``F`` -- as dead, and ``--fix`` then deletes them
    together with the rationale prose trailing them. Doing exactly that on this
    tree reported 891 instead of 864 and unmasked 24 real ``F401`` findings.

    An isolated select can therefore only ever find *at least* as many unused
    directives as the configured gate does, which is the direction asserted here:
    it stays true whichever way the tree moves, and it fails the moment someone
    hard-codes the dangerous invocation into the gate.
    """
    ruff_bin = shutil.which("ruff")
    if ruff_bin is None:
        pytest.skip("ruff not on PATH; the pinned ruff job in CI is the real gate")

    def _run(*extra_flags: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                ruff_bin,
                "check",
                "--no-cache",
                "--output-format",
                "concise",
                *extra_flags,
                ".",
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
        )

    def _findings(proc: subprocess.CompletedProcess[str]) -> int:
        # `concise` also prints "All checks passed!" and a "Found N errors."
        # summary, so counting stdout lines would report 1 on a clean tree.
        return len(
            [ln for ln in proc.stdout.splitlines() if re.search(r":\d+:\d+:\s+[A-Z]+\d", ln)]
        )

    configured = _run("--extend-select", "RUF100")
    isolated_findings = _findings(_run("--select", "RUF100"))

    assert configured.returncode == 0, (
        f"unused `# noqa` directives are back ({_findings(configured)}):\n{configured.stdout}"
    )
    assert isolated_findings >= _findings(configured), (
        f"`--select RUF100` alone found {isolated_findings} while the configured "
        "gate found 0; a lone select misjudges live directives as dead"
    )


#: Config files that install or require ruff, each of which must name exactly one
#: version. A second occurrence -- typically a commented-out old pin -- fails the
#: "exactly one" check on purpose, because that is how a stale version survives a
#: bump and gets copied back in later.
_RUFF_PIN_SITES = (
    ".github/workflows/ci.yml",
    ".github/workflows/ruff.yml",
    "pyproject.toml",
)
_RUFF_EXACT_PIN_SITES = (".github/workflows/ci.yml", ".github/workflows/ruff.yml")


def test_all_ruff_pins_agree_and_are_exact() -> None:
    """Every ruff pin names one exact, identical version.

    Two workflows both run ``ruff check .`` as a hard gate, and they used to
    install *different* ruffs: ``ci.yml`` pinned ``==0.11.10`` while ``ruff.yml``
    asked for ``>=0.15.12``. That is not one gate but two non-equivalent ones.
    Measured on the project config: ``PLC0415`` yielded **0** findings under
    0.11.10 (it is a preview rule there, and ruff itself prints "Selection
    `PLC0415` has no effect because preview is not enabled") against **1476** under
    0.15.12 where it is stable, and ``SLF001`` differed by 2 findings.

    The pin also used to sit **below** the dev-group floor in ``pyproject.toml``
    (``ruff>=0.15.12``), so CI ran a ruff that no developer reproduces locally --
    the failure a contributor sees and the failure CI reports are different tools.

    ``>=`` is rejected for the gate itself: a floating bound lets a new ruff
    release flip a hard gate red with no code change between two pushes, which is
    the non-reproducibility that made the discrepancy above expensive to diagnose.
    Bumping ruff is therefore a single-number change across three files, and this
    test fails if any of them falls out of step.
    """
    pin_re = re.compile(
        r"ruff\s*(?P<op>===|==|>=|<=|~=)\s*(?P<version>\d+(?:\.\d+)+)"
    )
    pins: dict[str, list[re.Match[str]]] = {}
    for rel in _RUFF_PIN_SITES:
        pins[rel] = list(pin_re.finditer((REPO / rel).read_text(encoding="utf-8")))
        assert len(pins[rel]) == 1, (
            f"{rel} must name exactly one ruff version; found {len(pins[rel])}: "
            f"{[m.group(0) for m in pins[rel]]}"
        )

    versions = {m.group("version") for m in (p[0] for p in pins.values())}
    assert len(versions) == 1, (
        "ruff pins disagree -- bump all of them together: "
        + ", ".join(f"{rel}={pins[rel][0].group(0)}" for rel in _RUFF_PIN_SITES)
    )

    for rel in _RUFF_EXACT_PIN_SITES:
        operator = pins[rel][0].group("op")
        assert operator == "==", (
            f"{rel} installs ruff with `{operator}`; a hard gate must pin an exact "
            "version, otherwise a new ruff release can flip it with no code change"
        )


# ---------------------------------------------------------------------------
# 8. CCD device-readiness guards survive `python -O`
# ---------------------------------------------------------------------------

#: Handle attributes whose nullness marks "device already open" in the CCD
#: drivers: ``self.cam`` (daheng / miicam), ``self._cap`` (ffmpeg video source),
#: ``self._folder_path`` (image-sequence replay).
_CCD_DEVICE_ATTRS = frozenset({"cam", "_cap", "_folder_path"})


def _ccd_device_state_asserts(path: Path) -> list[int]:
    """Line numbers of ``assert`` statements that test a CCD device handle.

    Walks the whole ``test`` subtree, so ``assert self.cam``,
    ``assert self._cap is not None and self._cap.isOpened()`` and
    ``assert self._folder_path is not None`` all count -- the guard must not be
    defeatable by re-spelling the condition.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    except (OSError, SyntaxError):
        return []

    found: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        if any(
            isinstance(sub, ast.Attribute)
            and sub.attr in _CCD_DEVICE_ATTRS
            and isinstance(sub.value, ast.Name)
            and sub.value.id == "self"
            for sub in ast.walk(node.test)
        ):
            found.append(node.lineno)
    return found


def test_ccd_drivers_never_assert_on_device_state() -> None:
    """No ``assert`` may guard CCD device readiness -- it dies under ``-O``.

    What these asserts guarded is **external device state** (did
    ``initialize()`` actually open the handle?), not a programmer invariant.
    ``python -O`` strips every ``assert`` statement wholesale, after which the
    driver would call the vendor SDK through a null handle. The collapse keeps
    one inherited entry point that always runs -- ``BaseCamera._ensure_open``
    raising ``CameraError``, mirroring the SLM side's ``Santec._ensure_open``
    precedent -- plus ``FFmpegCamera._device_open`` /
    ``ImageFolderCamera._ensure_recording`` for the backends whose readiness is
    not ``self.cam``.

    This guard sits here, not under ``tests/ao_shaping/drivers/ccd/``, because
    ``pyproject`` ``addopts --ignore`` prunes ``tests/ao_shaping/drivers/**``
    from default discovery -- a guard there would never run in CI.
    """
    offenders: list[str] = []
    for path in sorted((SRC / "ao_shaping" / "drivers" / "ccd").rglob("*.py")):
        for lineno in _ccd_device_state_asserts(path):
            offenders.append(f"{path.relative_to(REPO).as_posix()}:{lineno}")
    assert not offenders, (
        "`assert` on self.cam / self._cap / self._folder_path is back under "
        "src/ao_shaping/drivers/ccd/ -- use BaseCamera._ensure_open() "
        f"(or _ensure_recording) so the check survives `python -O`: {offenders}"
    )


#: Methods of ``ThorlabWFS`` allowed to touch the DLL handle without guarding.
#: ``__init__`` / ``open`` establish it and ``close`` releases it, so they must
#: run unguarded; ``is_connected`` is the pure query; ``handle_error`` serves the
#: open/close error paths; the two private helpers are reachable only from
#: already-guarded callers (``get_mla_name`` and ``get_wavefront``).
_WFS_GUARD_EXEMPT = frozenset({
    "__init__", "open", "close", "is_connected", "handle_error", "_ensure_open",
    "_try_get_mla_name_fallback", "_get_stable_samples",
})


def _wfs_unguarded_dll_methods(path: Path) -> list[str]:
    """``ThorlabWFS`` methods that use the DLL handle without guarding it first.

    The check is structural rather than a grep for ``_ensure_open``: a method
    counts as unguarded only if its **first statement after the docstring** is not
    ``self._ensure_open()``, which is what actually orders the guard before the
    handle reaches the DLL.
    """
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    cls = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "ThorlabWFS"
    )
    offenders: list[str] = []
    for fn in cls.body:
        if not isinstance(fn, ast.FunctionDef) or fn.name in _WFS_GUARD_EXEMPT:
            continue
        if not any(
            isinstance(sub, ast.Attribute) and sub.attr == "_instrument_handle"
            for sub in ast.walk(fn)
        ):
            continue
        body = [
            s for s in fn.body
            if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
        ]
        if not body or ast.unparse(body[0]) != "self._ensure_open()":
            offenders.append(f"{fn.name}:{fn.lineno}")
    return offenders


def test_wfs_driver_guards_every_dll_entry_point() -> None:
    """``ThorlabWFS`` must refuse a null handle in the driver, not in callers.

    Unlike the CCD drivers, the WFS driver had **no** open-state check at all:
    every public method handed ``self._instrument_handle`` to ``WFS_64.dll``, and
    a null handle is not a reliable failure there -- ``select_mla`` discards its
    return codes and the auto-exposure branch of ``take_image`` ignores the error
    code -- so an unopened call returned garbage rather than raising. The only
    thing preventing that was ``assert wfs.is_connected()`` spelled out at each
    call site, which ``python -O`` strips.

    So the mechanism now lives in the driver (``_ensure_open`` raising
    ``WfsError``, mirroring ``BaseCamera._ensure_open`` and
    ``Santec._ensure_open``) and the callers are free of device asserts -- which
    ``test_cartographer_never_asserts_on_device_state`` pins from the other side.
    The class is also required to hold **no** ``assert`` at all, since every one
    of its former asserts guarded either device readiness or a parameter bound
    that is handed to the DLL.

    Placed here for the same reason as the CCD guard: ``pyproject`` ``addopts
    --ignore`` prunes ``tests/ao_shaping/drivers/**`` from default discovery.
    """
    path = SRC / "ao_shaping" / "drivers" / "wfs" / "thorlab_wfs.py"
    rel = path.relative_to(REPO).as_posix()

    unguarded = _wfs_unguarded_dll_methods(path)
    assert not unguarded, (
        f"{rel}: ThorlabWFS methods reach self._instrument_handle without "
        f"calling self._ensure_open() first: {unguarded}"
    )

    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    cls = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "ThorlabWFS"
    )
    asserts = [n.lineno for n in ast.walk(cls) if isinstance(n, ast.Assert)]
    assert not asserts, (
        f"{rel}: `assert` is back inside ThorlabWFS at {asserts} -- it is "
        "stripped under `python -O`, so raise WfsError instead"
    )


#: Handlers that legitimately catch ``AssertionError`` because the assertion comes
#: from a **third-party** library inside the protected expression, not from this
#: repo. Keyed by ``<path>: <sorted handler names>`` rather than by line number, so
#: line drift cannot hide a new one and a changed handler turns the entry stale.
_CATCH_ASSERTION_ALLOWLIST = frozenset({
    # torch's own CUDA probe raises AssertionError on a broken driver stack; the
    # documented contract here is to degrade to ``pin_memory=False`` ("driver
    # hiccup"). ``src/ml/`` contains no driver asserts of ours, so this cannot be
    # one of the conversions this rule protects against.
    "src/ml/hwdataset/dataset.py: ('AssertionError', 'RuntimeError')",
})


def test_production_code_never_catches_assertion_error() -> None:
    """Nothing under ``src/`` or ``scripts/`` may have an ``except AssertionError``.

    Catching an assertion converts a programmer invariant into runtime control flow,
    and it fails twice over. Under ``python -O`` the assert that produced the
    exception no longer exists, so the handler is dead code -- and when a driver
    assert is converted to an explicit raise (what 8.9 asks for), a handler written
    against ``AssertionError`` **stops firing silently**, in the hardware-only paths
    that no container test executes. That is not hypothetical: converting
    ``assert x_offset >= 0`` to ``ValueError`` bypassed the narrow handler in
    ``calibration.py::apply_stored_window``, so an out-of-range stored window crashed
    the calibration step instead of degrading to the full frame its own docstring
    promises; converting the high-speed assert to ``WfsError`` dropped
    ``wfs_probe.py``'s actionable hint and turned it into a traceback. Both shipped
    green because the handlers sit on untestable paths -- which is exactly why this
    is a structural rule rather than something to catch by running tests.

    This is why the scan is AST-based and not a grep: the first grep for
    ``except AssertionError`` reported zero while three tuple-form handlers
    (``except (RuntimeError, ValueError, AssertionError)``) were present. Grep
    cannot see them; ``ast.ExceptHandler`` walks every name in the tuple.

    ``tests/`` is exempt on purpose: there ``pytest.raises(AssertionError)`` documents
    an intentional internal-invariant contract (8.9's "keep ``assert``" row), and a
    test may legitimately assert that an invariant still trips.
    """
    found: set[str] = set()
    for root in (SRC, REPO / "scripts"):
        for path in sorted(root.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            except (OSError, SyntaxError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ExceptHandler) or node.type is None:
                    continue
                names = sorted(
                    {
                        sub.id
                        for sub in ast.walk(node.type)
                        if isinstance(sub, ast.Name)
                    }
                )
                if "AssertionError" in names:
                    rel = path.relative_to(REPO).as_posix()
                    found.add(f"{rel}: {tuple(names)}")

    new = sorted(found - _CATCH_ASSERTION_ALLOWLIST)
    stale = sorted(_CATCH_ASSERTION_ALLOWLIST - found)
    assert not new and not stale, (
        f"{new = } -- `except AssertionError` is dead under `python -O` and detaches "
        "silently when a driver assert becomes an explicit raise; catch what the "
        f"callee actually raises. {stale = } -- no longer present, drop from the "
        "allowlist."
    )


#: Attribute reads that mean "is the device usable?" -- i.e. a device-state
#: precondition. `Panel_Res` counts because it is a driver **class constant**, so
#: asserting on it can only ever fire on a fake object.
_DEVICE_STATE_READS = (".is_open", ".is_connected(", ".Panel_Res")


def _cartographer_device_state_asserts(path: Path) -> list[int]:
    """Line numbers of ``assert`` statements testing device readiness or a driver constant."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    except (OSError, SyntaxError):
        return []
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Assert)
        and any(read in ast.unparse(node.test) for read in _DEVICE_STATE_READS)
    ]


def test_cartographer_never_asserts_on_device_state() -> None:
    """The cartographer must not re-implement device readiness checks.

    It carried 9 ``assert self.slm.is_open`` / ``assert self.wfs.is_connected()``
    sites. Those ask the driver a question the driver is supposed to answer itself,
    and ``python -O`` deletes the answer -- after which an unopened WFS hands a null
    handle to ``WFS_64.dll``, which (for ``select_mla`` and the auto-exposure branch
    of ``take_image``) reports no error at all and yields garbage numbers instead.

    The checks now live in the drivers -- ``Santec._ensure_open``,
    ``ThorlabWFS._ensure_open`` and ``BaseCamera._ensure_open`` -- so removing them
    here loses nothing. The 4 asserts that remain in this package guard **internal**
    state (``self.lut``, ``self._reference_saved``, ``self._last_measurement``),
    which is what ``assert`` is for under 8.9's own rule.
    """
    offenders: list[str] = []
    for path in sorted(
        (SRC / "ao_shaping" / "tools" / "slm" / "cartographer").rglob("*.py")
    ):
        for lineno in _cartographer_device_state_asserts(path):
            offenders.append(f"{path.relative_to(REPO).as_posix()}:{lineno}")
    assert not offenders, (
        "device-readiness `assert` is back in the cartographer -- the drivers guard "
        f"this now (Santec/ThorlabWFS._ensure_open): {offenders}"
    )


#: The complete set of ``assert`` statements allowed anywhere under ``drivers/``,
#: as ``<path under drivers/>: <expression>``. Each one tests an **internal**
#: invariant that no external input or device state can reach, which is precisely
#: the only thing Python's guidance reserves ``assert`` for.
#:
#: Matched on the parsed expression rather than the line number so that inserting
#: code above an allowed assert cannot take it off the allowlist, and adding a new
#: one cannot hide behind a shifted line.
_DRIVERS_ASSERT_ALLOWLIST = frozenset({
    "ccd/daheng/driver.py: width_val is not None and height_val is not None",
    "dm/micro/driver.py: isinstance(result, np.ndarray)",
    "dm/micro/driver.py: entry.physical_position is not None",
    "sim/wfs/simulated_wfs.py: self._raw is not None and self._flux is not None",
    "slm/santec/driver.py: max_grayscale is not None",
})


def _drivers_asserts() -> dict[str, list[int]]:
    """Every ``assert`` under ``drivers/``, keyed by path + parsed expression."""
    drivers = SRC / "ao_shaping" / "drivers"
    found: dict[str, list[int]] = {}
    for path in sorted(drivers.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        except (OSError, SyntaxError):
            continue
        rel = path.relative_to(drivers).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                found.setdefault(f"{rel}: {ast.unparse(node.test)}", []).append(node.lineno)
    return found


def test_drivers_only_assert_internal_invariants() -> None:
    """Under ``drivers/``, ``assert`` may only guard internal invariants.

    A driver sits on the boundary with the outside world, so what it checks there
    is either device state (did the SDK actually hand us a live handle?) or a
    caller-supplied argument (``n_sample > 0``). ``python -O`` strips every
    ``assert`` in the file, after which an unguarded driver either drives the SDK
    through a null handle or silently accepts a nonsense count. Each of the three
    driver domains now raises instead, through one entry point per concern:
    ``BaseCamera._ensure_open`` (29001826 / de5b9ccd), ``Santec._ensure_open``,
    ``ThorlabWFS._ensure_open`` (e4868b54), ``DeviceNotFoundError`` in the DM and
    TM drivers, and ``ValueError`` for argument bounds.

    The comparison is deliberately **two-way**: a new assert fails the test, and so
    does an allowlist entry whose assert has since been converted or deleted. A
    stale allowlist is how a guard quietly stops protecting anything, and this repo
    has been bitten repeatedly by stale entries in its own ledger.
    """
    found = _drivers_asserts()
    new = sorted(set(found) - _DRIVERS_ASSERT_ALLOWLIST)
    stale = sorted(_DRIVERS_ASSERT_ALLOWLIST - set(found))
    assert not new and not stale, (
        f"{new = } -- raise explicitly instead of asserting (stripped under "
        f"`python -O`); {stale = } -- no longer present, drop from the allowlist"
    )


def _numeric_registry_bounds(tree: ast.Module) -> dict[str, list[int]]:
    """Map ``file: param`` -> line numbers where a registered default is out of range.

    Only fully resolvable numeric declarations are judged: a bare literal, or a
    name bound to a numeric literal at module level (``EXP_TIME_LOW``). Anything
    dynamic is skipped rather than guessed at, so the guard cannot invent findings.
    """
    consts: dict[str, object] = {}
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else ([node.target] if isinstance(node, ast.AnnAssign) else [])
        for target in targets:
            if isinstance(target, ast.Name):
                try:
                    consts[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass

    def resolve(expr: ast.expr | None) -> float | None:
        if expr is None:
            return None
        try:
            value = ast.literal_eval(expr)
        except (ValueError, TypeError):
            if isinstance(expr, ast.Name):
                value = consts.get(expr.id)
            else:
                return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    found: dict[str, list[int]] = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "register_parameter"
        ):
            continue
        kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        positional = list(node.args)
        default = resolve(kwargs.get("default_value", positional[1] if len(positional) > 1 else None))
        if default is None:
            continue
        low = resolve(kwargs.get("min_value", positional[2] if len(positional) > 2 else None))
        high = resolve(kwargs.get("max_value", positional[3] if len(positional) > 3 else None))
        if (low is not None and default < low) or (high is not None and default > high):
            name = positional[0] if positional and isinstance(positional[0], ast.Constant) else kwargs.get("name")
            label = name.value if isinstance(name, ast.Constant) else "<dynamic>"
            found.setdefault(label, []).append(node.lineno)
    return found


def test_registered_parameter_defaults_satisfy_declared_bounds() -> None:
    """A ``register_parameter`` default must lie inside that same call's bounds.

    The registry is a *declaration*, and a declaration that contradicts itself is
    worse than no declaration: it reads like an enforced contract while nothing
    enforces it. ``ThorlabWFS._register_parameters`` registered
    ``exposure_time_ms`` with ``min_value=EXP_TIME_LOW`` (0.002) and
    ``default_value=0.0`` -- its own default outside its own floor. The driver is
    not wrong; ``open()`` treats 0.0 as the legal "auto exposure" sentinel and
    resets any non-zero out-of-range value back to it, so the accepted domain is
    ``{0.0} u [EXP_TIME_LOW, EXP_TIME_HIGH]`` and the registry has to declare the
    floor as 0.0 to stay truthful about the value it ships with.

    Scanned every ``register_parameter`` call under ``src/``: 62 sites, 1 offender
    before the fix, 0 after. No allowlist -- there is no legitimate way for a
    default to violate its own declared range, so a future hit is a real defect.
    """
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        except (OSError, SyntaxError):
            continue
        rel = path.relative_to(REPO).as_posix()
        for label, lines in _numeric_registry_bounds(tree).items():
            for line in lines:
                offenders.append(f"{rel}:{line} {label}")
    assert not offenders, (
        f"{offenders = } -- a registered default sits outside its own declared "
        "min/max; declare the range the driver actually accepts (including any "
        "sentinel it documents) or move the bound to where it is enforced."
    )


#: GUI-level hardware limits that must be imported, never re-typed as literals.
#: ``SINE_AMP_MAX`` is excluded: it is derived from these two, not duplicated.
_GUI_DRIVER_BOUNDS = ("HW_VOLTAGE_MIN", "HW_VOLTAGE_MAX")


def _numeric_literal(node: ast.expr | None) -> float | None:
    """Return the number a literal expresses, **including a leading unary minus**.

    ``-20.0`` parses as ``UnaryOp(USub, Constant(20.0))``, so an
    ``isinstance(node, ast.Constant)`` test silently reports a negative bound as
    "not a literal" -- the same blind spot as grepping for one spelling of a
    form that has several.
    """
    if (
        isinstance(node, ast.Constant)
        and not isinstance(node.value, bool)
        and isinstance(node.value, (int, float))
    ):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        inner = _numeric_literal(node.operand)
        if inner is None:
            return None
        return -inner if isinstance(node.op, ast.USub) else inner
    return None


def test_gui_never_redeclares_driver_voltage_bounds() -> None:
    """``gui/`` must take R50Power voltage limits from the driver, not restate them.

    The limits exist once, in ``drivers/dm/micro/micro_constants.py`` (VOLTAGE_MIN
    / VOLTAGE_MAX, from the R50Power datasheet), and ``gui/r50/r50_channel_select.py``
    already binds ``HW_VOLTAGE_MIN: float = VOLTAGE_MIN`` through the PEP-562 lazy
    re-export. ``gui/dm/micro_dm_ui.py`` instead re-typed ``-20.0``/``120.0``, and
    the copy drifted in the only direction that matters: its sine-amplitude widget
    allowed ``max_value=140.0``, 20 V past the hardware ceiling. Nothing downstream
    complained because the driver clips silently, so the panel accepts a setting the
    hardware cannot produce. Amplitude is now ``SINE_AMP_MAX``: with ``offset +/- amp``
    both required to stay in range, adding the two inequalities gives
    ``2*amp <= VOLTAGE_MAX - VOLTAGE_MIN``, so 70.0 is the exact ceiling and every
    larger value only reached the truncation warning.

    Judged on the AST, not the text: the offending form is an assignment whose
    *value* is a numeric constant, which covers both module level and the ``Cfg``
    dataclass body, and survives a rename of the driver constant it should use.
    """
    offenders: list[str] = []
    for path in sorted((SRC / "ao_shaping" / "gui").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        except (OSError, SyntaxError):
            continue
        rel = path.relative_to(REPO).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign):
                targets, value = [node.target], node.value
            else:
                continue
            literal = _numeric_literal(value)
            if literal is None:
                continue
            for target in targets:
                if isinstance(target, ast.Name) and target.id in _GUI_DRIVER_BOUNDS:
                    offenders.append(
                        f"{rel}:{node.lineno} {target.id} = {literal!r}"
                    )
    assert not offenders, (
        f"{offenders = } -- voltage limits are declared in "
        "`drivers/dm/micro/micro_constants.py`; import them so the GUI cannot drift "
        "away from what the driver enforces."
    )


#: Thread bodies in ``gui/`` that still reach ``st.session_state``. Each entry is
#: recorded debt: the guard fails if it grows, **and** if it stops being true, so a
#: fix cannot quietly leave a stale exemption behind.
_GUI_THREAD_SS_ALLOWLIST: set[str] = set()


def _scan_thread_func(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[bool, set[str], set[str]]:
    """Return (touches session_state directly, names called, Thread targets spawned)."""
    touches = False
    called: set[str] = set()
    targets: set[str] = set()
    for node in ast.walk(func):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "session_state"
            and isinstance(node.value, ast.Name)
            and node.value.id == "st"
        ):
            touches = True
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            called.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr == "Thread":
                target_kw = next(
                    (kw.value for kw in node.keywords if kw.arg == "target"), None
                )
                if isinstance(target_kw, ast.Name):
                    targets.add(target_kw.id)
                elif node.args and isinstance(node.args[0], ast.Name):
                    targets.add(node.args[0].id)
            called.add(node.func.attr)
    return touches, called, targets


def _gui_thread_body_offenders() -> set[str]:
    """``file::function`` for every ``Thread(target=...)`` body that reaches session_state.

    Reachability is transitive and resolved to a fixpoint over same-module calls,
    because that is how the violations actually happened: ``_matrix_continuous_loop``
    wrote session_state through ``set_tab2_feedback``, and the single-unit loops
    wrote it through ``_single_send_v``. A scan that only inspects the thread body's
    own attribute accesses reports all three as clean while shipping the race.
    """
    for path in sorted((SRC / "ao_shaping" / "gui").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        except (OSError, SyntaxError):
            continue
        rel = path.relative_to(REPO).as_posix()

        direct: set[str] = set()           # bodies with their own session_state access
        graph: dict[str, set[str]] = {}    # function -> names it calls
        thread_targets: set[str] = set()

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            touches, called, targets = _scan_thread_func(node)
            if touches:
                direct.add(node.name)
            graph.setdefault(node.name, set()).update(called)
            thread_targets.update(targets)

        reaching = set(direct)
        changed = True
        while changed:
            changed = False
            for caller, callees in graph.items():
                if caller in reaching:
                    continue
                if any(callee in reaching for callee in callees):
                    reaching.add(caller)
                    changed = True

        for target in thread_targets & reaching:
            yield f"{rel}::{target}"


def test_gui_thread_bodies_never_reach_session_state() -> None:
    """No ``Thread(target=...)`` body under ``gui/`` may touch ``st.session_state``.

    AGENTS.md states the rule and the approved shape (parameter snapshot in a
    plain dict, ``threading.Event`` for stop, ``queue.Queue`` drained by the main
    thread on rerun), but until now nothing enforced it. ``st.session_state`` is
    keyed off Streamlit's ``ScriptRunContext``; touching it from a hand-spawned
    thread produces the ``missing ScriptRunContext`` path and races the rerun that
    is reading the same value. The three loops in ``gui/dm/micro_dm_ui.py`` polled
    ``mdm_single_running`` / ``mdm_matrix_continuous_running`` **as the loop
    condition, every tick**, and appended to the history deques every tick -- the
    widest possible window -- while ``disconnect_dm`` never stopped the matrix
    thread at all, leaving it driving an already-closed device until an exception
    tripped it.

    The allowlist is two-way on purpose (same convention as
    ``test_drivers_only_assert_internal_invariants``): the remaining two offenders
    are real but pre-date this change and are tracked as debt in TODO.md 8.9.
    """
    found = set(_gui_thread_body_offenders())
    new = sorted(found - _GUI_THREAD_SS_ALLOWLIST)
    stale = sorted(_GUI_THREAD_SS_ALLOWLIST - found)
    assert not new and not stale, (
        f"{new = } -- thread bodies must receive their device handle, a parameter "
        "snapshot, a threading.Event and a queue.Queue as arguments; only the main "
        "thread touches st.session_state (AGENTS.md, 'Streamlit background threads'). "
        f"{stale = } -- fixed; drop it from the allowlist."
    )


_LOGURU_LEVELS = {
    "debug",
    "info",
    "success",
    "warning",
    "error",
    "critical",
    "exception",
    "log",
}
_PERCENT_SPEC = re.compile(r"%[-+ #0]*\d*(?:\.\d+)?[sdifge]")
_LOGURU_IMPORT = re.compile(r"^\s*from loguru import logger\b", re.M)
_LOGGER_REEXPORT = re.compile(r"^\s*from ao_shaping[\w.]* import [^(]*\blogger\b", re.M)
_STDLIB_LOGGER = re.compile(r"logging\.getLogger\(")


def _uses_loguru_logger(text: str) -> bool:
    """Whether this module's ``logger`` is loguru's -- by import, or by re-export.

    ``ao_shaping.utils`` re-exports the loguru logger, so filtering on the literal
    ``from loguru import logger`` alone leaves ~20 modules unscanned; a guard whose
    reach is narrower than its name is the 8.10 failure mode.
    """
    if _STDLIB_LOGGER.search(text):
        return False
    return bool(_LOGURU_IMPORT.search(text) or _LOGGER_REEXPORT.search(text))


def _dropped_arg_loguru_calls() -> list[str]:
    """Every loguru call that passes %-style arguments (which loguru discards).

    Covers ``src/``, ``tests/`` and ``scripts/`` -- the same claim the test name makes.

    Scoped to modules whose ``logger`` is loguru's (see :func:`_uses_loguru_logger`):
    under the stdlib ``logging`` module ``%``-args are the correct form, and
    ``gui/utils/phase_overlay.py`` uses that module -- flagging it would be wrong,
    not merely noisy.
    """
    offenders: list[str] = []
    roots = (SRC, REPO / "tests", REPO / "scripts")
    for path in sorted(p for root in roots for p in root.rglob("*.py")):
        text = path.read_text(encoding="utf-8-sig")
        if not _uses_loguru_logger(text):
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:  # ruff owns syntax errors
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in _LOGURU_LEVELS:
                continue
            receiver = node.func.value
            name = (
                receiver.id
                if isinstance(receiver, ast.Name)
                else getattr(receiver, "attr", "")
            )
            if "logger" not in str(name).lower():
                continue
            # logger.log(level, msg, ...) keeps the message in args[1], not args[0].
            msg_at = 1 if node.func.attr == "log" else 0
            if len(node.args) < msg_at + 2:
                continue
            msg = node.args[msg_at]
            if not isinstance(msg, ast.Constant) or not isinstance(msg.value, str):
                continue
            if "{" in msg.value:  # already slot-formatted
                continue
            if _PERCENT_SPEC.search(msg.value):
                offenders.append(f"{path.relative_to(REPO)}:{node.lineno}")
    return offenders


def test_loguru_calls_never_pass_percent_style_arguments() -> None:
    """A loguru message using ``%s`` loses every argument it was given.

    loguru interpolates with ``str.format``, so ``logger.info("x %s", v)`` prints
    the literal ``x %s`` and drops ``v`` -- a diagnostic that reports no values at
    all, which is the same unattributable-failure class as a swallowed exception.
    Measured **19 pre-existing** calls in 3 loguru modules (16 in the training-data
    collector, plus its zero-order-freeze line); the runtime proof is a captured log
    reading exactly ``零阶定位（冻结）：(%.1f, %.1f)``. 21 were converted in total, the
    extra 2 having been introduced by the fix that uncovered the problem -- i.e. the
    first version of that fix reproduced the very defect it was curing. Converted with
    precision preserved (``%.1f`` -> ``{:.1f}``). Non-vacuity: this guard reports those
    19 sites against the pre-fix tree and 0 after (TODO.md 8.9 item 7).
    """
    offenders = _dropped_arg_loguru_calls()
    assert not offenders, (
        f"{len(offenders)} loguru call(s) pass %-style arguments that loguru silently "
        f"discards: {offenders}. Use brace slots instead -- '{{}}' for a plain value, "
        "'{:.1f}' to keep precision."
    )


def test_the_loguru_guard_follows_the_logger_not_one_spelling_of_its_import() -> None:
    """The scan must cover re-exported loggers, and skip stdlib ``logging`` files.

    ``ao_shaping.utils`` re-exports the loguru logger, so filtering on the literal
    import line left ~20 modules unscanned -- this guard's reach first came up short of
    its own name. ``phase_overlay.py`` is the opposite case: it uses stdlib ``logging``,
    where ``%``-args are correct, so it must stay out of the scan rather than be
    converted. Each side of that boundary is pinned here, because a filter that quietly
    narrows turns a hard gate into a no-op.
    """
    reexport = (
        SRC / "ao_shaping" / "optimizer" / "combined_optimizer.py"
    ).read_text(encoding="utf-8-sig")
    stdlib = (SRC / "ao_shaping" / "gui" / "utils" / "phase_overlay.py").read_text(
        encoding="utf-8-sig"
    )

    assert "from loguru import logger" not in reexport, (
        "combined_optimizer now imports loguru directly -- this test's premise moved, "
        "so the re-export half of the filter is no longer being exercised"
    )
    assert _uses_loguru_logger(reexport), "a re-exported logger escaped the scan"
    assert not _uses_loguru_logger(stdlib), (
        "stdlib logging is being scanned -- % formatting is correct there, so this "
        "would report working code as a defect"
    )


# Measured 2026-10-25 by the same AST rule used below. Each of these handlers was read
# in context and judged legitimate (TODO.md 8.9 items 5-10); the number is a floor to
# defend, not an endorsement. Lowered 61 -> 60 on 2026-10-28: the collector's
# recognised-family probe stopped being a blind silent handler when its optional-import
# surface was narrowed to ImportError. Lowered again 60 -> 58 the same day after
# rebasing onto the #56 driver slice, which narrowed two of the silent ones.
# Lowered 58 -> 46 on 2026-10-28 by #56 slice (3) batches A/B/C/D/E: 12 more silent
# ones gained a concrete type instead of `except Exception` -- miicam's
# ctypes-SDK teardown (14 -> `OSError`), thorlab_wfs/`_sdk_setup` file I/O,
# the adc/santec guards, and the two `open()`+csv / `open()`+write sites in
# voltage_runner and adc_dm_adam. A typed silent handler is still silent, but it is
# no longer *unattributable*: the reader can see which failure class is being dropped.
# Lowered 46 -> 44 on 2026-10-29, twice over by both sides of a merge: `15ea0a22`
# moved pixel size into the driver, deleting the GUI's GenICam ``PixelSize`` probe and
# the daheng/miicam probe's blind fallbacks (two handlers stopped existing rather than
# becoming typed), and the src/ moves in 062a64e5..fae245db (transfer/validate.py split,
# exposure_control.py, record_mode.py) retired two more silent sites. The number below is
# taken from the guard's own walk, not recounted by hand.
_BLIND_SILENT_BASELINE = 44


def _blind_silent_handlers(root: Path) -> list[str]:
    """Blind ``except`` / ``except Exception`` handlers whose body only ignores.

    A handler that logs, re-raises, or runs recovery is **not** counted: the defect
    being pinned is an *unattributable* failure, not broad matching on its own -- most
    broad handlers in this tree are correct (teardown, enumeration control flow,
    retry-then-raise) and were individually confirmed as such.
    """
    found: list[str] = []
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        except (SyntaxError, UnicodeDecodeError):  # ruff owns these
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            blind = node.type is None or (
                isinstance(node.type, ast.Name)
                and node.type.id in {"Exception", "BaseException"}
            )
            if not blind:
                continue
            if all(isinstance(stmt, (ast.Pass, ast.Continue)) for stmt in node.body):
                found.append(f"{path.relative_to(REPO)}:{node.lineno}")
    return found


def test_blind_silent_exception_handlers_in_src_do_not_grow() -> None:
    """No new blind handler may swallow a failure in the production tree.

    Five sweeps read every such handler in ``src/``, ``tests/`` and ``scripts/`` in
    context (TODO.md 8.9 items 5-10) and the verdict was consistent: the count is not
    the target, the question is whether the swallow makes a wrong number or a lost
    measurement unattributable. Real defects found that way include an MIICAM frame
    callback that dropped frames with zero trace, a training-data collector that could
    lose a record from the persisted corpus, a lost ADC sample series, a swallowed DM
    zero-voltage safety reset, and a hardware report that silently published simulation
    numbers as device evidence.

    This ratchet keeps that verdict from decaying. It is deliberately **count-based
    rather than an allowlist of ``path:line``**: line pins drift on every unrelated
    insertion (this repo has already had to re-pin stale ones), so a pin list would rot
    while a count cannot. The offender list is printed so a regression stays diffable.

    Scope is ``src/`` only, matching the claim in the name: in ``tests/`` a swallow
    risks a vacuously green assertion and in ``scripts/`` a missing report row, but both
    also contain legitimate harness teardown, and folding them in would make the guard
    cry wolf on ordinary test churn.
    """
    offenders = _blind_silent_handlers(SRC)
    if len(offenders) > _BLIND_SILENT_BASELINE:
        raise AssertionError(
            f"blind silent exception handlers in src/ grew: "
            f"{len(offenders)} > baseline {_BLIND_SILENT_BASELINE}. Judge the new one in "
            "context first: if a failure it hides could make a number wrong or a "
            "measurement lost, report it (log with the reason and a count) instead of "
            f"silencing it. Current set:\n  " + "\n  ".join(sorted(offenders))
        )
    assert len(offenders) == _BLIND_SILENT_BASELINE, (
        f"count fell to {len(offenders)} < baseline {_BLIND_SILENT_BASELINE} -- good "
        "news, but lower the baseline deliberately so it stays a measured fact rather "
        "than a number nobody re-derives."
    )


#: Names that belong to a device's own lifecycle. An ``assert`` about one of these is a
#: precondition on hardware state, not an invariant about a locally computed value.
_DEVICE_STATE_NAMES = (
    "is_open", "is_connected", "is_ready", "_instrument_handle", "_dll", "_handle",
    "_device_open", "_ensure_open", "_sdk", "_miicam", "_session",
)


def _device_state_asserts_outside_drivers() -> list[str]:
    found: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC)
        if rel.parts[:2] == ("ao_shaping", "drivers"):  # pinned separately
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        except (SyntaxError, UnicodeDecodeError):  # ruff owns these
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assert):
                continue
            test = ast.unparse(node.test).lower()
            if any(name in test for name in _DEVICE_STATE_NAMES):
                found.append(
                    f"{rel}:{node.lineno}: assert {ast.unparse(node.test)[:60]}"
                )
    return found


def test_no_assert_on_device_state_outside_the_drivers_tree() -> None:
    """Device preconditions must be explicit raises, never ``assert``.

    Two independent reasons, both already paid for in this tree: an ``assert`` is
    stripped whole by ``python -O``, and even unstripped it raises the wrong class --
    callers cannot narrow on it, which is exactly the defect 92dbe716 chased through
    ``except AssertionError`` handlers. A guard that decides whether a device may be
    touched has to survive both.

    Measured 2026-10-25, whole ``src/``: 36 asserts, of which 5 live in ``drivers/``
    (all internal invariants, pinned by :data:`_DRIVERS_ASSERT_ALLOWLIST`) and 31
    elsewhere -- the last 31 guard values the calling function just computed
    (``img.ndim == 2``, ``G.ndim >= 2``, ``self.lut is not None``, shape equality) and
    **stay** ``assert`` per 8.10. Zero of them touch device state, so the baseline here
    is empty by measurement rather than by decree.

    Deliberately narrow: it fires on device-lifecycle identifiers only. Broadening it to
    "any assert in production" would re-litigate the 8.10 decision to keep internal
    invariants as asserts, which cost nothing and read better than a raise.
    """
    offenders = _device_state_asserts_outside_drivers()
    assert not offenders, (
        f"{len(offenders)} assert(s) outside drivers/ guard device state, which "
        "`python -O` deletes and which raises a type callers cannot catch. Move the "
        "check into the driver as an explicit raise of its own error class "
        "(CameraError / WfsError / DeviceError):\n  " + "\n  ".join(offenders)
    )


def _thread_boundary_capture_handlers() -> list[str]:
    """Handlers that stash a caught exception so another thread can re-raise it.

    The re-raise may be a name (``raise error[0]`` from a list) or a subscript
    (``raise box["error"]``); matching only the name form misses the second shape,
    and a guard that misses one of the three sites would enforce less than it claims.
    """
    found: list[str] = []
    for root in (SRC, REPO / "scripts", REPO / "tests"):
        for path in sorted(root.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            except (SyntaxError, UnicodeDecodeError):  # ruff owns these
                continue
            for fn in ast.walk(tree):
                if not isinstance(fn, ast.FunctionDef):
                    continue
                if not any(
                    isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "Thread"
                    for c in ast.walk(fn)
                ):
                    continue
                for handler in ast.walk(fn):
                    if not isinstance(handler, ast.ExceptHandler) or not handler.name:
                        continue
                    containers: set[str] = set()
                    for stmt in handler.body:
                        for node in ast.walk(stmt):
                            if isinstance(node, ast.Assign) and isinstance(
                                node.value, ast.Name
                            ) and node.value.id == handler.name:
                                for tgt in node.targets:
                                    if isinstance(tgt, ast.Name):
                                        containers.add(tgt.id)
                                    elif isinstance(tgt, ast.Subscript) and isinstance(
                                        tgt.value, ast.Name
                                    ):
                                        containers.add(tgt.value.id)
                            if (
                                isinstance(node, ast.Call)
                                and getattr(node.func, "attr", "") == "append"
                                and isinstance(node.func.value, ast.Name)
                                and any(
                                    isinstance(a, ast.Name) and a.id == handler.name
                                    for a in node.args
                                )
                            ):
                                containers.add(node.func.value.id)
                    if not containers:
                        continue
                    re_raised = any(
                        isinstance(r, ast.Raise)
                        and r.exc is not None
                        and any(
                            n.id in containers
                            for n in ast.walk(r.exc)
                            if isinstance(n, ast.Name)
                        )
                        for r in ast.walk(fn)
                    )
                    if re_raised:
                        kind = (
                            ast.unparse(handler.type) if handler.type else "bare except"
                        )
                        found.append(
                            f"{path.relative_to(REPO)}:{handler.lineno}: {kind}"
                        )
    return found


def test_thread_boundary_exception_capture_keeps_base_exception() -> None:
    """A worker thread's caught exception must be re-raisable in full, not just Exception.

    These handlers exist to move an exception across a thread boundary so the calling
    thread can re-raise it (``call_with_timeout`` and its script copies). Narrowing one
    to ``except Exception`` compiles, tests keep passing, and the behaviour change is
    silent: the worker's ``KeyboardInterrupt``/``SystemExit`` is never stored, so the
    caller sees a completed call instead of an interrupt -- the timeout helper returns
    ``None`` as if the SDK call had succeeded. 8.7 records this as forbidden prose;
    this pins it mechanically.

    Measured 2026-10-25: 3 such handlers, all already ``BaseException``, so the
    baseline is empty. The premise (that the scan finds any at all) is asserted too --
    a scan matching nothing would pass trivially.
    """
    handlers = _thread_boundary_capture_handlers()
    assert handlers, (
        "no thread-boundary exception-capture handler found anywhere: the shape this "
        "guard is about has changed, so it is enforcing nothing"
    )
    narrowed = [h for h in handlers if not h.endswith(": BaseException")]
    assert not narrowed, (
        f"{len(narrowed)} thread-boundary handler(s) narrowed below BaseException -- "
        "an interrupt inside the worker would then vanish and the caller would treat "
        f"the call as successful:\n  " + "\n  ".join(sorted(narrowed))
    )


def _ctypes_imports_in_utils() -> list[str]:
    """Files under ``utils/`` binding a native SDK through ``ctypes``."""
    found: list[str] = []
    for path in sorted((SRC / "ao_shaping" / "utils").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        except (SyntaxError, UnicodeDecodeError):  # ruff owns these
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(a.name == "ctypes" or a.name.startswith("ctypes.") for a in node.names):
                    found.append(f"{path.relative_to(REPO)}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "ctypes":
                found.append(f"{path.relative_to(REPO)}:{node.lineno}")
    return found


def test_the_utils_leaf_layer_never_binds_a_native_sdk() -> None:
    """``utils/`` is the leaf layer: no ``ctypes``, so no vendored driver bindings.

    ``utils/wavefront/vi.py`` (188 lines of ctypes VISA) was deleted in bbb3820b,
    which left the invariant behind as prose only -- nothing stopped the same shape
    from coming back under a new filename. Native SDK binding belongs in
    ``drivers/``, which is where the lazy-import contract, the ``hardware`` gate and
    the device error classes live; a ``ctypes`` import in ``utils/`` silently breaks
    the "import utils without a device or SDK" property the offline probes rely on.

    Measured 2026-10-25: 0 such imports in ``src/ao_shaping/utils/``. The baseline is
    an empty set, not a ``file:line`` allowlist, so there is nothing to re-pin.
    """
    offenders = _ctypes_imports_in_utils()
    assert not offenders, (
        f"{len(offenders)} file(s) under utils/ import ctypes. That layer is the "
        "import-light leaf the offline probes and tools depend on; move the binding "
        "into drivers/ (lazy registration in drivers/_lazy.py) instead:\n  "
        + "\n  ".join(sorted(offenders))
    )


_MD_SKIP = re.compile(
    r"(^|/)(call_graph_wiki|node_modules|\.venv|__pycache__|\.git)(/|$)"
)
_MD_FENCE = re.compile(r"^\s*(```|~~~)")
_MD_SEP_ROW = re.compile(r"^\|[\s:|-]*-[\s:|-]*\|?\s*$")
_MD_ENTRY = re.compile(r'"([^"]+\.md)":\s*ReportProvenance\(')
# The entry's OWN script field: a fixed window past the key spills into the next
# entry, and a neighbouring script=HANDWRITTEN then reclassifies a generated report
# as hand-written (measured: models_report/report.md, 4 offenders mis-bucketed).
_MD_SCRIPT_FIELD = re.compile(r'script\s*=\s*(HANDWRITTEN|"[^"]+")')

# Driven to zero in three stages (TODO 8.9 item 27): 24 -> 19 by correcting two wrong
# DELIMITER rows (generate_models_report declared 7 for an 8-column header,
# generate_pib_bench 10 for a 9-column header -- one line each cleared five rows);
# 19 -> 7 by escaping 8 hand-written row literals in the emitting generators so a
# regeneration matches the artifact; 7 -> 0 by wiring the miicam probe's render path.
# Kept as a SEPARATE bucket from the hand-written one even though both are empty, so a
# future split in a generated file names the generator as the place to fix, not the .md.
# This number may only go DOWN -- and it cannot, since DOWN from 0 is a failure.
_MD_GENERATED_SPLIT_BASELINE = 0


def _unfenced_md_lines(text: str) -> list[tuple[int, str]]:
    """Markdown lines with fenced bodies blanked, original line numbers kept.

    Table-shaped text inside a fence must not be read as a table: this repo documents
    row formats inside fenced blocks, and parsing those would report defects no
    renderer ever shows. A line is kept only while the fence is CLOSED -- inverted
    once already, which is why the caller bounds the block count by a measured
    magnitude instead of ``> 0``.
    """
    lines: list[tuple[int, str]] = []
    open_fence: str | None = None
    for lineno, line in enumerate(text.splitlines(), 1):
        mark = _MD_FENCE.match(line)
        if mark:
            if open_fence is None:
                open_fence = mark.group(1)
            elif mark.group(1) == open_fence:
                open_fence = None
            lines.append((lineno, ""))
            continue
        lines.append((lineno, line if open_fence is None else ""))
    return lines


def _md_pipe_count(line: str) -> int:
    """Cell delimiters in a table row: every ``|`` that is not GFM-escaped."""
    return len(re.findall(r"(?<!\\)\|", line))


def _generated_report_paths() -> tuple[set[str], set[str]]:
    """Report files split into generated vs hand-written, per provenance.REPORTS.

    The repo already records which report a script owns and which are 人工撰写, so the
    split reuses that single source instead of treating everything under report/ as
    generated: hand-written summaries live there too, and a directory-keyed exclusion
    would have protected their split cells forever.
    """
    text = (REPO / "scripts" / "_common" / "provenance.py").read_text(encoding="utf-8")
    generated: set[str] = set()
    handwritten: set[str] = set()
    for match in _MD_ENTRY.finditer(text):
        field = _MD_SCRIPT_FIELD.search(text, match.end())
        if field is None:
            continue  # no script field: unknown, falls into the must-be-zero bucket
        if field.group(1) == "HANDWRITTEN":
            handwritten.add(match.group(1))
        else:
            generated.add(match.group(1))
    return generated, handwritten


def _markdown_table_cell_splits() -> tuple[list[str], int]:
    """Rows writing more cells than their own separator row declares."""
    offenders: list[str] = []
    blocks = 0
    for path in sorted(REPO.rglob("*.md")):
        rel = path.relative_to(REPO).as_posix()
        if _MD_SKIP.search(rel):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:  # ruff owns encoding failures
            continue
        rows: list[tuple[int, str]] = []
        table_blocks: list[list[tuple[int, str]]] = []
        for lineno, line in _unfenced_md_lines(text):
            if line.startswith("|"):
                rows.append((lineno, line))
            elif rows:
                table_blocks.append(rows)
                rows = []
        if rows:
            table_blocks.append(rows)
        blocks += len(table_blocks)
        for block in table_blocks:
            if len(block) < 2 or not _MD_SEP_ROW.match(block[1][1]):
                continue  # no separator row: prose shaped like a table
            declared = _md_pipe_count(block[1][1])
            # The header is checked in BOTH directions: a header/delimiter mismatch is
            # not a cosmetic wobble, GFM refuses to see the table at all.
            header = _md_pipe_count(block[0][1])
            if header != declared:
                offenders.append(
                    f"{rel}:{block[0][0]}: header writes {header - 1} cell(s), the "
                    f"delimiter row declares {declared - 1} -- the block is not parsed "
                    f"as a table at all: {block[0][1][:60]}"
                )
            # Data rows only in ONE direction: GFM pads a short row and the trailing
            # pipe is optional, so fewer cells is legal and must not be flagged.
            for lineno, line in block[2:]:
                written = _md_pipe_count(line)
                if written > declared:
                    offenders.append(
                        f"{rel}:{lineno}: delimiter declares {declared - 1} cell(s), "
                        f"this row writes {written - 1}: {line[:70]}"
                    )
            # Zero tolerance, both directions: an escaped OUTER pipe means the row lost
            # its closing (or opening) delimiter. The arity rule above cannot see that --
            # a row with a missing closing pipe simply writes FEWER cells, which GFM
            # pads and this guard deliberately allows. It is the signature of a bad
            # mechanical escape, not of content.
            for lineno, line in block:
                body = line.rstrip()
                if body.endswith("\\|") or body.startswith("\\|"):
                    offenders.append(
                        f"{rel}:{lineno}: the row's outer delimiter was escaped, so the "
                        f"row lost its closing/opening pipe: {line[:70]}"
                    )
    return offenders, blocks


#: Ratchet for issue #174: blind ``except Exception`` / ``except BaseException`` handlers
#: under ``src/``. Counted by AST rather than by ruff because the test images do not install
#: ruff (a ruff-based count would skip in exactly the place the gate runs), and the check is
#: therefore unskippable. ``re_raise_only`` handlers are excluded, mirroring BLE001's own
#: exemption, so the number is comparable to ``ruff check src --select BLE001`` (measured
#: 363 today against this AST count of 362; the residual gap is ruff's contextual filtering).
#: This is a ratchet, not a claim that 362 is acceptable: most remaining handlers are
#: deliberate fallbacks, and each one still needs an individual judgement.
#:
#: The population shrank by one on 2026-10-28 rather than every handler staying blind: the R50
#: channel selector's wiring-CSV parse handler was narrowed to the three types its guarded
#: statements can actually raise (KeyError on a missing header, ValueError/TypeError on the
#: int() conversions). That is the first executed narrowing in the GUI layer, and it sits on a
#: pure-parse path -- no device call, so CI can see the whole failure surface. It is pinned by
#: tests/ao_shaping/gui/r50/test_r50_channel_select.py, which asserts both that malformed rows
#: still degrade to an empty table and that an unlisted error is no longer swallowed. Hardware
#: paths remain deliberately un-narrowed until the driver exception taxonomy (#145 / #45).
#: The second executed narrowing is the same shape on a different axis: the collector's
#: recognised-family probe wraps one optional-package import plus a defaulted getattr, so its
#: entire surface is ImportError and it was narrowed to that. Pinned by
#: tests/ao_shaping/tools/slm/test_slm_train_data_collect.py, whose last case is the one that
#: justifies the change -- a probe failure that is not an import problem now reaches the caller
#: instead of silently returning the built-in family list.
#: The population then fell by 18 more handlers, none of them narrowed by this workstream. #56
#: slices (3) A-E landed underneath, and their commit messages report 17 + 16 + 3 + 2 narrowed
#: sites -- a larger number than this guard can see, because those counts span scripts/ and tests/
#: while this ratchet scans src/ only. What is verifiable here is the net: 362 -> 344 inside src/.
#: Their miicam narrowing rests on real evidence (the vendor's HRESULTException is an OSError
#: subclass on win32), which is exactly the kind of justification the rule asks for, so the ceiling
#: is lowered to the integrated measurement rather than argued about. Of those 18 departures, **11
#: had been annotated and 7 had not**, which is why the marked count falls from 300 to 289 even
#: though this workstream added markers the whole time: an above-line marker survives the edit that
#: narrows its handler, but it stops counting once the handler leaves the blind population. Do not
#: read a fall in the marked number as lost work -- check the total in the same line.
_MAX_BLIND_EXCEPT_HANDLERS_IN_SRC = 344

#: Second ratchet over the same population: a broad catch that has actually been *judged*
#: carries a ``# policy:`` marker naming what the layer is protecting -- within two lines
#: above the ``except``, or in the first three lines of the handler body. Counting the
#: unmarked remainder is what makes progress reviewable. It rewards writing the judgement
#: down, and it deliberately does not press the tree toward mechanical ``except ValueError``
#: edits on hardware paths that CI cannot exercise: narrowing a driver call's handler without
#: the driver exception taxonomy (issue #145 / #45) would turn a working fallback into a crash.
#:
#:
#: Measured 2026-10-28 after twenty-five marker passes and two executed narrowings (runners/slm + config + utils, the three CCD
#: camera drivers, the R50 GUI loop files, the whole optimizer / runner CLI layer, the ml
#: training scripts, the CCD analyzer UI, the bench settings panel, the Micro-DM panel, the
#: target-shape helper, the R50 joint panel, the SLM calibration panel, the Micro-DM image
#: analysis panel, the shaping-controls panel, and the R50 channel selector, then the first
#: executed narrowing): 289 marked, 55
#: unmarked on the integrated tree (300 / 62 when the twenty-sixth pass landed, before #56 (3) A-E).
#: The fourth pass covered only 12 of the 17 open optimizer sites and the fifth had
#: to close the layer -- the ceiling is only as honest as the last count. One site left the
#: population instead of gaining a marker (narrowed out, see the total ratchet above), so the
#: marked count fell while this ceiling held: a real narrowing is worth more than a marker, and
#: unlike a marker it moves the total too.
#: The sixteenth marker pass (the R50 controller-UI family: connection helpers, group panel,
#: single-controller panel, tab host) moved this ceiling 194 -> 182 with 12 comment-only edits.
#: Every one of those sites guards a device call, so per the note above they stay broad and are
#: judged rather than narrowed; the recurring reason written into the markers is that teardown
#: and batch loops must attempt every step, because aborting midway leaves some channels
#: energised and others not. Control re-armed: one more unannotated blind catch makes 56 red.
#: The seventeenth marker pass (six scattered GUI files outside the R50 family: the transfer
#: progress panel, the Zernike debug viewer, both ceramic viewers, the pyarrow probe and the
#: phase overlay helper) moved this ceiling 182 -> 171 with 11 comment-only edits. Three of those
#: sites took a trailing comment on the except line and one converted a rationale that was
#: already there, so every line number those files are cited at stayed put.
#: The eighteenth marker pass (the multi-SLM controller panel, 19 sites) moved this ceiling
#: 171 -> 152. Ten of those sites sit above the line docs/issues_report.md pins, so they took a
#: trailing comment and one converted the rationale already sitting on its handler body; only the
#: eight below the pin grew the file. Nothing left the blind population: all 19 are SLM SDK or
#: session-state calls, whose failure type belongs to the vendor or to Streamlit, not to us.
#: The nineteenth marker pass (the Zernike response-matrix UI, 19 sites) moved this ceiling
#: 152 -> 133 and zeroed the GUI layer. Seventeen of those sites sit above a line TODO.md cites
#: by number, so they took trailing comments and the file grew by two lines in total -- only the
#: two sites below that citation could afford to grow. All 19 are SLM/WFS SDK, disk or Streamlit
#: calls: the narrowing question never arises, since the failure type belongs to the vendor, the
#: filesystem or Streamlit rather than to this code.
#: The twentieth marker pass (tools/zernike/common.py, 7 sites) moved this ceiling 133 -> 126 and
#: opened the tools layer. All seven guard a vendor SDK call or an SDK property read --
#: get_wavelength_info, get_hardware_info, pupil, get_mla_name, get_zernike twice, and a getattr on
#: an arbitrary handle -- so none has an enumerable failure surface and band-16 sends them to a
#: marker rather than a narrowing. The file's only line-number citations sit at :6, :8, :36, :40 and
#: :41, all above every handler, so the markers were written as trailing comments and the file stayed
#: exactly 700 lines.
#: The twenty-first marker pass (the SLM training-data collector, 13 sites) plus the second
#: executed narrowing (its recognised-family probe) moved this ceiling 126 -> 112. Every marked
#: site there guards a device call, an SLM-assisted phase conversion, a disk flush or a
#: caller-supplied callback, so none had an enumerable surface. The one narrowed site wraps a single
#: optional-package import, and it is the only handler in this file that left the blind population.
#: That file is cited by line number at :187, :683-687, :970, :975 and :980, so all thirteen markers
#: went onto the except line itself and the file stayed at exactly 987 lines.
#: The twenty-second marker pass (the SLM phase-capture calibration tool, 8 sites) moved this
#: ceiling 94 -> 86. All eight guard a device call -- capture_frame, Santec open, open_camera twice,
#: three closes and a display_phase over memory slots -- so the narrowing question never arises; the
#: recurring rationale is that either camera may be absent and teardown must attempt every device.
#: That file carries no line-number citation, but the markers still went onto the except lines so the
#: pass cannot shift anything a later document might cite: 605 lines before and after.
#: The twenty-third marker pass (the Micro-DM per-channel image collector, 8 sites) moved this
#: ceiling 86 -> 78. Two of those handlers log and then re-raise, and they are still counted: the
#: exemption here is as narrow as the guard's own test -- a handler is excused only when its entire
#: body is a bare ``raise``, so a log-then-re-raise wrapper stays in the population. TODO.md pins
#: this file at :465-467 (the ``with_params`` wiring), and that pin sits between handlers -- :380
#: above it in source order is :547 -- so all eight markers went onto their except lines: the file
#: stayed at 713 lines and the three pinned lines are byte-identical.
#: The twenty-fourth marker pass (the SLM cartographer panel, 7 sites) moved this ceiling 78 -> 71.
#: All seven guard a device connect/disconnect path or the LUT calibration worker. Two of them are
#: ``except Exception: pass`` -- genuinely silent -- and a marker does not change that, so the
#: blind-silent floor is untouched by this pass. TODO.md cites this file at :99 and :145, and both
#: are handler lines themselves, so a trailing comment keeps each citation on the statement it names
#: while any inserted line above would have moved it: 792 lines before and after.
#: The twenty-fifth marker pass (the WFS tool family: wfs_reference, wfs_probe, zernike_correction,
#: zernike_response -- 9 sites) moved this ceiling 71 -> 62. Four of those nine are top-level flow
#: guards that deliberately catch the ValueError raised by the WFS exposure safety ceiling on top of
#: the vendor SDK failures, so narrowing them would let the ceiling escape as a traceback instead of
#: the documented [FAIL] line plus exit code. That is a case where the broad catch is load-bearing by
#: contract, not by laziness. This family is also the most citation-dense one yet (:15, :27, :159,
#: :161, :219, :309, :430, :554 and :713 across the four files), and :713 is itself a handler line,
#: so all nine markers are trailing comments and every one of those cited lines was asserted either
#: byte-identical or, for the marked one, still holding the same statement.
#: Measured split of what is left after #56 slices (3) A-E landed: 32 in tools/, 23 in drivers/.
#: The first driver slice narrowed 17 handlers and added two blind handlers of its own in the
#: transfer tooling; slices A-E narrowed 18 more (16 + 3 + 2).
#: The convention gap flagged above is CLOSED (2026-10-29), and closing it changed what this
#: ceiling means far more than any further marking pass could. The guard now credits an *attached
#: rationale* -- trailing on the handler line, in the comment lines directly above it, or as the
#: handler body's leading comments, which is the shape #56 actually writes (``保留宽泛捕获: ...
#: 失败类型不可枚举``) -- and a handler whose last statement is ``raise`` (cleanup-then-propagate
#: is a decision: the exception still escapes, nothing is swallowed). Re-measured under that rule:
#: 340 blind handlers, 285 carrying ``# policy:``, 48 carrying an attached prose rationale, 6 that
#: re-raise, and **1** without a self-contained judgement -- ``# 同上`` at
#: ``tools/kernels/device_mutex.py:1052``, now written out in full. So the ceiling goes 55 -> 0:
#: any new blind handler under ``src/`` must record its judgement at the handler, in either idiom.
#: This does not soften the ratchet on the risk it exists for -- narrowing a hardware-path handler
#: without the driver exception taxonomy (#145 / #45) is still what the failure message warns
#: against, and a decorative or two-character comment is not a rationale (``_is_rationale``).
_MAX_UNMARKED_BLIND_EXCEPT_HANDLERS_IN_SRC = 0


#: Tool directives are not judgements. The character classes are ASCII on purpose: ``\w`` matches
#: CJK, so a ``pragma:[\w -]*`` stripper swallows the Chinese rationale that follows
#: ``pragma: no cover -`` and reports an annotated handler as unannotated.
_COMMENT_DIRECTIVE = re.compile(
    r"^(?:pragma:[A-Za-z0-9 -]*|noqa[A-Za-z0-9,: -]*|type:[A-Za-z0-9 \[\], -]*"
    r"|ruff:?[A-Za-z0-9 -]*)[-:]?\s*",
    re.IGNORECASE,
)
_DECORATION_ONLY = re.compile(r"^[\s\-=#*_+.·─━|~^]+$")


def _is_rationale(comment: str) -> bool:
    """Would this comment, read alone, tell a reader why the broad catch is intended?"""
    body = _COMMENT_DIRECTIVE.sub("", comment.lstrip("#").strip()).strip()
    if not body or _DECORATION_ONLY.match(body):
        return False
    if re.search(r"[\u4e00-\u9fff]", body):
        return len(body) >= 6            # 6 字 reads as a clause; ``同上`` (2 字) does not
    return len(body.split()) >= 3


def _comment_tokens(text: str) -> dict[int, str]:
    """Real COMMENT tokens keyed by start line.

    ``tokenize``, never ``'#' in line``: a ``#`` inside a string literal must not be able to
    pass as a rationale, and that is the one way this criterion could be gamed by accident.
    """
    out: dict[int, str] = {}
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type == tokenize.COMMENT:
                out.setdefault(token.start[0], token.string)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return out
    return out


def _attached_comment_lines(comments: dict[int, str], node: ast.ExceptHandler) -> list[int]:
    """Lines whose comments are attached to this handler: the ``except`` line itself, the
    contiguous comment run directly above it, and the contiguous comment run directly inside it.

    Attachment rather than a fixed window, so a comment sitting on unrelated code a few lines
    up cannot make an undocumented handler look judged.
    """
    found: list[int] = []
    if node.lineno in comments:
        found.append(node.lineno)
    line = node.lineno - 1
    while line >= 1 and line in comments:
        found.append(line)
        line -= 1
    last = node.body[-1].lineno
    line = node.lineno + 1
    while line <= last and line in comments:
        found.append(line)
        line += 1
    return found


def _blind_handlers_in_text(text: str, path: Path) -> list[tuple[Path, int, bool]]:
    """One file's blind handlers, judged. Split out so the detector itself can be tested."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    lines = text.splitlines()
    comments = _comment_tokens(text)
    out: list[tuple[Path, int, bool]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if not isinstance(node.type, ast.Name) or node.type.id not in ("Exception", "BaseException"):
            continue
        if len(node.body) == 1 and isinstance(node.body[0], ast.Raise) and node.body[0].exc is None:
            continue                       # re-raise only: BLE001 excuses this too
        above = lines[max(0, node.lineno - 3):node.lineno - 1]
        inside = lines[node.lineno - 1:min(len(lines), node.lineno + 2)]
        judged = any("# policy:" in w for w in above) or any("# policy:" in w for w in inside)
        if not judged and isinstance(node.body[-1], ast.Raise):
            judged = True
        if not judged:
            judged = any(_is_rationale(comments[ln]) for ln in _attached_comment_lines(comments, node))
        out.append((path, node.lineno, judged))
    return out


def _blind_except_handlers_in_src() -> list[tuple[Path, int, bool]]:
    """Every blind catch under ``src/``, as ``(path, except_line, judged)``.

    One walk feeds both ratchets. Two independent walks would drift apart, and that drift is
    the failure mode this file already documents for its formatting guards: a checker that
    counts something slightly different from the thing being guarded reports a clean tree.

    A handler counts as judged when any of these holds at the handler itself:

    * a ``# policy:`` token above, on, or just inside the ``except`` (the idiom this branch
      introduced, and still the one the failure message recommends);
    * its last statement is a ``raise`` -- cleanup-then-propagate is a decision, the original
      exception still escapes;
    * an attached rationale comment: trailing on the handler line, in the contiguous comment
      lines directly above it, or as the leading comments of the handler body. #56 writes its
      judgements in that last shape (``保留宽泛捕获: ... 失败类型不可枚举``), and crediting only
      ``# policy:`` counted 55 documented handlers as undecided.

    ``_is_rationale`` is what keeps this honest: decoration, tool directives (``pragma`` /
    ``noqa``), and two-character pointers (``同上``) do not judge anything.
    """
    out: list[tuple[Path, int, bool]] = []
    for path in sorted((REPO / "src").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (SyntaxError, UnicodeDecodeError):
            continue
        out.extend(_blind_handlers_in_text(text, path))
    return out


#: Live `path:NNN` pins that this refactor workflow re-pins whenever code moves (six were
#: re-pinned on 2026-10-28 after the #44 hoists in `slm_zernike_pib.py`).
#:
#: A repo-wide pin checker was measured first and REJECTED: the markdown tree holds 710
#: `path:NNN` citations, only a minority sit on a row carrying a code token able to validate
#: the target line, and several rows deliberately keep `原 :NNN` history for files that no
#: longer exist. Gating those emits noise, not signal. This registry instead names the anchors
#: known to move, each with the statement its pin must land on, and compares sets.
_DOC_LINE_ANCHORS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    # (doc, row locator, source file, statements whose current lines the row must pin)
    (
        "docs/slm/pre_run_characterization.md",
        "| 3 | `optimizer/wfless/slm_zernike_pib.py",
        "src/ao_shaping/optimizer/wfless/slm_zernike_pib.py",
        ("SLM_RESPONSE_TIME_S = 0.0",),
    ),
    (
        "docs/slm/pre_run_characterization.md",
        "| 9 | `optimizer/wfless/slm_zernike_pib.py",
        "src/ao_shaping/optimizer/wfless/slm_zernike_pib.py",
        ("def recenter", "set_reference_center(self.recenter",
         "shaping.set_reference_center(pos_center)"),
    ),
    (
        "TODO.md",
        "`--max_peak` 不要动",
        "src/ao_shaping/optimizer/wfless/slm_zernike_pib.py",
        ("_peak_observed = float(max(",),
    ),
    (
        "TODO.md",
        "在 `_apply_best_on_exit()` 退出清理里",
        "src/ao_shaping/optimizer/wfless/slm_zernike_pib.py",
        ("except BaseException as _exit_exc",),
    ),
    (
        "docs/issues_report.md",
        "| `IDEAL_SPOT_RADIUS` |",
        "src/ao_shaping/optimizer/wfless/slm_zernike_pib.py",
        # indented form: the bare name also appears in five call sites, which cannot be pinned
        ("    IDEAL_SPOT_RADIUS,",),
    ),
    (
        "TODO.md",
        "上报动作本身的失败**（`ccd_analyzer.py:",
        "src/ao_shaping/gui/ccd/ccd_analyzer.py",
        ('raw_frame_queue.put(("error", str(exc)), block=False)',
         'raw_frame_queue.put(("error", f"Capture loop crashed: {exc}"), block=False)'),
    ),
    (
        "TODO.md",
        "探测降级**（`bench_settings.py:",
        "src/ao_shaping/gui/slm/bench_settings.py",
        # Re-pinned 2026-10-29: `15ea0a22` moved pixel size into the driver as a model-level
        # single source, so `detect_camera_pixel_um` no longer probes Daheng at all
        # (`probe_daheng_models` + `lookup_model_pixel_size_um`) and both `# policy:` handlers
        # vanished with that code. One degradation guard remains -- MiiCam driver unavailable,
        # line 1282; the pin cites that line and nothing else.
        ("except (ImportError, OSError) as exc:",),
    ),
    (
        "TODO.md",
        "`r50_single.py:",
        "src/ao_shaping/gui/r50/r50_single.py",
        # re-pinned 2026-10-28: the #174 band-17 policy comment landed on the line above
        # this handler, pushing the `except Exception:` from 110 -> 111 (the comment the
        # pin points at is the bare-catch site, not the policy note above it)
        ("except Exception:",),
    ),
)


def _pinned_numbers(row: str, basename: str) -> set[int]:
    r"""Every line number a doc row pins to `basename`: `f:12`, `f:12,30`, `f:12-15`, `f:12/30`.

    A range contributes its start only -- the tables pin a statement or a block head, and a
    block's last line is not a stable anchor. Parsing the whole numeric run in one go is what
    keeps `f:1793-1803` from degrading into `179`, which is how `\d+(?!-)` backtracks.
    Slash-separated runs are the ledger's form for "these N handlers" (`f:297/321`); while `/` was
    not a separator, a registered pin of that shape validated only its first number and every
    later one could rot unnoticed.
    """
    out: set[int] = set()
    for m in re.finditer(re.escape(basename) + r":(\d+(?:\s*[-,/]\s*\d+)*)", row):
        for part in re.split(r"[,/]", m.group(1)):
            out.add(int(part.split("-")[0].strip()))
    return out


def test_live_doc_line_anchors_still_point_at_their_code() -> None:
    """Pinned line numbers in the guarded tables must match where the code now lives.

    Every doc row matching a registry locator is checked: the numbers it pins for the source
    file must equal the current lines of the registered statements. A row that was half
    re-pinned, over-pinned, or left behind by a code move all fail, and a human re-pin makes it
    pass again -- which is the workflow these tables actually follow. Multiple rows may share a
    locator (TODO.md cites the same anchor twice); every one of them has to agree, so the
    second copy cannot silently rot.
    """
    offenders: list[str] = []
    for doc_rel, locator, src_rel, statements in _DOC_LINE_ANCHORS:
        doc_p, src_p = REPO / doc_rel, REPO / src_rel
        if not doc_p.exists() or not src_p.exists():
            offenders.append(f"{doc_rel}[{locator}]: file missing ({src_rel})")
            continue
        src = src_p.read_text(encoding="utf-8").splitlines()
        want: set[int] = set()
        for stmt in statements:
            hits = [i + 1 for i, line in enumerate(src) if stmt in line]
            if len(hits) != 1:
                offenders.append(
                    f"{doc_rel}[{locator}]: statement {stmt!r} occurs {len(hits)}x in "
                    f"{src_rel} ({hits[:4]}) -- a pin cannot be attributed to an ambiguous "
                    "statement; register a rarer one"
                )
                continue
            want.add(hits[0])
        rows = [l for l in doc_p.read_text(encoding="utf-8").splitlines() if locator in l]
        if not rows:
            offenders.append(
                f"{doc_rel}: locator {locator!r} matched no row -- the table row was renamed "
                "or deleted and the registry is now stale"
            )
            continue
        for row in rows:
            have = _pinned_numbers(row, src_p.name)
            if not have:
                offenders.append(
                    f"{doc_rel}: row {locator!r} pins no `{src_p.name}:NNN` -- the citation "
                    "was dropped while the registry still expects it"
                )
            elif have != want:
                offenders.append(
                    f"{doc_rel}: row {locator!r} pins {sorted(have)} but the code sits at "
                    f"{sorted(want)} (stale: {sorted(have - want)}, unpinned: "
                    f"{sorted(want - have)}) -- re-pin the table"
                )
    assert not offenders, "\n  " + "\n  ".join(offenders)


#: The date-anchored ledger form (``**YYYY-MM-DD · sha · text**``) replaced numbered items on
#: 2026-10-27 and nothing in the suite parsed it. Four separate defects landed in it during
#: 2026-10-28: unbalanced bold, a nested full-width parenthesis, a shell/f-string escape residue
#: written literally before a code span, and a narration line reproducing a live doc-pin locator.
#: Legacy numbered items stay frozen -- rewriting them would falsify history -- so only this form is
#: checked.
_LEDGER_SECTION_START = "### 8.9"
_LEDGER_SECTION_END = "### 8.10"
_LEDGER_DATE_HEAD = re.compile(r"^\*\*(\d{4}-\d{2}-\d{2}) \u00b7 ([0-9a-f]{8}) \u00b7 ")
_LEDGER_HEAD_ANY = re.compile(r"^(?:\d{1,3}\. |\*\*\d{4}-\d{2}-\d{2} \u00b7 )")
_MIN_LEDGER_DATE_ENTRIES = 15


def _strip_code_spans(line: str) -> str:
    r"""Drop backticked spans: markdown does not read ``**`` inside ``...`` as bold."""
    return re.sub(r"`[^`]*`", "", line)


def _has_escape_residue(line: str) -> bool:
    r"""True when a backslash outside a code span sits directly before an opening backtick.

    A plain substring test for ``\`` + backtick is wrong: the legitimate span `` `\` `` (a code span
    whose content is one backslash) contains that exact byte pair at its closing delimiter, and a
    first version of this guard failed on such a line. Only a backslash in the prose segments -- what
    a quoted heredoc or an f-string writes when someone escapes a backtick -- is residue.
    """
    inside = False
    for i, ch in enumerate(line):
        if ch == "`":
            inside = not inside
        elif ch == "\\" and not inside and line[i + 1: i + 2] == "`":
            return True
    return False


def test_date_anchored_ledger_entries_stay_parseable_and_balanced() -> None:
    """Every date-anchored ledger entry must parse, balance, and never shadow a live doc pin.

    The heading shape is what ties a ledger entry to a commit, so drift (a third ``**``, a missing
    space around the separator, a truncated SHA) silently breaks the only mechanical link between a
    commit and its justification. Bold parity is measured outside code spans because a ``**`` inside a
    backtick span is not bold: raw counting rejected a correct line and passed a broken one earlier the
    same day.

    The locator clause is the one that pays for itself. A narration line containing a registered
    ``_DOC_LINE_ANCHORS`` locator makes that locator match a second, pin-less row, and the doc-pin
    guard then fails on an unrelated commit -- which is how a docs-only commit got pushed red.
    """
    lines = (REPO / "TODO.md").read_text(encoding="utf-8").splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith(_LEDGER_SECTION_START)), None)
    end = next((i for i, l in enumerate(lines) if l.startswith(_LEDGER_SECTION_END)), None)
    assert start is not None and end is not None and start < end, (
        f"ledger section not found between {_LEDGER_SECTION_START!r} and {_LEDGER_SECTION_END!r} "
        "-- the scope moved, so this guard would silently check nothing")

    body = lines[start:end]
    heads = [i for i, l in enumerate(body) if _LEDGER_HEAD_ANY.match(l)]
    dated = [i for i in heads if _LEDGER_DATE_HEAD.match(body[i])]
    assert len(dated) >= _MIN_LEDGER_DATE_ENTRIES, (
        f"only {len(dated)} date-anchored entries parsed, below the floor "
        f"{_MIN_LEDGER_DATE_ENTRIES} -- the ledger format moved and this guard went blind")

    offenders: list[str] = []
    for n, i in enumerate(dated):
        stop = dated[n + 1] if n + 1 < len(dated) else len(body)
        if _strip_code_spans(body[i]).count("**") != 2:
            offenders.append(
                f"TODO.md:{start + i + 1}: heading carries "
                f"{_strip_code_spans(body[i]).count('**')} bold markers, exactly 2 required "
                f"(inner bolding breaks the parse): {body[i][:64]}")
        for j in range(i, stop):
            line = body[j]
            where = f"TODO.md:{start + j + 1}"
            if line.count("`") % 2:
                offenders.append(f"{where}: unbalanced code span: {line[:64]}")
            elif _strip_code_spans(line).count("**") % 2:
                offenders.append(f"{where}: unbalanced bold outside code spans: {line[:64]}")
            if _has_escape_residue(line):
                offenders.append(
                    f"{where}: escape residue written literally before a code span "
                    f"(heredoc or f-string quoting): {line[:64]}")
            for _doc, locator, _src, _stmts in _DOC_LINE_ANCHORS:
                if locator in line:
                    offenders.append(
                        f"{where}: ledger narration reproduces the live locator {locator!r}, so the "
                        f"doc-pin guard matches this pin-less row and fails: {line[:64]}")

    assert not offenders, "\n  " + "\n  ".join(offenders[:12])


def test_blind_except_handlers_in_src_do_not_grow() -> None:
    """No new blind ``except Exception`` may enter ``src/`` (issue #174).

    Recorded 2026-10-27: the count had grown to 382 (ruff reports 363) since the issue was
    filed at 379 across the repo, and nothing in the suite could see that growth because
    BLE001 is not in ruff's ``select``. The floor on the found count is what keeps the walk
    itself honest: an extractor that stopped parsing would otherwise report zero violations
    and pass.
    """
    population = _blind_except_handlers_in_src()
    found = len(population)
    offenders = [f"{path.relative_to(REPO)}:{lineno}" for path, lineno, _m in population[:6]]
    assert found >= 100, (
        f"only {found} blind handler(s) parsed under src/ -- the walk stopped matching, so "
        "this ratchet would be enforcing nothing"
    )
    assert found <= _MAX_BLIND_EXCEPT_HANDLERS_IN_SRC, (
        f"{found} blind except handlers under src/, above the ratchet "
        f"{_MAX_BLIND_EXCEPT_HANDLERS_IN_SRC} (first offenders: {', '.join(offenders)}). "
        "Catch the types the guarded call can actually raise, or let the error propagate; "
        "if a broad catch is genuinely intended, the ceiling may be raised only with a "
        "reason recorded in TODO.md section 8.9."
    )


def test_unmarked_blind_except_handlers_do_not_grow() -> None:
    """A deliberate broad catch must record its judgement, or it stays counted as undecided.

    The ceiling moves only when a judgement exists at the handler: a ``# policy:`` line, an
    attached rationale comment, or a narrowed catch that leaves the population altogether. It is
    now zero -- every blind handler under ``src/`` carries one -- so a new undocumented blind
    catch fails immediately.

    Detector honesty is proved by negative controls in
    :func:`test_blind_handler_detector_still_flags_an_undocumented_handler`, not by the tree's
    own imperfection: the previous version asserted some handler must still read as unjudged,
    which demonstrates a working scanner only while the codebase stays defective.
    """
    population = _blind_except_handlers_in_src()
    unmarked = [(path, lineno) for path, lineno, judged in population if not judged]
    judged_count = sum(1 for _, _, judged in population if judged)
    assert len(population) >= 100 and judged_count >= 20, (
        f"the walk found {len(population)} handler(s) with {judged_count} judged -- an extractor "
        "that stopped parsing would report a clean tree, so the floor is what keeps this honest"
    )
    offenders = [f"{path.relative_to(REPO)}:{lineno}" for path, lineno in unmarked[:6]]
    assert len(unmarked) <= _MAX_UNMARKED_BLIND_EXCEPT_HANDLERS_IN_SRC, (
        f"{len(unmarked)} blind except handler(s) under src/ record no judgement at the handler, "
        f"above the ratchet {_MAX_UNMARKED_BLIND_EXCEPT_HANDLERS_IN_SRC} (first offenders: "
        f"{', '.join(offenders)}). Read the guarded call: either narrow it to the types that "
        "can actually be raised, or write the fallback policy in one line at the handler "
        "(``# policy: ...`` or plain prose). Narrowing a hardware-path handler is not a cleanup "
        "-- see issues #145 and #45."
    )


def test_blind_handler_detector_still_flags_an_undocumented_handler(tmp_path: Path) -> None:
    """Negative controls: the judgement scan must still answer NO when nothing judges it.

    Widening a marker rule is precisely where a guard starts passing what it should fail, so the
    detector is tested against known snippets instead of against the tree.
    """
    undocumented = _blind_handlers_in_text(
        "try:\n    read()\nexcept Exception:\n    pass\n", tmp_path / "s.py"
    )
    assert undocumented and not any(judged for _, _, judged in undocumented), (
        "an undocumented `except Exception` read as judged"
    )
    assert _blind_handlers_in_text(
        "try:\n    read()\nexcept Exception:\n    cleanup()\n    raise\n", tmp_path / "s.py"
    )[0][2], "cleanup-then-raise decides the failure: the exception still escapes"
    assert not _is_rationale("# 同上")
    assert not _is_rationale("# pragma: no cover")
    assert not _is_rationale("# ------------")
    assert _is_rationale("# pragma: no cover - 回落到 numpy 实现")
    assert _is_rationale("# 保留宽泛捕获: 失败类型不可枚举")
    assert _is_rationale("# a failed candidate must not abort the scan")


def test_markdown_table_rows_never_write_more_cells_than_declared() -> None:
    """Every markdown table row keeps the column count its separator declares.

    A bare ``|`` splits a cell and inline code does not protect it: ``|E|``,
    ``[spgd|ga]``, ``--protocol(file|objective)`` and a bare ``|Δz|`` each silently
    added a column. Two further shapes trip the same count -- two rows fused onto one
    physical line by a missing newline, and a note parked in a fifth cell of a
    four-column table -- so the count is a detector, not a diagnosis: each hit needs
    its cause read before choosing escape / split-line / merge-cell.

    The separator row, not the header, is the arity reference. A header can be the
    defective line itself (the hw_defocus_zernike reports put ``max|c| (rad)`` in the
    header: 14 cells against a 12-cell body), so a header-relative check blamed the
    wrong rows on those three files. Rows writing fewer cells than declared are legal
    -- GFM pads short rows and the trailing ``|`` is optional -- and are not flagged.
    Because that padding is legal, an arity rule alone cannot see a row that LOST its
    closing pipe, so the outer-delimiter escape is checked separately with zero
    tolerance -- a mechanical escaping pass once did exactly that, and only the header
    (which must match the delimiter exactly) betrayed it.

    Generated reports get their own bucket rather than an exclusion, so a split cannot
    arrive silently there either. That bucket is now empty: the emitting side was fixed
    (``markdown_table`` escapes cells; the generators that hand-write row strings
    escape their literals; the miicam probe escapes at render time -- TODO 8.9 item 27),
    so the baseline is zero tolerance. Hand-written files, hand-written reports
    included, are also held at zero.

    Measured 2026-10-25: 0 offenders of either kind over 1487 table blocks,
    and 0 rows whose outer delimiter is escaped.
    The block bound is a measured magnitude on purpose: an inverted fence test once
    blanked every real table and scanned 0 blocks, which a ``blocks > 0`` check would
    have reported as a passing guard.
    """
    offenders, blocks = _markdown_table_cell_splits()
    generated, handwritten = _generated_report_paths()
    assert len(generated) >= 20 and len(handwritten) >= 5, (
        f"provenance parse found {len(generated)} generated / {len(handwritten)} "
        "hand-written reports -- the map layout changed, so the two buckets below are "
        "not the buckets they claim to be"
    )
    assert blocks > 1000, (
        f"only {blocks} markdown table blocks scanned -- the file set or the skip "
        "pattern has stopped matching, so this guard is enforcing nothing"
    )
    split_gen = sorted(o for o in offenders if o.split(":")[0] in generated)
    split_hand = sorted(o for o in offenders if o.split(":")[0] not in generated)
    assert not split_hand, (
        f"{len(split_hand)} hand-written markdown table row(s) write more cells than "
        "their separator declares. Escape the pipe as backslash-| (inline code "
        "included), or merge the overflow back into the last cell if a note outgrew "
        "the schema:\n  " + "\n  ".join(split_hand)
    )
    assert len(split_gen) <= _MD_GENERATED_SPLIT_BASELINE, (
        f"{len(split_gen)} generated-report rows split cells, above the baseline of "
        f"{_MD_GENERATED_SPLIT_BASELINE}. A generated .md is re-written by its "
        "generator, so fix the emitter rather than the artifact (TODO 8.9 item 26):\n"
        "  " + "\n  ".join(split_gen)
    )


# ---------------------------------------------------------------------------
# CLI command inventory -- five sources that must name the same command set
# ---------------------------------------------------------------------------

#: Floor for every parsed set. Measured 19 (2026-10-25); the floor is deliberately far
#: below that so a legitimate new command cannot trip it, while a parse that silently
#: returns an empty set still cannot compare equal to a real set.
_MIN_CLI_COMMANDS = 8


def _docstring_commands(docstring: str) -> set[str]:
    """Command names from ``main.py``'s own ``Commands:`` block.

    The header may wrap, so scanning does not stop at the first unindented line until
    at least one entry has been seen -- otherwise a wrapped header yields an empty set,
    and an empty set equals another empty set.
    """
    lines = docstring.splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if l.strip().startswith("Commands"))
    except StopIteration:
        return set()
    found: set[str] = set()
    seen_entry = False
    for line in lines[start + 1:]:
        m = re.match(r"^ {4}([a-z][a-z-]+)\s{2,}\S", line)
        if m:
            found.add(m.group(1))
            seen_entry = True
        elif not line.strip():
            continue
        elif seen_entry and not line.startswith(" "):
            break                                  # next top-level section (Examples:)
    return found


def _cli_command_inventory() -> dict[str, set[str]]:
    """The five places that enumerate the top-level CLI commands."""
    main_path = SRC / "ao_shaping" / "main.py"
    main_src = main_path.read_text(encoding="utf-8")

    registered = set(re.findall(
        r'cli\.add_command\(\s*\w+\s*,\s*name="([a-z][a-z-]+)"\s*\)', main_src))

    frozen: set[str] = set()
    contract = REPO / "tests/ao_shaping/runners/test_cli_contract_freeze.py"
    if contract.exists():
        for node in ast.parse(contract.read_text(encoding="utf-8")).body:
            if (isinstance(node, ast.Assign)
                    and getattr(node.targets[0], "id", "") == "EXPECTED_COMMANDS"
                    and isinstance(node.value, (ast.Tuple, ast.List))):
                frozen = {e.value for e in node.value.elts if isinstance(e, ast.Constant)}

    golden: set[str] = set()
    golden_path = REPO / "tests/ao_shaping/runners/cli_help_golden.json"
    if golden_path.exists():
        import json
        # Schema of the merged snapshot (built by ``runners/cli_help_snapshot.py``): every key
        # is ``<module>:<name>`` with a companion ``<key>:exit``, and the probe tools add
        # ``tools.*`` entries. Only the ``main:`` group's names are top-level CLI commands.
        golden = {
            parts[1]
            for parts in (k.split(":") for k in json.loads(golden_path.read_text(encoding="utf-8")))
            if len(parts) == 2 and parts[0] == "main"
        }

    docstring = ast.get_docstring(ast.parse(main_src)) or ""
    in_docstring = _docstring_commands(docstring)

    tree: set[str] = set()
    agents = (REPO / "AGENTS.md").read_text(encoding="utf-8")
    at = agents.find("main (click.group)")
    if at >= 0:
        block = agents[at:agents.index("```", at)]
        # Only the first token of each branch row: subcommands are named later in the
        # same line (inside parentheses), and are NOT top-level commands.
        tree = set(re.findall(r"[├└]── ([a-z][a-z-]+)", block))

    return {"main.py add_command": registered, "EXPECTED_COMMANDS": frozen,
            "cli_help_golden.json": golden, "main.py docstring": in_docstring,
            "AGENTS.md tree": tree}


def test_cli_command_inventory_agrees_across_its_five_sources() -> None:
    """Every enumeration of the CLI command set must name the same commands.

    Measured 2026-10-25 (TODO 8.9 item 29): ``AGENTS.md`` claimed 20 commands while
    ``main.py`` registers 19, and it listed ``spgd-square`` as a top-level command when
    that is the ``square`` subcommand of the ``slm-pib`` group. ``main.py``'s own
    docstring advertised three commands that do not exist at all (``gs``, ``gs-square``,
    ``diff-beam``) and omitted nine -- so the text a reader copies from ``--help``
    fails on the first attempt. None of that is visible to any test, because nothing
    used to compare the enumerations against the registration.

    The stated count inside ``AGENTS.md`` is checked too, but only when present: a stale
    number must not survive, while dropping the number entirely stays legal.
    """
    sources = _cli_command_inventory()
    empty = [name for name, names in sources.items() if len(names) < _MIN_CLI_COMMANDS]
    assert not empty, (
        f"{len(empty)} CLI inventory source(s) parsed to almost nothing: {empty}. "
        "Either the command set really collapsed or a parser stopped matching the "
        "format -- both mean this guard is no longer comparing what it claims to."
    )
    reference = sources["main.py add_command"]
    disagreeing = {n: sorted(s ^ reference) for n, s in sources.items() if s != reference}
    assert not disagreeing, (
        "the CLI command inventory disagrees across sources; main.py registration is "
        f"the truth ({len(reference)} commands):\n"
        + "\n".join(f"  {name}: {sorted(diff)}" for name, diff in disagreeing.items())
    )

    agents = (REPO / "AGENTS.md").read_text(encoding="utf-8")
    stated = re.search(r"main\.py 注册 \*\*(\d+)\*\* 个命令", agents)
    assert not stated or int(stated.group(1)) == len(reference), (
        f"AGENTS.md states {stated.group(1)} registered commands but main.py registers "
        f"{len(reference)} -- update the number or drop it"
    )


def test_agents_md_never_pins_main_py_by_line_number() -> None:
    """``AGENTS.md`` cites ``main.py`` by symbol, never by line number.

    Not style: writing this very unit's first pass cited ``main.py:113`` / ``:120``, then
    growing ``main.py``'s docstring shifted the registration block by seven lines and the
    citations were stale on contact. ``main.py`` has no code above the registration that
    a reader needs a line number for, so an anchor like
    ``cli.add_command(combined_run, name="combined")`` is equally precise and cannot rot.

    Measured 2026-10-25: 0 such citations. Count-based baseline, so the rule can be
    relaxed deliberately by raising it -- never by leaving one behind.
    """
    agents = (REPO / "AGENTS.md").read_text(encoding="utf-8")
    pinned = re.findall(r"main\.py:\d+(?:-\d+)?", agents)
    assert not pinned, (
        f"{len(pinned)} absolute main.py line citation(s) in AGENTS.md: {sorted(set(pinned))}. "
        'Cite the symbol instead (e.g. `cli.add_command(..., name="...")`); line numbers '
        "here move on every docstring edit, which is how they were already wrong once."
    )


#: ``CHANGELOG.md`` is the one live document whose backticked hex tokens are commit
#: citations by convention -- batches like ``(``26d01013``, `fdc0696c``, ...)`` sitting
#: next to the issue numbers they close. Measured 2026-10-25: 71 such citations, one of
#: which was a one-character typo (``fdc0696c`` for ``fdc0696f``, "lift the T1/T2
#: composite into a shared _common.shaping_score (#154)") -- and #154 was the only item in
#: that batch no listed hash covered, which is how the typo was identifiable at all.
_CHANGELOG = REPO / "CHANGELOG.md"

#: A markdown link label names a *file*, not a commit. The CHANGELOG's only non-commit hex
#: tokens are link labels (``[`20261014`](./report/.../model_in_loop_live_20261014.md)``),
#: so they are excluded by CONTEXT and never by digit shape: 37 commits in this history
#: have all-digit short hashes (``91209176`` is one, and it resolves), so "all digits means
#: a date" would silently excuse a genuinely broken citation.
_MD_LINK_LABEL = re.compile(r"\[[^\]]*\]\([^)]*\)")

#: A floor, not an exact count -- citations grow with every entry. The point is to fail when
#: the grammar stops matching the file: a guard that parses 0 tokens passes vacuously, which
#: is how the markdown-table arity guard nearly shipped (it scanned 0 blocks, caught only by
#: an unrelated magnitude bound). Measured 71; floor 50 leaves room to reformat, none to vanish.
_MIN_CHANGELOG_CITATIONS = 50


def test_backticked_hex_citations_in_the_changelog_resolve_to_commits() -> None:
    """Every backticked hex token in ``CHANGELOG.md`` names a commit that exists.

    A citation to an unreachable commit voids its own evidence: the batch line reads as
    verified while one of its six hashes matches nothing. Same rot class as a stale path
    citation (TODO 8.9 items 23/24/29/30), same remedy -- resolve mechanically instead of
    trusting the prose.

    Scoped to ``CHANGELOG.md`` deliberately. The identical grammar over every ``*.md`` in
    the repo returned 185 candidates with 16 false positives in four shapes: dates, device
    serial numbers (``22030102``), upstream OOPAO revisions (OOPAO is a separate editable
    install, ``oopao = { path = "libs/OOPAO" }``, so its revs are legitimately absent from
    this object database), and recorded-checksum prefixes in test write-ups. A repo-wide
    rule would have to hard-code all four exemptions, and would then be measuring its own
    regex rather than the repository.
    """
    import subprocess

    text = _MD_LINK_LABEL.sub("", _CHANGELOG.read_text(encoding="utf-8"))
    citations = {token for token in re.findall(r"`([0-9a-f]{7,40})`", text)}
    assert len(citations) >= _MIN_CHANGELOG_CITATIONS, (
        f"parsed only {len(citations)} citation(s) from CHANGELOG.md, below the floor "
        f"{_MIN_CHANGELOG_CITATIONS}: the token grammar no longer matches the file's format, "
        "so a pass here would mean nothing was actually checked."
    )

    broken: list[str] = []
    for cite in sorted(citations):
        probe = subprocess.run(
            ["git", "cat-file", "-t", cite],
            capture_output=True,
            text=True,
            cwd=str(REPO),
            check=False,
        )
        kind = probe.stdout.strip()
        if kind == "commit":
            continue
        # `cat-file -t` is the one call that separates the three document defects, reporting an
        # ambiguous abbreviation on stderr rather than through an option. An earlier draft used
        # `git rev-parse --disambiguate <token>` as two argv entries, which git rejects (it needs
        # `--disambiguate=<token>`): rev-parse echoed the unknown option AND the token itself to
        # stdout, so a filtered scan matched the typo against its own text and labelled a
        # nonexistent hash "names a non-commit object" -- a wrong diagnosis shipped inside a
        # guard whose whole purpose is diagnosis.
        if kind:
            why = f"names a {kind}, not a commit"
        elif "ambiguous" in probe.stderr:
            why = "prefix not unique"
        else:
            why = "no such object"
        broken.append(f"{cite} ({why})")
    assert not broken, (
        f"{len(broken)} CHANGELOG.md citation(s) do not resolve to a commit: {broken}. "
        "For a typo, `git rev-parse --disambiguate=<prefix>` usually shows the intended "
        "commit next to the batch's issue number (the case this guard was written for: "
        "`fdc0696c` -> `fdc0696f`, batch item #154). If history was rewritten under the "
        "citation, re-verify against `git log` and correct the entry -- never delete the "
        "claim just to go green."
    )

#: Generated reports whose FINAL section is emitted by nobody: the heading stem appears in
#: no tracked emitter source while the owning script rewrites the whole file. Regenerating
#: such a report deletes that section -- these are hand-pasted findings (pitfalls, follow-up
#: ordering), i.e. exactly the content worth keeping. Measured 2026-10-25 over 85 generated
#: reports. Pinned rather than repaired because these generators cannot be run here (环境=硬件,
#: or data/debug which scripts/build/stage_test_data.sh excludes): moving prose into a
#: generator nobody can execute swaps one unverifiable claim for another. Each entry is
#: removed by moving the section into its owner and re-running that owner, not by editing
#: this dict -- a stale entry fails the guard.
_UNEMITTED_FINAL_SECTIONS: dict[str, str] = {
    'report/diff_beam/diff_beam_frame_analysis/README.md': '## 9. 关联',
    'report/shape_trials/gui_closed_loop_shaping/gui_closed_loop_shaping_20261014.md': '## 10. 相关文档',
    'report/shape_trials/cold_start/cold_start_20261026.md': '## 数据、复现与设备释放',
    'report/slm/differentiable_opt/differentiable_opt_animation_20261022.md': '## 6. 落盘数据与复现',
    'report/slm/diffraction_efficiency/hw_blaze_20261023/report.md': '## 衍射效率算法 (如何找出 0 级与 ±1 级)',
    'report/slm/model_in_loop_hw_runbook/model_in_loop_bench_calibration.md': '## 与 slm-model-in-loop 的关系',
    'report/slm/zernike_wfs_report/report.md': '## 4. 结论与注意',
    'report/slm_pib_bench/pib_bench_report/EXPERIMENT_REPORT.md': '## 7. 复现方式',
    'report/slm_pib_bench/pib_bench_report/report.md': '### 7.3 ROI 必须一次定位后冻结',
    'report/slm_pib_heuristic_hw/slm_pib_heuristic_hw_report/report.md': '### 5. 原始产物',
    'report/slm_zernike_shaping/slm_zernike_shaping_report/report.md': '## 4. SPGD',
    'report/zernike_r2_baseline/pod_ridge/report.md': '### 若仍要继续,建议的顺序',
}


def _report_owners() -> dict[str, str]:
    """``{artifact: generating script}`` for every report provenance marks generated.

    ``REPORTS`` is a dict literal whose values are ``ReportProvenance(...)`` calls, so the map
    is read off the call keywords: ``literal_eval`` over the whole literal raises on a Call node,
    and the assignment is annotated (``ast.AnnAssign``) rather than a plain ``ast.Assign``.
    ``script=HANDWRITTEN`` is a bare name, not a string, so both shapes count as 'generated'.
    """
    text = (REPO / "scripts" / "_common" / "provenance.py").read_text(encoding="utf-8")
    literal = None
    for node in ast.walk(ast.parse(text)):
        target = node.target if isinstance(node, ast.AnnAssign) else (
            node.targets[0] if isinstance(node, ast.Assign) else None
        )
        if target is not None and getattr(target, "id", "") == "REPORTS":
            literal = node.value
    assert isinstance(literal, ast.Dict), (
        "REPORTS is no longer a dict literal -- this parser and provenance.py have diverged."
    )
    owners: dict[str, str] = {}
    for key, value in zip(literal.keys, literal.values):
        if not isinstance(key, ast.Constant) or not isinstance(value, ast.Call):
            continue
        script = None
        for kw in value.keywords:
            if kw.arg == "script":
                script = kw.value.value if isinstance(kw.value, ast.Constant) else getattr(kw.value, "id", None)
        if isinstance(script, str) and script != "HANDWRITTEN":
            owners[str(key.value)] = script
    return owners


def test_generated_reports_do_not_end_with_a_section_nobody_emits() -> None:
    """No generated report may finish on prose that its generator cannot reproduce.

    The oracle is content-level -- the heading stem must appear in some tracked emitter -- not
    the declared owner, because provenance's ``script`` is the *reproduce entry point*: the
    MiiCam probe report names a hardware test that shells out to the script which actually
    writes the file (its ``运行后状态`` section is emitted there). Auditing through that field
    reports a clean section as drift. A heading is only a proxy, which is why the check is
    limited to the final section -- a repo-wide heading sweep flagged 346 of 1019 headings,
    nearly all of them composed from data inside a loop and therefore never literals.
    """
    import subprocess

    owners = _report_owners()
    generated, _handwritten = _generated_report_paths()
    assert set(owners) == generated, (
        f"the two provenance readers disagree ({len(set(owners) - generated)} / "
        f"{len(generated - set(owners))}): {sorted(set(owners) ^ generated)[:4]} -- one of them "
        "has stopped matching provenance.py's schema."
    )

    tracked = subprocess.run(
        ["git", "ls-files", "-z"], capture_output=True, cwd=str(REPO), check=False
    ).stdout.split(b"\0")
    emitters: dict[str, str] = {}
    self_rel = Path(__file__).relative_to(REPO).as_posix()
    for raw in tracked:
        rel = raw.decode("utf-8", errors="replace")
        # Only code and templates can emit prose: another report quoting a heading, or the
        # report/README.md index listing a title, is not evidence the section is reproducible.
        # This module is skipped as well -- its baseline dict quotes every baselined heading,
        # so counting it would let each recorded section look emitted by the guard recording it.
        if (
            rel != self_rel
            and rel.startswith(("scripts/", "src/", "tests/"))
            and (REPO / rel).is_file()
        ):
            emitters[rel] = (REPO / rel).read_text(encoding="utf-8", errors="replace")
    assert len(emitters) > 500, (
        f"only {len(emitters)} emitter file(s) scanned -- the corpus is empty or git ls-files "
        "failed, so 'emitted by nobody' would be measuring nothing."
    )

    scanned = 0
    orphans: dict[str, str] = {}
    for path in sorted(owners):
        artifact = REPO / path
        if not artifact.is_file():
            continue
        text = artifact.read_text(encoding="utf-8", errors="replace")
        heads = [ln.strip() for ln in text.splitlines() if re.match(r"^#{2,3} ", ln)]
        if not heads:
            continue
        scanned += 1
        last = heads[-1]
        stem = re.sub(r"^#+\s*", "", last)
        if not any(stem in src for src in emitters.values()):
            orphans[path] = last
    assert scanned >= 60, (
        f"only {scanned} generated report(s) scanned; provenance lists "
        f"{len(owners)} -- the sweep stopped covering the corpus."
    )

    fresh = {p: h for p, h in orphans.items() if p not in _UNEMITTED_FINAL_SECTIONS}
    assert not fresh, (
        f"{len(fresh)} generated report(s) end on a section no emitter can reproduce: {fresh}. "
        "Write the section in the generating script and re-run it; pasting it into the "
        "artifact loses it the next time anyone regenerates that report."
    )
    stale = {p: h for p, h in _UNEMITTED_FINAL_SECTIONS.items() if orphans.get(p) != h}
    assert not stale, (
        f"{len(stale)} baselined section(s) are no longer in that state: {stale}. Either the "
        "section was moved into its generator (good -- drop the entry) or the artifact was "
        "rewritten behind the baseline; check which before touching the dict."
    )


#: A link target that resolves against the *git index*, not the working tree. Targets are
#: skipped when they are URLs, bare anchors, or contain a template marker; a trailing
#: ``:NNN`` is stripped because ``[x](file.py:120)`` is a citation style used in these docs.
_LINK_TARGET = re.compile(r"!?\[((?:[^\[\]]|\n)*?)\]\(([^)\s]+)[^)]*\)")
_LINK_SKIP = re.compile(r"^(?:https?://|mailto:|#|data:|//)")
_LINK_LINE_CITE = re.compile(r":\d+(?:-\d+)?$")
#: Floor on the number of link targets actually resolved. A guard that parses nothing passes
#: vacuously, which is how the table-arity guard nearly shipped (it had scanned 0 blocks).
_MIN_LIVE_DOC_LINKS = 200


def _live_docs_link_targets() -> tuple[list[str], int]:
    """``(offenders, checked)`` for relative links in live docs, resolved against the index.

    Resolution uses the tracked file list rather than ``Path.exists()`` on purpose: ``logs/``
    and ``data/`` are gitignored, so a link into them is real on the author's machine and dead
    on every fresh clone -- which is the failure this found (16 links, several pointing at
    ``../logs/mamba_curriculum_*/report.md`` while the label already named the committed
    report). Historical docs are excluded by the repo's own marker list: dated records cite
    machine-local run artefacts by design.
    """
    import subprocess

    listing = subprocess.run(
        ["git", "ls-files", "-z"],
        capture_output=True,
        cwd=str(REPO),
        check=False,
    ).stdout
    tracked = {t.decode("utf-8", errors="replace") for t in listing.split(b"\0") if t}
    tracked_dirs = {
        "/".join(t.split("/")[:k]) for t in tracked for k in range(1, len(t.split("/")))
    }

    offenders: list[str] = []
    checked = 0
    for rel in sorted(tracked):
        if not rel.endswith(".md"):
            continue
        path = REPO / rel
        if not path.is_file() or _is_historical(path):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in _unfenced_md_lines(text):
            # Blank inline code before matching: a link written as literal text inside a
            # code span is documentation, not a link. Fences are already handled by
            # _unfenced_md_lines -- doing this the other way round lets a span swallow the
            # fence marker and unmask the block it was meant to hide.
            blanked = re.sub(r"`+[^`\n]*`+", lambda m: " " * len(m.group(0)), line)
            for match in _LINK_TARGET.finditer(blanked):
                target = match.group(2).strip()
                if _LINK_SKIP.match(target) or "<" in target:
                    continue
                target = target.split("#")[0].split("?")[0]
                target = _LINK_LINE_CITE.sub("", target)
                if not target:
                    continue
                checked += 1
                base = REPO / target.lstrip("/") if target.startswith("/") else path.parent / target
                try:
                    rp = base.resolve().relative_to(REPO).as_posix()
                except ValueError:
                    offenders.append(f"{rel}:{lineno} escapes the repo: {target!r}")
                    continue
                if rp in tracked or rp in tracked_dirs:
                    continue
                if any(t == rp or t.startswith(rp.rstrip("/") + "/") for t in tracked):
                    continue
                here = (REPO / rp).exists()
                offenders.append(
                    f"{rel}:{lineno} not in the git index: {target!r}"
                    + (" (exists locally but is untracked -- dead on a fresh clone)" if here else "")
                )
    return offenders, checked


def test_relative_links_in_live_docs_point_at_tracked_paths() -> None:
    """Every relative link in a live doc resolves to a path the next clone will actually have.

    Measured 2026-10-25: 0 offenders over 289 resolved targets, after 16 fixes. The two shapes
    found were a wrong ``../`` depth in ``docs/GUI/**`` (docs sit 4 levels down; the links
    climbed 5 or 6, leaving the repository entirely) and links into gitignored ``logs/`` which
    resolve on one machine and nowhere else.
    """
    offenders, checked = _live_docs_link_targets()
    assert checked >= _MIN_LIVE_DOC_LINKS, (
        f"resolved only {checked} link target(s) across live docs, below the floor "
        f"{_MIN_LIVE_DOC_LINKS} -- the grammar stopped matching the docs' format, so a pass "
        "here would mean nothing was checked."
    )
    assert not offenders, (
        f"{len(offenders)} link(s) in live docs point at something a fresh clone will not "
        "have:\n  " + "\n  ".join(offenders[:25])
        + "\nFix the path (relative to the *document*, and check the depth), or link the "
        "committed artefact the label already names -- never leave a link into gitignored "
        "logs/ or data/."
    )


#: An invocation claim -- ``python <repo-relative>.py`` inside a module docstring or a live
#: doc -- is an instruction, so a target missing from the git index is a command that cannot
#: run. The trailing ``(?![\w.])`` is load-bearing: without it ``src/calculators/gerchberg_
#: saxton.pyx`` (a tracked Cython module) matches as ``…/gerchberg_saxton.py`` and the guard
#: reports a real file as a defect. Historical docs are skipped by the repo's own marker list:
#: a dated record quoting a command that ran at the time is a record, not an instruction.
_INVOCATION = re.compile(
    r"(?:^|[\s(>])(?:uv run )?python3?[ \t]+((?:src|scripts|tests|libs)/[\w./-]+\.py)(?![\w.])"
)
#: Floor on resolved claims. Measured 533; guards that parse nothing pass vacuously (TODO 8.9
#: items 32 and 34), so a grammar break has to be loud.
_MIN_INVOCATION_CLAIMS = 400


def test_python_invocation_paths_in_docs_and_docstrings_are_tracked() -> None:
    """Every ``python <path>.py`` instruction names a file the next clone will have.

    Item 30 pinned ``python -m <module>`` claims; file-path invocations were the sibling hole,
    and the hole was self-perpetuating -- ``scripts/ml/train_ml.ps1`` ran
    ``python src/ao_shaping/ml/train.py`` while ``src/ml/train.py``'s own docstring taught the
    same dead path in five usage lines, and ``scripts/README.md`` quoted the launcher. The
    package is ``src/ml/`` (console script ``ao-train = ml.train:cli``), so all six now use
    ``python -m ml.train``, whose CWD-independent form was verified in the container: the fixed
    invocation prints its usage (exit 0) and the old file form fails with ``can't open file``
    (exit 2).
    """
    import subprocess

    listing = subprocess.run(
        ["git", "ls-files", "-z"], capture_output=True, cwd=str(REPO), check=False
    ).stdout
    tracked = {t.decode("utf-8", errors="replace") for t in listing.split(b"\0") if t}

    offenders: list[str] = []
    checked = 0
    for rel in sorted(tracked):
        path = REPO / rel
        if not path.is_file():
            continue
        if rel.endswith(".py"):
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            doc = ast.get_docstring(tree)
            if doc is None:
                continue
            lines = doc.splitlines()
            # Line numbers are relative to the docstring, not the file: the claim's location,
            # not a file offset. Stating that beats printing a number that jumps on any edit.
            where = f"{rel} (docstring line)"
        elif rel.endswith(".md") and not _is_historical(path):
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            where = rel
        else:
            continue
        for lineno, line in enumerate(lines, 1):
            for match in _INVOCATION.finditer(line):
                checked += 1
                if match.group(1) not in tracked:
                    offenders.append(f"{where}:{lineno} -> {match.group(1)!r}")

    assert checked >= _MIN_INVOCATION_CLAIMS, (
        f"resolved only {checked} invocation claim(s), below the floor "
        f"{_MIN_INVOCATION_CLAIMS} -- the grammar stopped matching the repo's usage style, "
        "so a pass here would mean nothing was checked."
    )
    assert not offenders, (
        f"{len(offenders)} invocation claim(s) name a file that is not in the git index:\n  "
        + "\n  ".join(offenders[:20])
        + "\nPoint them at the real module (prefer the CWD-independent `python -m <pkg.mod>` "
        "form -- `PYTHONPATH=src` plus a relative file path breaks as soon as the caller "
        "changes directory)."
    )


#: Reports whose generator carries no runtime data dependency -- the body is a
#: literal template, and it imports only the standard library. For those, the
#: registered reproduce command must produce byte-identical content, because
#: nothing about the machine or the corpus can legitimately change the output.
#: Every other generated report reads ``data/`` or ``logs/`` (gitignored) and
#: cannot be re-run here, so this list is deliberately short and explicit
#: (TODO 8.9 item 38).
_FROZEN_REPRODUCIBLE_REPORTS: dict[str, str] = {
    "report/fouriergsnet_pipeline/fouriergsnet_pipeline_report/report.md":
        "scripts/slm/generate_fouriergsnet_pipeline_report.py",
}

#: ``sync_report_provenance.py`` injects this block into the artifact; the
#: generator itself never emits it, so the comparison strips it from the
#: committed side.
_PROVENANCE_BLOCK = re.compile(
    r"<!-- provenance:start -->.*?<!-- provenance:end -->\n\n", re.S
)


def test_frozen_reports_reproduce_their_committed_artifact(tmp_path: Path) -> None:
    """Re-run an offline generator and require the committed artifact byte-for-byte.

    A ``script=`` registration is a claim that the named generator produces the
    artifact. When the generator's template has drifted from the committed file,
    that claim is false in both directions: the registered reproduce command no
    longer yields the artifact, and the artifact holds conclusions the generator
    cannot emit. The fouriergsnet pipeline report was exactly that -- its
    committed copy carries a 2026-10-01 review that *retracts* the generator's
    central finding (the ``phi.detach()`` bug was fixed at the source, and eight
    line numbers had drifted), so re-running the registered command would have
    overwritten a corrected record with a stale one.
    """
    import difflib

    for art_rel, script_rel in sorted(_FROZEN_REPRODUCIBLE_REPORTS.items()):
        art_path = REPO / art_rel
        assert art_path.is_file(), f"registered artifact {art_rel} is missing"
        artifact = art_path.read_text(encoding="utf-8")
        committed_body = _PROVENANCE_BLOCK.sub("", artifact, count=1)
        assert len(committed_body) > 1000, (
            f"{art_rel}: stripped body is only {len(committed_body)} chars -- the "
            "block regex matched more than the provenance header, so this comparison "
            "would be vacuous."
        )

        # ``-o`` takes a directory; an absolute path escapes ``_REPO_ROOT`` (pathlib
        # replaces on an absolute operand), so the repo is never written to.
        out_dir = tmp_path / art_rel.replace("/", "_")
        subprocess.run(
            [sys.executable, str(REPO / script_rel), "-o", str(out_dir)],
            check=True, cwd=str(REPO), capture_output=True,
        )
        emitted = (out_dir / "report.md").read_text(encoding="utf-8")

        diff = list(difflib.unified_diff(
            emitted.splitlines(), committed_body.splitlines(),
            fromfile=f"{script_rel} (generated)", tofile=art_rel, lineterm="",
        ))
        assert emitted == committed_body, (
            f"{script_rel} does not reproduce {art_rel}: "
            f"{sum(1 for d in diff if d.startswith('+') and not d.startswith('+++'))} "
            f"line(s) only in the artifact, "
            f"{sum(1 for d in diff if d.startswith('-') and not d.startswith('---'))} "
            f"only in the generator.\n" + "\n".join(diff[:40])
            + "\nThe generator owns the registration, so its template must match the "
            "committed record; fold any dated review note into the template rather "
            "than editing the artifact alone."
        )


#: ``.env`` and ``.env.example`` reach Python through the editor's terminal
#: environment: there is no dotenv in the tree and no CI step loads ``.env``, so a
#: declared key that no tracked code reads is a knob the guide advertises and
#: nothing honours. ``CENTER=577,655`` was exactly that -- declared in both files and
#: mirrored twice in ``AGENTS.md``, yet the literal ``"CENTER"`` appears in no tracked
#: file and never has (introduced in ``546bb610``, and ``git log -S'"CENTER"' -- '*.py'``
#: is empty), while the centre actually comes from the runner option ``-c/--center``.
_ENV_DECL = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")
#: Read by the interpreter or the OS rather than by repo code, so a literal search
#: cannot judge them. Only variables whose consumer is a tool belong here.
_ENV_TOOL_CONSUMED = frozenset({"PYTHONPATH", "PATH"})
#: Code extensions only. Report artifacts echo variable names in ``plan_*.json``, so a
#: corpus that accepted ``.json`` would let a phantom key validate itself from an output.
_ENV_CODE_GLOBS = ("*.py", "*.pyx", "*.sh", "*.ps1", "*.yml", "*.yaml", "*.toml", "*.cfg", "*.ini")
#: Floors: with only a handful of declared keys, a parse that found none would pass
#: vacuously (TODO 8.9 items 32, 34, 38).
_MIN_ENV_KEYS_DECLARED = 4
_MIN_ENV_FILES_SCANNED = 1000


def test_declared_dotenv_keys_are_read_by_tracked_code() -> None:
    """Every key declared in ``.env``/``.env.example`` must be read by tracked code.

    These two files are the repo's configuration contract -- ``AGENTS.md`` mirrors them,
    and a reader who sets a key expects behaviour. A key nothing reads is worse than an
    absent one: it looks authoritative, and the guide repeats it. The scan searches the
    quoted literal, so an indirection that builds the name at runtime would not count as
    a consumer; that is deliberate, because such a key is not discoverable either.
    """
    declared: dict[str, str] = {}
    for name in (".env", ".env.example"):
        path = REPO / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.lstrip().startswith("#"):
                continue
            match = _ENV_DECL.match(line)
            if match:
                declared.setdefault(match.group(1), name)
    assert len(declared) >= _MIN_ENV_KEYS_DECLARED, (
        f"parsed {len(declared)} declared key(s), below the floor "
        f"{_MIN_ENV_KEYS_DECLARED} -- the parser stopped matching these files, so a "
        "pass would mean nothing was checked."
    )

    listing = subprocess.run(
        ["git", "ls-files", *("*.py", "*.pyx", "*.sh", "*.ps1", "*.yml", "*.yaml",
                             "*.toml", "*.cfg", "*.ini")],
        capture_output=True, cwd=str(REPO), check=False,
    ).stdout.decode("utf-8", errors="replace").splitlines()
    assert len(listing) >= _MIN_ENV_FILES_SCANNED, (
        f"corpus is only {len(listing)} file(s), below the floor "
        f"{_MIN_ENV_FILES_SCANNED} -- the search would be near-vacuous."
    )

    # Self-exclusion is load-bearing: this file quotes the keys it hunts, so an
    # unfiltered search is satisfied by the hunt itself (the first version passed on a
    # HEAD that still declared ``CENTER``, its only hit being this module).
    self_rel = str(Path(__file__).resolve().relative_to(REPO.resolve()))

    unread: list[str] = []
    for key, where in sorted(declared.items()):
        if key in _ENV_TOOL_CONSUMED:
            continue
        found = subprocess.run(
            ["git", "grep", "-l", "-E", f'"{key}"|\'{key}\'', "--", *_ENV_CODE_GLOBS],
            capture_output=True, cwd=str(REPO), check=False,
        ).stdout.decode("utf-8", errors="replace").strip()
        consumers = [h for h in found.splitlines() if h and h != self_rel]
        if not consumers:
            unread.append(f"{key} (declared in {where})")

    assert not unread, (
        f"{len(unread)} declared env key(s) are read by no tracked code:\n  "
        + "\n  ".join(unread)
        + "\nEither read the key in code, or drop the declaration from both `.env` and "
        "`.env.example` (and any `AGENTS.md` mirror). A commented line does not count "
        "as declared."
    )


#: A reproduce command inside a fenced shell block is an instruction a reader pastes,
#: so its script path must exist. Prose and dated logs stay out on purpose: an issue log
#: naming a path that was correct when written, or a template name like ``<brand>.py``,
#: is not an executable claim -- that is the wall TODO 8.9 items 32 and 36 hit. Unlike
#: ``_INVOCATION`` this rule ignores how the interpreter is spelled, which matters:
#: ``.venv\Scripts\python.exe scripts\…`` hid from that guard no matter how its path
#: grammar was widened, because nothing there matches a bare ``python`` + space.
_SHELL_FENCE_OPEN = re.compile(
    r"^```[ \t]*(?:bash|sh|shell|powershell|ps1|cmd|batch|console)\b", re.IGNORECASE
)
#: The left boundary is load-bearing and must NOT exclude a backslash: the Windows fences
#: spell ``.\scripts\build.ps1``, and a lookbehind that rejects ``\`` silently drops every
#: one of those claims (measured: 292 claims / 2 offenders instead of 297 / 5).
_CMD_SCRIPT = re.compile(
    r"(?<![\w-])((?:scripts|src)[/\\][\w./\\*-]+\.(?:py|sh|ps1|bat))(?![\w.])"
)
#: Floor on checked claims. Measured 297; a fence parser that stopped matching would
#: otherwise pass by verifying nothing (TODO 8.9 items 32, 34, 38).
_MIN_CMD_SCRIPT_CLAIMS = 200


def test_reproduce_commands_in_live_docs_name_tracked_scripts() -> None:
    """Every script path in a live doc's shell fence must be in the git index.

    ``569ea514`` regrouped 139 flat scripts into subpackages. Section headings survived
    because they name a bare basename, but the fenced reproduce commands kept the old
    flat paths, so five of them now describe invocations that cannot run -- including the
    ``train_ml.ps1`` launcher whose dead body item 37 repaired while its documented call
    path stayed stale. When a claim misses, the report shows where that basename *is*
    tracked, because repointing the path is the whole repair.
    """
    import subprocess

    listing = subprocess.run(
        ["git", "ls-files", "-z"], capture_output=True, cwd=str(REPO), check=False
    ).stdout
    tracked = {t.decode("utf-8", errors="replace") for t in listing.split(b"\0") if t}

    offenders: list[str] = []
    checked = 0
    for rel in sorted(tracked):
        if not rel.endswith(".md") or _is_historical(REPO / rel):
            continue
        in_fence, shell_fence = False, False
        for lineno, line in enumerate(
            (REPO / rel).read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            stripped = line.strip()
            if stripped.startswith("```"):
                if not in_fence:
                    in_fence = True
                    shell_fence = bool(_SHELL_FENCE_OPEN.match(stripped))
                else:
                    in_fence = shell_fence = False
                continue
            if not shell_fence:
                continue
            for token in _CMD_SCRIPT.findall(line):
                checked += 1
                if any(ch in token for ch in "*<>"):
                    continue
                norm = token.replace("\\", "/")
                if norm in tracked:
                    continue
                # The path grammar deliberately starts a claim at ``scripts/`` or ``src/``
                # even mid-token, so a correct repo-root path for a script that lives
                # elsewhere -- e.g. a skill's own scripts directory under libs/ -- is
                # extracted as its trailing span. Accept that span only when this very line
                # spells a tracked path ending in it. A stale flat path spells no such
                # thing, so the detection the guard exists for is unchanged.
                line_norm = line.replace("\\", "/")
                if any(t.endswith("/" + norm) and t in line_norm for t in tracked):
                    continue
                base = norm.rsplit("/", 1)[-1]
                alt = sorted(t for t in tracked if t.rsplit("/", 1)[-1] == base)[:3]
                offenders.append(
                    f"{rel}:{lineno} -> {token!r}"
                    + (f" [same basename tracked at: {', '.join(alt)}]" if alt
                       else " [no tracked file has this basename]")
                )

    assert checked >= _MIN_CMD_SCRIPT_CLAIMS, (
        f"checked only {checked} fenced command path(s), below the floor "
        f"{_MIN_CMD_SCRIPT_CLAIMS} -- the fence or path grammar stopped matching, so a "
        "pass here would mean nothing was verified."
    )
    assert not offenders, (
        f"{len(offenders)} fenced reproduce command(s) name a script that is not tracked:\n  "
        + "\n  ".join(offenders[:20])
        + "\nA moved script leaves its old path behind in the usage fence; repoint the "
        "command at the tracked location, keeping the fence's own separator style."
    )


#: A boolean short-circuit containing a *dominant* constant throws away its own check:
#: ``x or True`` is always true and ``x and False`` is always false, however the rest
#: evaluates. Three were live -- a hardware tool gate, an assertion in a test, and a
#: generator metric whose key was never even rendered. Only dominant constants are
#: flagged: ``x or False`` and ``x and True`` are redundant but do not silence anything.
#: Scope is the repo's own trees; the vendored SDK code under ``libs/`` is not ours to
#: edit, which is the same boundary ``ruff.toml`` draws.
_TAUTOLOGY_SCOPE = ("src/", "scripts/", "tests/")
#: Floor on parsed files. Measured 1223; a scan that silently parsed nothing would
#: otherwise pass by verifying nothing (TODO 8.9 items 32, 34, 38, 40).
_MIN_TAUTOLOGY_FILES = 1000


def test_no_boolean_short_circuit_discards_its_own_check() -> None:
    """No tracked source file may neutralize a condition with ``or True`` / ``and False``.

    A guard written that way cannot fail, so it reports confidence the code never
    established. The test file case was worse than dead code: an assertion carrying
    ``or True`` looks like coverage while asserting nothing, and the correct assertion
    already sat on the next line.
    """
    import subprocess

    listing = subprocess.run(
        ["git", "ls-files", "-z"], capture_output=True, cwd=str(REPO), check=False
    ).stdout
    files = [
        t.decode("utf-8", errors="replace")
        for t in listing.split(b"\0")
        if t and t.decode("utf-8", errors="replace").endswith(".py")
        and t.decode("utf-8", errors="replace").startswith(_TAUTOLOGY_SCOPE)
    ]

    offenders: list[str] = []
    unparsed: list[str] = []
    for rel in sorted(files):
        text = (REPO / rel).read_text(encoding="utf-8-sig", errors="replace")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            # A file this repo's ruff gate parses must parse here too; skipping a
            # failure quietly would shrink the corpus the assertion speaks for.
            unparsed.append(rel)
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.BoolOp):
                continue
            dominant = True if isinstance(node.op, ast.Or) else False
            if any(isinstance(v, ast.Constant) and v.value is dominant
                   for v in node.values):
                seg = ast.get_source_segment(text, node) or ""
                offenders.append(f"{rel}:{node.lineno}: {' '.join(seg.split())[:100]}")

    assert not unparsed, (
        f"{len(unparsed)} file(s) failed to parse: {unparsed[:5]} -- the corpus the "
        "count below describes is smaller than the tree."
    )
    assert len(files) >= _MIN_TAUTOLOGY_FILES, (
        f"scanned {len(files)} file(s), below the floor {_MIN_TAUTOLOGY_FILES} -- the "
        "scope filter stopped matching the repo's own trees, so a pass would mean "
        "nothing was checked."
    )
    assert not offenders, (
        f"{len(offenders)} self-neutralizing boolean condition(s):\n  "
        + "\n  ".join(offenders[:15])
        + "\nDelete the dead operand (and the wrapper it disables) rather than leaving "
        "a check that can never fail."
    )


#: A list/dict/set literal, or a bare ``list()``/``dict()``/``set()``, in a parameter
#: default is constructed **once at def time** and then shared by every call that omits
#: that argument, so a future mutation leaks from one run into the next. ruff cannot see
#: this: the rule is B006, which lives in flake8-bugbear and is not installed here, so
#: the class needs its own check. ``default_factory`` is deliberately not flagged -- it
#: exists precisely to avoid this bug. The scanned trees are the same repo-owned
#: scope as the short-circuit guard above (_TAUTOLOGY_SCOPE) -- one definition, not two.
_MUTABLE_DEFAULT_CALLS = frozenset({"list", "dict", "set"})
#: Floor on scanned files; measured 1223. Guards that parse nothing pass vacuously
#: (TODO 8.9 items 32, 34, 38, 40, 41).
_MIN_MUTABLE_DEFAULT_FILES = 1000


def test_no_mutable_default_arguments() -> None:
    """No tracked source file may default a parameter to a mutable literal.

    All four occurrences found were read-only inside their bodies, so none was
    misbehaving yet -- the point is that a shared empty list is a trap primed for the
    first ``append``. The optimizers in the wavefront and PIB paths were the ones
    holding them, i.e. the layer whose whole contract is that two runs in one process
    must not share state.
    """
    import subprocess

    listing = subprocess.run(
        ["git", "ls-files", "-z"], capture_output=True, cwd=str(REPO), check=False
    ).stdout
    files = [
        t.decode("utf-8", errors="replace")
        for t in listing.split(b"\0")
        if t and t.decode("utf-8", errors="replace").endswith(".py")
        and t.decode("utf-8", errors="replace").startswith(_TAUTOLOGY_SCOPE)
    ]

    offenders: list[str] = []
    unparsed: list[str] = []
    for rel in sorted(files):
        text = (REPO / rel).read_text(encoding="utf-8-sig", errors="replace")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            unparsed.append(rel)
            continue
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            defaults = list(fn.args.defaults) + list(fn.args.kw_defaults)
            for d in defaults:
                if d is None:
                    continue
                mutable = isinstance(d, (ast.List, ast.Dict, ast.Set)) or (
                    isinstance(d, ast.Call) and isinstance(d.func, ast.Name)
                    and d.func.id in _MUTABLE_DEFAULT_CALLS
                )
                if mutable:
                    name = getattr(fn, "name", None) or "<lambda>"
                    offenders.append(f"{rel}:{d.lineno}: {name}() defaults to {ast.unparse(d)!r}")

    assert not unparsed, (
        f"{len(unparsed)} file(s) failed to parse: {unparsed[:5]} -- the corpus the count "
        "below describes is smaller than the tree."
    )
    assert len(files) >= _MIN_MUTABLE_DEFAULT_FILES, (
        f"scanned {len(files)} file(s), below the floor {_MIN_MUTABLE_DEFAULT_FILES} -- "
        "the scope filter stopped matching the repo's own trees."
    )
    assert not offenders, (
        f"{len(offenders)} mutable default argument(s):\n  " + "\n  ".join(offenders[:15])
        + "\nDefaults are evaluated once at import time, so the object is shared across "
        "calls. Use an immutable literal (tuple / frozenset), ``None`` with a body-level "
        "fallback, or ``field(default_factory=...)`` in a dataclass."
    )


#: ``pytest.raises(Exception)`` accepts *any* error, so the test keeps passing when the
#: code under test fails for an unrelated reason -- a ``TypeError`` from a refactor or an
#: ``AttributeError`` from a rename would both satisfy it. This is not a house style worth
#: preserving: the repo already names the exception almost everywhere (643 ``ValueError``,
#: 55 ``RuntimeError``, 37 ``TypeError``, 12 ``click.BadParameter``), leaving two bare
#: outliers. A tuple of specific exceptions is fine and is not flagged.
_RAISES_BARE_TYPES = frozenset({"Exception", "BaseException"})
#: Floor on collected ``pytest.raises`` call sites: measured 978 across 507 test files,
#: of which these 2 are bare -- outliers, not a house style. A finder that matched
#: nothing would otherwise pass by checking nothing (TODO 8.9 items 32, 34, 38, 40-42).
_MIN_PYTEST_RAISES_CALLS = 400


def test_pytest_raises_names_a_specific_exception() -> None:
    """Every ``pytest.raises`` in the suite must name a concrete exception type.

    The two bare sites were provable from the code they test: ``_reject`` returns
    ``click.BadParameter`` and every rejection path in ``resolve_exposure`` raises it,
    while ``ObjectiveResult`` is ``@dataclass(frozen=True)``, whose assignment raises
    ``FrozenInstanceError``. Both replacements were therefore derived, not guessed --
    which matters because one of the two tests needs real hardware to run.
    """
    import subprocess

    listing = subprocess.run(
        ["git", "ls-files", "-z"], capture_output=True, cwd=str(REPO), check=False
    ).stdout
    files = [
        t.decode("utf-8", errors="replace")
        for t in listing.split(b"\0")
        if t and t.decode("utf-8", errors="replace").endswith(".py")
        and t.decode("utf-8", errors="replace").startswith("tests/")
    ]

    offenders: list[str] = []
    unparsed: list[str] = []
    calls = 0
    for rel in sorted(files):
        text = (REPO / rel).read_text(encoding="utf-8-sig", errors="replace")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            unparsed.append(rel)
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not (isinstance(fn, ast.Attribute) and fn.attr == "raises"
                    and isinstance(fn.value, ast.Name) and fn.value.id == "pytest"):
                continue
            calls += 1
            for arg in node.args:
                if isinstance(arg, ast.Name) and arg.id in _RAISES_BARE_TYPES:
                    offenders.append(f"{rel}:{node.lineno}: pytest.raises({arg.id})")

    assert not unparsed, (
        f"{len(unparsed)} test file(s) failed to parse: {unparsed[:5]} -- the corpus the "
        "count below describes is smaller than the suite."
    )
    assert calls >= _MIN_PYTEST_RAISES_CALLS, (
        f"found only {calls} pytest.raises call site(s), below the floor "
        f"{_MIN_PYTEST_RAISES_CALLS} -- the finder stopped matching the suite's own idiom, "
        "so a pass here would mean nothing was checked."
    )
    assert not offenders, (
        f"{len(offenders)} pytest.raises site(s) accept any exception:\n  "
        + "\n  ".join(offenders[:15])
        + "\nName the exception the code actually raises (its own docstring, ``Raises:`` "
        "block or return annotation usually states it); a tuple of specific types is also "
        "acceptable."
    )


#: The class-based optimizer convention (AGENTS.md "Class-based Optimizer Convention",
#: ``src/ao_shaping/algorithm/README.md``) states that ``update()`` performs one step
#: **and returns the next state/solution**. The first half is used everywhere; the
#: second half was enforced nowhere, while 36 call sites assign its result -- a concrete
#: ``update()`` that forgets to return hands ``None`` to every one of them. Measured
#: corpus: 89 files, 20 ``update`` methods, 17 value-returning, 3 interface stubs,
#: 0 violations, so this guard is preventive rather than a bug fix.
_UPDATE_SCOPE = (SRC / "ao_shaping" / "algorithm", SRC / "ao_shaping" / "optimizer")
_MIN_UPDATE_SCOPE_FILES = 60
_MIN_UPDATE_METHODS = 15


def _is_interface_stub(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Exempt by shape: ``@abstractmethod`` / ``@overload``, or docstring + ``pass``/``...``.

    Deliberately NOT "the enclosing class derives from ``ABC``": an ABC may hold a fully
    concrete ``update()``, and excusing a method by class ancestry is how a real
    violation hides. All three current stubs are caught by shape alone.
    """
    for dec in fn.decorator_list:
        name = dec.id if isinstance(dec, ast.Name) else getattr(dec, "attr", "")
        if name in ("abstractmethod", "overload"):
            return True

    body = list(fn.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if not body:
        return True
    return all(
        isinstance(stmt, ast.Pass)
        or (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and stmt.value.value is Ellipsis
        )
        for stmt in body
    )


def _returns_a_value(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """A ``Return``/``Yield`` owned by *fn*, not by a nested function.

    ``ast.walk(fn)`` descends into inner closures, where a single ``return`` would
    falsely exempt an outer ``update()`` that itself returns nothing.
    """
    stack: list[ast.AST] = [fn]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Return):
            if node.value is not None:
                return True
        elif isinstance(node, ast.Yield):
            return True
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            stack.append(child)
    return False


def test_concrete_update_methods_return_a_value() -> None:
    """Every concrete ``update()`` in algorithm/ and optimizer/ returns a value.

    Callers in the wavefront, PIB and shaping loops all write
    ``update = optimizer.update(gradient)``, so an implicit ``None`` propagates without
    failing until something downstream indexes it.
    """
    files = sorted({p for root in _UPDATE_SCOPE for p in root.rglob("*.py")})

    offenders: list[str] = []
    unparsed: list[str] = []
    methods = 0
    for path in files:
        rel = path.relative_to(REPO).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig", errors="replace"))
        except SyntaxError:
            unparsed.append(rel)
            continue
        for owner in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
            for fn in owner.body:
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if fn.name != "update":
                    continue
                methods += 1
                if _is_interface_stub(fn):
                    continue
                if not _returns_a_value(fn):
                    offenders.append(
                        f"{rel}:{fn.lineno}: class {owner.name}.update() returns nothing"
                    )

    assert not unparsed, (
        f"{len(unparsed)} file(s) failed to parse: {unparsed[:5]} -- the corpus the counts "
        "below describe is smaller than the scope."
    )
    assert len(files) >= _MIN_UPDATE_SCOPE_FILES, (
        f"scanned {len(files)} file(s) in the update() scope, below the floor "
        f"{_MIN_UPDATE_SCOPE_FILES} -- the scope stopped matching the repo's own trees, so "
        "a pass here would mean nothing was checked."
    )
    assert methods >= _MIN_UPDATE_METHODS, (
        f"found only {methods} ``update`` method(s), below the floor "
        f"{_MIN_UPDATE_METHODS} -- the finder stopped matching the convention's own shape."
    )
    assert not offenders, (
        f"{len(offenders)} concrete ``update()`` method(s) return nothing:\n  "
        + "\n  ".join(offenders[:15])
        + "\n``update()`` must return the next state/solution (AGENTS.md class-based "
        "optimizer convention); callers assign its result, so an implicit ``None`` "
        "propagates silently. Interface stubs are exempt by shape, not by name."
    )


#: ``calc_n_zernike_terms`` legitimately has TWO numbers in this repo: the canonical term
#: count (which INCLUDES piston -- its own docstring says so) and the WFS family's array
#: length, one slot larger because the DLL fills ``coeff[1..N]`` 1-based and index 0 is
#: never written. The extra slot is padding, not a coefficient. Both are single implementations -- ``drivers/wfs/base.py``
#: keeps "只保留唯一一份实现" and ``drivers/wfs/AGENTS.md`` pins the ``+1`` relation
#: "防止两套实现无声漂移". ``ThorlabWFS`` nevertheless carried a third copy that re-derived the
#: arithmetic instead of inheriting ``BaseWFS``; that copy is deleted in the same commit that
#: adds this guard. So the rule is structural, not a name allowlist: any definition of the
#: entry point outside the canonical module must *call* it, which is exactly what a
#: backward-compat shim does. Measured: 716 files in scope, 3 definitions
#: (1 canonical + 2 delegating). ``tests/`` is out of scope by design -- a test-local
#: re-derivation is an independent oracle, not a second production implementation.
_TERM_COUNT_NAME = "calc_n_zernike_terms"
_TERM_COUNT_CANON = "src/ao_shaping/utils/wavefront/zernike_calc.py"
_MIN_TERM_COUNT_SCOPE_FILES = 600
_MIN_TERM_COUNT_DEFS = 3


def test_term_count_entry_point_is_not_reimplemented() -> None:
    """Every ``calc_n_zernike_terms`` outside the canonical module delegates to it.

    A copied body drifts in silence: the WFS array length and the canonical term count
    differ by exactly one padding slot, so a duplicate that keeps or drops it silently
    changes coefficient vector sizes -- the same class of defect that produced the
    documented 1.88x unit bug.
    """
    files = sorted(
        p.relative_to(REPO).as_posix()
        for root in (SRC, REPO / "scripts")
        for p in root.rglob("*.py")
        if "__pycache__" not in p.parts
    )

    unparsed: list[str] = []
    defs: list[tuple[str, int, bool]] = []
    for rel in files:
        try:
            tree = ast.parse((REPO / rel).read_text(encoding="utf-8-sig", errors="replace"))
        except SyntaxError:
            unparsed.append(rel)
            continue
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if fn.name != _TERM_COUNT_NAME:
                continue
            delegates = any(
                isinstance(call, ast.Call)
                and (
                    (isinstance(call.func, ast.Name)
                     and call.func.id.endswith(_TERM_COUNT_NAME))
                    or (isinstance(call.func, ast.Attribute)
                        and call.func.attr.endswith(_TERM_COUNT_NAME))
                )
                for call in ast.walk(fn)
            )
            defs.append((rel, fn.lineno, delegates))

    offenders = [
        f"{rel}:{line}"
        for rel, line, delegates in defs
        if rel != _TERM_COUNT_CANON and not delegates
    ]

    assert not unparsed, (
        f"{len(unparsed)} file(s) failed to parse: {unparsed[:5]} -- the corpus the counts "
        "below describe is smaller than the scope."
    )
    assert len(files) >= _MIN_TERM_COUNT_SCOPE_FILES, (
        f"scanned {len(files)} file(s), below the floor {_MIN_TERM_COUNT_SCOPE_FILES} -- the "
        "scope stopped matching the repo's own trees."
    )
    assert len(defs) >= _MIN_TERM_COUNT_DEFS, (
        f"found only {len(defs)} definition(s) of ``{_TERM_COUNT_NAME}``, below the floor "
        f"{_MIN_TERM_COUNT_DEFS} -- the finder stopped matching the entry point's own shape."
    )
    assert any(rel == _TERM_COUNT_CANON for rel, _, _ in defs), (
        f"no ``{_TERM_COUNT_NAME}`` definition in {_TERM_COUNT_CANON} -- the canonical entry "
        "point moved or was renamed, so this guard is checking nothing it claims to."
    )
    assert not offenders, (
        f"{len(offenders)} re-implementation(s) of ``{_TERM_COUNT_NAME}`` that do not delegate "
        "to the canonical module:\n  " + "\n  ".join(offenders[:15])
        + f"\nInherit from ``BaseWFS`` (whose own copy delegates) or call "
        f"``{_TERM_COUNT_CANON}`` directly; a body that re-derives ``(n+1)(n+2)//2`` can "
        "disagree with the canonical count by the padding slot and nothing will raise."
    )


#: The two WFS producers hand back Zernike coefficient vectors under index conventions that
#: are NOT interchangeable. Measured 2026-10-25, offline: defocus (2,0) lands at index 3 on
#: the twin and at slot 5 on the driver, with lengths 66 and 67 at order 10. The convention
#: is therefore declared on the class (``BaseWFS.zernike_layout``) so a caller can branch on
#: it instead of guessing. Validation is against the enum's own members -- not a hand-written
#: string list, which is the kind of allowlist that rots.
_WFS_BASE_NAME = "BaseWFS"
_WFS_LAYOUT_ATTR = "zernike_layout"
_WFS_LAYOUT_ENUM = "WfsZernikeLayout"
_WFS_BASE_FILE = "src/ao_shaping/drivers/wfs/base.py"
_MIN_WFS_DRIVERS = 2
_MIN_WFS_DRIVER_FILES = 60


def test_every_wfs_driver_declares_its_zernike_index_layout() -> None:
    """Each concrete WFS driver states which Zernike index layout ``get_zernike`` returns.

    A consumer that assumes one convention on the other device's vector does not crash: it
    silently reads a different mode (``w[1:]`` on the twin drops a real coefficient, ``w[5]``
    for defocus reads astigmatism). The declaration is only useful if it cannot be forgotten,
    left untyped, or quietly collapsed to one value.
    """
    base_tree = ast.parse((REPO / _WFS_BASE_FILE).read_text(encoding="utf-8-sig"))

    members: set[str] = set()
    base_declares_layout = False
    for node in ast.walk(base_tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name == _WFS_LAYOUT_ENUM:
            for st in node.body:
                if isinstance(st, ast.Assign):
                    members |= {t.id for t in st.targets if isinstance(t, ast.Name)}
        elif node.name == _WFS_BASE_NAME:
            for st in node.body:
                if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                    if st.target.id == _WFS_LAYOUT_ATTR:
                        base_declares_layout = True

    assert len(members) >= 2, (
        f"``{_WFS_LAYOUT_ENUM}`` defines {len(members)} member(s); the guard validates "
        f"declarations against them, so an empty or single-member enum proves nothing."
    )
    assert base_declares_layout, (
        f"``{_WFS_BASE_NAME}`` no longer declares ``{_WFS_LAYOUT_ATTR}``, so the attribute "
        "reads as unset on drivers that do not override it -- exactly the silent default this "
        "guard exists to prevent."
    )

    offenders: list[str] = []
    declared: dict[str, str] = {}
    scanned = 0
    for path in sorted((SRC / "ao_shaping" / "drivers").rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        scanned += 1
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig", errors="replace"))
        except SyntaxError:
            offenders.append(f"{rel}: <unparseable>")
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = {
                b.id if isinstance(b, ast.Name) else getattr(b, "attr", "")
                for b in node.bases
            }
            if _WFS_BASE_NAME not in bases:
                continue
            value: ast.expr | None = None
            for st in node.body:
                if isinstance(st, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == _WFS_LAYOUT_ATTR for t in st.targets
                ):
                    value = st.value
                elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                    if st.target.id == _WFS_LAYOUT_ATTR:
                        value = st.value
            if (
                not isinstance(value, ast.Attribute)
                or not isinstance(value.value, ast.Name)
                or value.value.id != _WFS_LAYOUT_ENUM
                or value.attr not in members
            ):
                offenders.append(
                    f"{rel}:{node.lineno}: class {node.name} declares no "
                    f"``{_WFS_LAYOUT_ENUM}.<member>`` as ``{_WFS_LAYOUT_ATTR}``"
                )
                continue
            declared[node.name] = value.attr

    assert scanned >= _MIN_WFS_DRIVER_FILES, (
        f"scanned {scanned} file(s) under ``src/ao_shaping/drivers``, below the floor "
        f"{_MIN_WFS_DRIVER_FILES} -- the scope stopped matching the tree."
    )
    assert len(declared) >= _MIN_WFS_DRIVERS, (
        f"only {len(declared)} WFS driver(s) declare a layout, below the floor "
        f"{_MIN_WFS_DRIVERS} -- the finder stopped matching ``{_WFS_BASE_NAME}`` subclasses."
    )
    assert not offenders, (
        f"{len(offenders)} WFS driver(s) without a valid ``{_WFS_LAYOUT_ATTR}`` "
        "declaration:\n  " + "\n  ".join(offenders[:15])
        + f"\nPick one ``{_WFS_LAYOUT_ENUM}.<member>`` that describes what ``get_zernike()`` "
        "actually returns; do not invent a string."
    )
    assert len(set(declared.values())) >= 2, (
        f"every WFS driver now declares the SAME layout ({declared}). They were measured to "
        "differ -- defocus sits at index 3 on the twin and slot 5 on the driver (66 vs 67 "
        "entries at order 10). Collapsing the declarations unifies a device-side boundary by "
        "accident: if that unification is deliberate, update ``drivers/wfs/AGENTS.md`` and "
        "ledger 8.9 items 49-50 in the same commit."
    )


#: Measured 2026-10-25 over 669 files: the ``w``/``w_before``/``wavefront`` unwrap existed
#: in two consumers, and the copies disagreed on their error path (one raised a
#: ``ClickException``, the other used a bare ``assert`` that ``python -O`` removes). The
#: fingerprint is the OR-chain of ``.get`` calls, NOT the key names: producers legitimately
#: write those keys, and ``h5py.File(path, "w")`` also contains ``"w"``.
_WFS_UNWRAP_KEYS = frozenset({"w", "w_before", "wavefront"})
_WFS_CHANNELS_MODULE = "src/ao_shaping/tools/zernike/common.py"
_MIN_WFS_UNWRAP_SCANNED = 600


def test_wfs_wavefront_unwrap_is_not_reimplemented() -> None:
    """Loading a saved WFS wavefront unwraps in exactly one place.

    ``wfs_channels_lambda`` owns the padding-slot rule, the length check and the piston
    zeroing; a second copy of the unwrap means a second opinion on all three.
    """
    chains: list[str] = []
    defining: list[str] = []
    unparsed: list[str] = []
    scanned = 0
    for path in (
        sorted((SRC / "ao_shaping").rglob("*.py")) + sorted((REPO / "scripts").rglob("*.py"))
    ):
        rel = path.relative_to(REPO).as_posix()
        scanned += 1
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig", errors="replace"))
        except SyntaxError:
            unparsed.append(rel)
            continue
        nodes = list(ast.walk(tree))
        for node in nodes:
            if not isinstance(node, ast.BoolOp) or not isinstance(node.op, ast.Or):
                continue
            keys = [
                call.args[0].value
                for call in node.values
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "get"
                and call.args
                and isinstance(call.args[0], ast.Constant)
                and call.args[0].value in _WFS_UNWRAP_KEYS
            ]
            if len(keys) >= 2:
                chains.append(rel)
                break
        if any(
            isinstance(n, ast.FunctionDef) and n.name == "wfs_channels_lambda" for n in nodes
        ):
            defining.append(rel)

    assert not unparsed, (
        f"{len(unparsed)} file(s) could not be parsed, so this guard saw nothing inside them: "
        f"{unparsed}"
    )
    assert scanned >= _MIN_WFS_UNWRAP_SCANNED, (
        f"scanned {scanned} file(s), below the floor {_MIN_WFS_UNWRAP_SCANNED} -- the scope "
        "stopped matching the tree."
    )
    assert defining == [_WFS_CHANNELS_MODULE], (
        f"``wfs_channels_lambda`` is defined in {defining}; it must have exactly one home."
    )
    offenders = sorted(set(chains) - set(defining))
    assert not offenders, (
        f"{len(offenders)} file(s) re-implement the WFS wavefront unwrap: {offenders}. Call "
        "``wfs_channels_lambda`` instead -- the duplicated copies had already drifted apart "
        "in how they reported a bad input."
    )


_SLOPE_FLATTEN_MODULE = "src/ao_shaping/utils/wavefront/wfs_utils.py"


def test_slope_flattening_is_not_reimplemented() -> None:
    """Stacking ``[dev_x; dev_y]`` into one slope vector happens in exactly one place.

    ``flatten_slopes`` owns the row order. Both sides of the closed loop read that order as a
    contract: a calibrated deviation matrix is filled through it, and ``from_response_matrix``
    expands its validity mask with ``np.concatenate([mask, mask])`` -- x rows first, then y
    rows. A hand-written stack is therefore a second opinion on the row order, and the two
    orderings have **identical shapes**: swapping x for y never raises, it silently feeds the
    pseudoinverse the wrong component of the measurement.

    The fingerprint is a one-argument ``concatenate``/``hstack`` over a 2-element sequence of
    ``.flatten()`` calls -- the shape the canonical helper itself compiles to, so anything
    matching it outside the helper is a duplicate rather than a different computation.
    """
    offenders: list[str] = []
    unparsed: list[str] = []
    defining: list[str] = []
    scanned = 0
    for path in (
        sorted((SRC / "ao_shaping").rglob("*.py")) + sorted((REPO / "scripts").rglob("*.py"))
    ):
        rel = path.relative_to(REPO).as_posix()
        scanned += 1
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig", errors="replace"))
        except SyntaxError:
            unparsed.append(rel)
            continue
        nodes = list(ast.walk(tree))
        if any(
            isinstance(n, ast.FunctionDef) and n.name == "flatten_slopes" for n in nodes
        ):
            defining.append(rel)
        if rel == _SLOPE_FLATTEN_MODULE:
            continue
        for node in nodes:
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr in {"concatenate", "hstack"}):
                continue
            if len(node.args) != 1 or not isinstance(node.args[0], ast.List | ast.Tuple):
                continue
            parts = node.args[0].elts
            if len(parts) == 2 and all(
                isinstance(part, ast.Call)
                and isinstance(part.func, ast.Attribute)
                and part.func.attr == "flatten"
                for part in parts
            ):
                offenders.append(f"{rel}:{node.lineno}")

    assert not unparsed, (
        f"{len(unparsed)} file(s) could not be parsed, so this guard saw nothing inside them: "
        f"{unparsed}"
    )
    assert scanned >= _MIN_WFS_UNWRAP_SCANNED, (
        f"scanned {scanned} file(s), below the floor {_MIN_WFS_UNWRAP_SCANNED} -- the scope "
        "stopped matching the tree."
    )
    assert defining == [_SLOPE_FLATTEN_MODULE], (
        f"``flatten_slopes`` is defined in {defining}; the row order must have one home."
    )
    assert not offenders, (
        f"{len(offenders)} hand-written x/y slope stack(s): {offenders}. Call "
        "``flatten_slopes`` instead -- a duplicate here changes the row order silently, "
        "because both orderings have the same shape."
    )


_TRUNCATED_MEAN_MODULE = "src/ao_shaping/utils/wavefront/wfs_utils.py"


def test_truncated_mean_is_not_reimplemented() -> None:
    """Dropping the extreme readings and averaging the rest happens in exactly one place.

    The rule was inlined seven times across the two matrix calibrators, gated on three
    different quantities (``n_averages``, ``len(responses)``, ``len(deviations)``) and explained
    in prose in only one of the two files -- the copies agreed numerically while their docs did
    not, which is how such a rule starts to disagree with itself.

    The fingerprint is a ``mean`` whose first argument is a ``[1:-1]`` slice: that is the shape
    of the helper's own body. A ``[1:-1]`` slice that is never averaged -- kernel trimming in
    ``exposure_fusion`` -- is deliberately **not** a match, so the guard does not tax unrelated
    slicing.
    """
    offenders: list[str] = []
    unparsed: list[str] = []
    defining: list[str] = []
    scanned = 0
    for path in (
        sorted((SRC / "ao_shaping").rglob("*.py")) + sorted((REPO / "scripts").rglob("*.py"))
    ):
        rel = path.relative_to(REPO).as_posix()
        scanned += 1
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig", errors="replace"))
        except SyntaxError:
            unparsed.append(rel)
            continue
        nodes = list(ast.walk(tree))
        if any(isinstance(n, ast.FunctionDef) and n.name == "truncated_mean" for n in nodes):
            defining.append(rel)
        if rel == _TRUNCATED_MEAN_MODULE:
            continue
        for node in nodes:
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "mean"):
                continue
            if not node.args or not isinstance(node.args[0], ast.Subscript):
                continue
            sl = node.args[0].slice
            if (
                isinstance(sl, ast.Slice)
                and isinstance(sl.lower, ast.Constant)
                and sl.lower.value == 1
                and isinstance(sl.upper, ast.UnaryOp)
                and isinstance(sl.upper.op, ast.USub)
                and isinstance(sl.upper.operand, ast.Constant)
                and sl.upper.operand.value == 1
            ):
                offenders.append(f"{rel}:{node.lineno}")

    assert not unparsed, (
        f"{len(unparsed)} file(s) could not be parsed, so this guard saw nothing inside them: "
        f"{unparsed}"
    )
    assert scanned >= _MIN_WFS_UNWRAP_SCANNED, (
        f"scanned {scanned} file(s), below the floor {_MIN_WFS_UNWRAP_SCANNED} -- the scope "
        "stopped matching the tree."
    )
    assert defining == [_TRUNCATED_MEAN_MODULE], (
        f"``truncated_mean`` is defined in {defining}; the truncation rule must have one home."
    )
    assert not offenders, (
        f"{len(offenders)} inline truncated-mean averaging site(s): {offenders}. Call "
        "``truncated_mean`` instead -- the copies had already drifted in how they expressed "
        "the gate and in which file explained it."
    )
def test_no_self_assignment_subscript() -> None:
    """A statement of the shape ``x[k] = x[k]`` is a conversion that converts nothing.

    Both matrix calibration results carried exactly that inside ``from_dict``, written right
    after real ``.tolist()`` conversions so it read like one of them -- while ``to_dict``
    went on advertising JSON output that ``json.dumps`` rejected, because the nested
    per-mode diagnostics kept their numpy arrays.

    The comparison is ``ast.unparse(target) == ast.unparse(value)``, not a (base, slice)
    tuple: an earlier hand-rolled key function collapsed ``ast.JoinedStr`` slices to an empty
    string and reported three legitimate GUI redirects of the form
    ``st.session_state[f"{p}_mode"] = st.session_state[f"{p}_mode_input"]`` as self-assignments.
    """
    offenders: list[str] = []
    unparsed: list[str] = []
    scanned = 0
    for path in (
        sorted((SRC / "ao_shaping").rglob("*.py")) + sorted((REPO / "scripts").rglob("*.py"))
    ):
        rel = path.relative_to(REPO).as_posix()
        scanned += 1
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig", errors="replace"))
        except SyntaxError:
            unparsed.append(rel)
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target, value = node.targets[0], node.value
            if not (isinstance(target, ast.Subscript) and isinstance(value, ast.Subscript)):
                continue
            if ast.unparse(target) == ast.unparse(value):
                offenders.append(f"{rel}:{node.lineno} {ast.unparse(node)}")

    assert not unparsed, (
        f"{len(unparsed)} file(s) could not be parsed, so this guard saw nothing inside them: "
        f"{unparsed}"
    )
    assert scanned >= _MIN_WFS_UNWRAP_SCANNED, (
        f"scanned {scanned} file(s), below the floor {_MIN_WFS_UNWRAP_SCANNED} -- the scope "
        "stopped matching the tree."
    )
    assert not offenders, (
        f"{len(offenders)} self-assignment site(s): {offenders}. A ``x[k] = x[k]`` statement "
        "either duplicates the conversion above it or hides the fact that none is happening."
    )
#: The paper's version is restated by hand in ``docs/INDEX.md``, and that restatement drifts:
#: upstream's v4.7 bump (00637811) left the index still saying v4.6, after the pin had already
#: needed its own catch-up commit once before (6bfde391).
_PAPER_DOC = REPO / "docs" / "paper_gs_refine_experiment.md"
_INDEX_DOC = REPO / "docs" / "INDEX.md"
#: Anchored to the head of the paper's version paragraph on purpose: the same paragraph ends
#: with a 版本沿革 list naming every older release, so an unanchored search would match the
#: oldest entry rather than the current one.
_PAPER_VERSION = re.compile(r"^版本\s+(v\d+\.\d+)（(\d{4}-\d{2}-\d{2}) 修订）", re.MULTILINE)
_INDEX_VERSION = re.compile(r"(v\d+\.\d+)（(\d{4}-\d{2}-\d{2}) 修订")


def test_the_paper_version_pin_in_index_tracks_the_paper() -> None:
    """``docs/INDEX.md`` must restate the paper's current version and revision date.

    Only those two tokens are pinned. The rest of the index row is prose about what a given
    revision added, and pinning it would force an index edit on every paper rewrite.
    """
    paper = _PAPER_VERSION.search(_PAPER_DOC.read_text(encoding="utf-8-sig"))
    assert paper, (
        "no version paragraph at the head of docs/paper_gs_refine_experiment.md -- the guard "
        "would be comparing nothing"
    )

    row = next(
        (
            line
            for line in _INDEX_DOC.read_text(encoding="utf-8-sig").split("\n")
            if "paper_gs_refine_experiment.md" in line
        ),
        None,
    )
    assert row, "docs/INDEX.md no longer carries a row for the paper -- the pin moved or renamed"
    pin = _INDEX_VERSION.search(row)
    assert pin, f"the paper's index row states no version: {row[:140]}"

    assert (pin.group(1), pin.group(2)) == (paper.group(1), paper.group(2)), (
        f"docs/INDEX.md pins the paper as {pin.group(1)}（{pin.group(2)} 修订） while the paper "
        f"opens with {paper.group(1)}（{paper.group(2)} 修订） -- restate the version in the "
        "index row in the same commit that bumps the paper"
    )
#: The generated call-graph wiki declares itself the authority on the CLI surface
#: (``SCHEMA.md``: ``entry/cli.md`` 的注册表为唯一权威), so its table, its per-command pages and
#: its stated count are assertions about ``main.py`` -- not prose that may remember a retired
#: name. ``AGENTS.md`` / ``CHANGELOG.md`` legitimately discuss ``spgd-square`` as history; this
#: scope deliberately excludes them (the repo-wide version of this guard was rejected once
#: already, TODO 8.9 item 30).
_WIKI_DIR = REPO / "docs" / "dev" / "call_graph_wiki"
_WIKI_COUNT_HEAD = re.compile(r"^## 注册的 (\d+) 个命令", re.MULTILINE)
#: Floor measured 2026-10-26: 19 registered commands and 19 pages. A guard that parses
#: nothing passes vacuously, so a shape change has to be loud.
_MIN_CLI_COMMANDS = 19


def test_call_graph_wiki_cli_registry_tracks_main() -> None:
    """The wiki's CLI registry must be exactly what ``main.py`` registers.

    Regression: the generator's command list kept the retired ``spgd-square`` after it stopped
    being a top-level command, and two of its three qualifier lookups ended in a bare ``else``
    that handed back the diagnose entry point -- so a page was published for a command that did
    not exist, claiming a source file it never touched.
    """
    from ao_shaping.main import cli  # the runtime registry, not a second parse of main.py

    registered = set(cli.commands)
    assert len(registered) >= _MIN_CLI_COMMANDS, (
        f"main.py registers {len(registered)} command(s), below the floor "
        f"{_MIN_CLI_COMMANDS} -- the import found a degraded CLI"
    )

    pages = {path.stem for path in (_WIKI_DIR / "commands").glob("*.md")}
    assert pages, "no command pages under docs/dev/call_graph_wiki/commands -- wrong wiki path"
    ghost_pages = sorted(pages - registered)
    assert not ghost_pages, (
        f"{len(ghost_pages)} wiki command page(s) name something main.py does not register: "
        f"{ghost_pages}. Drop the page and its index/diagram rows, or register the command."
    )
    missing_pages = sorted(registered - pages)
    assert not missing_pages, (
        f"{len(missing_pages)} registered command(s) have no wiki page: {missing_pages} -- "
        "regenerate the wiki (scripts/call_graph/gen_wiki_pages.py)"
    )

    cli_page = (_WIKI_DIR / "entry" / "cli.md").read_text(encoding="utf-8")
    table = set(re.findall(r"^\| `([a-z][a-z0-9-]+)` \| `", cli_page, re.MULTILINE))
    assert table, "the entry/cli.md registry table parsed to nothing -- its shape changed"
    assert table == registered, (
        f"entry/cli.md's registry table disagrees with main.py: only-in-table="
        f"{sorted(table - registered)} only-in-main={sorted(registered - table)}"
    )

    head = _WIKI_COUNT_HEAD.search(cli_page)
    assert head, "entry/cli.md lost its '## 注册的 N 个命令' heading -- the count went literal again"
    assert int(head.group(1)) == len(registered), (
        f"entry/cli.md states {head.group(1)} command(s) but main.py registers {len(registered)}"
    )
#: Every generated wiki page carries a ``source: <repo-relative>:<lineno>`` line, and the dev
#: handbook is read as fact. Two independent ways that line went wrong: the generator invented
#: one for a command that no longer exists (TODO 8.9 item 69), and a hand-written page pinned a
#: literal that drifted 43 lines behind the class it named (item 70). A citation is only worth
#: trusting if it lands on a definition's header -- decorator through ``def``/``class``.
_WIKI_DIR2 = REPO / "docs" / "dev" / "call_graph_wiki"
_WIKI_SOURCE = re.compile(r"^source:\s*(\S+?):(\d+)\s*$", re.MULTILINE)
#: Measured 2026-10-26: 42 citations across 50 pages. A guard that parses nothing passes
#: vacuously, so a front-matter format change has to be loud.
_MIN_WIKI_CITATIONS = 40


def test_wiki_source_citations_land_on_definitions() -> None:
    """Every ``source:`` citation in the call-graph wiki names a tracked file, at a definition.

    Citing a line inside a body is not merely imprecise: it points a reader at unrelated code,
    which is how a fabricated entry survived a whole review cycle -- the phantom page's citation
    was in range, so range checks alone could not have caught it.
    """
    offenders: list[str] = []
    checked = 0
    headers: dict[str, set[int]] = {}

    for page in sorted(_WIKI_DIR2.rglob("*.md")):
        rel_page = page.relative_to(REPO).as_posix()
        text = page.read_text(encoding="utf-8-sig", errors="replace")
        for match in _WIKI_SOURCE.finditer(text):
            checked += 1
            target, lineno = match.group(1), int(match.group(2))
            path = REPO / target
            if not path.is_file():
                offenders.append(f"{rel_page}: cited file {target} is not in the tree")
                continue
            if target not in headers:
                tree = ast.parse(path.read_text(encoding="utf-8-sig", errors="replace"), filename=target)
                good: set[int] = set()
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                        good.update(range(start, node.lineno + 1))
                headers[target] = good
            if lineno not in headers[target]:
                offenders.append(
                    f"{rel_page}: {target}:{lineno} is not a definition header "
                    f"(headers here: {sorted(headers[target])[:6]}...)"
                )

    assert checked >= _MIN_WIKI_CITATIONS, (
        f"parsed {checked} citation(s), below the floor {_MIN_WIKI_CITATIONS} -- the "
        "front-matter format stopped matching, so this guard is checking nothing"
    )
    assert not offenders, (
        f"{len(offenders)} wiki citation(s) point away from a definition:\n  "
        + "\n  ".join(offenders[:20])
        + "\nRegenerate with scripts/call_graph/build_call_graph.py + finalize_graph.py + "
        "gen_wiki_pages.py; if a page is hand-written, derive its citation instead of "
        "pinning a literal."
    )
#: Two docs restate the CLI inventory by hand as a table plus a count: ``README.md``'s command
#: overview and ``docs/user/cli-reference.md``'s reference table. Both were wrong at one point
#: (TODO 8.9 item 29 found 8 places naming a retired command as top-level), and nothing re-checks
#: them. Only table *rows* and the stated count are read, so prose that legitimately discusses a
#: retired command -- e.g. README's note that ``spgd-square`` is no longer top-level -- is never
#: treated as an inventory claim (the repo-wide prose version of this guard was rejected, item 30).
_CLI_DOC_README = REPO / "README.md"
_CLI_DOC_REFERENCE = REPO / "docs" / "user" / "cli-reference.md"
_CLI_README_ROW = re.compile(r"^\| `([a-z][a-z0-9-]+)` \| \[docs/", re.MULTILINE)
_CLI_README_COUNT = re.compile(r"注册 \*\*(\d+) 个顶层命令\*\*")
_CLI_REFERENCE_ROW = re.compile(r"^\| `([a-z][a-z0-9-]+)` \| [^|]+ \| \d+ \|$", re.MULTILINE)
_CLI_REFERENCE_COUNT = re.compile(r"当前共 \*\*(\d+) 个命令\*\*")
#: Measured 2026-10-26: 19 rows in each table. A guard that parses no rows passes vacuously.
_MIN_CLI_DOC_ROWS = 19


def test_cli_inventory_tables_in_docs_track_the_registered_commands() -> None:
    """Both hand-written CLI tables and their stated counts must equal the live registry.

    A command added to ``main.py`` without a doc row -- or a row left behind after removal --
    is a claim about an interface that does not exist, and users type what the table says.
    """
    from ao_shaping.main import cli  # the runtime registry, not a re-parse of main.py

    registered = set(cli.commands)
    assert len(registered) >= _MIN_CLI_DOC_ROWS, (
        f"main.py registers {len(registered)} command(s), below the floor "
        f"{_MIN_CLI_DOC_ROWS} -- the import found a degraded CLI"
    )

    sources = (
        (_CLI_DOC_README, _CLI_README_ROW, _CLI_README_COUNT, "**全局选项**"),
        (_CLI_DOC_REFERENCE, _CLI_REFERENCE_ROW, _CLI_REFERENCE_COUNT, "## 2. 全局选项"),
    )
    for path, row_re, count_re, tail_marker in sources:
        rel = path.relative_to(REPO).as_posix()
        text = path.read_text(encoding="utf-8-sig")
        table_text = text.split(tail_marker)[0] if tail_marker in text else text

        rows = row_re.findall(table_text)
        assert len(rows) >= _MIN_CLI_DOC_ROWS, (
            f"{rel}: parsed {len(rows)} command row(s), below the floor {_MIN_CLI_DOC_ROWS} "
            "-- the table's shape changed, so this guard is checking nothing"
        )
        extra = sorted(set(rows) - registered)
        missing = sorted(registered - set(rows))
        assert not extra and not missing, (
            f"{rel}'s CLI table disagrees with main.py: "
            f"listed-but-not-registered={extra} registered-but-unlisted={missing}"
        )

        stated = count_re.findall(text)
        assert stated, f"{rel} no longer states its command count -- the wording changed"
        wrong = [n for n in stated if int(n) != len(registered)]
        assert not wrong, (
            f"{rel} states {wrong} command(s) while main.py registers {len(registered)} -- "
            "restate the count in the same commit that changes the CLI"
        )
#: Environment variables are read by name and set by name, and the two ends never meet in a
#: type checker, so a casing slip produces a knob that silently does nothing. Measured on
#: 2026-10-26: production code reads 22 distinct names, and exactly one pair collided --
#: ``Far_Cam_ID`` (the spelling in ``.env``, ``.env.example``, ``AGENTS.md`` and the CLI
#: reference) and ``FAR_CAM_ID`` (read in three sites, set nowhere in the repo or CI). Those
#: three sites therefore ignored the configured camera id and always used their fallback.
#: A deliberate second spelling has to be registered in the allowlist below, with a reason.
_ENV_ALLOWLIST: tuple[str, ...] = ()
#: Measured 22 names read. A guard that parses nothing passes vacuously (TODO 8.9 #32/#34),
#: so the parse has to be loud about its own scope.
_MIN_ENV_NAMES_READ = 15

#: Platform-provided names that production code reads but this repo does not define.
#: Requiring a doc entry for them would mean documenting Windows or Linux itself.
_OS_ENV_NAMES: tuple[str, ...] = ("COMPUTERNAME", "USERPROFILE", "XDG_DATA_HOME")


def test_env_names_read_by_code_are_documented() -> None:
    """Every environment name production code reads is named in a live document.

    The case-collision guard checks one direction; this checks the other. A knob can be read at
    runtime, honoured, and still be undiscoverable: the census over 21 read names found 11
    repo-owned ones in no live document -- four ``*_CONFIG_DIR`` overrides, three vendor SDK/DLL
    path seams, one optional OOPAO source path, and three debug/tuning switches -- beside 3
    OS-provided names that are exempted rather than documented.
    """
    found, unparsed, scanned = _env_names_read_by_code()

    assert not unparsed, (
        f"{len(unparsed)} file(s) could not be parsed, so this guard saw nothing inside them: "
        f"{unparsed}"
    )
    assert len(found) >= _MIN_ENV_NAMES_READ, (
        f"only {len(found)} environment name(s) parsed over {scanned} file(s), below the floor "
        f"{_MIN_ENV_NAMES_READ} -- the extractor stopped matching how this repo reads env vars"
    )

    listing = subprocess.run(
        ["git", "ls-files", "-z"], capture_output=True, cwd=str(REPO), check=False
    ).stdout
    tracked = {t.decode("utf-8", errors="replace") for t in listing.split(b"\0") if t}

    corpus: dict[str, str] = {}
    for rel in tracked:
        if not rel.endswith(".md") or rel == "CHANGELOG.md":
            continue
        path = REPO / rel
        if _is_historical(path):
            continue
        corpus[rel] = path.read_text(encoding="utf-8", errors="replace")
    assert len(corpus) >= 40, (
        f"only {len(corpus)} live document(s) scanned -- the corpus disappeared, so a pass here "
        "would only prove the search was empty"
    )

    missing: dict[str, list[str]] = {}
    for name, sites in found.items():
        if name in _OS_ENV_NAMES:
            continue
        if not any(name in text for text in corpus.values()):
            missing[name] = sorted(sites)[:3]

    assert not missing, (
        f"{len(missing)} environment name(s) are read by production code but appear in no live "
        f"document ({len(corpus)} docs scanned, {len(_OS_ENV_NAMES)} OS-provided names exempted):\n  "
        + "\n  ".join(f"{n} <- {v}" for n, v in sorted(missing.items()))
        + "\nDocument each in the Environment Variables section of AGENTS.md with its default "
        "value and what it changes: a knob nobody can find is a knob nobody can set."
    )


_QUOTED_PY_PATH = re.compile(
    r"""[`'"]((?:src|scripts|tests)/[A-Za-z0-9_./-]+\.py)(?::\d+)?[`'"]"""
)

#: Citations whose referring text declares the target gone. Each key is
#: (citing file, cited path) and each value states the deletion commit or the record's own
#: self-disclaimer -- a bare stale path never belongs here, it gets repointed. Sized from the
#: census that found 11 of the 22 citations in this class, so the exemption is the minority.
_QUOTED_PY_PATH_HISTORICAL: dict[tuple[str, str], str] = {
    ("docs/issues_report.md", "src/ao_shaping/utils/wfs_utils.py"):
        "frozen 2026-05 snapshot whose own header declares the cited line numbers invalid",
    ("docs/issues_report.md", "src/ao_shaping/utils/pattern_helper.py"):
        "same frozen snapshot",
    ("docs/issues_report.md", "src/ao_shaping/drivers/ccd/miicam_driver.py"):
        "same frozen snapshot",
    ("scripts/ml/generate_inverse_design_report.py", "scripts/crosstalk_augmentation_train.py"):
        "emitted inside _removed_note(): the experiment is declared void and the script was "
        "deleted in 6b85858a, so naming it is the record",
    ("scripts/ml/generate_inverse_design_report.py", "scripts/inverse_crosstalk_check.py"):
        "same _removed_note() annotation",
    ("scripts/slm/generate_fouriergsnet_pipeline_report.py",
     "tests/ao_shaping/drivers/sim/test_sim_fouriergsnet_pipeline.py"):
        "self-dated (2026-09-22) frozen run snapshot; those tests were deleted in baefba2d -- "
        "annotating or regenerating this snapshot is an owner call",
    ("scripts/slm/generate_fouriergsnet_pipeline_report.py",
     "tests/ao_shaping/runners/test_fouriergsnet_optimize.py"):
        "same frozen snapshot",
    ("scripts/slm/generate_diff_shaping_report.py", "src/ao_shaping/runners/diff_shaping_runner.py"):
        "the runner was deleted in fbef1926 and the text says so, mirroring AGENTS.md's own "
        "convention for that entry",
    ("scripts/_common/provenance.py",
     "tests/ao_shaping/drivers/ccd/test_miicam_simulation_report.py"):
        "the note itself declares the generator never entered the index (全仓 git ls-files 无此 "
        "path), so the missing generator -- not this citation -- is what would need fixing",
}

#: The census resolved 362 claims; a floor this far below catches an extractor that stopped
#: matching without silently checking nothing.
_MIN_QUOTED_PY_PATH_CLAIMS = 250


def test_quoted_path_exemptions_are_still_needed() -> None:
    """Each entry of :data:`_QUOTED_PY_PATH_HISTORICAL` must still exempt a real miss.

    An exemption is justified only while the path it excuses is absent from the git index.
    When the path lands, the entry becomes unreachable (the consumer short-circuits on
    ``cited in tracked`` before consulting it) and its rationale starts asserting something
    false: the two pib_exit_safety entries deleted here still read "the script was never
    committed" after upstream committed the generator and its test. Nothing else in the suite
    notices, so dead justifications and their false claims accumulate silently.
    """
    listing = subprocess.run(
        ["git", "ls-files", "-z"], capture_output=True, cwd=str(REPO), check=False
    )
    tracked = {t.decode("utf-8", errors="replace") for t in listing.stdout.split(b"\0") if t}
    assert tracked, "git ls-files returned nothing; is this a git repository?"
    stale = sorted(
        (citing, cited) for (citing, cited) in _QUOTED_PY_PATH_HISTORICAL if cited in tracked
    )
    assert not stale, (
        f"{len(stale)} exemption(s) excuse a path that is tracked today: {stale[:5]} -- the "
        "citation resolves, so delete the entry rather than keep a rationale claiming the "
        "file does not exist."
    )

#: ``python <hub> main.py <command> [subcommand] [flags...]`` as written in live text. The tail is
#: captured on one line only: a backslash-continued block puts its remaining flags on the next line,
#: and letting ``\s`` cross the newline made the next line's ``python`` look like a subcommand.
_CLI_INVOCATION = re.compile(
    r"(?:python3?|uv run python)[^\n]*?\S*main\.py((?:[ \t]+\S+)*)"
)

#: 72 invocation claims resolved over live markdown and src/ + scripts/ string literals, 65 of them
#: runnable-looking and the rest click synopses; the floor leaves room for rewording but not for an
#: extractor that quietly matches nothing.
_MIN_CLI_INVOCATION_CLAIMS = 50


def test_cli_invocations_in_live_text_name_real_commands_and_options() -> None:
    """Every written-out CLI invocation names a command, subcommand and flag that exist.

    The sibling guards here check that an invoked *script* is tracked and that a quoted *path* is
    tracked; neither looks at what follows ``main.py``. This resolves the whole invocation against the
    live click tree: the command, the subcommand when the command is a group, and every flag. A bare
    group is an offender because click answers ``Missing command``, and a flag the command never
    declared is an offender because click answers ``no such option``. Flag spellings are the part docs
    drift on most easily -- ``docs/user/cli-reference.md`` already documents a live ``-d`` collision
    between group level and command level, and a group option placed after the subcommand is rejected
    exactly the way a made-up one is.
    """
    import click

    from ao_shaping.main import cli

    tracked = {
        t.decode("utf-8", errors="replace")
        for t in subprocess.run(
            ["git", "ls-files", "-z"], capture_output=True, cwd=str(REPO), check=False
        ).stdout.split(b"\0")
        if t
    }
    targets = sorted(
        rel
        for rel in tracked
        if (rel.endswith(".py") and (rel.startswith("src/") or rel.startswith("scripts/")))
        or (rel.endswith(".md") and rel != "CHANGELOG.md" and not _is_historical(REPO / rel))
    )

    group_allowed: set[str] = {"--help", "-h"}
    group_params: dict[str, click.Parameter] = {}
    for param in cli.params:
        group_allowed.update(param.opts)
        group_allowed.update(param.secondary_opts)
        for opt in (*param.opts, *param.secondary_opts):
            group_params[opt] = param

    def placeholder(token: str) -> bool:
        # ``main.py [OPTIONS] [COMMAND]`` is how click itself writes a synopsis, and ``…`` /
        # ``<command>`` are how this repo abbreviates one. Neither is a runnable invocation.
        return token.startswith(("<", "[")) or chr(8230) in token


    offenders: list[str] = []
    checked = 0
    for rel in targets:
        path = REPO / rel
        if not path.is_file():
            continue
        if rel.endswith(".py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8-sig", errors="replace"), filename=rel)
            except SyntaxError:
                continue
            claims = [
                (node.lineno, value)
                for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
                for value in node.value.splitlines()
            ]
        else:
            claims = list(
                enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1)
            )

        for lineno, line in claims:
            for match in _CLI_INVOCATION.finditer(line):
                checked += 1
                tokens = [t for t in match.group(1).split() if t != chr(92)]
                while tokens and tokens[0].startswith("-"):
                    raw = tokens.pop(0)
                    flag = raw.split("=", 1)[0]
                    param = group_params.get(flag)
                    if flag not in group_allowed:
                        offenders.append(
                            f"{rel}:{lineno} -> {flag!r} is not an option of the top-level group; "
                            f"a group flag must precede the command name. Declared: "
                            f"{sorted(group_allowed)}"
                        )
                    elif param is not None and not param.is_flag and "=" not in raw and tokens:
                        tokens.pop(0)
                if not tokens:
                    continue
                name = tokens.pop(0)
                rest = tokens
                if placeholder(name):
                    continue
                command = cli.commands.get(name)
                if command is None:
                    offenders.append(f"{rel}:{lineno} -> command {name!r} is not registered")
                    continue
                # Resolve a subcommand only through a group; a plain command takes none.
                if isinstance(command, click.Group):
                    nxt = next(
                        (t for t in rest if not t.startswith("-") and not placeholder(t)), None
                    )
                    if nxt is None and not any(t in ("--help", "-h") for t in rest):
                        offenders.append(
                            f"{rel}:{lineno} -> group {name!r} invoked with no subcommand "
                            f"(click: Missing command); its subcommands are "
                            f"{sorted(command.commands)}"
                        )
                        continue
                    if nxt is not None and nxt in command.commands:
                        command = command.commands[nxt]
                        rest = rest[rest.index(nxt) + 1:]
                    elif nxt is not None:
                        offenders.append(
                            f"{rel}:{lineno} -> {name} {nxt!r} is not a subcommand of that group; "
                            f"registered: {sorted(command.commands)}"
                        )
                        continue
                # click attaches --help lazily (get_help_option_names(ctx)), so it never shows up in
                # command.params. Accept the default plus the -h alias this hub's context_settings
                # declares; anything else has to be a real declared option.
                allowed: set[str] = {"--help", "-h"}
                for param in command.params:
                    allowed.update(param.opts)
                    allowed.update(param.secondary_opts)
                for token in rest:
                    if not token.startswith("-") or token.startswith("<"):
                        continue
                    flag = token.split("=", 1)[0].rstrip(").,;:`\"'")
                    if flag in ("--", "-"):
                        continue
                    if flag not in allowed:
                        offenders.append(
                            f"{rel}:{lineno} -> {flag!r} is not an option of "
                            f"{name}; "
                            f"declared: {sorted(a for a in allowed if a.startswith('--'))[:12]}"
                        )

    assert checked >= _MIN_CLI_INVOCATION_CLAIMS, (
        f"resolved only {checked} CLI invocation claim(s), below the floor "
        f"{_MIN_CLI_INVOCATION_CLAIMS} -- the grammar stopped matching how this repo writes its "
        "commands, so a pass here would mean nothing was checked."
    )
    assert not offenders, (
        f"{len(offenders)} written-out CLI invocation(s) cannot run as written:\n  "
        + "\n  ".join(offenders[:20])
        + "\nFix the text to the command tree that exists (run `python src/ao_shaping/main.py "
        "<command> --help` to see real flags); do not invent options."
    )



def test_quoted_source_paths_in_live_text_exist_in_the_git_index() -> None:
    """A quoted repo path is an existence claim: the file must be in the next clone.

    ``test_python_invocation_paths_in_docs_and_docstrings_are_tracked`` covers ``python <path>.py``
    instructions in module docstrings and live markdown. This covers the wider channel it cannot
    see: a path quoted as text -- in any string literal of ``src/`` and ``scripts/`` (which is where
    report generators write the tables their readers follow) and in live markdown. A census over 361
    such claims found 9 pointing at files that had moved or been deleted while the text still
    presented them as present: ``algorithm/differentiable_shaping.py`` (relocated into
    ``algorithm/signal_processing/`` in bbb3820b, three citations), ``optimizer/wfless/
    slm_zernike_shaping.py`` (the zero-importer re-export shim deleted in 8792ced1),
    ``tools/slm/slm_zernike_{response,correction,common}.py`` (regrouped into ``tools/wfs/`` and
    ``tools/zernike/`` in 699f149f and a later pass), ``tests/ao_shaping/runners/test_gsnet_cache.py``
    (moved under ``tests/ao_shaping/ml/``), and one citation in ``square_shaping.py`` to a
    ``test_square_shaping_fienup_variants.py`` that has never existed at all -- that last one was not
    a stale pointer but a claimed numerical verification with no test behind it.
    """
    import subprocess

    listing = subprocess.run(
        ["git", "ls-files", "-z"], capture_output=True, cwd=str(REPO), check=False
    ).stdout
    tracked = {t.decode("utf-8", errors="replace") for t in listing.split(b"\0") if t}

    offenders: list[str] = []
    checked = 0
    exempt = 0
    for rel in sorted(tracked):
        path = REPO / rel
        if not path.is_file():
            continue
        claims: list[tuple[int, str]] = []
        if rel.endswith(".py"):
            # tests/ is left out on purpose: a test docstring cannot mislead a reader about where
            # production code lives, and this mirrors the scope of the invocation guard.
            if not (rel.startswith("src/") or rel.startswith("scripts/")):
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8-sig", errors="replace"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    for match in _QUOTED_PY_PATH.finditer(node.value):
                        claims.append((node.lineno, match.group(1)))
        elif rel.endswith(".md") and rel != "CHANGELOG.md" and not _is_historical(path):
            # A changelog's job is to name the files it removed, so it is out of scope even though
            # _is_historical() does not list it.
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
            ):
                for match in _QUOTED_PY_PATH.finditer(line):
                    claims.append((lineno, match.group(1)))
        else:
            continue
        for lineno, cited in claims:
            # "tests/.../test_x.py" is a placeholder a human wrote, not a resolvable claim.
            if "..." in cited:
                continue
            checked += 1
            if cited in tracked:
                continue
            reason = _QUOTED_PY_PATH_HISTORICAL.get((rel, cited))
            if reason is not None:
                exempt += 1
                continue
            offenders.append(f"{rel}:{lineno} -> {cited!r} (not in the git index)")

    assert checked >= _MIN_QUOTED_PY_PATH_CLAIMS, (
        f"resolved only {checked} quoted-path claim(s), below the floor "
        f"{_MIN_QUOTED_PY_PATH_CLAIMS} -- the grammar stopped matching the repo's citation style, "
        "so a pass here would mean nothing was checked."
    )
    assert not offenders, (
        f"{len(offenders)} quoted path(s) name a file that is not in the git index "
        f"({exempt} exempted as self-declared history):\n  " + "\n  ".join(offenders[:20])
        + "\nRepoint each to where the file actually lives (git log --follow shows the move), or, "
        "if it really was deleted, make the text say so and register the exemption with the commit."
    )



def _env_names_read_by_code() -> tuple[dict[str, list[str]], list[str], int]:
    """Collect every literal environment variable name read under ``src/`` and ``scripts/``."""
    found: dict[str, list[str]] = {}
    unparsed: list[str] = []
    scanned = 0
    for path in sorted(SRC.rglob("*.py")) + sorted((REPO / "scripts").rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        scanned += 1
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig", errors="replace"), filename=rel)
        except SyntaxError:
            unparsed.append(rel)
            continue
        for node in ast.walk(tree):
            name: str | None = None
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ("get", "getenv", "setdefault"):
                    base = node.func.value
                    base_name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", "")
                    if base_name in ("os", "environ") and node.args and isinstance(node.args[0], ast.Constant):
                        name = node.args[0].value
            elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute):
                if node.value.attr == "environ" and isinstance(node.slice, ast.Constant):
                    name = node.slice.value
            if isinstance(name, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                found.setdefault(name, []).append(rel)
    return found, unparsed, scanned


def test_env_names_read_by_code_have_no_case_collisions() -> None:
    """No two environment variable names read by production code may differ only in case.

    A split spelling does not fail: the un-set name just keeps returning its default, so the
    code looks configured while honoring nothing. The camera id is the concrete case -- three
    sites read ``FAR_CAM_ID`` while ``.env`` sets ``Far_Cam_ID``, and one of those very
    declarations advertises ``Far_Cam_ID`` in its own ``--help`` text.
    """
    found, unparsed, scanned = _env_names_read_by_code()

    assert not unparsed, (
        f"{len(unparsed)} file(s) could not be parsed, so this guard saw nothing inside them: "
        f"{unparsed}"
    )
    assert len(found) >= _MIN_ENV_NAMES_READ, (
        f"only {len(found)} environment name(s) parsed over {scanned} file(s), below the floor "
        f"{_MIN_ENV_NAMES_READ} -- the extractor stopped matching how this repo reads env vars"
    )

    groups: dict[str, list[str]] = {}
    for name in found:
        groups.setdefault(name.upper(), []).append(name)
    collisions = {k: sorted(v) for k, v in groups.items() if len(v) > 1 and k not in _ENV_ALLOWLIST}

    assert not collisions, (
        f"{len(collisions)} env-name case collision(s): "
        + "; ".join(
            f"{names} read by {sorted(set().union(*(found[n] for n in names)))}"
            for names in sorted(collisions.values())
        )
        + " -- one spelling is settable and the other silently is not: pick the documented "
        "spelling, or register the pair in _ENV_ALLOWLIST with a reason."
    )


#: ``AGENTS.md`` pins its claims to specific guards with ``module.py::test_name``. A
#: pointer at a renamed or withdrawn guard is worse than no pointer: the paragraph then
#: asserts protection that no longer exists -- item 117 shipped exactly that, with every
#: gate green.
AGENTS_MD = REPO / "AGENTS.md"

#: The reference form is ``module.py::test_name``; both a bare module name and a
#: repo-relative path appear in the document, and the parser accepts both.
AGENT_REF_RE = re.compile(r"`(?P<module>[\w./-]+\.py)::(?P<name>test_\w+)`")


def _dangling_agent_refs(text: str) -> list[str]:
    dangling: list[str] = []
    for match in AGENT_REF_RE.finditer(text):
        module, name = match.group("module"), match.group("name")
        direct = REPO / module
        if direct.is_file():
            candidates = [direct]
        else:
            candidates = sorted((REPO / "tests").rglob(Path(module).name))
        if not candidates:
            dangling.append(f"{module}::{name} (no such module)")
            continue
        body = "\n".join(path.read_text(encoding="utf-8") for path in candidates)
        if not re.search(rf"^(?:async )?def {name}\b", body, re.MULTILINE):
            dangling.append(f"{module}::{name}")
    return dangling


def test_agents_md_guard_pointers_resolve() -> None:
    dangling = _dangling_agent_refs(AGENTS_MD.read_text(encoding="utf-8"))
    assert not dangling, (
        f"AGENTS.md cites guards that no longer exist: {dangling}. Point the sentence at "
        f"the guard that enforces the claim now, or drop the sentence.")


def test_agents_md_pointer_guard_is_not_vacuous() -> None:
    """Both accepted spellings must be parsed, and the live doc must yield matches.

    The first version of this control fed only the bare-module spelling, so it kept
    passing while the regex was too narrow to see the full-path form AGENTS.md actually
    uses -- a control that mirrors the bug it is guarding cannot report it.
    """
    sample = (
        "bare kept `test_conventions.py::test_agents_md_guard_pointers_resolve` "
        "bare gone `test_conventions.py::test_a_guard_that_was_withdrawn` "
        "path kept `tests/ao_shaping/test_conventions.py::test_all_ruff_pins_agree_and_are_exact` "
        "path gone `tests/ao_shaping/test_conventions.py::test_a_guard_that_was_withdrawn` "
        "missing `tests/ao_shaping/no_such_module.py::test_anything`"
    )
    assert _dangling_agent_refs(sample) == [
        "test_conventions.py::test_a_guard_that_was_withdrawn",
        "tests/ao_shaping/test_conventions.py::test_a_guard_that_was_withdrawn",
        "tests/ao_shaping/no_such_module.py::test_anything (no such module)",
    ], _dangling_agent_refs(sample)

    live = AGENTS_MD.read_text(encoding="utf-8")
    assert len(AGENT_REF_RE.findall(live)) >= 5, (
        f"only {len(AGENT_REF_RE.findall(live))} pointers parsed in AGENTS.md -- the "
        f"pattern stopped matching how the document spells its references")
