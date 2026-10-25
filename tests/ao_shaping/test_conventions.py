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
import os
import re
import shutil
import subprocess
import sys
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


def test_pytest_excludes_match_across_configs() -> None:
    """`run_pytest_ci.sh` re-states pyproject's `--ignore` list -- they must agree.

    The duplication is deliberate, not accidental: the runner starts pytest from a
    scratch directory, so pyproject's *relative* excludes (``tests/ao_shaping/ml``)
    resolve against that CWD and match nothing, which would pull the 65 GB hwdataset
    corpus and the hardware domains back into CI runs. So the script re-states them as
    absolute paths.

    Two lists that must mean the same thing is exactly how this repo has been bitten
    before, and it happened while this guard was being written: pyproject was narrowed
    to ``drivers/dm`` while the script still carried the whole ``drivers`` tree, so the
    narrowed discovery silently stayed off and every driver domain looked "excluded"
    even though the ledger said otherwise.
    """
    script = REPO / "scripts" / "build" / "run_pytest_ci.sh"
    shell_ignores = {
        match.removeprefix("$WORKDIR_IN_CONTAINER/")
        for match in re.findall(r'"--ignore=([^"]+)"', script.read_text(encoding="utf-8"))
    }
    addopts = (
        tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
        .get("tool", {})
        .get("pytest", {})
        .get("ini_options", {})
        .get("addopts", [])
    )
    pyproject_ignores = {
        entry.split("=", 1)[1]
        for entry in addopts
        if isinstance(entry, str) and entry.startswith("--ignore=")
    }
    assert shell_ignores == pyproject_ignores, (
        "pytest excludes drifted between pyproject.toml addopts and "
        f"scripts/build/run_pytest_ci.sh EXCLUDES -- only pyproject: "
        f"{sorted(pyproject_ignores - shell_ignores)}; only the script: "
        f"{sorted(shell_ignores - pyproject_ignores)}"
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
# defend, not an endorsement.
_BLIND_SILENT_BASELINE = 61


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
        keys = json.loads(golden_path.read_text(encoding="utf-8"))
        golden = {k for k in keys if ":" not in k}          # module-scoped keys carry ':'

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
