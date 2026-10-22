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

"""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import sys
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
    [REPO / "README.md"]
    + sorted(SRC.rglob("*.md"))
    + sorted((REPO / "scripts").glob("*.md"))
    + sorted((REPO / "docs").glob("*.md"))
)


def _is_historical(path: Path) -> bool:
    rel = path.relative_to(REPO).as_posix()
    return any(marker in rel for marker in _HISTORICAL_DOCS)


def _dotted(name: str) -> str:
    return name[: -len(".__init__")] if name.endswith(".__init__") else name


def _module_inventory() -> dict[str, tuple[bool, bool]]:
    """``{module: (has __main__ guard, declares a click command)}``."""
    out: dict[str, tuple[bool, bool]] = {}
    for path in sorted((SRC / "ao_shaping").rglob("*.py")):
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
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
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

_PY_M = re.compile(r"python\s+-m\s+(ao_shaping[\w.]*)")
_MD_M = re.compile(r"python\s+-m\s+(ao_shaping[\w.]*)")

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
        info = INVENTORY.get(mod)
        if info is None:
            problems.setdefault(mod, []).append(f"{where} (no such module)")
        elif not any(info):
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
#: needs a reason.
_ALLOWED_UNTRACKED_WRITERS = {
    "tests/ao_shaping/drivers/ccd/test_miicam_simulation_report.py": "targets report/miicam_simulation/, which git does not track, and needs no "
    "device (it is the simulated camera)",
}


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
    # or not it is tracked. Falling through to the name here would flag
    # TestReport("miicam", device_dir="report/miicam_simulation") as writing to
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
    merely leaving untracked files behind. test_miicam_simulation_report.py is
    therefore allowed -- it needs no device and writes to an untracked dir.
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
    text = writer.read_text(encoding="utf-8")
    assert "def write_table(" in text and "def write_artifacts(" in text
    # And the algorithm module must no longer expose them.
    algo = (
        SRC
        / "ao_shaping"
        / "algorithm"
        / "signal_processing"
        / "beam_shaping_benchmark.py"
    ).read_text(encoding="utf-8")
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
