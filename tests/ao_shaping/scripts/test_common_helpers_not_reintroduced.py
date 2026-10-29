"""R-25 — the shared helpers must not re-duplicate.

The extraction is only worth anything if the copies stay gone. These tests scan
``scripts/`` for the helper *definitions* that were removed and fail if any
reappears, and pin the migration contract:

* every report generator that uses a helper imports it from ``scripts._common``;
* no generator defines its own copy;
* ``scripts`` is an importable package (that is what makes
  ``from scripts._common import ...`` work at all).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = _ROOT / "scripts"

#: Definitions that were extracted. Keyed by the old local name; the value is
#: the canonical home (the assert message interpolates it verbatim).
EXTRACTED = {
    "_fmt": "scripts/_common: fmt_metric / fmt_ratio / fmt_general / fmt_fixed / fmt_signed (FIVE distinct formatters)",
    "_markdown_table": "scripts/_common: markdown_table",
    "_savefig": "scripts/_common: savefig",
    "iters_to_threshold": "scripts/_common: iters_to_threshold",
    "format_iters": "scripts/_common: format_iters",
    "oopao_package_dir": "scripts/_common/oopao_probe.py (both scripts/oopao/ generators)",
    # The device-mutex set: ``scripts/slm/run_hw_sweep_groups.py`` carried a
    # second copy of these while *already* importing the rest of the module
    # ("再迁复用", AGENTS.md 「scripts/ 目录」 rule 2). The private names are
    # imported across the module boundary on purpose -- forwarding, not
    # forking; a local copy here would drift from the classifier that the
    # live probe/queue paths actually consult.
    "_ensure_repo_paths": "src/ao_shaping/tools/kernels/device_mutex.py",
    "_extract_santec_error_code": "src/ao_shaping/tools/kernels/device_mutex.py",
    "_reason_text": "src/ao_shaping/tools/kernels/device_mutex.py",
    "classify_probe_setup_error": "src/ao_shaping/tools/kernels/device_mutex.py",
    "format_occupier_lines": "src/ao_shaping/tools/kernels/device_mutex.py",
    "parse_probe_cmd": "src/ao_shaping/tools/kernels/device_mutex.py",
    "query_windows_python_processes": "src/ao_shaping/tools/kernels/device_mutex.py",
    # Slot rotation: the 2..125 window and the "never the slot already being
    # displayed" exclusion *are* the firmware no-op fix (AGENTS.md -- display_memory
    # on the displayed slot leaves the LCOS unrefreshed), so a drifted copy is a
    # hardware bug, not a style issue. scripts/slm/diff_beam_capture_flat.py had one.
    "pick_slm_slot": "src/ao_shaping/utils/slm/phase_display.py",
    # CJK font setup: seven copies with three distinct bodies. The preference
    # chain is a parameter and the installed-font probe a flag, because merging
    # the chains would re-render tracked figures with a different font.
    "setup_cjk_font": "scripts/_common/plot_style.py: setup_cjk_font",
    "_setup_fonts": "scripts/_common/plot_style.py: setup_cjk_font(probe_installed=True)",
    # Leave-one-group-out folds over record paths: the grouping (key = str(path),
    # folds in sorted-key order, group-first train lists) *is* the paired-comparison
    # contract -- a drifted copy silently changes which records share a fold, so the
    # per-fold numbers in the diagnostic reports stop being comparable. Nine copies
    # existed; all now import the canonical ("再迁复用" batch 3). `ridge` is
    # deliberately NOT registered: legitimate different-purpose ridge copies remain.
    "folds_of": "src/ml/zernike/eval_stats.py: folds_of",
}

#: Generators migrated onto ``scripts._common``.
MIGRATED = (
    "benchmark_oopao_gpu.py",
    "compare_loss_algorithms.py",
    "compare_sim_vs_hw.py",
    "generate_diff_shaping_report.py",
    "generate_fouriergsnet_sim_report.py",
    "generate_gsnet_offline_report.py",
    "generate_heuristic_pib_report.py",
    "generate_inverse_design_report.py",
    "generate_multishape1200_report.py",
    "generate_oopao_impact_report.py",
    "generate_oopao_vs_numpy_report.py",
    "generate_pib_exit_safety_report.py",
    "generate_slm_pib_online_report.py",
    "generate_slm_pib_rms_pib_report.py",
    "generate_shape_objective_comparison.py",
    "generate_strehl_benchmark_report.py",
    "generate_zernike_coeff_report.py",
    "validate_sim2f_aberration_ensemble.py",
)


def _read(path: Path) -> str:
    """Read a script, tolerating a UTF-8 BOM.

    One checked-in script still carries a BOM, which makes ``ast.parse`` raise
    ``SyntaxError: invalid non-printable character U+FEFF`` for any tool that
    decodes with plain ``utf-8``. That is a latent hazard in its own right, so the
    scan must not be the thing that trips over it.
    """
    return path.read_text(encoding="utf-8-sig")


def _script_files() -> list[Path]:
    """Every generator under ``scripts/``, recursively, excluding ``_common``.

    Recursive because ``scripts/`` is grouped into domain subfolders
    (``slm/``, ``dm/``, ``wfs/``, ``ml/``, ``rl/``, ``zernike/``, ``oopao/``,
    ``build/``). A ``glob("*.py")`` would silently scan **nothing** once the
    generators move, and every guard below would pass vacuously -- a green test
    that checks nothing is worse than a red one.

    ``_common`` is excluded because it is where the extracted helpers are
    *defined*; scanning it would make this guard fire on its own canonical home.
    """
    return sorted(
        p
        for p in _SCRIPTS.rglob("*.py")
        if "__pycache__" not in p.parts
        and "_common" not in p.relative_to(_SCRIPTS).parts
    )


def _find_script(name: str) -> Path | None:
    """Locate ``name`` anywhere under ``scripts/``, at any depth."""
    matches = [p for p in _script_files() if p.name == name]
    if not matches:
        return None
    assert len(matches) == 1, f"{name} exists at {len(matches)} locations: {matches}"
    return matches[0]


def _repo_root_depth(path: Path) -> int:
    """``N`` such that ``Path(__file__).resolve().parents[N]`` is the repo root."""
    return len(path.relative_to(_ROOT).parts) - 1


def _local_defs(path: Path) -> set[str]:
    tree = ast.parse(_read(path), filename=str(path))
    return {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}


@pytest.mark.parametrize("name", sorted(EXTRACTED))
def test_no_generator_redefines_an_extracted_helper(name: str) -> None:
    offenders = [
        str(p.relative_to(_SCRIPTS))
        for p in _script_files()
        if name in _local_defs(p)
    ]
    assert not offenders, (
        f"`def {name}` is back in {offenders}; it lives in "
        f"{EXTRACTED[name]} and a second copy will drift from it"
    )


@pytest.mark.parametrize("name", MIGRATED)
def test_migrated_generator_imports_from_common(name: str) -> None:
    path = _require(name)
    text = _read(path)
    assert "from scripts._common import " in text, (
        f"{name} must import its helper from scripts/_common, not define it"
    )


@pytest.mark.parametrize("name", MIGRATED)
def test_migrated_generator_bootstraps_the_repo_root(name: str) -> None:
    """``from scripts._common import ...`` fails on a bare ``python scripts/x.py``.

    Depth-aware: the expected ``parents[N]`` is derived from where the file
    actually lives, so a generator still works after ``scripts/`` is regrouped
    into domain subfolders. Pinning ``parents[1]`` here would have made every
    legitimate move a test failure and invited someone to delete the guard.
    """
    path = _require(name)
    expected = _repo_root_depth(path)
    text = _read(path)
    bootstrap = f"sys.path.insert(0, str(Path(__file__).resolve().parents[{expected}]))"
    assert bootstrap in text, (
        f"{name} imports scripts._common without putting the repo root on "
        f"sys.path; at {path.relative_to(_ROOT)} that means parents[{expected}]"
    )


def _require(name: str) -> Path:
    """Resolve a MIGRATED generator, failing loudly when it has gone missing.

    A generator that cannot be found must NEVER read as a pass -- that is how a
    rename silently drops a migration contract.
    """
    path = _find_script(name)
    assert path is not None, (
        f"{name} is not anywhere under scripts/; if it was renamed or deleted, "
        "update the MIGRATED contract instead of letting this scan pass"
    )
    return path


def test_scripts_is_an_importable_package() -> None:
    assert (_SCRIPTS / "__init__.py").is_file(), (
        "scripts/__init__.py is required for `from scripts._common import ...`"
    )


@pytest.mark.parametrize("name", MIGRATED)
def test_migrated_generator_import_actually_resolves(name: str) -> None:
    """Execute the generator's module scope in isolation, then stop.

    Deliberately NOT ``--help``: most of these generators have **no argparse at
    all**, so ``python scripts/<name>.py --help`` ignores the flag and *runs the
    whole report*, overwriting the committed artefacts under ``docs/``. Importing
    the module instead proves the same thing (the ``scripts._common`` import
    resolves) without any side effect.
    """
    import importlib.util

    path = _require(name)
    spec = importlib.util.spec_from_file_location(f"_probe_{path.stem}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)


#: Single-home modules: content lives under ``src/`` and must not be copied here.
#:
#: ``pyarrow_probe`` is the case that motivated this guard (issue #73). It is a
#: **library module with no CLI and no __main__**, used by ``gui/slm``'s
#: ``pattern_controls.py`` and covered by ``test_pattern_controls.py``. A copy
#: had also appeared under ``scripts/`` with **zero importers** -- it was pure
#: duplication, and nobody noticed for as long as it existed.
SINGLE_HOME_MODULES = ("pyarrow_probe",)


@pytest.mark.parametrize("name", SINGLE_HOME_MODULES)
def test_pyarrow_probe_has_a_single_home(name: str) -> None:
    """The canonical copy lives in the GUI package; ``scripts/`` must not grow one."""
    canonical = _ROOT / "src" / "ao_shaping" / "gui" / "slm" / f"{name}.py"
    assert canonical.is_file(), f"canonical {name} is missing from {canonical}"

    stray = _SCRIPTS / f"{name}.py"
    assert not stray.exists(), (
        f"{stray} is a duplicate of {canonical}: {name} is a library module with "
        "no CLI, it belongs to the package that uses it. See issue #73."
    )


def test_no_script_duplicates_a_package_module() -> None:
    """No file under ``scripts/`` may be a byte-for-byte copy of one under ``src/``.

    Compares content with line endings normalised, because a CRLF/LF difference
    is what let the ``pyarrow_probe`` twin look "drifted" in a hash comparison when
    it was in fact identical -- an audit that concludes "these differ, decide
    later" is how the duplicate survived. A genuine copy shows up here regardless
    of which line ending each side was checked out with.

    Deliberately content-based rather than name-based: the failure mode is a copy
    of *whatever* a library module, and a new one would not be in any allow-list.
    """
    import hashlib

    package_root = _ROOT / "src" / "ao_shaping"

    def digest(path: Path) -> str:
        raw = path.read_bytes().replace(b"\r\n", b"\n")
        return hashlib.sha256(raw).hexdigest()

    package = {
        digest(p): p.relative_to(_ROOT).as_posix()
        for p in package_root.rglob("*.py")
        if p.is_file()
    }

    duplicates = [
        (p.relative_to(_ROOT).as_posix(), package[digest(p)])
        for p in sorted(_SCRIPTS.rglob("*.py"))
        if p.is_file() and digest(p) in package
    ]
    assert not duplicates, (
        "these scripts/ files are exact copies of a package module and will "
        f"drift: {duplicates}"
    )