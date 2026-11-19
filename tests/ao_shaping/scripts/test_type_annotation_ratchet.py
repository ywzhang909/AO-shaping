"""Ratchet for the type-annotation census and the coverage toolchain (#178).

Two half-promises of issue #178 were "annotation coverage is only ~62.6%" and
"there is no coverage toolchain". Both were unfalsifiable impressions; this
file makes them measured and one-way:

* the AST census (``scripts/_common/type_annotation_census.py``) pins a global
  floor; new code may raise it, silent drift down fails CI;
* the coverage sections in ``pyproject.toml`` (``pytest-cov`` in the dev group,
  ``[tool.coverage.*]``) must stay declared -- a cleanup that "tidies" them
  away would quietly remove the toolchain again.

Deliberately NOT a coverage-percentage gate: per TODO.md §6 decision B (#132)
the blocking gates stay ruff + pytest shards + convention guards; this ratchet
protects *annotation* progress and the *existence* of the measuring apparatus.
"""

from __future__ import annotations

import importlib.util
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CENSUS = ROOT / "scripts" / "_common" / "type_annotation_census.py"
BASELINE = ROOT / "scripts" / "_common" / "type_annotation_baseline.json"

#: Same tolerance the CLI ``--check`` uses, kept in one place.
TOLERANCE_PP = 0.2


def _load_census_module():
    spec = importlib.util.spec_from_file_location("type_annotation_census", CENSUS)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_census_runs_and_is_not_vacuous() -> None:
    """A guard that measures nothing is worse than no guard (repo culture)."""
    result = _load_census_module().census()
    assert result["defs"] > 5000, "census tree scan collapsed"
    assert 0.0 < result["pct_annotated"] < 100.0, result["pct_annotated"]
    # Both slot kinds must actually be counted, or the ratio means nothing.
    assert result["params_total"] > 5000 and result["returns_annotated"] > 1000


def test_annotation_coverage_has_not_regressed() -> None:
    """Floor is the *pinned baseline*, not the live number (one-way ratchet)."""
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    result = _load_census_module().census()
    assert result["pct_annotated"] >= baseline["pct_annotated"] - TOLERANCE_PP, (
        f"annotation coverage {result['pct_annotated']}% fell below the pinned "
        f"floor {baseline['pct_annotated']}% (tol {TOLERANCE_PP}). Annotate the "
        "new code, or -- if a module was genuinely deleted -- re-pin with "
        "`python3 scripts/_common/type_annotation_census.py --update-baseline` "
        "and justify it in the commit."
    )


def test_cli_check_mode_agrees_with_the_guard() -> None:
    """The script and the test must not drift into two different verdicts."""
    assert _load_census_module().main(["--check"]) == 0


def test_coverage_toolchain_is_declared() -> None:
    """#178's toolchain: pytest-cov in the dev group + [tool.coverage] config."""
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dev = cfg["dependency-groups"]["dev"]
    assert any("pytest-cov" in dep for dep in dev), dev
    run = cfg["tool"]["coverage"]["run"]
    assert run["source"] == ["src/ao_shaping"], run
    assert run["branch"] is True and run["relative_files"] is True, run
    # TYPE_CHECKING / abstract stubs must stay excluded, otherwise the first
    # real measurement buries signal under 400 impossible lines.
    excludes = cfg["tool"]["coverage"]["report"]["exclude_also"]
    assert any("TYPE_CHECKING" in x for x in excludes), excludes
    assert any("abstractmethod" in x for x in excludes), excludes
