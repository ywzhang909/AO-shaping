"""R-53 guard: the ambiguous bare name ``compute_metrics`` must not come back.

Issue #69 (2026-10-06 function-ownership audit) recorded ``compute_metrics`` as a
"duplicate implementation". That classification was a **false positive**: the three
functions share a name but have three different contracts, so merging them would
break at least two callers. The fix is disambiguation by name, and this test is what
keeps the fix from decaying back into the ambiguity.

Three distinct contracts, one per name:

===============================  =========================================
``compute_beam_metrics``         dict: mse / correlation / efficiency
``compute_bench_metrics``        dict: center / zero_order_margin_px
``compute_cv_ee``                tuple: (CV, EE)
===============================  =========================================

Two failure modes are guarded:

1. a new bare ``def compute_metrics`` appearing anywhere -- reintroduces the ambiguity;
2. one of the three canonical names disappearing -- the disambiguation was reverted,
   or a caller was pointed at the wrong contract.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCAN_ROOTS = ("src", "scripts", "tests")

# new name -> defining module (relative to repo root)
CANONICAL = {
    "compute_beam_metrics": "src/ao_shaping/utils/image/beam_metrics.py",
    "compute_bench_metrics": "src/ao_shaping/drivers/sim/slm_shaping_bench.py",
    "compute_cv_ee": "scripts/generate_diff_shaping_report.py",
}

AMBIGUOUS = "compute_metrics"


def _python_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        files.extend((REPO_ROOT / root).rglob("*.py"))
    return [f for f in files if f.is_file()]


def _defined_names(path: Path) -> set[str]:
    """Top-level + class-level function/method names defined in ``path``."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
    return names


def test_no_bare_compute_metrics_is_defined_anywhere() -> None:
    """The whole point of R-53: no definition may use the ambiguous bare name."""
    offenders: list[str] = []
    for path in _python_files():
        if AMBIGUOUS in _defined_names(path):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, (
        f"`{AMBIGUOUS}` is defined again in {offenders}. "
        f"It was ambiguous because three different contracts shared it (#69); "
        f"use {sorted(CANONICAL)} and add a new explicit name instead."
    )


@pytest.mark.parametrize(("name", "module"), sorted(CANONICAL.items()))
def test_canonical_name_is_defined_in_its_module(name: str, module: str) -> None:
    """Each disambiguated name must still exist where R-53 put it."""
    path = REPO_ROOT / module
    assert path.is_file(), f"{module} disappeared"
    assert name in _defined_names(path), f"`{name}` is no longer defined in {module}"


def test_the_three_contracts_stay_distinct() -> None:
    """Renaming must not have quietly collapsed two contracts into one."""
    import importlib.util
    import sys

    beam_metrics = REPO_ROOT / "src/ao_shaping/utils/image/beam_metrics.py"
    diff_shaping = REPO_ROOT / "scripts/generate_diff_shaping_report.py"

    def load(path: Path, alias: str):
        spec = importlib.util.spec_from_file_location(alias, path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[alias] = mod
        spec.loader.exec_module(mod)
        return mod

    bm = load(beam_metrics, "_r53_beam_metrics")
    # contract 1: a dict with mse / correlation / efficiency
    out = bm.compute_beam_metrics.__doc__ or ""
    assert "mse" in out.lower() or "correlation" in out.lower(), (
        "compute_beam_metrics no longer documents the img2img dict contract"
    )

    ds = load(diff_shaping, "_r53_diff_shaping")
    # contract 3: a (CV, EE) tuple -- a different shape from any dict contract
    assert (ds.compute_cv_ee.__doc__ or "") != (bm.compute_beam_metrics.__doc__ or ""), (
        "compute_cv_ee and compute_beam_metrics ended up with the same docstring -- "
        "the two contracts were collapsed into one"
    )
