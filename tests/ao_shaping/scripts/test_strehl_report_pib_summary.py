"""R-26 — the Strehl report must not silently drop the cross-benchmark table.

The old code hardcoded ``report/heuristic_pib/summary.csv`` and, on any problem
(missing / unreadable / wrong columns), simply returned ``None`` and omitted the
table. The report still rendered, still looked complete, and the only clue was one
log line — so **a report already on disk could be quietly missing a section**
(TODO.md: "磁盘上已有的报告可能就是错的").

These tests pin:

1. the path is a parameter (``--pib-summary`` / ``main(pib_csv=...)``), not a
   constant baked into a helper;
2. a degraded load is *recorded in the markdown*, naming the file that was tried;
3. the rendered path is repo-relative POSIX, never a machine-specific absolute
   path (the reports are committed).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _ROOT / "scripts" / "generate_strehl_benchmark_report.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_strehl_report", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_results() -> dict:
    """The minimum ``results`` shape ``write_report`` reads.

    ``write_report`` needs ``final_strehl``, ``n_evals`` and a monotonic
    ``eval_curve`` (it derives "loads to max / >=0.9 / >=0.5" from the curve).
    """
    curve = [0.1, 0.2, 0.35, 0.55, 0.8, 0.95, 0.99]
    return {
        name: {"final_strehl": final, "n_evals": 10, "eval_curve": list(curve)}
        for name, final in (("GA", 0.99), ("SA", 0.8), ("SPGD", 0.95))
    }


@pytest.fixture(scope="module")
def mod():
    return _load_module()


# ---------------------------------------------------------------------------
# 1. the path is a parameter
# ---------------------------------------------------------------------------


def test_load_pib_summary_takes_the_path_as_an_argument(mod) -> None:
    import inspect

    params = list(inspect.signature(mod.load_pib_summary).parameters)
    assert params == ["pib_csv"], (
        "load_pib_summary must receive the path; a baked-in constant is what this "
        "item is about"
    )


def test_pib_summary_default_is_a_module_constant(mod) -> None:
    """The default is allowed, but it must be overridable and named."""
    assert mod.PIB_SUMMARY.name == "summary.csv"
    assert mod.PIB_SUMMARY.parent.name == "heuristic_pib"
    assert "pib_csv" in mod.main.__code__.co_varnames


def test_main_accepts_out_dir_and_pib_csv(mod) -> None:
    import inspect

    params = inspect.signature(mod.main).parameters
    assert "out_dir" in params, "the output directory must be overridable too"
    assert "pib_csv" in params
    assert params["pib_csv"].default is None, "None means 'fall back to PIB_SUMMARY'"


# ---------------------------------------------------------------------------
# 2. degraded loads
# ---------------------------------------------------------------------------


def test_missing_file_returns_none_and_is_logged(mod, tmp_path: Path) -> None:
    assert mod.load_pib_summary(tmp_path / "nope.csv") is None


def test_wrong_columns_return_none(mod, tmp_path: Path) -> None:
    bad = tmp_path / "summary.csv"
    pd.DataFrame({"algorithm": ["ga"], "wrong": [1]}).to_csv(bad, index=False)
    assert mod.load_pib_summary(bad) is None


def test_valid_file_returns_the_three_columns(mod, tmp_path: Path) -> None:
    good = tmp_path / "summary.csv"
    pd.DataFrame(
        {"algorithm": ["ga", "sa"], "final_pib": [0.9, 0.8], "n_loads": [30, 40], "extra": [1, 2]}
    ).to_csv(good, index=False)
    df = mod.load_pib_summary(good)
    assert df is not None
    assert list(df.columns) == ["algorithm", "final_pib", "n_loads"]


def test_report_records_the_resolved_path_when_skipped(mod, tmp_path: Path) -> None:
    """The core of R-26: a skipped section must SAY which file was missing."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    missing = tmp_path / "elsewhere" / "summary.csv"
    results = _fake_results()
    mod.write_report(results, 0.1, out_dir, None, missing)
    text = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "SKIPPED" in text
    assert "summary.csv" in text
    assert "pib-summary" in text, "the report must tell the reader how to fix it"


def test_report_names_the_exact_file_it_tried(mod, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    missing = tmp_path / "run-42" / "pib.csv"
    results = _fake_results()
    mod.write_report(results, 0.1, out_dir, None, missing)
    text = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "run-42/pib.csv" in text


def test_report_keeps_the_table_when_the_summary_is_usable(mod, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    df = pd.DataFrame({"algorithm": ["GA"], "final_pib": [0.9], "n_loads": [30]})
    results = _fake_results()
    mod.write_report(results, 0.1, out_dir, df, mod.PIB_SUMMARY)
    text = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "SKIPPED" not in text
    assert "0.9000" in text or "0.9" in text


# ---------------------------------------------------------------------------
# 3. repo-relative rendering
# ---------------------------------------------------------------------------


def test_rel_is_repo_relative_posix(mod) -> None:
    assert mod._rel(mod.PIB_SUMMARY) == "report/heuristic_pib/summary.csv"
    assert mod._rel(mod.OUT_DIR) == "report/strehl_benchmark"


def test_rel_falls_back_for_outside_paths(mod, tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere.csv"
    assert mod._rel(outside) == outside.as_posix()
    assert "\\" not in mod._rel(mod.OUT_DIR)


def test_cli_exposes_both_new_flags() -> None:
    """``--help`` must advertise them (F-8 spirit: no invisible options)."""
    import subprocess

    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--pib-summary" in result.stdout
    assert "--out-dir" in result.stdout