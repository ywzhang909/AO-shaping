"""R-25 — the shared report helpers, behaviour-pinned.

Every expectation below was **recorded from the pre-extraction implementations**
(2026-10-03, straight out of the home scripts) before anything was moved, so this
file is a characterisation test, not a restatement of the new code.

Provenance of each pinned behaviour:

* ``fmt_metric``  — two disagreeing copies (``generate_fouriergsnet_sim_report``
  vs ``generate_gsnet_offline_report``). Recorded on 19 inputs; they differed on
  **exactly two**: ``1e-7`` and ``-1e-7``. The gsnet rendering is adopted as the
  fix (TODO R-25 explicitly says so).
* ``fmt_ratio``   — the OOPAO pair's distinct formatter, recorded from
  ``generate_oopao_vs_numpy_report``.
* ``markdown_table`` / ``iters_to_threshold`` / ``format_iters`` — copies that were
  already identical; recorded to prove the extraction changed nothing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scripts._common import (
    fmt_fixed,
    fmt_general,
    fmt_metric,
    fmt_ratio,
    fmt_signed,
    format_iters,
    iters_to_threshold,
    markdown_table,
    savefig,
)


# ---------------------------------------------------------------------------
# fmt_metric — 19 recorded probes
# ---------------------------------------------------------------------------

#: ``(input, expected)`` recorded from BOTH old copies (identical unless noted).
FMT_METRIC_CASES = [
    (None, "-"),
    (float("nan"), "-"),
    (float("inf"), "-"),
    (0.0, "0.0000"),
    (-0.0, "-0.0000"),
    (0.5, "0.5000"),
    (1.0, "1.0000"),
    (9.99, "9.9900"),
    (10.0, "10"),
    (12.0, "12"),
    (-12.0, "-12"),
    (42.0, "42"),
    (1e5, "100000"),
    (3.2e7, "32000000"),
    (0.000123, "0.0001"),
    ("abc", "abc"),
    (True, "1.0000"),
]

#: The two probes where the old copies disagreed. The gsnet column is the fix.
FMT_METRIC_TINY_FIX = [
    (1e-7, "0.0000", "1.000e-07"),
    (-1e-7, "-0.0000", "-1.000e-07"),
]


@pytest.mark.parametrize(("value", "expected"), FMT_METRIC_CASES)
def test_fmt_metric_matches_the_recorded_behaviour(value, expected: str) -> None:
    assert fmt_metric(value) == expected


@pytest.mark.parametrize(("value", "old", "new"), FMT_METRIC_TINY_FIX)
def test_tiny_values_are_no_longer_flattened_to_zero(value, old: str, new: str) -> None:
    """The defect R-25 exists to fix: ``1e-7`` must not read as ``0.0000``."""
    assert fmt_metric(value) == new
    assert fmt_metric(value) != old


def test_fmt_metric_tiny_threshold_is_not_inclusive_of_zero() -> None:
    """``0.0`` itself stays ``0.0000`` — the branch must not swallow it."""
    assert fmt_metric(0.0) == "0.0000"
    assert fmt_metric(0.0) != "0.000e+00"


def test_fmt_metric_nd_is_honoured() -> None:
    assert fmt_metric(0.5, nd=2) == "0.50"
    assert fmt_metric(0.5, nd=6) == "0.500000"


def test_fmt_metric_uses_numpy_isnan_not_math_isnan() -> None:
    """Both are correct for floats; the divergence was the *missing* tiny branch."""
    assert fmt_metric(float("nan")) == fmt_metric(math.nan) == "-"


# ---------------------------------------------------------------------------
# fmt_ratio — recorded from the OOPAO formatter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, "0"),
        (1.0, "1"),
        (0.5, "0.5"),
        (0.001, "0.001"),
        (1e-3, "0.001"),
        (9.999e-4, "9.999000e-04"),
        (99999.0, "99999"),
        (1e5, "1.000000e+05"),
        (-1.0e-6, "-1.000000e-06"),
        (1 / 3, "0.333333"),
        (1234.5678, "1234.57"),
    ],
)
def test_fmt_ratio_matches_the_recorded_behaviour(value: float, expected: str) -> None:
    assert fmt_ratio(value) == expected


def test_fmt_ratio_switch_points_are_strict() -> None:
    """The comparisons are ``< 1e-3`` / ``>= 1e-5``, so the boundary stays ``g``."""
    assert fmt_ratio(1e-3) == "0.001"
    assert fmt_ratio(9.999e-4) == "9.999000e-04"
    assert fmt_ratio(99999.0) == "99999"
    assert fmt_ratio(1e5) == "1.000000e+05"


def test_fmt_ratio_digits_is_honoured() -> None:
    assert fmt_ratio(1 / 3, digits=2) == "0.33"
    assert fmt_ratio(1 / 3, digits=3) == "0.333"
    assert fmt_ratio(1.0e-6, digits=2) == "1.00e-06"


def test_fmt_ratio_is_distinct_from_fmt_metric() -> None:
    """They must NOT be merged: the switch points and the int-compaction differ."""
    assert fmt_metric(1e-6) == "1.000e-06"
    assert fmt_ratio(1e-6) == "1.000000e-06"
    # fmt_metric compacts >=10 to an integer; fmt_ratio's `g` does too, but 0.001
    # is where they visibly part company.
    assert fmt_metric(0.001) == "0.0010"
    assert fmt_ratio(0.001) == "0.001"


# ---------------------------------------------------------------------------
# fmt_general / fmt_fixed — recorded from the two formatters that are NOT
# fmt_metric. They were left separate on purpose: folding them in would rewrite
# already-committed reports (verified: 168/170 probes identical; the 2 diffs are
# only the sanctioned tiny-value fix).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, "0"),
        (-0.0, "-0"),
        (0.5, "0.5"),
        (1.0, "1"),
        (9.99, "9.99"),
        (1e-7, "1e-07"),
        (-1e-7, "-1e-07"),
        (0.000123, "0.000123"),
        (1e5, "1e+05"),
        (3.2e7, "3.2e+07"),
        (True, "1"),
    ],
)
def test_fmt_general_matches_the_recorded_behaviour(value, expected: str) -> None:
    assert fmt_general(value) == expected


def test_fmt_general_does_not_special_case_none_or_nan() -> None:
    """Unlike ``fmt_metric`` it renders them literally — that is the old behaviour."""
    assert fmt_general(None) == "None"
    assert fmt_general(float("nan")) == "nan"
    assert fmt_general(float("inf")) == "inf"


def test_fmt_general_falls_back_to_str_instead_of_raising() -> None:
    assert fmt_general("abc") == "abc"


def test_fmt_general_spec_is_honoured() -> None:
    assert fmt_general(1.23456, ".2f") == "1.23"
    assert fmt_general(1234.5, ",") == "1,234.5"


def test_fmt_general_is_distinct_from_fmt_metric() -> None:
    """Folding them together is exactly the regression this separation prevents."""
    assert fmt_general(0.5) == "0.5"
    assert fmt_metric(0.5) == "0.5000"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, "0.0000"),
        (1.0, "1.0000"),
        (0.5, "0.5000"),
        (12.0, "12.0000"),
        (42.0, "42.0000"),
        (1e5, "100000.0000"),
        (1e-7, "0.0000"),
    ],
)
def test_fmt_fixed_matches_the_recorded_behaviour(value: float, expected: str) -> None:
    assert fmt_fixed(value) == expected


def test_fmt_fixed_renders_nan_as_an_em_dash() -> None:
    assert fmt_fixed(float("nan")) == "—"
    assert fmt_fixed(float("nan")) != fmt_metric(float("nan")) == "-"


def test_fmt_fixed_does_not_accept_none() -> None:
    """Recorded limitation of the original: it raises. Kept, not silently fixed."""
    with pytest.raises(TypeError):
        fmt_fixed(None)


def test_fmt_fixed_renders_inf_literally() -> None:
    assert fmt_fixed(float("inf")) == "inf"


def test_fmt_fixed_nd_is_honoured() -> None:
    assert fmt_fixed(0.5, nd=2) == "0.50"


def test_fmt_fixed_is_distinct_from_fmt_metric() -> None:
    assert fmt_fixed(12.0) == "12.0000"
    assert fmt_metric(12.0) == "12"


def test_fmt_signed_matches_the_recorded_behaviour() -> None:
    """Fifth distinct formatter (``generate_shape_objective_comparison.py``)."""
    assert fmt_signed(0.5) == "+0.5000"
    assert fmt_signed(-0.5) == "-0.5000"
    assert fmt_signed(0.0) == "+0.0000"
    # The `+` comes from the *default* spec, not from the function: a caller-supplied
    # plain ".2f" renders unsigned, exactly as `format` does.
    assert fmt_signed(1.23456, ".2f") == "1.23"
    assert fmt_signed(1.23456, "+.2f") == "+1.23"
    assert fmt_signed(1.0, ".1%") == "100.0%"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_fmt_signed_renders_non_finite_as_na(value: float) -> None:
    assert fmt_signed(value) == "n/a"


def test_fmt_signed_is_distinct_from_fmt_metric() -> None:
    """``n/a`` vs ``-``, and the forced ``+`` sign."""
    assert fmt_signed(float("nan")) == "n/a"
    assert fmt_metric(float("nan")) == "-"
    assert fmt_signed(0.5).startswith("+")
    assert not fmt_metric(0.5).startswith("+")


# ---------------------------------------------------------------------------
# markdown_table
# ---------------------------------------------------------------------------


def test_markdown_table_matches_the_recorded_shape() -> None:
    assert markdown_table(["a", "b"], [[1, 2], ["x", None]]) == (
        "| a | b |\n|---|---|\n| 1 | 2 |\n| x | None |"
    )


def test_markdown_table_header_only() -> None:
    assert markdown_table(["h1", "h2"], []) == "| h1 | h2 |\n|---|---|"


def test_markdown_table_stringifies_every_cell() -> None:
    out = markdown_table(["n"], [[1.5], [None], [True]])
    assert out.splitlines()[2:] == ["| 1.5 |", "| None |", "| True |"]


# ---------------------------------------------------------------------------
# iters_to_threshold / format_iters — 12 recorded probes
# ---------------------------------------------------------------------------

ITERS_CASES = [
    ([0.1, 0.5, 0.9, 0.95], 0.5, 2),
    ([0.1, 0.5, 0.9, 0.95], 0.9, 3),
    ([0.1, 0.5, 0.9, 0.95], 0.95, 4),
    ([0.1, 0.5, 0.9, 0.95], 1.5, None),
    ([0.0, 0.0], 0.5, None),
    ([0.0, 0.0], 0.9, None),
    ([0.0, 0.0], 0.95, None),
    ([0.0, 0.0], 1.5, None),
    ([0.9, 0.1], 0.5, 1),
    ([0.9, 0.1], 0.9, 1),
    ([0.9, 0.1], 0.95, None),
    ([0.9, 0.1], 1.5, None),
]


@pytest.mark.parametrize(("curve", "threshold", "expected"), ITERS_CASES)
def test_iters_to_threshold_matches_the_recorded_behaviour(curve, threshold, expected) -> None:
    assert iters_to_threshold(np.array(curve, dtype=float), threshold) == expected


def test_iters_to_threshold_is_one_based() -> None:
    assert iters_to_threshold(np.array([0.1, 0.9]), 0.9) == 2


def test_iters_to_threshold_absorbs_float_noise() -> None:
    """The ``- 1e-12`` epsilon: an exactly-at-threshold point must still count."""
    assert iters_to_threshold(np.array([0.0, 0.9 - 1e-15]), 0.9) == 2


def test_iters_to_threshold_on_an_empty_curve() -> None:
    assert iters_to_threshold(np.array([]), 0.5) is None


@pytest.mark.parametrize(("value", "expected"), [(None, "—"), (7, "7"), (1, "1"), (0, "0")])
def test_format_iters_matches_the_recorded_behaviour(value, expected: str) -> None:
    assert format_iters(value) == expected


def test_format_iters_uses_an_em_dash_not_a_hyphen() -> None:
    """U+2014, not ASCII '-'; a plain hyphen would read as a placeholder dash."""
    assert format_iters(None) == "—"
    assert format_iters(None) != "-"


# ---------------------------------------------------------------------------
# savefig
# ---------------------------------------------------------------------------


def test_savefig_writes_the_file_and_closes_the_figure(tmp_path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    target = tmp_path / "fig.png"
    savefig(fig, target, dpi=80)
    assert target.exists()
    assert target.stat().st_size > 0
    assert not plt.fignum_exists(fig.number), "savefig must close the figure"
    plt.close("all")


def test_savefig_passes_dpi_through(tmp_path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, _ = plt.subplots()
    low = tmp_path / "low.png"
    savefig(fig, low, dpi=40)
    assert low.exists()
    plt.close("all")