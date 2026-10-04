"""Shared helpers for the ``generate_*_report.py`` family.

Every helper here was duplicated across report generators. They were extracted
(TODO.md R-25) so that a fix lands once instead of N times — which had already
happened: two copies of ``_fmt`` disagreed, and one of them rendered ``1e-7`` as
``0.0000``, silently flattening a real measurement to zero in a committed report.

**Behaviour contract** (``tests/ao_shaping/scripts/test_common_helpers.py`` pins
every branch):

* :func:`fmt_metric` uses the **gsnet_offline** behaviour. Where the two old
  copies differed, gsnet wins — a tiny non-zero value must read as ``1.000e-07``,
  never ``0.0000``. This is a deliberate **fix**, not a refactor.
* :func:`fmt_ratio` is a *different* formatter (the OOPAO pair's), kept separate on
  purpose: it switches to scientific notation at ``1e-3``/``1e5`` and defaults to 6
  digits, which :func:`fmt_metric` does not. Merging them would change output.
* :func:`iters_to_threshold` / :func:`format_iters` are parameter-name-only
  variants of each other; the bodies were already identical.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

__all__ = [
    "fmt_fixed",
    "fmt_general",
    "fmt_metric",
    "fmt_ratio",
    "fmt_signed",
    "format_iters",
    "iters_to_threshold",
    "markdown_table",
    "savefig",
]


# ---------------------------------------------------------------------------
# Metric formatting
# ---------------------------------------------------------------------------


def fmt_metric(v: Any, nd: int = 4) -> str:
    """Format a metric for markdown: ints/large magnitudes compact, else fixed.

    ``None``, NaN and infinity render as ``-`` (the markdown "no data" cell), and
    non-numeric input passes through as ``str(v)`` rather than raising — a report
    generator must not die on one odd cell.

    The ``|v| < 1e-4`` branch is the whole point of this extraction: the previous
    copy in ``generate_fouriergsnet_sim_report.py`` had no such branch and printed
    ``0.0000`` for ``1e-7``.
    """
    if v is None:
        return "-"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if np.isnan(f) or np.isinf(f):
        return "-"
    if f.is_integer() and abs(f) >= 10.0:
        return f"{f:.0f}"
    if f != 0 and abs(f) < 1e-4:
        return f"{f:.3e}"
    if abs(f) >= 1e5:
        return f"{f:.3e}"
    return f"{f:.{nd}f}"


def fmt_ratio(value: float, digits: int = 6) -> str:
    """Format a ratio for the OOPAO reports (0 reads as ``0``).

    Deliberately NOT the same function as :func:`fmt_metric`: this one goes
    scientific below ``1e-3`` / at or above ``1e5`` and uses ``g`` formatting.
    """
    if value == 0.0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e5:
        return f"{value:.{digits}e}"
    return f"{value:.{digits}g}"


def fmt_general(v: Any, spec: str = ".4g") -> str:
    """Passthrough to Python's ``format`` with an explicit ``spec``.

    The ``generate_slm_pib_online_report.py`` formatter. Kept separate from
    :func:`fmt_metric` because it is genuinely different: ``.4g`` keeps four
    significant digits and prints ``1e+05`` rather than ``100000``, and it does
    **not** special-case ``None``/NaN (it renders them ``"None"`` / ``"nan"``).
    Folding it into :func:`fmt_metric` would silently rewrite committed reports.

    Non-numeric input falls back to ``str(v)`` rather than raising.
    """
    try:
        return format(float(v), spec)
    except (TypeError, ValueError):
        return str(v)


def fmt_fixed(v: float, nd: int = 4) -> str:
    """Fixed-point with ``nd`` decimals; an em dash when ``v`` is NaN.

    The ``generate_slm_pib_rms_pib_report.py`` formatter. Note it does **not**
    accept ``None`` (it raises ``TypeError`` on ``v == v``) and renders ``inf``
    as ``"inf"``; :func:`fmt_metric` handles both. Kept separate so extraction
    stays behaviour-preserving — see the module docstring.
    """
    return f"{v:.{nd}f}" if v == v else "—"


def fmt_signed(v: float, spec: str = "+.4f") -> str:
    """Signed ``format`` spec (``+.4f`` by default); ``"n/a"`` for non-finite.

    The ``generate_shape_objective_comparison.py`` formatter, used for the
    improvement columns where the ``+`` sign carries information.
    """
    return "n/a" if not np.isfinite(v) else format(v, spec)


# ---------------------------------------------------------------------------
# Convergence-curve helpers
# ---------------------------------------------------------------------------


def iters_to_threshold(curve: np.ndarray, threshold: float) -> int | None:
    """First **1-based** iteration where ``curve`` reaches ``threshold``, else None.

    ``- 1e-12`` absorbs the float noise that makes an "exactly at threshold"
    point compare False.
    """
    reached = np.flatnonzero(np.asarray(curve) >= threshold - 1e-12)
    return int(reached[0]) + 1 if reached.size else None


def format_iters(v: int | None) -> str:
    """Format an iteration count, or an em dash when the threshold was never reached."""
    return str(v) if v is not None else "—"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def savefig(fig: Any, path: Any, dpi: int = 150) -> None:
    """Save a matplotlib figure at the repo-standard DPI and close it.

    Every report generator used ``bbox_inches="tight"``; the DPI differed per
    script, so it is a parameter with the most common value as the default.
    """
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def markdown_table(headers: Sequence[Any], rows: Sequence[Sequence[Any]]) -> str:
    """Render a GitHub-flavoured markdown table from an iterable of rows."""
    lines = ["| " + " | ".join(str(h) for h in headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)