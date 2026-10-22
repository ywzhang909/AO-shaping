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
    "plot_summary_bars",
    "refuse_empty_overwrite",
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


def fmt_general(
    v: Any,
    spec: str = ".4g",
    *,
    missing_nonfinite: str | None = None,
    missing_none: str | None = None,
    nan_token: str | None = None,
    float_only: bool = False,
    bool_labels: tuple[str, str] | None = None,
    verbatim_int: bool = False,
) -> str:
    """Passthrough to Python's ``format`` with an explicit ``spec``.

    The ``generate_slm_pib_online_report.py`` formatter. Kept separate from
    :func:`fmt_metric` because it is genuinely different: ``.4g`` keeps four
    significant digits and prints ``1e+05`` rather than ``100000``, and it does
    **not** special-case ``None``/NaN (it renders them ``"None"`` / ``"nan"``).
    Folding it into :func:`fmt_metric` would silently rewrite committed reports.

    Non-numeric input falls back to ``str(v)`` rather than raising.

    ``missing_nonfinite`` (default ``None`` = today's behaviour) replaces only a
    **non-finite float** (NaN / ±inf) with the given token; ``None`` itself is
    still non-numeric and still falls through to ``str(v)`` → ``"None"``. That
    asymmetry is deliberate: it is the exact shape of
    ``scripts/slm/hw_differentiable_phase_opt.py``'s retired local formatter,
    whose saved reports already contain ``"None"`` cells, and the point of the
    extraction is byte-identical output, not a behaviour fix.

    ``missing_none`` and ``float_only`` (both default off = today's behaviour)
    exist for the third retired copy, ``scripts/slm/spgd_drift_cancel_loop.py``:

    * ``missing_none`` replaces ``None`` alone. With ``missing_nonfinite`` left
      unset, NaN/inf still print ``"nan"``/``"inf"`` — that report spells the
      sentinel out rather than rendering a "no data" cell.
    * ``float_only`` formats genuine ``float`` instances only and sends every
      other type through ``str(v)``, so a JSON-sourced integer renders ``3``, not
      ``3.000``. ``np.float64`` subclasses ``float`` and still formats;
      ``np.float32`` does not.
      * ``nan_token`` replaces NaN alone, before the spec is applied. With a sign
        flag (``"+.1f"``) ``format(nan, spec)`` renders ``"+nan"``, but that report
        wants ``"nan"``; ``inf`` still goes through the spec (``"+inf"``).
      * ``bool_labels`` and ``verbatim_int`` cover the fifth retired copy,
        ``scripts/slm/fit_forward_sampling_variants.py``, whose table cells are
        heterogeneous: ``bool_labels`` renders a flag as a language-specific pair
        (``"是"``/``"否"``) and ``verbatim_int`` prints an integer (``np.integer``
        included) as ``3`` instead of ``3.0000``.
      """
    if v is None and missing_none is not None:
        return missing_none
    if bool_labels is not None and isinstance(v, bool):
        return bool_labels[0] if v else bool_labels[1]
    if verbatim_int and isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return str(int(v))
    if float_only and not isinstance(v, float):
        return str(v)
    try:
        f = float(v)
        if nan_token is not None and np.isnan(f):
            return nan_token
        if missing_nonfinite is not None and not np.isfinite(f):
            return missing_nonfinite
        return format(f, spec)
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


def fmt_signed(v: float, spec: str = "+.4f", missing: str = "n/a") -> str:
    """Signed ``format`` spec (``+.4f`` by default); ``missing`` for unusable input.

    The ``generate_shape_objective_comparison.py`` formatter, used for the
    improvement columns where the ``+`` sign carries information.

    **Total by construction.** ``np.isfinite(None)`` raises ``TypeError``, and these
    generators feed it ``row.get(...)`` on cells that failed to train, so a bare
    ``isfinite`` guard turns a missing metric into a crash partway through writing a
    report. Non-numeric input is coerced, and anything that cannot yield a finite
    float renders as ``missing`` -- never as ``0.0000``, which would be
    indistinguishable from a real zero.
    """
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return missing if v is None else str(v)
    return missing if not np.isfinite(f) else format(f, spec)


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


def plot_summary_bars(
    results: dict[str, dict],
    out_path: Any,
    *,
    metric_key: str,
    xlabel: str,
    title: str,
) -> None:
    """Horizontal bar chart of one metric per algorithm, sorted descending.

    Extracted because ``generate_heuristic_pib_report`` and
    ``generate_strehl_benchmark_report`` carried the same block twice, differing only
    in ``metric_key``, ``xlabel`` and ``title``. Those three are the whole variation,
    so they became parameters here and every other constant is pinned to what both
    copies actually used -- ``figsize=(8, 5)``, ``color="steelblue"``, the value label
    at ``+0.01`` with three decimals, ``xlim=(0, 1.05)``, ``grid(axis="x",
    alpha=0.3)`` and ``dpi=150``. Verified byte-identical PNG output for both call
    sites, so no committed figure is rewritten.

    ``generate_slm_pib_rms_pib_report`` also defines a ``plot_summary_bars`` and is
    deliberately NOT routed through here: it scales figure height with the row count,
    derives a dynamic ``xlim`` from the data, uses dpi=110 without
    ``bbox_inches="tight"``, and returns the path instead of ``None``. Folding it in
    would force one contract onto both and rewrite a committed figure -- the same
    reason ``scripts/_common``'s five formatters are deliberately not merged.

    Args:
        results: ``{label: {metric_key: value}}``, sorted by that value descending.
        out_path: destination PNG path.
        metric_key: key inside each ``results`` entry holding the plotted value.
        xlabel: x-axis label.
        title: figure title.
    """
    import matplotlib.pyplot as plt

    from loguru import logger

    names = sorted(results, key=lambda n: results[n][metric_key], reverse=True)
    values = [results[n][metric_key] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(names, values, color="steelblue")
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel(xlabel)
    ax.set_xlim(0.0, 1.05)
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    savefig(fig, out_path, dpi=150)
    logger.info("Saved {}", out_path)


# ---------------------------------------------------------------------------
# Sweep output guard (issue #143 problem 2)
# ---------------------------------------------------------------------------
def refuse_empty_overwrite(out_path, results) -> bool:
    """Refuse to let an empty sweep result clobber a non-empty prior output.

    A crashed or aborted sweep must not overwrite a good prior result with ``{}``
    or a partial record (``report/loss_defects/padding_sweep.json`` was once
    clobbered exactly this way).

    Returns ``True`` when the write must be skipped (an empty result would
    overwrite an existing non-empty file); the caller should then refuse to write
    (``logger.warning`` / ``raise SystemExit``). Returns ``False`` otherwise:
    the result is non-empty (safe), or there is no existing file to protect
    (writing a fresh ``{}`` is fine), or the existing file is already empty.

    pure: performs no I/O beyond ``Path.stat`` on the single candidate path, so
    it is trivially unit-testable without running a sweep or loading heavy
    numerical dependencies.
    """
    from pathlib import Path

    if results not in (None, "", {}, [], ()):
        return False
    p = Path(out_path)
    try:
        if not p.exists() or p.stat().st_size <= 0:
            return False
    except OSError:
        return False
    return True
