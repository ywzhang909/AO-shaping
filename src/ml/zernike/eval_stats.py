"""Paired-comparison statistics and canary-subtracted skill for offline ML evaluation.

Every function here is **pure** (numpy only, no I/O, no hardware) and is the single
source of truth for the numbers that the Zernike reports quote. ``scripts/`` owns the
plotting and the markdown; this module owns the arithmetic, so a second report cannot
quietly disagree with the first.

The repo has neither scipy nor sklearn, so the permutation test is written out
exactly rather than delegated.

**The trap this module exists to prevent.** A naive paired test over grouped CV folds
looks rigorous and can be badly misleading, because the two axes people want to
separate are frequently *confounded*:

- **fold** is the data axis -- which samples are held out;
- **seed** is the initialisation axis -- which weight draw is used.

Only the *pairing* across folds kills the fold-difficulty term. But if every fold also
shares one ``seed``, then all folds share **one** weight initialisation, so N folds
measure *that single init pair* N times. They are not N independent samples of
"config A vs config B", and a sign-flip test will happily treat them as if they were,
returning an **anti-conservative** p-value. See :func:`seed_agreement` for the control
that detects this, and ``report/zernike_coeff2amp/report.md`` §"死杠杆" for a worked
case where ``p = 0.0144`` came out of exactly this.

The general rule: **a claim needs the axis you varied to be the axis you paired on.**
Vary seeds, pair on seeds. Vary folds, pair on folds. Never 18 folds at one seed and
call it 18 samples.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "cohens_dz",
    "folds_of",
    "holm_bonferroni",
    "paired_comparison",
    "pearson",
    "ridge",
    "seed_agreement",
    "sign_flip_pvalue",
    "skill_scores",
    "spearman",
]


def _as_float_pair(x, y) -> tuple[np.ndarray, np.ndarray]:
    """Validate two inputs as equal-length 1-D float64 vectors (copies, no aliasing).

    Raises ``ValueError`` on non-1D input or a length mismatch -- the same contract
    :func:`pearson` and :func:`spearman` document.
    """
    xa = np.array(x, dtype=np.float64, ndmin=0, copy=True)
    ya = np.array(y, dtype=np.float64, ndmin=0, copy=True)
    if xa.ndim != 1 or ya.ndim != 1:
        raise ValueError(
            f"expected 1-D vectors, got shapes {xa.shape} and {ya.shape}"
        )
    if xa.size != ya.size:
        raise ValueError(f"length mismatch: {xa.size} vs {ya.size}")
    return xa, ya


def _average_rank(values: np.ndarray) -> np.ndarray:
    """Average ranks of ``values`` (ties share the mean of their rank span).

    Matches the default ranking of ``scipy.stats.rankdata``: for a group of ``c``
    equal values occupying sorted positions ``p+1 .. p+c``, every member gets
    ``p + (c + 1) / 2``. Implemented with ``np.unique`` (sorted) plus cumulative
    counts -- no scipy, no pandas.
    """
    uniques, counts = np.unique(values, return_counts=True)
    offsets = np.concatenate(([0], np.cumsum(counts)))  # first sorted position per group
    group_rank = (offsets[:-1] + offsets[1:] + 1) / 2.0  # mean of (p+1 .. p+c)
    return group_rank[np.searchsorted(uniques, values)]


def pearson(x, y) -> float:
    """Pearson product-moment correlation of two 1-D vectors (numpy only).

    Contract: centre each vector, divide by the product of their L2 norms, return a
    float64 scalar. Inputs are never mutated (converted to float64 copies).

    - ``ValueError`` on length mismatch or non-1D input.
    - ``nan`` when ``n < 2`` (nothing to correlate) and when either vector has zero
      variance (a constant vector has no correlation *direction*). The degenerate
      return is ``nan``, **not 0.0** -- matching modern ``scipy.stats.pearsonr``,
      which likewise returns ``nan`` for constant input rather than inventing a
      zero correlation.
    - ``nan`` propagates: any non-finite entry makes the result ``nan`` rather than
      silently dropping the pair (unlike :func:`sign_flip_pvalue`, NaNs here are
      data errors, not missing folds).

    Deliberately not ``np.corrcoef``: that returns a 2x2 matrix and emits
    RuntimeWarnings (with ``nan`` off the diagonal) that callers routinely ignore.
    """
    xa, ya = _as_float_pair(x, y)
    if xa.size < 2:
        return float("nan")
    xc = xa - xa.mean()
    yc = ya - ya.mean()
    norm = float(np.linalg.norm(xc) * np.linalg.norm(yc))
    if norm == 0.0:
        return float("nan")  # zero variance in at least one vector
    return float(xc @ yc / norm)


def spearman(x, y) -> float:
    """Spearman rank correlation: Pearson on average ranks (ties get mean ranks).

    Matches ``scipy.stats.spearmanr`` default (average-rank ties, no tie correction
    debate -- the two agree because both reduce to Pearson over the same ranked
    vectors). Equivalent by construction to ``pearson(_average_rank(x),
    _average_rank(y))``.

    Same validation contract as :func:`pearson`: ``ValueError`` on length mismatch or
    non-1D input; ``nan`` when ``n < 2``; ``nan`` when either vector is constant
    (its ranks have zero variance) -- again ``nan``, not 0.0. Non-finite entries
    (``nan``/``inf``) sort into the ranking and corrupt it, so the result is
    undefined there; callers must pre-filter, as the Zernike eval scripts already do
    for their metric lists.
    """
    xa, ya = _as_float_pair(x, y)
    if xa.size < 2:
        return float("nan")
    return pearson(_average_rank(xa), _average_rank(ya))


def folds_of(records):
    """Leave-one-group-out folds over records keyed by ``str(record.path)``.

    Duck-typed on ``record.path`` (``str`` or ``Path`` both group to the same key);
    no import of the record type, so this module stays stdlib + numpy. Every
    distinct path is held out exactly once. Fold order is ``sorted`` key order --
    never insertion order -- so two runs over the same records get identical folds.

    Returns ``(index, key, train_positions, val_positions)`` where the position
    lists index into ``records``, ``train_positions`` concatenate in group-first
    order, and ``train + val`` is a permutation of ``range(len(records))``.
    Scripts whose loops consume a projection of this tuple (e.g. dropping ``key``)
    adapt at the call site; the canonical always carries both index and key.
    """
    groups: dict[str, list[int]] = {}
    for position, record in enumerate(records):
        groups.setdefault(str(record.path), []).append(position)
    keys = sorted(groups)
    return [(i, k, [p for kk, ps in groups.items() if kk != k for p in ps], groups[k])
            for i, k in enumerate(keys)]


def ridge(x: np.ndarray, z: np.ndarray, alpha: float = 1.0):
    """Dual-form multi-output ridge with an intercept: ``(weights, x_mean, z_mean)``.

    ``x`` is (n_samples, n_features), ``z`` is (n_samples, n_targets); the solve is
    on the n x n Gram, so it stays cheap when n << n_features. Prediction is
    ``(x_new - x_mean) @ weights + z_mean``. Inputs are never mutated.

    Degenerate inputs keep ``np.linalg.solve`` semantics rather than smoothing
    them: ``alpha == 0`` with a rank-deficient (or single-row) centred ``x`` is
    singular and raises ``LinAlgError`` -- the behaviour every migrated copy had,
    and silently returning a pseudo-inverse solution would hide a broken fold.
    """
    xm, zm = x.mean(0, keepdims=True), z.mean(0, keepdims=True)
    xc, zc = x - xm, z - zm
    g = xc @ xc.T
    g.flat[:: g.shape[0] + 1] += alpha
    return xc.T @ np.linalg.solve(g, zc), xm, zm


def sign_flip_pvalue(diffs: Sequence[float]) -> float:
    """Exact two-sided sign-flip permutation p-value on paired differences.

    Enumerates all ``2**n`` sign assignments of the observed differences and asks how
    often the mean of a random sign assignment is at least as extreme in magnitude as
    the observed one. No normality assumption, which matters because fold differences
    are routinely heavy-tailed.

    NaNs are dropped (an unusable fold is missing data, not a zero difference).

    The minimum attainable p-value is ``2 / 2**n``: ~7.6e-6 at 18 folds, but **0.125 at
    4 folds**. A 4-fold protocol therefore *structurally* cannot reach p < 0.05, no
    matter how large the effect -- report effect sizes there instead of a verdict.
    """
    clean = [d for d in diffs if not math.isnan(d)]
    n = len(clean)
    if n == 0:
        return float("nan")
    observed = abs(float(np.mean(clean)))
    total = 0
    extreme = 0
    for mask in range(1 << n):
        total += 1
        acc = 0.0
        for i in range(n):
            acc += clean[i] if (mask >> i) & 1 else -clean[i]
        if abs(acc / n) >= observed - 1e-15:
            extreme += 1
    return extreme / total


def min_attainable_pvalue(n_pairs: int) -> float:
    """Smallest p-value :func:`sign_flip_pvalue` can return for ``n_pairs`` pairs."""
    if n_pairs <= 0:
        return float("nan")
    return 2.0 / (2**n_pairs)


def cohens_dz(diffs: Sequence[float]) -> float:
    """Cohen's ``d_z`` = ``mean(diff) / sd(diff)`` for paired samples.

    The paired effect size to quote alongside a p-value. Returns ``+inf`` when the
    differences have no spread but a non-zero mean (a claim no replicate can support),
    and NaN when there is nothing to compare.

    The zero-spread test is **relative**, not ``sd <= 0.0``: floating-point noise makes
    ``std([0.1, 0.1, 0.1], ddof=1)`` come out as ~1.7e-17 rather than 0.0, so an exact
    comparison silently returns a ~6e15 "effect size" instead of ``inf``.
    """
    clean = np.asarray([d for d in diffs if not math.isnan(d)], dtype=np.float64)
    if clean.size < 2:
        return float("nan")
    sd = float(clean.std(ddof=1))
    mean = float(clean.mean())
    scale = max(abs(mean), float(np.abs(clean).max()))
    if sd <= 1e-12 * scale:
        return float("inf") if mean != 0.0 else float("nan")
    return float(mean / sd)


def holm_bonferroni(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values, monotone-enforced.

    Use whenever several configurations are ranked in one report: without it, the
    family-wise error rate grows with the size of the sweep, and a sweep that tests 20
    cells will produce a "winner" by chance. NaN inputs are passed through untouched so
    an unusable comparison is visibly unusable rather than silently ranked.
    """
    finite = {k: v for k, v in pvalues.items() if not math.isnan(v)}
    items = sorted(finite.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[key] = running
    for key, value in pvalues.items():
        if key not in adjusted:
            adjusted[key] = value
    return adjusted


def paired_comparison(diffs: Sequence[float], label: str = "") -> dict:
    """Summarise one set of paired differences into the fields a report quotes.

    Returns the mean difference, the between-replicate spread, ``d_z``, the exact
    sign-flip p-value, and ``min_p`` -- the smallest p this many pairs could ever
    produce, so a report can say "4 folds cannot reach significance" instead of
    printing a non-significant p as if it were a negative result.
    """
    clean = [d for d in diffs if not math.isnan(d)]
    return {
        "label": label,
        "n_pairs": len(clean),
        "mean_diff": float(np.mean(clean)) if clean else float("nan"),
        "std_diff": float(np.std(clean, ddof=1)) if len(clean) > 1 else float("nan"),
        "cohens_dz": cohens_dz(clean),
        "p_sign_flip": sign_flip_pvalue(clean),
        "min_attainable_p": min_attainable_pvalue(len(clean)),
    }


def skill_scores(
    scores: Sequence[float], null_scores: Sequence[float]
) -> np.ndarray:
    """Canary-subtracted skill: ``R^2`` minus the constant-predictor's ``R^2``.

    On a corpus with a large shared component, a per-fold *mean-image* predictor
    already scores a high R^2, so absolute R^2 overstates the model badly. Subtracting
    that baseline answers the question actually being asked -- "how much better than
    predicting the average frame?" -- and is the only form in which two corpora of
    different composition can be compared.

    ``scores`` and ``null_scores`` must be aligned per fold. The result is elementwise,
    so callers should feed per-fold scores (not dataset means) to keep the pairing.
    """
    model = np.asarray(scores, dtype=np.float64)
    null = np.asarray(null_scores, dtype=np.float64)
    if model.shape != null.shape:
        raise ValueError(
            f"shape mismatch: {model.shape} vs {null.shape}; "
            "skill_scores needs one aligned pair per fold"
        )
    return model - null


def seed_agreement(per_seed_diffs: Mapping[int, float]) -> dict:
    """Diagnose whether a paired effect survives the seed axis.

    ``per_seed_diffs`` maps a seed to the paired difference observed at that seed,
    typically with the data axis (fold) held fixed. Returns the signs observed, whether
    they disagree, and an interpretation string.

    **A sign that flips across seeds means the "effect" was an initialisation artefact.**
    This is the control that a fold-paired test cannot supply: if the effect's direction
    depends on which weight draw you happened to use, then N seeds (or N folds) of that
    configuration do not constitute N independent confirmations of it.

    A single seed is not evidence of anything on this axis; report the flip rather than
    averaging it away.
    """
    usable = {s: d for s, d in per_seed_diffs.items() if not math.isnan(d)}
    diffs = list(usable.values())
    signs = sorted({int(math.copysign(1, d)) for d in diffs if d != 0.0})
    flips = len(signs) > 1
    summary = {
        "n_seeds": len(usable),
        "mean_diff": float(np.mean(diffs)) if diffs else float("nan"),
        "std_diff": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else float("nan"),
        "signs_observed": signs,
        "sign_flips": flips,
        "per_seed": dict(sorted(usable.items())),
    }
    if len(usable) < 2:
        summary["interpretation"] = (
            "fewer than 2 seeds: the seed axis is unprobed, so an effect measured "
            "at one seed says nothing about whether it survives re-initialisation"
        )
    elif flips:
        summary["interpretation"] = (
            "sign flips across seeds => the effect is an initialisation artefact, "
            "not a property of the configuration"
        )
    else:
        summary["interpretation"] = (
            "sign is stable across seeds => consistent with a real effect; "
            "confirm with more seeds before relying on the magnitude"
        )
    return summary


def aggregate_by_arm(
    per_fold: Mapping[str, Mapping[int, float]],
) -> dict[str, dict[str, float]]:
    """Collapse ``{arm: {fold: score}}`` into per-arm mean/std.

    Kept next to the statistics so a sweep and a report cannot disagree about how an
    arm summary is formed.
    """
    out: dict[str, dict[str, float]] = {}
    for arm, folds in per_fold.items():
        values = [v for v in folds.values() if not math.isnan(v)]
        if not values:
            continue
        out[arm] = {
            "n_folds": len(values),
            "mean_r2": float(np.mean(values)),
            "std_r2": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
        }
    return out


def iter_paired_diffs(
    baseline: Mapping[int, float], arm: Mapping[int, float]
) -> Iterable[tuple[int, float]]:
    """Yield ``(fold, arm - baseline)`` over the folds present in both mappings.

    Folds missing from either arm are skipped rather than imputed: a fold that failed
    to train in one arm only is not a zero difference.
    """
    for fold in sorted(set(baseline) & set(arm)):
        b, a = baseline[fold], arm[fold]
        if not math.isnan(b) and not math.isnan(a):
            yield fold, a - b