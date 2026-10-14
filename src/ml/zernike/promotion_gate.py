"""Promotion gate: has a candidate variant *earned* the right to become the default?

**Why this exists.** Several experiments in this repo added an extra component to
a working pipeline, measured a number that looked fine, and were then found to be
statistically indistinguishable from the incumbent at a large parameter cost. The
recorded instance: a hybrid physics-envelope + CNN-residual forward model scored
``p >= 0.61`` against the physics-only model on two protocols and five metrics
while carrying 45x the parameters, so it was explicitly **not** promoted to a
training option. See ``report/zernike_phase2amp/report.md``.

A flag that defaults to OFF is the right shipping state for such a variant, but
"OFF" is a decision that needs a way to be revisited when evidence arrives. This
module is that way: it turns a set of held-out fold scores into a verdict with the
refusal reasons attached, so a promotion that does happen has to clear a bar that
was written down in advance.

**What the gate refuses, and why each refusal is a different failure.** These are
not variations of "not significant":

1. **Structural refusal.** An exact sign-flip test over ``k`` paired folds cannot
   return a p-value below ``2 / 2**k``. If that floor is already above the
   requested ``alpha``, the protocol *cannot* answer the question no matter how
   large the effect, so the gate refuses and reports the ``k`` that would be
   needed. Reporting an effect size is legitimate; reporting a verdict is not.
2. **Practical-significance refusal.** A tiny but real effect is not worth a
   promotion if it is below the caller's ``min_effect``.
3. **Sign refusal.** The candidate must be better on the stated axis; a
   statistically resolvable *regression* is a rejection, not a licence to flip.
4. **Axis refusal.** ``paired_on="seed"`` pairs along the seed axis with the fold
   held fixed, which does **not** cancel fold difficulty. A gate that promoted on
   that basis would be repeating the exact mistake the harness warns about, so it
   refuses unless the caller opts in explicitly.

Every statistic is delegated to :mod:`ml.zernike.eval_stats` and the fold
bookkeeping to :mod:`ml.zernike.cv_harness`; nothing is re-derived here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from loguru import logger

from ml.zernike.cv_harness import paired_fold_report
from ml.zernike.eval_stats import min_attainable_pvalue

__all__ = [
    "PromotionVerdict",
    "evaluate_promotion",
    "folds_needed_for_alpha",
]


def folds_needed_for_alpha(alpha: float) -> int:
    """Smallest ``k`` whose exact sign-flip floor ``2 / 2**k`` is ``<= alpha``.

    This is the "how many folds would it take" counterpart of
    :func:`ml.zernike.eval_stats.min_attainable_pvalue`, and it exists so a
    structural refusal can state the remedy rather than only the obstacle.

    Args:
        alpha: Target significance level in ``(0, 1)``.

    Returns:
        The minimum fold count; ``1`` if even one fold already clears ``alpha``
        (it cannot be reached in practice, so callers should treat small results
        as "no fold count is legitimate" -- the exact test needs >= 2).

    Raises:
        ValueError: if ``alpha`` is not in ``(0, 1)``.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")
    k = 1
    while min_attainable_pvalue(k) > alpha:
        k += 1
        if k > 1024:  # pragma: no cover - alpha is already validated above
            raise ValueError(f"alpha {alpha!r} is unreachable for any fold count")
    return k


@dataclass(frozen=True)
class PromotionVerdict:
    """Outcome of a promotion decision.

    Attributes:
        promoted: Whether the candidate may become the default.
        reasons: Why it was refused, or ``("promoted",)``. Empty is never valid --
            a verdict must always carry at least one human-readable cause.
        delta_mean: Mean paired difference ``candidate - incumbent``.
        cohens_dz: Paired effect size.
        pvalue: Exact sign-flip p-value for the paired differences.
        min_attainable_pvalue: Structural floor ``2 / 2**n_folds`` for this fold
            count.
        pvalue_is_at_floor: Whether the p-value sits exactly on that floor, i.e.
            the fold count is the binding constraint.
        n_folds: Number of paired folds.
        paired_on: Axis the comparison was paired on.
        folds_required: Folds needed for this protocol to be able to reach
            ``alpha`` at all.
    """

    promoted: bool
    reasons: tuple[str, ...]
    delta_mean: float
    cohens_dz: float
    pvalue: float
    min_attainable_pvalue: float
    pvalue_is_at_floor: bool
    n_folds: int
    paired_on: str
    folds_required: int

    def __bool__(self) -> bool:
        """Truthy only when the promotion is granted."""
        return self.promoted


def evaluate_promotion(
    candidate: Sequence[float],
    incumbent: Sequence[float],
    folds: Sequence[tuple[Sequence[int], Sequence[int]]],
    *,
    alpha: float = 0.05,
    min_effect: float = 0.0,
    paired_on: str = "fold",
    allow_seed_paired: bool = False,
    higher_is_better: bool = True,
) -> PromotionVerdict:
    """Decide whether ``candidate`` has earned promotion over ``incumbent``.

    Both series are one score per fold, positionally aligned with ``folds``. The
    decision is ``promoted`` only when **all** of the following hold; each failure
    appends its own reason so the caller can see every obstacle, not just the
    first.

    * the protocol can reach ``alpha`` at this fold count (else *structural*);
    * the pairing axis is legitimate (else *axis*);
    * the mean paired difference clears ``min_effect`` (else *practical*);
    * the mean difference points the right way (else *sign*);
    * the exact sign-flip p-value is below ``alpha`` (else *significance*).

    Args:
        candidate: Candidate score per fold.
        incumbent: Incumbent score per fold.
        folds: The partition both series were produced under.
        alpha: Required significance level.
        min_effect: Smallest mean paired difference worth promoting for, on the
            candidate's own scale.
        paired_on: ``"fold"`` or ``"seed"``; see the module docstring.
        allow_seed_paired: Permit promotion on a seed-paired comparison. Off by
            default because seed pairing does not cancel fold difficulty.
        higher_is_better: Sign convention of the score.

    Returns:
        The :class:`PromotionVerdict`.

    Raises:
        ValueError: On a length mismatch, a bad ``alpha``/``min_effect``, or an
            unknown ``paired_on``.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")
    if not math.isfinite(min_effect) or min_effect < 0.0:
        raise ValueError(f"min_effect must be >= 0 and finite, got {min_effect!r}")
    if paired_on not in ("fold", "seed"):
        raise ValueError(f"paired_on must be 'fold' or 'seed', got {paired_on!r}")

    report = paired_fold_report(
        {"incumbent": list(incumbent), "candidate": list(candidate)},
        list(folds),
        paired_on=paired_on,
    )
    comparison = report["comparison"]
    n_folds = int(report["n_folds"])
    floor = float(report["min_attainable_pvalue"])
    pvalue = float(comparison["pvalue"])
    delta_mean = float(comparison["delta_mean"])
    required = folds_needed_for_alpha(alpha)

    reasons: list[str] = []

    # 1. Structural: the protocol may be unable to answer at any effect size.
    if floor > alpha:
        reasons.append(
            f"structural: with {n_folds} folds the exact sign-flip floor is "
            f"{floor:g} > alpha={alpha:g}; this protocol cannot reach "
            f"significance for ANY effect. Use >= {required} folds "
            f"(or report the effect size only)."
        )

    # 2. Axis: seed pairing does not cancel fold difficulty.
    if paired_on == "seed" and not allow_seed_paired:
        reasons.append(
            "axis: paired on the seed axis with the fold held fixed, which does "
            "not cancel fold difficulty; pass allow_seed_paired=True to override "
            "knowing what that costs"
        )

    # 3. Practical significance.
    if abs(delta_mean) < min_effect:
        reasons.append(
            f"practical: mean paired delta {delta_mean:+.6g} is below the "
            f"required |effect| >= {min_effect:g}"
        )

    # 4. Sign: a resolvable regression is a rejection.
    if not higher_is_better and delta_mean > 0.0:
        reasons.append(
            f"sign: delta {delta_mean:+.6g} means the candidate is WORSE under a "
            "lower-is-better score"
        )
    elif higher_is_better and delta_mean < 0.0:
        reasons.append(
            f"sign: delta {delta_mean:+.6g} means the candidate is WORSE under a "
            "higher-is-better score"
        )

    # 5. Significance (only meaningful once 1 and 4 are satisfied, but reported
    #    either way so the caller sees the full picture).
    if pvalue >= alpha:
        reasons.append(
            f"significance: exact sign-flip p={pvalue:g} >= alpha={alpha:g}"
            + (" (at the structural floor)" if pvalue == floor else "")
        )

    verdict = PromotionVerdict(
        promoted=not reasons,
        reasons=tuple(reasons) if reasons else ("promoted",),
        delta_mean=delta_mean,
        cohens_dz=float(comparison["cohens_dz"]),
        pvalue=pvalue,
        min_attainable_pvalue=floor,
        pvalue_is_at_floor=bool(comparison["pvalue_is_at_floor"]),
        n_folds=n_folds,
        paired_on=paired_on,
        folds_required=required,
    )
    logger.info(
        "promotion verdict: promoted={} delta={:+.6g} p={:g} floor={:g} reasons={}",
        verdict.promoted,
        verdict.delta_mean,
        verdict.pvalue,
        verdict.min_attainable_pvalue,
        verdict.reasons,
    )
    return verdict