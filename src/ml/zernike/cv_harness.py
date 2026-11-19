"""K-fold cross-validation harness for the Zernike holdout.

The 8 bench calibration probes are currently split by a **fixed seed offset** in the
hardware twin (``ao_shaping.optimizer.wfless.slm_model_in_loop``): train probes are
seeded ``seed + index*1000 + k`` and holdout probes ``seed + 1_000_000 + index*1000 + k``.
That is one fixed partition, not cross-validation, so a model change is scored against
whichever probes happened to be held out -- an optimistic and fragile read.

This module builds a deterministic K-fold partition and reports paired statistics over
the folds. The statistics themselves are **not** re-derived here: every number comes
from :mod:`ml.zernike.eval_stats` (pure NumPy, no scipy/sklearn), the single source of
truth the reports quote.

**The correctness rule this harness exists to enforce** (a real conclusion in this repo
was overturned by violating it): *pair on the axis you varied.* Vary folds => pair on
folds with the seed held fixed; vary seeds => pair on seeds with the folds held fixed.
Folds that all share one ``seed`` share one weight-initialisation draw, so ``k`` folds
measure that single draw ``k`` times and a sign-flip test that treats them as ``k``
independent samples is **anti-conservative**. And because sign-flip p-values are exact
over ``2**k`` sign assignments, ``k`` folds cannot return anything below
``2 / 2**k`` -- :func:`paired_fold_report` always surfaces that floor so a plan cannot
over-claim significance.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from loguru import logger

from ml.zernike.eval_stats import (
    cohens_dz,
    min_attainable_pvalue,
    sign_flip_pvalue,
)

__all__ = [
    "build_folds",
    "min_attainable_pvalue_folds",
    "paired_fold_report",
]


def build_folds(n_items: int, k: int, *, seed: int) -> list[tuple[list[int], list[int]]]:
    """Deterministic K-fold partition as ``[(train_idx, test_idx), ...]``.

    The ``n_items`` indices are shuffled with ``np.random.default_rng(seed)`` before
    being cut into ``k`` contiguous chunks, so the test members are spread rather than
    blocked by position (a blocked split would make every fold a different regime).

    Every item appears in exactly one ``test_idx``; the remaining items form that
    fold's ``train_idx``. Indices are returned sorted for stable, comparable output.

    Args:
        n_items: Number of items to partition (must be >= ``k``).
        k: Number of folds (must be >= 2).
        seed: Seed for the shuffle. Fixed seed => identical folds; the fold partition
            itself must be held fixed when the seed is the varied axis.

    Returns:
        A list of ``k`` ``(train_idx, test_idx)`` pairs.

    Raises:
        ValueError: If ``k < 2`` or ``k > n_items``.
    """
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    if k > n_items:
        raise ValueError(f"k ({k}) cannot exceed n_items ({n_items})")

    rng = np.random.default_rng(seed)
    order = rng.permutation(n_items)
    chunks = np.array_split(order, k)

    folds: list[tuple[list[int], list[int]]] = []
    for i in range(k):
        test_idx = sorted(int(x) for x in chunks[i])
        train_idx = sorted(
            int(x) for j, chunk in enumerate(chunks) if j != i for x in chunk
        )
        folds.append((train_idx, test_idx))

    logger.debug(
        "built {} folds over {} items (seed={}); test sizes {}",
        k,
        n_items,
        seed,
        [len(test) for _, test in folds],
    )
    return folds


def min_attainable_pvalue_folds(k: int) -> float:
    """Smallest p-value a sign-flip test over ``k`` folds can ever produce.

    This is exactly ``2 / 2**k`` and is delegated to
    :func:`ml.zernike.eval_stats.min_attainable_pvalue` so the two cannot drift apart.
    Surface it next to any p-value: at 4 folds the floor is ``0.125``, so a
    non-significant result there is *structural*, not evidence of no effect.
    """
    return min_attainable_pvalue(k)


def paired_fold_report(
    arms: dict[str, Sequence[float]],
    folds: Sequence[tuple[Sequence[int], Sequence[int]]],
    *,
    paired_on: str = "fold",
) -> dict:
    """Per-arm paired statistics over ``folds``, using :mod:`ml.zernike.eval_stats` only.

    ``arms`` maps an arm name to one scalar per fold (e.g. the fold's validation R^2),
    positionally aligned with ``folds``.

    With ``paired_on="fold"`` (the default) the across-arm comparison is taken on the
    **same** folds, which cancels the fold-difficulty term -- the quantity a model
    change should be judged on. With ``paired_on="seed"`` the comparison is instead
    paired on the seed axis with the fold held fixed; that **does not** cancel fold
    difficulty, so a ``seed_note`` warning is attached to the result.

    For two arms the comparison is ``arms[second] - arms[first]`` (dict insertion
    order; the second arm is the treatment), matching
    :func:`ml.zernike.eval_stats.iter_paired_diffs`' ``arm - baseline`` convention.

    Args:
        arms: ``{arm_name: [one score per fold]}``.
        folds: The fold partition the scores were produced under.
        paired_on: Axis the comparison is paired on; ``"fold"`` or ``"seed"``.

    Returns:
        A dict with, for each arm, ``{"mean", "sd", "per_fold"}``; plus
        ``"min_attainable_pvalue"`` (``2 / 2**n_folds``), ``"paired_on"`` and
        ``"n_folds"``. For exactly two arms it also carries a ``"comparison"`` with
        ``"delta_mean"``, ``"cohens_dz"``, ``"pvalue"`` and ``"pvalue_is_at_floor"``.

    Raises:
        ValueError: If ``paired_on`` is not ``"fold"``/``"seed"``, or an arm's length
            does not match the number of folds (pairing would be misaligned).
    """
    if paired_on not in ("fold", "seed"):
        raise ValueError(f"paired_on must be 'fold' or 'seed', got {paired_on!r}")

    n_folds = len(folds)
    report: dict = {}
    for arm, values in arms.items():
        vals = [float(v) for v in values]
        if len(vals) != n_folds:
            raise ValueError(
                f"arm {arm!r} has {len(vals)} values but there are {n_folds} folds"
            )
        report[arm] = {
            "mean": float(np.mean(vals)) if vals else float("nan"),
            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"),
            "per_fold": vals,
        }

    floor = min_attainable_pvalue_folds(n_folds)
    report["min_attainable_pvalue"] = floor
    report["paired_on"] = paired_on
    report["n_folds"] = n_folds

    if paired_on == "seed":
        report["seed_note"] = (
            "paired_on='seed': the comparison is paired on the seed axis with the fold "
            "held fixed, so fold-difficulty cancellation does NOT apply and a small "
            "p-value says nothing about whether the effect survives a different fold "
            "partition."
        )

    if len(arms) == 2:
        names = list(arms)
        baseline = np.asarray([float(v) for v in arms[names[0]]], dtype=np.float64)
        treatment = np.asarray([float(v) for v in arms[names[1]]], dtype=np.float64)
        diffs = list(treatment - baseline)
        pvalue = sign_flip_pvalue(diffs)
        report["comparison"] = {
            "delta_mean": float(np.mean(diffs)) if diffs else float("nan"),
            "cohens_dz": cohens_dz(diffs),
            "pvalue": pvalue,
            "pvalue_is_at_floor": bool(pvalue == floor),
        }

    return report
