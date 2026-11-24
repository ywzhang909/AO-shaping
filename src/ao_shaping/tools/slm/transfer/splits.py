"""Split-tier vocabulary and pre-registered round assignment (#254 T2).

Three-tier split: ``train`` = fitting rows; ``validation`` (role
``selection_validation``) = the ONLY tier promotion and checkpoint selection
may read; ``test`` / ``final_test`` = sealed -- never fitted, never scored by
selection quality, evaluated once by ``adaptation.final_test_quality`` after
the candidate set is frozen. Older datasets legitimately lack the sealed
tiers, so all readers must go through the split value, never the row count.
"""
from __future__ import annotations

import numpy as np

SELECTION_SPLIT = "validation"
SEALED_SPLITS = frozenset({"test", "final_test"})

PROBE_ROUND = 6  # probe rows per pre-registered round (matches the mod-6 family cycle)
ROUND_SEED_SALT = 0x5EED2026  # rounds scheme: seed domain separated from probe grids


def assign_round_splits(n_rounds: int, seed: int, final_test_rounds: int = 1) -> dict[int, str]:
    """Pre-registered round -> split mapping (#254 T2, ``split_scheme="rounds"``).

    Whole rounds join one tier, so drift epochs cannot straddle train/validation
    (the legacy per-index interleave let a phase from the same block leak across
    tiers). The mapping is a pure function of (n_rounds, seed, final_test_rounds)
    and must be printed by ``--dry-run`` before any capture, per pre-registration.
    Sealed tier (final_test) takes the tail of the shuffled round order; only
    runs with >= 4 rounds seal (>= 3 rounds stay for train+selection).
    """
    if n_rounds < 4:
        raise ValueError(
            f"rounds split needs >= 4 rounds (count >= {4 * PROBE_ROUND}); got {n_rounds}")
    if not 1 <= final_test_rounds <= max(1, n_rounds - 3):
        raise ValueError(f"final_test_rounds must be in 1..{max(1, n_rounds - 3)}")
    order = np.random.default_rng(seed + ROUND_SEED_SALT).permutation(n_rounds)
    sealed = [int(r) for r in order[-final_test_rounds:]]
    rest = [int(r) for r in order[:-final_test_rounds]]
    n_val = max(1, -(-len(rest) // 3))
    val = rest[-n_val:]
    mapping = {r: "train" for r in rest[:-n_val]}
    mapping.update({r: "validation" for r in val})
    mapping.update({r: "final_test" for r in sealed})
    return {r: mapping[r] for r in range(n_rounds)}


def probe_round_of_label(label: str) -> int | None:
    """Round index of a probe row (None for calibration/ramp/reference rows)."""
    if not label.startswith("probe_"):
        return None
    return int(label.split("_")[1]) // PROBE_ROUND
