"""Block-indexed objective weights (issue #213 item 4c).

Why this exists
---------------
``ShapeObjective`` uses one constant weight set for a whole stage, so the same
trade-off ("be uniform" vs "keep the energy" vs "grow the shape") applies to the
first block and the last. In practice the early blocks are spent moving energy
into the target at all, the middle ones on making the spot *shaped*, and only
the tail benefits from chasing uniformity. A single setting cannot express that,
and the CV-only reading then happily converges to "uniform but dim" or "uniform
but no zero-order suppression" -- the failure mode in
``report/shape_trials/multishape_scale/next_round_20261029.md`` §6.

What it deliberately does NOT do
--------------------------------
It only moves the optimisation path. The acceptance criteria (frozen-aperture
CV, ``ee_rel``, zero-exclusion ratio, saturation ratio, shape coverage) are
frozen for a whole experiment: tuning ``energy`` weight until CV looks good is a
change of criterion, not a change of path, and must be argued separately
(``next_round_20261029.md`` §1.1). Every block records the weights it used, so a
later reader can always answer "why did this block go that way".

``constant`` (the default) returns the historical weights untouched.
"""
from __future__ import annotations

from collections.abc import Mapping

from ao_shaping.algorithm.signal_processing.multishape_objective import DEFAULT_WEIGHTS

#: Phase order of a staged run: energy first, shape in the middle, CV last.
PHASES: tuple[str, ...] = ("energy", "shape", "cv2")

#: Phase multipliers applied to the mode's base weights. Only these three keys
#: are staged; ``smooth`` and the formation-only terms (``partition`` /
#: ``peripheral`` / ``floor`` / ``boundary``) stay at their base value so a
#: comparison against ``constant`` differs in as few variables as possible.
PHASE_SCALE: dict[str, dict[str, float]] = {
    "energy": {"energy": 2.5, "shape": .5, "cv2": .25},
    "shape": {"energy": 1.0, "shape": 2.0, "cv2": .5},
    "cv2": {"energy": .5, "shape": .5, "cv2": 3.0},
}


def phase_for(block: int, blocks: int) -> str:
    """Phase name for ``block`` of a ``blocks``-long stage.

    Splits the stage into thirds by *position*, not by block index, so a
    15-block and a 4-block stage both spend a third each. A stage shorter than
    three blocks cannot be split -- it gets ``"shape"``, the middle term, rather
    than an arbitrary sliver of the first or last phase.
    """
    blocks = int(blocks)
    if blocks < len(PHASES):
        return PHASES[1]
    position = (int(block) + .5)/blocks
    if position < 1/3:
        return PHASES[0]
    if position > 2/3:
        return PHASES[2]
    return PHASES[1]


def staged_weights(mode: str, block: int, blocks: int, *,
                   base: Mapping[str, float] | None = None) -> dict[str, float]:
    """Return the full weight dict to use for ``block``.

    ``mode`` is ``ShapeObjective.mode`` (``"cv"`` / ``"shape"`` / ``"formation"``).
    Keys the mode does not use are skipped, so ``--objective cv`` gets its base
    weights back unchanged; the caller is expected to reject that combination
    instead of silently running an un-staged schedule.
    """
    if mode not in DEFAULT_WEIGHTS:
        raise ValueError(f"unknown objective mode {mode!r}; "
                         f"expected one of {sorted(DEFAULT_WEIGHTS)}")
    weights = dict(base if base is not None else DEFAULT_WEIGHTS[mode])
    if mode == "cv":
        # The CV-only loss is ``cv2 + smooth``: it has no energy/shape trade-off
        # to stage, and scaling cv2 alone would just re-weight the smoothness
        # term. Return the base unchanged; callers are expected to reject this
        # combination outright, which ``transfer_shapes_runner`` does.
        return weights
    scale = PHASE_SCALE[phase_for(block, blocks)]
    for key, factor in scale.items():
        if key in weights:
            weights[key] = float(weights[key])*factor
    return weights
