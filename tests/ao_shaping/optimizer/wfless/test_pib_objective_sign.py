"""Offline tests for the DM PIB optimizer's SPGD sign factor.

``optimizer.update()`` (Adam/AdaMOD/SGD) returns a *descent* step, so the
parameter update ``v - update`` descends whatever gradient it receives.
Maximisation objectives therefore require the negated gradient estimate.

The flip used to live inside the nested ``calc_objective`` closure, where the
assignment was closure-local and silently ineffective (commit 74c7d0f) -- so
``to_min`` stayed ``1`` and the optimizer *minimised* PIB. Keeping the decision
in a module-level function makes it directly testable, which is how this class
of scoping bug is caught.

No hardware is opened here.
"""

import numpy as np

from ao_shaping.optimizer.wfless.pib import _objective_to_min


class TestObjectiveToMin:
    def test_maximise_objectives_are_negated(self):
        assert _objective_to_min("pib") == -1
        assert _objective_to_min("avg_radiu") == -1

    def test_minimise_objective_is_positive(self):
        assert _objective_to_min("radiu") == 1

    def test_matches_declared_objective_mode(self):
        # optimize_pib declares pib/avg_radiu as "max" and radiu as "min".
        for objective, mode in (
            ("pib", "max"),
            ("avg_radiu", "max"),
            ("radiu", "min"),
        ):
            assert _objective_to_min(objective) == (-1 if mode == "max" else 1)

    def test_unknown_objective_defaults_to_minimise(self):
        assert _objective_to_min("not-an-objective") == 1


class TestUnsignedBucketSums:
    """Bucket sums are unsigned; the diff must be taken in float."""

    def test_float_diff_is_correctly_signed_when_neg_exceeds_pos(self):
        pos, neg = np.uint64(5), np.uint64(9)
        # The loop uses float(pos) - float(neg) so the result can go negative.
        assert float(pos) - float(neg) == -4.0

    def test_signed_diff_then_negation_gives_ascent_step(self):
        # maximise (to_min=-1): gradient = -(pos-neg)*disturb flips the sign.
        pos, neg, disturb, delta = np.uint64(5), np.uint64(9), 1.0, 0.1
        diff = (float(pos) - float(neg)) * _objective_to_min("pib")
        gradient = diff * disturb * delta
        assert gradient > 0.0
