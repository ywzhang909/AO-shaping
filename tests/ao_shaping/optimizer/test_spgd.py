"""Offline guards for the SPGD gradient-sign convention. No hardware needed.

Why this file exists
--------------------
Every SPGD loop forms ``pos = param + disturb`` / ``neg = param - disturb`` and
must then convert the two measured objectives into a gradient for a ``Base``
optimizer whose ``update()`` returns a **descent** step (paired with
``param -= update``). Getting the sign wrong silently optimises the OPPOSITE
objective. That has happened twice in this repo and both times was only caught
on hardware:

* ``optimizer/wfless/pib.py`` -- PIB was *minimised* (a ``to_min = -1`` inside a
  nested ``calc_objective`` closure was a no-op; commit 74c7d0f).
* ``optimizer/wf/rms.py`` -- RMS was *ascended*.

These tests check the convention itself, entirely offline:
1. the shared helper's contract for both objective directions;
2. the float cast that keeps unsigned bucket sums from wrapping;
3. an end-to-end simulation driving the REAL ``algorithm.adam`` optimizers over
   a synthetic objective -- maximise must increase it, minimise must decrease;
4. the failure mode itself -- the inverted sign must move the objective the
   wrong way, which is what makes (3) a meaningful guard.
"""

import numpy as np

from ao_shaping.algorithm.gradient.adam import AdaMOD
from ao_shaping.optimizer.spgd import spgd_gradient

OPTIMUM = np.array([1.0, -0.5, 2.0, 0.0])
START = np.full(4, -2.0)


def _peak(p) -> float:
    """Objective to MAXIMISE; peak value 0 at OPTIMUM."""
    return float(-np.sum((np.asarray(p) - OPTIMUM) ** 2))


def _bowl(p) -> float:
    """Objective to MINIMISE; minimum value 0 at OPTIMUM."""
    return float(np.sum((np.asarray(p) - OPTIMUM) ** 2))


def _spgd_run(
    objective,
    *,
    maximize: bool,
    steps: int = 300,
    delta: float = 0.1,
    lr: float = 0.5,
    seed: int = 0,
    invert: bool = False,
):
    """Run a real SPGD loop with a real algorithm-layer optimizer.

    ``invert=True`` deliberately flips the sign to emulate the historical bug.
    """
    rng = np.random.default_rng(seed)
    param = START.copy()
    optimizer = AdaMOD(dim=param.size, lr=lr)
    for _ in range(steps):
        disturb = (
            (rng.random(param.size) < 0.5).astype(np.float64) * 2.0 - 1.0
        ) * delta
        pos_obj = objective(param + disturb)
        neg_obj = objective(param - disturb)
        gradient = spgd_gradient(pos_obj, neg_obj, disturb, maximize=maximize)
        if invert:
            gradient = -gradient
        param = param - optimizer.update(gradient)
    return param, objective(param)


class TestHelperContract:
    def test_minimise_returns_the_ascent_estimate(self):
        gradient = spgd_gradient(10.0, 4.0, np.array([1.0]), maximize=False)
        assert gradient[0] == 6.0

    def test_maximise_negates_the_ascent_estimate(self):
        gradient = spgd_gradient(10.0, 4.0, np.array([1.0]), maximize=True)
        assert gradient[0] == -6.0

    def test_sign_follows_the_perturbation_direction(self):
        disturb = np.array([1.0, -1.0])
        gradient = spgd_gradient(10.0, 4.0, disturb, maximize=False)
        np.testing.assert_allclose(gradient, [6.0, -6.0])


class TestUnsignedObjectives:
    """Energy/bucket objectives are unsigned sums; the cast must prevent wrap."""

    def test_uint64_subtraction_does_not_wrap(self):
        pos, neg = np.uint64(5), np.uint64(9)
        gradient = spgd_gradient(pos, neg, np.array([1.0]), maximize=False)
        assert gradient[0] == -4.0

    def test_uint64_negation_does_not_overflow(self):
        pos, neg = np.uint64(5), np.uint64(9)
        gradient = spgd_gradient(pos, neg, np.array([1.0]), maximize=True)
        assert gradient[0] == 4.0

    def test_result_is_float64(self):
        gradient = spgd_gradient(
            np.uint64(5), np.uint64(9), np.array([1.0]), maximize=True
        )
        assert gradient.dtype == np.float64


class TestSimulatedSpgdDirection:
    """End-to-end: real AdaMOD + synthetic objective + the shared helper."""

    def test_maximise_objective_increases(self):
        j_before = _peak(START)
        param, j_after = _spgd_run(_peak, maximize=True)
        assert j_after > j_before + 1.0
        assert np.linalg.norm(param - OPTIMUM) < np.linalg.norm(START - OPTIMUM)

    def test_minimise_objective_decreases(self):
        j_before = _bowl(START)
        param, j_after = _spgd_run(_bowl, maximize=False)
        assert j_after < j_before - 1.0
        assert np.linalg.norm(param - OPTIMUM) < np.linalg.norm(START - OPTIMUM)

    def test_inverted_sign_moves_the_objective_the_wrong_way(self):
        """The exact historical bug: maximise objective, sign flipped."""
        j_before = _peak(START)
        _, j_after = _spgd_run(_peak, maximize=True, invert=True)
        assert j_after < j_before

    def test_inverted_minimise_climbs_instead_of_descending(self):
        j_before = _bowl(START)
        _, j_after = _spgd_run(_bowl, maximize=False, invert=True)
        assert j_after > j_before


class TestConsistencyWithModuleMappings:
    """The per-module objective->sign mapping must agree with the helper."""

    def test_pib_module_mapping_matches_shared_convention(self):
        from ao_shaping.optimizer.wfless.pib import _objective_to_min

        ascent = (10.0 - 4.0) * np.array([1.0])
        for objective in ("pib", "avg_radiu", "radiu"):
            maximize = _objective_to_min(objective) == -1
            np.testing.assert_allclose(
                spgd_gradient(10.0, 4.0, np.array([1.0]), maximize=maximize),
                -ascent if maximize else ascent,
            )
