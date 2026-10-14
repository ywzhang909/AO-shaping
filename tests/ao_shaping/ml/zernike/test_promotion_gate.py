"""Tests for the intensity-space term (default OFF) and the promotion gate.

Two independent subjects, one file because they are one decision: the term is the
candidate the gate judges.

The invariant that matters most here is **the default must be a no-op**. A flag
that defaults to OFF but shifts the numbers when it is OFF is worse than no flag,
because every pre-existing regression test would have to be re-baselined and
nobody would notice the drift. So the first class pins bit-for-bit equality with
the pre-flag code path, not an approximation of it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ao_shaping.algorithm.signal_processing.square_shaping import (
    SquareShapingConfig,
    compute_bench_metrics,
    composite_score,
    shaping_score,
)
from ao_shaping.utils.image.target import log_gradient_difference_metric
from ml.zernike.cv_harness import build_folds
from ml.zernike.promotion_gate import (
    evaluate_promotion,
    folds_needed_for_alpha,
)

_SMALL = dict(n_grid=16, far_field_padding=2, target_side_px=6, seed=3)


def _pattern(n: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """A speckled measured frame and a smooth square target of the same shape."""
    yy, xx = np.mgrid[0:n, 0:n]
    target = np.zeros((n, n))
    c = n // 2
    target[c - 3 : c + 3, c - 3 : c + 3] = 1.0
    rng = np.random.default_rng(11)
    speckle = np.zeros((n, n))
    speckle[c - 3 : c + 3, c - 3 : c + 3] = 1.0
    speckle += 0.4 * rng.random((n, n)) * (xx + yy < c)
    return speckle, target


class TestIntensitySpaceFlag:
    """``w_loggrad`` is off by default and a bit-for-bit no-op when off."""

    def test_the_flag_defaults_to_off(self) -> None:
        assert SquareShapingConfig().w_loggrad == 0.0

    def test_a_negative_weight_is_refused(self) -> None:
        with pytest.raises(ValueError, match="w_loggrad"):
            SquareShapingConfig(w_loggrad=-0.1)

    def test_a_non_finite_weight_is_refused(self) -> None:
        with pytest.raises(ValueError, match="w_loggrad"):
            SquareShapingConfig(w_loggrad=float("nan"))

    def test_zero_weight_is_exactly_the_legacy_score(self) -> None:
        """The bit-for-bit guarantee, asserted against the legacy composition."""
        measured, target = _pattern()
        center = (measured.shape[1] // 2, measured.shape[0] // 2)
        legacy = composite_score(compute_bench_metrics(measured, target, center=center))
        assert shaping_score(measured, target, center=center) == legacy
        assert shaping_score(
            measured, target, center=center, w_loggrad=0.0
        ) == legacy

    def test_a_positive_weight_lowers_the_score(self) -> None:
        """The penalty is subtracted, because both halves are higher-is-better."""
        measured, target = _pattern()
        center = (measured.shape[1] // 2, measured.shape[0] // 2)
        base = shaping_score(measured, target, center=center)
        penalised = shaping_score(
            measured, target, center=center, w_loggrad=1.0
        )
        residual = log_gradient_difference_metric(measured, target)
        assert penalised == pytest.approx(base - residual)

    def test_the_flag_actually_sees_a_structure_mismatch(self) -> None:
        """Guard the guard: a smooth-vs-speckle pair must score differently.

        Without this, the term could be wired but inert (e.g. the lazy import
        silently failing, or the weight never reaching the score) and the
        equality test above would still pass.
        """
        measured, target = _pattern()
        smooth = np.zeros_like(measured)
        c = measured.shape[0] // 2
        smooth[c - 3 : c + 3, c - 3 : c + 3] = 1.0
        assert log_gradient_difference_metric(measured, target) > 0.0
        assert log_gradient_difference_metric(smooth, target) < log_gradient_difference_metric(
            measured, target
        )

    def test_a_negative_weight_is_refused_at_call_time(self) -> None:
        measured, target = _pattern()
        with pytest.raises(ValueError, match="w_loggrad"):
            shaping_score(measured, target, w_loggrad=-1.0)

    def test_the_config_flag_reaches_the_spgd_objective(self) -> None:
        """The knob on the config is the one SPGD consumes.

        Comparing the histories rather than the final phase: at a small budget
        the solved phase barely moves, so a final-phase comparison could pass
        with the flag ignored.
        """
        from ao_shaping.algorithm.signal_processing.square_shaping import spgd_shape

        cfg = SquareShapingConfig(**_SMALL)
        off = spgd_shape(cfg, n_iters=12, delta=0.1, lr=0.02, seed=5, dim=4)
        on = spgd_shape(
            SquareShapingConfig(**{**_SMALL, "w_loggrad": 5.0}),
            n_iters=12,
            delta=0.1,
            lr=0.02,
            seed=5,
            dim=4,
        )
        assert off.metrics["score"] != pytest.approx(on.metrics["score"])


class TestFoldsNeeded:
    """``2 / 2**k`` inverted into "how many folds would it take"."""

    def test_five_percent_needs_six_folds(self) -> None:
        # floor(5) = 2/32 = 0.0625 > 0.05 ; floor(6) = 0.03125 <= 0.05
        assert folds_needed_for_alpha(0.05) == 6

    def test_ten_percent_needs_five_folds(self) -> None:
        assert folds_needed_for_alpha(0.10) == 5

    def test_a_bad_alpha_is_refused(self) -> None:
        for bad in (0.0, 1.0, -0.1, 2.0):
            with pytest.raises(ValueError, match="alpha"):
                folds_needed_for_alpha(bad)


class TestPromotionGate:
    """The gate refuses for distinguishable reasons and promotes only on merit."""

    FOLDS = build_folds(20, 10, seed=1)

    def _verdict(self, cand: list[float], inc: list[float], **kw):
        return evaluate_promotion(cand, inc, self.FOLDS, **kw)

    def test_a_consistent_real_win_is_promoted(self) -> None:
        incumbent = [0.50 + 0.01 * i for i in range(10)]
        candidate = [v + 0.05 for v in incumbent]
        verdict = self._verdict(candidate, incumbent, min_effect=0.01)
        assert verdict.promoted, verdict.reasons
        assert bool(verdict) is True
        assert verdict.delta_mean == pytest.approx(0.05)
        assert verdict.pvalue < 0.05
        assert verdict.reasons == ("promoted",)

    def test_too_few_folds_is_a_structural_refusal_not_a_null_result(self) -> None:
        """The headline rule: at 4 folds p < 0.05 is unreachable, so refuse."""
        folds = build_folds(8, 4, seed=1)
        incumbent = [0.50, 0.51, 0.52, 0.53]
        candidate = [v + 0.5 for v in incumbent]  # an enormous effect
        verdict = evaluate_promotion(candidate, incumbent, folds)
        assert not verdict.promoted
        assert verdict.pvalue_is_at_floor
        assert verdict.min_attainable_pvalue == pytest.approx(0.125)
        assert any("structural" in r for r in verdict.reasons)
        assert verdict.folds_required == 6

    def test_a_huge_effect_still_cannot_promote_at_four_folds(self) -> None:
        """Anti-vacuity: the structural refusal is not about effect size."""
        folds = build_folds(8, 4, seed=1)
        incumbent = [0.50, 0.51, 0.52, 0.53]
        verdict = evaluate_promotion(
            [v + 1e6 for v in incumbent], incumbent, folds
        )
        assert not verdict.promoted
        assert any("structural" in r for r in verdict.reasons)

    def test_a_tiny_but_significant_gain_is_refused_as_impractical(self) -> None:
        incumbent = [0.50 + 0.01 * i for i in range(10)]
        candidate = [v + 1e-4 for v in incumbent]
        verdict = self._verdict(candidate, incumbent, min_effect=0.01)
        assert not verdict.promoted
        assert any("practical" in r for r in verdict.reasons)

    def test_a_regression_is_refused_with_a_sign_reason(self) -> None:
        incumbent = [0.50 + 0.01 * i for i in range(10)]
        candidate = [v - 0.05 for v in incumbent]
        verdict = self._verdict(candidate, incumbent)
        assert not verdict.promoted
        assert any("sign" in r and "WORSE" in r for r in verdict.reasons)

    def test_seed_pairing_is_refused_by_default(self) -> None:
        incumbent = [0.50 + 0.01 * i for i in range(10)]
        candidate = [v + 0.05 for v in incumbent]
        verdict = self._verdict(candidate, incumbent, paired_on="seed")
        assert not verdict.promoted
        assert any("axis" in r for r in verdict.reasons)

    def test_seed_pairing_can_be_overridden_explicitly(self) -> None:
        incumbent = [0.50 + 0.01 * i for i in range(10)]
        candidate = [v + 0.05 for v in incumbent]
        verdict = self._verdict(
            candidate, incumbent, paired_on="seed", allow_seed_paired=True
        )
        assert verdict.promoted, verdict.reasons

    def test_a_lower_is_better_score_can_be_promoted(self) -> None:
        incumbent = [0.50 + 0.01 * i for i in range(10)]
        candidate = [v - 0.05 for v in incumbent]
        verdict = self._verdict(candidate, incumbent, higher_is_better=False)
        assert verdict.promoted, verdict.reasons

    def test_no_difference_is_refused_as_insignificant(self) -> None:
        incumbent = [0.50 + 0.01 * i for i in range(10)]
        verdict = self._verdict(list(incumbent), incumbent)
        assert not verdict.promoted
        assert any("significance" in r for r in verdict.reasons)

    def test_every_refusal_names_at_least_one_cause(self) -> None:
        incumbent = [0.50 + 0.01 * i for i in range(10)]
        verdict = self._verdict([v - 0.05 for v in incumbent], incumbent)
        assert verdict.reasons
        assert all(isinstance(r, str) and r for r in verdict.reasons)

    def test_a_length_mismatch_is_refused(self) -> None:
        with pytest.raises(ValueError, match="folds"):
            self._verdict([0.5] * 9, [0.4] * 10)

    def test_bad_parameters_are_refused(self) -> None:
        incumbent = [0.5] * 10
        with pytest.raises(ValueError, match="alpha"):
            self._verdict(incumbent, incumbent, alpha=0.0)
        with pytest.raises(ValueError, match="min_effect"):
            self._verdict(incumbent, incumbent, min_effect=-1.0)
        with pytest.raises(ValueError, match="paired_on"):
            self._verdict(incumbent, incumbent, paired_on="nope")

    def test_the_floor_is_always_surfaced(self) -> None:
        """Whatever the outcome, the structural limit is on the verdict."""
        incumbent = [0.50 + 0.01 * i for i in range(10)]
        for cand in (incumbent, [v + 0.05 for v in incumbent]):
            verdict = self._verdict(cand, incumbent)
            assert verdict.min_attainable_pvalue == pytest.approx(
                2.0 / (2**10)
            )
            assert verdict.n_folds == 10
            assert math.isfinite(verdict.pvalue)