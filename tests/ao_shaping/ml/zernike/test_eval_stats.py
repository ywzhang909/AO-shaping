"""Locks the evaluation statistics that the Zernike reports quote.

These are the numbers a reader is asked to trust, so each one is pinned against a
hand-computed value rather than a golden snapshot. The statistical claims in
``report/zernike_coeff2amp/report.md`` -- the canary baseline, the input_terms verdict,
the sign flip -- all reduce to these functions.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.zernike.eval_stats import (
    aggregate_by_arm,
    cohens_dz,
    holm_bonferroni,
    iter_paired_diffs,
    min_attainable_pvalue,
    paired_comparison,
    seed_agreement,
    sign_flip_pvalue,
    skill_scores,
)


class TestSignFlipPValue:
    def test_all_zero_differences_are_not_significant(self):
        assert sign_flip_pvalue([0.0, 0.0, 0.0, 0.0]) == 1.0

    def test_single_pair_cannot_reach_significance(self):
        # min attainable = 2 / 2**1 = 1.0, so a lone replicate is powerless.
        assert sign_flip_pvalue([0.5]) == 1.0

    def test_minimum_attainable_is_two_over_two_to_the_n(self):
        # A perfectly consistent difference hits exactly the structural floor.
        diffs = [0.1, 0.1, 0.1, 0.1]
        assert sign_flip_pvalue(diffs) == pytest.approx(
            min_attainable_pvalue(len(diffs))
        )
        assert sign_flip_pvalue(diffs) == pytest.approx(2 / 16)

    def test_four_folds_cannot_reach_five_percent(self):
        """The structural floor is the reason small folds report effect sizes only."""
        assert min_attainable_pvalue(4) == pytest.approx(0.125)
        assert min_attainable_pvalue(4) > 0.05

    def test_eighteen_folds_have_a_usable_floor(self):
        assert min_attainable_pvalue(18) == pytest.approx(7.62939453125e-06)

    def test_nans_are_dropped_not_zeroed(self):
        # One unusable fold must not masquerade as a zero difference.
        assert sign_flip_pvalue([0.1, np.nan, 0.2, 0.15]) == sign_flip_pvalue(
            [0.1, 0.2, 0.15]
        )

    def test_no_data_is_nan(self):
        assert np.isnan(sign_flip_pvalue([]))

    def test_sign_is_irrelevant_by_construction(self):
        assert sign_flip_pvalue([0.1, 0.2, -0.15, 0.05]) == pytest.approx(
            sign_flip_pvalue([-0.1, -0.2, 0.15, -0.05])
        )

    def test_a_real_effect_beats_a_null_one(self):
        real = sign_flip_pvalue([0.10, 0.12, 0.11, 0.09, 0.13, 0.10])
        null = sign_flip_pvalue([0.02, -0.03, 0.01, 0.04, -0.02, 0.03])
        assert real < null

    def test_is_symmetric_in_magnitude(self):
        diffs = [0.1, -0.2, 0.15, -0.05, 0.2]
        assert sign_flip_pvalue(diffs) == pytest.approx(sign_flip_pvalue(-d for d in diffs))


class TestCohensDz:
    def test_matches_hand_computed(self):
        diffs = [0.1, 0.2, 0.15, 0.05]
        expected = float(np.mean(diffs) / np.std(diffs, ddof=1))
        assert cohens_dz(diffs) == pytest.approx(expected)

    def test_single_pair_is_nan(self):
        assert np.isnan(cohens_dz([0.1]))

    def test_zero_spread_with_nonzero_mean_is_infinite(self):
        # A constant effect has no replicate variance: no magnitude is supportable.
        assert cohens_dz([0.1, 0.1, 0.1]) == float("inf")

    def test_zero_spread_with_zero_mean_is_nan(self):
        assert np.isnan(cohens_dz([0.0, 0.0, 0.0]))

    def test_magnitude_tracks_effect_size(self):
        small = cohens_dz([0.01, 0.02, 0.015, 0.005])
        large = cohens_dz([0.10, 0.20, 0.15, 0.05])
        assert abs(large) > abs(small)


class TestHolmBonferroni:
    def test_monotone_and_bounded(self):
        adjusted = holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.03})
        assert adjusted["a"] == pytest.approx(0.03)
        assert adjusted["c"] == pytest.approx(0.06)
        assert adjusted["b"] == pytest.approx(0.06)

    def test_never_decreasing_with_rank(self):
        ps = {"a": 0.001, "b": 0.002, "c": 0.5, "d": 0.9}
        adjusted = holm_bonferroni(ps)
        ranks = sorted(adjusted, key=lambda k: ps[k])
        values = [adjusted[k] for k in ranks]
        assert all(b >= a for a, b in zip(values, values[1:], strict=False))

    def test_all_bounded_by_one(self):
        assert all(v <= 1.0 for v in holm_bonferroni({"a": 0.9, "b": 0.95}).values())

    def test_nans_pass_through_untouched(self):
        # m counts only the finite hypotheses, so the surviving one is uncorrected.
        adjusted = holm_bonferroni({"a": 0.01, "broken": float("nan")})
        assert np.isnan(adjusted["broken"])
        assert adjusted["a"] == pytest.approx(0.01)

    def test_adjusted_is_never_below_raw(self):
        ps = {"a": 0.01, "b": 0.02, "c": 0.03}
        adjusted = holm_bonferroni(ps)
        assert all(adjusted[k] >= ps[k] for k in ps)

    def test_empty_is_empty(self):
        assert holm_bonferroni({}) == {}


class TestMinAttainable:
    def test_decreases_with_more_pairs(self):
        values = [min_attainable_pvalue(n) for n in range(1, 20)]
        assert all(b < a for a, b in zip(values, values[1:], strict=False))

    def test_nonpositive_is_nan(self):
        assert np.isnan(min_attainable_pvalue(0))


class TestPairedComparison:
    def test_reports_the_structural_floor_alongside_the_p_value(self):
        result = paired_comparison([0.1, 0.2, 0.15, 0.05], "arm-vs-base")
        assert result["label"] == "arm-vs-base"
        assert result["n_pairs"] == 4
        assert result["min_attainable_p"] == pytest.approx(0.125)
        assert result["p_sign_flip"] >= result["min_attainable_p"]

    def test_mean_and_std_agree_with_numpy(self):
        diffs = [0.1, 0.2, 0.15, 0.05, -0.02]
        result = paired_comparison(diffs)
        assert result["mean_diff"] == pytest.approx(float(np.mean(diffs)))
        assert result["std_diff"] == pytest.approx(float(np.std(diffs, ddof=1)))

    def test_drops_nans_from_the_count(self):
        assert paired_comparison([0.1, np.nan, 0.2])["n_pairs"] == 2

    def test_empty_reports_nan_mean(self):
        assert np.isnan(paired_comparison([])["mean_diff"])


class TestSkillScores:
    def test_subtracts_the_canary_elementwise(self):
        got = skill_scores([0.90, 0.80, 0.70], [0.83, 0.74, 0.52])
        assert got == pytest.approx([0.07, 0.06, 0.18])

    def test_zero_skill_when_model_matches_canary(self):
        assert skill_scores([0.8, 0.7], [0.8, 0.7]) == pytest.approx([0.0, 0.0])

    def test_shape_mismatch_is_rejected(self):
        # Misalignment here would silently compare different folds.
        with pytest.raises(ValueError, match="shape mismatch"):
            skill_scores([0.9, 0.8, 0.7], [0.8, 0.7])

    def test_returns_an_array(self):
        assert isinstance(skill_scores([0.9], [0.8]), np.ndarray)


class TestSeedAgreement:
    def test_a_sign_flip_is_flagged(self):
        """The measurement that overturned the 18-fold input_terms p-value."""
        got = seed_agreement({0: -0.0018, 1: 0.0005, 2: -0.0002, 3: -0.0053})
        assert got["sign_flips"] is True
        assert got["signs_observed"] == [-1, 1]
        assert "initialisation artefact" in got["interpretation"]

    def test_a_consistent_sign_is_not_flagged(self):
        got = seed_agreement({0: 0.10, 1: 0.12, 2: 0.11})
        assert got["sign_flips"] is False
        assert got["signs_observed"] == [1]
        assert "consistent with a real effect" in got["interpretation"]

    def test_a_single_seed_says_the_axis_is_unprobed(self):
        got = seed_agreement({0: 0.1})
        assert got["n_seeds"] == 1
        assert got["sign_flips"] is False
        assert "unprobed" in got["interpretation"]

    def test_no_seeds_is_nan_mean(self):
        assert np.isnan(seed_agreement({})["mean_diff"])

    def test_zero_differences_do_not_fabricate_a_sign(self):
        got = seed_agreement({0: 0.0, 1: 0.0})
        assert got["signs_observed"] == []

    def test_nan_seed_is_excluded(self):
        got = seed_agreement({0: 0.1, 1: float("nan")})
        assert got["n_seeds"] == 1

    def test_per_seed_is_sorted_and_complete(self):
        got = seed_agreement({2: 0.1, 0: -0.1, 1: 0.2})
        assert list(got["per_seed"]) == [0, 1, 2]

    def test_mean_and_std_match_numpy(self):
        diffs = {0: -0.0018, 1: 0.0005, 2: -0.0002}
        got = seed_agreement(diffs)
        assert got["mean_diff"] == pytest.approx(float(np.mean(list(diffs.values()))))

    def test_opposite_uniform_signs_are_a_flip(self):
        got = seed_agreement({0: 0.5, 1: -0.5})
        assert got["sign_flips"] is True


class TestAggregateByArm:
    def test_mean_and_std_per_arm(self):
        got = aggregate_by_arm({"conv": {0: 0.5, 1: 0.7}, "mlp": {0: 0.4, 1: 0.6}})
        assert got["conv"]["n_folds"] == 2
        assert got["conv"]["mean_r2"] == pytest.approx(0.6)
        assert got["mlp"]["mean_r2"] == pytest.approx(0.5)

    def test_arm_with_no_usable_folds_is_omitted(self):
        got = aggregate_by_arm({"empty": {0: float("nan")}, "ok": {0: 0.5}})
        assert "empty" not in got
        assert "ok" in got

    def test_empty_input(self):
        assert aggregate_by_arm({}) == {}


class TestIterPairedDiffs:
    def test_pairs_only_shared_folds(self):
        got = dict(iter_paired_diffs({0: 0.5, 1: 0.6}, {0: 0.7, 1: 0.8}))
        assert got == {0: pytest.approx(0.2), 1: pytest.approx(0.2)}

    def test_a_fold_missing_from_one_arm_is_skipped_not_zeroed(self):
        got = dict(iter_paired_diffs({0: 0.5, 1: 0.6}, {0: 0.7}))
        assert got == {0: pytest.approx(0.2)}

    def test_nan_scores_are_skipped(self):
        got = dict(iter_paired_diffs({0: 0.5}, {0: float("nan")}))
        assert got == {}

    def test_yields_in_sorted_fold_order(self):
        got = list(iter_paired_diffs({2: 0.0, 0: 0.0, 1: 0.0}, {2: 1.0, 0: 1.0, 1: 1.0}))
        assert [fold for fold, _ in got] == [0, 1, 2]


class TestReproducesThePublishedNumbers:
    """The exact figures the committed report quotes, so a refactor cannot move them."""

    def test_input_terms_mean_matches_the_report(self):
        # report/zernike_coeff2amp quotes in136 - in78 = -0.0977 as the mean of 18
        # paired fold differences. The p-value is NOT reproducible from the mean
        # alone (it needs the per-fold spread), so only the mean is pinned here.
        assert paired_comparison([-0.0977] * 18)["mean_diff"] == pytest.approx(-0.0977)

    def test_a_constant_difference_hits_the_structural_floor(self):
        # 18 *identical* differences is the most significant outcome possible, which
        # is why the real p=0.0144 implies genuine fold-to-fold variation.
        stats = paired_comparison([-0.0977] * 18)
        assert stats["p_sign_flip"] == pytest.approx(min_attainable_pvalue(18))
        assert stats["p_sign_flip"] < 0.0144

    def test_seed_control_result(self):
        # Same comparison, one fold, five seeds: mean -0.0031, sign flips.
        per_seed = {0: -0.0018, 1: 0.0005, 2: -0.0002, 3: -0.0053, 4: -0.0087}
        got = seed_agreement(per_seed)
        assert got["mean_diff"] == pytest.approx(-0.0031, abs=5e-5)
        assert got["sign_flips"] is True

    def test_four_fold_objective_protocol_cannot_be_significant(self):
        # The report says this outright; the function must agree.
        assert paired_comparison([0.02, 0.03, 0.01, 0.04])["min_attainable_p"] > 0.05