"""Locks the K-fold CV harness and its reuse-of-``eval_stats`` contract.

The harness exists because the bench holdout is a fixed partition, not a cross
validation -- these tests pin (a) that the fold partition is a true, deterministic,
seed-controlled partition, and (b) that every statistic it reports is delegated to
``ml.zernike.eval_stats`` rather than re-derived (the repo has neither scipy nor
sklearn, and a second copy of the arithmetic is exactly what the reports must not
drift from).
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

import ml.zernike.cv_harness as cv_harness
from ml.zernike.cv_harness import (
    build_folds,
    min_attainable_pvalue_folds,
    paired_fold_report,
)


def _synthetic_folds(k: int) -> list[tuple[list[int], list[int]]]:
    """A trivial ``k``-fold partition over ``range(k)`` for report-shape tests."""
    return [
        ([j for j in range(k) if j != i], [i])
        for i in range(k)
    ]


class TestBuildFolds:
    def test_leave_one_out_partitions_every_index_once(self):
        folds = build_folds(8, 8, seed=0)
        assert len(folds) == 8

        seen: list[int] = []
        for train, test in folds:
            assert len(test) == 1
            assert len(train) == 7  # n - one held-out member
            assert set(train).isdisjoint(test)
            assert sorted(train + test) == list(range(8))
            seen.extend(test)

        assert sorted(seen) == list(range(8))

    def test_four_folds_partition_all_items_exactly_once(self):
        folds = build_folds(8, 4, seed=0)
        assert len(folds) == 4

        all_test: list[int] = []
        for train, test in folds:
            assert set(train).isdisjoint(test)
            assert sorted(train + test) == list(range(8))
            all_test.extend(test)

        assert sorted(all_test) == list(range(8))

    def test_deterministic_for_a_fixed_seed(self):
        assert build_folds(8, 4, seed=7) == build_folds(8, 4, seed=7)

    def test_differs_across_seeds(self):
        assert build_folds(8, 4, seed=0) != build_folds(8, 4, seed=1)

    @pytest.mark.parametrize("k", [1, 0])
    def test_k_below_two_raises(self, k):
        with pytest.raises(ValueError):
            build_folds(8, k, seed=0)

    def test_k_above_n_items_raises(self):
        with pytest.raises(ValueError):
            build_folds(8, 9, seed=0)


class TestMinAttainablePvalueFolds:
    def test_eight_folds_floor(self):
        assert min_attainable_pvalue_folds(8) == 2 / 2**8
        assert min_attainable_pvalue_folds(8) == 0.0078125

    def test_four_folds_floor_is_above_five_percent(self):
        assert min_attainable_pvalue_folds(4) == 0.125
        assert min_attainable_pvalue_folds(4) > 0.05


class TestPairedFoldReport:
    def test_separated_arms_low_pvalue_and_positive_effect(self):
        folds = _synthetic_folds(8)
        baseline = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24]
        treatment = [0.30, 0.33, 0.36, 0.39, 0.42, 0.45, 0.48, 0.51]
        report = paired_fold_report(
            {"baseline": baseline, "treatment": treatment}, folds
        )

        comparison = report["comparison"]
        # treatment minus baseline, all positive -> positive paired effect size.
        assert comparison["delta_mean"] == pytest.approx(np.mean(
            np.asarray(treatment) - np.asarray(baseline)
        ))
        assert comparison["cohens_dz"] > 0
        assert comparison["pvalue"] < 0.05
        assert comparison["pvalue"] >= report["min_attainable_pvalue"]

    def test_identical_arms_have_zero_delta_and_no_effect(self):
        folds = _synthetic_folds(8)
        arm = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        report = paired_fold_report({"a": arm, "b": list(arm)}, folds)

        comparison = report["comparison"]
        assert comparison["delta_mean"] == 0.0
        assert comparison["pvalue"] == 1.0
        # NOTE: the task spec said "cohens_dz is 0" here, but the real
        # ml.zernike.eval_stats.cohens_dz deliberately returns NaN when the paired
        # differences have zero mean AND zero spread ("nothing to compare"). We
        # reuse that contract rather than re-deriving the statistic.
        assert np.isnan(comparison["cohens_dz"])

    def test_surfaces_the_structural_floor(self):
        folds = _synthetic_folds(5)
        report = paired_fold_report({"a": [0.0] * 5, "b": [1.0] * 5}, folds)
        assert report["min_attainable_pvalue"] == 2 / 2 ** len(folds)
        assert report["n_folds"] == len(folds)
        assert report["paired_on"] == "fold"

    def test_uniform_positive_diffs_hit_the_floor(self):
        folds = _synthetic_folds(8)
        report = paired_fold_report({"a": [0.0] * 8, "b": [0.5] * 8}, folds)
        comparison = report["comparison"]
        assert comparison["pvalue"] == report["min_attainable_pvalue"]
        assert comparison["pvalue_is_at_floor"] is True

    def test_per_arm_summary_fields(self):
        folds = _synthetic_folds(4)
        report = paired_fold_report(
            {"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 3.0, 4.0, 5.0]}, folds
        )
        assert report["a"]["mean"] == pytest.approx(2.5)
        assert report["a"]["per_fold"] == [1.0, 2.0, 3.0, 4.0]
        assert report["b"]["mean"] == pytest.approx(3.5)
        assert report["a"]["sd"] > 0

    def test_single_arm_has_no_comparison(self):
        folds = _synthetic_folds(4)
        report = paired_fold_report({"only": [1.0, 2.0, 3.0, 4.0]}, folds)
        assert "comparison" not in report

    def test_paired_on_seed_is_accepted_with_a_warning_note(self):
        folds = _synthetic_folds(4)
        report = paired_fold_report(
            {"a": [0.1, 0.2, 0.3, 0.4], "b": [0.2, 0.3, 0.4, 0.5]},
            folds,
            paired_on="seed",
        )
        assert report["paired_on"] == "seed"
        assert "seed_note" in report
        assert "fold-difficulty" in report["seed_note"]

    def test_invalid_paired_on_raises(self):
        folds = _synthetic_folds(4)
        with pytest.raises(ValueError):
            paired_fold_report(
                {"a": [0.1, 0.2, 0.3, 0.4], "b": [0.2, 0.3, 0.4, 0.5]},
                folds,
                paired_on="arm",
            )

    def test_arm_length_must_match_fold_count(self):
        folds = _synthetic_folds(4)
        with pytest.raises(ValueError):
            paired_fold_report({"a": [0.1, 0.2, 0.3], "b": [0.2, 0.3, 0.4]}, folds)


class TestReusesEvalStats:
    """Pins the 'never re-derive statistics' rule by inspecting the source AST."""

    def test_imports_statistics_from_eval_stats(self):
        source = Path(cv_harness.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)

        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "ml.zernike.eval_stats":
                imported.update(alias.name for alias in node.names)

        assert {"sign_flip_pvalue", "cohens_dz", "min_attainable_pvalue"} <= imported
