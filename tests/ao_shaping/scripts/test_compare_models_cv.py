"""Tests for the grouped cross-validation harness.

The permutation test is the load-bearing piece and had a real bug during
development: comparing ``|sum(+-d)|`` against ``sum(|d|)`` makes every p-value hit
the ``2/2**n`` floor, because only two sign-flips can align every term. These
tests pin the statistic so that cannot come back.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from compare_models_cv import (  # noqa: E402
    build_folds,
    exact_sign_flip_p,
    holm_bonferroni,
    objective_of,
)


class _Record:
    """Minimal stand-in exposing only what the fold builder reads."""

    def __init__(self, path: Path) -> None:
        self.path = path


def _corpus() -> list:
    """Two timestamps for three objectives -- 6 pickles, uneven group sizes."""
    names = [
        "slm_zernike_shaping_rms_pib_20260101_000000.pkl",
        "slm_zernike_shaping_rms_pib_20260101_010000.pkl",
        "slm_zernike_shaping_roi_pib_20260101_000000.pkl",
        "slm_zernike_shaping_shape_20260101_000000.pkl",
        "slm_zernike_shaping_shape_20260101_010000.pkl",
        "slm_zernike_shaping_roi_pib_20260101_010000.pkl",
    ]
    return [_Record(Path(n)) for n in names]


class TestObjectiveOf:
    def test_extracts_the_objective_between_family_and_timestamp(self) -> None:
        path = Path("slm_zernike_shaping_rms_pib_20260926_162917_20260926_162917.pkl")
        assert objective_of(path) == "rms_pib"

    @pytest.mark.parametrize(
        "stem,expected",
        [
            ("slm_zernike_shaping_rmse_out_20260926_163121_x.pkl", "rmse_out"),
            ("slm_zernike_shaping_roi_pib_20260926_171311_x.pkl", "roi_pib"),
        ],
    )
    def test_handles_every_objective(self, stem: str, expected: str) -> None:
        assert objective_of(Path(stem)) == expected

    def test_rejects_a_name_it_cannot_parse(self) -> None:
        with pytest.raises(ValueError, match="cannot parse an objective"):
            objective_of(Path("some_other_family_20260101_000000.pkl"))


class TestBuildFolds:
    def test_objective_protocol_keeps_every_record_exactly_once(self) -> None:
        records = _corpus()
        folds = build_folds(records, "objective")
        seen: list[int] = []
        for fold in folds:
            assert not ({objective_of(records[p].path) for p in fold.train}
                        & {objective_of(records[p].path) for p in fold.val})
            seen += fold.val
        assert sorted(seen) == list(range(len(records)))

    def test_objective_protocol_yields_one_fold_per_objective(self) -> None:
        folds = build_folds(_corpus(), "objective")
        assert len(folds) == 3
        # Each objective appears in exactly one validation set.
        assert len({next(iter(f.label.split()[2:])) for f in folds}) == 3

    def test_file_protocol_is_leave_one_pickle_out(self) -> None:
        records = _corpus()
        folds = build_folds(records, "file")
        assert len(folds) == len(records)
        for fold in folds:
            assert len(fold.val) == 1
            assert len(fold.train) == len(records) - 1

    def test_file_protocol_validates_every_record_exactly_once(self) -> None:
        """The property that removes the objective-mixture lottery."""
        records = _corpus()
        seen: list[int] = []
        for fold in build_folds(records, "file"):
            seen += fold.val
        assert sorted(seen) == list(range(len(records)))

    def test_rejects_an_unknown_protocol(self) -> None:
        with pytest.raises(ValueError, match="unknown protocol"):
            build_folds(_corpus(), "nonsense")


class TestExactSignFlip:
    def test_uniformly_signed_sample_reaches_the_two_over_n_floor(self) -> None:
        # Every sign-flip can only reach the observed |sum| in two of 2**4 ways.
        assert exact_sign_flip_p([1.0, 2.0, 3.0, 4.0]) == pytest.approx(2 / 16)

    def test_ten_folded_uniform_sample_has_the_smallest_attainable_p(self) -> None:
        assert exact_sign_flip_p([1.0] * 10) == pytest.approx(2 / 1024)

    def test_mixed_signs_are_not_floored(self) -> None:
        """Regression guard for the statistic bug.

        The broken implementation returned the 2/2**n floor for *every* input
        because it compared against the sum of absolute values. A mixed sample
        must come out strictly above the floor.
        """
        mixed = [0.11, 0.05, -0.02, 0.03, 0.04, -0.01, 0.02, 0.01, 0.09, 0.10]
        assert exact_sign_flip_p(mixed) > 2 / 1024

    def test_symmetry_under_sign_reversal(self) -> None:
        diffs = [0.3, -0.1, 0.2, 0.05, -0.4]
        assert exact_sign_flip_p(diffs) == pytest.approx(exact_sign_flip_p([-d for d in diffs]))

    def test_all_zero_gives_no_evidence(self) -> None:
        assert exact_sign_flip_p([0.0] * 10) == 1.0

    def test_single_nonzero_difference_cannot_be_significant(self) -> None:
        assert exact_sign_flip_p([0.0] * 9 + [0.5]) == 1.0

    def test_metric_matters_so_results_are_not_a_constant(self) -> None:
        """Two different metrics must not collapse onto the same p-value."""
        strong = [0.11, 0.05, -0.02, 0.03, 0.04, -0.01, 0.02, 0.01, 0.09, 0.10]
        weak = [0.01, 0.01, -0.01, 0.01, 0.0, -0.01, 0.01, 0.0, 0.01, -0.01]
        assert exact_sign_flip_p(strong) != exact_sign_flip_p(weak)


class TestHolmBonferroni:
    def test_is_monotone_and_never_below_the_raw_p(self) -> None:
        raw = {"a": 0.01, "b": 0.04, "c": 0.03}
        adjusted = holm_bonferroni(raw)
        # Sorted a(0.01) c(0.03) b(0.04); multipliers 3, 2, 1 give 0.03, 0.06,
        # 0.04 -- but Holm is step-*down*, so the running maximum carries b up to
        # 0.06 rather than letting it drop back.
        assert adjusted["a"] == pytest.approx(0.03)
        assert adjusted["c"] == pytest.approx(0.06)
        assert adjusted["b"] == pytest.approx(0.06)
        assert all(adjusted[k] >= raw[k] for k in raw)

    def test_clamps_to_one(self) -> None:
        assert holm_bonferroni({"a": 0.9, "b": 0.95})["b"] == 1.0

    def test_preserves_keys(self) -> None:
        assert set(holm_bonferroni({"x": 0.2})) == {"x"}