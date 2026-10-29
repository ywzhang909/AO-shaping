"""Locks the pearson / spearman primitives in ml.zernike.eval_stats.

Reference values are derived by hand, independently of the implementation; the
derivations are spelled out in the comments so a reviewer can check the arithmetic
without running anything. The scipy cross-check at the bottom is optional: the repo
ships no scipy dependency, so it is skipped when scipy is absent.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ml.zernike.eval_stats import pearson, spearman


class TestPearson:
    def test_identity_is_one(self):
        # b == a => centred vectors coincide => cos angle = 1.
        assert pearson([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_negation_is_minus_one(self):
        assert pearson([1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]) == pytest.approx(-1.0)

    def test_known_imperfect_case(self):
        # x = [1,2,3,4,5], y = [2,1,4,3,5]; both means are 3.
        # centred x = [-2,-1,0,1,2]; centred y = [-1,-2,+1,0,+2].
        #   dot                = 2 + 2 + 0 + 0 + 4 = 8
        #   ||x_c||^2 = 4+1+0+1+4 = 10;  ||y_c||^2 = 1+4+1+0+4 = 10
        #   r = 8 / sqrt(10*10) = 0.8.
        assert pearson([1, 2, 3, 4, 5], [2, 1, 4, 3, 5]) == pytest.approx(0.8)

    def test_scale_and_shift_invariance(self):
        a = [1.0, 2.0, 4.0, 7.0]
        b = [3.0, 5.0, 6.0, 9.0]
        assert pearson(a, b) == pytest.approx(pearson([2 * v + 10 for v in a], b))

    def test_constant_vector_is_nan_not_zero(self):
        # Degenerate: no correlation direction. nan (like modern scipy), never 0.0.
        assert math.isnan(pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))
        assert math.isnan(pearson([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]))

    @pytest.mark.parametrize("n", [0, 1])
    def test_fewer_than_two_points_is_nan(self, n):
        assert math.isnan(pearson(list(range(n)), list(range(n))))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            pearson([1, 2], [1, 2, 3])

    def test_two_dimensional_input_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            pearson([[1, 2], [3, 4]], [[1, 2], [3, 4]])

    def test_inputs_not_mutated(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([3.0, 2.0, 1.0])
        xb, yb = x.copy(), y.copy()
        pearson(x, y)
        np.testing.assert_array_equal(x, xb)
        np.testing.assert_array_equal(y, yb)

    def test_python_lists_and_int_arrays_accepted(self):
        assert pearson([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


class TestSpearman:
    def test_monotone_transform_is_one(self):
        # Any strictly increasing map => identical ranks => rho = 1.
        assert spearman([1, 2, 3, 4], [1, 4, 9, 16]) == pytest.approx(1.0)

    def test_perfectly_decreasing_is_minus_one(self):
        assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)

    def test_no_tie_case_by_sum_of_squared_rank_differences(self):
        # x = [1..5] ranks [1,2,3,4,5]; y = [2,1,4,3,5] ranks [2,1,4,3,5].
        # d = [-1,1,-1,1,0] => sum d^2 = 4;
        # rho = 1 - 6*4 / (5*(25-1)) = 1 - 24/120 = 0.8.
        assert spearman([1, 2, 3, 4, 5], [2, 1, 4, 3, 5]) == pytest.approx(0.8)

    def test_heavy_ties_use_average_ranks(self):
        # x = [1,2,2,3,4]: sorted positions of the two 2s are 2 and 3 => both rank 2.5.
        #   rank(x) = [1, 2.5, 2.5, 4, 5]
        # y = [1,3,3,2,4]: the two 3s occupy positions 3 and 4 => both rank 3.5.
        #   rank(y) = [1, 3.5, 3.5, 2, 5]
        # Both rank vectors have mean 3; centred:
        #   xc = [-2,-0.5,-0.5, 1, 2],  yc = [-2, 0.5, 0.5,-1, 2]
        #   dot = 4 - 0.25 - 0.25 - 1 + 4 = 6.5;  ||xc||^2 = ||yc||^2 = 9.5
        #   rho = 6.5 / 9.5 = 13/19.
        # (The sum-of-d^2 shortcut does NOT apply here -- ties break it.)
        rho = spearman([1, 2, 2, 3, 4], [1, 3, 3, 2, 4])
        assert rho == pytest.approx(13.0 / 19.0)

    def test_equals_pearson_on_average_ranks(self):
        # The identity the implementation is built on, checked on tie-heavy draws.
        from ml.zernike.eval_stats import _average_rank

        rng = np.random.default_rng(7)
        for _ in range(20):
            a = rng.integers(0, 5, size=15).astype(float)
            b = rng.integers(0, 5, size=15).astype(float)
            assert spearman(a, b) == pytest.approx(
                pearson(_average_rank(a), _average_rank(b))
            )

    def test_constant_vector_is_nan(self):
        assert math.isnan(spearman([1, 2, 3], [7, 7, 7]))

    @pytest.mark.parametrize("n", [0, 1])
    def test_fewer_than_two_points_is_nan(self, n):
        assert math.isnan(spearman(list(range(n)), list(range(n))))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            spearman([1, 2], [1, 2, 3])

    def test_two_dimensional_input_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            spearman([[1], [2]], [[1], [2]])

    def test_inputs_not_mutated(self):
        x = np.array([3.0, 1.0, 2.0])
        y = np.array([1.0, 1.0, 2.0])
        xb, yb = x.copy(), y.copy()
        spearman(x, y)
        np.testing.assert_array_equal(x, xb)
        np.testing.assert_array_equal(y, yb)


class TestScipyCrossCheck:
    """Optional: skip cleanly when scipy is absent (the repo does not depend on it)."""

    CASES = [
        ([1, 2, 3, 4, 5], [2, 1, 4, 3, 5]),
        ([1, 2, 2, 3, 4], [1, 3, 3, 2, 4]),
        ([5, 5, 5], [1, 2, 3]),
        ([3.2, -1.0, 7.7, 0.4, -9.1, 2.2], [1.1, 6.6, -2.3, 4.0, 8.8, -0.5]),
    ]

    @pytest.mark.parametrize(("a", "b"), CASES)
    def test_pearson_matches_scipy(self, a, b):
        stats = pytest.importorskip("scipy.stats")

        ref = float(stats.pearsonr(a, b)[0])
        if math.isnan(ref):
            assert math.isnan(pearson(a, b))
        else:
            assert pearson(a, b) == pytest.approx(ref)

    @pytest.mark.parametrize(("a", "b"), CASES)
    def test_spearman_matches_scipy(self, a, b):
        stats = pytest.importorskip("scipy.stats")

        ref = float(stats.spearmanr(a, b)[0])
        if math.isnan(ref):
            assert math.isnan(spearman(a, b))
        else:
            assert spearman(a, b) == pytest.approx(ref)
