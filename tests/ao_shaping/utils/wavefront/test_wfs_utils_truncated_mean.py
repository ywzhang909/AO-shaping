"""`truncated_mean` -- the rule the matrix calibrators each re-implemented.

Four call sites (two in ``dm_response_matrix``, two in ``zernike_response_matrix``) inlined
"sort the readings along axis 0, drop the first and last, average the rest", gated on more
than five readings. Only one of the four files explained the rule in prose, so the copies had
already drifted in *documentation* even while agreeing in behaviour.

The differential test is the one that matters for the refactor: it compares the helper against
the literal expression it replaces, exactly, so a convergence cannot quietly change a number.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.utils.wavefront.wfs_utils import truncated_mean


class TestSemantics:
    def test_plain_mean_at_or_below_five_rows(self) -> None:
        readings = np.arange(5 * 3, dtype=float).reshape(5, 3)
        np.testing.assert_array_equal(truncated_mean(readings), np.mean(readings, axis=0))

    def test_drops_one_extreme_per_side_at_six_rows(self) -> None:
        readings = np.arange(6 * 3, dtype=float).reshape(6, 3)
        # Row-major fill, so column 0 is [0, 3, 6, 9, 12, 15]: dropping the extreme
        # readings leaves 3+6+9+12 = 30 over 4 rows -> 7.5.
        assert truncated_mean(readings)[0] == pytest.approx(7.5)

    def test_extremes_are_dropped_per_column_not_per_row(self) -> None:
        # Every column's minimum AND maximum sit in a different row, so any row-wise
        # drop necessarily leaves a 0 or a 12 in some column, while a column-wise drop
        # leaves four 10s in every column.
        readings = np.array(
            [
                [0.0, 10.0, 10.0],
                [10.0, 0.0, 10.0],
                [10.0, 10.0, 0.0],
                [10.0, 12.0, 10.0],
                [10.0, 10.0, 12.0],
                [12.0, 10.0, 10.0],
            ]
        )
        result = truncated_mean(readings)
        np.testing.assert_array_equal(result, np.array([10.0, 10.0, 10.0]))

    def test_handles_the_reading_shapes_the_sites_produce(self) -> None:
        # zernike sites stack (n, 67); dm sites stack (n, ny, nx).
        assert truncated_mean(np.zeros((7, 67))).shape == (67,)
        assert truncated_mean(np.zeros((7, 8, 8))).shape == (8, 8)


class TestDifferentialAgainstTheReplacedExpression:
    @pytest.mark.parametrize("n", [6, 7, 11, 40])
    @pytest.mark.parametrize("shape", [(67,), (8, 8), (4, 6)])
    def test_above_the_gate_equals_sort_drop_average(self, n: int, shape: tuple) -> None:
        rng = np.random.default_rng(n)
        readings = rng.normal(size=(n, *shape))
        old = np.mean(np.sort(readings, axis=0)[1:-1], axis=0)
        np.testing.assert_array_equal(truncated_mean(readings), old)

    @pytest.mark.parametrize("n", [1, 3, 5])
    def test_at_or_below_the_gate_equals_plain_mean(self, n: int) -> None:
        rng = np.random.default_rng(n)
        readings = rng.normal(size=(n, 67))
        np.testing.assert_array_equal(truncated_mean(readings), np.mean(readings, axis=0))

    def test_accepts_a_list_of_readings_like_the_sites_do(self) -> None:
        readings = [np.full(3, float(i)) for i in range(7)]
        np.testing.assert_array_equal(
            truncated_mean(readings), np.mean(np.array(readings), axis=0)
        )
