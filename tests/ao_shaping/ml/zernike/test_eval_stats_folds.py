"""Locks the folds_of / ridge primitives migrated into ml.zernike.eval_stats.

The ridge reference is the *primal* normal-equations solve built explicitly
(augmented intercept system), which is an independent derivation from the dual
Gram form the implementation uses. The scipy cross-check at the bottom is
optional: the repo ships no scipy dependency, so it is skipped when absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml.zernike.eval_stats import folds_of, ridge


class _Rec:
    def __init__(self, path, family):
        self.path = path
        self.family = family


class TestFoldsOf:
    def test_empty_and_single(self):
        assert folds_of([]) == []
        (fold,) = folds_of([_Rec("a.pkl", "f")])
        assert fold == (0, "a.pkl", [], [0])

    def test_partition_is_exact_permutation(self):
        recs = [
            _Rec("b.pkl", "f1"), _Rec("a.pkl", "f2"), _Rec("b.pkl", "f3"),
            _Rec("c.pkl", "f1"), _Rec("a.pkl", "f1"),
        ]
        folds = folds_of(recs)
        assert len(folds) == 3
        for index, key, train, val in folds:
            assert sorted(train + val) == list(range(len(recs)))
            assert not set(train) & set(val)
            assert all(str(recs[p].path) == key for p in val)
            assert all(str(recs[p].path) != key for p in train)

    def test_fold_order_is_sorted_keys_and_index_matches_position(self):
        recs = [_Rec("z.pkl", "f"), _Rec("a.pkl", "f"), _Rec("m.pkl", "f")]
        folds = folds_of(recs)
        assert [k for _, k, _, _ in folds] == ["a.pkl", "m.pkl", "z.pkl"]
        assert [i for i, _, _, _ in folds] == [0, 1, 2]

    def test_path_and_str_group_together(self):
        # str(record.path) is the key: a Path and its string form are one group.
        recs = [_Rec(Path("d/a.pkl"), "f"), _Rec("d/a.pkl", "f"), _Rec("d/b.pkl", "f")]
        folds = folds_of(recs)
        assert [k for _, k, _, _ in folds] == ["d/a.pkl", "d/b.pkl"]
        assert folds[0][3] == [0, 1]

    def test_family_is_not_the_grouping_key(self):
        recs = [_Rec("a.pkl", "f1"), _Rec("a.pkl", "f2"), _Rec("b.pkl", "f1")]
        folds = folds_of(recs)
        assert [k for _, k, _, _ in folds] == ["a.pkl", "b.pkl"]

    def test_train_positions_follow_group_first_insertion_order(self):
        recs = [_Rec("a.pkl", "f"), _Rec("b.pkl", "f"), _Rec("a.pkl", "f"),
                _Rec("c.pkl", "f"), _Rec("b.pkl", "f")]
        folds = folds_of(recs)
        # held "a.pkl": train concatenates group b then group c in insertion order
        assert next(tr for _, k, tr, _ in folds if k == "a.pkl") == [1, 4, 3]


class TestRidge:
    @staticmethod
    def _primal_reference(x, z, alpha):
        """Independent closed form: intercept-augmented primal normal equations.

        Solve [(Xc^T Xc + aI) | ...] in feature space with an unpenalised
        intercept handled by centring -- dual form must match this to
        floating-point noise even when n << d.
        """
        xm = x.mean(0, keepdims=True)
        zm = z.mean(0, keepdims=True)
        xc, zc = x - xm, z - zm
        d = xc.shape[1]
        w = xc.T @ np.linalg.solve(xc @ xc.T + alpha * np.eye(xc.shape[0]), zc)
        # primal equivalent: (Xc^T Xc + alpha I) w = Xc^T zc (same w by the
        # dual identity; computed through the d x d system here).
        w_primal = np.linalg.solve(xc.T @ xc + alpha * np.eye(d), xc.T @ zc)
        assert np.allclose(w, w_primal, atol=1e-8)
        return w_primal, xm, zm

    def test_closed_form_matches_primal_reference(self):
        rng = np.random.default_rng(11)
        x = rng.standard_normal((10, 4))
        z = rng.standard_normal((10, 3))
        for alpha in (1.0, 0.5, 1e-3):
            w, xm, zm = ridge(x, z, alpha)
            w_ref, xm_ref, zm_ref = self._primal_reference(x, z, alpha)
            assert np.abs(w - w_ref).max() <= 1e-10
            assert np.abs(xm - xm_ref).max() == 0.0
            assert np.abs(zm - zm_ref).max() == 0.0
            # exact for the noiseless linear case
            x2 = rng.standard_normal((6, 4))
            pred = (x2 - xm) @ w + zm
            assert np.abs(pred - ((x2 - xm) @ w_ref + zm_ref)).max() <= 1e-10

    def test_perfect_recovery_with_zero_alpha_full_rank(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal((15, 4))
        w_true = rng.standard_normal((4, 2))
        z = x @ w_true
        w, xm, zm = ridge(x, z, 0.0)
        pred = (x - xm) @ w + zm
        assert np.abs(pred - z).max() < 1e-8

    def test_default_alpha_is_one(self):
        rng = np.random.default_rng(5)
        x, z = rng.standard_normal((8, 3)), rng.standard_normal((8, 1))
        a, b = ridge(x, z), ridge(x, z, 1.0)
        assert all(np.array_equal(p, q) for p, q in zip(a, b))

    def test_collinear_rank_deficient_alpha_zero_raises(self):
        v = np.linspace(-1.0, 1.0, 9)
        x = np.column_stack([v, 2.0 * v, v - 3.0])  # rank 1
        z = np.column_stack([v * 2.0])
        with pytest.raises(np.linalg.LinAlgError):
            ridge(x, z, 0.0)
        w, xm, zm = ridge(x, z, 1.0)  # regularised: solvable
        assert np.isfinite(w).all()

    def test_zero_variance_column_alpha_zero_raises(self):
        x = np.column_stack([np.arange(6.0), np.full(6, 2.5)])
        z = np.arange(6.0)[:, None]
        with pytest.raises(np.linalg.LinAlgError):
            ridge(x, z, 0.0)

    def test_single_row_alpha_zero_raises(self):
        with pytest.raises(np.linalg.LinAlgError):
            ridge(np.array([[1.0, 2.0]]), np.array([[3.0]]), 0.0)

    def test_inputs_never_mutated(self):
        rng = np.random.default_rng(9)
        x, z = rng.standard_normal((7, 3)), rng.standard_normal((7, 2))
        x0, z0 = x.copy(), z.copy()
        ridge(x, z, 1.0)
        assert np.array_equal(x, x0) and np.array_equal(z, z0)

    def test_multioutput_shapes(self):
        rng = np.random.default_rng(13)
        w, xm, zm = ridge(rng.standard_normal((6, 20)), rng.standard_normal((6, 4)), 0.7)
        assert w.shape == (20, 4) and xm.shape == (1, 20) and zm.shape == (1, 4)


def test_ridge_matches_scipy_when_available():
    scipy_linalg = pytest.importorskip("scipy.linalg")
    rng = np.random.default_rng(17)
    x = rng.standard_normal((9, 5))
    z = rng.standard_normal((9, 2))
    alpha = 0.25
    xm, zm = x.mean(0, keepdims=True), z.mean(0, keepdims=True)
    xc, zc = x - xm, z - zm
    w_scipy = xc.T @ scipy_linalg.solve(xc @ xc.T + alpha * np.eye(9), zc)
    w, _, _ = ridge(x, z, alpha)
    assert np.abs(w - w_scipy).max() <= 1e-10
