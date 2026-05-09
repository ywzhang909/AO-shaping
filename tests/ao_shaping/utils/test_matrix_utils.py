"""Tests for ao_shaping.utils.matrix_utils — pure math, no hardware."""

import numpy as np
import pytest

from ao_shaping.utils.matrix_utils import (
    calc_n_zernike_terms,
    compute_lstsq,
    compute_pinv,
    index_to_noll,
    noll_to_index,
)


class TestComputePinv:
    def test_square_matrix(self):
        A = np.random.randn(5, 5)
        pinv_A = compute_pinv(A)
        assert pinv_A.shape == (5, 5)
        assert np.allclose(A @ pinv_A @ A, A, atol=1e-10)

    def test_rectangular_overdetermined(self):
        A = np.random.randn(10, 5)
        pinv_A = compute_pinv(A)
        assert pinv_A.shape == (5, 10)
        assert np.allclose(A @ pinv_A @ A, A, atol=1e-10)

    def test_rectangular_underdetermined(self):
        A = np.random.randn(3, 8)
        pinv_A = compute_pinv(A)
        assert pinv_A.shape == (8, 3)

    def test_rcond_parameter(self):
        A = np.array([[1, 0], [0, 1e-12]])
        pinv_A = compute_pinv(A, rcond=1e-10)
        assert pinv_A.shape == (2, 2)


class TestComputeLstsq:
    def test_square_matrix(self):
        A = np.random.randn(5, 5)
        lstsq_A = compute_lstsq(A)
        expected = np.linalg.inv(A)
        assert np.allclose(lstsq_A, expected, atol=1e-10)

    def test_rectangular_overdetermined(self):
        A = np.random.randn(10, 5)
        lstsq_A = compute_lstsq(A)
        assert lstsq_A.shape == (5, 10)
        assert np.allclose(A @ lstsq_A @ A, A, atol=1e-10)

    def test_rectangular_underdetermined(self):
        A = np.random.randn(3, 8)
        lstsq_A = compute_lstsq(A)
        assert lstsq_A.shape == (8, 3)


class TestCalcNZernikeTerms:
    def test_n_max_0(self):
        assert calc_n_zernike_terms(0) == 1

    def test_n_max_1(self):
        assert calc_n_zernike_terms(1) == 3

    def test_n_max_2(self):
        assert calc_n_zernike_terms(2) == 6

    def test_n_max_4(self):
        assert calc_n_zernike_terms(4) == 15

    def test_n_max_10(self):
        assert calc_n_zernike_terms(10) == 66


class TestNollIndexConversion:
    def test_noll_to_index(self):
        assert noll_to_index(1) == 0
        assert noll_to_index(2) == 1
        assert noll_to_index(66) == 65

    def test_index_to_noll(self):
        assert index_to_noll(0) == 1
        assert index_to_noll(1) == 2
        assert index_to_noll(65) == 66

    def test_round_trip(self):
        for j in range(1, 67):
            assert index_to_noll(noll_to_index(j)) == j
