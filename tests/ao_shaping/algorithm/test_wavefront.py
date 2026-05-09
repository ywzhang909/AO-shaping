import numpy as np
import pytest

from ao_shaping.algorithm.wavefront import (
    zernike_piston_tilt,
    build_D_vectorized,
    reconstruct_wavefront,
    laplacian_2d,
    slope_to_rhs,
)


class TestZernikePistonTilt:
    def test_shape(self):
        N = 10
        Z = zernike_piston_tilt(N)
        assert Z.shape == (3, (N + 1) ** 2)

    def test_three_bases(self):
        Z = zernike_piston_tilt(5)
        assert Z.shape[0] == 3

    def test_orthogonality(self):
        Z = zernike_piston_tilt(10)
        gram = Z @ Z.T
        off_diag = gram - np.diag(np.diag(gram))
        assert np.max(np.abs(off_diag)) < 0.1


class TestBuildDVectorized:
    def test_shape(self):
        N = 5
        D = build_D_vectorized(N)
        M = N * N
        K = (N + 1) * (N + 1)
        assert D.shape == (2 * M, K)

    def test_sparse_format(self):
        D = build_D_vectorized(5)
        assert hasattr(D, "toarray")

    def test_row_sums_near_zero(self):
        D = build_D_vectorized(4).toarray()
        row_sums = D.sum(axis=1)
        assert np.allclose(row_sums, 0, atol=1e-10)


class TestReconstructWavefront:
    def test_flat_slopes_give_zero_rms(self):
        N = 8
        sx = np.zeros((N, N))
        sy = np.zeros((N, N))
        phi, rms = reconstruct_wavefront(sx, sy, remove_piston_tilt=True)
        assert rms < 1e-10

    def test_returns_correct_shape(self):
        N = 5
        sx = np.random.randn(N, N) * 0.01
        sy = np.random.randn(N, N) * 0.01
        phi, rms = reconstruct_wavefront(sx, sy)
        assert phi.shape == (N + 1, N + 1)
        assert isinstance(rms, float)

    def test_tilt_removal(self):
        N = 8
        x = np.linspace(-1, 1, N)
        X, _ = np.meshgrid(x, x)
        sx = np.ones((N, N)) * 0.1
        sy = np.zeros((N, N))
        _, rms_with = reconstruct_wavefront(sx, sy, remove_piston_tilt=False)
        _, rms_without = reconstruct_wavefront(sx, sy, remove_piston_tilt=True)
        assert rms_without < rms_with


class TestLaplacian2d:
    def test_shape(self):
        N = 5
        L = laplacian_2d(N)
        assert L.shape == (N * N, N * N)

    def test_first_diagonal_negative(self):
        L = laplacian_2d(5).toarray()
        assert L[0, 0] < 0

    def test_has_zero_eigenvalues(self):
        L = laplacian_2d(5).toarray()
        eigs = np.linalg.eigvalsh(L)
        assert np.min(np.abs(eigs)) < 1e-10


class TestSlopeToRhs:
    def test_shape(self):
        N = 8
        sx = np.random.randn(N, N)
        sy = np.random.randn(N, N)
        rho = slope_to_rhs(sx, sy)
        assert rho.shape == (N * N,)

    def test_zero_slopes_give_zero_rhs(self):
        N = 5
        sx = np.zeros((N, N))
        sy = np.zeros((N, N))
        rho = slope_to_rhs(sx, sy)
        assert np.allclose(rho, 0, atol=1e-10)
