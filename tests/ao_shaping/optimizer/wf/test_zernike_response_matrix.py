"""ZernikeResponseMatrix测试 - 使用uv run执行"""

import tempfile
from pathlib import Path

import numpy as np
import pytest


def calc_n_zernike_terms(n_max: int) -> int:
    """计算n_max对应的Zernike项数"""
    count = 0
    for n in range(n_max + 1):
        for m in range(-n, n + 1, 2):
            count += 1
    return count


def noll_to_index(j: int) -> int:
    return j - 1


def index_to_noll(i: int) -> int:
    return i + 1


class TestZernikeTerms:
    def test_calc_n_zernike_terms(self):
        assert calc_n_zernike_terms(0) == 1
        assert calc_n_zernike_terms(1) == 3
        assert calc_n_zernike_terms(2) == 6
        assert calc_n_zernike_terms(4) == 15
        assert calc_n_zernike_terms(10) == 66

    def test_noll_index_conversion(self):
        assert noll_to_index(1) == 0
        assert noll_to_index(2) == 1
        assert noll_to_index(66) == 65

        assert index_to_noll(0) == 1
        assert index_to_noll(1) == 2
        assert index_to_noll(65) == 66

    def test_round_trip_noll_index(self):
        for j in range(1, 67):
            assert index_to_noll(noll_to_index(j)) == j


class TestInverseMatrixComputation:
    """测试逆矩阵计算函数 (使用utils模块，避免硬件导入链)"""

    def test_compute_pinv_square(self):
        """测试方阵的SVD伪逆"""
        from ao_shaping.utils.matrix_utils import compute_pinv

        A = np.random.randn(5, 5)
        pinv_A = compute_pinv(A)

        # 验证 A @ pinv_A @ A ≈ A
        result = A @ pinv_A @ A
        assert np.allclose(result, A, atol=1e-10)

    def test_compute_pinv_rectangular(self):
        """测试非方阵的SVD伪逆"""
        from ao_shaping.utils.matrix_utils import compute_pinv

        A = np.random.randn(10, 5)
        pinv_A = compute_pinv(A)

        result = A @ pinv_A @ A
        assert np.allclose(result, A, atol=1e-10)
        assert pinv_A.shape == (5, 10)

    def test_compute_lstsq_square(self):
        """测试方阵的最小二乘逆"""
        from ao_shaping.utils.matrix_utils import compute_lstsq

        A = np.random.randn(5, 5)
        lstsq_A = compute_lstsq(A)

        expected = np.linalg.inv(A)
        assert np.allclose(lstsq_A, expected, atol=1e-10)

    def test_compute_lstsq_rectangular(self):
        """测试非方阵的最小二乘逆"""
        from ao_shaping.utils.matrix_utils import compute_lstsq

        A = np.random.randn(10, 5)
        lstsq_A = compute_lstsq(A)

        assert lstsq_A.shape == (5, 10)
        result = A @ lstsq_A @ A
        assert np.allclose(result, A, atol=1e-10)


class TestZernikeResponseMatrixResult:
    """测试ZernikeResponseMatrixResult数据类"""

    def test_dataclass_creation(self):
        """测试数据类创建"""
        from ao_shaping.optimizer.wf.zernike_response_matrix import (
            ZernikeResponseMatrixResult,
            calc_n_zernike_terms,
        )

        n_max = 10
        n_wfs = calc_n_zernike_terms(10) - 1
        n_slm = calc_n_zernike_terms(n_max) - 1

        matrix = np.random.randn(n_wfs, n_slm)
        variance = np.abs(np.random.randn(n_wfs, n_slm)) * 0.01

        result = ZernikeResponseMatrixResult(
            matrix=matrix,
            variance_matrix=variance,
            n_max=n_max,
            magnitude=0.5,
            wavelength_nm=1064,
            n_averages=3,
            n_cycles=2,
            timestamp="2026-04-25T12:00:00",
            excluded_piston=True,
        )

        assert result.n_wfs_terms == n_wfs
        assert result.n_slm_terms == n_slm
        assert result.mean_variance > 0
        assert result.max_variance > 0
        assert result.condition_number is None

    def test_dataclass_with_inverses(self):
        """测试带逆矩阵的数据类"""
        from ao_shaping.optimizer.wf.zernike_response_matrix import (
            ZernikeResponseMatrixResult,
            compute_pinv,
        )

        matrix = np.random.randn(5, 5)
        variance = np.abs(np.random.randn(5, 5)) * 0.01
        pinv_matrix = compute_pinv(matrix)

        result = ZernikeResponseMatrixResult(
            matrix=matrix,
            variance_matrix=variance,
            n_max=4,
            magnitude=0.5,
            wavelength_nm=1064,
            n_averages=3,
            n_cycles=2,
            timestamp="2026-04-25T12:00:00",
            excluded_piston=True,
            pinv_matrix=pinv_matrix,
        )

        assert result.condition_number is not None
        assert result.condition_number > 0

    def test_to_dict_and_from_dict(self):
        """测试字典序列化/反序列化"""
        from ao_shaping.optimizer.wf.zernike_response_matrix import (
            ZernikeResponseMatrixResult,
        )

        matrix = np.random.randn(5, 5)
        variance = np.abs(np.random.randn(5, 5)) * 0.01
        pinv = np.linalg.pinv(matrix)

        result = ZernikeResponseMatrixResult(
            matrix=matrix,
            variance_matrix=variance,
            n_max=4,
            magnitude=0.5,
            wavelength_nm=1064,
            n_averages=3,
            n_cycles=2,
            timestamp="2026-04-25T12:00:00",
            excluded_piston=True,
            pinv_matrix=pinv,
        )

        d = result.to_dict()
        assert "matrix" in d
        assert "variance_matrix" in d
        assert "pinv_matrix" in d
        assert isinstance(d["matrix"], list)

        restored = ZernikeResponseMatrixResult.from_dict(d)
        assert np.allclose(restored.matrix, matrix)
        assert np.allclose(restored.variance_matrix, variance)
        assert np.allclose(restored.pinv_matrix, pinv)
        assert restored.n_cycles == result.n_cycles


class TestSaveLoadEnhanced:
    """测试增强版保存/加载功能"""

    def test_save_and_load(self):
        """测试完整保存和加载"""
        from ao_shaping.optimizer.wf.zernike_response_matrix import (
            ZernikeResponseMatrixResult,
            save_zernike_response_matrix,
            load_zernike_response_matrix,
        )

        n_max = 10
        n_wfs = calc_n_zernike_terms(10) - 1
        n_slm = calc_n_zernike_terms(n_max) - 1

        matrix = np.random.randn(n_wfs, n_slm)
        variance = np.abs(np.random.randn(n_wfs, n_slm)) * 0.01
        pinv = np.linalg.pinv(matrix)

        result = ZernikeResponseMatrixResult(
            matrix=matrix,
            variance_matrix=variance,
            n_max=n_max,
            magnitude=0.5,
            wavelength_nm=1064,
            n_averages=3,
            n_cycles=2,
            timestamp="2026-04-25T12:00:00",
            excluded_piston=True,
            pinv_matrix=pinv,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "test_matrix"

            save_zernike_response_matrix(result, base, include_inverses=True)

            assert base.with_suffix(".response.npy").exists()
            assert base.with_suffix(".variance.npy").exists()
            assert base.with_suffix(".pinv.npy").exists()
            assert base.with_suffix(".json").exists()

            loaded = load_zernike_response_matrix(base)

            assert np.allclose(loaded.matrix, matrix)
            assert np.allclose(loaded.variance_matrix, variance)
            assert np.allclose(loaded.pinv_matrix, pinv)
            assert loaded.n_cycles == 2

    def test_load_without_inverses(self):
        """测试加载没有逆矩阵的结果"""
        from ao_shaping.optimizer.wf.zernike_response_matrix import (
            ZernikeResponseMatrixResult,
            save_zernike_response_matrix,
            load_zernike_response_matrix,
        )

        n_max = 4
        n_wfs = calc_n_zernike_terms(10) - 1
        n_slm = calc_n_zernike_terms(n_max) - 1

        matrix = np.random.randn(n_wfs, n_slm)
        variance = np.abs(np.random.randn(n_wfs, n_slm)) * 0.01

        result = ZernikeResponseMatrixResult(
            matrix=matrix,
            variance_matrix=variance,
            n_max=n_max,
            magnitude=0.5,
            wavelength_nm=1064,
            n_averages=3,
            n_cycles=2,
            timestamp="2026-04-25T12:00:00",
            excluded_piston=True,
            pinv_matrix=None,
            lstsq_matrix=None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "test_matrix"

            save_zernike_response_matrix(result, base, include_inverses=False)

            loaded = load_zernike_response_matrix(base)

            assert loaded.pinv_matrix is None
            assert loaded.lstsq_matrix is None


class TestMatrixOperations:
    def test_matrix_shape_calculation(self):
        """测试矩阵形状计算"""
        n_max = 10
        n_wfs = calc_n_zernike_terms(10) - 1
        n_slm = calc_n_zernike_terms(n_max) - 1

        matrix = np.random.randn(n_wfs, n_slm)
        assert matrix.shape == (65, 65)

    def test_matrix_multiply(self):
        """测试矩阵乘法 - 响应矩阵应用"""
        n_wfs = 65
        n_slm = 65

        response_matrix = np.random.randn(n_wfs, n_slm)
        zernike_command = np.zeros(n_slm)
        zernike_command[0] = 0.5

        measured_response = response_matrix @ zernike_command

        assert measured_response.shape == (n_wfs,)
        assert not np.allclose(measured_response, 0)

    def test_response_matrix_application(self):
        """测试响应矩阵在控制中的应用"""
        from ao_shaping.optimizer.wf.zernike_response_matrix import compute_pinv

        n_wfs = 65
        n_slm = 65

        response_matrix = np.random.randn(n_wfs, n_slm) * 0.1
        pinv_matrix = compute_pinv(response_matrix)

        target_response = np.zeros(n_wfs)
        target_response[0] = 0.5

        slm_command = pinv_matrix @ target_response

        assert slm_command.shape == (n_slm,)
        # 伪逆计算可能产生较大命令幅度，这是正常的
        assert not np.any(np.isnan(slm_command))


class TestVarianceTracking:
    """测试方差跟踪功能"""

    def test_variance_properties(self):
        """测试方差的数学性质"""
        from ao_shaping.optimizer.wf.zernike_response_matrix import (
            ZernikeResponseMatrixResult,
        )

        n_measurements = 5
        n_wfs = 10
        n_slm = 5

        all_responses = np.random.randn(n_measurements, n_wfs, n_slm) * 0.1
        mean_response = np.mean(all_responses, axis=0)
        variance_response = np.var(all_responses, axis=0)

        assert mean_response.shape == (n_wfs, n_slm)
        assert variance_response.shape == (n_wfs, n_slm)

        result = ZernikeResponseMatrixResult(
            matrix=mean_response,
            variance_matrix=variance_response,
            n_max=4,
            magnitude=0.5,
            wavelength_nm=1064,
            n_averages=3,
            n_cycles=n_measurements,
            timestamp="2026-04-25T12:00:00",
        )

        assert result.mean_variance > 0
        assert result.max_variance > 0
        assert result.mean_variance <= result.max_variance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])