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


class TestCallbackSupport:
    """测试callback回调功能"""

    def test_callback_called_correct_times(self):
        """测试callback被正确调用了预期的次数"""
        from unittest.mock import MagicMock, patch
        from ao_shaping.optimizer.wf.zernike_response_matrix import (
            calibrate_zernike_response_matrix,
            calc_n_zernike_terms,
        )

        n_max = 4
        n_slm_terms = calc_n_zernike_terms(n_max) - 1  # excluded_piston=True
        n_wfs_terms = calc_n_zernike_terms(n_max) - 1

        mock_callback = MagicMock()

        mock_zslm = MagicMock()
        mock_zslm.wavelength = 1064
        mock_zslm._slm = MagicMock()
        mock_wfs = MagicMock()
        mock_wfs.calc_n_zernike_terms.return_value = calc_n_zernike_terms(n_max)

        with patch(
            "ao_shaping.optimizer.wf.zernike_response_matrix.measure_zernike_mode_response"
        ) as mock_measure, patch(
            "ao_shaping.optimizer.wf.zernike_response_matrix.set_slm_flat"
        ), patch(
            "ao_shaping.optimizer.wf.zernike_response_matrix.time"
        ):
            mock_measure.return_value = (
                np.zeros(n_wfs_terms),
                np.zeros(n_wfs_terms),
            )

            result = calibrate_zernike_response_matrix(
                zslm=mock_zslm,
                wfs=mock_wfs,
                n_max=n_max,
                magnitude=0.5,
                n_cycles=1,
                n_averages=1,
                wait_time=0.01,
                excluded_piston=True,
                compute_inverses=False,
                verbose=False,
                display=None,
                callback=mock_callback,
            )

            assert mock_callback.call_count == n_slm_terms

            for call_idx, call_args in enumerate(mock_callback.call_args_list):
                args = call_args[0]
                assert len(args) == 4
                assert args[0] == call_idx
                assert args[1] == n_slm_terms
                assert isinstance(args[2], np.ndarray)
                assert isinstance(args[3], np.ndarray)

    def test_callback_skips_tqdm(self):
        """测试提供callback时跳过tqdm进度条"""
        from unittest.mock import MagicMock, patch
        from ao_shaping.optimizer.wf.zernike_response_matrix import (
            calibrate_zernike_response_matrix,
            calc_n_zernike_terms,
        )

        n_max = 3
        n_slm_terms = calc_n_zernike_terms(n_max) - 1
        n_wfs_terms = calc_n_zernike_terms(n_max) - 1

        mock_callback = MagicMock()
        mock_zslm = MagicMock()
        mock_zslm.wavelength = 1064
        mock_zslm._slm = MagicMock()
        mock_wfs = MagicMock()
        mock_wfs.calc_n_zernike_terms.return_value = calc_n_zernike_terms(n_max)

        with patch(
            "ao_shaping.optimizer.wf.zernike_response_matrix.measure_zernike_mode_response"
        ) as mock_measure, patch(
            "ao_shaping.optimizer.wf.zernike_response_matrix.tqdm"
        ) as mock_tqdm, patch(
            "ao_shaping.optimizer.wf.zernike_response_matrix.set_slm_flat"
        ), patch(
            "ao_shaping.optimizer.wf.zernike_response_matrix.time"
        ):
            mock_measure.return_value = (
                np.zeros(n_wfs_terms),
                np.zeros(n_wfs_terms),
            )
            mock_tqdm.side_effect = lambda x, **kwargs: x

            result = calibrate_zernike_response_matrix(
                zslm=mock_zslm,
                wfs=mock_wfs,
                n_max=n_max,
                magnitude=0.5,
                n_cycles=1,
                n_averages=1,
                wait_time=0.01,
                excluded_piston=True,
                compute_inverses=False,
                verbose=True,
                display=None,
                callback=mock_callback,
            )

            mock_tqdm.assert_not_called()

    def test_callback_without_callback_uses_tqdm(self):
        """测试不提供callback时正常使用tqdm"""
        from unittest.mock import MagicMock, patch
        from ao_shaping.optimizer.wf.zernike_response_matrix import (
            calibrate_zernike_response_matrix,
            calc_n_zernike_terms,
        )

        n_max = 3
        n_slm_terms = calc_n_zernike_terms(n_max) - 1
        n_wfs_terms = calc_n_zernike_terms(n_max) - 1

        mock_zslm = MagicMock()
        mock_zslm.wavelength = 1064
        mock_zslm._slm = MagicMock()
        mock_wfs = MagicMock()
        mock_wfs.calc_n_zernike_terms.return_value = calc_n_zernike_terms(n_max)

        with patch(
            "ao_shaping.optimizer.wf.zernike_response_matrix.measure_zernike_mode_response"
        ) as mock_measure, patch(
            "ao_shaping.optimizer.wf.zernike_response_matrix.tqdm"
        ) as mock_tqdm, patch(
            "ao_shaping.optimizer.wf.zernike_response_matrix.set_slm_flat"
        ), patch(
            "ao_shaping.optimizer.wf.zernike_response_matrix.time"
        ):
            mock_measure.return_value = (
                np.zeros(n_wfs_terms),
                np.zeros(n_wfs_terms),
            )
            mock_tqdm.side_effect = lambda x, **kwargs: x

            result = calibrate_zernike_response_matrix(
                zslm=mock_zslm,
                wfs=mock_wfs,
                n_max=n_max,
                magnitude=0.5,
                n_cycles=1,
                n_averages=1,
                wait_time=0.01,
                excluded_piston=True,
                compute_inverses=False,
                verbose=True,
                display=None,
                callback=None,
            )

            mock_tqdm.assert_called_once()

    def test_callback_backward_compatibility(self):
        """测试向后兼容性：display参数仍然正常工作"""
        from unittest.mock import MagicMock, patch
        from ao_shaping.optimizer.wf.zernike_response_matrix import (
            calibrate_zernike_response_matrix,
            calc_n_zernike_terms,
            ZernikeCalibrationDisplay,
        )

        n_max = 3
        n_slm_terms = calc_n_zernike_terms(n_max) - 1
        n_wfs_terms = calc_n_zernike_terms(n_max) - 1

        mock_zslm = MagicMock()
        mock_zslm.wavelength = 1064
        mock_zslm._slm = MagicMock()
        mock_wfs = MagicMock()
        mock_wfs.calc_n_zernike_terms.return_value = calc_n_zernike_terms(n_max)

        mock_display = MagicMock(spec=ZernikeCalibrationDisplay)
        mock_display.update.return_value = True

        with patch(
            "ao_shaping.optimizer.wf.zernike_response_matrix.measure_zernike_mode_response"
        ) as mock_measure, patch(
            "ao_shaping.optimizer.wf.zernike_response_matrix.set_slm_flat"
        ), patch(
            "ao_shaping.optimizer.wf.zernike_response_matrix.time"
        ):
            mock_measure.return_value = (
                np.zeros(n_wfs_terms),
                np.zeros(n_wfs_terms),
            )

            result = calibrate_zernike_response_matrix(
                zslm=mock_zslm,
                wfs=mock_wfs,
                n_max=n_max,
                magnitude=0.5,
                n_cycles=1,
                n_averages=1,
                wait_time=0.01,
                excluded_piston=True,
                compute_inverses=False,
                verbose=False,
                display=mock_display,
                callback=None,
            )

            assert mock_display.update.call_count == n_slm_terms
            mock_display.init_window.assert_called_once()
            mock_display.close.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestMatrixShapeCorrectness:
    """测试响应矩阵形状正确性 - 包括shift参数影响"""

    def test_matrix_shape_excluded_piston_true(self):
        """测试当excluded_piston=True时的矩阵形状"""
        from unittest.mock import MagicMock, patch
        from ao_shaping.optimizer.wf.zernike_response_matrix import (
            calibrate_zernike_response_matrix,
            calc_n_zernike_terms,
        )

        n_max = 10
        n_slm_terms = calc_n_zernike_terms(n_max) - 1
        n_wfs_terms = calc_n_zernike_terms(n_max) - 1

        mock_zslm = MagicMock()
        mock_zslm.wavelength = 1064
        mock_zslm._slm = MagicMock()
        mock_wfs = MagicMock()
        mock_wfs.calc_n_zernike_terms.return_value = calc_n_zernike_terms(n_max)

        with patch(
            "ao_shaping.optimizer.wf.zernike_response_matrix.measure_zernike_mode_response"
        ) as mock_measure, patch(
            "ao_shaping.optimizer.wf.zernike_response_matrix.set_slm_flat"
        ), patch(
            "ao_shaping.optimizer.wf.zernike_response_matrix.time"
        ):
            mock_measure.return_value = (
                np.zeros(n_wfs_terms),
                np.zeros(n_wfs_terms),
            )

            result = calibrate_zernike_response_matrix(
                zslm=mock_zslm,
                wfs=mock_wfs,
                n_max=n_max,
                magnitude=0.5,
                n_cycles=1,
                n_averages=1,
                wait_time=0.01,
                excluded_piston=True,
                compute_inverses=False,
                verbose=False,
                display=None,
                callback=None,
            )

            assert result.matrix.shape == (n_wfs_terms, n_slm_terms), \
                f"期望形状 ({n_wfs_terms}, {n_slm_terms}), 实际 {result.matrix.shape}"
            assert result.variance_matrix.shape == (n_wfs_terms, n_slm_terms)
            assert result.n_wfs_terms == n_wfs_terms
            assert result.n_slm_terms == n_slm_terms

    def test_matrix_shape_excluded_piston_false(self):
        """测试当excluded_piston=False时的矩阵形状"""
        from unittest.mock import MagicMock, patch
        from ao_shaping.optimizer.wf.zernike_response_matrix import (
            calibrate_zernike_response_matrix,
            calc_n_zernike_terms,
        )

        n_max = 10
        n_slm_terms = calc_n_zernike_terms(n_max)
        n_wfs_terms = calc_n_zernike_terms(n_max)

        mock_zslm = MagicMock()
        mock_zslm.wavelength = 1064
        mock_zslm._slm = MagicMock()
        mock_wfs = MagicMock()
        mock_wfs.calc_n_zernike_terms.return_value = calc_n_zernike_terms(n_max)

        with patch(
            "ao_shaping.optimizer.wf.zernike_response_matrix.measure_zernike_mode_response"
        ) as mock_measure, patch(
            "ao_shaping.optimizer.wf.zernike_response_matrix.set_slm_flat"
        ), patch(
            "ao_shaping.optimizer.wf.zernike_response_matrix.time"
        ):
            mock_measure.return_value = (
                np.zeros(n_wfs_terms),
                np.zeros(n_wfs_terms),
            )

            result = calibrate_zernike_response_matrix(
                zslm=mock_zslm,
                wfs=mock_wfs,
                n_max=n_max,
                magnitude=0.5,
                n_cycles=1,
                n_averages=1,
                wait_time=0.01,
                excluded_piston=False,
                compute_inverses=False,
                verbose=False,
                display=None,
                callback=None,
            )

            assert result.matrix.shape == (n_wfs_terms, n_slm_terms), \
                f"期望形状 ({n_wfs_terms}, {n_slm_terms}), 实际 {result.matrix.shape}"
            assert result.variance_matrix.shape == (n_wfs_terms, n_slm_terms)
            assert result.n_wfs_terms == n_wfs_terms
            assert result.n_slm_terms == n_slm_terms

    def test_calibrate_with_shift_parameters(self):
        """测试校准函数在shift参数存在时仍能正常工作"""
        from unittest.mock import MagicMock, patch
        from ao_shaping.optimizer.wf.zernike_response_matrix import (
            calibrate_zernike_response_matrix,
            calc_n_zernike_terms,
        )

        n_max = 4
        n_slm_terms = calc_n_zernike_terms(n_max) - 1
        n_wfs_terms = calc_n_zernike_terms(n_max) - 1

        mock_zslm = MagicMock()
        mock_zslm.wavelength = 1064
        mock_zslm._slm = MagicMock()
        mock_zslm.shift_x = 0
        mock_zslm.shift_y = 0
        mock_wfs = MagicMock()
        mock_wfs.calc_n_zernike_terms.return_value = calc_n_zernike_terms(n_max)

        with patch(
            "ao_shaping.optimizer.wf.zernike_response_matrix.measure_zernike_mode_response"
        ) as mock_measure, patch(
            "ao_shaping.optimizer.wf.zernike_response_matrix.set_slm_flat"
        ), patch(
            "ao_shaping.optimizer.wf.zernike_response_matrix.time"
        ):
            mock_measure.return_value = (
                np.zeros(n_wfs_terms),
                np.zeros(n_wfs_terms),
            )

            result = calibrate_zernike_response_matrix(
                zslm=mock_zslm,
                wfs=mock_wfs,
                n_max=n_max,
                magnitude=0.5,
                n_cycles=1,
                n_averages=1,
                wait_time=0.01,
                excluded_piston=True,
                compute_inverses=False,
                verbose=False,
                display=None,
                callback=None,
            )

            assert result.matrix.shape == (n_wfs_terms, n_slm_terms)

    def test_response_vector_dimension_matches_matrix(self):
        """测试响应向量维度与矩阵维度匹配"""
        from unittest.mock import MagicMock, patch
        from ao_shaping.optimizer.wf.zernike_response_matrix import (
            calibrate_zernike_response_matrix,
            calc_n_zernike_terms,
        )

        n_max = 10
        n_slm_terms = calc_n_zernike_terms(n_max) - 1
        n_wfs_terms = calc_n_zernike_terms(n_max) - 1

        mock_zslm = MagicMock()
        mock_zslm.wavelength = 1064
        mock_zslm._slm = MagicMock()
        mock_wfs = MagicMock()
        mock_wfs.calc_n_zernike_terms.return_value = calc_n_zernike_terms(n_max)

        with patch(
            "ao_shaping.optimizer.wf.zernike_response_matrix.measure_zernike_mode_response"
        ) as mock_measure, patch(
            "ao_shaping.optimizer.wf.zernike_response_matrix.set_slm_flat"
        ), patch(
            "ao_shaping.optimizer.wf.zernike_response_matrix.time"
        ):
            response_vec = np.random.randn(n_wfs_terms) * 0.1
            variance_vec = np.random.rand(n_wfs_terms) * 0.01
            mock_measure.return_value = (response_vec, variance_vec)

            result = calibrate_zernike_response_matrix(
                zslm=mock_zslm,
                wfs=mock_wfs,
                n_max=n_max,
                magnitude=0.5,
                n_cycles=1,
                n_averages=1,
                wait_time=0.01,
                excluded_piston=True,
                compute_inverses=False,
                verbose=False,
                display=None,
                callback=None,
            )

            assert result.matrix.shape == (n_wfs_terms, n_slm_terms)
            assert np.allclose(result.matrix[:, 0], response_vec)
            assert np.allclose(result.variance_matrix[:, 0], variance_vec)

    def test_n_wfs_terms_calculation_with_excluded_piston(self):
        """测试n_wfs_terms在不同excluded_piston设置下的计算"""
        from ao_shaping.optimizer.wf.zernike_response_matrix import calc_n_zernike_terms

        # calc_n_zernike_terms(10) 应该返回66 (包括piston)
        n_total = calc_n_zernike_terms(10)
        assert n_total == 66

        # 排除piston后应为65
        n_excluded = calc_n_zernike_terms(10) - 1
        assert n_excluded == 65

    def test_slm_zernike_terms_calculation(self):
        """测试SLM侧Zernike项数计算"""
        from ao_shaping.optimizer.wf.zernike_response_matrix import calc_n_zernike_terms

        # 不同n_max对应的项数
        assert calc_n_zernike_terms(4) == 15    # n_max=4
        assert calc_n_zernike_terms(10) == 66   # n_max=10

        # 排除piston后的项数
        assert calc_n_zernike_terms(4) - 1 == 14   # n_max=4, 排除piston
        assert calc_n_zernike_terms(10) - 1 == 65  # n_max=10, 排除piston