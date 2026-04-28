"""Unit tests for new Zernike SLM functions in interaction_matrix.py.

Tests ZernikeSLMResponseMatrixResult, save/load functions,
and mathematical operations using mock devices.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from ao_shaping.optimizer.wf.interaction_matrix import (
    ZernikeSLMResponseMatrixResult,
    DEFAULT_SLM_ZERNIKE_MODES,
    save_zernike_slm_response_matrix,
    load_zernike_slm_response_matrix,
)
from ao_shaping.utils.matrix_utils import compute_pinv


class TestZernikeSLMResponseMatrixResult:
    """Test ZernikeSLMResponseMatrixResult dataclass."""

    def test_dataclass_creation(self):
        """Test dataclass creation with all fields."""
        matrix = np.random.randn(100, 10)
        variance_matrix = np.abs(np.random.randn(100, 10)) * 0.01

        result = ZernikeSLMResponseMatrixResult(
            matrix=matrix,
            variance_matrix=variance_matrix,
            slm_wavelength_nm=1064,
            wfs_resolution="1024",
            pupil_diameter_mm=4.6,
            magnitude_rad=10,
            n_cycles=3,
            n_averages=15,
            timestamp="2026-04-28T12:00:00",
            pinv_matrix=None,
        )

        assert result.matrix.shape == (100, 10)
        assert result.variance_matrix.shape == (100, 10)
        assert result.slm_wavelength_nm == 1064
        assert result.wfs_resolution == "768"
        assert result.pupil_diameter_mm == 2.24
        assert result.magnitude_rad == 0.5
        assert result.n_cycles == 3
        assert result.n_averages == 5
        assert result.pinv_matrix is None

    def test_n_spots_property(self):
        """Test n_spots property."""
        matrix = np.zeros((2 * 50, 10))  # 50 spots * 2 for x,y
        result = ZernikeSLMResponseMatrixResult(
            matrix=matrix,
            variance_matrix=np.zeros_like(matrix),
            slm_wavelength_nm=1064,
            wfs_resolution="768",
            pupil_diameter_mm=2.24,
            magnitude_rad=0.5,
            n_cycles=3,
            n_averages=5,
            timestamp="2026-04-28T12:00:00",
        )

        assert result.n_spots == 50  # 100 / 2

    def test_n_modes_property(self):
        """Test n_modes property."""
        matrix = np.zeros((100, 10))
        result = ZernikeSLMResponseMatrixResult(
            matrix=matrix,
            variance_matrix=np.zeros_like(matrix),
            slm_wavelength_nm=1064,
            wfs_resolution="768",
            pupil_diameter_mm=2.24,
            magnitude_rad=0.5,
            n_cycles=3,
            n_averages=5,
            timestamp="2026-04-28T12:00:00",
        )

        assert result.n_modes == 10

    def test_to_dict(self):
        """Test to_dict() serialization."""
        matrix = np.random.randn(100, 10)
        result = ZernikeSLMResponseMatrixResult(
            matrix=matrix,
            variance_matrix=np.abs(np.random.randn(100, 10)) * 0.01,
            slm_wavelength_nm=1064,
            wfs_resolution="768",
            pupil_diameter_mm=2.24,
            magnitude_rad=0.5,
            n_cycles=3,
            n_averages=5,
            timestamp="2026-04-28T12:00:00",
            pinv_matrix=None,
        )

        d = result.to_dict()

        assert isinstance(d, dict)
        assert d["slm_wavelength_nm"] == 1064
        assert d["wfs_resolution"] == "768"
        assert d["pupil_diameter_mm"] == 2.24
        assert d["magnitude_rad"] == 0.5
        assert d["n_cycles"] == 3
        assert d["n_averages"] == 5
        assert d["matrix_shape"] == (100, 10)
        assert d["n_spots"] == 50
        assert d["n_modes"] == 10

    def test_from_dict(self):
        """Test from_dict() deserialization."""
        matrix = np.random.randn(100, 10)
        variance = np.abs(np.random.randn(100, 10)) * 0.01
        pinv = compute_pinv(matrix)

        result = ZernikeSLMResponseMatrixResult(
            matrix=matrix,
            variance_matrix=variance,
            slm_wavelength_nm=1064,
            wfs_resolution="768",
            pupil_diameter_mm=2.24,
            magnitude_rad=0.5,
            n_cycles=3,
            n_averages=5,
            timestamp="2026-04-28T12:00:00",
            pinv_matrix=pinv,
        )

        d = result.to_dict()

        # Simulate loading from dict + arrays
        restored = ZernikeSLMResponseMatrixResult.from_dict(
            d, matrix, variance, pinv
        )

        assert np.allclose(restored.matrix, matrix)
        assert np.allclose(restored.variance_matrix, variance)
        assert np.allclose(restored.pinv_matrix, pinv)
        assert restored.slm_wavelength_nm == 1064
        assert restored.n_cycles == 3


class TestDefaultSLMZernikeModes:
    """Test DEFAULT_SLM_ZERNIKE_MODES constant."""

    def test_default_modes_count(self):
        """Test that default modes list is not empty."""
        assert len(DEFAULT_SLM_ZERNIKE_MODES) > 0
        assert len(DEFAULT_SLM_ZERNIKE_MODES) == 10

    def test_default_modes_format(self):
        """Test that default modes are (n, m) tuples."""
        for mode in DEFAULT_SLM_ZERNIKE_MODES:
            assert isinstance(mode, tuple)
            assert len(mode) == 2
            n, m = mode
            assert isinstance(n, int)
            assert isinstance(m, int)
            assert -n <= m <= n
            assert (n + m) % 2 == 0  # Zernike requirement


class TestSaveLoadZernikeSLMMatrix:
    """Test save/load functions for Zernike SLM response matrix."""

    def test_save_and_load_with_pinv(self):
        """Test save and load with pseudoinverse."""
        matrix = np.random.randn(100, 10)
        variance = np.abs(np.random.randn(100, 10)) * 0.01
        pinv = compute_pinv(matrix)

        result = ZernikeSLMResponseMatrixResult(
            matrix=matrix,
            variance_matrix=variance,
            slm_wavelength_nm=1064,
            wfs_resolution="768",
            pupil_diameter_mm=2.24,
            magnitude_rad=0.5,
            n_cycles=3,
            n_averages=5,
            timestamp="2026-04-28T12:00:00",
            pinv_matrix=pinv,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "test_zernike_slm_matrix"

            save_zernike_slm_response_matrix(result, base)

            # Check files exist
            assert base.with_suffix(".matrix.npy").exists()
            assert base.with_suffix(".variance.npy").exists()
            assert base.with_suffix(".pinv.npy").exists()
            assert base.with_suffix(".json").exists()

            # Load and verify
            loaded = load_zernike_slm_response_matrix(base)

            assert np.allclose(loaded.matrix, matrix)
            assert np.allclose(loaded.variance_matrix, variance)
            assert np.allclose(loaded.pinv_matrix, pinv)
            assert loaded.slm_wavelength_nm == 1064
            assert loaded.n_cycles == 3
            assert loaded.n_averages == 5

    def test_save_and_load_without_pinv(self):
        """Test save and load without pseudoinverse."""
        matrix = np.random.randn(100, 10)
        variance = np.abs(np.random.randn(100, 10)) * 0.01

        result = ZernikeSLMResponseMatrixResult(
            matrix=matrix,
            variance_matrix=variance,
            slm_wavelength_nm=1064,
            wfs_resolution="768",
            pupil_diameter_mm=2.24,
            magnitude_rad=0.5,
            n_cycles=3,
            n_averages=5,
            timestamp="2026-04-28T12:00:00",
            pinv_matrix=None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "test_no_pinv"

            save_zernike_slm_response_matrix(result, base)

            assert base.with_suffix(".matrix.npy").exists()
            assert base.with_suffix(".variance.npy").exists()
            assert not base.with_suffix(".pinv.npy").exists()  # Should not exist
            assert base.with_suffix(".json").exists()

            loaded = load_zernike_slm_response_matrix(base)

            assert np.allclose(loaded.matrix, matrix)
            assert loaded.pinv_matrix is None


class TestComputePinvIntegration:
    """Test compute_pinv integration with response matrix shapes."""

    def test_pinv_shape_response_matrix(self):
        """Test pseudoinverse shape for SLM response matrix."""
        # Response matrix: (2*N_spots, N_modes)
        D = np.random.randn(100, 10)  # 50 spots * 2, 10 Zernike modes

        D_pinv = compute_pinv(D)

        # Pseudoinverse shape: (N_modes, 2*N_spots)
        assert D_pinv.shape == (10, 100)

    def test_pinv_reconstruction(self):
        """Test that pinv(D) @ D ≈ I for overdetermined full-rank system."""
        # For overdetermined system D (m>n), pinv(D) @ D = (D^T @ D)^-1 @ D^T @ D = I
        # Create a full column-rank matrix: D = [A; B] where A is 10x10 invertible
        np.random.seed(42)
        A = np.random.randn(10, 10)
        # Make A well-conditioned
        A = A + np.eye(10) * 10
        B = np.random.randn(90, 10)
        D = np.vstack([A, B])  # Shape (100, 10), full column rank

        D_pinv = compute_pinv(D)

        # pinv(D) @ D should be close to identity (10x10)
        identity_10 = np.eye(10)
        reconstruction = D_pinv @ D
        assert reconstruction.shape == (10, 10)
        assert np.allclose(reconstruction, identity_10, atol=1e-6)

    def test_pinv_svd_property(self):
        """Test SVD pseudoinverse properties."""
        D = np.random.randn(80, 8)

        D_pinv = compute_pinv(D)

        # D @ D_pinv should be close to identity (for overdetermined system)
        # Actually for rectangular matrices, D @ D_pinv is projection onto column space
        # Just verify shapes are correct
        assert D_pinv.shape == (8, 80)
        assert (D @ D_pinv).shape == (80, 80)


class TestCalculateZernikeSLMResponseMatrixMocked:
    """Test calculate_zernike_slm_response_matrix with mock devices.

    Note: This requires mock devices that implement the SantecSLM200
    and WFSManager interfaces. The current MockSLM and MockWFS
    in mock_devices.py have simpler interfaces and need extension.
    """

    def test_cannot_run_without_proper_mocks(self):
        """Document that we need interface-compatible mocks."""
        pytest.skip(
            "Integration test requires mock devices with SantecSLM200/WFSManager interfaces. "
            "Current MockSLM/MockWFS have simpler interfaces."
        )


class TestApplyZernikeCorrectionPIDMocked:
    """Test apply_zernike_correction with PID control using mocks."""

    @pytest.fixture
    def mock_slm(self):
        """Create a mock SLM with required interface."""
        class MockSLM:
            Panel_Res = (1920, 1200)
            wavelength = 1064

            def create_phase_from_array(self, phase):
                return phase.astype(np.uint16)

            def write_phase(self, gray, memory_number):
                self._last_phase = gray
                self._last_memory = memory_number

            def display_memory(self, memory_number):
                pass

        return MockSLM()

    @pytest.fixture
    def mock_wfs(self):
        """Create a mock WFS with required interface."""
        class MockWFS:
            num_spots_x = 10
            num_spots_y = 5
            d_x = 2.24

            def take_image(self, n_sample=1):
                pass

            def get_spot_deviation(self):
                # Return zero deviations (already corrected)
                return np.zeros((10, 5)), np.zeros((10, 5))

        return MockWFS()

    @pytest.fixture
    def response_matrix_file(self, tmp_path):
        """Create a temporary response matrix file for testing."""
        n_spots = 50  # 10x5
        n_modes = 10

        # Create a well-conditioned response matrix: (100, 10)
        # Make first 10 rows an invertible matrix for numerical stability
        D = np.random.randn(2 * n_spots, n_modes)
        D[:n_modes, :] += np.eye(n_modes) * 10

        variance = np.abs(np.random.randn(2 * n_spots, n_modes)) * 0.01
        pinv = compute_pinv(D)

        base = tmp_path / "test_response_matrix"

        np.save(base.with_suffix(".matrix.npy"), D)
        np.save(base.with_suffix(".variance.npy"), variance)
        np.save(base.with_suffix(".pinv.npy"), pinv)

        metadata = {
            "slm_wavelength_nm": 1064,
            "wfs_resolution": "768",
            "pupil_diameter_mm": 2.24,
            "magnitude_rad": 0.5,
            "n_cycles": 3,
            "n_averages": 5,
            "timestamp": "2026-04-28T12:00:00",
            "matrix_shape": list(D.shape),
            "n_spots": n_spots,
            "n_modes": n_modes,
        }

        with open(base.with_suffix(".json"), "w") as f:
            json.dump(metadata, f)

        return str(base)

    def test_pid_convergence(self, mock_slm, mock_wfs, response_matrix_file):
        """Test that PID loop converges when WFS returns zero slopes."""
        from ao_shaping.optimizer.wf.interaction_matrix import apply_zernike_correction

        final_coeffs, history = apply_zernike_correction(
            mock_slm, mock_wfs, response_matrix_file,
            Kp=1.0, Ki=0.1, Kd=0.01,
            max_iterations=50, convergence_threshold=1e-6,
            wait_time_s=0.01, n_averages=1,
        )

        assert len(history) > 0
        assert len(history) <= 50
        # Final coefficients should be close to zero (already corrected)
        assert np.linalg.norm(final_coeffs) < 1e-3

    def test_pid_max_iterations(self, mock_slm, response_matrix_file):
        """Test that PID stops after max_iterations."""
        from ao_shaping.optimizer.wf.interaction_matrix import apply_zernike_correction

        # Create WFS that always returns non-zero slopes (won't converge)
        class StubbornWFS:
            num_spots_x = 10
            num_spots_y = 5
            d_x = 2.24

            def take_image(self, n_sample=1):
                pass

            def get_spot_deviation(self):
                # Always return the same non-zero deviation
                return np.ones((10, 5)) * 0.1, np.ones((10, 5)) * 0.1

        wfs = StubbornWFS()

        final_coeffs, history = apply_zernike_correction(
            mock_slm, wfs, response_matrix_file,
            Kp=1.0, Ki=0.1, Kd=0.01,
            max_iterations=10, convergence_threshold=1e-10,
            wait_time_s=0.01, n_averages=1,
        )

        assert len(history) == 10  # Should hit max_iterations

    def test_pid_history_tracking(self, mock_slm, mock_wfs, response_matrix_file):
        """Test that history tracks coefficient vectors correctly."""
        from ao_shaping.optimizer.wf.interaction_matrix import apply_zernike_correction

        final_coeffs, history = apply_zernike_correction(
            mock_slm, mock_wfs, response_matrix_file,
            Kp=1.0, Ki=0.1, Kd=0.01,
            max_iterations=5, convergence_threshold=1e-10,
            wait_time_s=0.01, n_averages=1,
        )

        assert isinstance(history, list)
        assert len(history) <= 5
        for coeffs in history:
            assert isinstance(coeffs, np.ndarray)
            assert coeffs.shape == (10,)  # n_modes

    def test_pid_loads_from_file(self, mock_slm, mock_wfs, response_matrix_file):
        """Test that response matrix is loaded from file path."""
        from ao_shaping.optimizer.wf.interaction_matrix import apply_zernike_correction

        # This should not raise
        final_coeffs, history = apply_zernike_correction(
            mock_slm, mock_wfs, response_matrix_file,
            Kp=1.0, Ki=0.1, Kd=0.01,
            max_iterations=2, convergence_threshold=1e-10,
            wait_time_s=0.01, n_averages=1,
        )

        assert isinstance(final_coeffs, np.ndarray)

    def test_pid_with_target(self, mock_slm, mock_wfs, response_matrix_file):
        """Test PID with non-zero target (wavefront shaping, not flattening)."""
        from ao_shaping.optimizer.wf.interaction_matrix import apply_zernike_correction

        target = np.ones(10) * 0.5  # Non-zero target

        final_coeffs, history = apply_zernike_correction(
            mock_slm, mock_wfs, response_matrix_file,
            target=target,
            Kp=1.0, Ki=0.1, Kd=0.01,
            max_iterations=2, convergence_threshold=1e-10,
            wait_time_s=0.01, n_averages=1,
        )

        assert isinstance(final_coeffs, np.ndarray)
        assert len(history) <= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
