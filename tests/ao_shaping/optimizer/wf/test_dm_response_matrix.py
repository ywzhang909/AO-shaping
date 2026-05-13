"""Unit tests for DM response matrix module.

Tests DMResponseMatrixResult dataclass, HDF5 save/load roundtrips,
and mock-based tests for measure_actuator_response and calibration.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ===========================================================================
# Helper factories
# ===========================================================================


def _make_result(
    matrix: np.ndarray | None = None,
    variance_matrix: np.ndarray | None = None,
    n_actuators: int = 64,
    valid_actuator_indices: list[int] | None = None,
    pinv_matrix: np.ndarray | None = None,
    lstsq_matrix: np.ndarray | None = None,
    subaperture_mask: np.ndarray | None = None,
    device_config: dict | None = None,
    disturb_voltage: float = 50.0,
    n_averages: int = 20,
    n_cycles: int = 1,
    wait_time: float = 0.1,
    timestamp: str = "2026-05-13T12:00:00",
) -> "DMResponseMatrixResult":
    """Create a DMResponseMatrixResult with sensible defaults."""
    from ao_shaping.optimizer.wf.dm_response_matrix import DMResponseMatrixResult

    if matrix is None:
        matrix = np.random.randn(100, 63)
    if variance_matrix is None:
        variance_matrix = np.abs(np.random.randn(*matrix.shape)) * 0.01
    if valid_actuator_indices is None:
        valid_actuator_indices = list(range(1, n_actuators))

    return DMResponseMatrixResult(
        matrix=matrix,
        variance_matrix=variance_matrix,
        n_actuators=n_actuators,
        valid_actuator_indices=valid_actuator_indices,
        disturb_voltage=disturb_voltage,
        n_averages=n_averages,
        n_cycles=n_cycles,
        wait_time=wait_time,
        timestamp=timestamp,
        pinv_matrix=pinv_matrix,
        lstsq_matrix=lstsq_matrix,
        subaperture_mask=subaperture_mask,
        device_config=device_config,
    )


# ===========================================================================
# TestDMResponseMatrixResult
# ===========================================================================


class TestDMResponseMatrixResult:
    """Tests for the DMResponseMatrixResult dataclass."""

    def test_creation_default(self):
        """Create DMResponseMatrixResult with minimal args; verify defaults."""
        from ao_shaping.optimizer.wf.dm_response_matrix import DMResponseMatrixResult

        matrix = np.random.randn(100, 63)
        variance = np.abs(np.random.randn(100, 63)) * 0.01

        result = DMResponseMatrixResult(
            matrix=matrix,
            variance_matrix=variance,
        )

        assert result.matrix.shape == (100, 63)
        assert result.variance_matrix.shape == (100, 63)

        # Default values
        assert result.n_actuators == 64
        assert result.valid_actuator_indices is None
        assert result.disturb_voltage == 50.0
        assert result.n_averages == 20
        assert result.n_cycles == 1
        assert result.wait_time == 0.1
        assert result.timestamp == ""
        assert result.pinv_matrix is None
        assert result.lstsq_matrix is None
        assert result.subaperture_mask is None
        assert result.device_config is None
        assert result.amplitude_optimization is None

    def test_properties_n_slopes(self):
        """Verify n_slopes = matrix.shape[0], n_actuators_valid = matrix.shape[1]."""
        result = _make_result(
            matrix=np.random.randn(120, 63),
            variance_matrix=np.abs(np.random.randn(120, 63)) * 0.01,
        )
        assert result.n_slopes == 120
        assert result.n_actuators_valid == 63

        # Square matrix
        result2 = _make_result(
            matrix=np.random.randn(100, 100),
            variance_matrix=np.abs(np.random.randn(100, 100)) * 0.01,
        )
        assert result2.n_slopes == 100
        assert result2.n_actuators_valid == 100

    def test_properties_mean_max_variance(self):
        """Create with known variance, verify mean/max."""
        variance = np.full((100, 63), 0.05)
        variance[0, 0] = 0.5  # outlier for max
        result = _make_result(variance_matrix=variance)

        # mean includes the outlier: (6299*0.05 + 0.5) / 6300 ≈ 0.05007
        assert result.mean_variance == pytest.approx(0.05007, rel=1e-4)
        assert result.max_variance == pytest.approx(0.5)

    def test_properties_mean_variance_zero(self):
        """When variance is all zero, mean and max are both zero."""
        variance = np.zeros((100, 63))
        result = _make_result(variance_matrix=variance)
        assert result.mean_variance == 0.0
        assert result.max_variance == 0.0

    def test_condition_number_with_pinv(self):
        """Create with pinv_matrix; verify condition_number is not None."""
        matrix = np.random.randn(40, 40)
        pinv = np.linalg.pinv(matrix)
        result = _make_result(
            matrix=matrix,
            variance_matrix=np.abs(np.random.randn(40, 40)) * 0.01,
            pinv_matrix=pinv,
        )
        assert result.condition_number is not None
        assert result.condition_number > 0

    def test_condition_number_without_pinv(self):
        """Create without pinv_matrix; verify condition_number is None."""
        result = _make_result(pinv_matrix=None)
        assert result.condition_number is None

    def test_n_actuators_valid_from_shape(self):
        """n_actuators_valid is derived from matrix.shape[1]."""
        result = _make_result(
            matrix=np.random.randn(100, 60),
            variance_matrix=np.abs(np.random.randn(100, 60)) * 0.01,
            n_actuators=64,
            valid_actuator_indices=list(range(1, 61)),
        )
        assert result.n_actuators_valid == 60
        assert result.n_actuators == 64

    def test_to_dict_roundtrip(self):
        """to_dict() then from_dict() preserves all scalar fields."""
        from ao_shaping.optimizer.wf.dm_response_matrix import DMResponseMatrixResult

        matrix = np.random.randn(100, 63)
        variance = np.abs(np.random.randn(100, 63)) * 0.01

        result = DMResponseMatrixResult(
            matrix=matrix,
            variance_matrix=variance,
            n_actuators=64,
            valid_actuator_indices=list(range(1, 64)),
            disturb_voltage=50.0,
            n_averages=20,
            n_cycles=1,
            wait_time=0.1,
            timestamp="2026-05-13T12:00:00",
            device_config={"dm_model": "NLight"},
        )

        d = result.to_dict()
        restored = DMResponseMatrixResult.from_dict(d)

        assert restored.disturb_voltage == 50.0
        assert restored.n_averages == 20
        assert restored.n_cycles == 1
        assert restored.wait_time == 0.1
        assert restored.timestamp == "2026-05-13T12:00:00"
        assert restored.device_config == {"dm_model": "NLight"}
        assert restored.n_actuators == 64
        assert restored.valid_actuator_indices == list(range(1, 64))

        # Arrays are preserved
        assert np.allclose(restored.matrix, matrix)
        assert np.allclose(restored.variance_matrix, variance)

    def test_to_dict_ndarray_conversion(self):
        """Verify ndarrays become lists in dict and convert back."""
        from ao_shaping.optimizer.wf.dm_response_matrix import DMResponseMatrixResult

        matrix = np.random.randn(10, 5)
        variance = np.abs(np.random.randn(10, 5)) * 0.01
        pinv = np.linalg.pinv(matrix)

        result = DMResponseMatrixResult(
            matrix=matrix,
            variance_matrix=variance,
            n_actuators=6,
            valid_actuator_indices=[1, 2, 3, 4, 5],
            disturb_voltage=50.0,
            n_averages=20,
            n_cycles=1,
            wait_time=0.1,
            timestamp="2026-05-13T12:00:00",
            pinv_matrix=pinv,
        )

        d = result.to_dict()

        # Arrays serialized as lists
        assert isinstance(d["matrix"], list)
        assert isinstance(d["variance_matrix"], list)
        assert isinstance(d["pinv_matrix"], list)

        # Roundtrip
        restored = DMResponseMatrixResult.from_dict(d)
        assert np.allclose(restored.matrix, matrix)
        assert np.allclose(restored.variance_matrix, variance)
        assert restored.pinv_matrix is not None
        assert np.allclose(restored.pinv_matrix, pinv)

    def test_to_dict_without_optional_lists(self):
        """Verify optional None fields are handled correctly."""
        from ao_shaping.optimizer.wf.dm_response_matrix import DMResponseMatrixResult

        matrix = np.random.randn(10, 5)
        variance = np.abs(np.random.randn(10, 5)) * 0.01

        result = DMResponseMatrixResult(
            matrix=matrix,
            variance_matrix=variance,
        )

        d = result.to_dict()
        # Optional list fields should be in the dict as their defaults
        assert "pinv_matrix" in d
        assert "lstsq_matrix" in d
        assert "subaperture_mask" in d
        assert "amplitude_optimization" in d

        # pinv_matrix will be None or array - from_dict must handle both
        restored = DMResponseMatrixResult.from_dict(d)
        assert restored.pinv_matrix is None
        assert restored.lstsq_matrix is None
        assert restored.subaperture_mask is None
        assert restored.amplitude_optimization is None


# ===========================================================================
# TestDMSaveLoad  (HDF5 roundtrips)
# ===========================================================================


class TestDMSaveLoad:
    """Test HDF5 save/load roundtrips for DMResponseMatrixResult."""

    def test_save_load_basic(self):
        """Save result to temp HDF5, load back, verify core fields match."""
        from ao_shaping.optimizer.wf.dm_response_matrix import (
            DMResponseMatrixResult,
            save_dm_response_matrix,
            load_dm_response_matrix,
        )

        matrix = np.random.randn(100, 63)
        variance = np.abs(np.random.randn(100, 63)) * 0.01

        result = DMResponseMatrixResult(
            matrix=matrix,
            variance_matrix=variance,
            n_actuators=64,
            valid_actuator_indices=list(range(1, 64)),
            disturb_voltage=50.0,
            n_averages=20,
            n_cycles=1,
            wait_time=0.1,
            timestamp="2026-05-13T12:00:00",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dm_response_matrix.h5"
            save_dm_response_matrix(result, path)

            assert path.exists()

            loaded = load_dm_response_matrix(path)

            assert np.allclose(loaded.matrix, matrix)
            assert np.allclose(loaded.variance_matrix, variance)
            assert loaded.disturb_voltage == 50.0
            assert loaded.n_averages == 20
            assert loaded.n_cycles == 1
            assert loaded.wait_time == 0.1
            assert loaded.timestamp == "2026-05-13T12:00:00"
            assert loaded.n_actuators == 64
            assert loaded.pinv_matrix is None
            assert loaded.lstsq_matrix is None
            assert loaded.subaperture_mask is None
            assert loaded.device_config is None

    def test_save_load_with_inverses(self):
        """Save with pinv_matrix and lstsq_matrix, load back, verify inverses match."""
        from ao_shaping.optimizer.wf.dm_response_matrix import (
            DMResponseMatrixResult,
            save_dm_response_matrix,
            load_dm_response_matrix,
        )

        matrix = np.random.randn(80, 50)
        variance = np.abs(np.random.randn(80, 50)) * 0.01
        pinv = np.linalg.pinv(matrix)
        lstsq = np.linalg.pinv(matrix)  # stand-in for least-squares inverse

        result = DMResponseMatrixResult(
            matrix=matrix,
            variance_matrix=variance,
            n_actuators=52,
            valid_actuator_indices=list(range(1, 51)),
            disturb_voltage=50.0,
            n_averages=20,
            n_cycles=1,
            wait_time=0.1,
            timestamp="2026-05-13T12:00:00",
            pinv_matrix=pinv,
            lstsq_matrix=lstsq,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dm_response_matrix.h5"
            save_dm_response_matrix(result, path)

            loaded = load_dm_response_matrix(path)

            assert loaded.pinv_matrix is not None
            assert loaded.lstsq_matrix is not None
            assert np.allclose(loaded.pinv_matrix, pinv)
            assert np.allclose(loaded.lstsq_matrix, lstsq)

    def test_save_load_without_inverses(self):
        """Save with include_inverses=False; verify loaded inverses are None."""
        from ao_shaping.optimizer.wf.dm_response_matrix import (
            DMResponseMatrixResult,
            save_dm_response_matrix,
            load_dm_response_matrix,
        )

        matrix = np.random.randn(80, 50)
        variance = np.abs(np.random.randn(80, 50)) * 0.01
        pinv = np.linalg.pinv(matrix)

        result = DMResponseMatrixResult(
            matrix=matrix,
            variance_matrix=variance,
            n_actuators=52,
            valid_actuator_indices=list(range(1, 51)),
            disturb_voltage=50.0,
            n_averages=20,
            n_cycles=1,
            wait_time=0.1,
            timestamp="2026-05-13T12:00:00",
            pinv_matrix=pinv,
            lstsq_matrix=None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dm_response_matrix.h5"
            save_dm_response_matrix(result, path, include_inverses=False)

            loaded = load_dm_response_matrix(path)

            assert loaded.pinv_matrix is None
            assert loaded.lstsq_matrix is None
            # Core data should still be intact
            assert np.allclose(loaded.matrix, matrix)
            assert loaded.disturb_voltage == 50.0

    def test_save_load_with_mask(self):
        """Save with subaperture_mask, load back, verify mask matches."""
        from ao_shaping.optimizer.wf.dm_response_matrix import (
            DMResponseMatrixResult,
            save_dm_response_matrix,
            load_dm_response_matrix,
        )

        matrix = np.random.randn(100, 63)
        variance = np.abs(np.random.randn(100, 63)) * 0.01
        subap_mask = np.random.randn(12, 12) > 0.3  # boolean mask

        result = DMResponseMatrixResult(
            matrix=matrix,
            variance_matrix=variance,
            n_actuators=64,
            valid_actuator_indices=list(range(1, 64)),
            disturb_voltage=50.0,
            n_averages=20,
            n_cycles=1,
            wait_time=0.1,
            timestamp="2026-05-13T12:00:00",
            subaperture_mask=subap_mask,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dm_response_matrix.h5"
            save_dm_response_matrix(result, path)

            loaded = load_dm_response_matrix(path)

            assert loaded.subaperture_mask is not None
            assert np.array_equal(loaded.subaperture_mask, subap_mask)

    def test_save_load_device_config(self):
        """Save with device_config dict, load back, verify config matches."""
        from ao_shaping.optimizer.wf.dm_response_matrix import (
            DMResponseMatrixResult,
            save_dm_response_matrix,
            load_dm_response_matrix,
        )

        matrix = np.random.randn(100, 63)
        variance = np.abs(np.random.randn(100, 63)) * 0.01

        config = {
            "dm": {"n_actuators": 64, "voltage_min": 0, "voltage_max": 200},
            "wfs": {"num_spots_x": 10, "num_spots_y": 5},
        }

        result = DMResponseMatrixResult(
            matrix=matrix,
            variance_matrix=variance,
            n_actuators=64,
            valid_actuator_indices=list(range(1, 64)),
            disturb_voltage=50.0,
            n_averages=20,
            n_cycles=1,
            wait_time=0.1,
            timestamp="2026-05-13T12:00:00",
            device_config=config,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dm_response_matrix.h5"
            save_dm_response_matrix(result, path)

            loaded = load_dm_response_matrix(path)

            assert loaded.device_config is not None
            assert loaded.device_config["dm"]["n_actuators"] == 64
            assert loaded.device_config["wfs"]["num_spots_x"] == 10

    def test_save_load_auto_h5_extension(self):
        """Save without .h5 suffix, verify .h5 is appended automatically."""
        from ao_shaping.optimizer.wf.dm_response_matrix import (
            save_dm_response_matrix,
            load_dm_response_matrix,
        )

        result = _make_result()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "no_ext"
            save_dm_response_matrix(result, path)

            actual_path = path.with_suffix(".h5")
            assert actual_path.exists(), "Loader should auto-add .h5 suffix"

            loaded = load_dm_response_matrix(path)
            assert np.allclose(loaded.matrix, result.matrix)

    def test_save_load_invalid_path(self):
        """Loading a non-existent file raises FileNotFoundError."""
        from ao_shaping.optimizer.wf.dm_response_matrix import load_dm_response_matrix

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nonexistent.h5"
            with pytest.raises(FileNotFoundError):
                load_dm_response_matrix(path)


# ===========================================================================
# TestMeasureActuatorResponse  (mock-based)
# ===========================================================================


class TestMeasureActuatorResponse:
    """Mock-based tests for measure_actuator_response."""

    def _make_mock_dm(self, n_actuators: int = 64) -> MagicMock:
        """Create a mock DM with send_voltages."""
        dm = MagicMock()
        dm.DM_NUM = n_actuators
        dm.send_voltages.return_value = None
        return dm

    def _make_mock_wfs(
        self,
        n_spots_x: int = 10,
        n_spots_y: int = 5,
        dev_x_offset: float = 0.0,
        dev_y_offset: float = 0.0,
    ) -> MagicMock:
        """Create a mock WFS that returns constant deviations."""
        wfs = MagicMock()
        wfs.num_spots_x = n_spots_x
        wfs.num_spots_y = n_spots_y
        wfs.take_image.return_value = None

        def get_spot_deviation(cancel_tile=False):
            return (
                np.full((n_spots_x, n_spots_y), dev_x_offset, dtype=np.float64),
                np.full((n_spots_x, n_spots_y), dev_y_offset, dtype=np.float64),
            )

        wfs.get_spot_deviation.side_effect = get_spot_deviation
        return wfs

    def test_response_shape_simple(self):
        """Call measure_actuator_response with n_cycles=1, n_averages=3; verify shapes."""
        from ao_shaping.optimizer.wf.dm_response_matrix import measure_actuator_response

        n_spots_x, n_spots_y = 10, 5
        dm = self._make_mock_dm(64)
        wfs = self._make_mock_wfs(n_spots_x=n_spots_x, n_spots_y=n_spots_y)

        n_spots_total = n_spots_x * n_spots_y
        expected_len = 2 * n_spots_total  # x + y slopes

        mean_resp, var_resp, dev_x, dev_y = measure_actuator_response(
            dm=dm,
            wfs=wfs,
            actuator_idx=5,
            disturb_voltage=50.0,
            n_averages=3,
            n_cycles=1,
            wait_time=0.01,
        )

        assert isinstance(mean_resp, np.ndarray)
        assert isinstance(var_resp, np.ndarray)
        assert isinstance(dev_x, np.ndarray)
        assert isinstance(dev_y, np.ndarray)

        assert mean_resp.shape == (expected_len,)
        assert var_resp.shape == (expected_len,)
        assert dev_x.shape == (n_spots_x, n_spots_y)
        assert dev_y.shape == (n_spots_x, n_spots_y)

    def test_response_shape_multiple_cycles(self):
        """Test with n_cycles=5 and verify shapes."""
        from ao_shaping.optimizer.wf.dm_response_matrix import measure_actuator_response

        dm = self._make_mock_dm(64)
        wfs = self._make_mock_wfs(n_spots_x=8, n_spots_y=6)
        n_spots_total = 8 * 6
        expected_len = 2 * n_spots_total

        mean_resp, var_resp, dev_x, dev_y = measure_actuator_response(
            dm=dm,
            wfs=wfs,
            actuator_idx=10,
            disturb_voltage=50.0,
            n_averages=1,
            n_cycles=5,
            wait_time=0.01,
        )

        assert mean_resp.shape == (expected_len,)
        assert var_resp.shape == (expected_len,)
        assert dev_x.shape == (8, 6)
        assert dev_y.shape == (8, 6)

    def test_differential_cancellation(self):
        """Verify response = (pos - neg) / (2 * V) with known slopes."""
        from ao_shaping.optimizer.wf.dm_response_matrix import measure_actuator_response

        dm = self._make_mock_dm(64)
        n_spots_x, n_spots_y = 2, 2
        n_spots_total = n_spots_x * n_spots_y
        expected_len = 2 * n_spots_total

        # Alternating pos/neg pattern: first call returns pos, then neg, etc.
        call_count = {"idx": 0}
        pos_values = np.arange(1, n_spots_total + 1, dtype=np.float64).reshape(n_spots_x, n_spots_y)
        neg_values = -np.arange(1, n_spots_total + 1, dtype=np.float64).reshape(n_spots_x, n_spots_y)

        def get_spot_deviation(cancel_tile=False):
            call_count["idx"] += 1
            # Cycle: 1=pos, 2=neg, 3=pos, 4=neg
            if call_count["idx"] % 2 == 1:
                return (pos_values.copy(), pos_values.copy())
            else:
                return (neg_values.copy(), neg_values.copy())

        wfs = MagicMock()
        wfs.num_spots_x = n_spots_x
        wfs.num_spots_y = n_spots_y
        wfs.take_image.return_value = None
        wfs.get_spot_deviation.side_effect = get_spot_deviation

        disturb_voltage = 1.0
        mean_resp, var_resp, dev_x, dev_y = measure_actuator_response(
            dm=dm,
            wfs=wfs,
            actuator_idx=5,
            disturb_voltage=disturb_voltage,
            n_averages=1,
            n_cycles=1,
            wait_time=0.01,
        )

        # Expected: (pos - neg) / (2 * V)
        # For pos=n, neg=-n: (n - (-n)) / 2 = n
        expected_response = np.arange(1, n_spots_total + 1, dtype=np.float64)
        expected_full = np.concatenate([expected_response, expected_response])
        assert np.allclose(mean_resp, expected_full)

        # dev_x and dev_y should also be (pos - neg) / (2*V)
        expected_dev = np.arange(1, n_spots_total + 1, dtype=np.float64).reshape(n_spots_x, n_spots_y)
        assert np.allclose(dev_x, expected_dev)
        assert np.allclose(dev_y, expected_dev)

    def test_truncated_mean_ignores_outlier(self):
        """With n_averages=10 and one outlier, verify truncated mean works.

        When n_averages > 5, the function uses truncated mean (drops min/max
        after sorting), so a single extreme outlier should not affect the result.
        """
        from ao_shaping.optimizer.wf.dm_response_matrix import measure_actuator_response

        dm = self._make_mock_dm(64)
        n_spots_x, n_spots_y = 2, 2

        call_index = {"idx": 0}

        def get_spot_deviation(cancel_tile=False):
            call_index["idx"] += 1
            call_num = call_index["idx"]

            # Normal values = 0.5; one outlier at call 3
            # The first n_averages calls are for pos side.
            # Call 3 = 3rd sample of pos side = outlier
            value = 0.5
            if call_num == 3:
                value = 50.0  # extreme outlier

            return (
                np.full((n_spots_x, n_spots_y), value, dtype=np.float64),
                np.full((n_spots_x, n_spots_y), value, dtype=np.float64),
            )

        wfs = MagicMock()
        wfs.num_spots_x = n_spots_x
        wfs.num_spots_y = n_spots_y
        wfs.take_image.return_value = None
        wfs.get_spot_deviation.side_effect = get_spot_deviation

        n_averages = 10
        mean_resp, var_resp, dev_x, dev_y = measure_actuator_response(
            dm=dm,
            wfs=wfs,
            actuator_idx=5,
            disturb_voltage=1.0,
            n_averages=n_averages,
            n_cycles=1,
            wait_time=0.01,
        )

        # Truncated mean should exclude the outlier (50.0)
        # pos side avg of 9 normal values = 0.5 (outlier dropped)
        # neg side avg of 10 normal values = 0.5 (all same)
        # response = (0.5 - 0.5) / (2 * 1.0) = 0.0
        assert np.allclose(mean_resp, 0.0, atol=1e-6), (
            f"Truncated mean should exclude outlier; got max={np.max(np.abs(mean_resp))}"
        )


# ===========================================================================
# TestDMResponseMatrixCalibration  (mock-based)
# ===========================================================================


class TestDMResponseMatrixCalibration:
    """Mock-based tests for calibrate_dm_response_matrix."""

    def _make_mock_dm(self, n_actuators: int = 64) -> MagicMock:
        dm = MagicMock()
        dm.DM_NUM = n_actuators
        dm.send_voltages.return_value = None
        return dm

    def _make_mock_wfs(
        self,
        n_spots_x: int = 10,
        n_spots_y: int = 5,
    ) -> MagicMock:
        """WFS that returns deterministic deviations."""
        wfs = MagicMock()
        wfs.num_spots_x = n_spots_x
        wfs.num_spots_y = n_spots_y
        wfs.take_image.return_value = None

        def get_spot_deviation(cancel_tile=False):
            return (
                np.zeros((n_spots_x, n_spots_y), dtype=np.float64),
                np.zeros((n_spots_x, n_spots_y), dtype=np.float64),
            )

        wfs.get_spot_deviation.side_effect = get_spot_deviation

        # Mock build_subaperture_mask
        def build_subaperture_mask(**kwargs):
            mask = np.ones((n_spots_x, n_spots_y), dtype=bool)
            valid_idx = np.where(mask.ravel())[0]
            return mask, valid_idx

        wfs.build_subaperture_mask.side_effect = build_subaperture_mask
        wfs.save_user_ref.return_value = None
        wfs.load_user_ref.return_value = None

        return wfs

    def test_calibrate_inverses(self):
        """Call with compute_inverses=True; verify pinv and lstsq shapes."""
        from ao_shaping.optimizer.wf.dm_response_matrix import calibrate_dm_response_matrix

        n_actuators = 64
        dm = self._make_mock_dm(n_actuators)
        wfs = self._make_mock_wfs(n_spots_x=10, n_spots_y=5)

        dm_unit_mask = np.ones(n_actuators, dtype=bool)
        dm_unit_mask[0] = False  # actuator 0 excluded

        n_spots_total = wfs.num_spots_x * wfs.num_spots_y
        n_slopes = 2 * n_spots_total

        with patch(
            "ao_shaping.optimizer.wf.dm_response_matrix.measure_actuator_response"
        ) as mock_measure:
            def measure_side_effect(**kwargs):
                response = np.random.randn(n_slopes) * 0.1
                variance = np.full(n_slopes, 0.01)
                dev_x = np.random.randn(wfs.num_spots_x, wfs.num_spots_y) * 0.1
                dev_y = np.random.randn(wfs.num_spots_x, wfs.num_spots_y) * 0.1
                return response, variance, dev_x, dev_y

            mock_measure.side_effect = measure_side_effect

            result = calibrate_dm_response_matrix(
                dm=dm,
                wfs=wfs,
                dm_unit_mask=dm_unit_mask,
                disturb_voltage=50.0,
                n_averages=20,
                n_cycles=1,
                wait_time=0.1,
                compute_inverses=True,
            )

        n_valid = int(np.sum(dm_unit_mask))
        assert result.matrix.shape == (n_slopes, n_valid)
        assert result.pinv_matrix is not None
        assert result.pinv_matrix.shape == (n_valid, n_slopes)
        assert result.lstsq_matrix is not None
        assert result.lstsq_matrix.shape == (n_valid, n_slopes)

    def test_calibrate_no_inverses(self):
        """Same but compute_inverses=False; verify pinv_matrix is None."""
        from ao_shaping.optimizer.wf.dm_response_matrix import calibrate_dm_response_matrix

        n_actuators = 64
        dm = self._make_mock_dm(n_actuators)
        wfs = self._make_mock_wfs(n_spots_x=10, n_spots_y=5)

        dm_unit_mask = np.ones(n_actuators, dtype=bool)
        dm_unit_mask[0] = False

        with patch(
            "ao_shaping.optimizer.wf.dm_response_matrix.measure_actuator_response"
        ) as mock_measure:
            n_slopes = 2 * wfs.num_spots_x * wfs.num_spots_y

            def measure_side_effect(**kwargs):
                response = np.random.randn(n_slopes) * 0.1
                variance = np.full(n_slopes, 0.01)
                dev_x = np.random.randn(wfs.num_spots_x, wfs.num_spots_y) * 0.1
                dev_y = np.random.randn(wfs.num_spots_x, wfs.num_spots_y) * 0.1
                return response, variance, dev_x, dev_y

            mock_measure.side_effect = measure_side_effect

            result = calibrate_dm_response_matrix(
                dm=dm,
                wfs=wfs,
                dm_unit_mask=dm_unit_mask,
                disturb_voltage=50.0,
                n_averages=20,
                n_cycles=1,
                wait_time=0.1,
                compute_inverses=False,
            )

        assert result.pinv_matrix is None
        assert result.lstsq_matrix is None
        assert result.matrix is not None

    def test_calibrate_skips_actuator_zero(self):
        """Verify actuator 0 is not measured when dm_unit_mask[0] is False."""
        from ao_shaping.optimizer.wf.dm_response_matrix import calibrate_dm_response_matrix

        n_actuators = 8
        dm = self._make_mock_dm(n_actuators)
        wfs = self._make_mock_wfs(n_spots_x=3, n_spots_y=2)

        dm_unit_mask = np.ones(n_actuators, dtype=bool)
        dm_unit_mask[0] = False
        dm_unit_mask[3] = False  # Also skip actuator 3

        measured_indices: list[int] = []

        with patch(
            "ao_shaping.optimizer.wf.dm_response_matrix.measure_actuator_response"
        ) as mock_measure:
            n_slopes = 2 * wfs.num_spots_x * wfs.num_spots_y  # = 12

            def measure_side_effect(**kwargs):
                measured_indices.append(kwargs["actuator_idx"])
                response = np.random.randn(n_slopes) * 0.01
                variance = np.abs(np.random.randn(n_slopes)) * 0.001
                dev_x = np.random.randn(wfs.num_spots_x, wfs.num_spots_y) * 0.01
                dev_y = np.random.randn(wfs.num_spots_x, wfs.num_spots_y) * 0.01
                return response, variance, dev_x, dev_y

            mock_measure.side_effect = measure_side_effect

            result = calibrate_dm_response_matrix(
                dm=dm,
                wfs=wfs,
                dm_unit_mask=dm_unit_mask,
                disturb_voltage=50.0,
                n_averages=20,
                n_cycles=1,
                wait_time=0.1,
                compute_inverses=False,
            )

        # Check actuator 0 and 3 were NOT measured
        assert 0 not in measured_indices, "Actuator 0 should be skipped"
        assert 3 not in measured_indices, "Actuator 3 should be skipped"

        # Check only valid actuators were measured
        valid_indices = np.where(dm_unit_mask)[0].tolist()
        assert sorted(measured_indices) == sorted(valid_indices)

        # Matrix columns should match number of valid actuators
        n_valid = int(np.sum(dm_unit_mask))
        assert result.matrix.shape[1] == n_valid, (
            f"Matrix should have {n_valid} columns (valid actuators), "
            f"got {result.matrix.shape[1]}"
        )
        assert result.valid_actuator_indices is not None
        assert result.valid_actuator_indices == valid_indices


# ===========================================================================
# TestLoadFailures
# ===========================================================================


class TestLoadFailures:
    """Tests for loading failures."""

    def test_load_nonexistent(self):
        """Verify FileNotFoundError for non-existent file."""
        from ao_shaping.optimizer.wf.dm_response_matrix import load_dm_response_matrix

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "does_not_exist.h5"
            with pytest.raises(FileNotFoundError):
                load_dm_response_matrix(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
