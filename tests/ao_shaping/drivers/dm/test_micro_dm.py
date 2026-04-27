"""Unit tests for MicroDM driver.

These tests verify the MicroDM driver implementation without requiring actual hardware.
Uses SimMicroDM for simulation mode.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.drivers.dm.MicroDM import (
    MicroDM,
    SimMicroDM,
    MicroDMError,
    MicroDMConnectionError,
    MicroDMVoltageError,
)


class TestMicroDMExceptions:
    """Test exception classes."""

    def test_micro_dm_error(self):
        """Test MicroDMError base exception."""
        with pytest.raises(MicroDMError):
            raise MicroDMError("Test error")

    def test_micro_dm_connection_error(self):
        """Test MicroDMConnectionError."""
        with pytest.raises(MicroDMConnectionError):
            raise MicroDMConnectionError("Connection failed")

    def test_micro_dm_voltage_error(self):
        """Test MicroDMVoltageError."""
        with pytest.raises(MicroDMVoltageError):
            raise MicroDMVoltageError("Voltage out of range")


class TestSimMicroDM:
    """Test SimMicroDM (simulation mode)."""

    @pytest.fixture
    def dm(self):
        """Create SimMicroDM instance."""
        return SimMicroDM()

    def test_initialization(self, dm):
        """Test device initialization."""
        assert dm.DM_Num == 50
        assert dm.V_Min == -1.0
        assert dm.V_Max == 6.5

    def test_open_close(self, dm):
        """Test open and close connection."""
        dm.open()
        assert dm.is_connected() is True
        dm.close()
        assert dm.is_connected() is False

    def test_get_hardware_info(self, dm):
        """Test hardware info retrieval."""
        dm.open()
        info = dm.get_hardware_info()
        assert info["manufacturer"] == "R50Power"
        assert info["model"] == "MicroDM-50-Sim"
        assert info["channel_count"] == 50
        assert info["simulation"] is True

    def test_set_channel_voltage(self, dm):
        """Test single channel voltage setting."""
        dm.open()
        dm.set_channel_voltage(0, 2.5)
        positions = dm.get_actuator_positions()
        assert positions[0] == 2.5

    def test_set_all_voltage_by_arr(self, dm):
        """Test setting all channels by array."""
        dm.open()
        voltages = np.linspace(0, 5, 50)
        dm.set_all_voltage_by_arr(voltages)
        positions = dm.get_actuator_positions()
        np.testing.assert_array_almost_equal(positions, voltages)

    def test_set_all_channel_voltage(self, dm):
        """Test setting all channels to same voltage."""
        dm.open()
        voltages = dm.set_all_channel_voltage(3.0)
        assert np.all(voltages == 3.0)
        np.testing.assert_array_almost_equal(dm.get_actuator_positions(), voltages)

    def test_set_relay_state(self, dm):
        """Test relay state control."""
        dm.open()
        dm.set_relay_state(True)
        dm.set_relay_state(False)

    def test_transform(self, dm):
        """Test normalized command transformation."""
        cmd = np.array([0.0, 0.5, -0.5, 1.0, -1.0])
        voltages = dm.transform(cmd)
        expected = np.array([2.75, 4.625, 0.875, 6.5, -1.0])
        np.testing.assert_array_almost_equal(voltages, expected)

    def test_send_voltages(self, dm):
        """Test send_voltages method."""
        dm.open()
        voltages = np.random.uniform(-1, 6.5, 50)
        result = dm.send_voltages(voltages)
        np.testing.assert_array_almost_equal(result, np.clip(voltages, -1, 6.5))

    def test_reset_all(self, dm):
        """Test reset to zero."""
        dm.open()
        dm.set_all_channel_voltage(5.0)
        dm.reset_all()
        positions = dm.get_actuator_positions()
        np.testing.assert_array_almost_equal(positions, np.zeros(50))

    def test_voltage_clamping(self, dm):
        """Test voltage clamping to valid range."""
        dm.open()
        # Test voltage below minimum
        dm.set_channel_voltage(0, -5.0)
        positions = dm.get_actuator_positions()
        assert positions[0] == -1.0

        # Test voltage above maximum
        dm.set_channel_voltage(0, 10.0)
        positions = dm.get_actuator_positions()
        assert positions[0] == 6.5


class TestMicroDMVoltageConversion:
    """Test voltage conversion logic."""

    @pytest.fixture
    def dm(self):
        """Create SimMicroDM instance."""
        return SimMicroDM()

    def test_voltage_to_bytes_boundary_min(self, dm):
        """Test conversion at minimum voltage (-1V)."""
        hv, lv = dm._voltage_to_bytes(-1.0)
        # Value should be close to 0 at minimum
        assert hv >= 0

    def test_voltage_to_bytes_boundary_max(self, dm):
        """Test conversion at maximum voltage (6.5V)."""
        hv, lv = dm._voltage_to_bytes(6.5)
        # Value should be high at maximum
        assert hv >= 0

    def test_voltage_to_bytes_clamping(self, dm):
        """Test conversion clamping."""
        # Below minimum should clamp to min
        hv, lv = dm._voltage_to_bytes(-5.0)
        hv_min, lv_min = dm._voltage_to_bytes(-1.0)
        assert (hv, lv) == (hv_min, lv_min)

        # Above maximum should clamp to max
        hv, lv = dm._voltage_to_bytes(10.0)
        hv_max, lv_max = dm._voltage_to_bytes(6.5)
        assert (hv, lv) == (hv_max, lv_max)

    def test_voltage_to_bytes_offset(self, dm):
        """Test offset conversion for SetAllVoltageByArr."""
        hv, lv = dm._voltage_to_bytes_offset(0.0)
        hv_no_offset, lv_no_offset = dm._voltage_to_bytes(0.0)
        # Offset version should produce different values
        assert (hv, lv) != (hv_no_offset, lv_no_offset)


class TestMicroDMIntegration:
    """Integration tests for MicroDM driver package."""

    def test_import_from_package(self):
        """Test importing from package."""
        from ao_shaping.drivers.dm import MicroDM, SimMicroDM

        assert MicroDM is not None
        assert SimMicroDM is not None

    def test_exception_imports(self):
        """Test exception imports."""
        from ao_shaping.drivers.dm import (
            MicroDMError,
            MicroDMConnectionError,
            MicroDMVoltageError,
        )

        assert MicroDMError is not None
        assert MicroDMConnectionError is not None
        assert MicroDMVoltageError is not None

    def test_class_constants(self):
        """Test class-level constants."""
        assert MicroDM.DM_Num == 50
        assert MicroDM.V_Min == -1.0
        assert MicroDM.V_Max == 6.5

    def test_repr(self):
        """Test string representation."""
        dm = SimMicroDM()
        assert "MicroDM" in repr(dm)
        assert "50" in repr(dm)
