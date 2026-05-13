"""Unit tests for MicroDM driver.

Tests exception classes, voltage conversion, SimMicroDM interface,
and package imports. The real MicroDM requires async TCP and hardware,
so it is tested via the simulation class SimMicroDM.

SimMicroDM inherits from DM (pure ABC), so Device methods are NOT available
on SimMicroDM unless explicitly added.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.drivers.dm.MicroDM import (
    MicroDM,
    MicroDMError,
    MicroDMConnectionError,
    MicroDMVoltageError,
    RelayState,
    voltage_to_bytes,
    voltage_to_bytes_clipped,
    VOLTAGE_MIN,
    VOLTAGE_MAX,
    MAX_CHANNELS,
)
from ao_shaping.drivers.sim.dm import SimMicroDM


# =============================================================================
# Voltage Conversion
# =============================================================================

class TestVoltageConversion:
    """Tests for the voltage-to-byte conversion formula.

    Reference formula (MATLAB R50PowerV1.m):
        value = (voltage + 20) / 20 / 3.4 / 3.3 * 65535.0
        high_byte = floor(value / 255)
        low_byte = floor(mod(value, 256))
    """

    def test_voltage_minus_20(self):
        """At -20 V the raw value should be ~0 → high=0, low=0."""
        hv, lv = voltage_to_bytes(-20.0)
        assert hv == 0
        assert lv == 0

    def test_voltage_zero(self):
        """At 0 V the conversion should produce reasonable byte values."""
        hv, lv = voltage_to_bytes(0.0)
        # value = 20 / 20 / 3.4 / 3.3 * 65535 ≈ 5840.8
        # raw = int(5840.8 + 0.5) = 5841
        # high = 5841 // 256 = 22, low = 5841 % 256 = 209
        assert hv == 22
        assert lv == 209

    def test_voltage_120(self):
        """At 120 V the conversion should produce reasonable byte values."""
        hv, lv = voltage_to_bytes(120.0)
        # value = 140 / 20 / 3.4 / 3.3 * 65535 ≈ 40886.6
        # raw = int(40886.6 + 0.5) = 40886 (note: float precision)
        # high = 40886 // 256 = 159, low = 40886 % 256 = 182
        assert hv == 159
        assert lv == 182

    def test_voltage_clipped_low(self):
        """Values below -20 V should be clipped to -20 V."""
        hv, lv = voltage_to_bytes_clipped(-30.0)
        assert hv == 0
        assert lv == 0

    def test_voltage_clipped_high(self):
        """Values above 120 V should be clipped to 120 V."""
        hv, lv = voltage_to_bytes_clipped(150.0)
        assert hv == 159
        assert lv == 182


# =============================================================================
# Exception Classes
# =============================================================================

class TestMicroDMExceptions:
    def test_micro_dm_error(self):
        with pytest.raises(MicroDMError):
            raise MicroDMError("Test error")

    def test_micro_dm_connection_error(self):
        with pytest.raises(MicroDMConnectionError):
            raise MicroDMConnectionError("Connection failed")

    def test_micro_dm_voltage_error(self):
        with pytest.raises(MicroDMVoltageError):
            raise MicroDMVoltageError("Voltage out of range")


# =============================================================================
# SimMicroDM Tests
# =============================================================================

class TestSimMicroDM:
    @pytest.fixture
    def dm(self):
        return SimMicroDM()

    def test_initialization(self, dm):
        assert dm.DM_Num == MAX_CHANNELS
        assert dm.V_Min == VOLTAGE_MIN
        assert dm.V_Max == VOLTAGE_MAX

    def test_transform(self, dm):
        cmd = np.array([0.0, 0.5, -0.5, 1.0, -1.0])
        voltages = dm.transform(cmd)
        # V_Min = -20, V_Max = 120, midpoint = 50
        # f(x) = (x + 1) * (120 - (-20)) / 2 + (-20)
        #      = (x + 1) * 70 - 20
        # f(0.0)   = 1 * 70 - 20 = 50
        # f(0.5)   = 1.5 * 70 - 20 = 85
        # f(-0.5)  = 0.5 * 70 - 20 = 15
        # f(1.0)   = 2 * 70 - 20 = 120
        # f(-1.0)  = 0 * 70 - 20 = -20
        expected = np.array([50.0, 85.0, 15.0, 120.0, -20.0])
        np.testing.assert_array_almost_equal(voltages, expected)

    def test_send_voltages(self, dm):
        """Sending voltages should store them and return a copy."""
        vs = np.array([10.0, 20.0, 30.0])
        padded = np.zeros(50)
        padded[:3] = [10.0, 20.0, 30.0]
        result = dm.send_voltages(padded)
        np.testing.assert_array_equal(result, padded)
        np.testing.assert_array_equal(dm.get_actuator_positions(), padded)

    def test_send_voltages_wrong_shape(self, dm):
        """Sending wrong-sized array should raise."""
        with pytest.raises(MicroDMVoltageError):
            dm.send_voltages(np.array([1.0, 2.0, 3.0]))

    def test_send_scalar(self, dm):
        """Sending a scalar should set all channels."""
        result = dm.send(50.0)
        expected = np.full(50, 50.0)
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(dm.get_actuator_positions(), expected)

    def test_send_array(self, dm):
        """Sending an array should store and return it."""
        vs = np.full(50, 30.0)
        result = dm.send(vs)
        np.testing.assert_array_equal(result, vs)

    def test_set_channel_voltage(self, dm):
        """Setting a single channel should update only that channel."""
        dm.set_channel_voltage(5, 42.0)
        positions = dm.get_actuator_positions()
        assert positions[5] == 42.0
        assert positions[0] == 0.0  # Other channels unchanged

    def test_set_channel_voltage_out_of_range(self, dm):
        """Setting invalid channel should raise."""
        with pytest.raises(MicroDMVoltageError):
            dm.set_channel_voltage(999, 1.0)

    def test_relay_state(self, dm):
        """Relay state should track open/close."""
        dm.set_relay_state(True)
        assert dm.get_hardware_info()["relay_state"] == "ON"
        dm.set_relay_state(False)
        assert dm.get_hardware_info()["relay_state"] == "OFF"

    def test_reset_all(self, dm):
        """Reset should zero all voltages."""
        dm.send_voltages(np.full(50, 50.0))
        dm.reset_all()
        np.testing.assert_array_equal(
            dm.get_actuator_positions(),
            np.zeros(50),
        )

    def test_open_close(self, dm):
        """Open/close should transition state."""
        assert not dm.is_connected()
        dm.open()
        assert dm.is_connected()
        dm.close()
        assert not dm.is_connected()

    def test_transform_clipping(self, dm):
        """Transform should clip input to [-1, 1]."""
        cmd = np.array([-2.0, 2.0])
        voltages = dm.transform(cmd)
        # -2 → -1 → -20.0
        #  2 →  1 → 120.0
        expected = np.array([-20.0, 120.0])
        np.testing.assert_array_equal(voltages, expected)


# =============================================================================
# Integration / Import Tests
# =============================================================================

class TestMicroDMIntegration:
    def test_import_from_package(self):
        from ao_shaping.drivers.dm import MicroDM, SimMicroDM

        assert MicroDM is not None
        assert SimMicroDM is not None

    def test_exception_imports(self):
        from ao_shaping.drivers.dm import (
            MicroDMError,
            MicroDMConnectionError,
            MicroDMVoltageError,
        )

        assert MicroDMError is not None
        assert MicroDMConnectionError is not None
        assert MicroDMVoltageError is not None

    def test_class_constants(self):
        assert MicroDM.DM_Num == 39 * 39
        assert MicroDM.V_Min == VOLTAGE_MIN
        assert MicroDM.V_Max == VOLTAGE_MAX

    def test_relay_state_enum(self):
        assert RelayState.OFF.value != RelayState.ON.value
