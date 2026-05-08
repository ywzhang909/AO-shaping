"""Unit tests for MicroDM driver.

Tests exception classes, SimMicroDM basic interface, and package imports.
SimMicroDM inherits from DM (pure ABC), not Device — Device methods like
_set_state and register_parameter are not available.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.drivers.dm.MicroDM import (
    MicroDM,
    MicroDMError,
    MicroDMConnectionError,
    MicroDMVoltageError,
)
from ao_shaping.drivers.sim.dm import SimMicroDM


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


class TestSimMicroDM:
    @pytest.fixture
    def dm(self):
        return SimMicroDM()

    def test_initialization(self, dm):
        assert dm.DM_Num == 50
        assert dm.V_Min == -1.0
        assert dm.V_Max == 6.5

    def test_transform(self, dm):
        cmd = np.array([0.0, 0.5, -0.5, 1.0, -1.0])
        voltages = dm.transform(cmd)
        expected = np.array([2.75, 4.625, 0.875, 6.5, -1.0])
        np.testing.assert_array_almost_equal(voltages, expected)


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
        assert MicroDM.DM_Num == 50
        assert MicroDM.V_Min == -1.0
        assert MicroDM.V_Max == 6.5
