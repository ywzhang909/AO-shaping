from ao_shaping.drivers import NlightDM
from ao_shaping.drivers.dm.base import DM
from ao_shaping.utils.file import get_init_V_by_energy

import numpy as np
import pytest


def test_dm():
    """Test DM (Deformable Mirror) - requires hardware and Drv_UDPST.dll"""
    pytest.skip("Requires DM hardware")


def test_turn_off_dm():
    """Test turning off DM - requires hardware"""
    pytest.skip("Requires DM hardware")


def test_last_voltage():
    """Test getting last voltage - requires data file and hardware"""
    pytest.skip("Requires DM hardware and data files")


class TestDMBase:
    """Tests for the DM base class interface."""

    def test_transform_voltage(self):
        """Test transform_voltage maps [-1, 1] to [V_Min, V_Max]."""
        # Create a concrete subclass for testing
        from ao_shaping.drivers.sim.dm import SimMicroDM

        dm = SimMicroDM()
        cmd = np.array([0.0, 0.5, -0.5, 1.0, -1.0])
        voltages = dm.transform_voltage(cmd)
        # V_Min = -20, V_Max = 120
        # f(x) = (x + 1) * (120 - (-20)) / 2 + (-20) = (x + 1) * 70 - 20
        expected = np.array([50.0, 85.0, 15.0, 120.0, -20.0])
        np.testing.assert_array_almost_equal(voltages, expected)

    def test_transform_voltage_clipping(self):
        """Test transform_voltage clips input to [-1, 1]."""
        from ao_shaping.drivers.sim.dm import SimMicroDM

        dm = SimMicroDM()
        cmd = np.array([-2.0, 2.0])
        voltages = dm.transform_voltage(cmd)
        np.testing.assert_array_equal(voltages, np.array([-20.0, 120.0]))

    def test_safety_mode_default(self):
        """Test safety_mode defaults to True."""
        from ao_shaping.drivers.sim.dm import SimMicroDM

        dm = SimMicroDM()
        assert dm._safety_mode is True

    def test_safety_mode_off(self):
        """Test safety_mode can be disabled."""
        from ao_shaping.drivers.sim.dm import SimMicroDM

        dm = SimMicroDM(safety_mode=False)
        assert dm._safety_mode is False

    def test_dm_num_alias(self):
        """Test DM_Num alias matches DM_NUM."""
        from ao_shaping.drivers.sim.dm import SimMicroDM

        dm = SimMicroDM()
        assert dm.DM_Num == dm.DM_NUM
