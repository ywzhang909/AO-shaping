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


class TestDMRegistry:
    """Tests for the DM registry system."""

    def test_registry_has_types(self):
        """Test registry lists registered DM types."""
        from ao_shaping.drivers.dm._registry import get_dm_registry

        registry = get_dm_registry()
        types = registry.list_types()
        assert "nlight" in types
        assert "micro" in types
        assert "zernike" in types
        assert "hadamard" in types
        assert "sim_micro" in types

    def test_registry_has_type(self):
        """Test has_type validation."""
        from ao_shaping.drivers.dm._registry import get_dm_registry

        registry = get_dm_registry()
        assert registry.has_type("nlight")
        assert registry.has_type("NLIGHT")
        assert not registry.has_type("unknown")

    def test_registry_create(self):
        """Test creating DM via registry."""
        from ao_shaping.drivers.dm._registry import get_dm_registry

        registry = get_dm_registry()
        dm = registry.create("sim_micro")
        assert dm.DM_NUM == 50

    def test_registry_create_unknown(self):
        """Test creating unknown DM type raises ValueError."""
        from ao_shaping.drivers.dm._registry import get_dm_registry

        registry = get_dm_registry()
        with pytest.raises(ValueError, match="Unknown DM type"):
            registry.create("nonexistent")

    def test_create_dm_filters_kwargs(self):
        """Test create_dm filters kwargs per type."""
        from ao_shaping.drivers.dm import create_dm

        dm = create_dm("sim_micro", device_id="test", safety_mode=False, bogus=999)
        assert dm._device_id == "test"
        assert dm._safety_mode is False

    def test_create_dm_alias(self):
        """Test create_dm maps legacy dm_neibor_diff → max_neibor_diff for nlight."""
        from ao_shaping.drivers.dm._registry import _KWARG_ALIASES

        assert "dm_neibor_diff" in _KWARG_ALIASES.get("nlight", {})
        assert _KWARG_ALIASES["nlight"]["dm_neibor_diff"] == "max_neibor_diff"

    def test_is_reachable_software_dm(self):
        """Test software-only DMs always report reachable."""
        from ao_shaping.drivers.dm.zernike_dm import ZernikeDM
        from ao_shaping.drivers.dm.hadamard_dm import HadamardDM
        from ao_shaping.drivers.sim.dm import SimMicroDM

        assert ZernikeDM.is_reachable() is True
        assert HadamardDM.is_reachable() is True
        assert SimMicroDM.is_reachable() is True

    def test_list_reachable_types(self):
        """Test list_reachable_dm_types returns software DMs at minimum."""
        from ao_shaping.drivers.dm import list_reachable_dm_types

        reachable = list_reachable_dm_types()
        assert "zernike" in reachable
        assert "hadamard" in reachable
        assert "sim_micro" in reachable
