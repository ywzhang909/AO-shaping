from ao_shaping.drivers import NlightDM
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
