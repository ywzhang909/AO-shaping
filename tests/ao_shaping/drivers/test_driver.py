import os

import numpy as np
import matplotlib.pyplot as plt
import pytest

from ao_shaping.drivers import CameraStreamManager, NlightDM, Thorlab_WFS
from ao_shaping.utils import get_init_V_by_rms, get_init_V_by_energy
from ao_shaping.utils.spots_calc import centroid


def test_all():
    """Test all hardware together - requires all devices"""
    pytest.skip("Requires all hardware (DM, CCD, WFS)")
