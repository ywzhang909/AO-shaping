import matplotlib.pyplot as plt
import numpy as np
import pytest

from ao_shaping.drivers import Thorlab_WFS, MlaRes


def test_wfs():
    """Test WFS (Wavefront Sensor) - requires hardware"""
    pytest.skip("Requires WFS hardware")


def test_rms():
    """Test RMS measurement - requires hardware"""
    pytest.skip("Requires WFS hardware")
