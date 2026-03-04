import numpy as np
import matplotlib.pyplot as plt
import pytest

from ao_shaping.utils.spots_calc import centroid
from ao_shaping.drivers import CameraStreamManager


def test_cam_list():
    cam_list = CameraStreamManager.get_cam_list()
    for cam in cam_list:
        print(cam)


def test_cam(cam_id=0):
    """Test CCD camera - requires hardware"""
    pytest.skip("Requires CCD camera hardware")
