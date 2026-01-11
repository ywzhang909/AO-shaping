import os

import numpy as np
import matplotlib.pyplot as plt
import pytest

from ao_shaping.drivers import CameraStreamManager, NlightDM, Thorlab_WFS
from ao_shaping.utils import get_init_V_by_rms, get_init_V_by_energy
from ao_shaping.utils.spots_calc import centroid
        
@pytest.mark.experiment
def test_all():
    # vs = get_init_V_by_energy()
    vs = get_init_V_by_rms()
    with NlightDM(keep_when_exit=True) as dm, \
        CameraStreamManager(cam_id=os.environ.get('Far_Cam_ID', 0), exposure_time_ms=500) as cam_in, \
        Thorlab_WFS() as wfs:
        assert vs.shape == (dm.DM_Num,)
        dm.send_voltages(vs, 0.1)

        img_mla = wfs.get_spotfiled_image()
        img_in = cam_in.get_numpy_image()
        # img_out = cam_out.get_numpy_image()

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 4))
        ax1.imshow(img_in)
        ax1.set_title(f"in focal. center: {centroid(img_in)}")
        # ax2.imshow(img_out)
        # ax2.set_title("out focal")
        ax3.imshow(img_mla)
        ax3.set_title("MLA")
        ax4.bar(np.arange(len(vs)), vs)
        plt.show()


@pytest.mark.experiment
def test_init_V():
    vs = np.loadtxt('D:/workspace/AO-shaping/data/flatten_voltages/20251201/20251201-1.csv')
    with NlightDM(keep_when_exit=True) as dm, \
        CameraStreamManager(cam_id=os.environ.get('Far_Cam_ID', 0), exposure_time_ms=500) as cam_in:
        assert vs.shape == (dm.DM_Num,)
        dm.send_voltages(vs, 0.1)
        img_in = cam_in.get_numpy_image()

        center = centroid(img_in)
        print(f"in focal. center: {center}")
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 4))
        ax1.imshow(img_in)
        ax1.set_title(f"in focal. center: {centroid(np.where(img_in>np.mean(img_in), 1, 0))}")
        ax4.bar(np.arange(len(vs)), vs)
        plt.show()