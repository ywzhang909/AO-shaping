import os

import numpy as np
import pytest

from ao_shaping.utils.spots_calc import centroid


pytestmark = pytest.mark.skip(reason="Requires NlightDM + CameraStreamManager + Thorlab_WFS hardware")


def test_all():
    from ao_shaping.drivers import CameraStreamManager, NlightDM, Thorlab_WFS
    from ao_shaping.utils import get_init_V_by_rms

    vs = get_init_V_by_rms()
    with NlightDM(keep_when_exit=True) as dm, \
        CameraStreamManager(cam_id=os.environ.get('Far_Cam_ID', 0), exposure_time_ms=500) as cam_in, \
        Thorlab_WFS() as wfs:
        assert vs.shape == (dm.DM_Num,)
        dm.send_voltages(vs, 0.1)


def test_init_V():
    from ao_shaping.drivers import CameraStreamManager, NlightDM

    vs = np.loadtxt('D:/workspace/AO-shaping/data/flatten_voltages/20251201/20251201-1.csv')
    with NlightDM(keep_when_exit=True) as dm, \
        CameraStreamManager(cam_id=os.environ.get('Far_Cam_ID', 0), exposure_time_ms=500) as cam_in:
        assert vs.shape == (dm.DM_Num,)
        dm.send_voltages(vs, 0.1)
        img_in = cam_in.get_numpy_image()
        center = centroid(img_in)
        print(f"in focal. center: {center}")
