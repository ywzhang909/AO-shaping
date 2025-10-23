import numpy as np
import matplotlib.pyplot as plt

from ao_shaping.drivers import CameraStreamManager, NlightDM, Thorlab_WFS

def test_dm():
    with NlightDM() as dm:
        voltages = np.zeros((dm.DM_Num))
        for i in range(1_000_000):
            v = np.sin(2 * np.pi * i / 1_000_000) * 100
            voltages[1] = v
            dm.send_voltages(voltages, 0)
        
def turn_off_dm():
    with NlightDM(keep_when_exit=False) as dm:
        dm.send_voltages(np.zeros((dm.DM_Num)))
        dm.reset_all()
    
def test_cam():
    with CameraStreamManager() as cam:
        img = cam.get_numpy_image()
        plt.imshow(img)
        plt.show()
        
def test_all(vs: np.ndarray):
    with NlightDM(keep_when_exit=True) as dm, \
        CameraStreamManager(cam_id=0, exposure_time_ms=80) as cam_in, \
            CameraStreamManager(cam_id=1, exposure_time_ms=1000) as cam_out, \
                Thorlab_WFS() as wfs:
        assert vs.shape == (dm.DM_Num,)
        dm.send_voltages(vs, 0)
        
        img_mla = wfs.get_spotfiled_image()
        img_in = cam_in.get_numpy_image()
        img_out = cam_out.get_numpy_image()
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
        ax1.imshow(img_in)
        ax1.set_title("in focal")
        ax2.imshow(img_out)
        ax2.set_title("out focal")
        ax3.imshow(img_mla)
        ax3.set_title("MLA")
        plt.show()
        
        
if __name__ == "__main__":
    from ao_shaping.utils import get_init_V_by_rms
    vs = get_init_V_by_rms()
    test_all(vs)
    # turn_off_dm()