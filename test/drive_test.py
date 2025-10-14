import numpy as np
import matplotlib.pyplot as plt

from ao_shaping.drivers import CameraStreamManager, NlightDM

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
    with NlightDM() as dm, CameraStreamManager() as cam:
        assert vs.shape == (dm.DM_Num,)
        dm.send_voltages(vs, 0)
        img = cam.get_numpy_image()
        plt.imshow(img)
        plt.show()
        
        
if __name__ == "__main__":
    vs = np.loadtxt("data/to_load_V-9635.0.csv")
    test_all(vs)
    turn_off_dm()