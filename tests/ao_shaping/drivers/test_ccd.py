import numpy as np
import matplotlib.pyplot as plt

from ao_shaping.utils.spots_calc import centroid
from ao_shaping.drivers import CameraStreamManager

def test_cam_list():
        cam_list = CameraStreamManager.get_cam_list()
        for cam in cam_list:
            print(cam)
    
def test_cam(cam_id=0):
    
    with CameraStreamManager(cam_id, exposure_time_ms=50) as cam:
        img = cam.get_numpy_image(10)
        # center = np.unravel_index(np.argmax(img), img.shape)
        # center = (center[1], center[0])
        center = centroid(img)
        print(f'{center=}')
        plt.imshow(img)
        plt.title(f'{center=} = {img[center[::-1]]=}')
        plt.show()