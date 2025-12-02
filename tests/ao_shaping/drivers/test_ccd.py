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
        plt.title(f'{center=}')
        plt.show()

def test_autoset_exposure_time(cam_id=0):
    """
    测试自动设置曝光时间功能
    """
    with CameraStreamManager(cam_id, exposure_time_ms=50) as cam:
        # 测试自动设置曝光时间功能
        initial_exposure = cam.exposure_time_ms
        print(f"初始曝光时间: {initial_exposure}ms")
        
        # 调用自动设置曝光时间方法
        # 使用较低的目标亮度值以避免在测试环境中过度调整
        final_exposure = cam.autoset_exposure_time_ms(n_sample=10, target_max_brightness=100, threshold=0.2)
        # 验证曝光时间已被修改
        assert final_exposure != initial_exposure, "曝光时间应该被调整"

        # 测试自动曝光的范围正确
        img = cam.get_numpy_image(10)
        max_brightness = np.max(img)
        assert (1-0.2)*100 <= max_brightness <= (1+0.2)*100, f"自动曝光时间范围错误：{max_brightness:.2f}不在({(1-0.2)*100:.2f}ms, {(1+0.2)*100:.2f}ms)范围"