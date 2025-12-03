import numpy as np
import matplotlib.pyplot as plt

from ao_shaping.utils.spots_calc import centroid
from ao_shaping.drivers import CameraStreamManager

def test_cam_list():
        cam_list = CameraStreamManager.get_cam_list()
        for cam in cam_list:
            print(cam)
    
def test_cam(cam_id=0):
    
    with CameraStreamManager(cam_id, exposure_time_ms=500) as cam:
        exp_times = list(range(1, 51, 10))
        # center = np.unravel_index(np.argmax(img), img.shape)
        # center = (center[1], center[0])
        fig, ax_list = plt.subplots(1, len(exp_times))
        for i, t in enumerate(exp_times):
            img = cam.get_numpy_image(t)
            center = centroid(img)
            im = ax_list[i].imshow(img, vmin=0, vmax=255)
            ax_list[i].set_title(f'{t}: {center=} {np.max(img)=:.2f}')
        plt.colorbar(im, ax=ax_list, orientation='horizontal')
        plt.show()

def test_autoset_exposure_time(cam_id=0):
    """
    测试自动设置曝光时间功能
    """
    target_b = 100
    with CameraStreamManager(cam_id, exposure_time_ms=50) as cam:
        # 测试自动设置曝光时间功能
        initial_exposure = cam.exposure_time
        
        # 调用自动设置曝光时间方法
        # 使用较低的目标亮度值以避免在测试环境中过度调整
        cam.autoset_exposure_time_ms(target_max_brightness=target_b, threshold=20)
        # 验证曝光时间已被修改
        assert cam.exposure_time != initial_exposure, "曝光时间应该被调整"

        # 测试自动曝光的范围正确
        img = cam.get_numpy_image(10)
        max_brightness = np.max(img)
        assert target_b-20 <= max_brightness <= target_b+20, f"自动曝光时间范围错误：{max_brightness:.2f}不在({target_b-20:.2f}, {target_b+20:.2f})范围"
