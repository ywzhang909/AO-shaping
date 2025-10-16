import logging
import numpy as np

import gxipy as gx

log = logging.getLogger(__file__)

class CameraStreamManager:
    def __init__(self, cam_id:int=0, exposure_time_ms:int=20, skip_sampling=False):
        self.device_manager = gx.DeviceManager()
        self.cam_id = cam_id
        self.exposure_time_ms = exposure_time_ms
        self.skip_sampling = skip_sampling

        self.cam, self.__sn = None, None
        self.cam_width ,self.cam_height = 0, 0

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.cam:
            self.cam_width ,self.cam_height = 0, 0
            self.cam.stream_off()
            self.cam.close_device()
            self.cam, self.__sn = None, None

    def initialize(self):
        """
        初始化相机设备。

        此方法执行以下操作：
        1. 关闭之前打开的相机设备（如果有）。
        2. 更新设备列表并检查是否有足够的设备。
        3. 打开指定的相机设备。
        4. 设置相机的曝光时间、增益、像素格式、采样方式、偏移量、宽度和高度。
        5. 更新相机的属性并开启数据流。

        如果没有找到相机设备，将记录错误并抛出连接中止错误。

        参数:
            无

        返回:
            无
        """
        # 关闭之前打开的相机设备（如果有）
        self.__exit__(None, None, None)

        # 更新设备列表并获取设备信息列表
        _, dev_info_list = self.device_manager.update_device_list()
        # 检查设备列表长度是否小于等于指定的相机ID
        if len(dev_info_list) <= self.cam_id:
            log.error("No devices found.")
            raise ConnectionAbortedError("No cam devices found.")

        sn = dev_info_list[self.cam_id].get("sn")
        self.cam = self.device_manager.open_device_by_sn(sn)
        # 设置相机的曝光时间
        self.cam.ExposureTime.set(self.exposure_time_ms)
        # 设置相机的增益
        self.cam.Gain.set(0.0)
        # 设置相机的像素格式为MONO8
        self.cam.PixelFormat.set(gx.GxPixelFormatEntry.MONO8)
        if self.skip_sampling:
            # 设置相机的合并因子为2
            self.cam.BinningHorizontal.set(2)
            self.cam.BinningVertical.set(2)

        # 设置相机的水平偏移量为0
        self.cam.OffsetX.set(0)
        self.cam.OffsetY.set(0)
        # 设置相机的宽度为最大宽度
        self.cam.Width.set(self.cam.WidthMax.get())
        self.cam.Height.set(self.cam.HeightMax.get())

        self.__sn = sn
        self.__update_properties()
        self.cam.stream_on()

    def reset_exposure_time(self, time_ms:int):
        if time_ms >= 20:
            self.exposure_time_ms = time_ms
        else:
            self.exposure_time_ms = 20
            log.warning('exposure time must >= 20. set to 20.')
        self.cam.ExposureTime.set(self.exposure_time_ms)
        return self.exposure_time_ms

    def reset_window(self, center:tuple[int,int]|tuple[np.intp,...]=(0,0), size:tuple[int,int]=(0,0)) -> tuple[tuple[int,int], tuple[int,int]]:
        """
        重置相机的窗口大小和位置，以确保图像的中心位于指定的位置。

        参数:
        size (Tuple[int]): 期望的窗口大小，格式为 (宽度, 高度)。
        center (Tuple[int]): 期望的窗口中心位置，格式为 (x坐标, y坐标)。

        返回:
        Tuple[int]: 新的窗口中心位置，格式为 (x坐标, y坐标)。
        """
        # 中心坐标大于0
        center = tuple(int(c) for c in center)
        self.cam.stream_off()
        # 如果未指定窗口大小，则使用相机的最大宽度和高度
        if size == (0, 0):
            width, height = (self.cam.WidthMax.get(), self.cam.HeightMax.get())
            x_offset, y_offset = 0, 0
        else:
            width, height = size
            width_quatic = self.cam.Width.get_range()['inc']
            width_quatic = width_quatic*2 if width_quatic%2==1 else width_quatic
            height_quatic = self.cam.Height.get_range()['inc']
            height_quatic = height_quatic*2 if height_quatic%2==1 else height_quatic
            width, height = int(width//width_quatic*width_quatic), int(height//height_quatic*height_quatic)
            # 计算窗口的偏移量，确保中心位置在指定位置
            x_offset, y_offset = center[0]-(width//2), center[1]-(height//2)
            x_offset, y_offset = int(x_offset//width_quatic*width_quatic), int(y_offset//height_quatic*height_quatic)
        assert x_offset>=0 and y_offset>=0, f"窗口中心位置:{center}必须在图像内部，窗口大小:{size}"
        self.cam.Width.set(width)
        self.cam.Height.set(height)
        self.cam.OffsetX.set(x_offset)
        # 设置相机的垂直偏移量，确保偏移量是4的倍数
        self.cam.OffsetY.set(y_offset)

        self.__update_properties()
        self.cam.stream_on()

        # 返回新的窗口中心位置
        return (width, height), (width//2, height//2)

    def get_numpy_image(self, n_sample=1) -> np.ndarray:
        assert n_sample>0
        
        numpy_image = np.zeros((self.cam_height, self.cam_width))
        for _ in range(n_sample):
            while True:
                raw_image = self.cam.data_stream[0].get_image()
                if not raw_image:
                    continue
                
                numpy_image += raw_image.get_numpy_array()
                break
        avg_img = numpy_image/n_sample
        return avg_img.astype(np.uint8)

    def __update_properties(self):
        self.cam_width = self.cam.Width.get()
        self.cam_height = self.cam.Height.get()
        log.info(f"Open cam {self.__sn} success. width={self.cam_width}, height={self.cam_height}")
        self.xv, self.yv = self.__get_grid(self.cam_width, self.cam_height)

    @staticmethod
    def __get_grid(width, height):
        x = np.arange(0, width)
        y = np.arange(0, height)
        xv, yv = np.meshgrid(x, y)
        return xv, yv


if __name__ == '__main__':
    import numpy as np

    import matplotlib.pyplot as plt
   
    def test_cam(cam_id=0):
        with CameraStreamManager(cam_id, exposure_time_ms=80) as cam:
            img = cam.get_numpy_image()
            center = np.unravel_index(np.argmax(img), img.shape)
            center = (center[1], center[0])
            print(f'{center=}')
            plt.imshow(img)
            plt.title(f'{center=} = {img[center[::-1]]=}')
            plt.show()
            
    test_cam(0)


    