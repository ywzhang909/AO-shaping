from abc import ABC

import pygame

from .frames import  Image2DFrame, VoltageFrame
from . import __FRAME

class BaseDisplay(ABC):
    def __init__(self, total_size) -> None:
        self.total_size = total_size
        
    def render(self, info:str='') -> None:
        if info:
            pygame.display.set_caption(info)
        pygame.event.pump()
        pygame.display.update()
    
    def init_window(self) -> None:
        pygame.init()
        self.window = pygame.display.set_mode(self.total_size)
        
    def close(self) -> None:
        pygame.quit()
        
    def __enter__(self):
        self.init_window()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ImageVoltagesDisplay(BaseDisplay):
    def __init__(self, img_size, volt_height=200, v_min=-300, v_max=500):
        img_w, img_h = img_size
        super().__init__((img_w, img_h + volt_height*2))
        self.img_frame = Image2DFrame(self.window, (0,0), img_size)
        self.volt_frame = VoltageFrame(self.window, (0, img_h), (img_w, volt_height*2), v_min, v_max)

    def render(self, img, volts, center, r, info="") -> None:
        self.img_frame.render(img, center, r)
        self.volt_frame.render(volts)
        return super().render(info)


class AutoDisplay(BaseDisplay):
    def __init__(self, frame_list:list[str], frame_size=(300, 300)) -> None:
        # 获取屏幕分辨率
        screen_info = pygame.display.Info()
        screen_w, screen_h = screen_info.current_w, screen_info.current_h
        
        n_cols = screen_w // frame_size[0]
        n_rows = len(frame_list) // n_cols + 1
        if n_rows * frame_size[1] > screen_h:
            n_rows = screen_h // frame_size[1]
            n_cols = len(frame_list) // n_rows + 1
        
        total_size = (n_cols * frame_size[0],
                      n_rows * frame_size[1])
        super().__init__(total_size)
        
        self._frames = {}
        for i, frame_class_name in enumerate(frame_list):
            _row, _col = divmod(i, n_cols)
            _frame = __FRAME.get(frame_class_name)
            self._frames[frame_class_name] = _frame(self.window, (_col*frame_size[0], _row*frame_size[1]), frame_size)
            
    def render(self, frame_data:dict, info:str='') -> None:
        for name, frame in self._frames.items():
            frame.render(frame_data.get(name))
        return super().render(info)
