from abc import ABC
from typing import Iterable
from loguru import logger

import pygame

from . import  Image2DFrame, VoltageFrame
from .frames import Frame_Reg

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
    def __init__(self, frame_list:Iterable[str], frame_size=(300, 300), display_size=(1280, 720), margin=10) -> None:
        self.total_size = display_size
        self.frame_size = frame_size
        self.frame_list = frame_list
        self.margin = margin
        
    def init_window(self) -> None:
        super().init_window()
        screen_w, screen_h = self.total_size
        frame_w, frame_h = self.frame_size
        n_cols = screen_w // frame_w
        n_rows = len(self.frame_list) // n_cols + 1
        if n_rows * frame_h > screen_h:
            n_rows = screen_h // frame_h
            n_cols = len(self.frame_list) // n_rows + 1
        
        logger.info(f"AutoDisplay: {n_cols} x {n_rows} = {n_cols*n_rows} frames")
        total_size = (n_cols * frame_w,
                      n_rows * frame_h)
        super().__init__(total_size)
        
        self._frames = {}
        for i, frame_class_name in enumerate(self.frame_list):
            _row, _col = divmod(i, n_cols)
            _frame = Frame_Reg.get(frame_class_name)
            top = _row*(frame_h + self.margin)
            left = _col*(frame_w + self.margin)
            self._frames[frame_class_name] = _frame(window=self.window, render_pos=(top, left), frame_size=self.frame_size)

    def render(self, frame_data:dict[str, dict], info:str='') -> None:
        for name, frame in self._frames.items():
            frame.render(**frame_data.get(name))
        return super().render(info)
