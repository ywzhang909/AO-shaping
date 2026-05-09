from abc import ABC
from collections import namedtuple
from loguru import logger

import pygame
import importlib

from . import  Image2DFrame, VoltageFrame
from .frames import BaseFrame

FrameInfo = namedtuple('FrameInfo', ['name', 'title', 'frame', "kwargs"], defaults=[None, None, None, {}])


class BaseDisplay(ABC):
    def __init__(self, total_size) -> None:
        self.total_size = total_size

    def render(self, info:str='') -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        if info:
            pygame.display.set_caption(info)
        pygame.event.pump()
        pygame.display.update()
        return True

    def init_window(self) -> None:
        pygame.init()
        self.window = pygame.display.set_mode(self.total_size)

    def close(self) -> None:
        for frame in self._frames.values():
            frame.close()
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
    def __init__(self, frame_list:list[FrameInfo], frame_size=(300, 300), display_size=(1280, 720), margin=10) -> None:
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
        for i, frame_info in enumerate(self.frame_list):
            name, title, frame_class_name = frame_info.name, frame_info.title, frame_info.frame
            _row, _col = divmod(i, n_cols)
            _frame = self.__get_frame_by_name(frame_class_name)
            top = _row*(frame_h + self.margin)
            left = _col*(frame_w + self.margin)
            assert name not in self._frames, f"Frame name {name} is duplicated"
            self._frames[name] = _frame(window=self.window, render_pos=(top, left), frame_size=self.frame_size, title=title, **frame_info.kwargs)

    def render(self, frame_data:dict[str, dict], info:str='') -> bool:
        for name, frame in self._frames.items():
            frame.render(**frame_data.get(name))
            # frame.top, frame.left = frame.render_pos
        return super().render(info)

    @staticmethod
    def __get_frame_by_name(name:str) -> BaseFrame:
        module = importlib.import_module('ao_shaping.display.frames')
        return getattr(module, name)
