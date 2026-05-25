from abc import ABC

import numpy as np
from scipy.ndimage import zoom

from ao_shaping.utils import logger

BACKGROUND_COLOR = (0, 0, 0)
Title_height = 36
DPI = 96

__frame_registry = {}

def get_frame_names():
    return __frame_registry.keys()

def get_frame(frame_name:str):
    return __frame_registry[frame_name]

def register_frame(frame_name:str):
    def decorator(cls):
        assert frame_name not in __frame_registry, f"Frame name {frame_name} already registered"
        __frame_registry[frame_name] = cls
        return cls
    return decorator

class BaseFrame(ABC):
    def __init__(self, window, render_pos, frame_size, title:str="") -> None:
        import pygame
        self.window = window
        self.top, self.left = render_pos
        self.width, self.height = frame_size
        self.title = title

        if title:
            self.top += Title_height
            self.height -= Title_height

        self.plot_area = pygame.Rect(self.left, self.top, self.width, self.height)

    def close(self):
        pass

    def _render_title(self):
        import pygame
        font = pygame.font.Font(None, Title_height-6)
        text = font.render(self.title, True, (0, 255, 255))
        self.window.blit(text, (self.left, self.top - Title_height))

    def render(self):
        self._render_title()

@register_frame("Image2D")
class Image2DFrame(BaseFrame):
    def render(self, img:np.ndarray):
        import pygame
        assert img.ndim == 2, "ImageFrame only supports 2D images"
        img_h, img_w = img.shape
        zoom_factors = (self.height / img_h, self.width / img_w)
        img = zoom(img, zoom_factors, mode='nearest')
        img[np.isnan(img)] = 0
        img_surf = pygame.surfarray.make_surface(img.transpose())
        self.window.blit(img_surf, (self.left, self.top))
        super().render()


@register_frame("Image2DWithBucket")
class Image2DWithBucketFrame(BaseFrame):
    def render(self, img:np.ndarray, center:tuple[int,int], r:int):
        import pygame
        assert img.ndim == 2, "ImageFrame only supports 2D images"
        img_h, img_w = img.shape
        zoom_factors = (self.height / img_h, self.width / img_w)
        img = zoom(img, zoom_factors, mode='nearest')
        img_surf = pygame.surfarray.make_surface(img.transpose())
        pygame.draw.circle(img_surf, (255, 0, 0), center, r, 3)
        self.window.blit(img_surf, (self.left, self.top))
        super().render()


@register_frame("Voltage")
class VoltageFrame(BaseFrame):
    def __init__(self, v_min:int=-300, v_max:int=500, background_color=BACKGROUND_COLOR, **kwargs) -> None:
        super().__init__(**kwargs)
        self.background_color = background_color
        self.v_min = v_min
        self.v_max = v_max

        max_hight_ratio, total_scale = self.height / max(abs(v_max),abs(v_min)), (self.v_max - self.v_min)
        self.v_norm = lambda v: (v - self.v_min) / total_scale
        self.v_hight = lambda v: int(v * max_hight_ratio)

    def render(self, volts):
        import pygame
        _volts = np.clip(volts, self.v_min, self.v_max)
        self.window.fill(self.background_color, self.plot_area)
        bar_width = int(self.width / len(_volts))
        for i,v in enumerate(_volts):
            normed_v = self.v_norm(v)
            color = (int(normed_v*255), int((1-normed_v)*255), 0)
            x = int(self.left + i * bar_width)
            y = int(self.top + self.height/2)
            height = self.v_hight(v)
            pygame.draw.line(self.window, color, (x, y), (x, y - height), bar_width)
        super().render()


@register_frame("Log")
class LogFrame(BaseFrame):
    Line_Coler = (0, 255, 0)

    def __init__(self, background_color=BACKGROUND_COLOR, **kwargs) -> None:
        super().__init__(**kwargs)
        self.background_color = background_color
        self.__recorder = []

    def render(self, value):
        import pygame
        self.window.fill(self.background_color, self.plot_area)
        self.__recorder.append(value)
        if len(self.__recorder) > 1:
            min_sum = min(self.__recorder)
            max_sum = max(self.__recorder)
            points = []
            num_points = len(self.__recorder)
            for i, sum_value in enumerate(self.__recorder):
                # 均匀分布 x 轴坐标
                x = self.left + int(i * (self.width / (num_points - 1)))
                y = self.top + self.height - int(
                    (sum_value - min_sum) / (max_sum - min_sum) * self.height
                ) if max_sum != min_sum else self.height // 2
                points.append((x, y))
            pygame.draw.lines(self.window, self.Line_Coler, False, points, 2)
        super().render()

    def close(self):
        self.__recorder.clear()

    @property
    def data(self):
        return self.__recorder


@register_frame("Text")
class TextFrame(BaseFrame):
    def __init__(self, font_size:int=0, background_color=BACKGROUND_COLOR, **kwargs) -> None:
        super().__init__(**kwargs)
        self.font_size = font_size
        self.background_color = background_color
        if self.font_size == 0:
            logger.info("Font size not specified, using dynamic font size")

    def render(self, text:str, font_size:int=0):
        import pygame
        self.window.fill(self.background_color, self.plot_area)

        lines = text.splitlines()
        if font_size != 0:
            self.font_size = font_size

        if self.font_size == 0:
            max_line_len = max(len(line) for line in lines)
            font_pixel = min(self.height // len(lines)-6, self.width // max_line_len-6)
            self.font_size = self.__font_pixel_to_pt(font_pixel)

        font = pygame.font.Font(None, self.font_size)
        for i, line in enumerate(lines):
            text = font.render(line, True, (0, 255, 255))
            self.window.blit(text, (self.left, self.top + i * (font.get_height()+3)))
        super().render()

    @staticmethod
    def __font_pt_to_pixel(pt:int):
        return int(pt * DPI / 72)

    @staticmethod
    def __font_pixel_to_pt(px:int):
        return int(px * 72 / DPI)
