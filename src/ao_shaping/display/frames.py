from abc import ABC, abstractmethod

import pygame
import numpy as np
from scipy.ndimage import zoom

BACKGROUND_COLOR = (0, 0, 0)


class BaseFrame(ABC):
    def __init__(self, window, render_pos, frame_size) -> None:
        self.window = window
        self.top, self.left = render_pos
        self.width, self.height = frame_size
        
        self.plot_area = pygame.Rect(self.left, self.top, self.width, self.height)
        
    @abstractmethod
    def render(self):
        pass


class Image2DFrame(BaseFrame):
    def render(self, img:np.ndarray):
        assert img.ndim == 2, "ImageFrame only supports 2D images"
        img_h, img_w = img.shape
        zoom_factors = (self.height / img_h, self.width / img_w)
        img = zoom(img, zoom_factors, mode='nearest')
        img[np.isnan(img)] = 0
        img_surf = pygame.surfarray.make_surface(img.transpose())
        self.window.blit(img_surf, (self.left, self.top))
        
        
class Image2DWithBucketFrame(BaseFrame):
    def render(self, img:np.ndarray, center:tuple[int,int], r:int):
        assert img.ndim == 2, "ImageFrame only supports 2D images"
        img_h, img_w = img.shape
        zoom_factors = (self.height / img_h, self.width / img_w)
        img = zoom(img, zoom_factors, mode='nearest')
        img_surf = pygame.surfarray.make_surface(img.transpose())
        pygame.draw.circle(img_surf, (255, 0, 0), center, r, 3)
        self.window.blit(img_surf, (self.left, self.top))


class VoltageFrame(BaseFrame):
    def __init__(self, v_min:int=-300, v_max:int=500, background_color=BACKGROUND_COLOR, **kwargs) -> None:
        super().__init__(**kwargs)
        self.background_color = background_color
        self.v_min = v_min
        self.v_max = v_max
        
        max_hight_ratio, total_scale = self.height / max(v_max,v_min), (self.v_max - self.v_min)
        self.v_norm = lambda v: (v - self.v_min) / total_scale
        self.v_hight = lambda v: int(v * max_hight_ratio)
    
    def render(self, volts):
        self.window.fill(self.background_color, self.plot_area)
        bar_width = int(self.width / len(volts))
        for i,v in enumerate(volts):
            normed_v = self.v_norm(v)
            color = (int(normed_v*255), int((1-normed_v)*255), 0)
            x = int(self.left + i * bar_width)
            y = int(self.top + self.height/2)
            height = self.v_hight(v)
            pygame.draw.line(self.window, color, (x, y), (x, y - height), bar_width)
            

Frame_Reg = {
    "Image2D": Image2DFrame,
    "Image2DWithBucket": Image2DWithBucketFrame,
    "Voltage": VoltageFrame,
}

Frame = Frame_Reg.keys()