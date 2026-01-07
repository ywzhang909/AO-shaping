import numpy as np

from ao_shaping.utils.spots_calc import (
    center_of_mass_numpy,
    center_of_brightness,
)

class ImageTargetFunc:
    
    @classmethod
    def build_from_init_image(cls, init_img:np.ndarray):
        h, w = init_img.shape
        xv, yv = np.ogrid[:h, :w]
        _ret = cls(xv, yv, (h//2, w//2))
        center = _ret.intelligen_center(init_img)
        _ret.center = center
        
        return _ret
    
    def __init__(self, x, y, center):
        assert type(x) is type(y), "x and y must have the same type"
        self.init_coordinates(x, y, center)
        
        
    def init_coordinates(self, x, y, center):
        if isinstance(x, int):
            self.xv, self.yv = np.ogrid[:x, :y]
        elif isinstance(x, np.ndarray):
            if x.ndim == 2:
                self.xv, self.yv = x, y
            elif x.ndim == 1:
                self.xv, self.yv = np.meshgrid(x, y)
        self.shape = self.xv.shape
        self.center = center
        self.dist_mat = np.sqrt((self.xv - self.center[0])**2 + (self.yv - self.center[1])**2)
        self.masks = self.__gen_center_bucket_masks()
        
        self.npix = len(self.xv)
        if self.xv.ndim == 2 and self.xv.shape[1] > 1:
            self.dpix = self.xv[0, 1] - self.xv[0, 0]
        elif self.xv.ndim == 1:
            self.dpix = self.xv[1] - self.xv[0] if len(self.xv) > 1 else 1.0
        else:
            self.dpix = 1.0
        
    def __gen_center_bucket_masks(self):
        max_radius = min(
            self.center[0], 
            self.center[1],
            self.shape[0]-self.center[0]-1,
            self.shape[1]-self.center[1]-1
        )
        max_radius = max(1, max_radius)
        mask_mats = np.zeros((max_radius, *self.shape), dtype=bool)
        for r in range(1, max_radius):
            mask_mats[r] = self.dist_mat <= r
        return mask_mats

    def pib(self, img, pib_radius, normalize=True):
        pib_mask = self.__get_bucket_mask(pib_radius)
        pib = np.sum(img[pib_mask])
        return pib/np.sum(img) if normalize else pib
    
    def denoise_process(self, img):
        noise_sample = np.percentile(img, 5)
        denoised_img = img - noise_sample
        denoised_img[denoised_img < 0] = 0
        return denoised_img
    
    def intelligen_center(self, img, margin=5):
        # 如果环围半径较小，使用质心而非形心;如果中间存在空洞使用形心，否则质心 
        center = self.center_of_brightness(img)
        (cx, cy) = center
        if np.all(img[cy-margin: cy+margin, cx-margin: cx+margin] >= np.max(img) * 0.4): # 中心不是空洞
            center = self.center_of_mass(img)
            
        return center
    
    def center_of_brightness(self, img):
        center = center_of_brightness(img)
        return center
    
    def center_of_mass(self, img, moment=1):
        return center_of_mass_numpy(img, self.xv, self.yv, moment)
    
    def radius(self, intensity, energy=0.99):
        """
        以center为圆心，占总能量百分比为energy的圆的半径

        :param intensity: 强度分布
        :param energy: 圆内的能量比，默认0.99，取值范围0~1，常用0.5，0.865， 0.99
        :return radius: 圆的半径
        """
        power_in_circle = np.sum(intensity) * energy
        # intensity 复制扩展成3D 与 masks 维度一致
        intensity_3d = np.repeat(intensity[np.newaxis, ...], len(self.masks), axis=0)
        power_in_masks = np.sum(intensity_3d * self.masks, axis=(1, 2))
        radius = np.argmax(power_in_masks >= power_in_circle)+1
        return radius
    
    def __get_bucket_mask(self, radius):
        assert 0 < radius < len(self.masks), "Radius out of range"
        return self.masks[radius-1]