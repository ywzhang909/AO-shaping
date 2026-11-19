# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt

cnp.import_array()

ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.uint8_t BOOL_t

cdef class ImageTargetFunc:
    cdef public object xv  # Use object instead of buffer type
    cdef public object yv  # Use object instead of buffer type
    cdef public tuple center
    cdef public object dist_mat  # Use object instead of buffer type
    cdef public object masks  # Use object instead of buffer type
    cdef public int shape_0, shape_1
    cdef public int npix
    cdef public DTYPE_t dpix
    
    @classmethod
    def build_from_init_image(cls, cnp.ndarray[DTYPE_t, ndim=2] init_img):
        cdef int h = init_img.shape[0]
        cdef int w = init_img.shape[1]
        cdef cnp.ndarray[DTYPE_t, ndim=2] xv, yv
        xv, yv = np.ogrid[:h, :w]
        _ret = cls(xv, yv, (h//2, w//2))
        center = _ret.intelligen_center(init_img)
        _ret.center = center
        return _ret
    
    def __init__(self, xv, yv, center):
        self.init_coordinates(xv, yv, center)
        
    def init_coordinates(self, xv, yv, center):
        if isinstance(xv, int):
            self.xv, self.yv = np.ogrid[:xv, :yv]
        elif isinstance(xv, np.ndarray):
            if xv.ndim == 2:
                self.xv, self.yv = xv, yv
            elif xv.ndim == 1:
                self.xv, self.yv = np.meshgrid(xv, yv)

        self.shape_0 = self.xv.shape[0]
        self.shape_1 = self.xv.shape[1]
        self.center = center
        self.dist_mat = np.sqrt((self.xv - self.center[0])**2 + (self.yv - self.center[1])**2)
        self.masks = self.__gen_center_bucket_masks()

        # Use memoryviews for indexing
        cdef cnp.float64_t[:, :] xv_view = self.xv
        self.npix = self.xv.size
        self.dpix = xv_view[0, 1] - xv_view[0, 0]
        
    cdef cnp.ndarray[BOOL_t, ndim=2] __gen_center_bucket_masks(self):
        cdef int max_radius = min(
            self.center[0],
            self.center[1],
            self.shape_0-self.center[0],
            self.shape_1-self.center[1]
        )
        cdef cnp.ndarray[BOOL_t, ndim=3] mask_mats = np.zeros((max_radius, self.shape_0, self.shape_1), dtype=np.bool_)
        cdef cnp.float64_t[:, :] dist_mat_view = self.dist_mat  # Use memoryview
        cdef cnp.uint8_t[:, :, :] mask_mats_view = mask_mats  # Use memoryview
        cdef int r, i, j
        cdef DTYPE_t dist_val
        for r in range(1, max_radius):
            for i in range(self.shape_0):
                for j in range(self.shape_1):
                    dist_val = dist_mat_view[i, j]
                    if dist_val <= r:
                        mask_mats_view[r, i, j] = True
        return mask_mats

    def pib(self, cnp.ndarray[DTYPE_t, ndim=2] img, int pib_radius, bint normalize=True):
        cdef cnp.ndarray[BOOL_t, ndim=2] pib_mask = self.__get_bucket_mask(pib_radius)
        cdef DTYPE_t pib_val = 0.0
        cdef DTYPE_t img_sum = 0.0
        cdef int i, j
        
        # 计算掩码区域内的像素和
        for i in range(self.shape_0):
            for j in range(self.shape_1):
                if pib_mask[i, j]:
                    pib_val += img[i, j]
                    
        if normalize:
            # 计算整张图像的像素和
            for i in range(self.shape_0):
                for j in range(self.shape_1):
                    img_sum += img[i, j]
            return pib_val / img_sum if img_sum > 0 else 0.0
            
        return pib_val
    
    def denoise_process(self, cnp.ndarray[DTYPE_t, ndim=2] img):
        cdef DTYPE_t noise_sample = np.percentile(img, 5)
        cdef cnp.ndarray[DTYPE_t, ndim=2] denoised_img = img - noise_sample
        cdef int i, j
        # 将负值设为0
        for i in range(denoised_img.shape[0]):
            for j in range(denoised_img.shape[1]):
                if denoised_img[i, j] < 0:
                    denoised_img[i, j] = 0
        return denoised_img
    
    def intelligen_center(self, cnp.ndarray[DTYPE_t, ndim=2] img):
        # 如果环围半径较小，使用质心而非形心;如果中间存在空洞使用形心，否则质心
        # 计算环围半径（包含90%能量的半径）
        radius = self.radius(img, 0.9)
        # 如果半径较小（小于10个像素），使用质心
        if radius < 10:
            return self.center_of_brightness(img)
        # 检查中间是否存在空洞（中心区域强度低于周边平均值的30%）
        center_intensity = img[img.shape[0]//2, img.shape[1]//2]
        # 计算周边平均强度（排除中心区域）
        edge_sum = 0
        edge_count = 0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                # 距离中心超过半径的一半但不到半径的位置作为边缘样本
                dist_from_center = sqrt((i - img.shape[0]//2)**2 + (j - img.shape[1]//2)**2)
                if radius/2 < dist_from_center <= radius:
                    edge_sum += img[i, j]
                    edge_count += 1
        edge_avg = edge_sum / edge_count if edge_count > 0 else 0
        # 如果中心强度显著低于边缘平均强度，认为存在空洞
        if edge_avg > 0 and center_intensity < edge_avg * 0.3:
            return self.center_of_mass(img)
        # 否则使用质心
        return self.center_of_brightness(img)
    
    def center_of_brightness(self, cnp.ndarray[DTYPE_t, ndim=2] img):
        # 这里简化实现，实际应该调用numba优化的函数
        cdef DTYPE_t total_intensity = 0.0
        cdef DTYPE_t weighted_x = 0.0
        cdef DTYPE_t weighted_y = 0.0
        cdef int i, j
        cdef DTYPE_t intensity
        
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                intensity = img[i, j]
                total_intensity += intensity
                weighted_x += intensity * i
                weighted_y += intensity * j
                
        if total_intensity > 0:
            return (weighted_x / total_intensity, weighted_y / total_intensity)
        else:
            return (img.shape[0] / 2.0, img.shape[1] / 2.0)
    
    def center_of_mass(self, cnp.ndarray[DTYPE_t, ndim=2] img):
        # 这里简化实现，实际应该调用numba优化的函数
        cdef DTYPE_t total_intensity = 0.0
        cdef DTYPE_t weighted_x = 0.0
        cdef DTYPE_t weighted_y = 0.0
        cdef int i, j
        cdef DTYPE_t intensity
        cdef cnp.float64_t[:, :] xv_view = self.xv  # Use memoryview
        cdef cnp.float64_t[:, :] yv_view = self.yv  # Use memoryview
        cdef cnp.float64_t[:, :] img_view = img  # Use memoryview

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                intensity = img_view[i, j]
                total_intensity += intensity
                weighted_x += intensity * xv_view[i, j]
                weighted_y += intensity * yv_view[i, j]

        if total_intensity > 0:
            return (weighted_x / total_intensity, weighted_y / total_intensity)
        else:
            return (xv_view[img.shape[0] // 2, img.shape[1] // 2],
                    yv_view[img.shape[0] // 2, img.shape[1] // 2])
    
    def radius(self, cnp.ndarray[DTYPE_t, ndim=2] intensity, DTYPE_t energy=0.99):
        """
        以center为圆心，占总能量百分比为energy的圆的半径

        :param intensity: 强度分布
        :param energy: 圆内的能量比，默认0.99，取值范围0~1，常用0.5，0.865， 0.99
        :return radius: 圆的半径
        """
        cdef DTYPE_t power_in_circle = 0.0
        cdef DTYPE_t total_power = 0.0
        cdef int i, j
        cdef cnp.float64_t[:, :] intensity_view = intensity  # Use memoryview

        # 计算总功率
        for i in range(intensity.shape[0]):
            for j in range(intensity.shape[1]):
                total_power += intensity_view[i, j]

        power_in_circle = total_power * energy

        cdef int max_radius = self.masks.shape[0]
        cdef DTYPE_t current_power
        cdef int r
        cdef cnp.uint8_t[:, :, :] masks_view = self.masks  # Use memoryview

        # 计算每个半径的能量
        for r in range(1, max_radius):
            current_power = 0.0
            for i in range(self.shape_0):
                for j in range(self.shape_1):
                    if masks_view[r, i, j]:
                        current_power += intensity_view[i, j]

            if current_power >= power_in_circle:
                return r

        return max_radius

    cdef cnp.ndarray[BOOL_t, ndim=2] __get_bucket_mask(self, int radius):
        assert 0 < radius < self.masks.shape[0], "Radius out of range"
        return self.masks[radius-1]
