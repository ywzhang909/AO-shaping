import numpy as np
from scipy.optimize import curve_fit
import warnings

from ao_shaping.utils.spots_calc import (
    center_of_mass_numpy,
    center_of_brightness,
)


def _gaussian2d(
    xdata: np.ndarray,
    amplitude: float,
    x0: float,
    y0: float,
    sigma: float,
    offset: float,
) -> np.ndarray:
    """2D Gaussian function for curve fitting.
    
    Args:
        xdata: Coordinate array of shape (2, N)
        amplitude: Peak amplitude
        x0, y0: Center coordinates
        sigma: Standard deviation (waist radius)
        offset: Background offset
        
    Returns:
        np.ndarray: Gaussian values at each point
    """
    x = xdata[0]
    y = xdata[1]
    
    # Ensure sigma is positive
    sigma = max(sigma, 1e-6)
    
    return amplitude * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2)) + offset


class ImageTargetFunc:
    @classmethod
    def build_from_init_image(
        cls: "ImageTargetFunc", init_img: np.ndarray
    ) -> "ImageTargetFunc":
        h, w = init_img.shape
        # Initial center in (x, y) = (col, row) format: (w//2, h//2)
        _ret = cls(w, h, (w // 2, h // 2))
        center = _ret.intelligen_center(init_img)
        # Round center values to integers for consistent mask calculation
        # intelligen_center may return floats from center_of_mass calculation
        center_int = (round(center[0]), round(center[1]))
        # Recalculate dist_mat and masks with the correct center
        # since intelligen_center may update center based on actual image content
        _ret.init_coordinates(_ret.xv, _ret.yv, center_int)

        return _ret

    def __init__(self, x, y, center):
        assert type(x) is type(y), "x and y must have the same type"
        self.init_coordinates(x, y, center)

    def init_coordinates(self, x, y, center):
        if isinstance(x, int) and isinstance(y, int):
            _x, _y = np.arange(x), np.arange(y)
            self.xv, self.yv = np.meshgrid(_x, _y, indexing="xy")
            self.shape = y, x
        elif isinstance(x, np.ndarray):
            if x.ndim == 2:
                self.xv, self.yv = x, y
            elif x.ndim == 1:
                self.xv, self.yv = np.meshgrid(x, y, indexing="xy")
            self.shape = self.xv.shape

        self.center = center
        self.dist_mat = np.sqrt(
            (self.xv - self.center[0]) ** 2 + (self.yv - self.center[1]) ** 2
        )
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
            self.shape[0] - self.center[0] - 1,
            self.shape[1] - self.center[1] - 1,
        )
        max_radius = max(1, int(max_radius))
        mask_mats = np.zeros((max_radius, *self.shape), dtype=bool)
        for r in range(max_radius):
            mask_mats[r] = self.dist_mat <= r
        return mask_mats

    def pib(self, img, pib_radius):
        """计算桶中功率

        Args:
            img (np.ndarray[ndim=2, shape=(h,w)]): CCD 图片
            pib_radius (int): 桶半径

        Returns:
            tuple[float, float]: 桶功率/面积, 桶中/全部
        """
        pib_mask = self.__get_bucket_mask(pib_radius)
        pib = np.sum(img[pib_mask])
        return pib, pib / np.sum(img)

    def avg_radius(self, img, moment=1.0):
        r = np.sum(self.dist_mat**moment * img)
        return (
            r,
            r / np.sum(img),
        )

    def denoise_process(self, img):
        noise_sample = np.percentile(img, 5)
        denoised_img = img - noise_sample
        denoised_img[denoised_img < 0] = 0
        return denoised_img

    def intelligen_center(self, img, margin=5):
        # 如果环围半径较小，使用质心而非形心;如果中间存在空洞使用形心，否则质心
        center = self.center_of_brightness(img)
        (cx, cy) = center
        if np.all(
            img[cy - margin : cy + margin, cx - margin : cx + margin]
            >= np.max(img) * 0.4
        ):  # 中心不是空洞
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
        # FIX
        power_in_circle = np.sum(intensity) * energy
        # intensity 复制扩展成3D 与 masks 维度一致
        intensity_3d = np.repeat(intensity[np.newaxis, ...], len(self.masks), axis=0)
        power_in_masks = np.sum(intensity_3d * self.masks, axis=(1, 2))
        radius = int(np.argmax(power_in_masks >= power_in_circle) + 1)
        return radius

    def __get_bucket_mask(self, radius):
        assert 0 < radius < len(self.masks), f"Radius {radius} out of range"
        return self.masks[int(radius)]

    def fit_gaussian_radius(self, img: np.ndarray, center: tuple[float, float] | None = None) -> float | None:
        """拟合2D高斯曲线得到半腰半径（sigma）。
        
        Args:
            img (np.ndarray[ndim=2, shape=(h,w)]): CCD图片
            center (tuple[float, float] | None): 光斑中心坐标，默认为self.center
            
        Returns:
            float | None: 高斯半腰半径（sigma），拟合失败返回None
        """
        if center is None:
            center = self.center
        
        h, w = img.shape
        cy, cx = int(center[1]), int(center[0])
        
        # 提取感兴趣区域（ROI），中心48x48像素
        roi_size = 48
        y_start = max(0, cy - roi_size // 2)
        y_end = min(h, cy + roi_size // 2)
        x_start = max(0, cx - roi_size // 2)
        x_end = min(w, cx + roi_size // 2)
        
        if y_end - y_start < 5 or x_end - x_start < 5:
            return None
            
        roi = img[y_start:y_end, x_start:x_end]
        
        # 初始参数估计
        amplitude = float(np.max(roi) - np.min(roi))
        offset = float(np.min(roi))
        sigma_estimate = roi_size / 6
        
        # 构建2D网格并转换为 (2, N) 形状
        x_mesh, y_mesh = np.meshgrid(
            np.arange(roi.shape[1]) + x_start,
            np.arange(roi.shape[0]) + y_start,
            indexing='xy'
        )
        coords = np.vstack([x_mesh.ravel(), y_mesh.ravel()])
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(
                    _gaussian2d,
                    coords,
                    roi.ravel(),
                    p0=[amplitude, center[0], center[1], sigma_estimate, offset],
                    bounds=(
                        [0, x_start, y_start, 0.5, -np.inf],
                        [np.inf, x_end, y_end, roi_size, np.inf]
                    ),
                    maxfev=1000,
                )
            # popt = [amplitude, x0, y0, sigma, offset]
            return float(popt[3])
        except Exception:
            return None

    def second_moment_radius(self, img: np.ndarray, center: tuple[float, float] | None = None) -> float:
        """计算二阶矩半径。
        
        二阶矩半径定义：
        r^2 = Σ(r^2 * I) / Σ(I)
        其中 r 是像素到中心的距离，I 是强度值。
        
        Args:
            img (np.ndarray[ndim=2, shape=(h,w)]): CCD图片
            center (tuple[float, float] | None): 光斑中心坐标，默认为self.center
            
        Returns:
            float: 二阶矩半径
        """
        if center is None:
            center = self.center
        
        cx, cy = center
        
        # 计算距离矩阵
        y_coords, x_coords = np.ogrid[:img.shape[0], :img.shape[1]]
        dist_sq = (x_coords - cx)**2 + (y_coords - cy)**2
        
        # 计算二阶矩半径
        total_intensity = np.sum(img)
        if total_intensity <= 0:
            return 0.0
            
        second_moment_sq = np.sum(dist_sq * img) / total_intensity
        return float(np.sqrt(second_moment_sq))
