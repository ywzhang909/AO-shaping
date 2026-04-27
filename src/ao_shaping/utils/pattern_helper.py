# Pattern Helper - 光学相位图案生成工具
from __future__ import annotations

import numpy as np

from ao_shaping.utils.zernike_calc import ZernikeGenerator, nm_to_noll

from aotools.turbulence import PhaseScreenKolmogorov
from aotools import ft_phase_screen


class PatternHelper:
    """光学相位图案生成工具类。

    提供多种光学相位图案的生成方法，用于SLM（空间光调制器）等显示设备。

    用法示例:
    ```python
    from ao_shaping.utils.pattern_helper import PatternHelper

    ph = PatternHelper(resolution=(256, 256), bits=10)

    # 生成棋盘格图案
    checker = ph.generate_checkerboard(period=32)

    # 生成Zernike模式
    zernike = ph.generate_zernike(n=4, m=0, amplitude=1.0)

    # 生成聚焦透镜
    focus = ph.generate_focus(focal_length=0.5, wavelength=532e-9, pixel_size=8e-6)
    ```

    属性:
        x, y: 1D坐标 (中心为0)
        xx, yy: 2D网格坐标
        R: 径向距离
        Theta: 角向坐标
        mask: 圆形光阑掩模
        pixel_x, pixel_y: 像素坐标

    图案类型:
        - checkerboard: 棋盘格
        - binary_grating: 二值光栅
        - microlens_array: 微透镜阵列
        - turbulence: 湍流相位屏
        - zernike: Zernike模式
        - focus: 聚焦透镜
        - dammann: Dammann光栅
        - linear_grating: 线性光栅
        - circular_grating: 圆形光栅
        - lens: 透镜模式
        - hologram: 全息图
    """

    def __init__(self, resolution: tuple[int, int], bits: int = 10) -> None:
        self.resolution = resolution
        self.bits = bits
        height, width = resolution[1], resolution[0]
        self._max_val = 2**bits - 1
        self._height = height
        self._width = width

        # Cached coordinate arrays (lazy-computed)
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._xx: np.ndarray | None = None
        self._yy: np.ndarray | None = None
        self._R: np.ndarray | None = None
        self._Theta: np.ndarray | None = None
        self._mask: np.ndarray | None = None
        self._pixel_x: np.ndarray | None = None
        self._pixel_y: np.ndarray | None = None

    @property
    def x(self) -> np.ndarray:
        """1D x coordinates (centered at 0)."""
        if self._x is None:
            self._x = np.arange(self._width, dtype=np.float64) - self._width // 2
        return self._x

    @property
    def y(self) -> np.ndarray:
        """1D y coordinates (centered at 0)."""
        if self._y is None:
            self._y = np.arange(self._height, dtype=np.float64) - self._height // 2
        return self._y

    @property
    def xx(self) -> np.ndarray:
        """2D meshgrid x coordinates."""
        if self._xx is None:
            self._xx, self._yy = np.meshgrid(self.x, self.y)
        return self._xx

    @property
    def yy(self) -> np.ndarray:
        """2D meshgrid y coordinates."""
        if self._yy is None:
            self._xx, self._yy = np.meshgrid(self.x, self.y)
        return self._yy

    @property
    def R(self) -> np.ndarray:
        """Radial distance from center."""
        if self._R is None:
            self._R = np.sqrt(self.xx**2 + self.yy**2)
        return self._R

    @property
    def Theta(self) -> np.ndarray:
        """Azimuthal angle from center."""
        if self._Theta is None:
            self._Theta = np.arctan2(self.yy, self.xx)
        return self._Theta

    @property
    def mask(self) -> np.ndarray:
        """Circular pupil mask (R <= 1.0)."""
        if self._mask is None:
            radius = min(self._height, self._width) / 2
            self._mask = (self.R <= radius).astype(np.float64)
        return self._mask

    @property
    def pixel_x(self) -> np.ndarray:
        """Pixel x coordinates (centered at 0, in pixel units)."""
        if self._pixel_x is None:
            self._pixel_x = np.arange(self._width, dtype=np.float64) - self._width / 2
        return self._pixel_x

    @property
    def pixel_y(self) -> np.ndarray:
        """Pixel y coordinates (centered at 0, in pixel units)."""
        if self._pixel_y is None:
            self._pixel_y = np.arange(self._height, dtype=np.float64) - self._height / 2
        return self._pixel_y

    def generate_checkerboard(self, period: int = 100) -> np.ndarray:
        """生成棋盘格图案。

        Args:
            period: 棋盘格周期（像素）

        Returns:
            棋盘格图案 (uint16)
        """
        max_val = self._max_val

        y = np.arange(self._height) // period
        x = np.arange(self._width) // period
        X, Y = np.meshgrid(x, y)

        checker = (X + Y) % 2
        img = (checker * max_val).astype(np.uint16)

        return img

    def generate_binary_grating(
        self, a: int = 2, b: int = 3, direction: str = "horizontal"
    ) -> np.ndarray:
        """生成二值光栅。

        Args:
            a: 明条纹宽度
            b: 暗条纹宽度
            direction: "horizontal" 或 "vertical"

        Returns:
            二值光栅图案 (uint16)
        """
        height, width = self._height, self._width
        max_val = (2**self.bits - 1) // 2

        if direction == "horizontal":
            y = np.arange(height)
            grating = np.where(y % (a + b) < b, 0, max_val)
            img = np.tile(grating[:, np.newaxis], (1, width))
        else:
            x = np.arange(width)
            grating = np.where(x % (a + b) < b, 0, max_val)
            img = np.tile(grating[np.newaxis, :], (height, 1))

        return img.astype(np.uint16)

    def generate_microlens_array(
        self,
        lens_size: int = 200,
        focal_length: float = 0.1,
        wavelength: float = 532e-9,
        pixel_size: float = 8e-6,
    ) -> np.ndarray:
        """生成微透镜阵列。

        Args:
            lens_size: 单个透镜尺寸（像素）
            focal_length: 焦距 (m)
            wavelength: 波长 (m)
            pixel_size: 像素大小 (m)

        Returns:
            微透镜阵列图案 (uint16)
        """
        height, width = self._height, self._width
        max_val = self._max_val

        x = (np.arange(lens_size, dtype=np.float64) - lens_size / 2) * pixel_size
        y = (np.arange(lens_size, dtype=np.float64) - lens_size / 2) * pixel_size
        X, Y = np.meshgrid(x, y)
        r2 = X**2 + Y**2

        k = 2 * np.pi / wavelength
        phase = k * (focal_length - np.sqrt(r2 + focal_length**2))
        phase_wrapped = np.mod(phase, 2 * np.pi)
        lens_pattern = (phase_wrapped / (2 * np.pi) * max_val).astype(np.uint16)

        n_y = height // lens_size + 1
        n_x = width // lens_size + 1

        array = np.tile(lens_pattern, (n_y, n_x))

        return array[:height, :width]

    def generate_turbulence_screen(
        self,
        Cn2: float = 1e-14,
        L: float = 1000,
        wavelength: float = 532e-9,
        pixel_size: float = 8e-6,
        screen_size: float | None = None,
        L0: float | None = None,
        l0: float | None = None,
        random_seed: int | None = None,
        method: str = "kolmogorov",
    ) -> np.ndarray:
        """生成湍流相位���。

        Args:
            Cn2: 折射率结构常数 (m^(2/3))。
                默认 1e-14 对应弱湍流。
            L: 传播路径长度 (m)
            wavelength: 波长 (m)
            pixel_size: 像素大小 (m)
            screen_size: 屏幕物理尺寸 (m)。默认 max(h,w) * pixel_size
            L0: 外尺度 (m)。默认 10 * screen_size
            l0: 内尺度 (m)。默认 pixel_size * 2
            random_seed: 随机种子（用于 kolmogorov 方法）
            method: "kolmogorov" 或 "vankarman"

        Returns:
            湍流相位屏 (uint16, 0 到 2π)
        """
        height, width = self._height, self._width
        max_val = self._max_val

        if screen_size is None:
            screen_size = max(height, width) * pixel_size

        if L0 is None:
            L0 = 10 * screen_size
        if l0 is None:
            l0 = pixel_size * 2

        # Compute Fried parameter r0 from Cn2
        r0 = (wavelength**2 / (Cn2 * L * 0.033 * (2 * np.pi) ** 2)) ** (3 / 5)

        try:
            if method == "kolmogorov":
                # Use PhaseScreenKolmogorov for Kolmogorov turbulence
                screen = PhaseScreenKolmogorov(
                    nx_size=height,
                    pixel_scale=pixel_size,
                    r0=r0,
                    L0=L0,
                    random_seed=random_seed,
                )
                phase_screen = screen.scrn
            else:
                # Fallback to ft_phase_screen for Von Karman
                screen = ft_phase_screen(r0, height, pixel_size, L0, l0)
                phase_screen = screen[:height, :width]
        except np.linalg.LinAlgError:
            # PhaseScreenKolmogorov can fail for certain L0/pixel_scale combinations
            # Fall back to ft_phase_screen
            screen = ft_phase_screen(r0, height, pixel_size, L0, l0)
            phase_screen = screen[:height, :width]

        # Normalize to [0, 2π) and convert to uint16
        phase_min = phase_screen.min()
        phase_max = phase_screen.max()
        phase_normalized = (
            (phase_screen - phase_min)
            / (phase_max - phase_min + 1e-10)
            * 2 * np.pi
        )

        img = (phase_normalized / (2 * np.pi) * max_val).astype(np.uint16)
        return img

    def generate_zernike(
        self,
        n: int,
        m: int,
        amplitude: float = 1.0,
        radius: float | None = None,
    ) -> np.ndarray:
        """生成单个Zernike模式。

        Args:
            n: Zernike径向阶数
            m: Zernike角向阶数
            amplitude: 振幅
            radius: 瞳孔半径（像素）。默认 min(h,w)/2

        Returns:
            Zernike相位图案 (uint16, 0 到 2^bits-1)
        """
        gen = ZernikeGenerator(resolution=(self._width, self._height), radius=radius)
        gen.set_bits(self.bits)
        j = nm_to_noll(n, m)-1
        gen.precompute_bases(j)
        return gen.generate(n, m, amplitude)

    def generate_zernike_polynomial(
        self,
        coefficients: dict[tuple[int, int], float] | None = None,
        radius: float | None = None,
    ) -> np.ndarray:
        """生成多模式Zernike多项式。

        Args:
            coefficients: {(n, m): amplitude} 字典
            radius: 瞳孔半径（像素）

        Returns:
            Zernike相位图案 (uint16)
        """
        gen = ZernikeGenerator(resolution=(self._width, self._height), radius=radius)
        gen.set_bits(self.bits)

        if coefficients is None:
            coefficients = {}
        max_noll = max(nm_to_noll(n, m) for (n, m) in coefficients) if coefficients else 1
        n_terms = max_noll
        gen.precompute_bases(n_terms)

        if not coefficients:
            return np.zeros((self._height, self._width), dtype=np.uint16)

        return gen.generate_polynomial(coefficients)

    def to_uint16(self, phase_radians: np.ndarray) -> np.ndarray:
        """将弧度相位转换为uint16格式。

        Args:
            phase_radians: 弧度相位数组

        Returns:
            uint16格式相位数组 (0 到 2^bits-1)
        """
        phase_wrapped = np.mod(phase_radians, 2 * np.pi)
        img = (phase_wrapped / (2 * np.pi) * self._max_val).astype(np.uint16)
        return img

    def generate_focus(
        self,
        focal_length: float,
        wavelength: float = 532e-9,
        pixel_size: float = 8e-6,
        wrap_phase: bool = True,
    ) -> np.ndarray:
        """生成聚焦图案（透镜相位）。

        Args:
            focal_length: 焦距 (m)
            wavelength: 波长 (m)
            pixel_size: 像素大小 (m)
            wrap_phase: 是否包裹相位

        Returns:
            聚焦图案 (uint16 或弧度)
        """
        max_val = self._max_val
        R2 = self.xx**2 + self.yy**2
        phase = (np.pi / wavelength / focal_length) * (R2 * pixel_size**2)
        if not wrap_phase:
            return phase

        phase_wrapped = np.mod(phase, 2 * np.pi)
        img = (phase_wrapped / (2 * np.pi) * max_val).astype(np.uint16)
        return img

    def generate_dammann_grating(
        self, order: int = 3, fill_factor: float = 0.5
    ) -> np.ndarray:
        """生成Dammann光栅。

        Args:
            order: 衍射级次数量 (通常 2, 3, 4)
            fill_factor: 填充因子 (0.0 到 1.0)

        Returns:
            Dammann光栅图案 (uint16)
        """
        height, width = self.resolution[1], self.resolution[0]
        max_val = 2**self.bits - 1

        if order <= 0:
            order = 1

        elem_width = width // order
        elem_height = height // order

        img = np.zeros((height, width), dtype=np.uint16)

        for i in range(order):
            for j in range(order):
                y_start = i * elem_height
                y_end = min((i + 1) * elem_height, height)
                x_start = j * elem_width
                x_end = min((j + 1) * elem_width, width)

                if (i + j) % 2 == 0:
                    img[y_start:y_end, x_start:x_end] = max_val
                else:
                    img[y_start:y_end, x_start:x_end] = 0

        return img

    def linear_grating(
        self,
        period: float,
        phase_range: float = 2 * np.pi,
        wrap_phase: bool = True,
    ) -> np.ndarray:
        """生成线性（闪耀）光栅。

        Args:
            period: 光栅周期
            phase_range: 最大相位范围 (弧度)
            wrap_phase: 是否包裹相位

        Returns:
            线性光栅图案
        """
        max_val = self._max_val
        phase = (self.xx / period) * phase_range

        if not wrap_phase:
            return np.mod(phase, phase_range)

        phase_wrapped = np.mod(phase, phase_range)
        img = (phase_wrapped / phase_range * max_val).astype(np.uint16)
        return img

    def circular_grating(
        self,
        radius: float,
        phase_range: float = 2 * np.pi,
        wrap_phase: bool = True,
    ) -> np.ndarray:
        """生成圆形（径向）光栅。

        Args:
            radius: 光栅半径
            phase_range: 最大相位范围
            wrap_phase: 是否包裹相位

        Returns:
            圆形光栅图案
        """
        max_val = self._max_val
        rr = np.sqrt(self.xx**2 + self.yy**2)
        phase = (rr / radius) * phase_range

        if not wrap_phase:
            return np.mod(phase, phase_range)

        phase_wrapped = np.mod(phase, phase_range)
        img = (phase_wrapped / phase_range * max_val).astype(np.uint16)
        return img

    def lens(
        self,
        focal_length: float,
        wavelength: float = 532e-9,
        pixel_size: float = 8e-6,
    ) -> np.ndarray:
        """生成透镜相位模式。

        Args:
            focal_length: 焦距 (m)
            wavelength: 波长 (m)
            pixel_size: 像素大小 (m)

        Returns:
            透镜相位 (弧度, 未包裹)
        """
        xx = self.pixel_x * pixel_size
        yy = self.pixel_y * pixel_size
        xx, yy = np.meshgrid(xx, yy)
        r2 = xx**2 + yy**2
        k = 2 * np.pi / wavelength
        phase = k * (focal_length - np.sqrt(r2 + focal_length**2))
        return np.mod(phase, 2 * np.pi)

    def hologram(self, period: float, phase_range: float = 2 * np.pi) -> np.ndarray:
        """生成全息图（线性光栅的别名）。

        Args:
            period: 光栅周期 (像素)
            phase_range: 最大相位范围 (弧度)

        Returns:
            相位图案 (弧度)
        """
        return self.linear_grating(period=period, phase_range=phase_range, wrap_phase=False)

    def dammann_grating(
        self,
        width: int,
        height: int,
        order: int = 3,
        phase_range: float = 2 * np.pi,
    ) -> np.ndarray:
        """生成Dammann光栅相位图案。

        Args:
            width: 宽度
            height: 高度
            order: 衍射级次
            phase_range: 相位范围

        Returns:
            相位图案 (弧度)
        """
        if order <= 1:
            order = 2

        period_x = width // order
        period_y = height // order

        phase_x = (self.xx // period_x) % 2 * np.pi
        phase_y = (self.yy // period_y) % 2 * np.pi

        return np.mod(phase_x + phase_y, phase_range)