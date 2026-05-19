# Pattern Helper - 光学相位图案生成工具
from __future__ import annotations

import numpy as np

from ao_shaping.utils.zernike_calc import ZernikeGenerator
from ao_shaping.utils.phase_unwrap import PhaseUnwrapper, UnwrapStrategy, unwrap_phase
from ao_shaping.algorithm.phase_wrap import PhaseWrapOptimizer

from aotools.turbulence.infinitephasescreen import PhaseScreenKolmogorov


UNWRAP_STRATEGY = "iterative"
WRAP_STRATEGY = "hybrid"


class PhaseWrapOptimizerHelper:
    _instance: PhaseWrapOptimizer | None = None

    @classmethod
    def get_optimizer(cls, slm_height: int = 1600, slm_width: int = 2560, strategy: str = WRAP_STRATEGY) -> PhaseWrapOptimizer:
        if cls._instance is None or cls._instance.slm_height != slm_height or cls._instance.slm_width != slm_width:
            cls._instance = PhaseWrapOptimizer(slm_height=slm_height, slm_width=slm_width, oversample=2)
        return cls._instance

    @classmethod
    def set_strategy(cls, strategy: str):
        global WRAP_STRATEGY
        WRAP_STRATEGY = strategy

    @classmethod
    def wrap(cls, phase_unwrapped: np.ndarray, strategy: str | None = None) -> np.ndarray:
        s = strategy or WRAP_STRATEGY
        optimizer = cls.get_optimizer(phase_unwrapped.shape[0], phase_unwrapped.shape[1], s)
        return optimizer.optimize(phase_unwrapped, strategy=s)

    @classmethod
    def min_jump_wrap(cls, phase_unwrapped: np.ndarray) -> np.ndarray:
        optimizer = cls.get_optimizer(phase_unwrapped.shape[0], phase_unwrapped.shape[1])
        return optimizer.min_jump_wrap(phase_unwrapped)

    @classmethod
    def error_diffusion_wrap(cls, phase_unwrapped: np.ndarray, quantization_levels: int = 256) -> np.ndarray:
        optimizer = cls.get_optimizer(phase_unwrapped.shape[0], phase_unwrapped.shape[1])
        return optimizer.error_diffusion_wrap(phase_unwrapped, quantization_levels)

    @classmethod
    def oversample_smooth(cls, phase_unwrapped: np.ndarray, sigma_pixels: float = 0.8) -> np.ndarray:
        optimizer = cls.get_optimizer(phase_unwrapped.shape[0], phase_unwrapped.shape[1])
        return optimizer.oversample_smooth(phase_unwrapped, sigma_pixels)

    @classmethod
    def detect_jumps(cls, wrapped_phase: np.ndarray, threshold: float = 0.5 * np.pi) -> np.ndarray:
        return PhaseWrapOptimizer.detect_jumps(wrapped_phase, threshold)

    @classmethod
    def calculate_efficiency(cls, phase: np.ndarray) -> float:
        return PhaseWrapOptimizer.calculate_diffraction_efficiency(phase)


class PhaseUnwrapperHelper:
    _instance: PhaseUnwrapper | None = None

    @classmethod
    def get_unwrapper(cls, resolution: tuple[int, int], strategy: str = UNWRAP_STRATEGY) -> PhaseUnwrapper:
        if cls._instance is None or cls._instance.resolution != resolution:
            strategy_enum = UnwrapStrategy[strategy.upper()] if strategy.upper() in [e.name for e in UnwrapStrategy] else UnwrapStrategy.ITERATIVE
            cls._instance = PhaseUnwrapper(resolution=resolution, strategy=strategy_enum)
        return cls._instance

    @classmethod
    def set_strategy(cls, strategy: str):
        global UNWRAP_STRATEGY
        UNWRAP_STRATEGY = strategy
        if cls._instance is not None:
            strategy_enum = UnwrapStrategy[strategy.upper()] if strategy.upper() in [e.name for e in UnwrapStrategy] else UnwrapStrategy.ITERATIVE
            cls._instance.strategy = strategy_enum

    @classmethod
    def unwrap(cls, wrapped: np.ndarray, strategy: str | None = None) -> np.ndarray:
        s = strategy or UNWRAP_STRATEGY
        unwrapper = cls.get_unwrapper(wrapped.shape, s)
        return unwrapper.unwrap(wrapped)

    @classmethod
    def unwrap_fast(cls, wrapped: np.ndarray) -> np.ndarray:
        unwrapper = cls.get_unwrapper(wrapped.shape, "iterative")
        return unwrapper.unwrap_fast(wrapped)


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
        - vortex: 涡旋相位
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

        # Turbulence screen (lazy-initialized)
        self._turbulence_screen: PhaseScreenKolmogorov | None = None

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

    def init_turbulence_screen(
        self,
        r0: float,
        L0: float,
        pixel_scale: float,
        random_seed: int | None = None,
    ) -> None:
        """初始化湍流相屏成员变量。

        Args:
            r0: Fried参数 (米)
            L0: 外尺度 (米)
            pixel_scale: 每个像素的物理尺寸 (米)
            random_seed: 随机种子（可选）
        """
        self._turbulence_screen = PhaseScreenKolmogorov(
            nx_size=self._height,
            pixel_scale=pixel_scale,
            r0=r0,
            L0=L0,
            random_seed=random_seed,
        )

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

    def generate_turbulence_screen(self) -> np.ndarray:
        """生成湍流相位屏（通过add_row()迭代生成）。

        调用前必须先调用 init_turbulence_screen() 初始化湍流屏。

        Returns:
            湍流相位屏 (uint16, 0 到 2^bits-1)

        Raises:
            RuntimeError: 如果湍流屏未初始化
        """
        if self._turbulence_screen is None:
            raise RuntimeError(
                "Turbulence screen not initialized. "
                "Call init_turbulence_screen() first with r0, L0, and pixel_scale."
            )

        height, width = self._height, self._width
        max_val = self._max_val

        # Call add_row() to generate new phase and get updated screen
        self._turbulence_screen.add_row()

        # Get phase from .scrn property (in radians)
        phase_screen = self._turbulence_screen.scrn

        # Extract region matching our resolution
        phase_cropped = phase_screen[:height, :width]

        # Normalize to [0, 2π) and convert to uint16
        phase_min = phase_cropped.min()
        phase_max = phase_cropped.max()
        phase_normalized = (
            (phase_cropped - phase_min)
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
        gen = ZernikeGenerator(resolution=(self._width, self._height), radius=radius)
        gen.set_bits(self.bits)
        phase = gen.generate(n, m, amplitude)
        return self._zernike_to_uint16(phase)

    def generate_zernike_polynomial(
        self,
        coefficients: dict[tuple[int, int], float] | None = None,
        radius: float | None = None,
    ) -> np.ndarray:
        gen = ZernikeGenerator(resolution=(self._width, self._height), radius=radius)
        gen.set_bits(self.bits)

        if coefficients is None:
            coefficients = {}
        if not coefficients:
            return np.zeros((self._height, self._width), dtype=np.uint16)

        phase = gen.generate_polynomial(coefficients)
        return self._zernike_to_uint16(phase)

    def _zernike_to_uint16(self, phase: np.ndarray) -> np.ndarray:
        result = np.nan_to_num(phase, nan=0.0)
        result = (result - result.min()) / (result.max() - result.min() + 1e-10)
        return (result * self._max_val).astype(np.uint16)

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
        lens_radius: float | None = None,
    ) -> np.ndarray:
        """生成聚焦图案（透镜相位）。

        Args:
            focal_length: 焦距 (m)
            wavelength: 波长 (m)
            pixel_size: 像素大小 (m)
            wrap_phase: 是否包裹相位
            lens_radius: 透镜半径（像素），默认 min(height, width)/2

        Returns:
            聚焦图案 (uint16 或弧度)，超出透镜半径的区域置为0
        """
        max_val = self._max_val
        R2 = self.xx**2 + self.yy**2
        phase = (np.pi / wavelength / focal_length) * (R2 * pixel_size**2)

        # 应用透镜半径掩模
        if lens_radius is not None:
            # 创建圆形掩模 (R <= lens_radius)
            mask = self.R <= lens_radius
            phase = phase * mask.astype(np.float64)

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
        lens_radius: float | None = None,
    ) -> np.ndarray:
        """生成透镜相位模式。

        Args:
            focal_length: 焦距 (m)
            wavelength: 波长 (m)
            pixel_size: 像素大小 (m)
            lens_radius: 透镜半径（像素），默认 min(height, width)/2

        Returns:
            透镜相位 (弧度, 未包裹)，超出透镜半径的区域置为0
        """
        xx = self.pixel_x * pixel_size
        yy = self.pixel_y * pixel_size
        xx, yy = np.meshgrid(xx, yy)
        r2 = xx**2 + yy**2
        k = 2 * np.pi / wavelength
        phase = k * (focal_length - np.sqrt(r2 + focal_length**2))

        # 应用透镜半径掩模
        if lens_radius is not None:
            # 创建圆形掩模 (R <= lens_radius * pixel_size)
            mask = np.sqrt(xx**2 + yy**2) <= (lens_radius * pixel_size)
            phase = phase * mask.astype(np.float64)

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

    def generate_vortex(
        self,
        topological_charge: int = 1,
        wavelength: float = 532e-9,
        pixel_size: float = 8e-6,
        wrap_phase: bool = True,
    ) -> np.ndarray:
        """生成涡旋相位（螺旋相位）。
        
        生成具有拓扑荷的涡旋相位，位移与角度Theta成正比：phi = l * Theta
        
        Args:
            topological_charge: 拓扑荷 (l)，可以是正负整数
            wavelength: 波长 (m)
            pixel_size: 像素大小 (m)
            wrap_phase: 是否包裹相位到[0, 2π)
            
        Returns:
            涡旋相位图案 (uint16 或弧度)
        """
        # 涡旋相位：phi = l * theta，其中theta是角坐标
        phase = topological_charge * self.Theta

        if not wrap_phase:
            return phase

        # 将相位包裹到[0, 2π)范围并转换为uint16
        phase_wrapped = np.mod(phase, 2 * np.pi)
        img = (phase_wrapped / (2 * np.pi) * self._max_val).astype(np.uint16)
        return img

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

    def unwrap_phase(self, wrapped: np.ndarray, strategy: str | None = None) -> np.ndarray:
        """解包相位（将包裹相位转换为连续相位）。

        Args:
            wrapped: 包裹相位 [0, 2π)
            strategy: 解包策略 ("goldstein", "quality_guided", "iterative", "least_squares", "region_growing", "simple", "fast")

        Returns:
            连续相位 (弧度)
        """
        if wrapped.shape != self.resolution:
            resolution = tuple(wrapped.shape)
        else:
            resolution = self.resolution
        s = strategy if strategy is not None else UNWRAP_STRATEGY
        return unwrap_phase(wrapped, strategy=s, resolution=resolution)

    def wrap_phase(self, phase_unwrapped: np.ndarray, strategy: str = WRAP_STRATEGY) -> np.ndarray:
        """包裹相位（将连续相位转换为2π范围内的包裹相位）。

        使用 PhaseWrapOptimizer 进行包裹优化，减少2π跳变产生的高频衍射误差。

        Args:
            phase_unwrapped: 连续相位 (弧度)
            strategy: 包裹策略 ("min_jump", "error_diffusion", "oversample", "repair", "hybrid")

        Returns:
            包裹相位 [0, 2π)
        """
        optimizer = PhaseWrapOptimizer(slm_height=self._height, slm_width=self._width, oversample=2)
        return optimizer.optimize(phase_unwrapped, strategy=strategy)

    def wrap_phase_min_jump(self, phase_unwrapped: np.ndarray) -> np.ndarray:
        """最小跳变包裹。

        Args:
            phase_unwrapped: 连续相位

        Returns:
            包裹相位
        """
        optimizer = PhaseWrapOptimizer(slm_height=self._height, slm_width=self._width)
        return optimizer.min_jump_wrap(phase_unwrapped)

    def wrap_phase_error_diffusion(self, phase_unwrapped: np.ndarray, quantization_levels: int = 256) -> np.ndarray:
        """误差扩散包裹。

        Args:
            phase_unwrapped: 连续相位
            quantization_levels: 量化级数

        Returns:
            包裹相位
        """
        optimizer = PhaseWrapOptimizer(slm_height=self._height, slm_width=self._width)
        return optimizer.error_diffusion_wrap(phase_unwrapped, quantization_levels)

    def wrap_phase_oversample(self, phase_unwrapped: np.ndarray, sigma_pixels: float = 0.8) -> np.ndarray:
        """过采样平滑包裹。

        Args:
            phase_unwrapped: 连续相位
            sigma_pixels: 高斯平滑 sigma

        Returns:
            包裹相位
        """
        optimizer = PhaseWrapOptimizer(slm_height=self._height, slm_width=self._width, oversample=2)
        return optimizer.oversample_smooth(phase_unwrapped, sigma_pixels)

    def detect_phase_jumps(self, wrapped_phase: np.ndarray, threshold: float = 0.5 * np.pi) -> np.ndarray:
        """检测相位跳变位置。

        Args:
            wrapped_phase: 包裹相位
            threshold: 跳变检测阈值

        Returns:
            布尔掩模，True表示跳变边缘
        """
        return PhaseWrapOptimizer.detect_jumps(wrapped_phase, threshold)

    def calculate_diffraction_efficiency(self, phase: np.ndarray) -> float:
        """计算衍射效率估计。

        Args:
            phase: 相位图

        Returns:
            衍射效率 [0, 1]
        """
        return PhaseWrapOptimizer.calculate_diffraction_efficiency(phase)
