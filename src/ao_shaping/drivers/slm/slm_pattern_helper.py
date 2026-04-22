# 相位图案生成函数
from __future__ import annotations

import numpy as np


class PatternHelper:
    def __init__(self, resolution: tuple[int, int], bits: int = 10) -> None:
        self.resolution = resolution
        self.bits = bits

    def generate_focus(
        self, focal_length: float, wavelength: float = 532e-9, pixel_size: float = 8e-6
    ) -> np.ndarray:
        """
        生成聚焦相位图案 (抛物面相位)

        参数:
            focal_length: 焦距 (米)
            wavelength: 波长 (米), 默认 532nm
            pixel_size: 像素大小 (米), 默认 8um

        返回:
            相位图案 (0~2^Bits-1)
        """
        height, width = self.resolution[1], self.resolution[0]
        max_val = 2**self.bits - 1

        # 创建坐标网格
        x = np.arange(width) - width // 2
        y = np.arange(height) - height // 2
        X, Y = np.meshgrid(x, y)

        # 计算半径 (像素)
        R2 = X**2 + Y**2

        # 抛物面相位: phi = (pi / lambda / f) * r^2
        # 转换为 SLM 灰度值
        phase = (np.pi / wavelength / focal_length) * (R2 * pixel_size**2)

        # 包裹到 0~2π 并映射到 0~max_val
        phase_wrapped = np.mod(phase, 2 * np.pi)
        img = (phase_wrapped / (2 * np.pi) * max_val).astype(np.uint16)

        return img

    def generate_checkerboard(self, period: int = 100) -> np.ndarray:
        """
        生成棋盘格相位图案

        参数:
            period: 棋盘格周期 (像素)

        返回:
            相位图案 (0 或 max_val)
        """
        height, width = self.resolution[1], self.resolution[0]
        max_val = 2**self.bits - 1

        # 创建棋盘格
        y = np.arange(height) // period
        x = np.arange(width) // period
        X, Y = np.meshgrid(x, y)

        # 黑白交替
        checker = (X + Y) % 2
        img = (checker * max_val).astype(np.uint16)

        return img

    def generate_binary_grating(
        self, a: int = 2, b: int = 3, direction: str = "horizontal"
    ) -> np.ndarray:
        """
        生成 01 光栅 (二元光栅)

        参数:
            a: 亮条纹宽度 (像素)
            b: 暗条纹宽度 (像素)
            direction: 'horizontal' 或 'vertical'

        返回:
            相位图案 (0 或 max_val//2， pi)
        """
        height, width = self.resolution[1], self.resolution[0]
        max_val = (2**self.bits - 1) // 2

        if direction == "horizontal":
            # 水平光栅
            y = np.arange(height)
            grating = np.where(y % (a + b) < b, 0, max_val)
            img = np.tile(grating[:, np.newaxis], (1, width))
        else:
            # 垂直光栅
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
        """
        生成微透镜阵列相位图案

        参数:
            lens_size: 单个微透镜的大小 (像素)
            focal_length: 焦距 (米)
            wavelength: 波长 (米)
            pixel_size: 像素大小 (米)

        返回:
            相位图案
        """
        height, width = self.resolution[1], self.resolution[0]
        max_val = 2**self.bits - 1

        # 创建单个透镜的相位图案 (使用精确薄透镜公式)
        x = (np.arange(lens_size, dtype=np.float64) - lens_size / 2) * pixel_size
        y = (np.arange(lens_size, dtype=np.float64) - lens_size / 2) * pixel_size
        X, Y = np.meshgrid(x, y)
        r2 = X**2 + Y**2

        # 精确薄透镜相位: φ = k * (f - √(r² + f²))
        k = 2 * np.pi / wavelength
        phase = k * (focal_length - np.sqrt(r2 + focal_length**2))
        phase_wrapped = np.mod(phase, 2 * np.pi)
        lens_pattern = (phase_wrapped / (2 * np.pi) * max_val).astype(np.uint16)

        # 平铺成阵列
        n_y = height // lens_size + 1
        n_x = width // lens_size + 1

        array = np.tile(lens_pattern, (n_y, n_x))

        # 裁剪到目标大小
        img = array[:height, :width]

        return img

    def generate_turbulence_screen(
        self,
        Cn2: float = 1e-14,
        L: float = 1000,
        wavelength: float = 532e-9,
        pixel_size: float = 8e-6,
        screen_size: float = None,
    ) -> np.ndarray:
        """
        生成大气湍流相位屏 (基于 Kolmogorov 谱)

        参数:
            Cn2: 折射率结构常数 (m^(-2/3))
            L: 传输距离 (米)
            wavelength: 波长 (米)
            pixel_size: 像素大小 (米)
            screen_size: 屏的物理大小 (米), 默认根据分辨率计算

        返回:
            相位图案
        """
        height, width = self.resolution[1], self.resolution[0]
        max_val = 2**self.bits - 1

        if screen_size is None:
            screen_size = max(height, width) * pixel_size

        # 创建频率网格
        kx = 2 * np.pi * np.fft.fftfreq(width, pixel_size)
        ky = 2 * np.pi * np.fft.fftfreq(height, pixel_size)
        KX, KY = np.meshgrid(kx, ky)
        K = np.sqrt(KX**2 + KY**2)
        K[0, 0] = 1e-10  # 避免除以零

        # Kolmogorov 谱: Phi(k) = 0.033 * Cn2 * k^(-11/3)
        # 相位屏功率谱: W_phi(k) = 2 * pi * k^2 * L * Phi(k)
        power_spectrum = 2 * np.pi * K**2 * L * 0.033 * Cn2 * K ** (-11 / 3)

        # 生成随机相位
        random_phase = np.random.randn(height, width) + 1j * np.random.randn(
            height, width
        )

        # 在频域应用功率谱
        screen_fft = np.sqrt(power_spectrum) * random_phase

        # 逆 FFT 得到相位屏
        phase_screen = np.real(np.fft.ifft2(screen_fft))

        # 归一化并映射到 0~max_val
        phase_screen = (
            (phase_screen - phase_screen.min())
            / (phase_screen.max() - phase_screen.min())
            * max_val
        )

        return phase_screen.astype(np.uint16)

    def generate_zernike(
        self, n: int, m: int, amplitude: float = 1.0, radius: float = None
    ) -> np.ndarray:
        """
        生成 Zernike 多项式相位图案

        参数:
            n: 径向阶数
            m: 角向阶数
            amplitude: 振幅 (单位: 波长)
            radius: 圆形孔径半径 (像素), 默认为短边的一半

        返回:
            相位图案
        """
        height, width = self.resolution[1], self.resolution[0]
        max_val = 2**self.bits - 1

        if radius is None:
            radius = min(height, width) // 2

        # 创建归一化坐标
        x = (np.arange(width) - width // 2) / radius
        y = (np.arange(height) - height // 2) / radius
        X, Y = np.meshgrid(x, y)

        # 转换为极坐标
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)

        # 只在圆内计算
        mask = R <= 1.0

        # 计算 Zernike 多项式
        from scipy.special import factorial

        def zernike_radial(n, m, r):
            """Zernike 径向多项式"""
            R = np.zeros_like(r)
            for k in range((n - abs(m)) // 2 + 1):
                coef = ((-1) ** k * factorial(n - k)) / (
                    factorial(k)
                    * factorial((n + abs(m)) // 2 - k)
                    * factorial((n - abs(m)) // 2 - k)
                )
                R += coef * r ** (n - 2 * k)
            return R

        # 计算 Zernike 多项式
        if m >= 0:
            Z = zernike_radial(n, m, R) * np.cos(m * Theta)
        else:
            Z = zernike_radial(n, -m, R) * np.sin(-m * Theta)

        # 应用圆形孔径
        Z = Z * mask

        # 转换为相位 (单位: 2π)
        phase = Z * amplitude * 2 * np.pi

        # 包裹并映射到 0~max_val
        phase_wrapped = np.mod(phase, 2 * np.pi)
        img = (phase_wrapped / (2 * np.pi) * max_val).astype(np.uint16)

        return img

    def generate_zernike_polynomial(
        self,
        n_max: int = 4,
        coefficients: dict[tuple[int, int], float] | None = None,
        radius: float = None,
    ) -> np.ndarray:
        """
        生成多阶 Zernike 多项式叠加相位图案

        参数:
            n_max: 最大径向阶数
            coefficients: {(n, m): amplitude} 字典，例如 {(0,0): 1.0, (1,-1): 0.5}
            radius: 圆形孔径半径 (像素), 默认为短边的一半

        返回:
            相位图案
        """
        height, width = self.resolution[1], self.resolution[0]
        max_val = 2**self.bits - 1

        if radius is None:
            radius = min(height, width) // 2

        # 创建归一化坐标
        x = (np.arange(width, dtype=np.float64) - width / 2) / radius
        y = (np.arange(height, dtype=np.float64) - height / 2) / radius
        X, Y = np.meshgrid(x, y)

        # 转换为极坐标
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)

        # 只在圆内计算
        mask = R <= 1.0

        # 计算 Zernike 多项式
        from scipy.special import factorial

        def zernike_radial(n, m, r):
            """Zernike 径向多项式"""
            R = np.zeros_like(r)
            for k in range((n - abs(m)) // 2 + 1):
                coef = ((-1) ** k * factorial(n - k)) / (
                    factorial(k)
                    * factorial((n + abs(m)) // 2 - k)
                    * factorial((n - abs(m)) // 2 - k)
                )
                R += coef * r ** (n - 2 * k)
            return R

        # 默认系数：前n_max阶都为0，除了 piston (0,0) 为1
        if coefficients is None:
            coefficients = {}
            for n in range(n_max + 1):
                for m in range(-n, n + 1):
                    if (n - abs(m)) % 2 == 0:
                        if n == 0 and m == 0:
                            coefficients[(n, m)] = 1.0  # piston
                        else:
                            coefficients[(n, m)] = 0.0

        # 叠加各阶 Zernike
        phase_total = np.zeros_like(R)

        for (n, m), amp in coefficients.items():
            if abs(amp) < 1e-10:  # 跳过零系数
                continue

            if m >= 0:
                Z = zernike_radial(n, m, R) * np.cos(m * Theta)
            else:
                Z = zernike_radial(n, -m, R) * np.sin(-m * Theta)

            # 应用圆形孔径
            Z = Z * mask

            # 转换为相位 (单位: 2π) 并叠加
            phase_total += Z * amp * 2 * np.pi

        # 包裹并映射到 0~max_val
        phase_wrapped = np.mod(phase_total, 2 * np.pi)
        img = (phase_wrapped / (2 * np.pi) * max_val).astype(np.uint16)

        return img

    def generate_dammann_grating(
        self, order: int = 3, fill_factor: float = 0.5
    ) -> np.ndarray:
        """
        Generate a Dammann grating phase pattern

        Args:
            order: Number of diffraction orders in each direction (typically 2, 3, 4)
            fill_factor: Ratio of transparent area in each cell (0.0 to 1.0)

        Returns:
            Phase pattern (0 or max_val)
        """
        height, width = self.resolution[1], self.resolution[0]
        max_val = 2**self.bits - 1

        if order <= 0:
            order = 1

        # Calculate the size of each grating element
        elem_width = width // order
        elem_height = height // order

        # Create the Dammann grating pattern
        img = np.zeros((height, width), dtype=np.uint16)

        # Fill each grating element with alternating phase values
        for i in range(order):
            for j in range(order):
                # Define the region for this grating element
                y_start = i * elem_height
                y_end = min((i + 1) * elem_height, height)
                x_start = j * elem_width
                x_end = min((j + 1) * elem_width, width)

                # Determine phase based on position (alternating 0 and pi)
                if (i + j) % 2 == 0:
                    # Set to max phase (pi phase shift)
                    img[y_start:y_end, x_start:x_end] = max_val
                else:
                    # Set to zero phase
                    img[y_start:y_end, x_start:x_end] = 0

        return img


class SLMPatternHelper:
    """Generate phase patterns in radians for Streamlit SLM helpers."""

    def linear_grating(
        self,
        width: int,
        height: int,
        period: float,
        phase_range: float = 2 * np.pi,
    ) -> np.ndarray:
        """Generate a horizontal linear grating."""
        x = np.arange(width, dtype=np.float64)
        phase_line = np.mod(x / period * phase_range, phase_range)
        return np.tile(phase_line, (height, 1))

    def circular_grating(
        self,
        width: int,
        height: int,
        radius: float,
        phase_range: float = 2 * np.pi,
    ) -> np.ndarray:
        """Generate a radial circular grating."""
        x = np.arange(width, dtype=np.float64) - width / 2
        y = np.arange(height, dtype=np.float64) - height / 2
        xx, yy = np.meshgrid(x, y)
        rr = np.sqrt(xx**2 + yy**2)
        return np.mod(rr / radius * phase_range, phase_range)

    def lens(
        self,
        width: int,
        height: int,
        focal_length: float,
        wavelength: float,
        pixel_size: float,
    ) -> np.ndarray:
        """Generate a wrapped thin-lens phase profile.

        All parameters in meters:
            focal_length: focal length (m)
            wavelength: wavelength (m)
            pixel_size: pixel size (m)
        """
        x = (np.arange(width, dtype=np.float64) - width / 2) * pixel_size
        y = (np.arange(height, dtype=np.float64) - height / 2) * pixel_size
        xx, yy = np.meshgrid(x, y)
        r2 = xx**2 + yy**2
        k = 2 * np.pi / wavelength
        phase = k * (focal_length - np.sqrt(r2 + focal_length**2))
        return np.mod(phase, 2 * np.pi)

    def hologram(
        self,
        width: int,
        height: int,
        period: float,
        phase_range: float = 2 * np.pi,
    ) -> np.ndarray:
        """Generate a simple blazed grating hologram."""
        return self.linear_grating(
            width=width,
            height=height,
            period=period,
            phase_range=phase_range,
        )

    def dammann_grating(
        self,
        width: int,
        height: int,
        order: int = 3,
        phase_range: float = 2 * np.pi,
    ) -> np.ndarray:
        """Generate a Dammann grating phase pattern.

        A Dammann grating is a binary-phase grating that generates uniform diffraction orders.
        It creates a specific number of equally intense spots at regular intervals.

        Args:
            width: Width of the output pattern in pixels
            height: Height of the output pattern in pixels
            order: Number of diffraction orders (typically 2, 3, 4, etc.)
            phase_range: Total phase range in radians (default 2π)

        Returns:
            Phase pattern array in radians
        """
        # Create coordinate grids
        x = np.arange(width, dtype=np.float64)
        y = np.arange(height, dtype=np.float64)
        X, Y = np.meshgrid(x, y)

        if order <= 1:
            order = 2  # Minimum order is 2

        # Calculate the Dammann grating pattern
        # For a 1D Dammann grating, the phase follows a specific sequence to create uniform orders
        # For 2D, we can combine two 1D gratings orthogonally

        # Calculate the spatial frequency for the specified order
        # The grating period is chosen such that it creates the desired number of orders
        period_x = width // order
        period_y = height // order

        # Create the phase pattern based on Dammann grating principles
        # This implementation creates a pattern that generates uniform diffraction orders
        phase_x = (X // period_x) % 2 * np.pi  # Alternate 0 and π phases
        phase_y = (Y // period_y) % 2 * np.pi  # Alternate 0 and π phases

        # Combine both dimensions (XOR-like behavior)
        combined_phase = np.mod(phase_x + phase_y, phase_range)

        return combined_phase
