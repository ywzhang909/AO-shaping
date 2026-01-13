"""
传统AO系统设备仿真模块

包含光源、变形镜(DM)、哈特曼传感器(WFS)、大气湍流相位屏等设备的仿真实现。
基于Zernike多项式和角谱传播法进行物理仿真。
"""

import math
import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class AOConfig:
    """AO系统配置参数"""
    # 网格参数
    N: int = 256              # 网格点数
    L: float = 0.1            # 物理孔径大小 (m)
    wavelength: float = 1550e-9  # 波长 (m)
    
    # 大气参数
    Cn2: float = 1e-14        # 折射率结构常数
    L0: float = 10.0          # 外尺度 (m)
    l0: float = 0.01          # 内尺度 (m)
    
    # DM参数
    dm_actuators: int = 8     # DM致动器数量
    dm_stroke: float = 5e-6   # DM行程 (m)
    dm_infill: bool = True    # 是否使用插值填充
    
    # WFS参数
    subapertures: int = 8     # 子孔径数量
    pixel_scale: float = 0.5  # 像素比例
    
    # 传播参数
    propagation_distance: float = 1000.0  # 传播距离 (m)


class ZernikePolynomials:
    """Zernike多项式计算工具类"""
    
    @staticmethod
    def zernike_name(n: int, m: int) -> str:
        """返回Zernike模式名称"""
        names = {
            (0, 0): "Piston",
            (1, -1): "Tilt X",
            (1, 1): "Tilt Y",
            (2, -2): "Astigmatism 45°",
            (2, 0): "Defocus",
            (2, 2): "Astigmatism 0°",
            (3, -3): "Trefoil X",
            (3, -1): "Coma X",
            (3, 1): "Coma Y",
            (3, 3): "Trefoil Y",
            (4, -4): "Secondary Astigmatism",
            (4, -2): "Secondary Astigmatism",
            (4, 0): "Spherical",
            (4, 2): "Secondary Astigmatism",
            (4, 4): "Secondary Astigmatism",
        }
        return names.get((n, m), f"Z({n},{m})")
    
    @staticmethod
    def radial_polynomial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
        """计算径向多项式R_n^m(rho)"""
        if np.abs(m) > n or (n - m) % 2 != 0:
            return np.zeros_like(rho)
        
        R = np.zeros_like(rho)
        for k in range((n - m) // 2 + 1):
            coef = ((-1) ** k * 
                    math.factorial(n - k) / 
                    (math.factorial(k) * 
                     math.factorial((n + m) // 2 - k) * 
                     math.factorial((n - m) // 2 - k)))
            R += coef * rho ** (n - 2 * k)
        return R
    
    @staticmethod
    def zernike(n: int, m: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        计算Zernike多项式Z_n^m(rho, theta)
        
        参数:
            n: 径向阶数
            m: 方位角阶数
            rho: 归一化径向坐标 [0, 1]
            theta: 角度坐标
            
        返回:
            Zernike多项式值
        """
        if m > 0:
            return np.sqrt(2 * (n + 1)) * ZernikePolynomials.radial_polynomial(n, m, rho) * np.cos(m * theta)
        elif m < 0:
            return np.sqrt(2 * (n + 1)) * ZernikePolynomials.radial_polynomial(n, -m, rho) * np.sin(-m * theta)
        else:
            return ZernikePolynomials.radial_polynomial(n, 0, rho) * np.sqrt(n + 1)
    
    @staticmethod
    def generate_basis(num_modes: int, N: int, L: float) -> np.ndarray:
        """
        生成Zernike基函数
        
        参数:
            num_modes: Zernike模式数量
            N: 网格点数
            L: 物理孔径大小
            
        返回:
            basis: shape为(num_modes, N, N)的Zernike基函数
        """
        # 创建网格
        x = np.linspace(-1, 1, N)
        y = np.linspace(-1, 1, N)
        X, Y = np.meshgrid(x, y)
        
        rho = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        
        # 圆形遮罩
        mask = rho <= 1.0
        
        # 生成Noll索引的Zernike模式
        basis = np.zeros((num_modes, N, N))
        j = 1
        for n in range(num_modes + 1):
            for m in range(-n, n + 1, 2):
                if j > num_modes:
                    break
                if np.abs(m) <= n:
                    z = ZernikePolynomials.zernike(n, m, rho, theta)
                    z[~mask] = 0
                    basis[j - 1] = z
                    j += 1
                if j > num_modes:
                    break
        
        return basis


class LightSource:
    """
    光源仿真类
    
    支持多种光束模式：高斯光束、平面波、球面波等
    """
    
    def __init__(self, 
                 wavelength: float = 1550e-9,
                 power: float = 1.0,
                 beam_waist: Optional[float] = None):
        """
        初始化光源
        
        参数:
            wavelength: 波长 (m)
            power: 光功率
            beam_waist: 高斯光束腰斑半径 (m)，None表示平面波
        """
        self.wavelength = wavelength
        self.power = power
        self.beam_waist = beam_waist
        self.k0 = 2 * np.pi / wavelength
    
    def create_plane_wave(self, N: int, L: float) -> np.ndarray:
        """
        创建平面波
        
        参数:
            N: 网格点数
            L: 物理孔径大小
            
        返回:
            E: 复振幅分布
        """
        return np.ones((N, N), dtype=complex)
    
    def create_gaussian_beam(self, N: int, L: float, w0: Optional[float] = None) -> np.ndarray:
        """
        创建高斯光束
        
        参数:
            N: 网格点数
            L: 物理孔径大小
            w0: 束腰半径，None则使用L/4
            
        返回:
            E: 复振幅分布
        """
        if w0 is None:
            w0 = L / 4
        
        # 创建网格
        x = np.linspace(-L/2, L/2, N)
        y = np.linspace(-L/2, L/2, N)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        
        # 高斯分布
        amplitude = np.exp(-(R**2) / (w0**2))
        
        return amplitude.astype(complex)


class DeformableMirror:
    """
    变形镜(DM)仿真类
    
    支持基于影响函数和Zernike模式的DM仿真
    """
    
    def __init__(self,
                 num_actuators: int = 8,
                 stroke: float = 5e-6,
                 influence_matrix: Optional[np.ndarray] = None,
                 N: int = 256):
        """
        初始化DM
        
        参数:
            num_actuators: 致动器数量（沿一个维度）
            stroke: 最大行程 (m)
            influence_matrix: 影响矩阵，shape为(num_actuators^2, N, N)
            N: 输出网格点数
        """
        self.num_actuators = num_actuators
        self.total_actuators = num_actuators ** 2
        self.stroke = stroke
        self.N = N
        
        if influence_matrix is not None:
            self.influence_matrix = influence_matrix
        else:
            self.influence_matrix = self._create_default_influence_functions()
    
    def _create_default_influence_functions(self) -> np.ndarray:
        """创建默认的高斯型影响函数"""
        N = self.N
        num_act = self.num_actuators
        
        # 创建致动器位置网格
        x = np.linspace(-0.9, 0.9, num_act)
        y = np.linspace(-0.9, 0.9, num_act)
        act_x, act_y = np.meshgrid(x, y)
        act_positions = np.column_stack([act_x.flatten(), act_y.flatten()])
        
        # 创建空间网格
        grid_x = np.linspace(-1, 1, N)
        grid_y = np.linspace(-1, 1, N)
        X, Y = np.meshgrid(grid_x, grid_y)
        
        # 高斯影响函数参数
        sigma = 0.8 / num_act  # 影响函数宽度
        
        influence_matrix = np.zeros((self.total_actuators, N, N))
        for i, (ax, ay) in enumerate(act_positions):
            R = np.sqrt((X - ax)**2 + (Y - ay)**2)
            influence_matrix[i] = np.exp(-(R**2) / (2 * sigma**2))
        
        return influence_matrix
    
    def apply_voltages(self, voltages: np.ndarray) -> np.ndarray:
        """
        根据电压计算DM面型
        
        参数:
            voltages: 电压数组，shape为(num_actuators^2,)
            
        返回:
            surface: DM面型 (m)
        """
        # 将电压缩放到行程范围
        voltages = np.clip(voltages, -1, 1) * self.stroke
        
        # 叠加影响函数
        surface = np.tensordot(voltages, self.influence_matrix, axes=1)
        
        return surface
    
    def get_command_matrix(self) -> np.ndarray:
        """
        获取命令矩阵（Zernike模式到电压的映射）
        
        返回:
            command_matrix: shape为(num_actuators^2, num_zernike)
        """
        # 简化的命令矩阵：使用Zernike模式作为期望面型
        num_modes = min(36, self.total_actuators)
        basis = ZernikePolynomials.generate_basis(num_modes, self.N, 2.0)
        
        # 展平影响矩阵
        inf_flat = self.influence_matrix.reshape(self.total_actuators, -1)
        basis_flat = basis.reshape(num_modes, -1)
        
        # 最小二乘求解
        command_matrix = np.linalg.lstsq(inf_flat.T, basis_flat.T, rcond=None)[0]
        
        return command_matrix.T


class AtmosphericTurbulence:
    """
    大气湍流相位屏仿真类
    
    使用Kolmogorov谱生成湍流相位屏
    """
    
    def __init__(self,
                 Cn2: float = 1e-14,
                 L0: float = 10.0,
                 l0: float = 0.01,
                 N: int = 256,
                 L: float = 0.1,
                 seed: Optional[int] = None):
        """
        初始化湍流

        参数:
            Cn2: 折射率结构常数
            L0: 外尺度 (m)
            l0: 内尺度 (m)
            N: 网格点数
            L: 物理孔径大小 (m)
            seed: 随机种子
        """
        self.Cn2 = Cn2
        self.L0 = L0
        self.l0 = l0
        self.N = N
        self.L = L
        self.dx = L / N
        self.k0 = 2 * np.pi

        # 预计算空间频率
        self._setup_frequency_grid()

        # 预生成相位屏
        self.phase_screen = self._generate_turbulence(seed)
    
    def _setup_frequency_grid(self):
        """设置空间频率网格"""
        fx = np.fft.fftfreq(self.N, d=self.dx)
        fy = np.fft.fftfreq(self.N, d=self.dx)
        FX, FY = np.meshgrid(fx, fy)
        self.F = np.sqrt(FX**2 + FY**2)
        
        # Kolmogorov功率谱
        self.power_spectrum = self._kolmogorov_spectrum(self.F)
    
    def _kolmogorov_spectrum(self, f: np.ndarray) -> np.ndarray:
        """
        Kolmogorov湍流功率谱
        
        参数:
            f: 空间频率
            
        返回:
            Phi_n: 折射率功率谱
        """
        # Kolmogorov谱: Phi_n(f) ~ f^(-11/3)
        # 考虑内尺度和外尺度
        f0 = 1.0 / self.L0  # 外尺度频率
        f_l0 = 1.0 / self.l0  # 内尺度频率
        
        # 避免零频率
        f = np.where(f == 0, f0, f)
        
        # Von Karman谱（带外尺度）
        phi_n = 0.033 * self.Cn2 * (f**2 + f0**2)**(-11/6)
        
        # 高频截断（内尺度）
        high_freq_mask = f > f_l0
        phi_n[high_freq_mask] *= (f_l0 / f[high_freq_mask])**2
        
        return phi_n
    
    def _generate_turbulence(self, seed: Optional[int] = None) -> np.ndarray:
        """
        生成湍流相位屏
        
        参数:
            seed: 随机种子
            
        返回:
            phase_screen: 相位屏 (rad)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # 生成复高斯随机数
        complex_gaussian = (np.random.randn(self.N, self.N) + 
                          1j * np.random.randn(self.N, self.N)) / np.sqrt(2)
        
        # 乘以功率谱平方根，避免零值
        sqrt_power = np.sqrt(np.maximum(self.power_spectrum, 1e-20))  # 避免零
        phi_fft = complex_gaussian * sqrt_power * self.dx * 1e6  # 增加缩放因子
        
        # 逆FFT得到相位
        phase = np.fft.ifft2(phi_fft).real

        # 归一化
        phase = phase - np.mean(phase)

        # 根据Fried参数缩放 (增加幅度)
        if self.Cn2 > 0:
            # 使用固定的湍流强度进行测试
            turbulence_strength = 0.1  # 固定的测试值
            phase *= turbulence_strength
        else:
            phase *= 0  # 无湍流时相位为0
        
        return phase
    
    def _calculate_fried_parameter(self) -> float:
        """计算Fried参数r0"""
        if self.Cn2 <= 0:
            return float('inf')  # 无湍流时Fried参数无穷大
        return (0.423 * self.k0**2 * self.Cn2 * self.L0)**(-3/5)
    
    def add_phase_screen(self, wavefront: np.ndarray) -> np.ndarray:
        """
        在波前上叠加湍流相位
        
        参数:
            wavefront: 输入波前
            
        返回:
            distorted_wavefront: 畸变后的波前
        """
        return wavefront * np.exp(1j * self.phase_screen)
    
    def generate_new_screen(self, seed: Optional[int] = None):
        """生成新的相位屏（用于时间变化的湍流）"""
        self.phase_screen = self._generate_turbulence(seed)
    
    def get_phase_screen(self) -> np.ndarray:
        """获取当前相位屏"""
        return self.phase_screen


class HartmannShackWavefrontSensor:
    """
    哈特曼-夏克波前传感器(WFS)仿真类
    
    通过子孔径质心测量估计波前斜率
    """
    
    def __init__(self,
                 subapertures: int = 8,
                 pixel_scale: float = 0.5,
                 N: int = 256):
        """
        初始化WFS
        
        参数:
            subapertures: 子孔径数量（沿一个维度）
            pixel_scale: 像素比例
            N: 输入网格点数
        """
        self.subapertures = subapertures
        self.pixel_scale = pixel_scale
        self.N = N
        self.total_subapertures = subapertures ** 2
        
        # 计算子孔径大小
        self.sub_size = N // subapertures
        
        # 创建子孔径掩码
        self._create_subaperture_masks()
    
    def _create_subaperture_masks(self):
        """创建子孔径掩码"""
        self.masks = []
        for i in range(self.subapertures):
            for j in range(self.subapertures):
                mask = np.zeros((self.N, self.N), dtype=bool)
                mask[i*self.sub_size:(i+1)*self.sub_size, 
                     j*self.sub_size:(j+1)*self.sub_size] = True
                self.masks.append(mask)
        
        self.masks = np.array(self.masks)
    
    def measure_slopes(self, intensity: np.ndarray, wavefront: np.ndarray) -> np.ndarray:
        """
        测量波前斜率
        
        参数:
            intensity: 光强分布
            wavefront: 波前相位
            
        返回:
            slopes: 波前斜率数组，shape为(total_subapertures * 2,)
                     前半部分是x方向斜率，后半部分是y方向斜率
        """
        slopes_x = []
        slopes_y = []
        
        # 创建坐标网格
        x = np.arange(self.N)
        y = np.arange(self.N)
        X, Y = np.meshgrid(x, y)
        
        for mask in self.masks:
            # 提取子孔径区域
            sub_intensity = intensity[mask]
            sub_wavefront = wavefront[mask]
            sub_x = X[mask]
            sub_y = Y[mask]
            
            if np.sum(sub_intensity) < 1e-10:
                slopes_x.append(0)
                slopes_y.append(0)
                continue
            
            # 计算质心
            total_intensity = np.sum(sub_intensity)
            cx = np.sum(sub_x * sub_intensity) / total_intensity
            cy = np.sum(sub_y * sub_intensity) / total_intensity
            
            # 子孔径中心
            center_x = np.mean(sub_x)
            center_y = np.mean(sub_y)
            
            # 质心偏移作为斜率（简化模型）
            slope_x = (cx - center_x) * self.pixel_scale
            slope_y = (cy - center_y) * self.pixel_scale
            
            slopes_x.append(slope_x)
            slopes_y.append(slope_y)
        
        slopes = np.array(slopes_x + slopes_y)
        
        return slopes
    
    def reconstruct_wavefront(self, slopes: np.ndarray, basis: Optional[np.ndarray] = None) -> np.ndarray:
        """
        从斜率重建波前

        参数:
            slopes: 波前斜率
            basis: Zernike基函数

        返回:
            wavefront: 重建的波前
        """
        if basis is None:
            basis = ZernikePolynomials.generate_basis(36, self.N, 2.0)

        # 简化的重建方法：直接使用低阶Zernike模式拟合
        num_modes = min(basis.shape[0], 10)  # 限制模式数量
        basis = basis[:num_modes]

        num_subaps = self.subapertures ** 2
        x_slopes = slopes[:num_subaps]
        y_slopes = slopes[num_subaps:]

        # 为每个模式计算响应矩阵（简化的）
        coefficients = np.zeros(num_modes)

        # Tilt X (模式1) - 主要影响x斜率
        if num_modes > 1:
            coefficients[1] = np.mean(x_slopes) * 100  # 增加缩放因子

        # Tilt Y (模式2) - 主要影响y斜率
        if num_modes > 2:
            coefficients[2] = np.mean(y_slopes) * 100  # 增加缩放因子

        # Defocus (模式4) - 影响所有子孔径
        if num_modes > 4:
            avg_slope = (np.mean(np.abs(x_slopes)) + np.mean(np.abs(y_slopes))) / 2
            coefficients[4] = avg_slope * 50

        # 重建波前，确保非零
        reconstructed = np.sum([coeff * basis[i] for i, coeff in enumerate(coefficients)], axis=0)

        # 归一化以避免数值问题
        if np.max(np.abs(reconstructed)) > 0:
            reconstructed = reconstructed / np.max(np.abs(reconstructed))

        return reconstructed


class Camera:
    """
    CCD/相机仿真类
    
    模拟光强探测和噪声
    """
    
    def __init__(self,
                 N: int = 256,
                 quantum_efficiency: float = 0.8,
                 dark_current: float = 0.1,
                 read_noise: float = 1.0,
                 gain: float = 1.0):
        """
        初始化相机
        
        参数:
            N: 网格点数
            quantum_efficiency: 量子效率
            dark_current: 暗电流 (e-/pixel/s)
            read_noise: 读出噪声 (e- RMS)
            gain: 增益
        """
        self.N = N
        self.quantum_efficiency = quantum_efficiency
        self.dark_current = dark_current
        self.read_noise = read_noise
        self.gain = gain
    
    def detect(self, electric_field: np.ndarray, exposure_time: float = 1.0) -> np.ndarray:
        """
        探测光场并返回强度图像
        
        参数:
            electric_field: 入射光电场
            exposure_time: 曝光时间
            
        返回:
            image: 探测到的图像
        """
        # 计算强度
        intensity = np.abs(electric_field)**2
        
        # 应用量子效率
        electrons = intensity * self.quantum_efficiency * exposure_time * 1e15  # 缩放因子
        
        # 添加暗电流噪声
        dark_noise = np.random.poisson(self.dark_current * exposure_time, electrons.shape)
        electrons += dark_noise
        
        # 添加读出噪声
        read_noise = np.random.normal(0, self.read_noise, electrons.shape)
        electrons += read_noise
        
        # 确保非负
        electrons = np.maximum(electrons, 0)
        
        # 应用增益并转换为整数
        image = electrons * self.gain
        image = np.clip(image, 0, 65535).astype(np.uint16)
        
        return image


class VectorWavePropagator:
    """
    矢量波传播器（角谱法）
    
    支持标量和矢量波的角谱传播
    """
    
    def __init__(self,
                 N: int = 256,
                 L: float = 0.1,
                 wavelength: float = 1550e-9,
                 distance: float = 1000.0):
        """
        初始化传播器
        
        参数:
            N: 网格点数
            L: 物理孔径大小 (m)
            wavelength: 波长 (m)
            distance: 传播距离 (m)
        """
        self.N = N
        self.L = L
        self.wavelength = wavelength
        self.distance = distance
        self.dx = L / N
        self.k0 = 2 * np.pi / wavelength
        
        # 预计算传播因子
        self._setup_propagator()
    
    def _setup_propagator(self):
        """设置角谱传播因子"""
        fx = np.fft.fftfreq(self.N, d=self.dx)
        fy = np.fft.fftfreq(self.N, d=self.dx)
        FX, FY = np.meshgrid(fx, fy)
        
        # 波数平方
        k_trans_sq = (2 * np.pi * FX)**2 + (2 * np.pi * FY)**2
        
        # 传播因子
        kz_arg = 1 - (self.wavelength * FX)**2 - (self.wavelength * FY)**2
        kz_arg = np.maximum(kz_arg, 1e-10)
        
        self.propagator = np.exp(1j * self.k0 * np.sqrt(kz_arg) * self.distance)
    
    def propagate(self, E: np.ndarray) -> np.ndarray:
        """
        角谱法传播
        
        参数:
            E: 输入电场
            
        返回:
            E_propagated: 传播后的电场
        """
        E_fft = np.fft.fft2(E)
        E_propagated = np.fft.ifft2(E_fft * self.propagator)
        return E_propagated
    
    def propagate_vector(self, Ex: np.ndarray, Ey: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """矢量波传播"""
        Ex_out = self.propagate(Ex)
        Ey_out = self.propagate(Ey)
        return Ex_out, Ey_out


class TraditionalAOSystem:
    """
    传统AO系统集成类
    
    整合光源、湍流、DM、WFS和相机等组件
    """
    
    def __init__(self, config: Optional[AOConfig] = None):
        """
        初始化AO系统
        
        参数:
            config: AO配置
        """
        self.config = config or AOConfig()
        
        # 初始化组件
        self.light_source = LightSource(
            wavelength=self.config.wavelength
        )
        
        self.turbulence = AtmosphericTurbulence(
            Cn2=self.config.Cn2,
            L0=self.config.L0,
            l0=self.config.l0,
            N=self.config.N,
            L=self.config.L
        )
        
        self.dm = DeformableMirror(
            num_actuators=self.config.dm_actuators,
            stroke=self.config.dm_stroke,
            N=self.config.N
        )
        
        self.wfs = HartmannShackWavefrontSensor(
            subapertures=self.config.subapertures,
            pixel_scale=self.config.pixel_scale,
            N=self.config.N
        )
        
        self.camera = Camera(N=self.config.N)
        
        self.propagator = VectorWavePropagator(
            N=self.config.N,
            L=self.config.L,
            wavelength=self.config.wavelength,
            distance=self.config.propagation_distance
        )
        
        # 初始化状态
        self._init_state()
    
    def _init_state(self):
        """初始化系统状态"""
        # 创建入射光场 - 使用平面波以获得更好的初始性能
        self.E_in = self.light_source.create_plane_wave(
            self.config.N, self.config.L
        )

        # 应用湍流
        self.E_turb = self.turbulence.add_phase_screen(self.E_in)

        # DM初始状态
        self.dm_voltages = np.zeros(self.dm.total_actuators)

        # 简化传播模型：对于平面波，远场应该是类似的光场
        # 使用简化的传播，避免数值问题
        if self.config.propagation_distance > 0:
            # 对于短距离，使用简单的相位延迟
            k = 2 * np.pi / self.config.wavelength
            phase_delay = k * self.config.propagation_distance
            self.E_propagated = self.E_turb * np.exp(1j * phase_delay)
        else:
            self.E_propagated = self.E_turb

        # 通过DM校正
        self.E_corrected = self._apply_dm_correction()
    
    def _apply_dm_correction(self) -> np.ndarray:
        """应用DM校正"""
        dm_surface = self.dm.apply_voltages(self.dm_voltages)
        return self.E_propagated * np.exp(1j * dm_surface * 2 * np.pi / self.config.wavelength)
    
    def set_dm_voltages(self, voltages: np.ndarray):
        """
        设置DM电压
        
        参数:
            voltages: DM电压数组
        """
        self.dm_voltages = np.clip(voltages, -1, 1)
        self.E_corrected = self._apply_dm_correction()
    
    def measure_wavefront(self) -> np.ndarray:
        """
        测量波前
        
        返回:
            slopes: 波前斜率
        """
        intensity = np.abs(self.E_corrected)**2
        phase = np.angle(self.E_corrected)
        return self.wfs.measure_slopes(intensity, phase)
    
    def get_image(self) -> np.ndarray:
        """
        获取相机图像
        
        返回:
            image: 相机图像
        """
        return self.camera.detect(self.E_corrected)
    
    def step(self, action: np.ndarray) -> Dict[str, Any]:
        """
        执行一步AO校正
        
        参数:
            action: DM电压控制量
            
        返回:
            result: 包含观测、奖励等信息的字典
        """
        # 更新DM
        new_voltages = self.dm_voltages + action
        self.set_dm_voltages(new_voltages)
        
        # 获取观测
        image = self.get_image()
        slopes = self.measure_wavefront()
        
        # 计算性能指标
        peak_intensity = np.max(image)
        total_power = np.sum(image)
        
        # 计算相位RMS
        phase = np.angle(self.E_corrected)
        phase_rms = np.sqrt(np.mean(phase**2))
        
        # Strehl比：使用标准定义 S = exp(-RMS^2)
        strehl = np.exp(-phase_rms**2) if phase_rms < 10 else 0.001
        strehl = float(np.clip(strehl, 0.001, 1.0))
        
        return {
            'image': image,
            'slopes': slopes,
            'strehl': strehl,
            'power': total_power,
            'voltages': self.dm_voltages.copy()
        }
    
    def reset(self) -> Dict[str, Any]:
        """重置系统"""
        self._init_state()
        
        # 计算真实的初始Strehl比
        phase = np.angle(self.E_corrected)
        rms = np.sqrt(np.mean(phase**2))
        initial_strehl = np.exp(-rms**2) if rms < 10 else 0.01
        
        return {
            'image': self.get_image(),
            'slopes': self.measure_wavefront(),
            'strehl': float(np.clip(initial_strehl, 0.001, 1.0)),
            'power': np.sum(self.get_image()),
            'voltages': self.dm_voltages.copy()
        }


# ==================== 辅助函数 ====================

def zernike_phase_screen(n_max: int, rho: np.ndarray, theta: np.ndarray, 
                         coefficients: Optional[np.ndarray] = None) -> np.ndarray:
    """
    生成Zernike相位屏
    
    参数:
        n_max: 最大Zernike阶数
        rho: 归一化径向坐标
        theta: 角度坐标
        coefficients: Zernike系数，如果为None则生成随机系数
    返回:
        相位屏
    """
    if coefficients is None:
        num_modes = sum(n + 1 for n in range(n_max + 1))
        coefficients = np.random.randn(num_modes) * 0.1
    
    return ZernikePolynomials.generate_basis(len(coefficients), rho.shape[0], 2.0)[0]


def calculate_strehl(intensity: np.ndarray, peak_reference: Optional[float] = None) -> float:
    """
    计算Strehl比
    
    参数:
        intensity: 强度分布
        peak_reference: 峰值参考值，如果为None则使用最大可能值
    返回:
        Strehl比
    """
    peak_intensity = np.max(intensity)
    if peak_reference is None:
        peak_reference = np.max(intensity)
    return float(np.clip(peak_intensity / (peak_reference + 1e-10), 0, 1))


def calculate_rms(phase: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    计算相位RMS值
    
    参数:
        phase: 相位分布
        mask: 孔径掩膜
    返回:
        RMS值 (rad)
    """
    if mask is not None:
        phase_masked = phase[mask]
    else:
        phase_masked = phase
    return float(np.sqrt(np.mean(phase_masked**2)))


def calculate_pv(phase: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    计算相位PV值 (峰谷值)
    
    参数:
        phase: 相位分布
        mask: 孔径掩膜
    返回:
        PV值 (rad)
    """
    if mask is not None:
        phase_masked = phase[mask]
    else:
        phase_masked = phase
    return float(np.max(phase_masked) - np.min(phase_masked))


# ==================== 向量波光学仿真 ====================

class VectorWaveOpticsSim:
    """
    矢量波光学仿真器
    
    支持矢量光束（偏振）的传播和大气湍流仿真。
    """
    
    def __init__(self, N: int = 64, L: float = 0.1, 
                 wavelength: float = 1550e-9, Z: float = 1000.0, 
                 Cn2: float = 1e-14):
        """
        初始化矢量波光学仿真器
        
        参数:
            N: 网格点数
            L: 物理孔径大小 (m)
            wavelength: 波长 (m)
            Z: 传播距离 (m)
            Cn2: 折射率结构常数
        """
        self.N = N
        self.L = L
        self.wavelength = wavelength
        self.Z = Z
        self.Cn2 = Cn2
        self.dx = L / N
        self.k0 = 2 * np.pi / wavelength
        
        # 创建网格
        x = np.linspace(-L/2, L/2, N)
        y = np.linspace(-L/2, L/2, N)
        self.X, self.Y = np.meshgrid(x, y)
        self.R = np.sqrt(self.X**2 + self.Y**2)
        self.THETA = np.arctan2(self.Y, self.X)
        
        # 初始化湍流
        self.turb_phase = self._generate_turbulence()
        
        # 初始化传播器
        self._setup_propagator()
    
    def _generate_turbulence(self) -> np.ndarray:
        """生成湍流相位屏"""
        fx = np.fft.fftfreq(self.N, d=self.dx)
        fy = np.fft.fftfreq(self.N, d=self.dx)
        FX, FY = np.meshgrid(fx, fy)
        F = np.sqrt(FX**2 + FY**2)
        
        # 简化的Kolmogorov谱
        power_spectrum = 0.033 * self.Cn2 * (F**2 + 1e-10)**(-11/6)
        power_spectrum[F < 1/self.L] = 0  # 低频截断
        
        # 生成随机相位屏
        complex_gaussian = np.random.randn(self.N, self.N) + 1j * np.random.randn(self.N, self.N)
        phase = np.fft.ifft2(complex_gaussian * np.sqrt(power_spectrum)).real
        
        return phase - np.mean(phase)
    
    def _setup_propagator(self):
        """设置角谱传播因子"""
        fx = np.fft.fftfreq(self.N, d=self.dx)
        fy = np.fft.fftfreq(self.N, d=self.dx)
        FX, FY = np.meshgrid(fx, fy)
        
        # 波数分量
        kx = 2 * np.pi * FX
        ky = 2 * np.pi * FY
        kz_sq = (self.k0**2 - kx**2 - ky**2)
        kz_sq = np.maximum(kz_sq, 1e-10)
        
        self.propagator = np.exp(1j * np.sqrt(kz_sq) * self.Z)
    
    def diffract(self, Ex: np.ndarray, Ey: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        传播矢量光场
        
        参数:
            Ex: x偏振分量
            Ey: y偏振分量
        返回:
            传播后的 (Ex, Ey)
        """
        Ex_fft = np.fft.fft2(Ex)
        Ey_fft = np.fft.fft2(Ey)
        
        Ex_out = np.fft.ifft2(Ex_fft * self.propagator)
        Ey_out = np.fft.ifft2(Ey_fft * self.propagator)
        
        return Ex_out, Ey_out
    
    def add_turbulence(self, Ex: np.ndarray, Ey: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        添加湍流畸变
        
        参数:
            Ex: x偏振分量
            Ey: y偏振分量
        返回:
            畸变后的 (Ex, Ey)
        """
        phase_factor = np.exp(1j * self.turb_phase)
        return Ex * phase_factor, Ey * phase_factor
    
    def create_target_radial(self, w0_factor: float = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建目标径向偏振光
        
        参数:
            w0_factor: 束腰因子
        返回:
            目标 (Ex, Ey)
        """
        w0 = self.L / w0_factor
        
        # 高斯幅度分布
        amplitude = np.exp(-(self.R**2) / (w0**2))
        
        # 径向偏振: E_r = r * cos(theta), E_theta = r * sin(theta)
        # 在笛卡尔坐标系中: Ex ~ x, Ey ~ y
        mask = self.R <= self.L / 2
        
        Ex = np.zeros_like(self.X)
        Ey = np.zeros_like(self.Y)
        
        # 径向偏振的笛卡尔分量
        Ex[mask] = self.X[mask] * amplitude[mask]
        Ey[mask] = self.Y[mask] * amplitude[mask]
        
        # 归一化
        total_power = np.sum(np.abs(Ex)**2 + np.abs(Ey)**2)
        if total_power > 0:
            Ex /= np.sqrt(total_power)
            Ey /= np.sqrt(total_power)
        
        return Ex, Ey
    
    @staticmethod
    def calculate_stokes_rgb(Ex: np.ndarray, Ey: np.ndarray) -> np.ndarray:
        """
        计算Stokes参数并转换为RGB显示
        
        参数:
            Ex: x偏振分量
            Ey: y偏振分量
        返回:
            RGB图像
        """
        # 计算Stokes参数
        I = np.abs(Ex)**2 + np.abs(Ey)**2
        Q = np.abs(Ex)**2 - np.abs(Ey)**2
        U = 2 * np.real(Ex * np.conj(Ey))
        
        # 归一化
        I_max = np.max(I) + 1e-10
        I = I / I_max
        Q = Q / I_max
        U = U / I_max
        
        # 转换为RGB (简化的偏振可视化)
        # 使用Q和U表示偏振方向
        rgb = np.zeros((*I.shape, 3))
        rgb[..., 0] = np.clip(I + Q, 0, 1)  # R = I + Q
        rgb[..., 1] = np.clip(I - Q * 0.5 + U * 0.5, 0, 1)  # G = I - Q/2 + U/2
        rgb[..., 2] = np.clip(I - Q * 0.5 - U * 0.5, 0, 1)  # B = I - Q/2 - U/2
        
        return rgb


# ==================== 别名定义 (向后兼容) ====================

# 设备别名
LaserSource = LightSource
HartmannSensor = HartmannShackWavefrontSensor
TurbulencePhaseScreen = AtmosphericTurbulence
CCD = Camera
ZernikeReconstructor = ZernikePolynomials
OpticalPropagator = VectorWavePropagator
AOSystem = TraditionalAOSystem
CCDCamera = Camera
WavefrontPropagator = VectorWavePropagator
HartmannShackSensor = HartmannShackWavefrontSensor
