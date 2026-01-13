"""
传统AO系统设备仿真模块

包含光源、变形镜(DM)、哈特曼传感器(WFS)、大气湍流相位屏等设备的仿真实现。
基于Zernike多项式和角谱传播法进行物理仿真。
"""
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from functools import cached_property
import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.ndimage import zoom

# 全局常量定义
PI = np.pi
N_expansion = 3  # 扩展倍数
FTT_expande_time = 1  # FFT扩展时间

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


class ZernikePolynomials:
    """
    Zernike多项式计算类
    用于波前分析和光学像差建模
    """
    
    # Zernike模式名称映射
    ZERNIKE_NAMES = {
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

    @staticmethod
    def zernike_name(n: int, m: int) -> str:
        """
        获取Zernike模式名称
        
        参数:
            n: 径向阶数
            m: 角向阶数
            
        返回:
            名称字符串
        """
        return ZernikePolynomials.ZERNIKE_NAMES.get((n, m), f"Z_{n}_{m}")

    @staticmethod
    def radial_polynomial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
        """
        计算Zernike径向多项式
        
        参数:
            n: 径向阶数
            m: 角向阶数
            rho: 归一化径向坐标 (0-1)
            
        返回:
            径向多项式值
        """
        if (n - abs(m)) % 2 != 0:
            return np.zeros_like(rho)
        
        m = abs(m)
        R = np.zeros_like(rho, dtype=float)
        
        for k in range((n - m) // 2 + 1):
            numerator = (-1) ** k * np.math.factorial(n - k)
            denominator = (
                np.math.factorial(k) *
                np.math.factorial((n + m) // 2 - k) *
                np.math.factorial((n - m) // 2 - k)
            )
            coefficient = numerator / denominator
            R += coefficient * (rho ** (n - 2 * k))
        
        return R

    @staticmethod
    def generate_basis(num_modes: int, N: int, L: float = 1.0) -> np.ndarray:
        """
        生成Zernike基函数
        
        参数:
            num_modes: 模式数量
            N: 网格点数
            L: 孔径大小
            
        返回:
            基函数数组，shape为(num_modes, N, N)
        """
        # 计算径向阶数和角向阶数
        modes = []
        n = 0
        while len(modes) < num_modes:
            for m in range(-n, n + 1, 2):
                modes.append((n, m))
                if len(modes) >= num_modes:
                    break
            n += 1
        
        basis = np.zeros((num_modes, N, N))
        
        x = np.linspace(-L, L, N)
        y = np.linspace(-L, L, N)
        X, Y = np.meshgrid(x, y)
        rho = np.sqrt(X**2 + Y**2) / L
        theta = np.arctan2(Y, X)
        
        # 创建圆形掩码
        mask = rho <= 1.0
        
        for i, (n, m) in enumerate(modes[:num_modes]):
            if i >= num_modes:
                break
            R = ZernikePolynomials.radial_polynomial(n, m, rho)
            
            if m > 0:
                Z = R * np.cos(m * theta)
            elif m < 0:
                Z = R * np.sin(-m * theta)
            else:
                Z = R
            
            # 应用圆形掩码并归一化
            Z = Z * mask
            norm = np.sqrt(np.sum(Z**2))
            if norm > 0:
                Z = Z / norm
            
            basis[i] = Z
        
        return basis


class DeformableMirror:
    """
    变形镜(DM)仿真类
    
    支持基于影响函数和Zernike模式的DM仿真
    """
    
    def __init__(self,
                 act_positions: np.ndarray,
                 stroke: float = 5e-6,
                 N: int = 256,
                 actuator_coupling: float = 0.3):
        """
        初始化DM
        
        参数:
            num_actuators: 致动器数量（沿一个维度）
            stroke: 最大行程 (m)
            act_positions: 致动器位置，shape为(num_actuators, 2)
            N: 输出网格点数
            actuator_coupling: 致动器间的耦合系数
        """
        # 输入验证

        if stroke <= 0:
            raise ValueError("行程必须为正数")
        if N <= 0:
            raise ValueError("网格点数必须为正数")
        if actuator_coupling < 0 or actuator_coupling > 1:
            raise ValueError("致动器耦合系数必须在[0, 1]范围内")
        
        self.stroke = stroke
        self.N = N
        self.actuator_coupling = actuator_coupling
        
        self.act_positions = act_positions
        self.num_actuators = len(act_positions)
        self.influence_matrix = self._create_influence_functions()

    @classmethod
    def create_from_position_file(cls, filename: str, N: int=256, stroke: float = 5e-6, actuator_coupling: float = 0.3) -> 'DeformableMirror':
        """从文本文件创建DM实例"""
        act_positions = np.loadtxt(filename, delimiter=',')
        # TODO 位置归一化到[-0.9, 0.9]
        return cls(act_positions=act_positions, N=N, stroke=stroke, actuator_coupling=actuator_coupling)
    
    @classmethod
    def create_from_grid(cls, num_actuators_x: int, num_actuators_y: int, N: int=256, stroke: float = 5e-6, actuator_coupling: float = 0.3) -> 'DeformableMirror':
        """创建网格分布的DM实例"""
        act_positions = cls._create_grid_actuators(num_actuators_x, num_actuators_y)
        return cls(act_positions=act_positions, N=N, stroke=stroke, actuator_coupling=actuator_coupling)
    
    @classmethod
    def create_from_circle(cls, num_actuators: int, N: int=256, stroke: float = 5e-6, actuator_coupling: float = 0.3) -> 'DeformableMirror':
        """创建环形分布的DM实例"""
        act_positions = cls._create_circle_actuators(num_actuators)
        return cls(act_positions=act_positions, N=N, stroke=stroke, actuator_coupling=actuator_coupling)

    @staticmethod
    def _create_grid_actuators(num_actuators_x: int, num_actuators_y: int) -> np.ndarray:
        x = np.linspace(-0.9, 0.9, num_actuators_x)
        y = np.linspace(-0.9, 0.9, num_actuators_y)
        act_x, act_y = np.meshgrid(x, y)
        return np.column_stack([act_x.flatten(), act_y.flatten()])

    @staticmethod
    def _create_circle_actuators(num_actuators: int) -> np.ndarray:
        """创建环形分布的致动器位置"""
        radius = 0.9
        theta = np.linspace(0, 2*np.pi, num_actuators, endpoint=False)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        return np.column_stack([x, y])

    def _create_influence_functions(self) -> np.ndarray:
        N = self.N
        # 创建空间网格
        grid_x = np.linspace(-1, 1, N)
        grid_y = np.linspace(-1, 1, N)
        X, Y = np.meshgrid(grid_x, grid_y)
        
        # 基础高斯影响函数参数
        sigma_base = 0.8 / self.num_actuators
        influence_matrix = np.zeros((self.num_actuators, N, N))
        
        for i, (ax, ay) in enumerate(self.act_positions):
            R = np.sqrt((X - ax)**2 + (Y - ay)**2)
            
            # 主要影响函数
            primary_influence = np.exp(-(R**2) / (2 * sigma_base**2))
            
            # 耦合影响函数（相邻致动器的影响）
            coupling_influence = np.zeros_like(primary_influence)
            
            # 计算与相邻致动器的距离
            for j, (bx, by) in enumerate(self.act_positions):
                if i != j:
                    dist = np.sqrt((ax - bx)**2 + (ay - by)**2)
                    if dist < 2.5 * (1.8 / self.num_actuators):  # 只考虑近邻
                        R_cross = np.sqrt((X - bx)**2 + (Y - by)**2)
                        coupling_influence += self.actuator_coupling * \
                                             np.exp(-(R_cross**2) / (2 * (sigma_base * 1.2)**2))
            
            # 合并主要和耦合影响函数
            influence_matrix[i] = primary_influence + coupling_influence
        
        return influence_matrix
    
    def apply_voltages(self, voltages: np.ndarray) -> np.ndarray:
        """
        根据电压计算DM面型
        
        参数:
            voltages: 电压数组，shape为(num_actuators^2,)
            
        返回:
            surface: DM面型 (m)
        """
        # 输入验证
        if len(voltages) != self.num_actuators:
            raise ValueError(f"电压数组长度应为{self.num_actuators}，实际为{len(voltages)}")
        
        # 将电压缩放到行程范围，并使用tanh函数模拟非线性响应
        normalized_voltages = np.clip(voltages, -1, 1)
        # 使用tanh函数模拟非线性响应，使响应更真实
        nonlinear_response = np.tanh(normalized_voltages * 2) / np.tanh(2)
        scaled_voltages = nonlinear_response * self.stroke
        
        # 叠加影响函数
        surface = np.tensordot(scaled_voltages, self.influence_matrix, axes=1)
        
        return surface
    
    def get_surface_with_aperture(self, voltages: np.ndarray, aperture_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        获取带孔径掩模的DM表面
        
        参数:
            voltages: 电压数组
            aperture_mask: 孔径掩模，可选
            
        返回:
            masked_surface: 应用了孔径掩模的DM表面
        """
        surface = self.apply_voltages(voltages)
        
        if aperture_mask is not None:
            if aperture_mask.shape != surface.shape:
                raise ValueError(f"孔径掩模形状应为{surface.shape}，实际为{aperture_mask.shape}")
            return surface * aperture_mask
        else:
            # 默认使用圆形孔径
            x = np.linspace(-1, 1, self.N)
            y = np.linspace(-1, 1, self.N)
            X, Y = np.meshgrid(x, y)
            circular_mask = (X**2 + Y**2 <= 1.0)
            return surface * circular_mask
    
    def get_command_matrix(self, regularization: Optional[float] = None) -> np.ndarray:
        """
        获取命令矩阵（Zernike模式到电压的映射）
        
        参数:
            regularization: 正则化参数，如果为None则使用实例默认值
            
        返回:
            command_matrix: shape为(num_zernike, num_actuators^2)
        """
        # if regularization is None:
        #     regularization = self.regularization
            
        # # 简化的命令矩阵：使用Zernike模式作为期望面型
        # num_modes = min(36, self.total_actuators)
        # basis = ZernikePolynomials.generate_basis(num_modes, self.N, 2.0)
        
        # # 展平影响矩阵
        # inf_flat = self.influence_matrix.reshape(self.total_actuators, -1)
        # basis_flat = basis.reshape(num_modes, -1)
        
        # # 使用正则化最小二乘求解以提高数值稳定性
        # # 构建正规方程: (A^T*A + λ*I)*x = A^T*b
        # A = inf_flat.T  # A shape: (N*N, total_actuators)
        # B = basis_flat.T  # B shape: (N*N, num_modes)
        
        # # 计算 A^T*A + λ*I
        # AtA_reg = A.T @ A + regularization * np.eye(A.shape[1])
        # AtB = A.T @ B
        
        # # 求解命令矩阵
        # command_matrix = np.linalg.solve(AtA_reg, AtB)

        # return command_matrix.T
        # TODO: 实现用最小二乘拟合模式法响应矩阵
        raise NotImplementedError("获取命令矩阵未实现")

    def get_surface_rms(self, voltages: np.ndarray) -> float:
        """
        计算DM表面的RMS值
        
        参数:
            voltages: 电压数组
            
        返回:
            rms: 表面形变的RMS值
        """
        surface = self.apply_voltages(voltages)
        return np.sqrt(np.mean(surface**2))
    
    def get_surface_pv(self, voltages: np.ndarray) -> float:
        """
        计算DM表面的PV值（峰谷值）
        
        参数:
            voltages: 电压数组
            
        返回:
            pv: 表面形变的PV值
        """
        surface = self.apply_voltages(voltages)
        return np.max(surface) - np.min(surface)


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
        phi_fft = complex_gaussian * sqrt_power * self.dx * 1e6 * 1e10  # 增加缩放因子

        # 逆FFT得到相位
        phase = np.fft.ifft2(phi_fft).real

        # 归一化
        phase = phase - np.mean(phase)

        # 根据Fried参数缩放
        if self.Cn2 > 0:
            # 使用固定的湍流强度进行测试
            turbulence_strength = 20.0  # 调整为合适的测试值
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
    增强型哈特曼-夏克波前传感器(WFS)仿真类
    
    通过子孔径质心测量估计波前斜率，支持角谱传播
    """
    
    def __init__(self,
                 subapertures: int = 8,
                 pixel_scale: float = 0.5,
                 N: int = 256,
                 focal_length: float = 100e-3,  # 焦距 (m)
                 lenslet_pitch: float = 150e-6,  # 微透镜间距 (m)
                 wavelength: float = 1550e-9):   # 波长 (m)
        """
        初始化WFS
        
        参数:
            subapertures: 子孔径数量（沿一个维度）
            pixel_scale: 像素比例
            N: 输入网格点数
            focal_length: 微透镜焦距 (m)
            lenslet_pitch: 微透镜间距 (m)
            wavelength: 波长 (m)
        """
        self.subapertures = subapertures
        self.pixel_scale = pixel_scale
        self.N = N
        self.total_subapertures = subapertures ** 2
        self.focal_length = focal_length
        self.lenslet_pitch = lenslet_pitch
        self.wavelength = wavelength
        
        # 计算子孔径大小
        self.sub_size = N // subapertures
        
        # 预计算角谱传播参数
        self._setup_angular_spectrum_propagator()
        
        # 创建子孔径掩码
        self._create_subaperture_masks()
    
    def _setup_angular_spectrum_propagator(self):
        """设置角谱传播参数"""
        # 计算采样间隔
        self.dx = self.lenslet_pitch / self.sub_size  # 假设每个子孔径被均匀划分
        self.k0 = 2 * np.pi / self.wavelength
        
        # 预计算频率网格
        fx = np.fft.fftfreq(self.sub_size, d=self.dx)
        fy = np.fft.fftfreq(self.sub_size, d=self.dx)
        FX, FY = np.meshgrid(fx, fy)
        
        # 传播因子（针对微透镜焦距）
        k_trans_sq = (2 * np.pi * FX)**2 + (2 * np.pi * FY)**2
        kz_arg = self.k0**2 - k_trans_sq
        kz_arg = np.maximum(kz_arg, 1e-10)  # 避免除零
        
        # 在焦平面处的传播因子
        self.propagator_focus = np.exp(1j * np.sqrt(kz_arg) * self.focal_length)
    
    def _create_subaperture_masks(self):
        """创建子孔径掩码"""
        self.masks = []
        for i in range(self.subapertures):
            for j in range(self.subapertures):
                mask = np.zeros((self.N, self.N), dtype=bool)
                start_row = i * self.sub_size
                end_row = min((i + 1) * self.sub_size, self.N)
                start_col = j * self.sub_size
                end_col = min((j + 1) * self.sub_size, self.N)
                mask[start_row:end_row, start_col:end_col] = True
                self.masks.append(mask)
        
        self.masks = np.array(self.masks)
    
    def measure_slopes_with_propagation(self, electric_field: np.ndarray) -> np.ndarray:
        """
        使用角谱传播测量波前斜率
        
        参数:
            electric_field: 输入电场 (复数)
            
        返回:
            slopes: 波前斜率数组
        """
        slopes_x = []
        slopes_y = []
        
        for idx, mask in enumerate(self.masks):
            # 提取子孔径区域
            # 找到非零元素的索引
            rows, cols = np.where(mask)
            if len(rows) == 0:
                slopes_x.append(0)
                slopes_y.append(0)
                continue
                
            # 计算子孔径边界
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()
            
            # 提取子孔径区域
            sub_field_full = electric_field[min_row:max_row+1, min_col:max_col+1]
            sub_field = sub_field_full
            # 调整到正确的子孔径大小
            if sub_field_full.shape[0] != self.sub_size or sub_field_full.shape[1] != self.sub_size:
                # 重新采样到正确大小
                from scipy.ndimage import zoom
                zoom_factor = (self.sub_size/sub_field_full.shape[0], 
                              self.sub_size/sub_field_full.shape[1])
                sub_field = zoom(sub_field_full, zoom_factor, order=1)

            # 应用角谱传播到焦平面
            field_fft = np.fft.fft2(sub_field)
            propagated_field = np.fft.ifft2(field_fft * self.propagator_focus)
            
            # 计算焦平面上的强度分布
            intensity = np.abs(propagated_field)**2
            
            # 计算质心位置
            sub_x = np.arange(self.sub_size)
            sub_y = np.arange(self.sub_size)
            Sub_X, Sub_Y = np.meshgrid(sub_x, sub_y)
            
            if np.sum(intensity) < 1e-10:
                slopes_x.append(0)
                slopes_y.append(0)
                continue
            
            # 计算质心
            total_intensity = np.sum(intensity)
            cx = np.sum(Sub_X * intensity) / total_intensity
            cy = np.sum(Sub_Y * intensity) / total_intensity
            
            # 子孔径中心
            center_x = self.sub_size / 2
            center_y = self.sub_size / 2
            
            # 质心偏移作为斜率（考虑像素比例和物理尺寸）
            slope_x = (cx - center_x) * self.pixel_scale * (self.lenslet_pitch / self.sub_size) / self.focal_length
            slope_y = (cy - center_y) * self.pixel_scale * (self.lenslet_pitch / self.sub_size) / self.focal_length
            
            slopes_x.append(slope_x)
            slopes_y.append(slope_y)
        
        slopes = np.array(slopes_x + slopes_y)
        return slopes
    
    def reconstruct_wavefront(self, 
                              slopes: np.ndarray, 
                              zernike_modes: Optional[np.ndarray] = None,
                              piston: bool = True,
                              tile: bool = True) -> np.ndarray:
        """
        从斜率重建波前

        参数:
            slopes: 波前斜率
            zernike_modes: Zernike基函数列表

        返回:
            wavefront: 重建的波前
        """
        # TODO: 实现波前重建
        # 获取x和y方向的斜率
        num_subaps = self.subapertures ** 2
        x_slopes = slopes[:num_subaps]
        y_slopes = slopes[num_subaps:]

        # 创建重建网格
        reconstruction = np.zeros((self.N, self.N))

        # 简单的积分重建方法
        sub_size_recon = self.N // self.subapertures

        for i in range(self.subapertures):
            for j in range(self.subapertures):
                idx = i * self.subapertures + j
                if idx < len(x_slopes):
                    start_row = i * sub_size_recon
                    end_row = min((i + 1) * sub_size_recon, self.N)
                    start_col = j * sub_size_recon
                    end_col = min((j + 1) * sub_size_recon, self.N)

                    if i == 0 and j == 0:
                        # 参考点设为0
                        reconstruction[start_row:end_row, start_col:end_col] = 0
                    elif i == 0:
                        # 从左侧积分y斜率
                        prev_col_start = (j-1) * sub_size_recon
                        prev_col_end = min(j * sub_size_recon, self.N)
                        reconstruction[start_row:end_row, start_col:end_col] = \
                            reconstruction[start_row:end_row, prev_col_start:prev_col_end] + \
                            y_slopes[idx] * 0.5
                    elif j == 0:
                        # 从上方积分x斜率
                        prev_row_start = (i-1) * sub_size_recon
                        prev_row_end = min(i * sub_size_recon, self.N)
                        reconstruction[start_row:end_row, start_col:end_col] = \
                            reconstruction[prev_row_start:prev_row_end, start_col:end_col] + \
                            x_slopes[idx] * 0.5
                    else:
                        # 从上方和左侧平均积分
                        prev_row_start = (i-1) * sub_size_recon
                        prev_row_end = min(i * sub_size_recon, self.N)
                        prev_col_start = (j-1) * sub_size_recon
                        prev_col_end = min(j * sub_size_recon, self.N)

                        rec_from_x = reconstruction[prev_row_start:prev_row_end, start_col:end_col] + \
                                   x_slopes[idx] * 0.5
                        rec_from_y = reconstruction[start_row:end_row, prev_col_start:prev_col_end] + \
                                   y_slopes[idx] * 0.5
                        reconstruction[start_row:end_row, start_col:end_col] = (rec_from_x + rec_from_y) / 2

        return reconstruction


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
        electrons = intensity * self.quantum_efficiency * exposure_time * 1e6  # 缩放因子
        
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


# 定义配置类
class OptConfig:
    """
    Optical configuration parameters
    """
    pol = 1, 1
    shift = 0, 0


class GlDim:
    """
    Grid dimensions
    """
    x = 1
    y = 1


@dataclass
class FDTDIntervalConfig:
    """
    FDTD output field density interval
    """
    x: int
    y: int
    z: int

    def __post_init__(self):
        self.x = int(max(self.x, 1))
        self.y = int(max(self.y, 1))
        self.z = int(max(self.z, 1))


FDTDInterval = {"x": 50, "y": 50, "z": 50}
N_expansion = 3  # 扩展倍数
FTT_expande_time = 1  # FFT扩展时间

@dataclass(frozen=True)
class OpticParameters:
    """
    Optical parameters for calculations
    """
    _num_elements_x: int  # Number of elements
    _num_elements_y: int  # Number of elements
    period: float  # Period
    wavelength: float  # Wavelength
    focal_length: float  # Focal length

    @property
    def num_elements_x(self):
        return int(self._num_elements_x)

    @property
    def num_elements_y(self):
        return int(self._num_elements_y)

    @property
    def k0(self):
        return 1 / self.wavelength

    @property
    def Px(self):
        return self.num_elements_x * self.period

    @property
    def Py(self):
        return self.num_elements_y * self.period

    @cached_property
    def str_x(self):
        return np.linspace(-self.Px / 2, self.Px / 2, self.num_elements_x)

    @cached_property
    def str_y(self):
        return np.linspace(-self.Py / 2, self.Py / 2, self.num_elements_y)
    @cached_property
    def meshed_X(self):
        X, _ = np.meshgrid(self.str_x, self.str_y)
        return X

    @cached_property
    def meshed_Y(self):
        _, Y = np.meshgrid(self.str_x, self.str_y)
        return Y

    @property
    def R(self):
        return min(self.Px, self.Py) / 2

    # 出射场密度
    @property
    def x_fdtd(self):
        x_count = np.ceil(self.Px / FDTDInterval["x"]).astype(int) // 2 * 2 + 1
        return np.linspace(-self.Px / 2, self.Px / 2, x_count)

    @property
    def y_fdtd(self):
        y_count = np.ceil(self.Py / FDTDInterval["y"]).astype(int) // 2 * 2 + 1
        return np.linspace(-self.Py / 2, self.Py / 2, y_count)

    @property
    def F(self):
        return self.focal_length / (2 * self.R)

    @property
    def NA(self):
        return np.sin(np.arctan(self.R / self.focal_length))


@dataclass
class ElectronicField:
    """
    Electronic field representation
    """
    Ex: np.ndarray
    Ey: np.ndarray
    Ez: np.ndarray

    @property
    def G(self):
        """
        Calculate the intensity of the electronic field
        """
        g = np.abs(self.Ex) ** 2 + np.abs(self.Ey) ** 2 + np.abs(self.Ez) ** 2
        g[g < 0] = 0
        return g


class VectorWavePropagator:
    """
    矢量波传播器
    
    支持多种电磁波传播算法：
    1. 角谱法 (propagate) - 传统的快速传播方法
    2. 光束变换法 (trans_beam) - 高精度传播与效率分析方法（推荐）
    
    提供统一接口unified_propagate用于选择最适合的传播方法
    
    示例:
        # 创建传播器实例
        propagator = VectorWavePropagator(N=256, L=0.01, wavelength=1550e-9)
        
        # 使用传统角谱法传播
        result = propagator.propagate(field)
        
        # 使用高精度光束变换法传播（需提供光学参数）
        params = OpticParameters(...)  # 需要配置光学参数
        result, efficiency = propagator.trans_beam(phase, params, mon_dist=10e-3)
        
        # 使用统一接口（默认使用光束变换法）
        result, efficiency = propagator.unified_propagate(field_or_phase, 
                                                        method="beam_transform",
                                                        params=params, 
                                                        mon_dist=10e-3)
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
    
    def propagate(self, E: np.ndarray, distance: float = None) -> np.ndarray:
        """
        角谱法传播 - 传统的快速传播方法
        
        参数:
            E: 输入电场
            distance: 传播距离（如果为None，则使用初始化时的距离）
            
        返回:
            E_propagated: 传播后的电场
        """
        if distance is not None and distance != self.distance:
            # 如果指定了不同的距离，临时更新传播器
            old_distance = self.distance
            self.distance = distance
            self._setup_propagator()  # 重新设置传播器
            result = np.fft.ifft2(np.fft.fft2(E) * self.propagator)
            # 恢复原始距离设置
            self.distance = old_distance
            self._setup_propagator()
        else:
            # 使用预设的传播器
            result = np.fft.ifft2(np.fft.fft2(E) * self.propagator)
        
        return result
    
    def propagate_vector(self, Ex: np.ndarray, Ey: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """矢量波传播"""
        Ex_out = self.propagate(Ex)
        Ey_out = self.propagate(Ey)
        return Ex_out, Ey_out

    def propagate_with_beam_transformation(
        self,
        init_phase: np.ndarray,
        params: OpticParameters,
        mon_dist: float,
    ) -> tuple[np.ndarray, float]:
        """
        使用光束变换方法进行传播并计算效率
        
        Args:
            init_phase: 初始相位
            params: 光学参数
            mon_dist: 监测距离
            
        Returns:
            tuple: 传播后的电场和效率
        """
        # 将输入相位转换为电场
        e_field = self.phase_to_electronic_field(init_phase, params.Px, params.Py)
        
        # 使用FFT执行传播
        x_coords = np.linspace(-params.Px/2, params.Px/2, init_phase.shape[1])
        y_coords = np.linspace(-params.Py/2, params.Py/2, init_phase.shape[0])
        
        Eo, jmX, jmY = self.perform_fft(
            e_field, x_coords, y_coords, params.k0, params.focal_length, mon_dist
        )
        
        # 计算效率
        intensity = np.abs(Eo.Ex)**2 + np.abs(Eo.Ey)**2 + np.abs(Eo.Ez)**2
        efficiency, (fl, fr) = self.calculate_efficiency(intensity, jmX[0, :], jmY[:, 0], params.R)
        
        return Eo, efficiency

    def unified_propagate(self, 
                         field_or_phase: np.ndarray, 
                         method: str = "beam_transform",  # 更改默认方法为beam_transform
                         **kwargs) -> tuple:
        """
        统一的传播接口，支持多种传播方法，默认使用光束变换法
        
        Args:
            field_or_phase: 输入电场或相位
            method: 传播方法 ("angular_spectrum", "beam_transform")
            **kwargs: 其他参数
            
        Returns:
            传播后的电场和效率元组
        """
        if method == "angular_spectrum":
            # 使用传统的角谱方法
            if kwargs.get('distance'):
                result = self.propagate(field_or_phase, distance=kwargs['distance'])
                # 对于角谱方法，返回结果和单位效率
                return result, 1.0
            else:
                result = self.propagate(field_or_phase)
                return result, 1.0
        elif method == "beam_transform" and 'params' in kwargs and 'mon_dist' in kwargs:
            # 使用光束变换方法（现在是默认方法）
            return self.trans_beam(
                field_or_phase, 
                kwargs['params'], 
                kwargs['mon_dist']
            )
        else:
            # 默认使用光束变换方法
            if 'params' in kwargs and 'mon_dist' in kwargs:
                return self.trans_beam(
                    field_or_phase, 
                    kwargs['params'], 
                    kwargs['mon_dist']
                )
            else:
                # 如果没有提供必要的参数，回退到角谱方法
                result = self.propagate(field_or_phase)
                return result, 1.0

    def phase_to_electronic_field(self, phase, x, y):
        """
        Convert phase to electronic field

        Args:
            phase: Phase distribution
            x: Size in x direction
            y: Size in y direction

        Returns:
            Electronic field
        """
        g = np.exp(-1j * phase)
        g1 = np.zeros_like(phase)
        datax = np.linspace(-x / 2, x / 2, phase.shape[1])
        datay = np.linspace(-y / 2, y / 2, phase.shape[0])
        X, Y = np.meshgrid(datax, datay)
        r = np.sqrt(X**2 + Y**2)
        R = min(x, y) / 2
        g1[r <= R] = 1
        g = g * g1

        Ex = g * 1
        Ey = g * 1j
        Ez = np.zeros_like(Ex, dtype=np.complex64)
        return ElectronicField(Ex, Ey, Ez)

    def expande_electronic_field(self, init_phase, Px, Py, fdtd_x, fdtd_y):
        """
        Expand electronic field

        Args:
            init_phase: Initial phase
            Px: Size in x direction
            Py: Size in y direction
            fdtd_x: FDTD x coordinates
            fdtd_y: FDTD y coordinates

        Returns:
            Expanded electronic field
        """
        # Initialize optical field
        g = np.exp(-1j * init_phase)
        datax = np.linspace(-Px / 2, Px / 2, init_phase.shape[1])
        datay = np.linspace(-Py / 2, Py / 2, init_phase.shape[0])
        X, Y = np.meshgrid(datax, datay)
        g = np.where(np.sqrt(X**2 + Y**2) <= min(Px, Py) / 2, g, 0)

        Ex = g * 1
        Ey = g * 1j
        Ez = np.zeros_like(Ex, dtype=np.complex64)

        def interpolate_and_expand(E):
            from scipy.interpolate import RectBivariateSpline
            E_amp, E_phase = np.abs(E), np.angle(E)
            E_amp_interpolated = RectBivariateSpline(datax, datay, E_amp, kx=1, ky=1)(
                fdtd_x, fdtd_y
            )
            E_phase_interpolated = RectBivariateSpline(datax, datay, E_phase, kx=1, ky=1)(
                fdtd_x, fdtd_y
            )
            E_interpolated = E_amp_interpolated * np.exp(1j * E_phase_interpolated)
            E_expanded = np.tile(E_interpolated, (N_expansion, 1))
            return E_expanded

        Ex_expanded = interpolate_and_expand(Ex)
        Ey_expanded = interpolate_and_expand(Ey)
        Ez_expanded = interpolate_and_expand(Ez)
        return ElectronicField(Ex_expanded, Ey_expanded, Ez_expanded)

    def interp_phase(self, num_structure_x, num_structure_y, Px, Py, ws, ps, f, lambda_) -> tuple[np.ndarray, np.ndarray]:
        """
        Interpolate phase values

        Args:
            num_structure_x: Number of structures in x direction
            num_structure_y: Number of structures in y direction
            Px: Size in x direction
            Py: Size in y direction
            ws: Widths
            ps: Phases
            f: Focal length
            lambda_: Wavelength

        Returns:
            Interpolated phase values
        """
        from scipy.interpolate import interp1d
        # Generate grid
        x = np.linspace(-Px / 2, Px / 2, num_structure_x)
        y = np.linspace(-Py / 2, Py / 2, num_structure_y)
        X, Y = np.meshgrid(x, y)
        k = 360 / lambda_
        r2 = X**2 + Y**2
        phase = k * (np.sqrt(r2 + f**2) - f)
        phase = np.mod(phase, 360)

        # Interpolation
        P_a = interp1d(ps, ws, kind="nearest", fill_value=np.nan)(phase)
        phase_map = dict(zip(ws, np.deg2rad(ps)))
        dist_phase = np.vectorize(phase_map.get)(P_a)
        return P_a, dist_phase

    def trans_beam(
        self,
        init_phase,
        params: OpticParameters,
        mon_dist: float,
    ) -> tuple[ElectronicField, float]:
        """
        Transform beam and calculate efficiency

        Args:
            init_phase: Initial phase
            params: Optical parameters
            mon_dist: Monitor distance
            figure_path: Path to save figures

        Returns:
            tuple: Efficiency and paths to 2D and 1D plots
        """
        expanded_Ef = self.expande_electronic_field(
            init_phase,
            params.Px,
            params.Py,
            params.x_fdtd,
            params.y_fdtd,
        )

        x_FDTD = params.x_fdtd
        y_num = (len(x_FDTD) - 1) * N_expansion + 1
        y_FDTD = np.linspace(-params.Py / 2, params.Py / 2, y_num)

        Eo, jmX, jmY = self.perform_fft(
            expanded_Ef, x_FDTD, y_FDTD, params.k0, params.focal_length, mon_dist
        )
        efficiency, (fl, fr) = self.calculate_efficiency(Eo.G, jmX[0, :], jmY[:, 0], params.R)

        return Eo, efficiency

    def calculate_efficiency(self, G, x, y, R) -> tuple[float, tuple[float, float]]:
        """
        Calculate focusing efficiency

        Args:
            G: Intensity distribution
            x: X coordinates
            y: Y coordinates
            R: Radius

        Returns:
            Efficiency and FWHM values
        """
        from scipy.interpolate import interp1d
        A_half = G.shape[0] // 2
        If = G[A_half, :]
        If0 = If / np.max(G[A_half, :])

        pq = np.argmax(If0)
        # pq = A_half
        fl = interp1d(np.flip(If0[0 : pq + 1]), np.flip(x[0 : pq + 1]), kind="linear", fill_value=np.nan)(0.5)
        fr = interp1d(If0[pq:], x[pq:], kind="linear", fill_value=np.nan)(0.5)
        FWHM = abs(fl - fr)

        xc, yc = getattr(OptConfig, 'shift', (0, 0))
        rm1 = 3 * FWHM
        rm2 = min(np.max(x), np.max(y))
        rm1 = min(rm1, rm2)

        zmX, zmY = np.meshgrid(x, y)
        cycle = (zmX - xc) ** 2 * getattr(GlDim, 'x', 1) + (zmY - yc) ** 2 * getattr(GlDim, 'y', 1)
        I1 = np.where(cycle <= rm1**2, G, 0)
        I2 = np.where(cycle <= rm2**2, G, 0)
        efficiency = np.sum(I1) / np.sum(I2)

        return efficiency, (fl, fr)

    def perform_fft(self, e_field: ElectronicField, x_FDTD, y_FDTD, k0, focal_length, mon_dist, recover_scale=False):
        """
        Perform FFT transformation and calculate light intensity

        Args:
            e_field: Electronic field
            x_FDTD: FDTD x coordinates
            y_FDTD: FDTD y coordinates
            k0: Wave number
            focal_length: Focal length
            mon_dist: Monitor distance
            recover_scale: Whether to recover scale

        Returns:
            Electronic field and coordinates
        """
        from scipy.ndimage import zoom
        # Each pixel size
        px = abs(x_FDTD[-1] - x_FDTD[0]) / (len(x_FDTD) - 1)
        py = abs(y_FDTD[-1] - y_FDTD[0]) / (len(y_FDTD) - 1)

        # Expand field of view, a*b -> (jm*a)*3 * (jm*b)*3
        def expand(E, jm):
            E = zoom(E, jm, order=3) if jm != 1 else E
            a, b = E.shape
            E = np.pad(E,
                       ((FTT_expande_time*a, FTT_expande_time*a), (FTT_expande_time*b, FTT_expande_time*b)),
                       mode="constant", constant_values=0)
            return E

        jm = N_expansion
        jExi = expand(e_field.Ex, jm)
        jEyi = expand(e_field.Ey, jm)
        A, B = jExi.shape

        # FFT parameters
        lmaxX = (B / jm - 1) * px
        lmaxY = (A / jm - 1) * py
        jmx = np.linspace(-lmaxX / 2, lmaxX / 2, B)
        jmy = np.linspace(-lmaxY / 2, lmaxY / 2, A)
        jmX, jmY = np.meshgrid(jmx, jmy)
        jfx = np.pi * B / lmaxX
        jfy = np.pi * A / lmaxY
        jfmx = np.linspace(-jfx, jfx, B) / (2 * np.pi)
        jfmy = np.linspace(-jfy, jfy, A) / (2 * np.pi)
        jkx, jky = np.meshgrid(jfmx, jfmy)
        kr = np.sqrt(jkx**2 + jky**2) / k0

        # FFT transformation
        Ax = np.fft.fftshift(np.fft.fft2(jExi))
        Ay = np.fft.fftshift(np.fft.fft2(jEyi))
        Ax[kr > 1] = 0
        Ay[kr > 1] = 0
        q = np.sqrt(k0**2 - jkx**2 - jky**2, dtype=np.complex64)
        Az = -(jkx * Ax + jky * Ay) / q
        Az[kr > 1] = 0

        # Calculate focal plane light intensity
        zz = focal_length - mon_dist
        qq = np.exp(1j * 2 * PI * q * zz)

        Exo = np.fft.ifft2(np.fft.ifftshift(Ax * qq))
        Eyo = np.fft.ifft2(np.fft.ifftshift(Ay * qq))
        Ezo = np.fft.ifft2(np.fft.ifftshift(Az * qq))

        return (
            ElectronicField(Exo, Eyo, Ezo),
            jmX,
            jmY,
        )

    def rescale_EF(self, E:ElectronicField, jmX, jmY):
        """
        Rescale electronic field

        Args:
            E: Electronic field
            jmX: X coordinates
            jmY: Y coordinates

        Returns:
            Rescaled electronic field and coordinates
        """
        A = E.Ex.shape[0]
        expand_boarder = A //  int(2*FTT_expande_time+1)
        Exo = E.Ex[
            expand_boarder:-expand_boarder, expand_boarder:-expand_boarder
        ]
        Eyo = E.Ey[
            expand_boarder:-expand_boarder, expand_boarder:-expand_boarder
        ]
        Ezo = E.Ez[
            expand_boarder:-expand_boarder, expand_boarder:-expand_boarder
        ]
        return (
            ElectronicField(Exo, Eyo, Ezo),
            jmX[expand_boarder:-expand_boarder, expand_boarder:-expand_boarder],
            jmY[expand_boarder:-expand_boarder, expand_boarder:-expand_boarder],
        )


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

        # 创建瞳孔掩码
        self.pupil_mask = self._create_pupil_mask()

        # 初始化状态
        self._init_state()

        # 系统初始化完成

    def _create_pupil_mask(self):
        """创建圆形瞳孔掩码"""
        x = np.linspace(-1, 1, self.config.N)
        y = np.linspace(-1, 1, self.config.N)
        X, Y = np.meshgrid(x, y)
        mask = (X**2 + Y**2) <= 1.0
        return mask.astype(complex)

    def _init_state(self):
        """初始化系统状态"""
        # 创建入射光场 - 使用平面波并应用瞳孔掩码
        self.E_in = self.light_source.create_plane_wave(
            self.config.N, self.config.L
        ) * self.pupil_mask

        # 应用湍流
        self.E_turb = self.turbulence.add_phase_screen(self.E_in)

        # DM初始状态
        self.dm_voltages = np.zeros(self.dm.total_actuators)

        # 使用角谱传播
        self.E_propagated = self.propagator.propagate(self.E_turb)

        # 计算参考峰值（无湍流、无DM校正）
        E_ideal = self.propagator.propagate(self.E_in)
        self.reference_peak = np.max(np.abs(E_ideal)**2)

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

        # Strehl比：峰值强度相对于理想情况下的峰值
        raw_intensity = np.abs(self.E_corrected)**2
        strehl = float(np.clip(np.max(raw_intensity) / self.reference_peak, 0.001, 1.0))
        
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

        image = self.get_image()
        pupil_phase = np.angle(self.E_turb)
        pupil_phase_rms = np.sqrt(np.mean(pupil_phase**2))
        strehl = float(np.clip(np.exp(-pupil_phase_rms**2), 0.001, 1.0))

        return {
            'image': image,
            'slopes': self.measure_wavefront(),
            'strehl': strehl,
            'power': np.sum(image),
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


def calculate_strehl(intensity: np.ndarray, peak_reference: Optional[float]) -> float:
    """
    计算Strehl比
    
    参数:
        intensity: 强度分布
        peak_reference: 峰值参考值，如果为None则使用最大可能值
    返回:
        Strehl比
    """
    peak_intensity = np.max(intensity)
    peak_reference = peak_reference if peak_reference else peak_intensity
    assert peak_reference and peak_reference > 0, "峰值参考值必须大于0"
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
OpticalPropagator = VectorWavePropagator
AOSystem = TraditionalAOSystem
CCDCamera = Camera
WavefrontPropagator = VectorWavePropagator
HartmannShackSensor = HartmannShackWavefrontSensor

if __name__ == "__main__":
    # 测试代码
    pass