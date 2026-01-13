"""
Zernike多项式模块

提供Zernike多项式的计算功能，用于波前描述和自适应光学仿真。
"""

import math
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List


def zernike_radial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """
    计算Zernike径向多项式
    
    参数:
        n: 径向阶数
        m: 角向阶数
        rho: 归一化径向坐标 (0-1)
    返回:
        R_n^m(rho) 数组
    """
    if np.isscalar(rho):
        rho = np.array([rho])
    
    # 确保 m 的符号正确
    if m < 0:
        m = -m
        sign = (-1) ** ((n - m) // 2)
    else:
        sign = 1
    
    R = np.zeros_like(rho)
    
    # 计算径向多项式
    for k in range((n - m) // 2 + 1):
        coefficient = (-1) ** k * math.factorial(n - k)
        coefficient /= (math.factorial(k) *
                       math.factorial((n + m) // 2 - k) *
                       math.factorial((n - m) // 2 - k))
        R += coefficient * rho ** (n - 2 * k)
    
    return sign * R


def zernike_polynomial(n: int, m: int, rho: np.ndarray, 
                       theta: np.ndarray) -> np.ndarray:
    """
    计算Zernike多项式
    
    参数:
        n: 径向阶数
        m: 角向阶数
        rho: 归一化径向坐标 (0-1)
        theta: 角坐标 (rad)
    返回:
        Z_n^m(rho, theta) 数组
    """
    if np.isscalar(rho):
        rho = np.array([rho])
    if np.isscalar(theta):
        theta = np.array([theta])
    
    R = zernike_radial(n, np.abs(m), rho)
    
    if m >= 0:
        return R * np.cos(m * theta)
    else:
        return R * np.sin(-m * theta)


def normalize_zernike(n: int, m: int) -> float:
    """
    计算Zernike多项式的归一化因子
    
    参数:
        n: 径向阶数
        m: 角向阶数
    返回:
        归一化因子
    """
    if m == 0:
        return np.sqrt(n + 1)
    else:
        return np.sqrt(2 * (n + 1))


def generate_zernike_modes(n_max: int, rho: np.ndarray, 
                          theta: np.ndarray) -> list:
    """
    生成多个Zernike模式
    
    参数:
        n_max: 最大径向阶数
        rho: 归一化径向坐标
        theta: 角坐标
    返回:
        Zernike模式列表 [(n, m, 模式数组), ...]
    """
    modes = []
    for n in range(n_max + 1):
        for m in range(-n, n + 1, 2):
            norm = normalize_zernike(n, m)
            z = zernike_polynomial(n, m, rho, theta) * norm
            modes.append((n, m, z))
    return modes


def get_noll_indices(n_max: int) -> list:
    """
    获取Noll索引对应的 (n, m) 列表
    
    参数:
        n_max: 最大径向阶数
    返回:
        [(Noll_index, n, m), ...]
    """
    indices = []
    j = 1
    for n in range(n_max + 1):
        for m in range(-n, n + 1, 2):
            if n != 0:  # 跳过piston
                indices.append((j, n, m))
                j += 1
    return indices


def noll_to_nm(j: int) -> Tuple[int, int]:
    """
    将Noll索引转换为 (n, m)
    
    参数:
        j: Noll索引
    返回:
        (n, m)
    """
    if j == 0:
        return 0, 0
    
    n = 0
    while True:
        n += 1
        m_range = range(-n, n + 1, 2)
        count = len(list(m_range))
        if j <= count:
            m = -n + 2 * (j - 1)
            return n, m
        j -= count
    
    return n, 0


def compute_zernike_variance(n: int, r0: float, D: float) -> float:
    """
    计算Kolmogorov湍流下Zernike系数的方差
    
    参数:
        n: 径向阶数
        r0: Fried参数 (m)
        D: 望远镜口径 (m)
    返回:
        方差
    """
    if n == 0:
        return 0.0
    
    # Noll公式
    factor = 0.2944 * (n + 1) ** (-5/6) * (D / r0) ** (5/3)
    return factor


def reconstruct_wavefront(coefficients: np.ndarray, 
                         rho: np.ndarray,
                         theta: np.ndarray,
                         n_max: int = 10) -> np.ndarray:
    """
    从Zernike系数重建波前
    
    参数:
        coefficients: Zernike系数数组
        rho: 归一化径向坐标
        theta: 角坐标
        n_max: 最大径向阶数
    返回:
        波前相位分布
    """
    modes = generate_zernike_modes(n_max, rho, theta)
    wavefront = np.zeros_like(rho)
    
    for idx, (n, m, mode) in enumerate(modes):
        if idx < len(coefficients):
            wavefront += coefficients[idx] * mode
    
    return wavefront


def zernike_coeffs_to_rms(coefficients: np.ndarray, n_max: int) -> float:
    """
    计算Zernike系数的RMS值
    
    参数:
        coefficients: Zernike系数
        n_max: 最大径向阶数
    返回:
        RMS值 (rad)
    """
    # 对于归一化的Zernike多项式，RMS就是系数平方和的平方根
    return np.sqrt(np.sum(coefficients**2))


def fit_zernike_to_phase(phase: np.ndarray, 
                        mask: np.ndarray,
                        n_max: int = 10) -> np.ndarray:
    """
    将相位拟合为Zernike多项式
    
    参数:
        phase: 相位分布
        mask: 孔径掩膜
        n_max: 最大径向阶数
    返回:
        Zernike系数
    """
    # 获取网格坐标
    ny, nx = phase.shape
    y, x = np.mgrid[:ny, :nx].astype(float)
    
    # 归一化到 [-1, 1]
    x_norm = (x - nx/2) / (nx/2 - 1)
    y_norm = (y - ny/2) / (ny/2 - 1)
    
    rho = np.sqrt(x_norm**2 + y_norm**2)
    theta = np.arctan2(y_norm, x_norm)
    
    # 生成Zernike模式
    modes = generate_zernike_modes(n_max, rho, theta)
    
    # 构建矩阵
    A = []
    for n, m, mode in modes:
        A.append(mode[mask > 0])
    
    A = np.column_stack(A)
    b = phase[mask > 0]
    
    # 最小二乘拟合
    coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
    
    return coeffs


class Zernike:
    """
    统一的Zernike多项式类

    整合所有Zernike相关功能，包括计算、重建、拟合等。
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

    def __init__(self, n_max: int = 10, N: int = 256, L: float = 1.0,
                 use_cached: bool = False, cache_path: Optional[str] = None):
        """
        初始化Zernike类

        参数:
            n_max: 最大径向阶数
            N: 网格点数
            L: 物理孔径大小
            use_cached: 是否使用缓存的基函数
            cache_path: 缓存文件路径
        """
        self.n_max = n_max
        self.N = N
        self.L = L
        self.use_cached = use_cached
        self.cache_path = cache_path or 'scripts/tuning_devices/stdWavefront'

        # 创建网格
        self._create_grid()

        # 生成Noll索引
        self.noll_indices = self._generate_noll_indices()

        # 预计算基函数
        if use_cached:
            self._load_cached_basis()
        else:
            self._generate_basis()

    def _create_grid(self):
        """创建坐标网格"""
        x = np.linspace(-self.L/2, self.L/2, self.N)
        y = np.linspace(-self.L/2, self.L/2, self.N)
        X, Y = np.meshgrid(x, y)

        # 归一化到[-1, 1]
        self.X_norm = X / (self.L/2)
        self.Y_norm = Y / (self.L/2)

        # 极坐标
        self.rho = np.sqrt(self.X_norm**2 + self.Y_norm**2)
        self.theta = np.arctan2(self.Y_norm, self.X_norm)

        # 圆形孔径掩码
        self.mask = self.rho <= 1.0

    def _generate_noll_indices(self) -> List[Tuple[int, int, int]]:
        """生成Noll索引对应的(n, m)列表"""
        indices = []
        j = 1
        for n in range(self.n_max + 1):
            for m in range(-n, n + 1, 2):
                if n != 0:  # 跳过piston
                    indices.append((j, n, m))
                    j += 1
        return indices

    def _generate_basis(self):
        """生成Zernike基函数"""
        num_modes = len(self.noll_indices)
        self.basis = np.zeros((num_modes, self.N, self.N))

        for idx, (j, n, m) in enumerate(self.noll_indices):
            norm = normalize_zernike(n, m)
            z = zernike_polynomial(n, m, self.rho, self.theta) * norm
            z[~self.mask] = 0
            self.basis[idx] = z

    def _load_cached_basis(self):
        """从文件加载预计算的基函数"""
        try:
            txt_files = list(Path(self.cache_path).glob('*.txt'))
            num_files = len(txt_files)
            self.basis = np.zeros((num_files, self.N, self.N))

            for i, txt_file in enumerate(sorted(txt_files)):
                data = np.loadtxt(txt_file)
                if data.size == self.N * self.N:
                    self.basis[i] = data.reshape(self.N, self.N)
                else:
                    # 如果尺寸不匹配，重新生成
                    print(f"缓存文件尺寸{data.shape}不匹配期望尺寸({self.N},{self.N})，重新生成基函数")
                    self._generate_basis()
                    return
        except Exception as e:
            print(f"加载缓存失败: {e}，重新生成基函数")
            self._generate_basis()

    @staticmethod
    def get_name(n: int, m: int) -> str:
        """获取Zernike模式名称"""
        return Zernike.ZERNIKE_NAMES.get((n, m), f"Z({n},{m})")

    def get_mode(self, j: int) -> np.ndarray:
        """
        获取指定Noll索引的Zernike模式

        参数:
            j: Noll索引 (从1开始)

        返回:
            模式数组
        """
        if 1 <= j <= len(self.noll_indices):
            return self.basis[j - 1]
        else:
            raise ValueError(f"Noll索引 {j} 超出范围 [1, {len(self.noll_indices)}]")

    def get_mode_by_nm(self, n: int, m: int) -> np.ndarray:
        """
        根据(n, m)获取Zernike模式

        参数:
            n: 径向阶数
            m: 角向阶数

        返回:
            模式数组
        """
        for j, nn, mm in self.noll_indices:
            if nn == n and mm == m:
                return self.get_mode(j)
        raise ValueError(f"模式 ({n}, {m}) 不存在")

    def reconstruct(self, coefficients: np.ndarray) -> np.ndarray:
        """
        从系数重建波前

        参数:
            coefficients: Zernike系数数组

        返回:
            重建的波前
        """
        num_coeffs = min(len(coefficients), self.basis.shape[0])
        wavefront = np.sum([coefficients[i] * self.basis[i]
                           for i in range(num_coeffs)], axis=0)
        return wavefront

    def fit(self, phase: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        将相位拟合为Zernike多项式

        参数:
            phase: 相位分布
            mask: 孔径掩膜

        返回:
            Zernike系数
        """
        if mask is None:
            mask = self.mask

        # 展平
        phase_flat = phase[mask]
        basis_flat = self.basis[:, mask]

        # 最小二乘拟合
        coeffs = np.linalg.lstsq(basis_flat.T, phase_flat, rcond=None)[0]

        return coeffs

    def piston_tilt_basis(self) -> np.ndarray:
        """
        获取piston + x-tilt + y-tilt基函数

        返回:
            基函数数组，shape为(3, N, N)
        """
        # Piston (n=0, m=0) - 归一化到与Zernike基函数相同的范数
        piston = np.ones((self.N, self.N)) * self.mask
        piston_norm = piston / np.sqrt(np.sum(piston**2))

        # X-tilt (n=1, m=-1) - 已经归一化
        x_tilt = self.get_mode_by_nm(1, -1)

        # Y-tilt (n=1, m=1) - 已经归一化
        y_tilt = self.get_mode_by_nm(1, 1)

        return np.stack([piston_norm, x_tilt, y_tilt])

    def compute_rms(self, coefficients: np.ndarray) -> float:
        """
        计算Zernike系数的RMS值

        参数:
            coefficients: Zernike系数

        返回:
            RMS值
        """
        return zernike_coeffs_to_rms(coefficients, self.n_max)

    def compute_variance(self, n: int, r0: float, D: float) -> float:
        """
        计算Kolmogorov湍流下Zernike系数的方差

        参数:
            n: 径向阶数
            r0: Fried参数
            D: 望远镜口径

        返回:
            方差
        """
        return compute_zernike_variance(n, r0, D)

    @property
    def num_modes(self) -> int:
        """返回模式数量"""
        return len(self.noll_indices)
