"""
Zernike多项式模块

提供Zernike多项式的计算功能，用于波前描述和自适应光学仿真。
"""

import numpy as np


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
        coefficient = (-1) ** k * np.math.factorial(n - k)
        coefficient /= (np.math.factorial(k) * 
                       np.math.factorial((n + m) // 2 - k) * 
                       np.math.factorial((n - m) // 2 - k))
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
