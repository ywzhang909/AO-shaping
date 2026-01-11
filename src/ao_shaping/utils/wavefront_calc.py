import numpy as np
from pathlib import Path
from ..sim.zernike import Zernike

def normalize_01(matrix):
    """
    将矩阵归一化到[0, 1]范围
    
    参数:
    matrix: 输入矩阵
    
    返回:
    normalized: 归一化后的矩阵
    """
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    
    # 避免除以零的情况
    if max_val == min_val:
        normalized = np.zeros_like(matrix)
    else:
        normalized = (matrix - min_val) / (max_val - min_val)
    
    return normalized

def centroid_calculation(matrix):
    """
    计算矩阵的质心坐标
    
    参数:
    matrix: 输入矩阵
    
    返回:
    c_x: 质心x坐标
    c_y: 质心y坐标
    """
    # 获取矩阵尺寸
    rows, cols = matrix.shape
    
    # 创建坐标网格
    x, y = np.meshgrid(np.arange(1, cols + 1), np.arange(1, rows + 1))
    
    # 计算总和
    sum_intensity = np.sum(matrix)
    
    # 计算质心坐标 (加权平均)
    c_x = np.sum(matrix * x) / sum_intensity
    c_y = np.sum(matrix * y) / sum_intensity
    
    return c_x, c_y

def calculate_derotation(x_actual, y_actual, theta):
    """
    计算消旋坐标变换（反向旋转theta角）
    
    参数:
    x_actual: 实际x坐标
    y_actual: 实际y坐标  
    theta: 旋转角度
    
    返回:
    x_derotated: 消旋后的x坐标
    y_derotated: 消旋后的y坐标
    """
    # 步骤2：计算旋转角的余弦值和正弦值
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # 步骤3：执行消旋坐标变换（反向旋转theta角），公式依据专利消旋原理推导
    x_derotated = x_actual * cos_theta + y_actual * sin_theta
    y_derotated = -x_actual * sin_theta + y_actual * cos_theta
    
    # 步骤4：输出消旋后的坐标（保留6位小数，与专利实施例数据精度一致，如0.025mm、-0.144mm）
    x_derotated = np.round(x_derotated, 6)
    y_derotated = np.round(y_derotated, 6)
    
    return x_derotated, y_derotated

def get_zernike_base_matrixs(folder_path: str = 'scripts/tuning_devices/stdWavefront',
                           n_max: int = 10, N: int = 360) -> np.ndarray:
    """
    获取Zernike基函数矩阵

    参数:
        folder_path: 缓存文件路径
        n_max: 最大径向阶数
        N: 网格点数

    返回:
        wavefront_matrices: shape为(num_modes, N, N)的Zernike基函数
    """
    try:
        # 尝试从文件加载
        txt_files = list(Path(folder_path).glob('*.txt'))
        print(f"找到 {len(txt_files)} 个缓存文件")

        num_files = len(txt_files)
        wavefront_matrices = np.zeros((num_files, N, N))

        for i, txt_file in enumerate(sorted(txt_files)):
            data = np.loadtxt(txt_file)
            if data.size == N * N:
                wavefront_matrices[i] = data.reshape(N, N)
            else:
                raise ValueError(f"文件 {txt_file} 尺寸不匹配")

        return wavefront_matrices

    except Exception as e:
        print(f"加载缓存失败: {e}，使用Zernike类生成")
        # 使用Zernike类生成
        zernike_obj = Zernike(n_max=n_max, N=N, L=2.0)  # 范围[-1,1]对应L=2.0
        return zernike_obj.basis

def to_color(matrix, max_val=1):   
        # 将矩阵转换为RGB图像（归一化到0-255范围）
        normalized_matrix = (matrix) / (max_val + 1e-8)
        rgb_matrix = np.stack([normalized_matrix*255]*3, axis=-1).astype(np.uint8)
        return rgb_matrix

class ZernikeCentroidCalculator:
    """Zernike质心计算器"""

    # 计算100mm×100mm对应的像素尺寸  —（233，220）  （237，223）
    mm_size = 100  # 实际尺寸(mm)
    resolution_for_70mm = 360  # 70mm对应的像素数
    shape = (resolution_for_70mm, resolution_for_70mm)
    pixel_per_mm = resolution_for_70mm / 70  # 每毫米的像素数
    pixel_size = int(round(mm_size * pixel_per_mm))  # 100mm对应的像素数
    mm_per_pixel = 1 / pixel_per_mm  # 每像素对应的毫米数 (约0.1944mm/像素)

    def __init__(self, folder_path: str = 'scripts/tuning_devices/stdWavefront',
                 black_level: float = 0.0, n_max: int = 10):
        """
        初始化Zernike质心计算器

        参数:
            folder_path: Zernike基函数文件路径
            black_level: 黑电平
            n_max: 最大径向阶数
        """
        self.wavefront_matrices = get_zernike_base_matrixs(folder_path, n_max, self.resolution_for_70mm)
        self.num_files = self.wavefront_matrices.shape[0]
        self.black_level = black_level
        self.n_max = n_max
        
    def get_centroid(self, zernike_coef:np.ndarray):
        """
        计算给定Zernike系数组合的波前矩阵的质心坐标
        """
        zer_class = min(len(zernike_coef), self.num_files)
        _zernike_coef = zernike_coef[:zer_class]
        _wavefront_matrices = self.wavefront_matrices[:zer_class]
        _zernike_base_matrix = np.sum(_wavefront_matrices * _zernike_coef[:, np.newaxis, np.newaxis], axis=0)
        _zernike_base_matrix = normalize_01(_zernike_base_matrix)
        _zernike_base_matrix = _zernike_base_matrix - self.black_level
        _zernike_base_matrix = np.where(_zernike_base_matrix < 0, 0, _zernike_base_matrix)
        cx, cy = centroid_calculation(_zernike_base_matrix)
        return (cx, cy), _zernike_base_matrix
    
    def pix_to_mm(self, pix):
        """
        将像素坐标转换为毫米坐标
        """
        return (pix - self.resolution_for_70mm / 2) * self.mm_per_pixel
    
    def center_coordinate(self, cx, cy):
        """
        将像素坐标转换为毫米坐标
        """
        return (cx - self.resolution_for_70mm / 2), (cy - self.resolution_for_70mm / 2)