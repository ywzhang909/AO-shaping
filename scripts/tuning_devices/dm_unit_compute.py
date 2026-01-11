#!/usr/bin/env python3
"""
dmunitcompute.m 的Python实现
用于计算100mm×100mm对应的像素尺寸，并处理stdWavefront目录下的矩阵文件
"""

import numpy as np
from pathlib import Path
import pygame
from contextlib import contextmanager

from ao_shaping.drivers import Thorlab_WFS, MlaRes, NlightDM
from ao_shaping.utils import get_init_V_by_rms, get_init_V_by_energy
from ao_shaping.sim.zernike import Zernike

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
        
    def get_centroid(self, zernike_coef):
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
    
    # def __iter__(self):
    #     self.wfs.initialize()
        
        
@contextmanager
def visualize_with_pygame(title="Image Visualization"):
    """
    使用pygame实时显示矩阵图像并在图像上显示质心坐标
    
    参数:
    matrix: 要显示的矩阵
    cx: 质心x坐标
    cy: 质心y坐标
    title: 窗口标题
    """
    # 初始化pygame
    pygame.init()
    calculator = ZernikeCentroidCalculator()
    # 设置窗口大小（根据矩阵大小调整）
    height, width = calculator.shape
    window_width = min(width, 800)  # 最大800像素宽
    window_height = min(height, 600)  # 最大600像素高
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption(title)
    scaled_surface = pygame.Surface((width, height))
    
    # 缩放图像到窗口大小
    scaled_image = pygame.transform.scale(scaled_surface, (window_width, window_height))
    
    # 字体设置
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)
    
    # 创建退出按钮
    button_width = 100
    button_height = 40
    button_x = window_width - button_width - 10
    button_y = 10
    button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
    
    # 主循环
    running = True
    clock = pygame.time.Clock()
    
    # WFS init
    wfs = Thorlab_WFS(MlaRes.Res768)
    wfs.initialize()
    
    exposure_time, gain = wfs.optimize_exposure_time_and_gain()
    wfs.exposure_time = exposure_time
    wfs.optimize_pupil()
    wfs.pupil = wfs.optimize_pupil()
    
    try:
        assert running
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if button_rect.collidepoint(event.pos):
                    running = False
        
        zernike_coef = wfs.get_zernike()
        (dx, dy), img = calculator.get_centroid(zernike_coef)
        cx, cy = calculator.center_coordinate(dx, dy)
        cx, cy = calculate_derotation(cx, cy, np.deg2rad(45))
        # 绘制图像
        scaled_image = pygame.surfarray.make_surface(to_color(img))
        pygame.draw.circle(scaled_image, (255, 0, 0), (int(dx), int(dy)), 10, 15)
        screen.blit(scaled_image, (0, 0))
        
        # 显示质心坐标
        content = f'center: x={calculator.pix_to_mm(cx):.2f}mm, y={calculator.pix_to_mm(cy):.2f}mm'
        text = font.render(content, True, (255, 255, 255))  # 白色文字
        screen.blit(text, (10, window_height-font.size(content)[1]-10))
        
        # 绘制退出按钮
        pygame.draw.rect(screen, (255, 0, 0), button_rect)  # 红色按钮
        button_text = small_font.render("Quit", True, (255, 255, 255))  # 白色文字
        text_rect = button_text.get_rect(center=button_rect.center)
        screen.blit(button_text, text_rect)
        
        # 更新显示
        pygame.display.flip()
        clock.tick(30)  # 30 FPS
        yield
    except AssertionError as e:
        pass
    finally:
        wfs.close()
        pygame.quit()
        
    

if __name__ == "__main__":
    while True:
        visualize_with_pygame()
