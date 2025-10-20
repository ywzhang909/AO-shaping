#!/usr/bin/env python3
"""
dmunitcompute.m 的Python实现
用于计算100mm×100mm对应的像素尺寸，并处理stdWavefront目录下的矩阵文件
"""

import os
import numpy as np
from pathlib import Path

# 导入自定义函数
from .calculate_derotation import calculate_derotation
from .centroid_calculation import centroid_calculation
from .normalize_01 import normalize_01

def main():
    # 计算100mm×100mm对应的像素尺寸  —（233，220）  （237，223）  
    mm_size = 100  # 实际尺寸(mm)
    resolution_for_70mm = 360  # 70mm对应的像素数
    pixel_per_mm = resolution_for_70mm / 70  # 每毫米的像素数
    pixel_size = int(round(mm_size * pixel_per_mm))  # 100mm对应的像素数
    mm_per_pixel = 1 / pixel_per_mm  # 每像素对应的毫米数 (约0.1944mm/像素)
    
    print(f"像素尺寸: {pixel_size}x{pixel_size}")
    print(f"每像素毫米数: {mm_per_pixel:.4f} mm/pixel")
    
    # 读取stdWavefront下所有矩阵文件
    folder_path = 'scripts/tuning_devices/stdWavefront'
    file_pattern = '*.txt'
    
    # 获取所有txt文件
    txt_files = list(Path(folder_path).glob('*.txt'))
    print(f"找到 {len(txt_files)} 个文件")
    
    # 初始化矩阵
    zernike_base_matrix = np.zeros((360, 360))
    coeff = np.zeros((64, 1))
    
    # 处理每个文件
    for i, file_path in enumerate(txt_files):
        if i >= 64:  # 最多处理64个文件
            break
            
        # 读取数据
        data = np.loadtxt(file_path)
        A1 = data.reshape(360, 360)
        
        # 根据注释中的逻辑，这里应该是累加操作
        # 原始代码中使用了 coeff(i)，但没有定义coeff的值，这里假设为1
        zernike_base_matrix = zernike_base_matrix + A1 * coeff[i] if i < len(coeff) else zernike_base_matrix + A1
        
        print(f"处理文件: {file_path.name}")
    
    # 创建对应尺寸的二维数组
    matrix_data = np.ones((pixel_size, pixel_size))
    
    A = zernike_base_matrix
    A1 = A.reshape(360, 360)
    
    # 归一化
    A_norm = normalize_01(A1)
    A_norm = A_norm - 0.0
    
    # 设置负值为0
    A_norm[A_norm < 0] = 0
    
    # 计算质心
    cx_A, cy_A = centroid_calculation(A_norm)
    
    print(f'A1 质心: ({cx_A:.2f}, {cy_A:.2f})')
    
    # 可视化部分（如果需要的话）
    # 这里可以添加matplotlib相关代码来显示图像
    
    # 如果需要进行消旋计算，可以调用calculate_derotation函数
    # 例如：
    # theta = 0.1  # 示例角度
    # cx_A_derotated, cy_A_derotated = calculate_derotation(cx_A, cy_A, theta)
    # print(f'消旋后质心: ({cx_A_derotated:.2f}, {cy_A_derotated:.2f})')

if __name__ == "__main__":
    main()