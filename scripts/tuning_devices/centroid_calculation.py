def centroid_calculation(matrix):
    """
    计算矩阵的质心坐标
    
    参数:
    matrix: 输入矩阵
    
    返回:
    c_x: 质心x坐标
    c_y: 质心y坐标
    """
    import numpy as np
    
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