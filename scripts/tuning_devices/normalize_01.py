def normalize_01(matrix):
    """
    将矩阵归一化到[0, 1]范围
    
    参数:
    matrix: 输入矩阵
    
    返回:
    normalized: 归一化后的矩阵
    """
    import numpy as np
    
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    
    # 避免除以零的情况
    if max_val == min_val:
        normalized = np.zeros_like(matrix)
    else:
        normalized = (matrix - min_val) / (max_val - min_val)
    
    return normalized