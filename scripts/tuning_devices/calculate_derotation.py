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
    import numpy as np
    
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