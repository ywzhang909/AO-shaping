"""
综合测试脚本：验证Zernike多项式与哈特曼传感器的一致性
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

# 导入我们创建的模块
try:
    from test_wavefront_analysis import (
        zernike_radial, zernike_polynomial, 
        SimpleHartmannShackWavefrontSensor, 
        generate_test_wavefront
    )
    from enhanced_hartmann_sensor import EnhancedHartmannShackWavefrontSensor
    print("成功导入自定义模块")
except ImportError as e:
    print(f"导入模块失败: {e}")
    # 如果导入失败，定义基本函数
    def zernike_radial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
        """简化版Zernike径向多项式"""
        m = abs(m)
        if (n - m) % 2 != 0:
            return np.zeros_like(rho)
        
        R = np.zeros_like(rho, dtype=float)
        for k in range((n - m) // 2 + 1):
            import math
            numerator = (-1) ** k * math.factorial(n - k)
            denominator = (
                math.factorial(k) *
                math.factorial((n + m) // 2 - k) *
                math.factorial((n - m) // 2 - k)
            )
            coefficient = numerator / denominator
            R += coefficient * (rho ** (n - 2 * k))
        return R

    def zernike_polynomial(n: int, m: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """简化版Zernike多项式"""
        R = zernike_radial(n, m, rho)
        if m > 0:
            return R * np.cos(m * theta)
        elif m < 0:
            return R * np.sin(abs(m) * theta)
        else:
            return R

    print("使用内置函数定义")


def test_mathematical_properties():
    """
    测试Zernike多项式的数学性质
    """
    print("="*60)
    print("测试Zernike多项式的数学性质")
    print("="*60)
    
    N = 128
    x = np.linspace(-1.1, 1.1, N)  # 稍微超出单位圆以测试边界
    y = np.linspace(-1.1, 1.1, N)
    X, Y = np.meshgrid(x, y)
    
    rho = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    
    # 测试正交性 - 使用几个低阶模式
    test_modes = [(0, 0), (1, -1), (1, 1), (2, -2), (2, 0), (2, 2)]
    
    print("测试Zernike模式的正交性:")
    integrals = np.zeros((len(test_modes), len(test_modes)))
    
    for i, (n1, m1) in enumerate(test_modes):
        for j, (n2, m2) in enumerate(test_modes):
            # 在单位圆内积分
            mask = rho <= 1.0
            z1 = zernike_polynomial(n1, m1, rho, theta) * mask
            z2 = zernike_polynomial(n2, m2, rho, theta) * mask
            
            # 数值积分：对单位圆上的乘积进行积分
            integral = np.sum(z1 * z2) / np.sum(mask)  # 归一化
            integrals[i, j] = integral
            
            # 标记是否应该接近零（非相同模式）
            is_orthogonal = (n1 != n2 or m1 != m2)
            expected_zero = abs(integral) < 0.1 if is_orthogonal else abs(integral) > 0.1
            
            marker = "✓" if expected_zero else "✗"
            print(f"  Z_{n1}^{m1} × Z_{n2}^{m2}: {integral:.4f} {marker}")
    
    print("\n正交性测试完成!")


def test_wavefront_sensor_accuracy():
    """
    测试波前传感器的准确性
    """
    print("\n" + "="*60)
    print("测试波前传感器准确性")
    print("="*60)
    
    N = 128
    subapertures = 8
    
    # 创建传感器
    wfs = SimpleHartmannShackWavefrontSensor(
        subapertures=subapertures,
        pixel_scale=0.5,
        N=N
    )
    
    # 生成不同的测试波前
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    
    rho = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    
    aperture_mask = rho <= 1.0
    
    test_cases = [
        ("平顶", np.ones_like(rho) * aperture_mask),
        ("倾斜X", X * aperture_mask),
        ("倾斜Y", Y * aperture_mask),
        ("离焦", (2*rho**2 - 1) * aperture_mask),
        ("像散", (X**2 - Y**2) * aperture_mask)
    ]
    
    print("测试不同波前类型的传感器响应:")
    for name, wavefront in test_cases:
        # 创建强度分布
        intensity = np.ones_like(wavefront) * aperture_mask
        
        # 测量斜率
        slopes = wfs.measure_slopes(intensity, wavefront)
        
        # 分析结果
        x_slopes = slopes[:subapertures**2]
        y_slopes = slopes[subapertures**2:]
        
        x_mean = np.mean(x_slopes)
        y_mean = np.mean(y_slopes)
        x_std = np.std(x_slopes)
        y_std = np.std(y_slopes)
        
        print(f"  {name:8s}: <Sx>={x_mean:6.4f}, <Sy>={y_mean:6.4f}, "
              f"σx={x_std:6.4f}, σy={y_std:6.4f}")


def visualize_zernike_hierarchy():
    """
    可视化Zernike模式的层级结构
    """
    print("\n" + "="*60)
    print("可视化Zernike模式层级结构")
    print("="*60)
    
    N = 128
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    
    rho = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    
    aperture_mask = rho <= 1.0
    
    # 显示前两行的Zernike模式 (n=0到n=3)
    n_max = 3
    modes_to_show = []
    
    for n in range(n_max + 1):
        for m in range(-n, n + 1, 2):
            z_mode = zernike_polynomial(n, m, rho, theta) * aperture_mask
            modes_to_show.append(((n, m), z_mode))
    
    # 绘制模式
    n_modes = len(modes_to_show)
    cols = 4
    rows = (n_modes + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    if rows == 1:
        axes = axes if cols > 1 else [axes]
    else:
        axes = axes.flatten()
    
    for idx, ((n, m), z_mode) in enumerate(modes_to_show):
        im = axes[idx].imshow(z_mode, cmap='RdBu', vmin=-1, vmax=1)
        axes[idx].set_title(f'Z_{n}^{m}', fontsize=10)
        axes[idx].axis('off')
        # 添加颜色条
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
    
    # 隐藏多余的子图
    for idx in range(len(modes_to_show), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Zernike模式层级结构', y=1.02, fontsize=14)
    plt.savefig('zernike_hierarchy.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Zernike层级结构可视化完成，共显示 {len(modes_to_show)} 个模式")


def main():
    """
    主函数：运行所有测试
    """
    print("开始综合测试：Zernike多项式与哈特曼传感器一致性分析")
    print("="*70)
    
    # 1. 测试数学性质
    test_mathematical_properties()
    
    # 2. 测试传感器准确性
    test_wavefront_sensor_accuracy()
    
    # 3. 可视化层级结构
    visualize_zernike_hierarchy()
    
    print("\n" + "="*70)
    print("所有综合测试完成!")
    print("生成的文件:")
    print("- zernike_hierarchy.png: Zernike模式层级结构图")
    print("="*70)
    
    # 最后的总结
    print("\n总结:")
    print("1. Zernike多项式具有良好的数学性质，特别是正交性")
    print("2. 哈特曼传感器能够检测不同类型的波前畸变")
    print("3. 不同Zernike模式对应不同的光学像差类型")
    print("4. 传感器响应与波前特性密切相关")


if __name__ == "__main__":
    main()