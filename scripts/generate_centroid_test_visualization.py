"""质心计算测试可视化脚本

生成高斯分布测试图像和质心位置的可视化结果
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 添加src到路径以便导入ao_shaping模块
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ao_shaping.utils.spots_calc import centroid


def create_gaussian_2d(shape: tuple, center: tuple, sigma: float) -> np.ndarray:
    """创建2D高斯分布
    
    Args:
        shape: 矩阵形状 (height, width)
        center: 高斯中心 (x, y)
        sigma: 高斯标准差
        
    Returns:
        2D高斯分布矩阵
    """
    h, w = shape
    cy, cx = np.ogrid[:h, :w]
    x, y = center
    gaussian = np.exp(-((cx - x) ** 2 + (cy - y) ** 2) / (2 * sigma ** 2))
    return gaussian


def generate_test_visualization():
    """生成测试可视化图像"""
    
    # 创建输出目录
    output_dir = Path(__file__).parent.parent / "reports" / "centroid_test_visualization"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 测试用例配置
    test_cases = [
        {
            "name": "centered_100x100",
            "size": 100,
            "center": (50, 50),
            "sigma": 10.0,
            "noise": 0.0,
            "description": "中心位置 - 100x100"
        },
        {
            "name": "shifted_100x100",
            "size": 100,
            "center": (30, 70),
            "sigma": 8.0,
            "noise": 0.0,
            "description": "偏移位置 - 100x100"
        },
        {
            "name": "noise_100x100",
            "size": 100,
            "center": (45, 55),
            "sigma": 12.0,
            "noise": 0.05,
            "description": "高斯+噪声 - 100x100"
        },
        {
            "name": "large_noise_120x120",
            "size": 120,
            "center": (60, 60),
            "sigma": 15.0,
            "noise": 0.1,
            "description": "较大噪声 - 120x120"
        },
        {
            "name": "small_sigma_100x100",
            "size": 100,
            "center": (40, 60),
            "sigma": 5.0,
            "noise": 0.0,
            "description": "窄高斯(sigma=5) - 100x100"
        },
        {
            "name": "large_sigma_100x100",
            "size": 100,
            "center": (50, 50),
            "sigma": 30.0,
            "noise": 0.0,
            "description": "宽高斯(sigma=30) - 100x100"
        },
        {
            "name": "threshold_100x100",
            "size": 100,
            "center": (50, 50),
            "sigma": 10.0,
            "noise": 0.1,
            "background": 0.1,
            "description": "带阈值 - 100x100"
        },
        {
            "name": "150x150",
            "size": 150,
            "center": (75, 80),
            "sigma": 15.0,
            "noise": 0.03,
            "description": "大尺寸 - 150x150"
        },
        {
            "name": "200x200",
            "size": 200,
            "center": (100, 100),
            "sigma": 20.0,
            "noise": 0.02,
            "description": "大尺寸 - 200x200"
        }
    ]
    
    np.random.seed(42)
    results = []
    
    # 创建大图
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle("Centroid Test Visualization\n质心计算测试可视化", fontsize=14, fontweight='bold')
    
    for idx, test_case in enumerate(test_cases):
        ax = axes[idx // 3, idx % 3]
        
        size = test_case["size"]
        center = test_case["center"]
        sigma = test_case["sigma"]
        noise = test_case["noise"]
        
        # 创建高斯分布
        intensity = create_gaussian_2d((size, size), center, sigma)
        
        # 添加噪声
        if noise > 0:
            noise_array = np.random.normal(0, noise, (size, size))
            intensity = np.clip(intensity + noise_array, 0, None)
        
        # 添加背景
        if "background" in test_case:
            intensity = intensity + test_case["background"]
        
        # 计算质心
        cx_float, cy_float = centroid(intensity, return_float=True)
        cx_int, cy_int = centroid(intensity, return_float=False)
        
        # 计算误差
        error_x = abs(cx_float - center[0])
        error_y = abs(cy_float - center[1])
        
        results.append({
            "name": test_case["description"],
            "expected": center,
            "calculated": (cx_float, cy_int),
            "error_x": error_x,
            "error_y": error_y
        })
        
        # 绘制图像
        im = ax.imshow(intensity, cmap='viridis', origin='lower')
        
        # 标记预期中心 (绿色x)
        ax.scatter(center[0], center[1], c='green', marker='x', s=100, 
                   linewidths=2, label='Expected', zorder=5)
        
        # 标记计算质心 (红色+)
        ax.scatter(cx_float, cy_float, c='red', marker='+', s=100, 
                   linewidths=2, label='Calculated', zorder=5)
        
        ax.set_title(f"{test_case['description']}\nError: ({error_x:.2f}, {error_y:.2f})")
        ax.set_xlabel(f"Expected: {center}, Calc: ({cx_float:.1f}, {cy_float:.1f})")
        ax.legend(loc='upper right', fontsize=8)
        
        # 添加颜色条
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    
    # 保存大图
    output_path = output_dir / "centroid_test_overview.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # 为每个测试用例生成单独的高分辨率图像
    for test_case in test_cases:
        fig, ax = plt.subplots(figsize=(8, 8))
        
        size = test_case["size"]
        center = test_case["center"]
        sigma = test_case["sigma"]
        noise = test_case["noise"]
        
        # 创建高斯分布
        intensity = create_gaussian_2d((size, size), center, sigma)
        
        # 添加噪声
        if noise > 0:
            np.random.seed(idx)  # 为每个图使用不同的种子以获得不同的噪声模式
            noise_array = np.random.normal(0, noise, (size, size))
            intensity = np.clip(intensity + noise_array, 0, None)
        
        # 添加背景
        if "background" in test_case:
            intensity = intensity + test_case["background"]
        
        # 计算质心
        cx_float, cy_float = centroid(intensity, return_float=True)
        
        # 计算误差
        error_x = abs(cx_float - center[0])
        error_y = abs(cy_float - center[1])
        
        # 绘制图像
        im = ax.imshow(intensity, cmap='viridis', origin='lower')
        
        # 标记预期中心 (绿色x)
        ax.scatter(center[0], center[1], c='lime', marker='x', s=200, 
                   linewidths=3, label='Expected Center', zorder=5)
        
        # 标记计算质心 (红色+)
        ax.scatter(cx_float, cy_float, c='red', marker='+', s=200, 
                   linewidths=3, label='Calculated Centroid', zorder=5)
        
        # 添加标题和说明
        ax.set_title(f"Centroid Test: {test_case['description']}\n"
                     f"Size: {size}x{size}, Sigma: {sigma}, Noise: {noise}\n"
                     f"Expected: ({center[0]}, {center[1]}) | "
                     f"Calculated: ({cx_float:.2f}, {cy_float:.2f}) | "
                     f"Error: ({error_x:.2f}, {error_y:.2f})",
                     fontsize=10)
        
        ax.legend(loc='upper right', fontsize=10)
        plt.colorbar(im, ax=ax, label='Intensity')
        
        # 保存单独图像
        output_path = output_dir / f"centroid_{test_case['name']}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    # 生成测试报告
    report_path = output_dir / "centroid_test_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Centroid Test Visualization Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Test Results Summary\n\n")
        f.write("| Test Case | Expected (x, y) | Calculated (x, y) | Error (x, y) |\n")
        f.write("|-----------|------------------|--------------------| ------------|\n")
        
        for result in results:
            f.write(f"| {result['name']} | {result['expected']} | "
                    f"({result['calculated'][0]:.2f}, {result['calculated'][1]:.2f}) | "
                    f"({result['error_x']:.2f}, {result['error_y']:.2f}) |\n")
        
        f.write("\n## Visualizations\n\n")
        f.write("### Overview\n")
        f.write("![Overview](centroid_test_overview.png)\n\n")
        
        f.write("### Individual Tests\n")
        for test_case in test_cases:
            f.write(f"#### {test_case['description']}\n")
            f.write(f"![{test_case['description']}](centroid_{test_case['name']}.png)\n\n")
    
    print(f"\nReport saved: {report_path}")
    print(f"\nAll visualizations saved to: {output_dir}")
    
    return output_dir


if __name__ == "__main__":
    generate_test_visualization()
