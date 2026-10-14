# Centroid Test Visualization Report

<!-- provenance:start -->
> **生成脚本**: [`scripts/slm/generate_centroid_test_visualization.py`](../../scripts/slm/generate_centroid_test_visualization.py)
> **复现命令**: `python scripts/slm/generate_centroid_test_visualization.py`
> **运行环境**: 离线
> **说明**: 同目录 10 张 png 为本脚本产物
<!-- provenance:end -->

**本轮结论**：质心计算在 9 组测试用例（含偏移、噪声、不同 σ 高斯、阈值、150/200 大尺寸）上全部通过——无噪声用例误差为 0，噪声与阈值用例最大误差仅 1.12 px（高斯+噪声用例），全部亚像素级。质心估计器对噪声、光斑宽度与图像尺寸均稳健，满足波前斜率/倾斜检测精度要求。

Generated: 2026-03-25 13:07:59

## Test Results Summary

| Test Case | Expected (x, y) | Calculated (x, y) | Error (x, y) |
|-----------|------------------|--------------------| ------------|
| 中心位置 - 100x100 | (50, 50) | (50.00, 50.00) | (0.00, 0.00) |
| 偏移位置 - 100x100 | (30, 70) | (30.00, 70.00) | (0.00, 0.00) |
| 高斯+噪声 - 100x100 | (45, 55) | (45.89, 54.00) | (0.89, 1.12) |
| 较大噪声 - 120x120 | (60, 60) | (60.00, 60.00) | (0.00, 0.29) |
| 窄高斯(sigma=5) - 100x100 | (40, 60) | (40.00, 60.00) | (0.00, 0.00) |
| 宽高斯(sigma=30) - 100x100 | (50, 50) | (49.82, 50.00) | (0.18, 0.18) |
| 带阈值 - 100x100 | (50, 50) | (49.60, 50.00) | (0.40, 0.24) |
| 大尺寸 - 150x150 | (75, 80) | (74.95, 79.00) | (0.05, 0.86) |
| 大尺寸 - 200x200 | (100, 100) | (99.92, 100.00) | (0.08, 0.09) |

## Visualizations

### Overview
![Overview](centroid_test_overview.png)

### Individual Tests
#### 中心位置 - 100x100
![中心位置 - 100x100](centroid_centered_100x100.png)

#### 偏移位置 - 100x100
![偏移位置 - 100x100](centroid_shifted_100x100.png)

#### 高斯+噪声 - 100x100
![高斯+噪声 - 100x100](centroid_noise_100x100.png)

#### 较大噪声 - 120x120
![较大噪声 - 120x120](centroid_large_noise_120x120.png)

#### 窄高斯(sigma=5) - 100x100
![窄高斯(sigma=5) - 100x100](centroid_small_sigma_100x100.png)

#### 宽高斯(sigma=30) - 100x100
![宽高斯(sigma=30) - 100x100](centroid_large_sigma_100x100.png)

#### 带阈值 - 100x100
![带阈值 - 100x100](centroid_threshold_100x100.png)

#### 大尺寸 - 150x150
![大尺寸 - 150x150](centroid_150x150.png)

#### 大尺寸 - 200x200
![大尺寸 - 200x200](centroid_200x200.png)
