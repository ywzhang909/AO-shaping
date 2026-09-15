# Zernike 响应矩阵报告 (创建 / 分析 / 检测)

**生成时间**: 2026-09-16 00:10:13  
**数据源**: `zm_slm22030102_wfsM01219666_532nm_20260915_235207.h5`, `report_20260915_235549.json`, `raw_scan_20260915_235549.json`

## 1. 创建 (Creation)

响应矩阵由 `tools/slm/slm_zernike_response.py` 以**推拉法**标定: 对每个 SLM Zernike 模式施加 ±A 扰动, 测 WFS Zernike 读数增量, `matrix[wfs_coeff, slm_mode] = Δ(WFS)/Δ(SLM 幅度)`。

| 项 | 值 |
|---|---|
| 设备 | SLM #22030102 + WFS M01219666 |
| 波长 / 2π 灰度 | 532 nm / 998 |
| SLM shift | [106, 40] |
| Zernike 半径 | 250.0 px |
| 扰动幅度 | 5.00 rad (0.7958 λ) |
| 推拉循环 / 帧平均 | 2 / 2 |
| WFS pupil | center=(-0.155, 0.133) mm, diameter=(3.570, 3.786) mm |
| 参考平面 | custom user ref @ flat phase (shift 已应用) |
| 矩阵形状 | (66, 14) (WFS × SLM) |
| SLM 模式 (DLL 索引) | [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] |

> ⚠️ **索引约定**: 矩阵行 = DLL 系数索引 (**顺序 m 枚举, 非标准 Noll**); `[5](2,0)defocus` `[9](3,1)coma` `[13](4,0)spherical`。手册佐证 `roCMm` "derived from Zernike coefficient Z[5]"。

## 2. 分析 (Analysis)

### 2.1 响应矩阵热图

![matrix heatmap](figures/01_matrix_heatmap.png)

绿框 = 对角线 (同索引 SLM 模式 → WFS 系数)。

### 2.2 对角主导性

![diagonal](figures/02_diagonal_dominance.png)

**13/14** 个模式的列主导项即对角线项。

### 2.3 奇异值谱 / 条件数

![svd](figures/03_singular_values.png)

条件数 **5.12** (良态, 伪逆稳定)。

### 2.4 重复性方差

![variance](figures/04_variance_map.png)

平均方差 **3.838e-06**, 最大 **3.853e-04**。

### 2.5 线性度 (多尺寸扫描)

![linearity](figures/05_linearity.png)

**正确判据**: 响应已按单位幅度归一化 → 线性响应表现为 `|resp|` **恒定**, 故用幅度离散度 **CV** 与响应向量方向一致性 **cos** (原 slope/R² 判据在线性时 slope≈0, 无意义)。

| 模式 | 尺寸 R(px) | CV(%) | cos_min | 判定 |
|---|---|---|---|---|
| [2] (1,-1)tiltA | 200 | 0.3 | 0.979 | ✅ |
| [2] (1,-1)tiltA | 300 | 1.6 | 0.998 | ✅ |
| [2] (1,-1)tiltA | 400 | 2.2 | 0.997 | ✅ |
| [3] (1,1)tiltB | 200 | 6.3 | 0.994 | ✅ |
| [3] (1,1)tiltB | 300 | 1.6 | 0.998 | ✅ |
| [3] (1,1)tiltB | 400 | 6.0 | 0.996 | ✅ |
| [4] (2,-2)astig | 200 | 3.2 | 0.982 | ✅ |
| [4] (2,-2)astig | 300 | 2.3 | 0.997 | ✅ |
| [4] (2,-2)astig | 400 | 0.9 | 0.994 | ✅ |
| [5] (2,0)defocus | 200 | 102.8 | 0.479 | ❌ |
| [5] (2,0)defocus | 300 | 3.5 | 0.979 | ✅ |
| [5] (2,0)defocus | 400 | 4.6 | 0.988 | ✅ |
| [6] (2,2)astig | 200 | 14.4 | 0.565 | ❌ |
| [6] (2,2)astig | 300 | 1.7 | 0.997 | ✅ |
| [6] (2,2)astig | 400 | 5.1 | 0.994 | ✅ |
| [7] (3,-3)tf | 200 | 56.2 | 0.092 | ❌ |
| [7] (3,-3)tf | 300 | 6.5 | 0.994 | ✅ |
| [7] (3,-3)tf | 400 | 4.3 | 0.986 | ✅ |
| [8] (3,-1)coma | 200 | 95.8 | -0.670 | ❌ |
| [8] (3,-1)coma | 300 | 7.2 | 0.864 | ❌ |
| [8] (3,-1)coma | 400 | 1.0 | 0.998 | ✅ |
| [9] (3,1)coma | 200 | 141.3 | 0.321 | ❌ |
| [9] (3,1)coma | 300 | 2.6 | 0.993 | ✅ |
| [9] (3,1)coma | 400 | 1.7 | 0.999 | ✅ |
| [10] (3,3)tf | 200 | 13.5 | 0.939 | ✅ |
| [10] (3,3)tf | 300 | 4.4 | 0.995 | ✅ |
| [10] (3,3)tf | 400 | 8.8 | 0.975 | ✅ |
| [11] (4,-4)qf | 200 | 121.0 | 0.304 | ❌ |
| [11] (4,-4)qf | 300 | 6.9 | 0.993 | ✅ |
| [11] (4,-4)qf | 400 | 2.1 | 0.951 | ✅ |
| [12] (4,-2)2a | 200 | 141.0 | 0.410 | ❌ |
| [12] (4,-2)2a | 300 | 2.7 | 0.994 | ✅ |
| [12] (4,-2)2a | 400 | 1.9 | 0.997 | ✅ |
| [13] (4,0)spherical | 200 | 46.2 | -0.681 | ❌ |
| [13] (4,0)spherical | 300 | 5.7 | 0.987 | ✅ |
| [13] (4,0)spherical | 400 | 5.1 | 0.956 | ✅ |
| [14] (4,2)2a | 200 | 128.1 | -0.213 | ❌ |
| [14] (4,2)2a | 300 | 3.7 | 0.990 | ✅ |
| [14] (4,2)2a | 400 | 1.2 | 0.997 | ✅ |
| [15] (4,4)qf | 200 | 86.8 | 0.159 | ❌ |
| [15] (4,4)qf | 300 | 0.9 | 0.974 | ✅ |
| [15] (4,4)qf | 400 | 4.4 | 0.935 | ✅ |

## 3. 检测 (Detection)

### 3.1 异常点诊断

![outlier](figures/06_outlier_diagnosis.png)

**根因**: Zernike 半径 ≈ 光束半径 (实测 200px) 时, 大幅度高阶模式把子孔径光斑推出有效区 → DLL Zernike 拟合崩溃, `|resp|` 暴涨 (实测最高 4288)。故标定应取 **R ≈ 1.5×光束半径**。

### 3.2 离线反解验证

![inverse](figures/07_inverse_demo.png)

合成像差 `[5]defocus=+0.30λ, [9]coma=−0.20λ` → 反解 `c = pinv(M) @ w`, 残差 ‖Mc−w‖=0.0455 / ‖w‖=0.3606 → **降低 87.4%**。

### 3.3 实测闭环矫正

![closed loop](figures/08_closed_loop.png)

恢复 WFS 内部参考后, 矫正前 RMS=0.3493λ → 矫正后 0.3290λ (**5.8%**)。

> 该轮使用被 R=200 异常点污染的矩阵 (见 3.1), 改善有限; 修正后应以 R≈300px 重跑阶段 2/3。

## 4. 产物

| 产物 | 路径 |
|---|---|
| 响应矩阵 h5 | `D:\Projects\TIFO\AO-shaping\data\zernike_response_matrix\zm_slm22030102_wfsM01219666_532nm_20260915_235207.h5` |
| 矩阵 + 原始读数 json | `D:\Projects\TIFO\AO-shaping\data\zernike_response_matrix\zm_slm22030102_wfsM01219666_532nm_20260915_235207.json` |
| 多尺寸扫描报告 | `D:\Projects\TIFO\AO-shaping\data\zernike_correction\report_20260915_235549.json` |
| 多尺寸原始扫描 | `D:\Projects\TIFO\AO-shaping\data\zernike_correction\raw_scan_20260915_235549.json` |
| 标定工具 | `src/ao_shaping/tools/slm/slm_zernike_response.py` |
| 三阶段工具 | `src/ao_shaping/tools/slm/slm_zernike_correction.py` |
| 共享模块 | `src/ao_shaping/tools/slm/slm_zernike_common.py` |
