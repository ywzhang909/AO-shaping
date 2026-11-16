# 物理预测模型缺陷拟合分析 (M3 绝对保真度数据)

<!-- provenance:start -->
> **生成脚本**: [`scripts/slm/analyze_fwdmodel_fit.py`](../../../../../scripts/slm/analyze_fwdmodel_fit.py)
> **复现命令**: `python scripts/slm/analyze_fwdmodel_fit.py --out report/slm/differentiable_opt/artifacts/fwdmodel_fit_20261116`
> **数据/关联脚本**: [`scripts/_common/differentiable_problem.py`](../../../../../scripts/_common/differentiable_problem.py)
> **数据/关联脚本**: [`src/ao_shaping/algorithm/signal_processing/differentiable_shaping.py`](../../../../../src/ao_shaping/algorithm/signal_processing/differentiable_shaping.py)
> **运行环境**: 离线
> **说明**: 前向物理预测模型缺陷的离线拟合归因 (驱动无关, 只读已落盘的相位/实测/模型三联数据): 头号结论是冻结模型的光轴**陈旧 ~179 个 CCD 像素** (2026-11-07 M3 原点 1306,944 vs 2026-11-16 实测零级 1224,1105) —— 仅此一项把 Pearson 从 -0.003 拉到 0.95~0.98、验证集形状残差降 65%, **离焦不可辨识** (目标敏感度 1e-4), 993/1023 灰度假设**在方向上被证伪且量级上无关紧要** (3% 相位标度只动远场 NRMSE 0.012)。给出拟合的横向尺度/中心/模糊等全局量、验证集前后对照、联合优化可辨识性判定与补数据清单
<!-- provenance:end -->

- 生成时间 (UTC): `2026-11-16T18:23:23+00:00`
- 脚本: `scripts/slm/analyze_fwdmodel_fit.py`
- 离线契约: `sys.modules` 里的设备驱动包 = 无

## 结论速览

- **absolute_fidelity_12bit_fixed_20261116: 拟合集形状残差**: 冻结 5.9807 → 拟合 2.6734 (降 55.3 %)
- **absolute_fidelity_12bit_fixed_20261116: 验证集形状残差 (大核)**: 冻结 0.9955 → 拟合 0.3437 (降 65.5 %)
- **absolute_fidelity_12bit_fixed_20261116: 冻结模型零级错位**: 模型内容峰相对实测零级的偏移 (CCD px): 均值 178.8, 最大 181.6
- **absolute_fidelity_12bit_fixed_20261116: 参数可辨识性**: 目标函数约束得住: ['lateral_scale_x', 'centre_dx_px', 'centre_dy_px']; 约束不住 (不可辨识): ['defocus_delta_diopter', 'lateral_scale_y', 'phase_scale', 'blur_sigma_samples']
- **absolute_fidelity_12bit_fixed_20261116: 联合可辨识性**: 仅全局 2.6734 / 仅全频随机相位 5.7009 / 联合 2.6657; 联合后仍然稳定的全局参数: ['lateral_scale_x', 'lateral_scale_y', 'centre_dx_px', 'centre_dy_px', 'phase_scale']; 漂移最大的是 defocus_delta_diopter (0.290 个取值区间)
- **absolute_fidelity_12bit_fixed_20261116: 灰阶材料性**: 1023/993 过调制让远场形状的相关性只掉到 0.99988 (NRMSE 0.0121) ⇒ **灰阶在远场强度分布上几乎不材料**; 相对最优的残差增量 0.2262 (最强具名缺陷 gray_1023_over_993, 增量 0.2262) (⚠ 拟合的 phase_scale 顶在搜索边界上, 所以这些增量是**边界值之间的差**, 不是「灰阶比离焦更重要」的证据)
- **absolute_fidelity_20261116: 拟合集形状残差**: 冻结模型在拟合核里**没有信号** (代价退化 = 1e+06); 拟合后 3.0516
- **absolute_fidelity_20261116: 验证集形状残差 (大核)**: 冻结 nan → 拟合 nan (降 nan %)
- **absolute_fidelity_20261116: 冻结模型零级错位**: 模型内容峰相对实测零级的偏移 (CCD px): 均值 148.5, 最大 254.1
- **absolute_fidelity_20261116: 参数可辨识性**: 目标函数约束得住: ['lateral_scale_x', 'centre_dx_px', 'centre_dy_px', 'blur_sigma_samples']; 约束不住 (不可辨识): ['defocus_delta_diopter', 'lateral_scale_y', 'phase_scale']
- **absolute_fidelity_20261116: 灰阶材料性**: 1023/993 过调制让远场形状的相关性只掉到 0.99997 (NRMSE 0.0045) ⇒ **灰阶在远场强度分布上几乎不材料**; 相对最优的残差增量 8.712e-05 (最强具名缺陷 blur_minus_0p5_sample, 增量 0.008346) (⚠ 拟合的 phase_scale 顶在搜索边界上, 所以这些增量是**边界值之间的差**, 不是「灰阶比离焦更重要」的证据)

## 数据集 `absolute_fidelity_12bit_fixed_20261116`

| 项 | 值 |
| --- | --- |
| 来源 | real |
| 根目录 | D:\Projects\TIFO\AO-shaping\data\debug\slm_transfer\absolute_fidelity_12bit_fixed_20261116 |
| 零点 spot_xy (x, y) | [1224.0, 1105.0] |
| 曝光 (ms) | 0.034 |
| 采样窗 | [261, 261] @ step 2.0 px |
| 图样数 | 12 |
| 模型复算最大偏差 (counts) | 0 |
| 配对问题 | 无 |

### 拟合到的全局缺陷参数

| 参数 | 冻结值 | 拟合值 | 物理含义 |
| --- | --- | --- | --- |
| defocus_delta_diopter | 0 | 0.00841 | 模型总离焦 0.030897032469511032 + Δ D (屈光度) |
| lateral_scale_x | 1 | 1.05 | 模型内容相对实测窗的 x 向放大倍率 |
| lateral_scale_y | 1 | 0.7333 | 模型内容相对实测窗的 y 向放大倍率 |
| centre_dx_px | 0 | -72.19 | 模型相机网格相对实测窗的 x 向平移 (CCD px, 正 = 网格右挪) |
| centre_dy_px | 0 | 162.1 | 模型相机网格相对实测窗的 y 向平移 (CCD px, 正 = 网格下挪) |
| phase_scale | 1 | 0.85 | 命令相位 → 有效相位的乘子 (1.0302 = 993/1023 过调制) |
| blur_sigma_samples | 0 | 0.9065 | 模型窗上的高斯展宽 (采样点; = 1.81 CCD px) |

- 拟合集形状残差合计: 冻结 **5.981** → 拟合 **2.673** (下降 55.3 %)
- ⚠ 拟合值顶到搜索边界的参数: `['phase_scale']` (这些参数的数值只是边界值, 不是真值)

### 逐图样: 冻结模型 vs 拟合模型

| 图样 | 集合 | 残差(冻结) | 残差(拟合) | Pearson(冻结) | Pearson(拟合) | FWHM比x(冻结) | FWHM比x(拟合) | FWHM比y(冻结) | FWHM比y(拟合) | 峰值偏移px(冻结) | 峰值偏移px(拟合) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zero | fit | 0.9972 | 0.22 | -0.003429 | 0.9754 | 1.312 | 1.301 | 0.651 | 1.257 | 178.9 | 0 |
| rand_rms0p5_seed0 | holdout | 0.9969 | 0.3608 | -0.003268 | 0.9322 | 1.369 | 1.357 | 0.6419 | 1.245 | 180.8 | 2 |
| rand_rms0p5_seed1 | holdout | 0.9968 | 0.2643 | -0.003377 | 0.9642 | 1.344 | 1.333 | 0.6397 | 1.234 | 180.8 | 2 |
| rand_rms1p5_seed0 | holdout | 0.9941 | 0.3268 | 0.007098 | 0.9444 | 1.511 | 1.473 | 0.6141 | 1.245 | 180.8 | 2 |
| rand_rms1p5_seed1 | holdout | 0.994 | 0.4059 | 0.003056 | 0.9128 | 1.532 | 1.488 | 0.6208 | 1.206 | 180.8 | 4 |
| anchored_m3_best | holdout | 0.9971 | 0.2946 | -0.003416 | 0.9553 | 1.399 | 1.387 | 0.6009 | 1.136 | 180.8 | 2 |
| anchored_m3_rs15_best | holdout | 0.994 | 0.4096 | -0.005808 | 0.9111 | 1.573 | 1.537 | 0.6138 | 1.168 | 181.6 | 2 |
| tilt_x_1rad | fit | 0.9968 | 0.2257 | -0.003585 | 0.974 | 1.323 | 1.311 | 0.6236 | 1.196 | 177.1 | 2 |
| tilt_x_3rad | fit | 0.9964 | 0.7496 | -0.003722 | 0.6589 | 1.374 | 1.363 | 0.4823 | 0.9218 | 169.9 | 10 |
| tilt_y_1rad | fit | 0.9968 | 0.352 | -0.003592 | 0.9356 | 1.314 | 1.308 | 0.6126 | 1.182 | 177.3 | 4 |
| tilt_y_3rad | fit | 0.9966 | 0.7098 | -0.003758 | 0.702 | 0.9961 | 0.9894 | 0.6605 | 1.271 | 176.8 | 10.2 |
| defocus_z4_1rad | fit | 0.9968 | 0.4164 | -0.003211 | 0.9086 | 1.153 | 1.143 | 0.5235 | 1.005 | 180.7 | 4 |

> 残差与相关性在**大核** (拟合核, 半径 240 CCD px) 上算; 小核 (半径 100 CCD px, 罩住实测零级) 在冻结模型上是空的 —— 冻结模型的零级根本不在小核里, 那一列会全是 NaN。

### 可辨识性 (最优点的代价敏感度)

- 敏感度 (代价能被她整个取值范围改变多少): {'defocus_delta_diopter': '9.573e-05', 'lateral_scale_x': '0.7643', 'lateral_scale_y': '0.1248', 'centre_dx_px': '3.575', 'centre_dy_px': '15.01', 'phase_scale': '0.3881', 'blur_sigma_samples': '0.02055'}
- 目标函数约束得住: `['lateral_scale_x', 'centre_dx_px', 'centre_dy_px']`
- 目标函数约束不住 (不可辨识): `['defocus_delta_diopter', 'lateral_scale_y', 'phase_scale', 'blur_sigma_samples']`

### 联合优化 (低阶相位 + 全局参数)

| 项 | 值 |
| --- | --- |
| 相位基 (Noll) | n ≤ 3 (9 个模态) |
| 仅全局参数 (拟合集) 残差 | 2.673 |
| 仅全频随机相位 (全局冻结) 残差 | 5.701 |
| 相位 + 全局 联合 残差 | 2.666 |
| 联合后 defocus_delta_diopter | 0.02 |
| 联合后 lateral_scale_x | 1.05 |
| 联合后 lateral_scale_y | 0.7333 |
| 联合后 centre_dx_px | -72.19 |
| 联合后 centre_dy_px | 162.1 |
| 联合后 phase_scale | 0.85 |
| 联合后 blur_sigma_samples | 0 |
| 残差相位系数 RMS (rad) | 0.1715 |

### 灰阶 / 离焦材料性

| 具名假设 | 形状残差合计 | 相对最优的增量 |
| --- | --- | --- |
| gray_1023_over_993 | 2.949 | 0.2262 |
| gray_993_over_1023 | 2.876 | 0.1533 |
| gray_frozen_1p0 | 2.913 | 0.1894 |
| defocus_plus_0p002D | 2.698 | -0.02504 |
| defocus_minus_0p002D | 2.756 | 0.03251 |
| scale_x_plus_1pct | 2.797 | 0.07403 |
| scale_x_minus_1pct | 2.812 | 0.08857 |
| scale_y_plus_1pct | 2.805 | 0.08213 |
| scale_y_minus_1pct | 2.804 | 0.08038 |
| blur_plus_0p5_sample | 2.709 | -0.01462 |
| blur_minus_0p5_sample | 2.743 | 0.01961 |

- 灰阶过调制因子 1023/993 = 1.030211
- 等效相位误差: {'rms_0.5rad_phase_error': 0.015105740181268867, 'rms_1.5rad_phase_error': 0.0453172205438066, 'rms_3.0rad_phase_error': 0.0906344410876132}

## 数据集 `absolute_fidelity_20261116`

| 项 | 值 |
| --- | --- |
| 来源 | real |
| 根目录 | D:\Projects\TIFO\AO-shaping\data\debug\slm_transfer\absolute_fidelity_20261116 |
| 零点 spot_xy (x, y) | [1220.0, 606.0] |
| 曝光 (ms) | 1000 |
| 采样窗 | [261, 261] @ step 2.0 px |
| 图样数 | 12 |
| 模型复算最大偏差 (counts) | 0 |
| 配对问题 | 无 |

### 拟合到的全局缺陷参数

| 参数 | 冻结值 | 拟合值 | 物理含义 |
| --- | --- | --- | --- |
| defocus_delta_diopter | 0 | 0.02 | 模型总离焦 0.030897032469511032 + Δ D (屈光度) |
| lateral_scale_x | 1 | 0.7 | 模型内容相对实测窗的 x 向放大倍率 |
| lateral_scale_y | 1 | 1.575 | 模型内容相对实测窗的 y 向放大倍率 |
| centre_dx_px | 0 | -98.32 | 模型相机网格相对实测窗的 x 向平移 (CCD px, 正 = 网格右挪) |
| centre_dy_px | 0 | -25.89 | 模型相机网格相对实测窗的 y 向平移 (CCD px, 正 = 网格下挪) |
| phase_scale | 1 | 0.85 | 命令相位 → 有效相位的乘子 (1.0302 = 993/1023 过调制) |
| blur_sigma_samples | 0 | 6 | 模型窗上的高斯展宽 (采样点; = 12 CCD px) |

- 拟合集形状残差合计: 冻结 **1e+06** → 拟合 **3.052** (下降 100 %)
- ⚠ 拟合值顶到搜索边界的参数: `['defocus_delta_diopter', 'lateral_scale_x', 'phase_scale', 'blur_sigma_samples']` (这些参数的数值只是边界值, 不是真值)

### 逐图样: 冻结模型 vs 拟合模型

| 图样 | 集合 | 残差(冻结) | 残差(拟合) | Pearson(冻结) | Pearson(拟合) | FWHM比x(冻结) | FWHM比x(拟合) | FWHM比y(冻结) | FWHM比y(拟合) | 峰值偏移px(冻结) | 峰值偏移px(拟合) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zero | fit | nan | 0.5957 | 0.2767 | 0.1628 | 1.206 | 0.09691 | 2.022 | 0.1202 | 180.7 | 132.5 |
| rand_rms0p5_seed0 | holdout | nan | 0.57 | 0.4838 | 0.1798 | 1.248 | 0.101 | 1.8 | 0.1078 | 220.6 | 125.2 |
| rand_rms0p5_seed1 | holdout | nan | 0.5662 | 0.5379 | 0.1882 | 1.232 | 0.09949 | 1.811 | 0.1083 | 107.7 | 110.1 |
| rand_rms1p5_seed0 | holdout | nan | nan | 0.49 | 0.3422 | 1.035 | 0.09618 | 1 | 0.06916 | 140.8 | 96.54 |
| rand_rms1p5_seed1 | holdout | nan | nan | 0.4882 | 0.3192 | 1 | 0.0906 | 1.043 | 0.07003 | 70.11 | 114.4 |
| anchored_m3_best | holdout | nan | 0.5515 | 0.2916 | 0.2178 | 1.088 | 0.08742 | 2.382 | 0.1408 | 147.9 | 76.55 |
| anchored_m3_rs15_best | holdout | nan | 0.4257 | 0.4974 | 0.2883 | 1.011 | 0.09072 | 1.088 | 0.07237 | 254.1 | 67.68 |
| tilt_x_1rad | fit | nan | 0.5199 | 0.4102 | 0.2276 | 1 | 0.08038 | 1.657 | 0.0983 | 120.8 | 55.57 |
| tilt_x_3rad | fit | nan | 0.5179 | 0.3276 | 0.2256 | 1 | 0.08039 | 1.647 | 0.09779 | 126.1 | 59.93 |
| tilt_y_1rad | fit | nan | 0.4906 | 0.2933 | 0.229 | 1 | 0.08038 | 1.639 | 0.09739 | 112.2 | 41.23 |
| tilt_y_3rad | fit | nan | 0.4679 | 0.3149 | 0.2261 | 1 | 0.08038 | 1.662 | 0.09872 | 210.2 | 58.14 |
| defocus_z4_1rad | fit | nan | 0.4597 | 0.3387 | 0.2305 | 1 | 0.07971 | 1.662 | 0.09688 | 90.97 | 29.53 |

> 残差与相关性在**大核** (拟合核, 半径 240 CCD px) 上算; 小核 (半径 100 CCD px, 罩住实测零级) 在冻结模型上是空的 —— 冻结模型的零级根本不在小核里, 那一列会全是 NaN。

### 可辨识性 (最优点的代价敏感度)

- 敏感度 (代价能被她整个取值范围改变多少): {'defocus_delta_diopter': '0.004734', 'lateral_scale_x': '0.06713', 'lateral_scale_y': '0.001199', 'centre_dx_px': '0.01711', 'centre_dy_px': '0.01993', 'phase_scale': '0.0001453', 'blur_sigma_samples': '0.1027'}
- 目标函数约束得住: `['lateral_scale_x', 'centre_dx_px', 'centre_dy_px', 'blur_sigma_samples']`
- 目标函数约束不住 (不可辨识): `['defocus_delta_diopter', 'lateral_scale_y', 'phase_scale']`

### 灰阶 / 离焦材料性

| 具名假设 | 形状残差合计 | 相对最优的增量 |
| --- | --- | --- |
| gray_1023_over_993 | 3.052 | 8.712e-05 |
| gray_993_over_1023 | 3.052 | 5.837e-05 |
| gray_frozen_1p0 | 3.052 | 7.253e-05 |
| defocus_plus_0p002D | 3.051 | -0.0002414 |
| defocus_minus_0p002D | 3.052 | 0.0002346 |
| scale_x_plus_1pct | 3.052 | 0.0005646 |
| scale_x_minus_1pct | 3.051 | -0.0001995 |
| scale_y_plus_1pct | 3.052 | 0.0002003 |
| scale_y_minus_1pct | 3.052 | 0.0002273 |
| blur_plus_0p5_sample | 3.043 | -0.008921 |
| blur_minus_0p5_sample | 3.06 | 0.008346 |

- 灰阶过调制因子 1023/993 = 1.030211
- 等效相位误差: {'rms_0.5rad_phase_error': 0.015105740181268867, 'rms_1.5rad_phase_error': 0.0453172205438066, 'rms_3.0rad_phase_error': 0.0906344410876132}

## 最小额外数据需求 (要把剩下的失配也吃掉)

1. **同会话的零级位置 / 模型原点**: 冻结检查点的 `origin_xy` 来自 2026-11-07 的
   M3 照明拟合, 而采集是 2026-11-16, 期间零级漂了 ~180 px。只要在**同一次会话**
   里重跑一次零级定心 (probe 已经用同帧定心), 或把零级像素坐标一起存进检查点,
   这项失配就完全消失 —— 不需要任何新硬件数据, 只需要**同会话**的零点坐标。
2. **离焦**: 这一批数据对离焦**不敏感** (代价对 ΔD 几乎无梯度), 所以拟合不出
   来。离焦必须靠**独立的焦点扫描** (改变 z 位移, 量 FWHM vs 焦点) 或者
   `phys_20261116_01/focus_sweep.csv` 那类数据来定, 不能靠远场强度形状反演。
3. **照明形状**: 冻结模型假设面板上的照明是高斯 (width 383×142 panel px, 各向异性
   2.7:1), 而实测零级是**圆**的 (13.8×13.6 px) 而模型是 **20×10 px 的椭圆**。
   要解释这个, 需要同会话量一次**面板上的光斑形状** (或者干脆把照明振幅从
   「各向异性高斯」换成实测形状)。
4. **灰阶**: 3% 的相位标度差在远场强度分布上只有 ~1% (NRMSE), 而零级错位是
   100% (Pearson 从 0 到 0.97)。所以灰阶**不需要**额外数据, 但也**测不出来** ——
   要测它得做**斑点级**的对比 (同一张图上零级之外的散斑), 而不是零级轮廓。

## 产物

- `fwdmodel_fit.json`
- `fwdmodel_fit.md`
- `run.log`
- `analyze_fwdmodel_fit.py`
- `fwdfit_radial_profiles.png`
- `fwdfit_marginal_curves.png`
- `fwdfit_cost_maps.png`
- `fwdfit_holdout_images.png`
- `fwdfit_gray_materiality.png`
