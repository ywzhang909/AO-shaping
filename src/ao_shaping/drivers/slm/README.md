# SLM响应标定

本模块提供多种SLM（空间光调制器相位-灰度）标定方法，用于确定SLM的相位-灰度响应特性。

## 目录

- [概述](#概述)
- [标定方法](#标定方法)
  - [闪耀光栅法](#闪耀光栅法)
  - [干涉法](#干涉法)
  - [衍射效率法](#衍射效率法)
- [光路设置](#光路设置)
- [自动曝光功能](#自动曝光功能)
- [使用示例](#使用示例)
- [可能遇到的问题及解决方法](#可能遇到的问题及解决方法)

## 概述

SLM是一种用于调制光波前的器件，其相位调制能力由液晶材料决定。不同灰度值对应不同的相位延迟，因此需要对SLM进行标定，建立灰度值与相位之间的准确对应关系。

典型的2π相位对应的灰度值范围在200-1023之间，具体取决于SLM型号和工作波长。

## 标定方法

### 闪耀光栅法（Blazed Grating Method）

#### 原理

闪耀光栅法利用衍射原理进行标定：

1. 在SLM上显示线性相位梯度（闪耀光栅），相位从0增加到φ
2. 光栅将入射光衍射到特定方向（一级衍射）
3. 根据衍射理论，当相位深度φ = 2π时，一级衍射效率达到最大值
4. 通过扫描不同灰度值，找到最大衍射效率点，该灰度值即为2π对应的灰度值

**衍射效率公式：**

$\eta_1 = \left(\frac{2 \cdot J_1(\phi)}{\phi}\right)^2$

其中$J_1$是第一类贝塞尔函数，$\phi$是相位深度。

当$\phi = 2\pi$时，$J_1(2\pi) \approx 0.339$，衍射效率达到理论最大值约11.4%。

#### 优点

- 装置简单，只需SLM、相机和激光光源
- 不需要参考光路
- 测量相对强度，无需绝对校准

#### 缺点

- 需要精确对准衍射光斑
- 对环境振动敏感

### 零级光强比值法（Zero-Order Ratio Method）

#### 原理

零级光强比值法通过比较有无光栅时的零级光强来确定2π相位：

1. 先测量全0相位（无光栅）时的零级光强作为参考 $I_0$
2. 然后测量不同灰度深度的光栅图案时的零级光强 $I$
3. 计算比值 $\eta = I / I_0$
4. 当相位深度为2π时，一级衍射最强，零级光强最低（比值最小）
5. 找到零级比值最小的点，即为2π相位

**物理原理：**
- 全0相位：光直接通过，零级光强最大
- 闪耀光栅：部分光被衍射到一级，零级光强降低
- 2π相位：衍射到一级的效率最高（约11.4%），零级光强最低

#### 优点

- 消除光源功率波动影响（使用比值）
- 只需测量零级光强，无需对准一级衍射
- 测量更稳定，重复性好
- 光路调试相对简单

#### 缺点

- 每次测量需要切换全0和光栅图案，耗时较长
- 需要SLM支持快速切换

#### 使用方法

```python
# 使用零级光强比值法标定
result = calibrator.calibrate_with_zero_order_ratio(
    grayscale_range=(100, 1023),
    step=10,
    n_samples=3
)
```

### 干涉法（Interferometer Method）

#### 原理

干涉法利用光的干涉效应：

1. 构建马赫-泽德干涉仪或类似干涉系统
2. SLM调制光与参考光进行干涉
3. 改变SLM上的灰度值，观察干涉条纹变化
4. 从干涉条纹的移动量计算相位变化

**相位计算：**

$$\Delta\phi = \frac{2\pi \cdot \Delta x}{d}$$

其中$\Delta x$是条纹移动距离，$d$是条纹间距。

#### 优点

- 精度高，可达到λ/10甚至更高
- 可测量完整的2π相位范围

#### 缺点

- 需要稳定的干涉仪装置
- 对环境要求高（防振、隔热）
- 光路调试复杂

### 衍射效率法（Diffraction Efficiency Method）

#### 原理

通过测量一级衍射光相对于零级（透射光）的效率变化：

1. 在SLM上显示周期性光栅图案
2. 测量一级衍射光强度
3. 测量零级（未衍射光）强度作为参考
4. 计算衍射效率：$\eta = I_1 / I_0$
5. 根据贝塞尔函数拟合得到相位-灰度曲线

#### 优点

- 相比闪耀光栅法，使用相对效率更稳定
- 可以获得完整的响应曲线

#### 缺点

- 需要精确测量零级和一级光强
- 计算相对复杂

## 光路设置

### 闪耀光栅法光路

```
激光器 → 扩束器 → SLM → 透镜 → 孔径光阑 → CCD/相机
```

**关键点：**

1. **扩束器**：将激光束扩展至SLM尺寸，确保均匀照明
2. **SLM**：相位调制面朝向光路
3. **透镜**：傅里叶变换透镜，将SLM的远场衍射图案成像到相机
4. **孔径光阑**：阻挡高级次衍射，只允许一级衍射通过
5. **相机**：放置在透镜焦平面上，采集衍射光斑

### 推荐参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 光栅周期 | 8-16像素 | 太短会产生高级次衍射，太长则光斑分散 |
| ROI大小 | 50×50~100×100像素 | 根据光斑大小调整 |
| 曝光时间 | 10-100ms | 根据光强调整，避免饱和 |
| 采样次数 | 3-5次 | 平均以降低噪声 |

## 自动曝光功能

模块提供`AutoExposureController`类用于自动调整相机曝光：

```python
from ao_shaping.drivers.slm.slm_calibration import AutoExposureController

# 创建自动曝光控制器
auto_expo = AutoExposureController(
    camera=camera,
    target_min=80,    # 目标最小灰度
    target_max=220,   # 目标最大灰度
    min_exposure=1,   # 最小曝光时间
    max_exposure=1000 # 最大曝光时间
)

# 自动调整曝光
optimal_exposure = auto_expo.auto_adjust(n_samples=3)
```

**工作原理：**

1. 拍摄初始图像，检测最大灰度值
2. 如果饱和（>220），降低曝光时间
3. 如果信号过弱（<80），增加曝光时间
4. 重复迭代直到达到目标范围

## 使用示例

### 闪耀光栅法标定（推荐）

```python
from ao_shaping.drivers.slm.santec_slm200 import SantecSLM200
from ao_shaping.drivers.slm.slm_calibration import SantecSLM200Calibrator
from ao_shaping.drivers.ccd.daheng import CameraStreamManager

# 连接设备
with SantecSLM200(slm_number=1) as slm:
    slm.set_wavelength(1064, 200)
    
    with CameraStreamManager(cam_id=0, exposure_time_ms=50) as camera:
        # 创建标定器
        calibrator = SantecSLM200Calibrator(
            slm=slm,
            camera=camera,
            grating_period=8
        )
        
        # 带自动曝光的标定
        result = calibrator.calibrate_with_auto_exposure(
            grayscale_range=(100, 1023),
            step=10,
            n_samples=3,
            auto_exposure=True
        )
        
        # 保存结果
        calibrator.save_calibration('calibration_result.json')
        
        # 绘制曲线
        from ao_shaping.drivers.slm.slm_calibration import plot_calibration_result
        plot_calibration_result(result)
```

### 零级光强比值法标定（更稳定）

```python
from ao_shaping.drivers.slm.santec_slm200 import SantecSLM200
from ao_shaping.drivers.slm.slm_calibration import SantecSLM200Calibrator
from ao_shaping.drivers.ccd.daheng import CameraStreamManager

with SantecSLM200(slm_number=1) as slm:
    slm.set_wavelength(1064, 200)
    
    with CameraStreamManager(cam_id=0, exposure_time_ms=50) as camera:
        calibrator = SantecSLM200Calibrator(
            slm=slm,
            camera=camera,
            grating_period=8
        )
        
        # 使用零级光强比值法标定
        result = calibrator.calibrate_with_zero_order_ratio(
            grayscale_range=(100, 1023),
            step=10,
            n_samples=3,
            fine_search=True
        )
        
        calibrator.save_calibration('zero_order_calibration.json')
```

### 使用其他标定方法

```python
# 干涉法标定
from ao_shaping.drivers.slm.slm_calibration import InterferometerCalibrator

interferometer_calib = InterferometerCalibrator(
    slm=slm,
    camera=camera,
    roi_center=(512, 512),
    roi_size=(100, 100)
)
result = interferometer_calib.calibrate()

# 衍射效率法标定
from ao_shaping.drivers.slm.slm_calibration import DiffractionEfficiencyCalibrator

diffraction_calib = DiffractionEfficiencyCalibrator(
    slm=slm,
    camera=camera,
    grating_period=16,
    roi_center=(640, 512),
    roi_size=(50, 50)
)
result = diffraction_calib.calibrate()
```

## 可能遇到的问题及解决方法

### 1. 衍射光斑不稳定

**现象：** 测量过程中衍射光斑位置或强度波动较大

**可能原因：**
- 激光器功率不稳定
- 环境振动
- SLM响应延迟

**解决方法：**
- 使用功率稳定的激光器
- 减少环境振动（光学平台、防振台）
- 增加采样次数取平均
- 等待SLM稳定（设置合适的延迟时间）

### 2. 找不到衍射光斑

**现象：** 相机上看不到一级衍射光斑

**可能原因：**
- 光栅周期设置不当
- 光路未对准
- 曝光时间不合适

**解决方法：**
- 尝试不同的光栅周期（8、12、16像素）
- 调整光路，确保激光垂直入射SLM
- 先用较长曝光时间找到光斑位置
- 使用最大灰度值(1023)测试，应该能看到最亮的衍射光斑

### 3. 标定曲线有多个峰值

**现象：** 衍射效率曲线出现多个局部最大值

**可能原因：**
- 高级次衍射进入ROI
- 光斑大小超过ROI
- 背景光干扰

**解决方法：**
- 减小光栅周期，避免高级次衍射
- 增大ROI尺寸或调整ROI位置
- 测量并扣除背景光
- 使用更小的孔径光阑阻挡杂散光

### 4. 标定结果与预期差异大

**现象：** 2π相位对应灰度值与参考值偏差大

**可能原因：**
- 波长设置错误
- 温度影响（液晶特性随温度变化）
- SLM非线性响应

**解决方法：**
- 确认激光波长与SLM波长设置一致
- 在稳定温度环境下进行标定
- 进行多点标定（不同波长、不同温度）
- 考虑使用多项式拟合获得更精确的响应曲线

### 5. 图像饱和

**现象：** 相机图像最大灰度值达到255

**原因：** 曝光时间过长

**解决方法：**
- 使用自动曝光功能
- 手动降低曝光时间
- 减小激光功率
- 在SLM前添加衰减片

### 6. 标定结果重复性差

**现象：** 多次标定结果不一致

**可能原因：**
- 设备连接不稳定
- 环境条件变化
- SLM初始化不充分

**解决方法：**
- 预热设备（SLM、激光器）
- 多次测量取平均
- 记录并控制环境参数（温度、湿度）
- SLM初始化后等待足够时间

## 文件说明

| 文件 | 说明 |
|------|------|
| `slm_calibration.py` | 标定模块主文件 |
| `slm_calibration.py::CalibrationResult` | 标定结果数据类 |
| `slm_calibration.py::SantecSLM200Calibrator` | 闪耀光栅法/零级比值法标定器 |
| `slm_calibration.py::InterferometerCalibrator` | 干涉法标定器 |
| `slm_calibration.py::DiffractionEfficiencyCalibrator` | 衍射效率法标定器 |
| `slm_calibration.py::AutoExposureController` | 自动曝光控制器 |
| `slm_calibration.py::SLMCalibratorBase.measure_zero_order_ratio()` | 零级光强比值测量方法 |

## 参考资料

1. E. G. v. d. L. B. W. H. B. Johnson, "Phase-only modulation with twisted nematic liquid crystal displays," Opt. Commun. 156, 199-203 (1998)
2. J. W. Goodman, Introduction to Fourier Optics, 3rd ed. (Roberts & Company, 2005)
