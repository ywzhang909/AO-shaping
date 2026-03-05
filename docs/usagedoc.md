# 使用文档

**项目名称:** AO-shaping  
**文档版本:** 1.0.0  
**创建日期:** 2026-03-05  
**最后更新:** 2026-03-05

---

## 目录

1. [安装配置](#1-安装配置)
2. [快速开始](#2-快速开始)
3. [硬件驱动使用](#3-硬件驱动使用)
4. [波前控制使用](#4-波前控制使用)
5. [无波前传感使用](#5-无波前传感使用)
6. [工具模块使用](#6-工具模块使用)
7. [可视化界面](#7-可视化界面)
8. [常见用例](#8-常见用例)

---

## 1. 安装配置

### 1.1 环境要求

| 要求 | 最低版本 |
|-----|---------|
| Python | 3.13 |
| CUDA | 11.8 (GPU 支持) |
| 操作系统 | Windows/Linux |

### 1.2 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/[organization]/AO-shaping.git
cd AO-shaping

# 2. 使用 uv 安装依赖
uv sync

# 3. 验证安装
python -c "import ao_shaping; print('Installation successful')"
```

### 1.3 数据目录挂载

项目需要访问数据存储：

```bash
# Linux/macOS
sudo mount -t cifs -o user=tifo,password=TIFO1234,uid=tifo,gid=tifo,iocharset=utf8,vers=3.0 //10.10.0.53/storage/AO_data data

# Windows
# 使用映射网络驱动器
```

---

## 2. 快速开始

### 2.1 基本工作流

```python
# 1. 导入所需模块
from ao_shaping.drivers.slm import SantecSLM200
from ao_shaping.drivers.dm import NLightDM
from ao_shaping.drivers.ccd import MiiCAM
from ao_shaping.wfless.gready_cam import GreedyCAM

# 2. 初始化硬件
with SantecSLM200() as slm:
    with MiiCAM() as ccd:
        # 3. 创建优化器
        optimizer = GreedyCAM(slm, ccd)
        
        # 4. 运行优化
        result = optimizer.optimize(iterations=100)
        
        print(f"优化完成: {result}")
```

### 2.2 波前控制工作流

```python
from ao_shaping.drivers.dm import NLightDM
from ao_shaping.drivers.wfs import ThorlabsWFS
from ao_shaping.wf.DM_wfs import DMWFSController

# 初始化硬件
with NLightDM() as dm:
    with ThorlabsWFS() as wfs:
        # 创建控制器
        controller = DMWFSController(dm, wfs)
        
        # 校准
        controller.calibrate()
        
        # 波前校正
        controller.correct(accuracy=0.01)
```

---

## 3. 硬件驱动使用

### 3.1 空间光调制器 (SLM)

#### Santec SLM-200

```python
from ao_shaping.drivers.slm import SantecSLM200
import numpy as np

# 初始化
slm = SantecSLM200()
slm.open()

# 加载图案
phase_pattern = np.random.rand(1920, 1080) * 2 * np.pi
slm.write_pattern(phase_pattern)

# 获取状态
status = slm.get_status()
print(f"SLM 状态: {status}")

# 关闭
slm.close()

# 上下文管理器方式
with SantecSLM200() as slm:
    slm.write_pattern(phase_pattern)
# 自动关闭
```

#### 图案辅助工具

```python
from ao_shaping.drivers.slm import SLMPatternHelper
import numpy as np

# 创建光栅图案
grating = SLMPatternHelper.grating(angle=10, period=50)

# 创建闪耀光栅
blazed = SLMPatternHelper.blazed_grating(blaze_angle=15)

# 创建随机相位
random_phase = SLMPatternHelper.random_phase(resolution=(1920, 1080))

# 生成 GS 算法图案
from ao_shaping.utils.phase_patterns import gs_iterate
target = np.random.rand(512, 512)
gs_pattern = gs_iterate(target, iterations=50)
```

#### SLM 校准

```python
from ao_shaping.drivers.slm import SLMCalibration

calibrator = SLMCalibration(slm, ccd)
calibration_data = calibrator.calibrate()

# 保存校准数据
calibrator.save("slm_calibration.json")

# 加载校准数据
calibrator.load("slm_calibration.json")
```

### 3.2 变形镜 (DM)

#### NLight DM

```python
from ao_shaping.drivers.dm import NLightDM
import numpy as np

# 初始化
dm = NLightDM()
dm.open()

# 获取 DM 信息
info = dm.get_info()
print(f"驱动器数量: {info['actuators']}")

# 设置电压
voltages = np.random.rand(info['actuators']) * 5.0  # 0-5V
dm.set_voltage(voltages)

# 获取当前状态
current = dm.get_voltage()

# 关闭
dm.close()

# 上下文管理器
with NLightDM() as dm:
    dm.set_voltage(voltages)
```

#### 模拟 DM

用于无硬件时的测试：

```python
from ao_shaping.drivers.dm import SimulateDM

# 创建模拟 DM
dm = SimulateDM(n_actuators=97)

# 设置电压
dm.set_voltage(voltages)

# 模拟响应
response = dm.get_response()
```

### 3.3 波前传感器 (WFS)

#### Thorlabs WFS

```python
from ao_shaping.drivers.wfs import ThorlabsWFS

# 初始化
wfs = ThorlabsWFS()
wfs.open()

# 获取波前
wavefront = wfs.get_wavefront()
print(f"RMS: {wavefront.rms} waves")

# 获取 Zernike 系数
coefficients = wfs.get_zernike()
print(f"Zernike 系数: {coefficients}")

# 关闭
wfs.close()
```

### 3.4 CCD/相机

#### Mii CAM

```python
from ao_shaping.drivers.ccd import MiiCAM
import numpy as np

# 初始化
ccd = MiiCAM()
ccd.open()

# 配置
ccd.set_exposure(0.001)  # 1ms
ccd.set_gain(1.0)

# 采集图像
image = ccd.capture()

print(f"图像尺寸: {image.shape}")
print(f"最大强度: {image.max()}")

# 连续采集
for i in range(10):
    frame = ccd.capture()
    # 处理 frame

# 关闭
ccd.close()
```

#### 大恒相机

```python
from ao_shaping.drivers.ccd import DahengCamera

ccd = DahengCamera()
ccd.open()
ccd.set_exposure(0.005)
image = ccd.capture()
ccd.close()
```

### 3.5 温度管理

```python
from ao_shaping.drivers.tm import SerialPortFSM

# 初始化温度控制器
tm = SerialPortFSM(port="COM3")
tm.open()

# 设置目标温度
tm.set_temperature(25.0)

# 获取当前温度
current_temp = tm.get_temperature()
print(f"当前温度: {current_temp}°C")

# 启动温度稳定
tm.enable_control(True)

tm.close()
```

---

## 4. 波前控制使用

### 4.1 DM 波前传感控制

```python
from ao_shaping.drivers.dm import NLightDM
from ao_shaping.drivers.wfs import ThorlabsWFS
from ao_shaping.wf.DM_wfs import DMWFSController
import numpy as np

# 初始化硬件
with NLightDM() as dm, ThorlabsWFS() as wfs:
    # 创建控制器
    controller = DMWFSController(dm, wfs)
    
    # 校准（建立响应矩阵）
    controller.calibrate(response_matrix=None)  # 或提供预计算的响应矩阵
    
    # 执行波前校正
    result = controller.correct(
        target_rms=0.01,      # 目标 RMS
        max_iterations=50,    # 最大迭代次数
        convergence=0.001     # 收敛阈值
    )
    
    print(f"校正结果: RMS = {result['final_rms']}")
    print(f"迭代次数: {result['iterations']}")
```

### 4.2 线性回归波前传感

```python
from ao_shaping.drivers.dm import NLightDM
from ao_shaping.drivers.ccd import MiiCAM
from ao_shaping.wf.lr_wfs import LrWFS

with NLightDM() as dm, MiiCAM() as ccd:
    # 创建控制器
    wfs = LrWFS(dm, ccd)
    
    # 校准
    wfs.calibrate(n_samples=100)
    
    # 校正
    wfs.correct(target_rms=0.02)
```

### 4.3 强化学习波前传感

```python
from ao_shaping.drivers.dm import NLightDM
from ao_shaping.drivers.ccd import MiiCAM
from ao_shaping.wf.rl_wfs import RLWFS

with NLightDM() as dm, MiiCAM() as ccd:
    # 创建强化学习控制器
    wfs = RLWFS(
        dm, 
        ccd,
        algorithm="ppo",       # 或 "sac", "td3"
        n_episodes=1000,
        reward_function="intensity"
    )
    
    # 训练
    wfs.train(write_path="models/rl_wfs")
    
    # 使用训练好的模型
    wfs.load("models/rl_wfs/best_model")
    wfs.correct()
```

---

## 5. 无波前传感使用

### 5.1 贪婪相机方法

```python
from ao_shaping.drivers.slm import SantecSLM200
from ao_shaping.drivers.ccd import MiiCAM
from ao_shaping.wfless.gready_cam import GreedyCAM
import numpy as np

with SantecSLM200() as slm, MiiCAM() as ccd:
    # 创建优化器
    optimizer = GreedyCAM(slm, ccd)
    
    # 配置
    optimizer.set_config(
        n_patterns=100,       # 每轮模式数
        iterations=50,        # 迭代次数
        exposure_time=0.001   # 曝光时间
    )
    
    # 优化
    result = optimizer.optimize(
        target_region=None,   # 目标区域 (x, y, w, h)
        metric="brightness"   # 优化指标: "brightness", "contrast", "custom"
    )
    
    print(f"最终强度: {result['final_intensity']}")
    print(f"收敛曲线: {result['history']}")
```

### 5.2 线性回归无波前传感

```python
from ao_shaping.drivers.slm import SantecSLM200
from ao_shaping.drivers.ccd import MiiCAM
from ao_shaping.wfless.lr import LrWFLess

with SantecSLM200() as slm, MiiCAM() as ccd:
    optimizer = LrWFLess(slm, ccd)
    
    # 校准
    optimizer.calibrate(n_samples=200)
    
    # 优化
    result = optimizer.optimize(iterations=100)
```

### 5.3 ADC DM Adam 优化

```python
from ao_shaping.drivers.slm import SantecSLM200
from ao_shaping.drivers.dm import NLightDM
from ao_shaping.drivers.ccd import MiiCAM
from ao_shaping.wfless.adc_dm_adam import ADCDMAdam

with SantecSLM200() as slm, NLightDM() as dm, MiiCAM() as ccd:
    optimizer = ADCDMAdam(slm, dm, ccd)
    
    # 优化
    result = optimizer.optimize(
        lr=0.01,              # 学习率
        iterations=200,       # 迭代次数
        beta1=0.9,            # Adam 参数
        beta2=0.999
    )
```

### 5.4 相位恢复

```python
from ao_shaping.wfless.phase_retraive import PhaseRetrieve
import numpy as np

# 创建相位恢复器
pr = PhaseRetrieve(resolution=(512, 512))

# 定义目标强度
target_intensity = np.random.rand(512, 512)

# 运行 GS 算法
phase = pr.gs_iterate(target_intensity, iterations=100)

# 运行混合输入输出 (HIO)
phase = pr.hio(target_intensity, iterations=100)
```

---

## 6. 工具模块使用

### 6.1 波前计算

```python
from ao_shaping.utils.wavefront_calc import Zernike, Wavefront
import numpy as np

# Zernike 多项式
zernike = Zernike(n_terms=15)

# 生成 Zernike 模式
coefficients = np.random.rand(15) * 0.1
phase_map = zernike.generate(coefficients)

# 从相位图拟合 Zernike 系数
fitted_coefficients = zernike.fit(phase_map)

# 波前分析
wf = Wavefront(phase_map)
print(f"RMS: {wf.rms}")
print(f"PV: {wf.pv}")
print(f"倾斜: {wf.tilt}")
```

### 6.2 相位图案生成

```python
from ao_shaping.utils.phase_patterns import (
    grating_pattern,
    blazed_grating,
    gs_iterate,
    dmd_pattern
)

# 光栅图案
grating = grating_pattern(resolution=(1920, 1080), period=50, angle=0)

# 闪耀光栅
blazed = blazed_grating(resolution=(1920, 1080), blaze_angle=10)

# GS 算法迭代
target = np.random.rand(512, 512) + 0j
phase = gs_iterate(target, iterations=50)

# DMD 图案
dmd = dmd_pattern(binary_phase=True)
```

### 6.3 光斑计算

```python
from ao_shaping.utils.spots_calc import (
    centroid,
    spot_radius,
    correlation
)

# 图像
image = np.random.rand(512, 512)

# 计算质心
cx, cy = centroid(image)

# 计算光斑半径
radius = spot_radius(image, cx, cy)

# 相关性计算
corr = correlation(image1, image2)
```

### 6.4 文件操作

```python
from ao_shaping.utils.file import save_data, load_data
import numpy as np

# 保存数据
data = np.random.rand(100, 100)
save_data(data, "output.dat")

# 加载数据
loaded = load_data("output.dat")

# 保存为 numpy
np.save("output.npy", data)
loaded = np.load("output.npy")
```

---

## 7. 可视化界面

### 7.1 Streamlit 可视化

```bash
# 启动 Streamlit 可视化界面
streamlit run scripts/streamlit_visualizer.py
```

功能包括：
- 实时图像显示
- 参数配置
- 优化过程监控
- 结果分析

### 7.2 自定义显示

```python
from ao_shaping.display.windows import DisplayWindow
from ao_shaping.display.frames import ImageFrame

# 创建窗口
window = DisplayWindow(title="AO Shaping Display")

# 添加图像帧
frame = ImageFrame(image_data)
window.add_frame(frame)

# 显示
window.show()

# 关闭
window.close()
```

---

## 8. 常见用例

### 8.1 完整波前校正流程

```python
from ao_shaping.drivers.slm import SantecSLM200
from ao_shaping.drivers.dm import NLightDM
from ao_shaping.drivers.wfs import ThorlabsWFS
from ao_shaping.wf.DM_wfs import DMWFSController
import numpy as np

def wavefront_correction():
    """完整的波前校正流程"""
    
    # 初始化硬件
    with SantecSLM200() as slm, \
         NLightDM() as dm, \
         ThorlabsWFS() as wfs:
        
        # 1. 校准 DM 响应矩阵
        controller = DMWFSController(dm, wfs)
        print("正在校准...")
        response_matrix = controller.calibrate(n_samples=50)
        
        # 2. 保存响应矩阵
        np.save("response_matrix.npy", response_matrix)
        
        # 3. 执行校正
        print("正在校正波前...")
        result = controller.correct(
            target_rms=0.01,
            max_iterations=30
        )
        
        print(f"校正完成!")
        print(f"  初始 RMS: {result['initial_rms']:.6f}")
        print(f"  最终 RMS: {result['final_rms']:.6f}")
        print(f"  迭代次数: {result['iterations']}")
        
        return result

if __name__ == "__main__":
    wavefront_correction()
```

### 8.2 无波前传感优化流程

```python
from ao_shaping.drivers.slm import SantecSLM200
from ao_shaping.drivers.ccd import MiiCAM
from ao_shaping.wfless.gready_cam import GreedyCAM
from ao_shaping.utils.phase_patterns import gs_iterate
import numpy as np

def wfless_optimization():
    """无波前传感优化流程"""
    
    with SantecSLM200() as slm, MiiCAM() as ccd:
        # 1. 设置目标
        ccd.set_exposure(0.001)
        
        # 2. 创建优化器
        optimizer = GreedyCAM(slm, ccd)
        
        # 3. 执行优化
        print("开始优化...")
        result = optimizer.optimize(
            iterations=100,
            n_patterns=50
        )
        
        print(f"优化完成!")
        print(f"  最终强度: {result['final_intensity']:.2f}")
        
        # 4. 获取最终相位图案
        final_pattern = optimizer.get_best_pattern()
        
        # 5. 保存结果
        np.save("best_pattern.npy", final_pattern)
        
        return result

if __name__ == "__main__":
    wfless_optimization()
```

### 8.3 数据采集流程

```python
from ao_shaping.drivers.slm import SantecSLM200
from ao_shaping.drivers.ccd import MiiCAM
from ao_shaping.train_data_collect import DataCollector
import numpy as np

def collect_training_data():
    """训练数据采集"""
    
    with SantecSLM200() as slm, MiiCAM() as ccd:
        collector = DataCollector(slm, ccd)
        
        # 配置采集
        collector.set_config(
            n_samples=1000,
            pattern_type="random",
            save_path="training_data"
        )
        
        # 开始采集
        print("开始采集数据...")
        dataset = collector.collect()
        
        print(f"采集完成!")
        print(f"  样本数: {len(dataset)}")
        
        return dataset

if __name__ == "__main__":
    collect_training_data()
```

---

## 附录

### A. 错误处理

```python
from ao_shaping.drivers.slm import SantecSLM200
from ao_shaping.drivers.slm import SantecSLM200Error

try:
    with SantecSLM200() as slm:
        slm.write_pattern(pattern)
except SantecSLM200Error as e:
    print(f"SLM 错误: {e}")
except Exception as e:
    print(f"未知错误: {e}")
```

### B. 日志配置

```python
from loguru import logger

# 配置日志
logger.add("ao_shaping.log", rotation="10 MB", retention="7 days")
logger.info("AO Shaping 启动")
```

### C. 性能优化

```python
# 使用 Numba 加速
from ao_shaping.utils.wavefront_calc import Zernike

# Zernike 计算已优化
zernike = Zernike(n_terms=15)
phase = zernike.generate(coefficients)  # 自动使用 JIT 编译
```

---

*本文档由 AI 自动生成，最后更新于 2026-03-05*
