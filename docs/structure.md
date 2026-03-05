# 项目结构与功能说明

**项目名称:** AO-shaping  
**文档版本:** 1.0.0  
**创建日期:** 2026-03-05  
**最后更新:** 2026-03-05

---

## 目录

1. [概述](#1-概述)
2. [目录结构](#2-目录结构)
3. [核心模块说明](#3-核心模块说明)
4. [硬件驱动](#4-硬件驱动)
5. [算法模块](#5-算法模块)
6. [工具模块](#6-工具模块)
7. [显示模块](#7-显示模块)
8. [配置文件](#8-配置文件)

---

## 1. 概述

AO-shaping 是一个基于 Python 的自适应光学（Adaptive Optics）项目，用于控制硬件设备（SLM、DM、WFS、CCD）并结合 PyTorch 深度学习进行波前控制与无波前传感优化。

### 1.1 项目目标

- 实现高精度波前校正
- 支持多种自适应光学硬件设备
- 结合深度学习优化控制算法
- 提供可视化界面与分析工具

### 1.2 技术栈

| 类别 | 技术 |
|-----|------|
| 核心语言 | Python 3.13+ |
| 数值计算 | NumPy, SciPy |
| 深度学习 | PyTorch, Timm, Stable-Baselines3 |
| 硬件控制 | PyVISA, PySerial, NI-DAQmx |
| 可视化 | Matplotlib, Plotly, Streamlit |
| 依赖管理 | uv |

---

## 2. 目录结构

```
AO-shaping/
├── src/ao_shaping/          # 源代码主目录
│   ├── drivers/             # 硬件驱动模块
│   │   ├── slm/             # 空间光调制器 (SLM)
│   │   ├── dm/              # 变形镜 (DM)
│   │   ├── wfs/             # 波前传感器 (WFS)
│   │   ├── ccd/             # CCD/相机
│   │   ├── tm/              # 温度管理器
│   │   ├── device_base.py   # 设备基类
│   │   ├── device_registry.py # 设备注册表
│   │   └── visa_base.py     # VISA 基础类
│   ├── algorithm/           # 优化算法
│   │   └── adam.py          # Adam 优化器变体
│   ├── wf/                  # 波前控制
│   │   ├── DM_wfs.py        # DM 波前传感
│   │   ├── lr_wfs.py        # 线性回归波前传感
│   │   └── rl_wfs.py        # 强化学习波前传感
│   ├── wfless/              # 无波前传感
│   │   ├── gready_cam.py    # 贪婪相机方法
│   │   ├── lr.py            # 线性回归方法
│   │   ├── adc_dm_adam.py  # ADC DM Adam 方法
│   │   └── phase%20retraive.py # 相位恢复
│   ├── utils/               # 工具模块
│   │   ├── wavefront_calc.py # 波前计算
│   │   ├── phase_patterns.py # 相位图案生成
│   │   ├── spots_calc.py    # 光斑计算
│   │   ├── display.py       # 显示工具
│   │   ├── file.py          # 文件操作
│   │   └── TM.py            # 温度管理工具
│   ├── display/             # GUI/可视化
│   │   ├── windows.py       # 窗口组件
│   │   └── frames.py        # 帧组件
│   ├── base/                # 基础类
│   │   └── optimize.py      # 优化基类
│   ├── __init__.py
│   ├── DM_flatten.py        # DM 校平脚本
│   ├── DM_combined.py       # DM 组合控制
│   └── train_data_collect.py # 训练数据收集
├── tests/ao_shaping/        # 测试目录 (镜像 src 结构)
├── scripts/                 # 独立应用程序
│   └── streamlit_visualizer.py # Streamlit 可视化
├── docs/                    # 文档目录
├── pyproject.toml           # uv 项目配置
└── README.md                # 项目说明
```

---

## 3. 核心模块说明

### 3.1 硬件驱动层 (drivers/)

硬件驱动层负责与物理设备通信，提供统一的接口规范。

**设计原则：**
- 所有驱动继承自统一的基类
- 必须实现 `open()`, `close()`, 上下文管理器
- 使用自定义异常处理错误

**基类：**
- `DeviceBase`: 设备基类，定义通用接口
- `VisaBase`: 基于 PyVISA 的设备基类

### 3.2 算法层 (algorithm/)

优化算法的实现模块。

| 模块 | 功能 |
|-----|------|
| `adam.py` | Adam 优化器变体，用于迭代优化 |

### 3.3 波前控制层 (wf/)

基于波前传感的自适应光学控制。

| 模块 | 功能 |
|-----|------|
| `DM_wfs.py` | 变形镜波前传感控制 |
| `lr_wfs.py` | 线性回归波前传感 |
| `rl_wfs.py` | 强化学习波前传感 |

### 3.4 无波前传感层 (wfless/)

无需波前传感器的优化方法。

| 模块 | 功能 |
|-----|------|
| `gready_cam.py` | 贪婪相机方法 |
| `lr.py` | 线性回归无波前传感 |
| `adc_dm_adam.py` | ADC DM Adam 优化 |
| `phase retrieve.py` | 相位恢复算法 |

---

## 4. 硬件驱动

### 4.1 空间光调制器 (SLM)

**目录:** `src/ao_shaping/drivers/slm/`

| 文件 | 功能 |
|-----|------|
| `santec_slm200.py` | Santec SLM-200 驱动 |
| `santec_slm200_visa.py` | Santec SLM-200 VISA 驱动 |
| `slm_pattern_helper.py` | SLM 图案辅助工具 |
| `slm_calibration.py` | SLM 校准工具 |
| `_slm_win.py` | SLM 窗口控制 |

**支持的设备:**
- Santec SLM-200

### 4.2 变形镜 (DM)

**目录:** `src/ao_shaping/drivers/dm/`

| 文件 | 功能 |
|-----|------|
| `base.py` | DM 基础类 |
| `NLight.py` | NLight DM 驱动 |
| `simulateDM.py` | 模拟 DM（用于测试） |

### 4.3 波前传感器 (WFS)

**目录:** `src/ao_shaping/drivers/wfs/`

| 文件 | 功能 |
|-----|------|
| `thorlab_wfs.py` | Thorlabs 波前传感器驱动 |

### 4.4 CCD/相机

**目录:** `src/ao_shaping/drivers/ccd/`

| 文件 | 功能 |
|-----|------|
| `base.py` | 相机基础类 |
| `miicam.py` | Mii相机驱动 |
| `miicam_device.py` | Mii相机设备实现 |
| `daheng.py` | 大恒相机驱动 |

### 4.5 温度管理器 (TM)

**目录:** `src/ao_shaping/drivers/tm/`

| 文件 | 功能 |
|-----|------|
| `serial_port_fsm.py` | 串口有限状态机温度控制 |

---

## 5. 算法模块

### 5.1 优化算法

**目录:** `src/ao_shaping/algorithm/`

```python
# 使用示例
from ao_shaping.algorithm.adam import CustomAdam

optimizer = CustomAdam(lr=0.01)
```

### 5.2 波前控制

**目录:** `src/ao_shaping/wf/`

#### DM_wfs.py

变形镜波前传感控制实现。

```python
from ao_shaping.wf.DM_wfs import DMWFSController

controller = DMWFSController(dm, wfs)
controller.correct(wavefront_error)
```

#### lr_wfs.py

线性回归波前传感。

```python
from ao_shaping.wf.lr_wfs import LrWFS

wfs_controller = LrWFS(dm, ccd)
wfs_controller.calibrate()
```

#### rl_wfs.py

强化学习波前传感。

```python
from ao_shaping.wf.rl_wfs import RLWFS

wfs_controller = RLWFS(dm, ccd, algorithm="ppo")
wfs_controller.train(episodes=1000)
```

### 5.3 无波前传感

**目录:** `src/ao_shaping/wfless/`

#### gready_cam.py

贪婪相机方法。

```python
from ao_shaping.wfless.gready_cam import GreedyCAM

optimizer = GreedyCAM(slm, ccd)
optimizer.optimize(iterations=100)
```

#### lr.py

线性回归无波前传感。

```python
from ao_shaping.wfless.lr import LrWFLess

optimizer = LrWFLess(slm, ccd)
optimizer.calibrate()
optimizer.optimize()
```

#### adc_dm_adam.py

ADC DM Adam 优化方法。

```python
from ao_shaping.wfless.adc_dm_adam import ADCDMAdam

optimizer = ADCDMAdam(slm, dm, ccd)
optimizer.optimize(lr=0.01, iterations=200)
```

---

## 6. 工具模块

### 6.1 目录

**目录:** `src/ao_shaping/utils/`

### 6.2 主要工具

| 模块 | 功能 |
|-----|------|
| `wavefront_calc.py` | 波前计算（Zernike 多项式、干涉图分析） |
| `phase_patterns.py` | 相位图案生成（光栅、LUT、GS 算法） |
| `spots_calc.py` | Shack-Hartmann 光斑计算 |
| `display.py` | 显示辅助工具 |
| `file.py` | 文件读写工具 |
| `TM.py` | 温度监控工具 |

### 6.3 使用示例

```python
# 波前计算
from ao_shaping.utils.wavefront_calc import Zernike

zernike = Zernike(n_terms=15)
coefficients = zernike.fit(phase_map)

# 相位图案
from ao_shaping.utils.phase_patterns import gs_iterate

pattern = gs_iterate(target, iterations=50)
```

---

## 7. 显示模块

### 7.1 目录

**目录:** `src/ao_shaping/display/`

### 7.2 组件

| 模块 | 功能 |
|-----|------|
| `windows.py` | GUI 窗口组件 |
| `frames.py` | 帧组件（用于图像显示） |

### 7.3 Streamlit 可视化

```bash
# 启动 Streamlit 可视化界面
streamlit run scripts/streamlit_visualizer.py
```

---

## 8. 配置文件

### 8.1 pyproject.toml

项目配置文件，定义依赖、构建选项等。

```toml
[project]
name = "ao_shaping"
version = "0.1.0"
requires-python = ">=3.13"

dependencies = [
    "numpy>=2.2.6",
    "torch>=2.8.0",
    # ... 更多依赖
]
```

### 8.2 数据目录挂载

项目使用网络存储保存数据：

```bash
# 挂载数据目录
sudo mount -t cifs -o user=tifo,password=TIFO1234,uid=tifo,gid=tifo,iocharset=utf8,vers=3.0 //10.10.0.53/storage/AO_data data
```

---

## 附录

### A. 模块依赖关系图

```
┌─────────────────────────────────────────────────────────────────┐
│                        AO-shaping 架构                           │
└─────────────────────────────────────────────────────────────────┘

                        ┌──────────────┐
                        │   scripts/   │  (Streamlit UI)
                        └──────┬───────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ▼                     ▼                     ▼
┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│    display/    │  │     utils/      │  │    algorithm/  │
│  (可视化界面)   │  │   (波前计算等)   │  │   (优化算法)    │
└───────┬────────┘  └────────┬────────┘  └───────┬────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│      wf/       │  │    wfless/     │  │    drivers/    │
│  (波前控制)    │  │ (无波前传感)    │  │  (硬件驱动)    │
└────────────────┘  └────────────────┘  └────────────────┘
                                              │
                      ┌───────────┬───────────┼───────────┬───────────┐
                      ▼           ▼           ▼           ▼           ▼
               ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
               │   slm   │ │   dm    │ │   wfs   │ │   ccd   │ │   tm    │
               └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘
```

### B. 命名约定

| 类型 | 约定 | 示例 |
|-----|------|------|
| 文件 | 小写下划线 | `santec_slm200.py` |
| 类 | 大驼峰 | `SantecSLM200` |
| 函数 | 小写下划线 | `calculate_wavefront` |
| 常量 | 全大写下划线 | `MAX_VOLTAGE` |

### C. 相关文档

- [Git 文档](gitdoc.md)
- [使用文档](usagedoc.md)
- [部署文档](deploydoc.md)

---

*本文档由 AI 自动生成，最后更新于 2026-03-05*
