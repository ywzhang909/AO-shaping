# AO-Shaping 自适应光学整形系统

## 项目简介

AO-Shaping是一个基于强化学习的自适应光学(AO)系统，用于波前校正和光束整形。该项目集成了多种优化算法，包括基于波前传感器(WFS)的优化和无波前优化方法，能够通过变形镜(DM)对光波前进行精确控制，实现高质量的光束输出。

## 主要特性

- **多优化算法**: 支持基于波前传感器的RMS优化和无波前的PIB优化
- **强化学习集成**: 使用SAC算法进行波前优化
- **硬件支持**: 兼容Thorlabs WFS波前传感器、NLight变形镜、R50Power MicroDM、大恒相机、MIICAM和Santec SLM200
- **可视化工具**: 提供实时波前和电压可视化功能
- **数据处理**: 集成Dask进行高性能数据处理和分析
- **实验跟踪**: 支持WandB和SwanLab进行实验管理和可视化
- **ML训练**: 支持U-Net+GAN相位预测模型训练和推理

## 项目结构

```
AO-shaping/
├── src/
│   ├── ao_shaping/          # 主程序包
│   │   ├── main.py              # CLI入口点 (Click-based)
│   │   ├── runners/             # 运行器包
│   │   │   ├── wf_runner.py     # 波前RMS优化器
│   │   │   ├── axis_beam_runner.py  # PIB优化器
│   │   │   ├── pipeline_runner.py  # 串行 WF→PIB 流水线
│   │   │   └── zernike_matrix_runner.py  # Zernike响应矩阵校准
│   │   ├── algorithm/           # 优化算法 (Adam, SGD, Muon等)
│   │   ├── drivers/             # 硬件驱动
│   │   │   ├── ccd/             # 相机 (Daheng, MiiCam)
│   │   │   ├── dm/              # 变形镜 (NLight, R50Power MicroDM)
│   │   │   ├── slm/             # 空间光调制器 (Santec)
│   │   │   ├── wfs/             # 波前传感器 (Thorlabs)
│   │   │   ├── tm/              # 定时模块 (Serial/FSM)
│   │   │   ├── sim/             # 数字孪生仿真
│   │   │   └── mock_devices.py  # 测试用模拟设备
│   │   ├── optimizer/           # 高层优化器
│   │   │   ├── wf/              # 波前优化 (RMS)
│   │   │   ├── wfless/          # 无波前优化 (PIB)
│   │   │   └── rl/              # 强化学习 (SAC, LR-WFS)
│   │   ├── utils/               # 工具函数 (spots_calc, wavefront_calc)
│   │   ├── ml/                  # 机器学习 (U-Net+GAN, 训练, 模型)
│   │   │   ├── trainer/         # 训练器
│   │   │   ├── models/          # 神经网络模型
│   │   │   └── wandb_logger.py  # WandB日志
│   │   ├── tools/               # 独立工具 (SLM相位捕获, 数据采集)
│   │   ├── display/             # 可视化 (窗口, GUI帧)
│   │   └── gui/                 # GUI组件 (Streamlit)
│   ├── calculators/             # Cython扩展 (独立)
│   └── optical_ui/              # [已废弃]
├── tests/ao_shaping/             # 测试 (镜像src结构)
├── scripts/                      # 实用脚本
├── libs/                         # 第三方SDK二进制 (gxipy, Drv_UDPST)
└── AGENTS.md                     # 开发指南
```

## 安装指南

### 环境要求

- Python 3.12+
- Windows/Linux/macOS
- CUDA (可选，用于GPU加速)

### 安装步骤

1. 克隆项目仓库:
```bash
git clone <repository_url>
cd AO-shaping
```

2. 创建虚拟环境:
```bash
uv venv .venv
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate     # Windows
```

3. 安装依赖:
```bash
# 仅安装基础依赖
uv sync

# 安装 ML 相关包 (torch, torchvision, wandb)
uv sync --extra ml

# 安装 RL 相关包 (gymnasium, stable-baselines3)
uv sync --extra rl

# 安装所有可选依赖
uv sync --extra ml --extra rl
```

## 使用说明

### CLI命令 (基于Click)

项目提供了统一的命令行界面，通过`main.py`作为入口点：

```bash
python src/ao_shaping/main.py [OPTIONS] COMMAND [ARGS]...
```

所有运行器位于 `src/ao_shaping/runners/` 包中，通过 main CLI 统一调用：

#### 全局选项
- `--dir`: 指定数据保存根目录 (默认: data)
- `DEBUG`: 环境变量控制调试模式 (export DEBUG=1 或 DEBUG=true)

#### 调试模式

所有命令支持通过环境变量开启调试模式：

```bash
# 方式1: export
export DEBUG=1
python src/ao_shaping/main.py wf --epochs 10000

# 方式2: 内联
DEBUG=1 python src/ao_shaping/main.py pib --epochs 5000
```

支持的DEBUG值: `1`, `true`, `yes` (不区分大小写)

#### 波前优化器 (wf)
```bash
python src/ao_shaping/main.py wf [OPTIONS]
```
等同于: `python -m ao_shaping.runners.wf_runner`

选项:
- `-e, --epochs`: 优化迭代次数 (默认: 20000)
- `-r, --wfs_res`: WFS分辨率 (默认: 768)
- `-p, --pupil_diameter`: 瞳孔直径 (默认: 2.7)
- `-t, --early_stop_threshold`: 早停阈值 (默认: 0.0)

示例:
```bash
DEBUG=1 python src/ao_shaping/main.py wf --epochs 10000
```

#### 轴向光束优化器 (pib)
```bash
python src/ao_shaping/main.py pib [OPTIONS]
```
等同于: `python -m ao_shaping.runners.axis_beam_runner`

选项:
- `-f, --load_file`: 加载优化结果文件
- `--cam_id`: 远场光斑CCD设备ID (默认: 0)
- `-c, --center`: 场光斑CCD中心位置
- `-t, --exposure_time_ms`: 远场光斑CCD曝光时间(毫秒) (默认: 800)
- `-e, --epochs`: 优化迭代次数 (默认: 4000)
- `-r, --r_bucket`: 渲染半径桶大小 (默认: 18)
- `--delta`: 优化步长 (默认: 2)
- `--lr`: 优化学习率 (默认: 2)
- `--weight_decay`: 权重衰减 (默认: 0.0)
- `--shrink_iter`: 优化迭代次数后收缩半径桶和步长 (默认: 300)
- `--shrink_ratio`: 收缩半径桶和步长比例 (默认: 0.8)
- `-s, --cam_size`: 相机开窗大小 (默认: 200)

示例:
```bash
DEBUG=1 python src/ao_shaping/main.py pib --epochs 5000 --cam_id 1
```

#### 串行流水线优化器 (pipeline)
```bash
python src/ao_shaping/main.py pipeline [OPTIONS]
```
等同于: `python -m ao_shaping.runners.pipeline_runner`

选项:
- `-f, --load_file`: 加载优化结果文件
- `-e, --epochs`: 优化迭代次数 (默认: 8000)
- `-R, --wfs_res`: WFS分辨率 (默认: 768)
- `-p, --pupil_diameter`: 瞳孔直径 (默认: 2.7)
- `-c, --cam_id`: 远场光斑CCD设备ID (默认: 0)
- `-t, --exposure_time_ms`: 远场光斑CCD曝光时间(毫秒) (默认: 500)
- `-s, --cam_size`: 相机开窗大小 (默认: 160)
- `-r, --rms_threshold`: RMS阈值 (默认: 0.12)
- `-u, --dm_unit_mask`: DM单元掩码 (默认: all)

示例:
```bash
DEBUG=1 python src/ao_shaping/main.py pipeline --epochs 6000
```

#### Zernike响应矩阵校准 (zernike-matrix)
```bash
python src/ao_shaping/main.py zernike-matrix [OPTIONS]
```
等同于: `python -m ao_shaping.runners.zernike_matrix_runner`

#### Zernike波前优化器 - 贪婪局部搜索 (greedy-zernike)
```bash
python src/ao_shaping/main.py greedy-zernike [OPTIONS]
```
使用贪婪局部搜索算法进行Zernike波前校正。

选项:
- `-e, --epochs`: 优化迭代次数 (默认: 2000)
- `-n, --n-max`: Zernike最大阶数 (默认: 4)
- `-r, --wfs_res`: WFS分辨率 (默认: 1024)
- `-p, --pupil_diameter`: 瞳孔直径 (默认: 2.7)
- `-c, --pupil_center`: 瞳孔中心坐标 (默认: (0,0))
- `-t, --early_stop_threshold`: 早停阈值 (默认: 0.12)
- `--wavelength`: SLM波长 (nm, 默认: 532)
- `--slm-number`: SLM设备编号 (默认: 1)
- `--remove-tilt`: 移除波前测量中的倾斜项
- `--n-init`: 初始随机位置数量 (默认: 10)
- `--n-directions`: 每次迭代的随机方向数量 (默认: 5)
- `--perturbation-scale`: 扰动幅度缩放因子 (默认: 5.0)

算法流程:
1. 随机初始化N个位置，选取最优作为起始点
2. 每次迭代采样n个随机扰动方向
3. 评估所有候选(当前位置+n个扰动)，选择最优
4. 重复直到收敛或达到最大迭代次数

示例:
```bash
DEBUG=1 python src/ao_shaping/main.py greedy-zernike --n-init 20 --n-directions 8
```

### 串行流水线优化器处理流程详解

串行流水线优化器采用分阶段优化策略，集成了波前传感器和CCD相机的优势，通过两个阶段的优化实现高质量的光束输出。

#### 1. 使用的设备

组合优化器集成了多种硬件设备协同工作：

1. **波前传感器(WFS)**：Thorlabs WFS系列设备，用于测量光波前的畸变
2. **变形镜(DM)**：NLight DM设备，用于校正光波前
3. **CCD相机**：大恒相机/MIICAM系统，用于捕捉远场光斑图像
4. **空间光调制器(SLM)**：Santec SLM200，用于相位调制
5. **计算单元**：运行优化算法的计算机系统

#### 2. 算法流程

组合优化器采用分阶段优化策略：

##### 第一阶段：波前优化(RMS优化)
1. 初始化变形镜电压为零或从文件加载初始电压
2. 使用波前传感器测量当前波前，计算RMS值
3. 应用扰动法，分别向正负方向施加随机扰动电压
4. 测量扰动后的波前RMS值
5. 根据RMS差异计算梯度，更新变形镜电压
6. 重复上述过程直到RMS达到阈值或完成预定迭代次数

##### 第二阶段：PIB优化(远场光斑优化)
1. 基于第一阶段优化结果初始化变形镜电压
2. 使用CCD相机捕获远场光斑图像
3. 计算光斑中心和桶内功率(PIB)
4. 应用扰动法，分别向正负方向施加随机扰动电压
5. 测量扰动后的光斑图像，计算PIB值
6. 根据PIB差异计算梯度，更新变形镜电压
7. 重复上述过程直到完成预定迭代次数

#### 3. CCD中心计算方法

CCD中心计算采用了多种方法来精确定位光斑中心：

##### 方法一：最大值法(Max)
```python
center = np.unravel_index(np.argmax(img), img.shape)[::-1]
```
直接寻找图像中像素强度最大的位置作为中心点。

##### 方法二：质心法(Centroid)
使用图像强度加权计算质心位置，更能反映光斑的整体分布。

##### 方法三：形状识别法
通过阈值分割识别光斑区域，然后计算该区域的质心。

#### 4. 优化目标

1. **第一阶段目标**：最小化波前RMS值，使光波前尽可能接近理想平面波
2. **第二阶段目标**：最大化PIB值(桶内功率)，即将更多光能集中到目标区域内

#### 5. 关键技术细节

1. **自适应学习率**：根据当前优化状态动态调整学习率和扰动幅度
2. **桶半径自适应收缩**：随着优化进展逐步缩小功率计算区域，提高聚焦精度
3. **安全检查机制**：监控相邻变形镜单元间的电压差，防止损坏设备
4. **曝光时间自适应调节**：根据图像亮度自动调整相机曝光时间，保证图像质量
5. **可视化监控**：实时显示优化过程中的图像、波前和电压变化

整个组合优化器通过这种分阶段、多目标的优化策略，能够有效地实现自适应光学系统的波前校正和光束整形双重目标。

### 单独运行脚本

除了使用统一入口，也可以直接运行各个优化器脚本:

1. 波前优化器:
```bash
python -m ao_shaping.runners.wf_runner [OPTIONS]
```

2. 轴向光束优化器:
```bash
python -m ao_shaping.runners.axis_beam_runner [OPTIONS]
```

3. 流水线优化器:
```bash
python -m ao_shaping.runners.pipeline_runner [OPTIONS]
```

4. Zernike校准:
```bash
python -m ao_shaping.runners.zernike_matrix_runner [OPTIONS]
```

注意: `combined_runner.py` 已废弃，请使用 `pipeline_runner`

### ML训练 (U-Net+GAN相位预测)

项目支持使用深度学习模型进行相位预测训练和推理：

```bash
# 训练
python scripts/train_phase_prediction.py --config configs/train_config.yaml

# 推理
python scripts/inference_phase.py --model models/best_model.pth --input input.png
```

### GUI界面 (Streamlit)

项目提供了基于Streamlit的图形界面：

```bash
streamlit run src/ao_shaping/gui/app.py
```

## 硬件支持

### 波前传感器
- **Thorlabs WFS系列**: 支持自动图像采集和倾斜去除

### 变形镜
- **统一 DM 接口**: 所有变形镜继承自 `ao_shaping.drivers.dm.base.DM`，提供 `transform`/`send`/`open`/`close`/`is_connected`/`get_actuator_positions` 等标准方法
- **NLight系列**: 支持电压控制和电压差安全检查
- **R50Power MicroDM**: 通过异步 TCP 控制多路 R50Power 控制器（每路 50 通道，-20V~120V）
  - 自动从 `libs/micro_drive1300/wiring_map.json` 加载控制器 IP 和通道映射
  - 支持 39×39 阵列坐标 `(x, y)` 到控制器通道的双向查询
  - 支持 `(ip_suffix, payload_position)` 到物理位置的映射查询
  - 类型安全的 WiringMap dataclass 解析（`WiringMap`, `ChannelEntry`, `ChannelInfo`）
  - 可通过 `use_wiring_map=False` 回退到默认 IP 配置
  - **容错连接**: `open()` 允许个别控制器连接失败，记录 warning 后继续，仅当全部失败时抛出异常
  - **连接状态检查**: `get_connection_status()` 返回所有控制器的 ping 可达性和 TCP 连接状态
  - **单控制器管理**: 支持 `connect_controller(id)`、`disconnect_controller(id)`、`reconnect_controller(id)` 独立控制
  - **排除控制器**: 初始化时可通过 `exclude_ips` 或 `exclude_ids` 跳过指定控制器

  ```python
  from ao_shaping.drivers.dm.MicroDM import MicroDM

  # 默认加载 wiring map，自动识别控制器 IP
  dm = MicroDM()

  # 排除特定控制器（按 IP 或 ID）
  dm = MicroDM(exclude_ips=["192.168.0.103"], exclude_ids=[4, 5])

  # 通过 39×39 阵列坐标查询通道信息
  info = dm.get_channel_by_xy(x=1, y=3)
  print(info.ip_address, info.payload_position, info.physical_label)

  # 通过控制器 IP 和通道标号查询
  info = dm.get_channel_by_ip_position(ip_suffix=101, payload_position=13)
  print(info.physical_position, info.physical_label)

  # 发送电压指令
  dm.open()
  dm.send_voltages(np.zeros(dm.DM_Num))

  # 检查所有控制器连接状态
  for status in dm.get_connection_status():
      print(f"ID={status.controller_id} IP={status.ip} "
            f"ping={status.ping_reachable} tcp={status.tcp_connected}")

  # 单独管理控制器
  dm.connect_controller(3)       # 连接控制器 3
  dm.disconnect_controller(2)    # 断开控制器 2
  dm.reconnect_controller(1)     # 重连控制器 1

  dm.close()
  ```

- **ZernikeDM**: Zernike 系数驱动的 DM/SLM 接口，支持 Zernike 多项式相位生成
- **HadamardDM**: Hadamard 系数驱动的 DM/SLM 接口，支持 Walsh-Hadamard 模式相位生成
- **PIB 优化器多 DM 支持**: `optimize_pib` 现接受任意 `DM` 子类实例，命令行支持 `--dm_type` 参数
  - 支持类型: `nlight`, `micro`, `zernike`, `hadamard`
  - 自动检测: 未指定 `--dm_type` 时自动探测在线 DM，仅一个时自动选取，多个时报错提示

### 相机
- **大恒相机系列**: DahengCamManager，支持14位和16位模式
- **MIICAM系列**: MIICamDriver，支持高速采集

### 空间光调制器
- **Santec SLM200**: 支持相位图案生成、缓存和CSV加载

### 数据采集卡
- **NI DAQ设备**: 用于多设备同步控制

## 仿真环境

项目包含完整的数字孪生仿真环境，支持无硬件测试：

```bash
python -c "from ao_shaping.drivers.sim import SimTurbulenceAOEnv; env = SimTurbulenceAOEnv()"
```

仿真模块包括:
- 变形镜仿真 (带迟滞特性)
- 波前传播仿真
- 湍流生成
- 光束传播

## 开发指南

### 编码规范

#### 1. Python 版本与导入

- **Python 3.12+** 是必需的（见 `pyproject.toml`）
- **强制使用 `from __future__ import annotations`**：所有模块文件第一行应包含此导入，启用 PEP 604 延迟求值语法
- **导入顺序**（按标准库 → 第三方 → 本地有序分组）：
  ```python
  from __future__ import annotations  # 总在第一行

  import uuid
  from abc import ABC, abstractmethod
  from collections.abc import Callable, Sequence
  from dataclasses import dataclass
  from typing import Any, ClassVar

  import numpy as np

  from loguru import logger

  from ao_shaping.config import DM_N_ACTUATORS
  from ao_shaping.drivers import CameraStreamManager
  ```

#### 2. 绝对导入（项目强制规则）

- **所有包内引用必须使用绝对导入**，禁止 `from .xxx import yyy` 形式的相对导入
- 格式：`from ao_shaping.子包.模块 import 名称`
- 对于 Cython 回退等特殊场景，使用 try/except 包在绝对导入中：
  ```python
  try:
      from ao_shaping.algorithm._adam_cython import Adam  # type: ignore
  except ImportError:
      from ao_shaping.algorithm.adam import Adam
  ```

#### 3. 类型注解（Python 3.12+ 语法）

- **所有公开函数/方法必须标注参数和返回类型**
- 使用 `|` 语法替代 `Optional` / `Union`：
  ```python
  # 正确
  def get_parameter(self, name: str) -> float | None:
      ...

  # 错误
  def get_parameter(self, name: str) -> Optional[float]:
      ...
  ```
- 使用 `list[X]` 替代 `List[X]`：
  ```python
  def process(items: list[float]) -> dict[str, int]:
      ...
  ```
- 避免 `Any` 作为逃逸手段——尽可能精确定义类型

#### 4. 命名约定

| 元素 | 规范 | 示例 |
|------|------|------|
| 类名 | PascalCase | `NLightDM`, `BaseFrame`, `PhaseWrapOptimizer` |
| 函数/方法 | snake_case | `calculate_sharpness`, `get_centroid` |
| 变量 | snake_case | `exposure_time_ms`, `dm_unit_mask` |
| 常量 | SCREAMING_SNAKE | `MAX_VOLTAGE`, `DEFAULT_THRESHOLD` |
| 私有属性/方法 | `_` 前缀 | `_device_id`, `_set_state()` |
| 类型变量 | PascalCase | `T`, `T_co` |

#### 5. 日志（必须使用 loguru）

- **禁止使用 `print()` 输出调试/状态信息**——全部使用 `loguru.logger`
- 禁止使用标准库 `import logging` / `logging.getLogger()`
- 使用 loguru 的格式化字符串（惰性求值）：
  ```python
  # 正确 — 惰性求值，日志级别抑制时不格式化
  logger.info("Device {} initialized with {} actuators", device_id, n)

  # 错误 — 非惰性求值，即使不输出也会格式化
  logger.info(f"Device {device_id} initialized with {n} actuators")
  ```
- 日志级别规范：
  - `logger.debug()`：详细调试信息（函数入口/出口、中间变量）
  - `logger.info()`：重要状态变更（设备连接/断开、优化启动/完成）
  - `logger.warning()`：可恢复的异常（设备连接失败但继续、参数超范围）
  - `logger.error()`：不可恢复的异常（设备断开、关键数据缺失）
  - `logger.exception()`：在 `except` 块中记录完整异常堆栈

#### 6. 异常处理

- **禁止宽泛的异常捕获**：不使用 `except:` 或 `except Exception:` 而不指定类型
  ```python
  # 正确
  except ConnectionError:
      logger.error("Device connection lost")
  except ValueError as e:
      logger.warning("Invalid parameter: {}", e)

  # 错误 — 掩盖所有错误
  except Exception:
      pass
  ```
- 自定义异常使用 `*Error` 后缀，继承 `Exception`：
  ```python
  class DeviceError(Exception): ...
  class DeviceNotFoundError(DeviceError): ...
  ```
- 资源管理使用上下文管理器（`__enter__` / `__exit__`）

#### 7. 配置管理

- **所有 `os.environ` 读取集中在 `config.py`**，其他文件从 `ao_shaping.config` 导入
- 避免在多个文件中重复读取相同的环境变量
- 对于必须使用时再确定的配置（如硬件 ID），通过参数传递而非全局读取
  ```python
  # config.py
  @dataclass
  class Config:
      far_cam_id: int = 0
      near_cam_id: int = 1
      ideal_spot_radius: int = 7

  # 其他文件
  from ao_shaping.config import ao_config
  cam_id = ao_config.far_cam_id
  ```

#### 8. DataClass 与 Enum

- 结构化数据使用 `@dataclass`（无继承需求时）或 `@dataclass` + `ABC`（有继承时）
  ```python
  @dataclass
  class DeviceParameter:
      name: str
      value: Any
      value_type: type = float
      min_value: float | None = None
      max_value: float | None = None
  ```
- 状态/类型定义使用 `Enum` 配合 `auto()`：
  ```python
  from enum import Enum, auto

  class DeviceState(Enum):
      UNKNOWN = auto()
      DISCONNECTED = auto()
      READY = auto()
      ERROR = auto()
  ```

#### 9. 模块与包结构

- **每个子目录必须包含 `__init__.py`**（即使是空文件或仅 docstring）
- 模块文件行数建议：工具/算法模块 < 500 行，驱动/优化器 < 800 行。超过 800 行应考虑拆分为子模块
- `utils/` 是叶子模块：不能反向依赖 `algorithm/`、`drivers/`、`optimizer/` 等高阶包
  如果必须引用，使用以下模式之一：
  - `TYPE_CHECKING` 保护（仅类型检查时导入）
  - 函数内部的延迟导入（deferred local import）

#### 10. 性能优化模式

- **默认使用 NumPy** 实现数值计算
- 对热点循环，提供 Numba JIT 加速版本：
  ```python
  @numba.njit(cache=True)
  def _calculate_sharpness_numba(img: np.ndarray) -> float:
      ...
  ```
- 对 GPU 加速场景，提供 CuPy 版本并包含降级回退：
  ```python
  try:
      import cupy as cp
      CUPY_AVAILABLE = cp.cuda.is_available()
  except (ImportError, AttributeError):
      CUPY_AVAILABLE = False
  ```
- 避免深层嵌套循环（3+ 层），优先使用向量化操作

#### 11. 测试规范

- **测试文件必须可脱机运行**：在 `tests/` 中使用模拟设备（`MockDM`、`SimTurbulenceAOEnv`）
- 需要硬件的测试使用 `pytest.skip("Requires hardware")` 条件跳过
- 测试验证优化器输出字典中是否包含预期的字段（Recorder 模式）
- 禁止删除失败测试——应修复代码而非测试
- 使用 pytest 而非 unittest

### 贡献流程

1. Fork项目
2. 创建功能分支
3. 提交更改 (遵循conventional commits)
4. 发起Pull Request

### 测试

```bash
# 运行所有测试
pytest -v

# 运行特定测试文件
pytest tests/ao_shaping/utils/test_spots_calc.py

# 运行测试并查看输出
pytest -s

# 运行特定测试函数
pytest tests/ao_shaping/utils/test_spots_calc.py::TestCentroid::test_centroid_uniform
```

### 文档

- [AGENTS.md](AGENTS.md): 开发指南和项目架构
- [drivers/AGENTS.md](src/ao_shaping/drivers/AGENTS.md): 硬件驱动文档

## 近期更新

### v0.3.0 (2026-05)
- **DM 统一接口重构**: 所有 DM 继承自 `base.DM`，提供标准方法 (V_Min/V_Max, DM_NUM, default_dm_unit_mask, check_dm_unit_grad_safe, send_voltages)
- MicroDM 容错连接: `open()` 允许个别控制器连接失败，记录 warning 后继续
- MicroDM 连接状态检查: `get_connection_status()` 返回 ping 可达性和 TCP 连接状态
- MicroDM 单控制器管理: `connect_controller()` / `disconnect_controller()` / `reconnect_controller()`
- MicroDM 排除控制器: 初始化支持 `exclude_ips` 和 `exclude_ids` 参数
- MicroDM 新增 `ControllerStatus` dataclass 和 `_ping_host()` 静态方法
- PIB 优化器多 DM 支持: `optimize_pib` 接受 `dm` 参数，兼容任意 DM 子类
- 轴向光束优化器 `--dm_type` 参数: 支持 nlight/micro/zernike/hadamard，自动检测在线 DM

### v0.2.0 (2026-03)
- 新增SLM (Santec SLM200) 支持
- 重构相机驱动 (DahengCamManager, MIICamDriver)
- 支持14位相机模式
- 新增U-Net+GAN相位预测训练
- 集成WandB实验跟踪
- 重构仿真模块到 `drivers/sim/`
- 新增Zernike多项式计算工具
- 添加AGENTS.md开发文档

### v0.1.0
- 基础波前优化功能
- PIB优化功能
- 串行流水线优化
- SAC强化学习集成