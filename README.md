# AO-Shaping 自适应光学整形系统

## 项目简介

AO-Shaping是一个基于强化学习的自适应光学(AO)系统，用于波前校正和光束整形。该项目集成了多种优化算法，包括基于波前传感器(WFS)的优化和无波前优化方法，能够通过变形镜(DM)对光波前进行精确控制，实现高质量的光束输出。

## 主要特性

- **多优化算法**: 支持基于波前传感器的RMS优化和无波前的PIB优化
- **强化学习集成**: 使用SAC算法进行波前优化
- **硬件支持**: 兼容Thorlabs WFS波前传感器、NLight变形镜、大恒相机、MIICAM和Santec SLM200
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
│   │   │   ├── dm/              # 变形镜 (NLight)
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
- **NLight系列**: 支持电压控制和电压差安全检查

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

### 代码规范

- 遵循PEP8代码风格
- 使用类型提示 (Python 3.12+)
- 使用`loguru`进行日志记录
- 编写单元测试

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