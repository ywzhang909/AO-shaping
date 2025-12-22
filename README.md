# AO-Shaping 自适应光学整形系统

## 项目简介

AO-Shaping是一个先进的自适应光学(AO)系统，用于波前校正和光束整形。该项目集成了多种优化算法，包括基于波前传感器(WFS)的优化和无波前优化方法，能够通过变形镜(DM)对光波前进行精确控制，实现高质量的光束输出。

## 主要特性

- **多优化算法**: 支持基于波前传感器的RMS优化、无波前的PIB优化、贝叶斯优化和启发式搜索
- **图形用户界面**: 提供基于PyQt6的可视化控制界面
- **硬件支持**: 兼容Thorlabs WFS波前传感器、NLight变形镜和大恒相机
- **可视化工具**: 提供实时波前和电压可视化功能
- **灵活的参数配置**: 支持丰富的命令行参数和GUI参数调节
- **模块化设计**: 清晰的代码结构，易于扩展和维护

## 项目结构

```
src/
└── ao_shaping/                 # 主程序包
    ├── main.py                 # 命令行入口点
    ├── wf_runner.py            # 波前优化器
    ├── axis_beam_runner.py     # 轴向光束优化器
    ├── combined_runner.py      # 组合优化器
    ├── heuristic_search_runner.py # 启发式搜索优化器
    ├── algorithm/              # 优化算法实现
    │   ├── adam.py             # ADAM优化算法
    │   ├── heuristic_search.py # 启发式搜索算法
    │   └── target_func.py      # 目标函数
    ├── drivers/                # 硬件驱动
    │   ├── ccd/                # CCD相机驱动
    │   ├── dm/                 # 变形镜驱动
    │   ├── tm/                 # 串口通信驱动
    │   └── wfs/                # 波前传感器驱动
    ├── optimizer/              # 优化器模块
    │   ├── rl/                 # 强化学习优化器
    │   ├── wf/                 # 波前优化器
    │   └── wfless/             # 无波前优化器
    ├── utils/                  # 工具函数
    ├── display/                # 显示和可视化模块
    └── gui/                    # 图形用户界面
        ├── main.py             # GUI入口点
        ├── main_window.py      # 主窗口
        ├── panels/             # 各种面板组件
        └── workers/            # 后台工作线程
```


## 安装指南

### 环境要求

- Python 3.13+
- Windows/Linux/macOS

### 安装步骤

1. 克隆项目仓库:
```bash
git clone <repository_url>
cd AO-shaping
```

2. 推荐使用uv工具创建虚拟环境并安装依赖:
```bash
# 安装uv工具
pip install uv

# 创建虚拟环境并安装依赖
uv sync
```

或者使用传统的pip方式:

2. 创建虚拟环境:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate     # Windows
```

3. 安装依赖:
```bash
pip install -e .
```

## 使用说明

### 图形用户界面 (推荐)

项目提供了一个功能完整的图形用户界面，可以通过以下命令启动：

```bash
python -m src.ao_shaping.gui
```

或者直接运行主窗口文件：

```bash
python src/ao_shaping/gui/main.py
```

GUI界面提供了以下功能：
- 变形镜单元可视化控制
- 多种优化算法支持（波前优化、轴向光束优化、组合优化、贝叶斯优化、启发式搜索）
- 实时参数调整
- CCD相机参数设置
- 变形镜参数配置
- 实时可视化显示

### 命令行界面

项目提供了统一的命令行界面，通过`main.py`作为入口点：

```bash
python src/ao_shaping/main.py [OPTIONS] COMMAND [ARGS]...
```

#### 全局选项
- `--help`: 显示帮助信息

#### 波前优化器 (wf)
```bash
python src/ao_shaping/main.py wf [OPTIONS]
```

选项:
- `-d, --dir`: 数据保存根目录 (默认: data)
- `-e, --epochs`: 优化迭代次数 (默认: 20000)
- `-r, --wfs_res`: WFS分辨率 (默认: 768)
- `-p, --pupil_diameter`: 瞳孔直径 (默认: 2.7)
- `-t, --early_stop_threshold`: 早停阈值 (默认: 0.0)
- `--debug`: 开启调试模式
- `--show`: 显示远场光斑CCD图像和优化历史

示例:
```bash
python src/ao_shaping/main.py wf --epochs 10000 --debug
```

#### 轴向光束优化器 (pib)
```bash
python src/ao_shaping/main.py pib [OPTIONS]
```

选项:
- `-d, --root_dir`: 数据保存根目录 (默认: data)
- `-f, --load_file`: 加载优化结果文件 (默认: rms)
- `--cam_id`: 远场光斑CCD设备ID (默认: 环境变量Far_Cam_ID或0)
- `-c, --center`: 场光斑CCD中心位置 (默认: mass)
- `-t, --exposure_time_ms`: 远场光斑CCD曝光时间(毫秒) (默认: 60)
- `-e, --epochs`: 优化迭代次数 (默认: 4000)
- `-r, --r_bucket`: 渲染半径桶大小 (默认: 0)
- `--delta`: 优化步长 (默认: 2)
- `--lr`: 优化学习率 (默认: 0.0)
- `--weight_decay`: 权重衰减 (默认: 0.0)
- `--shrink_iter`: 优化迭代次数后收缩半径桶和步长 (默认: 300)
- `--shrink_ratio`: 收缩半径桶和步长比例 (默认: 0.8)
- `-s, --cam_size`: 相机开窗大小 (默认: 200)
- `-b, --target_max_brightness`: 目标最大亮度值 (默认: 90)
- `--debug`: 开启调试模式
- `--show`: 显示远场光斑CCD图像和优化历史

示例:
```bash
python src/ao_shaping/main.py pib --epochs 5000 --cam_id 1 --debug
```

#### 组合优化器 (combine)
```bash
python src/ao_shaping/main.py combine [OPTIONS]
```

选项:
- `-d, --dir`: 数据保存根目录 (默认: data)
- `-f, --load_file`: 加载优化结果文件 (默认: None)
- `-e, --epochs`: 优化迭代次数 (默认: 8000)
- `-E, --wf_epochs`: WF优化迭代次数 (默认: 8000)
- `-R, --wfs_res`: WFS分辨率 (默认: 768)
- `-p, --pupil_diameter`: 瞳孔直径 (默认: 2.7)
- `-c, --cam_id`: 远场光斑CCD设备ID (默认: 环境变量Far_Cam_ID或0)
- `-t, --exposure_time_ms`: 远场光斑CCD曝光时间(毫秒) (默认: 500)
- `-s, --cam_size`: 相机开窗大小 (默认: 160)
- `-r, --rms_threshold`: RMS阈值 (默认: 0.12)
- `-u, --dm_unit_mask`: DM单元掩码 (默认: all)
- `--debug`: 开启调试模式

示例:
```bash
python src/ao_shaping/main.py combine --epochs 6000 --debug
```

#### 贝叶斯优化器 (bayes-opt)
```bash
python src/ao_shaping/main.py bayes-opt [OPTIONS]
```

#### 启发式搜索优化器 (heuristic)
```bash
python src/ao_shaping/main.py heuristic [OPTIONS]
```

### 单独运行脚本

除了使用统一入口，也可以直接运行各个优化器脚本:

1. 波前优化器:
```bash
python src/ao_shaping/wf_runner.py [OPTIONS]
```

2. 轴向光束优化器:
```bash
python src/ao_shaping/axis_beam_runner.py [OPTIONS]
```

3. 组合优化器:
```bash
python src/ao_shaping/combined_runner.py [OPTIONS]
```

4. 贝叶斯优化器:
```bash
python src/ao_shaping/optimizer/wfless/bayes_opt_runner.py [OPTIONS]
```

5. 启发式搜索优化器:
```bash
python src/ao_shaping/heuristic_search_runner.py [OPTIONS]
```

## 硬件支持

- **波前传感器**: Thorlabs WFS系列
- **变形镜**: NLight系列
- **相机**: 大恒相机系列
- **数据采集卡**: NI DAQ设备
- **串口设备**: 用于与TM设备通信

## 开发指南

### 代码规范

- 遵循PEP8代码风格
- 使用类型提示
- 编写单元测试

### 贡献流程

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 发起Pull Request

### 测试

运行测试套件:
```bash
pytest
```

### 依赖管理

本项目使用`uv`工具进行依赖管理，配置文件为`pyproject.toml`。主要依赖包括：

- **科学计算**: numpy, pandas, scipy
- **可视化**: matplotlib, PyQt6
- **硬件接口**: pyserial, nidaqmx
- **机器学习**: scikit-learn, scikit-optimize, gymnasium
- **工具库**: click, loguru, tqdm
- **开发工具**: pytest, pytest-cov, coredumpy

如需添加新依赖，请在`pyproject.toml`文件中添加，并运行`uv lock`更新锁定文件。