# AO-Shaping 自适应光学整形系统

## 项目简介

AO-Shaping是一个基于强化学习的自适应光学(AO)系统，用于波前校正和光束整形。该项目集成了多种优化算法，包括基于波前传感器(WFS)的优化和无波前优化方法，能够通过变形镜(DM)对光波前进行精确控制，实现高质量的光束输出。

## 主要特性

- **多优化算法**: 支持基于波前传感器的RMS优化和无波前的PIB优化
- **强化学习集成**: 使用SAC算法进行波前优化
- **硬件支持**: 兼容Thorlabs WFS波前传感器、NLight变形镜和大恒相机
- **可视化工具**: 提供实时波前和电压可视化功能
- **数据处理**: 集成Dask进行高性能数据处理和分析
- **实验跟踪**: 使用SwanLab进行实验管理和可视化

## 项目结构

src/ 
├── ao_shaping/ # 主程序包 
│ ├── main.py # 命令行入口点 
│ ├── wf_runner.py # 波前优化器 
│ ├── axis_beam_runner.py # 轴向光束优化器 
│ ├── combined_runner.py # 组合优化器 
│ ├── algorithm/ # 优化算法实现 
│ ├── drivers/ # 硬件驱动 
│ ├── optimizer/ # 优化器模块 
│ ├── utils/ # 工具函数 
│ └── display/ # 显示和可视化模块


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

2. 创建虚拟环境:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate     # Windows
```

3. 安装依赖:
```bash
pip install -r requirements.txt
```

## 使用说明

### 主要操作命令

项目提供了统一的命令行界面，通过`main.py`作为入口点：

```bash
python src/ao_shaping/main.py [OPTIONS] COMMAND [ARGS]...
```

#### 全局选项
- `--debug`: 开启调试模式
- `--dir`: 指定数据保存根目录 (默认: data)
- `--show`: 显示远场光斑CCD图像和优化历史

#### 波前优化器 (wf)
```bash
python src/ao_shaping/main.py wf [OPTIONS]
```

选项:
- `-e, --epochs`: 优化迭代次数 (默认: 20000)
- `-r, --wfs_res`: WFS分辨率 (默认: 768)
- `-p, --pupil_diameter`: 瞳孔直径 (默认: 2.7)
- `-t, --early_stop_threshold`: 早停阈值 (默认: 0.0)

示例:
```bash
python src/ao_shaping/main.py wf --epochs 10000 --debug
```

#### 轴向光束优化器 (pib)
```bash
python src/ao_shaping/main.py pib [OPTIONS]
```

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
python src/ao_shaping/main.py pib --epochs 5000 --cam_id 1 --debug
```

#### 组合优化器 (combine)
```bash
python src/ao_shaping/main.py combine [OPTIONS]
```

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
python src/ao_shaping/main.py combine --epochs 6000 --debug
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

## 硬件支持

- **波前传感器**: Thorlabs WFS系列
- **变形镜**: NLight系列
- **相机**: 大恒相机系列
- **数据采集卡**: NI DAQ设备

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