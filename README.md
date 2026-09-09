# AO-Shaping 自适应光学整形系统

## 项目简介

AO-Shaping是一个基于强化学习的自适应光学(AO)系统，用于波前校正和光束整形。该项目集成了多种优化算法，包括基于波前传感器(WFS)的优化和无波前优化方法，能够通过变形镜(DM)对光波前进行精确控制，实现高质量的光束输出。

## 主要特性

- **多优化算法**: 支持基于波前传感器的RMS优化和无波前的PIB优化
- **强化学习集成**: 使用SAC算法进行波前优化
- **硬件支持**: 兼容Thorlabs WFS波前传感器、NLight变形镜、R50Power MicroDM、大恒相机、MIICAM、Santec SLM200和NI DAQ ADC
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
│   │   │   ├── zernike_matrix_runner.py  # Zernike响应矩阵校准与闭环控制
│   │   │   ├── gs_hologram_runner.py   # Gerchberg-Saxton全息图生成器
│   │   │   ├── gs_square_runner.py     # GS闭环光束整形优化器 (方形)
│   │   │   ├── diff_shaping_runner.py  # 可微分闭环光束整形优化器 (PyTorch)
│   │   │   ├── dm_matrix_runner.py     # DM响应矩阵标定
│   │   │   ├── alt_voltage_runner.py   # 交替电压下发 (R50Power + ADC采集)
│   │   │   ├── full_voltage_runner.py  # 全量交替电压下发 (AsyncMicroDM)
│   │   │   └── combined_runner.py      # [已废弃] 使用pipeline_runner代わり
│   │   ├── algorithm/           # 优化算法 (Adam, SGD, Muon, 可微分光束整形等)
│   │   │   ├── gerchberg_saxton.py     # Gerchberg-Saxton 相位恢复算法
│   │   │   ├── differentiable_beam.py  # 可微光束整形 (双向传播模拟)
│   │   │   └── beam_shaping_utils.py   # 光束整形共享工具 (目标生成/指标/方形尺寸)
│   │   ├── drivers/             # 硬件驱动
│   │   │   ├── ccd/             # 相机 (Daheng, MiiCam)
│   │   │   ├── dm/              # 变形镜 (NLight, R50Power MicroDM)
│   │   │   ├── slm/             # 空间光调制器 (Santec, WavefrontCorrection)
│   │   │   ├── wfs/             # 波前传感器 (Thorlabs)
│   │   │   ├── adc/             # NI DAQ ADC 电压采集
│   │   │   ├── tm/              # 定时模块 (Serial/FSM)
│   │   │   ├── sim/             # 数字孪生仿真
│   │   │   └── mock_devices.py  # 测试用模拟设备
│   │   ├── optimizer/           # 高层优化器
│   │   │   ├── wf/              # 波前优化 (RMS)
│   │   │   ├── wfless/          # 无波前优化 (PIB)
│   │   │   └── rl/              # 强化学习 (SAC, LR-WFS)
│   │   ├── utils/               # 工具函数 (spots_calc, wavefront_calc, resample)
│   │   ├── ml/                  # 机器学习 (U-Net+GAN, 训练, 模型)
│   │   │   ├── trainer/         # 训练器
│   │   │   ├── models/          # 神经网络模型
│   │   │   └── wandb_logger.py  # WandB日志
│   │   ├── tools/               # 独立工具 (tools/slm 包: SLM相位捕获, LUT校准; Micro-DM逐单元图像采集)
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

#### 全息图生成器 (gs)
```bash
python src/ao_shaping/main.py gs [OPTIONS]
```
等同于: `python -m ao_shaping.runners.gs_hologram_runner`

基于Gerchberg-Saxton算法的全息图生成和优化。使用Santec SLM200空间光调制器和Daheng CCD相机。

选项:
- `--target-image`: 目标图像路径 (灰度图，将转为振幅分布)
- `--target-shape`: 预设目标形状 (gaussian, circle, square, annular, grid, cross) (默认: gaussian)
- `-i, --iterations`: GS算法迭代次数 (默认: 50)
- `-d, --distance`: 传播距离 (米) (默认: 0.1)
- `-l, --wavelength`: 激光波长 (纳米) (默认: 1064)
- `--slm-wavelength`: SLM工作波长 (纳米) (默认: 1064)
- `--slm-number`: SLM设备编号 (默认: 1)
- `--cam-id`: CCD相机ID (默认: FAR_CAM_ID/0)
- `--cam-center`: CCD中心位置 'x,y' (默认: 自动检测)
- `--cam-size`: CCD开窗大小 (像素) (默认: 400)
- `--cam-exposure`: CCD曝光时间 (毫秒) (默认: 50)
- `-s, --save-dir`: 结果保存目录 (默认: data/gs_hologram)
- `--use-hardware`: 使用实际硬件 (SLM+CCD)，否则仅模拟计算
- `--adaptive`: 启用自适应GS (使用CCD反馈迭代优化)
- `--adaptive-iterations`: 自适应迭代次数 (默认: 3)
- `--show`: 显示结果图像

示例:
```bash
DEBUG=1 python src/ao_shaping/main.py gs --target-shape gaussian --iterations 100 --use-hardware
```

#### GS闭环光束整形 (gs-square)
```bash
python src/ao_shaping/main.py gs-square [OPTIONS]
```
等同于: `python -m ao_shaping.runners.gs_square_runner`

基于 **Gerchberg-Saxton 算法 + CCD 反馈**的闭环光束整形：迭代优化 SLM 相位图案，将远场光斑整形为方形。每个外迭代：CCD 采集 → 测光斑直径 → 计算方形目标尺寸 → 运行 GS 生成相位 → 下发 SLM（内存槽轮换）→ 重新采集 → 计算方形质量评分 → 收敛判断。实测光束成为下一次 GS 的光源振幅，形成真正闭环。

选项:
- `--camera-type`: 相机类型 (daheng / miicam, 默认: daheng)
- `--cam-id`: 相机 ID (默认: FAR_CAM_ID/0)
- `--exposure-ms`: 相机曝光时间 (毫秒, 默认: 50); **miicam 建议 ≥0.2ms** (<0.1ms 信号淹没在传感器噪声中)
- `--slm-number`: SLM 设备编号 (默认: 1)
- `--slm-wavelength`: SLM 工作波长 nm (默认: 1064)
- `-i, --gs-iterations`: GS 内迭代次数 (默认: 100)
- `--gs-factor`: 方形/光斑尺寸因子 (默认: 1.5) — 仅当未指定 `--target-px` 时使用
- `--target-px`: **目标方形在相机上的像素宽度** (推荐显式指定, 默认: 按 `--gs-factor×光斑直径`)。光斑 90% 环围能量直径 (spot_d) 在光束超出传感器时被裁剪而失真膨胀, 依它计算会把方形推到超过传感器尺寸 (全帧点亮/指标失真)。应选一个小于传感器高度 (MiiCam 1520px) 的像素宽度, 例如 1200。
- `--focal-length`: 焦距/传播距离 m (默认: 0.1)
- `--gs-energy`: 光斑测量环围能量 (默认: 0.90)
- `--p-cam`: 相机像素间距 m (默认: 使用SLM间距)
- `--pixel-scale`: 像素缩放比 k=SLM网格边长/相机亮区宽度 (默认: 自动标定)。实测 SLM→MiiCam k≈0.414 (每 SLM 8µm 像素 ≈2.417 相机像素)。**光路调整后必须重新标定**。
- `--propagation`: GS 传播模型 (asm / fft, 默认: asm)
- `-n, --outer-iterations`: 最大外迭代次数 (默认: 10)
- `--convergence-threshold`: 收敛评分阈值 0~1 (默认: 0.95)
- `--settle-time`: SLM 稳定等待时间 s (默认: 0.5)
- `--n-sample`: 相机每次采样平均帧数 (默认: 3)
- `-o, --output`: 输出目录 (默认: data/gs_square)
- `--display/--no-display`: 启用 pygame 实时可视化 GS 迭代过程 (默认: False)

`--display` 启用后弹出 pygame 窗口，四面板实时显示: GS 相位图案、目标方形振幅、远场光斑图像（伪彩）、GS 误差收敛曲线，标题栏显示当前评分/边长/光斑直径。

**每外迭代记录 (pkl)**: 每次迭代将相位、相机图像及质量指标增量保存为 `data/gs_square/gs_square_records.pkl` (zip 压缩)，Ctrl+C 中断也不丢失已保存记录。记录字段: `quality_score`, `iteration`, `side`, `spot_d`, `center`, `gs_final_error`, `aspect_ratio`, `squareness`, `uniformity_cv`, `encircled_energy`, `phase`(float64), `phase_gray`(uint16), `image`(float64), `target`(float64)。

示例:
```bash
# 基本闭环整形, 启用pygame可视化
DEBUG=1 python src/ao_shaping/main.py gs-square --outer-iterations 10 --display

# MiiCam + 更多GS迭代, 不显示
python src/ao_shaping/main.py gs-square --camera-type miicam --gs-iterations 150 -o data/gs_square

# MiiCam 方形直接以相机像素指定 (推荐), 自动标定像素缩放
python src/ao_shaping/main.py gs-square --camera-type miicam --exposure-ms 0.5 \
    --gs-iterations 60 --outer-iterations 5 --target-px 1200 -o data/gs_square
```

**GS 传播模型 (`propagation`)**: `gerchberg_saxton` 支持两种传播模型:
- `asm` (默认): Angular Spectrum Method，精确的近/远场传播，逐迭代正向 + 反向各一次传播。传播子仅依赖 (网格形状, 间距, 距离, 波长)，已做 **lru_cache 预计算缓存**，避免逐迭代重建 meshgrid/exp 传播因子，数值不变但大幅提速。
- `fft`: 单 FFT Fraunhofer 焦平面模型（同 differential_shaping 的 GS），焦平面视为光源平面的傅里叶变换，完全去除传播子构造开销，仅有单 FFT/IFFT 对，适合远场整形追求最高吞吐的场景。

gs-square 的核心 GS 调用位于 `src/ao_shaping/algorithm/gerchberg_saxton.py`。

#### 可微分光束整形 (diff-shaping)
```bash
python src/ao_shaping/main.py diff-shaping [OPTIONS]
```
等同于: `python -m ao_shaping.runners.diff_shaping_runner`

基于 **PyTorch 可微分优化 + CCD 反馈**的闭环光束整形：将远场光斑整形为 square / circle / gaussian / spot 目标形状。核心算法 `src/ao_shaping/algorithm/differentiable_shaping.py` 用梯度下降直接优化 SLM 相位图，损失 = 均匀性(CV) + 效率(EE) + 零级惩罚 + 平滑正则的加权和，支持 `fft` (单 FFT 夫琅禾费焦平面, 高速) 与 `asm` (角谱法, 精确) 两种传播模型; CCD 实测光束反馈到目标/质量评分, 形成闭环。

选项:
- `--camera-type`: 相机类型 (daheng / miicam, 默认: daheng)
- `--cam-id`: 相机 ID (默认: FAR_CAM_ID/0)
- `--exposure-ms`: 相机曝光时间 (毫秒, 默认: 自动解析 — **miicam=0.02ms (1064nm 近饱和基线)**, daheng=50ms); miicam 建议 ≥0.011ms 且通常 0.02~1ms 防饱和
- `--cam-bit-depth`: MiiCam 输出位深 (默认: 8)
- `--slm-number`: SLM 设备编号 (默认: 1)
- `--slm-wavelength`: SLM 工作波长 nm (默认: 1064)
- `--target-shape`: 目标形状 (square / circle / gaussian / spot, 默认: square)
- `--target-size`: 目标尺寸 (SLM 网格 px); 0=自动 (默认: 0)
- `--target-px`: 目标在相机上的像素宽度; 推荐显式指定 (默认: 按 `--gs-factor×光斑直径`)
- `--gs-factor`: 目标/光斑尺寸因子 (默认: 1.5)
- `--focal-length`: 焦距/传播距离 m (默认: 0.1)
- `--gs-energy`: 光斑测量环围能量 (默认: 0.90)
- `--cell-spacing`: SLM 像素间距 um (默认: 8)
- `--p-cam`: 相机像素间距 m (默认: 使用SLM间距)
- `--pixel-scale`: 像素缩放比 k=SLM网格边长/相机亮区宽度 (默认: 自动标定)。**光路调整后必须重新标定**。
- `--propagation`: 传播模型 (fft / asm, 默认: fft)
- `-i, --dl-iterations`: 每次外迭代的梯度优化内迭代次数 (默认: 500; **600 达 CV<0.1**)
- `--optimizer`: 梯度优化器 (adam / lbfgs, 默认: adam)
- `--lr`: 梯度优化学习率 (默认: 3e-2)
- `--w-uniformity`: 均匀性损失权重 (默认: 0.4)
- `--w-efficiency`: 效率损失权重 (默认: 0.6; 取 ≥ 均匀性权重可先集中能量再展平)
- `--w-zero-order`: 零级损失权重 (默认: **0.0**; 非零会把能量推出居中目标, 详见下)
- `--w-smoothness`: 平滑损失权重 (默认: **0.0**; 抑制方形锐边所需的高频相位)
- `--seed`: 随机种子 (默认: None)
- `--device`: 计算设备 (auto / cuda / cpu, 默认: auto, 自动用本机 GPU 否则 CPU)
- `-n, --outer-iterations`: 最大外迭代次数 (默认: 10)
- `--convergence-threshold`: 收敛评分阈值 0~1 (默认: 0.95)
- `--settle-time`: SLM 稳定等待时间 s (默认: 0.5)
- `--n-sample`: 相机每次采样平均帧数 (默认: 3)
- `--refine/--no-refine`: 每次外迭代后运行额外细化梯度通道 (默认: False)
- `-o, --output`: 输出目录 (默认: data/diff_shaping)
- `--display/--no-display`: 启用 pygame 实时可视化迭代过程 (默认: False)

**权重默认值经实验标定** (详见 `docs/slm_differential_shaping/` 验证报告):
- 默认权重 `[.4,.4,.1,.1]` + `lr=1e-2` 曾经是坏的: 零级惩罚把能量从居中目标推出 (EE 0.84→0.07), 平滑项抑制方形锐边所需高频相位, 低学习率还使 `asm` 困在平凡均匀临界点 (loss 不降反升)。
- 成功配置 `w=[.4,.6,0,0]`, `lr=3e-2`, 600 内迭代: fft/adam 方形 CV<0.1 / EE≈0.84 (seed 1-3 稳健), asm CV≈0.001 / EE≈0.90, spot CV≈0 / EE≈0.87; `--optimizer lbfgs --lr 1.0` 60 步即近平顶 (CV≈0)。
- 相位初始化使用 0.1×randn 小噪声, 逃离零相位处的零梯度退化点; torch 版 ASM 与 numpy 参考实现数值一致 (~1e-11), 可互换对比。

示例:
```bash
# 方形成形 (默认配置)
python src/ao_shaping/main.py diff-shaping

# spot 聚焦, MiiCam, 更多内迭代达高均匀性
python src/ao_shaping/main.py diff-shaping --camera-type miicam --target-shape spot \
    --dl-iterations 600 -o data/diff_shaping

# lbfgs 快速近平顶方形
python src/ao_shaping/main.py diff-shaping --optimizer lbfgs --lr 1.0 -o data/diff_shaping
```

#### SLM 灰度→相位 LUT 校准 (slm-lut)
```bash
python src/ao_shaping/main.py slm-lut [OPTIONS]
```
等同于: `python -m ao_shaping.tools.slm.slm_lut_runner`

在 SLM 上同时写入**半屏参考光栅 + 半屏测试光栅**(上半屏恒定满深度闪耀参考, 下半屏扫描深度/偏移), 同帧测量两半 +1 级衍射效率的比值 (相互抵消激光漂移), 由 sinc² 效率曲线反演灰度→相位映射, 输出正向/逆向 LUT (`lut_forward.csv`/`lut_inverse.csv`/`lut.npz`), 之后可通过 `slm.load_lut(dir)` 加载, 由驱动在相位写入时自动补偿灰度↔相位非线性。

选项:
- `--method`: 扫描方法 (depth=缩放闪耀峰值灰度 / offset=均匀灰度偏移, 默认: depth)
- `--period-ref`: 参考半屏闪耀光栅周期 (SLM px, 默认: 64)
- `--period-test`: 测试半屏闪耀光栅周期 (SLM px, 默认: 32)
- `--gray-step`: 灰度扫描步长 (默认: 16)
- `--exposure-ms`: 初始相机曝光 (ms, 默认: 0.03)
- `--n-frames`: 每灰度点平均帧数 (默认: 10)
- `--camera-type`: 相机类型 (miicam/daheng, 默认: miicam)
- `--cam-id`: 相机 ID (默认: 0)
- `--settle-time`: SLM 写入后稳定等待 s (默认: 0.3)
- `--slm-number`: SLM 设备编号 (默认: 1)
- `--slm-wavelength`: SLM 工作波长 nm; 2π 对应灰度由设备动态查询, 禁硬编码 (默认: 1064)
- `--spot-window`: 光斑 ROI 窗口 (奇数, 默认: 41)
- `--bright-floor` / `--saturation-stop`: 联合自动曝光阈值 (默认: 0.02 / 0.9)
- `-o, --output`: 输出目录 (默认: data/slm_lut)
- `--display/--no-display`: 是否弹出 matplotlib 图窗 (默认: False)

输出目录 `data/slm_lut/run-<时间戳>/`: `lut_calibration.png` (η/g、φ/g、逆LUT三图), `lut/` (LUT 文件), `calibration_frame.npy`, `records.pkl` (g/eta/phi/inverse_gray/meta/p_ref/p_test 全量记录)。

示例:
```bash
# 默认 depth 扫描 (推荐)
python src/ao_shaping/main.py slm-lut --period-ref 64 --period-test 32 -o data/slm_lut

# offset 对照方法
python src/ao_shaping/main.py slm-lut --method offset -o data/slm_lut
```

**注意**: 校准图案 (半屏闪耀光栅) 使用 uint16 原始灰度直接 `display_data` 写入, 严禁经过 `create_phase_from_array()` (弧度转换会损坏灰度值)。

#### SLM 硬件自检 (slm-diagnose)
```bash
python src/ao_shaping/main.py slm-diagnose [OPTIONS]
```
等同于: `python -m ao_shaping.tools.slm.slm_diagnose`

在 2f Fourier 光路下对 SLM + MiiCam 做逐级硬件自检, 定位"面板不调制光"类故障 (2026-09 诊断固化, 三步证据链):

1. **freeze (面板冻结检测)**: flat/全屏光栅/上下半屏光栅写入**轮换内存槽**, 对比各帧是否随图案变化 — 全同 ⇒ LCOS 冻结。
2. **modulate (调制能力检测)**: `set_grayscale` 0→1023 扫描, 0 级桶能量须有 ~993 灰度周期 — 无周期 ⇒ 面板不调制光。此模式下 `get_displayed_memory_number` 报错码 1 是**正常**行为。
3. **linearity (到达光强检测)**: 曝光 ×4/×20, 峰值亮度须增长 — 恒定峰值 ⇒ 到达相机光强比已知 ~0.02ms 近饱和基线弱 >100×。

选项:
- `--slm-number`: SLM 设备编号 (默认: 1)
- `--slm-wavelength`: SLM 工作波长 nm (默认: 1064)
- `--cam-id`: MiiCam 相机 ID (默认: 0)
- `--period-ref` / `--period-test`: 光栅周期 px (默认: 64 / 32)
- `--exposure-ms`: 自检曝光 ms (默认: 2.0)
- `--settle-s`: SLM/相机稳定等待 s (默认: 1.0)
- `--step`: 只跑某步 freeze/modulate/linearity (默认: all)
- `-o, --output`: 保存诊断报告目录 (默认: 不保存)

**已知约束**: DVI 模式 (`video_mode=1`) 的 `open()` 可能挂起, 且挂起后 memory 模式也挂直到**物理断电** —— 本工具只用 memory 模式, 绝不自动尝试 DVI。

示例:
```bash
# 全量三步自检
python src/ao_shaping/main.py slm-diagnose

# 只查面板是否冻结
python src/ao_shaping/main.py slm-diagnose --step freeze
```

#### 可微光束整形 (diff-beam)
```bash
python src/ao_shaping/main.py diff-beam [OPTIONS]
```
等同于: `python -m ao_shaping.runners.diff_beam_runner`

双算法可微光束整形：**backprop**（PyTorch 前向/反向传播，角谱衍射模拟 + Adam 优化相位）或 **gs**（Gerchberg-Saxton）。将 SLM 相位优化为任意目标强度图案（高斯、圆形、或自定义图片）。相同指标定义（`beam_shaping_utils.compute_metrics`）保证两算法结果可直接对比。

选项:
- `--algorithm`: 优化算法 (backprop / gs, 默认: backprop)
- `--target-image`: 目标图像路径 (灰度图 / .npy，归一化后作为目标强度)
- `--target-shape`: 预设目标形状 (gaussian / circle, 默认: gaussian)
- `-e, --epochs`: backprop 的 Adam 优化步数 (默认: 200)
- `--lr`: backprop 学习率 (默认: 0.01)
- `-i, --iterations`: gs 迭代次数 (默认: 50)
- `-d, --distance`: 传播距离 m (默认: 0.1)
- `-l, --wavelength`: 激光波长 nm (默认: 1064)
- `--cam-id`: CCD 相机 ID (默认: FAR_CAM_ID/0)
- `--cam-center`: CCD 中心 'x,y' (默认: 自动检测)
- `--cam-size`: CCD 开窗大小 像素 (默认: 400)
- `--adaptive`: gs 算法启用 CCD 反馈自适应 (需 --use-hardware)
- `--seed`: backprop 初始相位随机种子 (默认: 0，可复现)
- `-s, --save-dir`: 结果保存目录 (默认: data/diff_beam)
- `--use-hardware`: 使用实际硬件 (SLM+CCD)，否则仅模拟
- `--show`: 显示结果图像

示例:
```bash
# 纯模拟: backprop 整形为高斯目标
python src/ao_shaping/main.py diff-beam --target-shape gaussian -e 200 --show

# 自定义目标图片 + GS 算法
python src/ao_shaping/main.py diff-beam --algorithm gs --target-image target.png -i 100 --show

# 硬件闭环: CCD 反馈自适应
python src/ao_shaping/main.py diff-beam --algorithm gs --adaptive --use-hardware
```

#### 闭环波前优化 (closed-loop)
```bash
python src/ao_shaping/main.py closed-loop [OPTIONS]
```
等同于: `python -m ao_shaping.runners.zernike_matrix_runner closed-loop`

基于已保存的Zernike响应矩阵进行闭环波前优化。

选项:
- `--load-file`: 已保存的响应矩阵 .h5 文件路径 (必需)
- `--output`: 结果保存路径 (默认: 在load-file同目录生成)
- `--control-law`: 控制律 (pid, leaky, qg, lqg, mpc, adaptive) (默认: leaky)
- `--gain`: 控制增益覆盖 (控制律依赖)
- `--leak`: 泄漏因子覆盖
- `--kp`: PID比例增益
- `--ki`: PID积分增益
- `--kd`: PID微分增益
- `--dt`: 采样周期 [s] (默认: 0.067)
- `--rms-target`: 目标RMS [λ] (默认: 0.05)
- `--max-iter`: 最大迭代次数 (默认: 100)
- `--delay-steps`: 延时补偿步数 (默认: 1)
- `--cancel-tile/--no-cancel-tile`: 测量时去除WFS tip/tilt (默认: False)
- `--display/--no-display`: 显示实时pygame显示 (默认: False)
- `--debug`: 启用调试模式

示例:
```bash
DEBUG=1 python src/ao_shaping/main.py closed-loop --load-file data/zm.h5 --control-law leaky --max-iter 50
```

#### 交替电压下发 (alt-voltage)
```bash
python src/ao_shaping/main.py alt-voltage [OPTIONS]
```
等同于: `python -m ao_shaping.runners.alt_voltage_runner`

在 0V 和指定电压之间循环交替发送到 R50Power 控制器的指定单元。可选同步采集 NI DAQ ADC 信号。

选项:
- `-i, --ip`: R50Power 控制器 IP 地址 (默认: 192.168.0.101)
- `-p, --port`: 控制器端口 (默认: 8080)
- `-v, --voltage`: 高电平电压 (V, 默认: 20.0)
- `-f, --freq`: 交替频率 (Hz, 默认: 1.0)
- `-d, --duration`: 运行时长 (秒, 0=持续运行直到 Ctrl+C) (默认: 0)
- `-c, --channels`: 通道列表 (逗号分隔, 默认: 全部 50 通道)
- `--no-ping-first`: 跳过启动前 ping 检查
- `--no-relay-on`: 跳过自动上电 (relay on)
- `--adc-enabled`: 启用 NI DAQ ADC 同步采集
- `--adc-device`: NI DAQ 设备名 (默认: Dev1)
- `--adc-channel`: 模拟输入通道 (默认: ai0)
- `--adc-sample-rate`: ADC 采样率 (Hz, 默认: 5000)
- `--adc-samples-per-read`: 每次读取的样本数 (默认: 10)

ADC 采集数据自动保存到 `data/alt_voltage_adc_<timestamp>.csv`。

示例:
```bash
# 全部50个通道交替 20V, 1Hz, 持续运行
python src/ao_shaping/main.py alt-voltage --ip 192.168.0.101 --voltage 20

# 通道 0-5, 30V, 2Hz, 持续 10 秒, 同步 ADC 采集
python src/ao_shaping/main.py alt-voltage --ip 192.168.0.101 --voltage 30 --freq 2.0 --duration 10 --channels 0,1,2,3,4,5 --adc-enabled
```

#### 全量交替电压下发 (full-voltage)
```bash
python src/ao_shaping/main.py full-voltage [OPTIONS]
```
等同于: `python -m ao_shaping.runners.full_voltage_runner`

基于 **AsyncMicroDM 异步驱动**的全量交替电压工具：所有单元的电压**同时、均匀**地在 0V 和指定电压之间交替（无逐通道选择），可用于变形镜老化测试、寿命验证等场景。

高实时性设计:
- asyncio 非阻塞 TCP + `TCP_NODELAY`（禁用 Nagle，消除延迟 ACK 引入的每帧几十 ms 等待）
- 两种状态 (0V / 指定电压) 的命令字节**一次性预构建**，热循环零编码、零分配，只做 `write`+`drain`
- deadline 节拍调度，下发/打印耗时不累积相位漂移；Ctrl+C 响应 ≤50ms
- 进度输出节流 + 实时打印每帧平均下发延迟 (µs)

选项:
- `--ips`: 控制器 IP 列表 (逗号分隔, 默认: 192.168.0.101)
- `--voltage`: 高电平电压 (V, -20~120, 必需)
- `--freq`: 交替频率 (Hz, 默认: 1.0)
- `--duration`: 运行时长 (秒, 0=持续运行直到 Ctrl+C) (默认: 0)
- `--relay-on/--no-relay-on`: 自动继电器上电 (默认: True)
- `--home-voltage`: 关闭时归位电压 (V, 默认: 0.0)
- `--timeout`: 控制器连接/下发超时 (秒, 默认: 10.0)
- `--debug`: 启用调试日志

示例:
```bash
# 单个控制器全部单元交替 20V, 1Hz, 持续运行
python src/ao_shaping/main.py full-voltage --voltage 20

# 两个控制器, 30V, 2Hz, 持续 10 秒
python src/ao_shaping/main.py full-voltage --ips 192.168.0.101,192.168.0.102 --voltage 30 --freq 2.0 --duration 10

# 关闭自动上电, 5V, 0.5Hz
python src/ao_shaping/main.py full-voltage --voltage 5 --freq 0.5 --no-relay-on
```

#### DM响应矩阵标定 (dm-matrix)
```bash
python src/ao_shaping/main.py dm-matrix [OPTIONS]
```
等同于: `python -m ao_shaping.runners.dm_matrix_runner`

通过推拉电压扰动测量DM-to-WFS响应矩阵。

选项:
- `--voltage`: 扰动电压 (0=自动优化, 默认: 0.1)
- `--n-averages`: 每次WFS读取次数 M (默认: 10)
- `--n-cycles`: 正负交替循环次数 N (默认: 10)
- `--wait`: 电压施加后等待时间 (秒, 默认: 0.1)
- `--output`: 输出文件路径 (默认: data/dm_response_matrix)
- `--dm-unit-mask`: DM单元掩码 (逗号分隔的0/1列表, 默认: 全部有效)
- `--mla-index`: MLA分辨率 (512, 540, 600, 768, 1280) (默认: 512)
- `--exp-time`: WFS曝光时间 (ms, 0=自动)
- `--auto-exposure/--no-auto-exposure`: 启用WFS自动曝光 (默认: True)
- `--high-speed`: 启用高速模式
- `--use-custom-ref`: 使用自定义参考文件
- `--pupil-diameter`: 瞳孔直径 (mm, 默认: 2.0)
- `--pupil-center`: 瞳孔中心坐标 (默认: (0,0))
- `--no-inverses`: 不计算逆矩阵 (默认: False)
- `--cancel-tile`: 测量时去除WFS的tip/tilt (默认: False)
- `--auto-optimize/--no-auto-optimize`: 自动优化每路扰动电压 (voltage=0时, 默认: True)
- `--optimize-n-avg`: 电压优化时的WFS读取次数 (默认: 10)
- `--display/--no-display`: 显示实时pygame显示 (暂未实现)
- `--debug`: 启用调试模式 (保存原始测量数据)

示例:
```bash
DEBUG=1 python src/ao_shaping/main.py dm-matrix --voltage 0.2 --n-averages 5 --output data/dm_response.h5
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

5. 全息图生成:
```bash
python -m ao_shaping.runners.gs_hologram_runner [OPTIONS]
```

6. GS闭环光束整形:
```bash
python -m ao_shaping.runners.gs_square_runner [OPTIONS]
```

7. 可微分闭环光束整形 (PyTorch):
```bash
python -m ao_shaping.runners.diff_shaping_runner [OPTIONS]
```

8. 交替电压下发:
```bash
python -m ao_shaping.runners.alt_voltage_runner [OPTIONS]
```

9. 全量交替电压下发 (AsyncMicroDM):
```bash
python -m ao_shaping.runners.full_voltage_runner [OPTIONS]
```

10. DM响应矩阵标定:
```bash
python -m ao_shaping.runners.dm_matrix_runner [OPTIONS]
```

10. Micro-DM 逐单元图像采集:
```bash
python -m ao_shaping.tools.micro_dm_image_collect [OPTIONS]
```

遍历一个或多个 R50Power 控制器, 对每个通道 (单元) 依次下发电压并用 MiiCam 相机采集图像, 每通道采集后归位。未指定 `--ip` 时遍历所有控制器 (wiring map 或默认 192.168.0.101-126), 某个控制器连接失败则跳过并继续下一个。

常用选项:
- `--ip`: 控制器 IP, 可多次指定; 不指定则遍历所有控制器
- `-v, --voltage`: 下发电压 V (必需, -20~120)
- `--home-voltage`: 归位电压 V (默认: 0.0)
- `--channels`: 通道列表 逗号分隔 或 'all' 全部 50 通道 (默认: all)
- `-o, --output`: 输出目录 (默认: data/micro_dm_images)
- `--cam-id`: MiiCam 相机 ID (默认: config 中 far_cam_id)
- `--exposure-ms`: MiiCam 曝光时间 ms (默认: 20)
- `--bit-depth`: 输出位深 8 或 16 (默认: 8)
- `--n-sample`: 每帧平均采样数 (默认: 1)
- `--n-frames`: 每通道采集图像张数 (默认: 1)
- `--settle-time`: 电压下发后等待时间 s (默认: 0.5)
- `--ping-first/--no-ping-first`: 连接前先 ping 测试 (默认: 开启)
- `--relay-on/--no-relay-on`: 连接后自动继电器上电 (默认: 开启)
- `--save-npy`: 额外保存 .npy 原始数组

示例:
```bash
# 单控制器, 全部 50 通道 20V
python -m ao_shaping.tools.micro_dm_image_collect --ip 192.168.0.101 --voltage 20 -o data/micro_dm_images

# 不指定 IP → 遍历所有控制器
python -m ao_shaping.tools.micro_dm_image_collect --voltage 20

# 每通道采集 3 张
python -m ao_shaping.tools.micro_dm_image_collect --ip 192.168.0.101 --voltage 20 --n-frames 3
```

### Micro-DM 数据目录

采集的图像数据存储在 `data/md_test/` 目录下，结构如下：

```
data/md_test/
├── md_img/                          # 原始灰度图像
│   └── 192.168.0.{101~126}/         # 按 IP 分组
│       └── 192.168.0.{ip}-{seq:03d}.png
├── md_img-100v_processed/diff/      # 100V 差分图像 (FFT 去条纹)
│   └── 192.168.0.{ip}/{ip}-{seq:03d}_cx{X}_cy{Y}.png
└── md_img-100v_gif/                 # GIF 动画 (逐通道)
```

**映射关系**: 网格坐标 (row, col) → CSV/Excel → IP 组 + 序号 → 图像文件

详细说明见: `data/md_test/README.md`

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

项目提供了基于Streamlit的图形界面，按设备域分包于 `src/ao_shaping/gui/` 下，各 UI 独立运行：

```bash
# R50 控制器 (单单元/单控制器/组/联合控制)
streamlit run src/ao_shaping/gui/r50/r50_controller_ui.py

# Micro-DM 驱动控制
streamlit run src/ao_shaping/gui/dm/micro_dm_ui.py

# Zernike 响应矩阵校准
streamlit run src/ao_shaping/gui/zernike/zernike_response_matrix_ui.py

# SLM 校准
streamlit run src/ao_shaping/gui/slm/slm_calibration_ui.py

# 多 SLM 控制器 (相位图案生成/下发, 含 GS方形整形)
streamlit run src/ao_shaping/gui/slm/multi_slm_controller.py

# 1300 陶瓷单元查看器 (网格浏览 + 图片标注)
streamlit run src/ao_shaping/gui/r50/ceramic_viewer.py
```

`multi_slm_controller.py` 提供多种全息相位图案生成：平场、闪耀光栅、达曼光栅、涡旋相位、**GS方形整形**等。其中 GS方形整形模式：上传远场光斑图片 → 自动测量光斑直径并计算方形边长（边长 = 光斑直径 × 尺寸因子，自动换算相机/SLM 像素间距）→ 在 SLM 分辨率网格上运行 Gerchberg-Saxton → 下发 uint16 相位到 SLM。支持实时迭代进度显示与逐轮相位下发（内存槽自动轮换）。方形尺寸/相位正确性由仿真测试验证（`tests/ao_shaping/gui/slm/test_gs_square_shaping.py`，角谱传播断言）。

## 硬件支持

### 波前传感器
- **Thorlabs WFS系列**: 支持自动图像采集和倾斜去除

### 变形镜
- **统一 DM 接口**: 所有变形镜继承自 `ao_shaping.drivers.dm.base.DM`，提供 `transform`/`send`/`open`/`close`/`is_connected`/`get_actuator_positions` 等标准方法
- **NLight系列**: 支持电压控制和电压差安全检查
- **R50Power MicroDM (同步)**: 通过 TCP 控制多路 R50Power 控制器（每路 50 通道，-20V~120V）
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

- **R50Power AsyncMicroDM (异步)**: 基于 asyncio 的高性能异步 TCP 驱动，专为 AO 快速闭环优化
  - LUT-based 预查表电压转换，零 GC 稳态运行（`VoltageConverter`）
  - `asyncio.StreamReader/StreamWriter` 非阻塞 TCP 通信
  - **TCP_NODELAY**：连接时禁用 Nagle，消除延迟 ACK 对小火花的缓冲延迟
  - 预分配命令缓冲区，避免帧间内存分配
  - **内联命令字节快路径**：`prebuild_command()` / `build_frame_commands()` 将电压帧一次性编码为命令字节，热循环用 `send_bytes()` / `send_frame_commands()` 直接重放——零编码、零分配、仅 `write`+`drain`（见 `full-voltage` 工具）
  - 支持同步/异步双模式使用（`open()`/`close()` 同步桥接）
  - 并行控制器通信，独立超时控制

  ```python
  from ao_shaping.drivers.dm.asyn_micro_dm import AsyncMicroDM

  # 同步用法（内部桥接到异步）
  dm = AsyncMicroDM(ips=["192.168.0.101", "192.168.0.102"])
  dm.open()
  dm.send_voltages(np.zeros(dm.DM_Num))
  dm.close()

  # 异步用法（原生 asyncio）
  dm = AsyncMicroDM(ips=["192.168.0.101"])
  await dm.connect_all()
  await dm.send_frame(np.zeros(dm.DM_Num))
  await dm.shutdown()

  # 内联快路径：预构建状态字节，热循环零开销重放
  cmd_off = dm.build_frame_commands(np.zeros(dm.DM_Num))
  cmd_on = dm.build_frame_commands(np.full(dm.DM_Num, 20.0))
  await dm.send_frame_commands(cmd_off)  # 仅 write + drain
  await dm.send_frame_commands(cmd_on)

  # 工厂创建
  from ao_shaping.drivers.dm._registry import create_dm
  dm = create_dm("asyn_micro", ips=["192.168.0.101"])
  ```

- **ZernikeDM**: Zernike 系数驱动的 DM/SLM 接口，支持 Zernike 多项式相位生成
- **HadamardDM**: Hadamard 系数驱动的 DM/SLM 接口，支持 Walsh-Hadamard 模式相位生成
- **PIB 优化器多 DM 支持**: `optimize_pib` 现接受任意 `DM` 子类实例，命令行支持 `--dm_type` 参数
  - 支持类型: `nlight`, `micro`, `asyn_micro`, `zernike`, `hadamard`
  - 自动检测: 未指定 `--dm_type` 时自动探测在线 DM，仅一个时自动选取，多个时报错提示

### 相机
- **大恒相机系列**: DahengCamManager，支持14位和16位模式
- **MIICAM系列**: MIICamDriver，支持高速采集

### 空间光调制器
- **Santec SLM200**: 支持相位图案生成、缓存和CSV加载
  - `open()` 方法已重构为子方法 (`_apply_config_params`, `_load_correction`, `_setup_wavelength`)，逻辑更清晰
  - 波前误差矫正通过独立 `WavefrontCorrection` 类管理（CSV加载→异常点检测→矫正映射图）
  - 矫正数据自动按优先级加载: `__init__` 显式指定 > 配置文件 > 默认路径

> **⚠️ SLM 平场灰度生成注意事项**
>
> Santec SLM200在1064nm附近存在**振幅耦合**效应——SLM加载不同灰度值的平场相位时，相机采集到的光斑亮度会随灰度值变化（周期 ≈ 2π，即约993灰度值）。这是SLM的固有特性，已在实验中验证（参见 `scripts/validate_flat_phase_gray.py`）。
>
> **关键规则1（灰度值路径）**: 平场相位（以及其他直接灰度图案）**必须**使用 `np.full((height, width), gray, dtype=np.uint16)` 生成，**不能**通过 `create_phase_from_array()` 传递。因为 `create_phase_from_array()` 将输入作为**弧度**处理（mod 2π → 弧度/2π × 1023），uint16灰度值会经过不必要的弧度转换而被静默损坏。
>
> **关键规则2（内存模式槽轮换）**: Santec SLM 在内存模式下，**前后两次写入不能使用同一个内存槽**（memory slot）。当 `display_memory(slot)` 被调用时，如果该槽已经在显示，设备会将此调用视为空操作（no-op），LCOS 面板不会刷新，屏幕上仍显示上一次的相位图案。连续写入时必须使用不同的槽位——diff-shaping runner 在 **2~125 槽范围内随机选取**并排除当前显示槽（启动时 `get_displayed_memory_number()` 续接，跨进程也不冲突）。`display_data()` 内置的 127 槽循环机制同样满足这一约束。
>
> 验证命令:
> ```bash
> python scripts/validate_flat_phase_gray.py --exposure-ms 0.8 --wait-time-s 0.3
> ```

### 数据采集卡
- **NI DAQ (nidaqmx)**: 用于多设备同步控制和模拟电压采集
- **ADC Driver** (`ao_shaping.drivers.adc.NidaqADC`): NI DAQ模拟输入电压采集驱动
  - 基于 nidaqmx 的 `HW_TIMED_SINGLE_POINT` 采样模式
  - 可配置设备名 (Dev1)、通道 (ai0)、采样率和每批采样数
  - 提供 `read(samples)` 返回原始电压数组和 `read_mean()` 返回均值
  - 无硬件时可使用 `MockADC` 进行开发和测试

  ```python
  from ao_shaping.drivers.adc import NidaqADC

  with NidaqADC(device_name="Dev1", channel="ai0", sample_rate=5000, samples_per_channel=10) as adc:
      voltages = adc.read()        # shape (10,) array of voltages
      mean_v = adc.read_mean()     # float
  ```

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

### v0.9.0 (2026-08-27)
- **全量交替电压工具** (`full-voltage`): 新增 `full_voltage_runner.py`，基于 AsyncMicroDM 异步驱动，所有单元电压同时、均匀地在 0V 与指定电压间交替（无逐通道选择），用于老化/寿命测试
- **AsyncMicroDM 延迟优化**:
  - `TCP_NODELAY`: 连接时禁用 Nagle，消除延迟 ACK 引入的每帧几十 ms 缓冲
  - **内联命令字节快路径**: `prebuild_command()` / `build_frame_commands()` 预编码电压帧为命令字节，`send_bytes()` / `send_frame_commands()` 在热循环中零编码、零分配重放（仅 `write`+`drain`）
  - `send_voltages()` 重构为复用预构建+发送快路径（字节级兼容）
- **full-voltage runner 高实时性设计**: deadline 节拍调度（无相位漂移）、进度输出节流 + 平均下发延迟统计、分片睡眠保证 Ctrl+C 响应 ≤50ms

### v0.7.0 (2026-08-27)
- **R50 GUI 全面重构**: 单控制器 / 分组控制 / 全部控制 (联合) 三大 Tab 模块化拆分
  - 单控制器 Tab: 支持单次发送、持续保持、正弦/交替/逐序波形下发
  - 联合控制 Tab: 36×36 矩阵全量编辑、单单元格/行列/矩形批量填充、一键归零下发
  - 分组控制 Tab: 按 wiring map 组别选择控制器并批量下发
  - 侧边栏调试面板集成: 仿真状态指示、指令日志、操作日志（含各控制器 IP 详情）
- **单控制器增强**: 支持在「联合控制」连接模式下选择单个控制器 IP 进行独立操作
- **控制器序号自动映射**: 输入 1-26 序号自动设置 IP (192.168.0.101~126) 和端口 (10101~10126)
- **波形下发优化**: 持续保持 / 正弦 / 交替 / 逐序模式统一按钮状态管理，仿真/正式模式均可正常下发
- **矩阵可视化统一**: 36×36 表格全部加粗逻辑统一，下发后（含归零）所有单元格均为粗体
- **Styler 兼容性修复**: 修复 pandas 2.1+ `Styler.applymap` 移除导致的 AttributeError

### v0.8.0 (2026-08-27)
- **1300 陶瓷单元查看器** (`ceramic_viewer.py`):
  - 36×36 网格浏览，点击单元格查看详细信息
  - 原始图像 + 差分图像双列显示
  - **Circle 标注模式**: 在原始图像上绘制圆形标注缺陷区域
  - **Transform 模式**: 移动/调整已绘制圆的位置和大小
  - 标注坐标自动映射回原图尺寸 (支持缩放显示)
  - CSV 导出标注数据
- **Micro-DM 数据文档**: 新增 `data/md_test/README.md` 和 `docs/micro deformable mirror/docs/README.md`，详细说明 1300 单元映射关系
- **streamlit-drawable-canvas 集成**: 安装并修补兼容 Streamlit 1.56+ 的绘图组件

### v0.6.0 (2026-08-25)
- **AsyncMicroDM 异步驱动**: 新增 `asyn_micro_dm.py`，基于 asyncio 的高性能异步 TCP 驱动
  - LUT-based 预查表电压转换（`VoltageConverter`），零 GC 稳态运行
  - `AsyncR50Controller` 使用 `asyncio.StreamReader/StreamWriter` 非阻塞 TCP
  - 预分配命令缓冲区，避免帧间内存分配
  - 支持同步/异步双模式使用（`open()`/`close()` 同步桥接到异步内部）
  - 并行控制器通信，独立超时控制
  - 复用 `WiringMap` 接线映射系统
  - 注册为 `"asyn_micro"` 类型，支持 `create_dm()` 工厂创建
- **Micro-DM 逐通道响应分析脚本**: 新增 `scripts/md_img_diff_centroid.py`（FFT 去条纹 → signed diff → 阈值去噪 → 主暗斑质心 → jet 伪彩色渲染）、`scripts/md_img_diff_overlay.py`（逐像素最大值合并分析，输出覆盖率统计）、`scripts/md_img_diff_to_gif.py`（按控制器 IP 将 50 通道 diff 图合成为动画 GIF，帧标注通道号与质心坐标）
- **文档完善**: `scripts/README.md` 新增 Micro-DM Diff Analysis Pipeline 章节，详细阐述 diff 计算与合并分析的算法、处理流程和阈值选取方法（经验阈值 15、主暗斑质心替代全图质心的原因、jet 色标 vmax 归一化）

### v0.5.0 (2026-07)
- **SLM波前误差矫正重构**: 引入独立 `WavefrontCorrection` 类，封装CSV加载、异常点检测（Z-score）、中值滤波剔除和矫正映射图计算
- **SLM open() 方法重构**: 拆分为 `_apply_config_params`, `_load_correction`, `_setup_wavelength` 三个子方法，提升可维护性
- **矫正数据加载优先级**: `__init__` 显式指定 > 配置文件 > 默认路径，支持自定义 `calc_fn` 工厂方法

### v0.4.0 (2026-05)
- **根目录访问重构**: 引入 `ROOT_DIR` 常量，统一项目根目录访问，替代多处 `Path(__file__).resolve().parents[N]` 写法
- **WFS配置管理器简化**: 使用 `PROJECT_ROOT` 简化 WFS 配置管理器初始化
- **ZernikeSLM增强**: 添加 `length` 属性，返回 Zernike 多项式数量
- **文档完善**: 更新 Zernike 响应矩阵标定与闭环控制的文档，新增响应矩阵测试用例
- **编码规范更新**: 在 README 中添加详细的编码指南和最佳实践

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