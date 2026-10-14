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

```
AO-shaping/
├── src/
│   ├── ao_shaping/               # 主程序包
│   │   ├── __init__.py
│   │   ├── main.py               # CLI入口点 (Click-based), 注册 20 个命令
│   │   ├── config.py             # 中央配置 (DM_N_ACTUATORS, DEFAULTS, PATHS, ao_config)
│   │   ├── runners/              # 运行器包 (硬件编排层: Click CLI, 设备生命周期)
│   │   │   ├── __init__.py       # Lazy imports + re-exports (via __getattr__)
│   │   │   ├── runner_common.py  # 共享参数 dataclass + with_params click 集成
│   │   │   ├── closed_loop.py    # [向后兼容 shim] AOClosedLoop re-export (canonical: optimizer/wf/)
│   │   │   ├── matrix_runner.py        # DM响应矩阵 (dm-matrix) + Hadamard响应矩阵 (hadamard-matrix)
│   │   │   ├── zernike_search_runner.py  # GA Zernike (ga-zernike) + 贪婪局部搜索 (greedy-zernike)
│   │   │   ├── gsnet_cache.py             # GSNet缓存
│   │   │   ├── gsnet_dataset.py           # GSNet数据集
│   │   │   ├── gsnet_offline.py           # GSNet离线评估
│   │   │   ├── gsnet_train.py             # GSNet训练
│   │   │   ├── nlight_dm/                 # NLight DM 相关 runner
│   │   │   │   ├── wf_runner.py           # Wavefront RMS优化 (wf)
│   │   │   │   ├── axis_beam_runner.py    # PIB优化 (pib)
│   │   │   │   ├── pipeline_runner.py     # 串行WF→PIB (pipeline)
│   │   │   │   └── combined_runner.py     # AdaMOD+SPGD (combined)
│   │   │   ├── micro_drive/               # 微驱 (R50Power) 相关 runner
│   │   │   │   ├── voltage_runner.py      # 交替电压 (alt-voltage) + 全量交替电压 (full-voltage)
│   │   │   └── slm/                       # SLM 相关 runner
│   │   │       ├── rms_zernike_runner.py     # SLM Zernike RMS (rms-zernike)
│   │   │       ├── zernike_matrix_runner.py  # Zernike响应矩阵 (zernike-matrix) + closed-loop
│   │   │       ├── slm_shaping_runner.py          # SLM Zernike PIB (slm-pib) + 方形整形 (spgd-square)
│   │   │       ├── gsnet_runner.py           # SLM自由相位方形整形 (slm-gsnet)
│   │   │       ├── gs_refine_runner.py       # GS 预矫正 + SPGD 细化 (slm-gs-refine)
│   │   │       ├── model_in_loop_runner.py   # 正向模型闭环校正 (slm-model-in-loop)
│   │   │       └── offset_runner.py          # SLM偏移校准 (standalone, 未注册)
│   │   ├── algorithm/           # 优化算法 (5 子包: gradient/heuristic/signal_processing/tabu/goal_functions)
│   │   │   ├── gradient/        # 梯度优化 (Adam, SGD, Muno, MuonW, etc.)
│   │   │   ├── heuristic/       # 启发式搜索 (GA, PSO, SA, CEM, DE, HC, RS)
│   │   │   ├── signal_processing/  # 信号处理 (GS, 相位包裹, 可微分光束整形, 控制律)
│   │   │   ├── tabu/            # Tabu 搜索
│   │   │   ├── goal_functions/  # 目标函数 (target_func, image_metrics)
│   │   │   ├── base.py          # Base 梯度优化器基类
│   │   │   └── README.md        # 优化器 class-based 约定
│   │   ├── model/                 # 物理量数据类与物理坐标重采样
│   │   │   ├── __init__.py        # 包级 re-exports
│   │   │   ├── field.py           # PhaseMap, AmplitudeMap, ComplexField + OOPAO 转换
│   │   │   ├── quantities.py      # OPDMap, DmCommands, PSFImage 等物理量
│   │   │   ├── spatial.py         # 以物理坐标为基准的插值
│   │   │   └── metadata.py        # FieldMetadata (timestamp, source, wavelength)
│   │   ├── drivers/              # 硬件驱动 (see drivers/AGENTS.md)
│   │   │   ├── device_base.py    # Device 基类 + _load_sdk()/_ensure_sdk() 惰性契约
│   │   │   ├── device_registry.py
│   │   │   ├── _lazy.py          # PEP 562 install_lazy_attrs (包级惰性)
│   │   │   ├── mock_devices.py
│   │   │   ├── adc/             # NI DAQ ADC 电压采集
│   │   │   ├── ccd/             # 相机 (Daheng, MiiCam, ffmpeg)
│   │   │   ├── dm/              # 变形镜 (NLight, MicroDM, AsyncMicroDM, HadamardDM, ZernikeDM, _registry)
│   │   │   ├── slm/             # 空间光调制器 (Santec, ZernikeSLM)
│   │   │   ├── wfs/             # 波前传感器 (Thorlabs; base/_registry + ThorlabWFS)
│   │   │   ├── tm/              # 定时模块 (Serial/FSM)
│   │   │   └── sim/             # 数字孪生仿真 (atmos/ccd/dm/laser/optics/wfs + dm_optics/disturbance)
│   │   ├── optimizer/           # 高层优化器 (策略实现层)
│   │   │   ├── constants.py, spgd.py, combined_optimizer.py
│   │   │   ├── wf/              # 波前优化 (RMS, Zernike, GA, 响应矩阵, closed_loop)
│   │   │   ├── wfless/          # 无波前优化 (PIB, SPGD, 可微分整形)
│   │   │   └── rl/              # 强化学习 (SAC, LR-WFS)
│   │   ├── utils/               # 工具函数 (io/image/wavefront/slm 子包 + root-level shims)
│   │   │   └── image/
│   │   │       └── target/      # 目标形状: ccd, metrics, objective, patterns, square
│   │   ├── tools/               # 硬件交互工具 (tools/slm/, tools/micro_dm/, slm_train_data_collect.py)
│   │   ├── display/             # 可视化 (窗口, GUI帧, pygame)
│   │   └── gui/                 # GUI组件 (Streamlit), 按设备域分包: r50/dm/slm/zernike/ccd
│   ├── ml/                      # 机器学习独立包
│   │   ├── gsnet/               # GSNet: GS + SPGD 整形训练
│   │   ├── gsnet_debug/         # GSNet 调试/离线评估
│   │   ├── phase/               # U-Net+GAN 相位预测 (dataset, discriminator, trainer, unet)
│   │   ├── hwdataset/           # 硬件相位→相机图像 Dataset (index, records, transforms, cache, dataset, zernike_dataset)
│   │   ├── zernike/             # Zernike forward/inverse model (models, forward_model, train_coeff, train_amp, losses, metrics, augment, dataset, inverse_design, eval_stats)
│   │   ├── train.py             # ML 训练入口
│   │   └── wandb_logger.py      # W&B 集成
│   ├── calculators/               # Cython扩展 (standalone)
│   └── optical_ui/                # [DEPRECATED] Empty package
├── tests/ao_shaping/              # Tests (镜像 src 结构)
├── scripts/                      # 实用脚本 (含报告生成 generate_*_report.py)
├── docs/                         # 设备说明文档 (设备规格/SDK用法/操作手册/图集)
├── report/                       # 实验报告 (结论/测量/基准), 每份带生成脚本溯源块
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
# 仅安装基础依赖 (注意: uv sync 是精确同步, 会移除已安装的 ml/rl 等依赖组)
uv sync

# 安装 ML 相关包 (torch, torchvision, wandb)
# 注意: ml/rl 定义在 [dependency-groups] 中, 必须用 --group, 不能用 --extra
uv sync --group ml

# 安装 RL 相关包 (gymnasium, stable-baselines3)
uv sync --group rl

# 安装全部依赖组 (dev/ml/rl/disp/data) —— 推荐用于完整开发环境
uv sync --all-groups
```

OOPAO 从 `libs/OOPAO` 以 editable 依赖安装。首次使用前确认已获取子模块，
然后运行 `uv sync`；当前 `pyproject.toml` 已声明该依赖，无需再次执行 `uv add`：

```bash
git submodule update --init libs/OOPAO
uv sync
```

## 使用说明

> **CLI 命令文档已按设备域拆分到 `docs/` 下**，本节只保留总览与入口说明。详细用法、选项与示例见各设备域文档：

| 命令 | 文档 | 功能 |
|------|------|------|
| `wf` | [docs/dm/wf.md](docs/dm/wf.md) | 波前RMS优化 (DM电压 + WFS) |
| `pib` | [docs/dm/pib.md](docs/dm/pib.md) | PIB优化 (DM电压 + CCD) |
| `pipeline` | [docs/dm/pipeline.md](docs/dm/pipeline.md) | 串行 WF RMS → PIB 流水线 |
| `zernike-matrix` | [docs/slm/zernike_matrix.md](docs/slm/zernike_matrix.md) | Zernike响应矩阵校准 |
| `closed-loop` | [docs/wfs/closed_loop.md](docs/wfs/closed_loop.md) | 基于响应矩阵的闭环波前优化 |
| `rms-zernike` | [docs/slm/rms_zernike.md](docs/slm/rms_zernike.md) | SLM Zernike RMS优化 |
| `ga-zernike` | [docs/wfs/ga_zernike.md](docs/wfs/ga_zernike.md) | 遗传算法 Zernike优化 |
| `greedy-zernike` | [docs/wfs/greedy_zernike.md](docs/wfs/greedy_zernike.md) | 贪婪局部搜索 Zernike优化 |
| `dm-matrix` | [docs/dm/dm_matrix.md](docs/dm/dm_matrix.md) | DM响应矩阵标定 |
| `hadamard-matrix` | [docs/dm/hadamard_matrix.md](docs/dm/hadamard_matrix.md) | Hadamard响应矩阵标定 |
| `alt-voltage` | [docs/micro_drive/alt_voltage.md](docs/micro_drive/alt_voltage.md) | 交替电压下发 (R50Power + ADC) |
| `full-voltage` | [docs/micro_drive/full_voltage.md](docs/micro_drive/full_voltage.md) | 全量交替电压 (AsyncMicroDM) |
| `combined` | [docs/dm/combined.md](docs/dm/combined.md) | AdaMOD+SPGD 混合PIB (DM+CCD) |
| `slm-lut` | [docs/slm/slm_lut.md](docs/slm/slm_lut.md) | SLM灰度→相位LUT校准 |
| `slm-diagnose` | [docs/slm/slm_diagnose.md](docs/slm/slm_diagnose.md) | SLM硬件自检 |
| `spgd-square` | [docs/slm/spgd_square.md](docs/slm/spgd_square.md) | SLM方形光斑SPGD整形 |
| `slm-gsnet` | [docs/slm/slm_gsnet.md](docs/slm/slm_gsnet.md) | SLM自由相位方形整形 (SPGD/启发式) |
| `slm-pib` | [docs/slm/slm_pib.md](docs/slm/slm_pib.md) | SLM Zernike PIB优化 (`spgd` / `heuristic`) |
| `slm-gs-refine` | [docs/slm/slm_gs_refine.md](docs/slm/slm_gs_refine.md) | GS 预矫正 + 自由相位 SPGD 整形 |
| `slm-model-in-loop` | [docs/slm/slm_model_in_loop.md](docs/slm/slm_model_in_loop.md) | 正向模型闭环校正 + 目标光斑相位合成 (反复迭代) |

项目提供了统一的命令行界面，通过`main.py`作为入口点：

```bash
python src/ao_shaping/main.py [OPTIONS] COMMAND [ARGS]...
```

所有运行器位于 `src/ao_shaping/runners/` 包及其子包中，通过 main CLI 统一调用。目前共注册 **20 个命令**。

> **注意**: `gs`、`gs-square`、`diff-shaping`、`diff-beam` 命令已从 CLI 中移除 (运行器文件不再存在)。其功能已并入 `slm-gsnet` (自由相位整形)、`algorithm/signal_processing/gerchberg_saxton.py` (GS算法) 和 `algorithm/signal_processing/differentiable_shaping.py` (可微分整形)。

#### 全局选项
- `--dir` / `-d`: 指定数据保存根目录 (默认: data)
- `--debug`: 启用调试模式 (保存 pkl/json/图片等调试产物; 等价于 `DEBUG=1` 环境变量)

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

> 完整选项与示例见 [docs/dm/wf.md](docs/dm/wf.md)

```bash
python src/ao_shaping/main.py wf [OPTIONS]
```
等同于: `python -m ao_shaping.runners.nlight_dm.wf_runner`

#### 轴向光束优化器 (pib)

> 完整选项与示例见 [docs/dm/pib.md](docs/dm/pib.md)
```bash
python src/ao_shaping/main.py pib [OPTIONS]
```
等同于: `python -m ao_shaping.runners.nlight_dm.axis_beam_runner`

#### 串行流水线优化器 (pipeline)

> 完整选项与示例见 [docs/dm/pipeline.md](docs/dm/pipeline.md)
```bash
python src/ao_shaping/main.py pipeline [OPTIONS]
```
等同于: `python -m ao_shaping.runners.nlight_dm.pipeline_runner`

#### Zernike响应矩阵校准 (zernike-matrix)

> 完整选项与示例见 [docs/slm/zernike_matrix.md](docs/slm/zernike_matrix.md)
```bash
python src/ao_shaping/main.py zernike-matrix [OPTIONS]
```
等同于: `python -m ao_shaping.runners.slm.zernike_matrix_runner`

#### SLM Zernike RMS 优化 (rms-zernike)

> 完整选项与示例见 [docs/slm/rms_zernike.md](docs/slm/rms_zernike.md)
```bash
python src/ao_shaping/main.py rms-zernike [OPTIONS]
```
等同于: `python -m ao_shaping.runners.slm.rms_zernike_runner`

通过 SLM 加载 Zernike 相位, 以 WFS 测量的 RMS 为目标进行优化 (梯度法 SPGD), 实现波前校正。支持 delta 自动检测 (数量级扫描) 与多起点优化。

#### 遗传算法 Zernike 优化 (ga-zernike)

> 完整选项与示例见 [docs/wfs/ga_zernike.md](docs/wfs/ga_zernike.md)
```bash
python src/ao_shaping/main.py ga-zernike [OPTIONS]
```
等同于: `python -m ao_shaping.runners.zernike_search_runner`

基于遗传算法 (GA) 搜索最优 Zernike 系数组合, 以 WFS 测量 RMS 为适应度。适合无梯度/多峰搜索场景。

#### Zernike波前优化器 - 贪婪局部搜索 (greedy-zernike)

> 完整选项与示例见 [docs/wfs/greedy_zernike.md](docs/wfs/greedy_zernike.md)
```bash
python src/ao_shaping/main.py greedy-zernike [OPTIONS]
```
使用贪婪局部搜索算法进行Zernike波前校正。

#### SLM 灰度→相位 LUT 校准 (slm-lut)

> 完整选项与示例见 [docs/slm/slm_lut.md](docs/slm/slm_lut.md)
```bash
python src/ao_shaping/main.py slm-lut [OPTIONS]
```
等同于: `python -m ao_shaping.tools.slm.calibration.lut_runner`

#### SLM 硬件自检 (slm-diagnose)

> 完整选项与示例见 [docs/slm/slm_diagnose.md](docs/slm/slm_diagnose.md)

```bash
python src/ao_shaping/main.py slm-diagnose [OPTIONS]
```
等同于: `python -m ao_shaping.tools.slm.panel.diagnose`

#### SLM 台架探针 (独立运行, 不注册为 CLI)

以下探针**不注册为 CLI 命令**, 用 `python -m ao_shaping.tools.<子包>.<名字>` 直接运行,
且都支持 `--no-hw` (只打印采集计划、不碰硬件)。它们固化了 2026-09-30 在大恒 +
Santec SLM-200 台架上踩出来的台架常数与测量陷阱, **不要用第一性原理重新推导几何**。

| 探针 (`python -m` 路径) | 文档 | 用途 |
|---|---|---|
| `ao_shaping.tools.slm.panel.tilt_probe` | [docs/tools/slm_tilt_probe.md](docs/tools/slm_tilt_probe.md) | **判定面板是否真的在调制** (倾斜斜坡)。比光栅可靠: 光斑位移只取决于斜坡周期, 与衍射效率无关 |
| `ao_shaping.tools.slm.panel.panel_locate` | [docs/tools/slm_panel_locate.md](docs/tools/slm_panel_locate.md) | 面板坐标上定位光斑中心。相机 0 阶**不是**面板坐标 (两轴互换 90°, 尺度差 >10 倍) |
| `ao_shaping.tools.slm.panel.beam_extent` | [docs/tools/slm_beam_extent.md](docs/tools/slm_beam_extent.md) | 半平面随机相位边界扫描测光斑中心/半径。实测 r=450 px @ (960,600) |
| `ao_shaping.tools.slm.panel.phase_resolution` | [docs/tools/slm_phase_resolution.md](docs/tools/slm_phase_resolution.md) | 比较逐像素随机相位与光滑 Zernike 相位, 判定面板**等效相位分辨率** |
| `ao_shaping.tools.ccd.exposure_check` | [docs/tools/slm_exposure_check.md](docs/tools/slm_exposure_check.md) | 相机自动曝光状态 + 固定设置下漂移 (区分"相机漂移"与"SLM 保留上次图案") |
| `ao_shaping.tools.zernike.sweep_probe` | [docs/tools/slm_zernike_sweep_probe.md](docs/tools/slm_zernike_sweep_probe.md) | **光滑 Zernike 扫描探针**: ramp + tilt + defocus + astig + coma + spherical 共 42 点, 逐点稳定判据读帧, 落盘 npz + Recorder (含相位与 CCD 帧) |
| `ao_shaping.tools.ccd.drift_probe` | [docs/tools/slm_drift_probe.md](docs/tools/slm_drift_probe.md) | **平场漂移 + 曝光阶梯线性**。用区域范数判漂移 (**不用峰值** — 实测同设置两次运行峰值读到 100 与 23, 而 box sum 稳到 0.2%) |
| `ao_shaping.tools.ccd.floor_probe` | [docs/tools/slm_floor_probe.md](docs/tools/slm_floor_probe.md) | **测量本底 + 稳定时间 + SNR-vs-K**。回答噪声是读噪声 (`noise_limited`) 还是漂移 (`drift_limited`) |
| `ao_shaping.tools.zernike.abba_probe` | [docs/tools/slm_abba_probe.md](docs/tools/slm_abba_probe.md) | **稠密随机相位是否可分辨**。ABBA (`+ - - +`) 消一阶漂移并先量本底。`verdict=unusable` 时不要去测转移矩阵 |

> 🔑 **启动 GS / GSNet / SPGD runner 之前先跑表征探针**:
> `ccd.drift_probe` → `ccd.floor_probe` → `zernike.abba_probe`。
> 完整流程、验收阈值、**危险默认值清单**、**曝光 `0` 的真实语义 (§4.1)** 与
> **四条台架铁律** (暗帧不可裸 `argmax` / SLM 保留上次图案 / 固定 `memory_number`
> 是固件 no-op / 液晶要等"稳定"而非"够久") 见
> [docs/slm/pre_run_characterization.md](docs/slm/pre_run_characterization.md)。

共用内核 (无 CLI, 纯 numpy, 设备由参数传入): `tools/kernels/bench_kernels.py`
(帧数学 / 光斑测量 / 面板相位放置 / `display_and_average` / `capture_settled`) 与
`tools/kernels/sweep_analysis.py` (扫描线性 / `delta` 探索 / SNR)。Zernike 共用量在
`tools/zernike/common.py`。`scripts/model_in_loop_hw_runbook.py --stage sweep` 的采集内核
收敛自 `tools/zernike/sweep_probe.py`; 面板与几何常数来源见
[`report/slm/model_in_loop_bench_calibration.md`](report/slm/model_in_loop_bench_calibration.md)。

#### SLM 方形光斑 SPGD 整形 (spgd-square)

> 完整选项与示例见 [docs/slm/spgd_square.md](docs/slm/spgd_square.md)

```bash
python src/ao_shaping/main.py spgd-square [OPTIONS]
```
等同于: `python -m ao_shaping.runners.slm.slm_shaping_runner square` (也可写成 `main.py slm-pib square` —— 三个入口等价)

#### SLM 自由相位方形光斑整形 (slm-gsnet)

> 完整选项与示例见 [docs/slm/slm_gsnet.md](docs/slm/slm_gsnet.md)

```bash
python src/ao_shaping/main.py slm-gsnet [COMMAND] [OPTIONS]
```
等同于: `python -m ao_shaping.runners.slm.gsnet_runner`

#### GS 预整形 + 自由相位 SPGD 细化 (slm-gs-refine)

> 完整选项与示例见 [docs/slm/slm_gs_refine.md](docs/slm/slm_gs_refine.md)

```bash
python src/ao_shaping/main.py slm-gs-refine [OPTIONS]
```
等同于: `python -m ao_shaping.runners.slm.gs_refine_runner`

#### 正向模型闭环整形 (slm-model-in-loop)

> 完整选项与示例见 [docs/slm/slm_model_in_loop.md](docs/slm/slm_model_in_loop.md)

```bash
python src/ao_shaping/main.py slm-model-in-loop [OPTIONS]
```
等同于: `python -m ao_shaping.runners.slm.model_in_loop_runner`

#### Hadamard响应矩阵标定 (hadamard-matrix)

> 完整选项与示例见 [docs/dm/hadamard_matrix.md](docs/dm/hadamard_matrix.md)

```bash
python src/ao_shaping/main.py hadamard-matrix [OPTIONS]
```

#### SLM Zernike PIB优化 (slm-pib)

> 完整选项与示例见 [docs/slm/slm_pib.md](docs/slm/slm_pib.md)

```bash
python src/ao_shaping/main.py slm-pib [spgd|heuristic] [OPTIONS]
```
等同于: `python -m ao_shaping.runners.slm.slm_shaping_runner`

#### 闭环波前优化 (closed-loop)

> 完整选项与示例见 [docs/wfs/closed_loop.md](docs/wfs/closed_loop.md)

```bash
python src/ao_shaping/main.py closed-loop [OPTIONS]
```
等同于: `python -m ao_shaping.runners.slm.zernike_matrix_runner closed-loop`

#### 交替电压下发 (alt-voltage)

> 完整选项与示例见 [docs/micro_drive/alt_voltage.md](docs/micro_drive/alt_voltage.md)

```bash
python src/ao_shaping/main.py alt-voltage [OPTIONS]
```
等同于: `python -m ao_shaping.runners.micro_drive.voltage_runner`

#### 全量交替电压下发 (full-voltage)

> 完整选项与示例见 [docs/micro_drive/full_voltage.md](docs/micro_drive/full_voltage.md)

```bash
python src/ao_shaping/main.py full-voltage [OPTIONS]
```
等同于: `python -m ao_shaping.runners.micro_drive.voltage_runner full-voltage`

#### DM响应矩阵标定 (dm-matrix)

> 完整选项与示例见 [docs/dm/dm_matrix.md](docs/dm/dm_matrix.md)

```bash
python src/ao_shaping/main.py dm-matrix [OPTIONS]
```
等同于: `python -m ao_shaping.runners.matrix_runner`

#### AdaMOD+SPGD 混合 PIB 优化 (combined)

> 完整选项与示例见 [docs/dm/combined.md](docs/dm/combined.md)

```bash
python src/ao_shaping/main.py combined [OPTIONS]
```
等同于: `python -m ao_shaping.runners.nlight_dm.combined_runner`

### 串行流水线优化器处理流程详解

> 完整内容见 [docs/dm/pipeline.md](docs/dm/pipeline.md)

### 单独运行脚本

所有 runner 与工具都可以绕过 `main.py` 直接以模块方式运行。每条命令的
`python -m` 等价入口见上文各节末的 `等同于:` 行; 详细选项见对应 `docs/` 页面。

```bash
# main.py 之外的直接入口 (与上文各节 "等同于" 一一对应)
python -m ao_shaping.runners.nlight_dm.wf_runner            # wf
python -m ao_shaping.runners.nlight_dm.axis_beam_runner    # pib
python -m ao_shaping.runners.nlight_dm.pipeline_runner     # pipeline
python -m ao_shaping.runners.slm.zernike_matrix_runner     # zernike-matrix
python -m ao_shaping.runners.slm.rms_zernike_runner        # rms-zernike
python -m ao_shaping.runners.zernike_search_runner         # ga-zernike
python -m ao_shaping.runners.matrix_runner                 # dm-matrix / hadamard-matrix
python -m ao_shaping.runners.nlight_dm.combined_runner     # combined
python -m ao_shaping.runners.micro_drive.voltage_runner    # alt-voltage / full-voltage
python -m ao_shaping.runners.slm.slm_shaping_runner       # slm-pib / spgd-square
python -m ao_shaping.runners.slm.gsnet_runner             # slm-gsnet
python -m ao_shaping.runners.slm.gs_refine_runner         # slm-gs-refine
python -m ao_shaping.runners.slm.model_in_loop_runner     # slm-model-in-loop
python -m ao_shaping.tools.slm.calibration.lut_runner     # slm-lut
python -m ao_shaping.tools.slm.panel.diagnose             # slm-diagnose
python -m ao_shaping.tools.micro_dm.micro_dm_image_collect # Micro-DM 逐单元采集
```

探针 (`slm.panel.*` / `ccd.*` / `wfs.*` / `zernike.*`) 见上方「SLM 台架探针」表。

### Micro-DM 数据目录

`micro_dm_image_collect` 采集的图像按 `IP-通道号` 落盘, 磁盘布局 (`data/md_test/`)
承重于 `scripts/` 侧的 Micro-DM diff 分析流水线, **改动落盘命名/目录前先看
`scripts/md_img_*.py`**。完整布局 (原始灰度图 / 100V 差分图 / GIF) 与
网格坐标 → IP 组 → 图像文件的映射表见
[`data/md_test/README.md`](data/md_test/README.md);
工具说明见 [docs/tools/micro_dm_image_collect.md](docs/tools/micro_dm_image_collect.md)。

### ML训练

ML 代码在独立包 `src/ml/` (可选依赖组), 与硬件层解耦。训练器、模型与指标的
**权威说明在实验报告里** (报告是结论的唯一事实源, 脚本只做渲染):

| 任务 | 入口 | 报告 |
|---|---|---|
| Zernike 系数 → 远场前向模型 (`ml/zernike/`) | `python -m ml.zernike.train_coeff` | [`report/zernike_coeff2amp/report.md`](report/zernike_coeff2amp/report.md) |
| 前向模型搜索 (physics × U-Net × loss) | `python scripts/forward_search.py` | [`report/loss_defects/forward_search_report.md`](report/loss_defects/forward_search_report.md) |
| U-Net+GAN 相位预测 (`ml/phase/`) | `python scripts/train_phase_prediction.py --config configs/train_config.yaml` | — |

两个关键口径 (完整论证见报告):

- **只用 R² / corr / 光斑域指标判断拟合质量**, 不要用 MSE/PSNR/SSIM。实测同一配置下
  `normalization="sum"` 给出 PSNR 72.1 dB / SSIM 0.9996, 而 R² 反而更差 —— 任何含 sum
  归一化的指标都在度量归一化本身。
- **fold 与 seed 不能混配**: 同 seed 的 N 折共享同一条初始化随机流, 配对检验的 p 值
  偏乐观。判定前跑多种子。

`ml/hwdataset/` 把 `data/debug/` 下的硬件调试产物转成 PyTorch Dataset
(输入 = SLM 相位 + 曝光, 输出 = 相机画面); 语料统计与抽样自检:

```bash
python -m ml.hwdataset.inspect --index-cache data/hw_index_cache.json --sample 5
python -m ml.hwdataset.inspect --index-cache data/hw_index_cache.json --build-cache
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

# 多 SLM 控制器 (相位图案生成/下发, 含闭环整形 (GS+SPGD / 正向模型))
streamlit run src/ao_shaping/gui/slm/multi_slm_controller.py

# 1300 陶瓷单元查看器 (网格浏览 + 图片标注)
streamlit run src/ao_shaping/gui/r50/ceramic_viewer.py
```

`multi_slm_controller.py` 提供多种全息相位图案生成：平场、闪耀光栅、达曼光栅、涡旋相位等。支持**从 CSV 加载相位**（格式：1200×1920，值 0~1023，首行/首列为 Y/X 索引），走 `load_gray_from_csv` → `csv_to_phase` → `display_phase` 三步管线，与 GUI 预览共用 `create_phase_from_array()` 保证字节级一致；也支持将当前相位或相位 A/B **导出为弧度 CSV**（`Santec.save_phase_to_csv()`，保留 Y/X 行列索引）。导出文件的数据区是弧度值而非灰度值，不能交给 `load_gray_from_csv()` 或 `csv_to_phase()`；重新使用时应按弧度读取并传入 `create_phase_from_array()`。

旧的开环方形整形功能（**GS方形整形** / **GS方形整形(高斯版)** / **稳像法整形**，含 `_gs_square_phase_radians` / `generate_gs_square_phase`）已移除：在本 2f-Fourier 台架上，用未标定的台架参数做开环方形整形已被实测证明行不通 —— Zernike SPGD 与未标定 GS 从未产出过均匀方形（见 [`report/shape_trials/README.md`](report/shape_trials/README.md) 与 [`report/shape_trials/online_20261010.md`](report/shape_trials/online_20261010.md)）。取而代之的是新的 **闭环整形** 区（`src/ao_shaping/gui/slm/shaping_controls.py::render_closed_loop_shaping`，单/双 SLM 布局均渲染），提供两种方法：

1. **GS+SPGD 闭环整形（已验证配方）**：与 CLI `slm-gs-refine` 同一代码路径（`SlmGsRefineParams` → `build_gs_refine_config` → `optimize_slm_gs_refine` → `save_gs_refine_results`），GUI 默认值即交替重放验证过的配方（[`report/shape_trials/gs_refine_live_20261012.md`](report/shape_trials/gs_refine_live_20261012.md)）：cam_type daheng、曝光 0.05 ms（大恒硬限 ≤0.4 ms）、pupil_center `960,690`、beam_radius_px 450、target_side 100、cam_size 250、gs_iters 12、far_field_padding 3、phase_grid 12、epochs 12、seed 0、confirm_best 开、gs_method plain。验证结果：flat 0.527 → plain GS 0.775 → SPGD best 0.780（两次一致的交替重放）。
2. **正向模型闭环整形（前沿）**：CLI `slm-model-in-loop` 链路（`SlmModelInLoopParams` → `build_model_in_loop_config` → `optimize_slm_model_in_loop`），需要 torch；目前最佳均匀性 CV 0.643 / EE 0.786，但尚未成为被接受的均匀方形（见 [`report/shape_trials/model_in_loop_live_20261013.md`](report/shape_trials/model_in_loop_live_20261013.md)）。

设备生命周期注意：上述优化器在内部自行打开/关闭 SLM 与相机（Santec SDK 句柄不可重入），因此 GUI 在整形运行期间会**暂时释放目标 SLM 句柄**、运行结束后重新获取（只释放目标 SLM，另一台 SLM 保持连接）。结束时的最佳相位（raw 弧度）经常规 `display_phase` 路径下发显示。GUI 显示 flat/GS/best 各阶段得分、PIB、CV 与 GS bake-off 判定（adopted/rejected）；debug 开启时产物存于 `data/debug/slm_gs_refine/<ts>/`。

## Zernike 使用指南

> **Canonical 入口 (单一事实源)**: 全项目所有 Zernike 纯数学 (模式枚举 / 索引换算 / 相位生成 / 系数解析 / 单位换算) 统一走 `utils/wavefront/` 两个模块。**任何脚本、runner、工具、GUI 不得自行实现** Noll↔(n,m) 查表、Zernike 多项式求值或相位生成 —— 2026-09 已完成去重重构, 此前散落在 `gui/slm/`、`optimizer/wfless/`、`tools/slm/` 的重复实现已全部收敛到这两层。

### 两层入口

| 层 | 模块 | 公开 API | 何时用 |
|---|---|---|---|
| API 层 (首选) | `ao_shaping.utils.wavefront.zernike_utils` | `parse_zernike_coefficients` (Noll dict / (n,m) dict / Noll 序数组 → {(n,m):amp}), `generate_zernike_phase` (系数 → **raw 弧度**相位图), `list_zernike_modes` (→ [(noll,n,m,name)]), `coefficients_to_array` ((n,m) dict → Noll 序数组), `um_to_waves` (WFS µm→λ) | 绝大多数场景: 解析 / 生成 / 枚举 / 单位换算 |
| 引擎层 (复用/底层) | `ao_shaping.utils.wavefront.zernike_calc` | `ZernikeGenerator` (带网格缓存: `generate_noll` / `generate_polynomial` / `generate` / `fit`), `noll_to_nm` / `ZernikeGenerator.nm_to_noll` (支持任意 Noll), `zernike_modes` / `noll_indices` (模式枚举), `calc_n_zernike_terms`, `get_zernike_name`, `fit_zernike` | 同一分辨率反复生成相位 (复用实例, 省 RZern 建网格开销), 或需要底层索引换算 / 拟合 |

单向依赖: `zernike_utils` → `zernike_calc` (API 层内部创建/复用引擎)。上层 (optimizer / runner / scripts / GUI) 只 import 这两层, 不得反向。

### 最小示例

```python
from ao_shaping.utils.wavefront.zernike_utils import (
    parse_zernike_coefficients,
    generate_zernike_phase,
    list_zernike_modes,
    um_to_waves,
)

# 1) 解析系数 — 3 种输入格式, 输出统一为 {(n, m): amplitude}
coeffs = parse_zernike_coefficients({"5": 1.0, "13": 0.5})   # Noll 索引 dict
coeffs = parse_zernike_coefficients({(2, 0): 1.0, (4, 0): 0.5})  # (n, m) dict
coeffs = parse_zernike_coefficients([0.0, 0.0, 0.0, 1.0])    # Noll 序扁平数组

# 2) 生成相位 — 输出 **raw 未包裹弧度** float64 (孔径外 NaN), 非 uint16
phase = generate_zernike_phase(coeffs, resolution=(1920, 1200), n_max=4)

# 3) 枚举模式 → [(noll, n, m, name), ...] (Noll 序)
for noll, n, m, name in list_zernike_modes(4):
    print(noll, (n, m), name)
```

同一分辨率反复生成时, 直接持有 `ZernikeGenerator` 实例复用 (内部缓存 RZern 坐标网格):

```python
from ao_shaping.utils.wavefront.zernike_calc import ZernikeGenerator

gen = ZernikeGenerator((1920, 1200), n_orders=4)   # 一次建网格, 多次复用
gen.set_bits(10)
img = gen.generate_polynomial({(2, 0): 1.0, (4, 0): 0.5})
```

### 红线 (违反 = 重新引入重复 / 约定漂移 / 单位 bug)

1. **禁止自写 Noll↔(n,m) 查表或模式枚举**。历史教训: `noll_to_nm_legacy` (那里 Noll 5=(2,0), 与 canonical 相反) 曾与 canonical 并存导致两套索引混用, 已删除; 新代码一律 `zernike_calc.noll_to_nm()` / `zernike_utils.list_zernike_modes()`。
2. **禁止生成器自行 `mod 2π`**。`generate_zernike_phase()` / `ZernikeGenerator.*` 返回 **raw 未包裹弧度**; 唯一 wrap 点在 SLM 驱动 `Santec.create_phase_from_array()` (弧度→灰度)。弧度→灰度统一走 `utils/slm/phase_display.phase_to_slm_grayscale(phase, slm=slm)`。
3. **禁止 min-max 归一化相位**。`(p−min)/(max−min)` 归一化使图案**尺度无关** (系数 ×1 与 ×4 输出字节完全相同, 幅度不可控), 是已知反模式, 不得用于新代码。
   历史例子: `ZernikeDM.generate_phase` (于 2026-09 修复) 与 `PatternHelper._zernike_to_uint16` (**于 2026-09 删除**)。
4. **WFS 系数单位必须统一**。WFS `get_zernike()` 返回 **µm**; 参与响应矩阵 / 矫正运算前必须 `um_to_waves()` (µm→λ, ÷0.532); 反解出的 λ 系数在喂给 `generate_zernike_phase` / `make_phase` 前必须 **×2π** (λ→rad)。两个真实 bug (2026-09-16, 均因单位混用: 系数放大 1.88× / 相位缩小 6.28×) 修复后闭环 RMS 改善 13.8% → 42.1%。
5. **Noll 约定 = Noll 1976 (aotools)**: Noll 4=(2,0) defocus, Noll 5=(2,-2) astig, Noll 11=(4,0) spherical, Noll 13=(4,-2) ⚠ (不是 (2,0))。`zernike_calc.noll_indices` (Noll 序) 与 `zernike_modes` ((n,m) 字典序)**顺序不同, 不可互换**; 完整前 15 阶映射表见 `zernike_utils` 模块 docstring。

## 物理-optical 数据类模型 (`ao_shaping.model`)

`ao_shaping.model` 将二维采样值与像素间距、波长及来源信息放在一起，明确区分
相位 (rad)、光程差 (m)、场振幅、强度和电压。数据类已经实现；跨模块调用仍在分阶段迁移，
不要把类型存在理解为所有驱动和优化器都已返回该类型。

| 类 | 单位与用途 |
|---|---|
| `PhaseMap`, `OPDMap` | 未包裹相位 (rad) 与光程差 (m)，用明确的波长相互转换 |
| `AmplitudeMap`, `ComplexField`, `PSFImage` | 场振幅 `|E|`、复电场与焦面强度；强度转振幅使用平方根 |
| `WfsSlopes` | 分开的 `sx`/`sy`，单位为 `arcsec` 或 `rad`，支持 `to_units()` |
| `DmCommands` | 电压数组、致动器数量和允许范围，支持 `clipped()` |
| `ZernikeCoefficients` | 带显式 Noll 索引的 `rad`/`waves` 系数 |
| `CoordinateGrid2D` | 坐标、像素间距、原点和单位 (`um`/`m`/`pixels`) |
| `BeamMetrics`, `TurbulenceParameters`, `WavefrontStatistics` | 光束指标、湍流参数和波前统计 |
| `FieldMetadata` | 时间戳、来源、波长及附加元数据 |

**物理空间重采样**: `PhaseMap`、`OPDMap`、`AmplitudeMap`、`ComplexField` 和
`PSFImage` 都提供 `resample_to(shape, pitch_size_um)` (PSF 参数名为
`pixel_scale_um`)。`model/spatial.py` 将输入和输出网格的几何中心都设为
`(0, 0)`，像素中心坐标为 `(index - (size - 1) / 2) * pitch`；默认双线性插值，
超出输入网格的采样值填零。这是点采样插值，**不保证积分光通量守恒**。
`utils/image/resample.py` 的亮斑居中裁剪用于相机图像对齐，语义不同。

```python
import numpy as np

from ao_shaping.model import AmplitudeMap, ComplexField, PhaseMap

phase = PhaseMap(np.zeros((5, 5)), pitch_size_um=8.0)
opd = phase.to_opd_map(wavelength_m=532e-9)  # metres
phase_16um = opd.resample_to((3, 3), pitch_size_um=16.0).to_phase()

amplitude = AmplitudeMap.from_intensity(np.ones((5, 5)), pitch_size_um=8.0)
field = ComplexField.from_maps(phase, amplitude)  # 检查形状和像素间距
```

**OOPAO**: `PhaseMap`、`AmplitudeMap` 和 `ComplexField` 提供
`from_oopao_source()` / `to_oopao_source()`。实际调用时才通过
`ao_shaping.drivers.sim._oopao_compat` 加载 OOPAO，避免模型包与模拟驱动的循环导入。
可运行以下无硬件测试检查当前环境：

```bash
uv run pytest tests/ao_shaping/model tests/ao_shaping/drivers/sim/test_oopao_backend.py -q
```

**当前接入范围**: DM 基类、`SimMicroDM`、`SimulateDM` 接受 `DmCommands`；
`phase_to_slm_grayscale()` 接受 `PhaseMap`。这些接口仍接受原始数组，原始数组调用
保持原有返回类型。目前没有统一添加弃用警告。硬件驱动覆盖方法、WFS、优化器、GUI
和其余工具尚待迁移。

#### 2f-Fourier 整形台架的唯一入口

前向模型、目标函数指标与参考优化器**只在** `drivers/sim/slm_shaping_bench.py`。
`optimizer/wfless/slm_shaping_bench.py` 是**向后兼容 re-export shim**。

> ⚠️ **不要再手抄第二份实现**。2026-10-01 的 merge 曾在两个模块各留一份
> `gs_shape`/`differentiable_shape`/`spgd_shape`，而副本用的是
> `beam_backend.focal_plane` 走**未加零填充**的瞳孔，canonical 版则零填充到
> `far_field_size` 再做 Fraunhofer FFT —— 同一个方法**按 import 来源返回不同指标**，
> 且不报任何错。拆分模块时只加 re-export，不要复制函数体。

同理 `iterative_zernike_shaping._zernike_basis()` 直接返回 `__init__` 里由
canonical `ZernikeGenerator` 预计算的基，**不得**在算法层重新推导径向多项式
(那会构成第二套会漂移的 Zernike 数学)。


### 波前传感器
- **Thorlabs WFS系列**: 支持自动图像采集和倾斜去除
- **WFS 注册表** (`drivers/wfs/_registry.py`): `register_wfs` / `create_wfs` / `list_wfs_types` / `resolve_wfs`，镜像 DM 的注册表；采用延迟绑定，硬件包**不会** import 仿真包。`resolve_wfs(None)` 仍返回 `ThorlabWFS`，故硬件运行行为不变。
- **`--wfs_type [thorlab|sim]`** 定义在共享的 `WfsParams` 上，8 个已注册命令**既暴露也真正透传**该 flag: `wf`、`pipeline`、`rms-zernike`、`ga-zernike`、`greedy-zernike`、`dm-matrix`、`hadamard-matrix`、`zernike-matrix`。`closed-loop` **刻意没有** —— 它回放已保存的响应矩阵。

#### 仿真 WFS (`SimulatedWFS`)

`drivers/sim/wfs/simulated_wfs.py`，OOPAO Shack-Hartmann 实现 `BaseWFS`，可直接顶替 `ThorlabWFS`，使所有 WFS 类 runner 无需硬件即可执行。

- **测瞳面梯度而非焦平面传播**: SH 通过微透镜阵列成像瞳孔，探测器上光斑位移编码各子孔径的局部 tip/tilt，因此测量量是**瞳面相位梯度**。早前计划复用本仓 `wave.py` 的焦平面传播 —— 那产出的是**相机图像**而非 slope，二者不可互换。
- **slope 数组是行块状布局**: 第 `0:n_subap` 行为 x-slope，第 `n_subap:` 行为 y-slope (纯倾斜探针实测: 纯 x 倾斜 → 上半 rms 1.29e-4, 下半 0)。
- 🔴 **SH 测不到 piston**: 子孔径的绝对相位偏移不移动光斑。`list_zernike_modes` 从 Noll 1 = piston 起，把它放进拟合基会留下一个近零空间列，`pinv` 把它放大成巨大的伪系数 (实测纯离焦瞳孔下 **+0.50 rad**)。故 piston 必须排除出拟合基 (`_FIT_FIRST_MODE`) —— **这不是 off-by-one**。
- **单位**: `get_wavefront()` → **waves** = `phase_rad / 2π`; `get_zernike()` → **µm** = `phase_rad * λ_nm*1e-3 / 2π`。`um_to_waves()` 写死 532 nm，故默认波长必须 532 nm。
- ⚠️ **保真度不足以当波前基准 (实测，勿高估)**: 注入已知模式再读回系数 —— noll 2 tilt 单独 24% / 与其他模式同时 **83%**; noll 4 defocus 1.4%; noll 11 spherical 32%。tilt 单独注入只差 24%，与其他模式同注入却差到 83% ⇒ 主误差是**模式间串扰 (cross-talk)**，不是逐模式噪声。已排除质心量化与标定 pass 污染 (平场读数严格为 0)。**任何由 `get_zernike()` 推出的 RMS 改善率或 Strehl 都必须标注为「未验证」**; 要可信数值请用硬件标定。
- **接口契约必须与 `ThorlabWFS` 完全对齐**，缺一个就直接崩: `get_wavefront()` 的 stats 键必须是 `min/max/diff/mean/rms/wighted_rms`; `build_subaperture_mask()` 返回 **2 元组** `(mask_2d, valid_indices_flat)` 且按 **subaperture 网格**定尺寸 (**不能**按 flux 定 —— OOPAO 的 flux 报在 8×8 lenslet 网格而 slope 跨 6×6 网格); runner/优化器直接读取 12 个成员 (`num_spots_x/num_spots_y`、`mla_index`、`serial_num`、`device_name`、`exposure_time`、`high_speed`、`use_custom_ref`、`pupil`、`d_x`、`get_mla_name()`、`set_ref_plane()`)。锁定测试: `tests/ao_shaping/drivers/sim/test_simulated_wfs.py::TestRunnerFacingSurface`。

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
  from ao_shaping.drivers.dm.micro import MicroDM

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
  from ao_shaping.drivers.dm.micro import AsyncMicroDM

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
- **大恒相机系列**: DahengCamera，支持14位和16位模式
- **MIICAM系列**: MIICamera，支持高速采集

### 空间光调制器
- **Santec SLM200**: 支持相位图案生成、缓存和CSV加载/导出
  - `open()` 方法已重构为子方法 (`_apply_config_params`, `_load_correction`, `_setup_wavelength`)，逻辑更清晰
  - 波前误差矫正通过独立 `WavefrontCorrection` 类管理（CSV加载→异常点检测→矫正映射图）
  - 矫正数据自动按优先级加载: `__init__` 显式指定 > 配置文件 > 默认路径
  - **CSV 相位加载与导出**（`multi_slm_controller.py` GUI）：加载格式为 1200×1920、值 0~1023、首行/首列为 `Y/X` 索引；管线为 `load_gray_from_csv`（驱动层格式校验）→ `csv_to_phase`（灰度→弧度）→ `display_phase`（`create_phase_from_array()` 弧度→灰度+矫正+LUT+平移），与 GUI 预览共用同一路径保证字节级一致。导出使用 `Santec.save_phase_to_csv(phase_rad, destination)`，数据区为弧度值并保留 `Y/X` 行列索引，目标可为路径、`BytesIO` 或文本流；导出文件不能交给灰度加载管线。详见 `docs/slm/slm_gui_manual.md`。

> **⚠️ SLM 平场灰度生成注意事项**
>
> Santec SLM200在1064nm附近存在**振幅耦合**效应——SLM加载不同灰度值的平场相位时，相机采集到的光斑亮度会随灰度值变化（周期 ≈ 2π，即约993灰度值）。这是SLM的固有特性，已在实验中验证（参见 `scripts/validate_flat_phase_gray.py`）。
>
> **关键规则1（灰度值路径）**: 平场相位（以及其他直接灰度图案）**必须**使用 `np.full((height, width), gray, dtype=np.uint16)` 生成，**不能**通过 `create_phase_from_array()` 传递。因为 `create_phase_from_array()` 将输入作为**弧度**处理（mod 2π → 弧度/2π × 1023），uint16灰度值会经过不必要的弧度转换而被静默损坏。
>
> **关键规则2（内存模式槽轮换）**: Santec SLM 在内存模式下，**前后两次写入不能使用同一个内存槽**（memory slot）。当 `display_memory(slot)` 被调用时，如果该槽已经在显示，设备会将此调用视为空操作（no-op），LCOS 面板不会刷新，屏幕上仍显示上一次的相位图案。连续写入时必须使用不同的槽位——`slm-gsnet` runner 在 **2~125 槽范围内随机选取**并排除当前显示槽（启动时 `get_displayed_memory_number()` 续接，跨进程也不冲突）。`display_data()` 内置的 127 槽循环机制同样满足这一约束。
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
- 模拟 WFS (`SimulatedWFS`, OOPAO Shack-Hartmann，见上文「仿真 WFS」)
- DM→瞳孔相位耦合 (`SimDmOptics`, 可分离高斯影响力函数)
- 2f-Fourier SLM-PIB 数字孪生 (`SimPibSystem`)

#### DM→相位耦合已接入仿真 (2026-10-01)

`SimDmOptics` (`sim/dm_optics.py`) 把 DM 电压映射为瞳孔相位，经 `SimPibSystem.dm_optics`
在 `far_field()` 的 **cache-miss 分支**叠加 (与 `disturbance` 同一约定: 相位贡献者在**求值时**相加,
绝不烘进 `self._phase`)。此前 `pib` / `combined` 的仿真结果**没有物理意义** (DM 动作对远场无影响)。

| runner | 驱动量 | 仿真下是否有物理意义 |
|--------|--------|---------------------|
| `slm-pib` / `slm-gsnet` / `spgd-square` | SLM 相位 | **有** —— `set_phase_rad` 直接改变远场 |
| `pib` / `combined` | DM 电压 | **有** —— `SimulateDM.send_voltages` → `SimDmOptics` → `far_field()` |
| `wf` / `rms-zernike` 等 | DM 电压 → WFS | **有** —— DM 同时发布给当前活跃的 `SimulatedWFS.dm_optics` |

实现要点:
- **可分离高斯影响力函数**: 每个致动器一个高斯凸包, x/y 可分离, 故用一次矩阵乘求值。稠密 influence 矩阵在真实 1920×1200 面板下需 ~1.2 GB, 不可接受。
- **单位链**: `opd_um = stroke_um * v / v_max` → `phase_rad = opd_um * 2π / (λ_nm * 1e-3)`。0 V 为平场 (与真实驱动一致)。`V_Min=-300` 与 `V_Max=499` **不对称**, 故负向半程行程短于正向 —— 裁剪到负轨**不是**正轨的镜像。
- **相位是 raw 未包裹弧度**, 不做 `mod 2π`: 全项目唯一 wrap 点在 SLM 驱动。
- **cache 失效靠 `dm_optics.version`**: DM 是独立光学状态, 电压变化必须让 `_far_field` 失效, 否则相机继续返回旧图, 看起来"DM 毫无作用"。
- 传感器按自己的瞳孔网格 (48×48) 持有一份 `SimDmOptics`, 因此**无需**把 1200×1920 重采样。
- ⚠️ 以前 DM 只发布到远场, 于是 `wf` 永远读到平 pupil、无论 DM 怎么动 RMS 都是 0。

修复后实测 `main.py pib --cam_type sim --dm_type sim -e 200`, 目标函数真实上升: pib 0.66 → 2.33 → **2.98** (修复前 200 epoch 恒为 0.0145; pib 单位为 %, 故可 >1)。

⚠️ **相机噪声的地板项会把功率比指标淹没**: `far_field_noisy()` 曾以 `clip(img, 0, None)` 收尾。
`far_field()` 是 `|FFT|²` 本身非负, 帧内所有负值**只可能来自读出噪声** —— 在 0 处裁剪会把对称分布
**整流**, 凭空造出与**像素数**成正比的 DC 地板: shot noise 地板/信号 **1.005×** (正确),
read noise 经 clip 后 **4598×** (缺陷)。光斑只占 ~150 px 而画幅 2.3 M px, 地板携带 ~3000× 的光,
`pib` 的分母是整帧总功率, 于是它量的是这块地板。

修复: **不再逐帧裁剪**。真实探测器的本底位于其阈值**之下**, 只在量化时裁剪一次, 去黑电平靠暗帧扣除。
判据用**暗区均值** (≈0) 与**桶内信号占比**, **不要**用整帧 total 判定噪声 (零均值噪声在 2.3 M px 上
求和标准差 ≈ σ√N ≈ 758, 与 152 的信号总量同量级)。参见 `tests/ao_shaping/drivers/sim/test_sim_noise_model.py`。

### 波前干扰 (湍流 + 热晕): 静态 / 动态

`ao_shaping.drivers.sim.disturbance` 为 SLM 类仿真提供**大气湍流 + 热晕 (热晕/halo)** 波前干扰，
`SimPibSystem` 通过关键字参数 `disturbance=` 接入 (`far_field()` 在傅里叶变换前叠加，
且只在 cache-miss 分支推进，故每次真实光学评估只消耗一次；`self._phase` 保持纯 SLM 命令，
不会被重复计入)。两种体制：

| 体制 | 含义 |
|---|---|
| `static` | 全程复用**唯一一张冻结**相位屏 (对应仓库 `closed` 湍流) |
| `dynamic` | **每次光学评估重抽一张独立**相位屏 (对应 `open`/sliding，即完全去相关 "white-in-time" 极限) |

> `dynamic` 不是风场平流模型: 真实大气去相关时间 ~10–50 ms 远短于本环路每评估 ~0.375 s，故完全去相关
> 是此处的合理渐近。

湍流复用 canonical `beam_backend.turbulence_phase` (von Karman，相位屏方阵生成后按面板原生
8 µm 像素居中裁剪，避免粗网格丢失约 41% 相位幅度)；热晕由 canonical
`zernike_utils.generate_zernike_phase` 构造 (Noll 4 离焦 + Noll 11 球差，负号=负热透镜)，
经 smoothstep 光晕窗延展并按 waves 归一化峰谷值。全程 raw 未包裹弧度。

**注意**: `cn2` 与 `distance` 是**退化**旋钮 (`r0` 只依赖二者乘积，单层薄屏无传播物理，本台架是
~0.3 m 实验室 2f 光路) —— 属参数化应力测试而非大气传输仿真；numpy 相位屏缺少次谐波补偿，
实测 σ 低于同 r0 的解析值，故报告一律引用**实测** σ。`SimulatedThermalScreen`(热晕相位屏)依赖未安装的
`sim.digitaltwin` 且回退为静默 no-op，**不可使用**。

配套运行与报告 (完全离线，见 `scripts/README.md`):
```bash
python scripts/slm_pib_sim_run.py --epochs 300 --disturbance static
python scripts/slm_pib_sim_run.py --epochs 300 --disturbance dynamic
python scripts/generate_slm_pib_sim_report.py --max-runs 2   # → report/slm_pib_sim/report.md
```

实测 (300 epochs，各 604 次光学评估): static 用 1 张屏、逐次 RMS 恒定 (σ≈0.566 rad ≈0.090 waves)；
dynamic 用 604 张屏、逐次 RMS 变化 (σ≈0.580 rad ≈0.092 waves)；目标改善 static +0.0286 > dynamic +0.0202
(冻结扰动更易被校正)。

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
  from ao_shaping.drivers import MIICamera, DahengCamera
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

#### 9.1 驱动惰性加载契约（2026-10-01）

`import ao_shaping` 及其任何子包**都不得加载原生 SDK、也不得做磁盘 I/O**。这不是性能优化，而是正确性问题 —— 仓库里已因此真实炸过两次。

**两条独立的惰性**：

| 层级 | 机制 | 位置 |
|------|------|------|
| **包级** | PEP 562 模块 `__getattr__` | `drivers/_lazy.py::install_lazy_attrs` |
| **构造级** | `Device._load_sdk()` / `Device._ensure_sdk()` | `drivers/device_base.py` |

```python
class MyDriver(Device):
    @staticmethod
    def _load_sdk():        # 覆写点; 默认返回 None (仿真设备无 SDK)
        return load_dll()

    @property
    def _lib(self):          # 只读属性 → 所有 self._lib.<fn> 读点零改动
        return self._ensure_sdk()   # 解析一次并缓存
```

红线:
- `install_lazy_attrs` 内部**必须**写 `module_globals[name] = value` 缓存，否则后续直接 `import` 该子模块会重新绑定全局变量，惰性被彻底绕过。
- 惰性名字**绝不能**在模块作用域出现同名赋值 —— 那会遮蔽 `__getattr__` 并退回 eager。
- **禁止在任何类体/模块作用域做 I/O**。仓库**输入**资产 (如 `dm_adj.txt`) 经 `drivers/dm/_adjacency.py::load_adjacency()` 加载，路径锚定包位置；运行**输出** (如 `PATHS.root_dir`) CWD 相对是**正确**的。
- **分层方向单一**: 硬件包不得 import 仿真包。仿真 DM 经同一个 `install_lazy_attrs` 惰性解析，`dm/_registry.py::_ensure_sim_dms_bound()` 保证 `"sim"` 在**首次注册表使用**时绑定。

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
- [drivers/AGENTS.md](src/ao_shaping/drivers/AGENTS.md): 硬件驱动文档 (含驱动惰性加载契约)
- [drivers/sim/AGENTS.md](src/ao_shaping/drivers/sim/AGENTS.md): 仿真模块文档 (含 `SimulatedWFS` 保真度实测表)
- [scripts/README.md](scripts/README.md): 脚本说明 (含报告生成架构)
- [report/](report/): **实验报告总索引** (2026-10-05 从 `docs/` 迁出，与设备说明文档分开)。
  每份报告都带 `生成脚本` / `复现命令` / `运行环境` 溯源块，索引表给出
  **报告 → 生成脚本（仓库根相对路径）→ 离线/硬件** 的完整对应关系。
  新增报告 = 在 `scripts/_common/provenance.py` 的 `REPORTS` 加一条，再跑
  `python scripts/sync_report_provenance.py`（幂等；`--check` 只校验不写入）。
- [docs/](docs/): **设备说明文档** — 设备规格、SDK/驱动用法、装配与操作手册、图集。
  不再放实验结论 (含仍在原地的每设备硬件测试报告 `<device>/<device>_report.md`，
  由 `tests/ao_shaping/utils/test_report.py` 写出)：
  - [SLM 操作手册](docs/slm/slm_gui_manual.md)、[PatternHelper 指南](docs/slm/slm_pattern_helper.md)、[开机表征指南](docs/slm/pre_run_characterization.md)、[SLM 相位图集](docs/slm/slm_patterns/)
  - **每设备硬件评测报告** (2026-09 重新生成): [WFS](docs/wfs/wfs_report.md) / [MiiCam](docs/miicam/miicam_report.md) / [SLM-200](docs/slm-200/slm-200_report.md) / [Micro-DM](docs/micro-dm/micro-dm_report.md)
  - 厂商手册: [大恒相机 SDK](docs/daheng/) · [Thorlabs WFS](docs/thorlab-wfs/) · [Santec SLM-200/DLL](docs/slm-200/) · [微驱动器接线表](docs/micro%20deformable%20mirror/)
  - [驱动层架构](docs/drivers_architecture.md)、[AO 仿真指南](docs/simulation.md)、[Tabu 算法](docs/tabu_search_algorithm.md)、[已知问题](docs/issues_report.md)、[待办索引](TODO.md) · [已完成档案](docs/dev/todo_archive.md)

## 近期更新

### v0.17.0 (2026-10-06)

**Zernike 系数→远场前向模型训练** (15 个提交 `0694a0e` → `493ef1e`):

- **物理-optical 前向模型** (`ml/zernike/models.py`): `ZernikeAmpModel` —— 用单个全局 Zernike 系数向量 (raw 弧度, Noll 1 piston 排除) 表示整个台面的波到远场的确定性映射: `phasor(cos,sin) → complex pupil → exp(i·Σc_k·Z_k) → centre-pad + fftshift(fft2(ifftshift)) → |F|² (intensity)`。可观测量 `intensity` (默认) / `amplitude`, 归一化 `peak`/`sum`/`none`。默认 `n_max=15` (135 系数), `grid=64`, `observable="intensity"`, `normalization="peak"`。详见 [`report/zernike_coeff2amp/report.md`](report/zernike_coeff2amp/report.md)。

- **可微前向模型** (`ml/zernike/forward_model.py`): `ZernikeCoeffConvNet` (coefficient projection → conv decoder) + `ZernikeCoeffMLP` (baseline)。两者都输出 raw unbounded 单通道图像, peak-normalize 由 caller 完成。ConvNet 拥有空间局部性优势 (移形/尺度), MLP 作为对照臂。

- **训练器** (`ml/zernike/train_coeff.py`): `CoeffTrainConfig` (冻结 dataclass, `--input-terms`/`--protocol`/`--fold`/`--held-out-path`, W&B 集成, `--w-ellipse`). 两种折叠协议: `"file"` (18 折 leave-one-pickle-out) / `"objective"` (4 折 leave-one-objective-out). R² 选择 checkpoint (PSNR/SSIM/MSE 仅作诊断 —— 它们在 per-frame peak-normalised 目标上是退化的)。`--input-terms` 修剪输入到前 N 项 (语料最长向量 78 项, 136 维中 58 维恒为 0)。

- **物理感知损失** (`ml/zernike/losses.py`): `LossConfig` + `composite_loss`。移植自 `ao_shaping.utils.image.target.metrics` 的 PIB/uniformity/energy-loss (torch 版, ROI mask 复用 canonical builder)。新增:
  - `shape_gap_term` (anchored: pred 与 reference 的 ROI 形状统计绝对误差 —— 解决 unanchored PIB/uniformity 在"拟合"任务上的劣性: val R² +0.78 → -0.86)
  - `spot_moment_gap_term` (锚定的径向二阶矩相对误差, MSE 对核/晕分布盲)
  - `ellipse_parameters`/`ellipse_gap_term` (5 参数椭圆 gap, 捕捉位置/aspect/tilt 而径向矩无法)
  - `speckle_detail_term` (log-intensity 梯度差, 惩罚过度细微结构)
  - `poisson_nll` (FourierGSNet Eq.28)

- **数据增广** (`ml/zernike/augment.py`): `phasor_noise` (感应级 phasor 噪声), `phase_flip`, `phase_shift`, `strong_phase_pairs` (GS 提案 / 大振幅 Zernike / 空间粗糙随机相位 → OOD 合成对, 解决 inverse design 外分布问题)。全部显式 seeded (generator), 模型系数 Init 为 0。

- **配对统计来源** (`src/ml/zernike/eval_stats.py`): `sign_flip_pvalue` (精确 2^n 枚举, 最小 p = 2/2^n), `cohens_dz`, `holm_bonferroni`, `paired_comparison`, `skill_scores`, `seed_agreement`。纯 numpy (无 scipy/sklearn), 唯一来源。scripts/ 只做渲染。⚠️ **fold 与 seed 不能混配**: 同 seed 的 N 折共享一条初始化随机流, sign-flip 检验 p 偏乐观 (实测 `input_terms` 136→78 18 折 p=0.0144, 但固定一折 vary seed 后符号翻转, 差值量级小 30 倍 —— 初始化噪声)。

- **目标形状度量** (`utils/image/target/` 子包):
  - **结构项 `log_gradient_difference_metric`** (`utils/image/target/metrics.py`): `log_gradient_field`/`log_gradient_difference_from_fields`/`log_gradient_difference_metric` + `LOGGRAD_EPS`/`LOGGRAD_SIGMA` 常量。惩罚过度细微结构 (speckle), 能量-in-bucket 目标无法捕捉。
  - **`--w_loggrad` CLI** (`runner_common.py`, `objective.py`, `slm_zernike_pib.py`): 新增 `--w_loggrad` (默认 0), 串接 log-gradient 差异度量进 PIB 目标函数。

- **slm-model-in-loop** (`6b85858` + `efedabe`): `slm_model_in_loop` 闭环优化器 recorder pkl 修复 (写入现在真正生效), 负效应臂移除, README 补充硬件使用指南 (pkl 总是写出, 曝光陷阱, 运行前探针)。

- **CLI 帮助测试** (`493ef1e`): CLI 帮助 golden-file 测试增强, SLM 训练数据采集器 (`slm_train_data_collect.py`) 测试引入。

### v0.16.0 (2026-10-05)
- **`slm-pib` / `spgd-square` 合并进单个 runner**: `runners/slm/pib_runner.py` → `runners/slm/slm_shaping_runner.py`, `runners/slm/square_runner.py` 作为其 `square` 子命令合入。命令行名与全部选项不变; 新增两个等价入口 `main.py slm-pib square` 与 `python -m ...slm_shaping_runner square`。
- 修复: **`--cam_type sim` 在两半都是坏的** —— `spgd-square` 从未注册模拟相机 (所谓"离线试跑"会去开真实 Santec 并以 `SantecError -10002` 收场), `slm-pib` 只换了相机却仍构造真实 `Santec`, 且是在烧掉自动曝光探针之后。现统一走 `runner_common.patch_sim_square_shaping` / `patch_sim_pib_shaping`; 两半都能在 2f-Fourier 数字孪生上跑完 2 epoch。`--slm_type sim` 同时从"声明但无效"变成有效 (缺 `--cam_type sim` 时报 UsageError, 而不是去连硬件)。两个 helper 有一处**故意不同**: pib 那半**不** reset 模拟系统, 因为 `slm_pib_sim_run.py` harness 自己装了带干扰的定种子系统并包装了 `reset_system` 以保住干扰 —— 实测重跑 harness 干扰量不变 (`sigma_total=0.5658 rad`)。
- 修复: `spgd-square --delta` 是**空操作** —— `SlmSquareParams.delta` 硬编码 `0.1`, Click 无法区分"没传"与"传了 0.1", 于是默认 `lr=0` 的自适应调度每轮改写它。现改为 `None` + `resolve_spgd_delta` (省略=自适应, 显式=钉住), 与 `slm-gsnet` / `slm-pib` 一致。
- `spgd-square --debug` 过去不产出任何东西; 现写 `data/debug/slm_square/<ts>/` 的 PNG/PKL/JSON (带自描述 sidecar)。历史上的 CSV / 最优系数 / 最优图输出不变。⚠️ 实测这些记录目前**只有 `--basis freeform` 能进 `ml/hwdataset`** (Zernike 基 0/3, 因 `_c` 存的是活动模式向量而被判 `odd_coefficient_length`), 见 TODO H-23。
- 顺带修复了一个**上一提交遗留的破仓**: `gsnet_runner` 已 import `patch_sim_square_shaping` 但定义没跟着进 `runner_common`, 导致 `import ao_shaping.main` 直接 ImportError (clean checkout 无法跑 CLI)。
- 合并动机不是"文件挪个窝": 两个模块驱动同一对设备, 漂移已经产生了**行为差异**而非仅仅排版差异。详见 commit `278a3bc`; 待上设备复测项见 TODO H-20~H-24。

### v0.15.0 (2026-10-01)

**merge `banckend` (16419e5) 整合 + review 修复**

- **模拟 WFS** (`drivers/sim/wfs/simulated_wfs.py`): 新增 `SimulatedWFS` —— OOPAO Shack-Hartmann, 实现 `BaseWFS`, 可直接顶替 `ThorlabWFS`, 使所有 WFS 类 runner 无需硬件即可执行。测**瞳面相位梯度**(非焦平面传播)，排除 piston (SH 测不到绝对相位偏移，否则 `pinv` 放大成 +0.50 rad 伪系数)；`get_wavefront()` 返回 **waves**，`get_zernike()` 返回 **µm**。⚠️ 保真度不足以当波前基准 (tilt 单独注入只差 24%、与其他模式同注入差 83% ⇒ 主误差是模式间串扰)，任何由 `get_zernike()` 推出的 RMS 改善率或 Strehl **必须标注为「未验证」**。
- **WFS 注册表** (`drivers/wfs/_registry.py`): `register_wfs`/`create_wfs`/`list_wfs_types`/`resolve_wfs`，镜像 DM 注册表，延迟绑定故硬件包不 import 仿真包。`--wfs_type [thorlab|sim]` 定义在共享 `WfsParams` 上，8 个命令**既暴露也真正透传** (`wf`/`pipeline`/`rms-zernike`/`ga-zernike`/`greedy-zernike`/`dm-matrix`/`hadamard-matrix`/`zernike-matrix`)；`closed-loop` 刻意没有 (回放已保存的响应矩阵)。
- **DM→瞳孔相位耦合接入仿真** (`sim/dm_optics.py`): `SimDmOptics` 用**可分离高斯影响力函数**把 DM 电压映射为瞳面相位 (稠密矩阵在 1920×1200 下需 ~1.2 GB 不可接受)，`SimulateDM.send_voltages` 同时发布给远场与当前活跃 WFS。此前 `pib`/`combined`/`wf` 的仿真结果**没有物理意义**。
- **驱动惰性加载契约**: `import ao_shaping` 及其任何子包**不得加载原生 SDK、也不得做磁盘 I/O**。包级用 PEP 562 `__getattr__` (`drivers/_lazy.py::install_lazy_attrs`)，构造级用 `Device._load_sdk()`/`_ensure_sdk()` + 只读 `_lib` 属性。`ThorlabWFS` 曾把 `load_dll()` 写在 `__init__` 里使**构造**就 `OSError`; `NLight.py` 曾**类体**执行 `np.loadtxt("data/dm_adj.txt")` 使 `import ao_shaping` 直接失败 (CWD 相对路径) —— 仓库输入资产现经 `drivers/dm/_adjacency.py::load_adjacency()` 锚定包位置。
- **分层方向修正**: `drivers/dm/__init__.py` 曾 import `drivers.sim.dm` (硬件 → 仿真，方向反了) 形成循环，该循环此前靠 `drivers/__init__.py` 的 eager 导入顺序侥幸未暴露 —— 一旦改成惰性就立刻炸。
- **`--disturbance-cn2`** 注入像差 (经 `SimDisturbance` 注入**瞳孔**，默认 0 = 不注入)；`wf` 新增 `--lr`/`--delta` 覆盖。
- 修复: `far_field_noisy` 的 `clip(img, 0, None)` **整流**读出噪声、造出与像素数成正比的 DC 地板 (read noise 地板/信号 **4598×**)，使 `pib`/`combined` 停在 0.0145。改为不再逐帧裁剪 (真实探测器的本底在阈值**之下**，只在量化时裁剪一次)。
- 修复 (review): merge 残留的 `pipeline_runner` 引用未定义的 `wfs`(应为 `wfs_params`)；`sim/slm_shaping_bench.py` 的 `differentiable_shape` 在 torch import 被删后残留 19 处 `F821`；`optimizer/wfless/slm_shaping_bench.py` 是一份**手抄副本**且用的是**未加零填充的旧前向模型** —— 现改为对 canonical 实现的 re-export shim (同一方法此前按 import 来源返回不同指标)；`iterative_zernike_shaping` 的 `_zernike_basis` 调用了已被删除的 `_zernike_radial`(3 个测试失败) —— 现直接返回 canonical `ZernikeGenerator` 预计算的基。
- 规模: 93 文件 / +7232 −547。

### v0.14.0 (2026-10-01)
- **SLM-PIB 仿真的波前干扰** (`ao_shaping.drivers.sim.disturbance`): 新增 `SimDisturbance` / `DisturbanceConfig` —— **大气湍流** (复用 canonical `beam_backend.turbulence_phase`, von Karman 相位屏) + **热晕** (canonical `generate_zernike_phase`: Noll 4 离焦 + Noll 11 球差, 负热透镜, smoothstep 光晕窗 + waves 峰谷归一化), 全程 raw 未包裹弧度。提供 **static** (全程冻结屏 ≡ `closed`) 与 **dynamic** (每次光学评估重抽独立屏 ≡ `open`/white-in-time 极限, 非风场模型) 两种体制; `SimPibSystem` 以关键字参数 `disturbance=` 接入, 仅在 `far_field()` 的 cache-miss 分支推进 (每次真实光学评估一次), `self._phase` 保持纯命令不重复计入。`slm_pib_sim_run.py` 新增 `--disturbance/--cn2/--halo-pv-waves/...` 并写出 companion `disturbance.json`+`.npz`; `generate_slm_pib_sim_report.py` 渲染静态/动态对比 (配置派生 tag + 干扰相位屏图 + 逐次 RMS 轨迹图, 证明 static 恒定 / dynamic 变化) 与 8 条口径说明。53 个新测试用例
- 修复: 相位屏粗网格的系统性幅度损失 (方阵生成→居中裁剪至面板原生 8 µm 像素; 粗网格实测丢 ~41% 相位幅度)
- 修复: `slm_pib_runner._maybe_sim_patch` 的 `reset_system(seed=42)` 会丢弃已注入的干扰导致**静默无干扰运行** —— 现由 harness 包装该调用重新挂载干扰 (manifest 不再谎报有干扰)
- 修复: dynamic 模式曾保留全部相位屏 (~18 MB × 604 ≈ 11 GB) —— 改为仅累积标量
- 修复: 报告生成器 config 派生 tag 消除 `run0`/`run1` 图文件名冲突; `disturbance.json` 不再遮蔽 runner 自身 sidecar

### v0.13.0 (2026-09-29)
- **物理-optical 数据类模型** (`ao_shaping.model`): 新增 `model/` 包，包含 `PhaseMap`, `AmplitudeMap`, `ComplexField` 和 `FieldMetadata` 数据类 —— 封装 2-D 相位/振幅/复电场数组 + 像素尺寸 + 元数据(timestamp/source/wavelength)，并通过 `ao_shaping.drivers.sim._oopao_compat` 兼容层提供 OOPAO `Source` 转换 (`from_oopao_source`/`to_oopao_source`/`to_opd`)。71 个测试用例 (69 passed / 4 OOPAO 条件跳过)
- 修复: `scipy.ndimage.unwrap_phase` → 项目自己的 `ao_shaping.utils.wavefront.phase_unwrap.unwrap_phase`
- 修复: `ComplexField.field` 属性名冲突 → 使用 `dc_field` 别名

### v0.12.0 (2026-09-17)
- **共享扫描分析助手** (`tools/slm/slm_scan_analysis.py`): 纯 numpy 提取 `outlier_mask` (Z-score 异常点剔除)、`group_raw_scan` (灰度扫描分批求均值/标准差)、`analyze_linearity` (线性度指标)、`LINEARITY_AMPS`、`latest_match` 等 7 个公共符号; `zernike_matrix_runner` 改用 `outlier_mask` 剔除伪影点; 报告生成脚本 (`generate_zernike_response_matrix_report.py` / `generate_zernike_linearity_report.py`) 委托同一助手, 消除 `calibration.py`/`slm_lut_runner` 中的复刻逻辑
- **共享相机/相位工具** (`utils/image/hardware_utils.py`, `utils/slm_phase.py`): `open_camera()` 统一相机工厂 + `flat_gray`/`capture_frame` 等; 所有相机打开调用点 (slm_lut_runner, phase_capture, slm_diagnose, micro_dm_image_collect) 统一走 `hardware_utils.open_camera`, 消除 `slm_camera.py` 中间层 (注: `slm_camera.py` 模块随后已删除, 相机打开功能统一收敛至 `utils.image.hardware_utils.open_camera`；该模块原先还有一个 `utils/hardware_utils.py` 别名 shim，已于 2026-10-03 删除)
- **slm_slot 助手并入 Santec 驱动**: `utils/slm_slot.py` 删除, `SLOT_MIN`/`SLOT_MAX`、`SlotRotator`、`choose_slot`、`read_current_slot`、`apply_lut_remap` 移至驱动内部 (经 `santec/__init__.py` re-export 保持公共面)
- **calibration.py 拆分**: 离散几何标定 (`SLMCCDCalibrator`, 现行主流程) 与 LUT 标定 (`SLMLUTCalibrator`, `DeprecationWarning` 废弃) 分离; LUT canonical 路径收敛到 `slm_lut_runner` + `utils/slm_lut` → `Santec.load_lut`
- **tools/slm 迁移到 raw-grayscale 契约**: 扫描分析/校准工具统一走 uint16 直接灰度 (不经弧度转换, 2π=993 周期), 消除 `PatternHelper` 遗留 min-max 归一化
- **回归锚点测试 + 硬件自检 pytest 包装** (`a715937`): 纯逻辑回归锚点 (scan/lut/zernike/calibration/cartographer) + `AO_RUN_HARDWARE` 环境变量门控硬件自检用例 (默认 skip, 避免 CI 挂起)
- **文档迁移**: 根目录报告文档 (`beam_shaping_benchmark_metrics.md`, `issues_report.md`) 与 `performance_comparison` 输出迁移至 `docs/`; 重新生成硬件测试报告 (wfs/miicam/slm-200/micro-dm); `scripts/README.md` 补充报告生成架构文档
- **README 同步**: 补全缺失 CLI 文档 (`rms-zernike`, `ga-zernike`, `combined`, `spgd-square`), 修正项目结构树 (`ml/` 独立包位置、新增共享模块), 修复 `combined_runner` 废弃标注矛盾与脚本编号重复

### v0.11.0 (2026-09-16)
- **SLM 相位生成 raw-only 契约**: 所有 SLM 相位生成函数只产生 **raw 未包裹弧度**，不再自行 `mod 2π`——唯一 wrap 点在驱动 `Santec.create_phase_from_array()` 的弧度→灰度转换 (`santec/driver.py` L1382)。涉及 `optimizer/wfless/slm_square_shaping.py: _freeform_phase_radians` 移除末尾 `np.mod`、`_params_to_gray` 与 `slm_zernike_pib._zernike_to_phase` docstring 同步为 raw-only (`_zernike_phase_radians` 保留 wrapped 输出仅作 test-only 参考实现)
- **相位→灰度统一入口**: `utils/slm/phase_display.phase_to_slm_grayscale(phase, max_grayscale=None, slm=None)` — 传入已打开 SLM 时委托 `slm.create_phase_from_array()`（驱动统一管线：弧度→灰度 + 波前矫正 + LUT + 平移, 2π 灰度取设备波长相关 `_max_gray`）；`slm=None`（纯模拟/离线保存/单测）回退内置纯数学转换 (wrap→scale→clip→uint16, 默认 1023)。`gs_hologram_runner` / `diff_beam_runner` 保存路径迁移为 `phase_to_slm_grayscale(phase, slm=slm)`
- **测试修复**: `test_rms_zernike_runner` / `test_rms_by_zernike` 旧函数名 `optimizer_rms` → `optimizer_rms_slm`（对 `rms_by_zernike.py` 既有重命名的同步，5 个预存 ImportError 修复）；`test_gray_csv` roundtrip 断言改为 **mod-2π 相位等价**（`slm._max_gray` 反向换算 roundtrip 相位）；全套 SLM 相关测试通过：drivers/slm+runners 185 passed / wfless+gui/slm 136 passed（各 1 个硬件 skip）/ optimizer-wf 99 passed

### v0.10.0 (2026-09-13)
- **代码整合 (runners/utils 去重)**: `gs_square_runner`/`diff_beam_runner` 复用的质量指标、SLM 相位下发/槽轮换、超时看门狗、自动曝光、帧记录等辅助逻辑统一迁入 `utils/image/beam_metrics.py`、`utils/slm/phase_display.py`、`utils/image/hardware_utils.py` (原 `algorithm/beam_shaping_utils` 保留为兼容 re-export 层)
- **共享相机工厂**: 新增 `utils/image/hardware_utils.open_camera(camera_type, cam_id, exposure_ms, bit_depth)`，消除 `gs_square_runner`/`diff_shaping_runner` 中字节级重复的 daheng/miicam 初始化代码 (驱动延迟导入，保持 utils 叶子层约束)
- **wfless 内部去重**: `slm_zernike_pib` 的 `_zernike_indices` 改为复用 `slm_square_shaping` 同源实现 (字节级一致性校验通过)
- **回归锚点测试**: 新增 4 个 TDD 锚点测试文件 (`tests/ao_shaping/utils/test_{beam_metrics,phase_display,hardware_utils,targets}.py`, 共 153 例)，锁定全部迁移函数行为；`record_frame` 新增 `include_spot` 参数记录真实 0 级光斑 (argmax) 位置
- **代码评审修复**: ruff 清理、异常类型修正 (如 `ConnectionRefusedError`)、`SimDM` 补齐 `open/close` 接口并对齐 DM registry API；全套测试 1506 passed / 9 failed (仅限 Windows-only WFS/DM SDK 环境绑定用例) / 273 skipped

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
- 重构相机驱动 (DahengCamera, MIICamera)
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
