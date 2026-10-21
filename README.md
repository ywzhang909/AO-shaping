# AO-Shaping 自适应光学整形系统

> 本 README 已于 2026-10-13 精简为**索引**：详细指南拆分到 `docs/`，版本历史在
> [CHANGELOG.md](CHANGELOG.md)，实验结论在 [`report/`](report/)（结论的唯一事实源）。

## 项目简介

AO-Shaping 是一个基于强化学习的自适应光学 (AO) 系统，用于波前校正和光束整形。项目集成了多种优化算法，包括基于波前传感器 (WFS) 的优化和无波前优化方法，能够通过变形镜 (DM) 或空间光调制器 (SLM) 对光波前进行精确控制，实现高质量的光束输出。

**主要特性**

- **多优化算法**: 支持基于波前传感器的 RMS 优化和无波前的 PIB 优化 (SPGD/GA/PSO/CEM/Tabu/可微分)
- **强化学习集成**: 使用 SAC 算法进行波前优化
- **硬件支持**: Thorlabs WFS、NLight 变形镜、R50Power MicroDM、大恒相机、MIICAM、Santec SLM200、NI DAQ ADC
- **数字孪生**: 2f-Fourier SLM-PIB 仿真系统 (湍流 + 热晕扰动、相机噪声模型)，无硬件即可回归
- **ML 训练**: U-Net+GAN 相位预测、Zernike 前向/逆向模型、FourierGSNet
- **实验跟踪**: WandB / SwanLab；**数据处理**: Dask

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

## 文档索引

| 文档 | 内容 |
|------|------|
| [docs/architecture.md](docs/architecture.md) | 项目结构（详细）、物理-optical 数据类模型 (`ao_shaping.model`)、2f-Fourier 实验系统唯一入口约定 |
| [docs/drivers_architecture.md](docs/drivers_architecture.md) | 驱动分层架构、设备注册表、惰性加载契约、2026-10 更新 |
| [docs/hardware_drivers.md](docs/hardware_drivers.md) | 硬件驱动详情：WFS / DM / 相机 / SLM / 数据采集卡 |
| [docs/simulation.md](docs/simulation.md) | 数字孪生仿真：2f-Fourier 实验系统、DM→相位耦合、噪声模型红线、湍流+热晕扰动 |
| [docs/zernike_usage.md](docs/zernike_usage.md) | Zernike 两层入口、最小示例、单位/约定红线 |
| [docs/ml_training.md](docs/ml_training.md) | ML 训练入口与报告索引、指标口径红线 (R²/corr, 不用 PSNR/SSIM) |
| [docs/gui.md](docs/gui.md) | Streamlit GUI 入口一览、multi_slm_controller 与 GS 方形整形 |
| [docs/dev/coding_conventions.md](docs/dev/coding_conventions.md) | 编码规范、测试规范、贡献流程 |
| [docs/dev/call_graph_wiki/index.md](docs/dev/call_graph_wiki/index.md) | main.py 函数调用图开发手册：20 命令逐命令 + 骨架函数逐函数文档（AST 静态分析，脚本在 `scripts/call_graph/` 可重跑） |
| [docs/slm/pre_run_characterization.md](docs/slm/pre_run_characterization.md) | 实验系统表征流程、验收阈值、四条实验系统铁律 |
| [docs/slm/slm_gui_manual.md](docs/slm/slm_gui_manual.md) | SLM GUI 完整操作手册 |
| [CHANGELOG.md](CHANGELOG.md) | 版本历史 (v0.1.0 → v0.19.0) |
| [report/](report/) | 实验报告 (结论/测量/基准)，每份带生成脚本溯源块 |

各设备域 CLI 命令文档在 `docs/{dm,slm,wfs,micro_drive,tools}/` 下（见下方命令总览表）。

## CLI 命令总览

统一入口 `main.py`（Click-based），目前注册 **20 个命令**；所有 runner 位于
`src/ao_shaping/runners/`，可绕过 main.py 以 `python -m` 直接运行。

```bash
python src/ao_shaping/main.py [OPTIONS] COMMAND [ARGS]...
```

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
| `slm-gsnet` | [docs/slm/slm_gsnet.md](docs/slm/slm_gsnet.md) | SLM全频随机相位方形整形 (SPGD/启发式) |
| `slm-pib` | [docs/slm/slm_pib.md](docs/slm/slm_pib.md) | SLM Zernike PIB优化 (`spgd` / `heuristic`) |
| `slm-gs-refine` | [docs/slm/slm_gs_refine.md](docs/slm/slm_gs_refine.md) | GS 预矫正 + 全频随机相位 SPGD 整形 |
| `slm-model-in-loop` | [docs/slm/slm_model_in_loop.md](docs/slm/slm_model_in_loop.md) | 正向模型闭环校正 + 目标光斑相位合成 (反复迭代) |

> **注意**: `gs`、`gs-square`、`diff-shaping`、`diff-beam` 命令已从 CLI 中移除 (运行器文件不再存在)。其功能已并入 `slm-gsnet` (全频随机相位整形)、`algorithm/signal_processing/gerchberg_saxton.py` (GS算法) 和 `algorithm/signal_processing/differentiable_shaping.py` (可微分整形)。

**全局选项**

- `--dir` / `-d`: 指定数据保存根目录 (默认: data)
- `--debug`: 启用调试模式 (保存 pkl/json/图片等调试产物; 等价于 `DEBUG=1` 环境变量; 支持 `1`/`true`/`yes`，不区分大小写)

**直接模块入口** (与上表一一对应)

```bash
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

### SLM 实验系统探针 (独立运行, 不注册为 CLI)

以下探针**不注册为 CLI 命令**，用 `python -m ao_shaping.tools.<子包>.<名字>` 直接运行，
且都支持 `--no-hw` (只打印采集计划、不碰硬件)。它们固化了 2026-09-30 在大恒 +
Santec SLM-200 实验系统上踩出来的实验系统常数与测量陷阱，**不要用第一性原理重新推导几何**。

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
> **四条实验系统铁律** (暗帧不可裸 `argmax` / SLM 保留上次图案 / 固定 `memory_number`
> 是固件 no-op / 液晶要等"稳定"而非"够久") 见
> [docs/slm/pre_run_characterization.md](docs/slm/pre_run_characterization.md)。

共用内核 (无 CLI, 纯 numpy, 设备由参数传入): `tools/kernels/bench_kernels.py`
(帧数学 / 光斑测量 / 面板相位放置 / `display_and_average` / `capture_settled`) 与
`tools/kernels/sweep_analysis.py` (扫描线性 / `delta` 探索 / SNR)。Zernike 共用量在
`tools/zernike/common.py`。`scripts/slm/model_in_loop_hw_runbook.py --stage sweep` 的采集内核
收敛自 `tools/zernike/sweep_probe.py`; 面板与几何常数来源见
[`report/slm/model_in_loop_hw_runbook/model_in_loop_bench_calibration.md`](./report/slm/model_in_loop_hw_runbook/model_in_loop_bench_calibration.md)。

### Micro-DM 数据目录

`micro_dm_image_collect` 采集的图像按 `IP-通道号` 落盘, 磁盘布局 (`data/md_test/`)
承重于 `scripts/dm/` 侧的 Micro-DM diff 分析流水线, **改动落盘命名/目录前先看
`scripts/dm/md_img_*.py`**。完整布局 (原始灰度图 / 100V 差分图 / GIF) 与
网格坐标 → IP 组 → 图像文件的映射表见
`data/md_test/README.md` (随采集数据落盘, 不在仓库中);
工具说明见 [docs/tools/micro_dm_image_collect.md](docs/tools/micro_dm_image_collect.md)。

## GUI 界面 (Streamlit)

项目提供了基于 Streamlit 的图形界面，**代码**按设备域分包于 `src/ao_shaping/gui/` 下，各 UI 独立运行；对应的**操作手册与需求规格说明书**在 `docs/GUI/<子包>/<页面名>/` 下（与 py 文件同名的包目录）：

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

# CCD 实时图像分析 (大恒/MIICAM + 椭圆拟合 + 截面)
streamlit run src/ao_shaping/gui/ccd/ccd_analyzer.py

# Zernike 调试查看器 (响应矩阵/系数可视化)
streamlit run src/ao_shaping/gui/zernike/zernike_debug_viewer.py

# Micro-DM 图像分析 (diff 图/网格)
streamlit run src/ao_shaping/gui/dm/md_img_analysis_ui.py
```

各页面的用户手册与需求规格说明书见 [`docs/gui.md`](docs/gui.md) 的文档索引。其中 `multi_slm_controller.py` 是 Santec SLM-200 空间光调制器的 Streamlit 前端页面，支持最多两台 SLM、相位图案生成/下发、CSV 加载与导出、周期切换后台任务，以及闭环远场光束整形（GS+SPGD 与正向模型闭环）。完整操作手册与需求规格说明书见 [`docs/GUI/slm/multi_slm_controller/multi_slm_controller_user_manual.md`](docs/GUI/slm/multi_slm_controller/multi_slm_controller_user_manual.md) 与 [`docs/GUI/slm/multi_slm_controller/multi_slm_controller_srs.md`](docs/GUI/slm/multi_slm_controller/multi_slm_controller_srs.md)。

## 项目结构（简版）

```
AO-shaping/
├── src/
│   ├── ao_shaping/     # 主程序包: runners/ algorithm/ model/ drivers/
│   │                   #   optimizer/ utils/ tools/ display/ gui/
│   ├── ml/             # 机器学习独立包: gsnet/ phase/ hwdataset/ zernike/
│   ├── calculators/    # Cython扩展 (standalone)
│   └── optical_ui/     # [DEPRECATED] Empty package
├── tests/ao_shaping/   # Tests (镜像 src 结构)
├── scripts/            # 实用脚本, 按处理对象分组 (含报告生成 generate_*_report.py)
│                     # slm/ zernike_bench/ dm/ wfs/ = 设备域; ml/ rl/ oopao/ build/ = 数据与用途
├── docs/               # 设备说明与指南文档
├── report/             # 实验报告 (结论的唯一事实源)
├── libs/               # 第三方SDK二进制 (gxipy, Drv_UDPST) + OOPAO 子模块
└── AGENTS.md           # 开发指南
```

完整目录树见 [docs/architecture.md](docs/architecture.md#项目结构详细)。

## 近期更新

版本历史已迁至 [CHANGELOG.md](CHANGELOG.md)（当前最新 **v0.19.0**, 2026-10-19）。
最近一轮要点：正向模型在环（model-in-loop）闭环成型、GS→Zernike 阶数扫描立项、
scripts/ 按处理对象重组、ruff 硬门禁（2026-10-21 恢复入库，见 CHANGELOG v0.19.0 节）。
