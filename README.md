# AO-Shaping 自适应光学整形系统

> **从数字孪生到实验系统：一套可复现的自适应光学（AO）波前校正与光束整形代码库。**
>
> 第一次接触 → [项目简介](#项目简介) → [快速开始](#快速开始)；
> 要改代码 → [二次开发与使用](#二次开发与使用) 与
> [`docs/dev/extension_guide.md`](docs/dev/extension_guide.md)；
> 查命令 → [`docs/user/cli-reference.md`](docs/user/cli-reference.md)；
> 查结论 → [`report/`](report/)（实验结论的唯一事实源）。

## 项目简介

AO-Shaping 解决一个具体问题：**让激光远场变成想要的样子**。它把
“测量/不测量波前 → 计算相位或电压 → 驱动变形镜（DM）或空间光调制器（SLM）
→ 用相机（CCD）或波前传感器（WFS）读回 → 更新控制量”的闭环，做成可配置、
可复现、并且能在无硬件条件下回归的工程系统。

项目同时覆盖三条技术路线：

1. **基于波前传感器（WFS）的闭环**：以 DM 电压或 SLM Zernike 相位为控制量，
   用 RMS / 波前残差类指标驱动优化。入口：`wf`、`pipeline`、`zernike-matrix`、
   `closed-loop`、`rms-zernike`、`ga-zernike`、`greedy-zernike`。
2. **无波前传感器（WFS-less）的闭环**：只看 CCD 远场，用 PIB（Power-in-Bucket）、
   方形光斑均匀度、环围能量等目标驱动 SPGD、启发式搜索、可微分优化或
   “物理预测模型在环”合成。入口：`pib`、`combined`、`slm-pib`、`slm-pib square`、
   `slm-gsnet`、`slm-gs-refine`、`slm-model-in-loop`、`slm-transfer-shapes`。
3. **数字孪生与学习型模型**：2f-Fourier 实验系统的仿真后端，带湍流、
   热晕扰动与相机噪声模型；`src/ml/` 提供相位预测、Zernike 正向/逆向模型与
   GSNet 训练。没有硬件时，同一套 runner 用 `--cam-type sim` 即可回归。

**核心能力**

| 能力 | 从哪里开始 | 说明 |
|---|---|---|
| WFS 闭环校正 | `wf` / `pipeline` / `zernike-matrix` | DM 电压或 SLM 相位；响应矩阵标定 + 闭环 |
| 无波前整形 | `pib` / `slm-pib` / `slm-model-in-loop` | CCD 反馈；PIB、方斑、全频随机相位、物理预测模型在环 |
| 器件标定与自检 | `dm-matrix` / `hadamard-matrix` / `slm-lut` / `slm-diagnose` | 响应矩阵、灰度-相位 LUT、逐级硬件自检 |
| 数字孪生回归 | `--cam-type sim`（部分命令支持 `--slm-type sim`） | 无硬件跑完整闭环；可选 OOPAO 后端 |
| 机器学习 | `src/ml/`、`ao-train` | U-Net+GAN、Zernike 前向/逆向、GSNet、硬件语料 Dataset |
| 实验记录 | `data/`、`Recorder`、[`report/`](report/) | 逐轮历史、调试产物、带溯源块的实验报告 |
| 图形界面 | `src/ao_shaping/gui/` | 按设备域拆分的 Streamlit 页面 |

**这个仓库适合谁**

- 需要把 AO 闭环接到自己的 DM / SLM / 相机光路，并希望算法与设备解耦的人；
- 想在同一实验系统上比较优化算法、目标函数或波前传感方案的人；
- 没有硬件、但需要开发、回归或教学演示的人；
- 想复用 Zernike 数学、响应矩阵、SPGD/启发式优化器等组件的人。

**边界与前提**

- 它不是光学设计软件（不做 Zemax / CODE V 式的镜头设计），而是**控制与优化层**。
- 实验系统常数、标定文件、安全阈值必须按当前光路重新测量；上硬件前先跑
  [表征探针](docs/slm/pre_run_characterization.md)。
- 强化学习（`optimizer/rl/`）是其中一个模块，不是全系统的范式。
- 仓库当前聚焦单一实验系统（2f-Fourier SLM / DM + WFS + CCD）的工程实现，
  换光路时需要重新标定几何与噪声地板。

## 快速开始

### 1. 环境要求

- Python **3.12+**
- Linux / Windows / macOS
- [uv](https://docs.astral.sh/uv/)（推荐）
- Docker（仅测试需要；不用容器也可以跑 CLI 与仿真）
- CUDA（可选，用于 `src/ml/` 或可微分整形）

### 2. 获取代码与安装

```bash
git clone --recurse-submodules https://github.com/ywzhang909/AO-shaping.git
cd AO-Shaping

uv venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

```bash
# 只跑 CLI 与数字孪生：基础依赖
uv sync

# 开发 / 测试：含 pytest、ruff、ml、rl 等依赖组
uv sync --all-groups
```

> **PyTorch 是 extras，不在 `ml` 依赖组里。** 需要训练或可微分整形时，按显卡算力
> **只选一个** torch extra：
>
> ```bash
> uv sync --all-groups --extra torch-cpu     # 无 GPU
> uv sync --all-groups --extra torch-cu128   # RTX 50xx / sm_120
> uv sync --all-groups --extra torch-cu126   # GTX 10xx / sm_61（cu126 是最后支持 Pascal 的 12.x）
> ```
>
> 完整算力对照与索引配置见 [`pyproject.toml`](pyproject.toml) 的
> “PyTorch build selection” 注释。

OOPAO 以 `libs/OOPAO` 的 editable 路径依赖安装。克隆时带了
`--recurse-submodules` 即可；若仓库已存在，先补子模块：

```bash
git submodule update --init libs/OOPAO
uv sync
```

### 3. 验证安装

```bash
python src/ao_shaping/main.py --help
```

无硬件跑一条完整的 SPGD 闭环（数字孪生，约 10 秒）：

```bash
python src/ao_shaping/main.py slm-pib spgd --cam-type sim -e 3
```

预期日志以 `SLM PIB done` 结束，不打开任何真实设备。

### 4. Python API 最小示例

生成一张 Zernike 相位并转成 SLM 灰度图（纯离线，不碰硬件）：

```python
import numpy as np

from ao_shaping.utils.slm.phase_display import phase_to_slm_grayscale
from ao_shaping.utils.wavefront.zernike_utils import generate_zernike_phase

# Noll 4 = defocus，Noll 5 = astigmatism；单位是弧度（raw，未包裹）
phase = generate_zernike_phase({4: 0.3, 5: -0.15}, resolution=(256, 256), n_max=4)
phase = np.nan_to_num(phase, nan=0.0)   # 圆形孔径外是 NaN

gray = phase_to_slm_grayscale(phase)    # 离线纯数学转换
print(gray.dtype, gray.shape)           # uint16 (256, 256)
```

连接实验系统时把已打开的 `slm` 传给 `phase_to_slm_grayscale(phase, slm=slm)`，
由驱动统一完成 2π 包裹、波前校正与 LUT。

### 5. 跑测试

测试**必须在容器里跑**：数值库会按可见核数开线程池，不受限的 `pytest`
会把宿主机跑满；`conftest.py` 会直接拒绝宿主机运行。

```bash
# GPU 机器
docker build -f docker/Dockerfile.test -t ao-shaping-test:gpu .
./run_pytest.sh                                   # 默认子集
./run_pytest.sh tests/ao_shaping/algorithm -- -q     # 指定子树

# 无 GPU / CI
docker build -f docker/Dockerfile.test.cpu -t ao-shaping-test:cpu .
AO_IMAGE=ao-shaping-test:cpu ./scripts/build/run_pytest_ci.sh
```

默认排除硬件、`ml` 与个别会挂起的 GUI 测试（见 `pyproject.toml` 的 `addopts`）；
显式指定路径仍会正常收集。硬件测试使用 `@pytest.mark.hardware` 标记。

## 二次开发与使用

先判断你的改动属于哪一类，再跳到对应入口。**详细的扩展契约、最小模板与提交
检查单见 [`docs/dev/extension_guide.md`](docs/dev/extension_guide.md)。**

| 你的目标 | 改哪里 | 先读什么 |
|---|---|---|
| 用现成命令做实验 | `src/ao_shaping/main.py` | [`docs/user/cli-reference.md`](docs/user/cli-reference.md)、[`docs/slm/pre_run_characterization.md`](docs/slm/pre_run_characterization.md) |
| 新增/替换一台设备 | `src/ao_shaping/drivers/<domain>/` | `drivers/AGENTS.md`、[`docs/drivers_architecture.md`](docs/drivers_architecture.md) |
| 新增一个纯数学优化器 | `src/ao_shaping/algorithm/<category>/` | [`src/ao_shaping/algorithm/README.md`](src/ao_shaping/algorithm/README.md) |
| 新增一种闭环策略/目标函数 | `src/ao_shaping/optimizer/{wf,wfless}/` | 对应模块 docstring + `Recorder` 约定 |
| 新增一条 CLI 命令 | `src/ao_shaping/runners/<domain>/` + `main.py` | [`docs/dev/extension_guide.md`](docs/dev/extension_guide.md) §4.4 |
| 训练新模型 / 换数据集 | `src/ml/` | [`docs/ml_training.md`](docs/ml_training.md) |
| 新增 GUI 页面 | `src/ao_shaping/gui/<domain>/` | [`docs/gui.md`](docs/gui.md) |

**三条不可协商的契约**

1. **惰性加载**：`import ao_shaping` 及其子包不得加载原生 SDK、不得做磁盘 I/O；
   SDK 延迟到 `open()`，通过 `Device._load_sdk()` / `_ensure_sdk()` 解析。
2. **仿真先行**：算法与优化器先用仿真/合成目标验证，测试不得 import 硬件 SDK；
   需要实验系统的测试加 `@pytest.mark.hardware`。
3. **单一事实源**：Zernike 数学、目标方向、复合质量分、报告溯源映射都只有一份
   实现；扩展时调用 canonical 入口，不要复制第二份。

**新增一条命令的最短路径**

1. 在 `runners/<domain>/` 写 runner，参数用
   `Annotated[T, option("-x", "--xxx")]` + `@with_params(...)` 声明；
2. 在 `main.py` 用 `cli.add_command(run, name="your-command")` 注册；
3. 更新 CLI `--help` 快照（`AO_UPDATE_HELP_GOLDEN=1` 与
   `AO_CLI_CONTRACT_UPDATE=1`，见扩展指南）；
4. 在 `docs/<domain>/` 写命令页并登记到 [`docs/INDEX.md`](docs/INDEX.md)。

## 架构速览

```
CLI (src/ao_shaping/main.py)
  └── runners/          命令注册、设备 open/close、结果落盘
        ├── optimizer/  闭环策略、目标函数编排、Recorder 历史
        │     ├── algorithm/  纯数学优化器（无硬件知识）
        │     └── drivers/    设备 I/O（惰性加载 SDK）
        └── utils/ + model/ 叶子工具与物理量数据类
```

依赖方向**只能向下**：`utils/` 是叶子层，不得反向 import `algorithm/`、
`drivers/`、`optimizer/`；硬件包不得 import 仿真包。`src/ml/` 是独立可选包，
`scripts/` 只做离线分析/报告，不进入运行时依赖。

详细目录树与物理数据类见 [`docs/architecture.md`](docs/architecture.md)。

## 文档索引

| 文档 | 内容 |
|------|------|
| [`docs/INDEX.md`](docs/INDEX.md) | **文档总索引**（按主题分域，全部文档的入口） |
| [`docs/user/cli-reference.md`](docs/user/cli-reference.md) | CLI 正式参考：全部 flag、取值域、默认值 |
| [`docs/dev/extension_guide.md`](docs/dev/extension_guide.md) | **二次开发指南**：分层、扩展点、最小模板、测试与文档门禁 |
| [`docs/dev/coding_conventions.md`](docs/dev/coding_conventions.md) | 编码规范、测试规范、贡献流程 |
| [`docs/dev/call_graph_wiki/index.md`](docs/dev/call_graph_wiki/index.md) | main.py 函数调用图开发手册：逐命令 + 逐函数文档（AST 静态分析，可重跑） |
| [`docs/architecture.md`](docs/architecture.md) | 项目结构（详细）、物理-optical 数据类模型 (`ao_shaping.model`)、2f-Fourier 实验系统唯一入口约定 |
| [`docs/drivers_architecture.md`](docs/drivers_architecture.md) | 驱动分层架构、设备注册表、惰性加载契约 |
| [`docs/hardware_drivers.md`](docs/hardware_drivers.md) | 硬件驱动详情：WFS / DM / 相机 / SLM / 数据采集卡 |
| [`docs/simulation.md`](docs/simulation.md) | 数字孪生仿真：2f-Fourier 实验系统、DM→相位耦合、噪声模型红线、湍流+热晕扰动 |
| [`docs/zernike_usage.md`](docs/zernike_usage.md) | Zernike 两层入口、最小示例、单位/约定红线 |
| [`docs/ml_training.md`](docs/ml_training.md) | ML 训练入口与报告索引、指标口径红线（R²/corr，不用 PSNR/SSIM 选模型） |
| [`docs/gui.md`](docs/gui.md) | Streamlit GUI 入口一览、页面手册与 SRS 索引 |
| [`docs/slm/pre_run_characterization.md`](docs/slm/pre_run_characterization.md) | 实验系统表征流程、验收阈值、四条实验系统铁律 |
| [`docs/slm/slm_gui_manual.md`](docs/slm/slm_gui_manual.md) | SLM GUI 完整操作手册 |
| [`docs/terminology_glossary.md`](docs/terminology_glossary.md) | **术语规范表**（中文文档唯一权威） |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | 分支、提交信息、质量门禁、Issue / PR 流程 |
| [`CHANGELOG.md`](CHANGELOG.md) | 版本历史 |
| [`report/`](report/) | 实验报告（结论/测量/基准），每份带生成脚本溯源块 |
| [`LICENSE`](LICENSE) | MIT（占位，待最终确认） |

各设备域 CLI 命令文档在 `docs/{dm,slm,wfs,micro_drive,tools}/` 下（见下方命令总览表）。

## CLI 命令总览

统一入口 `main.py`（Click-based），目前注册 **20 个顶层命令**；所有 runner 位于
`src/ao_shaping/runners/`，可绕过 `main.py` 以 `python -m` 直接运行。

```bash
python src/ao_shaping/main.py [OPTIONS] COMMAND [ARGS]...
```

| 命令 | 文档 | 功能 |
|------|------|------|
| `wf` | [docs/dm/wf.md](docs/dm/wf.md) | 波前 RMS 优化（DM 电压 + WFS） |
| `pib` | [docs/dm/pib.md](docs/dm/pib.md) | PIB 优化（DM 电压 + CCD） |
| `pipeline` | [docs/dm/pipeline.md](docs/dm/pipeline.md) | 串行 WF RMS → PIB 流水线 |
| `zernike-matrix` | [docs/slm/zernike_matrix.md](docs/slm/zernike_matrix.md) | Zernike 响应矩阵校准 |
| `closed-loop` | [docs/wfs/closed_loop.md](docs/wfs/closed_loop.md) | 基于响应矩阵的闭环波前优化 |
| `rms-zernike` | [docs/slm/rms_zernike.md](docs/slm/rms_zernike.md) | SLM Zernike RMS 优化 |
| `ga-zernike` | [docs/wfs/ga_zernike.md](docs/wfs/ga_zernike.md) | 遗传算法 Zernike 优化 |
| `greedy-zernike` | [docs/wfs/greedy_zernike.md](docs/wfs/greedy_zernike.md) | 贪婪局部搜索 Zernike 优化 |
| `dm-matrix` | [docs/dm/dm_matrix.md](docs/dm/dm_matrix.md) | DM 响应矩阵标定 |
| `hadamard-matrix` | [docs/dm/hadamard_matrix.md](docs/dm/hadamard_matrix.md) | Hadamard 响应矩阵标定 |
| `alt-voltage` | [docs/micro_drive/alt_voltage.md](docs/micro_drive/alt_voltage.md) | 交替电压下发（R50Power + ADC） |
| `full-voltage` | [docs/micro_drive/full_voltage.md](docs/micro_drive/full_voltage.md) | 全量交替电压（AsyncMicroDM） |
| `combined` | [docs/dm/combined.md](docs/dm/combined.md) | AdaMOD + SPGD 混合 PIB（DM + CCD） |
| `slm-lut` | [docs/slm/slm_lut.md](docs/slm/slm_lut.md) | SLM 灰度→相位 LUT 校准 |
| `slm-diagnose` | [docs/slm/slm_diagnose.md](docs/slm/slm_diagnose.md) | SLM + 相机逐级硬件自检 |
| `slm-gsnet` | [docs/slm/slm_gsnet.md](docs/slm/slm_gsnet.md) | SLM 全频随机相位方形整形（SPGD / 启发式） |
| `slm-pib` | [docs/slm/slm_pib.md](docs/slm/slm_pib.md) | SLM Zernike PIB 优化（`spgd` / `heuristic` / `square`） |
| `slm-gs-refine` | [docs/slm/slm_gs_refine.md](docs/slm/slm_gs_refine.md) | GS 预矫正 + 全频随机相位 SPGD 细化 |
| `slm-model-in-loop` | [docs/slm/slm_model_in_loop.md](docs/slm/slm_model_in_loop.md) | 物理预测模型闭环校正 + 目标光斑相位合成（反复迭代） |
| `slm-transfer-shapes` | [docs/slm/slm_transfer_workflow.md](docs/slm/slm_transfer_workflow.md) | 物理预测模型多形状整形（model → shaping → replay → record 四阶段） |

> `slm-pib` 是 Click group：不带子命令等价于 `spgd`；`square` 是方形远场整形
> （文档页 [`docs/slm/spgd_square.md`](docs/slm/spgd_square.md)）。
> 早期文档中的 `spgd-square` 已不是独立顶层命令，请使用
> `python src/ao_shaping/main.py slm-pib square ...`。

> **注意**：`gs`、`gs-square`、`diff-shaping`、`diff-beam` 命令已从 CLI 中移除
> （运行器文件不再存在）。其功能已并入 `slm-gsnet`（全频随机相位整形）、
> `algorithm/signal_processing/gerchberg_saxton.py`（GS 算法）和
> `algorithm/signal_processing/differentiable_shaping.py`（可微分整形）。

**全局选项**

- `--dir` / `-d`：指定数据保存根目录（默认：`data`）
- `--debug`：启用调试模式（保存 pkl/json/图片等调试产物；等价于 `DEBUG=1`
  环境变量；支持 `1`/`true`/`yes`，不区分大小写）

**直接模块入口**（与上表一一对应）

```bash
python -m ao_shaping.runners.nlight_dm.wf_runner               # wf
python -m ao_shaping.runners.nlight_dm.axis_beam_runner       # pib
python -m ao_shaping.runners.nlight_dm.pipeline_runner        # pipeline
python -m ao_shaping.runners.slm.zernike_matrix_runner        # zernike-matrix
python -m ao_shaping.runners.slm.rms_zernike_runner           # rms-zernike
python -m ao_shaping.runners.zernike_search_runner            # ga-zernike / greedy-zernike
python -m ao_shaping.runners.matrix_runner                    # dm-matrix / hadamard-matrix
python -m ao_shaping.runners.nlight_dm.combined_runner        # combined
python -m ao_shaping.runners.micro_drive.voltage_runner       # alt-voltage / full-voltage
python -m ao_shaping.runners.slm.slm_shaping_runner spgd      # slm-pib spgd
python -m ao_shaping.runners.slm.slm_shaping_runner square    # slm-pib square
python -m ao_shaping.runners.slm.gsnet_runner                 # slm-gsnet
python -m ao_shaping.runners.slm.gs_refine_runner             # slm-gs-refine
python -m ao_shaping.runners.slm.model_in_loop_runner         # slm-model-in-loop
python -m ao_shaping.runners.slm.transfer_shapes_runner      # slm-transfer-shapes
python -m ao_shaping.tools.slm.calibration.lut_runner         # slm-lut
python -m ao_shaping.tools.slm.panel.diagnose                 # slm-diagnose
python -m ao_shaping.tools.micro_dm.micro_dm_image_collect    # Micro-DM 逐单元采集
```

### SLM 实验系统探针（独立运行，不注册为 CLI）

以下探针**不注册为 CLI 命令**，用 `python -m ao_shaping.tools.<子包>.<名字>` 直接运行，
且都支持 `--no-hw`（只打印采集计划、不碰硬件）。它们固化了 2026-09-30 在大恒 +
Santec SLM-200 实验系统上踩出来的实验系统常数与测量陷阱，**不要用第一性原理重新推导几何**。

| 探针（`python -m` 路径） | 文档 | 用途 |
|---|---|---|
| `ao_shaping.tools.slm.panel.tilt_probe` | [docs/tools/slm_tilt_probe.md](docs/tools/slm_tilt_probe.md) | **判定面板是否真的在调制**（倾斜斜坡）。比光栅可靠：光斑位移只取决于斜坡周期，与衍射效率无关 |
| `ao_shaping.tools.slm.panel.panel_locate` | [docs/tools/slm_panel_locate.md](docs/tools/slm_panel_locate.md) | 面板坐标上定位光斑中心。相机 0 阶**不是**面板坐标（两轴互换 90°，尺度差 >10 倍） |
| `ao_shaping.tools.slm.panel.beam_extent` | [docs/tools/slm_beam_extent.md](docs/tools/slm_beam_extent.md) | 半平面随机相位边界扫描测光斑中心/半径。实测 r=450 px @ (960,600) |
| `ao_shaping.tools.slm.panel.phase_resolution` | [docs/tools/slm_phase_resolution.md](docs/tools/slm_phase_resolution.md) | 比较逐像素随机相位与光滑 Zernike 相位，判定面板**等效相位分辨率** |
| `ao_shaping.tools.ccd.exposure_check` | [docs/tools/slm_exposure_check.md](docs/tools/slm_exposure_check.md) | 相机自动曝光状态 + 固定设置下漂移（区分“相机漂移”与“SLM 保留上次图案”） |
| `ao_shaping.tools.zernike.sweep_probe` | [docs/tools/slm_zernike_sweep_probe.md](docs/tools/slm_zernike_sweep_probe.md) | **光滑 Zernike 扫描探针**：ramp + tilt + defocus + astig + coma + spherical 共 42 点，逐点稳定判据读帧，落盘 npz + Recorder（含相位与 CCD 帧） |
| `ao_shaping.tools.ccd.drift_probe` | [docs/tools/slm_drift_probe.md](docs/tools/slm_drift_probe.md) | **平场漂移 + 曝光阶梯线性**。用区域范数判漂移（**不用峰值** — 实测同设置两次运行峰值读到 100 与 23，而 box sum 稳到 0.2%） |
| `ao_shaping.tools.ccd.floor_probe` | [docs/tools/slm_floor_probe.md](docs/tools/slm_floor_probe.md) | **测量本底 + 稳定时间 + SNR-vs-K**。回答噪声是读噪声（`noise_limited`）还是漂移（`drift_limited`） |
| `ao_shaping.tools.zernike.abba_probe` | [docs/tools/slm_abba_probe.md](docs/tools/slm_abba_probe.md) | **稠密随机相位是否可分辨**。ABBA（`+ - - +`）消一阶漂移并先量本底。`verdict=unusable` 时不要去测转移矩阵 |

> 🔑 **启动 GS / GSNet / SPGD runner 之前先跑表征探针**：
> `ccd.drift_probe` → `ccd.floor_probe` → `zernike.abba_probe`。
> 完整流程、验收阈值、**危险默认值清单**、**曝光 `0` 的真实语义（§4.1）** 与
> **四条实验系统铁律**（暗帧不可裸 `argmax` / SLM 保留上次图案 / 固定 `memory_number`
> 是固件 no-op / 液晶要等“稳定”而非“够久”）见
> [docs/slm/pre_run_characterization.md](docs/slm/pre_run_characterization.md)。

共用内核（无 CLI，纯 numpy，设备由参数传入）：`tools/kernels/bench_kernels.py`
（帧数学 / 光斑测量 / 面板相位放置 / `display_and_average` / `capture_settled`）与
`tools/kernels/sweep_analysis.py`（扫描线性 / `delta` 探索 / SNR）。Zernike 共用量在
`tools/zernike/common.py`。`scripts/slm/model_in_loop_hw_runbook.py --stage sweep` 的采集内核
收敛自 `tools/zernike/sweep_probe.py`；面板与几何常数来源见
[`report/slm/model_in_loop_hw_runbook/model_in_loop_bench_calibration.md`](report/slm/model_in_loop_hw_runbook/model_in_loop_bench_calibration.md)。

### Micro-DM 数据目录

`micro_dm_image_collect` 采集的图像按 `IP-通道号` 落盘，磁盘布局（`data/md_test/`）
承重于 `scripts/dm/` 侧的 Micro-DM diff 分析流水线，**改动落盘命名/目录前先看
`scripts/dm/md_img_*.py`**。完整布局（原始灰度图 / 100V 差分图 / GIF）与
网格坐标 → IP 组 → 图像文件的映射表见
`data/md_test/README.md`（随采集数据落盘，不在仓库中）；
工具说明见 [docs/tools/micro_dm_image_collect.md](docs/tools/micro_dm_image_collect.md)。

## GUI 界面（Streamlit）

项目提供了基于 Streamlit 的图形界面，**代码**按设备域分包于
`src/ao_shaping/gui/` 下，各 UI 独立运行；对应的**操作手册与需求规格说明书**在
`docs/GUI/<子包>/<页面名>/` 下（与 py 文件同名的包目录）：

```bash
# R50 控制器（单单元/单控制器/组/联合控制）
streamlit run src/ao_shaping/gui/r50/r50_controller_ui.py

# Micro-DM 驱动控制
streamlit run src/ao_shaping/gui/dm/micro_dm_ui.py

# Zernike 响应矩阵校准
streamlit run src/ao_shaping/gui/zernike/zernike_response_matrix_ui.py

# SLM 校准
streamlit run src/ao_shaping/gui/slm/slm_calibration_ui.py

# 多 SLM 控制器（相位图案生成/下发，含闭环整形：GS+SPGD / 物理预测模型）
streamlit run src/ao_shaping/gui/slm/multi_slm_controller.py

# 1300 陶瓷单元查看器（网格浏览 + 图片标注）
streamlit run src/ao_shaping/gui/r50/ceramic_viewer.py

# CCD 实时图像分析（大恒/MIICAM + 椭圆拟合 + 截面）
streamlit run src/ao_shaping/gui/ccd/ccd_analyzer.py

# Zernike 调试查看器（响应矩阵/系数可视化）
streamlit run src/ao_shaping/gui/zernike/zernike_debug_viewer.py

# Micro-DM 图像分析（diff 图/网格）
streamlit run src/ao_shaping/gui/dm/md_img_analysis_ui.py
```

各页面的用户手册与需求规格说明书见 [`docs/gui.md`](docs/gui.md) 的文档索引。
其中 `multi_slm_controller.py` 是 Santec SLM-200 的 Streamlit 前端页面，支持最多
两台 SLM、相位图案生成/下发、CSV 加载与导出、周期切换后台任务，以及闭环远场
光束整形（GS+SPGD 与物理预测模型闭环）。完整操作手册与需求规格说明书见
[`docs/GUI/slm/multi_slm_controller/multi_slm_controller_user_manual.md`](docs/GUI/slm/multi_slm_controller/multi_slm_controller_user_manual.md)
与
[`docs/GUI/slm/multi_slm_controller/multi_slm_controller_srs.md`](docs/GUI/slm/multi_slm_controller/multi_slm_controller_srs.md)。

## 项目结构（简版）

```
AO-shaping/
├── src/
│   ├── ao_shaping/     # 主程序包: runners/ algorithm/ model/ drivers/
│   │                   #   optimizer/ utils/ tools/ display/ gui/
│   ├── ml/             # 机器学习独立包: gsnet/ phase/ hwdataset/ zernike/
│   ├── calculators/    # Cython 扩展 (standalone)
│   └── optical_ui/     # [DEPRECATED] Empty package
├── tests/ao_shaping/   # 测试（镜像 src 结构）
├── scripts/            # 实用脚本，按处理对象分组（含 generate_*_report.py）
│                       # slm/ zernike_bench/ dm/ wfs/ = 设备域；ml/ rl/ oopao/ build/ = 数据与用途
├── docs/               # 设备说明与开发指南
├── report/             # 实验报告（结论的唯一事实源）
├── libs/               # 第三方 SDK 二进制 (gxipy, Drv_UDPST) + OOPAO 子模块
├── docker/             # 测试镜像（GPU / CPU）
├── CONTRIBUTING.md     # 贡献流程与质量门禁
└── AGENTS.md           # 仓库级开发约定（AI 工具与人类共同遵守）
```

完整目录树见 [docs/architecture.md](docs/architecture.md#项目结构详细)。

## 约定与贡献

- 中文叙述必须遵守 [`docs/terminology_glossary.md`](docs/terminology_glossary.md)；
  提交前运行 `python3 scripts/_common/term_unify_20261016.py --check`。
- 实验结论写 [`report/`](report/)，设备说明与操作手册写 `docs/`；报告生成脚本
  只能放在 `scripts/`（命名 `generate_*_report.py`），并登记到
  `scripts/_common/provenance.py` 的 `REPORTS` 映射。
- 提交前跑 `ruff check .`、相关测试与术语门禁；分支/提交信息规范见
  [`CONTRIBUTING.md`](CONTRIBUTING.md)。

## 近期更新

版本历史已迁至 [CHANGELOG.md](CHANGELOG.md)（当前最新 **v0.19.0**，2026-10-19）。
最近一轮要点：物理预测模型在环（model-in-loop）闭环成型、GS→Zernike 阶数扫描立项、
`scripts/` 按处理对象重组、ruff 硬门禁恢复入库。
