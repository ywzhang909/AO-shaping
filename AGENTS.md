# AGENTS.md - AO-Shaping Development Guide

**Generated:** 2026-06-20

## Project Overview

AO-Shaping is an Adaptive Optics (AO) system using reinforcement learning for wavefront correction and beam shaping. It integrates multiple optimization algorithms including WFS-based and wavefront-sensorless methods.

## Project Structure

```
AO-shaping/
├── src/
│   ├── ao_shaping/          # Main package
│   │   ├── main.py              # CLI entry point (Click-based)
│   │   ├── runners/             # Runner scripts package
│   │   │   ├── __init__.py     # Re-exports for backward compatibility
│   │   │   ├── wf_runner.py         # Wavefront RMS optimizer
│   │   │   ├── axis_beam_runner.py  # PIB optimizer
│   │   │   ├── pipeline_runner.py   # Serial WF→PIB pipeline
│   │   │   └── zernike_matrix_runner.py  # Zernike response matrix
│   │   ├── algorithm/            # Optimization algorithms (Adam, SGD, etc.)
│   │   ├── drivers/              # Hardware drivers (see drivers/AGENTS.md)
│   │   │   ├── ccd/              # Cameras (Daheng, MiiCam)
│   │   │   ├── dm/               # Deformable Mirrors (NLight)
│   │   │   ├── slm/              # Spatial Light Modulators (Santec, WavefrontCorrection)
│   │   │   ├── wfs/              # Wavefront Sensors (Thorlabs)
│   │   │   ├── tm/               # Timing modules (Serial/FSM)
│   │   │   ├── sim/              # Digital twin simulation
│   │   │   └── mock_devices.py   # Mock devices for testing
│   │   ├── optimizer/            # High-level optimizers
│   │   │   ├── wf/               # Wavefront-based (RMS)
│   │   │   ├── wfless/           # Wavefront-sensorless (PIB)
│   │   │   │   └── slm_square_shaping.py  # SLM方形光斑 SPGD 整形优化器
│   │   │   └── rl/               # Reinforcement learning (SAC)
│   │   ├── utils/                # Utilities (spots_calc, wavefront_calc, zernike_calc, zernike_utils, wfs_utils)
│   │   ├── ml/                  # Machine learning (U-Net+GAN, training, models) — NOTE: lives at src/ml/ as a separate standalone package
│   │   │   └── hwdataset/       # 硬件相位→相机图像 DataLoader (index/transforms/records/dataset/cache/inspect)
│   │   │   ├── trainer/         # Training utilities
│   │   │   ├── models/          # Neural network models
│   │   │   └── wandb_logger.py  # WandB integration
│   │   ├── tools/                # Standalone tools (SLM phase capture, Micro-DM image collection, data collection)
│   │   ├── display/              # Visualization (Windows, frames for GUI)
│   │   └── gui/                  # GUI components (Streamlit), 按设备域分包:
│   │       ├── r50/              #   R50Power 控制器 UI (r50_controller_ui.py 入口)
│   │       ├── dm/               #   变形镜 UI (micro_dm_ui.py, ceramic_viewer.py)
│   │       ├── slm/              #   SLM UI (multi_slm_controller.py, slm_calibration_ui.py)
│   │       ├── zernike/          #   Zernike UI (response matrix 校准/调试查看)
│   │       └── ccd/              #   CCD 图像分析 UI (ccd_analyzer.py)
│   ├── calculators/               # Cython extensions (standalone)
│   └── optical_ui/                # [DEPRECATED] Empty package
├── tests/ao_shaping/              # Tests (mirrors src structure)
├── libs/                          # Third-party SDK binaries (gxipy, Drv_UDPST)
└── scripts/                       # Utility scripts
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Hardware drivers | `src/ao_shaping/drivers/` | See drivers/AGENTS.md |
| CLI runner scripts | `src/ao_shaping/runners/` | 硬件编排层: Click CLI 命令注册, 设备生命周期 (open/close), 结果保存 |
| Optimization algorithms | `src/ao_shaping/algorithm/` | 5 子包: gradient/, heuristic/, signal_processing/, tabu/, goal_functions/. Class-based optimizer convention: see src/ao_shaping/algorithm/README.md |
| Wavefront optimizers | `src/ao_shaping/optimizer/wf/` | RMS optimization |
| Zernike response matrix | `src/ao_shaping/optimizer/wf/zernike_response_matrix.py` | SLM→WFS Zernike校准 |
| PIB optimizers | `src/ao_shaping/optimizer/wfless/` | Power-in-bucket |
| SLM Zernike PIB / 方形整形 (同一模块) | `src/ao_shaping/optimizer/wfless/slm_zernike_pib.py` + `slm_square_shaping.py` + `runners/slm/slm_shaping_runner.py` | `slm_shaping_runner` 一个 click 组装下两个家族: `slm-pib`(`spgd`/`heuristic`) → `slm_zernike_pib.py`; `spgd-square`(同 `square` 子命令) → `slm_square_shaping.py`。`--cam_type sim` 在两半都走数字孪生 (`runner_common.patch_sim_*_shaping`) |
| SLM方形光斑整形 (SPGD, freeform) | `src/ao_shaping/optimizer/wfless/slm_square_shaping.py` + `runners/slm/gsnet_runner.py` | SPGD 优化 Zernike 系数 → 均匀方形远场 (CLI: `slm-gsnet`) |
| GS 预整形 + 自由相位 SPGD 细化 (硬件) | `src/ao_shaping/optimizer/wfless/slm_gs_refine.py` + `runners/slm/gs_refine_runner.py` | 仿真 `iterative_zernike_shaping.py` 的硬件移植: GS 开环预矫正 (仅当实测优于平场才采用) + 无感知 SPGD 细化 (CLI: `slm-gs-refine`) |
| 正向模型闭环校正 + 反复迭代 (硬件) | `src/ao_shaping/optimizer/wfless/slm_model_in_loop.py` + `runners/slm/model_in_loop_runner.py` | 仿真 `model_in_loop_shaping.simulate_iterative_shaping` 的硬件移植: 每轮用强随机探针重拟合正向模型的 Zernike 像差 (Step A), 再冻结该像差合成目标方斑相位 (Step B), 用 trust region + 逐轮验收抑制两者互相追�� (CLI: `slm-model-in-loop`) |
| Zernike 工具 | `src/ao_shaping/utils/wavefront/zernike_utils.py` | 系数解析 (Noll/(n,m)/数组) + 相位生成，Noll 1976 约定 |
| RL training | `src/ao_shaping/optimizer/rl/` | SAC, LR-WFS |
| Simulation | `src/ao_shaping/drivers/sim/` | Digital twin devices |
| Utilities | `src/ao_shaping/utils/{io,image,wavefront,slm}/` + root `cli_params.py` | 4 子包: io/, image/, wavefront/, slm/ (spots_calc, wavefront_calc, zernike_calc, display 等) + 零导入叶子 `cli_params.py` |
| Standalone runners (未注册) | `src/ao_shaping/runners/` | `slm_offset_runner` 计划迁移至 `tools/slm/`。⚠️ `slm_shaping_runner` **曾经**在此列 (2026-10-05 前它未注册); 现已注册为 `slm-pib` + `spgd-square`, 不再属于本行 |
| ML training | `src/ml/` (standalone, not inside `ao_shaping/`) | U-Net+GAN, trainer, wandb_logger |
| 硬件相位→相机图像 DataLoader | `src/ml/hwdataset/` | `data/debug` 全量转 PyTorch Dataset: 输入=SLM 相位+曝光, 输出=CCD 画面 (见 `硬件调试转 Dataset` 节) |
| Standalone tools | `src/ao_shaping/tools/` | SLM phase capture, Micro-DM per-channel image collection, train data collection |
| Visualization | `src/ao_shaping/display/` | Windows, frames for GUI |
| GUI | `src/ao_shaping/gui/{r50,dm,slm,zernike,ccd}/` | Streamlit components, 按设备域分包 (见上方目录树) |
| Tests | `tests/ao_shaping/` | Mirror of src structure |

---

## optimizer/ Module

高层次优化策略层, 负责实现特定优化目标 (RMS、PIB、Zernike 标定等) 并编排硬件与算法的协作。

### 职责

- 实现优化策略 (RMS via DM电压 / RMS via SLM Zernike / PIB / GA Zernike / 响应矩阵标定等)
- 管理优化过程中的硬件状态 (电压/相位/图像记录)
- 调用 algorithm 包中的基础算法进行参数更新
- 子包:
  - `wf/`: 基于波前传感器的优化 (DM 电压 RMS, SLM Zernike RMS, Zernike 响应矩阵, GA Zernike)
  - `wfless/`: 无波前传感器优化 (PIB via DM电压, SPGD, 模拟 SPGD, 微分波束整形, SLM 方形光斑 SPGD)
  - `rl/`: 强化学习优化 (SAC, LR-WFS)

### 优化策略与 Runner 对应关系

| Runner (main.py 命令) | Optimizer 函数 | 子包 | 优化方式 | 硬件 |
|---|---|---|---|---|
| `wf` | `optimizer.wf.rms:optimizer_rms_dm()` | wf | DM 电压 RMS (SPGD) | DM + WFS |
| `pib` | `optimizer.wfless.pib:optimize_pib()` | wfless | DM 电压 PIB (SPGD) | DM + CCD |
| `pipeline` | `wf.rms:optimizer_rms_dm()` + `wfless.pib:optimize_pib()` | wf + wfless | WF RMS → PIB 串行 | DM + WFS + CCD |
| \zernike-matrix\ | \optimizer.wf.zernike_response_matrix:calibrate_zernike_response_matrix\ | wf | Zernike 响应矩阵标定 + 闭环优化 | SLM + WFS |
| `rms-zernike` | `optimizer.wf.rms_by_zernike:optimizer_rms_slm()` | wf | SLM Zernike RMS | SLM + WFS |
| `ga-zernike` | `optimizer.wf.ga_zernike:optimizer_ga()` | wf | GA Zernike | SLM + WFS |
| `combined` | `optimizer.combined_optimizer:optimize_pib()` | wfless | AdaMOD + SPGD 混合 PIB | DM + CCD |
| `slm-pib` (`spgd`) | `optimizer.wfless.slm_zernike_pib:optimize_slm_zernike_pib()` | wfless | Zernike 系数 SPGD 梯度 (PIB / RMS / Pearson… 目标形状) | SLM + CCD |
| `slm-pib` (`heuristic`) | 同上 (`algorithm=ga/pso/sa/hc/rs/cem/de`) | wfless | 黑盒启发式搜索 | SLM + CCD |
| `spgd-square` | `optimizer.wfless.slm_square_shaping:optimize_slm_square()` | wfless | 均匀方形远场 (CV + EE + AR 综合质量分) | SLM + CCD |
| `slm-gs-refine` | `optimizer.wfless.slm_gs_refine:optimize_slm_gs_refine()` | wfless | GS 预矫正 (bake-off) + 自由相位 SPGD 细化 | SLM + CCD |
| `slm-model-in-loop` | `optimizer.wfless.slm_model_in_loop:optimize_slm_model_in_loop()` | wfless | 探针 refit 正向模型 (Step A) + 目标光斑相位合成 (Step B), 带 trust region 与逐轮验收 | SLM + CCD |

> **`slm-pib` 配置容器** (2026-09): `optimize_slm_zernike_pib()` 已收敛为**纯 dataclass 单参数 API**:
> `def optimize_slm_zernike_pib(config: SlmZernikePibConfig)` (`optimizer/wfless/slm_zernike_pib.py`)。
> `SlmZernikePibConfig` 是普通 `@dataclass` (无自定义 `__init__`), 必填 `center`/`epochs` 在前,
> 嵌套 `camera: CameraParamsPib` / `slm: SlmParamsPib` (惰性默认, 定义于 `runners/runner_common.py`),
> 保留 `kwargs` 逃生口 (`**config.kwargs`)。**不再接受任何平铺关键字参数 / `cam=` / `slm=`**;
> 设备由优化器内部经 `create_camera(config.camera)` / `Santec.from_params(config.slm)` 上下文管理器
> 自行打开/关闭 (禁止跨 run 复用设备)。调用方: `runners/slm/slm_shaping_runner.py`、`scripts/compare_shape_objectives.py`、
> `scripts/generate_slm_pib_heuristic_hw_report.py`、`scripts/repeat_shape_objectives.py`、
> `tests/ao_shaping/optimizer/wfless/test_slm_zernike_*`。
> 内部重命名 (非公开 API): `test_pib`→`ideal_pib_ratio`、`intellij_center`→`_smart_center`
> (本文件内, 与 `pib.py` 的 `intellij_center` 无关)、`to_min` 标量删除 → 由 `objective_mode`
> 派生 `_spgd_sign`。

> **注意**: `optimizer/wf/rms.py` 和 `optimizer/wf/rms_by_zernike.py` 的函数名冲突已通过重命名解决:
> - `rms.py:optimizer_rms_dm()`: DM 电压控制 + WFS 测量 (用于 `wf` 和 `pipeline` 命令)
> - `rms_by_zernike.py:optimizer_rms_slm()`: SLM Zernike 相位控制 + WFS 测量 (用于 `rms-zernike` 命令)

---

### algorithm/ Module — 基础算法层

`src/ao_shaping/algorithm/` 提供纯数学优化算法, 不包含任何硬件知识。

#### Class-based Optimizer Convention

New optimizers added to `src/ao_shaping/algorithm/` MUST follow the class-based API: `__init__` does validation + state setup, `update()` performs one step and returns the next state/solution, and an optional `run()` returns a result dataclass. A one-shot function is kept only as a thin wrapper for backward compatibility. Torch/numpy **simulation-first tests are required before any hardware use**. Canonical example: `DifferentiableBeamOptimizer` (`src/ao_shaping/algorithm/differentiable_beam.py`). Full principle: `src/ao_shaping/algorithm/README.md`.

#### 算法分类

| 类别 | 算法 | 说明 |
|------|------|------|
| 梯度优化 | `Base`, `SGD`, `Adam`, `AdamW`, `AdaMOD`, `Muno`, `MuonW`, `AdamNS` | `update(grad) → next_step` 模式 |
| 启发式搜索 | `GeneticAlgorithm`, `PSO`, `SA`, `CEM`, `DE`, `HC`, `RandomSearch` | 无梯度全局搜索 |
| Tabu 搜索 | `TabuMemory`, `AdaptiveSearchState`, `TabuSearchRunner` | 禁忌搜索 |
| 信号处理 | `PhaseWrapOptimizer`, `GerchbergSaxton`, `SLMPhaseController`, `ControlLaw` | 相位包裹/光强重建/控制律 |
| 可微分 shaping | `DifferentiableBeamOptimizer`, `DifferentiableShapingResult` | PyTorch 可微分波前优化 (需 GPU) |
| 目标函数 | `ImageTargetFunc` | 图像质量指标 |

#### 子包结构

| 子包 | 内容 | 说明 |
|------|------|------|
| `gradient/` | `adam`, `acceleration` | 梯度优化器 |
| `heuristic/` | `ga`, `pso`, `sa`, `hc`, `rs`, `cem`, `de` | 无梯度启发式搜索 |
| `signal_processing/` | `gerchberg_saxton`, `phase_wrap`, `controller`, `iterative_base`, `differentiable_beam`, `differentiable_shaping`, `beam_shaping_utils`, `wavefront`, `beam_shaping_benchmark` | 相位恢复/控制律/可微分整形 |
| `tabu/` | Tabu 搜索 | 禁忌搜索 |
| `goal_functions/` | `target_func`, `image_metrics` | 目标函数与图像质量指标 |

> **注意**: top-level `algorithm/*.py` 文件是 *-re-export shims (向后兼容)。

---

### 三层架构关系

```
CLI (main.py Click 命令)
  │
  ├─ wf ──────────────→ runners/wf_runner.py ──→ optimizer/wf/rms.py:optimizer_rms_dm() ──→ algorithm: Adam/AdaMOD
  ├─ pib ─────────────→ runners/axis_beam_runner.py ──→ optimizer/wfless/pib.py:optimize_pib() ──→ algorithm: AdaMOD/Adam/SGD/Muno
  ├─ pipeline ────────→ runners/pipeline_runner.py ──→ optimizer/wf/rms.py:optimizer_rms_dm() + wfless/pib.py ──→ algorithm: Adam/AdaMOD
  ├─ zernike-matrix ──→ runners/zernike_matrix_runner.py ──→ optimizer/wf/zernike_response_matrix.py ──→ (标定)
  ├─ rms-zernike ────→ runners/rms_zernike_runner.py ──→ optimizer/wf/rms_by_zernike.py ──→ algorithm: Adam/AdaMOD
  ├─ ga-zernike ─────→ runners/zernike_search_runner.py ──→ optimizer/wf/ga_zernike.py ──→ algorithm: GA
  ├─ slm-pib ─────────→ runners/slm/slm_shaping_runner.py ──→ optimizer/wfless/slm_zernike_pib.py ──→ algorithm: Adam/AdaMOD/GA/PSO…
  ├─ spgd-square ─────→ runners/slm/slm_shaping_runner.py (square) ──→ optimizer/wfless/slm_square_shaping.py ──→ algorithm: SPGD (同名 click 组)
  └─ combined ────────→ runners/combined_runner.py ──→ optimizer/combined_optimizer.py ──→ algorithm: AdaMOD/SPGD

runners/       硬件编排层  — Click CLI, 设备生命周期 (open/close), 结果保存
optimizer/     策略实现层  — 优化流程编排, 硬件状态管理, 调用 algorithm 更新参数
algorithm/     算法基础层  — 纯数学优化器 (update/grad), 无硬件知识
```

**调用链**: `Runner` 打开硬件 → 调用 `Optimizer` 函数 → `Optimizer` 创建 `Algorithm` 实例 → `Algorithm.update(grad)` 返回参数更新 → `Optimizer` 应用更新并记录历史 → `Runner` 关闭硬件并保存结果。

---

## runners/ Module

硬件编排层 — Click CLI 命令注册, 设备生命周期 (open/close), 结果保存。

### 职责

- 注册 CLI 命令 (main.py Click group) 并解析参数
- 打开/关闭硬件设备, 管理设备生命周期
- 调用 optimizer 层函数执行优化流程
- 保存优化结果与调试产物

### 已注册 CLI 命令 vs 独立 Runner

| 类型 | 说明 |
|------|------|
| 已注册 CLI 命令 | main.py 注册 20 个命令 (含 `slm-gsnet`, `combined` 等), 见 Entry Points 节 |
| 独立 Runner (未注册) | 需直接运行 `python -m ao_shaping.runners.xxx` 或 standalone 脚本; `slm_offset_runner` 计划迁移至 `tools/slm/`。⚠️ `slm_shaping_runner` **曾经**在此列 (2026-10-05 前它未注册); 现已注册为 `slm-pib` + `spgd-square`, 不再属于本行 |

### 共享辅助

`runner_common.py` 是共享辅助函数的主目录: `resolve_dm` (DM 解析), debug-artifact 写入器等。

> **注意**: `closed_loop.py` 不是 runner — 它是 `AOClosedLoop` 控制类, 归属 `optimizer/wf/`。

---

## utils/ Module

Utility functions for image processing and calculations, organized into 4 subpackages:

| Subpackage | Contents | Notes |
|------|---------|-------|
| `utils/io/` | `file`, `timestamp`, `cli_helpers`, `device_config`, `network`, `handler` | `handler` becomes a backward-compat shim to `display/` |
| `utils/image/` | `spots_calc`, `beam_metrics`, `targets`, `resample`, `display`, `hardware_utils` | `gs_visualization` 已迁至 `display/`（2026-10-03）；`utils/hardware_utils.py` 别名 shim 已删（2026-10-03）。**仍开放**: `display.py` 本身仍是 utils 里的渲染器，需连同 `ImageVoltagesDisplay` / `plot_funcs` / `VOLT_HEIGHT` 的 re-export 一起搬 |
| `utils/wavefront/` | `zernike_calc`, `zernike_utils`, `wavefront_calc`, `wfs_utils`, `phase_unwrap`, `hadamard_calc`, `matrix_utils` | |
| `utils/slm/` | `pattern_helper`, `slm_lut`, `phase_display` | |
| `utils/cli/params.py` (`utils/cli/` 子包) | `option`, `with_params`, `ClickGroup` | **零 `ao_shaping` 导入**（2026-10-03 从 `runners/runner_common.py` 抽出，TODO R-36）。见下方红线 |

> **注意**: legacy top-level `ao_shaping.utils.X` paths remain importable via shims.

### 🔴 `utils/cli/params.py` 是零导入叶子 (2026-10-03 抽出 R-36, 2026-10-05 移入 `utils/cli/`)

dataclass→click 参数绑定机制 (`Annotated[T, option(...)]` + `with_params` 收集器 +
对象投递) 的**唯一**实现。它同时被 `runners/` 和 `tools/slm/` 消费，所以:

- **禁止任何 `ao_shaping.*` 导入** —— 模块级、函数体内、`TYPE_CHECKING` 下都不行。
  只允许 **stdlib + `click`**。
- 理由链: `tools/slm/` 探针要在**无设备**路径上可用。若机制留在 `runners/runner_common.py`，
  `tools/` 就得 import `runners/`，而后者 import 期就 `list_dm_types()` 并
  `import ao_shaping.drivers.dm.asyn_micro_dm` 触发注册副作用 —— 台架探针会在
  没设备时先碰硬件包，正是 R-20 惰性契约要避免的。
- 机制**不得在别处重新定义**。import 是正常消费（12 个 runner 都 import
  `with_params`），**定义**才是重复 —— 那正是 R-25 里两份 `_fmt` 漂移的成因。
- 守卫是 **AST 而非 grep**：`tests/ao_shaping/utils/test_cli_params_leaf.py`
  遍历全树，能挡住藏在函数体里的延迟 import；grep 挡不住。
  ⚠️ `ruff` **查不出**「从错误模块导入」（它只报「导入未使用」），所以这一项
  必须靠自己的测试，不能只靠 lint。
- 私有名（如 `_collect_click_annotations`）**不跨模块 re-export**：测试直接
  从本模块 import。

---

## Zernike 使用规范 (canonical 入口 — 禁止重复实现)

> **单一事实源**: 全项目所有 Zernike 纯数学 (模式枚举 / 索引换算 / 相位生成 / 系数解析 / 单位换算) 统一走 `utils/wavefront/` 两层。**任何脚本、runner、tools、GUI、optimizer 不得自行实现** Noll↔(n,m) 查表、Zernike 多项式求值或相位生成 (2026-09 去重重构, 此前散落在 `gui/slm/`、`optimizer/wfless/`、`tools/slm/` 的重复实现已全部收敛)。详见 README `## Zernike 使用指南`。

| 层 | 模块 | 公开 API | 何时用 |
|---|---|---|---|
| API 层 (首选) | `utils/wavefront/zernike_utils.py` | `parse_zernike_coefficients`, `generate_zernike_phase`, `list_zernike_modes` (4元组 noll,n,m,name), `coefficients_to_array`, `um_to_waves`, `LAMBDA_UM` | 绝大多数场景 |
| 引擎层 (复用/底层) | `utils/wavefront/zernike_calc.py` | `ZernikeGenerator` (网格缓存: `generate_noll`/`generate_polynomial`/`generate`/`fit`), `noll_to_nm`/`nm_to_noll`, `zernike_modes`/`noll_indices`, `calc_n_zernike_terms`, `get_zernike_name`, `fit_zernike` | 同分辨率反复生成 (复用实例), 或底层索引/拟合 |

单向依赖: `zernike_utils` → `zernike_calc`; 上层只 import 这两层, 不得反向。

**红线** (对应 ANTI-PATTERNS):
1. 自写 Noll↔(n,m) 查表 / 模式枚举 — `noll_to_nm_legacy` (Noll 5=(2,0), 与 canonical 相反) 已删除, 勿再引入第二套索引
2. 生成器自行 `mod 2π` — 一律返回 **raw 未包裹弧度**; 唯一 wrap 点 `Santec.create_phase_from_array()`
3. min-max 归一化相位 — 已知反模式 (尺度无关, 幅度不可控): `ZernikeDM.generate_phase`（于 2026-09 修复）。同樾的 `PatternHelper._zernike_to_uint16` 于 2026-09 **已删除**（仅作历史参考，不再是现存代码）
4. WFS µm 系数未经 `um_to_waves()` 直接参与运算; λ 系数未 ×2π 直接喂 `make_phase`/`generate_zernike_phase`
5. `zernike_calc.noll_indices` (Noll 序) 与 `zernike_modes` ((n,m) 字典序) 顺序不同, 不可互换

---

## ml/hwdataset/ Module — 硬件调试转 Dataset

`src/ml/hwdataset/` 把 `data/debug/` 下的**全部**硬件调试产物转成 PyTorch Dataset:
**输入 = 下发给 SLM 的相位 + 相机曝光参数, 输出 = 相机画面**。独立包 (`src/ml/`),
不 import 任何硬件驱动即可 import (`ml` 是可选依赖组)。

### 语料实测 (2026-10-02, 已核验)

| 项 | 值 |
|---|---|
| 语料 | `data/debug/` 996 文件 / **265 个 `.pkl` / 65.7 GB** / 单文件最大 **3.42 GB** / 中位数 55 MB |
| 可用记录 | **11,393** / 261 文件 / 7 个 family (总记录 11,441) |
| 相位来源 | `panel_gray` 7866 (uint16 灰度) · `panel_rad` 2199 (float32 弧度) · `zernike` 1202 (由 `_c` 反演) · `freeform` 126 (由 `_c` 上采样) |
| 排除 | `missing_phase` 48 (`bench_stability` 42 条平场漂移无指令相位 + `sim_calib_abba` 6/18) · `no_records` 3 (`None` 占位) |
| 曝光 | 11,393/11,393 全部可解析; 0.1–80.0 ms, 9 个不同值 |
| 索引构建 | 全量 35 s; 带 JSON cache **0.09 s** |
| 整轮 epoch | 11,393 条 / 179 batch (bs=64) / 单进程 **450 s** |

### 模块

| 文件 | 职责 |
|---|---|
| `index.py` | 递归发现全部 `*.pkl` + 合并 sidecar JSON + **逐记录**判定相位来源。`PhaseSource` / `ExclusionReason` / `HwRecordRef` / `HwCorpusIndex` / `build_hw_index` / `family_of` |
| `transforms.py` | 纯 array→array: 灰度→弧度、ROI 裁剪、**相干**块平均、Zernike/freeform 面板还原、远场裁剪 |
| `records.py` | 单条 `HwRecordRef` → `HwSample` 的唯一实现 (dataset 与 cache 共用, 保证两条路径**逐位一致**); `MaterialiserConfig` / `PayloadStore`(有界 LRU) / `Materialiser`(也是缓存读取点, `use_cache` 默认 True) |
| `train_amp.py` | `python -m ml.zernike.train_amp` 训练入口: DataLoader 驱动, 记录 loss / grad norm / R² / PSNR / SSIM / NRMSE / perplexity / corr / efficiency / 质心偏移 / 光斑直径比, 出 true-vs-pred 对比图, 推 wandb (无 key 时 offline) |
| `metrics.py` | img2img 标准指标 (MSE/RMSE/MAE/NRMSE/PSNR/SSIM) + 光斑域指标 (质心偏移 / 90% 环围直径 / 峰比), 复用 `beam_metrics.compute_metrics` / `measure_spot_diameter_cam` |
| `dataset.py` | `HwPhaseImageDataset` / `FileGroupedSampler` / `build_hw_dataloader` / `create_hw_dataloaders` |
| `cache.py` | 派生网格的 mmap 缓存 (`.hw_cache/<stem>/`), 消除每轮 212 s Zernike 反演 + 35 s 读盘 |
| `inspect.py` | `python -m ml.hwdataset.inspect` 语料统计 / 抽样自检 / 建缓存 |

#### 缓存是**被读取**的, 不是只能写 (2026-10-02 实测并接线)

`Materialiser(use_cache=True)` (默认) 在 `materialise()` 里先查缓存, 命中就走 mmap,
不命中 (无缓存 / 配置不符 / 该 ref 未缓存 / 缓存损坏) 逐条回落到直接路径。`prepare_hw_cache`
与 Dataset 共用同一个 `Materialiser`, 所以两条路径**逐位一致**, `use_cache` 只改速度与内存。

真语料 `slm_zernike_shaping` (1010 记录 / 10 pickle) 实测:

| 路径 | 耗时 | 打开的 pickle 数 |
|---|---|---|
| `use_cache=False` | 127.0 s | 10 |
| `use_cache=True` | **0.4 s** | **0** |

**317× 加速, 且一个 pickle 都不再打开** —— 峰值 RAM 从「一个整 pickle (最大 3.42 GB)」
降到「一页」。实测 1010/1010 样本 (含 `exposure_ms` / `fov_px`) 与直接路径逐位一致。

接线时踩到并修掉的四个真问题 (勿回退):

1. **`prepare_hw_cache` 必须 `use_cache=False`。** 否则 builder 会拿正要写的缓存当输入,
   Windows 下已打开的 mmap 让目标文件不可覆写, 每个 `np.save` 报 `Invalid argument` (errno 22),
   整族被静默跳过。
2. **`exposure.npy` 必须是 float64, 不是 float32。** float32 把 `0.1 ms` 变成
   `0.10000000149011612`, 真语料 1010/1010 样本只在这一个字段上不一致 (合成语料的
   1.0/2.0/0.5 恰好可被 float32 精确表示, 所以测试盲区掩盖了它)。数组是 `(N,)`, 全库 91 KB。
3. **`fov_px` 必须走缓存。** 原先 `Dataset._fov_px` 从 payload LRU 反查 `_img` 形状, 靠
   「`materialise` 刚好填过一次 LRU」成立 —— 缓存命中时 LRU 从未被填, 于是每条记录都要**重新
   打开整个 pickle** (最大 3.42 GB), 恰好抵消缓存的意义。现在几何尺寸由 `HwSample.fov_px` 携带。
4. **`Materialiser.__getstate__` 必须丢掉 `_cached`。** `np.memmap` 不可 pickle;
   Windows `spawn` 每 worker 重建 Dataset, 丢了 flag 会静默退回慢路径整个 epoch。

> **缓存槽位 = 该 pickle 内的成功物化序号 (rank); 唯一定址方式是
> `HwCachedSource.position_of(ref.key if ref.key is not None else ref.position)`。**
> 缓存只为**成功物化**的 ref 存一个槽, 顺序跟随 index 顺序; 每个槽写入的是**该 ref 自己的
> stored key** (`keys.npy`), 所以上面这个查表对任何语料都精确 —— 即使
> `key != position` (真语料 `sim_calib_abba_20261001_163743.pkl` 存 key 10000-10011 于
> position 6-17), 即使有记录被排除 (两者都会让 rank ≠ position, 但都不动 stored key)。
> 实测真语料 1010/1010 记录按此取槽逐位一致。
> ⚠️ 键在单个 pickle 内唯一, 故 `position_of()` 单射; 但它只在「该 ref 确实被缓存」时可用,
> `KeyError` 表示回落到直接路径, 不是错误。

### 样本契约

```python
{
  "phase_cos":     (1, g, g) f32,   # 相位网格的 Re —— 连续表示, 无 arctan2 分支切口
  "phase_sin":     (1, g, g) f32,   # 相位网格的 Im
  "image":         (1, g, g) f32,   # 远场画面, [0,1] 绝对强度 (未做 peak 归一)
  "exposure_log10": (1,)      f32,   # (log10(ms) - mean) / std, 按 index 级统计标准化
  "exposure_ms":   float,           # 原始 ms (default_collate 会转成 (B,) f64 张量)
  "contrast":      (g, g)    f32,   # hypot(cos, sin) = 每格相位相干度 [0,1]
  "source": str, "family": str, "sample_idx": int, "path": str, "fov_px": int | None,
}
```

> `fov_px` 由 `HwSample.fov_px` 携带 (`records.py`), 不再由 Dataset 回查 payload ——
> 缓存命中时 payload 根本不在内存里, 回查会重新打开整个 pickle。

### 🔴 四条不可回退的实测结论 (改动前必读)

> **单指标会骗人 —— 用 `ZernikeAmpModel` 时三个归一化/可观测量选项各自埋了一个坑,
> 都是实测出来的 (真语料 `slm_zernike_shaping`, 固定 split, 逐项单变量):**
>
> | 选项 | val MSE | R² | PSNR | SSIM | 结论 |
> |---|---|---|---|---|---|
> | `normalization="peak"` | 0.00429 | **+0.706** | 24.8 dB | 0.594 | ✅ 唯一正确选择 |
> | `normalization="sum"` | **0.00000** | +0.632 | **72.1 dB** | **0.9996** | ⚠️ **陷阱** |
> | `normalization="none"` | 0.47534 | **−158.98** | 3.2 dB | 0.037 | ❌ 不可用 |
>
> `sum` 把预测与目标各自除以总能量, 两者于是"几乎一致" —— MSE/PSNR/SSIM 全部好到离谱,
> 而 **R² 反而更差**。任何含 sum 归一化的指标都在度量归一化本身, 不是拟合质量。
> **判据: 只看 R² / corr / 光斑域指标, 永远不要用 MSE/PSNR/SSIM 选这个开关。**
>
> | `observable` | n_max=4 | n_max=11 | n_max=15 |
> |---|---|---|---|
> | `"amplitude"` (原始需求写的 `amp`) | +0.634 | +0.659 | +0.675 |
> | `"intensity"` (**默认**) | **+0.750** | **+0.792** | **+0.797** |
>
> **CCD 测的是强度, 不是场振幅**, 所以 `intensity` 既是物理上正确的, 也是实测最优的,
> 三个 `n_max` 上都稳定领先 ~0.12 R²。默认值已改为 `intensity`, `amplitude` 保留可选。
>
> **噪声地板: 只换 split seed, R² 就从 0.780 跑到 0.923 (σ≈0.073)。** 同一 config
> 同 seed 重复 3 次结果**逐位相同** (确定性没问题), 所以这个方差全部来自
> "val 只有 2 个文件 / 128 条"。**单次 run 的 R² 差 ±0.13 是噪声, 不要据此排序**。
> 逐项单变量扫描里只有 `n_max` (0.48→0.80) 明显高于噪声; `l2_penalty` /
> `grad_clip` / `momentum` / `optimizer` 的差异全在噪声内 —— 想要可信排序必须多种子取均值。

> **噪声地板的真因 (2026-10-04): 主要不是"验证集只有 128 条", 而是折间难度差。**
> `slm_zernike_shaping` 的 1010 条记录均匀分布在 10 个 pickle (每份恰好 101 条),
> 但这 10 份只来自 **4 个优化目标** (`rms_pib` 4 / `rmse_out` 3 / `shape` 2 / `roi_pib` 1)。
> **四个目标的图像分布确实不同** —— 用 A 目标均值图预测 B 目标, 峰值归一后 R² 落在
> **+0.54 ~ +0.97** (最低 `shape` → `roi_pib` = +0.538), 对角线为 1。
> ⚠️ 但**没有一对是负的**。早期文档称 `rms_pib → roi_pib = −0.670`「比预测常数还差」
> —— **那个数是算错的** (分母误用了另一个 group 的方差, 量纲不一致而放大; 正确分母是
> 被预测组自身的 `SS_tot`)。结论方向 (目标异质 → 折的组成会变) 成立, 但"比常数还差"作废。
> **真正的大头是折难度本身**: 留一 pickle 的逐折 R² 从 0.737 到 0.941 (σ≈0.08),
> 同一模型换一折就差 0.2 ⇒ 任何**非配对**比较都淹没在折间方差里。
> **所以必须分组 + 配对**: 三个模型共用同一批折, 折难度在差值里抵消, 配对 σ 降到 ~0.01。
> ⚠️ 另: `_select_records` 的划分是**按文件的 75/25** (`val_fraction=0.25`,
> train_amp.py:302), 之后验证集再被截断到 `max_val=128` (:315) —— **不是 80/20**。
> ⚠️ `HWRecordRef.source` 是**相位表示** (`panel_gray`/`panel_rad`/`zernike`/`freeform`),
> **不是来源文件**; 文件标识是 `HWRecordRef.path`。

> **与 U-Net 的对比 — 权威结论是 grouped CV (`scripts/compare_models_cv.py`)**
> 仓库**完全没有 sklearn 依赖** (全树零引用), 故折按 `str(record.path)` 手工构造。
> 统计用**精确 sign-flip 置换检验** (2¹⁰=1024 次枚举, 最小 p=0.00195, 不假设正态)
> + Cohen's `d_z` + Holm-Bonferroni。同 split / 同输入 / 同 loss / 同 50 epoch 预算:
>
> | 模型 | 参数 | val R² (10 折) | val SSIM | val PSNR |
> |---|---|---|---|---|
> | `hybrid` (physics+CNN 残差) | 10,408 | +0.8721 ± 0.0799 | 0.7404 ± 0.0817 | 29.07 ± 2.51 |
> | `physics(n_max=15)` | **135** | +0.8727 ± 0.0867 | 0.7320 ± 0.0704 | 29.21 ± 2.64 |
> | `unet`[16…256] | 7,778,465 | **+0.9004 ± 0.0610** | **+0.8419 ± 0.0618** | **+31.05 ± 2.44** |
>
> 配对差 (physics − unet, 负 = unet 更好): **SSIM −0.110 (`d_z`=−2.49, p=0.0020)**,
> PSNR −1.84 dB (p=0.0039), R² −0.0277 (p=0.0117) ⇒ **U-Net 显著更好, 不是打平**。
> 4 折 leave-one-objective-out 的**最小可达 p = 2/2⁴ = 0.125**, 结构上无法判显著 ——
> 该协议问题最对但功效不足, 要靠**增加目标数**而非增加折数解决。
>
> ⚠️ **physics 的最优是 (n_max, lr) 联合点, 单变量扫描会扫错** (10 折 CV 实测):
> `lr=0.02` 单独用 (n_max=15) **毫无改善** (+0.8719 vs +0.8719 基线),
> 但同一个 `lr` 配 `n_max=20` 就是全表最好 (+0.8803 / SSIM 0.7646) ⇒ 非可加性
> 在**有功效的协议下同样成立**, 贪心坐标下降必然失败。配对检验 R² +0.0076
> (d_z=0.79, p=0.023)、SSIM +0.033 (d_z=1.98, p=0.004)。
> ⚠️ 该配置是**在这 10 折上从 5 个候选里挑出来的** ⇒ 带 winner's curse,
> 只能当"提示性"证据; 要确证需嵌套 CV。保守可选 `n_max=20, lr=0.01`。
> ⚠️ **U-Net 的配置没有过拟合到噪声 split**: 在 10 折下复验 5 个候选
> (ep 50/100 × lr 0.005/0.01 × 宽度 16/24), 原选择 [16…256]/50ep/0.01 仍最优,
> 全部候选 R² 只差 0.010、SSIM 差 0.031, 远在 fold-σ 0.06 内。
>
> ⚠️ **三条被推翻的结论 (勿再犯, 两次都因为噪声来源没查清)**:
> (1) "U-Net R² 高 +0.022" —— 单 seed + **不公平预算** (U-Net 只给 25 epoch, 而物理模型第 13 epoch 就收敛)。
> (2) "SSIM 是真实且区间不重叠的差距" —— 同一单 seed; 3 seed 下区间重叠。
> (3) **"两者打平, 无任何指标能区分"** —— **过度纠正**, 是我把 ±0.07 的配比噪声当成了
>     模型差异。真实效应一度被掩盖、又被否认, 两次都源于**未先定位方差来源**。
>     最终结论: U-Net 显著更好 (R² p=0.012, SSIM p=0.002), 且调优 physics 后仍领先。
>
> ⚠️ **hybrid(物理包络 + 零初始化 CNN 残差) 没有补上差距**: 两个协议、5 个指标
> 全部 p ≥ 0.61, 与 physics **统计不可区分**, 却多 45× 参数 ⇒ **不提升为训练入口的
> 模型选项**, 只作为已记录的反面结果留在对比脚本里。
>
> 但两者**用途不可互换**: U-Net 出的是**图像**, 反解成可下发的 SLM 相位是另一个问题;
> 就"预测要下发的修正"而言, 230 参数物理模型是唯一能**闭式给出可实现相位**
> (`Σ Z_k B_k`, 230 个可解释弧度) 的, 且参数少 34000×、墙钟少 40%。
> 完整报告见 [`report/zernike_phase2amp/report.md`](report/zernike_phase2amp/report.md),
> 含 11 节"被推翻结论"存档。旧对比表见
> [`report/zernike_phase2amp/unet_comparison.md`](report/zernike_phase2amp/unet_comparison.md)。

0. **拿 `hwdataset` 的 `image` 当 Zernike 前向模型的 target 时, 必须校准远场尺度。**
   `_anchored_window` (`transforms.py:440`) 取的是**以 0 阶为中心的 `grid×grid` 窗口**,
   它**不缩放** —— 真语料 `slm_zernike_shaping` 的相机窗口是 248 px, 所以 target 只覆盖
   `64/248 ≈ 26%` 的视场。而 FFT 输出若取 `far_field_padding=1` (整孔衍射全展), 预测与
   target 的**角尺度差约 4 倍**, 逐像素不可比, 实测 `Z=0` 时 **R² = −0.20 (比直接预测均值还差)**。
   修法: `far_field_padding = 10` + **中心裁剪**回 `grid×grid`。128 张真样本、`n_max=4`、
   `Z=0` 时的 R² 实测: `pad=1 → −0.20`, `4 → −0.12`, `8 → +0.33`, **`10 → +0.46`**,
   `12 → +0.53`, `16 → +0.49`, `20 → +0.18` —— 明显的**内点最优**, 不是"越锐越好",
   所以它是真正的物理标定。训练后 (25 epoch) **R² = +0.63, PSNR = 22.7 dB**, 且
   **`max|c|` 从失控的 1.77 rad 降到 0.165 rad** —— 标定对了之后小系数就能解释数据,
   不再需要靠巨大相位硬凑 loss。换 family (即换 `fov_px`) 必须**重跑这个 sweep**。

   ⚠️ `ZernikeAmpConfig.far_field_padding` 的 CLI 默认值必须**从 dataclass 读**
   (`train_amp.py::_build_parser`), 不要写字面量: 曾因 CLI 写死 `default=1` 静默覆盖
   标定值 10, 每轮都从 R² = −0.38 起步。

   ⚠️ **逐项单变量扫描的结果不可叠加。** 实测 `far_field_padding=12` 与 `lr=0.1`
   各自单跑都比默认好, 但与 `n_max=11` 组合后反而都变差 (R² 0.794 → 0.753)。
   贪心逐坐标下降在这个问题上不成立 —— **改完必须联合复测**。
   另: `optimizer` 杠杆曾因 `train()` 里硬写 `torch.optim.Adam` 而三个选项
   逐位相同 (白测), 现已改为转发 `cfg.optimizer`; 注意 `weight_decay=0` 时
   `adamw` 与 `adam` **本就等价**, 不是 bug。

1. **相位必须用相干 (复数相量) 块平均, 不能用算术平均。**
   语料里的相位**已被 writer 卷到 [0, 2π)** (整块面板实测 max 6.2605 < 2π = 6.2832)。
   对被卷绕的相位取算术均值, 每一格都偏向 ~π, 结构被完全抹掉; 实测算术均值给
   `[2.82 3.32 2.78 3.29 …]` 而相干恢复 `[0.90 -2.97 1.08 2.98 …]`。
   **输出 `(cos, sin)` 而非 `(angle, contrast)`**: `arctan2` 在 ±π 有分支切口,
   会让 CNN 面对一个输入不连续面。

2. **填充必须填在 cos/sin 上, 不能填在相位上。**
   先给相位补 0 再取 cos, 填充区变成 `cos(0)=1` —— 一个指向 +Re 的单位相量,
   会在**所有边界块**引入约 **+0.42** 的伪 DC 项 (实测 0.414 vs 正确 -0.007)。
   `pupil_phase_to_grid` 正是对 cos/sin 各自补 0。

3. **ROI 默认半径是 500 px, 不是照明半径 450。**
   语料里指令相位的最宽结构达 **半径 479 px** (`slm_pib_*`, sidecar `zernike_radius=480`),
   占可用记录的 69%。450 会整族裁掉。1000×1000 ROI 落在 1200×1920 面板内
   (行 100..1099, 列 460..1459)。**且必须先裁后降采样**: 整面板降到 64 格时
   每格 ~19×30 px, 500 px 光斑只占 ~53×34 格, 其余全是平场本底。
   裁剪框按样本的非零 bbox 自适应是**错的** —— 那会让同一个格子在不同样本里
   对应不同的物理尺度, 卷积网格的对齐就毁了。

4. **`_img` 的 dtype 不是量纲。** 语料同时有 `uint8` / `float32` / `float64` 帧,
   但逐帧实测最大值是 255 (uint8) / 239.6 (float32) / 100.9 (float64) —— **同一个
   0–255 探测器量纲**, 全库无一处超过 255。所以 `/255` 对所有 family 都正确,
   `[0,1]` 截断实际从不触发。把浮点帧当成已归一化会把它压缩最多 255 倍。

### 其他已定决策

- **FOV 不统一, 这是已知口径而非 bug。** 各 family 的相机窗口真的不同
  (64 px `region=32` / 248 / 320 / 1944 px 整幅传感器), 所以同一个输出格对应不同的
  物理角尺度。**不做全局 resize 去掩盖它** (会破坏 0 级对齐), 而是把 `fov_px`
  随样本返回, 供使用方按 family 过滤或分头训练。
- **默认 `image_mode="abs255"`, 不做 peak 归一。** 曝光是模型**输入**, 目标里必须
  保留编码它的绝对亮度。
- **`exposure_ms` 走 log10 再标准化。** 原始线性值跨 2.9 个数量级 (0.1–80 ms),
  直接喂小 CNN 不合适; 统计量在 `__init__` 里由 index 一次算出, 不做逐样本估计。
- **`len(_c)==36` 二义性按 family 裁决。** 36 既是三角 Zernike 数 (n_max=7) 又是
  完全平方 (6×6)。`_FREEFORM_FAMILIES` (`slm_gsnet_square` / `sim_calib_abba`) 里
  平方优先, 其余 family 三角优先 —— 否则 `slm_pib_online` 的 192 条 (占 1.7%)
  会被按完全不同的基重建。
- **排除 `bench_stability` (42 条)。** 平场漂移序列没有指令相位; 用全零相位混进去
  会把"指令平场"和"无指令"混为一谈, 污染 `相位 → 画面` 的监督映射。
- **`FileGroupedSampler` 不是优化而是必需。** 每个 pickle 整体读取, 随机采样会让
  几乎每次 `__getitem__` 都重新反序列化最多 3.42 GB; 按文件分组后每轮每文件只读一次。
- **Windows `spawn` 契约。** `__getstate__` 必须丢掉 LRU 与全部派生数组, 且 `self`
  上不得持有任何打开的文件句柄; 否则每个 worker 都会通过管道复制一份几百 MB。
- **分层说明 (有意为之)。** 本包 import `ao_shaping.runners.gsnet_offline` 的纯函数
  (`infer_n_max` / `reconstruct_pupil_phase_rad` / `pupil_phase_to_grid` /
  `farfield_to_grid`) 而非复刻 —— 本仓库已多次被"重复实现漂移"咬伤。
  `gsnet_offline` 只依赖 `ao_shaping.utils.*`, 且 `runners/__init__.py` 用 PEP-562
  惰性 `__getattr__`, 因此无循环。

### 用法

```python
from ml.hwdataset import (
    build_hw_index, build_hw_dataloader, prepare_hw_cache,
    HwPhaseImageDataset, MaterialiserConfig,
)

index = build_hw_index(index_cache="data/hw_index_cache.json")   # 35 s 首次, 0.09 s 之后

# 一次性预建派生网格缓存, 之后每轮省 212 s Zernike 反演 + 35 s 读盘 (实测 317×)
prepare_hw_cache(index, config=MaterialiserConfig(grid=64))

loader = build_hw_dataloader(
    index,
    config=MaterialiserConfig(grid=64),   # 输入/输出都是 (1, 64, 64)
    batch_size=64,
    num_workers=4,
    use_cache=True,                       # 默认已为 True; 无缓存时逐条回落直接路径
)
for batch in loader:
    phase = torch.cat([batch["phase_cos"], batch["phase_sin"]], dim=1)  # (B, 2, 64, 64)
    exposure = batch["exposure_log10"]                                  # (B, 1)
    target = batch["image"]                                             # (B, 1, 64, 64)
```

```bash
# 语料统计 / 抽样自检 (JSON cache 已建时是瞬时的)
python -m ml.hwdataset.inspect --index-cache data/hw_index_cache.json --sample 5

# 预建派生网格缓存 (消除每轮 212 s Zernike 反演 + 35 s 读盘; 之后 loader 自动读取)
python -m ml.hwdataset.inspect --index-cache data/hw_index_cache.json --build-cache
```

---

## Configuration

### Environment Variables (.env)

Project uses `.env` file for environment configuration:

```bash
# Hardware device IDs
Far_Cam_ID=0
Near_Cam_ID=1

# Optical parameters
IDEAL_SPOT_RADIUS=7
CENTER=577,655

# Library paths
PYTHONPATH=src;libs
PATH=libs\Drv_UDPST\x64\Release;libs\gxipy;${PATH}
```

### VSCode Settings

VSCode settings are configured in `.vscode/settings.json`:
- Python path includes `src/` and `libs/`
- Pytest integration enabled
- Terminal environment variables from `.env`

### Config Module

Centralized configuration in `src/ao_shaping/config.py`:

```python
from ao_shaping.config import DM_N_ACTUATORS, DEFAULTS, PATHS

# Hardware constants
DM_N_ACTUATORS = 64

# Default optimization parameters
defaults = DEFAULTS
print(defaults.WF_EPOCHS)  # 20000

# Path configuration
paths = PATHS
print(paths.root_dir)  # data/
```

---

## Entry Points

**CLI Commands (Click-based):**
```bash
# Via main.py hub
python src/ao_shaping/main.py wf
python src/ao_shaping/main.py pib
python src/ao_shaping/main.py pipeline
python src/ao_shaping/main.py zernike-matrix
python src/ao_shaping/main.py rms-zernike
python src/ao_shaping/main.py ga-zernike
python src/ao_shaping/main.py slm-pib spgd          # SLM Zernike PIB (SLM + CCD)
python src/ao_shaping/main.py spgd-square           # SLM 方形远场整形 (同上模块的 square 子命令)
python src/ao_shaping/main.py combined
```

**CLI Structure (main.py 注册关系):**
```
main (click.group)
├── wf             ← wf_runner.run        [Wavefront RMS via DM电压 + WFS]
├── pib            ← axis_beam_runner.run  [Power-in-Bucket via DM电压 + CCD]
├── pipeline       ← pipeline_runner.run   [Serial WF RMS → PIB]
├── zernike-matrix ← zernike_matrix_runner.run [Zernike响应矩阵标定 + 闭环优化 (closed_loop_run)]
├── rms-zernike    ← rms_zernike_runner.run    [SLM Zernike RMS]
├── ga-zernike     ← ga_zernike_runner.run     [GA Zernike]
├── slm-pib         ← slm_pib_run                [SLM Zernike PIB (spgd / heuristic 子命令)]
├── spgd-square     ← slm_square_run             [SLM 方形远场均匀性整形 (slm_shaping_runner 的 square 子命令)]
├── slm-gsnet      ← slm_gsnet_run             [SLM方形光斑 SPGD 整形 (freeform)]
├── slm-gs-refine  ← slm_gs_refine_run         [GS 预矫正 + 自由相位 SPGD 细化]
├── slm-model-in-loop ← slm_model_in_loop_run   [正向模型闭环校正 + 目标光斑相位合成]
└── combined       ← combined_runner.run       [AdaMOD+SPGD 混合 PIB]
```

**Note:** `combined` 命令仍注册于 main.py (main.py:106) 且功能可用, 作为 legacy 保留。`pipeline_runner.py` 是推荐的 WF→PIB 串行方案。

> **未注册到 main.py 的独立 Runner** (需直接运行 `python -m ao_shaping.runners.xxx` 或 standalone 脚本):
> `slm_offset_runner` (计划迁移至 `tools/slm/`), `hadamard_matrix_runner` (正在注册为 `hadamard-matrix` 命令)。
> ⚠️ `slm_shaping_runner` 曾列于此 (2026-10-05 前未注册), 现已注册为 `slm-pib` + `spgd-square`。

**Refactoring Notes:**
- All runner scripts now use centralized config from `config.py` (DM_N_ACTUATORS, PATHS, DEFAULTS)
- Common CLI helpers moved to `utils/cli_helpers.py` (parse_tuple, setup_coredumpy)
- Duplicate code eliminated across runner files

---

## Build/Lint/Test Commands

### Running Tests

```bash
# Run all tests
pytest

# Run all tests with verbose output
pytest -v

# Run tests with coverage
pytest --cov=ao_shaping

# Run a specific test file
pytest tests/ao_shaping/utils/test_spots_calc.py

# Run a specific test class
pytest tests/ao_shaping/utils/test_spots_calc.py::TestCentroid

# Run a specific test function
pytest tests/ao_shaping/utils/test_spots_calc.py::TestCentroid::test_centroid_uniform

# Run tests matching a pattern
pytest -k "spots_calc"
pytest -k "test_centroid"

# Run with live output (no capture)
pytest -s

# Run with specific markers
pytest -m "not slow"
```

### Environment Setup

```bash
# Create virtual environment with uv
uv venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
uv add -e .
```

---

## Code Style Guidelines

### Python Version
- **Python 3.12+ required** (see `pyproject.toml`)

### Imports

**Ordering (PEP 8 standard library ordering):**
```python
from __future__ import annotations  # Future imports first

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, ClassVar

import numpy as np

from loguru import logger
```

**Avoid relative imports in package code:**
```python
# Good
from ao_shaping.drivers import MIICamera, DahengCamera
from ao_shaping.utils.image.spots_calc import centroid

# Avoid (unless necessary)
from .drivers import ...
```

### Type Hints

**Use modern type hints with `|` syntax (Python 3.12+):**
```python
def set_parameter_value(self, name: str, value: Any) -> bool:
    min_value: float | None = None
    error_message: str | None = None
```

**Return type hints on all public methods:**
```python
def get_parameter_value(self, name: str) -> Any:
    pass

def is_connected(self) -> bool:
    pass
```

**Generic types:**
```python
from typing import TypeVar

T = TypeVar('T')

def get_item(self, key: str) -> DeviceParameter | None:
    return self._parameters.get(key)
```

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Classes | PascalCase | `DeviceBase`, `NLightDM` |
| Functions | snake_case | `calculate_sharpness`, `get_centroid` |
| Variables | snake_case | `exposure_time_ms`, `dm_unit_mask` |
| Constants | SCREAMING_SNAKE | `MAX_VOLTAGE`, `DEFAULT_THRESHOLD` |
| Private attrs | _leading_underscore | `_device_id`, `_parameters` |
| Type vars | PascalCase | `T`, `T_co` |

### Data Classes

Use `@dataclass` for structured data containers:
```python
@dataclass
class DeviceParameter:
    name: str
    value: Any
    value_type: type = float
    min_value: float | None = None
    max_value: float | None = None
    unit: str = ""
    description: str = ""
    writable: bool = True
```

### Enums

Use `Enum` with `auto()` for state/type definitions:
```python
class DeviceState(Enum):
    UNKNOWN = auto()
    DISCONNECTED = auto()
    CONNECTING = auto()
    READY = auto()
    BUSY = auto()
    ERROR = auto()
```

### Error Handling

**Custom exceptions with `*Error` suffix:**
```python
class DeviceError(Exception):
    pass

class DeviceNotFoundError(DeviceError):
    pass

class DeviceBusyError(DeviceError):
    pass
```

**Context manager support for resources:**
```python
class BaseDM(ABC):
    @abstractmethod
    def open(self) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    def __enter__(self) -> "BaseDM":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
```

**Usage:**
```python
with NlightDM() as dm:
    dm.send_voltages(vs, 0.1)
# Automatically closed
```

**Graceful exception handling:**
```python
try:
    import cupy as cp
    CUPY_AVAILABLE = cp.cuda.is_available()
except (ImportError, AttributeError):
    CUPY_AVAILABLE = False
```

### Docstrings

Use docstrings for public APIs (Google style):
```python
def validate(self, value: Any) -> bool:
    """Validate if value is within allowed range.
    
    Args:
        value: The value to validate.
        
    Returns:
        True if valid, False otherwise.
    """
    pass
```

### Logging

Use `loguru.logger` (configured in pyproject.toml):
```python
from loguru import logger

logger.debug(f"Device {self._device_id} initialized")
logger.info("Starting optimization")
logger.warning(f"Invalid value {value} for parameter '{name}'")
logger.error(f"Device {self._device_id} error: {error_msg}")
```

### Hardware Drivers

**Required interface for all drivers:**
```python
class Device(ABC):
    @abstractmethod
    def open(self) -> None:
        """Open connection to the device."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close connection and release resources."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if device is connected and ready."""
        pass
```

**State tracking:**
- Use `DeviceState` enum for state management
- Use `_set_state(state, error_msg)` helper method
- Track `self.is_open` or similar for connection state

### Performance-Critical Code

**Numba JIT compilation:**
```python
@numba.njit(cache=True)
def calculate_sharpness_numba(img: np.ndarray):
    # JIT-compiled code here
    pass
```

**NumPy as default, provide alternatives:**
```python
def calculate_sharpness(img: np.ndarray):
    # NumPy version (default)
    pass

def calculate_sharpness_numba(img: np.ndarray):
    # Numba-accelerated version
    pass

def calculate_sharpness_cupy(img: cp.ndarray):
    # CuPy GPU version
    pass
```

---

## Environment Variables

Configuration via `.env` file:
```
Far_Cam_ID=0
Near_Cam_ID=1
IDEAL_SPOT_RADIUS=7
CENTER=577,655
```

Access in code:
```python
import os
cam_id = int(os.environ.get('Far_Cam_ID', 0))
```

---

## Configuration

Pytest configuration in `pyproject.toml`:
```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
```

VS Code settings in `.vscode/settings.json` set PYTHONPATH to `src` and `libs` directories.

---

## ANTI-PATTERNS (THIS PROJECT)

| Pattern | Forbidden Because |
|---------|------------------|
| Relative imports in package | Use `from ao_shaping.xxx import yyy` instead of `from .xxx import yyy` |
| `as any`, `@ts-ignore` | Never suppress type errors |
| Empty catch blocks | Always handle exceptions or log |
| Deleting failing tests | Fix the code, not the test |
| `combined_runner.py` with main CLI | Prefer `pipeline_runner.py` for new serial flows; `combined` remains registered as legacy (main.py:106) |
| Passing uint16 grayscale through `create_phase_from_array()` | `create_phase_from_array()` treats input as **radians** (mod 2π → grayscale = rad/2π × 1023). uint16 grayscale values get silently corrupted. Use `np.full((h,w), gray, dtype=np.uint16)` for flat phase or direct grayscale patterns. |
| SLM 相位生成函数自行 `np.mod(phase, 2π)` (2026-09 raw-only 契约) | All phase generators must return **raw unwrapped radians** — the only mod-2π wrap lives in the driver `Santec.create_phase_from_array()` on radian→grayscale conversion (`santec/driver.py` L1382). Generators that self-wrap duplicate the driver contract and hide the true phase. Convert via `utils/slm/phase_display.phase_to_slm_grayscale(phase, slm=slm)` (hardware) or the pure fallback (offline/tests). Exception: `_zernike_phase_radians` keeps a wrapped output **only** as a test-only reference. |
| Consecutive `write_phase` + `display_memory` to the **same** memory slot | Santec SLM firmware treats `display_memory(slot)` as a no-op when that slot is already being displayed — the LCOS panel does **not** refresh. Consecutive writes must ALWAYS target different slots. Preferred pattern (used by diff-shaping runner): pick a **random slot in 2..125** each write, excluding the currently displayed slot — this also survives process restarts (`get_displayed_memory_number()` before the first write). Older tools rotate a small pool like `itertools.cycle([3,4,5])` — works within one process only. The built-in `display_data()` cycles through all 127 slots. |
| Opening the SLM in DVI mode (`video_mode=1`) for diagnostics | `Santec(..., video_mode=1).open()` can **hang** (observed 120s/300s timeouts), and a hung controller then also hangs memory-mode `open()` until a **physical power cycle**. Never auto-try DVI mode; use memory mode (`video_mode=0`) only. See `tools/slm/slm_diagnose.py` |
| Treating `get_displayed_memory_number` error-code 1 as a fault | In `set_grayscale` mode there is no memory slot being displayed, so `SLM_Ctrl_ReadDS` returns error-code 1 — this is **normal**, not a failure. Slots only exist in memory mode. |
| Re-writing a **cached displayed phase** back through `write_phase` | Cached phases (`get_displayed_phase()`) already contain base overlay + wavefront correction; `write_phase` re-applies both → corrected **twice**. Rewriting shifted phases must go through the driver-level `Santec.apply_shift()` (raw write via `_write_to_memory` + slot rotation + config save), which is the single public entry point for shift-and-redisplay (identical semantics to the multi-SLM controller "应用平移" button). Shift math's only implementation is the static `Santec.shift_phase()`. See `src/ao_shaping/drivers/slm/AGENTS.md`. |
| Assuming the 0-order spot sits at the camera frame center | In the 2f Fourier bench the optical axis (0-order = frame **global maximum**) lands at the camera center only by luck. Observed: frame center (1344,760) vs 0-order spot (1441-1443, 705-706). Always locate 0-order by `argmax`, never by geometry. |
| diff-shaping 硬件闭环挂起（日志止于 `成功打开SLM #1`） | 2026-09-08 实测 (与相机无关): SLM open() 成功后, 首次 `camera.get_numpy_image()` 前无任何日志输出即无限阻塞 (900s 超时被强杀; 分步探针脚本同挂)。`WaitImageV3` 的原生等待由 SDK 内部驱动, 不受 Python 侧超时保护。处置: 强杀后先确认无残留 python 进程 (Get-Process python*)；重跑前 SLM memory 模式 open() 若超过数秒无日志, 对 SLM 控制器物理断电重置 (与 DVI 挂起同一处置)。见 `diff_shaping_runner.py` SLM 连接段注释与 `report/slm/slm_shaping_diff/readme.md` 故障排查。 |
| Generating SLM phase via **min-max normalisation** | It **min-max normalises** the phase (`(p-pmin)/(pmax-pmin)*1023`) instead of `mod 2π` radians→grayscale, making the pattern **scale-invariant** (coefficients ×1 and ×4 produce byte-identical patterns; verified `np.array_equal == True`). Always convert radian phase through the SLM driver `slm.create_phase_from_array()` (2π=993 + wavefront correction + LUT). See `docs/slm_square_spgd/README.md`. ⚠️ 历史例子: `PatternHelper._zernike_to_uint16` 与 `ZernikeDM.generate_phase` —— **两者已于 2026-09 修复/删除**, 本行保留为反模式记录。 |
| **`ZernikeDM.generate_phase` min-max normalises the phase** (same anti-pattern, still live) | `drivers/dm/zernike_dm.py:106-129` does `(raw−min)/(max−min)` then `×2π` (rad) / `×max_val` (gray) → output is **scale-invariant**: coefficients ×1 and ×4 give **byte-identical** phases (verified `np.array_equal == True`, PV always 2π). Since `ZernikeSLM.send_zernike` routes through it, **the Zernike coefficient amplitude is uncontrollable** for every consumer (`zernike-matrix`, `rms-zernike`, `ga-zernike`, `greedy-zernike`, `runners/zernike_matrix_runner.py`, `runners/rms_zernike_runner.py`, `gui/zernike/`). Fix: drop the normalisation (treat coefficients as radians) — note this intentionally changes those callers' behaviour. See `report/slm/report2.md` §2.3. |
| **Mixing units across the WFS→SLM correction loop** | The WFS `get_zernike()` returns **µm**; the response matrix must be built in the **same unit** as the correction's `w` (λ). Two real bugs (2026-09-16): matrix built in µm vs `w` in λ → coefficients inflated **1/0.532 = 1.88×**; and the solved coefficients are in **λ (waves)** but `make_phase`/`generate_zernike_polynomial` take **radians** → applied phase shrunk **2π = 6.28×**. Both fixed → closed-loop RMS improvement 13.8% → **42.1%**. Always route through `tools/slm/slm_zernike_common.um_to_waves()` and multiply λ→rad by `2π` before `make_phase`. See `report/slm/report2.md` §0. |
| **Full-frame centre leaking into window-local metrics** (`slm_zernike_pib.py`) | `cam.reset_window(...)` returns the window centre in **full-frame sensor coordinates**, but every downstream metric (`rms_pib_terms`, `ImageTargetFunc.radius`, `spot_waist_sigma`, `reference_center`) operates on the **re-windowed** image and needs **window-local** coordinates. Hardware-verified bug (2026-09-23): the runner's `-c auto` probe locates the 0-order on the un-windowed camera (e.g. (633,934) on 1944×2592) → optimizer used it directly as window-local → every metric read **0** (pib=0, r_bucket pinned at 0.9 = 1px radius × shrink 0.9) across the whole heuristic matrix. Fix: after capturing the re-windowed `init_img`, re-locate `reference_center = zero_order_center(init_img)` and use `reference_center` (NOT the post-`reset_window` `center`) for the r-bucket radius too (L1396/L1441). Rule: whenever a centre is handed to a metric/ROI builder, verify which coordinate frame it belongs to first. |
| Square shaping with low-order Zernike (n≤4) | Zernike modes are a **circularly symmetric smooth** basis; they physically cannot synthesise a square far-field (needs 2D-sinc-like near field / high spatial frequencies). Use full-pixel phase freedom (GS / differentiable / free-form), not Zernike. |
| Optimising `-CV` alone as the SPGD objective for square shaping | With no energy term the optimizer **empties the target box** to minimise CV (hardware observed EE→0.002). The objective must include encircled energy (use the combined quality score). |
| Computing `PIB` / `CV` on a **raw** CCD frame (read noise not removed) | A CCD frame carries symmetric read noise, so ~half its pixels are negative (measured 7164/14400 on a sim frame). `power_in_bucket` divides the in-target sum by the **whole-frame** sum, so a negative background drives the ratio **above 1** (measured `PIB=1.0120`) and corrupts `CV` the same way — the optimiser then chases read noise. Subtract the frame median and clip at 0 (`slm_gs_refine._prepare_frame`); clipping the *raw* frame instead rectifies noise into a pixel-count-sized DC pedestal (see the `clip(·,0,None)` defect in `sim/AGENTS.md`). |
| Re-locating the target ROI by `argmax` on **every** iteration | On a speckle field the global maximum hops between near-equal grains under a ~1e-3 perturbation, so an `argmax`-rolled box makes `PIB`/`CV` **discontinuous** and the optimizer chases a box that no longer covers the beam (documented on `compute_metrics`). Locate the 0-order **once** on the unshaped frame and freeze the ROI for the whole run. |
| Committing a GS warm-start phase to hardware without a bake-off | The GS phase comes from a *model* of the bench (aperture, focal length, camera pixel pitch). If any input is wrong, GS makes the real spot **worse**. Measure flat and GS and keep the better one — a mis-calibrated model then costs two measurements instead of wrecking the run (`slm_gs_refine` Stage 1). |
| Treating an exact run-to-run reproduction as a property of a closed loop | The seed pins the SPGD perturbation signs, **not** the measurements: every far-field read carries device noise. Two same-seed runs disagree at ~1e-3 *at the flat baseline*, before any optimisation. Assert approximate reproducibility, never equality. |
| Trusting `reset_window()`'s returned centre | When the spot is near the frame edge the ROI offset is clamped but the returned `(w//2, h//2)` is not the true spot position → the target box lands off the beam (hardware observed epoch-0 `mean_b=0.01`). Re-locate the spot by `argmax`/centroid on the **windowed** image. |
| Function-only optimizers in ao_shaping/algorithm (no class API) | New optimizers must expose __init__ (validation + state) + update() (one step) + optional run() (result dataclass); one-shot functions are legacy/thin wrappers only. See src/ao_shaping/algorithm/README.md. |
| Placing markdown/report **generation** under `src/ao_shaping/tools/` | **All markdown/illustrated-report generation MUST live in `scripts/`** (naming: `scripts/generate_*_report.py`, e.g. `generate_zernike_wfs_report.py`, `generate_diff_shaping_report.py`). `src/ao_shaping/tools/` is reserved for hardware-interaction tools (CLI + driver orchestration), not report writers. See scripts/README.md. |
| Report generation inside `algorithm/` | `signal_processing/beam_shaping_benchmark` writes CSV/MD/GIF reports — report generation MUST live in `scripts/` (see scripts/README.md), not in the algorithm layer. |
| Pygame/viz code inside `utils/` | Visualization code belongs in `display/`, not the leaf utils layer. **Done for `gs_visualization`** (moved to `display/gs_visualization.py`, 2026-10-03; re-exported from `ao_shaping.display`). **Still open for `utils/image/display.py`** — it is the package's own matplotlib/pygame renderer and needs the same move plus a re-point of `utils/__init__.py`'s `ImageVoltagesDisplay` / `plot_funcs` / `VOLT_HEIGHT` … re-exports. |
| Re-implementing the GS loop | `gs_visualization.gerchberg_saxton_with_visualization` must drive the canonical `gerchberg_saxton` via a callback, not duplicate the loop. |
| Dead vendored ctypes VISA code | `utils/wavefront/vi.py` is dead vendored ctypes VISA code — remove it or move it out of utils. |
| Min-max scale-invariant phase conversion | Min-max normalising the phase makes patterns scale-invariant (coefficients ×1 and ×4 → byte-identical). Use `utils/slm/phase_display.phase_to_slm_grayscale(phase, slm=slm)` instead. ⚠️ 历史例子 `PatternHelper._zernike_to_uint16` 已删除。 |
| Vector-beam demo `sim.py` living in `algorithm/` | The vector-beam demo `sim.py` lives in `algorithm/` — it belongs in `scripts/`. |
| `utils/slm/pattern_helper.py` importing `from ao_shaping.algorithm.phase_wrap` at module top level | utils is the leaf layer and must not depend on `algorithm/` at import time — use deferred function-local imports. |
| 脚本/runner/tools/GUI 内**重复实现 Zernike 数学** (Noll↔(n,m) 查表、多项式求值、相位生成) | Canonical 入口唯一: `utils/wavefront/zernike_utils.py` (API 层) + `utils/wavefront/zernike_calc.py` (引擎层), 单向依赖 `zernike_utils → zernike_calc`。2026-09 去重重构前 `gui/slm/`、`optimizer/wfless/`、`tools/slm/` 各有一套, 已全数收敛。新代码一律 `from ao_shaping.utils.wavefront.zernike_utils import ...`, 禁止 `import aotools`/自写 `RZern`/自写 `noll2nm` 表。详见 README `## Zernike 使用指南` 与 AGENTS `## Zernike 使用规范`。 |

> 方形光斑 SPGD 整形的完整分析、硬件实测与修复记录见 [`report/slm/slm_square_spgd/README.md`](report/slm/slm_square_spgd/README.md)。

---

## UNIQUE STYLES

- **Mock-first testing**: Tests use simulation classes (`SimTurbulenceAOEnv`, `sim_spgd`) to avoid hardware
- **Zernike Noll 约定统一** (aotools Noll 1976): Noll 4 = (2,0) defocus, Noll 5 = (2,-2) astig, Noll 11 = (4,0) spherical, Noll 13 = (4,-2)。`optimizer/wf/` 三个优化器 (ga_zernike / greedy_zernike / rms_by_zernike) 的模式枚举与 Noll 索引→(n,m) 映射已全部使用 canonical `zernike_calc.noll_to_nm()` / `zernike_calc.zernike_modes()`; 历史 legacy 硬编码查表 (`noll_to_nm_legacy`, 那里 Noll 5 = (2,0)) 已删除, 不再存在。新代码一律用 `zernike_calc.noll_to_nm()` / `zernike_utils.list_zernike_modes()`, 勿混用两套索引。zernike_utils 模块文档含完整前 15 阶映射表。
- **Zernike 纯数学 canonical 两层入口** (2026-09 去重重构): API 层 `zernike_utils.py` (`parse_zernike_coefficients` / `generate_zernike_phase` / `list_zernike_modes` / `coefficients_to_array` / `um_to_waves`) → 引擎层 `zernike_calc.py` (`ZernikeGenerator` 网格缓存 / `noll_to_nm` / `zernike_modes` / `noll_indices`)。生成器输出 **raw 未包裹弧度**, 弧度→灰度统一走 `phase_display.phase_to_slm_grayscale(phase, slm=slm)`。任何脚本/runner/tools/GUI 禁止自写第二套 Zernike 数学 — 见 `## Zernike 使用规范` 节与 README `## Zernike 使用指南`。
- **Hardware skip pattern**: Tests requiring physical hardware use `pytest.skip("Requires DM hardware")`
- **Recorder pattern**: Optimization tests validate history dictionaries with expected fields
- **Optional backend testing**: CuPy/Numba tested conditionally with try/except guards
- **No fixtures**: No `conftest.py`, fixtures defined inline in test methods
- **SLM flat-phase gray RAW path**: Always send raw uint16 grayscale values to SLM via `np.full((h,w), gray, dtype=np.uint16)`. Never route flat phase through `create_phase_from_array()` (radian conversion). The SLM has amplitude coupling: different flat-phase gray levels produce different camera intensities at 1064nm (periodic with 2π ≈ 993 gray). Use `scripts/validate_flat_phase_gray.py` to verify. Parameters for stable observation: `--exposure-ms 0.8 --wait-time-s 0.3 --discard-count 3`.
- **SLM phase raw-only contract + unified grayscale entry** (2026-09): Every SLM phase generator (optimizer/runner layer) returns **raw unwrapped radians**; do NOT `mod 2π` in generators — the driver `create_phase_from_array()` is the single wrap point (radian→grayscale, `santec/driver.py` L1382). Convert radian phase to grayscale through `utils/slm/phase_display.phase_to_slm_grayscale(phase, slm=slm)` — passes a live SLM it delegates to the driver pipeline (grayscale + wavefront correction + LUT + shift, 2π = device `_max_gray`); `slm=None` (simulation/offline save/unit tests) falls back to the built-in pure math conversion (wrap→scale→clip→uint16, default 1023). Counter-example **history** (both since fixed/removed in 2026-09 — do NOT reintroduce): `PatternHelper._zernike_to_uint16` (**deleted**, `pattern_helper.py:499-500`) and `ZernikeDM.generate_phase` (**fixed**, `zernike_dm.py:117-118`). A live scale-invariant min-max normaliser remains a defect; the two names above no longer exist in the tree.
- **SLM memory-slot rotation**: When writing consecutive phases to memory mode, always target **different** slots — calling `display_memory(slot)` for the slot already displayed is a no-op and the LCOS panel will not refresh. Preferred pattern (diff-shaping runner): random slot in **2..125**, excluding the currently displayed one (`get_displayed_memory_number()` on start, survives process restarts). Older tools rotate a small pool (`itertools.cycle([3,4,5])`) — in-process only. `display_data()` cycles all 127 slots internally.
- **Streamlit background threads**: Any background loop that drives hardware must never touch `st.session_state` directly — it causes `missing ScriptRunContext` warnings and race conditions. Use the R50 `run_loop` pattern: pass a snapshot of parameters as a plain `dict`, communicate state changes via `threading.Event` for stop signals and mutable containers (e.g. `list`) for live-updated values like frequency, and drain feedback through a `queue.Queue`. The main thread reads/writes `st.session_state` only on rerun. See `src/ao_shaping/gui/r50/r50_voltage_send.py:run_loop` and `src/ao_shaping/gui/slm/multi_slm_controller.py:_toggle_phases_task` for implementations.
- **Timing in background loops**: Prefer `time.time()` wall-clock deltas over counters for state machines. Example: `int(elapsed * 2.0 * freq) % 2 == 0` toggles at exactly the requested frequency without drift, instead of sleeping fixed half-periods and accumulating error.
- **2f Fourier bench geometry** (SLM front focus → f=125mm lens → CCD back focus): the lens Fourier-transforms the SLM field, so CCD coordinates represent **spatial frequency** — the +1 orders of an upper-half-grating and a lower-half-grating land on the **same CCD row (optical-axis row)**, differing only in x-offset. Expected diffraction offset `Δx_px = λ·f/(d_SLM·p_cam) ≈ 5021/Λ` (P64→78px, P32→157px, P96→52, P40→126). Do NOT assume half-screen gratings separate in y. 0-order = frame global max (`argmax`), re-check its location after any bench change.
- **SLM panel "not modulating" diagnostic chain** (2026-09, encoded in `tools/slm/slm_diagnose.py`): ① freeze check — write flat/full-grating/top-half/bottom-half to **rotated memory slots**, frames must differ (all-identical ⇒ LCOS frozen); ② modulation check — `set_grayscale` sweep 0..1023, 0-order bucket must vary with ~993-gray period (flat ⇒ no amplitude coupling ⇒ panel not modulating); ③ linearity check — exposure ×4, ×20 must grow peak brightness (constant peak incl. at 0.1ms/2ms ⇒ light is >100× weaker than the known ~0.02ms near-saturation baseline). Verdict thresholds: freeze ≥2/3 frames differ, modulation bucket rel-spread >15%, linearity growth >2×.
- **Class-based optimizer convention**: optimizers in ao_shaping/algorithm expose __init__ (validate + set state), update() (one step → next state), optional run() → result dataclass; the one-shot function stays a thin wrapper. Simulation-first (torch/numpy) tests before hardware. Canonical example: DifferentiableBeamOptimizer. See src/ao_shaping/algorithm/README.md.
- **Report generation lives in `scripts/`**: any functionality that writes markdown/illustrated reports MUST live in `scripts/` (naming `generate_*_report.py`, e.g. `generate_zernike_wfs_report.py`, `generate_zernike_response_matrix_report.py`, `generate_diff_shaping_report.py`) — NEVER under `src/ao_shaping/tools/`, which is reserved for hardware-interaction tools (CLI + driver orchestration). Scripts follow the repo script conventions: `matplotlib.use("Agg")` BEFORE importing pyplot, `sys.path` bootstrap (`ROOT = Path(__file__).resolve().parents[1]`), CJK font rcParams (`Microsoft YaHei`/`SimHei` + `axes.unicode_minus = False`), output under `report/<topic>/` or `logs/<timestamp>/`, and MUST be documented in `scripts/README.md`. Prefer **offline** report generators (read saved artefacts) so a report can be regenerated without hardware.
- **`docs/` vs `report/` (2026-10-05 拆分, 红线)**: 文档分两类，**不可混放** —
  - `report/<topic>/` = **实验报告**：一次实验/基准/仿真的**结论**（测量值、参数、口径、被推翻的假设）。原先在 `docs/` 下，2026-10-05 全部迁到仓库根的 `report/`。
  - `docs/` = **设备说明文档**：设备规格、SDK/驱动用法、装配/操作手册、图集，外加**每设备硬件测试报告** `<device>/<device>_report.md`（由 `tests/ao_shaping/utils/test_report.py` 的 `TestReport` 写出，位置绑定该 harness 默认值与 `test_conventions.py` 的 tracked-docs 守卫，故**不迁**）。
  - **每份报告必须能追到生成它的脚本**：`scripts/_common/provenance.py::REPORTS` 是**唯一映射源**，渲染出每份报告的 `生成脚本 / 复现命令 / 运行环境` 溯源块，以及 `report/README.md` 总索引。改动报告或脚本后跑 `python scripts/sync_report_provenance.py`（幂等）；`--check` 只校验。**手工编辑溯源块或索引 = 会被 `tests/ao_shaping/scripts/test_report_provenance.py` 判失败。**
  - **代码注释不承载实验叙述**：实测数字/踩坑结论写进对应 `report/`，代码处只留**不变量 + 报告链接**（例如 `ZernikeDM` 的系数即弧度 → `report/slm/report2.md`；读出噪声致 `PIB>1` → `report/slm_pib_bench/report.md` §7）。函数 docstring 仍保留**契约**（这个函数做什么、为什么这样裁剪），只把长实测叙述外迁。

---

## MISSING INFRASTRUCTURE

- **No CI/CD**: No GitHub Actions, no automated testing on push
- **No linting**: No ruff/mypy/flake8 configured
- **No pre-commit**: No hooks for lint/format before commits
- **No requirements.txt**: Only `pyproject.toml` and `uv.lock`

Consider adding: `.github/workflows/ci.yml`, `ruff.toml`, `.pre-commit-config.yaml`

---

## CodeGraph (Pre-indexed Knowledge Graph)

This project has CodeGraph initialized (`.codegraph/` exists, 6,111 nodes, 12,339 edges).

### For Explore agents
Use `codegraph_explore` as your PRIMARY tool — it returns full source code sections from all relevant files in one call.

**Rules:**
1. Follow the explore call budget in the `codegraph_explore` tool description.
2. Do NOT re-read files that `codegraph_explore` already returned source code for.
3. Only fall back to grep/glob/read for files listed under "Additional relevant files" if you need more detail.

### For the main session
Only use these lightweight tools directly:

| Tool | Use For |
|------|---------|
| `codegraph_search` | Find symbols by name |
| `codegraph_callers` / `codegraph_callees` | Trace call flow |
| `codegraph_impact` | Check what's affected before editing |
| `codegraph_node` | Get a single symbol's details |
| `codegraph_context` | Build relevant context for a task |
| `codegraph_files` | Get indexed file structure |
| `codegraph_status` | Check index health

<!-- CODEGRAPH_START -->
## CodeGraph

In repositories indexed by CodeGraph (a `.codegraph/` directory exists at the repo root), reach for it BEFORE grep/find or reading files when you need to understand or locate code:

- **MCP tool** (when available): `codegraph_explore` answers most code questions in one call — the relevant symbols' verbatim source plus the call paths between them, including dynamic-dispatch hops grep can't follow. Name a file or symbol in the query to read its current line-numbered source. If it's listed but deferred, load it by name via tool search.
- **Shell** (always works): `codegraph explore "<symbol names or question>"` prints the same output.

If there is no `.codegraph/` directory, skip CodeGraph entirely — indexing is the user's decision.
<!-- CODEGRAPH_END -->


