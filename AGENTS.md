# AGENTS.md - AO-Shaping Development Guide

**Generated:** 2026-06-20

## 🔴 术语规范（所有 AI 工具必须先读）

本仓库**所有**文档写作、改写、摘要、翻译，以及任何 AI 工具（Claude Code /
Codex / Cursor / Copilot / Hermes / 子代理）生成的中文叙述，**必须**遵守
[术语规范表 `docs/terminology_glossary.md`](docs/terminology_glossary.md)。

- **单一事实源**：该表是 AO-shaping 术语的唯一权威。左列"口语/非规范"一律
  禁止，右列"规范术语"必须使用（如 `台架→实验系统`、`实机/真机→实验`、
  `平场泄漏→加性背景（杂散光）项`、`瞳孔→瞳面`、`跑通→在线实现`、
  `打散→弥散`、`金标准→参考（真值）`、`bake-off→对比择优`、`留出→验证集`）。
- **只改中文叙述**，绝不改代码标识符 / 围栏代码块 / mermaid / 行内
  `` `code` `` / 纯英文叙述 / 厂商型号 / 数值参数 / 生成文档
  （`docs/dev/call_graph_wiki/**`）。
- **首现夹注**：`pupil（瞳面）` 在中文叙述中首现加一次中文释义，不得双重夹注。
  `bake-off` 自 2026-10-17 起中文叙述一律用**对比择优**，不再夹带英文。
- **数据守恒**：术语替换是纯措辞改写，数值/单位/公式/参数/路径/DOI/引用编号
  逐字节保留。
- **提交前校验**（必须全绿）：
  `python3 scripts/_common/term_unify_20261016.py`（dry-run，输出 `0 个文件`）；
  `git grep -E 'pupil（瞳面）（|bake-off（对比择优）|bake-off（强制对比择优）'` 须为空；
  残留 `平场泄漏|台架|实机|真机|上机|瞳孔|跑通|打散|灌进|钻空子|金标准|围栅|真实硬件`
  的命中须全部落在术语表 §2 的 DO-NOT-TOUCH 集合内。
  残留 `臂`（T-65）的命中只允许落在术语表 §1.9 / §3 的登记行与
  `scripts/_common/term_unify_20261016.py` 的规则表里；正文一律「变体」
  （A/B 对照语境）或「两个仿真后端分支 / 仿真后端分支」（numpy / OOPAO 两个仿真
  后端分支语境）。
- **新词须登记**：出现未收录的口语词，先在术语表 §3 登记（T-60 起）再替换。

> 详细规则、150+ 条对照、保留清单见 [`docs/terminology_glossary.md`](docs/terminology_glossary.md)。
> 自动统一脚本：[`scripts/_common/term_unify_20261016.py`](scripts/_common/term_unify_20261016.py)。

## Project Overview

AO-Shaping is an Adaptive Optics (AO) system using reinforcement learning for wavefront correction and beam shaping. It integrates multiple optimization algorithms including WFS-based and wavefront-sensorless methods.

## Project Structure

```
AO-shaping/
├── src/
│   ├── ao_shaping/          # Main package
│   │   ├── main.py              # CLI entry point (Click-based)
│   │   ├── runners/             # Runner scripts package
│   │   │   ├── __init__.py          # 惰性 re-export (_LAZY_RUNNERS / R-27, PEP-562)
│   │   │   ├── runner_common.py     # 共享辅助 (resolve_dm / patch_sim_*_shaping)
│   │   │   ├── matrix_runner.py     # dm-matrix + hadamard-matrix (DM 响应矩阵标定)
│   │   │   ├── zernike_search_runner.py  # ga-zernike
│   │   │   ├── nlight_dm/           # DM 家族 runners
│   │   │   │   ├── wf_runner.py         # Wavefront RMS optimizer
│   │   │   │   ├── axis_beam_runner.py  # PIB optimizer
│   │   │   │   ├── pipeline_runner.py   # Serial WF→PIB pipeline
│   │   │   │   └── combined_runner.py   # AdaMOD+SPGD 混合 PIB (legacy)
│   │   │   ├── micro_drive/         # R50Power (Micro-DM) 电压驱动
│   │   │   │   └── voltage_runner.py    # alt-voltage / full-voltage
│   │   │   └── slm/                 # SLM 光路 runners
│   │   │       ├── slm_shaping_runner.py    # slm-pib 命令组 (spgd / heuristic / square)
│   │   │       ├── gsnet_runner.py          # slm-gsnet
│   │   │       ├── gs_refine_runner.py      # slm-gs-refine
│   │   │       ├── model_in_loop_runner.py  # slm-model-in-loop
│   │   │       ├── model_in_loop_session.py # 硬件 session 预检 (入射定位 / 曝光)
│   │   │       ├── rms_zernike_runner.py    # rms-zernike
│   │   │       ├── zernike_matrix_runner.py # zernike-matrix + closed-loop
│   │   │       └── offset_runner.py         # 未注册 (计划迁 tools/slm/)
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
│   │   │   ├── gsnet/         # GSNet: GS + SPGD 整形训练
│   │   │   ├── gsnet_debug/   # GSNet 调试/离线评估
│   │   │   ├── phase/         # U-Net+GAN 相位预测 (dataset, discriminator, trainer, unet)
│   │   │   ├── hwdataset/     # 硬件相位→相机图像 DataLoader (index/transforms/records/dataset/cache/inspect/zernike_dataset)
│   │   │   ├── zernike/       # Zernike forward/inverse model (models, forward_model, train_coeff, train_amp, losses, metrics, augment, dataset, inverse_design, eval_stats)
│   │   │   ├── train.py       # ML 训练入口
│   │   │   └── wandb_logger.py  # WandB integration
│   │   ├── tools/                # Standalone tools (SLM phase capture, Micro-DM image collection, slm_train_data_collect)
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
└── scripts/                       # Utility scripts, grouped by what they process
    ├── slm/ zernike_bench/ dm/ wfs/    # device domains (SLM bench / Zernike calib / DM / WFS)
    ├── ml/ rl/ oopao/ build/            # data+model / RL+turbulence / OOPAO backend / build
    └── _common/ call_graph/ dm_sim/ tuning_devices/   # shared helpers + pipelines + MATLAB
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
| SLM Zernike PIB / 方形整形 (同一模块) | `src/ao_shaping/optimizer/wfless/slm_zernike_pib.py` + `slm_square_shaping.py` + `runners/slm/slm_shaping_runner.py` | `slm_shaping_runner` 一个 click 组装下两个家族: `slm-pib`(`spgd`/`heuristic`) → `slm_zernike_pib.py`; `slm-pib square`(`square` 子命令，不是顶层命令) → `slm_square_shaping.py`。`--cam-type sim` 在两半都走数字孪生 (`runner_common.patch_sim_*_shaping`) |
| SLM方形光斑整形 (SPGD, freeform) | `src/ao_shaping/optimizer/wfless/slm_square_shaping.py` + `runners/slm/gsnet_runner.py` | SPGD 优化 Zernike 系数 → 均匀方形远场 (CLI: `slm-gsnet`) |
| GS 预整形 + 自由相位 SPGD 细化 (硬件) | `src/ao_shaping/optimizer/wfless/slm_gs_refine.py` + `runners/slm/gs_refine_runner.py` | 仿真 `iterative_zernike_shaping.py` 的硬件移植: GS 开环预矫正 (仅当实测优于平场才采用) + 无感知 SPGD 细化 (CLI: `slm-gs-refine`) |
| 物理预测模型闭环校正 + 反复迭代 (硬件) | `src/ao_shaping/optimizer/wfless/slm_model_in_loop.py` + `runners/slm/model_in_loop_runner.py` | 仿真 `model_in_loop_shaping.simulate_iterative_shaping` 的硬件移植: 每轮用强随机探针重拟合物理预测模型的 Zernike 像差 (Step A), 再冻结该像差合成目标方斑相位 (Step B), 用 trust region + 逐轮验收抑制两者相互耦合振荡 (CLI: `slm-model-in-loop`) |
| **迁移学习闭环整形（物理预测模型）** | `src/ao_shaping/tools/slm/transfer/` + `src/ao_shaping/gui/slm/shaping_controls.py` | 单一入口 `run_session` / `TransferSession`（CLI `python -m ao_shaping.tools.slm.transfer`，GUI 闭环整形「迁移学习」同入口）: `capture.py` 采集 + `to_panel`、`workflow.py` 阶段编排与 journal、`validate.py` A–B–B–A 验收、`fit.py` / `adaptation.py` / `seeding.py` / `pretrain.py`。接受相位自带载波（`ramp_matrix`），下发时跳过侧栏载波层 |
| Zernike 工具 | `src/ao_shaping/utils/wavefront/zernike_utils.py` | 系数解析 (Noll/(n,m)/数组) + 相位生成，Noll 1976 约定 |
| **SLM 闪耀光栅衍射效率 / 相位响应标定** | `src/ao_shaping/tools/slm/calibration/diffraction_efficiency/` | 2f 实验系统里用一维闪耀光栅测器件的相位调制深度与衍射效率: `core.py` (纯算法: 图案/阶次识别/**离散器件模型**/拟合/诊断) + `simulate.py` (数字孪生: 填充因子/LUT/面板相差/离焦) + `measure.py` (Santec+Daheng 硬件后端) + `artifacts.py` (CSV/JSON/PNG)。CLI: `python -m ao_shaping.tools.slm.calibration.diffraction_efficiency {simulate,measure,analyze}`。**硬件前置条件**: 必须先有光路标定 `.npz` (几何标定产出), 否则无法预测 ±1 级落点。⚠️ **可信范围** (见 `report/slm/diffraction_efficiency/sim_nonideality_20261022.md`): **相对效率曲线可信, 但拟合出的相位深度只在"幂律器件 + 在焦 + 填充因子≈1"三前提同时成立时可信** —— 填充因子 0.8 会把标称器件报成过调制 (+16%), 非幂律 LUT 偏 +42%, 离焦偏 +11%。**光斑宽度门禁** (`--fwhm-gate-ratio`, 默认 3×; `--reference-fwhm-px` 给一条**在焦**的绝对参照) 把这类失真变成显式失败: 门禁触发时越限点与"零级被熄灭"的点**一并**退出拟合, 有效点不足即**拟合报错**而不是给出一个偏了的深度 |
| RL training | `src/ao_shaping/optimizer/rl/` | SAC, LR-WFS |
| Simulation | `src/ao_shaping/drivers/sim/` | Digital twin devices |
| **2f 数字孪生（可复用）** | `src/ao_shaping/drivers/sim/bench_twin.py` | 参数化 2f 实验系统：把物理非理想因素注入传播链 —— SLM 的 LUT 非线性+量化、面板静态相差（Zernike+粗糙度）、装调（相机离焦/光路倾斜/透镜像差）。口径=瞳面区域→CCD 远场（与 region/padding/像元标度约定一致），默认全零误差=理想系统。与既有 `slm_pib_sim.py`（面板级 FFT 仿真）、`fouriergsnet_env.py`（带真实 λ/f/像元节距的傅里叶环境）并存，互不替代 |
| Utilities | `src/ao_shaping/utils/{io,image,wavefront,slm}/` + root `cli_params.py` | 4 子包: io/, image/, wavefront/, slm/ (spots_calc, wavefront_calc, zernike_calc, display 等) + 零导入叶子 `cli_params.py` |
| Standalone runners (未注册) | `src/ao_shaping/runners/` | `runners/slm/offset_runner.py` 计划迁移至 `tools/slm/`。⚠️ `slm_shaping_runner` **曾经**在此列 (2026-10-05 前它未注册); 现以 `slm-pib` **命令组**注册（子命令 `spgd` / `heuristic` / `square`）；⚠️ 无 `spgd-square` 顶层命令 ⇒ 写 `slm-pib square` |
| ML training | `src/ml/` (standalone, not inside `ao_shaping/`) | U-Net+GAN, trainer, wandb_logger |
| 硬件相位→相机图像 DataLoader | `src/ml/hwdataset/` | `data/debug` 全量转 PyTorch Dataset: 输入=SLM 相位+曝光, 输出=CCD 画面 (见 `硬件调试转 Dataset` 节) |
| **Zernike 系数→远场强度 Dataset** | `src/ml/hwdataset/zernike_dataset.py` | 输入=136 维 Noll 序系数 (弧度), 输出=归一化远场强度; **绕过 `materialise()` 的瞳面重建**, 实测 136 → 10.5 ms/item 且输出逐位一致 |
| 离线评估统计 (配对检验/canary skill) | `src/ml/zernike/eval_stats.py` | **纯 numpy, 报告引用的统计唯一来源** (sign-flip 精确置换检验 / `cohens_dz` / `holm_bonferroni` / `min_attainable_pvalue` / `skill_scores` / `seed_agreement`)。scripts/ 只做渲染, **不得复制第二份** |
| Standalone tools | `src/ao_shaping/tools/` | SLM phase capture, Micro-DM per-channel image collection, train data collection |
| Visualization | `src/ao_shaping/display/` | Windows, frames for GUI 与渲染器全集: `windows.py` 窗口管理 (旧注记 `Windows` 的实现文件, 无 docstring, 头部为 ABC 基类), `frames.py`, `beam_profile.py` 远场/SLM 相位/XY 截面渲染器 (单一实现) |
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

### optimizer/ 模块逐名登记 (2026-10-26 普查)

49 个模块逐一对照本文件叙述：此前有 32 个零提及 (顶级 4 / `rl/` 8 / `wf/` 3 / `wfless/` 17)，全部补录如下；其余 17 个已由职责表、调用链与 runner 对应表叙述覆盖。全量覆盖由守卫 `tests/ao_shaping/test_optimizer_doc_coverage.py` 锁定。

| 分组 | 模块 | 职责 (据 docstring 或 import/class 面) |
|---|---|---|
| 顶级共享 | `_camera_inject.py` | 无波前传感器 PIB 优化器的单一共享相机注入点 |
| 顶级共享 | `constants.py` | optimizer 包共享常量集中定义 (如 `KEEP_VOLTAGE_WHEN_EXIT`) |
| 顶级共享 | `spgd.py` | SPGD 梯度符号共享助手 (扰动成对评估 → 差分定号) |
| 顶级共享 | `spid_pib_loop.py` | 两条 `optimize_pib` 路径 (`wfless/pib.py` 等) 共享的 bench + SPGD-epoch 机制 |
| `rl/` | `device_registry.py` | AO RL 训练的设备注册表 (虚拟 + 物理设备) |
| `rl/` | `envs.py` | Gymnasium RL 训练环境集 (`sac_train` / `sac_turbulence_train` 消费, 湍流环境有专测) |
| `rl/` | `lr.py` | SAC+LSTM 训练竖切: `TensorboardCallback` / `BinomialActionNoise` / `LSTMEncoder` / 图像+向量组合提取器 |
| `rl/` | `lr_wfs.py` | LR-WFS 训练竖切: `WFS` 类 + `LaserCastEnv` (gym.Env) + 同款训练配套 |
| `rl/` | `rl_wfs.py` | RL-WFS 训练竖切: `WFSLaserCastEnv` (gym.Env) + 波前+向量组合提取器 |
| `rl/` | `sac_common.py` | AO 仿真训练的 SAC 共享工具 |
| `rl/` | `sac_train.py` | SAC 训练入口: 校正随机静态 Zernike 像差 |
| `rl/` | `sac_turbulence_train.py` | SAC 训练入口: 缓慢漂移湍流下的闭环 AO 校正 |
| `wf/` | `dm_response_matrix.py` | DM 致动器响应矩阵标定 (推-拉激励逐致动器实测) |
| `wf/` | `greedy_zernike.py` | 贪婪局部搜索: SLM Zernike 系数控制 |
| `wf/` | `interaction_matrix.py` | 斜率法交互矩阵 (响应矩阵) 获取 |
| `wfless/` | `_blaze_calibration.py` | SLM 整形会话的离轴载波选择实测 (私有辅助) |
| `wfless/` | `_convergence_diagnostics.py` | 动态 LR/delta 调度的共享收敛诊断阈值族 (三个 wfless 调度器共用) |
| `wfless/` | `_measure.py` | SLM+CCD 闭环测量辅助 (私有) |
| `wfless/` | `adaptive_spgd_core.py` | 自适应 SPGD 核 (`AdaptiveSpgdConfig` frozen dataclass) |
| `wfless/` | `adaptive_trust_region.py` | model-in-the-loop 轮循的自适应信任域半径策略 |
| `wfless/` | `adc_dm_adam.py` | ADC 驱动族的参考实现 (`drivers/adc/driver.py` 文档指名引用): Adam + numba 加速 + pandas 记录 |
| `wfless/` | `bayes_hyperparam.py` | skopt `gp_minimize` 贝叶斯超参搜索工具 |
| `wfless/` | `bayes_opt_lr_delta.py` | 对 `wfless/pib.optimize_pib` 的 LR/delta 做贝叶斯选型 |
| `wfless/` | `defocus_uniformity_shaping.py` | 离焦可微整形块级策略: 初值 / 方向检验 / A-B-B-A 重放 / 接受-拒绝 (纯逻辑, 硬件 I/O 由协议脚本注入) |
| `wfless/` | `differentiable_shaping.py` | 可微光束整形优化器封装 (闭环实现的薄 re-export) |
| `wfless/` | `gready_cam.py` | 相机+DM 贪婪搜索 (`MIICamera` + `NlightDM` + `KEEP_VOLTAGE_WHEN_EXIT`; 模块名 `gready` 为现状拼写) |
| `wfless/` | `gs_refine_zernike_fit.py` | 把 GS-refine SLM 相位拟合为 Noll 序 Zernike 系数 |
| `wfless/` | `model_in_loop_shaping.py` | SLM 方形整形 model-in-the-loop 迭代校正 (仅仿真; Step A 波前拟合 / Step B 交替) |
| `wfless/` | `pib_sim_eval.py` | PIB 仿真离线评估 (`eval_pib_hybrid_sim` / heuristic PIB 报告脚本消费) |
| `wfless/` | `sim_spgd.py` | 基于 `ao_shaping.sim` 的 SPGD 仿真优化 |
| `wfless/` | `slm_zernike_primitives.py` | SLM+Zernike 闭环共享原语 (`slm_zernike_pib` 与 `slm_zernike_shaping` 两目标族共用) |
| `wfless/` | `strehl_sim_eval.py` | DM 电压 Strehl 景观离线基准评估 (封装离线物理仿真) |

`lr.py` / `lr_wfs.py` / `rl_wfs.py` 三个训练竖切复写同款脚手架 (回调 / 编码器 / 组合提取器)，此处如实登记现状；脚手架去重与 `gready` 拼写属代码级变更，另行条目。

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
| `slm-pib square` | `optimizer.wfless.slm_square_shaping:optimize_slm_square()` | wfless | 均匀方形远场 (CV + EE + AR 综合质量分) | SLM + CCD |
| `slm-gs-refine` | `optimizer.wfless.slm_gs_refine:optimize_slm_gs_refine()` | wfless | GS 预矫正 (对比择优) + 自由相位 SPGD 细化 | SLM + CCD |
| `slm-model-in-loop` | `optimizer.wfless.slm_model_in_loop:optimize_slm_model_in_loop()` | wfless | 探针 refit 物理预测模型 (Step A) + 目标光斑相位合成 (Step B), 带 trust region 与逐轮验收 | SLM + CCD |
| （工具入口，非 main.py 命令） | `ao_shaping.tools.slm.transfer.workflow:run_session()` | tools | 少参数物理预测模型 + 当前光路重测几何 + 检查点迁移 + CCD 锚定 A–B–B–A 验收 | SLM + CCD |

> **`slm-pib` 配置容器** (2026-09): `optimize_slm_zernike_pib()` 已收敛为**纯 dataclass 单参数 API**:
> `def optimize_slm_zernike_pib(config: SlmZernikePibConfig)` (`optimizer/wfless/slm_zernike_pib.py`)。
> `SlmZernikePibConfig` 是普通 `@dataclass` (无自定义 `__init__`), 必填 `center`/`epochs` 在前,
> 嵌套 `camera: CameraParamsPib` / `slm: SlmParamsPib` (惰性默认, 定义于 `runners/runner_common.py`),
> 保留 `kwargs` 逃生口 (`**config.kwargs`)。**不再接受任何平铺关键字参数 / `cam=` / `slm=`**;
> 设备由优化器内部经 `create_camera(config.camera)` / `Santec.from_params(config.slm)` 上下文管理器
> 自行打开/关闭 (禁止跨 run 复用设备)。调用方: `runners/slm/slm_shaping_runner.py`、`scripts/slm/compare_shape_objectives.py`、
> `scripts/slm/generate_slm_pib_heuristic_hw_report.py`、`scripts/slm/repeat_shape_objectives.py`、
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

New optimizers added to `src/ao_shaping/algorithm/` MUST follow the class-based API: `__init__` does validation + state setup, `update()` performs one step and returns the next state/solution, and an optional `run()` returns a result dataclass. A one-shot function is kept only as a thin wrapper for backward compatibility. Torch/numpy **simulation-first tests are required before any hardware use**. Canonical example: `DifferentiableBeamOptimizer` (`src/ao_shaping/algorithm/signal_processing/differentiable_beam.py`). Full principle: `src/ao_shaping/algorithm/README.md`.

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
| `heuristic/` | `heuristic_base`, `search`, `gm`, `ga`, `pso`, `simulated_annealing`, `hill_climbing`, `random_search`, `cross_entropy`, `differential_evolution` | 无梯度启发式搜索; 注: 原行的 `sa`/`hc`/`rs`/`cem`/`de` 为别名口径, 盘上这些名字自有史以来从未作为文件存在——盘上与全仓 import 皆用全名模块;`heuristic_base` 基类 / `gm` 引导变异 / `search` 共享黑盒驱动 (`heuristic_algorithm_choices`, runner_common 与多个 optimizer 流消费) 同样漏登 |
| `signal_processing/` | `gerchberg_saxton`, `phase_wrap`, `controller`, `iterative_base`, `differentiable_beam`, `differentiable_shaping`, `beam_shaping_utils`, `wavefront`, `beam_shaping_benchmark`, `sim`, `iterative_zernike_shaping`, `zernike_coefficient_optimizer`, `angular_spectrum`, `square_shaping`, `uncertainty_probe_selector`, `differentiable_defocus_uniformity`, `multishape_objective`, `physical_forward_fit`, `defocus_forward_core` | 相位恢复/控制律/可微分整形; 补齐的 10 个模块各认其引入提交: `sim` (`1663614e`, 2026-09-18, 向量光束演示; `__main__` 演示体已于 `14b1cda` 迁至 `scripts/signal_processing/demo_wavefront_sim.py` (+ `demo_wavefront_reconstruction.py`), 本体保留 import-only 库, 拆分即终态), `iterative_zernike_shaping` (`04e6fc78`, 2026-09-29), `zernike_coefficient_optimizer` (`a94cadbb`, 2026-09-29), `angular_spectrum` (`c75aa3da`, 2026-10-09), `square_shaping` (`4edc8b84`, 2026-10-11), `uncertainty_probe_selector` (`080e0ac9`, 2026-10-16), `differentiable_defocus_uniformity` (`fe0f98b2`, 2026-10-23), `multishape_objective` 与 `physical_forward_fit` (`6a5e1067`, 2026-10-23), `defocus_forward_core` (`edbad379`, 2026-10-25, issue #199 step 4) |
| `tabu/` | `tabu_search` | 禁忌搜索 |
| `goal_functions/` | `target_func` | 目标函数与图像质量指标; 行内原列的 `image_metrics` 为幻影 token: 文件从未存在 (git `--follow` 零历史), 全仓 import 均走 `goal_functions.target_func` 的 `ImageTargetFunc` |

> **注意**: 顶层 `algorithm/` 现仅存 `__init__` 与 `base` (基类 `RegisteredBase`); 分包拆分带来的顶层 re-export shims 已于 `bbb3820b` (2026-09-20) 删除

---

### 三层架构关系

```
CLI (main.py Click 命令)
  │
  ├─ wf ──────────────→ runners/nlight_dm/wf_runner.py ──→ optimizer/wf/rms.py:optimizer_rms_dm() ──→ algorithm: Adam/AdaMOD
  ├─ pib ─────────────→ runners/nlight_dm/axis_beam_runner.py ──→ optimizer/wfless/pib.py:optimize_pib() ──→ algorithm: AdaMOD/Adam/SGD/Muno
  ├─ pipeline ────────→ runners/nlight_dm/pipeline_runner.py ──→ optimizer/wf/rms.py:optimizer_rms_dm() + wfless/pib.py ──→ algorithm: Adam/AdaMOD
  ├─ zernike-matrix ──→ runners/slm/zernike_matrix_runner.py ──→ optimizer/wf/zernike_response_matrix.py ──→ (标定)
  ├─ rms-zernike ────→ runners/slm/rms_zernike_runner.py ──→ optimizer/wf/rms_by_zernike.py ──→ algorithm: Adam/AdaMOD
  ├─ ga-zernike ─────→ runners/zernike_search_runner.py ──→ optimizer/wf/ga_zernike.py ──→ algorithm: GA
  ├─ slm-pib ─────────→ runners/slm/slm_shaping_runner.py ──→ optimizer/wfless/slm_zernike_pib.py ──→ algorithm: Adam/AdaMOD/GA/PSO…
  ├─ slm-pib square ──→ runners/slm/slm_shaping_runner.py (square 子命令) ──→ optimizer/wfless/slm_square_shaping.py ──→ algorithm: SPGD (同名 click 组)
  └─ combined ────────→ runners/nlight_dm/combined_runner.py ──→ optimizer/combined_optimizer.py ──→ algorithm: AdaMOD/SPGD

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
| 已注册 CLI 命令 | main.py 注册 **19** 个命令（逐个 `cli.add_command`，见 `main.py` 的 `# Register subcommands` 段；名单由 `tests/ao_shaping/runners/test_cli_contract_freeze.py::EXPECTED_COMMANDS` 冻结），见 Entry Points 节 |
| 独立 Runner (未注册) | 需直接运行 `python -m ao_shaping.runners.<name>` 或 standalone 脚本; `runners/slm/offset_runner.py` 计划迁移至 `tools/slm/`。⚠️ `slm_shaping_runner` **曾经**在此列 (2026-10-05 前它未注册); 现以 `slm-pib` **命令组**注册（子命令 `spgd` / `heuristic` / `square`）；⚠️ 无 `slm-pib` 之外的 `spgd-square` 顶层命令 ⇒ 写 `slm-pib square` |

### 共享辅助

`runner_common.py` 是共享辅助函数的主目录: `resolve_dm` (DM 解析), debug-artifact 写入器等。

> **注意**: `closed_loop.py` 不是 runner — 它是 `AOClosedLoop` 控制类, 归属 `optimizer/wf/`。

---

## utils/ Module

Utility functions for image processing and calculations, organized into 4 subpackages:

| Subpackage | Contents | Notes |
|------|---------|-------|
| `utils/io/` | `file`, `timestamp`, `cli_helpers`, `device_config`, `network`, `handler`, `recorder_keys`, `npz_summary` | `handler` becomes a backward-compat shim to `display/`; `recorder_keys` (2026-10-20 新增, 跟进 `1a11e60e`, #130 item 2): Recorder 记录字典键名的单一来源——11 个 `RECORD_*` 值成了一份盘面列契约 (HDF5/CSV sidecar 列名), 全仓 runners/optimizer/tools/ml 广泛 import;零导入叶子 (理由同 `utils/cli/params`), 由 `tests/ao_shaping/utils/test_recorder_keys.py` 钉住, 改键必须走显式迁移; `npz_summary` (2026-10-25 新增, 跟进 `d14d9dc0`): 探针脚本共用的压缩 `.npz` 摘要写盘单一实现 (`save_summary_npz`: 建父目录 + `np.savez_compressed` + log), `tools/zernike/abba_probe` 与 `tools/ccd/floor_probe` 的两份逐字节相同副本收拢于此, 域内旧名保留薄封装 |
| `utils/image/` | `spots_calc`, `beam_metrics`, `targets` (+ `target/`), `resample`, `hardware_utils`, `displacement`, `exposure_control`, `exposure_fusion`, `exposure_response`, `frame_analysis`, `hdr`, `centroid`, `ranks` | `gs_visualization` 已迁至 `display/`（2026-10-03）；`utils/hardware_utils.py` 别名 shim 已删（2026-10-03）；`display.py` 亦已迁至 `display/image_display.py`（2026-10-09, `a9236e9`）⇒ 旧路径**无 compat shim**，由 `test_utils_leaf_no_rendering_import.py` 钉住（2026-10-25 实测：`utils/image/display.py` 已不存在）; `centroid` (2026-10-24 新增, 跟进 `124444e6`, issue #199 R6 单一来源): 窗口内强度加权质心原语 `windowed_centroid`——以全帧 argmax 定窗 (0 级光斑常偏心数百像素, 不按几何中心定窗), 减窗口最小值作局部本底, 无正向动态范围时回落峰像素;`frame_analysis` 与 `displacement` 的三份副本收拢于此; 有意不是二阶矩/FWHM 估计器, 也不是相位相关平移估计——两类不同问题; `ranks` (2026-10-27 新增, 跟进 `4c4513c2`): 图像数组秩的零导入叶子 `NDIM_SINGLE_CHANNEL` (=2, 裸图像素面) / `NDIM_SPATIAL_STACK` (=3, `(C,H,W)` 空间栈), `beam_metrics`/`target.metrics`/`exposure_fusion` 的形状守卫单一来源, 由 `tests/ao_shaping/utils/image/test_rank_constants_single_source.py` 钉住 |
| `utils/wavefront/` | `zernike_calc`, `zernike_utils`, `wavefront_calc`, `wfs_utils`, `phase_unwrap`, `hadamard_calc`, `matrix_utils`, `zero_order_overlay`, `model_error_terms`, `calibration_terms`, `pattern_helper`, `beam_amplitude`, `fraunhofer`, `link_design` | `zero_order_overlay` (2026-10-15 新增): 2f 实验系统零级衍射叠加 (BLAZE/FRESNEL 模式), 用实测 `TILT_SHIFT_SCALE` 预测光斑位移, 不 import drivers/gui/algorithm; `model_error_terms` (2026-10-22 新增, 跟进 `3d8ef224`): model-in-loop 物理预测模型误差项 `ModelErrorTerms` / `load_error_terms` / `save_error_terms` (LUT γ / 相机离焦 / 倾斜 / 面板相位), 孪生 `bench_twin` 与 MIL 优化器共用同一份误差口径; `calibration_terms` (2026-10-22 新增, 跟进 `7e1a2716`): 自标定量 瞳面平面内仿射 τ_θ (6 标量 amp_x/amp_y/off_x/off_y/rot/grid) + 逐模式族行程缩放 γ, 依据 GRAPHYCS (Cho 等 2026) 思路, 供 `slm_model_in_loop` Step A 自标定 (CLI `--selfcal-affine` / `--selfcal-travel`); torch 走函数内延迟导入, 保持 utils 叶子层零 torch 依赖; `link_design` (2026-10-26 新增, 跟进 `d10bb483`, issue #204 单一信仰源): 望远镜链路 (M_t=10) 远场设计入口——`far_to_ccd`/`ccd_to_far` 双向换算 (f_eff = f_c·M_t 口径) 与纯 numpy DFT 远场引擎 (与 `fraunhofer` 同物理, 保持 utils 叶子层零 torch 依赖); 锁定设计点与 golden 验收值由 `tests/ao_shaping/utils/test_link_design_param_sweep.py` 逐值断言, 消费方为 `gui/slm/link_design_assist.py` 与 `gui/slm/shaping_controls.py` |
| `utils/slm/` | `slm_lut`, `phase_display`, `bench_constants`, `panel_geometry`, `phase_disturbance`, `phase_embed`, `phase_uniformity`, `gs_debug_optics` | `pattern_helper` 实际在 `utils/wavefront/`（2026-10-25 `ls` 实测，已移到上面那行）；本行原写 `utils/slm/pattern_helper.py` 的引用同属误标 |
| `utils/cli/params.py` (`utils/cli/` 子包) | `option`, `with_params`, `ClickGroup` | **零 `ao_shaping` 导入**（2026-10-03 从 `runners/runner_common.py` 抽出，TODO R-36）。见下方红线 |

> **注意**: legacy top-level `ao_shaping.utils.X` paths remain importable via shims.

### 🔴 `utils/cli/params.py` 是零导入叶子 (2026-10-03 抽出 R-36, 2026-10-05 移入 `utils/cli/`)

dataclass→click 参数绑定机制 (`Annotated[T, option(...)]` + `with_params` 收集器 +
对象投递) 的**唯一**实现。它同时被 `runners/` 和 `tools/slm/` 消费，所以:

- **禁止任何 `ao_shaping.*` 导入** —— 模块级、函数体内、`TYPE_CHECKING` 下都不行。
  只允许 **stdlib + `click`**。
- 理由链: `tools/slm/` 探针要在**无设备**路径上可用。若机制留在 `runners/runner_common.py`，
  `tools/` 就得 import `runners/`，而后者 import 期就 `list_dm_types()` 并
  `import ao_shaping.drivers.dm.asyn_micro_dm` 触发注册副作用 —— 实验系统探针会在
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

### utils/ 模块逐名登记 (2026-10-26 普查)

65 个模块逐一对照本文件叙述：子包表与树注释已以 stem 形式列举 43 个，名面真缺口 7 个；另有 6 个 stem 命中属撞词假覆盖（`exposure` 只命中 `exposure.npy` 与 env 注记、`target/` 五叶命中 drivers 树与反模式行的普通词），一并补录为 13 行表。全量覆盖由守卫 `tests/ao_shaping/test_utils_display_doc_coverage.py` 锁定（token 合同见守卫 docstring）。

| 位置 | 模块 | 职责 (docstring 原句或类面) |
|---|---|---|
| `image/target/` | `shapes.py` | 闭环 SLM 整形链的 canonical 形状词表 (单一来源) |
| `image/target/` | `geometry.py` | 形状→掩膜纯几何原语 (顶点/面积/包含判定/腐蚀) |
| `image/target/` | `masks.py` | 多形状目标掩膜 (正方形/长方形/三角形/五角星) 按面积归一; 非方形目标若仍用外接方框做 CV 会混入背景且读数无意义, 故单独成模块 |
| `image/target/` | `patterns.py` | SLM/图像网格上的参数化目标图样生成 |
| `image/target/` | `square.py` | 方形光束整形目标的 sizing 与振幅掩膜 (GS 式) |
| `image/target/` | `ccd.py` | CCD 帧驱动目标: 裁剪/缩放、任意中心掩膜、帧→SLM 网格管线 |
| `image/target/` | `objective.py` | 整形目标选择 + 自适应 `rms_pib` 权重 + 指标面板 |
| `math/` | `array_shape.py` | 精确形状数组拟合 (不感知图像/波前/设备) |
| `math/` | `fitting.py` | 最小二乘拟合助手 (纯 numpy, 原样搬迁) |
| `math/` | `stability.py` | settle/drift/linear 特征化判据 (纯 numpy 序列级) |
| 顶级 | `numba_compat.py` | 算法+utils 两层共享的可选 numba 装饰器垫片 (零依赖叶子) |
| 顶级 | `objective_direction.py` | 目标名→搜索方向的单一事实源 |
| `cli/` | `exposure.py` | `--exposure-time-ms`/`--exposure-ms` 一旗标三状态一解析器 (旗标曾于 runner 声明 7 处 + `panel_locate` 1 处, 且 0 值在两类声明中含义不同) |

尾注：`targets.py` 是 compat shim（其 docstring 的叶子枚举已过时——写了并不存在的 `metrics`、缺 `geometry`/`masks`/`shapes` 三个盘面叶），代码级 docstring 修正另行条目；`target/` 包消费方为 5 个 wfless 优化器与 `slm_shaping_runner`。

## tools/ 模块逐名登记 (2026-10-26 普查)

86 个模块逐一对照本文件叙述：名面（`name.py` 全名）未列 58 个 —— 46 个真零提及 + 12 个 stem 命中（4 个真命中：`floor_probe` / `abba_probe` 见 io 行收拢注记、`lut_runner` 见注册映射树、`panel_locate` 见 `cli/exposure` 行注记；3 个 `__main__.py` 通用 token 按同 `__init__.py` 的 floor 合同通过；4 个撞普通词假命中：`bootstrap` / `probes` / `response` / `simulation`；1 个特例 `calibration.py` 只被包目录路径串 `tools/slm/calibration/` 命中）。46 真缺 + 4 假命中 + `calibration.py` 共 51 行补录如下；全量覆盖由守卫 `tests/ao_shaping/test_tools_doc_coverage.py` 锁定（token 合同同 `utils/` 守卫）。**2026-10-26 复核**：上游合入 10 个新 transfer 模块（tools 总数 86 → 96），守卫当时恰红这 10 个，已一并登记，本表现共 61 行。

| 位置 | 模块 | 职责 (docstring 原句或类面) |
|---|---|---|
| 顶级 | `train_data_collect.py` | DM+WFS 训练数据采集器 (无 docstring; 由 `slm/calibration/train_data_collect.py` 的对标句背书) |
| `ccd/` | `drift_probe.py` | 平场漂移 + 曝光阶梯探针 (SLM 2f 实验系统, 一次运行回答两个问题) |
| `ccd/` | `exposure_check.py` | 相机是否真在手动曝光及其漂移检查 (实验系统工具而非库) |
| `ccd/` | `short_exposure_probe.py` | 短曝光固定相位探针 (raw 帧 + 曝光读回) |
| `kernels/` | `bench_kernels.py` | SLM 实验系统共享测量核 (由 `slm_bench_probe` 等多个先行探针模块合并) |
| `kernels/` | `device_mutex.py` | 设备互斥锁: 谁占着设备从猜测变为可判定 (2026-10-22 实测事故驱动) |
| `kernels/` | `sweep_analysis.py` | SLM Zernike 扫描的共享扫描/扫描分析助手 (多模块合并) |
| `micro_dm/` | `micro_dm_image_collect.py` | Micro-DM 逐单元图像采集: 遍历多台 R50Power 控制器 (按 IP) 逐通道下发电压并以 MiiCam/Daheng 采集 |
| `slm/` | `analyze_disturbance.py` | 光轴按各平场冻结后重算配对运行指标 |
| `slm/` | `bench_twin_calibrate.py` | 数字孪生参数反演标定 (复用 `drivers.sim.bench_twin` 的 2f 孪生核心, 上层反向拟合) |
| `slm/` | `compare_disturbance.py` | 固定 SLM 像差下的 GS-refine / model-in-loop 配对比较 |
| `slm/` | `gs_refine_replay.py` | 就一个运行目录交错重放 GS-refine 相位 |
| `slm/` | `model_in_loop_replay.py` | 两个已存 model-in-loop 相位的交错 CCD 重放 |
| `slm/` | `model_in_loop_summary.py` | 单个 model-in-loop 会话的离线免相机复盘 |
| `slm/` | `replay_disturbance.py` | 四个已存比较相位的交错定曝光 CCD 重放 |
| `slm/calibration/` | `calibration.py` | SLM 标定工具包 (单文件合并版): 2026-09 由 3 来源合并 (含 `drivers/slm/slm_calibration` 的相位-灰度响应标定) |
| `slm/calibration/` | `gray_response.py` | SLM 灰度→相机最大亮度响应采集 (出响应曲线图) |
| `slm/calibration/` | `phase_capture.py` | SLM 随机相位 (湍流屏/Zernike) + Daheng/MiiCam 画面三路采集 |
| `slm/calibration/` | `train_data_collect.py` | 面向 `ml/hwdataset` 语料的 SLM+CCD 训练数据采集器 |
| `slm/cartographer/` | `cosine_pattern.py` | SLM Hartmann 标定的中心 cosine 灰度图样生成 |
| `slm/cartographer/` | `dynamic_compensation.py` | 按实测 SLM 相位响应做动态像差补偿 |
| `slm/cartographer/` | `hartmann_capture.py` | Hartmann-Shack 光斑场采集 + 质心跟踪 (Thorlabs WFS) |
| `slm/cartographer/` | `phase_grayscale_lut.py` | 以中心 cosine 图样标定相位→灰度 LUT |
| `slm/cartographer/` | `slm_cartographer_ui.py` | SLM Hartmann 标定 + 动态像差补偿的 Streamlit UI |
| `slm/cartographer/` | `wavefront_reconstruction.py` | 由稀疏质心位移做傅里叶波前重建 |
| `slm/panel/` | `alignment_probe.py` | 定位 CCD 光斑前先测焦距与两处彗差校正 (复用 canonical Fresnel 扫描) |
| `slm/panel/` | `beam_extent.py` | 在面板上按边界扫描测光束中心与半径 |
| `slm/panel/` | `bench_fingerprint.py` | 标定后光路是否变过——实测指纹比对而非假定 |
| `slm/panel/` | `display_state.py` | 探针借用临时存储槽期间保全 SLM 显示状态 |
| `slm/panel/` | `focus_preflight.py` | 采集前置阶段: 焦面(离焦)补偿 + 去转向自标切 (2f 实验系统 CCD 常驻离焦) |
| `slm/panel/` | `focus_probe.py` | 找回把探测器拉回焦面的 Fresnel 焦长 |
| `slm/panel/` | `phase_resolution.py` | 面板能否施加像素尺度相位——对照平滑模式判定 |
| `slm/panel/` | `phase_response.py` | SLM 相位→CCD 响应探针 (核对相位图案确实作用于光) |
| `slm/panel/` | `probe_policy.py` | 面板探针族共享的固定预算采集策略 (`beam_extent`/`panel_locate` 各持一策略) |
| `slm/panel/` | `session_profile.py` | 会话级实验系统配置校验与按需重建 (2026-10-23 实测事故驱动) |
| `slm/panel/` | `target_size.py` | 方形目标边长自动标定与推荐 (过小则零级能量堆中心/散斑) |
| `slm/panel/` | `tilt_probe.py` | 以 SLM 倾斜给出面板是否真在调制的判定性证据 |
| `slm/transfer/` | `blur.py` | 用实测形状标定等效模糊/辐射度量 (带形状族验证) |
| `slm/transfer/` | `bootstrap.py` | 由新鲜 signed-ramp 采集建名义物理检查点 (零设备 import) |
| `slm/transfer/` | `capture_response.py` | 交错定相位曝光响应采集, 保留每一帧 (需显式执行旗标) |
| `slm/transfer/` | `direction.py` | 已存相位上的全局模型 vs 实测锚点方向性目标比较 |
| `slm/transfer/` | `paired.py` | 定曝光配对判定 (带显式实测验收指标) |
| `slm/transfer/` | `probes.py` | 确定性相位族 + canonical Zernike 求值 |
| `slm/transfer/` | `response.py` | 固定 CCD 网格上的照明曝光响应拟合与独立校验 |
| `slm/transfer/` | `simulation.py` | 生成已知真值的线性相机探针, 供免设备迁移测试 |
| `slm/transfer/` | `summary.py` | 免开设备的配对实验运行汇总 |
| `slm/transfer/` | `axial.py` | 已记录相位/CCD 对上的条件 ABCD 输入面扫描 (免设备; 拟合距离是相对冻结标称标定的有效修正, 非 SLM-透镜绝对距离) |
| `slm/transfer/` | `camera_probe.py` | 不开 SLM 记录相机实际格式与原始精度 |
| `slm/transfer/` | `cold_start.py` | 无训练整形种子 (无相位恢复、无历史检查点) |
| `slm/transfer/` | `exploration.py` | 已记录的非单调 CCD 探索 (champion 单独保留) |
| `slm/transfer/` | `geometric_seed.py` | 可分离辐照传输 + 拟合光学给出的非 GS 方形种子 |
| `slm/transfer/` | `local_response.py` | 由已记录 signed CCD 相位探针标定局部 Jacobian 增益/模糊 |
| `slm/transfer/` | `preflight.py` | 迁移学习闭环整形的启动前参数检查 (2026-10-26 实测: `empty target core` 报错实为载波周期参数问题而非几何问题) |
| `slm/transfer/` | `progress.py` | 由增量实验记录离线出指标轨迹 (免设备) |
| `slm/transfer/` | `radiometry.py` | 共享光学增益、辐照背景与曝光无关计数 |
| `slm/transfer/` | `replay_phases.py` | 在共同原生位深与曝光下重放精确归档相位 |
| `wfs/` | `wfs_probe.py` | SLM+WFS 光强与瞳面探针 — 标定前硬件状态检查 (光强是否足够且不饱和) |
| `wfs/` | `wfs_reference.py` | SLM+WFS 参考波前标定与倾斜线性度 (三步流程) |
| `wfs/` | `zernike_correction.py` | SLM Zernike 模式法波前矫正三阶段流程 (2026-09-15, 按实测调优) |
| `wfs/` | `zernike_response.py` | SLM Zernike 模式→WFS 读数响应矩阵标定 (各模式 ±A 推拉) |
| `zernike/` | `sweep_probe.py` | Smooth-Zernike 扫描探针: 以稳定读数采实验系统响应曲线 |

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
| `metrics.py` | img2img 标准指标 (MSE/RMSE/MAE/NRMSE/PSNR/SSIM) + 光斑域指标 (质心偏移 / 90% 环围直径 / 峰比), 复用 `beam_metrics.compute_beam_metrics` / `measure_spot_diameter_cam` |
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
> ⚠️ 另: `train_amp._select_records` 的划分是**按 objective 分层的按文件划分**
> (2026-10-07 改为 objective-stratified, 此前是"按文件随机 75/25 + `max_val=128` 截断")。
> **但分层并没有消解 train/val 缺口 —— 别再宣称它消解了。** 6 seed × 2 arm × 40 epoch
> 配对实测 (同 epoch 对比, `n_max=4`, 唯一变量是划分):
>
> | | gap@best_ep 均值±sd | val R² 均值±sd |
> |---|---|---|
> | 旧划分 | 1.16 ± 0.71 (0.63–2.32) | +0.8418 ± 0.0774 |
> | 分层划分 | 1.13 ± 0.33 (0.74–1.62) | +0.8499 ± 0.0201 |
>
> 配对差 (旧−新) `p=0.9375` / `d_z=+0.03` ⇒ **均值无效应**, 两边缺口都 ≈1.15。
> ⚠️ **早期"缺口 1.97 → 0.78 已消解"的说法是单 seed 噪声, 已作废**: 旧划分的 val 组成
> **随 seed 剧变** (seed 0 抽到 2 个 `rmse_out` → gap 1.99; seed 2 抽到单个 `rms_pib`
> → gap 0.63), 当时只跑了 seed 0, 正好抽到最差情形。
> **分层真正修掉的是方差与可复现性, 不是均值**: val 组成恒为 `{rms_pib, rmse_out, shape}`
> (旧划分 6 个 seed 出现 5 种不同组合), gap sd 0.71→0.33, val R² sd 0.077→0.020 (3.9×),
> 且逐目标 val R² 每 epoch 可见 (旧划分常只有 1 个目标, 看不见别的目标崩)。
> 单一目标 (`roi_pib`, 仅 1 文件) 无法拆分, 只进 train。"train on one objective"
> 用 `--file-contains <substring>`。
> ⚠️ **`gap` 指标本身有缺陷**: `best_val_mse / final_train_mse` 取的是**不同 epoch**
> (early-stop 的 val vs 最后一轮 train), 而 history 里的 `train_mse` 是**轮内平均**
> (权重还在动)。要看真实泛化缺口必须用 `train_eval_mse` / `train_eval_r2` / `gap_mse`
> (同 epoch、同一权重、同一 peak 归一)。真实缺口很小 (val R² ≈ +0.85), 无过拟合
> (14 参数 vs 512 样本) ⇒ **"差异很大"主要是单 seed + val 组成漂移的错觉**。
> ⚠️ `HWRecordRef.source` 是**相位表示** (`panel_gray`/`panel_rad`/`zernike`/`freeform`),
> **不是来源文件**; 文件标识是 `HWRecordRef.path`。

> **与 U-Net 的对比 — 权威结论是 grouped CV (`scripts/ml/compare_models_cv.py`)**
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
> (1) "U-Net R² 高 +0.022" —— 单 seed + **不公平预算** (U-Net 只给 25 epoch, 而物理预测模型第 13 epoch 就收敛)。
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
> 就"预测要下发的修正"而言, 230 参数物理预测模型是唯一能**闭式给出可实现相位**
> (`Σ Z_k B_k`, 230 个可解释弧度) 的, 且参数少 34000×、墙钟少 40%。
> 完整报告见 [`report/zernike_phase2amp/zernike_amp_report/report.md`](./report/zernike_phase2amp/zernike_amp_report/report.md),
> 含 11 节"被推翻结论"存档。旧对比表见
> [`report/zernike_phase2amp/models_cv/unet_comparison.md`](./report/zernike_phase2amp/models_cv/unet_comparison.md)。

0. **拿 `hwdataset` 的 `image` 当 Zernike 物理预测模型的 target 时, 必须校准远场尺度。**
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
# CENTER is not an env knob -- no code has ever read it. The analysis centre
# comes from the runner option -c/--center (auto = locate the 0-order by argmax).

# Camera protection exposure ceiling (ms). Optical-path dependent, set per
# session; unset or 0 = code probes the CCD image (doubling ladder to saturation)
# AO_DAHENG_MAX_EXPOSURE_MS=0.6

# SLM 2π gray: NOT an env knob any more (2026-10-23). MAX_GRAYSCALE_VALUE=1023 is
# only the radian→gray scale; the real 2π gray comes from the device phase table
# (SLM_Ctrl_ReadWL → 1064 nm phase=206 ⇒ 2.06π ⇒ 993) and the device **rejects**
# smaller values (196/200 snap to 206), so it cannot be forced to 1023.
# Zero-order floor saturates at 2.86% for phase ≥ 2.06π (ff≈0.854, η±1_max≈97.1%),
# so the device value is self-consistent. Full logic + evidence pointers live in
# the module comment above `src/ao_shaping/drivers/slm/santec/driver.py:MAX_PIXEL_FLIP_TIME_S`;
# probes: scripts/slm/probe_wl_phase_vs_efficiency.py, probe_gs_ceiling_vs_efficiency.py

# Library paths
PYTHONPATH=src;libs
PATH=libs\Drv_UDPST\x64\Release;libs\gxipy;${PATH}
```

### VSCode Settings

VSCode settings are configured in `.vscode/settings.json` (machine-local, **not tracked** since 2026-10-19 — create it yourself on a fresh clone if you use VS Code):
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
python src/ao_shaping/main.py slm-pib square        # 方形远场整形 (slm-pib 的子命令；无 spgd-square 顶层命令)
python src/ao_shaping/main.py combined              # legacy AdaMOD+SPGD 混合 PIB
python src/ao_shaping/main.py hadamard-matrix       # Hadamard 响应矩阵
```

**Console script (`pyproject.toml [project.scripts]`):** `ao-train = ml.train:cli` —— 唯一随包安装的入口，指向独立包 `src/ml/`（不在 `main.py` 的 click 组里）。

**CLI Structure (main.py 注册关系):**
```
main (click.group)                                  # 19 个命令，逐个由 cli.add_command 注册
├── wf                ← runners.nlight_dm.wf_runner
├── pib               ← runners.nlight_dm.axis_beam_runner
├── pipeline          ← runners.nlight_dm.pipeline_runner
├── zernike-matrix    ← runners.slm.zernike_matrix_runner      (attr `run`)
├── closed-loop       ← runners.slm.zernike_matrix_runner      (attr `closed_loop_run`)
├── rms-zernike       ← runners.slm.rms_zernike_runner
├── ga-zernike        ← runners.zernike_search_runner          (attr `ga_zernike_run`)
├── greedy-zernike    ← runners.zernike_search_runner          (attr `greedy_zernike_run`)
├── dm-matrix         ← runners.matrix_runner                  (attr `dm_matrix_run`)
├── hadamard-matrix   ← runners.matrix_runner                  (attr `hadamard_matrix_run`)
├── alt-voltage       ← runners.micro_drive.voltage_runner     (同模块两命令，靠 attr 区分)
├── full-voltage      ← runners.micro_drive.voltage_runner
├── combined          ← runners.nlight_dm.combined_runner      [legacy]
├── slm-pib           ← runners.slm.slm_shaping_runner         (click 组: spgd / heuristic / square)
├── slm-gsnet         ← runners.slm.gsnet_runner
├── slm-gs-refine     ← runners.slm.gs_refine_runner
├── slm-model-in-loop ← runners.slm.model_in_loop_runner
├── slm-lut           ← tools.slm.calibration.lut_runner       (来自 tools/，不经 _LAZY_RUNNERS)
└── slm-diagnose      ← tools.slm.panel.diagnose               (同上)
# 一个模块导出多个命令时必须显式给出 attr（`_IMPORT_ATTRS`），
# 否则落到默认 attr `run` 会取回错误的命令。
```

**Note:** `combined` 命令仍注册于 main.py（cli.add_command(combined_run, name="combined")）且功能可用, 作为 legacy 保留。`pipeline_runner.py` 是推荐的 WF→PIB 串行方案。

> **未注册到 main.py 的独立 Runner** (需直接运行 `python -m ao_shaping.runners.<name>` 或 standalone 脚本):
> `runners/slm/offset_runner.py` (计划迁移至 `tools/slm/`)。
> ⚠️ `hadamard_matrix_runner` 曾在此列并标注「正在注册」—— 2026-10-25 实测**已注册**（cli.add_command(hadamard_matrix_run, name="hadamard-matrix")，已进入 `EXPECTED_COMMANDS` 与 help golden），故移出本行。
> ⚠️ `slm_shaping_runner` 曾列于此 (2026-10-05 前未注册), 现以 `slm-pib` **命令组**注册（其 `square` 子命令即方形整形）。

**Refactoring Notes:**
- All runner scripts now use centralized config from `config.py` (DM_N_ACTUATORS, PATHS, DEFAULTS)
- Common CLI helpers moved to `utils/cli_helpers.py` (parse_tuple, setup_coredumpy)
- Duplicate code eliminated across runner files

---

## Build/Lint/Test Commands

### Running Tests

> 🔴 **测试必须在容器里跑。** 本机 384 核 + 755 GiB 内存，数值库按「能看到的核数」
> 开线程池；不受限的 `pytest` 会把整机跑满，**2026-10-21 就是这样把宿主机跑死、
> 强制断电才恢复**。`conftest.py` 现在会**直接拒绝**宿主机上的测试运行（报
> `Refusing to run the test suite directly on the host`），CI 同样走容器。

```bash
# 整棵树（唯一推荐入口）
./run_pytest.sh

# 子树
./run_pytest.sh tests/ao_shaping/runners

# 附加参数放在 -- 之后
./run_pytest.sh -k "not slow" -- -x

# 指定单个文件/类/函数
./run_pytest.sh tests/ao_shaping/utils/test_spots_calc.py
./run_pytest.sh tests/ao_shaping/utils/test_spots_calc.py::TestCentroid
./run_pytest.sh tests/ao_shaping/utils/test_spots_calc.py::TestCentroid::test_centroid_uniform

# 按 pattern / marker 过滤
./run_pytest.sh -k "spots_calc"
./run_pytest.sh -k "test_centroid"
./run_pytest.sh -m "not slow"

# 带覆盖率（容器内）
./run_pytest.sh -- --cov=ao_shaping
```

首次使用需先构建测试镜像（`run_pytest.sh` 只用现成镜像，不自己 build）：

```bash
docker build -f docker/Dockerfile.test -t ao-shaping-test:gpu .
```

**宿主机上确实只想跑一个文件时**（调试用），显式放行 —— 线程与显存上限仍然生效：

```bash
AO_ALLOW_HOST_PYTEST=1 .venv/bin/python -m pytest tests/ao_shaping/utils/test_spots_calc.py
```

CI 走同一套容器口径（CPU 镜像，无 GPU）：

```bash
docker build -f docker/Dockerfile.test.cpu -t ao-shaping-test:cpu .
./scripts/build/run_pytest_ci.sh
```

### 常规测试默认排除（pytest addopts，2026-10-16；2026-10-24 按实测撤销 drivers 排除；2026-10-26 按实测撤销 AppTest 排除）

`pyproject.toml` 的 `addopts` 默认 `--ignore` 一类**会阻塞**（挂死、无输出）的测试，
容器里的 `pytest` / `pytest tests` 不再踩到它们（从仓库根目录运行）：

| 排除 | 原因 |
|---|---|
| `tests/ao_shaping/ml` | 65 GB hwdataset 重 I/O + torch：多进程/mmap 曾把整树 `pytest tests` 挂死 |
| ~~`tests/ao_shaping/drivers`~~ **已撤销** | **原排除整棵 drivers 树过度保守一个量级，且造成了真实的可见性事故**：`drivers/ccd` 曾烂到 **6 failed + 1 error** 而 CI 毫无察觉（`29001826` 修）。逐域实测后全部恢复进发现集：`ccd` 204/25、`sim` 394/2（63.33 s）、`wfs` 110/11、`slm` 158/127（7.95 s）、`adc` 49（2.84 s）、`hardware` 37 skipped（0.06 s）、`dm` 139/5（**40.22 s**）、根文件 13/9；整棵 `drivers` 单 session **1067 passed / 216 skipped / 124.43 s** |
| ~~`tests/ao_shaping/gui/slm/test_zernike_abort_repro.py`~~ **已撤销** | 原排除理由是「唯一跑**真 `AppTest`** 的测试，需网页自动化配合，否则挂死」—— **该理由不成立**：它是**无浏览器**的 AppTest，跑真实 GUI 函数（`ZernikeControl.render` / `render_pattern_controls`），2026-10-26 实测 **2.06 s / 3 passed**。它当初是「复现」Zernike 切换时的 abort，而现在的代码已不会触发。作为**唯一**端到端渲染真实 GUI 运行时的测试，排除它等于白丢覆盖率 |

🔴 **`drivers/` 的 `--hardware` 门必须在 `drivers/conftest.py`**：跳过钩子原先只存在于
`drivers/slm/conftest.py` 与 `drivers/hardware/conftest.py`，而 `pytest_collection_modifyitems`
是**会话级**钩子 ⇒ 某条 `@pytest.mark.hardware` 测试是否被跳过，取决于**同会话里还收集了哪棵无关子树**
（`pytest drivers/dm drivers/slm` 会跳过，单独 `pytest drivers/dm` 不跳过）。后果不是观感问题：
`dm` 域曾因 `TestMicroDMReport` 的标记无人执行，fixture 把 26 个不可达控制器逐个扫到底 ——
**每测试 260 s 的 setup，520 s 占掉该域 560 s 的绝大部分**。新增 `drivers/conftest.py`
（复用 `drivers/_hardware_gate.py` 单源）后 `dm` 变 **40.22 s / 0 failed**。
⇒ 判"某域慢/挂起"必须先用 `--durations` 归因，不要按域整体归因。

🔴 **本地剪除与 CI 收集必须同向，但两份清单不必相等**：`pyproject.toml` 的 `--ignore` 只剪**本地默认**的
发现集（当前仅 `tests/ao_shaping/ml`：65 GB 语料 + torch 多进程曾把全量运行挂死），而
CI 按 `.github/workflows/pytest.yml` 的 `run_shard` 目标**显式收集**这些区域。两侧清单曾被要求
逐路径相等（脚本从 scratch 目录启动 pytest，pyproject 的**相对**排除在那种 CWD 下匹配不到任何
东西），该等式在 CI 开始有意收集 `ml` 之后即作废：强行相等就等于把整区 `--ignore` 塞回 harness，
让 CI 重新对该树变盲——那正是 2026-10-24 那次漂移与 TODO.md 8.9 items 102–109 的成因。
现由 `tests/ao_shaping/scripts/test_pytest_workflow_shards_cover_tree.py::test_every_test_area_is_sharded`
逐文件钉住（`tests/ao_shaping` 下每个测试模块都必须落在某条 `run_shard` 目标里，被本地剪除的 `ml`
树因而仍由 CI 收集）；harness 自身的 `EXCLUDES` 由同文件的
`tests/ao_shaping/scripts/test_pytest_workflow_shards_cover_tree.py::test_no_harness_ignore_blanks_a_whole_test_area`
与其同伴守卫逐条审（不得悬空、必须带理由、不得整区豁免、解析器不得空洞）。

⚠️ `--ignore` 只作用于 **discovery**（从 `testpaths` 发现的裸 `pytest`）。2026-10-24 实测：
裸 `pytest` 收集 **9293**；被排除的是 `ml` **1415** + AppTest **3**（两者亦实测）。
撤销 drivers 排除前为 **8011**，即整棵 drivers 树原本藏了 **1286** 个测试。
⚠️ **显式指定的路径不受 `--ignore` 影响**——实测 `pytest tests/ao_shaping/drivers/dm`
在该路径正被排除时仍收集 **144** 个（现已不排除，此性质仍成立）。要在实验系统上运行
硬件域，用 `--hardware` 打开标记门，而不是改 `addopts`。要连 discovery 也全部恢复：

```bash
./run_pytest.sh -- -o "addopts=-ra --strict-markers"
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
# CENTER 不是环境变量: 分析中心由 runner 选项 -c/--center 给出。
# Daheng 保护曝光上限 (ms), 随光路按会话设置; 缺省/0 = 由画面探测 (倍增阶梯到饱和)
# AO_DAHENG_MAX_EXPOSURE_MS=0.6
```

### 不在 `.env` 里的旋钮 (设备路径与诊断开关)

下列名字由生产代码直接读取, 不在 `.env` 中; 每个都在**模块作用域**求值, 所以必须在启动
Python 之前设好, import 之后再 `os.environ[...]` 赋值不会生效。`test_env_names_read_by_code_are_documented`
把这张表与代码里实际读取的名字对齐。

| 名称 | 读取位置 | 未设置时的默认 | 作用 |
|---|---|---|---|
| `SLM_CONFIG_DIR` | `src/ao_shaping/drivers/slm/santec/driver.py` | `<repo>/data/slm_configs` | SLM 相位表 / LUT 配置目录 |
| `DM_CONFIG_DIR` | `src/ao_shaping/drivers/dm/nlight/driver.py` | `<repo>/data/dm_configs` | NLight 变形镜配置目录 |
| `MICRO_DM_CONFIG_DIR` | `src/ao_shaping/drivers/dm/micro/driver.py` | `<repo>/data/micro_dm_configs` | 微变形镜 (R50Power) 配置目录 |
| `CCD_CONFIG_DIR` | `src/ao_shaping/drivers/ccd/daheng/driver.py` | `<repo>/data/ccd_configs` | Daheng 相机配置目录 |
| `AO_SHAPING_SLM_DLL_DIR` | `src/ao_shaping/drivers/slm/santec/_slm_win.py` | 无 (按候选目录搜索) | `SLMFunc.dll` 的第一候选目录; 未设则再找 `docs/SLM_DLL_ver.2.51`, 仍找不到就 `OSError` |
| `MIICAM_SDK_PATH` | `src/ao_shaping/drivers/ccd/miicam/_sdk_setup.py` | 无 (回落链) | MiiCam SDK 目录, 优先级最高; 否则依次用随仓的 `src/ao_shaping/drivers/ccd/_miicam_sdk` 与 `libs/miicamsdk.20240728/python` |
| `AO_SHAPING_SDK_PATH` | `src/ao_shaping/tools/slm/transfer/hardware.py` | `<repo>/libs` | 存放 `gxipy` 厂商绑定的父目录 |
| `AO_OOPAO_SOURCE_DIR` | `src/ao_shaping/drivers/sim/_oopao_compat.py` | 未设 (通常不需要) | 追加到 OOPAO 影子包 `__path__` 的额外搜索路径; 核心 18 个模块不靠它也能加载 |
| `WFS_DEBUG` | `src/ao_shaping/drivers/wfs/thorlab_wfs.py` | `0` | 置 `1` 时在 Thorlabs WFS 读取路径上记录每帧波前的方差 / 标准差 |
| `RMS_ZERNIKE_DEBUG` | `src/ao_shaping/optimizer/wf/rms_by_zernike.py` | `0` | 置 `1` 时打开 SLM Zernike RMS 优化器的逐轮调试输出 |
| `RMS_ZERNIKE_WEIGHTS` | `src/ao_shaping/optimizer/wf/rms_by_zernike.py` | `(n+1)/n_max` | 逗号分隔的逐阶扰动权重; 解析失败会记 warning 并保留默认值 |
| `NUMBA_CACHE_DIR` | `src/ao_shaping/__init__.py` 与 `src/ao_shaping/utils/numba_compat.py` (import 期) | 未设或非绝对路径 ⇒ `<tempdir>/ao_shaping_numba_cache` | numba JIT 缓存目录: 把 `@njit(cache=True)` 的 `.nbi`/`.nbc` 挪出 `src/`, 既守住「import 不落盘」契约也让只读挂载可用; 用户显式设**绝对路径**时不覆盖 |

平台自带变量 (`COMPUTERNAME`, `USERPROFILE`, `XDG_DATA_HOME`) 也被驱动用来定位配置落盘位置,
但它们由操作系统提供而非本仓定义, 因此不列入上表, 并在该守卫里被显式豁免。

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

VS Code settings in `.vscode/settings.json` (machine-local, untracked) set PYTHONPATH to `src` and `libs` directories.

---

## ANTI-PATTERNS (THIS PROJECT)

| Pattern | Forbidden Because |
|---------|------------------|
| Relative imports in package | Use `from ao_shaping.xxx import yyy` instead of `from .xxx import yyy` |
| `as any`, `@ts-ignore` | Never suppress type errors |
| Empty catch blocks | Always handle exceptions or log |
| Deleting failing tests | Fix the code, not the test |
| `combined_runner.py` with main CLI | Prefer `pipeline_runner.py` for new serial flows; `combined` remains registered as legacy (cli.add_command(combined_run, name="combined")) |
| Passing uint16 grayscale through `create_phase_from_array()` | `create_phase_from_array()` treats input as **radians** (mod 2π → grayscale = rad/2π × 1023). uint16 grayscale values get silently corrupted. Use `np.full((h,w), gray, dtype=np.uint16)` for flat phase or direct grayscale patterns. |
| SLM 相位生成函数自行 `np.mod(phase, 2π)` (2026-09 raw-only 契约) | All phase generators must return **raw unwrapped radians** — the only mod-2π wrap lives in the driver `Santec.create_phase_from_array()` on radian→grayscale conversion (`santec/driver.py` L1382). Generators that self-wrap duplicate the driver contract and hide the true phase. Convert via `utils/slm/phase_display.phase_to_slm_grayscale(phase, slm=slm)` (hardware) or the pure fallback (offline/tests). Exception: `_zernike_phase_radians` keeps a wrapped output **only** as a test-only reference. |
| Consecutive `write_phase` + `display_memory` to the **same** memory slot | Santec SLM firmware treats `display_memory(slot)` as a no-op when that slot is already being displayed — the LCOS panel does **not** refresh. Consecutive writes must ALWAYS target different slots. Preferred pattern (used by diff-shaping runner): pick a **random slot in 2..125** each write, excluding the currently displayed slot — this also survives process restarts (`get_displayed_memory_number()` before the first write). Older tools rotate a small pool like `itertools.cycle([3,4,5])` — works within one process only. The built-in `display_data()` cycles through all 127 slots. |
| Opening the SLM in DVI mode (`video_mode=1`) for diagnostics | `Santec(..., video_mode=1).open()` can **hang** (observed 120s/300s timeouts), and a hung controller then also hangs memory-mode `open()` until a **physical power cycle**. Never auto-try DVI mode; use memory mode (`video_mode=0`) only. See `tools/slm/panel/diagnose.py` |
| Treating `get_displayed_memory_number` error-code 1 as a fault | In `set_grayscale` mode there is no memory slot being displayed, so `SLM_Ctrl_ReadDS` returns error-code 1 — this is **normal**, not a failure. Slots only exist in memory mode. |
| Re-writing a **cached displayed phase** back through `write_phase` | Cached phases (`get_displayed_phase()`) already contain base overlay + wavefront correction; `write_phase` re-applies both → corrected **twice**. Rewriting shifted phases must go through the driver-level `Santec.apply_shift()` (raw write via `_write_to_memory` + slot rotation + config save), which is the single public entry point for shift-and-redisplay (identical semantics to the multi-SLM controller "应用平移" button). Shift math's only implementation is the static `Santec.shift_phase()`. See `src/ao_shaping/drivers/slm/AGENTS.md`. |
| Assuming the 0-order spot sits at the camera frame center | In the 2f Fourier bench the optical axis (0-order = frame **global maximum**) lands at the camera center only by luck. Observed: frame center (1344,760) vs 0-order spot (1441-1443, 705-706). Always locate 0-order by `argmax`, never by geometry. |
| diff-shaping 硬件闭环挂起（日志止于 `成功打开SLM #1`） | 2026-09-08 实测 (与相机无关): SLM open() 成功后, 首次 `camera.get_numpy_image()` 前无任何日志输出即无限阻塞 (900s 超时被强杀; 分步探针脚本同挂)。`WaitImageV3` 的原生等待由 SDK 内部驱动, 不受 Python 侧超时保护。处置: 强杀后先确认无残留 python 进程 (Get-Process python*)；重跑前 SLM memory 模式 open() 若超过数秒无日志, 对 SLM 控制器物理断电重置 (与 DVI 挂起同一处置)。故障排查与处置记录见 `report/slm/slm_shaping_diff/readme.md`（该条目当时引用的 `diff_shaping_runner.py` 已在 `fbef1926` 删除，仅存回归测试 `tests/ao_shaping/runners/test_diff_shaping_runner.py`；挂起结论本身仍然成立）。 |
| Generating SLM phase via **min-max normalisation** | It **min-max normalises** the phase (`(p-pmin)/(pmax-pmin)*1023`) instead of `mod 2π` radians→grayscale, making the pattern **scale-invariant** (coefficients ×1 and ×4 produce byte-identical patterns; verified `np.array_equal == True`). Always convert radian phase through the SLM driver `slm.create_phase_from_array()` (2π=993 + wavefront correction + LUT). See `README.md`. ⚠️ 历史例子: `PatternHelper._zernike_to_uint16` 与 `ZernikeDM.generate_phase` —— **两者已于 2026-09 修复/删除**, 本行保留为反模式记录。 |
| **`ZernikeDM.generate_phase` min-max normalises the phase** (same anti-pattern, still live) | `drivers/dm/zernike_dm.py:106-129` does `(raw−min)/(max−min)` then `×2π` (rad) / `×max_val` (gray) → output is **scale-invariant**: coefficients ×1 and ×4 give **byte-identical** phases (verified `np.array_equal == True`, PV always 2π). Since `ZernikeSLM.send_zernike` routes through it, **the Zernike coefficient amplitude is uncontrollable** for every consumer (`zernike-matrix`, `rms-zernike`, `ga-zernike`, `greedy-zernike`, `runners/slm/zernike_matrix_runner.py`, `runners/slm/rms_zernike_runner.py`, `gui/zernike/`). Fix: drop the normalisation (treat coefficients as radians) — note this intentionally changes those callers' behaviour. See `report/slm/report2.md` §2.3. |
| **Mixing units across the WFS→SLM correction loop** | The WFS `get_zernike()` returns **µm**; the response matrix must be built in the **same unit** as the correction's `w` (λ). Two real bugs (2026-09-16): matrix built in µm vs `w` in λ → coefficients inflated **1/0.532 = 1.88×**; and the solved coefficients are in **λ (waves)** but `make_phase`/`generate_zernike_polynomial` take **radians** → applied phase shrunk **2π = 6.28×**. Both fixed → closed-loop RMS improvement 13.8% → **42.1%**. Always route through the canonical `utils/wavefront/zernike_utils.py::um_to_waves`（`tools/zernike/common.py:36` 只做 re-export，并注明 canonical 在 zernike_utils；原写法指向的 `tools/slm/slm_zernike_common.py` 已不存在，2026-10-25 实测） and multiply λ→rad by `2π` before `make_phase`. See `report/slm/report2.md` §0. |
| **Full-frame centre leaking into window-local metrics** (`slm_zernike_pib.py`) | `cam.reset_window(...)` returns the window centre in **full-frame sensor coordinates**, but every downstream metric (`rms_pib_terms`, `ImageTargetFunc.radius`, `spot_waist_sigma`, `reference_center`) operates on the **re-windowed** image and needs **window-local** coordinates. Hardware-verified bug (2026-09-23): the runner's `-c auto` probe locates the 0-order on the un-windowed camera (e.g. (633,934) on 1944×2592) → optimizer used it directly as window-local → every metric read **0** (pib=0, r_bucket pinned at 0.9 = 1px radius × shrink 0.9) across the whole heuristic matrix. Fix: after capturing the re-windowed `init_img`, re-locate `reference_center = zero_order_center(init_img)` and use `reference_center` (NOT the post-`reset_window` `center`) for the r-bucket radius too (L1396/L1441). Rule: whenever a centre is handed to a metric/ROI builder, verify which coordinate frame it belongs to first. |
| Square shaping with low-order Zernike (n≤4) | Zernike modes are a **circularly symmetric smooth** basis; they physically cannot synthesise a square far-field (needs 2D-sinc-like near field / high spatial frequencies). Use full-pixel phase freedom (GS / differentiable / free-form), not Zernike. |
| Optimising `-CV` alone as the SPGD objective for square shaping | With no energy term the optimizer **empties the target box** to minimise CV (hardware observed EE→0.002). The objective must include encircled energy (use the combined quality score). |
| Computing `PIB` / `CV` on a **raw** CCD frame (read noise not removed) | A CCD frame carries symmetric read noise, so ~half its pixels are negative (measured 7164/14400 on a sim frame). `power_in_bucket` divides the in-target sum by the **whole-frame** sum, so a negative background drives the ratio **above 1** (measured `PIB=1.0120`) and corrupts `CV` the same way — the optimiser then chases read noise. Subtract the frame median and clip at 0 (`ao_shaping.utils.image.beam_metrics.prepare_frame`); clipping the *raw* frame instead rectifies noise into a pixel-count-sized DC pedestal (see the `clip(·,0,None)` defect in `sim/AGENTS.md`). |
| Re-locating the target ROI by `argmax` on **every** iteration | On a speckle field the global maximum hops between near-equal grains under a ~1e-3 perturbation, so an `argmax`-rolled box makes `PIB`/`CV` **discontinuous** and the optimizer chases a box that no longer covers the beam (documented on `compute_beam_metrics`). Locate the 0-order **once** on the unshaped frame and freeze the ROI for the whole run. |
| Committing a GS warm-start phase to hardware without a bake-off | The GS phase comes from a *model* of the bench (aperture, focal length, camera pixel pitch). If any input is wrong, GS makes the real spot **worse**. Measure flat and GS and keep the better one — a mis-calibrated model then costs two measurements instead of wrecking the run (`slm_gs_refine` Stage 1). |
| Treating an exact run-to-run reproduction as a property of a closed loop | The seed pins the SPGD perturbation signs, **not** the measurements: every far-field read carries device noise. Two same-seed runs disagree at ~1e-3 *at the flat baseline*, before any optimisation. Assert approximate reproducibility, never equality. |
| Trusting `reset_window()`'s returned centre | When the spot is near the frame edge the ROI offset is clamped but the returned `(w//2, h//2)` is not the true spot position → the target box lands off the beam (hardware observed epoch-0 `mean_b=0.01`). Re-locate the spot by `argmax`/centroid on the **windowed** image. |
| Function-only optimizers in ao_shaping/algorithm (no class API) | New optimizers must expose __init__ (validation + state) + update() (one step) + optional run() (result dataclass); one-shot functions are legacy/thin wrappers only. See src/ao_shaping/algorithm/README.md. |
| Placing markdown/report **generation** under `src/ao_shaping/tools/` | **All markdown/illustrated-report generation MUST live in `scripts/`** (naming: `scripts/generate_*_report.py`, e.g. `generate_zernike_wfs_report.py`, `generate_diff_shaping_report.py`). `src/ao_shaping/tools/` is reserved for hardware-interaction tools (CLI + driver orchestration), not report writers. See scripts/README.md. |
| Report generation inside `algorithm/` | `signal_processing/beam_shaping_benchmark` writes CSV/MD/GIF reports — report generation MUST live in `scripts/` (see scripts/README.md), not in the algorithm layer. |
| Pygame/viz code inside `utils/` | Visualization code belongs in `display/`, not the leaf utils layer. **Done for `gs_visualization`** (moved to `display/gs_visualization.py`, 2026-10-03; re-exported from `ao_shaping.display`) **and done for `display.py`** (moved to `display/image_display.py`, 2026-10-09, commit `a9236e9`; `utils/__init__.py` re-exports re-pointed to `ao_shaping.display.image_display`). No compat shim left — the old `ao_shaping.utils.image.display` path now raises `ImportError` (guarded by `test_utils_leaf_no_rendering_import.py::test_old_display_path_raises`). |
| Re-implementing the GS loop | `gs_visualization.gerchberg_saxton_with_visualization` must drive the canonical `gerchberg_saxton` via a callback, not duplicate the loop. |
| Native SDK binding inside `utils/` | `utils/` is the import-light leaf layer — it must not `import ctypes` to reach a device SDK; that binding belongs in `drivers/` (lazy registration via `drivers/_lazy.py`). The old `utils/wavefront/vi.py` (188 lines of ctypes VISA) was deleted in `bbb3820b`, which left the rule as prose only; it is now pinned by `test_conventions.py::test_the_utils_leaf_layer_never_binds_a_native_sdk` (measured baseline 0, no allowlist). |
| Min-max scale-invariant phase conversion | Min-max normalising the phase makes patterns scale-invariant (coefficients ×1 and ×4 → byte-identical). Use `utils/slm/phase_display.phase_to_slm_grayscale(phase, slm=slm)` instead. ⚠️ 历史例子 `PatternHelper._zernike_to_uint16` 已删除。 |
| Relocating `algorithm/signal_processing/sim.py` into `scripts/` | **Obsolete instruction (verified 2026-10-25)** — the vector-beam demo already lives in `scripts/`: `14b1cda` moved the `__main__` body to `scripts/signal_processing/demo_wavefront_sim.py` (+ `demo_wavefront_reconstruction.py`), leaving `sim.py` as an import-only library (`numpy` / `matplotlib.colors` / `loguru`; no `__main__`, `argparse` or `plt.show`). That script imports 4 symbols from it, so moving the module itself would break the import. The split is already correct: library in `algorithm/`, demo in `scripts/`. |
| `utils/**` 模块级 import `algorithm/`（原例：`pattern_helper` 曾 top-level import `ao_shaping.algorithm.phase_wrap`） | utils is the leaf layer and must not depend on `algorithm/` at import time — use deferred function-local imports. **原例已合规**（2026-10-25 AST 实测：该文件模块级只有 `__future__` / `typing` / `numpy` / `utils.wavefront.phase_unwrap:8` / `utils.wavefront.zernike_calc:13`，`algorithm` 全部延迟到函数内 `:20/:36/:92/:98/:837/:854/:871/:890/:903`）⇒ 由 `test_conventions.py::test_utils_has_no_module_scope_import_of_a_higher_layer` 钉住，其非空性由 `test_the_deferred_forms_still_exist_so_the_guard_is_not_vacuous` 守住（现 14 处延迟形式）。另：本行原写 `utils/slm/pattern_helper.py` 系误标，文件在 `utils/wavefront/`（见上方结构表）。 |
| 脚本/runner/tools/GUI 内**重复实现 Zernike 数学** (Noll↔(n,m) 查表、多项式求值、相位生成) | Canonical 入口唯一: `utils/wavefront/zernike_utils.py` (API 层) + `utils/wavefront/zernike_calc.py` (引擎层), 单向依赖 `zernike_utils → zernike_calc`。2026-09 去重重构前 `gui/slm/`、`optimizer/wfless/`、`tools/slm/` 各有一套, 已全数收敛。新代码一律 `from ao_shaping.utils.wavefront.zernike_utils import ...`, 禁止 `import aotools`/自写 `RZern`/自写 `noll2nm` 表。详见 README `## Zernike 使用指南` 与 AGENTS `## Zernike 使用规范`。 |
| **配对检验只在"变化的轴"上配对** — 拿 N 折当 N 个独立样本 | 全部折共用同一个 `seed` 时, 各折共享**同一条权重初始化随机流**, 于是 N 折是"同一对初始化"被测了 N 次, **不是**该对比的 N 次独立采样; sign-flip 检验仍按独立处理, p 值**偏保守 (过于乐观)**。实测: `input_terms` 136→78 的 18 折配对给出 `p = 0.0144`, 但固定一折只换 seed 后**符号翻转**、差值量级小 30 倍 ⇒ 那是初始化噪声不是效应。规则: **换 seed 就对 seed 配对, 换折就对折配对**, 两者不可互相冒充; 判定前跑 `scripts/ml/seed_sensitivity_coeff.py`。另: `min_attainable_pvalue(n) = 2/2**n`, 故 4 折协议**结构上**到不了 p<0.05, 那里只报效应量不报结论。统计实现唯一入口 `ml/zernike/eval_stats.py` |
| 用 **MSE/PSNR/SSIM** 给 `normalization` 选开关 | 任何含 `sum` 归一化的指标都在度量**归一化本身**而不是拟合质量。实测同一配置: `normalization="sum"` 给 PSNR **72.1 dB** / SSIM 0.9996, 而 R² 反而**更差**于 `peak`。判据只看 **R² / corr / 光斑域指标** |

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
- **SLM flat-phase gray RAW path**: Always send raw uint16 grayscale values to SLM via `np.full((h,w), gray, dtype=np.uint16)`. Never route flat phase through `create_phase_from_array()` (radian conversion). The SLM has amplitude coupling: different flat-phase gray levels produce different camera intensities at 1064nm (periodic with 2π ≈ 993 gray). Use `scripts/slm/validate_flat_phase_gray.py` to verify. Parameters for stable observation: `--exposure-ms 0.8 --wait-time-s 0.3 --discard-count 3`.
- **SLM phase raw-only contract + unified grayscale entry** (2026-09): Every SLM phase generator (optimizer/runner layer) returns **raw unwrapped radians**; do NOT `mod 2π` in generators — the driver `create_phase_from_array()` is the single wrap point (radian→grayscale, `santec/driver.py` L1382). Convert radian phase to grayscale through `utils/slm/phase_display.phase_to_slm_grayscale(phase, slm=slm)` — passes a live SLM it delegates to the driver pipeline (grayscale + wavefront correction + LUT + shift, 2π = device `_max_gray`); `slm=None` (simulation/offline save/unit tests) falls back to the built-in pure math conversion (wrap→scale→clip→uint16, default 1023). Counter-example **history** (both since fixed/removed in 2026-09 — do NOT reintroduce): `PatternHelper._zernike_to_uint16` (**deleted**, `pattern_helper.py:499-500`) and `ZernikeDM.generate_phase` (**fixed**, `zernike_dm.py:117-118`). A live scale-invariant min-max normaliser remains a defect; the two names above no longer exist in the tree.
- **SLM memory-slot rotation**: When writing consecutive phases to memory mode, always target **different** slots — calling `display_memory(slot)` for the slot already displayed is a no-op and the LCOS panel will not refresh. Preferred pattern (diff-shaping runner): random slot in **2..125**, excluding the currently displayed one (`get_displayed_memory_number()` on start, survives process restarts). Older tools rotate a small pool (`itertools.cycle([3,4,5])`) — in-process only. `display_data()` cycles all 127 slots internally.
- **Streamlit background threads**: Any background loop that drives hardware must never touch `st.session_state` directly — it causes `missing ScriptRunContext` warnings and race conditions. Use the R50 `run_loop` pattern: pass a snapshot of parameters as a plain `dict`, communicate state changes via `threading.Event` for stop signals and mutable containers (e.g. `list`) for live-updated values like frequency, and drain feedback through a `queue.Queue`. The main thread reads/writes `st.session_state` only on rerun. See `src/ao_shaping/gui/r50/r50_voltage_send.py:run_loop` and `src/ao_shaping/gui/slm/multi_slm_controller.py:_toggle_phases_task` for implementations.
- **Timing in background loops**: Prefer `time.time()` wall-clock deltas over counters for state machines. Example: `int(elapsed * 2.0 * freq) % 2 == 0` toggles at exactly the requested frequency without drift, instead of sleeping fixed half-periods and accumulating error.
- **2f Fourier bench geometry** (SLM front focus → f=125mm lens → CCD back focus): the lens Fourier-transforms the SLM field, so CCD coordinates represent **spatial frequency** — the +1 orders of an upper-half-grating and a lower-half-grating land on the **same CCD row (optical-axis row)**, differing only in x-offset. Expected diffraction offset `Δx_px = λ·f/(d_SLM·p_cam) ≈ 5021/Λ` (P64→78px, P32→157px, P96→52, P40→126). Do NOT assume half-screen gratings separate in y. 0-order = frame global max (`argmax`), re-check its location after any bench change.
- **SLM panel "not modulating" diagnostic chain** (2026-09, encoded in `tools/slm/panel/diagnose.py`): ① freeze check — write flat/full-grating/top-half/bottom-half to **rotated memory slots**, frames must differ (all-identical ⇒ LCOS frozen); ② modulation check — `set_grayscale` sweep 0..1023, 0-order bucket must vary with ~993-gray period (flat ⇒ no amplitude coupling ⇒ panel not modulating); ③ linearity check — exposure ×4, ×20 must grow peak brightness (constant peak incl. at 0.1ms/2ms ⇒ light is >100× weaker than the known ~0.02ms near-saturation baseline). Verdict thresholds: freeze ≥2/3 frames differ, modulation bucket rel-spread >15%, linearity growth >2×.
- **Class-based optimizer convention**: optimizers in ao_shaping/algorithm expose __init__ (validate + set state), update() (one step → next state), optional run() → result dataclass; the one-shot function stays a thin wrapper. Simulation-first (torch/numpy) tests before hardware. Canonical example: DifferentiableBeamOptimizer. See src/ao_shaping/algorithm/README.md.
- **Report generation lives in `scripts/`**: any functionality that writes markdown/illustrated reports MUST live in `scripts/` (naming `generate_*_report.py`, e.g. `generate_zernike_wfs_report.py`, `generate_zernike_response_matrix_report.py`, `generate_diff_shaping_report.py`) — NEVER under `src/ao_shaping/tools/`, which is reserved for hardware-interaction tools (CLI + driver orchestration). Scripts follow the repo script conventions: `matplotlib.use("Agg")` BEFORE importing pyplot, `sys.path` bootstrap (`ROOT = Path(__file__).resolve().parents[1]`), CJK font rcParams (`Microsoft YaHei`/`SimHei` + `axes.unicode_minus = False`), output under `report/<topic>/<子目录>/` or `logs/<timestamp>/`, and MUST be documented in `scripts/README.md`. Prefer **offline** report generators (read saved artefacts) so a report can be regenerated without hardware.
- **`docs/` vs `report/` (2026-10-05 拆分, 红线)**: 文档分两类，**不可混放** —
  - `report/<topic>/` = **实验报告**：一次实验/基准/仿真的**结论**（测量值、参数、口径、被推翻的假设）。原先在 `docs/` 下，2026-10-05 全部迁到仓库根的 `report/`。
  - `docs/` = **设备说明文档**：设备规格、SDK/驱动用法、装配/操作手册、图集，外加**每设备硬件测试报告** `<device>/<device>_report.md`（由 `tests/ao_shaping/utils/test_report.py` 的 `TestReport` 写出，位置绑定该 harness 默认值与 `test_conventions.py` 的 tracked-docs 守卫，故**不迁**）。
  - **每份报告必须能追到生成它的脚本**：`scripts/_common/provenance.py::REPORTS` 是**唯一映射源**，渲染出每份报告的 `生成脚本 / 复现命令 / 运行环境` 溯源块，以及 `report/README.md` 总索引。改动报告或脚本后跑 `python scripts/sync_report_provenance.py`（幂等）；`--check` 只校验。**手工编辑溯源块或索引 = 会被 `tests/ao_shaping/scripts/test_report_provenance.py` 判失败。**
  - **代码注释不承载实验叙述**：实测数字/踩坑结论写进对应 `report/`，代码处只留**不变量 + 报告链接**（例如 `ZernikeDM` 的系数即弧度 → `report/slm/report2.md`；读出噪声致 `PIB>1` → `report/slm_pib_bench/pib_bench_report/report.md` §7）。函数 docstring 仍保留**契约**（这个函数做什么、为什么这样裁剪），只把长实测叙述外迁。
- **`report/` 目录结构 (2026-10-21 重组, 红线)**: 一份报告 = 一个子目录，**按生成脚本/实验系列**归类，报告与其独占资产同住 ——
  - `report/<topic>/<子目录>/<报告名>.md` + `figures/` + `images/` + `artifacts/` + `assets/`
  - **不**平铺在 `report/<topic>/` 根下；**不**建 `shared/` 公共图目录（跨报告复用 → 各自复制）
  - 生成脚本的 `REPORT_KEY` / `OUT_DIR` / `FIG_DIR` 常量必须**写死新子目录**（重跑不会写回旧位置）
  - 完整规则见 [`report/AGENTS.md`](report/AGENTS.md)（该文件是 `report/` 下的专用规范，与本节冲突时以它为准）

---

## MISSING INFRASTRUCTURE

- **CI**: 2026-10-15 曾建立 ruff 硬门禁（`.github/workflows/ruff.yml`，push + PR 触发，
  `ruff check .`，issue #132 决策 B）与参考级 `pylint.yml`（`continue-on-error: true`）；
  2026-10-19 `.github/` 曾整体移出 git 跟踪（本机保留）；
  **2026-10-21 恢复入库**（`1d33d078`，现 `.github/workflows/ci.yml`：ruff 硬门禁
  + best-effort pytest 作业 `continue-on-error`），新克隆重新自带 CI 工作流 ——
  本地 `ruff check .` 仍须保持全绿。
  ✅ **2026-10-24：pytest 已成硬门禁**（`.github/workflows/pytest.yml`，push + PR）：
  一次镜像构建 + 分片（2026-10-24 建立时为 **8 片**；2026-10-27 由 TODO items 108–109 把 `ml`
  区域补齐、新增 `ml-gsnet-debug-offline` 与 `ml-gsnet-debug-rest` 后现 **11 片**），每片一个 pytest 进程，`AO_CPUS=4 / AO_MEMORY=6g`
  （ubuntu-latest 4 vCPU / 16 GB，留头寸）。`ci.yml` 里那个整目录单进程的
  `continue-on-error` 作业已删除 —— 它在 runner 上**不可能通过**（实测无目标参数的
  `run_pytest_ci.sh` 在 16g 上限下 34% 处 SIGKILL，exit 137）。分片清单只列**目录**、
  不列文件（文件清单会在新增测试时静默失盖），由
  `tests/ao_shaping/scripts/test_pytest_workflow_shards_cover_tree.py` 两向钉住：
  每个顶层测试区必须被某片覆盖、每个分片目标必须存在、且至少要有一个阻塞分片
  （否则"永远红不了的作业"等于没有门禁）。`optimizer` 片用 `AO_TEST_PER_FILE=1`
  （见下），实测覆盖率与单进程一致（1259 passed + 3 skipped == 1262 collected）。
   **2026-10-25 起 8 片全部 blocking，无 advisory**：最后一片 `scripts` 的名义原因是
   wt-190 的 `_fmt` 去重 —— 该去重**已入库**（`741ea0e5` 是 master 祖先），但 `273181e6`
   又**重新引入**一份本地 `def _fmt`，守卫在 master 上**仍红**；本账本 §8.9 第 44 项把该
   副本改走 `scripts/_common` 的 `fmt_general` 后实测整片全绿
   （**2423 passed / 64 skipped / 0 failed**，`AO_CPUS=4 / AO_MEMORY=6g`），当日转 blocking。
   `runners-gui-misc` 曾因 `slm-gsnet --help` 基线红而挂 advisory（wt-193 在办），
  **2026-10-25 已转 blocking**：基线由上游 `c5cd68a9` 重做后实测整片全绿
  （1149 passed / 30 skipped / 0 failed，`AO_CPUS=4 / AO_MEMORY=6g`）。
  ⚠️ 门禁是 **fail-closed** 的：镜像构建失败会让整个作业变红，这是有意取舍
  （宁可红也不允许"没跑测试却绿"）。
  仍缺：类型检查门禁（mypy / pyright 未配置）。
  注：本机工作树还残留两个未跟踪的 `ruff.yml`/`pylint.yml`（2026-10-19 移出跟踪的
  本地遗留，现已被 `ci.yml` 取代）—— 属本机状态，勿提交。
  ⚠️ **pytest 门禁必须分片跑，不能单进程整目录跑。** 2026-10-20 实测：sim 后端
  每次测量做 **7680×4800 complex128 FFT**（面板 1920×1200 × padding 4 = 36.9 M 点，
  单缓冲 0.59 GB，一次测量含 spectrum/`abs**2`/`fftshift` 副本约 2 GB 瞬时分配），
  而 numpy/Python 分配器不把已释放内存还给 OS —— 整目录单进程跑到几百个测量时
  工作集涨到 **20–27 GB**（63.7 GB 机器上页文件开始换页）。两个这样的 pytest
  同时跑就会互相拖成「假挂起」（实测进程 99% user CPU、页错误 0、可用内存 36 GB
  —— 是分配抖动，不是死锁）。分片（按文件）跑则每个文件都在秒级～分钟级完成，
  全部文件合计约 20–25 分钟。
  ⇒ 该口径已进机器：`scripts/build/run_pytest_ci.sh` 的 `AO_TEST_PER_FILE=1` 就是它，
  **每文件一个 pytest 进程**，清单由 `--collect-only` 现生成（自动遵守 EXCLUDES，不会因
  新增文件而静默漏测），**收集到 0 个文件即 FATAL** —— 一片「什么都没跑却绿」比红更糟。
  ⚠️ 另：**别把「日志里点数停在某个位置」当成挂起** —— `Tee-Object`/`Select-Object`
  管道会缓冲，实测「停在 216」的那次运行其实以 28 点/分钟正常推进，点数是缓冲
  量化出来的假象。判据要看进程 CPU 与内存，而不是日志末尾的点数。
- **Linting**: ruff 已配置（`ruff.toml`，`select = [E, F, RUF100, B023, PLR0124]`，
  `libs/**` 与
  `notebooks/**` 显式排除，守卫见
  `tests/ao_shaping/ml/test_train_cli_runnable.py::test_ruff_excludes_nothing_under_src`）；
  mypy / flake8 未配置。
  🔴 **ruff 版本只有一把**：`ci.yml`、`ruff.yml` 与 `pyproject.toml` 的 dev 组下限必须
  同为一个**精确版**（现 `0.15.12`），由
  `tests/ao_shaping/test_conventions.py::test_all_ruff_pins_agree_and_are_exact` 钉住。
  此前两处不一致（`ci.yml==0.11.10` vs `ruff.yml>=0.15.12`）实测**不等价**：`PLC0415`
  在 0.11.10 是 preview 规则（**0** 命中，ruff 自己会打印 "Selection `PLC0415` has no
  effect because preview is not enabled"），在 0.15.12 已稳定（**1476** 命中）；
  `SLF001` 差 2 条。硬门禁不得写 `>=`（新版本能在两次推送之间无代码变化地把门禁翻红）。
- **machine-local 开发工具配置**: `.mcp.json`、`.pre-commit-config.yaml`、`opencode.jsonc`、`pyrightconfig.json`、`skills-lock.json` 已移出 git 跟踪并加入根 `.gitignore`（2026-10-19，仅保留在开发者本机，不再入库）；同日 7 个 machine-local 目录整体移出跟踪并忽略：`.auto_doc_review`/`.claude`/`.codegraph`/`.slim`/`.streamlit`/`.vscode`/`_scripts`（本地副本保留，不随仓库分发）。**注意 `.github/` 不在其列**：它 2026-10-19 曾一并移出，2026-10-21 已恢复入库（见上文 CI 条，`1d33d078`），根 `.gitignore` 中 `.github` 的忽略行同日删除。
- **No requirements.txt**: Only `pyproject.toml` and `uv.lock`

Consider adding: 类型检查门禁（mypy / pyright 未配置）。pytest 硬门禁已于
2026-10-24 入库（见上文 CI 条），ruff 门禁已入库，无需重新提交工作流文件。

---

## CodeGraph (Pre-indexed Knowledge Graph)

This project has CodeGraph initialized (`.codegraph/` exists on the dev machine, 6,111 nodes, 12,339 edges). Note: `.codegraph/` is machine-local and **not tracked** (2026-10-19, in `.gitignore`), so a fresh clone will not have it — in that case skip the CodeGraph tools and fall back to grep/find/read.

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


