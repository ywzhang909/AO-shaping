# Module Inventory — AO-Shaping

> 本文件是根 `AGENTS.md` 的子文件：把各模块的详细登记（optimizer / runners /
> utils / tools）移出根文件，以精简根入口。根 `AGENTS.md` 仅保留索引与指针；
> 完整口径以本文件为准。术语、环境变量、CLI 命令等其它专题见
> `docs/dev/` 下对应子文件。

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
| `signal_processing/` | `gerchberg_saxton`, `phase_wrap`, `controller`, `iterative_base`, `differentiable_beam`, `differentiable_shaping`, `beam_shaping_utils`, `wavefront`, `beam_shaping_benchmark`, `sim`, `iterative_zernike_shaping`, `zernike_coefficient_optimizer`, `angular_spectrum`, `square_shaping`, `uncertainty_probe_selector`, `differentiable_defocus_uniformity`, `multishape_objective`, `physical_forward_fit`, `defocus_forward_core`, `amplitude_residual`, `full_chain_forward`, `monotonic_response`, `slm_phase_nonlinearity`, `spectral_regulariser`, `weight_schedule` | 相位恢复/控制律/可微分整形; 补齐的 10 个模块各认其引入提交: `sim` (`1663614e`, 2026-09-18, 向量光束演示; `__main__` 演示体已于 `14b1cda` 迁至 `scripts/signal_processing/demo_wavefront_sim.py` (+ `demo_wavefront_reconstruction.py`), 本体保留 import-only 库, 拆分即终态), `iterative_zernike_shaping` (`04e6fc78`, 2026-09-29), `zernike_coefficient_optimizer` (`a94cadbb`, 2026-09-29), `angular_spectrum` (`c75aa3da`, 2026-10-09), `square_shaping` (`4edc8b84`, 2026-10-11), `uncertainty_probe_selector` (`080e0ac9`, 2026-10-16), `differentiable_defocus_uniformity` (`fe0f98b2`, 2026-10-23), `multishape_objective` 与 `physical_forward_fit` (`6a5e1067`, 2026-10-23), `defocus_forward_core` (`edbad379`, 2026-10-25, issue #199 step 4) |
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
| 已注册 CLI 命令 | main.py 注册 **20** 个命令（逐个 `cli.add_command`，见 `main.py` 的 `# Register subcommands` 段；名单由 `tests/ao_shaping/runners/test_cli_contract_freeze.py::EXPECTED_COMMANDS` 冻结），见 Entry Points 节 |
| 独立 Runner (未注册) | 需直接运行 `python -m ao_shaping.runners.<name>` 或 standalone 脚本; `runners/slm/offset_runner.py` 计划迁移至 `tools/slm/`。⚠️ `slm_shaping_runner` **曾经**在此列 (2026-10-05 前它未注册); 现以 `slm-pib` **命令组**注册（子命令 `spgd` / `heuristic` / `square`）；⚠️ 无 `slm-pib` 之外的 `spgd-square` 顶层命令 ⇒ 写 `slm-pib square` |

### 共享辅助

`runner_common.py` 是共享辅助函数的主目录: `resolve_dm` (DM 解析), debug-artifact 写入器等。

> **注意**: `closed_loop.py` 不是 runner — 它是 `AOClosedLoop` 控制类, 归属 `optimizer/wf/`。

---

## utils/ Module

Utility functions for image processing and calculations, organized into 4 subpackages:

| Subpackage | Contents | Notes |
|------|---------|-------|
| `utils/io/` | `file`, `timestamp`, `cli_helpers`, `device_config`, `network`, `handler`, `recorder_keys`, `npz_summary`, `record_mode` | `handler` becomes a backward-compat shim to `display/`; `recorder_keys` (2026-10-20 新增, 跟进 `1a11e60e`, #130 item 2): Recorder 记录字典键名的单一来源——11 个 `RECORD_*` 值成了一份盘面列契约 (HDF5/CSV sidecar 列名), 全仓 runners/optimizer/tools/ml 广泛 import;零导入叶子 (理由同 `utils/cli/params`), 由 `tests/ao_shaping/utils/test_recorder_keys.py` 钉住, 改键必须走显式迁移; `npz_summary` (2026-10-25 新增, 跟进 `d14d9dc0`): 探针脚本共用的压缩 `.npz` 摘要写盘单一实现 (`save_summary_npz`: 建父目录 + `np.savez_compressed` + log), `tools/zernike/abba_probe` 与 `tools/ccd/floor_probe` 的两份逐字节相同副本收拢于此, 域内旧名保留薄封装; `record_mode` (2026-10-29 新增, 跟进 `2ad2e58e`): Recorder 模式词汇 `max`/`min` 的单一来源 —— `RecordMode(str, Enum)` + `coerce` 回落方向与旧 `== "max"` 判据逐位一致 (None/未知 → MIN), `maximizing`/`sign` 属性收拢原先 6 个优化器文件各自拼的 `objective_mode` |
| `utils/image/` | `spots_calc`, `beam_metrics`, `targets` (+ `target/`), `resample`, `hardware_utils`, `displacement`, `exposure_control`, `exposure_fusion`, `exposure_response`, `frame_analysis`, `hdr`, `centroid`, `ranks`, `physical_model_metrics`, `model_response` | `gs_visualization` 已迁至 `display/`（2026-10-03）；`utils/hardware_utils.py` 别名 shim 已删（2026-10-03）；`display.py` 亦已迁至 `display/image_display.py`（2026-10-09, `a9236e9`）⇒ 旧路径**无 compat shim**，由 `test_utils_leaf_no_rendering_import.py::test_old_display_path_raises` 钉住（2026-10-25 实测：`utils/image/display.py` 已不存在）; `centroid` (2026-10-24 新增, 跟进 `124444e6`, issue #199 R6 单一来源): 窗口内强度加权质心原语 `windowed_centroid`——以全帧 argmax 定窗 (0 级光斑常偏心数百像素, 不按几何中心定窗), 减窗口最小值作局部本底, 无正向动态范围时回落峰像素;`frame_analysis` 与 `displacement` 的三份副本收拢于此; 有意不是二阶矩/FWHM 估计器, 也不是相位相关平移估计——两类不同问题; `ranks` (2026-10-27 新增, 跟进 `4c4513c2`): 图像数组秩的零导入叶子 `NDIM_SINGLE_CHANNEL` (=2, 裸图像素面) / `NDIM_SPATIAL_STACK` (=3, `(C,H,W)` 空间栈), `beam_metrics`/`target.metrics`/`exposure_fusion` 的形状守卫单一来源, 由 `tests/ao_shaping/utils/image/test_rank_constants_single_source.py` 钉住; `physical_model_metrics` (2026-10-27 新增): 物理前向模型与整形质量度量 (NumPy-only, 离线, 不碰相机/SLM), 把模型预测质量与物理整形质量分开——`affine_radiometry`/`compute_forward_metrics`/`compute_directional_metrics` 均在显式核心掩膜上计算并保留零级排除 |
| `utils/wavefront/` | `zernike_calc`, `zernike_utils`, `wavefront_calc`, `wfs_utils`, `phase_unwrap`, `hadamard_calc`, `matrix_utils`, `zero_order_overlay`, `model_error_terms`, `calibration_terms`, `pattern_helper`, `beam_amplitude`, `fraunhofer`, `link_design`, `sampling_limit` | `zero_order_overlay` (2026-10-15 新增): 2f 实验系统零级衍射叠加 (BLAZE/FRESNEL 模式), 用实测 `TILT_SHIFT_SCALE` 预测光斑位移, 不 import drivers/gui/algorithm; `model_error_terms` (2026-10-22 新增, 跟进 `3d8ef224`): model-in-loop 物理预测模型误差项 `ModelErrorTerms` / `load_error_terms` / `save_error_terms` (LUT γ / 相机离焦 / 倾斜 / 面板相位), 孪生 `bench_twin` 与 MIL 优化器共用同一份误差口径; `calibration_terms` (2026-10-22 新增, 跟进 `7e1a2716`): 自标定量 瞳面平面内仿射 τ_θ (6 标量 amp_x/amp_y/off_x/off_y/rot/grid) + 逐模式族行程缩放 γ, 依据 GRAPHYCS (Cho 等 2026) 思路, 供 `slm_model_in_loop` Step A 自标定 (CLI `--selfcal-affine` / `--selfcal-travel`); torch 走函数内延迟导入, 保持 utils 叶子层零 torch 依赖; `link_design` (2026-10-26 新增, 跟进 `d10bb483`, issue #204 单一信仰源): 望远镜链路 (M_t=10) 远场设计入口——`far_to_ccd`/`ccd_to_far` 双向换算 (f_eff = f_c·M_t 口径) 与纯 numpy DFT 远场引擎 (与 `fraunhofer` 同物理, 保持 utils 叶子层零 torch 依赖); 锁定设计点与 golden 验收值由 `tests/ao_shaping/utils/test_link_design_param_sweep.py` 逐值断言, 消费方为 `gui/slm/link_design_assist.py` 与 `gui/slm/shaping_controls.py` |
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
- 相关越界守卫另见 `test_conventions.py::test_the_utils_leaf_layer_never_binds_a_native_sdk` 与 `test_conventions.py::test_utils_has_no_module_scope_import_of_a_higher_layer`（utils 叶子层不得绑定原生 SDK、不得模块作用域导入高层）。

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

86 个模块逐一对照本文件叙述：名面（`name.py` 全名）未列 58 个 —— 46 个真零提及 + 12 个 stem 命中（4 个真命中：`floor_probe` / `abba_probe` 见 io 行收拢注记、`lut_runner` 见注册映射树、`panel_locate` 见 `cli/exposure` 行注记；3 个 `__main__.py` 通用 token 按同 `__init__.py` 的 floor 合同通过；4 个撞普通词假命中：`bootstrap` / `probes` / `response` / `simulation`；1 个特例 `calibration.py` 只被包目录路径串 `tools/slm/calibration/` 命中）。46 真缺 + 4 假命中 + `calibration.py` 共 51 行补录如下；全量覆盖由守卫 `tests/ao_shaping/test_tools_doc_coverage.py` 锁定（token 合同同 `utils/` 守卫）。**2026-10-26 复核**：上游合入 10 个新 transfer 模块（tools 总数 86 → 96），守卫当时恰红这 10 个，已一并登记，本表现共 61 行。**2026-10-29 复核**：上游 10-26 后又陆续合入 `jacobian_guard` / `parallel_fit` / `protocol` 三个 transfer 模块（此前漏登记），连同 `validate.py` 为过模块规模门禁拆分出的 `session_plan.py` 一并补登记 4 行，本表现共 64 行；同日 `48b8b55a` 把曝光阶梯自 `panel/focus_probe.py` 拆出为 `panel/exposure_adapt.py`，补登记 1 行，本表现共 65 行。**2026-10-29 二轮**：本轮 campaign 界面新增的 4 个 transfer 模块（`campaign_control` / `checkpoint_registry` / `cost_table` / `optical_session`）与拆分出的 `session_record.py` 补登记 5 行，本表现共 70 行。**2026-10-30 复核**：上游 `f6b3e186` 合入时 `jacobian_guard` / `parallel_fit` / `protocol` 三条被两条提交（`adbc2016` 短版 / `54782eec` 长版）重复登记，已去重各留一条（长版，与 docstring 一致），本表实为 75 行。

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
| `slm/panel/` | `exposure_adapt.py` | 曝光阶梯单一来源 (2026-10-29 自 `focus_probe` 拆出): 选出让帧峰值落在满量程可接受带内的曝光并读出 `robust_peak`/`frame_full_scale`/`adapt_exposure`; 免设备 (`set_exposure_ms`/`read_frame` 注入), `focus_probe` 旧路径 re-export 同组名 |
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
| `slm/transfer/` | `device.py` | Which compute device the torch stages use (fit, synthesis, validation) |
| `slm/transfer/` | `cold_start.py` | 无训练整形种子 (无相位恢复、无历史检查点) |
| `slm/transfer/` | `exploration.py` | 已记录的非单调 CCD 探索 (champion 单独保留) |
| `slm/transfer/` | `geometric_seed.py` | 可分离辐照传输 + 拟合光学给出的非 GS 方形种子 |
| `slm/transfer/` | `local_response.py` | 由已记录 signed CCD 相位探针标定局部 Jacobian 增益/模糊 |
| `slm/transfer/` | `jacobian_guard.py` | 逐块 Jacobian 方向校验与它驱动的信任域收缩 (免设备): 用每个 CCD 校验块本来就采的 `A1`/`B` 两端差分, 拿模型预测差与实测差的**相关性**判方向是否可信, 不可信就收缩信任域 |
| `slm/transfer/` | `light_guard.py` | 光路无光的早判与专用退出码: `OpticalPathInvalid` (`exit_code = 3`) + 曝光阶梯内的 `on_step` 守卫 (`LightGuard`, 峰值停在噪声底即中止) + 启动前 `verify_light`; 判据是「峰值 vs 该帧自己的噪声底」, 与探针的 `weak_path` 分工 (后者管「能测出来但已衰减」) |
| `slm/transfer/` | `parallel_fit.py` | 拟合并行监视进程 (不碰设备): capture 独占 SLM, 本进程流式写 `loss_trace.jsonl` / `parallel_fit_provenance.json`, 损失够好或停止下降时落 `stop_acquisition.request` 让 capture 提前收工 |
| `slm/transfer/` | `protocol.py` | 孔径固定的指标口径与重复统计原语 (CV 单独用无法区分「均匀」与「空」, 故每条结论都配一个相对的桶内能量数) |
| `slm/transfer/` | `preflight.py` | 迁移学习闭环整形的启动前参数检查 (2026-10-26 实测: `empty target core` 报错实为载波周期参数问题而非几何问题) |
| `slm/transfer/` | `progress.py` | 由增量实验记录离线出指标轨迹 (免设备) |
| `slm/transfer/` | `radiometry.py` | 共享光学增益、辐照背景与曝光无关计数 |
| `slm/transfer/` | `replay_phases.py` | 在共同原生位深与曝光下重放精确归档相位 |
| `slm/transfer/` | `session_plan.py` | 免设备会话规划、续跑记账与 model-only 验收 (由 `validate.py` 拆出过模块规模门禁, `validate` 按旧路径重导入) |
| `slm/transfer/` | `stages.py` | 四阶段 CCD 锚定整形配方表 (单一事实源, 不碰设备): `STAGE_RECIPES` 从 `runners/slm/transfer_shapes_runner.py` 搬出, 免设备消费者 (GUI 计划页 / 进度读取 / 计划脚本) 不必 import 一整个 click 绑定的 runner |
| `slm/transfer/` | `campaign.py` | 多形状 campaign 的「形状 + 尺寸 → 其余全部参数」派生层 (纯函数, 不碰设备): `SIZE_WINDOWS`/`window_for_side` 与计划脚本共用同一张尺寸→窗口表 (200/600/1200 → 420/620/900), `derive_settings` 逐字段标 `measured`/`derived`/`default`/`manual` 来源并记下被覆盖的原值, `campaign_matrix`/`estimate_seconds`/`estimate_disk_gb` 供计划页预览; 存在的理由是「哪些数是测出来的、哪些只是 CLI 默认值」不能靠操作员记忆 |
| `slm/transfer/` | `campaign_progress.py` | campaign 输出目录的逐块进度读取器 (纯函数, 不碰 streamlit/设备; 产物缺失、只写一半、阶段被 `stopped` 都不抛): `collect_campaign_progress` 读 `journal.json` + `shape/<shape>/<stage>/manifest.json` → 16 段 / 112 块视图 (与报告 §3.2 表同源), `peak_full` 对齐报告「段内峰值」, `peak_inside` 另答核心内峰值 |
| `slm/transfer/` | `campaign_control.py` | 段间暂停闸: 「本段结束后停下」写进目录而非内存 (换页面/重开浏览器仍看得见), 文件约定同 `parallel_fit` 的 `stop_acquisition.request` |
| `slm/transfer/` | `checkpoint_registry.py` | 检查点索引: `register_session` / `scan_checkpoint_index` / `verify_checkpoint` 三件事分开做, 失配标 `stale` 而不是删 (「过期」本身有用), 只写 `session/checkpoint_index.json` 不动模型文件 |
| `slm/transfer/` | `cost_table.py` | 逐段实测成本: `cost.json` 落在段目录, `read_campaign_costs` 反向归并; 估算只用**每块秒**外推 (段长秒数不可比), 无历史时返回全 0 并留空基准, 让页面能说「首次运行, 无历史成本」 |
| `slm/transfer/` | `optical_session.py` | 光路指纹与兼容评分: 指纹 (窗口/零级/曝光/相机格式/控制网格) + 五维加权打分 + 两档阈值 (`COMPAT_THRESHOLD` 0.60 / `AUTO_APPLY_THRESHOLD` 0.80) 与唯一判定 `compatibility_verdict` |
| `slm/transfer/` | `session_record.py` | 会话收尾与段级记账 (指纹落盘 + 检查点登记 + 段成本 + 段间暂停), 免设备免 click, 只读会话目录已有产物 (由 `transfer_shapes_runner.py` 拆出过模块规模门禁) |
| `wfs/` | `wfs_probe.py` | SLM+WFS 光强与瞳面探针 — 标定前硬件状态检查 (光强是否足够且不饱和) |
| `wfs/` | `wfs_reference.py` | SLM+WFS 参考波前标定与倾斜线性度 (三步流程) |
| `wfs/` | `zernike_correction.py` | SLM Zernike 模式法波前矫正三阶段流程 (2026-09-15, 按实测调优) |
| `wfs/` | `zernike_response.py` | SLM Zernike 模式→WFS 读数响应矩阵标定 (各模式 ±A 推拉) |
| `zernike/` | `sweep_probe.py`, `common.py` | Smooth-Zernike 扫描探针: 以稳定读数采实验系统响应曲线 |

