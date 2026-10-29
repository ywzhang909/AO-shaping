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

> 逐文件详解（drivers / optimizer / ml / gui / scripts 等完整展开）见 [docs/dev/project_structure.md](docs/dev/project_structure.md)。

```
AO-shaping/
├── src/
│   ├── ao_shaping/               # Main package
│   │   ├── main.py                   # CLI entry point (Click-based, 20 子命令)
│   │   ├── runners/                  # CLI 编排层 (设备生命周期 + 结果保存); 逐文件见 docs/dev/project_structure.md
│   │   │   ├── __init__.py               # 惰性 re-export (_LAZY_RUNNERS / R-27, PEP-562)
│   │   │   ├── runner_common.py          # 共享辅助 (resolve_dm / patch_sim_*_shaping)
│   │   │   ├── matrix_runner.py          # dm-matrix + hadamard-matrix
│   │   │   ├── zernike_search_runner.py  # ga-zernike
│   │   │   ├── nlight_dm/                # wf_runner.py / axis_beam_runner.py / pipeline_runner.py / combined_runner.py
│   │   │   ├── micro_drive/              # voltage_runner.py (alt / full-voltage)
│   │   │   └── slm/                      # slm_shaping_runner.py / gsnet_runner.py / gs_refine_runner.py / model_in_loop_runner.py / model_in_loop_session.py / rms_zernike_runner.py / zernike_matrix_runner.py / offset_runner.py
│   │   ├── algorithm/                # 优化算法 (gradient / heuristic / signal_processing / tabu / goal_functions)
│   │   ├── drivers/                  # 硬件抽象层 (ccd / dm / slm / wfs / tm / sim); 详见 drivers/AGENTS.md
│   │   ├── optimizer/                # 高层优化策略 (wf / wfless / rl)
│   │   ├── utils/                    # 工具 (io / image / wavefront / slm + cli_params.py)
│   │   ├── ml/                       # ML (gsnet / hwdataset / phase / zernike); 详见 docs/dev/project_structure.md
│   │   ├── tools/                    # 独立工具 (SLM 相位采集 / Micro-DM 采图 / 训练数据采集)
│   │   ├── display/                  # 可视化 (窗口 / 帧 / 渲染器)
│   │   └── gui/                      # Streamlit GUI, 按设备域分包 (r50 / dm / slm / zernike / ccd)
│   ├── calculators/                  # Cython 扩展 (独立)
│   └── optical_ui/                   # [DEPRECATED] 空包
├── tests/ao_shaping/                 # 测试 (镜像 src 结构)
├── libs/                             # 第三方 SDK 二进制 (gxipy, Drv_UDPST)
└── scripts/                          # AI 一次性脚本, 按处理对象分组 (slm / dm / wfs / ml / rl / oopao / build / _common ...)
```

## scripts/ 目录：AI 一次性脚本（2026-10-29）

`scripts/` 不是产品代码，是**给 AI 用的脚本** —— 它记录"某次实验/某份报告/某次排查
是怎么算出来的"。使用者是发起那次任务的 AI 或人，不是终端用户。因此：

1. **不做易用性优化。** 不加交互式参数、不做宽容错兜底、不为好看而拆文件、不追求
   可复用 API。一个 4000 行的报告脚本只要能重现它那份报告，**就不该**被为了行数
   好看而拆开。`scripts/check_module_size.py` 因此**只对 `src/` 判红**，
   `scripts/` 计入报告（哪个脚本在膨胀一眼可见）但不参与退出码；baseline 里也
   只有 `src/` 条目。
2. **有复用价值的函数迁移到主项目。** 判据是"**第二个消费者出现了**"，不是"写得
   漂亮"。同一个判据/统计量/几何换算在第二个脚本里再抄一遍，就该搬进
   `src/ao_shaping/` 的对应层（纯算法 → `algorithm/` 或 `utils/`；IO/报告基建 →
   `utils/io/`；硬件交互 → `tools/` 对应子包），scripts 侧改成 import。
   **禁止**在 scripts 里维护第二份实现 —— 已有先例：`src/ml/zernike/eval_stats.py`
   是离线统计的唯一来源，scripts/ 只做渲染。
3. **确认过期的删除。** 一次性探针在结论入 `report/` 或 issue 之后就是死代码；
   留着只会让"哪个脚本是现在的口径"变成考古。删之前确认三件事：
   ① 它产出的结论已被某份 report/issue 引用；② 没有别的脚本 import 它；
   ③ `scripts/README.md` 里它那一行条目同时删掉。暂时删不掉的（还要重跑），
   在文件头 docstring 写明"什么条件下可以删"。

配套的**动作顺序**（不要反过来做）：先删过期 → 再迁复用 → 最后才考虑拆。
拆分排在最后，是因为前两步经常让某个大脚本直接消失，拆了等于白干。

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
| **迁移学习闭环整形（物理预测模型）** | `src/ao_shaping/tools/slm/transfer/` + `src/ao_shaping/gui/slm/shaping_controls.py` | 单一入口 `run_session` / `TransferSession`（CLI `python -m ao_shaping.tools.slm.transfer`，GUI 闭环整形「迁移学习」同入口）: `capture.py` 采集 + `to_panel`、`workflow.py` 阶段编排与 journal、`validate.py` A–B–B–A 验收、`fit.py` / `adaptation.py` / `seeding.py` / `pretrain.py`。每次 CCD 采集同时写入 Recorder 记录（`recording.py`：`process.csv` + `process.h5` + 过程 GIF，设备关闭后才导出），大尺寸目标用 `--window-span/--window-step/--zero-radius/--core-margin/--control-grid` 打开采样窗与控制分辨率。接受相位自带载波（`ramp_matrix`），下发时跳过侧栏载波层 |
| **多形状物理预测模型闭环整形（从零训练）** | `src/ao_shaping/runners/slm/transfer_shapes_runner.py` | 单一 runner 编排「从零训练物理预测模型 → 逐形状 CCD 锚定整形 → 共同重放 → Recorder 记录」：探针采集 (`capture`) → 名义初始化 (`bootstrap`) → 拟合 (`fit`) → 控制分辨率导出 (`geometric_seed`) → 四段整形 (`validate`：非 GS 初相位 / Mono12 / 续跑 / 稠密) → 同曝光同位深重放 (`replay_phases`) → 逐 CCD 块的 Recorder 记录 (`transfer/recording.py`，CSV + HDF5 + 过程 GIF)。CLI: `slm-transfer-shapes`。初相位有三种构造并**实测择优**：传输 (`square_transport_phase`，仅正方形精确) / 反投影 (`target_backprojection_phase`，形状通用但受控制 Nyquist 限制) / GS (`gs_seed`) |
| Zernike 工具 | `src/ao_shaping/utils/wavefront/zernike_utils.py` | 系数解析 (Noll/(n,m)/数组) + 相位生成，Noll 1976 约定 |
| **SLM 闪耀光栅衍射效率 / 相位响应标定** | `src/ao_shaping/tools/slm/calibration/diffraction_efficiency/` | 2f 实验系统里用一维闪耀光栅测器件的相位调制深度与衍射效率: `core.py` (纯算法: 图案/阶次识别/**离散器件模型**/拟合/诊断) + `simulate.py` (数字孪生: 填充因子/LUT/面板相差/离焦) + `measure.py` (Santec+Daheng 硬件后端) + `artifacts.py` (CSV/JSON/PNG)。CLI: `python -m ao_shaping.tools.slm.calibration.diffraction_efficiency {simulate,measure,analyze}`。**硬件前置条件**: 必须先有光路标定 `.npz` (几何标定产出), 否则无法预测 ±1 级落点。⚠️ **可信范围** (见 `report/slm/diffraction_efficiency/sim_nonideality_20261022.md`): **相对效率曲线可信, 但拟合出的相位深度只在"幂律器件 + 在焦 + 填充因子≈1"三前提同时成立时可信** —— 填充因子 0.8 会把标称器件报成过调制 (+16%), 非幂律 LUT 偏 +42%, 离焦偏 +11%。**光斑宽度门禁** (`--fwhm-gate-ratio`, 默认 3×; `--reference-fwhm-px` 给一条**在焦**的绝对参照) 把这类失真变成显式失败: 门禁触发时越限点与"零级被熄灭"的点**一并**退出拟合, 有效点不足即**拟合报错**而不是给出一个偏了的深度 |
| RL training | `src/ao_shaping/optimizer/rl/` | SAC, LR-WFS |
| Simulation | `src/ao_shaping/drivers/sim/` | Digital twin devices |
| **2f 数字孪生（可复用）** | `src/ao_shaping/drivers/sim/bench_twin.py` | 参数化 2f 实验系统：把物理非理想因素注入传播链 —— SLM 的 LUT 非线性+量化、面板静态相差（Zernike+粗糙度）、装调（相机离焦/光路倾斜/透镜像差）。口径=瞳面区域→CCD 远场（与 region/padding/像元标度约定一致），默认全零误差=理想系统。与既有 `slm_pib_sim.py`（面板级 FFT 仿真）、`fouriergsnet_env.py`（带真实 λ/f/像元节距的傅里叶环境）并存，互不替代 |
| Utilities | `src/ao_shaping/utils/{io,image,wavefront,slm}/` + root `cli_params.py` | 4 子包: io/, image/, wavefront/, slm/ (spots_calc, wavefront_calc, zernike_calc, display 等) + 零导入叶子 `cli_params.py` |
| Standalone runners (未注册) | `src/ao_shaping/runners/` | `runners/slm/offset_runner.py` 计划迁移至 `tools/slm/`。⚠️ `slm_shaping_runner` **曾经**在此列 (2026-10-05 前它未注册); 现以 `slm-pib` **命令组**注册（子命令 `spgd` / `heuristic` / `square`）；⚠️ 无 `spgd-square` 顶层命令 ⇒ 写 `slm-pib square` |
| ML training | `src/ml/` (standalone, not inside `ao_shaping/`) | U-Net+GAN, trainer, wandb_logger |
| 硬件相位→相机图像 DataLoader | `src/ml/hwdataset/` | `data/debug` 全量转 PyTorch Dataset: 输入=SLM 相位+曝光, 输出=CCD 画面 (见 [docs/ml_hwdataset.md](docs/ml_hwdataset.md) 的 `## ml/hwdataset/ Module — 硬件调试转 Dataset` 节) |
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
| `utils/io/` | `file`, `timestamp`, `cli_helpers`, `device_config`, `network`, `handler`, `recorder_keys`, `npz_summary` | `handler` becomes a backward-compat shim to `display/`; `recorder_keys` (2026-10-20 新增, 跟进 `1a11e60e`, #130 item 2): Recorder 记录字典键名的单一来源——11 个 `RECORD_*` 值成了一份盘面列契约 (HDF5/CSV sidecar 列名), 全仓 runners/optimizer/tools/ml 广泛 import;零导入叶子 (理由同 `utils/cli/params`), 由 `tests/ao_shaping/utils/test_recorder_keys.py` 钉住, 改键必须走显式迁移; `npz_summary` (2026-10-25 新增, 跟进 `d14d9dc0`): 探针脚本共用的压缩 `.npz` 摘要写盘单一实现 (`save_summary_npz`: 建父目录 + `np.savez_compressed` + log), `tools/zernike/abba_probe` 与 `tools/ccd/floor_probe` 的两份逐字节相同副本收拢于此, 域内旧名保留薄封装 |
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
| `slm/transfer/` | `jacobian_guard.py` | 逐块 Jacobian 方向校验与它驱动的信任域收缩 (免设备): 用每个 CCD 校验块本来就采的 `A1`/`B` 两端差分, 拿模型预测差与实测差的**相关性**判方向是否可信, 不可信就收缩信任域 |
| `slm/transfer/` | `light_guard.py` | 光路无光的早判与专用退出码: `OpticalPathInvalid` (`exit_code = 3`) + 曝光阶梯内的 `on_step` 守卫 (`LightGuard`, 峰值停在噪声底即中止) + 启动前 `verify_light`; 判据是「峰值 vs 该帧自己的噪声底」, 与探针的 `weak_path` 分工 (后者管「能测出来但已衰减」) |
| `slm/transfer/` | `parallel_fit.py` | 拟合并行监视进程 (不碰设备): capture 独占 SLM, 本进程流式写 `loss_trace.jsonl` / `parallel_fit_provenance.json`, 损失够好或停止下降时落 `stop_acquisition.request` 让 capture 提前收工 |
| `slm/transfer/` | `protocol.py` | 孔径固定的指标口径与重复统计原语 (CV 单独用无法区分「均匀」与「空」, 故每条结论都配一个相对的桶内能量数) |
| `slm/transfer/` | `preflight.py` | 迁移学习闭环整形的启动前参数检查 (2026-10-26 实测: `empty target core` 报错实为载波周期参数问题而非几何问题) |
| `slm/transfer/` | `progress.py` | 由增量实验记录离线出指标轨迹 (免设备) |
| `slm/transfer/` | `radiometry.py` | 共享光学增益、辐照背景与曝光无关计数 |
| `slm/transfer/` | `replay_phases.py` | 在共同原生位深与曝光下重放精确归档相位 |
| `wfs/` | `wfs_probe.py` | SLM+WFS 光强与瞳面探针 — 标定前硬件状态检查 (光强是否足够且不饱和) |
| `wfs/` | `wfs_reference.py` | SLM+WFS 参考波前标定与倾斜线性度 (三步流程) |
| `wfs/` | `zernike_correction.py` | SLM Zernike 模式法波前矫正三阶段流程 (2026-09-15, 按实测调优) |
| `wfs/` | `zernike_response.py` | SLM Zernike 模式→WFS 读数响应矩阵标定 (各模式 ±A 推拉) |
| `zernike/` | `sweep_probe.py`, `common.py` | Smooth-Zernike 扫描探针: 以稳定读数采实验系统响应曲线 |

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
main (click.group)                                  # 20 个命令，逐个由 cli.add_command 注册
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
├── slm-transfer-shapes ← runners.slm.transfer_shapes_runner  (物理预测模型多形状闭环整形: 从零训练 + 逐形状整形)
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


> 🔴 **测试排除与分片策略**已外迁到 [docs/dev/ci_testing.md](docs/dev/ci_testing.md)（由 `tests/ao_shaping/scripts/test_pytest_workflow_shards_cover_tree.py::test_every_test_area_is_sharded` 与 `tests/ao_shaping/scripts/test_pytest_workflow_shards_cover_tree.py::test_no_harness_ignore_blanks_a_whole_test_area` 钉住）：
> `pyproject.toml` 的 `addopts` 默认 `--ignore` 区域、drivers 排除的撤销与可见性事故、
> `--hardware` 门必须位于 `drivers/conftest.py`、本地剪除与 CI 收集必须同向等细节。

### Environment Setup

```bash
# Create virtual environment with uv
uv venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
uv add -e .
```
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
## Configuration

Pytest configuration in `pyproject.toml`:
```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
```

VS Code settings in `.vscode/settings.json` (machine-local, untracked) set PYTHONPATH to `src` and `libs` directories.

## Code Style Guidelines

完整编码规范（Python 3.12+、绝对导入、类型注解、命名、日志、异常处理、DataClass/Enum、
文档字符串、硬件驱动接口、性能优化）见 [docs/dev/coding_conventions.md](docs/dev/coding_conventions.md)（ruff 单一精确版门禁由 `tests/ao_shaping/test_conventions.py::test_all_ruff_pins_agree_and_are_exact` 钉住；ML 训练入口的 ruff 范围由 `tests/ao_shaping/ml/test_train_cli_runnable.py::test_ruff_excludes_nothing_under_src` 钉住）
（前 11 节为完整版）。原 AGENTS.md 的 `Code Style Guidelines` 全节已外迁。

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
| **Full-frame centre leaking into window-local metrics** (`slm_zernike_pib.py`) | `cam.reset_window(...)` returns the window centre in **full-frame sensor coordinates**, but every downstream metric (`rms_pib_terms`, `ImageTargetFunc.radius`, `spot_waist_sigma`, `reference_center`) operates on the **re-windowed** image and needs **window-local** coordinates. Hardware-verified bug (2026-09-23): the runner's `-c auto` probe locates the 0-order on the un-windowed camera (e.g. (633,934) on 1944×2592) → optimizer used it directly as window-local → every metric read **0** (pib=0, r_bucket pinned at 0.9 = 1px radius × shrink 0.9) across the whole heuristic matrix. Fix: after capturing the re-windowed `init_img`, re-locate `reference_center = zero_order_center(init_img)` and use `reference_center` (NOT the post-`reset_window` `center`) for the r-bucket radius too (L869/L916, 2026-10-28 重钉：`2ee69348` 把几何段上提到模块级 `_prepare_geometry`，原 L1396/L1441 失效). Rule: whenever a centre is handed to a metric/ROI builder, verify which coordinate frame it belongs to first. |
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

完整反模式表（SLM 相位 min-max 归一化、Zernike 系数单位混用、PIB 未去本底、重复实现 Zernike
数学等）已外迁到 [docs/dev/coding_conventions.md](docs/dev/coding_conventions.md) 的
`## 反模式红线（ANTI-PATTERNS）` 节。修改相关代码前务必先读。

## UNIQUE STYLES（项目特有风格）

项目特有风格（Mock-first 测试、Zernike Noll 约定、SLM 相位 raw-only 契约、内存槽轮转、
Streamlit 后台线程、2f 傅里叶面几何、报告生成必须位于 `scripts/` 等）已外迁到
[docs/dev/coding_conventions.md](docs/dev/coding_conventions.md) 的 `## 项目特有风格（UNIQUE STYLES）` 节。

## MISSING INFRASTRUCTURE（CI / 门禁现状）

CI 工作流（ruff 硬门禁、pytest 分片 11 片、fail-closed）、linting（ruff 单一精确版
`0.15.12`）、machine-local 配置、类型检查门禁缺失等现状已外迁到
[docs/dev/ci_testing.md](docs/dev/ci_testing.md) 的 `## 缺失基础设施 / CI 现状` 节。

## CodeGraph（预索引知识图谱）

本仓库已初始化 CodeGraph（开发机上 `.codegraph/` 目录，6111 节点 / 12339 边）。注意：
`.codegraph/` 是 machine-local，**不入库**（2026-10-19，在 `.gitignore`），新克隆不会有它——
此时跳过 CodeGraph 工具，回退到 grep/find/read。

- **Explore 用 `codegraph_explore`** 作为首选工具：一次返回相关文件完整源码段。遵循其调用预算，
  已返回的源码不要重读，仅在 "Additional relevant files" 需深入时用 grep/glob/read 补充。
- **主会话**用轻量工具：`codegraph_search`（按名找符号）/ `codegraph_callers`·`callees`（追踪调用流）/
  `codegraph_impact`（改动前看影响面）/ `codegraph_node` / `codegraph_context` / `codegraph_files` /
  `codegraph_status`（索引健康）。
- 无 `.codegraph/` 目录时完全跳过 CodeGraph——索引是用户自己的决定。
