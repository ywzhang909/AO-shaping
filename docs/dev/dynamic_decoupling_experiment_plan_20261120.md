# 动态整形·速度解耦采集实验方案（冻结版 2026-11-20）

## 摘要

本方案冻结「动态整形速度解耦采集」实验的目标矩阵、相位方法、时间与曝光策略、数据采集口径和验收记录要求。实验目的：在一台 SLM 与一台 CCD 单进程开闭的前提下，把 **SLM 灰度帧派发节拍**与 **CCD 测量节拍**解耦，量化不同形状/尺寸目标在过渡过程中的动态响应。用户已于 2026-11-20 通过交互页确认全部关键参数，本文件为冻结记录；后续如需改动任一冻结项，须重新走用户确认并更新本文件。

**重要边界**：本实验与 transfer `light/none` 科学验收是两条独立工作线。P4 双门（强度门、方向门）均为有效 `fail`，transfer `light/none` 继续暂停，不进入 P5，不启动其硬件 `--execute`。本实验中的 transfer 模型相位**仅作诊断**，不得当作 P4 资格通过证据，也不得据此启动 `light/none`。

## 1. 实验目标与判据

| 项 | 内容 |
|---|---|
| 主目标 | 验证 SLM 帧派发与 CCD 采样两路节拍可独立推进，互不串行阻塞 |
| 观测量 | 每个过渡的 SLM 派发帧数、CCD 实测帧数、实测采样率、逐帧峰值/饱和像素/均匀度、原始 CCD 帧 |
| 判据 | 采集完整（每条 transition 有 `frames.npy` + `frames_meta.json` + `transition.json`）、设备开闭各一次且正常释放、`run_summary.json` 中 48 条均 `status=ok` |
| 不判据 | 本实验**不**重判强度门 δ=0.20、方向门 10/14；不用它给模型或 checkpoint "翻案" |

## 2. 冻结参数（用户 2026-11-20 确认，不得擅改）

### 2.1 目标矩阵

- 形状（6）：`square`、`circle`、`triangle`、`star`、`rectangle`、`annular`
- 尺寸（4）：`diameter = 200, 500, 1000, 1500`，单位为 **CCD 目标平面原生像元**（CCD 2592×1944，像元 2.2 µm）
- 尺寸语义：
  - `circle`/`triangle`/`star`/`annular`：diameter = 外接圆直径；
  - `square`：size = 边长（等于该数值）；
  - `rectangle`：长边 = 该数值，短边 = 1/2（长宽比 2:1，沿用现有 `--aspect-ratio` 默认值）；
  - `annular`：外径 = 该数值，内径 = 0.6×外径。
- 目标总数：6 × 4 = 24；相位方法 2 种 → 48 条相位/transition。

### 2.2 相位方法（用户确认「两种方法都采集」）

1. `gs`：Gerchberg–Saxton，从目标振幅离线反演相位，独立基线。
2. `model`：用现有冻结物理预测模型（`data/debug/slm_transfer/p3_fit_20261120/frozen_parameters.json`）做可微优化离线生成相位，**仅作诊断，标记为 `model_phase_diagnostic_only=true`**。

> 约束：`slm-transfer-shapes` 是 CCD 锚定的硬件闭环（每个 block 靠实测 core CV 接受/拒绝），不是离线相位生成器；其 `initial-phase` 枚举的是初相位种子算法（auto/transport/backprojection/gs/plane/random），不是优化器。本实验不走该 runner 生成相位，只用底层 `load_frozen_parameters` + `DifferentiableBeamOptimizer` 做离线 model 相位。P3 checkpoint 双门 fail，不得当作已资格化的控制模型。

### 2.3 时间与节拍

| 参数 | 冻结值 |
|---|---|
| 起点 | flat gray 0（`--start-flat 0`，绝不过 `create_phase_from_array`） |
| 单次过渡时长 | 5 s（`--duration-s 5`） |
| SLM 派发帧率 | 10 fps（脚本默认，计划阶段复核） |
| CCD 测量策略 | 每帧测量（`--measure every --measure-every 1`） |
| 位深 | 12-bit |
| 曝光 | 先做独立 12-bit 曝光探测，按结果冻结统一曝光值，再用于全系列（不做逐帧自动曝光，避免横向比较失真） |
| 原始帧 | 保存（`--raw-frames stack`，每帧 uint16 + 时间戳 + 统计量） |
| 过渡间隔 | 0.5 s SLM settle（背对背 open/close 会触发驱动板假死，本实验单进程开闭，间隔仅稳定面板） |

### 2.4 设备与资源

- 设备：Santec SLM #1（1064 nm，`22030108`）+ Daheng MER2-507-23GM CCD（`FJB24112232`，12-bit，全幅 2592×1944，不做物理开窗）。
- 单进程开 SLM + CCD 各一次，48 条 transition 串行共用；`--wait-until-idle 120` 持锁，`try/finally` 释放。
- 存储估算：单帧 2592×1944×2 B ≈ 9.62 MiB；每过渡约 51 帧 → 48 × 51 ≈ 2448 帧 ≈ **22.98 GiB**（未压缩、uint16）。D 盘可用空间需在执行前复核（冻结时约 124 GiB）。

## 3. 分阶段任务

| 阶段 | 任务 | 产物 | 门禁 |
|---|---|---|---|
| T0 | 冻结本方案（本文件） | 方案 md | 用户已确认 |
| T1 | 离线生成 24 目标 × 2 方法的相位与 manifest，核查网格代表性 | `data/debug/slm_transfer/dyn_phase_matrix_20261120/` | 聚焦测试 + ruff + 代表性结论 |
| T2 | 扩展采集器支持多端点计划 + 原始帧存储 + 批次记录 | `scripts/slm/dynamic_shaping_capture_20261120.py` + 测试 | 聚焦测试 + ruff + plan-only 冒烟 |
| T3 | 曝光探测（12-bit，实验） | 探测记录 | 单独设备窗口，冻结统一曝光 |
| T4 | 48 条 transition 串行采集（实验） | `<out>/<label>/` + `run_summary.json` | 设备纪律 + 完整产物 |
| T5 | 中文正式报告（摘要/图表/任务项/指标参数，存 `report/shape_trials/`） | 报告 md + 图 | 用户偏好格式 |

T1/T2 为离线车道，可并行；T3/T4 需独占设备，串行执行；**T3/T4 开始前须再次获得用户明确的执行许可**（本方案冻结不等于启动 `--execute`）。

## 4. 硬件设备纪律

占用 SLM/CCD/激光前后，须在回复中明确报告：占用了什么设备、做了什么、何时释放、是否正常关闭。设备被占用期间不并行启动第二个占用同一设备的进程。所有设备命令带 `--wait-until-idle 120`。采集期间保持全幅，不做物理开窗。实验前按 `ao-optical-path-calibration` skill 完成设备在线/光路有效/曝光不超量程/SLM 真调制探测。

## 5. 验收与记录

- 每条 transition 必含：`frames.npy`（memmap，容量 ≥ 实测帧数）、`frames_meta.json`（含 `n_measured`、`effective_full_scale`、`exposure_ms`、`achieved_measure_fps`、逐条时间戳/峰值/饱和/均匀度）、`transition.json`（起止路径 SHA256、group、帧数、时长、easing、measure、exposure、bit_depth）。
- 顶层：`run_plan.json`（解析后绝对路径 + SHA256）、`run_summary.json`（48 条状态）、`frames.jsonl`、`device_log.json`（含 open/close 结果与 settle）。
- 报告按用户偏好：简体中文，含摘要、图表、任务项、指标参数，保存到 `report/shape_trials/`。

## 6. 冻结哈希与出处

- 用户确认交互页（2026-11-20）：目标形状/尺寸、方法=GS+模型、尺寸=CCD 目标平面、暴露策略=先探测后冻结、输出=保存原始帧、时长=5 s、measure=every 1、12-bit、起点 flat 0、曝光=自动探测。
- 前序记录：`docs/dev/agent_notes_20261120.md` §9.1（速度解耦脚本离线审查）；P4 双门 fail 见同文件 §9 与 `data/debug/slm_transfer/p3_gate_20261120/{strength_gate,direction_gate}.json`。
- 本方案冻结日期：2026-11-20。实现脚本/产物哈希在 T1/T2 完成后补录到 §6。

### 6.1 T1 / T2 / T3 实测补录（2026-11-20）

- T1 相位矩阵：`data/debug/slm_transfer/dyn_phase_matrix_20261120/manifest.json`，
  `fingerprint=29d287e76202fd2c70eab017f4b1bb06824a31e7da24e28cc8ee04c72b61b160`；
  checkpoint SHA256 `380b9b87885fe70a4279dbe00fc500dcc3893b9422ddc84ccd0115e491c6619a`。
  48 条计划 = 30 条有效相位（24 GS + 6 model@200 px）+ 18 条 `skipped_unrepresentable`，
  `failed=0`；逐文件 SHA256 与形状核验 0 问题。报告见
  `report/shape_trials/target_phase_matrix_20261120.md`。
- T1 聚焦测试：`pytest ... test_target_phase_matrix... test_dynamic_shaping_capture_plan...
  test_dynamic_shaping_capture... -q` → 19 passed；ruff → All checks passed。
- T3 12-bit 曝光探测：`data/debug/optical_path_probe/speed_decouple_20261120/20261120_030220Z/`，
  `probe_status=ok`，可用 0.03–0.12 ms，**冻结 0.08 ms**（peak 2212.6 / 饱和阈值 3890.25，
  未饱和），离焦响应 `responds=true`；SLM/CCD 于 `2026-11-20T03:04:46Z` 正常释放。
- 计划构建器：`scripts/slm/build_speed_decouple_plan_20261120.py`；模型 128×128 → 面板
  `(1200,1920)` 复用 P3 规范 `capture.to_panel`（块复制 pitch=15，裁 `[:1200]`），只是诊断相位。
- 最终批次计划：`data/debug/slm_transfer/speed_decouple_plan_20261120.json`，30 条 transition
  （24 GS + 6 model），`panel=[1920,1200]`、`duration_s=5`、`fps=10`、`easing=smooth`、
  `measure={every,1}`、`exposure_ms=0.08`、`bit_depth=12`、起点 flat gray 0。
  plan-only 复核：`total_frames=1530`、`expected_total_duration_s=156.000`、
  输出目录未创建、exit 0。

## 7. 未决项（不阻塞离线阶段）

- 12-bit 曝光探测的具体档位与最终冻结曝光值（T3 产出，执行前复核）。
  **已结案**：冻结 0.08 ms（见 §6.1）。
- 网格代表性：模型 `grid_size=128/fft_size=512` 对 1500 CCD px 目标是否可表示，由 T1 给出具体数字；不可表示的 model 相位须标诊断专用，不静默产出。
  **已结案**：模型可用半径 245.35 CCD px，500/1000/1500 均不可表示（18 条 `skipped_unrepresentable`），仅 200 px 的 6 条 model 相位产出并标诊断专用。

### 7.1 执行口径澄清：30 条而非 48 条

方案 §2.1 名义 24×2=48 条相位；T1 实测模型网格只能表示 `size=200`。按 §7「不可表示的
model 相位须标诊断专用，不静默产出」，实际带相位的 transition = 24 GS + 6 model(200 px) = **30 条**。
24 个目标全部由 GS 覆盖（目标矩阵完整）；model 只在可表示的 200 px 上作诊断对照，不伪造缺失相位，
也不因缺相位删减 GS 目标。采集输出目录若因中断/失败产生，必须换新目录，不得复用。

### 7.2 执行记录与实测结论（2026-11-20，T4/T5 已完成）

- **T4 采集完成**：新目录 `data/debug/slm_transfer/speed_decouple_capture_20261120_v2/`，
  30 条 transition（24 GS + 6 model 诊断）全部 `status=ok`，无错误日志；各条 `frames.npy` 为
  `(52,1944,2592)` uint16、全非空，冻结曝光 0.08 ms、`effective_full_scale=4095.0`。
  首轮 `RawFrameStack` 按面板 shape 预分配致原始帧全零的缺陷已修复（改按首帧 CCD 全幅惰性预分配），
  首轮目录改名 `..._INVALID_firstrun` 保留证据、禁止复用。
- **实测节拍**：实测约 **2.3 fps**（fps 中位 2.284 / 均值 2.319），每条 5 s 实下发 12–14 帧，
  远低于计划的 10 fps / 51 帧（后者是计划值，不代表实验帧数）。瓶颈是 **SLM 写入+换槽**（中位 0.312 s，
  最大 0.562 s），不是 CCD 曝光读出（中位 0.075 s）或 LCOS 像素翻转等待（中位 7.8 ms）。
  墙钟驱动循环在硬件慢时跳帧而不拉长过渡，末帧 `progress_t=1.0`。
- **T5 正式报告已落盘**：`report/shape_trials/speed_decouple_capture_20261120.md`（含图
  `report/shape_trials/figures/speed_decouple_20261120_{fps,timing}.png`）。
- **边界不变**：model 相位仅诊断（`model_phase_diagnostic_only=true`），P4 双门 fail 不受影响，
  transfer `light/none` 继续暂停；GUI 接入属独立工作线（issue #252），不在本方案范围。
