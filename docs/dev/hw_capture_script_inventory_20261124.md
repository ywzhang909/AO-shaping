# 硬件采集与分析脚本入口清单（transfer/MIL 判定总账 §8 配套）

**日期**：2026-11-24
**目的**：`transfer_mil_issue_validity_verdicts_20261124.md` §8 十行"缺数据采集与分析计划"此前只写冻结口径、没有可执行入口。本文把仓库中实际存在的硬件采集 / 回放 / 离线评估脚本逐一挂到 §8 各行，并记录 CLI 契约要点与安全语义。
**来源**：只读盘点（子代理核验 `--help` 与源码）+ 落笔前主会话逐文件存在性复核。未连接任何设备，未运行任何带 `--execute` 的命令。

## 0. 状态快照与通用纪律

- **并发写入警告**：落笔时同工作树有多个并行车道在增删文件。本清单所有路径在 2026-11-24 深夜落笔时逐一 `Test-Path` 核实存在；被并发车道删除的入口标 **【已消失】**，恢复或迁移由持有该车道者决定。
- **2026-11-25 复核（文档守护）**：上表 5 个「工作树暂缺」条目（`cross_session_check` / `analyze_dark_frame_exposures` / `analyze_content_axis_2x2` / `content_axis_2x2_capture` / `reproducibility_prereg`）现已全部回到工作树；其中 `cross_session_check` 已被 `33049c6c` 修改（import 降级为 `DEPS_MISSING` 哨兵），**禁止**再执行旧版 `git checkout 6c686a00 -- <path>`（会用 `6c686a00` 旧版覆盖回退代码）。其余【已消失】条目（ABBA replay、s1 mapping 脚本、cross_session prereg、ABBA 计划文档）截至 2026-11-25 仍未入 HEAD，保持"真丢失"判定。
- **开设备门禁**：绝大多数采集脚本遵循 `--execute` 才开设备的约定。**例外**：`issue247_cross_session_capture_20261124.py` 根本没有 `--execute` 开关，真实路径自动向三个子进程注入 `--execute`；它的安全演练模式是显式 `--dry-run`。不要套用"不带 --execute 就安全"的惯例。
- **`--help` 不是普遍安全查询**：少数固定入口（如 `analyze_cold_start_cv.py`、`report/sim_tue_methods/t*.py`、`evaluate_psf.py`）没有 argparse，裸跑会执行离线分析并写产物。
- **释放证明**：预注册要求的 open/close/release 以采集目录里的 `released.json` 等产物为准，不能只看进程 exit 0。
- **位深与曝光**：逐入口核对，不得为没有位深参数的脚本编造 Mono12 命令（历史教训见 `efficiency_match_experiment_plan_20261121.md` §181–192 的量纲问题）。

## 1. #247 跨会话复现

| 角色 | 入口 | 契约要点 |
|---|---|---|
| 三会话采集控制器 | `scripts/slm/issue247_cross_session_capture_20261124.py` | 必填 `--out-root` 与三次 `--probe-dir`（s1/s2/s3 各自六步光路预检）；`--dry-run` 只打印三条命令；`--expect-plan-sha256` 冻结计划哈希；`--wait-until-idle` 默认在飞保护。**无 `--execute` 开关，真实运行自动注入，演练必须 `--dry-run`。** |
| 单会话几何采集 | `scripts/slm/issue247_geometry_capture_20261120.py` | `--plan {frozen,waveform,attribution,reproduction}`；plan-only 默认，`--execute` 才开设备；`--self-test` 离线自检。 |
| 跨会话复现检查 | `scripts/slm/issue247_reproducibility_check_20261124.py` | 离线；必填 `--dir --baseline-dir --probe-dir --out-dir`；消费 `released.json`；输出目录必须与采集目录分离。 |
| 盲评审计 | `scripts/slm/issue247_core_blind_audit_20261124.py` | 离线；必填 `--dir --out-dir`；默认拒绝覆盖。 |
| 总账离线判定器 | `scripts/slm/issue247_cross_session_check_20261124.py`（2026-11-25 复核：已在工作树，且被 `33049c6c` 更新为 import 降级 `DEPS_MISSING` 哨兵版） | 离线；必填 `--root --out-dir`；`--root` 需含 `cross_session_manifest.json` 与 s1/s2/s3。 |
| 波形求解 / 轴映射 | `scripts/slm/issue247_waveform_probe_solve_20261122.py`；`scripts/slm/issue247_axis_map_solve_20261121.py` | 离线；注意 waveform 求解默认把分析产物写回采集目录。 |

缺口：三会话复现、轴映射冻结、校正前后实测配对是三个递进实验，控制器只覆盖第一步。

## 2. #234/#236/#254 强度/方向门与效率配对

| 角色 | 入口 | 契约要点 |
|---|---|---|
| 效率帧矩阵采集 | `scripts/slm/efficiency_match_capture_20261123.py` | `--execute` 才开设备；`--out` 新目录、`--cooldown-s`（≥20 s）；help **没有位深参数**，Mono12 口径需另行核实；`--dark-frames-dir` 复用既有暗场。 |
| 效率分析 | `scripts/slm/efficiency_match_analyze_20261123.py` | 离线；`--dir`；bootstrap 默认 2000、seed 0；写回采集目录 `analysis.json/EM_REPORT.md`。 |
| 内容轴四帧诊断 | `scripts/slm/content_axis_2x2_capture_20261124.py`（盘点时曾被删、落笔复核时已恢复；已入 HEAD） + `analyze_content_axis_2x2_20261124.py`（2026-11-25 复核：已在工作树） | capture 契约：`--execute` 才开设备，支持 `--bit-depth`。 |
| 暗帧评估 | `scripts/slm/analyze_dark_frame_exposures_20261124.py`（2026-11-25 复核：已在工作树，`8032b3f6` 新增） | — |
| 闭环/replay 通用外壳 | `scripts/slm/validate_transfer_shapes_hardware.py`；`scripts/slm/replay_transfer_shapes_hardware.py` | validator 安全演练模式是 `--plan-only`，支持 `--bit-depth {8,12}`、`--initial-phase`、`--seed-run`；replay 外壳无 `--execute` 拒绝且不导入设备模块，持锁贯穿、finally 释放并写 `device_lock.json`。 |

缺口：efficiency-match 固定 16 帧与内容轴四帧都是**诊断入口**，不等同于 train/selection_validation/final_test 三层隔离、适配总墙钟与完整算法 2×2 交错的终验执行器。

## 3. #238 灰度 g₂π 与串扰验收

| 角色 | 入口 | 契约要点 |
|---|---|---|
| 灰度相位探针 | `scripts/slm/issue238_gray_phase_probes.py` | 必填 `--out`；`--execute` 才采集；`--mode {blaze,binary}`；`--exposure-bracket MS...` ≥4 档；`--power-reference {off,abba}`；`--dark-frames/--dark-at {pre,post,both}`。**help 自述无熄灯 hook：同曝光熄灯暗场仍须独立采集并记录来源。** |
| 二值拟合 | `scripts/slm/issue238_binary_fit.py` | 离线；必填 `--run`；默认拒绝覆盖 `<run>/fit.json`。 |
| LUT 棋盘格 | `scripts/slm/calib_slm_lut_checkerboard.py` | 不传 `--execute` 只打印计划。 |
| LUT 探针 | `scripts/slm/capture_slm_lut_probes.py` | 有 `--execute` 门；**help 无 `--wait-until-idle`、无曝光 bracket、无位深**，不满足 §8 正式 g₂π 验收硬门。 |
| 周期扫描 | `scripts/experiments/run_issue238_period_sweep_20261119.py` | 顶层执行门禁未核定，裸跑安全性未确认。 |

## 4. PSF / refit 闭环

| 角色 | 入口 | 契约要点 |
|---|---|---|
| refit 前向 | `scripts/slm/refit_transfer_forward.py` | 离线；必填 `--dataset --checkpoint --out`（输出目录必须不存在）；`--dry-run` 只评估一个 loss、不写文件。 |
| 跨会话评估 | `scripts/slm/evaluate_transfer_on_new_session.py` | 离线；必填 `--original --new --fit --out`；`--skip-validation` 只用于调试，不得作为正式验收命令。 |
| ABBA 回放 | `scripts/slm/replay_closed_loop_abba_20261124.py` **【已消失；未入 HEAD，删除即真丢失】** | 原契约：`--phase-a/--phase-b/--mask-npz/--out`，`--execute` 才开设备，支持 `--bit-depth`。 |
| PSF 评估 | `data/debug/slm_transfer/20261124_transfer_psf_v1/evaluate_psf.py` | 冻结实验固定入口，无常规 argparse，不补造参数。 |

缺口：静态 PSF 双门与 refit 实测闭环收益是两个实验；refit ABBA 计划文档 `report/shape_trials/physical_refine/closed_loop_refit_abba_plan_20261124.md` 与其入口同时被并发车道删除。

## 5. #137 热启动交错

**无完整契约执行器**（同连接 cold/hot 各 N≥3、探针预算 4→3→2 的交错编排尚不存在）。可复用组件：`scripts/slm/bootstrap_transfer_checkpoint.py`（离线构造 checkpoint，必填 `--transfer-dir --geometry-calib --out`）、`scripts/slm/generate_cold_start_report.py`（离线报告，`--render-only` 只回看）；`analyze_cold_start_cv.py` 为硬编码路径的固定离线入口，无 argparse。`validate_transfer_shapes_hardware.py` 的 `--initial-phase/--seed-run` 可承载初相位控制，但本身不构成该契约。

## 6. #139 不确定性探针

`scripts/slm/transfer_uncertainty_experiment.py`：必填 `--root --checkpoint`；`--execute` 开设备、`--render-only` 只渲染已有结果（help 自述不拟合、不开设备）；支持 `--iterations --span --seed`。离线对照：`report/sim_tue_methods/t4_uncertainty_sampling.py`（固定入口，无 argparse）。缺口：U(p) 与相机窗口脱节修正、几何门、N≥3 同种子配对是否已在脚本内实现，本轮未确认，不能仅因入口存在认定计划可完整执行。

## 7. #124/#126 正向增量

按 §8 契约**先 sim 达门、未达门不安排硬件**，本就没有硬件配对入口。离线复现入口即总账 §9 指定的 `report/sim_tue_methods/t1_noise_relaxation.py`、`t2_mixed_fitting.py`（固定入口）；辅助筛选 `scripts/slm/analyze_sim_differentiable.py`、`sweep_differentiable_lr.py`、`sweep_differentiable_multistart_batch.py`（`--out` 运行时必填）。

## 8. #119/#121/#165 模型准入

| 角色 | 入口 | 契约要点 |
|---|---|---|
| 硬件保真阶梯 | `scripts/slm/hw_forward_model_fidelity_ladder.py` | 无 `--execute` 默认 dry-run，另有显式 `--dry-run`；`--wait-until-idle` 在飞保护；help 未见 Mono12 位深开关。 |
| 离线保真分析 | `scripts/slm/analyze_model_hw_fidelity.py` | `--run-dir` 可重复；`--all-differentiable-opt`。 |
| 模型对比 | `scripts/ml/freeform_vs_zernike.py` | 固定离线入口，无 argparse。 |
| 闭环/replay 外壳 | `scripts/slm/validate_transfer_shapes_hardware.py`；`replay_transfer_shapes_hardware.py` | 同 §2。 |

缺口：多次独立运行准入与 #165 ≥3 对交错收益终验没有统一执行器，需用上述组件自行编排。

## 9. #213 L-BFGS 物理 ABBA

离线冻结筛选：`scripts/slm/sweep_kaxis_frozen_20261123.py`（`--run-dir --baseline-json --ks --seeds --iters --lr`；**CLI 无 `--optimizer`，不能编造 L-BFGS 开关**）。硬件侧可复用 `scripts/slm/hw_differentiable_phase_opt.py` 与通用 validator/replay 外壳，但 Adam vs L-BFGS 同预算物理 ABBA 执行器未确认存在；`hw_plan_consolidated_20261118.md:96` 已登记"Adam 120 iter vs L-BFGS 早停 ABBA ≥4 次待物理复验"。

## 10. #156 七因子归因

无覆盖 phase-stepping 帧栈、jitter 周期量化、闭环内去相关因子的专用采集 CLI。可复用：`scripts/slm/hw_forward_model_fidelity_ladder.py`（采集保真诊断数据）、`scripts/slm/analyze_model_hw_fidelity.py`、`analyze_session_time_budget.py`（必填 `--session`）、`audit_physical_test_time.py`（离线但非纯只读，默认可能测量落盘耗时）。七因子实现在 `src/ao_shaping/utils/slm/fidelity_diag.py`（模块，非 CLI）。保真阶梯与时间审计不能替代七因子真实测量；默认 1.0 因子不能当实测证据。

## 11. 落笔时被并发车道删除的入口汇总

| 路径 | 状态（以 HEAD + 落笔复核为准） |
|---|---|
| `scripts/slm/issue247_cross_session_check_20261124.py` | 已入 HEAD（`6c686a00`）；落笔时工作树暂缺 → 2026-11-25 复核已回到工作树，且已被 `33049c6c` 更新（`DEPS_MISSING` 哨兵降级）；**勿用 `6c686a00` 旧版 checkout 覆盖** |
| `scripts/slm/analyze_dark_frame_exposures_20261124.py` | 已入 HEAD（`8032b3f6`）；2026-11-25 复核已在工作树 |
| `scripts/slm/analyze_content_axis_2x2_20261124.py` | 已入 HEAD（`2c76e6b4`）；2026-11-25 复核已在工作树 |
| `scripts/slm/content_axis_2x2_capture_20261124.py` | 已入 HEAD（`c5b48717`）；盘点时曾被删、复核时已在工作树（波动） |
| `scripts/slm/s1e_same_bytes_probe_capture_20261124.py` | 已入 HEAD（`94282236`）；复核时已在工作树 |
| `scripts/slm/replay_closed_loop_abba_20261124.py` | **未入 HEAD，删除即真丢失**（2026-11-25 复核仍未恢复） |
| `scripts/slm/capture_s1_mapping_disambiguation_20261124.py` | **未入 HEAD，删除即真丢失**（仅剩 `__pycache__` 编译产物；2026-11-25 复核仍未恢复） |
| `docs/dev/hw_window_20261124_issue247_cross_session_prereg.md`（总账 §8 #247 行引用的预注册计划） | **未入 HEAD，真丢失**；§8 该行为悬空引用，待持有车道补回或改指（2026-11-25 复核仍未恢复） |
| `docs/dev/hw_window_20261124_issue247_reproducibility_prereg.md` | 已入 HEAD（`b0096beb`）；落笔时工作树暂缺 → 2026-11-25 复核已在工作树 |
| `report/shape_trials/physical_refine/closed_loop_refit_abba_plan_20261124.md`（总账 §8 PSF/refit 行引用） | **未入 HEAD，真丢失**；§8 该行为悬空引用（2026-11-25 复核仍未恢复） |

以上删除均非本车道所为；本车道未恢复、未改写任何被删文件。
