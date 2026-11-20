# Model-in-loop 链条移除报告

<!-- provenance:start -->
> **生成脚本**: 人工撰写，无生成脚本
> **运行环境**: 离线记录（移除操作当时的边界说明，不占用设备）
> **说明**: 2026-11 model-in-loop 优化链移除（af9a55a0）时的边界记录：保留哪些中立实验系统标定、删除哪些实验链、哪些文档引用随之改写。后由 b0108349 恢复优化链，本文仅作历史移除边界的事实记录，不代表当前代码状态。人工撰写。
<!-- provenance:end -->

## 结论

移除实验系统 SLM 整形中的 model-in-loop（MIL）优化执行链，保留传递侧（transfer）路径，并把仍被实验系统标定、几何诊断与报告生成使用的通用能力迁入中立模块。历史脚本作为只读复现材料归档，不再作为当前产品入口或受支持的运行路径。

本决定表示：现有证据没有证明 MIL 相对保留路径带来可接受且可复现的收益；并非宣称所有模型辅助方法在原理上都无价值。

## 决策依据

截至本次清理前，MIL 相关候选和实测结果未达到预先设定的验收标准：

- 变体 A/B 的硬件 对比择优 均为 `bake_off_rejected`，整形后得分未超过平场基线；放开验证集门槛没有改变结局。
- 均匀度实验中的目标框 CV 从 `0.5627` 到 `0.4454`，但这项单次指标改善不足以建立端到端、可重复且优于保留路径的收益。
- 全频随机相位 holdout 的模型相关性约为 `0.108–0.277`，低于 `0.5` 的接受阈值；不能支撑将预测模型用于该参数空间中的优化闭环。
- 没有建立逐位可复现的 MIL 结果链。相较之下，transfer 路径仍是当前应保留的实验路径。

证据概述及物理预测模型的经验教训见[模型闭环总述](model_in_loop_overview.md)和[物理预测经验总结](../physical_forward_experience.md)。上述结论限定于仓库现有记录和验收门槛；不得把单个失败实验外推为对所有模型辅助优化方法的普遍否定。

## 移除边界与迁移

| 区域 | 处理 | 说明 |
| --- | --- | --- |
| MIL 优化核心 | 删除 | 删除 MIL 专用优化器、信赖域、探针不确定度选择器、运行器、重放/汇总与扰动对比分析工具，以及对应 CLI/显示/导出入口。 |
| 标定与几何能力 | 抽取并保留 | `optimizer/wfless/bench_geometry.py` 保留实验系统几何定位、扫掠判定与区域逻辑；`optimizer/wfless/frozen_aberration_shaping.py` 保留冻结像差整形所需的中立逻辑。 |
| 会话标定配置 | 保留并中立化 | 由独立 `runners/slm/session_calibration.py` 承载实验系统标定所需的窄类型；GUI 控件及基准设置改为依赖中立标定模块，而不是 MIL 会话模块。 |
| 共享 Zernike 优化 | 保留 | `ZernikeCoefficientOptimizer` 仍被 `calibrate_bench_geometry` / `calibrate_shared_aberration` 等标定运行时依赖，不能因名称或旧调用链而一并删除。 |
| 几何发现 | 保留 | `discover_model_in_loop_geometry` 仍用于既有几何发现/标定消费者，并改用中立 `bench_geometry` 接口；该名称兼容旧语境，不表示仍存在 MIL 优化器。 |
| 脚本 | 归档例外 | MIL 专用脚本移入 `scripts/slm/_archive/`，仅作为历史复现材料；实验系统校准 runbook、transfer 不确定度实验及实验系统探针报告生成脚本仍是活跃工具。 |
| 实验报告 | 合并与精简 | 中间探测报告/文件和不再维护的 MIL 细节记录移除；实验结论归入总述，物理预测模型记录压缩为经验总结。 |
| 文档与索引 | 清理 | 删除已失效的 MIL 用户指南和命令页面，改为说明保留的实验系统标定能力；更新命令、报告来源及术语/模块索引。第二轮清理覆盖 README、CLI 参考（计数 20 → 19）、GUI 操作图/手册/SRS、相位分辨率决策表、对比实验方案与双优化方案的状态注记，并移除 `ruff.toml` 中已删测试文件的 per-file 豁免。 |

`compare_disturbance.py`、`analyze_disturbance.py` 和 `replay_disturbance.py` 的引用核查结果是：三者只服务于 MIL 评估/重放，不被 transfer 路径调用，因此随 MIL 源码一并删除，而不是保留为 transfer 工具。

## 代码范围

源代码中移除的 MIL 专属模块包括：

- `optimizer/wfless/slm_model_in_loop.py`
- `optimizer/wfless/adaptive_trust_region.py`
- `optimizer/wfless/model_in_loop_shaping.py`
- `algorithm/signal_processing/uncertainty_probe_selector.py`
- `runners/slm/model_in_loop_runner.py`
- `tools/slm/model_in_loop_replay.py`、`model_in_loop_summary.py`
- `tools/slm/compare_disturbance.py`、`analyze_disturbance.py`、`replay_disturbance.py`

同时移除 `slm-model-in-loop` CLI 注册、MIL 显示适配器、结果配置/保存辅助函数和引用这些入口的生成映射。保留文件中的消费者已改为从中立几何/冻结整形模块导入需要的实验系统功能。

测试随受删 API 与执行路径删除；留下的标定、GUI、报告和 transfer 测试继续验证保留接口。测试命令、结果与最终扫描记录以本报告更新后的验证小节为准。

## 风险与缓解

1. **旧脚本导入失效。** 归档脚本仍可能引用已删除源码，归档物不承诺直接运行；应从归档说明中明确这一点，避免被误认为受支持入口。
2. **外部调用依赖被删除的内部 API。** 本次删除了 MIL CLI、运行器和配置/结果保存接口。仓库内活跃消费者须由导入扫描与导入冒烟测试覆盖；仓库外调用方若存在，需迁移到 transfer 或中立标定 API。
3. **名称残留造成误解。** 保留的 `discover_model_in_loop_geometry` 是几何发现兼容能力，不是 MIL 闭环。清理说明应保留此例外，避免后续误删或误认为 MIL 仍可运行。
4. **报告历史压缩。** 汇总报告取代部分中间探测材料；原始实验产物若不在本次归档/保留范围内，不应在结论中暗示仍可从汇总页完整重建。结论必须保留门槛、失败类型与数据限制。
5. **回滚。** 所有改动限定在 `remove-model-in-loop` 隔离 worktree，尚未提交。需要恢复时按变更路径从版本控制恢复对应文件；不要通过恢复整个忙碌主 checkout 或重建已删除 API 的兼容层来回滚。

## 验证记录

所有并行修改汇合后的最终验证记录如下（离线执行，未占用任何实验设备）：

### 导入与入口

- **删除导入扫描**：`src/`、活跃 `scripts/`（排除 `_archive/`）与 `tests/` 中对已删 MIL 模块的
  `from`/`import` 语句为 **0 命中**；`src/` 内剩余提及仅出现在标注「已移除」的注释/文档字符串中，
  用于保留行为来源，不再指向可打开的文件。
- **CLI 冒烟**：`python -m ao_shaping.main --help` 列出 **19** 个命令，不含 `slm-model-in-loop`。
- **导入冒烟**：`bench_geometry`、`bench_geometry_common`、`bench_sweep`、`bench_sweep_fit`、
  `bench_shared_aberration`、`frozen_aberration_shaping`、`runners/slm/session_calibration`、
  `gui/ccd/target_shape_helper`、`gui/ccd/target_shape_plotly`、
  `zernike_coefficient_optimizer`、`tools/slm/transfer/jacobian_guard` 共 11 个模块导入成功。
- **活跃脚本**：`model_in_loop_hw_runbook.py`、`generate_bench_probe_report.py`、
  `probe_optical_path_validity.py`、`transfer_uncertainty_experiment.py` 全部 `py_compile` 通过。

### 模块规模门禁与拆分

`bench_geometry.py` 拆分后五个模块均低于 800 行、依赖严格向下无环：

| 模块 | 行数 | 角色 |
| --- | --- | --- |
| `bench_geometry_common.py` | 97 | 共享叶子（`_pupil_amplitude` / `_spot_fwhm` / `CalibrationRecord` / `square_metrics_at_zero_order`） |
| `bench_geometry.py` | 468 | `BenchGeometry` / `load_bench_geometry` / `calibrate_bench_geometry` |
| `bench_sweep.py` | 481 | `SweepRecord` / ramp 与 tilt 原始量 |
| `bench_sweep_fit.py` | 577 | `calibrate_bench_geometry_from_sweep`（扫掠拟合） |
| `bench_shared_aberration.py` | 687 | `calibrate_shared_aberration` / `FitAberrationResult` |

依赖方向为 `common → {geometry, sweep, sweep_fit, shared_aberration}`，`sweep_fit → geometry + sweep`；
9 个消费点改为直接导入，不设兼容 facade。拆分相关测试 **84 passed**。

`gui/ccd/target_shape_helper.py` 由 803 行拆为 `target_shape_helper.py`（665 行）+
`target_shape_plotly.py`（160 行），Plotly 渲染段整体移出，行为不变；相关测试 **15 passed**。
该文件越线源于基线提交 `3edd0a06`（698 行）之后的历史提交 `ae788a42`，与本次 MIL 移除无关，
按门禁要求先拆分再结算，而非登记豁免。

`python scripts/check_module_size.py --check` → **EXIT=0**（87 个受管 `src/` 模块）。
基线结算方向：收紧 4（含 `runner_common` -1019、`shaping_controls` -489）、移除 2（已删 MIL 模块）、
抬高 7（均为基线提交后的历史已提交增长，非本次引入）。

### 静态检查与测试

- `ruff check`：本次变更的 **69** 个 Python 文件全部通过。
- `test_report_provenance.py`：**816 passed**。
- `sync_report_provenance.py --check`：**clean**（202 份登记报告；顺带补写 `report/data_reference_inventory.md`
  缺失的 provenance 头）。
- 术语统一：`python3 scripts/_common/term_unify_20261016.py --check` → 0 个文件 0 行。
- 报告链接：`docs/paper_gs_refine_experiment.md` 引用的 7 张
  `report/shape_trials/model_in_loop/images/*.png` 全部存在；文档内 `raw/master/...` 图片链接无缺失。

已知范围外项（不影响本次结论）：全量 `ruff check .` 在
`report/slm/differentiable_opt/artifacts/` 的报告产物脚本中有 18 个 E402，为本分支未触碰的存量问题；
`test_model_in_loop_hw_runbook.py` 有两个 `bit_identical` 用例引用分支 HEAD 中的旧 runbook（其导入已删
`model_in_loop_shaping`），属提交前状态问题，与本次源码清理无关。

另有三类历史材料按决策保留原文，不再逐处改写：

- 生成物：`docs/dev/call_graph_wiki/relations/*` 与 `raw/**`、`assets/**` 仍含 `build_model_in_loop_config`
  等旧符号，由下次 `scripts/call_graph/build_similarity.py` 重建刷新。
- 历史记录：`CHANGELOG.md`、`docs/dev/todo_archive.md`、
  `report/slm_shaping_sim/slm_shaping_sim_report/report.md`、`report/model_free_ao_survey/report.md`、
  `docs/slm/闭环光束整形_文献与优化建议_20261014.md` 中的 MIL 叙述属当时结果与建议；
  `scripts/_common/provenance.py` 中登记的 2026-10 复现命令仍列出已删测试文件，
  作为历史复现记录不改写。
- 外部数据：`data/debug/slm_model_in_loop/` 等运行目录不在仓库范围内，
  `report/data_reference_inventory.md` 已按「本机缺失」登记。

**硬件状态：本次清理与验证仅离线进行，未打开或占用 SLM、CCD、激光或其他实验设备。**
