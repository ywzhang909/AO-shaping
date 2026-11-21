# Agent 当日记录 2026-11-20（§1–§8 纯离线；§9 起占用设备完成 P2/P3 采集与 P4 双门）

## 1. §0.3 预注册重签（已完成）

`hw_window_20261119_preregistration.md` §0 表 + §0.3 记录已按磁盘原始字节（CRLF）重新登记哈希；
`hw_window_20261119_execution_checklist.md` 内散落的 §0.1/旧哈希表述已全部同步为 §0.3。

评分口径语义未变，故评分不作废、不重算；复跑 `RECIPE_SELFCHECK=0.000e+00`，score JSON SHA256 不变。

## 2. P1 零级定心四口径离线对账（已完成）

- 脚本：`scripts/slm/issue235_zero_order_four_method_reconcile_20261120.py`
- 产物：`report/shape_trials/issue235_zero_order_four_method_20261120/`（报告 `.md` + `reconciliation.json`）
- 数据：`data/debug/issue235_center_20261117/flat.npy`（归档，本次未采集）

四法统一复用生产函数 `utils.image.zero_order.zero_order_center_by_method`（与测试同一实现），
六组两两欧氏距离最大 **3.0 px**（argmax vs centroid / box_centroid，恰在阈值上，边界通过），
判定 `verdict=ellipse_centre_usable`，默认椭圆中心 `(x=1227, y=1106)` 可用。

注意：归档 `summary.json` 里的 box_centroid 是浮点值 `(1106.23, 1225.99)`，与 argmax 直接相减
Δx≈3.008 px 会微超 3 px；本对账统一生产函数口径（同取整）判为 3.0 px，差异已在
`reconciliation.json` 的 `summary_crosscheck` 披露。CCD 帧中心与 SLM panel 刀口帧中心
（`find_beam_on_slm` / `fit_centre`，panel 1920×1200）不可混减。

## 3. 暗场 manifest 序列号复核：假阳性

工具把真实序列号脱敏渲染成 `FJB24112232`，文件中并不存在 `{{...}}` 模板字面量，manifest 无需修。

## 4. 新发现缺陷（未修，待决策）

`data/ccd_configs/ccd/FJB24112232/darks/manifest.json` 混存 3 条 8-bit（0.024/0.043/0.047 ms）
与 8 条 12-bit 暗场，而 `DarkFrameSet.match` 不按位深过滤，可能让 8-bit 帧进入 12-bit 判定。

修法见 `report/shape_trials/dark_frame_record_20261119.md`（文件名加 `-bd12`、`match(bit_depth=)`
显式过滤、旧条目标归档）。**未擅自实施**：修 manifest 或代码会改代码/数据哈希，触发 §0.3 重签连锁。

## 5. 门禁基线状态（非本轮引入）

- `sync_report_provenance.py --check`：`headers updated: 0`；仍有 6 份未登记报告（`tmp_sac_run_showcase.md`、
  `dark_frame_record_20261119.md`、`ladder_summary.md`、`stepb_optimizer_lr.md`、`sensitivity.md`、
  `online_hw/shape_ob...`），均为既有遗留。
- `test_report_provenance.py`：28 项失败全部落在 `model_in_loop` 区与 `physical_refine/fit_compare_m3_20261118.md`
  （声明 script 但无 repro command），以及上述 6 份未登记报告——HEAD 既有基线，本轮新增条目 0 失败。
- `ruff check scripts/_common/provenance.py`：`F601` 重复键 `report/slm/differentiable_opt/status_20261117.md`，
  HEAD 基线即存在（合并 `3d0fe7f2` 遗留），非本轮引入。
- 本轮新增/编辑行经 `term_unify_20261016.transform(_py)` 复核 term-clean；`hw_plan_consolidated_20261118.md`
  唯一 offender 在 line 126，属并行会话 hunk。

## 6. #242 focused test 阻塞根因与修复（2026-11-20）

初次复跑 `.venv\Scripts\python.exe -m pytest tests/ao_shaping/gui/slm/test_transfer_desktop_gates_ui.py -q --import-mode=importlib` 为 **11 failed, 8 passed，exit 1**；`py_compile` 同样因 `workflow.py` 游离的 `should_stop` 文档片段而失败。根因是已提交 `b7aa4746` 合并后 `_session_prepare_stage` 函数头缺失，函数体粘到 `_session_write_summary` 之后；Stop callback 契约在 `run_session` 文档中仍然存在。

已在 `src/ao_shaping/tools/slm/transfer/workflow.py` 恢复 `_session_prepare_stage(config, offline, on_stage)` 函数边界与原有本地校验（`config.validate()`、offline 数据集要求、`ccd_mode in ("light", "none")` no-camera 合约），并移除 `run_session` 内重复的校验片段。未改变 public signature、返回值、stage 顺序、日志/observer 行为；保留 `_session_write_summary` 定义以满足 conventions ratchet。复验：`.venv\Scripts\python.exe -m py_compile src/ao_shaping/tools/slm/transfer/workflow.py` **exit 0**；`.venv\Scripts\python.exe -m pytest tests/ao_shaping/gui/slm/test_transfer_desktop_gates_ui.py -q --import-mode=importlib` **19 passed in 4.41s，exit 0**。这修复了独立 transfer workflow 基线缺陷，不改变 #242 已 CLOSED 的功能判定。

**遗留（未修，需单独决定）**：`_session_write_summary(accepted, config, ..., summary)` 是合并 `3d0fe7f2`/`b7aa4746` 留下的**过期副本**——全库无调用者，且它缺少 `run_session` 现有的 `ccd_mode == "none"` → `open_loop.write_summary` 分支，若被调用会静默丢掉 open-loop 摘要。删除它必须同时删除 `test_conventions.py` `_PRINT_CONTRACT_ALLOWLIST` 中 `("tools/slm/transfer/workflow.py", "_session_write_summary"): 1` 这一条（该测试双向 pin，删函数不改 allowlist 会以 gone 失败），属测试基线改动，本轮不越界。

顺带发现的**既有**门禁漂移（本轮未修，跨车道）：`test_src_print_calls_stay_within_cli_contract_allowlist` 仍失败，extra 为 `tools/ccd/capture_darks.py:main`、`tools/slm/transfer/validate.py:_run_validate_blocks`、`tools/slm/transfer/workflow.py:run_session`，gone 为 `validate.py:_validate_block_loop`。其中 `workflow.py` 的 `run_session` 打印在 HEAD 版本即存在（合并 `3d0fe7f2`/`b7aa4746` 留下的内联 summary 与 `_session_write_summary` 助手重复），本轮恢复函数边界未增删任何 `print`；另两项位于其他会话的未提交改动文件，故不在本任务内改动他人车道。

## 7. #238 §6.6 第一阶段执行（清单核验 + 固定 ROI 描述量）

- 新脚本 `scripts/slm/issue238_run4_spatial_diagnostic_20261120.py`（未提交，`??`）：只读、不 import 硬件模块、不碰 `DeviceLock`、不写回原 run；`--self-test` 6/6 通过，`ruff` clean，`py_compile` exit 0，无 BOM。
- 全帧运行：两段各 `n_analyzed=960`、`levels=48`、`frames_failed=0`、`complete=true`，逐帧 shape/dtype/曝光/饱和/顺序检查全 ok，零丢帧；耗时 80.2 s / 83.6 s（流式，`allow_pickle=False`）。
- 产物只写 Temp：`C:/Users/zhangh/AppData/Local/Temp/opencode/issue238_run4_joint_diagnostic_20261120/diagnostic.json` + `run.log`；**未写 `report/`**。
- 途中修掉一个自造缺陷：`r50/r90` 原先对未截断的 `patch − bg_pedestal` 求累计（含负值 → 累计非单调 → `np.interp` 给出 93 px > 孔径的荒谬值），改为对 `clip(·,0,None)` 求累计，并在自测加「`r50 ≤ r90 ≤ aperture_r`」硬检查，改后复跑全量。
- 结果与边界（period 与时段/漂移完全共线，M0/M1 未实现 → §6.6 永久报告门槛未触发）见 `issue238_binary_gray_capture_plan_20261119.md` §6.7。判定不变：**不更新 LUT / 不冻结 2π 灰度 / 不宣称串扰定标**。

## 8. #235 / #247 剩余离线判别力复核（只读）

- **#235**：三个剩余子项均**零/极低判别力**——`shift_phase` 在归档中与理想无平移编码字节一致（mismatch=0）、`apply_shift` 在 shift 不变时不重写相位、`FrozenDefocusForward` 无该输入；独立 pupil（瞳面）-center 反演分辨率仅 Δx ±20 px / Δy ±30–40 px；物理 LUT 只有 `data/debug/lut_checker_modulation_20261118b/summary.json`（24 cells 棋盘格调制比，全库无消费脚本），且 Q1/Q2 的「实测点状 vs 模型满幅平顶」形态差不可能由逐像元振幅加权产生。**无离线项能改变 OPEN 判定。**
- **#247**：5 次重复逐帧角度 SD **0.088°**（稳定测量值），标签交换被 `AxisRotationError` 拒绝；但全局翻号给出 **−91.457°** ⇒ 估计量属 **180° 对称族**，符号分支只由 `image_alignment.convention` 钉住。`slm_geometry_calib.npz` 缺 panel/CCD pitch 与焦距放大率。`zero_order_overlay.py` 为死代码，`FrozenDefocusForward` 无零级项，唯一可微零级只在 sim2f 代理且未接入物理预测。梯度单测因缺 checkpoint 被 skip、batch/vmap 一致性无测试。**88.544° 可复现但不可冻结。**
- 详细记录已写入 `open_issue_offline_followup_20261119.md` 的「2026-11-20 纯离线可执行性复核」、`next_hw_plan_transfer_candidates_issue247_20261117.md` 的任务 B 追加节、`issue_validity_verdict_20261119.md` §7.1。

## 9. P2/P3 上游资格设备窗口 + P4 双门（2026-11-20，已占设备）

按 `transfer_upstream_qualification_plan_20261119.md` §6.1 冻结顺序单进程串行执行，全部带 `--wait-until-idle 120`、全新 `--out`、正常关闭，无并行占用。详见该文档 §5.1 表。

- **S0** `probe_optical_path_validity.py --bit-depth 8` → `data/debug/optical_path_probe/p3_upstream_20261120/20261120_012500Z/`：`probe_status=ok`、非饱和、离焦 +19.2%/+27.0%（SLM 真调制）、8-bit/255。
- **P3a** pass1/pass2 `direction_holdout_capture_20261119.py`：各 24 行全 `ok`（14 `test` × Z5–Z11 ±0.05 + 10 calibration），`split` 捕获时写死，段间 60 s 冷却。
- **P3b** `zernike_probe_hardware.py` → `p3_zernike_20261120/`：78 行（56 train / 12 test / 10 calibration），全 `0.087 ms`、Mono8/8-bit、全 `ok`。
- **P2** 用户确认物理挡光后，`capture_darks --exposures-ms 0.087 --frames 16 --bit-depth 8` → `FJB24112232/darks/dark-exp-0.087ms.npz`（`peak=2.0` 纯读出噪声，追加进共享库，其它档未动）。**仅开 CCD，SLM/激光未开**；09:52:08→09:52:13 正常关闭、`DeviceLock` 释放。
- **P4(a)** `refit_transfer_forward.py` → `p3_fit_20261120/frozen_parameters.json`（`checkpoint_sha256=380b9b87…`，train-only fit 门禁全绿，选迭代 209）。裸 loss 诊断：refit 在 test 上劣于基座（0.7866 vs 0.7188）——如实记录，未换模型。
- **P4(b) 强度门 fail**（exit 1）：12/12 holdout 全部可评估（暗场逐行匹配 0.087ms、8-bit/255、validity 前四项 ok），最差 `relL2=0.8849` ≫ δ=0.20。
- **P4(c) 方向门 fail**（exit 1）：pooled 4/14（需 10），pass1 2/7、pass2 2/7。
- **结论：双门均未通过（有效 fail，非 not_assessable）。transfer `light/none` 科学验收继续暂停，不进入 P5，未调阈值/换模型操纵通过。**
- 证据：`p3_gate_20261120/{strength_gate,direction_gate}.json`；运行日志同名 `*_run.log`。

### 9.1 速度解耦采集脚本（后台子代理产出，已审查）

- 新增 `scripts/slm/dynamic_shaping_capture_20261120.py` + `tests/ao_shaping/scripts/test_dynamic_shaping_capture_20261120.py`（5 用例）+ `src/ao_shaping/utils/slm/transition_timeline.py`（Qt-free 时间轴数学，GUI 模块 re-export 保持接口不变）。
- 设计：端点相位管线各跑一次（`create_phase_from_array`），中间帧纯灰度 `blend_gray`；SLM 主线程按墙钟派发，CCD 采样在独立 `CCDSampler` 线程（`--measure at-end|every|periodic`），阻塞假 CCD 不卡派发（测试锁定）；plan-only 默认不 import 驱动/不建目录。
- 验证：5/5 测试通过、`ruff` 四文件 exit 0、BOM/CRLF/AST 全绿。**纯离线，未占设备，未跑 `--execute`。**

### 9.2 速度解耦实验方案冻结（用户确认参数，见 `dynamic_decoupling_experiment_plan_20261120.md`）

- 用户 2026-11-20 交互页确认：6 形状（square/circle/triangle/star/rectangle/annular）× 4 尺寸（diameter 200/500/1000/1500，**CCD 目标平面原生像元**）；方法 = GS + transfer 模型两种都采；起点 flat gray 0；每帧测量（`every 1`）；12-bit 先探测后冻结曝光；保存原始 CCD 帧；单次过渡 5 s。
- 冻结方案：`docs/dev/dynamic_decoupling_experiment_plan_20261120.md`。48 条 transition，估算原始帧 ~22.98 GiB。**T3 曝光探测 / T4 采集仍须用户再次明确许可，本冻结不等于启动 `--execute`。**
- 与 transfer 科学验收严格分开：本实验的 model 相位仅作诊断，不得作为 P4 双门通过证据。
- **T2 采集器扩展已完成并独立复核**（`ses_e0938cf12ffez5yRxrvRrH8lcq`）：`dynamic_shaping_capture_20261120.py` 729→1278 行，新增 `--plan speed_decouple_plan/1`（多端点批次）、`--raw-frames stack`（memmap uint16 逐帧）、`--inter-transition-settle-s`、`--on-error`；新增 `tests/ao_shaping/scripts/test_dynamic_shaping_capture_plan_20261120.py`（351 行 / 7 用例）。主复核：12 tests pass（exit 0）、ruff clean（exit 0）；独立 plan-only 冒烟：2 transition 摘要正确、`DIR_EXISTS=False`、另测越界相位在 plan 阶段即被拒（exit 1）。**未占设备、未跑 `--execute`。** 全系列 plan-only 冒烟：48 条 transition（占位 panel-shaped 灰度，因相位矩阵车道尚未产出）全部打印摘要、`total_frames=2448`、exit 0、`DIR_EXISTS=NO`、stderr 空，临时文件已清理。

## 10. transfer 补资格路线调查（两条只读路线，均无文件/设备改动）

### 10.1 模型过拟合 / 早停路线（`ses_e094cfb32ffejqHDFDpRoBsGsy`）

- **选点机制**：`fit.py:345-372` 纯 train loss 选点（`best_loss` 创新低才覆盖 `result.json`），validation/held-out 不参与；本轮 `patience=999, min_delta=0`，早停不可能触发，跑满 0..220 共 221 迭代，选中 iteration 209（train 最低点）。
- **过拟合确证**（事后从 `iterations.pkl` 重建 test 走势，非用 test 选点）：test relL2 在 iter 18 最低 0.5500，选中点 209 退化到 0.5928；train relL2 同期 0.4092→0.3771。选中点恰在 test 已明显退化之后。
- **换迭代不能通关**：即使取 held-out 最优 test relL2 0.5500 仍 ≫ δ=0.20；方向门 pooled 4/14 与逐模态符号相关有关，非换迭代可保。失败根因是模型/观测算子层面，不是实现 bug 或选点噪声。
- **合法早停须重采**：当前 P3 只有 calibration/train/test，无 validation（calibration 明确排除）。从现有 train 事后切 validation 违反协议冻结；须在新一轮采数前预注册独立 validation split（train 更新 / validation 选迭代 / test 仅最终单次打分），δ=0.20 与两门定义不变。即便如此，早停本身仍不足以通关。

### 10.2 数据与曝光窗口路线（`ses_e094cfb2fffefxhIyx0KIbK4AC`）

- **P3b 信号有效**：78/78 行 `status=ok`、无饱和；8-bit 峰值 79–192（门限 250），spot 标准差 train ≈(0.97,1.19) px、test ≈(1.17,2.14) px，reference 起终相距 0.42 px。当前数据并非欠曝。
- **曝光窗口窄**：S0 阶梯 0.05 ms 峰值 90/95 无饱和，0.2 ms 已饱和 58 px；线性外推 120–180 counts 需约 0.063–0.095 ms，P3b 的 0.087 ms 已达该量级。适度提到 ~0.10 ms 可行（外推最大峰 ~221 < 250），但不可一律拉到 0.2 ms。自适应逻辑是宽接受带（25%–80% 满量程即接受），非逐行强制命中。
- **方向门 4/14 更像 checkpoint 问题**：新采两 pass 的 `model_phase` 与归档逐项 14/14 完全相同（无相位错配），新数据曝光不低于归档且峰值同量级或更高；refit 只有 4/14，归档诊断 checkpoint `hybrid320_fresh_all` 在归档 pass 上 9/14（仍 <10）。
- **重优先级**：① 先离线用较好的候选 checkpoint 在新 P3 两 pass 评估（不动设备/阈值）；② 若验证曝光影响，另建新数据集并预注册逐行目标峰值（120–180）、250-count 上限、匹配暗场，原 P3b test 继续冻结。重采可降低噪声但不能替代模型诊断。
- **共同结论**：两条路线都不支持"抢救"当前 fail；双门 fail 如实上报，不调阈值、不用 test 选点、不剔除行。补资格须先提升模型类/观测口径并预冻结新协议，再重采数。

## 11. 进程级 access violation（GUI 车道，2026-11-20）

- **现象**：间歇性进程级 `access violation`，Python 栈停在 `transfer_desktop/check_task.py` 的 `CcdDetectTask` → `hardware_check` 顶层 `import ao_shaping.drivers`（`drivers.adc` → `nidaqmx`、`drivers.ccd` → `gxipy`），dump 落在 `Garbage-collecting`。同类崩溃在 `docs/dev/todo_archive.md`（tqdm monitor 线程）已有记录：**CPython 3.13.2 的 GC 与后台原生线程**，不是逻辑错误。
- **爆点路径**：`beam_quality/shell.py` 的 `_transfer()` 用默认 `TransferPage()`（`auto_detect_ccd=True`）→ `_start_ccd_detection()` → `CcdDetectTask` 在页的 `QThreadPool` 线程里首次 import 驱动。窗口/页面先被回收、线程还在 import，就是原生崩溃而不是异常。
- **处置**（只离线，未碰 SLM/CCD/激光）：
  1. 外壳里改 `TransferPage(auto_detect_ccd=False)`：光束质量页在同一窗口，`_seed_camera_from_beam_quality` 已给出初值，重复枚举只是重复付原生 import 代价。独立 `transfer_desktop` 应用保留自动探测。
  2. `page.py`：新增 `_closing`（关窗后迟到探测结论丢弃），`closeEvent` 里 `_pool.clear()` + 有界 `waitForDone`（`_CLOSE_GRACE_S`），保证池线程落地后才让页被回收。
  3. `shell.py`：`_WarmTask` 改为 `setAutoDelete(False)` + `self._warm_task` 强引用（结论回来再放手），`closeEvent` 有界等待 `_warm_pool`（`_WARM_CLOSE_GRACE_MS = 15 s`）。
- **回归**：`test_beam_quality_shell.py::test_transfer_page_opens_without_hardware_probe`（外壳一次都不启动探测）；`test_transfer_desktop_ccd_auto.py::test_close_waits_for_inflight_detection_and_drops_late_result`（关窗等在飞任务落地、迟到结论不写控件）。
- **验证**：5 文件聚焦 `80 passed`；shell+ccd_auto 6 轮压力 27 passed/轮，exit 0，无 access violation；`ruff` 四文件 `All checks passed!`；四文件无 BOM、CRLF 一致。
- **未解决的环境层**：CPython 3.13.2 上"GC × 后台原生线程"仍可能在别处偶发（整库全量单进程跑约 32% 崩）。本车道做的是**不再无谓制造该路径**，不是修 CPython。
