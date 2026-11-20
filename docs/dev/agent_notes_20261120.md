# Agent 当日记录 2026-11-20（纯离线，未占 SLM/CCD/激光）

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
